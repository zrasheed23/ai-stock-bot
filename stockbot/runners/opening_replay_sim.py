"""
Historical opening replay — Steps 1–5.5 + simulated entry/exit (no broker, no ledger).

Orchestration only: reuses the same Step 1–5.5 functions as paper morning runs; never calls Step 6.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stockbot.config import Settings
from stockbot.execution.exit_engine import MINIMUM_HOLD_SECONDS_DEFAULT
from stockbot.execution.opening_allocation import build_step2_index
from stockbot.execution.paper_deployable import PAPER_DEPLOYABLE_EQUITY_FRACTION
from stockbot.ingestion.rth_minute_bars import fetch_rth_1min_full_session
from stockbot.runners.managed_position_ledger import (
    STOP_LOSS_PCT,
    step2_row_by_symbol,
    strong_stock_deterministic,
    take_profit_pct_for,
)
from stockbot.runners.opening_attribution_report import (
    analyze_opening_records,
    build_opening_attrib_record,
    print_opening_attribution_summary,
    relaxed_opening_env_label,
    write_opening_attribution_json,
)
from stockbot.runners.paper_open_run import run_paper_opening_through_5_5
from stockbot.strategy.engine import StrategyEngine

_LOG = logging.getLogger("stockbot.runners.opening_replay_sim")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def _die(msg: str) -> None:
    print(msg, file=__import__("sys").stderr)
    raise SystemExit(1)


def _is_weekday(d: date) -> bool:
    return d.weekday() < 5


def _require_env_replay(*, need_anthropic: bool) -> None:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    sec = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    missing: list[str] = []
    if not key:
        missing.append("ALPACA_API_KEY")
    if not sec:
        missing.append("ALPACA_SECRET_KEY")
    if missing:
        _die(f"Replay requires Alpaca market data credentials: {', '.join(missing)}")
    if need_anthropic:
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            _die("Missing ANTHROPIC_API_KEY (or run with --use-ai-cache after populating var/replay/opening_ai_cache/).")


def fetch_rth_1min_bars(symbol: str, session_date: date, settings: Settings) -> pd.DataFrame:
    """Regular-session 1-minute bars [09:30 ET, 16:05 ET) via shared Alpaca helper."""
    return fetch_rth_1min_full_session(symbol, session_date, settings)


def _entry_bar_and_price(df: pd.DataFrame, session_date: date) -> tuple[pd.Timestamp | None, float | None]:
    if df is None or df.empty:
        return None, None
    idx = pd.to_datetime(df.index, utc=True).tz_convert(_ET)
    open_et = datetime.combine(session_date, time(9, 30), tzinfo=_ET)
    for ts_idx, ts_et in zip(df.index, idx):
        if ts_et.date() == session_date and ts_et >= open_et:
            o = float(df.loc[ts_idx]["open"])
            if math.isfinite(o) and o > 0:
                return pd.Timestamp(ts_idx), o
    return None, None


def _simulate_intraday_exit(
    *,
    df: pd.DataFrame,
    session_date: date,
    entry_ts: pd.Timestamp,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[float, str]:
    """
    Returns (exit_price, reason) with reasons aligned to metrics:
    TAKE_PROFIT_HIT, STOP_LOSS_HIT, EOD_FLATTEN, SIM_DATA_MISSING.
    """
    if df is None or df.empty:
        return entry_price, "SIM_DATA_MISSING"

    stop_px = float(entry_price) * (1.0 + float(stop_loss_pct))
    tp_px = float(entry_price) * (1.0 + float(take_profit_pct))

    entry_t = pd.Timestamp(entry_ts).tz_convert(_UTC)

    eod_cutoff = datetime.combine(session_date, time(15, 55), tzinfo=_ET)

    sub = df.sort_index()
    idx_et = pd.to_datetime(sub.index, utc=True).tz_convert(_ET)

    last_valid_close: float | None = None
    for row_ts, ts_et in zip(sub.index, idx_et):
        bar = sub.loc[row_ts]
        if pd.Timestamp(row_ts).tz_convert(_UTC) < entry_t:
            continue

        bar_close = float(bar["close"])
        hi = float(bar["high"])
        lo = float(bar["low"])
        if not all(math.isfinite(x) for x in (bar_close, hi, lo)):
            return entry_price, "SIM_DATA_MISSING"

        last_valid_close = bar_close

        held_s = (pd.Timestamp(row_ts).tz_convert(_UTC) - entry_t).total_seconds()
        hold_ok = held_s >= float(MINIMUM_HOLD_SECONDS_DEFAULT)

        if ts_et >= eod_cutoff:
            return bar_close, "EOD_FLATTEN"

        if hold_ok:
            if lo <= stop_px:
                return stop_px, "STOP_LOSS_HIT"
            if hi >= tp_px:
                return tp_px, "TAKE_PROFIT_HIT"

    # Tape ended before 15:55 ET (early close) or last bars lacked hold-eligible TP/SL —
    # still have minute data; flatten at last regular-session close (checkpoint had SIM_DATA_MISSING: 0).
    if last_valid_close is not None and math.isfinite(last_valid_close):
        return float(last_valid_close), "EOD_FLATTEN"

    return entry_price, "SIM_DATA_MISSING"


def _replay_opening_midmorning_gate(
    *,
    df: pd.DataFrame,
    session_date: date,
    entry_ts: pd.Timestamp,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    notional_usd: float,
) -> tuple[bool, float]:
    """
    Opening-only gate: skip mid-morning when tape missing, entry invalid, or position still held
    at the 10:30 ET decision boundary. Otherwise returns incremental realized PnL from exits
    strictly before that boundary (same TP/SL/min-hold assumptions as full-day replay).
    """
    if df is None or df.empty or not math.isfinite(notional_usd) or notional_usd <= 0:
        return True, 0.0

    stop_px = float(entry_price) * (1.0 + float(stop_loss_pct))
    tp_px = float(entry_price) * (1.0 + float(take_profit_pct))
    entry_t = pd.Timestamp(entry_ts).tz_convert(_UTC)

    midmorning_cutoff = datetime.combine(session_date, time(10, 30), tzinfo=_ET)
    eod_cutoff = datetime.combine(session_date, time(15, 55), tzinfo=_ET)

    sub = df.sort_index()
    idx_et = pd.to_datetime(sub.index, utc=True).tz_convert(_ET)

    for row_ts, ts_et in zip(sub.index, idx_et):
        bar = sub.loc[row_ts]
        if pd.Timestamp(row_ts).tz_convert(_UTC) < entry_t:
            continue

        bar_close = float(bar["close"])
        hi = float(bar["high"])
        lo = float(bar["low"])
        if not all(math.isfinite(x) for x in (bar_close, hi, lo)):
            return True, 0.0

        if ts_et >= midmorning_cutoff:
            return True, 0.0

        held_s = (pd.Timestamp(row_ts).tz_convert(_UTC) - entry_t).total_seconds()
        hold_ok = held_s >= float(MINIMUM_HOLD_SECONDS_DEFAULT)

        if ts_et >= eod_cutoff:
            pnl = float(notional_usd) * ((bar_close / float(entry_price)) - 1.0)
            return False, pnl

        if hold_ok:
            if lo <= stop_px:
                pnl = float(notional_usd) * ((stop_px / float(entry_price)) - 1.0)
                return False, pnl
            if hi >= tp_px:
                pnl = float(notional_usd) * ((tp_px / float(entry_price)) - 1.0)
                return False, pnl

    return True, 0.0


def _entry_bar_midmorning(df: pd.DataFrame, session_date: date) -> tuple[pd.Timestamp | None, float | None]:
    """First regular-session bar at or after 10:30 ET."""
    if df is None or df.empty:
        return None, None
    target = datetime.combine(session_date, time(10, 30), tzinfo=_ET)
    idx = pd.to_datetime(df.index, utc=True).tz_convert(_ET)
    for ts_idx, ts_et in zip(df.index, idx):
        if ts_et.date() != session_date:
            continue
        if ts_et >= target:
            o = float(df.loc[ts_idx]["open"])
            if math.isfinite(o) and o > 0:
                return pd.Timestamp(ts_idx), o
    return None, None


def _cache_path(base: Path, trade_date: date) -> Path:
    return base / f"{trade_date.isoformat()}.json"


def _load_ai_cache(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("opening_decision_raw_text"), str):
        return str(obj["opening_decision_raw_text"])
    if isinstance(obj, str):
        return obj
    return None


def _save_ai_cache(path: Path, opening_decision_raw_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"opening_decision_raw_text": opening_decision_raw_text}, indent=2),
        encoding="utf-8",
    )


def _print_opening_diagnostic_summary(summary: Mapping[str, Any]) -> None:
    """Human-readable aggregate of Step 4 diagnostics (stdout after JSON summary)."""
    print("")
    print("=== Opening allocation diagnostic summary (replay aggregate) ===")
    print(f"days_tested={summary.get('days_tested')}")
    orc = summary.get("opening_reject_reason_counts") or {}
    print("opening_reject_reason_counts:")
    if not orc:
        print("  (none)")
    else:
        for k in sorted(orc.keys()):
            print(f"  {k}: {orc[k]}")
    print(f"opening_surgical_pm_return_reject_total={summary.get('opening_surgical_pm_return_reject_total', 0)}")
    print(f"opening_surgical_expected_move_reject_total={summary.get('opening_surgical_expected_move_reject_total', 0)}")
    print(f"decision_status_not_trade_days={summary.get('decision_status_not_trade_days', 0)}")
    print(f"no_candidates_empty_days={summary.get('no_candidates_empty_days', 0)}")
    print(f"confidence_pass_direction_fail_count={summary.get('confidence_pass_direction_fail_count', 0)}")
    print(
        "confidence_direction_pass_volume_fail_count="
        f"{summary.get('confidence_direction_pass_volume_fail_count', 0)}"
    )
    print(f"soft_band_candidate_count={summary.get('soft_band_candidate_count', 0)}")
    print(f"soft_band_expected_move_pass_count={summary.get('soft_band_expected_move_pass_count', 0)}")
    print(f"soft_band_accepted_count={summary.get('soft_band_accepted_count', 0)}")
    print(f"allocation_ready_plan_blocked_days={summary.get('allocation_ready_plan_blocked_days', 0)}")
    print("(allocation_ready_plan_blocked_days = Step 4 ready but validated plan had no instructions)")
    oeb = summary.get("opening_execution_plan_block_failure_counts") or {}
    print("opening_execution_plan_block_failure_counts:")
    if not oeb:
        print("  (none)")
    else:
        for k in sorted(oeb.keys()):
            print(f"  {k}: {oeb[k]}")
    msr = summary.get("midmorning_skip_primary_reason_counts") or {}
    print("midmorning_skip_primary_reason_counts:")
    if not msr:
        print("  (none)")
    else:
        for k in sorted(msr.keys()):
            print(f"  {k}: {msr[k]}")
    print(f"opening_trade_count={summary.get('opening_trade_count', 0)}")
    print(f"midmorning_trade_count={summary.get('midmorning_trade_count', 0)}")
    print(f"opening_sim_data_missing_count={summary.get('opening_sim_data_missing_count', 0)}")
    print(f"midmorning_sim_data_missing_count={summary.get('midmorning_sim_data_missing_count', 0)}")
    print(f"opening_step2_missing_or_invalid_fields_count={summary.get('opening_step2_missing_or_invalid_fields_count', 0)}")
    print(
        "(opening_step2_missing = Step 3 candidate symbols with bad Step 2 rows only; "
        "see opening_step2_watchlist_symbols_missing_or_invalid_total for full watchlist)"
    )
    print(
        "opening_step2_watchlist_symbols_missing_or_invalid_total="
        f"{summary.get('opening_step2_watchlist_symbols_missing_or_invalid_total', 0)}"
    )
    print("opening_step2_not_ok_reason_counts:")
    s2n = summary.get("opening_step2_not_ok_reason_counts") or {}
    if not s2n:
        print("  (none)")
    else:
        for k in sorted(s2n.keys()):
            print(f"  {k}: {s2n[k]}")
    print("opening_initial_no_trade_subtype_counts:")
    ositc = summary.get("opening_initial_no_trade_subtype_counts") or {}
    if not ositc:
        print("  (none)")
    else:
        for k in sorted(ositc.keys()):
            print(f"  {k}: {ositc[k]}")
    print(f"source_override_count={summary.get('source_override_count', 0)}")
    print(f"source_override_budget_max={summary.get('source_override_budget_max', 0)}")
    dbg = summary.get("debug_comparison_summary") or {}
    print("")
    print("=== Temporary debug comparison summary (opening replay) ===")
    print(f"num_trade_days={dbg.get('num_trade_days')}")
    print(f"num_trades={dbg.get('num_trades')}")
    print(f"sim_data_missing_dates={dbg.get('sim_data_missing_dates')}")
    print(f"source_override_dates={dbg.get('source_override_dates')}")
    print(f"no_trade_dates={dbg.get('no_trade_dates')}")
    print(f"allocation_blocked_dates={dbg.get('allocation_blocked_dates')}")
    print(f"step2_missing_dates={dbg.get('step2_missing_dates')}")
    print("===========================================================")
    print(f"slot2_trade_count={summary.get('slot2_trade_count', 0)}")
    print(f"slot2_avg_return={summary.get('slot2_avg_return', 0)}")
    print(f"slot1_full_sleeve_count={summary.get('slot1_full_sleeve_count', 0)}")
    print("slot2_reject_reason_counts:")
    s2r = summary.get("slot2_reject_reason_counts") or {}
    if not s2r:
        print("  (none)")
    else:
        for k in sorted(s2r.keys()):
            print(f"  {k}: {s2r[k]}")
    print("================================================================")


def _would_submit(validated_plan: Mapping[str, Any]) -> bool:
    if validated_plan.get("execution_status") == "no_execution":
        return False
    instr = validated_plan.get("instructions")
    return isinstance(instr, list) and len(instr) > 0


def _instruction_notional_usd(inst: Mapping[str, Any], entry_price: float) -> float:
    mode = inst.get("mode")
    if mode == "notional_market":
        nu = inst.get("notional_usd")
        if nu is None:
            return 0.0
        v = float(nu)
        return v if math.isfinite(v) and v > 0 else 0.0
    if mode == "shares_market":
        sh = inst.get("shares")
        if sh is None:
            return 0.0
        q = int(sh)
        return max(0.0, float(q) * float(entry_price))
    return 0.0


def _count_step2_watchlist_symbols_missing_or_invalid(step2_packet: Mapping[str, Any] | None) -> int:
    """
    All Step 2 watchlist rows failing status/volume shape (aggregate diagnostic only — does not gate trades).
    """
    if not isinstance(step2_packet, Mapping):
        return 0
    syms = step2_packet.get("symbols")
    if not isinstance(syms, list):
        return 0
    n = 0
    for row in syms:
        if not isinstance(row, Mapping):
            n += 1
            continue
        if row.get("status") != "ok":
            n += 1
            continue
        pmv = row.get("pm_volume")
        try:
            v = float(pmv)
        except (TypeError, ValueError):
            n += 1
            continue
        if not math.isfinite(v) or v < 0:
            n += 1
            continue
    return n


def _count_step2_opening_candidates_missing_or_invalid(
    step2_packet: Mapping[str, Any] | None,
    validated_decision: Mapping[str, Any] | None,
) -> int:
    """Step 2 shape issues for symbols in Step 3 trade candidates only (opening-relevant scope)."""
    if not isinstance(validated_decision, Mapping):
        return 0
    if validated_decision.get("decision_status") != "trade":
        return 0
    cands = validated_decision.get("candidates")
    if not isinstance(cands, list):
        return 0
    idx = build_step2_index(step2_packet)
    n = 0
    for c in cands:
        if not isinstance(c, Mapping):
            n += 1
            continue
        sym = str(c.get("symbol", "")).strip().upper()
        if not sym:
            n += 1
            continue
        row = idx.get(sym)
        if not isinstance(row, Mapping):
            n += 1
            continue
        if row.get("status") != "ok":
            n += 1
            continue
        pmv = row.get("pm_volume")
        try:
            v = float(pmv)
        except (TypeError, ValueError):
            n += 1
            continue
        if not math.isfinite(v) or v < 0:
            n += 1
            continue
    return n


def _accumulate_step2_not_ok_reasons(step4: Mapping[str, Any] | None, counter: Counter[str]) -> None:
    if not isinstance(step4, Mapping):
        return
    rej = step4.get("rejected")
    if not isinstance(rej, list):
        return
    for item in rej:
        if not isinstance(item, Mapping):
            continue
        rc = item.get("reason_code")
        if not isinstance(rc, str):
            continue
        if "STEP2_NOT_OK" in rc:
            counter[rc] += 1


def run_opening_replay_range(
    start: date,
    end: date,
    *,
    starting_equity: float,
    settings: Settings,
    strategy: StrategyEngine,
    use_ai_cache: bool,
    cache_dir: Path,
    replay_out_dir: Path,
    enable_midmorning: bool = True,
) -> dict[str, Any]:
    """
    Iterate weekdays in [start, end]; write CSV + JSON summary under ``replay_out_dir``.
    """
    replay_out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0

    csv_path = replay_out_dir / f"replay_{start.isoformat()}_{end.isoformat()}.csv"
    rows_out: list[dict[str, Any]] = []
    exit_counts: Counter[str] = Counter()
    days_tested = 0
    trade_days = 0
    no_trade_days = 0
    error_days = 0
    n_trades = 0
    trade_returns: list[float] = []
    trade_pnls: list[float] = []
    opening_trade_returns: list[float] = []
    opening_trade_pnls: list[float] = []
    midmorning_trade_returns: list[float] = []
    midmorning_trade_pnls: list[float] = []
    daily_returns: list[float] = []
    day_pnls: list[float] = []
    slot2_returns: list[float] = []

    opening_reject_agg: Counter[str] = Counter()
    slot2_reject_agg: Counter[str] = Counter()
    slot1_full_sleeve_count = 0
    soft_band_candidate_total = 0
    soft_band_expected_move_pass_total = 0
    soft_band_accepted_total = 0
    confidence_pass_direction_fail_total = 0
    confidence_direction_pass_volume_fail_total = 0
    no_candidates_empty_total = 0
    decision_status_not_trade_total = 0
    allocation_ready_plan_blocked_days = 0
    opening_initial_no_trade_subtype_agg: Counter[str] = Counter()
    source_override_budget: dict[str, int] = {"max": 8, "used": 0}
    midmorning_skip_position_open_days = 0
    midmorning_skip_filters_days = 0
    midmorning_plan_trade_days = 0
    opening_sim_data_missing_count = 0
    midmorning_sim_data_missing_count = 0
    opening_step2_not_ok_reason_agg: Counter[str] = Counter()
    opening_step2_missing_or_invalid_fields_count = 0
    opening_step2_watchlist_symbols_missing_or_invalid_total = 0
    debug_sim_missing_dates_set: set[str] = set()
    debug_source_override_dates: list[str] = []
    debug_no_trade_dates: list[str] = []
    debug_allocation_blocked_dates: list[str] = []
    debug_step2_candidate_issue_dates: list[str] = []
    opening_execution_plan_block_diag_agg: Counter[str] = Counter()
    midmorning_skip_primary_reason_agg: Counter[str] = Counter()
    opening_surgical_pm_reject_total = 0
    opening_surgical_em_reject_total = 0

    relaxed_opening_env_snapshot = relaxed_opening_env_label()
    opening_attrib_records: list[dict[str, Any]] = []

    d = start
    while d <= end:
        if not _is_weekday(d):
            d += timedelta(days=1)
            continue

        equity_day_start = float(equity)
        days_tested += 1
        deployable_sim = equity_day_start * float(PAPER_DEPLOYABLE_EQUITY_FRACTION)

        try:
            cache_file = _cache_path(cache_dir, d)
            opening_raw: str | None = None
            if use_ai_cache:
                opening_raw = _load_ai_cache(cache_file)

            if opening_raw is None:
                through = run_paper_opening_through_5_5(
                    d,
                    settings=settings,
                    strategy=strategy,
                    deployable_usd_override=deployable_sim,
                    allocation_diagnostics=True,
                    source_no_trade_override_budget=source_override_budget,
                )
                opening_raw = str(through["opening_decision_raw_text"])
                if use_ai_cache:
                    _save_ai_cache(cache_file, opening_raw)
            else:
                through = run_paper_opening_through_5_5(
                    d,
                    settings=settings,
                    strategy=strategy,
                    deployable_usd_override=deployable_sim,
                    opening_decision_raw_text=opening_raw,
                    allocation_diagnostics=True,
                    source_no_trade_override_budget=source_override_budget,
                )

            step2 = through["step2_packet"]
            validated_decision = through["validated_decision"]
            validated_plan = through["validated_plan"]
            odm = through.get("opening_decision_meta")
            src_ov_applied = bool(isinstance(odm, Mapping) and odm.get("source_override_applied"))
            if isinstance(odm, Mapping):
                if odm.get("initial_decision_status") == "no_trade":
                    opening_initial_no_trade_subtype_agg[str(odm.get("no_trade_subtype") or "unknown")] += 1
                if odm.get("source_override_applied"):
                    debug_source_override_dates.append(d.isoformat())

            exec_dx_open = through.get("execution_plan_block_diagnosis")
            if isinstance(exec_dx_open, Mapping) and exec_dx_open.get("ok") is False:
                opening_execution_plan_block_diag_agg[str(exec_dx_open.get("failure_code") or "unknown")] += 1

            rth_cache: dict[str, pd.DataFrame] = {}

            def _day_rth(sym_u: str) -> pd.DataFrame:
                if sym_u not in rth_cache:
                    rth_cache[sym_u] = fetch_rth_1min_bars(sym_u, d, settings)
                return rth_cache[sym_u]

            cand_miss = _count_step2_opening_candidates_missing_or_invalid(step2, validated_decision)
            opening_step2_missing_or_invalid_fields_count += cand_miss
            opening_step2_watchlist_symbols_missing_or_invalid_total += (
                _count_step2_watchlist_symbols_missing_or_invalid(step2)
            )
            if cand_miss > 0:
                debug_step2_candidate_issue_dates.append(d.isoformat())

            step4 = through.get("step4_allocation")
            _accumulate_step2_not_ok_reasons(step4, opening_step2_not_ok_reason_agg)
            if isinstance(step4, Mapping):
                adx = step4.get("allocation_diagnostics")
                if isinstance(adx, Mapping):
                    for rk, rv in (adx.get("opening_reject_reason_counts") or {}).items():
                        opening_reject_agg[str(rk)] += int(rv)
                    opening_surgical_pm_reject_total += int(adx.get("opening_surgical_pm_return_reject_count") or 0)
                    opening_surgical_em_reject_total += int(adx.get("opening_surgical_expected_move_reject_count") or 0)
                    soft_band_candidate_total += int(adx.get("soft_band_candidate_count") or 0)
                    soft_band_expected_move_pass_total += int(adx.get("soft_band_expected_move_pass_count") or 0)
                    soft_band_accepted_total += int(adx.get("soft_band_accepted_count") or 0)
                    confidence_pass_direction_fail_total += int(
                        adx.get("confidence_pass_direction_fail_count") or 0
                    )
                    confidence_direction_pass_volume_fail_total += int(
                        adx.get("confidence_direction_pass_volume_fail_count") or 0
                    )
                    no_candidates_empty_total += int(adx.get("no_candidates_empty_count") or 0)
                    decision_status_not_trade_total += int(adx.get("decision_status_not_trade_count") or 0)
                    for sk, sv in (adx.get("slot2_reject_reason_counts") or {}).items():
                        slot2_reject_agg[str(sk)] += int(sv)
                    if bool(adx.get("slot1_full_sleeve_only")):
                        slot1_full_sleeve_count += 1
                if step4.get("preparation_status") == "ready" and not _would_submit(validated_plan):
                    allocation_ready_plan_blocked_days += 1
                    debug_allocation_blocked_dates.append(d.isoformat())

            equity_start_day = equity_day_start
            day_pnl_opening = 0.0
            if not _would_submit(validated_plan):
                no_trade_days += 1
                debug_no_trade_dates.append(d.isoformat())
                exit_counts["NO_TRADE"] += 1
                rows_out.append(
                    {
                        "trade_date": d.isoformat(),
                        "symbol": "",
                        "rank": "",
                        "notional_usd": "",
                        "entry_price": "",
                        "exit_price": "",
                        "return_pct": "",
                        "pnl_usd": "",
                        "exit_reason": "NO_TRADE",
                        "session": "opening",
                        "equity_after": f"{equity:.6f}",
                    }
                )
            else:
                trade_days += 1
                s2_by = step2_row_by_symbol(step2)
                cum_day = 0.0

                for inst in validated_plan.get("instructions") or []:
                    if not isinstance(inst, Mapping):
                        continue
                    sym = str(inst.get("symbol", "")).strip().upper()
                    if not sym:
                        continue
                    rank = inst.get("rank", "")
                    try:
                        rank_i = int(rank) if rank is not None and rank != "" else 0
                    except (TypeError, ValueError):
                        rank_i = 0
                    df = _day_rth(sym)
                    entry_ts, entry_px = _entry_bar_and_price(df, d)
                    if entry_ts is None or entry_px is None:
                        exit_counts["SIM_DATA_MISSING"] += 1
                        opening_sim_data_missing_count += 1
                        debug_sim_missing_dates_set.add(d.isoformat())
                        rows_out.append(
                            {
                                "trade_date": d.isoformat(),
                                "symbol": sym,
                                "rank": rank,
                                "notional_usd": "",
                                "entry_price": "",
                                "exit_price": "",
                                "return_pct": "",
                                "pnl_usd": "",
                                "exit_reason": "SIM_DATA_MISSING",
                                "session": "opening",
                                "equity_after": f"{equity_start_day + cum_day:.6f}",
                            }
                        )
                        continue

                    notional = _instruction_notional_usd(inst, entry_px)
                    if notional <= 0:
                        exit_counts["SIM_DATA_MISSING"] += 1
                        opening_sim_data_missing_count += 1
                        debug_sim_missing_dates_set.add(d.isoformat())
                        rows_out.append(
                            {
                                "trade_date": d.isoformat(),
                                "symbol": sym,
                                "rank": rank,
                                "notional_usd": "0",
                                "entry_price": f"{entry_px:.6f}",
                                "exit_price": "",
                                "return_pct": "",
                                "pnl_usd": "",
                                "exit_reason": "SIM_DATA_MISSING",
                                "session": "opening",
                                "equity_after": f"{equity_start_day + cum_day:.6f}",
                            }
                        )
                        continue

                    cands = (
                        validated_decision.get("candidates") if isinstance(validated_decision, Mapping) else []
                    )
                    ai_conf = 0.0
                    if isinstance(cands, list):
                        for c in cands:
                            if not isinstance(c, Mapping):
                                continue
                            if str(c.get("symbol", "")).strip().upper() == sym:
                                try:
                                    ai_conf = float(c.get("confidence", 0.0))
                                except (TypeError, ValueError):
                                    ai_conf = 0.0
                                break
                    strong = strong_stock_deterministic(ai_confidence=ai_conf, step2_row=s2_by.get(sym))
                    tp_pct = take_profit_pct_for(strong=strong)
                    exit_px, reason = _simulate_intraday_exit(
                        df=df,
                        session_date=d,
                        entry_ts=entry_ts,
                        entry_price=entry_px,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=STOP_LOSS_PCT,
                    )
                    if reason == "SIM_DATA_MISSING":
                        opening_sim_data_missing_count += 1
                        debug_sim_missing_dates_set.add(d.isoformat())
                    ret = (exit_px / entry_px) - 1.0 if entry_px > 0 else float("nan")
                    pnl = notional * ret if math.isfinite(ret) else 0.0
                    exit_counts[reason] += 1
                    n_trades += 1
                    opening_attrib_records.append(
                        build_opening_attrib_record(
                            trade_date=d,
                            symbol=sym,
                            rank_i=rank_i,
                            ai_confidence=float(ai_conf),
                            exit_reason=str(reason),
                            ret=float(ret) if math.isfinite(ret) else 0.0,
                            pnl_usd=float(pnl),
                            notional_usd=float(notional),
                            step2_row=s2_by.get(sym),
                            validated_decision=validated_decision
                            if isinstance(validated_decision, Mapping)
                            else None,
                            step2_packet=step2 if isinstance(step2, Mapping) else None,
                            source_override_applied=src_ov_applied,
                            relaxed_opening_env_snapshot=relaxed_opening_env_snapshot,
                        )
                    )
                    if math.isfinite(ret) and reason in (
                        "TAKE_PROFIT_HIT",
                        "STOP_LOSS_HIT",
                        "EOD_FLATTEN",
                    ):
                        trade_returns.append(float(ret))
                        trade_pnls.append(float(pnl))
                        opening_trade_returns.append(float(ret))
                        opening_trade_pnls.append(float(pnl))
                        if rank_i == 2:
                            slot2_returns.append(float(ret))
                    day_pnl_opening += pnl
                    cum_day += pnl

                    rows_out.append(
                        {
                            "trade_date": d.isoformat(),
                            "symbol": sym,
                            "rank": rank,
                            "notional_usd": f"{notional:.6f}",
                            "entry_price": f"{entry_px:.6f}",
                            "exit_price": f"{exit_px:.6f}",
                            "return_pct": f"{ret:.8f}" if math.isfinite(ret) else "",
                            "pnl_usd": f"{pnl:.6f}",
                            "exit_reason": reason,
                            "session": "opening",
                            "equity_after": f"{equity_start_day + cum_day:.6f}",
                        }
                    )

            skip_midmorning = False
            early_open_pnl_mm_gate = 0.0
            if enable_midmorning:
                if _would_submit(validated_plan):
                    s2_gate = step2_row_by_symbol(step2)
                    cum_early = 0.0
                    for inst_g in validated_plan.get("instructions") or []:
                        if not isinstance(inst_g, Mapping):
                            continue
                        sym_g = str(inst_g.get("symbol", "")).strip().upper()
                        if not sym_g:
                            continue
                        df_g = _day_rth(sym_g)
                        entry_ts_g, entry_px_g = _entry_bar_and_price(df_g, d)
                        if entry_ts_g is None or entry_px_g is None:
                            skip_midmorning = True
                            break
                        notional_g = _instruction_notional_usd(inst_g, entry_px_g)
                        if notional_g <= 0:
                            skip_midmorning = True
                            break
                        cands_g = (
                            validated_decision.get("candidates") if isinstance(validated_decision, Mapping) else []
                        )
                        ai_conf_g = 0.0
                        if isinstance(cands_g, list):
                            for c in cands_g:
                                if not isinstance(c, Mapping):
                                    continue
                                if str(c.get("symbol", "")).strip().upper() == sym_g:
                                    try:
                                        ai_conf_g = float(c.get("confidence", 0.0))
                                    except (TypeError, ValueError):
                                        ai_conf_g = 0.0
                                    break
                        strong_g = strong_stock_deterministic(
                            ai_confidence=ai_conf_g, step2_row=s2_gate.get(sym_g)
                        )
                        tp_pct_g = take_profit_pct_for(strong=strong_g)
                        sk, pnl_leg = _replay_opening_midmorning_gate(
                            df=df_g,
                            session_date=d,
                            entry_ts=entry_ts_g,
                            entry_price=entry_px_g,
                            take_profit_pct=tp_pct_g,
                            stop_loss_pct=STOP_LOSS_PCT,
                            notional_usd=notional_g,
                        )
                        if sk:
                            skip_midmorning = True
                            break
                        cum_early += pnl_leg
                    if skip_midmorning:
                        midmorning_skip_position_open_days += 1
                        _LOG.info("MIDMORNING_SKIP reason=position_open trade_date=%s", d.isoformat())
                    else:
                        early_open_pnl_mm_gate = cum_early

            day_pnl_mm = 0.0
            if enable_midmorning and not skip_midmorning:
                deploy_mm = (equity_day_start + early_open_pnl_mm_gate) * float(
                    PAPER_DEPLOYABLE_EQUITY_FRACTION
                )
                from stockbot.runners.paper_open_run import run_paper_midmorning_through_5_5

                mm_through = run_paper_midmorning_through_5_5(
                    d,
                    settings=settings,
                    strategy=strategy,
                    deployable_usd_override=deploy_mm,
                    allocation_diagnostics=False,
                )

                mm_plan = mm_through["validated_plan"]

                if not _would_submit(mm_plan):
                    midmorning_skip_filters_days += 1
                    mp = mm_through.get("midmorning_pipeline_outcome") or {}
                    if not mp.get("sector_rs_filter_pass"):
                        midmorning_skip_primary_reason_agg[
                            f"sector_rs:{mp.get('sector_rs_skip_reason') or 'unknown'}"
                        ] += 1
                    elif mp.get("step4_preparation_status") != "ready":
                        midmorning_skip_primary_reason_agg[
                            f"step4:{mp.get('step4_preparation_status') or 'unknown'}"
                        ] += 1
                    elif mp.get("validated_execution_status") == "no_execution":
                        ddx = mp.get("execution_plan_diagnosis")
                        fc = (
                            str(ddx.get("failure_code") or "unknown")
                            if isinstance(ddx, Mapping)
                            else "no_diagnosis"
                        )
                        midmorning_skip_primary_reason_agg[f"plan_validate:{fc}"] += 1
                    else:
                        midmorning_skip_primary_reason_agg["no_instructions_other"] += 1
                else:
                    midmorning_plan_trade_days += 1
                    s2_mm = step2_row_by_symbol(mm_through["step2_packet"])
                    vd_mm = mm_through["validated_decision"]
                    cum_mm = 0.0
                    base_after_opening = equity_start_day + day_pnl_opening
                    for inst_m in mm_plan.get("instructions") or []:
                        if not isinstance(inst_m, Mapping):
                            continue
                        sym_m = str(inst_m.get("symbol", "")).strip().upper()
                        if not sym_m:
                            continue
                        rank_m = inst_m.get("rank", "")
                        try:
                            rank_mi = int(rank_m) if rank_m is not None and rank_m != "" else 0
                        except (TypeError, ValueError):
                            rank_mi = 0
                        df_m = _day_rth(sym_m)
                        entry_ts_m, entry_px_m = _entry_bar_midmorning(df_m, d)
                        if entry_ts_m is None or entry_px_m is None:
                            exit_counts["SIM_DATA_MISSING"] += 1
                            midmorning_sim_data_missing_count += 1
                            rows_out.append(
                                {
                                    "trade_date": d.isoformat(),
                                    "symbol": sym_m,
                                    "rank": rank_m,
                                    "notional_usd": "",
                                    "entry_price": "",
                                    "exit_price": "",
                                    "return_pct": "",
                                    "pnl_usd": "",
                                    "exit_reason": "SIM_DATA_MISSING",
                                    "session": "midmorning",
                                    "equity_after": f"{base_after_opening + cum_mm:.6f}",
                                }
                            )
                            continue

                        notional_m = _instruction_notional_usd(inst_m, entry_px_m)
                        if notional_m <= 0:
                            exit_counts["SIM_DATA_MISSING"] += 1
                            midmorning_sim_data_missing_count += 1
                            rows_out.append(
                                {
                                    "trade_date": d.isoformat(),
                                    "symbol": sym_m,
                                    "rank": rank_m,
                                    "notional_usd": "0",
                                    "entry_price": f"{entry_px_m:.6f}",
                                    "exit_price": "",
                                    "return_pct": "",
                                    "pnl_usd": "",
                                    "exit_reason": "SIM_DATA_MISSING",
                                    "session": "midmorning",
                                    "equity_after": f"{base_after_opening + cum_mm:.6f}",
                                }
                            )
                            continue

                        cands_m = vd_mm.get("candidates") if isinstance(vd_mm, Mapping) else []
                        ai_conf_m = 0.0
                        if isinstance(cands_m, list):
                            for c in cands_m:
                                if not isinstance(c, Mapping):
                                    continue
                                if str(c.get("symbol", "")).strip().upper() == sym_m:
                                    try:
                                        ai_conf_m = float(c.get("confidence", 0.0))
                                    except (TypeError, ValueError):
                                        ai_conf_m = 0.0
                                    break
                        strong_m = strong_stock_deterministic(
                            ai_confidence=ai_conf_m, step2_row=s2_mm.get(sym_m)
                        )
                        tp_pct_m = take_profit_pct_for(strong=strong_m)
                        exit_px_m, reason_m = _simulate_intraday_exit(
                            df=df_m,
                            session_date=d,
                            entry_ts=entry_ts_m,
                            entry_price=entry_px_m,
                            take_profit_pct=tp_pct_m,
                            stop_loss_pct=STOP_LOSS_PCT,
                        )
                        ret_m = (exit_px_m / entry_px_m) - 1.0 if entry_px_m > 0 else float("nan")
                        pnl_m = notional_m * ret_m if math.isfinite(ret_m) else 0.0
                        exit_counts[reason_m] += 1
                        n_trades += 1
                        if math.isfinite(ret_m) and reason_m in (
                            "TAKE_PROFIT_HIT",
                            "STOP_LOSS_HIT",
                            "EOD_FLATTEN",
                        ):
                            trade_returns.append(float(ret_m))
                            trade_pnls.append(float(pnl_m))
                            midmorning_trade_returns.append(float(ret_m))
                            midmorning_trade_pnls.append(float(pnl_m))
                            if rank_mi == 2:
                                slot2_returns.append(float(ret_m))
                        day_pnl_mm += pnl_m
                        cum_mm += pnl_m
                        w_m = ""
                        for trw in mm_through["step4_allocation"].get("trades") or []:
                            if isinstance(trw, Mapping) and str(trw.get("symbol") or "").strip().upper() == sym_m:
                                w_m = str(trw.get("capital_weight") or "")
                                break
                        _LOG.info(
                            "MIDMORNING_EXECUTION symbol=%s weight=%s trade_date=%s",
                            sym_m,
                            w_m or "",
                            d.isoformat(),
                        )

                        rows_out.append(
                            {
                                "trade_date": d.isoformat(),
                                "symbol": sym_m,
                                "rank": rank_m,
                                "notional_usd": f"{notional_m:.6f}",
                                "entry_price": f"{entry_px_m:.6f}",
                                "exit_price": f"{exit_px_m:.6f}",
                                "return_pct": f"{ret_m:.8f}" if math.isfinite(ret_m) else "",
                                "pnl_usd": f"{pnl_m:.6f}",
                                "exit_reason": reason_m,
                                "session": "midmorning",
                                "equity_after": f"{base_after_opening + cum_mm:.6f}",
                            }
                        )

            equity = equity_start_day + day_pnl_opening + day_pnl_mm
            if equity > peak:
                peak = equity
            elif peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("replay failed trade_date=%s", d.isoformat())
            error_days += 1
            exit_counts["REPLAY_ERROR"] += 1
            rows_out.append(
                {
                    "trade_date": d.isoformat(),
                    "symbol": "",
                    "rank": "",
                    "notional_usd": "",
                    "entry_price": "",
                    "exit_price": "",
                    "return_pct": "",
                    "pnl_usd": "",
                    "exit_reason": f"REPLAY_ERROR:{exc!r}",
                    "session": "",
                    "equity_after": f"{equity:.6f}",
                }
            )
        finally:
            dr = ((equity / equity_day_start) - 1.0) if equity_day_start > 0 else 0.0
            daily_returns.append(dr)
            day_pnls.append(equity - equity_day_start)

        d += timedelta(days=1)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trade_date",
                "symbol",
                "rank",
                "notional_usd",
                "entry_price",
                "exit_price",
                "return_pct",
                "pnl_usd",
                "exit_reason",
                "session",
                "equity_after",
            ],
        )
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    net_ret = (equity / float(starting_equity)) - 1.0 if starting_equity > 0 else float("nan")
    avg_ret = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
    wins = sum(1 for r in trade_returns if r > 0)
    win_rate = wins / len(trade_returns) if trade_returns else 0.0

    avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0
    wins_r = [r for r in trade_returns if r > 0]
    losses_r = [r for r in trade_returns if r < 0]
    avg_win_return = sum(wins_r) / len(wins_r) if wins_r else 0.0
    avg_loss_return = sum(losses_r) / len(losses_r) if losses_r else 0.0
    win_sum = sum(p for p in trade_pnls if p > 0)
    loss_sum = sum(p for p in trade_pnls if p < 0)
    profit_factor: float | None
    if loss_sum < 0:
        profit_factor = win_sum / abs(loss_sum)
    else:
        profit_factor = None
    slot2_trade_count = len(slot2_returns)
    slot2_avg_return = sum(slot2_returns) / len(slot2_returns) if slot2_returns else 0.0
    max_single_day_loss = min(day_pnls) if day_pnls else 0.0

    opening_trade_count = len(opening_trade_returns)
    midmorning_trade_count = len(midmorning_trade_returns)
    opening_avg_return = (
        sum(opening_trade_returns) / len(opening_trade_returns) if opening_trade_returns else 0.0
    )
    midmorning_avg_return = (
        sum(midmorning_trade_returns) / len(midmorning_trade_returns) if midmorning_trade_returns else 0.0
    )
    opening_win_rate = (
        sum(1 for r in opening_trade_returns if r > 0) / len(opening_trade_returns)
        if opening_trade_returns
        else 0.0
    )
    midmorning_win_rate = (
        sum(1 for r in midmorning_trade_returns if r > 0) / len(midmorning_trade_returns)
        if midmorning_trade_returns
        else 0.0
    )
    mm_win_sum = sum(p for p in midmorning_trade_pnls if p > 0)
    mm_loss_sum = sum(p for p in midmorning_trade_pnls if p < 0)
    midmorning_profit_factor: float | None
    if mm_loss_sum < 0:
        midmorning_profit_factor = mm_win_sum / abs(mm_loss_sum)
    else:
        midmorning_profit_factor = None

    opening_reject_json = dict(sorted(opening_reject_agg.items()))

    keyed_counts: dict[str, int] = {}
    for k in (
        "TAKE_PROFIT_HIT",
        "STOP_LOSS_HIT",
        "EOD_FLATTEN",
        "NO_TRADE",
        "SIM_DATA_MISSING",
        "REPLAY_ERROR",
    ):
        keyed_counts[k] = int(exit_counts.get(k, 0))
    for k, v in exit_counts.items():
        if k not in keyed_counts:
            keyed_counts[k] = int(v)

    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "starting_equity": starting_equity,
        "ending_equity": equity,
        "days_tested": days_tested,
        "trade_days": trade_days,
        "no_trade_days": no_trade_days,
        "error_days": error_days,
        "num_trades": n_trades,
        "win_rate": win_rate,
        "average_trade_return": avg_ret,
        "opening_trade_count": opening_trade_count,
        "opening_avg_return": opening_avg_return,
        "opening_win_rate": opening_win_rate,
        "opening_sim_data_missing_count": opening_sim_data_missing_count,
        "midmorning_trade_count": midmorning_trade_count,
        "midmorning_avg_return": midmorning_avg_return,
        "midmorning_win_rate": midmorning_win_rate,
        "midmorning_profit_factor": midmorning_profit_factor,
        "net_return": net_ret,
        "max_drawdown": max_dd,
        "deployable_fraction": float(PAPER_DEPLOYABLE_EQUITY_FRACTION),
        "avg_daily_return": avg_daily_return,
        "avg_win_return": avg_win_return,
        "avg_loss_return": avg_loss_return,
        "profit_factor": profit_factor,
        "slot2_trade_count": slot2_trade_count,
        "slot2_avg_return": slot2_avg_return,
        "slot2_reject_reason_counts": dict(sorted(slot2_reject_agg.items())),
        "slot1_full_sleeve_count": slot1_full_sleeve_count,
        "max_single_day_loss": max_single_day_loss,
        "exit_reason_counts": keyed_counts,
        "csv_path": str(csv_path.resolve()),
        "use_ai_cache": use_ai_cache,
        "opening_reject_reason_counts": opening_reject_json,
        "opening_surgical_pm_return_reject_total": opening_surgical_pm_reject_total,
        "opening_surgical_expected_move_reject_total": opening_surgical_em_reject_total,
        "soft_band_candidate_count": soft_band_candidate_total,
        "soft_band_expected_move_pass_count": soft_band_expected_move_pass_total,
        "soft_band_accepted_count": soft_band_accepted_total,
        "confidence_pass_direction_fail_count": confidence_pass_direction_fail_total,
        "confidence_direction_pass_volume_fail_count": confidence_direction_pass_volume_fail_total,
        "no_candidates_empty_days": no_candidates_empty_total,
        "decision_status_not_trade_days": decision_status_not_trade_total,
        "allocation_ready_plan_blocked_days": allocation_ready_plan_blocked_days,
        "opening_execution_plan_block_failure_counts": dict(sorted(opening_execution_plan_block_diag_agg.items())),
        "midmorning_skip_primary_reason_counts": dict(sorted(midmorning_skip_primary_reason_agg.items())),
        "opening_initial_no_trade_subtype_counts": dict(sorted(opening_initial_no_trade_subtype_agg.items())),
        "source_override_count": int(source_override_budget.get("used", 0)),
        "source_override_budget_max": int(source_override_budget.get("max", 0)),
        "enable_midmorning": enable_midmorning,
        "midmorning_skip_position_open_days": midmorning_skip_position_open_days,
        "midmorning_skip_filters_days": midmorning_skip_filters_days,
        "midmorning_plan_trade_days": midmorning_plan_trade_days,
        "opening_step2_not_ok_reason_counts": dict(sorted(opening_step2_not_ok_reason_agg.items())),
        "opening_step2_missing_or_invalid_fields_count": opening_step2_missing_or_invalid_fields_count,
        "opening_step2_watchlist_symbols_missing_or_invalid_total": opening_step2_watchlist_symbols_missing_or_invalid_total,
        "midmorning_sim_data_missing_count": midmorning_sim_data_missing_count,
        "debug_comparison_summary": {
            "num_trade_days": trade_days,
            "num_trades": n_trades,
            "sim_data_missing_dates": sorted(debug_sim_missing_dates_set),
            "source_override_dates": debug_source_override_dates,
            "no_trade_dates": debug_no_trade_dates,
            "allocation_blocked_dates": debug_allocation_blocked_dates,
            "step2_missing_dates": debug_step2_candidate_issue_dates,
        },
    }
    json_path = replay_out_dir / f"replay_summary_{start.isoformat()}_{end.isoformat()}.json"
    attrib_path = replay_out_dir / f"replay_opening_attribution_{start.isoformat()}_{end.isoformat()}.json"
    attrib_payload = analyze_opening_records(opening_attrib_records, starting_equity=float(starting_equity))
    write_opening_attribution_json(attrib_path, attrib_payload)

    summary["opening_attribution_report_path"] = str(attrib_path.resolve())
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["json_path"] = str(json_path.resolve())
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Opening replay simulation (no broker orders).")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--starting-equity", type=float, default=1000.0)
    p.add_argument(
        "--use-ai-cache",
        "--skip-ai-cache",
        action="store_true",
        help="Read/write var/replay/opening_ai_cache/<date>.json; skip Anthropic when that file exists.",
    )
    p.add_argument(
        "--no-midmorning",
        dest="midmorning",
        action="store_false",
        help="Opening-only replay: skip ~10:30 ET mid-morning simulation.",
    )
    p.set_defaults(midmorning=True)
    args = p.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        _die("--end must be on or after --start")

    use_ai_cache = bool(args.use_ai_cache)
    need_ai = not use_ai_cache
    settings = Settings.from_env()
    _require_env_replay(need_anthropic=need_ai)

    repo = Path(__file__).resolve().parents[2]
    cache_dir = repo / "var" / "replay" / "opening_ai_cache"
    out_dir = repo / "var" / "replay"

    settings = replace(settings, dry_run=True, enable_premarket_signals=True)
    strat = StrategyEngine()

    summary = run_opening_replay_range(
        start,
        end,
        starting_equity=float(args.starting_equity),
        settings=settings,
        strategy=strat,
        use_ai_cache=use_ai_cache,
        cache_dir=cache_dir,
        replay_out_dir=out_dir,
        enable_midmorning=bool(args.midmorning),
    )
    print(json.dumps(summary, indent=2, default=str))
    _print_opening_diagnostic_summary(summary)
    ap = summary.get("opening_attribution_report_path")
    if ap:
        try:
            print_opening_attribution_summary(
                json.loads(Path(ap).read_text(encoding="utf-8")),
                max_symbol_rows=45,
            )
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("opening attribution report read failed path=%s err=%s", ap, exc)


if __name__ == "__main__":
    main()
