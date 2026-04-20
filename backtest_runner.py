#!/usr/bin/env python3
"""
Multi-day simulation: run the existing daily pipeline many times without changing core logic.

How it works:
  1. Forces STOCKBOT_DRY_RUN=true so no real broker orders are sent.
  2. For each calendar day in a range, calls stockbot.pipeline.run_daily_pipeline(trade_date=...).
  3. Writes JSON with per-day digest + PnL summary; CSV lists **executed trades only** (one row
     per successful execution, so two fills on the same decision day appear as two rows) with
     next-session open → same-day close (no lookahead on the decision day). Each row includes
     ``position_weight`` (1.0 for a single pick; two picks: 0.7 for slot 1 and 0.30 / 0.25 / 0.20
     for slot 2 from raw score gap vs slot 1 plus an absolute slot-2 floor — see
     ``stockbot.execution.orders``). ``return_pct`` is raw; ``pnl_summary.total_return``
     compounds **daily** blended ``w1*r1 + w2*r2``.

Run from the project root, for example:
  python backtest_runner.py --days 21
  python backtest_runner.py --start 2026-04-01 --end 2026-04-18
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any
import csv
import json
import logging
import math
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

# Simulation must never place live orders — set before Settings / pipeline load.
os.environ["STOCKBOT_DRY_RUN"] = "true"

from stockbot.config import Settings
from stockbot.execution.orders import (
    OVERLAP_PAIR_ELITE_MAX_SCORE_GAP,
    capital_fraction_for_slot,
    get_overlap_capped_slot2_to_25_count,
    get_overlap_elite_full_sleeve_count,
    get_overlap_two_leg_sessions,
    reset_overlap_slot2_stats,
)
from stockbot.ingestion.market import fetch_market_snapshots
from stockbot.models import DailyReasoningRecord
from stockbot.pipeline import run_daily_pipeline


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _daterange_inclusive(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _next_trading_day(d: date) -> date:
    """First U.S.-style weekday session after *d* (Sat/Sun skipped)."""
    cur = d + timedelta(days=1)
    while cur.weekday() >= 5:
        cur += timedelta(days=1)
    return cur


def _open_close_on_session_day(
    symbol: str,
    session_day: date,
    settings: Settings,
    log: logging.Logger,
) -> tuple[float, float, dict[str, object]] | None:
    """
    Same-day open → close from **real** daily bars only (``allow_synthetic=False``).

    If Alpaca data is missing or fails validation, returns None and logs loudly — no toy OHLC.
    """
    sym = symbol.upper()
    as_of = datetime.combine(session_day, time(20, 0, 0), tzinfo=timezone.utc)
    market, meta = fetch_market_snapshots([sym], settings, as_of=as_of, allow_synthetic=False)
    pm: dict[str, Any] = dict(meta.get("per_symbol", {}).get(sym, {}))
    raw_src = str(pm.get("source") or "none")
    if raw_src.startswith("synthetic") or raw_src == "synthetic_disabled":
        data_lineage = "synthetic"
    elif raw_src == "alpaca" and bool(pm.get("real_data_valid")):
        data_lineage = "alpaca"
    else:
        data_lineage = "none"

    detail: dict[str, object] = {
        "data_source_used": raw_src,
        "real_data_valid": False,
        "bar_count": int(pm.get("bar_count") or 0),
        "first_bar_ts": pm.get("first_bar_ts"),
        "last_bar_ts": pm.get("last_bar_ts"),
    }

    snap = market.get(sym)
    if snap is None or snap.bars is None or snap.bars.empty:
        log.warning(
            "REAL DATA MISSING - TRADE SKIPPED: %s on %s (reason=%s)",
            sym,
            session_day,
            pm.get("skip_reason", "no_snapshot"),
        )
        log.info(
            "[backtest OHLC] symbol=%s session_day=%s data_lineage=%s raw_source=%s bars=%s first=%s last=%s "
            "series_valid=%s session_bar=missing trade=skipped",
            sym,
            session_day,
            data_lineage,
            raw_src,
            detail["bar_count"],
            detail["first_bar_ts"],
            detail["last_bar_ts"],
            pm.get("real_data_valid"),
        )
        return None
    bars = snap.bars
    dts = pd.to_datetime(bars.index, utc=True)
    for i, ts in enumerate(dts):
        if ts.date() == session_day:
            row = bars.iloc[i]
            o = float(row["open"])
            c = float(row["close"])
            detail["real_data_valid"] = True
            log.info(
                "[backtest OHLC] symbol=%s session_day=%s data_lineage=alpaca raw_source=%s bars=%s first=%s last=%s "
                "series_valid=%s session_bar=found trade=priced open=%.6f close=%.6f",
                sym,
                session_day,
                raw_src,
                len(bars),
                detail["first_bar_ts"],
                detail["last_bar_ts"],
                True,
                o,
                c,
            )
            return o, c, detail
    log.warning(
        "REAL DATA MISSING - TRADE SKIPPED: no bar row for %s on session %s (index range issue)",
        sym,
        session_day,
    )
    log.info(
        "[backtest OHLC] symbol=%s session_day=%s data_lineage=%s raw_source=%s bars=%s first=%s last=%s "
        "series_valid=%s session_bar=missing trade=skipped",
        sym,
        session_day,
        data_lineage,
        raw_src,
        len(bars),
        str(bars.index[0]) if len(bars) else None,
        str(bars.index[-1]) if len(bars) else None,
        pm.get("real_data_valid"),
    )
    return None


def summarize_day(record: DailyReasoningRecord) -> dict[str, object]:
    """Pull a small set of fields from the audit record (no scoring changes)."""
    strat = record.strategy or {}
    ranked = strat.get("ranked") or []
    top_score: float | None = None
    if ranked:
        top_score = float(ranked[0]["score"])

    chosen = strat.get("chosen")
    reason_codes = list(strat.get("reason_codes") or [])

    if record.inputs_trace.get("note") == "weekend_skip":
        reason_codes = ["WEEKEND_SKIP"] if not reason_codes else reason_codes

    executions = list(record.executions or [])
    trade_made = len(executions) > 0
    chosen_symbols: list[str]
    if chosen is None:
        chosen_symbols = []
    elif isinstance(chosen, str):
        chosen_symbols = [chosen]
    else:
        chosen_symbols = [str(x) for x in chosen]

    broker_ids = [str(x.get("broker_order_id") or "") for x in executions if x.get("success")]

    return {
        "date": record.trade_date,
        "chosen_symbols": chosen_symbols,
        "top_score": top_score,
        "trade_made": trade_made,
        "reason_codes": reason_codes,
        "dry_run": bool(record.meta.get("dry_run", True)),
        "broker_order_ids": broker_ids,
    }


def _pnl_summary(blended_daily_returns: list[float], raw_trade_returns: list[float]) -> dict[str, float]:
    """
    total_return / win_rate use **blended daily** returns: per calendar trade_date,
    sum(position_weight * return_pct) so two fills on one day count as one portfolio step.

    avg_return_per_trade is the unweighted mean of raw per-fill return_pct (audit / per-name view).
    """
    if not blended_daily_returns:
        return {
            "total_return": 0.0,
            "win_rate": 0.0,
            "avg_return_per_trade": 0.0,
            "avg_blended_day_return": 0.0,
            "trade_count": 0.0,
            "pnl_day_count": 0.0,
        }
    wins = sum(1 for r in blended_daily_returns if r > 0)
    compound = math.prod(1.0 + r for r in blended_daily_returns) - 1.0
    return {
        "total_return": round(compound, 6),
        "win_rate": round(wins / len(blended_daily_returns), 6),
        "avg_return_per_trade": round(
            sum(raw_trade_returns) / len(raw_trade_returns), 6
        )
        if raw_trade_returns
        else 0.0,
        "avg_blended_day_return": round(
            sum(blended_daily_returns) / len(blended_daily_returns), 6
        ),
        "trade_count": float(len(raw_trade_returns)),
        "pnl_day_count": float(len(blended_daily_returns)),
    }


def _blended_daily_returns(trade_rows: list[dict[str, Any]]) -> list[float]:
    """One value per trade_date: sum(weight * raw return) for that session's rows."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trade_rows:
        by_day[str(t["trade_date"])].append(t)
    out: list[float] = []
    for day in sorted(by_day.keys()):
        rows = by_day[day]
        out.append(
            sum(float(r["position_weight"]) * float(r["return_pct"]) for r in rows)
        )
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("backtest_runner")

    parser = argparse.ArgumentParser(
        description="Run run_daily_pipeline over a date range (dry-run only) and save results.",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="First date (YYYY-MM-DD). Default: end minus --days",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="Last date inclusive (YYYY-MM-DD). Default: today (UTC)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=21,
        help="If --start is omitted, go back this many days from --end (default: 21)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("var/backtest"),
        help="Directory for JSON/CSV output (default: var/backtest)",
    )
    args = parser.parse_args()

    end = args.end or datetime.now(timezone.utc).date()
    if args.start is not None:
        start = args.start
    else:
        start = end - timedelta(days=max(1, args.days) - 1)

    dates = _daterange_inclusive(start, end)
    log.info("Simulating %d days from %s to %s (STOCKBOT_DRY_RUN forced on)", len(dates), start, end)

    prev_real = os.environ.get("STOCKBOT_REQUIRE_REAL_MARKET")
    os.environ["STOCKBOT_REQUIRE_REAL_MARKET"] = "true"

    settings = Settings.from_env()
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    reset_overlap_slot2_stats()
    try:
        for d in dates:
            record = run_daily_pipeline(trade_date=d)
            row = summarize_day(record)
            daily_rows.append(row)
            log.info(
                "%s chosen=%s top_score=%s trade_made=%s reasons=%s",
                row["date"],
                row["chosen_symbols"],
                row["top_score"],
                row["trade_made"],
                row["reason_codes"],
            )

            if not row["trade_made"]:
                continue

            strat = record.strategy or {}
            chosen_syms: list[str]
            ch = strat.get("chosen")
            if ch is None:
                chosen_syms = []
            elif isinstance(ch, str):
                chosen_syms = [ch]
            else:
                chosen_syms = [str(x) for x in ch]

            exec_day = _next_trading_day(d)
            for ex in record.executions or []:
                if not ex.get("success"):
                    continue
                symbol = str(ex.get("symbol") or "")
                if not symbol:
                    continue
                pw = ex.get("position_weight")
                if pw is None:
                    slot = chosen_syms.index(symbol) if symbol in chosen_syms else 0
                    pw = capital_fraction_for_slot(slot, len(chosen_syms))
                position_weight = float(pw)

                ohlc = _open_close_on_session_day(symbol, exec_day, settings, log)
                if ohlc is None:
                    continue

                entry_px, exit_px, detail = ohlc
                if entry_px <= 0:
                    log.warning("PnL skip: non-positive open for %s on %s", symbol, exec_day)
                    continue

                ret_pct = (exit_px - entry_px) / entry_px
                trade_rows.append(
                    {
                        "trade_date": d.isoformat(),
                        "execution_date": exec_day.isoformat(),
                        "symbol": symbol,
                        "entry_price": round(entry_px, 6),
                        "exit_price": round(exit_px, 6),
                        "return_pct": round(ret_pct, 6),
                        "position_weight": round(position_weight, 4),
                        "data_source_used": detail["data_source_used"],
                        "real_data_valid": detail["real_data_valid"],
                        "bar_count": detail["bar_count"],
                        "first_bar_ts": detail["first_bar_ts"],
                        "last_bar_ts": detail["last_bar_ts"],
                    }
                )
                log.info(
                    "PnL trade trade_date=%s execution_date=%s %s open=%.4f close=%.4f return_pct=%.4f",
                    d,
                    exec_day,
                    symbol,
                    entry_px,
                    exit_px,
                    ret_pct,
                )
    finally:
        if prev_real is None:
            os.environ.pop("STOCKBOT_REQUIRE_REAL_MARKET", None)
        else:
            os.environ["STOCKBOT_REQUIRE_REAL_MARKET"] = prev_real

    raw_returns = [float(t["return_pct"]) for t in trade_rows]
    blended_daily = _blended_daily_returns(trade_rows)
    pnl_summary = _pnl_summary(blended_daily, raw_returns)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"simulation_{stamp}.json"
    csv_trades_path = args.out_dir / f"simulation_{stamp}_trades.csv"

    payload = {
        "meta": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "day_count": len(daily_rows),
            "dry_run_forced": True,
            "require_real_market": True,
            "position_sizing": {
                "trade_slot_1_weight": 0.7,
                "trade_slot_2_weight": "0.30_default_0.25_0.20_when_weak_gap_and_low_abs_score",
                "single_trade_weight": 1.0,
                "pnl_total_return_basis": "compounded_daily_blended_weight_times_raw_return",
            },
            "overlap_slot2_sizing": {
                "overlap_two_leg_sessions": int(get_overlap_two_leg_sessions()),
                "overlap_elite_full_sleeve_count": int(get_overlap_elite_full_sleeve_count()),
                "overlap_capped_0_30_to_0_25_from_30_count": int(
                    get_overlap_capped_slot2_to_25_count()
                ),
                "overlap_elite_max_score_gap": float(OVERLAP_PAIR_ELITE_MAX_SCORE_GAP),
            },
        },
        "daily_rows": daily_rows,
        "trades": trade_rows,
        "pnl_summary": pnl_summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    trade_fields = [
        "trade_date",
        "execution_date",
        "symbol",
        "entry_price",
        "exit_price",
        "return_pct",
        "position_weight",
        "data_source_used",
        "real_data_valid",
        "bar_count",
        "first_bar_ts",
        "last_bar_ts",
    ]
    with csv_trades_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=trade_fields)
        w.writeheader()
        for t in trade_rows:
            w.writerow(t)

    log.info("PnL summary: %s", pnl_summary)
    log.info(
        "meta.overlap_slot2_sizing "
        "overlap_two_leg_sessions=%d "
        "overlap_elite_full_sleeve_count=%d "
        "overlap_capped_0_30_to_0_25_from_30_count=%d "
        "overlap_elite_max_score_gap=%.2f",
        int(get_overlap_two_leg_sessions()),
        int(get_overlap_elite_full_sleeve_count()),
        int(get_overlap_capped_slot2_to_25_count()),
        float(OVERLAP_PAIR_ELITE_MAX_SCORE_GAP),
    )
    log.info("Wrote %s", json_path.resolve())
    log.info("Wrote %s (executed trades only)", csv_trades_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
