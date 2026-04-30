"""
Paper-trading morning orchestration: wires Steps 1 → 6 without changing their logic.

Design: start this process *before* 09:30 ET so Steps 1–5.5 finish first; Step 6 runs at the open
(09:30:00 America/New_York) after a deterministic wait.

Requires env (see main). Paper trading host only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time as time_module
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from stockbot.ai.anthropic_messages_client import anthropic_messages_text, resolve_opening_decision_model
from stockbot.ai.opening_decision_prompt import OPENING_DECISION_SYSTEM_PROMPT, build_opening_decision_user_content
from stockbot.ai.opening_decision_validate import validate_opening_decision_response_detailed
from stockbot.config import Settings
from stockbot.execution.alpaca_opening_submit import AlpacaHttpOpeningClient, submit_opening_execution_plan
from stockbot.execution.broker import AlpacaBroker
from stockbot.execution.midmorning_sector_strategy import (
    build_midmorning_step2_packet,
    deterministic_midmorning_confidence,
    select_midmorning_long,
)
from stockbot.execution.opening_allocation import OpeningAllocationConfig, build_step2_index, prepare_opening_execution
from stockbot.execution.opening_source_override import try_source_no_trade_rank1_override
from stockbot.execution.opening_execution_plan import build_opening_execution_plan
from stockbot.execution.opening_execution_plan_validate import validate_opening_execution_plan
from stockbot.execution.paper_deployable import deployable_usd_from_alpaca_broker
from stockbot.ingestion.market import fetch_market_snapshots
from stockbot.ingestion.premarket import fetch_premarket_for_watchlist
from stockbot.ingestion.premarket_packet import build_ai_premarket_packet
from stockbot.ingestion.premarket_wait import wait_until_premarket_decision_et
from stockbot.models import MarketSnapshot
from stockbot.runners.managed_position_ledger import (
    SqliteManagedPositionLedger,
    default_managed_position_ledger_path,
    record_submitted_opening_buys,
)
from stockbot.runners.persistent_opening_idempotency import SqliteOpeningIdempotencyStore
from stockbot.strategy.engine import StrategyEngine

_LOG = logging.getLogger("stockbot.runners.paper_open_run")

_ET = ZoneInfo("America/New_York")


def _paper_diag_emit(title: str, payload: Any) -> None:
    """Diagnostic only: stderr + log marker (does not affect trading)."""
    print(f"\n=== paper_open_run diagnostic: {title} ===", file=sys.stderr, flush=True)
    print(json.dumps(payload, indent=2, default=str), file=sys.stderr, flush=True)
    _LOG.info("paper_open_run diagnostic section: %s", title)


def _paper_diag_step2_summary(step2_packet: Mapping[str, Any]) -> dict[str, Any]:
    syms_raw = step2_packet.get("symbols")
    rows_in: list[Any] = list(syms_raw) if isinstance(syms_raw, list) else []
    per_symbol: list[dict[str, Any]] = []
    ok_count = 0
    for r in rows_in:
        if not isinstance(r, Mapping):
            continue
        st = r.get("status")
        if st == "ok":
            ok_count += 1
        per_symbol.append(
            {
                "symbol": r.get("symbol"),
                "status": st,
                "reason": r.get("reason"),
                "pm_volume": r.get("pm_volume"),
                "pm_session_return_pct": r.get("pm_session_return_pct"),
                "pm_close_position_in_range": r.get("pm_close_position_in_range"),
                "gap_close_vs_prior_close_pct": r.get("gap_close_vs_prior_close_pct"),
            }
        )
    return {
        "trade_date": step2_packet.get("trade_date"),
        "symbol_count": len(per_symbol),
        "ok_status_count": ok_count,
        "symbols": per_symbol,
    }


def _paper_diag_step4_focus(step4_allocation: Mapping[str, Any]) -> dict[str, Any]:
    rej = step4_allocation.get("rejected")
    reason_codes: list[Any] = []
    if isinstance(rej, list):
        for item in rej:
            if isinstance(item, Mapping):
                reason_codes.append(item.get("reason_code"))
    return {
        "preparation_status": step4_allocation.get("preparation_status"),
        "trades": step4_allocation.get("trades"),
        "rejected_reason_codes": reason_codes,
        "rejected": step4_allocation.get("rejected"),
    }


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _open_no_wait() -> bool:
    v = os.environ.get("STOCKBOT_OPEN_NO_WAIT", "")
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _midmorning_no_wait() -> bool:
    v = os.environ.get("STOCKBOT_MIDMORNING_NO_WAIT", "")
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _wait_until_midmorning_et(trade_date: date) -> None:
    """Sleep until 10:30:00 America/New_York when ``trade_date`` is today; skipped for historical dates."""
    now_et = datetime.now(_ET)
    if now_et.date() != trade_date:
        return
    if _midmorning_no_wait():
        _LOG.info("STOCKBOT_MIDMORNING_NO_WAIT set; skipping wait until 10:30 ET.")
        return
    if _open_no_wait():
        _LOG.info("STOCKBOT_OPEN_NO_WAIT set; skipping wait until 10:30 ET (mid-morning).")
        return
    target = datetime.combine(trade_date, time(10, 30, 0), tzinfo=_ET)
    if now_et >= target:
        _LOG.warning(
            "Current ET time already at or after mid-morning target %s (now=%s); continuing.",
            target.isoformat(),
            now_et.isoformat(),
        )
        return
    delay = (target - now_et).total_seconds()
    _LOG.info("Waiting %.1f s until mid-morning %s ET.", delay, target.isoformat())
    time_module.sleep(delay)


def _wait_until_market_open_et(trade_date: date) -> None:
    """
    After Steps 1–5.5, sleep until 09:30:00 America/New_York on trade_date (regular session open).

    Skipped when trade_date is not \"today\" in ET (historical runs) or STOCKBOT_OPEN_NO_WAIT=1.
    If the process is already at or past the open instant, returns immediately (no negative sleep).
    """
    now_et = datetime.now(_ET)
    if now_et.date() != trade_date:
        return
    if _open_no_wait():
        _LOG.info("STOCKBOT_OPEN_NO_WAIT set; skipping wait until 09:30 ET.")
        return
    target = datetime.combine(trade_date, time(9, 30, 0), tzinfo=_ET)
    if now_et >= target:
        _LOG.warning(
            "Current ET time is already at or after 09:30 (now=%s); submitting without further delay.",
            now_et.isoformat(),
        )
        return
    delay = (target - now_et).total_seconds()
    _LOG.info("Waiting %.1f s until market open %s (Step 6 follows).", delay, target.isoformat())
    time_module.sleep(delay)


def _require_env() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    sec = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    trading = os.environ.get("ALPACA_TRADING_BASE_URL", "").strip().rstrip("/")
    data = os.environ.get("ALPACA_DATA_BASE_URL", "").strip().rstrip("/")
    missing = [
        n
        for n, v in (
            ("ALPACA_API_KEY", key),
            ("ALPACA_SECRET_KEY", sec),
            ("ALPACA_TRADING_BASE_URL", trading),
            ("ALPACA_DATA_BASE_URL", data),
        )
        if not v
    ]
    if missing:
        _die(f"Missing required environment variables: {', '.join(missing)}. Not submitting.")
    p = urlparse(trading)
    if p.scheme != "https" or p.hostname != "paper-api.alpaca.markets":
        _die(
            f"ALPACA_TRADING_BASE_URL must be https://paper-api.alpaca.markets (got host={p.hostname!r}). "
            "Not submitting."
        )
    d = urlparse(data)
    if d.scheme != "https" or d.hostname != "data.alpaca.markets":
        _die(
            f"ALPACA_DATA_BASE_URL must be https://data.alpaca.markets (got host={d.hostname!r}). "
            "Not submitting."
        )
    anth = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not anth:
        _die("Missing ANTHROPIC_API_KEY (required for opening decision). Not submitting.")
    return trading, data


def _settings_for_paper(trading_base_url: str) -> Settings:
    """Point Alpaca REST at paper trading; disable dry_run so equity is read from Alpaca."""
    os.environ["ALPACA_BASE_URL"] = trading_base_url
    base = Settings.from_env()
    return replace(base, dry_run=False)


def _dedupe_symbols(symbols: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in symbols:
        u = str(s).strip().upper()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _last_close_by_symbol(market: dict[str, MarketSnapshot | None]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, snap in market.items():
        if snap is None or snap.bars is None or snap.bars.empty:
            continue
        if "close" not in snap.bars.columns:
            continue
        last = float(snap.bars["close"].astype(float).iloc[-1])
        if math.isfinite(last) and last > 0:
            out[str(k).upper()] = last
    return out


def _extract_json_object(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model response")
    return m.group(0)


def _anthropic_decision_json_raw(
    *,
    anthropic_api_key: str,
    packet: dict[str, Any],
    watchlist: list[str],
    expected_trade_date: str,
    build_user_content: Callable[..., str],
) -> str:
    packet_json = json.dumps(packet, default=str)
    user = build_user_content(
        packet_json=packet_json,
        expected_trade_date=expected_trade_date,
        allowed_symbols=watchlist,
    )
    model = resolve_opening_decision_model()
    raw = anthropic_messages_text(
        api_key=anthropic_api_key,
        system=OPENING_DECISION_SYSTEM_PROMPT,
        user_text=user,
        model=model,
        max_tokens=2048,
    )
    return _extract_json_object(raw)


def _anthropic_opening_raw(
    *,
    anthropic_api_key: str,
    packet: dict[str, Any],
    watchlist: list[str],
    expected_trade_date: str,
) -> str:
    return _anthropic_decision_json_raw(
        anthropic_api_key=anthropic_api_key,
        packet=packet,
        watchlist=watchlist,
        expected_trade_date=expected_trade_date,
        build_user_content=build_opening_decision_user_content,
    )


def _default_idempotency_path() -> Path:
    raw = os.environ.get("STOCKBOT_OPENING_IDEMPOTENCY_DB", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("var/state/opening_idempotency.sqlite3").resolve()


def _paper_would_submit_summary(validated_plan: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    if validated_plan.get("execution_status") == "no_execution":
        return False, []
    instr = validated_plan.get("instructions")
    if not isinstance(instr, list) or len(instr) == 0:
        return False, []
    would_orders: list[dict[str, Any]] = []
    for inst in instr:
        if not isinstance(inst, Mapping):
            continue
        would_orders.append(
            {
                "symbol": inst.get("symbol"),
                "side": inst.get("side"),
                "mode": inst.get("mode"),
                "notional_usd": inst.get("notional_usd"),
                "shares": inst.get("shares"),
                "rank": inst.get("rank"),
                "scheduled_for": inst.get("scheduled_for"),
            }
        )
    return True, would_orders


def run_paper_opening_through_5_5(
    trade_date: date,
    *,
    settings: Settings,
    strategy: StrategyEngine,
    deployable_usd_override: float | None = None,
    opening_decision_raw_text: str | None = None,
    allocation_diagnostics: bool = False,
    source_no_trade_override_budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Steps 1 → 5.5 only (same ingestion, AI, allocation, planning, validation as paper morning run).

    ``deployable_usd_override``: when set (replay / tests), skip Alpaca equity read and use this value.
    ``opening_decision_raw_text``: when set, skip Anthropic and validate this JSON text as Step 3 output.
    ``allocation_diagnostics``: when True, attach per-day Step 4 diagnostics under ``step4_allocation``.
    ``source_no_trade_override_budget``: mutable ``{"max": N, "used": n}`` enables capped rank-1-only
    recovery when Step 3 returns ``no_trade`` (replay only; omit for production paper).
    """
    watchlist = [s.upper() for s in strategy.watchlist]
    if not watchlist:
        _die("Strategy watchlist is empty. Not submitting.")

    step1_symbols = _dedupe_symbols(list(watchlist) + ["SPY", "QQQ"])

    wait_until_premarket_decision_et(trade_date, settings)

    step1_by_symbol = fetch_premarket_for_watchlist(settings, trade_date, step1_symbols)

    ingest_as_of = datetime.combine(trade_date, time(20, 0, 0, tzinfo=timezone.utc))
    market, _market_meta = fetch_market_snapshots(
        step1_symbols,
        settings,
        as_of=ingest_as_of,
        allow_synthetic=None,
    )

    step2_packet = build_ai_premarket_packet(
        trade_date,
        watchlist,
        step1_by_symbol,
        {k: market.get(k) for k in watchlist},
    )
    _paper_diag_emit("Step 2 summary", _paper_diag_step2_summary(step2_packet))

    expected_td = str(step2_packet.get("trade_date", trade_date.isoformat()))
    if opening_decision_raw_text is not None:
        raw_text = opening_decision_raw_text
    else:
        raw_text = _anthropic_opening_raw(
            anthropic_api_key=settings.anthropic_api_key,
            packet=step2_packet,
            watchlist=watchlist,
            expected_trade_date=expected_td,
        )
    validated_decision, opening_meta = validate_opening_decision_response_detailed(
        raw_text,
        expected_trade_date=expected_td,
        allowed_symbols=watchlist,
    )
    override_applied = False
    override_detail: dict[str, Any] = {}
    if source_no_trade_override_budget is not None and opening_meta.get(
        "initial_decision_status"
    ) == "no_trade":
        step2_by_override = build_step2_index(step2_packet)
        mr = validated_decision.get("market_read")
        mr_summary = str(mr.get("summary")) if isinstance(mr, Mapping) else None
        synth, override_detail = try_source_no_trade_rank1_override(
            raw_model_text=raw_text,
            opening_meta=opening_meta,
            trade_date_str=expected_td,
            watchlist=watchlist,
            step2_by_symbol=step2_by_override,
            budget=source_no_trade_override_budget,
            config=OpeningAllocationConfig(),
            market_read_summary=mr_summary,
        )
        if synth is not None:
            validated_decision = synth
            override_applied = True

    opening_decision_meta: dict[str, Any] = {
        **opening_meta,
        "source_override_applied": override_applied,
        "source_override_detail": override_detail,
    }
    _paper_diag_emit("Step 3.5 validated_decision", dict(validated_decision))

    step4_allocation = prepare_opening_execution(
        validated_decision,
        step2_packet=step2_packet,
        allocation_diagnostics=allocation_diagnostics,
    )
    _paper_diag_emit(
        "Step 4 allocation",
        {
            "step4_allocation": step4_allocation,
            "focus": _paper_diag_step4_focus(step4_allocation),
        },
    )

    if deployable_usd_override is not None:
        deployable_usd = float(deployable_usd_override)
    else:
        broker = AlpacaBroker(settings)
        try:
            deployable_usd = deployable_usd_from_alpaca_broker(broker)
        except Exception as exc:  # noqa: BLE001
            _die(f"Could not read account equity from Alpaca paper: {exc!r}. Not submitting.")

    if not math.isfinite(deployable_usd) or deployable_usd <= 0:
        _die(f"Invalid deployable_usd={deployable_usd!r} from equity. Not submitting.")

    ref_prices = _last_close_by_symbol({k: market.get(k) for k in watchlist})

    plan = build_opening_execution_plan(
        step4_allocation,
        deployable_usd=deployable_usd,
        reference_prices=ref_prices,
    )
    _paper_diag_emit("Step 5 execution plan (before Step 5.5 validation)", plan)
    validated_plan = validate_opening_execution_plan(plan)
    _paper_diag_emit("Step 5.5 validated_plan", dict(validated_plan))

    return {
        "trade_date": trade_date,
        "watchlist": watchlist,
        "step1_symbols": step1_symbols,
        "step1_by_symbol": step1_by_symbol,
        "market": market,
        "step2_packet": step2_packet,
        "opening_decision_raw_text": raw_text,
        "validated_decision": validated_decision,
        "step4_allocation": step4_allocation,
        "deployable_usd": deployable_usd,
        "ref_prices": ref_prices,
        "plan": plan,
        "validated_plan": validated_plan,
        "opening_decision_meta": opening_decision_meta,
    }


def _opening_buy_symbols_submitted(submission: Mapping[str, Any] | None) -> set[str]:
    out: set[str] = set()
    if submission is None:
        return out
    orders = submission.get("orders")
    if not isinstance(orders, list):
        return out
    for o in orders:
        if not isinstance(o, Mapping):
            continue
        if o.get("status") != "submitted":
            continue
        sym = str(o.get("symbol") or "").strip().upper()
        if sym:
            out.add(sym)
    return out


def run_paper_midmorning_through_5_5(
    trade_date: date,
    *,
    settings: Settings,
    strategy: StrategyEngine,
    deployable_usd_override: float | None = None,
    allocation_diagnostics: bool = False,
) -> dict[str, Any]:
    """
    Mid-morning (~10:30 ET): sector leadership + RTH relative strength → Steps 4–5.5 only.

    Does not use opening allocation, Anthropic, or premarket ingestion. Caller runs Step 6 after
    the 10:30 wait and overlap checks.
    """
    _LOG.info("MIDMORNING_PIPELINE_START trade_date=%s", trade_date.isoformat())
    watchlist = [s.upper() for s in strategy.watchlist]
    if not watchlist:
        _die("Strategy watchlist is empty. Not submitting.")

    sel, pick_stats = select_midmorning_long(trade_date, settings, watchlist)
    td_str = trade_date.isoformat()

    if deployable_usd_override is not None:
        deployable_usd = float(deployable_usd_override)
    else:
        broker = AlpacaBroker(settings)
        try:
            deployable_usd = deployable_usd_from_alpaca_broker(broker)
        except Exception as exc:  # noqa: BLE001
            _die(f"Could not read account equity from Alpaca paper: {exc!r}. Not submitting.")

    if not math.isfinite(deployable_usd) or deployable_usd <= 0:
        _die(f"Invalid deployable_usd={deployable_usd!r} from equity. Not submitting.")

    pick_sym = sel.midmorning_candidate_symbol if sel.midmorning_filter_pass else None
    step2_packet = build_midmorning_step2_packet(trade_date, pick_sym, pick_stats)

    ref_prices: dict[str, float] = {}
    if pick_sym and pick_stats is not None and pick_stats.price_close is not None:
        pc = float(pick_stats.price_close)
        if math.isfinite(pc) and pc > 0:
            ref_prices[pick_sym] = pc

    mm_refs = dict(ref_prices)

    conf = (
        deterministic_midmorning_confidence(float(sel.midmorning_relative_strength_vs_spy or 0.0))
        if sel.midmorning_filter_pass and pick_sym
        else 0.0
    )

    if sel.midmorning_filter_pass and pick_sym:
        validated_decision = {
            "trade_date": td_str,
            "decision_status": "trade",
            "market_read": {"summary": "midmorning_sector_leadership_rs"},
            "candidates": [
                {
                    "rank": 1,
                    "symbol": pick_sym,
                    "direction": "long",
                    "confidence": conf,
                    "reason": "sector_rs_rank1",
                }
            ],
        }
        step4_allocation: dict[str, Any] = {
            "trade_date": td_str,
            "preparation_status": "ready",
            "source_decision_status": "trade",
            "accepted_count": 1,
            "trades": [
                {
                    "ai_rank": 1,
                    "symbol": pick_sym,
                    "direction": "long",
                    "ai_confidence": conf,
                    "capital_weight": 1.0,
                    "included": True,
                    "notes": "MIDMORNING_SECTOR_RS",
                }
            ],
            "weights": [1.0],
            "rejected": [],
        }
    else:
        validated_decision = {
            "trade_date": td_str,
            "decision_status": "no_trade",
            "market_read": {"summary": sel.midmorning_skip_reason or "midmorning_skip"},
            "candidates": [],
        }
        step4_allocation = {
            "trade_date": td_str,
            "preparation_status": "no_trades",
            "source_decision_status": "no_trade",
            "accepted_count": 0,
            "trades": [],
            "weights": [],
            "rejected": [
                {
                    "ai_rank": 0,
                    "symbol": "",
                    "reason_code": "MIDMORNING_SKIP",
                    "detail": sel.midmorning_skip_reason or "",
                }
            ],
        }

    if allocation_diagnostics:
        step4_allocation = dict(step4_allocation)
        step4_allocation["allocation_diagnostics"] = {"midmorning_sector_rs": True}

    plan = build_opening_execution_plan(
        step4_allocation,
        deployable_usd=deployable_usd,
        reference_prices=ref_prices,
        scheduled_session_open_et=time(10, 30),
    )
    validated_plan = validate_opening_execution_plan(plan)

    opening_decision_meta = {
        "session_phase": "midmorning",
        "strategy": "sector_leadership_rs",
        "initial_decision_status": validated_decision.get("decision_status"),
        "midmorning_selection": sel.log_fields(),
    }

    _paper_diag_emit(
        "Mid-morning sector RS selection",
        {"selection": sel.log_fields(), "step4_allocation": step4_allocation},
    )

    ds = str(validated_decision.get("decision_status") or "")
    _LOG.info(
        "MIDMORNING_DECISION decision=%s symbol=%s midmorning_filter_pass=%s midmorning_skip_reason=%s",
        ds,
        pick_sym or "",
        sel.midmorning_filter_pass,
        sel.midmorning_skip_reason or "",
    )

    if (
        step4_allocation.get("preparation_status") != "ready"
        or validated_plan.get("execution_status") == "no_execution"
    ):
        _LOG.info(
            "MIDMORNING_SKIP reason=filters_failed preparation_status=%s execution_status=%s",
            step4_allocation.get("preparation_status"),
            validated_plan.get("execution_status"),
        )
    else:
        trades_pm = step4_allocation.get("trades")
        if isinstance(trades_pm, list):
            for tr in trades_pm:
                if not isinstance(tr, Mapping):
                    continue
                if tr.get("included") is True:
                    _LOG.info(
                        "MIDMORNING_EXECUTION symbol=%s weight=%s",
                        str(tr.get("symbol") or "").strip().upper(),
                        tr.get("capital_weight"),
                    )

    return {
        "trade_date": trade_date,
        "watchlist": watchlist,
        "step2_packet": step2_packet,
        "validated_decision": validated_decision,
        "step4_allocation": step4_allocation,
        "deployable_usd": deployable_usd,
        "ref_prices": ref_prices,
        "plan": plan,
        "validated_plan": validated_plan,
        "opening_decision_meta": opening_decision_meta,
        "midmorning_reference_prices": mm_refs,
        "midmorning_selection": sel.log_fields(),
    }


def run_paper_opening_morning(
    trade_date: date,
    *,
    settings: Settings | None = None,
    strategy: StrategyEngine | None = None,
    dry_run_no_submit: bool = False,
    enable_midmorning: bool = False,
) -> dict[str, Any]:
    """
    Orchestration only: Step 1 → … → Step 5.5, then wait for 09:30 ET, then Step 6.

    Start this job *before* 09:30 ET so ingestion + planning complete before the open wait.

    With ``dry_run_no_submit=True``: runs Steps 1–5.5 unchanged, skips the open wait and Step 6,
    does not submit orders or write the managed-position ledger; prints audit diagnostics and a
    ``would_submit`` summary to stderr.

    With ``enable_midmorning=True``: after opening Step 6 (unless dry-run), waits until 10:30 ET
    and runs a second independent pipeline when opening inventory allows it (see ``MIDMORNING_SKIP``
    logs).
    """
    trading_url, _data_url = _require_env()
    if settings is None:
        settings = _settings_for_paper(trading_url)
    else:
        pu = urlparse(settings.alpaca_base_url.rstrip("/"))
        if pu.scheme != "https" or pu.hostname != "paper-api.alpaca.markets":
            _die("settings.alpaca_base_url must be https://paper-api.alpaca.markets when settings is passed in.")
        settings = replace(settings, dry_run=False)
    strategy = strategy or StrategyEngine()

    through = run_paper_opening_through_5_5(trade_date, settings=settings, strategy=strategy)
    deployable_usd = float(through["deployable_usd"])
    step2_packet = through["step2_packet"]
    validated_decision = through["validated_decision"]
    step4_allocation = through["step4_allocation"]
    validated_plan = through["validated_plan"]

    would_submit, would_orders = _paper_would_submit_summary(validated_plan)
    dry_summary: dict[str, Any] = {
        "dry_run_no_submit": True,
        "trade_date": trade_date.isoformat(),
        "would_submit": would_submit,
        "would_orders": would_orders,
        "deployable_usd": deployable_usd,
        "step4_preparation_status": step4_allocation.get("preparation_status"),
    }

    if dry_run_no_submit:
        _paper_diag_emit("final would_submit summary (dry-run-no-submit)", dry_summary)
        base_out: dict[str, Any] = {
            "trade_date": trade_date.isoformat(),
            "deployable_usd": deployable_usd,
            "dry_run_no_submit": True,
            "would_submit": would_submit,
            "would_orders": would_orders,
            "step4_preparation_status": step4_allocation.get("preparation_status"),
            "validated_execution_plan_ok": validated_plan.get("execution_status") != "no_execution",
        }
        if enable_midmorning:
            mm = run_paper_midmorning_through_5_5(trade_date, settings=settings, strategy=strategy)
            ws_mm, wo_mm = _paper_would_submit_summary(mm["validated_plan"])
            base_out["midmorning"] = {
                "would_submit": ws_mm,
                "would_orders": wo_mm,
                "step4_preparation_status": mm["step4_allocation"].get("preparation_status"),
                "validated_execution_plan_ok": mm["validated_plan"].get("execution_status") != "no_execution",
            }
        return base_out

    _wait_until_market_open_et(trade_date)

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    client = AlpacaHttpOpeningClient(settings.alpaca_base_url, headers)
    store = SqliteOpeningIdempotencyStore(_default_idempotency_path())
    submission = submit_opening_execution_plan(
        plan=validated_plan,
        alpaca_client=client,
        idempotency_store=store,
    )

    ledger = SqliteManagedPositionLedger(default_managed_position_ledger_path())
    managed_ledger_writes = record_submitted_opening_buys(
        ledger,
        trade_date=trade_date.isoformat(),
        submission=submission,
        validated_decision=validated_decision,
        step2_packet=step2_packet,
    )

    midmorning_out: dict[str, Any] | None = None
    if enable_midmorning:
        _wait_until_midmorning_et(trade_date)
        submitted_syms = _opening_buy_symbols_submitted(submission)
        broker_chk = AlpacaBroker(settings)
        open_now = broker_chk.list_open_position_symbols()
        overlap = submitted_syms & open_now
        if overlap:
            _LOG.info(
                "MIDMORNING_SKIP reason=position_open symbols=%s",
                ",".join(sorted(overlap)),
            )
            midmorning_out = {"skipped": True, "reason": "position_open", "symbols": sorted(overlap)}
        else:
            mm_through = run_paper_midmorning_through_5_5(trade_date, settings=settings, strategy=strategy)
            mm_plan = mm_through["validated_plan"]
            mm_submit_ok = (
                mm_through["step4_allocation"].get("preparation_status") == "ready"
                and mm_plan.get("execution_status") != "no_execution"
            )
            if not mm_submit_ok:
                midmorning_out = {
                    "skipped": True,
                    "reason": "filters_failed",
                    "step4_preparation_status": mm_through["step4_allocation"].get("preparation_status"),
                }
            else:
                mm_submission = submit_opening_execution_plan(
                    plan=mm_plan,
                    alpaca_client=client,
                    idempotency_store=store,
                    client_order_id_prefix="OPEN-MM",
                )
                mm_ledger_writes = record_submitted_opening_buys(
                    ledger,
                    trade_date=trade_date.isoformat(),
                    submission=mm_submission,
                    validated_decision=mm_through["validated_decision"],
                    step2_packet=mm_through["step2_packet"],
                )
                for tr in mm_through["step4_allocation"].get("trades") or []:
                    if isinstance(tr, Mapping) and tr.get("included") is True:
                        _LOG.info(
                            "MIDMORNING_EXECUTION symbol=%s weight=%s",
                            str(tr.get("symbol") or "").strip().upper(),
                            tr.get("capital_weight"),
                        )
                midmorning_out = {
                    "skipped": False,
                    "submission": mm_submission,
                    "managed_position_ledger_writes": mm_ledger_writes,
                    "deployable_usd": mm_through["deployable_usd"],
                }

    out = {
        "trade_date": trade_date.isoformat(),
        "deployable_usd": deployable_usd,
        "step4_preparation_status": step4_allocation.get("preparation_status"),
        "validated_execution_plan_ok": validated_plan.get("execution_status") != "no_execution",
        "submission": submission,
        "managed_position_ledger_writes": managed_ledger_writes,
    }
    if midmorning_out is not None:
        out["midmorning"] = midmorning_out
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Paper opening-bell run (Steps 1–6 orchestration).")
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="YYYY-MM-DD (default: today in America/New_York)",
    )
    parser.add_argument(
        "--dry-run-no-submit",
        action="store_true",
        help="Run Steps 1–5.5 only: no open wait, no orders, no managed-position ledger writes.",
    )
    parser.add_argument(
        "--midmorning",
        action="store_true",
        help="After opening Step 6, wait until 10:30 ET and run the independent mid-morning pipeline "
        "when opening inventory is flat.",
    )
    args = parser.parse_args()
    if args.trade_date:
        td = date.fromisoformat(args.trade_date)
    else:
        td = datetime.now(_ET).date()
    try:
        result = run_paper_opening_morning(
            td,
            dry_run_no_submit=bool(args.dry_run_no_submit),
            enable_midmorning=bool(args.midmorning),
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("paper_open_run failed")
        _die(f"paper_open_run failed: {exc!r}")
    print(json.dumps(result, indent=2, default=str))
    if result.get("dry_run_no_submit"):
        _LOG.info("dry_run_no_submit: Step 6 skipped; would_submit=%s", result.get("would_submit"))
    else:
        _LOG.info("Step 6 submission: %s", json.dumps(result.get("submission"), default=str))


if __name__ == "__main__":
    main()
