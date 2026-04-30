"""Single daily pipeline — pure orchestration; business rules live in engines."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, datetime, time, timezone

from stockbot import __version__
from stockbot.ai.processor import NewsFilingsProcessor
from stockbot.audit.logger import AuditLogger
from stockbot.config import Settings
from stockbot.execution.broker import AlpacaBroker
from stockbot.execution.orders import (
    build_buy_market,
    capital_fraction_for_slot,
    capital_fractions_for_chosen_slots,
)
from stockbot.features.premarket import neutral_symbol_row, neutral_spy_row
from stockbot.features.sentiment import sentiment_features_from_llm
from stockbot.features.technical import technical_features
from stockbot.ingestion.filings import fetch_filings_refs
from stockbot.ingestion.market import fetch_market_snapshots
from stockbot.ingestion.premarket import fetch_premarket_for_watchlist
from stockbot.ingestion.premarket_wait import wait_until_premarket_decision_et
from stockbot.ingestion.news import fetch_news
from stockbot.models import DailyReasoningRecord, IngestionBundle
from stockbot.risk.engine import RiskEngine, load_daily_trade_count, record_trade_execution
from stockbot.risk.kill_switch import KillSwitch
from stockbot.strategy.engine import StrategyEngine

_LOG = logging.getLogger("stockbot")


def is_weekday(d: date) -> bool:
    return d.weekday() < 5


def run_daily_pipeline(
    trade_date: date | None = None,
    settings: Settings | None = None,
    strategy: StrategyEngine | None = None,
) -> DailyReasoningRecord:
    settings = settings or Settings.from_env()
    strategy = strategy or StrategyEngine()
    symbols = list(strategy.watchlist)
    trade_date = trade_date or datetime.now(timezone.utc).date()

    audit = AuditLogger(settings.audit_dir)
    kill = KillSwitch(settings.kill_switch_path)
    risk_engine = RiskEngine(settings, kill_switch=kill)
    broker = AlpacaBroker(settings)
    llm = NewsFilingsProcessor(settings)

    if not is_weekday(trade_date):
        plain = f"{trade_date}: skipped — weekend."
        rec = DailyReasoningRecord(
            trade_date=trade_date.isoformat(),
            pipeline_version=__version__,
            inputs_trace={"note": "weekend_skip"},
            llm_outputs=[],
            strategy={},
            risk={"skipped": True},
            executions=[],
            plain_english=plain,
            meta={"kill_switch": kill.is_active(), "dry_run": settings.dry_run},
        )
        audit.write_reasoning(rec)
        return rec

    # Anchor ingestion to the pipeline's trade_date (critical for multi-day simulation).
    # Using "now" here made every historical day fetch the same bar window ending today →
    # identical technicals and scores across the backtest range.
    ingest_as_of = datetime.combine(trade_date, time(20, 0, 0, tzinfo=timezone.utc))

    market, market_meta = fetch_market_snapshots(symbols, settings, as_of=ingest_as_of)

    watchlist_market = {s: market[s] for s in symbols if s in market}

    if settings.enable_premarket_signals:
        wait_until_premarket_decision_et(trade_date, settings)
        try:
            step1 = fetch_premarket_for_watchlist(settings, trade_date, symbols)
            pm_state = {
                "disabled_session": False,
                "trade_date": trade_date.isoformat(),
                "feed": (settings.alpaca_data_feed or "sip").strip().lower() or "sip",
                "step1": step1,
                "symbols": {
                    s.upper(): neutral_symbol_row(fetch_error="STEP1_INGESTION_ONLY") for s in symbols
                },
                "spy": neutral_spy_row(fetch_error="STEP1_INGESTION_ONLY"),
                "errors": [],
            }
            for sym, row in step1.items():
                st = row.get("status")
                if st and st != "ok":
                    pm_state["errors"].append(f"{sym}:{st}:{row.get('reason')}")
        except Exception as exc:  # noqa: BLE001 — never fail the whole pipeline on pre-market
            _LOG.exception("[premarket] session fetch failed; disabling for this run only")
            pm_state = {
                "disabled_session": True,
                "trade_date": trade_date.isoformat(),
                "feed": None,
                "errors": [f"pipeline_wrap:{exc!r}"],
                "symbols": {s.upper(): neutral_symbol_row(fetch_error="session_fetch_failed") for s in symbols},
                "spy": neutral_spy_row(fetch_error="session_fetch_failed"),
            }
    else:
        pm_state = {
            "disabled_session": True,
            "trade_date": trade_date.isoformat(),
            "feed": None,
            "errors": ["STOCKBOT_ENABLE_PREMARKET_SIGNALS off"],
            "symbols": {s.upper(): neutral_symbol_row(fetch_error="FEATURE_FLAG_OFF") for s in symbols},
            "spy": neutral_spy_row(fetch_error="FEATURE_FLAG_OFF"),
        }

    news, news_meta = fetch_news(
        symbols, run_date=ingest_as_of, trade_date=trade_date, settings=settings
    )
    filings = fetch_filings_refs(symbols, as_of=trade_date)
    bundle = IngestionBundle(
        run_date=trade_date,
        market=watchlist_market,
        news=news,
        filings=filings,
        premarket=pm_state,
    )

    llm_views = llm.analyze_watchlist(symbols, news, filings)
    decision = strategy.decide(trade_date, bundle, llm_views, settings)

    # Baseline for audit; in-loop counter increments after each successful non-dry fill.
    initial_daily_trade_count = load_daily_trade_count(settings.state_dir, trade_date)
    daily_trades_executed = initial_daily_trade_count

    account = broker.get_account()
    open_syms = broker.list_open_position_symbols()

    risk_evaluations: list[dict[str, object]] = []
    executions_out: list[dict[str, object]] = []

    # Up to two independent passes: risk sizing + submit does not share one verdict across symbols.
    # A block on the first pick must not prevent attempting the second (unless shared limits apply).
    # Capital: slot 1 fixed at 0.7 of max sleeve when two picks; slot 2 is 0.30 / 0.25 / 0.20 (see orders)
    # to slot 1 raw score (see ``capital_fractions_for_chosen_slots``); single pick = 100%.
    n_picks = len(decision.chosen)
    capital_plan = (
        decision.capital_fractions
        if len(decision.capital_fractions) == n_picks and n_picks
        else capital_fractions_for_chosen_slots(decision.chosen)
    )
    for slot, cand in enumerate(decision.chosen):
        if len(capital_plan) == n_picks:
            position_weight = capital_plan[slot]
        else:
            position_weight = capital_fraction_for_slot(slot, n_picks)
        account = broker.get_account()
        open_syms = broker.list_open_position_symbols()
        pm_mult = float(getattr(decision, "risk_notional_multiplier", 1.0) or 1.0)
        effective_weight = float(position_weight) * pm_mult
        verdict = risk_engine.evaluate(
            trade_date,
            cand,
            account,
            daily_trades_executed=daily_trades_executed,
            open_position_symbols=open_syms,
            notional_fraction=effective_weight,
        )
        risk_evaluations.append(
            {
                "symbol": cand.symbol,
                "trade_slot": slot + 1,
                "position_weight": position_weight,
                "premarket_notional_multiplier": pm_mult,
                "effective_notional_fraction": effective_weight,
                "allowed": verdict.allowed,
                "block_reasons": list(verdict.block_reasons),
                "qty": verdict.position_qty,
                "notional_usd": verdict.notional_usd,
            }
        )
        if not verdict.allowed:
            continue
        intent = build_buy_market(cand.symbol, verdict)
        if not intent:
            continue
        execution = broker.submit_order(intent)
        executions_out.append(
            {
                "symbol": cand.symbol,
                "trade_slot": slot + 1,
                "position_weight": position_weight,
                "premarket_notional_multiplier": pm_mult,
                "effective_notional_fraction": effective_weight,
                "success": execution.success,
                "broker_order_id": execution.broker_order_id,
                "error": execution.error,
            }
        )
        if execution.success and not settings.dry_run:
            record_trade_execution(settings.state_dir, trade_date)
            daily_trades_executed += 1

    _placeholder_news_sources = frozenset({"stub", "stub_fallback", "finnhub_empty"})
    real_news_article_rows = sum(1 for n in news if n.source not in _placeholder_news_sources)

    inputs_trace = {
        "watchlist": symbols,
        "effective_watchlist": list(symbols),
        "ingest_as_of": ingest_as_of.isoformat(),
        "market_symbols": list(watchlist_market.keys()),
        "market_ingestion": market_meta,
        "news_count": len(news),
        "news_real_article_rows": real_news_article_rows,
        "news_ingestion": news_meta,
        "news_sample": [
            {
                "symbol": n.symbol,
                "source": n.source,
                "headline": (n.headline or "")[:240],
                "note": n.note,
            }
            for n in news[:24]
        ],
        "filings_count": len(filings),
        "account": asdict(account),
        "open_positions": sorted(open_syms),
        "daily_trades_executed_before_run": initial_daily_trade_count,
    }

    _LOG.info(
        "[ingest] trade_date=%s ingest_as_of=%s real_news_rows=%d news_provider=%s market_sources=%s",
        trade_date,
        ingest_as_of.isoformat(),
        real_news_article_rows,
        news_meta.get("provider"),
        {s: market_meta.get("per_symbol", {}).get(s, {}).get("source") for s in symbols},
    )

    llm_outputs = [asdict(v) for v in llm_views.values()]

    audit_reason_codes = list(decision.reason_codes)
    executed_symbols = {str(e["symbol"]) for e in executions_out}
    for ev in risk_evaluations:
        if not ev["allowed"]:
            audit_reason_codes.append("RISK_BLOCKED")
        elif str(ev["symbol"]) not in executed_symbols:
            audit_reason_codes.append("ORDER_INTENT_SKIPPED")

    strategy_payload = {
        "reason_codes": audit_reason_codes,
        "premarket": bundle.premarket,
        "risk_notional_multiplier": float(getattr(decision, "risk_notional_multiplier", 1.0) or 1.0),
        "ranked": [
            {
                "symbol": c.symbol,
                "score": c.score,
                "adjusted_score": c.adjusted_score,
                "breakdown": c.score_breakdown,
                "technical": technical_features(bundle.market[c.symbol].bars),
                "sentiment_features": sentiment_features_from_llm(c.features.raw_llm),
            }
            for c in decision.ranked
        ],
        "chosen": [c.symbol for c in decision.chosen],
        "selected_trade_1": decision.chosen[0].symbol if len(decision.chosen) >= 1 else None,
        "selected_trade_2": decision.chosen[1].symbol if len(decision.chosen) >= 2 else None,
        "candidate_outcomes": decision.candidate_outcomes,
        "cooldown_skips": decision.cooldown_skips,
        "soft_modifiers": decision.soft_modifiers,
        "capital_fractions": list(decision.capital_fractions) if decision.capital_fractions else None,
    }

    risk_payload: dict[str, object] = {"evaluations": risk_evaluations}

    plain = _plain_english(
        trade_date=trade_date,
        decision=decision,
        risk_evaluations=risk_evaluations,
        executions=executions_out,
        kill_active=kill.is_active(),
        dry_run=settings.dry_run,
    )

    record = DailyReasoningRecord(
        trade_date=trade_date.isoformat(),
        pipeline_version=__version__,
        inputs_trace=inputs_trace,
        llm_outputs=llm_outputs,
        strategy=strategy_payload,
        risk=risk_payload,
        executions=executions_out,
        plain_english=plain,
        meta={
            "dry_run": settings.dry_run,
            "kill_switch": kill.is_active(),
        },
    )
    path = audit.write_reasoning(record)
    _LOG.info("Wrote audit: %s", path)
    return record


def _plain_english(**kwargs) -> str:
    trade_date = kwargs["trade_date"]
    decision = kwargs["decision"]
    risk_evaluations: list[dict[str, object]] = kwargs["risk_evaluations"]
    executions: list[dict[str, object]] = kwargs["executions"]
    parts = [
        f"Trade date {trade_date}: pipeline completed.",
        f"Watchlist: {', '.join(decision.watchlist)}.",
    ]
    if kwargs["kill_active"]:
        parts.append("Kill-switch file is active; risk layer should block.")
    if decision.chosen:
        desc = "; ".join(
            f"trade #{i} {c.symbol} score {c.score:.3f}" for i, c in enumerate(decision.chosen, start=1)
        )
        parts.append(f"Strategy selected ({len(decision.chosen)}): {desc}.")
    else:
        parts.append("No candidate met strategy thresholds.")
    for ev in risk_evaluations:
        sym = ev["symbol"]
        if ev["allowed"]:
            parts.append(
                f"Risk approved {sym} qty={ev['qty']} (~${ev['notional_usd']})."
            )
        else:
            parts.append(f"Risk blocked {sym}: {', '.join(ev['block_reasons'])}.")
    if decision.chosen and not executions and any(ev["allowed"] for ev in risk_evaluations):
        parts.append("At least one name was risk-approved but no execution record (see ORDER_INTENT_SKIPPED).")
    for ex in executions:
        if ex["success"]:
            parts.append(
                f"Execution {ex['symbol']}: success id={ex['broker_order_id']} (dry_run={kwargs['dry_run']})."
            )
        else:
            parts.append(f"Execution {ex['symbol']} failed: {ex['error']}")
    return " ".join(parts)
