"""
Step 2 — AI premarket input: pure transforms from Step 1 rows + daily bars (no I/O).

No fetching, retries, strategy logic, or Step 1 changes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from stockbot.models import MarketSnapshot

_ET = "America/New_York"


def _trade_date_str(trade_date: date) -> str:
    return trade_date.isoformat()


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and abs(v) != float("inf")


def _clean_derived_float(x: Any) -> float | None:
    """Coerce to finite float for payload; never NaN or inf."""
    if not _finite(x):
        return None
    return float(x)


def _prior_rth_close_out(p: float | None) -> float | None:
    if p is None:
        return None
    if not _finite(p) or float(p) <= 0:
        return None
    return float(p)


def _prior_rth_close_from_daily(bars: pd.DataFrame | None, trade_date: date) -> float | None:
    """Last daily close strictly before ``trade_date`` (bar calendar date in America/New_York)."""
    if bars is None or bars.empty or "close" not in bars.columns:
        return None
    idx = pd.to_datetime(bars.index, utc=True).tz_convert(_ET)
    mask = [bool(ts.date() < trade_date) for ts in idx]
    eligible = bars.loc[mask]
    if eligible.empty:
        return None
    last = float(eligible["close"].astype(float).iloc[-1])
    if not _finite(last) or last <= 0:
        return None
    return last


def _premarket_return_pct(pm_open: Any, pm_close: Any) -> float | None:
    if not _finite(pm_open) or not _finite(pm_close):
        return None
    o = float(pm_open)
    c = float(pm_close)
    if o == 0:
        return None
    return _clean_derived_float((c / o) - 1.0)


def _close_position_in_range(pm_low: Any, pm_high: Any, pm_close: Any) -> float | None:
    if not _finite(pm_low) or not _finite(pm_high) or not _finite(pm_close):
        return None
    lo, hi, cl = float(pm_low), float(pm_high), float(pm_close)
    if hi <= lo:
        return None
    return _clean_derived_float((cl - lo) / (hi - lo))


def _step1_volume(row: dict[str, Any]) -> float | None:
    return _clean_derived_float(row.get("pm_volume"))


def build_market_context(
    trade_date: date,
    step1_spy: dict[str, Any] | None,
    step1_qqq: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Benchmark fields from SPY/QQQ Step 1 rows only.
    Any missing or non-ok row → nulls for that ticker's three derived fields.
    """
    spy_ret = spy_pos = spy_vol = None
    qqq_ret = qqq_pos = qqq_vol = None

    if step1_spy is not None and step1_spy.get("status") == "ok":
        spy_ret = _premarket_return_pct(step1_spy.get("pm_open"), step1_spy.get("pm_close"))
        spy_pos = _close_position_in_range(
            step1_spy.get("pm_low"), step1_spy.get("pm_high"), step1_spy.get("pm_close")
        )
        spy_vol = _step1_volume(step1_spy)

    if step1_qqq is not None and step1_qqq.get("status") == "ok":
        qqq_ret = _premarket_return_pct(step1_qqq.get("pm_open"), step1_qqq.get("pm_close"))
        qqq_pos = _close_position_in_range(
            step1_qqq.get("pm_low"), step1_qqq.get("pm_high"), step1_qqq.get("pm_close")
        )
        qqq_vol = _step1_volume(step1_qqq)

    return {
        "trade_date": _trade_date_str(trade_date),
        "spy_premarket_return_pct": spy_ret,
        "qqq_premarket_return_pct": qqq_ret,
        "spy_close_position_in_range": spy_pos,
        "qqq_close_position_in_range": qqq_pos,
        "spy_premarket_volume": spy_vol,
        "qqq_premarket_volume": qqq_vol,
    }


def _raw_pm_field(step1_row: dict[str, Any], key: str) -> float | None:
    return _clean_derived_float(step1_row.get(key))


def _bar_count(x: Any) -> int:
    if not _finite(x):
        return 0
    try:
        return max(0, int(float(x)))
    except (ValueError, OverflowError):
        return 0


def build_symbol_ai_row(
    trade_date: date,
    step1_row: dict[str, Any],
    prior_rth_close: float | None,
    *,
    include_raw_pm_ohlc: bool = True,
) -> dict[str, Any]:
    sym = str(step1_row.get("symbol", "")).upper()
    td = _trade_date_str(trade_date)
    status = step1_row.get("status")
    ok = status == "ok"
    prior_out = _prior_rth_close_out(prior_rth_close)

    gap: float | None = None
    pm_ret: float | None = None
    pm_pos: float | None = None
    pm_vol: float | None = None

    if ok:
        pm_ret = _premarket_return_pct(step1_row.get("pm_open"), step1_row.get("pm_close"))
        pm_pos = _close_position_in_range(
            step1_row.get("pm_low"), step1_row.get("pm_high"), step1_row.get("pm_close")
        )
        pm_vol = _step1_volume(step1_row)
        pc = step1_row.get("pm_close")
        if prior_out is not None and _finite(pc):
            gap = _clean_derived_float((float(pc) / prior_out) - 1.0)

    row: dict[str, Any] = {
        "symbol": sym,
        "trade_date": td,
        "status": status,
        "reason": step1_row.get("reason"),
        "alpaca_feed": step1_row.get("alpaca_feed"),
        "bar_count": _bar_count(step1_row.get("bar_count")),
        "first_bar_ts": step1_row.get("first_bar_ts"),
        "last_bar_ts": step1_row.get("last_bar_ts"),
        "prior_rth_close": prior_out,
        "gap_close_vs_prior_close_pct": gap,
        "pm_session_return_pct": pm_ret,
        "pm_close_position_in_range": pm_pos,
        "pm_volume": pm_vol,
    }

    if include_raw_pm_ohlc:
        row["pm_open"] = _raw_pm_field(step1_row, "pm_open")
        row["pm_high"] = _raw_pm_field(step1_row, "pm_high")
        row["pm_low"] = _raw_pm_field(step1_row, "pm_low")
        row["pm_close"] = _raw_pm_field(step1_row, "pm_close")

    return row


def build_ai_premarket_packet(
    trade_date: date,
    watchlist: Sequence[str],
    step1_by_symbol: Mapping[str, dict[str, Any]],
    market_by_symbol: Mapping[str, MarketSnapshot | None],
) -> dict[str, Any]:
    """
    Assemble the AI-ready premarket packet. ``watchlist`` order is preserved in ``symbols``.

    ``step1_by_symbol`` must contain Step 1 dicts for each watchlist symbol (and SPY/QQQ if used
    for ``market_context``); missing keys are not synthesized here.
    """
    spy = step1_by_symbol.get("SPY")
    qqq = step1_by_symbol.get("QQQ")
    market_context = build_market_context(trade_date, spy, qqq)

    symbols_out: list[dict[str, Any]] = []
    for sym in watchlist:
        u = sym.upper()
        snap = market_by_symbol.get(u)
        bars = snap.bars if snap is not None else None
        prior = _prior_rth_close_from_daily(bars, trade_date)
        step1_row = step1_by_symbol[u]
        symbols_out.append(build_symbol_ai_row(trade_date, step1_row, prior))

    return {
        "trade_date": _trade_date_str(trade_date),
        "market_context": market_context,
        "symbols": symbols_out,
    }
