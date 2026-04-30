"""
Mid-morning Step 2 enrichment: merge fresh RTH tape through ~10:30 ET into the AI packet.

Leaves allocation contract intact by overwriting excursion/volume fields the Step 4 gates already read.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import date, time
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from stockbot.config import Settings
from stockbot.ingestion.premarket_packet import _prior_rth_close_from_daily
from stockbot.ingestion.rth_minute_bars import fetch_rth_1min_bars_range
from stockbot.models import MarketSnapshot

_LOG = logging.getLogger("stockbot.ingestion.midmorning_packet")

_ET = ZoneInfo("America/New_York")

# Through end of 10:30 bar (exclusive API end at 10:31 ET).
_MIDM_RTH_START = time(9, 30)
_MIDM_RTH_END = time(10, 31)


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _agg_intraday(df: pd.DataFrame, session_date: date) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index, utc=True).tz_convert(_ET)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    vol_sum = 0
    last_close: float | None = None
    first_open: float | None = None
    for row_ts, ts_et, (_, row) in zip(df.index, idx, df.iterrows()):
        if ts_et.date() != session_date:
            continue
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        if not all(map(math.isfinite, (o, h, l, c))):
            continue
        if first_open is None:
            first_open = o
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vol_sum += int(row.get("volume", 0))
        last_close = c
    if first_open is None or last_close is None or not highs or not lows:
        return None
    hi = max(highs)
    lo = min(lows)
    pos_in_range = None
    if hi > lo:
        pos_in_range = (last_close - lo) / (hi - lo)
    return {
        "open_rth": first_open,
        "last_close": last_close,
        "high": hi,
        "low": lo,
        "volume_sum": float(vol_sum),
        "close_position_in_range": pos_in_range,
    }


def enrich_step2_packet_midmorning(
    packet: dict[str, Any],
    trade_date: date,
    settings: Settings,
    watchlist: list[str],
    market_by_symbol: Mapping[str, MarketSnapshot | None],
) -> dict[str, float]:
    """
    Mutate ``packet`` symbol rows + ``market_context`` benchmarks using RTH data through 10:30 ET.

    Returns ``reference_prices`` (upper symbol -> last regular-session print through cutoff) for
    execution-plan sizing.
    """
    syms_out = packet.get("symbols")
    if not isinstance(syms_out, list):
        return {}

    ref: dict[str, float] = {}
    bench_syms = ("SPY", "QQQ")

    def _update_row(sym_u: str, row: dict[str, Any], snap: MarketSnapshot | None) -> None:
        bars = snap.bars if snap is not None else None
        prior = _prior_rth_close_from_daily(bars, trade_date)
        df = fetch_rth_1min_bars_range(sym_u, trade_date, settings, start_et=_MIDM_RTH_START, end_et=_MIDM_RTH_END)
        agg = _agg_intraday(df, trade_date)
        if agg is None:
            _LOG.warning("[midmorning_packet] no RTH aggregate for symbol=%s date=%s", sym_u, trade_date)
            return
        last_c = float(agg["last_close"])
        ref[sym_u] = last_c
        if row.get("status") != "ok":
            return
        o = float(agg["open_rth"])
        if prior is not None and _finite(prior) and float(prior) > 0:
            row["gap_close_vs_prior_close_pct"] = (last_c / float(prior)) - 1.0
        if _finite(o) and o > 0:
            row["pm_session_return_pct"] = (last_c / o) - 1.0
        row["pm_volume"] = float(agg["volume_sum"])
        pir = agg.get("close_position_in_range")
        if pir is not None and _finite(pir):
            row["pm_close_position_in_range"] = float(pir)

    for row in syms_out:
        if not isinstance(row, dict):
            continue
        sym_u = str(row.get("symbol", "")).strip().upper()
        if not sym_u:
            continue
        snap = market_by_symbol.get(sym_u) if isinstance(market_by_symbol, Mapping) else None
        _update_row(sym_u, row, snap)

    mc = packet.get("market_context")
    if isinstance(mc, dict):
        for bench in bench_syms:
            snap_b = market_by_symbol.get(bench) if isinstance(market_by_symbol, Mapping) else None
            df_b = fetch_rth_1min_bars_range(
                bench, trade_date, settings, start_et=_MIDM_RTH_START, end_et=_MIDM_RTH_END
            )
            agg_b = _agg_intraday(df_b, trade_date)
            if agg_b is None:
                continue
            o = float(agg_b["open_rth"])
            c = float(agg_b["last_close"])
            if bench == "SPY" and _finite(o) and o > 0:
                mc["spy_premarket_return_pct"] = (c / o) - 1.0
                mc["spy_close_position_in_range"] = agg_b.get("close_position_in_range")
                mc["spy_premarket_volume"] = float(agg_b["volume_sum"])
            if bench == "QQQ" and _finite(o) and o > 0:
                mc["qqq_premarket_return_pct"] = (c / o) - 1.0
                mc["qqq_close_position_in_range"] = agg_b.get("close_position_in_range")
                mc["qqq_premarket_volume"] = float(agg_b["volume_sum"])

    return ref
