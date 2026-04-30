"""
Mid-morning (~10:30 ET) selection: sector leadership + relative strength (RTH 09:30–10:30).

Independent from opening allocation / opening AI. Long-only; rank-1 only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, time
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo

from stockbot.ingestion.rth_minute_bars import fetch_rth_1min_bars_range

_LOG = logging.getLogger("stockbot.execution.midmorning_sector_strategy")

_ET = ZoneInfo("America/New_York")

_RTH_START = time(9, 30)
_RTH_END_MM = time(10, 31)

_BENCHMARKS = ("SPY", "QQQ", "IWM")
_LEADERSHIP_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "IWM")

MIN_LEADING_SECTOR_RETURN = 0.007
MIN_SYMBOL_RTH_RETURN = 0.008
MIN_RANGE_POSITION = 0.80
MIN_VOLUME = 750_000
MAX_PULLBACK_FROM_HIGH_OF_RANGE = 0.35

# Watchlist symbol → sector/style ETF (must be one of _LEADERSHIP_ETFS). None = not eligible as MM stock pick.
_MIDMORNING_SYMBOL_SECTOR_ETF: dict[str, str] = {}
for _etf in _LEADERSHIP_ETFS:
    _MIDMORNING_SYMBOL_SECTOR_ETF[_etf.upper()] = _etf.upper()

_MIDMORNING_SYMBOL_SECTOR_ETF.update(
    {
        # Mega-cap tech / semis / software
        "AAPL": "XLK",
        "MSFT": "XLK",
        "NVDA": "XLK",
        "AMD": "XLK",
        "GOOGL": "XLK",
        "META": "XLK",
        "AVGO": "XLK",
        "SMCI": "XLK",
        "SNOW": "XLK",
        "CRWD": "XLK",
        "PANW": "XLK",
        "DDOG": "XLK",
        "PLTR": "XLK",
        "COIN": "XLK",
        # Consumer discretionary
        "AMZN": "XLY",
        "TSLA": "XLY",
        "NFLX": "XLY",
        "SHOP": "XLY",
        "RIVN": "XLY",
        "LCID": "XLY",
        "AFRM": "XLY",
        # Financials
        "JPM": "XLF",
        "GS": "XLF",
        "BAC": "XLF",
        # Cyclicals / diversified — no XLI in mandate; use broad IWM proxy
        "CAT": "IWM",
        "BA": "IWM",
        # Benchmarks: tradable as ETFs only when leadership matches
        "QQQ": "XLK",
        "SPY": "SPY",
    }
)


def midmorning_sector_etf_for_symbol(symbol: str) -> str | None:
    """Sector/style ETF ticker for MM alignment, or None if unknown / not tradable under MM rules."""
    s = str(symbol).strip().upper()
    etf = _MIDMORNING_SYMBOL_SECTOR_ETF.get(s)
    if etf is None:
        return None
    if etf == "SPY":
        return None
    return etf


@dataclass
class MidmorningBarStats:
    symbol: str
    price_open: float | None
    price_close: float | None
    rth_high: float | None
    rth_low: float | None
    rth_return_pct: float | None
    rth_close_position_in_range: float | None
    rth_volume_total: float | None
    rth_vwap: float | None


@dataclass
class MidmorningSelectionResult:
    midmorning_leading_sector: str | None
    midmorning_sector_return: float | None
    midmorning_candidate_symbol: str | None
    midmorning_candidate_sector: str | None
    midmorning_relative_strength_vs_spy: float | None
    midmorning_relative_strength_vs_sector: float | None
    midmorning_rth_return_pct: float | None
    midmorning_range_position: float | None
    midmorning_volume: float | None
    midmorning_vwap: float | None
    midmorning_price_vs_vwap_ok: bool | None
    midmorning_filter_pass: bool
    midmorning_skip_reason: str
    spy_rth_return_pct: float | None
    qqq_rth_return_pct: float | None

    def log_fields(self) -> dict[str, Any]:
        base = asdict(self)
        return base


def compute_rth_window_stats(df: pd.DataFrame, session_date: date, symbol: str) -> MidmorningBarStats | None:
    """Aggregate 09:30–10:30 ET regular-session bars into strategy stats."""
    if df is None or df.empty:
        return None
    idx_et = pd.to_datetime(df.index, utc=True).tz_convert(_ET)

    o_prices: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    vols: list[int] = []
    tp_vol: list[tuple[float, int]] = []

    for i in range(len(df)):
        ts_et = idx_et[i]
        if ts_et.date() != session_date:
            continue
        row = df.iloc[i]
        o, h, l, c = (
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
        )
        v = int(row.get("volume", 0) or 0)
        if not all(math.isfinite(x) for x in (o, h, l, c)):
            continue
        o_prices.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(v)
        tp = (h + l + c) / 3.0
        if math.isfinite(tp) and v > 0:
            tp_vol.append((tp, v))

    if not closes:
        return None

    price_open = o_prices[0]
    price_close = closes[-1]
    hi = max(highs)
    lo = min(lows)
    vol_sum = float(sum(vols))

    r_ret: float | None = None
    if price_open > 0 and math.isfinite(price_open) and math.isfinite(price_close):
        r_ret = (price_close / price_open) - 1.0

    rng_pos: float | None = None
    if hi > lo and math.isfinite(price_close):
        rng_pos = (price_close - lo) / (hi - lo)

    vwap: float | None = None
    if tp_vol:
        num = sum(t * float(v) for t, v in tp_vol)
        den = sum(float(v) for _, v in tp_vol)
        if den > 0 and math.isfinite(num):
            vwap = num / den

    sym_u = str(symbol).strip().upper()
    return MidmorningBarStats(
        symbol=sym_u,
        price_open=float(price_open) if math.isfinite(price_open) else None,
        price_close=float(price_close) if math.isfinite(price_close) else None,
        rth_high=float(hi) if math.isfinite(hi) else None,
        rth_low=float(lo) if math.isfinite(lo) else None,
        rth_return_pct=r_ret,
        rth_close_position_in_range=rng_pos,
        rth_volume_total=vol_sum if vol_sum > 0 else None,
        rth_vwap=vwap,
    )


def _skip_result(
    *,
    reason: str,
    spy_ret: float | None,
    qqq_ret: float | None,
    leader: str | None = None,
    leader_ret: float | None = None,
) -> tuple[MidmorningSelectionResult, MidmorningBarStats | None]:
    r = MidmorningSelectionResult(
        midmorning_leading_sector=leader,
        midmorning_sector_return=leader_ret,
        midmorning_candidate_symbol=None,
        midmorning_candidate_sector=None,
        midmorning_relative_strength_vs_spy=None,
        midmorning_relative_strength_vs_sector=None,
        midmorning_rth_return_pct=None,
        midmorning_range_position=None,
        midmorning_volume=None,
        midmorning_vwap=None,
        midmorning_price_vs_vwap_ok=None,
        midmorning_filter_pass=False,
        midmorning_skip_reason=reason,
        spy_rth_return_pct=spy_ret,
        qqq_rth_return_pct=qqq_ret,
    )
    _emit_selection_log(r)
    return r, None


def _passes_tape_filters(st: MidmorningBarStats) -> tuple[bool, str, bool | None]:
    sr = st.rth_return_pct
    rp = st.rth_close_position_in_range
    vol = st.rth_volume_total
    close_p = st.price_close
    hi = st.rth_high
    lo = st.rth_low
    vw = st.rth_vwap

    if sr is None or sr <= MIN_SYMBOL_RTH_RETURN:
        return False, "symbol_return_below_min", None
    if rp is None or rp < MIN_RANGE_POSITION:
        return False, "range_position_below_min", None
    if vol is None or vol < MIN_VOLUME:
        return False, "volume_below_min", None

    vwap_ok: bool | None = None
    if vw is not None and close_p is not None and math.isfinite(vw) and math.isfinite(close_p):
        vwap_ok = close_p >= vw
        if not vwap_ok:
            return False, "below_vwap", False
    elif vw is None:
        vwap_ok = None
    else:
        return False, "price_missing_for_vwap", None

    if hi is not None and lo is not None and close_p is not None and hi > lo:
        pullback = (hi - close_p) / (hi - lo)
        if pullback > MAX_PULLBACK_FROM_HIGH_OF_RANGE:
            return False, "exhaustion_pullback_from_high", vwap_ok

    return True, "", vwap_ok


def select_midmorning_long(
    trade_date: date,
    settings: Any,
    watchlist: Sequence[str],
) -> tuple[MidmorningSelectionResult, MidmorningBarStats | None]:
    """
    Deterministic mid-morning long: sector leadership + RS vs SPY + RS vs sector ETF + tape filters.
    """
    wl_raw = [str(s).strip().upper() for s in watchlist if str(s).strip()]
    need_syms = sorted(set(wl_raw) | set(_BENCHMARKS) | set(_LEADERSHIP_ETFS))

    stats_by_sym: dict[str, MidmorningBarStats] = {}
    for sym in need_syms:
        df = fetch_rth_1min_bars_range(sym, trade_date, settings, start_et=_RTH_START, end_et=_RTH_END_MM)
        st = compute_rth_window_stats(df, trade_date, sym)
        if st is not None:
            stats_by_sym[sym] = st

    spy_st = stats_by_sym.get("SPY")
    spy_ret = spy_st.rth_return_pct if spy_st else None
    qqq_st = stats_by_sym.get("QQQ")
    qqq_ret = qqq_st.rth_return_pct if qqq_st else None

    if spy_ret is None:
        return _skip_result(reason="no_spy_rth_stats", spy_ret=spy_ret, qqq_ret=qqq_ret)

    leadership_candidates: list[tuple[str, float]] = []
    for etf in _LEADERSHIP_ETFS:
        st = stats_by_sym.get(etf)
        if st is None or st.rth_return_pct is None:
            continue
        leadership_candidates.append((etf, float(st.rth_return_pct)))

    if not leadership_candidates:
        return _skip_result(reason="no_leadership_etf_stats", spy_ret=spy_ret, qqq_ret=qqq_ret)

    leadership_candidates.sort(key=lambda x: (-x[1], x[0]))
    leader, leader_ret = leadership_candidates[0]
    if leader_ret <= MIN_LEADING_SECTOR_RETURN:
        return _skip_result(
            reason="leading_sector_return_below_min",
            spy_ret=spy_ret,
            qqq_ret=qqq_ret,
            leader=leader,
            leader_ret=leader_ret,
        )

    macro_ok = (spy_ret is not None and spy_ret > 0) or (qqq_ret is not None and qqq_ret > 0)
    if not macro_ok:
        return _skip_result(
            reason="spy_qqq_not_positive",
            spy_ret=spy_ret,
            qqq_ret=qqq_ret,
            leader=leader,
            leader_ret=leader_ret,
        )

    spy_r = float(spy_ret)
    eligible: list[tuple[str, MidmorningBarStats, str, float, float, float]] = []

    for sym in wl_raw:
        sector_etf = midmorning_sector_etf_for_symbol(sym)
        if sector_etf is None or sector_etf != leader:
            continue
        st = stats_by_sym.get(sym)
        if st is None:
            continue
        sr = st.rth_return_pct
        if sr is None:
            continue
        st_sector = stats_by_sym.get(sector_etf)
        sec_ret = st_sector.rth_return_pct if st_sector else None
        if sec_ret is None:
            continue
        rs_spy = float(sr) - spy_r
        rs_sec = float(sr) - float(sec_ret)
        if rs_spy <= 0.0 or rs_sec <= 0.0:
            continue
        eligible.append((sym, st, sector_etf, rs_spy, rs_sec, float(sr)))

    if not eligible:
        return _skip_result(
            reason="no_watchlist_symbol_relative_strength_or_sector_alignment",
            spy_ret=spy_ret,
            qqq_ret=qqq_ret,
            leader=leader,
            leader_ret=leader_ret,
        )

    passed: list[tuple[str, MidmorningBarStats, str, float, float, float, bool | None]] = []
    for sym, st, sector_etf, rs_spy, rs_sec, sr in eligible:
        ok_f, _reason_f, vw_ok = _passes_tape_filters(st)
        if ok_f:
            passed.append((sym, st, sector_etf, rs_spy, rs_sec, sr, vw_ok))

    if not passed:
        return _skip_result(
            reason="no_watchlist_symbol_passes_tape_filters",
            spy_ret=spy_ret,
            qqq_ret=qqq_ret,
            leader=leader,
            leader_ret=leader_ret,
        )

    passed.sort(
        key=lambda x: (
            -x[3],
            -x[5],
            -(x[1].rth_close_position_in_range if x[1].rth_close_position_in_range is not None else 0.0),
            x[0],
        )
    )

    pick_sym, pick_st, sector_etf, rs_spy_f, rs_sec_f, sr_f, vw_ok_f = passed[0]

    out = MidmorningSelectionResult(
        midmorning_leading_sector=leader,
        midmorning_sector_return=leader_ret,
        midmorning_candidate_symbol=pick_sym,
        midmorning_candidate_sector=sector_etf,
        midmorning_relative_strength_vs_spy=rs_spy_f,
        midmorning_relative_strength_vs_sector=rs_sec_f,
        midmorning_rth_return_pct=pick_st.rth_return_pct,
        midmorning_range_position=pick_st.rth_close_position_in_range,
        midmorning_volume=pick_st.rth_volume_total,
        midmorning_vwap=pick_st.rth_vwap,
        midmorning_price_vs_vwap_ok=vw_ok_f,
        midmorning_filter_pass=True,
        midmorning_skip_reason="",
        spy_rth_return_pct=spy_ret,
        qqq_rth_return_pct=qqq_ret,
    )
    _emit_selection_log(out)
    return out, pick_st


def _emit_selection_log(result: MidmorningSelectionResult) -> None:
    d = result.log_fields()
    parts = [f"{k}={d[k]!r}" for k in sorted(d.keys())]
    _LOG.info("MIDMORNING_SELECTION %s", " ".join(parts))


def synthetic_step2_row_for_midmorning_pick(stats: MidmorningBarStats, trade_date: date) -> dict[str, Any]:
    """Minimal Step-2-shaped row for ledger / TP strength wiring (not opening allocation)."""
    td = trade_date.isoformat()
    sym = stats.symbol
    return {
        "symbol": sym,
        "trade_date": td,
        "status": "ok",
        "reason": None,
        "alpaca_feed": None,
        "bar_count": 0,
        "first_bar_ts": None,
        "last_bar_ts": None,
        "prior_rth_close": None,
        "gap_close_vs_prior_close_pct": None,
        "pm_session_return_pct": stats.rth_return_pct,
        "pm_close_position_in_range": stats.rth_close_position_in_range,
        "pm_volume": stats.rth_volume_total,
        "pm_open": stats.price_open,
        "pm_high": stats.rth_high,
        "pm_low": stats.rth_low,
        "pm_close": stats.price_close,
        "midmorning_rth_vwap": stats.rth_vwap,
    }


def build_midmorning_step2_packet(
    trade_date: date,
    pick_sym: str | None,
    pick_stats: MidmorningBarStats | None,
) -> dict[str, Any]:
    """Minimal packet for managed-position ledger snapshot."""
    syms_out: list[dict[str, Any]] = []
    if pick_sym and pick_stats:
        syms_out.append(synthetic_step2_row_for_midmorning_pick(pick_stats, trade_date))
    return {
        "trade_date": trade_date.isoformat(),
        "market_context": {},
        "symbols": syms_out,
    }


def deterministic_midmorning_confidence(rs_vs_spy: float) -> float:
    """Bounded synthetic confidence for ledger / downstream (not opening gates)."""
    return float(min(0.86, max(0.62, 0.62 + min(max(rs_vs_spy, 0.0), 0.024) * 10.0)))
