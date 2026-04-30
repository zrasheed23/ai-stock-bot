"""
Step 1 — premarket ingestion only: Alpaca 1-minute bars, 04:00–09:29 America/New_York.

No strategy / gap / ATR / RVOL / SPY injection. Intended feed for this workflow is SIP.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from stockbot.config import Settings

_LOG = logging.getLogger("stockbot.ingestion.premarket")

_ET = ZoneInfo("America/New_York")

_DATA_BASE = "https://data.alpaca.markets/v2/stocks"

_BODY_PREVIEW_LIMIT = 500


def _body_preview(text: str, limit: int = _BODY_PREVIEW_LIMIT) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def _feed(settings: Settings) -> str:
    # SIP is the normal case for this workflow (not IEX).
    return (settings.alpaca_data_feed or "sip").strip().lower() or "sip"


def _premarket_window_utc_bounds(session_trade_date: date) -> tuple[datetime, datetime]:
    """API window [04:00 ET, 09:30 ET) UTC so minute bars starting at 09:29 ET are included; 09:30 excluded."""
    start_local = datetime.combine(session_trade_date, time(4, 0), tzinfo=_ET)
    end_local = datetime.combine(session_trade_date, time(9, 30), tzinfo=_ET)
    return (
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )


def _bars_block_row_counts(bars_block: Mapping[str, Any], symbols: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym in symbols:
        su = sym.upper()
        rows = bars_block.get(su)
        if rows is None:
            rows = bars_block.get(sym)
        if rows is None and bars_block:
            for k, v in bars_block.items():
                if str(k).upper() == su:
                    rows = v
                    break
        counts[su] = len(rows) if isinstance(rows, list) else 0
    return counts


def _fetch_minute_bars_chunk(
    symbols: list[str],
    start_utc: datetime,
    end_utc: datetime,
    settings: Settings,
) -> dict[str, pd.DataFrame]:
    """Returns per-symbol DataFrames (possibly empty)."""
    out: dict[str, pd.DataFrame] = {s.upper(): pd.DataFrame() for s in symbols}
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        return out
    sym_param = ",".join(sorted({s.upper() for s in symbols}))
    feed = _feed(settings)
    start_s = start_utc.isoformat().replace("+00:00", "Z")
    end_s = end_utc.isoformat().replace("+00:00", "Z")
    params: dict[str, Any] = {
        "symbols": sym_param,
        "timeframe": "1Min",
        "start": start_s,
        "end": end_s,
        "limit": 10000,
        "adjustment": "raw",
        "feed": feed,
        "sort": "asc",
    }
    url = f"{_DATA_BASE}/bars"
    next_token: str | None = None
    page_num = 0
    while True:
        if next_token:
            params["page_token"] = next_token
        elif "page_token" in params:
            del params["page_token"]
        try:
            r = requests.get(url, headers=_headers(settings), params=params, timeout=60)
        except requests.RequestException as exc:
            _LOG.warning(
                "[premarket] bars network error feed=%s window_utc=[%s,%s] symbols=%s err=%s",
                feed,
                start_s,
                end_s,
                sym_param,
                exc,
            )
            return out
        page_num += 1
        body_preview = _body_preview(r.text)
        try:
            payload = r.json() if r.text else {}
        except ValueError:
            _LOG.warning(
                "[premarket] bars non-JSON feed=%s status=%s url=%s body_preview=%s",
                feed,
                r.status_code,
                r.url,
                body_preview,
            )
            return out
        if r.status_code >= 400:
            _LOG.warning(
                "[premarket] bars HTTP error status=%s feed=%s window_utc=[%s,%s] url=%s "
                "body_preview=%s",
                r.status_code,
                feed,
                start_s,
                end_s,
                r.url,
                body_preview,
            )
            return out
        if not isinstance(payload, dict):
            _LOG.warning(
                "[premarket] bars unexpected JSON type feed=%s url=%s body_preview=%s",
                feed,
                r.url,
                body_preview,
            )
            return out
        bars_block = payload.get("bars") or {}
        if not isinstance(bars_block, dict):
            bars_block = {}
        chunk_counts = _bars_block_row_counts(bars_block, symbols)
        total_chunk_rows = sum(chunk_counts.values())
        _LOG.info(
            "[premarket] Alpaca GET bars page=%s status=%s feed=%s window_utc=[%s,%s] "
            "symbols=%s rows_this_page_by_symbol=%s next_page_token=%s url=%s",
            page_num,
            r.status_code,
            feed,
            start_s,
            end_s,
            sym_param,
            chunk_counts,
            bool(payload.get("next_page_token")),
            r.url,
        )
        if total_chunk_rows == 0 and sym_param:
            _LOG.info(
                "[premarket] Alpaca returned zero bar rows for this page; body_preview=%s",
                body_preview,
            )
            if feed == "iex":
                _LOG.warning(
                    "[premarket] feed=iex: minute bars are IEX-venue only; premarket (04:00–09:30 ET) "
                    "often has no bars for many symbols vs consolidated SIP. "
                    "For full extended-hours tape set STOCKBOT_ALPACA_DATA_FEED=sip (requires SIP entitlement)."
                )
        for sym, rows in bars_block.items():
            sym_u = sym.upper()
            if sym_u not in out:
                continue
            if not rows:
                continue
            frame_rows = []
            for b in rows:
                frame_rows.append(
                    {
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": int(b.get("v", 0)),
                        "time": pd.Timestamp(b["t"]),
                    }
                )
            df = pd.DataFrame(frame_rows).set_index("time").sort_index()
            out[sym_u] = pd.concat([out[sym_u], df]).sort_index()
        next_token = payload.get("next_page_token")
        if not next_token:
            break
    return out


def _filter_premarket_session(df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Keep only bars whose start is in [04:00, 09:30) ET on ``trade_date`` (i.e. through 09:29)."""
    if df is None or df.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(df.index, utc=True).tz_convert(_ET)
    start_et = datetime.combine(trade_date, time(4, 0), tzinfo=_ET)
    open_et = datetime.combine(trade_date, time(9, 30), tzinfo=_ET)
    mask = pd.Series(
        [bool(ts.date() == trade_date and start_et <= ts < open_et) for ts in idx],
        index=df.index,
    )
    sub = df.loc[mask]
    if df is not None and not df.empty and sub.empty:
        idx = pd.to_datetime(df.index, utc=True).tz_convert(_ET)
        first_et = idx.min()
        last_et = idx.max()
        _LOG.warning(
            "[premarket] session filter removed all %s bars for trade_date=%s "
            "(first_bar_et=%s last_bar_et=%s window_et=[04:00,09:30))",
            len(df),
            trade_date.isoformat(),
            first_et.isoformat(),
            last_et.isoformat(),
        )
    return sub.sort_index() if not sub.empty else sub


def _ts_iso_utc(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat().replace("+00:00", "Z")


def _row_from_filtered(symbol: str, trade_date: date, sub: pd.DataFrame) -> dict[str, Any]:
    trade_date_s = trade_date.isoformat()
    sym_u = symbol.upper()
    nulls = {
        "pm_open": None,
        "pm_high": None,
        "pm_low": None,
        "pm_close": None,
        "pm_volume": None,
        "first_bar_ts": None,
        "last_bar_ts": None,
        "bar_count": 0,
    }
    if sub is None or sub.empty:
        return {
            "symbol": sym_u,
            "trade_date": trade_date_s,
            "status": "empty",
            "reason": "no_bars_in_premarket_window",
            **nulls,
        }
    o = sub["open"].astype(float)
    h = sub["high"].astype(float)
    lo = sub["low"].astype(float)
    c = sub["close"].astype(float)
    v = sub["volume"].astype(float)
    first_i = sub.index[0]
    last_i = sub.index[-1]
    return {
        "symbol": sym_u,
        "trade_date": trade_date_s,
        "status": "ok",
        "reason": None,
        "pm_open": float(o.iloc[0]),
        "pm_high": float(h.max()),
        "pm_low": float(lo.min()),
        "pm_close": float(c.iloc[-1]),
        "pm_volume": float(v.sum()),
        "first_bar_ts": _ts_iso_utc(first_i),
        "last_bar_ts": _ts_iso_utc(last_i),
        "bar_count": int(len(sub)),
    }


def _empty_row(symbol: str, trade_date: date, status: str, reason: str, alpaca_feed: str) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "trade_date": trade_date.isoformat(),
        "status": status,
        "reason": reason,
        "alpaca_feed": alpaca_feed,
        "pm_open": None,
        "pm_high": None,
        "pm_low": None,
        "pm_close": None,
        "pm_volume": None,
        "first_bar_ts": None,
        "last_bar_ts": None,
        "bar_count": 0,
    }


def fetch_premarket_for_watchlist(
    settings: Settings,
    trade_date: date,
    symbols: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Per watchlist symbol: minute bars for ``trade_date`` premarket (04:00–09:29 ET), aggregated OHLCV.

    ``status`` reflects tape availability (``ok`` vs ``empty``, etc.). Feed choice is recorded on each row
    and in ``premarket_ingestion_diag``; non-SIP feeds are warned once but no longer force ``wrong_feed``.
    """
    wanted = [s.upper() for s in symbols]
    seen: set[str] = set()
    ordered: list[str] = []
    for s in wanted:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    feed = _feed(settings)
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        return {
            s: _empty_row(s, trade_date, "config_error", "missing_alpaca_keys", feed) for s in ordered
        }

    start_utc, end_utc = _premarket_window_utc_bounds(trade_date)
    window_start_utc = start_utc.isoformat().replace("+00:00", "Z")
    window_end_utc = end_utc.isoformat().replace("+00:00", "Z")
    et_today = datetime.now(_ET).date()
    if trade_date > et_today:
        _LOG.warning(
            "[premarket] trade_date=%s is after America/New_York today=%s — Alpaca typically "
            "returns no minute bars for sessions that have not occurred yet.",
            trade_date.isoformat(),
            et_today.isoformat(),
        )
    _LOG.info(
        "[premarket] Step1 window trade_date=%s feed=%s window_utc=[%s,%s) (04:00–09:30 ET)",
        trade_date.isoformat(),
        feed,
        window_start_utc,
        window_end_utc,
    )
    frames: dict[str, pd.DataFrame] = {s: pd.DataFrame() for s in ordered}
    chunk_size = 12
    try:
        for i in range(0, len(ordered), chunk_size):
            chunk_syms = ordered[i : i + chunk_size]
            part = _fetch_minute_bars_chunk(chunk_syms, start_utc, end_utc, settings)
            for sym in chunk_syms:
                frames[sym] = part.get(sym, pd.DataFrame())
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("[premarket] unexpected failure fetching minute bars: %s", exc)
        return {
            s: _empty_row(s, trade_date, "fetch_failed", f"fetch_exception:{exc!r}", feed) for s in ordered
        }

    out: dict[str, dict[str, Any]] = {}
    for sym in ordered:
        raw = frames.get(sym, pd.DataFrame())
        filtered = _filter_premarket_session(raw, trade_date)
        row = _row_from_filtered(sym, trade_date, filtered)
        row["alpaca_feed"] = feed
        row["premarket_ingestion_diag"] = {
            "alpaca_endpoint": f"{_DATA_BASE}/bars",
            "request_params_template": {
                "timeframe": "1Min",
                "start": window_start_utc,
                "end": window_end_utc,
                "adjustment": "raw",
                "feed": feed,
                "sort": "asc",
                "limit": 10000,
            },
            "api_raw_bar_count_before_filter": int(len(raw)),
            "bar_count_after_session_filter": int(len(filtered)),
            "iex_premarket_note": (
                "IEX feed is venue-limited; extended-hours 1Min bars are often empty vs SIP "
                "(consolidated tape). Alpaca SIP subscription required for reliable premarket bars."
                if feed == "iex"
                else None
            ),
        }
        out[sym] = row
    if feed != "sip":
        _LOG.warning(
            "[premarket] Alpaca data feed is %s (not sip); SIP is recommended for consolidated "
            "premarket tape. Step 1 rows use returned bars only.",
            feed,
        )
    return out
