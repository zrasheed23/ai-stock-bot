"""Regular-session 1-minute OHLCV from Alpaca (shared by replay and mid-morning enrichment)."""

from __future__ import annotations

import logging
import time as _time_stdlib
from datetime import date, datetime, time, timezone
from typing import Any

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from stockbot.config import Settings

_LOG = logging.getLogger("stockbot.ingestion.rth_minute_bars")

_ET = ZoneInfo("America/New_York")
_DATA_BASE = "https://data.alpaca.markets/v2/stocks"


def _feed(settings: Settings) -> str:
    return (settings.alpaca_data_feed or "sip").strip().lower() or "sip"


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


def _body_preview(text: str, limit: int = 400) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _get_with_429_retries(
    url: str,
    headers: dict[str, str],
    params: dict[str, Any],
    *,
    symbol: str,
    max_attempts: int = 8,
) -> requests.Response:
    """Alpaca data tier often returns 429 under burst replay; retry without changing bar semantics."""
    last: requests.Response | None = None
    for attempt in range(max_attempts):
        try:
            last = requests.get(url, headers=headers, params=params, timeout=60)
        except requests.RequestException as exc:
            _LOG.warning("[rth_1min] fetch failed %s: %s", symbol, exc)
            raise
        if last.status_code != 429:
            return last
        if attempt + 1 >= max_attempts:
            break
        delay_s = min(60.0, 1.5**attempt)
        _LOG.warning(
            "[rth_1min] HTTP 429 symbol=%s page retry %s/%s sleep=%.1fs",
            symbol,
            attempt + 1,
            max_attempts,
            delay_s,
        )
        _time_stdlib.sleep(delay_s)
    assert last is not None
    return last


def fetch_rth_1min_bars_range(
    symbol: str,
    session_date: date,
    settings: Settings,
    *,
    start_et: time,
    end_et: time,
) -> pd.DataFrame:
    """
    Minute bars [start_et, end_et) on ``session_date`` in America/New_York.

    ``end_et`` is exclusive (e.g. time(10, 31) includes the 10:30 bar).
    Returns ascending UTC-indexed OHLCV (empty if unavailable).
    """
    sym_u = symbol.strip().upper()
    start_local = datetime.combine(session_date, start_et, tzinfo=_ET)
    end_local = datetime.combine(session_date, end_et, tzinfo=_ET)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    start_s = start_utc.isoformat().replace("+00:00", "Z")
    end_s = end_utc.isoformat().replace("+00:00", "Z")
    params: dict[str, Any] = {
        "symbols": sym_u,
        "timeframe": "1Min",
        "start": start_s,
        "end": end_s,
        "limit": 10000,
        "adjustment": "raw",
        "feed": _feed(settings),
        "sort": "asc",
    }
    url = f"{_DATA_BASE}/bars"
    frames: list[pd.DataFrame] = []
    next_token: str | None = None
    while True:
        if next_token:
            params["page_token"] = next_token
        elif "page_token" in params:
            del params["page_token"]
        try:
            r = _get_with_429_retries(url, _headers(settings), params, symbol=sym_u)
        except requests.RequestException as exc:
            _LOG.warning("[rth_1min] fetch failed %s: %s", sym_u, exc)
            break
        try:
            payload = r.json() if r.text else {}
        except ValueError:
            _LOG.warning("[rth_1min] non-JSON status=%s body=%s", r.status_code, _body_preview(r.text))
            break
        if r.status_code >= 400:
            _LOG.warning("[rth_1min] HTTP %s symbol=%s body=%s", r.status_code, sym_u, _body_preview(r.text))
            break
        bars_block = payload.get("bars") if isinstance(payload, dict) else None
        if not isinstance(bars_block, dict):
            break
        rows_raw = bars_block.get(sym_u) or bars_block.get(symbol)
        if rows_raw:
            chunk: list[dict[str, Any]] = []
            for b in rows_raw:
                chunk.append(
                    {
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": int(b.get("v", 0)),
                        "time": pd.Timestamp(b["t"]),
                    }
                )
            df = pd.DataFrame(chunk).set_index("time").sort_index()
            frames.append(df)
        next_token = payload.get("next_page_token") if isinstance(payload, dict) else None
        if not next_token:
            break
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def fetch_rth_1min_full_session(symbol: str, session_date: date, settings: Settings) -> pd.DataFrame:
    """Convenience: full regular session [09:30 ET, 16:05 ET)."""
    return fetch_rth_1min_bars_range(
        symbol,
        session_date,
        settings,
        start_et=time(9, 30),
        end_et=time(16, 5),
    )
