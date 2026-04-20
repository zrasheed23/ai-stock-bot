"""
News ingestion — real headlines when FINNHUB_API_KEY is set, otherwise safe placeholders.

Flow (high level):
  1. Pipeline calls fetch_news(...) with your watchlist and dates.
  2. This module returns NewsItem rows (symbol, headline, source, time, url).
  3. NewsFilingsProcessor passes those headlines to Claude (or to a small offline fallback)
     so sentiment + risk_flags stay structured JSON in the audit trail.

Finnhub company-news docs: https://finnhub.io/docs/api/company-news
(Free tier has rate limits; we fetch once per symbol and cap article count.)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from stockbot.config import Settings
from stockbot.models import NewsItem

_LOG = logging.getLogger("stockbot.ingestion.news")

# NOTE ON DEBUGGING 401s
# ---------------------
# If curl works but requests gets 401/403, a common cause is that Python is
# implicitly using proxy environment variables (HTTPS_PROXY, etc.) while your
# curl test is not. To make our request behave like a simple curl invocation,
# we set Session.trust_env=False (disables proxy/env injection by requests).

# How far back to search relative to the pipeline's trade_date (calendar days).
_LOOKBACK_DAYS = 7
# Cap per symbol so prompts and API stay bounded.
_MAX_ARTICLES_PER_SYMBOL = 15


def _utc_from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc)


def _clean_value(value: str) -> str:
    """Trim whitespace and a single layer of matching quotes from env/file values."""
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _clean_token(token: str) -> tuple[str, bool]:
    """
    Finnhub tokens should not contain whitespace/newlines. If they do, we remove
    whitespace characters as a last-resort cleanup and report that we did so.
    """
    t = _clean_value(token)
    had_ws = any(ch.isspace() for ch in t)
    if had_ws:
        t = "".join(ch for ch in t if not ch.isspace())
    return t, had_ws


def _mask_token_in_url(url: str) -> str:
    """Log-friendly URL with the Finnhub token redacted."""
    parts = urlsplit(url)
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key == "token":
            query_items.append((key, "***MASKED***"))
        else:
            query_items.append((key, value))
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query_items), parts.fragment)
    )


def _stub_items(symbols: Iterable[str], run_dt: datetime, reason: str) -> list[NewsItem]:
    """Last-resort rows so the rest of the pipeline always has a list to iterate."""
    out: list[NewsItem] = []
    for sym in symbols:
        sym_u = sym.upper()
        out.append(
            NewsItem(
                symbol=sym_u,
                headline=f"{sym_u}: no live headlines available ({reason}).",
                source="stub",
                published_at=run_dt,
                url=None,
                note=reason,
            )
        )
    return out


def _fetch_finnhub_for_symbol(
    symbol: str,
    api_key: str,
    date_from: date,
    date_to: date,
) -> tuple[list[NewsItem], str | None]:
    """
    One Finnhub request per symbol. Returns (items, error_message_or_none).
    On failure we do not raise — the caller merges errors into the audit meta.
    """
    url = "https://finnhub.io/api/v1/company-news"
    clean_symbol = _clean_value(symbol).upper()
    clean_token, token_had_ws = _clean_token(api_key)
    # IMPORTANT: use two completely separate dicts.
    # - `request_params` holds the REAL token and is the ONLY dict passed into requests.
    # - `log_params` is for humans/logs only and must never be passed to requests.Request.
    #
    # Why: `requests.Request(..., params=some_dict)` keeps a reference to that dict. If we
    # ever mutated the same dict to mask the token for logging *before* prepare(), we could
    # accidentally send token=***MASKED*** to Finnhub (401). Keeping dicts separate prevents that.
    date_from_s = date_from.isoformat()
    date_to_s = date_to.isoformat()
    request_params = {
        "symbol": clean_symbol,
        "from": date_from_s,
        "to": date_to_s,
        "token": clean_token,
    }
    log_params = {
        "symbol": clean_symbol,
        "from": date_from_s,
        "to": date_to_s,
        "token": "***MASKED***",
    }
    try:
        req = requests.Request("GET", url, params=request_params)
        prepared = req.prepare()
        prepared_url = prepared.url or url

        # Required debug prints (token masked). These should match curl format:
        # https://finnhub.io/api/v1/company-news?symbol=SPY&from=YYYY-MM-DD&to=YYYY-MM-DD&token=...
        _LOG.info("Finnhub params for %s (log copy, token masked): %s", clean_symbol, log_params)
        _LOG.info(
            "Finnhub prepared URL for %s (masked for log): %s",
            clean_symbol,
            _mask_token_in_url(prepared_url),
        )
        if token_had_ws:
            _LOG.warning("Finnhub token contained whitespace; it was removed before request.")

        with requests.Session() as session:
            session.trust_env = False  # avoid proxy/env surprises; closer to plain curl
            r = session.send(prepared, timeout=25)

        _LOG.info("Finnhub status for %s: %s", clean_symbol, r.status_code)
        if r.status_code == 429:
            _LOG.warning("Finnhub non-200 body for %s: %s", clean_symbol, (r.text or "")[:300])
            return [], "rate_limited"
        if r.status_code != 200:
            _LOG.warning("Finnhub non-200 body for %s: %s", clean_symbol, (r.text or "")[:300])
            return [], f"http_{r.status_code}"
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        # Keep this quiet-ish: we capture the failure in `meta` for the audit trail.
        _LOG.warning("Finnhub request failed for %s: %s", clean_symbol, exc)
        return [], "request_failed"

    if not isinstance(data, list):
        # Finnhub should return a JSON array for this endpoint.
        _LOG.warning(
            "Finnhub unexpected JSON shape for %s: %s (first 300 chars: %s)",
            clean_symbol,
            type(data).__name__,
            str(data)[:300],
        )
        return [], "unexpected_response_shape"

    _LOG.info("Finnhub articles for %s: %d", clean_symbol, len(data))

    items: list[NewsItem] = []
    for row in data[:_MAX_ARTICLES_PER_SYMBOL]:
        if not isinstance(row, dict):
            continue
        headline = (row.get("headline") or row.get("summary") or "").strip()
        if not headline:
            continue
        ts = row.get("datetime")
        try:
            pub = _utc_from_unix(ts) if ts is not None else datetime.now(timezone.utc)
        except (TypeError, ValueError, OSError):
            pub = datetime.now(timezone.utc)
        src = str(row.get("source") or "finnhub")
        link = row.get("url")
        items.append(
            NewsItem(
                symbol=clean_symbol,
                headline=headline[:2000],
                source=src,
                published_at=pub,
                url=str(link)[:2000] if link else None,
                note=None,
            )
        )

    return items, None


def fetch_news(
    symbols: Iterable[str],
    run_date: datetime | None = None,
    trade_date: date | None = None,
    settings: Settings | None = None,
) -> tuple[list[NewsItem], dict[str, Any]]:
    """
    Pull recent company news for each symbol.

    Returns:
        (news_items, meta) — meta is small JSON-friendly dict for your audit file.

    If FINNHUB_API_KEY is missing or Finnhub errors out, we still return stub rows
    so strategy/risk code never crashes on an empty list.
    """
    run_date = run_date or datetime.now(timezone.utc)
    trade_date = trade_date or run_date.date()
    sym_list = [s.upper() for s in symbols]
    settings = settings or Settings.from_env()

    meta: dict[str, Any] = {
        "provider": "finnhub",
        "trade_date": trade_date.isoformat(),
        "window_days": _LOOKBACK_DAYS,
        "symbols": sym_list,
        "per_symbol": {},
        "errors": [],
    }

    api_key = (settings.finnhub_api_key or "").strip()
    if not api_key:
        meta["provider"] = "stub"
        meta["errors"].append("FINNHUB_API_KEY not set — using placeholder headlines")
        _LOG.info("News: no FINNHUB_API_KEY; using stub headlines")
        return _stub_items(sym_list, run_date, "no FINNHUB_API_KEY"), meta

    date_to = trade_date
    date_from = date_to - timedelta(days=_LOOKBACK_DAYS)
    if date_from > date_to:
        date_from = date_to

    all_items: list[NewsItem] = []
    real_article_count = 0
    for sym in sym_list:
        rows, err = _fetch_finnhub_for_symbol(sym, api_key, date_from, date_to)
        meta["per_symbol"][sym] = {"count": len(rows), "error": err}
        if err:
            meta["errors"].append(f"{sym}: {err}")
            # Graceful degradation: one clear placeholder for this symbol only.
            all_items.append(
                NewsItem(
                    symbol=sym,
                    headline=f"{sym}: Finnhub returned no usable news ({err}).",
                    source="stub_fallback",
                    published_at=run_date,
                    url=None,
                    note=f"finnhub:{err}",
                )
            )
        elif not rows:
            all_items.append(
                NewsItem(
                    symbol=sym,
                    headline=f"{sym}: no articles in Finnhub window "
                    f"({date_from.isoformat()} .. {date_to.isoformat()}).",
                    source="finnhub_empty",
                    published_at=run_date,
                    url=None,
                    note="empty_window",
                )
            )
        else:
            all_items.extend(rows)
            real_article_count += len(rows)

    _LOG.info(
        "News: Finnhub ingested %d real articles for %d symbols",
        real_article_count,
        len(sym_list),
    )
    return all_items, meta
