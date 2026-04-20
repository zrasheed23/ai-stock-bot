"""Market bars: Alpaca IEX/SIP when keys exist; optional synthetic sample for local dry-run."""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd
import requests

from stockbot.config import Settings
from stockbot.models import MarketSnapshot

_LOG = logging.getLogger("stockbot.ingestion.market")


def _synthetic_bars(symbol: str, end: datetime, n: int = 40) -> pd.DataFrame:
    """
    Reproducible pseudo-price series when no live market API is used.

    The seed includes the series *end* calendar day so multi-day simulations do not
    reuse the same OHLCV path for every trade_date (which made technical scores flat).
    """
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    anchor = end_utc.date().isoformat()
    seed = int(hashlib.sha256(f"{symbol}:{anchor}".encode()).hexdigest()[:8], 16)
    rng = pd.Series(range(n), dtype=float)
    base = 100.0 + (seed % 50)
    noise = (rng * 0.37 + seed % 7) * 0.02
    close = base + noise.cumsum() * 0.5
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": (1e6 + (seed % 1000) * 1000 + rng * 1e3).astype(int),
        }
    )
    idx = pd.date_range(end=end_utc, periods=n, freq="B", tz="UTC")
    df.index = idx[: len(df)]
    return df


def _ohlc_passes_realism_check(bars: pd.DataFrame) -> bool:
    """
    Reject constant / degenerate series that are not credible for evaluation.

    Synthetic bars are smooth and highly regular; real markets usually show more
    variation in daily returns when enough history is present.
    """
    if bars is None or len(bars) < 3:
        return False
    c = bars["close"].astype(float)
    if float(c.min()) <= 0:
        return False
    r = c.pct_change().dropna()
    if len(r) < 2:
        return False
    if float(r.std()) < 1e-9:
        return False
    # Near-constant returns (typical of toy series)
    if float(r.std()) < 1e-6 and float((r - r.mean()).abs().max()) < 1e-6:
        return False
    return True


def _alpaca_bars(
    symbol: str,
    settings: Settings,
    end: datetime,
    lookback_days: int,
) -> pd.DataFrame:
    """Fetch daily bars from Alpaca data API v2. Returns empty DataFrame if no bars (never synthetic)."""
    start = end - timedelta(days=lookback_days * 2)
    data_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "limit": 10000,
        "adjustment": "split",
    }
    auth = (settings.alpaca_api_key, settings.alpaca_secret_key)
    r = requests.get(data_url, params=params, auth=auth, timeout=30)
    r.raise_for_status()
    payload = r.json()
    bars = payload.get("bars") or []
    if not bars:
        return pd.DataFrame()
    rows = []
    for b in bars:
        rows.append(
            {
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "volume": int(b["v"]),
                "time": pd.Timestamp(b["t"]),
            }
        )
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


def fetch_market_snapshots(
    symbols: Iterable[str],
    settings: Settings,
    as_of: datetime | None = None,
    lookback_days: int = 120,
    allow_synthetic: bool | None = None,
) -> tuple[dict[str, MarketSnapshot], dict[str, Any]]:
    """
    Returns (snapshots, meta) so the pipeline can audit whether bars came from Alpaca or synthetic.

    ``allow_synthetic``:
      - ``None`` (default): synthetic allowed unless env ``STOCKBOT_REQUIRE_REAL_MARKET`` is truthy
        (backtest runner sets this for evaluation runs).
      - ``False``: never substitute synthetic bars; symbols without valid Alpaca OHLC are omitted.

    Always pass ``as_of`` anchored to the *simulated* trade day when backtesting.
    """
    if allow_synthetic is None:
        allow_synthetic = not os.environ.get("STOCKBOT_REQUIRE_REAL_MARKET", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    as_of = as_of or datetime.now(timezone.utc)
    out: dict[str, MarketSnapshot] = {}
    meta: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "allow_synthetic": allow_synthetic,
        "force_synthetic": bool(os.environ.get("STOCKBOT_FORCE_SYNTHETIC_MARKET")),
        "per_symbol": {},
    }
    use_alpaca = bool(settings.alpaca_api_key and settings.alpaca_secret_key)
    force_synth = bool(os.environ.get("STOCKBOT_FORCE_SYNTHETIC_MARKET"))

    for sym in symbols:
        sym_u = sym.upper()
        sym_meta: dict[str, Any] = {
            "source": "none",
            "bar_count": 0,
            "first_bar_ts": None,
            "last_bar_ts": None,
            "real_data_valid": False,
            "skip_reason": None,
        }

        if use_alpaca and not force_synth:
            try:
                bars = _alpaca_bars(sym_u, settings, as_of, lookback_days)
                if bars.empty:
                    sym_meta["source"] = "alpaca_empty"
                    sym_meta["skip_reason"] = "alpaca_returned_zero_bars"
                    if allow_synthetic:
                        bars = _synthetic_bars(sym_u, as_of)
                        sym_meta["source"] = "synthetic_empty_fallback"
                        sym_meta["bar_count"] = int(len(bars))
                        sym_meta["first_bar_ts"] = str(bars.index[0])
                        sym_meta["last_bar_ts"] = str(bars.index[-1])
                        sym_meta["real_data_valid"] = False
                    else:
                        _LOG.warning(
                            "REAL DATA MISSING - skipping %s: Alpaca returned no bars (as_of=%s)",
                            sym_u,
                            as_of.date(),
                        )
                elif not _ohlc_passes_realism_check(bars):
                    sym_meta["source"] = "alpaca_rejected"
                    sym_meta["skip_reason"] = "failed_realism_check"
                    sym_meta["bar_count"] = int(len(bars))
                    sym_meta["first_bar_ts"] = str(bars.index[0])
                    sym_meta["last_bar_ts"] = str(bars.index[-1])
                    if allow_synthetic:
                        bars = _synthetic_bars(sym_u, as_of)
                        sym_meta["source"] = "synthetic_validation_fallback"
                        sym_meta["bar_count"] = int(len(bars))
                        sym_meta["first_bar_ts"] = str(bars.index[0])
                        sym_meta["last_bar_ts"] = str(bars.index[-1])
                        sym_meta["real_data_valid"] = False
                    else:
                        _LOG.warning(
                            "REAL DATA INVALID - skipping %s: OHLC failed validation (as_of=%s)",
                            sym_u,
                            as_of.date(),
                        )
                        bars = pd.DataFrame()
                else:
                    sym_meta["source"] = "alpaca"
                    sym_meta["bar_count"] = int(len(bars))
                    sym_meta["first_bar_ts"] = str(bars.index[0])
                    sym_meta["last_bar_ts"] = str(bars.index[-1])
                    sym_meta["real_data_valid"] = True
            except (requests.RequestException, KeyError, ValueError) as exc:
                sym_meta["source"] = "alpaca_error"
                sym_meta["skip_reason"] = repr(exc)
                if allow_synthetic:
                    bars = _synthetic_bars(sym_u, as_of)
                    sym_meta["source"] = "synthetic_fallback"
                    sym_meta["bar_count"] = int(len(bars))
                    sym_meta["first_bar_ts"] = str(bars.index[0])
                    sym_meta["last_bar_ts"] = str(bars.index[-1])
                    sym_meta["real_data_valid"] = False
                else:
                    _LOG.warning(
                        "REAL DATA MISSING - skipping %s: Alpaca error (%s)",
                        sym_u,
                        exc,
                    )
                    bars = pd.DataFrame()
        else:
            if allow_synthetic:
                bars = _synthetic_bars(sym_u, as_of)
                sym_meta["source"] = "synthetic_forced" if use_alpaca else "synthetic_no_keys"
                sym_meta["bar_count"] = int(len(bars))
                sym_meta["first_bar_ts"] = str(bars.index[0])
                sym_meta["last_bar_ts"] = str(bars.index[-1])
                sym_meta["real_data_valid"] = False
            else:
                sym_meta["source"] = "synthetic_disabled"
                sym_meta["skip_reason"] = "no_alpaca_credentials_or_forced_off"
                bars = pd.DataFrame()
                _LOG.warning(
                    "REAL DATA MISSING - skipping %s: synthetic disabled and Alpaca unavailable",
                    sym_u,
                )

        meta["per_symbol"][sym_u] = sym_meta

        if bars is not None and not bars.empty:
            out[sym_u] = MarketSnapshot(symbol=sym_u, as_of=as_of, bars=bars)

    return out, meta

