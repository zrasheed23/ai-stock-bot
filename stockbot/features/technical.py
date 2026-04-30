"""Technical features from OHLCV — small set, easy to extend."""

from __future__ import annotations

import pandas as pd


def _sma(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return float(series.iloc[-1])
    return float(series.iloc[-window:].mean())


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0 or pd.isna(avg_loss):
        return 70.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr14_from_daily_bars(bars: pd.DataFrame) -> float | None:
    """
    Simple 14-period ATR: mean of the last 14 true ranges (daily bars).
    Returns None if insufficient data.
    """
    if bars is None or len(bars) < 15:
        return None
    h = bars["high"].astype(float)
    l = bars["low"].astype(float)
    c = bars["close"].astype(float)
    tr = _true_range(h, l, c)
    tail = tr.iloc[-14:]
    if tail.isna().any() or len(tail) < 14:
        return None
    v = float(tail.mean())
    if not (v > 0.0) or v != v:  # finite, positive
        return None
    return v


def _volatility(close: pd.Series, window: int = 20) -> float:
    if len(close) < 2:
        return 0.0
    use = close.iloc[-window:] if len(close) >= window else close
    rets = use.pct_change().dropna()
    if rets.empty:
        return 0.0
    return float(rets.std() * (252**0.5))  # annualized rough scale


def technical_features(bars: pd.DataFrame) -> dict[str, float]:
    """Return latest-row technical scalars for scoring."""
    close = bars["close"].astype(float)
    last = float(close.iloc[-1])
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    momentum_20 = float(last / close.iloc[-20] - 1.0) if len(close) >= 20 else 0.0
    out: dict[str, float] = {
        "last_close": last,
        "sma20": sma20,
        "sma50": sma50,
        "sma20_distance": float((last - sma20) / sma20) if sma20 else 0.0,
        "sma50_distance": float((last - sma50) / sma50) if sma50 else 0.0,
        "rsi14": _rsi(close, 14),
        "volatility_ann": _volatility(close, 20),
        "momentum_20d": momentum_20,
    }
    atr = atr14_from_daily_bars(bars)
    if atr is not None:
        out["atr14"] = atr
    return out
