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
    return {
        "last_close": last,
        "sma20": sma20,
        "sma50": sma50,
        "sma20_distance": float((last - sma20) / sma20) if sma20 else 0.0,
        "sma50_distance": float((last - sma50) / sma50) if sma50 else 0.0,
        "rsi14": _rsi(close, 14),
        "volatility_ann": _volatility(close, 20),
        "momentum_20d": momentum_20,
    }
