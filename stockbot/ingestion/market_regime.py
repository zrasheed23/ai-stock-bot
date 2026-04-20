"""QQQ-based market regime gate + defensive sleeve symbols (ingestion only, no scoring)."""

from __future__ import annotations

from datetime import date
from typing import Literal, Sequence

import pandas as pd

from stockbot.features.technical import technical_features
from stockbot.models import MarketSnapshot

RegimeTier = Literal["STRONG", "WEAK", "VERY_WEAK"]

# Must match ``DEFENSIVE`` names in ``stockbot.strategy.engine`` sector map (WEAK-regime demotion).
_DEFENSIVE_EQUITY_SYMBOLS = frozenset({"UNH", "LLY", "SPY"})


def is_defensive_equity(symbol: str) -> bool:
    """True for symbols treated as defensive sleeve vs risk-on tech (WEAK regime rank demotion)."""
    return symbol.upper() in _DEFENSIVE_EQUITY_SYMBOLS


def defensive_trade_symbols(watchlist: Sequence[str]) -> list[str]:
    """Watchlist names that count as defensive references (audit / regime payload)."""
    return [s for s in watchlist if is_defensive_equity(s)]


def _return_10d(close: pd.Series) -> float | None:
    if len(close) < 11:
        return None
    return float(close.iloc[-1] / close.iloc[-11] - 1.0)


def evaluate_qqq_regime_from_market(
    market: dict[str, MarketSnapshot],
    trade_date: date,
) -> tuple[RegimeTier, dict[str, object]]:
    """
    Classify broad tape using QQQ daily bars already loaded with the watchlist.

    Returns ``(tier, audit_dict)``. Audit keys are consumed by ``pipeline`` logging and JSON.
    """
    snap = market.get("QQQ")
    failures: list[str] = []
    audit: dict[str, object] = {
        "regime_tier": "STRONG",
        "trade_date": trade_date.isoformat(),
        "close_price": None,
        "sma20": None,
        "return_10d": None,
        "failure_reasons": failures,
    }

    if snap is None or snap.bars is None or snap.bars.empty:
        failures.append("missing_qqq")
        audit["regime_tier"] = "VERY_WEAK"
        audit["failure_reasons"] = list(failures)
        return "VERY_WEAK", audit

    bars = snap.bars
    close_s = bars["close"].astype(float)
    tech = technical_features(bars)
    last = float(tech["last_close"])
    sma20 = float(tech["sma20"])
    mom20 = float(tech["momentum_20d"])
    ret10 = _return_10d(close_s)

    audit["close_price"] = last
    audit["sma20"] = sma20
    audit["return_10d"] = ret10
    audit["sma20_distance"] = float(tech["sma20_distance"])
    audit["momentum_20d"] = mom20

    # VERY_WEAK: no meaningful QQQ edge — skip stock session (rare; multiple simultaneous stresses).
    if len(close_s) < 15:
        failures.append("insufficient_qqq_history")
        audit["regime_tier"] = "VERY_WEAK"
        audit["failure_reasons"] = list(failures)
        return "VERY_WEAK", audit

    if ret10 is None:
        failures.append("return_10d_unavailable")
        audit["regime_tier"] = "VERY_WEAK"
        audit["failure_reasons"] = list(failures)
        return "VERY_WEAK", audit

    very_weak = last < sma20 and ret10 < -0.05 and mom20 < -0.04
    if very_weak:
        failures.append("qqq_below_sma20_and_negative_return_10d_and_momentum")
        audit["regime_tier"] = "VERY_WEAK"
        audit["failure_reasons"] = list(failures)
        return "VERY_WEAK", audit

    # WEAK: soft risk-off tilt (rank demotion on non-defensive names); still trade.
    weak = last < sma20 or ret10 < -0.015 or mom20 < -0.02
    if weak:
        audit["regime_tier"] = "WEAK"
        audit["failure_reasons"] = list(failures)
        return "WEAK", audit

    audit["regime_tier"] = "STRONG"
    audit["failure_reasons"] = list(failures)
    return "STRONG", audit
