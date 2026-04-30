"""Pure pre-market feature transforms — no network I/O (unit-test friendly)."""

from __future__ import annotations

import math
from typing import Any

# Small additive score nudges (keeps existing score scale intact).
PM_SCORE_GAP_WARN = -0.08
PM_SCORE_GAP_CLEAN = 0.04
PM_SCORE_RVOL_STRONG = 0.05


def gap_fraction(prior_rth_close: float, pm_ref_price: float) -> float:
    """(ref - prior_close) / prior_close — signed gap as a fraction."""
    if prior_rth_close <= 0 or not math.isfinite(prior_rth_close) or not math.isfinite(pm_ref_price):
        return 0.0
    return float((pm_ref_price - prior_rth_close) / prior_rth_close)


def dollar_gap(prior_rth_close: float, pm_ref_price: float) -> float:
    return float(pm_ref_price - prior_rth_close)


def compute_gap_atr(
    prior_rth_close: float,
    pm_ref_price: float,
    atr14: float | None,
) -> tuple[float, bool, float]:
    """
    Returns (gap_atr, used_atr_fallback, gap_fraction).

    If ATR is missing or non-positive, gap_atr is derived from gap fraction capped at ±5%,
    scaled so that |gap_frac|==5% maps to |gap_atr|==2.5 (aligns with PM_GAP_ATR_HARD_SKIP).
    """
    gf = gap_fraction(prior_rth_close, pm_ref_price)
    if not math.isfinite(gf):
        return 0.0, True, 0.0
    if atr14 is not None and atr14 > 0.0 and math.isfinite(atr14):
        dg = dollar_gap(prior_rth_close, pm_ref_price)
        return float(dg / atr14), False, gf
    capped = max(-0.05, min(0.05, gf))
    # 5% -> 2.5 in gap_atr units for threshold compatibility
    return float(capped / 0.02), True, gf


def compute_pm_rvol(
    pm_session_volume: float,
    baseline_volume: float,
    *,
    baseline_is_placeholder: bool,
) -> float:
    if baseline_volume > 0.0 and math.isfinite(baseline_volume) and math.isfinite(pm_session_volume):
        return float(pm_session_volume / baseline_volume)
    return 1.0


def classify_premarket_hard_skip(
    gap_atr: float,
    pm_rvol: float,
    *,
    pm_gap_atr_hard_skip: float,
    pm_rvol_min_on_gap: float,
    pm_gap_atr_warn: float,
) -> str | None:
    """Return skip reason code or None."""
    if not math.isfinite(gap_atr) or not math.isfinite(pm_rvol):
        return None
    if gap_atr > pm_gap_atr_hard_skip or gap_atr < -pm_gap_atr_hard_skip:
        return "PM_GAP_TOO_LARGE"
    # Large extension (either direction) with weak pre-market participation.
    if abs(gap_atr) > pm_gap_atr_warn and pm_rvol < pm_rvol_min_on_gap:
        return "PM_LOW_VOLUME_ON_GAP"
    return None


def premarket_score_adjustment(
    gap_atr: float,
    pm_rvol: float,
    *,
    pm_gap_atr_hard_skip: float,
    pm_gap_atr_warn: float,
    pm_rvol_strong: float,
) -> float:
    """
    Additive adjustment to composite score (applied only if symbol is not hard-skipped).
    """
    if not math.isfinite(gap_atr) or not math.isfinite(pm_rvol):
        return 0.0
    adj = 0.0
    # Warn band: extension between warn and hard skip (hard skip removed earlier)
    if pm_gap_atr_warn <= gap_atr < pm_gap_atr_hard_skip:
        adj += PM_SCORE_GAP_WARN
    if -pm_gap_atr_hard_skip < gap_atr <= -pm_gap_atr_warn:
        adj += PM_SCORE_GAP_WARN
    # Clean open
    if abs(gap_atr) < 0.5:
        adj += PM_SCORE_GAP_CLEAN
    # Strong participation on moderate gap
    if pm_rvol > pm_rvol_strong and pm_gap_atr_warn > abs(gap_atr) >= 0.5:
        adj += PM_SCORE_RVOL_STRONG
    return float(adj)


def neutral_symbol_row(*, fetch_error: str | None = None) -> dict[str, Any]:
    return {
        "gap_atr": 0.0,
        "pm_rvol": 1.0,
        "gap_fraction": 0.0,
        "prior_rth_close": None,
        "pm_ref_price": None,
        "pm_session_volume": 0.0,
        "baseline_pm_volume": None,
        "baseline_is_placeholder": False,
        "hard_skip_reason": None,
        "fetch_error": fetch_error,
    }


def neutral_spy_row(*, fetch_error: str | None = None) -> dict[str, Any]:
    return {
        "gap_atr": 0.0,
        "gap_fraction": 0.0,
        "prior_rth_close": None,
        "pm_ref_price": None,
        "pm_session_volume": 0.0,
        "pm_rvol": 1.0,
        "fetch_error": fetch_error,
    }
