"""Unit tests for pure pre-market feature helpers."""

from __future__ import annotations

import unittest

from stockbot.config import PM_GAP_ATR_HARD_SKIP, PM_GAP_ATR_WARN, PM_RVOL_MIN_ON_GAP, PM_RVOL_STRONG
from stockbot.features.premarket import (
    classify_premarket_hard_skip,
    compute_gap_atr,
    compute_pm_rvol,
    gap_fraction,
    premarket_score_adjustment,
)


class TestPremarketFeatures(unittest.TestCase):
    def test_gap_fraction(self) -> None:
        self.assertAlmostEqual(gap_fraction(100.0, 102.0), 0.02)

    def test_gap_atr_with_atr(self) -> None:
        ga, fb, gf = compute_gap_atr(100.0, 102.5, 2.0)
        self.assertFalse(fb)
        self.assertAlmostEqual(gf, 0.025)
        self.assertAlmostEqual(ga, 1.25)

    def test_gap_atr_fallback_no_atr(self) -> None:
        ga, fb, _gf = compute_gap_atr(100.0, 105.0, None)
        self.assertTrue(fb)
        self.assertAlmostEqual(ga, 2.5)

    def test_pm_rvol(self) -> None:
        self.assertAlmostEqual(compute_pm_rvol(1000.0, 500.0, baseline_is_placeholder=True), 2.0)

    def test_hard_skip_gap(self) -> None:
        r = classify_premarket_hard_skip(
            3.0,
            1.0,
            pm_gap_atr_hard_skip=PM_GAP_ATR_HARD_SKIP,
            pm_rvol_min_on_gap=PM_RVOL_MIN_ON_GAP,
            pm_gap_atr_warn=PM_GAP_ATR_WARN,
        )
        self.assertEqual(r, "PM_GAP_TOO_LARGE")

    def test_hard_skip_low_vol_gap(self) -> None:
        r = classify_premarket_hard_skip(
            1.5,
            0.1,
            pm_gap_atr_hard_skip=PM_GAP_ATR_HARD_SKIP,
            pm_rvol_min_on_gap=PM_RVOL_MIN_ON_GAP,
            pm_gap_atr_warn=PM_GAP_ATR_WARN,
        )
        self.assertEqual(r, "PM_LOW_VOLUME_ON_GAP")

    def test_score_warn_positive(self) -> None:
        adj = premarket_score_adjustment(
            1.5,
            1.0,
            pm_gap_atr_hard_skip=PM_GAP_ATR_HARD_SKIP,
            pm_gap_atr_warn=PM_GAP_ATR_WARN,
            pm_rvol_strong=PM_RVOL_STRONG,
        )
        self.assertLess(adj, 0.0)

    def test_score_clean(self) -> None:
        adj = premarket_score_adjustment(
            0.2,
            1.0,
            pm_gap_atr_hard_skip=PM_GAP_ATR_HARD_SKIP,
            pm_gap_atr_warn=PM_GAP_ATR_WARN,
            pm_rvol_strong=PM_RVOL_STRONG,
        )
        self.assertGreater(adj, 0.0)


if __name__ == "__main__":
    unittest.main()
