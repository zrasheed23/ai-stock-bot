"""Map strategy + risk outputs to broker-neutral order intents (never from LLM text)."""

from __future__ import annotations

import logging
from typing import Sequence

from stockbot.models import OrderIntent, RiskVerdict, ScoredCandidate

_LOG = logging.getLogger("stockbot.execution.orders")

# Coarse sector buckets (must match ``stockbot.strategy.engine`` theme overlap map).
_SYMBOL_SECTOR_THEME: dict[str, str] = {}
for _s in ("AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "TSLA"):
    _SYMBOL_SECTOR_THEME[_s] = "TECH"
for _s in ("XOM", "CVX", "XLE"):
    _SYMBOL_SECTOR_THEME[_s] = "ENERGY"
for _s in ("JPM", "GS", "XLF"):
    _SYMBOL_SECTOR_THEME[_s] = "FINANCIALS"
for _s in ("UNH", "LLY", "SPY"):
    _SYMBOL_SECTOR_THEME[_s] = "DEFENSIVE"


def _slot1_slot2_same_sector_theme(c1: ScoredCandidate, c2: ScoredCandidate) -> bool:
    b1 = _SYMBOL_SECTOR_THEME.get(c1.symbol.upper())
    b2 = _SYMBOL_SECTOR_THEME.get(c2.symbol.upper())
    return b1 is not None and b2 is not None and b1 == b2


# -----------------------------------------------------------------------------
# Same-sector two-leg days: slot-2 weight is 0.30 only when slot-1 and slot-2 raw scores are
# very close (score_gap = slot1_raw - slot2_raw <= OVERLAP_PAIR_ELITE_MAX_SCORE_GAP); otherwise
# slot-2 weight is 0.25. Non-overlapping pairs: gap sizing only (0.30 / 0.25 / 0.20).
# -----------------------------------------------------------------------------
OVERLAP_PAIR_ELITE_MAX_SCORE_GAP = 0.60

overlap_two_leg_sessions = 0
overlap_elite_full_sleeve_count = 0
overlap_capped_slot2_to_25_count = 0


def reset_overlap_slot2_stats() -> None:
    global overlap_two_leg_sessions, overlap_elite_full_sleeve_count, overlap_capped_slot2_to_25_count
    overlap_two_leg_sessions = 0
    overlap_elite_full_sleeve_count = 0
    overlap_capped_slot2_to_25_count = 0


def get_overlap_two_leg_sessions() -> int:
    return overlap_two_leg_sessions


def get_overlap_elite_full_sleeve_count() -> int:
    return overlap_elite_full_sleeve_count


def get_overlap_capped_slot2_to_25_count() -> int:
    return overlap_capped_slot2_to_25_count


# When two names are selected the same day, allocate more sleeve to trade #1 (strategy rank).
TRADE_ONE_CAPITAL_FRACTION = 0.7
TRADE_TWO_CAPITAL_FRACTION = 0.3

# -----------------------------------------------------------------------------
# Slot-2 conviction sizing only (raw ``ScoredCandidate.score``; slot 1 fixed at 0.7).
# Default 0.30. Shrink only when slot-2 is BOTH (a) meaningfully behind slot-1 on raw score gap
# AND (b) not “strong” in absolute terms (below STRONG_MIN). If (b) is false, gap is ignored —
# avoids punishing large gaps when #2 is still a high-conviction trade.
# When (b) is true, gap tiers are tighter than the “strong #2” path so a weak second with a
# moderate gap (e.g. #1 also softer) can still step down to 0.25 / 0.20. No 0.15 sleeve.
#
# STRONG_MIN ≈ 1.48: ~0.28 above strategy ``min_second_trade_score`` (1.2); clears the clutter
# of marginal seconds while keeping almost all dual-winner days at full 0.30.
# Weak-branch gaps: keep-full ≤0.28, mild ≤0.48 — calibrated on two-leg audit days so only a
# handful of genuinely soft seconds leave 0.30.
# -----------------------------------------------------------------------------
SLOT2_ABSOLUTE_STRONG_MIN_RAW = 1.48
SLOT2_GAP_WHEN_WEAK_KEEP_FULL = 0.28
SLOT2_GAP_WHEN_WEAK_REDUCE_MILD = 0.48


def slot2_capital_fraction_from_raw_score_gap(slot1_raw_score: float, slot2_raw_score: float) -> float:
    """Two picks: 0.30 default; 0.25 / 0.20 only when #2 raw score is below strong floor AND gap says so."""
    s1 = float(slot1_raw_score)
    s2 = float(slot2_raw_score)
    if s2 >= SLOT2_ABSOLUTE_STRONG_MIN_RAW:
        return 0.30
    gap = max(0.0, s1 - s2)
    if gap <= SLOT2_GAP_WHEN_WEAK_KEEP_FULL:
        return 0.30
    if gap <= SLOT2_GAP_WHEN_WEAK_REDUCE_MILD:
        return 0.25
    return 0.20


def capital_fractions_for_chosen_slots(chosen: Sequence[ScoredCandidate]) -> list[float]:
    """Per-slot fractions for ``decision.chosen`` order; single name = full 1.0 sleeve."""
    global overlap_two_leg_sessions, overlap_elite_full_sleeve_count, overlap_capped_slot2_to_25_count
    n = len(chosen)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    w2 = slot2_capital_fraction_from_raw_score_gap(chosen[0].score, chosen[1].score)
    s1 = float(chosen[0].score)
    s2 = float(chosen[1].score)
    sym1 = chosen[0].symbol
    sym2 = chosen[1].symbol
    sec1 = _SYMBOL_SECTOR_THEME.get(sym1.upper())
    sec2 = _SYMBOL_SECTOR_THEME.get(sym2.upper())
    overlap = _slot1_slot2_same_sector_theme(chosen[0], chosen[1])

    _LOG.info(
        "OVERLAP_CHECK slot1=%s slot2=%s sector_slot1=%s sector_slot2=%s overlap=%s",
        sym1,
        sym2,
        sec1,
        sec2,
        overlap,
    )

    score_gap = s1 - s2
    elite_met = False
    reason = "non-overlap"
    if overlap:
        overlap_two_leg_sessions += 1
        elite_met = score_gap <= OVERLAP_PAIR_ELITE_MAX_SCORE_GAP
        if elite_met:
            w2 = 0.30
            overlap_elite_full_sleeve_count += 1
            reason = "overlap + elite"
        else:
            w2 = 0.25
            overlap_capped_slot2_to_25_count += 1
            reason = "overlap + reduced"

    _LOG.info(
        "SLOT2_WEIGHT_DECISION slot1_raw_score=%.6f slot2_raw_score=%.6f score_gap=%.6f elite=%s "
        "final_slot2_weight=%.4f reason=%s",
        s1,
        s2,
        score_gap,
        elite_met,
        w2,
        reason,
    )
    return [TRADE_ONE_CAPITAL_FRACTION, w2]


def capital_fraction_for_slot(trade_slot_index: int, total_strategy_picks: int) -> float:
    """
    Scale factor applied to ``equity * max_position_fraction`` for this order only.

    Single pick: full sleeve (1.0). Two picks: first slot 0.7, second slot 0.3 — does not change
    strategy selection; sizing is applied at risk/execution time (see pipeline + RiskEngine).

    When the strategy attaches ``StrategyDecision.capital_fractions``, the pipeline uses that
    plan instead (slot 2: gap-based 0.30/0.25/0.20; same-sector pairs use 0.30 vs 0.25 from raw
    score gap only — see ``capital_fractions_for_chosen_slots``).
    """
    if total_strategy_picks <= 1:
        return 1.0
    return TRADE_ONE_CAPITAL_FRACTION if trade_slot_index == 0 else TRADE_TWO_CAPITAL_FRACTION


def build_buy_market(symbol: str, verdict: RiskVerdict) -> OrderIntent | None:
    if not verdict.allowed or verdict.position_qty is None:
        return None
    return OrderIntent(symbol=symbol, side="buy", qty=verdict.position_qty, order_type="market")
