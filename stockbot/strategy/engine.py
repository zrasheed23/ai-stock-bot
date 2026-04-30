"""Rule-based scoring and ranking — LLM never selects trades."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import date
from typing import Any, Sequence

from stockbot.config import (
    PM_GAP_ATR_HARD_SKIP,
    PM_GAP_ATR_WARN,
    PM_RVOL_STRONG,
    SPY_GAP_ATR_NO_TRADE,
    SPY_GAP_ATR_REDUCE,
    Settings,
)
from stockbot.features.premarket import premarket_score_adjustment
from stockbot.features.sentiment import sentiment_features_from_llm
from stockbot.features.technical import technical_features
from stockbot.models import (
    FeatureVector,
    IngestionBundle,
    LLMInstrumentView,
    ScoredCandidate,
    StrategyDecision,
)
from stockbot.execution.orders import (
    SLOT2_ABSOLUTE_STRONG_MIN_RAW,
    SLOT2_GAP_WHEN_WEAK_KEEP_FULL,
    SLOT2_GAP_WHEN_WEAK_REDUCE_MILD,
    capital_fractions_for_chosen_slots,
)
from stockbot.risk.engine import MAX_LIVE_TRADES_PER_DAY
from stockbot.runners.managed_position_ledger import STOP_LOSS_PCT

_LOG = logging.getLogger("stockbot.strategy")

# -----------------------------------------------------------------------------
# Strategy refinement (extensions as rank enhancers, minimal hard vetoes)
#
# Core ``score()`` output is unchanged. Extensions below only adjust ``adjusted_score``
# used for sort order, except for two hard trend vetoes (deep negative mom/SMA).
#
# Soft modifiers:
# - Mild pullback rank demotion when mom/SMA are above hard floors but still negative.
#
# Hard gates kept minimal:
# - Raw composite score floor (min_score_to_trade / min_second_trade_score).
# - Deep trend veto only (clearly broken trend), not mom>=0 / sma>=0.
# - Trade #2 vol cap + score floor (unchanged intent); mom/sma floor relaxed to 0.0.
# -----------------------------------------------------------------------------

SECOND_TRADE_MAX_VOLATILITY_ANN = 0.30

# Mild negative mom/SMA (still above hard vetoes): single nudge, not a skip.
PULLBACK_SOFT_RANK_PENALTY = 0.10

# Hard vetoes: only extreme deterioration (replaces stacked mom>=0 AND sma>=0).
ENTRY_HARD_MOMENTUM_FLOOR = -0.14
ENTRY_HARD_SMA_FLOOR = -0.10

# -----------------------------------------------------------------------------
# Slot-1 selection quality only (higher bar than generic min_score_to_trade).
# Mirrors exit stop magnitude via ``STOP_LOSS_PCT`` for asymmetry vs expected range.
# Slot-2 gates and pairing logic are unchanged.
# -----------------------------------------------------------------------------
SLOT1_MIN_RAW_SCORE = 1.14
SLOT1_MIN_LLM_CONFIDENCE = 0.68
SLOT1_MIN_SENTIMENT_SCORE = 0.12
# Typical daily range proxy as fraction of price (ATR14/close or vol fallback).
SLOT1_MIN_EXPECTED_MOVE_PCT = 0.012
SLOT1_MIN_MOVE_TO_STOP_RATIO = 1.50

# Trade #2: same vol/score gates; relax mom/sma from 0.01 to 0.0 so slot-2 is not double-strict vs slot-1.
TRADE2_MOM_SMA_TECH_FLOOR = 0.0

# -----------------------------------------------------------------------------
# Slot-2 weak same-theme pairing (portfolio construction only — NOT ranking/signals)
#
# Trade #1 uses extra slot-1 quality gates (score/confidence/expected-move vs stop); see constants above.
# Natural slot 2 = first other name in adjusted-rank order passing all existing trade-#2 gates (unchanged).
#
# If natural slot 2 is NOT in the same mapped sector bucket as slot 1 → keep natural.
# If same sector: keep natural when slot-2 raw score is already strong (shares
# ``SLOT2_ABSOLUTE_STRONG_MIN_RAW`` with sizing — same "strong second leg" idea). Otherwise
# (weak + redundant theme) optionally replace with the best-scoring candidate that still passes
# the same gates, maps to a *different* sector than slot 1, and has raw score at least
# ``natural_score − SLOT2_WEAK_THEME_MAX_SCORE_DEFICIT`` (tight — no material downgrades).
# If none qualifies → keep natural. Never skip slot 2; always two picks when natural exists.
# -----------------------------------------------------------------------------

# Static coarse sectors for overlap detection only (not scoring).
_SYMBOL_SECTOR: dict[str, str] = {}
for _s in ("AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "TSLA"):
    _SYMBOL_SECTOR[_s] = "TECH"
for _s in ("XOM", "CVX", "XLE"):
    _SYMBOL_SECTOR[_s] = "ENERGY"
for _s in ("JPM", "GS", "XLF"):
    _SYMBOL_SECTOR[_s] = "FINANCIALS"
for _s in ("UNH", "LLY", "SPY"):
    _SYMBOL_SECTOR[_s] = "DEFENSIVE"


def _sector_bucket(symbol: str) -> str | None:
    return _SYMBOL_SECTOR.get(symbol.upper())


def _expected_intraday_move_pct(technical: dict[str, float]) -> float | None:
    """Fraction-of-price proxy for typical daily range (ATR/price preferred)."""
    last = float(technical.get("last_close", 0.0))
    atr = technical.get("atr14")
    if last > 0 and atr is not None:
        a = float(atr)
        if math.isfinite(a) and a > 0:
            return a / last
    vol_ann = float(technical.get("volatility_ann", 0.0))
    if vol_ann > 0 and math.isfinite(vol_ann):
        daily_sigma = vol_ann / math.sqrt(252.0)
        return float(daily_sigma * 1.28)
    return None


def _slot1_quality_gate(
    cand: ScoredCandidate,
    *,
    stop_distance_abs: float,
) -> tuple[bool, float, float, str]:
    """Returns (accepted, llm_confidence, expected_move_pct, reject_reason_code)."""
    conf = float(cand.features.sentiment.get("llm_confidence", 0.0))
    sent = float(cand.features.sentiment.get("sentiment_score", 0.0))
    raw = float(cand.score)
    tech = cand.features.technical
    exp = _expected_intraday_move_pct(tech)
    exp_for_metrics = exp if exp is not None else 0.0

    if raw < SLOT1_MIN_RAW_SCORE:
        return False, conf, exp_for_metrics, "LOW_RAW_SCORE"
    if conf < SLOT1_MIN_LLM_CONFIDENCE:
        return False, conf, exp_for_metrics, "LOW_CONFIDENCE"
    if sent < SLOT1_MIN_SENTIMENT_SCORE:
        return False, conf, exp_for_metrics, "LOW_SENTIMENT_SCORE"
    if exp is None:
        return False, conf, 0.0, "NO_MOVE_ESTIMATE"
    floor = max(SLOT1_MIN_EXPECTED_MOVE_PCT, stop_distance_abs * SLOT1_MIN_MOVE_TO_STOP_RATIO)
    if exp < floor:
        return False, conf, exp, "LOW_EXPECTED_MOVE"
    return True, conf, exp, ""


def _slot2_meaningful_cluster(trade_one: ScoredCandidate, cand: ScoredCandidate) -> bool:
    """True when both symbols map to the same sector bucket (coarse theme overlap)."""
    b1 = _sector_bucket(trade_one.symbol)
    b2 = _sector_bucket(cand.symbol)
    return b1 is not None and b2 is not None and b1 == b2


# Max allowed raw-score drop vs natural slot-2 for a replacement (``orders`` strong floor ties
# "strong duplication OK"; this threshold stays tight so we almost never swap in a worse name).
SLOT2_WEAK_THEME_MAX_SCORE_DEFICIT = 0.085


def _trade_two_gate_failures(
    cand: ScoredCandidate,
    min_score: float,
    tech_floor: float,
    max_vol_ann: float,
) -> list[str]:
    failures: list[str] = []
    sc = float(cand.score)
    mom = float(cand.features.technical.get("momentum_20d", 0.0))
    sma = float(cand.features.technical.get("sma20_distance", 0.0))
    vol = float(cand.features.technical.get("volatility_ann", 0.0))
    if not all(math.isfinite(x) for x in (sc, mom, sma, vol)):
        failures.append("non_finite_score_or_technicals")
        return failures
    if sc < min_score:
        failures.append(f"score<{min_score:.4f} (got {sc:.6f})")
    if mom < tech_floor:
        failures.append(f"momentum_20d<{tech_floor:.4f} (got {mom:.6f})")
    if sma < tech_floor:
        failures.append(f"sma20_distance<{tech_floor:.4f} (got {sma:.6f})")
    if vol >= max_vol_ann:
        failures.append(f"volatility_ann>={max_vol_ann:.4f} (got {vol:.6f})")
    return failures


# Manually curated trade universe (single source of truth for default watchlist).
SYMBOL_UNIVERSE: tuple[str, ...] = (
    # Core ETFs
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLY",
    # Mega caps
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    # High-volatility large caps
    "AMD",
    "SMCI",
    "AVGO",
    "NFLX",
    "SHOP",
    "PLTR",
    "COIN",
    # Financial / macro movers
    "JPM",
    "GS",
    "BAC",
    # Industrial / cyclical
    "CAT",
    "BA",
    # Growth / momentum names
    "SNOW",
    "CRWD",
    "PANW",
    "DDOG",
    # High-beta names
    "RIVN",
    "LCID",
    "AFRM",
)


def default_watchlist() -> list[str]:
    return list(SYMBOL_UNIVERSE)


class StrategyEngine:
    """
    Combined score = weighted sum of normalized technical + sentiment features.
    ``adjusted_score`` applies mild pullback demotion for sort order; deep trend + score floors hard-block.
    """

    def __init__(
        self,
        watchlist: Sequence[str] | None = None,
        max_candidates: int = 5,
    ):
        self._watchlist = [s.upper() for s in (watchlist or default_watchlist())]
        self._max_candidates = max_candidates
        _LOG.info("UNIVERSE_SIZE total_symbols=%d", len(self._watchlist))
        _LOG.info("UNIVERSE_SYMBOLS symbols=%s", self._watchlist)

    @property
    def watchlist(self) -> list[str]:
        return list(self._watchlist)

    def build_features(
        self,
        bundle: IngestionBundle,
        llm_by_symbol: dict[str, LLMInstrumentView],
    ) -> list[FeatureVector]:
        vecs: list[FeatureVector] = []
        for sym in self._watchlist:
            snap = bundle.market.get(sym)
            if snap is None:
                continue
            llm = llm_by_symbol.get(sym)
            if llm is None:
                continue
            tech = technical_features(snap.bars)
            sent = sentiment_features_from_llm(llm)
            vecs.append(
                FeatureVector(
                    symbol=sym,
                    as_of=bundle.run_date,
                    technical=tech,
                    sentiment=sent,
                    raw_llm=llm,
                )
            )
        return vecs

    def score(
        self,
        fv: FeatureVector,
        premarket_row: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        t = fv.technical
        s = fv.sentiment
        breakdown: dict[str, float] = {}

        breakdown["mom20"] = max(min(t.get("momentum_20d", 0.0) * 6.75, 2.35), -2.35)
        breakdown["sma_bias"] = max(min(t.get("sma20_distance", 0.0) * 12.0, 1.85), -1.85)
        rsi = t.get("rsi14", 50.0)
        rsi_balance = (50.0 - abs(rsi - 50.0)) / 50.0 * 0.58
        rsi_trend = max(-0.52, min(0.78, (rsi - 50.0) / 50.0 * 1.12))
        breakdown["rsi"] = rsi_balance + rsi_trend
        breakdown["sentiment"] = s.get("sentiment_score", 0.0) * 0.70
        breakdown["confidence_adj"] = (s.get("llm_confidence", 0.0) - 0.5) * 0.26

        vol = t.get("volatility_ann", 0.0)
        if vol > 0.45:
            breakdown["vol_penalty"] = -2.0
        elif vol > 0.30:
            breakdown["vol_penalty"] = -1.0
        else:
            breakdown["vol_penalty"] = 0.0

        sev = float(s.get("risk_severity", 0.0))
        breakdown["risk_severity"] = sev
        breakdown["risk_penalty"] = -min(1.35, 0.42 * sev)

        if s.get("sentiment_bucket", 0.0) < 0 and s.get("llm_confidence", 0.0) > 0.65:
            breakdown["neg_highconf_veto"] = -3.0
        else:
            breakdown["neg_highconf_veto"] = 0.0

        mom20_v = breakdown["mom20"]
        sma_v = breakdown["sma_bias"]
        rsi_v = breakdown["rsi"]
        tech_tailwind = mom20_v + sma_v + rsi_v
        technical_supportive = tech_tailwind > 0.28
        not_mixed = float(s.get("mixed_news_multiplier", 1.0)) >= 0.999
        clearly_bullish_sentiment = (
            not_mixed
            and s.get("sentiment_bucket", 0.0) > 0.5
            and float(s.get("sentiment_score", 0.0)) >= 0.22
        )
        vol_ok_for_boost = breakdown["vol_penalty"] > -1.01
        if technical_supportive and clearly_bullish_sentiment and vol_ok_for_boost:
            breakdown["bullish_boost"] = 0.20
        else:
            breakdown["bullish_boost"] = 0.0

        total = float(sum(breakdown.values()))
        if (
            premarket_row
            and not premarket_row.get("fetch_error")
            and premarket_row.get("pm_ref_price") is not None
            and premarket_row.get("prior_rth_close") is not None
        ):
            pm_adj = premarket_score_adjustment(
                float(premarket_row.get("gap_atr", 0.0)),
                float(premarket_row.get("pm_rvol", 1.0)),
                pm_gap_atr_hard_skip=PM_GAP_ATR_HARD_SKIP,
                pm_gap_atr_warn=PM_GAP_ATR_WARN,
                pm_rvol_strong=PM_RVOL_STRONG,
            )
            breakdown["premkt_score_adj"] = float(pm_adj)
            total += float(pm_adj)
        return total, breakdown

    def decide(
        self,
        trade_date: date,
        bundle: IngestionBundle,
        llm_by_symbol: dict[str, LLMInstrumentView],
        settings: Settings | None = None,
    ) -> StrategyDecision:
        settings = settings or Settings.from_env()
        reason_codes: list[str] = []
        cooldown_skips: list[dict[str, Any]] = []
        soft_modifiers: list[dict[str, Any]] = []
        risk_mult = 1.0
        prem = bundle.premarket or {}
        active_pm = bool(
            settings.enable_premarket_signals and prem and not prem.get("disabled_session")
        )
        if active_pm:
            spy_ga = float((prem.get("spy") or {}).get("gap_atr", 0.0))
            if spy_ga < SPY_GAP_ATR_NO_TRADE:
                return StrategyDecision(
                    trade_date=trade_date,
                    watchlist=self._watchlist,
                    ranked=[],
                    chosen=[],
                    reason_codes=["NO_TRADE_MARKET_CONDITIONS"],
                    candidate_outcomes=[],
                    cooldown_skips=cooldown_skips,
                    soft_modifiers=soft_modifiers,
                    capital_fractions=[],
                    risk_notional_multiplier=1.0,
                )
            if spy_ga < SPY_GAP_ATR_REDUCE:
                risk_mult = 0.5
                reason_codes.append("SPY_RISK_REDUCTION")

        skip_syms: set[str] = set()
        if active_pm:
            for sym, row in (prem.get("symbols") or {}).items():
                hr = row.get("hard_skip_reason")
                if hr == "PM_GAP_TOO_LARGE":
                    skip_syms.add(sym.upper())
                    reason_codes.append("PM_GAP_TOO_LARGE")
                elif hr == "PM_LOW_VOLUME_ON_GAP":
                    skip_syms.add(sym.upper())
                    reason_codes.append("PM_LOW_VOLUME_ON_GAP")

        fvecs_all = self.build_features(bundle, llm_by_symbol)
        fvecs = [fv for fv in fvecs_all if fv.symbol.upper() not in skip_syms]
        if not fvecs_all:
            reason_codes.append("NO_FEATURES")
            return StrategyDecision(
                trade_date=trade_date,
                watchlist=self._watchlist,
                ranked=[],
                chosen=[],
                reason_codes=reason_codes,
                candidate_outcomes=[],
                cooldown_skips=cooldown_skips,
                soft_modifiers=soft_modifiers,
                capital_fractions=[],
                risk_notional_multiplier=risk_mult,
            )
        if not fvecs:
            # All watchlist names filtered (typically pre-market hard skips); PM_* codes already added.
            return StrategyDecision(
                trade_date=trade_date,
                watchlist=self._watchlist,
                ranked=[],
                chosen=[],
                reason_codes=reason_codes,
                candidate_outcomes=[],
                cooldown_skips=cooldown_skips,
                soft_modifiers=soft_modifiers,
                capital_fractions=[],
                risk_notional_multiplier=risk_mult,
            )

        sym_rows = (prem.get("symbols") or {}) if active_pm else {}
        full_ranked: list[ScoredCandidate] = []
        for fv in fvecs:
            pm_row = sym_rows.get(fv.symbol) if active_pm else None
            total, br = self.score(fv, pm_row)
            full_ranked.append(
                ScoredCandidate(
                    symbol=fv.symbol,
                    score=total,
                    adjusted_score=total,
                    score_breakdown=br,
                    features=fv,
                )
            )

        for i, cand in enumerate(full_ranked):
            adj = float(cand.score)
            sym = cand.symbol
            t = cand.features.technical
            mom = float(t.get("momentum_20d", 0.0))
            sma = float(t.get("sma20_distance", 0.0))

            if (
                mom >= ENTRY_HARD_MOMENTUM_FLOOR
                and sma >= ENTRY_HARD_SMA_FLOOR
                and (mom < 0.0 or sma < 0.0)
            ):
                adj -= PULLBACK_SOFT_RANK_PENALTY
                soft_modifiers.append(
                    {
                        "reason": "MILD_PULLBACK_RANK_DEMOTION",
                        "symbol": sym,
                        "penalty": PULLBACK_SOFT_RANK_PENALTY,
                        "momentum_20d": mom,
                        "sma20_distance": sma,
                        "adjusted_score_after": round(adj, 6),
                    }
                )

            full_ranked[i] = replace(cand, adjusted_score=adj)

        full_ranked.sort(key=lambda c: (-c.adjusted_score, -c.score))

        min_score_to_trade = 1.0
        min_second_trade_score = min_score_to_trade + 0.2
        second_trade_max_vol_ann = SECOND_TRADE_MAX_VOLATILITY_ANN

        chosen: list[ScoredCandidate] = []
        candidate_outcomes: list[dict[str, Any]] = []

        best_raw = max(c.score for c in full_ranked) if full_ranked else float("-inf")
        if not full_ranked or best_raw < min_score_to_trade:
            reason_codes.append("NO_TRADE_LOW_SCORE")
        else:
            trade_one: ScoredCandidate | None = None
            stop_abs = abs(float(STOP_LOSS_PCT))
            slot1_hard_gate_passed_any = False
            for cand in full_ranked:
                if cand.score < min_score_to_trade:
                    continue
                t = cand.features.technical
                mom = float(t.get("momentum_20d", 0.0))
                sma_bias = float(t.get("sma20_distance", 0.0))
                if mom < ENTRY_HARD_MOMENTUM_FLOOR or sma_bias < ENTRY_HARD_SMA_FLOOR:
                    continue
                slot1_hard_gate_passed_any = True
                ok, conf, exp_move, rej = _slot1_quality_gate(cand, stop_distance_abs=stop_abs)
                _LOG.info(
                    "HIGH_QUALITY_TRADE symbol=%s confidence=%.4f expected_move=%.6f accepted=%s",
                    cand.symbol,
                    conf,
                    exp_move,
                    ok,
                )
                if not ok:
                    soft_modifiers.append(
                        {
                            "reason": "SLOT1_QUALITY_REJECT",
                            "symbol": cand.symbol,
                            "reject_code": rej,
                            "confidence": round(conf, 4),
                            "expected_move_pct": round(exp_move, 6),
                            "raw_score": round(float(cand.score), 6),
                        }
                    )
                    continue
                trade_one = cand
                break

            if trade_one is None:
                if slot1_hard_gate_passed_any:
                    reason_codes.append("SKIP_SLOT1_QUALITY_FILTERS")
                else:
                    reason_codes.append("SKIP_SHORT_TERM_DOWNTREND")
            else:
                chosen.append(trade_one)
                reason_codes.append("CANDIDATE_SELECTED")

                def _slot2_passes_base_gates(cand: ScoredCandidate) -> tuple[bool, float, float, float]:
                    if cand.symbol == trade_one.symbol:
                        return False, 0.0, 0.0, 0.0
                    if cand.score < min_score_to_trade:
                        return False, 0.0, 0.0, 0.0
                    t = cand.features.technical
                    mom = float(t.get("momentum_20d", 0.0))
                    sma_bias = float(t.get("sma20_distance", 0.0))
                    if mom < ENTRY_HARD_MOMENTUM_FLOOR or sma_bias < ENTRY_HARD_SMA_FLOOR:
                        return False, 0.0, 0.0, 0.0
                    vol_ann = float(t.get("volatility_ann", 0.0))
                    return True, mom, sma_bias, vol_ann

                slot2: ScoredCandidate | None = None
                natural_slot2: ScoredCandidate | None = None
                for cand in full_ranked:
                    ok, mom, sma_bias, vol_ann = _slot2_passes_base_gates(cand)
                    if not ok:
                        continue
                    failures = _trade_two_gate_failures(
                        cand,
                        min_second_trade_score,
                        TRADE2_MOM_SMA_TECH_FLOOR,
                        second_trade_max_vol_ann,
                    )
                    if failures:
                        _LOG.info(
                            "[strategy trade#2] symbol=%s REJECTED score=%.6f momentum_20d=%.6f "
                            "sma20_distance=%.6f volatility_ann=%.6f failures=[%s]",
                            cand.symbol,
                            float(cand.score),
                            mom,
                            sma_bias,
                            vol_ann,
                            "; ".join(failures),
                        )
                        if any("volatility_ann" in f for f in failures):
                            reason_codes.append("SECOND_TRADE_REJECTED_HIGH_VOLATILITY")
                        elif float(cand.score) < min_second_trade_score:
                            reason_codes.append("SECOND_TRADE_REJECTED_LOW_SCORE")
                        else:
                            reason_codes.append("SECOND_TRADE_REJECTED_WEAK_TECHNICALS")
                        continue
                    natural_slot2 = cand
                    break

                if natural_slot2 is not None:
                    slot2 = natural_slot2
                    if _slot2_meaningful_cluster(trade_one, natural_slot2):
                        s_nat = float(natural_slot2.score)
                        if s_nat >= float(SLOT2_ABSOLUTE_STRONG_MIN_RAW):
                            _LOG.info(
                                "[strategy trade#2] SLOT2_KEEP_SAME_THEME_STRONG symbol=%s score=%.4f "
                                "(>= strong floor %.4f vs slot1=%s sector=%s)",
                                natural_slot2.symbol,
                                s_nat,
                                float(SLOT2_ABSOLUTE_STRONG_MIN_RAW),
                                trade_one.symbol,
                                _sector_bucket(natural_slot2.symbol),
                            )
                        else:
                            min_alt_score = s_nat - float(SLOT2_WEAK_THEME_MAX_SCORE_DEFICIT)
                            best_alt: ScoredCandidate | None = None
                            best_alt_score = float("-inf")
                            for cand in full_ranked:
                                if cand.symbol == natural_slot2.symbol:
                                    continue
                                ok2, _, _, _ = _slot2_passes_base_gates(cand)
                                if not ok2:
                                    continue
                                if _trade_two_gate_failures(
                                    cand,
                                    min_second_trade_score,
                                    TRADE2_MOM_SMA_TECH_FLOOR,
                                    second_trade_max_vol_ann,
                                ):
                                    continue
                                if _slot2_meaningful_cluster(trade_one, cand):
                                    continue
                                sc = float(cand.score)
                                if sc < min_alt_score:
                                    continue
                                if sc > best_alt_score:
                                    best_alt_score = sc
                                    best_alt = cand
                            if best_alt is not None:
                                slot2 = best_alt
                                reason_codes.append("SECOND_TRADE_SLOT2_WEAK_THEME_SWAP")
                                soft_modifiers.append(
                                    {
                                        "reason": "SLOT2_WEAK_THEME_NEAR_PEER_REPLACEMENT",
                                        "trade_1": trade_one.symbol,
                                        "natural_slot2": natural_slot2.symbol,
                                        "picked_slot2": best_alt.symbol,
                                        "natural_score": round(s_nat, 6),
                                        "picked_score": round(float(best_alt.score), 6),
                                        "min_alt_score": round(min_alt_score, 6),
                                        "max_score_deficit": SLOT2_WEAK_THEME_MAX_SCORE_DEFICIT,
                                        "strong_same_theme_floor": SLOT2_ABSOLUTE_STRONG_MIN_RAW,
                                    }
                                )
                                _LOG.info(
                                    "[strategy trade#2] SLOT2_WEAK_THEME_SWAP natural=%s@%.4f -> %s@%.4f "
                                    "(min_alt=%.4f)",
                                    natural_slot2.symbol,
                                    s_nat,
                                    best_alt.symbol,
                                    float(best_alt.score),
                                    min_alt_score,
                                )
                            else:
                                _LOG.info(
                                    "[strategy trade#2] SLOT2_KEEP_WEAK_SAME_THEME_NO_PEER symbol=%s "
                                    "score=%.4f sector=%s (no different-sector peer >= %.4f)",
                                    natural_slot2.symbol,
                                    s_nat,
                                    _sector_bucket(natural_slot2.symbol),
                                    min_alt_score,
                                )

                if slot2 is not None:
                    _t = slot2.features.technical
                    _mom = float(_t.get("momentum_20d", 0.0))
                    _sma = float(_t.get("sma20_distance", 0.0))
                    _vol = float(_t.get("volatility_ann", 0.0))
                    _LOG.info(
                        "[strategy trade#2] symbol=%s SELECTED score=%.6f momentum_20d=%.6f "
                        "sma20_distance=%.6f volatility_ann=%.6f (all gates pass)",
                        slot2.symbol,
                        float(slot2.score),
                        _mom,
                        _sma,
                        _vol,
                    )
                    chosen.append(slot2)
                    reason_codes.append("CANDIDATE_SELECTED")

        if len(chosen) > 1:
            inv = _trade_two_gate_failures(
                chosen[1],
                min_second_trade_score,
                TRADE2_MOM_SMA_TECH_FLOOR,
                second_trade_max_vol_ann,
            )
            if inv:
                _LOG.error(
                    "[strategy trade#2] INVARIANT VIOLATION: stripping second pick %s — %s",
                    chosen[1].symbol,
                    "; ".join(inv),
                )
                chosen.pop()
                reason_codes.append("SECOND_TRADE_STRIPPED_INVALID")
                for i in range(len(reason_codes) - 1, -1, -1):
                    if reason_codes[i] == "CANDIDATE_SELECTED":
                        reason_codes.pop(i)
                        break

        rank_cap = self._max_candidates or len(full_ranked)
        for idx, cand in enumerate(full_ranked[:rank_cap]):
            t = cand.features.technical
            mom = float(t.get("momentum_20d", 0.0))
            sma_bias = float(t.get("sma20_distance", 0.0))
            hard_bad = mom < ENTRY_HARD_MOMENTUM_FLOOR or sma_bias < ENTRY_HARD_SMA_FLOOR
            if cand.score < min_score_to_trade:
                rc = "LOW_SCORE"
                selected = False
                trade_slot: int | None = None
            elif len(chosen) >= 1 and cand is chosen[0]:
                rc = "SELECTED_TRADE_1"
                selected = True
                trade_slot = 1
            elif len(chosen) >= 2 and cand is chosen[1]:
                rc = "SELECTED_TRADE_2"
                selected = True
                trade_slot = 2
            elif hard_bad:
                rc = "SHORT_TERM_DOWNSIDE"
                selected = False
                trade_slot = None
            elif len(chosen) == 2:
                rc = "TOP_TWO_CAP"
                selected = False
                trade_slot = None
            elif len(chosen) != 1:
                rc = "NOT_SELECTED"
                selected = False
                trade_slot = None
            else:
                vol_ann = float(t.get("volatility_ann", 0.0))
                if cand.score < min_second_trade_score:
                    rc = "SECOND_TRADE_REJECTED_LOW_SCORE"
                    selected = False
                    trade_slot = None
                elif mom < TRADE2_MOM_SMA_TECH_FLOOR or sma_bias < TRADE2_MOM_SMA_TECH_FLOOR:
                    rc = "SECOND_TRADE_REJECTED_WEAK_TECHNICALS"
                    selected = False
                    trade_slot = None
                elif vol_ann >= second_trade_max_vol_ann:
                    rc = "SECOND_TRADE_REJECTED_HIGH_VOLATILITY"
                    selected = False
                    trade_slot = None
                else:
                    rc = "TOP_TWO_CAP"
                    selected = False
                    trade_slot = None

            candidate_outcomes.append(
                {
                    "rank": idx + 1,
                    "symbol": cand.symbol,
                    "score": cand.score,
                    "adjusted_score": cand.adjusted_score,
                    "momentum_20d": mom,
                    "sma20_distance": sma_bias,
                    "volatility_ann": float(t.get("volatility_ann", 0.0)),
                    "selected": selected,
                    "trade_slot": trade_slot,
                    "reason_code": rc,
                }
            )

        ranked = full_ranked[: self._max_candidates] if self._max_candidates else full_ranked

        capital_fractions = capital_fractions_for_chosen_slots(chosen)
        if len(chosen) == 2 and capital_fractions:
            gap = max(0.0, float(chosen[0].score) - float(chosen[1].score))
            soft_modifiers.append(
                {
                    "reason": "SLOT2_RELATIVE_NOTIONAL_FRACTION",
                    "slot1_symbol": chosen[0].symbol,
                    "slot2_symbol": chosen[1].symbol,
                    "slot1_raw_score": round(float(chosen[0].score), 6),
                    "slot2_raw_score": round(float(chosen[1].score), 6),
                    "raw_score_gap": round(gap, 6),
                    "slot2_capital_fraction": capital_fractions[1],
                    "slot2_absolute_strong_min_raw": SLOT2_ABSOLUTE_STRONG_MIN_RAW,
                    "gap_when_weak_keep_full_max": SLOT2_GAP_WHEN_WEAK_KEEP_FULL,
                    "gap_when_weak_reduce_mild_max": SLOT2_GAP_WHEN_WEAK_REDUCE_MILD,
                }
            )

        return StrategyDecision(
            trade_date=trade_date,
            watchlist=self._watchlist,
            ranked=ranked,
            chosen=chosen,
            reason_codes=reason_codes,
            candidate_outcomes=candidate_outcomes,
            cooldown_skips=cooldown_skips,
            soft_modifiers=soft_modifiers,
            capital_fractions=capital_fractions,
            risk_notional_multiplier=risk_mult,
        )
