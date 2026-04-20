"""Map LLM structured outputs into numeric features for the rule-based strategy."""

from __future__ import annotations

from stockbot.models import LLMInstrumentView

# Headline / summary vocabulary for tone checks (LLM news_summary is usually headlines).
# Broader bullish list helps macro “risk-on” days register; bearish list stays for balance checks.
_POS_WORDS = (
    "beat",
    "beats",
    "surge",
    "surges",
    "soar",
    "rally",
    "rallies",
    "rallying",
    "gain",
    "gains",
    "jump",
    "jumps",
    "upgrade",
    "upgrades",
    "bullish",
    "optimism",
    "optimistic",
    "breakout",
    "record",
    "highs",
    "outperform",
    "strong",
    "strength",
    "growth",
    "rebound",
    "recovery",
    "momentum",
    "tailwind",
    "risk-on",
    "risk on",
    "green",
    "higher",
    "pop",
    "pops",
    "approval",
    "partnership",
    "expansion",
    "win",
    "wins",
    "earnings",
    "revenue",
    "guidance raise",
    "raises guidance",
    "raises outlook",
    "all-time",
    "all time",
    "ath",
)
_NEG_WORDS = (
    "lawsuit",
    "investigation",
    "subpoena",
    "recall",
    "bankrupt",
    "layoff",
    "layoffs",
    "downgrade",
    "downgrades",
    "miss",
    "misses",
    "warning",
    "fraud",
    "probe",
    "halt",
    "crash",
    "plunge",
    "plunges",
    "selloff",
    "sell-off",
    "bearish",
    "pessimism",
    "weak",
    "weakness",
    "cuts guidance",
    "lowers guidance",
    "lawsuits",
)


def _count_hits(text: str, words: tuple[str, ...]) -> int:
    blob = text.lower()
    return sum(blob.count(w) for w in words)


def _mixed_tone_multiplier(summary: str) -> tuple[float, float, float]:
    """
    When both bullish and bearish words appear, dampen less than before.

    If bullish hits clearly lead (typical rally + a few scary headlines), keep most of the
    positive signal. If bearish leads, dampen more. Balanced noise → mild dampening only.
    """
    pos = float(_count_hits(summary, _POS_WORDS))
    neg = float(_count_hits(summary, _NEG_WORDS))
    if pos < 1.0 or neg < 1.0:
        return 1.0, pos, neg

    mix = min(pos, neg) / max(max(pos, neg), 1.0)
    ratio = pos / max(neg, 0.01)

    if ratio >= 2.2:
        mult = 1.0 - 0.10 * mix
    elif ratio >= 1.35:
        mult = 1.0 - 0.18 * mix
    elif ratio <= 0.45:
        mult = 1.0 - 0.38 * mix
    elif ratio <= 0.75:
        mult = 1.0 - 0.30 * mix
    else:
        mult = 1.0 - 0.24 * mix

    if pos >= 4.0 and neg >= 4.0:
        mult *= 0.92

    return float(max(0.55, min(1.0, mult))), pos, neg


def _risk_weight_for_flag(flag: str) -> float:
    """
    Per-flag severity for scoring (not all flags are equal).
    Tune keywords to match labels your LLM / offline fallback emits.
    """
    f = flag.lower()
    # Serious / specific legal or distress signals
    if any(k in f for k in ("litigation", "lawsuit", "subpoena", "recall", "bankrupt", "distress", "fraud")):
        return 1.0
    if any(k in f for k in ("restatement", "restructuring")):
        return 0.55
    # Routine large-cap noise: SEC / regulatory *mentions* are common — stay light
    if any(k in f for k in ("regulatory", "sec", "investigation")):
        return 0.28
    if "guidance" in f:
        return 0.18
    # Unknown label: small nudge, not a full "major incident"
    return 0.22


def _risk_severity(flags: list[str]) -> float:
    """Sum of per-flag weights, capped so many tiny flags do not explode the penalty."""
    total = sum(_risk_weight_for_flag(fl) for fl in flags)
    return min(float(total), 2.5)


def _derived_bucket(raw_bucket: float, pos_hits: float, neg_hits: float) -> float:
    """
    Bucket used by the strategy layer. Mixed news no longer forces pure neutral when
    bullish language clearly outweighs bearish language (macro risk-on days).
    """
    if pos_hits < 1.0 or neg_hits < 1.0:
        return raw_bucket
    if raw_bucket < 0:
        if neg_hits >= pos_hits * 1.6:
            return raw_bucket
        return 0.0
    if raw_bucket > 0:
        if pos_hits >= neg_hits * 1.55:
            return 1.0
        if neg_hits >= pos_hits * 1.55:
            return 0.0
        return 0.0
    if pos_hits >= neg_hits * 1.75:
        return 1.0
    if neg_hits >= pos_hits * 1.75:
        return -1.0
    return 0.0


def sentiment_features_from_llm(view: LLMInstrumentView) -> dict[str, float]:
    """
    Deterministic mapping — strategy never reads raw LLM text.

    Goals:
      - Rally / risk-on language in news_summary registers as meaningfully positive.
      - Mixed headlines: if bulls clearly lead, keep a positive tilt; if bears lead, stay cautious.
      - Raw model negatives stay conservative (no “positive stretch” on bearish labels).
    """
    sent_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    raw_bucket = sent_map.get(view.sentiment, 0.0)
    summary = view.news_summary or ""

    mixed_mult, pos_hits, neg_hits = _mixed_tone_multiplier(summary)
    raw_score = float(view.sentiment_score)
    conf = float(view.confidence)

    # Softer confidence blend than before so neutral confidence does not crush real positives.
    effective_score = raw_score * mixed_mult * (0.62 + 0.38 * conf)

    if raw_bucket >= 0 and raw_score > 0.0:
        stretch = 1.0 + 0.22 * min(raw_score, 1.0)
        effective_score = min(1.0, max(0.0, effective_score * stretch))

    effective_score = max(-1.0, min(1.0, effective_score))

    bucket = _derived_bucket(raw_bucket, pos_hits, neg_hits)

    risk_count = float(len(view.risk_flags))
    risk_severity = _risk_severity(view.risk_flags)

    return {
        "sentiment_bucket": bucket,
        "sentiment_score": effective_score,
        "sentiment_score_raw": raw_score,
        "sentiment_bucket_raw": raw_bucket,
        "llm_confidence": conf,
        "risk_flag_count": risk_count,
        "risk_severity": risk_severity,
        "has_high_risk": 1.0 if risk_severity >= 1.2 else 0.0,
        "mixed_news_hits_pos": pos_hits,
        "mixed_news_hits_neg": neg_hits,
        "mixed_news_multiplier": mixed_mult,
    }
