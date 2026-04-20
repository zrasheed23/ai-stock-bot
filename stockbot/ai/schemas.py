"""Strict JSON shape for LLM responses — validate before any downstream use."""

from __future__ import annotations

from typing import Any

from stockbot.models import FilingRef, NewsItem

ALLOWED_SENTIMENT = {"positive", "neutral", "negative"}

# --- Simple offline signals (used when ANTHROPIC_API_KEY is not set) ---
# These are not "AI" — they make the audit JSON reflect real headlines anyway.
_PLACEHOLDER_SOURCES = frozenset({"stub", "stub_fallback", "finnhub_empty"})

_NEGATIVE_HINTS = (
    "lawsuit",
    "investigation",
    "subpoena",
    "recall",
    "bankrupt",
    "layoff",
    "downgrade",
    "miss",
    "warning",
    "fraud",
    "probe",
    "halt",
    "crash",
    "plunge",
    "selloff",
    "bearish",
    "weakness",
    "cuts guidance",
)

_POSITIVE_HINTS = (
    "beat",
    "beats",
    "surge",
    "soar",
    "rally",
    "gain",
    "gains",
    "jump",
    "upgrade",
    "bullish",
    "optimism",
    "breakout",
    "growth",
    "approval",
    "partnership",
    "record",
    "expansion",
    "win",
    "strong",
    "rebound",
    "recovery",
    "momentum",
    "risk-on",
    "earnings",
    "revenue",
    "all-time",
    "outperform",
)

_RISK_HINTS = (
    ("sec", "regulatory_sec_mention"),
    ("investigation", "investigation_mention"),
    ("lawsuit", "litigation_mention"),
    ("subpoena", "subpoena_mention"),
    ("recall", "product_recall_mention"),
    ("bankrupt", "distress_mention"),
    ("restructuring", "restructuring_mention"),
    ("guidance", "guidance_mention"),
    ("restat", "restatement_mention"),
)


def validate_llm_payload(data: Any, symbol: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("LLM output must be a JSON object")
    required = [
        "news_summary",
        "filings_summary",
        "sentiment",
        "sentiment_score",
        "risk_flags",
        "confidence",
    ]
    for k in required:
        if k not in data:
            raise ValueError(f"Missing key: {k}")
    sent = data["sentiment"]
    if sent not in ALLOWED_SENTIMENT:
        raise ValueError(f"Invalid sentiment: {sent}")
    score = float(data["sentiment_score"])
    if score < -1.0 or score > 1.0:
        raise ValueError("sentiment_score out of range")
    conf = float(data["confidence"])
    if conf < 0.0 or conf > 1.0:
        raise ValueError("confidence out of range")
    flags = data["risk_flags"]
    if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
        raise ValueError("risk_flags must be list of strings")
    return {
        "symbol": symbol,
        "news_summary": str(data["news_summary"])[:4000],
        "filings_summary": str(data["filings_summary"])[:4000],
        "sentiment": sent,
        "sentiment_score": score,
        "risk_flags": [str(x)[:500] for x in flags][:50],
        "confidence": conf,
    }


def structured_fallback_from_news_and_filings(
    symbol: str,
    news: list[NewsItem],
    filings: list[FilingRef],
) -> dict[str, Any]:
    """
    Build the same JSON shape the LLM returns, without calling Claude.

    Why this exists:
      - Lets you run the full pipeline without an Anthropic key.
      - If Finnhub (or another source) filled NewsItem rows, summaries and coarse
        sentiment/flags still land in the audit JSON.

    This is intentionally dumb (keyword counting). It is NOT a substitute for
    the model when you care about nuance — turn on ANTHROPIC_API_KEY for that.
    """
    sym = symbol.upper()
    rows = [n for n in news if n.symbol == sym or n.symbol is None]
    live = [n for n in rows if n.source not in _PLACEHOLDER_SOURCES]

    if live:
        lines = []
        for i, n in enumerate(live[:12], start=1):
            lines.append(f"{i}. ({n.source}) {n.headline}")
        news_summary = "\n".join(lines)
        blob = " ".join(h.lower() for h in (n.headline for n in live))
        neg = sum(blob.count(w) for w in _NEGATIVE_HINTS)
        pos = sum(blob.count(w) for w in _POSITIVE_HINTS)
        raw = (pos - neg) / 2.5
        sentiment_score = max(-1.0, min(1.0, raw))
        # Mixed: dampen lightly; if bulls clearly lead, keep more of the upside (risk-on days).
        if pos >= 1 and neg >= 1:
            mix = min(pos, neg) / float(max(max(pos, neg), 1))
            ratio = pos / max(float(neg), 0.01)
            if ratio >= 2.0:
                damp = 0.10 * mix
            elif ratio >= 1.35:
                damp = 0.18 * mix
            elif ratio <= 0.5:
                damp = 0.38 * mix
            else:
                damp = 0.26 * mix
            sentiment_score *= 1.0 - damp
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        if sentiment_score > 0.0:
            sentiment_score = min(1.0, sentiment_score * 1.10 + 0.03)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        if sentiment_score > 0.15:
            sentiment = "positive"
        elif sentiment_score < -0.15:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        flags: list[str] = []
        for needle, label in _RISK_HINTS:
            if needle in blob:
                if label not in flags:
                    flags.append(label)
        confidence = min(0.85, 0.35 + 0.05 * min(len(live), 10))
    else:
        headlines = [n.headline for n in rows[:5]]
        if headlines:
            news_summary = "No live headlines parsed; placeholder rows only:\n" + "\n".join(
                f"- {h}" for h in headlines
            )
        else:
            news_summary = f"No news rows for {sym} (ingestion may have failed or returned empty)."
        sentiment = "neutral"
        sentiment_score = 0.0
        flags = []
        confidence = 0.25

    sym_filings = [f for f in filings if f.symbol == sym]
    if sym_filings:
        flines = [
            f"- {f.form_type} filed {f.filed_at.isoformat()} ({f.accession})" for f in sym_filings[:8]
        ]
        filings_summary = "Filing references (not full text):\n" + "\n".join(flines)
    else:
        filings_summary = f"No filing references for {sym} in this run."

    return {
        "news_summary": news_summary[:4000],
        "filings_summary": filings_summary[:4000],
        "sentiment": sentiment,
        "sentiment_score": float(sentiment_score),
        "risk_flags": flags[:50],
        "confidence": float(confidence),
    }


def mock_llm_json(symbol: str) -> dict[str, Any]:
    """Backward-compatible name: empty inputs → same shape as offline fallback."""
    return structured_fallback_from_news_and_filings(symbol, [], [])
