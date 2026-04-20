"""Shared data shapes used across the pipeline (plain dataclasses, easy to serialize)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd


@dataclass
class MarketSnapshot:
    symbol: str
    as_of: datetime
    bars: pd.DataFrame  # columns: open, high, low, close, volume (index or column time)


@dataclass
class NewsItem:
    symbol: str | None
    headline: str
    source: str
    published_at: datetime
    url: str | None = None
    # Short audit hint, e.g. why we fell back to a placeholder row (optional).
    note: str | None = None


@dataclass
class FilingRef:
    symbol: str
    form_type: str
    filed_at: date
    accession: str
    url: str | None = None


@dataclass
class IngestionBundle:
    """Everything collected for one daily run before feature generation."""

    run_date: date
    market: dict[str, MarketSnapshot]
    news: list[NewsItem]
    filings: list[FilingRef]


@dataclass
class LLMInstrumentView:
    """Strict structured output from the AI layer (no trade instructions)."""

    symbol: str
    news_summary: str
    filings_summary: str
    sentiment: Literal["positive", "neutral", "negative"]
    sentiment_score: float  # -1.0 .. 1.0
    risk_flags: list[str]
    confidence: float  # 0.0 .. 1.0


@dataclass
class FeatureVector:
    symbol: str
    as_of: date
    technical: dict[str, float]
    sentiment: dict[str, float]
    raw_llm: LLMInstrumentView


@dataclass
class ScoredCandidate:
    symbol: str
    score: float
    # Rank key after soft extensions (e.g. mild pullback demotion); ``score`` stays the raw composite.
    adjusted_score: float
    score_breakdown: dict[str, float]
    features: FeatureVector


@dataclass
class StrategyDecision:
    """Up to two actionable candidates per day (score order); may be zero trades."""

    trade_date: date
    watchlist: list[str]
    ranked: list[ScoredCandidate]
    chosen: list[ScoredCandidate]
    reason_codes: list[str]
    # One row per entry in `ranked` (audit): selection result for that rank slice only.
    candidate_outcomes: list[dict[str, Any]] = field(default_factory=list)
    # Legacy audit field; loss cooldown no longer hard-skips — kept empty for compatibility.
    cooldown_skips: list[dict[str, Any]] = field(default_factory=list)
    # Soft extension applications (rank demotions, not hard blocks).
    soft_modifiers: list[dict[str, Any]] = field(default_factory=list)
    # Per-leg capital fractions vs max daily sleeve (e.g. [0.7, 0.2]); empty => pipeline defaults.
    capital_fractions: list[float] = field(default_factory=list)


@dataclass
class RiskVerdict:
    allowed: bool
    block_reasons: list[str]
    position_qty: int | None = None
    notional_usd: float | None = None


@dataclass
class OrderIntent:
    symbol: str
    side: Literal["buy", "sell"]
    qty: int
    order_type: Literal["market", "limit"] = "market"
    limit_price: float | None = None
    time_in_force: str = "day"


@dataclass
class ExecutionResult:
    success: bool
    broker_order_id: str | None
    raw_response: dict[str, Any] | None
    error: str | None = None


@dataclass
class DailyReasoningRecord:
    """Single audit record: JSON-serializable summary of the run."""
    plain_english: str
    trade_date: str
    pipeline_version: str
    inputs_trace: dict[str, Any]
    llm_outputs: list[dict[str, Any]]
    strategy: dict[str, Any]
    risk: dict[str, Any]
    # One dict per broker submit attempt (0..2); empty list means no execution tried.
    executions: list[dict[str, Any]] = field(default_factory=list)
    
    meta: dict[str, Any] = field(default_factory=dict)
