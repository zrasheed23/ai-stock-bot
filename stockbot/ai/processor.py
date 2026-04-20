"""Call Claude for summaries/sentiment only; enforce JSON-only responses."""

from __future__ import annotations

import json
import re
from typing import Iterable

import requests

from stockbot.ai.schemas import structured_fallback_from_news_and_filings, validate_llm_payload
from stockbot.config import Settings
from stockbot.models import FilingRef, LLMInstrumentView, NewsItem


_SYSTEM = """You assist a trading research pipeline. You MUST NOT recommend trades or actions.
Output a single JSON object only, no markdown, no prose outside JSON.
Keys: news_summary (string), filings_summary (string), sentiment (one of: positive, neutral, negative),
sentiment_score (number -1 to 1), risk_flags (array of short strings), confidence (0 to 1).
Use risk_flags for concrete concerns (e.g. litigation, regulatory, guidance, liquidity); use short snake_case labels."""


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group())


class NewsFilingsProcessor:
    def __init__(self, settings: Settings):
        self._settings = settings

    def analyze_symbol(
        self,
        symbol: str,
        news: list[NewsItem],
        filings: list[FilingRef],
    ) -> LLMInstrumentView:
        sym_news = [n for n in news if n.symbol == symbol or n.symbol is None]
        sym_filings = [f for f in filings if f.symbol == symbol]
        if not self._settings.anthropic_api_key:
            # No Claude: still emit validated JSON from real headlines when present.
            payload = structured_fallback_from_news_and_filings(symbol, sym_news, sym_filings)
        else:
            user = self._build_user_prompt(symbol, sym_news, sym_filings)
            raw = self._anthropic_messages(user)
            payload = _extract_json_object(raw)
        normalized = validate_llm_payload(payload, symbol)
        return LLMInstrumentView(
            symbol=normalized["symbol"],
            news_summary=normalized["news_summary"],
            filings_summary=normalized["filings_summary"],
            sentiment=normalized["sentiment"],
            sentiment_score=normalized["sentiment_score"],
            risk_flags=normalized["risk_flags"],
            confidence=normalized["confidence"],
        )

    def analyze_watchlist(
        self,
        symbols: Iterable[str],
        news: list[NewsItem],
        filings: list[FilingRef],
    ) -> dict[str, LLMInstrumentView]:
        return {s.upper(): self.analyze_symbol(s.upper(), news, filings) for s in symbols}

    def _build_user_prompt(
        self,
        symbol: str,
        news: list[NewsItem],
        filings: list[FilingRef],
    ) -> str:
        lines = [f"Symbol: {symbol}", "", "News headlines:"]
        for n in news[:20]:
            lines.append(f"- ({n.source}) {n.headline}")
        lines.extend(["", "Filings (references only):"])
        for f in filings[:10]:
            lines.append(f"- {f.form_type} filed {f.filed_at} accession {f.accession}")
        lines.append("")
        lines.append("Return the JSON object as specified.")
        return "\n".join(lines)

    def _anthropic_messages(self, user_text: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": user_text}],
        }
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        parts = data.get("content") or []
        texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
        return "".join(texts)
