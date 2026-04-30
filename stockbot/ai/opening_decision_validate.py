"""
Step 3.5 — strict JSON parse + schema validation for opening-bell AI decisions.

Pure validation only: no I/O, logging, retries, prompts, or execution.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from typing import Any

_TOP_LEVEL_KEYS = frozenset({"trade_date", "decision_status", "market_read", "candidates"})
_CANDIDATE_KEYS = frozenset({"rank", "symbol", "direction", "confidence", "reason"})


def _fallback_no_trade(trade_date: str) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "decision_status": "no_trade",
        "market_read": {"summary": "invalid output"},
        "candidates": [],
    }


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _word_count(text: str) -> int:
    if not text.strip():
        return 0
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _finite_number(x: Any) -> bool:
    if type(x) not in (int, float):
        return False
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(v)


def _validate_payload(
    obj: Mapping[str, Any],
    *,
    expected_trade_date: str,
    allowed: frozenset[str],
) -> dict[str, Any] | None:
    if not _TOP_LEVEL_KEYS.issubset(obj.keys()):
        return None

    trade_date = obj["trade_date"]
    if not isinstance(trade_date, str) or trade_date != expected_trade_date:
        return None

    decision_status = obj["decision_status"]
    if decision_status not in ("trade", "no_trade"):
        return None

    market_read = obj["market_read"]
    if not isinstance(market_read, dict) or "summary" not in market_read:
        return None
    summary = market_read["summary"]
    if not isinstance(summary, str) or not summary.strip():
        return None

    candidates = obj["candidates"]
    if not isinstance(candidates, list):
        return None
    if len(candidates) > 3:
        return None

    if decision_status == "no_trade":
        if candidates:
            return None
        return {
            "trade_date": trade_date,
            "decision_status": decision_status,
            "market_read": {"summary": summary},
            "candidates": [],
        }

    if not candidates:
        return None

    out_candidates: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        if not isinstance(c, dict) or not _CANDIDATE_KEYS.issubset(c.keys()):
            return None
        rank = c["rank"]
        if type(rank) is not int:
            return None
        if rank != i + 1:
            return None

        symbol = c["symbol"]
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        sym_u = symbol.strip().upper()
        if sym_u not in allowed:
            return None

        direction = c["direction"]
        if direction not in ("long", "short"):
            return None

        conf = c["confidence"]
        if type(conf) not in (int, float):
            return None
        if not _finite_number(conf):
            return None
        cf = float(conf)
        if cf < 0.0 or cf > 1.0:
            return None

        reason = c["reason"]
        if reason is None or not isinstance(reason, str) or not reason.strip():
            return None
        if _word_count(reason) > 40:
            return None

        out_candidates.append(
            {
                "rank": rank,
                "symbol": sym_u,
                "direction": direction,
                "confidence": cf,
                "reason": reason.strip(),
            }
        )

    return {
        "trade_date": trade_date,
        "decision_status": decision_status,
        "market_read": {"summary": summary},
        "candidates": out_candidates,
    }


_LOW_CONF_HINT = re.compile(
    r"\b(confidence|cautious|uncertain|risk[- ]?off|avoid trading|stay sideline|no conviction|too risky)\b",
    re.I,
)


def _explicit_no_trade_subtype(summary: str) -> str:
    """Subtype when schema-valid AI payload chose ``no_trade``."""
    if _LOW_CONF_HINT.search(summary or ""):
        return "low_overall_confidence"
    return "explicit_ai_returned_no_trade"


def _diagnose_invalid_payload(
    parsed: Mapping[str, Any],
    *,
    expected_trade_date: str,
    allowed: frozenset[str],
) -> str:
    """Why strict validation failed (fallback invalid output)."""
    if not _TOP_LEVEL_KEYS.issubset(parsed.keys()):
        return "invalid_or_missing_ai_response"

    trade_date = parsed.get("trade_date")
    if not isinstance(trade_date, str) or trade_date != expected_trade_date:
        return "invalid_or_missing_ai_response"

    decision_status = parsed.get("decision_status")
    if decision_status not in ("trade", "no_trade"):
        return "invalid_or_missing_ai_response"

    market_read = parsed.get("market_read")
    if not isinstance(market_read, dict) or "summary" not in market_read:
        return "invalid_or_missing_ai_response"
    summary = market_read.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "invalid_or_missing_ai_response"

    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        return "invalid_or_missing_ai_response"
    if len(candidates) > 3:
        return "invalid_or_missing_ai_response"

    if decision_status == "no_trade":
        return "invalid_or_missing_ai_response"

    if not candidates:
        return "no_symbol_selected"

    for i, c in enumerate(candidates):
        if not isinstance(c, dict) or not _CANDIDATE_KEYS.issubset(c.keys()):
            return "invalid_or_missing_ai_response"
        rank = c.get("rank")
        if type(rank) is not int or rank != i + 1:
            return "invalid_or_missing_ai_response"

        symbol = c.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            return "invalid_or_missing_ai_response"
        sym_u = symbol.strip().upper()
        if sym_u not in allowed:
            return "risk_or_symbol_rejection"

        direction = c.get("direction")
        if direction not in ("long", "short"):
            return "invalid_or_missing_ai_response"

        conf = c.get("confidence")
        if type(conf) not in (int, float) or not _finite_number(conf):
            return "invalid_or_missing_ai_response"
        cf = float(conf)
        if cf < 0.0 or cf > 1.0:
            return "invalid_or_missing_ai_response"

        reason = c.get("reason")
        if reason is None or not isinstance(reason, str) or not reason.strip():
            return "invalid_or_missing_ai_response"
        if _word_count(reason) > 40:
            return "invalid_or_missing_ai_response"

    return "invalid_or_missing_ai_response"


def validate_opening_decision_response_detailed(
    raw_model_text: str,
    *,
    expected_trade_date: str,
    allowed_symbols: Collection[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns ``(validated_decision, meta)``. Meta keys:

    - ``initial_decision_status``: ``trade`` | ``no_trade``
    - ``no_trade_subtype``: diagnostic bucket when initial outcome is ``no_trade`` (fallback or valid explicit).
    """
    allowed = frozenset(s.strip().upper() for s in allowed_symbols if str(s).strip())
    parsed = _parse_json_object(raw_model_text.strip())
    if parsed is None:
        meta = {
            "initial_decision_status": "no_trade",
            "no_trade_subtype": "invalid_or_missing_ai_response",
        }
        return _fallback_no_trade(expected_trade_date), meta

    ok = _validate_payload(parsed, expected_trade_date=expected_trade_date, allowed=allowed)
    if ok is None:
        sub = _diagnose_invalid_payload(
            parsed, expected_trade_date=expected_trade_date, allowed=allowed
        )
        meta = {"initial_decision_status": "no_trade", "no_trade_subtype": sub}
        return _fallback_no_trade(expected_trade_date), meta

    if ok["decision_status"] == "no_trade":
        summary = str(ok["market_read"]["summary"])
        meta = {
            "initial_decision_status": "no_trade",
            "no_trade_subtype": _explicit_no_trade_subtype(summary),
        }
        return ok, meta

    meta = {"initial_decision_status": "trade", "no_trade_subtype": None}
    return ok, meta


def validate_opening_decision_response(
    raw_model_text: str,
    *,
    expected_trade_date: str,
    allowed_symbols: Collection[str],
) -> dict[str, Any]:
    decision, _meta = validate_opening_decision_response_detailed(
        raw_model_text,
        expected_trade_date=expected_trade_date,
        allowed_symbols=allowed_symbols,
    )
    return decision
