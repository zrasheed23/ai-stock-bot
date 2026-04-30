"""
Limited rank-1-only recovery when Step 3 chose ``no_trade`` but raw JSON still implies a strong setup.

Does not replace AI validation globally; capped per replay budget (live callers omit budget → no override).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Collection, Mapping
from typing import Any

from stockbot.execution.opening_allocation import OpeningAllocationConfig

_LOG = logging.getLogger("stockbot.execution.opening_source_override")

# Override uses hard minimums aligned with allocation quality (not soft-band).
SOURCE_NO_TRADE_OVERRIDE_MIN_EM = 0.011


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _premarket_expected_move_proxy(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    gap = _as_float(row.get("gap_close_vs_prior_close_pct"))
    pm = _as_float(row.get("pm_session_return_pct"))
    total = 0.0
    got = False
    if gap is not None:
        total += abs(gap)
        got = True
    if pm is not None:
        total += abs(pm)
        got = True
    return total if got else None


def _step2_volume_for_rank_override(
    row: dict[str, Any] | None, config: OpeningAllocationConfig, rank: int
) -> tuple[float | None, str]:
    min_v = float(config.min_pm_volume)
    if rank == 1:
        vol = _as_float(row.get("pm_volume")) if isinstance(row, Mapping) else None
        if vol is not None and math.isfinite(vol) and vol >= min_v:
            return vol, ""
        return min_v, ""
    prefix = "RANK2" if rank == 2 else "RANK3"
    if row is None:
        return None, f"{prefix}_STEP2_NOT_OK"
    if row.get("status") != "ok":
        return None, f"{prefix}_STEP2_NOT_OK"
    vol = _as_float(row.get("pm_volume"))
    if vol is None:
        return None, f"{prefix}_PM_VOLUME_BELOW_MIN"
    if vol < min_v:
        return None, f"{prefix}_PM_VOLUME_BELOW_MIN"
    return vol, ""


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


def _extract_rank1_candidate(
    parsed: Mapping[str, Any],
    *,
    allowed: frozenset[str],
) -> dict[str, Any] | None:
    """First list element that looks like rank 1 with strict shape (same intent as Step 3 schema)."""
    cands = parsed.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    c0 = cands[0]
    if not isinstance(c0, dict):
        return None
    need = frozenset({"rank", "symbol", "direction", "confidence", "reason"})
    if not need.issubset(c0.keys()):
        return None
    if c0.get("rank") != 1:
        return None
    symbol = c0.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    sym_u = symbol.strip().upper()
    if sym_u not in allowed:
        return None
    direction = c0.get("direction")
    if direction not in ("long", "short"):
        return None
    conf = c0.get("confidence")
    if type(conf) not in (int, float):
        return None
    cf = float(conf)
    if not math.isfinite(cf) or cf < 0.0 or cf > 1.0:
        return None
    reason = c0.get("reason")
    if reason is None or not isinstance(reason, str) or not reason.strip():
        return None
    if _word_count(reason) > 40:
        return None
    return {
        "rank": 1,
        "symbol": sym_u,
        "direction": direction,
        "confidence": cf,
        "reason": reason.strip(),
    }


def try_source_no_trade_rank1_override(
    *,
    raw_model_text: str,
    opening_meta: Mapping[str, Any],
    trade_date_str: str,
    watchlist: Collection[str],
    step2_by_symbol: Mapping[str, dict[str, Any]],
    budget: dict[str, int] | None,
    config: OpeningAllocationConfig,
    market_read_summary: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    If Step 3 outcome was ``no_trade`` but rank1 in raw JSON clears strict quality gates,
    emit a synthetic ``trade`` payload with **rank 1 only** (slot 2 untouched — absent).

    ``budget``: mutable ``{"max": int, "used": int}``; when ``None``, overrides are disabled.
    """
    info: dict[str, Any] = {"applied": False}
    if budget is None:
        return None, info
    if opening_meta.get("initial_decision_status") != "no_trade":
        return None, info

    max_n = int(budget.get("max", 0))
    used = int(budget.get("used", 0))
    if max_n <= 0 or used >= max_n:
        info["blocked"] = "budget_exhausted"
        return None, info

    allowed = frozenset(s.strip().upper() for s in watchlist if str(s).strip())
    parsed = _parse_json_object(raw_model_text.strip())
    if parsed is None:
        return None, info

    rank1 = _extract_rank1_candidate(parsed, allowed=allowed)
    if rank1 is None:
        return None, info

    cf = float(rank1["confidence"])
    if cf < float(config.c1_hard):
        return None, info

    sym_u = str(rank1["symbol"])
    row1 = step2_by_symbol.get(sym_u) if isinstance(step2_by_symbol, Mapping) else None
    if not isinstance(row1, dict):
        row1 = None

    em = _premarket_expected_move_proxy(row1)
    if em is None or em < float(SOURCE_NO_TRADE_OVERRIDE_MIN_EM):
        return None, info

    vol1, vfail = _step2_volume_for_rank_override(row1, config, 1)
    if vol1 is None:
        return None, info

    budget["used"] = used + 1
    summary_fallback = (
        (market_read_summary or "").strip()
        or "SOURCE_NO_TRADE_OVERRIDE synthetic trade (rank1 only)."
    )
    synthetic = {
        "trade_date": trade_date_str,
        "decision_status": "trade",
        "market_read": {"summary": summary_fallback[:400]},
        "candidates": [rank1],
    }
    _LOG.info(
        "SOURCE_NO_TRADE_OVERRIDE symbol=%s confidence=%.6f expected_move=%.6f",
        sym_u,
        cf,
        float(em),
    )
    info["applied"] = True
    info["symbol"] = sym_u
    return synthetic, info
