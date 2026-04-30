"""
Step 5 — deterministic broker execution plan from Step 4 allocation.

No price fetching, no Step 4 mutation, no capital redistribution on skips.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class OpeningExecutionPlanConfig:
    """Tunable execution planning constants."""

    min_order_notional: float = 100.0
    notional_market_supported: bool = True


def _scheduled_for_et_iso(trade_date_str: str, session_open_et: time | None = None) -> str:
    """Session open time America/New_York on trade_date (ISO 8601 with offset). Default 09:30."""
    d = date.fromisoformat(trade_date_str.strip())
    t = session_open_et if session_open_et is not None else time(9, 30, 0)
    dt = datetime.combine(d, t, tzinfo=_ET)
    return dt.isoformat()


def _deployable_cents_floor(deployable_usd: float) -> int:
    if deployable_usd <= 0 or not math.isfinite(deployable_usd):
        return 0
    return int(math.floor(deployable_usd * 100.0 + 1e-9))


def _base_cents_for_line(weight: float, deployable_usd: float) -> int:
    """Per-line floor(weight * deployable_usd * 100) cents (integer)."""
    if deployable_usd <= 0 or not math.isfinite(deployable_usd):
        return 0
    if not math.isfinite(weight) or weight < 0:
        return 0
    return int(math.floor(weight * deployable_usd * 100.0 + 1e-9))


def _leftover_bonuses_by_index(
    trades: list[Any],
    deployable_usd: float,
    min_order_notional: float,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Base cents for every long row; leftover-cent bonuses only for early-eligible long rows.

    Early-eligible (for leftover pool): mapping, long, and base_dollars >= min_order_notional
    where base_dollars = floor(weight * deployable * 100) / 100.

    Leftover = deployable_cents - sum(base cents over early-eligible rows only).
    Bonuses: +1 cent per step in round-robin rank order among early-eligible (rank 1, 2, 3, repeat).
    """
    all_long_base: dict[int, int] = {}
    eligible: list[tuple[int, int, int]] = []
    for idx, row in enumerate(trades):
        if not isinstance(row, Mapping):
            continue
        if row.get("direction") != "long":
            continue
        rank = row.get("ai_rank")
        try:
            rank_i = int(rank)
        except (TypeError, ValueError):
            rank_i = 0
        w = float(row.get("capital_weight", 0.0) or 0.0)
        bc = _base_cents_for_line(w, float(deployable_usd))
        all_long_base[idx] = bc
        base_dollars = bc / 100.0
        if base_dollars >= float(min_order_notional):
            eligible.append((idx, rank_i, bc))

    bonus_by_idx = {idx: 0 for idx in all_long_base}

    d_cents = _deployable_cents_floor(float(deployable_usd))
    sum_base = sum(t[2] for t in eligible)
    leftover = d_cents - sum_base
    if leftover < 0:
        leftover = 0
    if eligible and leftover > 0:

        def _rank_sort_key(rank_i: int) -> int:
            if 1 <= rank_i <= 3:
                return rank_i
            return 999

        order = sorted(eligible, key=lambda t: (_rank_sort_key(t[1]), t[0]))
        n = len(order)
        for k in range(leftover):
            i = order[k % n][0]
            bonus_by_idx[i] = bonus_by_idx.get(i, 0) + 1

    return all_long_base, bonus_by_idx


def _ref_price_ok(px: Any) -> bool:
    try:
        v = float(px)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0


def _as_of_policy(
    deployable_usd: float,
    cfg: OpeningExecutionPlanConfig,
) -> dict[str, Any]:
    return {
        "deployable_usd": float(deployable_usd),
        "min_order_notional": float(cfg.min_order_notional),
        "execution_mode_priority": ["notional_market", "shares_market"],
        "pricing_source": "caller_supplied_reference_prices_only",
    }


def build_opening_execution_plan(
    step4_allocation: Mapping[str, Any],
    *,
    deployable_usd: float,
    reference_prices: Mapping[str, float],
    config: OpeningExecutionPlanConfig | None = None,
    scheduled_session_open_et: time | None = None,
) -> dict[str, Any]:
    """
    Build execution instructions from Step 4 output.

    ``reference_prices`` is only consulted when ``shares_market`` fallback is required.
    Keys are matched case-insensitively on ``symbol``.
    """
    cfg = config or OpeningExecutionPlanConfig()
    trade_date = str(step4_allocation.get("trade_date", ""))
    instructions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    policy = _as_of_policy(deployable_usd, cfg)

    prep = str(step4_allocation.get("preparation_status", ""))
    trades = step4_allocation.get("trades")
    if prep != "ready" or not isinstance(trades, list) or len(trades) == 0:
        return {
            "trade_date": trade_date,
            "as_of_policy": policy,
            "instructions": [],
            "skipped": [],
            "totals": {
                "requested_notional_usd": 0.0,
                "instruction_count": 0,
                "skipped_count": 0,
            },
        }

    ref_upper = {str(k).strip().upper(): float(v) for k, v in reference_prices.items() if _ref_price_ok(v)}
    scheduled_for = _scheduled_for_et_iso(trade_date, scheduled_session_open_et)

    base_by_idx, bonus_by_idx = _leftover_bonuses_by_index(
        trades, float(deployable_usd), float(cfg.min_order_notional)
    )

    total_requested = 0.0

    for idx, row in enumerate(trades):
        if not isinstance(row, Mapping):
            continue
        sym = str(row.get("symbol", "")).strip().upper()
        direction = row.get("direction")
        rank = row.get("ai_rank")
        try:
            rank_i = int(rank)
        except (TypeError, ValueError):
            rank_i = 0

        if direction != "long":
            skipped.append(
                {
                    "rank": rank_i,
                    "symbol": sym,
                    "reason_code": "SHORT_NOT_SUPPORTED",
                    "detail": "",
                }
            )
            continue

        base_c = base_by_idx.get(idx, 0)
        bonus_c = bonus_by_idx.get(idx, 0)
        total_cents = base_c + bonus_c
        dollars_i = total_cents / 100.0

        if dollars_i < cfg.min_order_notional:
            skipped.append(
                {
                    "rank": rank_i,
                    "symbol": sym,
                    "reason_code": "NOTIONAL_TOO_SMALL",
                    "detail": "",
                }
            )
            continue

        use_notional = bool(cfg.notional_market_supported)
        if use_notional:
            instructions.append(
                {
                    "symbol": sym,
                    "side": "buy",
                    "direction": "long",
                    "mode": "notional_market",
                    "notional_usd": dollars_i,
                    "shares": None,
                    "ref_price": None,
                    "scheduled_for": scheduled_for,
                    "rank": rank_i,
                }
            )
            total_requested += dollars_i
            continue

        px = ref_upper.get(sym)
        if px is None or not _ref_price_ok(px):
            skipped.append(
                {
                    "rank": rank_i,
                    "symbol": sym,
                    "reason_code": "NO_VALID_REFERENCE_PRICE",
                    "detail": "",
                }
            )
            continue

        shares = math.floor(dollars_i / px + 1e-12)
        if shares < 1:
            skipped.append(
                {
                    "rank": rank_i,
                    "symbol": sym,
                    "reason_code": "INSUFFICIENT_NOTIONAL_FOR_ONE_SHARE",
                    "detail": "",
                }
            )
            continue

        instructions.append(
            {
                "symbol": sym,
                "side": "buy",
                "direction": "long",
                "mode": "shares_market",
                "notional_usd": None,
                "shares": int(shares),
                "ref_price": float(px),
                "scheduled_for": scheduled_for,
                "rank": rank_i,
            }
        )
        total_requested += round(shares * px, 2)

    return {
        "trade_date": trade_date,
        "as_of_policy": policy,
        "instructions": instructions,
        "skipped": skipped,
        "totals": {
            "requested_notional_usd": round(total_requested, 2),
            "instruction_count": len(instructions),
            "skipped_count": len(skipped),
        },
    }
