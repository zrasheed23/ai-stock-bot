"""
Step 5.5 — strict validation of Step 5 opening execution plans (no broker I/O).
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

_NY = ZoneInfo("America/New_York")

_TOP_LEVEL_KEYS = frozenset({"trade_date", "as_of_policy", "instructions", "skipped", "totals"})
_POLICY_KEYS = frozenset(
    {"deployable_usd", "min_order_notional", "execution_mode_priority", "pricing_source"}
)
_MODE_PRIORITY = ["notional_market", "shares_market"]
_INSTRUCTION_KEYS = frozenset(
    {
        "symbol",
        "side",
        "direction",
        "mode",
        "notional_usd",
        "shares",
        "ref_price",
        "scheduled_for",
        "rank",
    }
)
_TOTALS_KEYS = frozenset({"requested_notional_usd", "instruction_count", "skipped_count"})
_SKIP_REASONS = frozenset(
    {
        "SHORT_NOT_SUPPORTED",
        "NOTIONAL_TOO_SMALL",
        "NO_VALID_REFERENCE_PRICE",
        "INSUFFICIENT_NOTIONAL_FOR_ONE_SHARE",
    }
)
_SKIP_ALLOWED_KEYS = frozenset({"rank", "symbol", "reason_code", "detail"})

# Reject a '.' immediately after HH:MM:SS (fractional seconds in the timestamp text).
_FRACTIONAL_SECOND_IN_TEXT = re.compile(r"T\d{2}:\d{2}:\d{2}\.")


def _fallback_trade_date(plan: Any) -> str:
    if isinstance(plan, Mapping):
        td = plan.get("trade_date")
        if isinstance(td, str):
            return td
    return ""


def _no_execution(plan: Any) -> dict[str, Any]:
    return {
        "trade_date": _fallback_trade_date(plan),
        "execution_status": "no_execution",
        "instructions": [],
        "skipped": [],
        "reason": "invalid_execution_plan",
    }


def _exact_keys(obj: Mapping[str, Any], expected: frozenset[str]) -> bool:
    return set(obj.keys()) == expected


def _valid_yyyy_mm_dd(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 10:
        return False
    try:
        date.fromisoformat(s)
    except ValueError:
        return False
    return True


def _is_int_not_bool(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _finite_real(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float):
        return math.isfinite(x)
    return False


def _finite_non_negative(x: Any) -> bool:
    return _finite_real(x) and float(x) >= 0.0


def _finite_positive(x: Any) -> bool:
    return _finite_real(x) and float(x) > 0.0


def _scheduled_for_valid(scheduled_for: Any, trade_date: str) -> bool:
    if not isinstance(scheduled_for, str):
        return False
    if _FRACTIONAL_SECOND_IN_TEXT.search(scheduled_for):
        return False
    try:
        dt = datetime.fromisoformat(scheduled_for.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        return False
    if dt.microsecond != 0:
        return False
    local = dt.astimezone(_NY)
    if local.date() != date.fromisoformat(trade_date):
        return False
    if (local.hour, local.minute, local.second) not in ((9, 30, 0), (10, 30, 0)):
        return False
    return True


def _instruction_leg_cents(inst: Mapping[str, Any]) -> int | None:
    mode = inst.get("mode")
    if mode == "notional_market":
        nu = inst.get("notional_usd")
        if not _finite_positive(nu):
            return None
        return int(round(float(nu) * 100.0))
    if mode == "shares_market":
        sh = inst.get("shares")
        px = inst.get("ref_price")
        if not _is_int_not_bool(sh) or sh < 1:
            return None
        if not _finite_positive(px):
            return None
        return int(round(float(sh) * float(px) * 100.0))
    return None


def validate_opening_execution_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate an untrusted Step 5 execution plan.

    On success returns a deep copy of the plan. On any failure returns the
    deterministic no-execution object (no repair, no mutation of input).
    """
    if not isinstance(plan, Mapping):
        return _no_execution(plan)

    if not _exact_keys(plan, _TOP_LEVEL_KEYS):
        return _no_execution(plan)

    trade_date = plan["trade_date"]
    if not isinstance(trade_date, str) or not _valid_yyyy_mm_dd(trade_date):
        return _no_execution(plan)

    policy = plan["as_of_policy"]
    if not isinstance(policy, Mapping) or not _exact_keys(policy, _POLICY_KEYS):
        return _no_execution(plan)

    du = policy["deployable_usd"]
    if not _finite_non_negative(du):
        return _no_execution(plan)

    mo = policy["min_order_notional"]
    if not _finite_positive(mo):
        return _no_execution(plan)

    emp = policy["execution_mode_priority"]
    if not isinstance(emp, list) or emp != _MODE_PRIORITY:
        return _no_execution(plan)

    ps = policy["pricing_source"]
    if not isinstance(ps, str) or ps.strip() == "":
        return _no_execution(plan)

    instructions = plan["instructions"]
    if not isinstance(instructions, list):
        return _no_execution(plan)
    if len(instructions) > 3:
        return _no_execution(plan)

    seen_symbols: set[str] = set()
    total_leg_cents = 0

    for inst in instructions:
        if not isinstance(inst, Mapping) or not _exact_keys(inst, _INSTRUCTION_KEYS):
            return _no_execution(plan)

        sym = inst["symbol"]
        if not isinstance(sym, str) or sym.strip() == "":
            return _no_execution(plan)
        key = sym.strip().upper()
        if key in seen_symbols:
            return _no_execution(plan)
        seen_symbols.add(key)

        if inst["side"] != "buy":
            return _no_execution(plan)
        if inst["direction"] != "long":
            return _no_execution(plan)

        mode = inst["mode"]
        if mode not in ("notional_market", "shares_market"):
            return _no_execution(plan)

        rk = inst["rank"]
        if not _is_int_not_bool(rk) or rk < 1 or rk > 3:
            return _no_execution(plan)

        sf = inst["scheduled_for"]
        if not _scheduled_for_valid(sf, trade_date):
            return _no_execution(plan)

        if mode == "notional_market":
            if inst["shares"] is not None:
                return _no_execution(plan)
            if inst["ref_price"] is not None:
                return _no_execution(plan)
            if not _finite_positive(inst["notional_usd"]):
                return _no_execution(plan)
        else:
            if not _is_int_not_bool(inst["shares"]) or inst["shares"] < 1:
                return _no_execution(plan)
            if not _finite_positive(inst["ref_price"]):
                return _no_execution(plan)
            if inst["notional_usd"] is not None:
                return _no_execution(plan)

        lc = _instruction_leg_cents(inst)
        if lc is None:
            return _no_execution(plan)
        total_leg_cents += lc

    skipped = plan["skipped"]
    if not isinstance(skipped, list):
        return _no_execution(plan)

    for item in skipped:
        if not isinstance(item, Mapping):
            return _no_execution(plan)
        ks = set(item.keys())
        if not ks.issubset(_SKIP_ALLOWED_KEYS):
            return _no_execution(plan)
        if "rank" not in item or "symbol" not in item or "reason_code" not in item:
            return _no_execution(plan)
        if not _is_int_not_bool(item["rank"]):
            return _no_execution(plan)
        if not isinstance(item["symbol"], str):
            return _no_execution(plan)
        if "detail" in item and not isinstance(item["detail"], str):
            return _no_execution(plan)
        rc = item["reason_code"]
        if not isinstance(rc, str) or rc not in _SKIP_REASONS:
            return _no_execution(plan)

    totals = plan["totals"]
    if not isinstance(totals, Mapping) or not _exact_keys(totals, _TOTALS_KEYS):
        return _no_execution(plan)

    rnu = totals["requested_notional_usd"]
    if not _finite_non_negative(rnu):
        return _no_execution(plan)

    ic = totals["instruction_count"]
    sc = totals["skipped_count"]
    if not _is_int_not_bool(ic) or ic < 0 or ic != len(instructions):
        return _no_execution(plan)
    if not _is_int_not_bool(sc) or sc < 0 or sc != len(skipped):
        return _no_execution(plan)

    deployable_cents = int(round(float(du) * 100.0))
    if total_leg_cents > deployable_cents:
        return _no_execution(plan)

    return copy.deepcopy(dict(plan))


def diagnose_opening_execution_plan(plan: Any) -> dict[str, Any]:
    """
    Explain why ``validate_opening_execution_plan`` would reject a plan (read-only; no mutation).

    Returns ``{"ok": bool, "failure_code": str | None, "failure_detail": str | None}``.
    """

    def _fail(code: str, detail: str | None = None) -> dict[str, Any]:
        return {"ok": False, "failure_code": code, "failure_detail": detail}

    if not isinstance(plan, Mapping):
        return _fail("plan_not_mapping")

    if not _exact_keys(plan, _TOP_LEVEL_KEYS):
        return _fail("bad_top_level_keys", f"got={sorted(plan.keys())}")

    trade_date = plan["trade_date"]
    if not isinstance(trade_date, str) or not _valid_yyyy_mm_dd(trade_date):
        return _fail("bad_trade_date", repr(trade_date))

    policy = plan["as_of_policy"]
    if not isinstance(policy, Mapping) or not _exact_keys(policy, _POLICY_KEYS):
        return _fail("bad_as_of_policy")

    du = policy["deployable_usd"]
    if not _finite_non_negative(du):
        return _fail("bad_deployable_usd", repr(du))

    mo = policy["min_order_notional"]
    if not _finite_positive(mo):
        return _fail("bad_min_order_notional", repr(mo))

    emp = policy["execution_mode_priority"]
    if not isinstance(emp, list) or emp != _MODE_PRIORITY:
        return _fail("bad_execution_mode_priority")

    ps = policy["pricing_source"]
    if not isinstance(ps, str) or ps.strip() == "":
        return _fail("bad_pricing_source")

    instructions = plan["instructions"]
    if not isinstance(instructions, list):
        return _fail("instructions_not_list")
    if len(instructions) > 3:
        return _fail("too_many_instructions", str(len(instructions)))

    seen_symbols: set[str] = set()
    total_leg_cents = 0

    for i, inst in enumerate(instructions):
        if not isinstance(inst, Mapping) or not _exact_keys(inst, _INSTRUCTION_KEYS):
            return _fail("bad_instruction_shape", f"index={i}")

        sym = inst["symbol"]
        if not isinstance(sym, str) or sym.strip() == "":
            return _fail("bad_instruction_symbol", f"index={i}")

        key = sym.strip().upper()
        if key in seen_symbols:
            return _fail("duplicate_instruction_symbol", key)
        seen_symbols.add(key)

        if inst["side"] != "buy":
            return _fail("instruction_side_not_buy", f"index={i}")
        if inst["direction"] != "long":
            return _fail("instruction_direction_not_long", f"index={i}")

        mode = inst["mode"]
        if mode not in ("notional_market", "shares_market"):
            return _fail("bad_instruction_mode", f"index={i} mode={mode!r}")

        rk = inst["rank"]
        if not _is_int_not_bool(rk) or rk < 1 or rk > 3:
            return _fail("bad_instruction_rank", f"index={i} rank={rk!r}")

        sf = inst["scheduled_for"]
        if not _scheduled_for_valid(sf, trade_date):
            return _fail(
                "bad_scheduled_for",
                f"index={i} scheduled_for={sf!r} (expect 09:30 or 10:30 ET on {trade_date})",
            )

        if mode == "notional_market":
            if inst["shares"] is not None:
                return _fail("notional_has_shares", f"index={i}")
            if inst["ref_price"] is not None:
                return _fail("notional_has_ref_price", f"index={i}")
            if not _finite_positive(inst["notional_usd"]):
                return _fail("bad_notional_usd", f"index={i} nu={inst.get('notional_usd')!r}")
        else:
            if not _is_int_not_bool(inst["shares"]) or inst["shares"] < 1:
                return _fail("bad_shares", f"index={i}")
            if not _finite_positive(inst["ref_price"]):
                return _fail("bad_ref_price", f"index={i}")
            if inst["notional_usd"] is not None:
                return _fail("shares_mode_has_notional", f"index={i}")

        lc = _instruction_leg_cents(inst)
        if lc is None:
            return _fail("instruction_leg_cents_none", f"index={i}")
        total_leg_cents += lc

    skipped = plan["skipped"]
    if not isinstance(skipped, list):
        return _fail("skipped_not_list")

    for j, item in enumerate(skipped):
        if not isinstance(item, Mapping):
            return _fail("bad_skip_item", f"index={j}")
        ks = set(item.keys())
        if not ks.issubset(_SKIP_ALLOWED_KEYS):
            return _fail("bad_skip_keys", f"index={j}")
        if "rank" not in item or "symbol" not in item or "reason_code" not in item:
            return _fail("skip_missing_required", f"index={j}")
        if not _is_int_not_bool(item["rank"]):
            return _fail("bad_skip_rank", f"index={j}")
        if not isinstance(item["symbol"], str):
            return _fail("bad_skip_symbol", f"index={j}")
        if "detail" in item and not isinstance(item["detail"], str):
            return _fail("bad_skip_detail", f"index={j}")
        rc = item["reason_code"]
        if not isinstance(rc, str) or rc not in _SKIP_REASONS:
            return _fail("bad_skip_reason_code", f"index={j} rc={rc!r}")

    totals = plan["totals"]
    if not isinstance(totals, Mapping) or not _exact_keys(totals, _TOTALS_KEYS):
        return _fail("bad_totals")

    rnu = totals["requested_notional_usd"]
    if not _finite_non_negative(rnu):
        return _fail("bad_totals_requested_notional", repr(rnu))

    ic = totals["instruction_count"]
    sc = totals["skipped_count"]
    if not _is_int_not_bool(ic) or ic < 0 or ic != len(instructions):
        return _fail(
            "totals_instruction_count_mismatch",
            f"ic={ic!r} len_instructions={len(instructions)}",
        )
    if not _is_int_not_bool(sc) or sc < 0 or sc != len(skipped):
        return _fail("totals_skipped_count_mismatch", f"sc={sc!r} len_skipped={len(skipped)}")

    deployable_cents = int(round(float(du) * 100.0))
    if total_leg_cents > deployable_cents:
        return _fail(
            "total_notional_exceeds_deployable",
            f"leg_cents={total_leg_cents} deployable_cents={deployable_cents}",
        )

    return {"ok": True, "failure_code": None, "failure_detail": None}
