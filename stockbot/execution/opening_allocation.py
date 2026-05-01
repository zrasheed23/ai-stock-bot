"""
Step 4 — deterministic opening-bell trade acceptance and capital weights.

Pure rules: no broker calls, no AI, no ingestion.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from stockbot.execution.orders import symbols_same_sector_theme

_LOG = logging.getLogger("stockbot.execution.opening_allocation")


def _log_opening_accept(
    *,
    confidence: float,
    threshold: float,
    expected_move: float | None,
) -> None:
    em_str = "na" if expected_move is None else f"{expected_move:.6f}"
    _LOG.info(
        "OPENING_ACCEPT decision=accept confidence=%.6f threshold=%.6f expected_move=%s",
        confidence,
        threshold,
        em_str,
    )


def _log_opening_reject(
    reason: str,
    *,
    confidence: float | None,
    threshold: float | None,
    expected_move: float | None = None,
) -> None:
    c_str = "na" if confidence is None else f"{confidence:.6f}"
    t_str = "na" if threshold is None else f"{threshold:.6f}"
    em_str = "na" if expected_move is None else f"{expected_move:.6f}"
    _LOG.info(
        "OPENING_REJECT reason=%s confidence=%s threshold=%s expected_move=%s",
        reason,
        c_str,
        t_str,
        em_str,
    )


def _log_opening_skip_primary(
    reason: str,
    *,
    symbol: str = "",
    confidence: float | None = None,
    threshold: float | None = None,
) -> None:
    c_str = "na" if confidence is None else f"{confidence:.6f}"
    t_str = "na" if threshold is None else f"{threshold:.6f}"
    _LOG.info(
        "OPENING_SKIP_PRIMARY reason=%s symbol=%s confidence=%s threshold=%s",
        reason,
        symbol.strip().upper() if symbol else "",
        c_str,
        t_str,
    )


def _log_slot2_reject(
    reason: str,
    *,
    rank2_confidence: float | None,
    rank2_expected_move: float | None,
    score_gap: float | None,
) -> None:
    c_str = "na" if rank2_confidence is None else f"{rank2_confidence:.6f}"
    em_str = "na" if rank2_expected_move is None else f"{rank2_expected_move:.6f}"
    g_str = "na" if score_gap is None else f"{score_gap:.6f}"
    _LOG.info(
        "SLOT2_REJECT reason=%s rank2_confidence=%s rank2_expected_move=%s score_gap=%s",
        reason,
        c_str,
        em_str,
        g_str,
    )


@dataclass(frozen=True)
class OpeningAllocationConfig:
    # Rank 2 — elite-only gates (opening AI path); modestly tighter than legacy.
    c2: float = 0.667
    c3: float = 0.72
    # Rank2 must stay close to rank1 on confidence (replaces loose d12=0.12 for slot2).
    slot2_max_gap_vs_rank1: float = 0.045
    d23: float = 0.08
    r32: float = 0.92
    min_pm_volume: float = 100_000.0
    # Rank 1: lowered slightly vs legacy 0.605 to improve deployment; soft band recovers strong movers.
    c1_hard: float = 0.565
    c1_soft_margin: float = 0.042
    # Below this confidence rank 1 never accepts (even soft band).
    c1_absolute_floor: float = 0.52
    # Premarket excursion proxy = |gap| + |pm_session_return|; soft band requires >= this.
    c1_soft_min_expected_move: float = 0.0105
    rank2_abs_min: float = 0.688
    slot2_min_expected_move: float = 0.0115
    slot2_same_sector_max_gap: float = 0.028
    slot2_same_sector_min_confidence: float = 0.708
    dominance_margin: float = 0.042
    # If rank1 fails the strict gap rule, allow pass when rank1 is strong and tape supports upside.
    dominance_bypass_rank1_min: float = 0.72
    dominance_bypass_em_min: float = 0.015
    # --- Mid-morning-only (opening keeps defaults so behavior is unchanged) ---
    # When set: rank1 must satisfy excursion proxy >= this on both hard_pass and soft_pass paths.
    rank1_min_expected_move_absolute: float | None = None
    apply_midmorning_strict_gates: bool = False
    # Signed intraday session return (fraction): long needs pm >= min; short needs pm <= -min.
    mm_min_signed_pm_return_pct: float = 0.0
    # Long rejected if gap vs prior close worse than -this; short rejected if gap > +this.
    mm_max_gap_counter_trend_pct: float = 0.0
    # SPY intraday return from packet market_context must align with direction (risk-on / risk-off).
    mm_spy_min_signed_alignment_pct: float = 0.0
    # Set True when ``opening_allocation_config_from_env()`` sees STOCKBOT_SLOT2_RELAX_OPENING (conservative slot2).
    slot2_relaxed_opening: bool = False
    # Optional: opening_two_leg only — extra gates + small slot-2 weight cap (env STOCKBOT_SLOT2_ELITE_ONLY_OPENING).
    slot2_elite_only_opening: bool = False
    slot2_elite_min_confidence: float = 0.718
    slot2_elite_max_confidence_gap: float = 0.028
    slot2_elite_min_pm_volume_multiplier: float = 1.3
    slot2_elite_max_expected_move_gap: float = 0.008
    slot2_elite_overlap_max_gap: float = 0.026
    slot2_elite_overlap_min_confidence: float = 0.725
    slot2_elite_max_slot2_weight: float = 0.12


# Stricter Step 4 profile for ~10:30 ET session only (does not alter opening defaults above).
MIDMORNING_ALLOCATION_CONFIG = OpeningAllocationConfig(
    c1_hard=0.618,
    c1_soft_margin=0.026,
    c1_absolute_floor=0.588,
    c1_soft_min_expected_move=0.0185,
    rank1_min_expected_move_absolute=0.017,
    min_pm_volume=800_000.0,
    apply_midmorning_strict_gates=True,
    mm_min_signed_pm_return_pct=0.003,
    mm_max_gap_counter_trend_pct=0.007,
    mm_spy_min_signed_alignment_pct=0.0006,
    c2=0.94,
    rank2_abs_min=0.94,
    slot2_max_gap_vs_rank1=0.012,
    slot2_min_expected_move=0.026,
    slot2_same_sector_max_gap=0.018,
    slot2_same_sector_min_confidence=0.82,
    dominance_margin=0.036,
    dominance_bypass_rank1_min=0.78,
    dominance_bypass_em_min=0.019,
    slot2_relaxed_opening=False,
)


def opening_allocation_config_from_env() -> OpeningAllocationConfig:
    """Opening Step 4 config; optional slot2 flags via env (relax / elite-only weight path)."""
    import os

    cfg = OpeningAllocationConfig()
    if os.environ.get("STOCKBOT_SLOT2_RELAX_OPENING", "").strip().lower() in {"1", "true", "yes", "on"}:
        cfg = replace(
            cfg,
            slot2_relaxed_opening=True,
            slot2_min_expected_move=min(0.0105, cfg.slot2_min_expected_move),
            slot2_max_gap_vs_rank1=min(0.054, cfg.slot2_max_gap_vs_rank1 + 0.006),
        )
    if os.environ.get("STOCKBOT_SLOT2_ELITE_ONLY_OPENING", "").strip().lower() in {"1", "true", "yes", "on"}:
        cfg = replace(cfg, slot2_elite_only_opening=True)
    return cfg


@dataclass
class OpeningAllocationDiagnostics:
    """Per-day Step 4 diagnostics for replay / auditing (rank-1 soft band + reject breakdown)."""

    opening_reject_reason_counts: dict[str, int] = field(default_factory=dict)
    slot2_reject_reason_counts: dict[str, int] = field(default_factory=dict)
    slot1_full_sleeve_only: bool = False
    confidence_pass_direction_fail_count: int = 0
    confidence_direction_pass_volume_fail_count: int = 0
    soft_band_candidate_count: int = 0
    soft_band_expected_move_pass_count: int = 0
    soft_band_accepted_count: int = 0
    no_candidates_empty_count: int = 0
    decision_status_not_trade_count: int = 0

    def finalize_from_rejected(self, rejected: Sequence[Mapping[str, Any]]) -> None:
        """Fill ``opening_reject_reason_counts`` from the final rejected list (includes slot2/slot3)."""
        counts: dict[str, int] = {}
        for r in rejected:
            rc = r.get("reason_code")
            if not rc:
                continue
            k = str(rc)
            counts[k] = counts.get(k, 0) + 1
        self.opening_reject_reason_counts = counts
        s2c: dict[str, int] = {}
        for r in rejected:
            rc = r.get("reason_code")
            if not rc:
                continue
            k = str(rc)
            ar = r.get("ai_rank")
            if k.startswith("SLOT2_") or ar == 2:
                s2c[k] = s2c.get(k, 0) + 1
        self.slot2_reject_reason_counts = s2c

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _premarket_expected_move_proxy(row: dict[str, Any] | None) -> float | None:
    """
    Single-session excursion proxy (fraction of price): |overnight gap| + |premarket return|.
    Favors names with enough tape to support asymmetric trades vs stop; not a forecast.
    """
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


def _rank1_confidence_pass(
    conf1: float | None,
    row1: dict[str, Any] | None,
    config: OpeningAllocationConfig,
) -> tuple[bool, str, float | None]:
    """
    Returns (accept, mode, expected_move_proxy).

    mode is ``hard_pass``, ``soft_pass``, or a rejection token starting with ``rej_``.
    """
    em = _premarket_expected_move_proxy(row1)
    thr = float(config.c1_hard)
    margin = float(config.c1_soft_margin)
    soft_hi = thr
    soft_lo = max(float(config.c1_absolute_floor), thr - margin)
    min_em = float(config.c1_soft_min_expected_move)

    if conf1 is None:
        return False, "rej_invalid_confidence", em

    if conf1 < soft_lo:
        return False, "rej_below_confidence_floor", em

    mode: str
    if conf1 >= soft_hi:
        mode = "hard_pass"
    else:
        if em is None or em < min_em:
            return False, "rej_soft_band_expected_move", em
        mode = "soft_pass"

    floor_abs = config.rank1_min_expected_move_absolute
    if floor_abs is not None:
        if em is None or em < float(floor_abs):
            return False, "rej_rank1_expected_move_absolute_fail", em

    return True, mode, em


def _midmorning_rank1_strict(
    direction: str,
    row1: dict[str, Any] | None,
    market_context: Mapping[str, Any] | None,
    config: OpeningAllocationConfig,
) -> tuple[bool, str]:
    """Tape + benchmark alignment gates for mid-morning session only."""
    if row1 is None:
        return False, "MM_REJECT_RANK1_NO_ROW"
    pm = _as_float(row1.get("pm_session_return_pct"))
    gap = _as_float(row1.get("gap_close_vs_prior_close_pct"))
    min_pm = float(config.mm_min_signed_pm_return_pct)
    max_ct = float(config.mm_max_gap_counter_trend_pct)
    spy_need = float(config.mm_spy_min_signed_alignment_pct)

    if pm is None:
        return False, "MM_REJECT_PM_RET_MISSING"
    if direction == "long":
        if pm < min_pm:
            return False, "MM_REJECT_PM_DRIFT_LONG"
        if gap is not None and gap < -max_ct:
            return False, "MM_REJECT_GAP_COUNTER_LONG"
    elif direction == "short":
        if pm > -min_pm:
            return False, "MM_REJECT_PM_DRIFT_SHORT"
        if gap is not None and gap > max_ct:
            return False, "MM_REJECT_GAP_COUNTER_SHORT"
    else:
        return False, "MM_REJECT_DIRECTION"

    spy = _as_float(market_context.get("spy_premarket_return_pct")) if isinstance(market_context, Mapping) else None
    if spy is None:
        return False, "MM_REJECT_SPY_CONTEXT_MISSING"
    if direction == "long":
        if spy < spy_need:
            return False, "MM_REJECT_SPY_NOT_ALIGNED_LONG"
    else:
        if spy > -spy_need:
            return False, "MM_REJECT_SPY_NOT_ALIGNED_SHORT"

    return True, ""


def build_step2_index(step2_packet: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map upper(symbol) -> Step 2 symbol row from packet['symbols'] list."""
    if step2_packet is None:
        return {}
    syms = step2_packet.get("symbols")
    if not isinstance(syms, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in syms:
        if not isinstance(row, Mapping):
            continue
        s = row.get("symbol")
        if not isinstance(s, str) or not s.strip():
            continue
        out[s.strip().upper()] = dict(row)
    return out


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _direction_ok(d: Any) -> bool:
    return d == "long" or d == "short"


def _step2_volume_for_rank(row: dict[str, Any] | None, config: OpeningAllocationConfig, rank: int) -> tuple[float | None, str]:
    min_v = float(config.min_pm_volume)
    # Opening rank 1: never hard-reject on Step 2 envelope (status/volume); checkpoint trades did not
    # stall on premarket row quality for slot 1. Slot 2/3 unchanged.
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


def weight_accepted_trades(accepted: Sequence[Mapping[str, Any]]) -> list[float]:
    n = len(accepted)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    if n == 2:
        return [0.7, 0.3]
    if n == 3:
        return [0.7, 0.2, 0.1]
    raise ValueError(f"weight_accepted_trades: accepted count must be 0–3, got {n}")


_OPENING_SLOT2_MIN_CONF_KEEP = 0.685
_OPENING_SLOT2_GAP_SKIP = 0.068


def _slot2_elite_only_gate_failure(
    *,
    c2: float,
    gap: float,
    row2: dict[str, Any] | None,
    em1: float | None,
    em2: float | None,
    overlap: bool,
    config: OpeningAllocationConfig,
) -> str | None:
    """Elite-only opening slot-2 gate (``slot2_elite_only_opening``). None means pass."""
    if row2 is None or row2.get("status") != "ok":
        return "STEP2_STATUS_NOT_OK"
    pmv = _as_float(row2.get("pm_volume"))
    need_vol = float(config.min_pm_volume) * float(config.slot2_elite_min_pm_volume_multiplier)
    if pmv is None or pmv < need_vol:
        return "PM_VOLUME_BELOW_ELITE_MULT"
    if c2 < float(config.slot2_elite_min_confidence):
        return "CONFIDENCE_BELOW_ELITE_FLOOR"
    if gap > float(config.slot2_elite_max_confidence_gap):
        return "CONFIDENCE_GAP_TOO_WIDE"
    if em1 is None or em2 is None:
        return "EXPECTED_MOVE_PROXY_MISSING"
    if em2 < em1 - float(config.slot2_elite_max_expected_move_gap):
        return "EXPECTED_MOVE_TOO_WEAK_VS_RANK1"
    if overlap:
        if gap > float(config.slot2_elite_overlap_max_gap) or c2 < float(config.slot2_elite_overlap_min_confidence):
            return "OVERLAP_NOT_ELITE_PEER"
    return None


def opening_two_leg_capital_weights(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    config: OpeningAllocationConfig,
    step2_by_symbol: Mapping[str, dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Two accepted legs only: tighten or drop slot 2 by confidence gap, tape (expected-move proxy),
    and overlap elite rules. Full sleeve to rank 1 when slot 2 is stripped.
    """
    if len(accepted) != 2:
        return weight_accepted_trades(accepted), accepted, rejected

    a1, a2 = accepted[0], accepted[1]
    sym1 = str(a1["symbol"]).strip().upper()
    sym2 = str(a2["symbol"]).strip().upper()
    c1 = float(a1["ai_confidence"])
    c2 = float(a2["ai_confidence"])
    gap = c1 - c2
    row2 = step2_by_symbol.get(sym2) if isinstance(step2_by_symbol, Mapping) else None
    if not isinstance(row2, dict):
        row2 = None
    em2 = _premarket_expected_move_proxy(row2)
    overlap = symbols_same_sector_theme(sym1, sym2)
    rx = config.slot2_relaxed_opening
    elite_gap_lim = 0.034 if rx else 0.032
    elite_c2_floor = 0.702 if rx else 0.705
    elite = overlap and gap <= elite_gap_lim and c2 >= elite_c2_floor

    min_conf_keep = 0.678 if rx else _OPENING_SLOT2_MIN_CONF_KEEP
    gap_skip = 0.072 if rx else _OPENING_SLOT2_GAP_SKIP

    def _log_weight(final_w2: float, reason: str) -> None:
        em_str = "na" if em2 is None else f"{em2:.6f}"
        _LOG.info(
            "SLOT2_WEIGHT_DECISION slot1_raw_score=%.6f slot2_raw_score=%.6f score_gap=%.6f "
            "expected_move=%s elite=%s final_slot2_weight=%.4f reason=%s",
            c1,
            c2,
            gap,
            em_str,
            elite,
            final_w2,
            reason,
        )

    rej_out = list(rejected)

    min_em = float(config.slot2_min_expected_move)
    if em2 is None or em2 < min_em:
        _log_slot2_reject(
            "SLOT2_REJECT_STRIP_EXPECTED_MOVE_WEAK",
            rank2_confidence=c2,
            rank2_expected_move=em2,
            score_gap=gap,
        )
        _log_weight(0.0, "slot2_expected_move_strip")
        rej_out.append(
            {
                "ai_rank": 2,
                "symbol": sym2,
                "reason_code": "SLOT2_SKIP_EXPECTED_MOVE_OPENING",
                "detail": "",
            }
        )
        return [1.0], [a1], rej_out

    if c2 < min_conf_keep:
        _log_slot2_reject(
            "SLOT2_REJECT_STRIP_LOW_CONFIDENCE",
            rank2_confidence=c2,
            rank2_expected_move=em2,
            score_gap=gap,
        )
        _log_weight(0.0, "slot2_low_confidence_skip")
        rej_out.append(
            {
                "ai_rank": 2,
                "symbol": sym2,
                "reason_code": "SLOT2_SKIP_LOW_CONFIDENCE_OPENING",
                "detail": "",
            }
        )
        return [1.0], [a1], rej_out

    if not elite and (gap >= gap_skip or (gap > 0.048 and c2 < 0.675)):
        _log_slot2_reject(
            "SLOT2_REJECT_STRIP_LARGE_GAP_OR_WEAK",
            rank2_confidence=c2,
            rank2_expected_move=em2,
            score_gap=gap,
        )
        _log_weight(0.0, "slot2_large_gap_skip")
        rej_out.append(
            {
                "ai_rank": 2,
                "symbol": sym2,
                "reason_code": "SLOT2_SKIP_LARGE_GAP_OPENING",
                "detail": "",
            }
        )
        return [1.0], [a1], rej_out

    if elite and c2 >= 0.69:
        w1, w2 = 0.75, 0.25
        reason = "overlap_elite_strong_slot2"
    elif gap <= 0.05:
        w1, w2 = 0.75, 0.25
        reason = "tight_confidence_gap"
    elif rx:
        w1, w2 = 0.88, 0.12
        reason = "moderate_gap_slot2_weight_12_relaxed"
    else:
        w1, w2 = 0.85, 0.15
        reason = "moderate_gap_slot2_weight_15"

    if config.slot2_elite_only_opening:
        row1_elite = step2_by_symbol.get(sym1) if isinstance(step2_by_symbol, Mapping) else None
        if not isinstance(row1_elite, dict):
            row1_elite = None
        em1 = _premarket_expected_move_proxy(row1_elite)
        em_gap_str = (
            "na" if em1 is None or em2 is None else f"{float(em1) - float(em2):.6f}"
        )
        elite_fail = _slot2_elite_only_gate_failure(
            c2=c2,
            gap=gap,
            row2=row2,
            em1=em1,
            em2=em2,
            overlap=overlap,
            config=config,
        )
        if elite_fail:
            _LOG.info(
                "SLOT2_ELITE_REJECT_REASON=%s rank1_score=%.6f rank2_score=%.6f score_gap=%.6f "
                "expected_move_gap=%s final_slot2_weight=0.0000",
                elite_fail,
                c1,
                c2,
                gap,
                em_gap_str,
            )
            rej_out.append(
                {
                    "ai_rank": 2,
                    "symbol": sym2,
                    "reason_code": "SLOT2_SKIP_ELITE_ONLY_OPENING",
                    "detail": elite_fail,
                }
            )
            return [1.0], [a1], rej_out
        cap_w2 = float(config.slot2_elite_max_slot2_weight)
        if w2 > cap_w2:
            w2 = cap_w2
            w1 = 1.0 - w2
            reason = f"{reason}_elite_w2_cap"
        _LOG.info(
            "SLOT2_ELITE_ACCEPT rank1_score=%.6f rank2_score=%.6f score_gap=%.6f expected_move_gap=%s "
            "final_slot2_weight=%.4f",
            c1,
            c2,
            gap,
            em_gap_str,
            w2,
        )

    _log_weight(w2, reason)
    return [w1, w2], accepted, rej_out


def accept_opening_candidates(
    validated_decision: Mapping[str, Any],
    *,
    step2_by_symbol: Mapping[str, dict[str, Any]],
    config: OpeningAllocationConfig,
    diagnostics: OpeningAllocationDiagnostics | None = None,
    market_context: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (accepted, rejected) where each item uses ai_rank, symbol, direction, ai_confidence
    for accepted; rejected entries have ai_rank, symbol, reason_code, detail.
    """
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    ds = validated_decision.get("decision_status")
    cands = validated_decision.get("candidates")

    if ds != "trade":
        _log_opening_skip_primary("SOURCE_DECISION_NO_TRADE")
        rejected.append(
            {
                "ai_rank": 0,
                "symbol": "",
                "reason_code": "SOURCE_DECISION_NO_TRADE",
                "detail": "",
            }
        )
        if diagnostics is not None:
            diagnostics.decision_status_not_trade_count += 1
        return accepted, rejected

    if not isinstance(cands, list) or len(cands) == 0:
        _log_opening_skip_primary("NO_CANDIDATES")
        rejected.append(
            {
                "ai_rank": 0,
                "symbol": "",
                "reason_code": "NO_CANDIDATES",
                "detail": "",
            }
        )
        if diagnostics is not None:
            diagnostics.no_candidates_empty_count += 1
        return accepted, rejected

    c1 = cands[0]
    sym1 = str(c1.get("symbol", "")).strip().upper()
    row1 = step2_by_symbol.get(sym1) if isinstance(step2_by_symbol, Mapping) else None
    if not isinstance(row1, dict):
        row1 = None
    conf1 = _as_float(c1.get("confidence"))

    if diagnostics is not None and conf1 is not None:
        thr_s = float(config.c1_hard)
        soft_lo_s = max(float(config.c1_absolute_floor), thr_s - float(config.c1_soft_margin))
        if soft_lo_s <= conf1 < thr_s:
            diagnostics.soft_band_candidate_count += 1
            emz = _premarket_expected_move_proxy(row1)
            if emz is not None and emz >= float(config.c1_soft_min_expected_move):
                diagnostics.soft_band_expected_move_pass_count += 1

    ok_conf, conf_mode, em_proxy = _rank1_confidence_pass(conf1, row1, config)
    if not ok_conf:
        if conf_mode == "rej_invalid_confidence":
            rcode = "RANK1_INVALID_CONFIDENCE"
        elif conf_mode == "rej_below_confidence_floor":
            rcode = "RANK1_BELOW_CONFIDENCE_FLOOR"
        elif conf_mode == "rej_soft_band_expected_move":
            rcode = "RANK1_SOFT_BAND_EXPECTED_MOVE_FAIL"
        elif conf_mode == "rej_rank1_expected_move_absolute_fail":
            rcode = "RANK1_EXPECTED_MOVE_ABSOLUTE_FAIL"
        else:
            rcode = "RANK1_CONFIDENCE_REJECT"
        _log_opening_reject(rcode, confidence=conf1, threshold=config.c1_hard, expected_move=em_proxy)
        rejected.append({"ai_rank": 1, "symbol": sym1, "reason_code": rcode, "detail": conf_mode})
        return accepted, rejected

    dir1 = c1.get("direction")
    if not _direction_ok(dir1):
        if diagnostics is not None:
            diagnostics.confidence_pass_direction_fail_count += 1
        _log_opening_reject(
            "RANK1_INVALID_DIRECTION",
            confidence=conf1,
            threshold=config.c1_hard,
            expected_move=em_proxy,
        )
        rejected.append(
            {
                "ai_rank": 1,
                "symbol": sym1,
                "reason_code": "RANK1_INVALID_DIRECTION",
                "detail": "",
            }
        )
        return accepted, rejected

    if config.apply_midmorning_strict_gates:
        ok_mm, mm_code = _midmorning_rank1_strict(str(dir1).strip().lower(), row1, market_context, config)
        if not ok_mm:
            _log_opening_reject(
                mm_code,
                confidence=conf1,
                threshold=None,
                expected_move=em_proxy,
            )
            rejected.append({"ai_rank": 1, "symbol": sym1, "reason_code": mm_code, "detail": ""})
            return accepted, rejected

    vol1, vfail = _step2_volume_for_rank(row1, config, 1)
    if vol1 is None:
        if diagnostics is not None:
            diagnostics.confidence_direction_pass_volume_fail_count += 1
        thr = config.min_pm_volume if "PM_VOLUME" in vfail else None
        _log_opening_reject(vfail, confidence=conf1, threshold=thr, expected_move=em_proxy)
        rejected.append({"ai_rank": 1, "symbol": sym1, "reason_code": vfail, "detail": ""})
        return accepted, rejected

    if diagnostics is not None and conf_mode == "soft_pass":
        diagnostics.soft_band_accepted_count += 1

    _log_opening_accept(
        confidence=float(conf1),
        threshold=float(config.c1_hard),
        expected_move=em_proxy,
    )
    accepted.append(
        {
            "ai_rank": 1,
            "symbol": sym1,
            "direction": str(dir1),
            "ai_confidence": float(conf1),
        }
    )

    if len(cands) < 2:
        return accepted, rejected

    c2 = cands[1]
    sym2 = str(c2.get("symbol", "")).strip().upper()
    dir2 = c2.get("direction")
    if not _direction_ok(dir2):
        cf = _as_float(c2.get("confidence"))
        _log_slot2_reject(
            "SLOT2_REJECT_INVALID_DIRECTION",
            rank2_confidence=cf,
            rank2_expected_move=None,
            score_gap=(float(conf1) - float(cf)) if cf is not None else None,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "RANK2_INVALID_DIRECTION", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    conf2 = _as_float(c2.get("confidence"))
    if conf2 is None:
        _log_slot2_reject(
            "SLOT2_REJECT_INVALID_CONFIDENCE",
            rank2_confidence=None,
            rank2_expected_move=None,
            score_gap=None,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "RANK2_BELOW_C2", "detail": "invalid_confidence"})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    gap_val = float(conf1) - float(conf2)

    if conf2 < config.c2:
        _log_slot2_reject(
            "SLOT2_REJECT_BELOW_C2",
            rank2_confidence=conf2,
            rank2_expected_move=None,
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "RANK2_BELOW_C2", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    if conf2 < config.rank2_abs_min:
        _log_slot2_reject(
            "SLOT2_REJECT_BELOW_ABS_MIN",
            rank2_confidence=conf2,
            rank2_expected_move=None,
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "RANK2_BELOW_ABS_MIN", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    max_peer_gap = float(config.slot2_max_gap_vs_rank1)
    if gap_val > max_peer_gap:
        _log_slot2_reject(
            "SLOT2_REJECT_PEER_GAP_TOO_LARGE",
            rank2_confidence=conf2,
            rank2_expected_move=None,
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "SLOT2_REJECT_PEER_GAP", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected
    thresh = float(config.dominance_margin)
    passes_standard = float(conf1) >= float(conf2) + thresh
    em1_dom = _premarket_expected_move_proxy(row1)
    bypass = (
        (not passes_standard)
        and float(conf1) >= float(config.dominance_bypass_rank1_min)
        and em1_dom is not None
        and em1_dom >= float(config.dominance_bypass_em_min)
    )
    dominance_pass = passes_standard or bypass
    _LOG.info(
        "RANK1_DOMINANCE_DECISION pass=%s rank1=%.6f rank2=%.6f gap=%.6f threshold=%.6f bypass=%s",
        dominance_pass,
        float(conf1),
        float(conf2),
        gap_val,
        thresh,
        bypass,
    )
    if not dominance_pass:
        _log_slot2_reject(
            "SLOT2_REJECT_RANK1_DOMINANCE",
            rank2_confidence=conf2,
            rank2_expected_move=None,
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "RANK1_DOMINANCE_FAIL", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    row2 = step2_by_symbol.get(sym2) if isinstance(step2_by_symbol, Mapping) else None
    if not isinstance(row2, dict):
        row2 = None
    vol2, vfail2 = _step2_volume_for_rank(row2, config, 2)
    if vol2 is None:
        _log_slot2_reject(
            f"SLOT2_REJECT_VOLUME::{vfail2}",
            rank2_confidence=conf2,
            rank2_expected_move=_premarket_expected_move_proxy(row2),
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": vfail2, "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    em2 = _premarket_expected_move_proxy(row2)
    min_s2_em = float(config.slot2_min_expected_move)
    if em2 is None or em2 < min_s2_em:
        _log_slot2_reject(
            "SLOT2_REJECT_EXPECTED_MOVE_WEAK",
            rank2_confidence=conf2,
            rank2_expected_move=em2,
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "SLOT2_REJECT_EXPECTED_MOVE_WEAK", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    overlap = symbols_same_sector_theme(sym1, sym2)
    sector_elite = (
        overlap
        and gap_val <= float(config.slot2_same_sector_max_gap)
        and float(conf2) >= float(config.slot2_same_sector_min_confidence)
    )
    if overlap and not sector_elite:
        _log_slot2_reject(
            "SLOT2_REJECT_SAME_THEME_NOT_ELITE",
            rank2_confidence=conf2,
            rank2_expected_move=em2,
            score_gap=gap_val,
        )
        rejected.append({"ai_rank": 2, "symbol": sym2, "reason_code": "SLOT2_REJECT_SAME_THEME_NOT_ELITE", "detail": ""})
        _reject_rank3_cascade(cands, rejected)
        return accepted, rejected

    accepted.append(
        {
            "ai_rank": 2,
            "symbol": sym2,
            "direction": str(dir2),
            "ai_confidence": conf2,
        }
    )

    if len(cands) < 3:
        return accepted, rejected

    c3 = cands[2]
    sym3 = str(c3.get("symbol", "")).strip().upper()
    dir3 = c3.get("direction")
    if not _direction_ok(dir3):
        rejected.append({"ai_rank": 3, "symbol": sym3, "reason_code": "RANK3_INVALID_DIRECTION", "detail": ""})
        return accepted, rejected

    conf3 = _as_float(c3.get("confidence"))
    if conf3 is None:
        rejected.append({"ai_rank": 3, "symbol": sym3, "reason_code": "RANK3_BELOW_C3", "detail": "invalid_confidence"})
        return accepted, rejected

    if conf3 < config.c3:
        rejected.append({"ai_rank": 3, "symbol": sym3, "reason_code": "RANK3_BELOW_C3", "detail": ""})
        return accepted, rejected

    if conf3 < conf2 - config.d23:
        rejected.append({"ai_rank": 3, "symbol": sym3, "reason_code": "RANK3_TOO_FAR_BELOW_RANK2", "detail": ""})
        return accepted, rejected

    if conf3 < conf2 * config.r32:
        rejected.append({"ai_rank": 3, "symbol": sym3, "reason_code": "RANK3_RATIO_FAIL", "detail": ""})
        return accepted, rejected

    row3 = step2_by_symbol.get(sym3) if isinstance(step2_by_symbol, Mapping) else None
    if not isinstance(row3, dict):
        row3 = None
    vol3, vfail3 = _step2_volume_for_rank(row3, config, 3)
    if vol3 is None:
        rejected.append({"ai_rank": 3, "symbol": sym3, "reason_code": vfail3, "detail": ""})
        return accepted, rejected

    accepted.append(
        {
            "ai_rank": 3,
            "symbol": sym3,
            "direction": str(dir3),
            "ai_confidence": conf3,
        }
    )

    return accepted, rejected


def _reject_rank3_cascade(cands: list[Any], rejected: list[dict[str, Any]]) -> None:
    if len(cands) >= 3:
        c3 = cands[2]
        sym3 = str(c3.get("symbol", "")).strip().upper()
        rejected.append(
            {
                "ai_rank": 3,
                "symbol": sym3,
                "reason_code": "RANK3_NOT_EVALUATED",
                "detail": "rank_2_not_accepted",
            }
        )


def prepare_opening_execution(
    validated_decision: Mapping[str, Any],
    *,
    step2_packet: Mapping[str, Any] | None = None,
    config: OpeningAllocationConfig | None = None,
    allocation_diagnostics: bool = False,
) -> dict[str, Any]:
    cfg = config or OpeningAllocationConfig()
    step2_by = build_step2_index(step2_packet)
    mc: Mapping[str, Any] | None = None
    if isinstance(step2_packet, Mapping):
        raw_mc = step2_packet.get("market_context")
        if isinstance(raw_mc, Mapping):
            mc = raw_mc
    diag = OpeningAllocationDiagnostics() if allocation_diagnostics else None
    accepted, rejected = accept_opening_candidates(
        validated_decision,
        step2_by_symbol=step2_by,
        config=cfg,
        diagnostics=diag,
        market_context=mc,
    )
    if len(accepted) == 2:
        weights, accepted, rejected = opening_two_leg_capital_weights(
            accepted, rejected, config=cfg, step2_by_symbol=step2_by
        )
    else:
        weights = weight_accepted_trades(accepted)
    trade_date = str(validated_decision.get("trade_date", ""))
    source_ds = str(validated_decision.get("decision_status", ""))

    trades: list[dict[str, Any]] = []
    for i, a in enumerate(accepted):
        rnk = int(a["ai_rank"])
        trades.append(
            {
                "ai_rank": rnk,
                "symbol": str(a["symbol"]),
                "direction": str(a["direction"]),
                "ai_confidence": float(a["ai_confidence"]),
                "capital_weight": float(weights[i]),
                "included": True,
                "notes": f"ACCEPT_R{rnk}",
            }
        )

    n = len(trades)
    preparation_status = "no_trades" if n == 0 else "ready"

    out: dict[str, Any] = {
        "trade_date": trade_date,
        "preparation_status": preparation_status,
        "source_decision_status": source_ds,
        "accepted_count": n,
        "trades": trades,
        "weights": list(weights),
        "rejected": rejected,
    }
    if diag is not None:
        diag.finalize_from_rejected(rejected)
        diag.slot1_full_sleeve_only = n == 1 and preparation_status == "ready"
        out["allocation_diagnostics"] = diag.to_json_dict()
    return out
