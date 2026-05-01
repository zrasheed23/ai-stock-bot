"""
Opening-session replay attribution (diagnostics only).

Aggregates per-trade opening legs into bucket metrics and pruning candidates.
Does not influence execution or scoring.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from stockbot.execution.opening_allocation import _premarket_expected_move_proxy
from stockbot.strategy.engine import _sector_bucket


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def env_truthy(name: str) -> bool:
    import os

    v = os.environ.get(name)
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on"}


def relaxed_opening_env_label() -> str:
    bits: list[str] = []
    if env_truthy("STOCKBOT_SLOT2_RELAX_OPENING"):
        bits.append("slot2_relax")
    if env_truthy("STOCKBOT_SLOT2_ELITE_ONLY_OPENING"):
        bits.append("slot2_elite_only")
    if env_truthy("STOCKBOT_MIDMORNING_RELAX_FILTERS"):
        bits.append("midmorning_relax")
    return "|".join(bits) if bits else "none"


def band_confidence(conf: float | None) -> str:
    if conf is None:
        return "unknown"
    c = float(conf)
    edges = [0.0, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 1.01]
    labels = [
        "[0.00,0.58)",
        "[0.58,0.62)",
        "[0.62,0.66)",
        "[0.66,0.70)",
        "[0.70,0.74)",
        "[0.74,0.78)",
        "[0.78,1.01]",
    ]
    for i in range(len(edges) - 1):
        if edges[i] <= c < edges[i + 1]:
            return labels[i]
    return "[0.78,1.01]" if c >= 0.78 else "unknown"


def band_expected_move(em: float | None) -> str:
    if em is None:
        return "unknown"
    x = abs(float(em))
    if x < 0.010:
        return "em_[0,1.0pct)"
    if x < 0.015:
        return "em_[1.0,1.5pct)"
    if x < 0.020:
        return "em_[1.5,2.0pct)"
    if x < 0.030:
        return "em_[2.0,3.0pct)"
    return "em_>=3.0pct"


def band_pm_return(pm: float | None) -> str:
    if pm is None:
        return "unknown"
    x = float(pm) * 100.0
    if x < -2.0:
        return "pm_ret_<-2pct"
    if x < -1.0:
        return "pm_ret_[-2,-1)pct"
    if x < 0.0:
        return "pm_ret_[-1,0)pct"
    if x < 1.0:
        return "pm_ret_[0,1)pct"
    if x < 2.0:
        return "pm_ret_[1,2)pct"
    return "pm_ret_>=2pct"


def band_pm_volume(vol: float | None) -> str:
    if vol is None:
        return "unknown"
    v = float(vol)
    if v < 50_000:
        return "pm_vol_<50k"
    if v < 100_000:
        return "pm_vol_[50k,100k)"
    if v < 300_000:
        return "pm_vol_[100k,300k)"
    if v < 800_000:
        return "pm_vol_[300k,800k)"
    if v < 2_000_000:
        return "pm_vol_[800k,2M)"
    return "pm_vol_>=2M"


def band_gap(gap: float | None) -> str:
    """Gap vs prior close (fraction); signed."""
    if gap is None:
        return "unknown"
    x = float(gap) * 100.0
    if x < -3.0:
        return "gap_<-3pct"
    if x < -2.0:
        return "gap_[-3,-2)pct"
    if x < -1.0:
        return "gap_[-2,-1)pct"
    if x < 0.0:
        return "gap_[-1,0)pct"
    if x < 1.0:
        return "gap_[0,1)pct"
    if x < 2.0:
        return "gap_[1,2)pct"
    if x < 3.0:
        return "gap_[2,3)pct"
    return "gap_>=3pct"


def band_rank12_gap(validated_decision: Mapping[str, Any] | None) -> str:
    if not isinstance(validated_decision, Mapping):
        return "n/a"
    cands = validated_decision.get("candidates")
    if not isinstance(cands, list) or len(cands) < 2:
        return "n/a_lt2_candidates"
    c1 = _as_float(cands[0].get("confidence"))
    c2 = _as_float(cands[1].get("confidence"))
    if c1 is None or c2 is None:
        return "n/a_bad_confidence"
    g = float(c1) - float(c2)
    if g < 0.020:
        return "gap12_[0,0.020)"
    if g < 0.035:
        return "gap12_[0.020,0.035)"
    if g < 0.050:
        return "gap12_[0.035,0.050)"
    return "gap12_>=0.050"


def band_market_context(step2_packet: Mapping[str, Any] | None) -> str:
    if not isinstance(step2_packet, Mapping):
        return "missing_packet"
    mc = step2_packet.get("market_context")
    if not isinstance(mc, Mapping):
        return "mc_missing"
    spy = _as_float(mc.get("spy_premarket_return_pct"))
    qqq = _as_float(mc.get("qqq_premarket_return_pct"))
    if spy is None:
        spy_b = "spy_na"
    else:
        sx = float(spy) * 100.0
        if sx < -1.0:
            spy_b = "spy_pm_ret_<-1pct"
        elif sx < -0.25:
            spy_b = "spy_pm_ret_[-1,-0.25)pct"
        elif sx < 0.25:
            spy_b = "spy_pm_ret_[-0.25,0.25)pct"
        elif sx < 1.0:
            spy_b = "spy_pm_ret_[0.25,1)pct"
        else:
            spy_b = "spy_pm_ret_>=1pct"
    if qqq is None:
        qqq_b = "qqq_na"
    else:
        qx = float(qqq) * 100.0
        if qx < -1.0:
            qqq_b = "qqq_pm_ret_<-1pct"
        elif qx < 0:
            qqq_b = "qqq_pm_ret_[-1,0)pct"
        elif qx < 1.0:
            qqq_b = "qqq_pm_ret_[0,1)pct"
        else:
            qqq_b = "qqq_pm_ret_>=1pct"
    return f"{spy_b}|{qqq_b}"


def dow_label(trade_date: date) -> str:
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return names[trade_date.weekday()]


def build_opening_attrib_record(
    *,
    trade_date: date,
    symbol: str,
    rank_i: int,
    ai_confidence: float,
    exit_reason: str,
    ret: float,
    pnl_usd: float,
    notional_usd: float,
    step2_row: Mapping[str, Any] | None,
    validated_decision: Mapping[str, Any] | None,
    step2_packet: Mapping[str, Any] | None,
    source_override_applied: bool,
    relaxed_opening_env_snapshot: str,
) -> dict[str, Any]:
    row = dict(step2_row) if isinstance(step2_row, Mapping) else {}
    em = _premarket_expected_move_proxy(row if row else None)
    pm_ret = _as_float(row.get("pm_session_return_pct"))
    pm_vol = _as_float(row.get("pm_volume"))
    gap = _as_float(row.get("gap_close_vs_prior_close_pct"))
    st = row.get("status")
    if st == "ok":
        step2_band = "clean_ok"
    elif st is None or st == "":
        step2_band = "missing_row"
    else:
        step2_band = f"not_ok:{st}"

    sector = _sector_bucket(symbol.upper()) or "UNMAPPED"

    slot_band = "rank1" if rank_i == 1 else "rank2" if rank_i == 2 else f"rank_other_{rank_i}"

    return {
        "trade_date": trade_date.isoformat(),
        "month": trade_date.strftime("%Y-%m"),
        "symbol": symbol.upper(),
        "exit_reason": exit_reason,
        "source_override_band": "source_override" if source_override_applied else "normal",
        "confidence_band": band_confidence(ai_confidence),
        "expected_move_band": band_expected_move(em),
        "pm_return_band": band_pm_return(pm_ret),
        "pm_volume_band": band_pm_volume(pm_vol),
        "gap_band": band_gap(gap),
        "step2_band": step2_band,
        "market_context_band": band_market_context(step2_packet),
        "sector_bucket": sector,
        "rank1_rank2_gap_band": band_rank12_gap(validated_decision),
        "day_of_week": dow_label(trade_date),
        "relaxed_opening_env_band": relaxed_opening_env_snapshot,
        "slot_band": slot_band,
        "ret": float(ret),
        "pnl_usd": float(pnl_usd),
        "notional_usd": float(notional_usd),
    }


DIMENSION_KEYS: tuple[tuple[str, str], ...] = (
    ("month", "month"),
    ("symbol", "symbol"),
    ("exit_reason", "exit_reason"),
    ("source_override_band", "source_override_vs_normal"),
    ("confidence_band", "confidence_band"),
    ("expected_move_band", "expected_move_band"),
    ("pm_return_band", "premarket_return_band"),
    ("pm_volume_band", "premarket_volume_band"),
    ("gap_band", "gap_vs_prior_close_band"),
    ("step2_band", "step2_clean_missing"),
    ("market_context_band", "market_context"),
    ("sector_bucket", "sector_bucket"),
    ("rank1_rank2_gap_band", "rank1_vs_rank2_score_gap"),
    ("day_of_week", "day_of_week"),
    ("relaxed_opening_env_band", "relaxed_opening_env"),
    ("slot_band", "slot1_vs_slot2"),
)


def _bucket_metrics(rows: list[dict[str, Any]], opening_total_pnl: float) -> dict[str, Any]:
    tc = len(rows)
    if tc == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "weighted_pnl_contribution": float("nan"),
            "profit_factor": None,
            "stop_loss_count": 0,
            "take_profit_count": 0,
            "eod_flatten_count": 0,
            "pnl_usd_sum": 0.0,
        }
    rets = [float(r["ret"]) for r in rows]
    pnls = [float(r["pnl_usd"]) for r in rows]
    wins = sum(1 for x in rets if x > 0)
    losses = sum(1 for x in rets if x < 0)
    win_rate = wins / tc
    avg_ret = sum(rets) / tc
    pnl_sum = sum(pnls)
    wcontrib = (pnl_sum / opening_total_pnl) if opening_total_pnl != 0 else float("nan")

    win_pnl = sum(p for p, ret in zip(pnls, rets) if ret > 0)
    loss_pnl = sum(p for p, ret in zip(pnls, rets) if ret < 0)
    pf: float | None
    if loss_pnl < 0:
        pf = win_pnl / abs(loss_pnl)
    else:
        pf = None

    reasons = [str(r["exit_reason"]) for r in rows]
    return {
        "trade_count": tc,
        "win_rate": win_rate,
        "avg_return": avg_ret,
        "weighted_pnl_contribution": wcontrib,
        "profit_factor": pf,
        "stop_loss_count": sum(1 for x in reasons if x == "STOP_LOSS_HIT"),
        "take_profit_count": sum(1 for x in reasons if x == "TAKE_PROFIT_HIT"),
        "eod_flatten_count": sum(1 for x in reasons if x == "EOD_FLATTEN"),
        "pnl_usd_sum": pnl_sum,
        "winner_count": wins,
        "loser_count": losses,
    }


def analyze_opening_records(
    records: Sequence[dict[str, Any]],
    *,
    starting_equity: float,
) -> dict[str, Any]:
    rows = list(records)
    opening_total_pnl = sum(float(r["pnl_usd"]) for r in rows)
    total_trades = len(rows)
    total_wins = sum(1 for r in rows if float(r["ret"]) > 0)

    by_dimension: dict[str, dict[str, Any]] = {}
    for field, dim_name in DIMENSION_KEYS:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            key = str(r.get(field, "unknown"))
            buckets[key].append(r)
        by_dimension[dim_name] = {
            k: _bucket_metrics(v, opening_total_pnl) for k, v in sorted(buckets.items())
        }

    # Pruning candidates: scan flat list of (dimension, bucket, metrics)
    pruning_raw: list[dict[str, Any]] = []
    meaningful_floor = max(25.0, 0.015 * max(abs(opening_total_pnl), 1.0))

    for field, dim_name in DIMENSION_KEYS:
        for bucket_key, metrics in by_dimension[dim_name].items():
            tc = int(metrics["trade_count"])
            if tc < 5:
                continue
            ar = float(metrics["avg_return"])
            wr = float(metrics["win_rate"])
            pnl_sum = float(metrics["pnl_usd_sum"])
            if ar >= 0 or wr >= 0.45:
                continue
            if pnl_sum >= 0 or abs(pnl_sum) < meaningful_floor:
                continue
            winners_lost = int(metrics.get("winner_count") or 0)
            trades_removed = tc
            stops_removed = int(metrics["stop_loss_count"])
            est_equity_delta = -pnl_sum / float(starting_equity) if starting_equity > 0 else float("nan")
            denom = total_trades - trades_removed
            new_wr = (
                (total_wins - winners_lost) / denom
                if denom > 0
                else float("nan")
            )
            wr_impact = (
                new_wr - (total_wins / total_trades if total_trades else 0.0)
                if total_trades > 0 and denom > 0
                else float("nan")
            )

            pruning_raw.append(
                {
                    "dimension": dim_name,
                    "bucket": bucket_key,
                    "trade_count": tc,
                    "win_rate": wr,
                    "avg_return": ar,
                    "pnl_usd_sum": pnl_sum,
                    "weighted_pnl_contribution": metrics["weighted_pnl_contribution"],
                    "profit_factor": metrics["profit_factor"],
                    "stop_loss_count": stops_removed,
                    "take_profit_count": int(metrics["take_profit_count"]),
                    "eod_flatten_count": int(metrics["eod_flatten_count"]),
                    "estimate_trades_removed": trades_removed,
                    "estimate_stops_removed": stops_removed,
                    "estimate_winners_lost": winners_lost,
                    "estimate_net_equity_fraction_delta_if_removed_linear": est_equity_delta,
                    "estimate_opening_win_rate_delta_if_removed_linear": wr_impact,
                    "_note": (
                        "Linear counterfactual: assumes removing bucket trades leaves other trades unchanged; "
                        "equity path dependence ignored."
                    ),
                }
            )

    pruning_sorted = sorted(pruning_raw, key=lambda x: float(x["pnl_usd_sum"]))

    return {
        "meta": {
            "opening_attrib_trade_rows": total_trades,
            "opening_attrib_pnl_usd_sum": opening_total_pnl,
            "starting_equity_for_estimates": starting_equity,
            "meaningful_negative_pnl_floor_usd": meaningful_floor,
            "relaxed_opening_env_snapshot": relaxed_opening_env_label(),
        },
        "by_dimension": by_dimension,
        "pruning_candidates": pruning_sorted,
    }


def write_opening_attribution_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def print_opening_attribution_summary(payload: Mapping[str, Any], *, max_symbol_rows: int = 45) -> None:
    """Stdout tables (abbreviated)."""
    meta = payload.get("meta") or {}
    print("")
    print("=== Opening attribution report (diagnostics only; opening legs only) ===")
    print(f"opening_attrib_trade_rows={meta.get('opening_attrib_trade_rows', 0)}")
    print(f"opening_attrib_pnl_usd_sum={meta.get('opening_attrib_pnl_usd_sum', 0):.6f}")
    print(f"relaxed_opening_env_snapshot={meta.get('relaxed_opening_env_snapshot', '')}")

    by_dim = payload.get("by_dimension") or {}
    summary_dims = (
        "month",
        "symbol",
        "exit_reason",
        "source_override_vs_normal",
        "confidence_band",
        "expected_move_band",
        "premarket_return_band",
        "premarket_volume_band",
        "gap_vs_prior_close_band",
        "step2_clean_missing",
        "market_context",
        "sector_bucket",
        "rank1_vs_rank2_score_gap",
        "day_of_week",
        "relaxed_opening_env",
        "slot1_vs_slot2",
    )
    for dim_name in summary_dims:
        block = by_dim.get(dim_name)
        if not isinstance(block, dict) or not block:
            continue
        print("")
        print(f"--- by {dim_name} (trade_count>=1) ---")
        items = list(block.items())
        lim = max_symbol_rows if dim_name == "symbol" else min(40, max(35, max_symbol_rows))
        items = sorted(items, key=lambda kv: -kv[1].get("trade_count", 0))[:lim]
        for k, m in items:
            pf = m.get("profit_factor")
            pf_s = f"{pf:.3f}" if isinstance(pf, (int, float)) and pf is not None else "na"
            wcontrib = m.get("weighted_pnl_contribution")
            wc_s = f"{float(wcontrib):.4f}" if isinstance(wcontrib, (int, float)) and math.isfinite(float(wcontrib)) else "na"
            print(
                f"  {k}: n={m.get('trade_count')} wr={m.get('win_rate', 0):.3f} "
                f"avg_ret={m.get('avg_return', 0):.6f} pnl_sum={m.get('pnl_usd_sum', 0):.2f} "
                f"w_contrib={wc_s} PF={pf_s} "
                f"TP={m.get('take_profit_count')} SL={m.get('stop_loss_count')} EOD={m.get('eod_flatten_count')}"
            )

    pruning = payload.get("pruning_candidates") or []
    print("")
    print("=== Candidate pruning report (NOT applied; buckets dragging opening PnL) ===")
    if not pruning:
        print("  (no buckets met trade_count>=5, avg_ret<0, win_rate<45%, meaningful negative pnl)")
    else:
        for p in pruning[:40]:
            print(
                f"  dim={p.get('dimension')} bucket={p.get('bucket')} n={p.get('trade_count')} "
                f"wr={p.get('win_rate', 0):.3f} avg_ret={p.get('avg_return', 0):.6f} "
                f"pnl_sum={p.get('pnl_usd_sum', 0):.2f} "
                f"est_equity_delta≈{p.get('estimate_net_equity_fraction_delta_if_removed_linear')} "
                f"est_wr_delta≈{p.get('estimate_opening_win_rate_delta_if_removed_linear')}"
            )
        if len(pruning) > 40:
            print(f"  ... {len(pruning) - 40} more rows in JSON")
    print("================================================================")
