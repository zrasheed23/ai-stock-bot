"""
Step 7 — automated exit engine (polling). Manages only managed-position ledger rows.

No buys, no AI, no changes to Steps 1–6.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, time as dt_time
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from stockbot.runners.managed_position_ledger import (
    SqliteManagedPositionLedger,
    TAKE_PROFIT_MULTIPLIER,
    build_exit_client_order_id,
    count_managed_positions_for_date,
    ledger_row_exit_pending,
    load_open_positions_for_date,
    mark_position_closed_flat,
    record_exit_submission,
    update_entry_fill_data,
)

_LOG = logging.getLogger("stockbot.execution.exit_engine")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

SCAN_INTERVAL_SECONDS_DEFAULT = 9
MINIMUM_HOLD_SECONDS_DEFAULT = 120
STOP_LOSS_PCT_DEFAULT = -0.008


@runtime_checkable
class ExitBrokerPort(Protocol):
    def get_order_by_id(self, order_id: str) -> dict[str, Any] | None: ...
    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None: ...
    def get_position(self, symbol: str) -> dict[str, Any] | None: ...
    def get_latest_trade_price(self, symbol: str) -> tuple[float | None, str | None]: ...
    def submit_market_sell_day(
        self, *, symbol: str, qty: float, client_order_id: str
    ) -> tuple[str | None, str | None]: ...


def _float_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _parse_ts_for_validation(raw: Any) -> str | None:
    from stockbot.runners.managed_position_ledger import _normalize_and_validate_entry_timestamp_utc

    if raw is None:
        return None
    return _normalize_and_validate_entry_timestamp_utc(str(raw))


def _hold_eligible(entry_ts_str: str | None, now_utc: datetime, min_hold: int) -> bool:
    if not entry_ts_str:
        return False
    s = str(entry_ts_str).strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return False
    if dt.tzinfo is None:
        return False
    entry_u = dt.astimezone(_UTC)
    now_u = now_utc.astimezone(_UTC) if now_utc.tzinfo else now_utc.replace(tzinfo=_UTC).astimezone(_UTC)
    return (now_u - entry_u).total_seconds() >= float(min_hold)


def _at_or_after_eod_flatten(now_et: datetime) -> bool:
    cutoff = datetime.combine(now_et.date(), dt_time(15, 55), tzinfo=_ET)
    return now_et >= cutoff


def _current_return_pct(entry: float, price: float) -> float:
    if not math.isfinite(entry) or entry <= 0.0 or not math.isfinite(price) or price <= 0.0:
        return float("nan")
    return (price / entry) - 1.0


def _output_quantity(broker_qty: float | None, ledger_filled: Any) -> float | None:
    if broker_qty is not None:
        return float(broker_qty)
    if ledger_filled is None:
        return None
    try:
        v = float(ledger_filled)
    except (TypeError, ValueError):
        return None
    return v


def _position_result(
    *,
    symbol: str,
    buy_rank: int,
    quantity: float | None,
    entry_price: float | None,
    current_price: float | None,
    current_return_pct: float | None,
    strong_stock: bool,
    take_profit_pct: float,
    stop_loss_pct: float,
    hold_eligible: bool,
    exit_status: str,
    exit_reason: str | None,
    exit_client_order_id: str,
    alpaca_exit_order_id: str | None,
    error: str | None,
) -> dict[str, Any]:
    """Single schema for scan output (one ``quantity`` field)."""
    return {
        "symbol": symbol,
        "buy_rank": buy_rank,
        "quantity": quantity,
        "entry_price": entry_price,
        "current_price": current_price,
        "current_return_pct": current_return_pct,
        "strong_stock": strong_stock,
        "take_profit_pct": take_profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "hold_eligible": hold_eligible,
        "exit_status": exit_status,
        "exit_reason": exit_reason,
        "exit_client_order_id": exit_client_order_id,
        "alpaca_exit_order_id": alpaca_exit_order_id,
        "error": error,
    }


def reconcile_row_from_alpaca(
    ledger: SqliteManagedPositionLedger,
    row: dict[str, Any],
    client: ExitBrokerPort,
) -> None:
    """Refresh entry_price / filled_qty / entry_timestamp_utc from broker when available."""
    symbol = str(row["symbol"]).strip().upper()
    buy_cid = str(row["buy_client_order_id"]).strip()
    oid = row.get("buy_alpaca_order_id")
    row_status = str(row.get("status") or "open")

    pos = client.get_position(symbol)
    pos_qty: float | None = None
    if pos is not None:
        q = _float_or_none(pos.get("qty"))
        if q is not None:
            pos_qty = abs(q)

    if pos_qty is not None and pos_qty <= 1e-12:
        mark_position_closed_flat(ledger, symbol=symbol, buy_client_order_id=buy_cid)
        return
    if row_status == "exit_submitted" and pos is None:
        mark_position_closed_flat(ledger, symbol=symbol, buy_client_order_id=buy_cid)
        return

    entry_px: float | None = None
    entry_ts: str | None = None
    filled_qty: float | None = pos_qty

    if oid:
        ord_d = client.get_order_by_id(str(oid))
        if isinstance(ord_d, dict):
            fap = _float_or_none(ord_d.get("filled_avg_price"))
            if fap is not None and fap > 0.0:
                entry_px = fap
            entry_ts = _parse_ts_for_validation(ord_d.get("filled_at"))
            fq = _float_or_none(ord_d.get("filled_qty"))
            if fq is not None and fq > 0.0 and filled_qty is None:
                filled_qty = abs(fq)

    if pos is not None:
        pep = _float_or_none(pos.get("avg_entry_price"))
        if entry_px is None and pep is not None and pep > 0.0:
            entry_px = pep
        pq = _float_or_none(pos.get("qty"))
        if pq is not None:
            filled_qty = abs(pq)

    update_entry_fill_data(
        ledger,
        symbol,
        buy_cid,
        entry_px,
        filled_qty,
        entry_ts,
    )


def _broker_sell_qty(client: ExitBrokerPort, symbol: str) -> tuple[float | None, str | None]:
    pos = client.get_position(symbol)
    if pos is None:
        return None, "no_position"
    q = _float_or_none(pos.get("qty"))
    if q is None:
        return None, "invalid_qty"
    return abs(q), None


def run_exit_scan(
    *,
    trade_date: str,
    ledger: SqliteManagedPositionLedger,
    client: ExitBrokerPort,
    scan_interval_seconds: int = SCAN_INTERVAL_SECONDS_DEFAULT,
    minimum_hold_seconds: int = MINIMUM_HOLD_SECONDS_DEFAULT,
    eod_flatten_enabled: bool = True,
) -> dict[str, Any]:
    """
    Single poll: reconcile all open ledger rows, then evaluate exits.

    Returns one JSON-serializable result document (see spec).
    """
    td = str(trade_date).strip()
    now_utc = datetime.now(_UTC)
    now_et = now_utc.astimezone(_ET)

    rows_before = load_open_positions_for_date(ledger, td)
    if not rows_before:
        any_row = count_managed_positions_for_date(ledger, td) > 0
        return {
            "trade_date": td,
            "exit_engine_status": "complete" if any_row else "no_positions",
            "scan_interval_seconds": scan_interval_seconds,
            "minimum_hold_seconds": minimum_hold_seconds,
            "eod_flatten_enabled": eod_flatten_enabled,
            "positions": [],
        }

    for row in list(rows_before):
        reconcile_row_from_alpaca(ledger, row, client)

    open_rows = load_open_positions_for_date(ledger, td)
    if not open_rows:
        return {
            "trade_date": td,
            "exit_engine_status": "complete",
            "scan_interval_seconds": scan_interval_seconds,
            "minimum_hold_seconds": minimum_hold_seconds,
            "eod_flatten_enabled": eod_flatten_enabled,
            "positions": [],
        }

    positions_out: list[dict[str, Any]] = []
    eod_now = _at_or_after_eod_flatten(now_et)

    for row in open_rows:
        sym = str(row["symbol"]).strip().upper()
        rank = int(row["buy_rank"])
        buy_cid = str(row["buy_client_order_id"]).strip()
        exit_cid = build_exit_client_order_id(td, sym, rank)
        stop_pct = float(row.get("stop_loss_pct") or STOP_LOSS_PCT_DEFAULT)
        tp_pct = float(row["take_profit_pct"])
        strong = bool(row["strong_stock"])
        entry_px = row.get("entry_price")
        entry_ts = row.get("entry_timestamp_utc")
        if entry_px is not None:
            entry_px = float(entry_px)

        ledger_st = str(row.get("status") or "open")
        ec_pending, ea_pending = ledger_row_exit_pending(
            ledger, symbol=sym, buy_client_order_id=buy_cid
        )
        monitored_exit = ledger_st == "exit_submitted" or bool(ea_pending)

        if monitored_exit:
            qty_mon, qmon_err = _broker_sell_qty(client, sym)
            if qty_mon is not None and qty_mon <= 1e-12:
                mark_position_closed_flat(ledger, symbol=sym, buy_client_order_id=buy_cid)
                positions_out.append(
                    _position_result(
                        symbol=sym,
                        buy_rank=rank,
                        quantity=0.0,
                        entry_price=entry_px,
                        current_price=None,
                        current_return_pct=None,
                        strong_stock=strong,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=stop_pct,
                        hold_eligible=False,
                        exit_status="exited",
                        exit_reason=None,
                        exit_client_order_id=ec_pending or exit_cid,
                        alpaca_exit_order_id=ea_pending,
                        error=None,
                    )
                )
                continue
            if qty_mon is None and qmon_err == "no_position":
                mark_position_closed_flat(ledger, symbol=sym, buy_client_order_id=buy_cid)
                positions_out.append(
                    _position_result(
                        symbol=sym,
                        buy_rank=rank,
                        quantity=0.0,
                        entry_price=entry_px,
                        current_price=None,
                        current_return_pct=None,
                        strong_stock=strong,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=stop_pct,
                        hold_eligible=False,
                        exit_status="exited",
                        exit_reason=None,
                        exit_client_order_id=ec_pending or exit_cid,
                        alpaca_exit_order_id=ea_pending,
                        error=None,
                    )
                )
                continue
            qdisp = _output_quantity(qty_mon, row.get("filled_qty"))
            positions_out.append(
                _position_result(
                    symbol=sym,
                    buy_rank=rank,
                    quantity=qdisp,
                    entry_price=entry_px,
                    current_price=None,
                    current_return_pct=None,
                    strong_stock=strong,
                    take_profit_pct=tp_pct,
                    stop_loss_pct=stop_pct,
                    hold_eligible=False,
                    exit_status="exit_submitted",
                    exit_reason=None,
                    exit_client_order_id=ec_pending or exit_cid,
                    alpaca_exit_order_id=ea_pending,
                    error=None,
                )
            )
            continue

        if ledger_st == "open" and ec_pending and not ea_pending:
            recovered = client.get_order_by_client_id(ec_pending)
            rid = recovered.get("id") if isinstance(recovered, dict) else None
            if rid:
                record_exit_submission(
                    ledger,
                    symbol=sym,
                    buy_client_order_id=buy_cid,
                    exit_client_order_id=ec_pending,
                    exit_alpaca_order_id=str(rid),
                )
                positions_out.append(
                    _position_result(
                        symbol=sym,
                        buy_rank=rank,
                        quantity=_output_quantity(None, row.get("filled_qty")),
                        entry_price=entry_px,
                        current_price=None,
                        current_return_pct=None,
                        strong_stock=strong,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=stop_pct,
                        hold_eligible=False,
                        exit_status="exit_submitted",
                        exit_reason=None,
                        exit_client_order_id=ec_pending,
                        alpaca_exit_order_id=str(rid),
                        error=None,
                    )
                )
                continue
            positions_out.append(
                _position_result(
                    symbol=sym,
                    buy_rank=rank,
                    quantity=_output_quantity(None, row.get("filled_qty")),
                    entry_price=entry_px,
                    current_price=None,
                    current_return_pct=None,
                    strong_stock=strong,
                    take_profit_pct=tp_pct,
                    stop_loss_pct=stop_pct,
                    hold_eligible=False,
                    exit_status="open",
                    exit_reason=None,
                    exit_client_order_id=ec_pending,
                    alpaca_exit_order_id=None,
                    error="exit_client_order_pending_sync",
                )
            )
            continue

        qty_sell, qerr = _broker_sell_qty(client, sym)
        if qty_sell is not None and qty_sell <= 1e-12:
            mark_position_closed_flat(ledger, symbol=sym, buy_client_order_id=buy_cid)
            positions_out.append(
                _position_result(
                    symbol=sym,
                    buy_rank=rank,
                    quantity=0.0,
                    entry_price=entry_px,
                    current_price=None,
                    current_return_pct=None,
                    strong_stock=strong,
                    take_profit_pct=tp_pct,
                    stop_loss_pct=stop_pct,
                    hold_eligible=False,
                    exit_status="exited",
                    exit_reason=None,
                    exit_client_order_id=exit_cid,
                    alpaca_exit_order_id=None,
                    error=None,
                )
            )
            continue

        exit_reason: str | None = None
        want_submit = False
        err_msg: str | None = None
        cur_px: float | None = None
        ret_pct: float | None = None
        hold_ok = _hold_eligible(
            str(entry_ts) if entry_ts else None, now_utc, minimum_hold_seconds
        )

        if eod_now and eod_flatten_enabled:
            if qty_sell is None:
                err_msg = qerr or "eod_qty_unknown"
                positions_out.append(
                    _position_result(
                        symbol=sym,
                        buy_rank=rank,
                        quantity=_output_quantity(None, row.get("filled_qty")),
                        entry_price=entry_px,
                        current_price=None,
                        current_return_pct=None,
                        strong_stock=strong,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=stop_pct,
                        hold_eligible=hold_ok,
                        exit_status="not_manageable",
                        exit_reason=None,
                        exit_client_order_id=exit_cid,
                        alpaca_exit_order_id=None,
                        error=err_msg,
                    )
                )
                continue
            exit_reason = "EOD_FLATTEN"
            want_submit = True
        elif hold_ok and entry_px is not None and float(entry_px) > 0.0:
            _LOG.info(
                "EXIT_PARAMS symbol=%s stop_loss_pct=%s take_profit_pct=%s tp_multiplier=%s",
                sym,
                stop_pct,
                tp_pct,
                TAKE_PROFIT_MULTIPLIER,
            )
            cur_px, perr = client.get_latest_trade_price(sym)
            if cur_px is None:
                err_msg = perr or "price_unavailable"
            else:
                ret = _current_return_pct(float(entry_px), cur_px)
                if not math.isfinite(ret):
                    err_msg = "return_nan"
                else:
                    ret_pct = ret
                    if ret <= stop_pct:
                        exit_reason = "STOP_LOSS_HIT"
                        want_submit = True
                    elif ret >= tp_pct:
                        exit_reason = "TAKE_PROFIT_HIT"
                        want_submit = True

        if want_submit and qty_sell is None:
            err_msg = err_msg or (qerr or "qty_unknown_for_exit")

        if want_submit and qty_sell is not None and qty_sell > 0.0:
            oid, serr = client.submit_market_sell_day(
                symbol=sym, qty=qty_sell, client_order_id=exit_cid
            )
            if oid is not None:
                record_exit_submission(
                    ledger,
                    symbol=sym,
                    buy_client_order_id=buy_cid,
                    exit_client_order_id=exit_cid,
                    exit_alpaca_order_id=oid,
                )
                positions_out.append(
                    _position_result(
                        symbol=sym,
                        buy_rank=rank,
                        quantity=qty_sell,
                        entry_price=entry_px,
                        current_price=cur_px,
                        current_return_pct=ret_pct,
                        strong_stock=strong,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=stop_pct,
                        hold_eligible=hold_ok,
                        exit_status="exit_submitted",
                        exit_reason=exit_reason,
                        exit_client_order_id=exit_cid,
                        alpaca_exit_order_id=oid,
                        error=None,
                    )
                )
            else:
                positions_out.append(
                    _position_result(
                        symbol=sym,
                        buy_rank=rank,
                        quantity=qty_sell,
                        entry_price=entry_px,
                        current_price=cur_px,
                        current_return_pct=ret_pct,
                        strong_stock=strong,
                        take_profit_pct=tp_pct,
                        stop_loss_pct=stop_pct,
                        hold_eligible=hold_ok,
                        exit_status="open",
                        exit_reason=exit_reason,
                        exit_client_order_id=exit_cid,
                        alpaca_exit_order_id=None,
                        error=serr or "submit_failed",
                    )
                )
        else:
            positions_out.append(
                _position_result(
                    symbol=sym,
                    buy_rank=rank,
                    quantity=_output_quantity(qty_sell, row.get("filled_qty")),
                    entry_price=entry_px,
                    current_price=cur_px,
                    current_return_pct=ret_pct,
                    strong_stock=strong,
                    take_profit_pct=tp_pct,
                    stop_loss_pct=stop_pct,
                    hold_eligible=hold_ok,
                    exit_status="open",
                    exit_reason=None,
                    exit_client_order_id=exit_cid,
                    alpaca_exit_order_id=None,
                    error=err_msg,
                )
            )

    return {
        "trade_date": td,
        "exit_engine_status": "running",
        "scan_interval_seconds": scan_interval_seconds,
        "minimum_hold_seconds": minimum_hold_seconds,
        "eod_flatten_enabled": eod_flatten_enabled,
        "positions": positions_out,
    }


def run_exit_engine_loop(
    *,
    trade_date: str,
    ledger: SqliteManagedPositionLedger,
    client: ExitBrokerPort,
    scan_interval_seconds: int = SCAN_INTERVAL_SECONDS_DEFAULT,
    minimum_hold_seconds: int = MINIMUM_HOLD_SECONDS_DEFAULT,
    eod_flatten_enabled: bool = True,
    max_scans: int | None = None,
) -> dict[str, Any]:
    """
    Poll until no active ledger rows remain (``open`` / ``exit_submitted``) or ``max_scans``
    is reached (None = unlimited).
    """
    td = str(trade_date).strip()
    last: dict[str, Any] | None = None
    scans = 0
    while True:
        scans += 1
        last = run_exit_scan(
            trade_date=td,
            ledger=ledger,
            client=client,
            scan_interval_seconds=scan_interval_seconds,
            minimum_hold_seconds=minimum_hold_seconds,
            eod_flatten_enabled=eod_flatten_enabled,
        )
        st = last.get("exit_engine_status")
        if st in ("no_positions", "complete"):
            assert last is not None
            return last
        if max_scans is not None and max_scans > 0 and scans >= max_scans:
            assert last is not None
            last = {**last, "exit_engine_status": "running", "scan_count": scans}
            return last
        _LOG.info(
            "exit engine scan %d: %d open ledger rows (reported)",
            scans,
            len(last.get("positions") or []),
        )
        time.sleep(float(scan_interval_seconds))
