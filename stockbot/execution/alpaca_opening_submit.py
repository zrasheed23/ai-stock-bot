"""
Step 6 — thin Alpaca submission for validated opening execution plans.

No retries, no reinterpretation of instructions, no broker logic beyond POST mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import math

import requests


def opening_client_order_id(trade_date: str, symbol: str, rank: Any, *, prefix: str = "OPEN") -> str:
    """Stable client_order_id; ``prefix`` distinguishes sessions (e.g. OPEN vs OPEN-MM)."""
    sym_u = str(symbol).strip().upper()
    pfx = str(prefix).strip().upper() or "OPEN"
    return f"{pfx}-{trade_date}-{sym_u}-{rank}"


@runtime_checkable
class OpeningIdempotencyStore(Protocol):
    """Minimal persistence: remember client_order_id -> Alpaca order id after success."""

    def get_existing_order_id(self, client_order_id: str) -> str | None:
        ...

    def record_submission(self, client_order_id: str, alpaca_order_id: str) -> None:
        ...


@runtime_checkable
class OpeningAlpacaTradesClient(Protocol):
    """Submit one market DAY buy; exactly one of notional_usd or qty is non-None."""

    def submit_market_buy_day(
        self,
        *,
        symbol: str,
        client_order_id: str,
        notional_usd: float | None,
        qty: int | None,
    ) -> tuple[str | None, str | None]:
        """Returns (alpaca_order_id, error). error is None on success."""


class InMemoryIdempotencyStore:
    """In-process idempotency ledger (tests / simple runners)."""

    def __init__(self) -> None:
        self._by_cid: dict[str, str] = {}

    def get_existing_order_id(self, client_order_id: str) -> str | None:
        return self._by_cid.get(client_order_id)

    def record_submission(self, client_order_id: str, alpaca_order_id: str) -> None:
        self._by_cid[client_order_id] = alpaca_order_id


class AlpacaHttpOpeningClient:
    """
    POST /v2/orders with market + day + client_order_id + notional XOR qty.
    Caller supplies base URL and auth headers (e.g. from Settings).
    """

    def __init__(self, base_url: str, headers: dict[str, str], timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {**headers, "Content-Type": "application/json"}
        self._timeout = timeout

    def submit_market_buy_day(
        self,
        *,
        symbol: str,
        client_order_id: str,
        notional_usd: float | None,
        qty: int | None,
    ) -> tuple[str | None, str | None]:
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        }
        if notional_usd is not None and qty is None:
            body["notional"] = str(notional_usd)
        elif qty is not None and notional_usd is None:
            body["qty"] = str(qty)
        else:
            return None, "internal_error: exactly one of notional_usd or qty required"

        r = requests.post(
            f"{self._base}/v2/orders",
            headers=self._headers,
            json=body,
            timeout=self._timeout,
        )
        if r.status_code >= 400:
            return None, f"{r.status_code}: {r.text}"
        d = r.json()
        oid = d.get("id")
        if oid is None:
            return None, "missing order id in Alpaca response"
        return str(oid), None


def _trade_date_out(plan: Mapping[str, Any]) -> str:
    td = plan.get("trade_date")
    return td if isinstance(td, str) else ""


def _output_rank(rank: Any) -> int:
    if isinstance(rank, int) and not isinstance(rank, bool):
        return int(rank)
    return 0


def _submission_status(orders: list[dict[str, Any]]) -> str:
    if len(orders) == 0:
        return "none"
    if any(o.get("status") == "failed" for o in orders):
        return "partial"
    return "full"


def submit_opening_execution_plan(
    *,
    plan: Mapping[str, Any],
    alpaca_client: OpeningAlpacaTradesClient,
    idempotency_store: OpeningIdempotencyStore,
    clock: Callable[[], Any] | None = None,
    client_order_id_prefix: str = "OPEN",
) -> dict[str, Any]:
    """
    Submit a validated Step 5.5 execution plan. Does not mutate ``plan``.

    ``clock`` is reserved for tests/diagnostics; unused in submission logic.

    ``client_order_id_prefix``: opening session uses ``OPEN``; mid-morning uses ``OPEN-MM``.
    """
    _ = clock
    cid_prefix = str(client_order_id_prefix).strip().upper() or "OPEN"

    orders_out: list[dict[str, Any]] = []

    if not isinstance(plan, Mapping):
        td = ""
        return {
            "trade_date": td,
            "submission_status": "none",
            "orders": [],
            "submitted_count": 0,
            "failed_count": 0,
        }

    td_out = _trade_date_out(plan)

    if plan.get("execution_status") == "no_execution":
        return {
            "trade_date": td_out,
            "submission_status": "none",
            "orders": [],
            "submitted_count": 0,
            "failed_count": 0,
        }

    instructions = plan.get("instructions")
    if not isinstance(instructions, list) or len(instructions) == 0:
        return {
            "trade_date": td_out,
            "submission_status": "none",
            "orders": [],
            "submitted_count": 0,
            "failed_count": 0,
        }

    trade_date = td_out
    if not trade_date:
        return {
            "trade_date": "",
            "submission_status": "none",
            "orders": [],
            "submitted_count": 0,
            "failed_count": 0,
        }

    for i, inst in enumerate(instructions):
        if not isinstance(inst, Mapping):
            bad_cid = opening_client_order_id(trade_date, "INVALID", i + 1, prefix=cid_prefix)
            orders_out.append(
                {
                    "symbol": "",
                    "rank": 0,
                    "status": "failed",
                    "client_order_id": bad_cid,
                    "alpaca_order_id": None,
                    "error": "instruction_not_a_mapping",
                }
            )
            continue

        symbol = str(inst.get("symbol", "")).strip()
        rank = inst.get("rank")
        mode = inst.get("mode")
        client_order_id = opening_client_order_id(trade_date, symbol, rank, prefix=cid_prefix)

        if not symbol:
            orders_out.append(
                {
                    "symbol": "",
                    "rank": _output_rank(rank),
                    "status": "failed",
                    "client_order_id": client_order_id,
                    "alpaca_order_id": None,
                    "error": "missing_symbol",
                }
            )
            continue

        rank_out = _output_rank(rank)
        existing = idempotency_store.get_existing_order_id(client_order_id)
        if existing is not None:
            orders_out.append(
                {
                    "symbol": symbol,
                    "rank": rank_out,
                    "status": "submitted",
                    "client_order_id": client_order_id,
                    "alpaca_order_id": existing,
                    "error": None,
                }
            )
            continue

        notional: float | None = None
        qty: int | None = None
        if mode == "notional_market":
            nu = inst.get("notional_usd")
            if isinstance(nu, bool) or not isinstance(nu, (int, float)):
                orders_out.append(
                    {
                        "symbol": symbol,
                        "rank": rank_out,
                        "status": "failed",
                        "client_order_id": client_order_id,
                        "alpaca_order_id": None,
                        "error": "invalid_notional_usd",
                    }
                )
                continue
            notional = float(nu)
            if not math.isfinite(notional) or notional <= 0.0:
                orders_out.append(
                    {
                        "symbol": symbol,
                        "rank": rank_out,
                        "status": "failed",
                        "client_order_id": client_order_id,
                        "alpaca_order_id": None,
                        "error": "invalid_notional_usd",
                    }
                )
                continue
            qty = None
        elif mode == "shares_market":
            sh = inst.get("shares")
            if not isinstance(sh, int) or isinstance(sh, bool) or sh < 1:
                orders_out.append(
                    {
                        "symbol": symbol,
                        "rank": rank_out,
                        "status": "failed",
                        "client_order_id": client_order_id,
                        "alpaca_order_id": None,
                        "error": "invalid_shares",
                    }
                )
                continue
            qty = sh
            notional = None
        else:
            orders_out.append(
                {
                    "symbol": symbol,
                    "rank": rank_out,
                    "status": "failed",
                    "client_order_id": client_order_id,
                    "alpaca_order_id": None,
                    "error": "invalid_mode",
                }
            )
            continue

        oid, err = alpaca_client.submit_market_buy_day(
            symbol=symbol,
            client_order_id=client_order_id,
            notional_usd=notional,
            qty=qty,
        )
        if err is not None or oid is None:
            orders_out.append(
                {
                    "symbol": symbol,
                    "rank": rank_out,
                    "status": "failed",
                    "client_order_id": client_order_id,
                    "alpaca_order_id": None,
                    "error": err or "unknown_error",
                }
            )
            continue

        idempotency_store.record_submission(client_order_id, oid)
        orders_out.append(
            {
                "symbol": symbol,
                "rank": rank_out,
                "status": "submitted",
                "client_order_id": client_order_id,
                "alpaca_order_id": oid,
                "error": None,
            }
        )

    submitted_count = sum(1 for o in orders_out if o.get("status") == "submitted")
    failed_count = sum(1 for o in orders_out if o.get("status") == "failed")

    status = _submission_status(orders_out)

    return {
        "trade_date": trade_date,
        "submission_status": status,
        "orders": orders_out,
        "submitted_count": submitted_count,
        "failed_count": failed_count,
    }
