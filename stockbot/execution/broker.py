"""Minimal Alpaca REST client: account + market order. Extend for limits/stops."""

from __future__ import annotations

import logging
from typing import Any

import requests

from stockbot.config import Settings
from stockbot.models import ExecutionResult, OrderIntent
from stockbot.risk.engine import AccountSummary

_LOG = logging.getLogger("stockbot.execution.broker")

# Deterministic sizing for dry-run / backtests (no Alpaca account API).
_SIM_EQUITY = 100_000.0

# One INFO line per process when we skip account/positions HTTP in dry-run.
_dry_run_sim_account_logged: bool = False


class AlpacaBroker:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._base = settings.alpaca_base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self._settings.alpaca_secret_key,
            "Content-Type": "application/json",
        }

    def get_account(self) -> AccountSummary:
        global _dry_run_sim_account_logged
        # Dry-run (including multi-day backtests): never call /v2/account — avoids timeouts and
        # keeps PnL simulation independent of paper/live connectivity.
        if self._settings.dry_run:
            if not _dry_run_sim_account_logged:
                _LOG.info(
                    "[broker] Simulated account for dry-run: skipping Alpaca /v2/account (and "
                    "/v2/positions when listing open symbols). Using stable equity=cash=buying_power=%.0f.",
                    _SIM_EQUITY,
                )
                _dry_run_sim_account_logged = True
            return AccountSummary(equity=_SIM_EQUITY, cash=_SIM_EQUITY, buying_power=_SIM_EQUITY)
        if not self._settings.alpaca_api_key:
            return AccountSummary(equity=_SIM_EQUITY, cash=_SIM_EQUITY, buying_power=_SIM_EQUITY)
        r = requests.get(f"{self._base}/v2/account", headers=self._headers(), timeout=30)
        r.raise_for_status()
        d = r.json()
        return AccountSummary(
            equity=float(d.get("equity", 0)),
            cash=float(d.get("cash", 0)),
            buying_power=float(d.get("buying_power", 0)),
        )

    def list_open_position_symbols(self) -> set[str]:
        if self._settings.dry_run:
            return set()
        if not self._settings.alpaca_api_key:
            return set()
        r = requests.get(f"{self._base}/v2/positions", headers=self._headers(), timeout=30)
        r.raise_for_status()
        rows = r.json()
        return {str(x.get("symbol", "")).upper() for x in rows}

    def submit_order(self, intent: OrderIntent) -> ExecutionResult:
        if self._settings.dry_run:
            return ExecutionResult(
                success=True,
                broker_order_id="DRY_RUN",
                raw_response={"intent": intent.__dict__},
                error=None,
            )
        body: dict[str, Any] = {
            "symbol": intent.symbol,
            "qty": intent.qty,
            "side": intent.side,
            "type": intent.order_type,
            "time_in_force": intent.time_in_force,
        }
        if intent.order_type == "limit" and intent.limit_price is not None:
            body["limit_price"] = str(intent.limit_price)
        r = requests.post(
            f"{self._base}/v2/orders",
            headers=self._headers(),
            json=body,
            timeout=30,
        )
        if r.status_code >= 400:
            return ExecutionResult(
                success=False,
                broker_order_id=None,
                raw_response=None,
                error=f"{r.status_code}: {r.text}",
            )
        d = r.json()
        return ExecutionResult(
            success=True,
            broker_order_id=str(d.get("id", "")),
            raw_response=d,
            error=None,
        )
