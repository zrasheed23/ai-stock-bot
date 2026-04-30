"""
Step 7 — Alpaca REST helpers for exit reconciliation and market sells.

No buy orders. No Step 1–6 changes.
"""

from __future__ import annotations

import math
import os
from typing import Any

import requests


class AlpacaExitHttpClient:
    """Trading + market-data endpoints used by the exit engine (polling v1)."""

    def __init__(
        self,
        *,
        trading_base_url: str,
        data_base_url: str,
        headers: dict[str, str],
        data_feed: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._trade = trading_base_url.rstrip("/")
        self._data = data_base_url.rstrip("/")
        self._headers = {**headers, "Content-Type": "application/json"}
        self._timeout = timeout
        raw = (data_feed or os.environ.get("STOCKBOT_ALPACA_DATA_FEED", "iex") or "iex").strip().lower()
        self._data_feed = raw or "iex"

    def get_order_by_id(self, order_id: str) -> dict[str, Any] | None:
        oid = str(order_id).strip()
        if not oid:
            return None
        r = requests.get(
            f"{self._trade}/v2/orders/{oid}",
            headers=self._headers,
            timeout=self._timeout,
        )
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            return None
        d = r.json()
        return d if isinstance(d, dict) else None

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        cid = str(client_order_id).strip()
        if not cid:
            return None
        r = requests.get(
            f"{self._trade}/v2/orders:by_client_order_id",
            headers=self._headers,
            params={"client_order_id": cid},
            timeout=self._timeout,
        )
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            return None
        d = r.json()
        return d if isinstance(d, dict) else None

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        sym = str(symbol).strip().upper()
        if not sym:
            return None
        r = requests.get(
            f"{self._trade}/v2/positions/{sym}",
            headers=self._headers,
            timeout=self._timeout,
        )
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            return None
        d = r.json()
        return d if isinstance(d, dict) else None

    def list_positions(self) -> list[dict[str, Any]]:
        """GET /v2/positions — all open positions (paper account sync)."""
        r = requests.get(
            f"{self._trade}/v2/positions",
            headers=self._headers,
            timeout=self._timeout,
        )
        if r.status_code >= 400:
            return []
        d = r.json()
        if not isinstance(d, list):
            return []
        return [x for x in d if isinstance(x, dict)]

    def get_latest_trade_price(self, symbol: str) -> tuple[float | None, str | None]:
        """Latest sale price from Alpaca market data v2. Returns (price, error)."""
        sym = str(symbol).strip().upper()
        if not sym:
            return None, "missing_symbol"
        r = requests.get(
            f"{self._data}/v2/stocks/{sym}/trades/latest",
            headers=self._headers,
            params={"feed": self._data_feed},
            timeout=self._timeout,
        )
        if r.status_code >= 400:
            return None, f"{r.status_code}: {r.text}"
        d = r.json()
        if not isinstance(d, dict):
            return None, "invalid_trade_response"
        trade = d.get("trade")
        if not isinstance(trade, dict):
            return None, "missing_trade"
        p = trade.get("p")
        try:
            px = float(p)
        except (TypeError, ValueError):
            return None, "invalid_trade_price"
        if not math.isfinite(px) or px <= 0.0:
            return None, "invalid_trade_price"
        return px, None

    def submit_market_sell_day(
        self,
        *,
        symbol: str,
        qty: float,
        client_order_id: str,
    ) -> tuple[str | None, str | None]:
        """POST market DAY sell. ``qty`` may be fractional; sent as string."""
        sym = str(symbol).strip().upper()
        if not sym:
            return None, "missing_symbol"
        if not math.isfinite(qty) or qty <= 0.0:
            return None, "invalid_qty"
        cid = str(client_order_id).strip()
        if not cid:
            return None, "missing_client_order_id"
        body: dict[str, Any] = {
            "symbol": sym,
            "qty": str(qty),
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": cid,
        }
        r = requests.post(
            f"{self._trade}/v2/orders",
            headers=self._headers,
            json=body,
            timeout=self._timeout,
        )
        if r.status_code >= 400:
            return None, f"{r.status_code}: {r.text}"
        d = r.json()
        oid = d.get("id") if isinstance(d, dict) else None
        if oid is None:
            return None, "missing order id in Alpaca response"
        return str(oid), None
