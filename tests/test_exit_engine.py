"""Tests for Step 7 exit engine (mocked broker)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from stockbot.execution.exit_engine import (
    _at_or_after_eod_flatten,
    _current_return_pct,
    _hold_eligible,
    reconcile_row_from_alpaca,
    run_exit_scan,
)
from stockbot.runners.managed_position_ledger import load_open_positions_for_date
from stockbot.runners.managed_position_ledger import (
    STOP_LOSS_PCT,
    SqliteManagedPositionLedger,
    build_exit_client_order_id,
    ledger_row_exit_pending,
    record_exit_submission,
    take_profit_pct_for,
    update_entry_fill_data,
)


class _MockExitBroker:
    def __init__(self) -> None:
        self.positions: dict[str, dict] = {}
        self.orders_by_id: dict[str, dict] = {}
        self.orders_by_cid: dict[str, dict] = {}
        self.prices: dict[str, float] = {}
        self.sells: list[tuple[str, float, str]] = []

    def get_order_by_id(self, order_id: str) -> dict | None:
        return self.orders_by_id.get(order_id)

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        return self.orders_by_cid.get(client_order_id)

    def get_position(self, symbol: str) -> dict | None:
        return self.positions.get(symbol.upper())

    def get_latest_trade_price(self, symbol: str) -> tuple[float | None, str | None]:
        p = self.prices.get(symbol.upper())
        if p is None:
            return None, "no_price"
        return p, None

    def submit_market_sell_day(
        self, *, symbol: str, qty: float, client_order_id: str
    ) -> tuple[str | None, str | None]:
        self.sells.append((symbol.upper(), qty, client_order_id))
        return "exit-order-1", None


class ExitEngineTests(unittest.TestCase):
    def test_hold_eligible_requires_aware_ts_and_delay(self) -> None:
        utc = ZoneInfo("UTC")
        entry = datetime(2026, 4, 24, 14, 0, 0, tzinfo=utc)
        now = datetime(2026, 4, 24, 14, 1, 0, tzinfo=utc)
        self.assertFalse(_hold_eligible(entry.isoformat(), now, 120))
        now2 = datetime(2026, 4, 24, 14, 2, 1, tzinfo=utc)
        self.assertTrue(_hold_eligible(entry.isoformat(), now2, 120))
        self.assertFalse(_hold_eligible("2026-04-24T14:00:00", now2, 120))

    def test_stop_before_take_profit(self) -> None:
        self.assertLess(_current_return_pct(100.0, 99.0), -0.008)
        eff_tp = take_profit_pct_for(strong=False)
        self.assertGreater(
            _current_return_pct(100.0, 100.0 * (1.0 + eff_tp) + 0.01),
            eff_tp,
        )

    def test_eod_cutoff_et(self) -> None:
        et = ZoneInfo("America/New_York")
        self.assertFalse(_at_or_after_eod_flatten(datetime(2026, 4, 24, 15, 54, tzinfo=et)))
        self.assertTrue(_at_or_after_eod_flatten(datetime(2026, 4, 24, 15, 55, tzinfo=et)))

    def test_run_exit_scan_no_positions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "x.db")
            mock = _MockExitBroker()
            out = run_exit_scan(trade_date="2026-04-24", ledger=ledger, client=mock)
            self.assertEqual(out["exit_engine_status"], "no_positions")

    def test_reconcile_updates_entry_from_position_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "x.db")
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.8,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=True,
                take_profit_pct=take_profit_pct_for(strong=True),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="buy-1",
                status="open",
            )
            mock = _MockExitBroker()
            mock.positions["AMD"] = {"qty": "2.5", "avg_entry_price": "100.0"}
            mock.orders_by_id["buy-1"] = {
                "filled_avg_price": "99.5",
                "filled_qty": "2.5",
                "filled_at": "2026-04-24T14:00:00+00:00",
            }
            row = {
                "symbol": "AMD",
                "buy_client_order_id": "OPEN-2026-04-24-AMD-1",
                "buy_alpaca_order_id": "buy-1",
            }
            reconcile_row_from_alpaca(ledger, row, mock)
            rows = load_open_positions_for_date(ledger, "2026-04-24")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["entry_price"], 99.5)
            self.assertEqual(rows[0]["filled_qty"], 2.5)
            self.assertEqual(rows[0]["entry_timestamp_utc"], "2026-04-24T14:00:00+00:00")

    def test_stop_loss_triggers_sell(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "x.db")
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.8,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="buy-1",
                status="open",
            )
            utc = ZoneInfo("UTC")
            entry = (datetime.now(utc) - timedelta(minutes=10)).isoformat()
            update_entry_fill_data(
                ledger,
                "AMD",
                "OPEN-2026-04-24-AMD-1",
                100.0,
                1.0,
                entry,
            )
            mock = _MockExitBroker()
            mock.positions["AMD"] = {"qty": "1", "avg_entry_price": "100"}
            mock.orders_by_id["buy-1"] = {
                "filled_avg_price": "100",
                "filled_qty": "1",
                "filled_at": entry,
            }
            mock.prices["AMD"] = 99.0
            out = run_exit_scan(
                trade_date="2026-04-24",
                ledger=ledger,
                client=mock,
                minimum_hold_seconds=0,
                eod_flatten_enabled=False,
            )
            self.assertEqual(out["exit_engine_status"], "running")
            self.assertEqual(len(mock.sells), 1)
            self.assertEqual(mock.sells[0][0], "AMD")
            self.assertEqual(mock.sells[0][2], build_exit_client_order_id("2026-04-24", "AMD", 1))
            pos = out["positions"][0]
            self.assertEqual(pos["exit_reason"], "STOP_LOSS_HIT")
            self.assertEqual(pos["exit_status"], "exit_submitted")

    def test_idempotent_skip_second_submit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "x.db")
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.8,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="buy-1",
                status="open",
            )
            record_exit_submission(
                ledger,
                symbol="AMD",
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                exit_client_order_id=build_exit_client_order_id("2026-04-24", "AMD", 1),
                exit_alpaca_order_id="already",
            )
            mock = _MockExitBroker()
            ec, ea = ledger_row_exit_pending(
                ledger, symbol="AMD", buy_client_order_id="OPEN-2026-04-24-AMD-1"
            )
            self.assertEqual(ea, "already")
            out = run_exit_scan(trade_date="2026-04-24", ledger=ledger, client=mock)
            self.assertEqual(len(mock.sells), 0)
            self.assertEqual(out["exit_engine_status"], "complete")
            self.assertEqual(out["positions"], [])


if __name__ == "__main__":
    unittest.main()
