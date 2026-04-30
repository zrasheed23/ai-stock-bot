"""Tests for Alpaca paper positions → managed ledger sync (no sells)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime as real_datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import stockbot.execution.exit_engine as exit_engine_mod
from stockbot.execution.exit_engine import run_exit_scan
from stockbot.runners.managed_position_ledger import (
    STOP_LOSS_PCT,
    SqliteManagedPositionLedger,
    ledger_has_active_row_for_symbol_date,
    load_open_positions_for_date,
    sync_alpaca_positions_to_managed_ledger,
    take_profit_pct_for,
)
from tests.test_exit_engine import _MockExitBroker


class PaperLedgerSyncTests(unittest.TestCase):
    def test_sync_creates_row_for_missing_position(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "m.db")
            out = sync_alpaca_positions_to_managed_ledger(
                ledger,
                trade_date="2026-04-24",
                positions=[{"symbol": "amd", "qty": "3", "avg_entry_price": "50"}],
            )
            self.assertEqual(out["synced"], ["AMD"])
            self.assertEqual(out["skipped"], [])
            rows = load_open_positions_for_date(ledger, "2026-04-24")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "AMD")
            self.assertEqual(rows[0]["buy_client_order_id"], "SYNC-2026-04-24-AMD")

    def test_second_sync_skips_same_symbol_no_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "m.db")
            pos = [{"symbol": "AMD", "qty": "1", "avg_entry_price": "10"}]
            self.assertEqual(
                sync_alpaca_positions_to_managed_ledger(
                    ledger, trade_date="2026-04-24", positions=pos
                )["synced"],
                ["AMD"],
            )
            out2 = sync_alpaca_positions_to_managed_ledger(
                ledger, trade_date="2026-04-24", positions=pos
            )
            self.assertEqual(out2["synced"], [])
            self.assertEqual(
                out2["skipped"], [{"symbol": "AMD", "reason": "active_ledger_exists"}]
            )

    def test_sync_skips_when_active_ledger_row_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "m.db")
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
            out = sync_alpaca_positions_to_managed_ledger(
                ledger,
                trade_date="2026-04-24",
                positions=[{"symbol": "AMD", "qty": "2", "avg_entry_price": "40"}],
            )
            self.assertEqual(out["synced"], [])
            self.assertEqual(
                out["skipped"],
                [{"symbol": "AMD", "reason": "active_ledger_exists"}],
            )

    def test_sync_ignores_non_positive_qty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "m.db")
            out = sync_alpaca_positions_to_managed_ledger(
                ledger,
                trade_date="2026-04-24",
                positions=[
                    {"symbol": "ZERO", "qty": "0", "avg_entry_price": "10"},
                    {"symbol": "NEG", "qty": "-1", "avg_entry_price": "10"},
                ],
            )
            self.assertEqual(out["synced"], [])
            reasons = {s["reason"] for s in out["skipped"]}
            self.assertEqual(reasons, {"non_positive_qty"})

    def test_synced_row_buy_rank_99_and_null_entry_ts_and_tp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            sync_alpaca_positions_to_managed_ledger(
                ledger,
                trade_date="2026-04-24",
                positions=[{"symbol": "NVDA", "qty": "1", "avg_entry_price": "900"}],
            )
            with sqlite3.connect(path) as con:
                row = con.execute(
                    "SELECT buy_rank, entry_timestamp_utc, take_profit_pct, stop_loss_pct, strong_stock "
                    "FROM managed_positions WHERE symbol = ?",
                    ("NVDA",),
                ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[0], 99)
            self.assertIsNone(row[1])
            self.assertEqual(row[2], take_profit_pct_for(strong=False))
            self.assertEqual(row[3], -0.008)
            self.assertEqual(row[4], 0)

    def test_synced_row_not_stop_tp_but_eod_flatten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "m.db")
            sync_alpaca_positions_to_managed_ledger(
                ledger,
                trade_date="2026-04-24",
                positions=[{"symbol": "AMD", "qty": "2", "avg_entry_price": "100"}],
            )
            mock = _MockExitBroker()
            mock.positions["AMD"] = {"qty": "2", "avg_entry_price": "100"}
            mock.prices["AMD"] = 50.0

            not_eod = real_datetime(2026, 4, 24, 14, 0, 0, tzinfo=ZoneInfo("UTC"))

            class _DtNotEod:
                combine = real_datetime.combine
                fromisoformat = real_datetime.fromisoformat

                @staticmethod
                def now(tz=None):
                    if tz == exit_engine_mod._UTC:
                        return not_eod
                    return real_datetime.now(tz)

            with patch.object(exit_engine_mod, "datetime", _DtNotEod):
                out = run_exit_scan(
                    trade_date="2026-04-24",
                    ledger=ledger,
                    client=mock,
                    eod_flatten_enabled=False,
                    minimum_hold_seconds=0,
                )
            self.assertEqual(mock.sells, [])
            pos = out["positions"][0]
            self.assertEqual(pos["exit_status"], "open")
            self.assertFalse(pos["hold_eligible"])

            eod_utc = real_datetime(2026, 4, 24, 19, 55, 0, tzinfo=ZoneInfo("UTC"))

            class _DtEod:
                combine = real_datetime.combine
                fromisoformat = real_datetime.fromisoformat

                @staticmethod
                def now(tz=None):
                    if tz == exit_engine_mod._UTC:
                        return eod_utc
                    return real_datetime.now(tz)

            with patch.object(exit_engine_mod, "datetime", _DtEod):
                out2 = run_exit_scan(
                    trade_date="2026-04-24",
                    ledger=ledger,
                    client=mock,
                    eod_flatten_enabled=True,
                    minimum_hold_seconds=0,
                )
            self.assertEqual(len(mock.sells), 1)
            self.assertEqual(mock.sells[0][0], "AMD")
            self.assertEqual(out2["positions"][0]["exit_reason"], "EOD_FLATTEN")

    def test_ledger_has_active_includes_exit_submitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ledger = SqliteManagedPositionLedger(Path(td) / "m.db")
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.8,
                step2_row_snapshot={},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="b1",
                status="exit_submitted",
            )
            self.assertTrue(
                ledger_has_active_row_for_symbol_date(ledger, "2026-04-24", "AMD")
            )
