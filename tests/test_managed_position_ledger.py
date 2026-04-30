"""Tests for Step 7 Stage 1 managed-position ledger."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from stockbot.runners.managed_position_ledger import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT_NORMAL,
    TAKE_PROFIT_PCT_STRONG,
    TAKE_PROFIT_MULTIPLIER,
    SqliteManagedPositionLedger,
    load_open_positions_for_date,
    record_exit_submission,
    record_submitted_opening_buys,
    strong_stock_deterministic,
    take_profit_pct_for,
    update_entry_fill_data,
)


class ManagedPositionLedgerTests(unittest.TestCase):
    def test_strong_stock_all_conditions(self) -> None:
        row = {
            "symbol": "AMD",
            "status": "ok",
            "pm_session_return_pct": 0.021,
            "pm_volume": 1_500_000,
            "pm_close_position_in_range": 0.55,
        }
        self.assertTrue(strong_stock_deterministic(ai_confidence=0.72, step2_row=row))
        self.assertEqual(take_profit_pct_for(strong=True), TAKE_PROFIT_PCT_STRONG * TAKE_PROFIT_MULTIPLIER)

    def test_strong_stock_fails_confidence(self) -> None:
        row = {
            "status": "ok",
            "pm_session_return_pct": 0.03,
            "pm_volume": 2_000_000,
            "pm_close_position_in_range": 0.6,
        }
        self.assertFalse(strong_stock_deterministic(ai_confidence=0.71, step2_row=row))
        self.assertEqual(
            take_profit_pct_for(strong=False), TAKE_PROFIT_PCT_NORMAL * TAKE_PROFIT_MULTIPLIER
        )

    def test_strong_stock_fails_volume(self) -> None:
        row = {
            "status": "ok",
            "pm_session_return_pct": 0.03,
            "pm_volume": 999_999,
            "pm_close_position_in_range": 0.6,
        }
        self.assertFalse(strong_stock_deterministic(ai_confidence=0.8, step2_row=row))

    def test_record_submitted_opening_buys_writes_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            step2 = {
                "trade_date": "2026-04-24",
                "symbols": [
                    {
                        "symbol": "AMD",
                        "status": "ok",
                        "pm_session_return_pct": 0.025,
                        "pm_volume": 2e6,
                        "pm_close_position_in_range": 0.55,
                    }
                ],
            }
            decision = {
                "candidates": [
                    {
                        "rank": 1,
                        "symbol": "AMD",
                        "direction": "long",
                        "confidence": 0.75,
                        "reason": "test",
                    }
                ]
            }
            submission = {
                "orders": [
                    {
                        "symbol": "AMD",
                        "rank": 1,
                        "status": "submitted",
                        "client_order_id": "OPEN-2026-04-24-AMD-1",
                        "alpaca_order_id": "ord-123",
                        "error": None,
                    }
                ]
            }
            written = record_submitted_opening_buys(
                ledger,
                trade_date="2026-04-24",
                submission=submission,
                validated_decision=decision,
                step2_packet=step2,
            )
            self.assertEqual(written, ["OPEN-2026-04-24-AMD-1"])

            with sqlite3.connect(path) as con:
                cur = con.execute("SELECT * FROM managed_positions")
                row = cur.fetchone()
                self.assertIsNotNone(row)
                cols = [d[0] for d in cur.description]
                d = dict(zip(cols, row))
            self.assertEqual(d["symbol"], "AMD")
            self.assertEqual(d["buy_rank"], 1)
            self.assertEqual(d["strong_stock"], 1)
            self.assertEqual(d["take_profit_pct"], TAKE_PROFIT_PCT_STRONG * TAKE_PROFIT_MULTIPLIER)
            self.assertEqual(d["stop_loss_pct"], STOP_LOSS_PCT)
            self.assertEqual(d["status"], "open")
            self.assertIsNone(d["entry_price"])
            self.assertEqual(json.loads(d["step2_row_snapshot"])["symbol"], "AMD")

    def test_update_entry_fill_data_partial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"symbol": "AMD", "status": "ok"},
                strong_stock=True,
                take_profit_pct=take_profit_pct_for(strong=True),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            ok = update_entry_fill_data(
                ledger,
                "AMD",
                "OPEN-2026-04-24-AMD-1",
                100.5,
                None,
                None,
            )
            self.assertTrue(ok)
            with sqlite3.connect(path) as con:
                row = con.execute(
                    "SELECT entry_price, filled_qty, entry_timestamp_utc FROM managed_positions"
                ).fetchone()
            self.assertEqual(row[0], 100.5)
            self.assertIsNone(row[1])
            self.assertIsNone(row[2])

            ok2 = update_entry_fill_data(
                ledger,
                "AMD",
                "OPEN-2026-04-24-AMD-1",
                None,
                1.25,
                "2026-04-24T13:35:12+00:00",
            )
            self.assertTrue(ok2)
            with sqlite3.connect(path) as con:
                row = con.execute(
                    "SELECT entry_price, filled_qty, entry_timestamp_utc FROM managed_positions"
                ).fetchone()
            self.assertEqual(row[0], 100.5)
            self.assertEqual(row[1], 1.25)
            self.assertEqual(row[2], "2026-04-24T13:35:12+00:00")

    def test_update_entry_fill_timezone_aware_z_suffix_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            self.assertTrue(
                update_entry_fill_data(
                    ledger,
                    "AMD",
                    "OPEN-2026-04-24-AMD-1",
                    None,
                    None,
                    "2026-04-24T13:35:12Z",
                )
            )
            with sqlite3.connect(path) as con:
                ts = con.execute(
                    "SELECT entry_timestamp_utc FROM managed_positions"
                ).fetchone()[0]
            self.assertEqual(ts, "2026-04-24T13:35:12+00:00")

    def test_update_entry_fill_rejects_naive_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            self.assertFalse(
                update_entry_fill_data(
                    ledger,
                    "AMD",
                    "OPEN-2026-04-24-AMD-1",
                    None,
                    None,
                    "2026-04-24T13:35:12",
                )
            )
            with sqlite3.connect(path) as con:
                ts = con.execute(
                    "SELECT entry_timestamp_utc FROM managed_positions"
                ).fetchone()[0]
            self.assertIsNone(ts)

    def test_update_entry_fill_rejects_invalid_timestamp_string(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            self.assertFalse(
                update_entry_fill_data(
                    ledger,
                    "AMD",
                    "OPEN-2026-04-24-AMD-1",
                    None,
                    None,
                    "not-a-valid-instant",
                )
            )

    def test_update_entry_fill_naive_timestamp_does_not_block_other_columns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            self.assertTrue(
                update_entry_fill_data(
                    ledger,
                    "AMD",
                    "OPEN-2026-04-24-AMD-1",
                    50.0,
                    2.5,
                    "2026-04-24T13:35:12",
                )
            )
            with sqlite3.connect(path) as con:
                ep, fq, ts = con.execute(
                    "SELECT entry_price, filled_qty, entry_timestamp_utc FROM managed_positions"
                ).fetchone()
            self.assertEqual(ep, 50.0)
            self.assertEqual(fq, 2.5)
            self.assertIsNone(ts)

    def test_load_open_positions_filters_date_and_open(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            for cid, day, st in (
                ("OPEN-2026-04-24-AMD-1", "2026-04-24", "open"),
                ("OPEN-2026-04-24-NVDA-1", "2026-04-24", "open"),
                ("OPEN-2026-04-24-TSLA-1", "2026-04-24", "open"),
                ("OPEN-2026-04-23-AMD-1", "2026-04-23", "open"),
            ):
                parts = cid.split("-")
                sym = parts[-2]
                rk = int(parts[-1])
                ledger.upsert_after_opening_buy(
                    trade_date=day,
                    symbol=sym,
                    buy_rank=rk,
                    ai_confidence=0.8,
                    step2_row_snapshot={"symbol": sym, "status": "ok"},
                    strong_stock=True,
                    take_profit_pct=take_profit_pct_for(strong=True),
                    stop_loss_pct=STOP_LOSS_PCT,
                    buy_client_order_id=cid,
                    buy_alpaca_order_id="x",
                    status=st,
                )
            with sqlite3.connect(path) as con:
                con.execute(
                    "UPDATE managed_positions SET status = 'exit_submitted' WHERE buy_client_order_id = ?",
                    ("OPEN-2026-04-24-NVDA-1",),
                )
                con.execute(
                    "UPDATE managed_positions SET status = 'exited' WHERE buy_client_order_id = ?",
                    ("OPEN-2026-04-24-TSLA-1",),
                )
                con.commit()
            rows = load_open_positions_for_date(ledger, "2026-04-24")
            self.assertEqual(len(rows), 2)
            ids = {r["buy_client_order_id"] for r in rows}
            self.assertEqual(ids, {"OPEN-2026-04-24-AMD-1", "OPEN-2026-04-24-NVDA-1"})
            st_by = {r["buy_client_order_id"]: r["status"] for r in rows}
            self.assertEqual(st_by["OPEN-2026-04-24-AMD-1"], "open")
            self.assertEqual(st_by["OPEN-2026-04-24-NVDA-1"], "exit_submitted")

    def test_record_exit_submission_sets_exit_submitted_not_exited(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
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
                buy_alpaca_order_id="b1",
                status="open",
            )
            self.assertTrue(
                record_exit_submission(
                    ledger,
                    symbol="AMD",
                    buy_client_order_id="OPEN-2026-04-24-AMD-1",
                    exit_client_order_id="EXIT-2026-04-24-AMD-1",
                    exit_alpaca_order_id="e1",
                )
            )
            with sqlite3.connect(path) as con:
                st = con.execute(
                    "SELECT status FROM managed_positions WHERE buy_client_order_id = ?",
                    ("OPEN-2026-04-24-AMD-1",),
                ).fetchone()[0]
            self.assertEqual(st, "exit_submitted")

    def test_load_open_positions_deserializes_step2_and_strong_bool(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"symbol": "AMD", "status": "ok", "pm_volume": 1e6},
                strong_stock=True,
                take_profit_pct=take_profit_pct_for(strong=True),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            rows = load_open_positions_for_date(ledger, "2026-04-24")
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertIsInstance(r["step2_row_snapshot"], dict)
            self.assertEqual(r["step2_row_snapshot"]["symbol"], "AMD")
            self.assertIs(r["strong_stock"], True)
            json.dumps(rows)

    def test_load_open_positions_invalid_step2_json_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.75,
                step2_row_snapshot={"ok": True},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            with sqlite3.connect(path) as con:
                con.execute(
                    "UPDATE managed_positions SET step2_row_snapshot = ? WHERE buy_client_order_id = ?",
                    ("{not json", "OPEN-2026-04-24-AMD-1"),
                )
                con.commit()
            rows = load_open_positions_for_date(ledger, "2026-04-24")
            self.assertEqual(rows[0]["step2_row_snapshot"], {})

    def test_load_open_positions_strong_stock_false_bool(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            ledger.upsert_after_opening_buy(
                trade_date="2026-04-24",
                symbol="AMD",
                buy_rank=1,
                ai_confidence=0.5,
                step2_row_snapshot={"symbol": "AMD"},
                strong_stock=False,
                take_profit_pct=take_profit_pct_for(strong=False),
                stop_loss_pct=STOP_LOSS_PCT,
                buy_client_order_id="OPEN-2026-04-24-AMD-1",
                buy_alpaca_order_id="ord-1",
                status="open",
            )
            rows = load_open_positions_for_date(ledger, "2026-04-24")
            self.assertIs(rows[0]["strong_stock"], False)

    def test_failed_order_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.db"
            ledger = SqliteManagedPositionLedger(path)
            submission = {
                "orders": [
                    {
                        "symbol": "AMD",
                        "rank": 1,
                        "status": "failed",
                        "client_order_id": "OPEN-2026-04-24-AMD-1",
                        "alpaca_order_id": None,
                        "error": "oops",
                    }
                ]
            }
            written = record_submitted_opening_buys(
                ledger,
                trade_date="2026-04-24",
                submission=submission,
                validated_decision={"candidates": []},
                step2_packet={"symbols": []},
            )
            self.assertEqual(written, [])
            with sqlite3.connect(path) as con:
                n = con.execute("SELECT COUNT(*) FROM managed_positions").fetchone()[0]
            self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
