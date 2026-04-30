"""
Persistent idempotency for opening-bell order client_order_id (runner path only).

Implements the same interface Step 6 expects: get_existing_order_id, record_submission.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteOpeningIdempotencyStore:
    """SQLite-backed store; survives process restarts."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS opening_client_orders (
                    client_order_id TEXT PRIMARY KEY,
                    alpaca_order_id TEXT NOT NULL
                )
                """
            )
            con.commit()

    def get_existing_order_id(self, client_order_id: str) -> str | None:
        with sqlite3.connect(self._path) as con:
            row = con.execute(
                "SELECT alpaca_order_id FROM opening_client_orders WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def record_submission(self, client_order_id: str, alpaca_order_id: str) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO opening_client_orders (client_order_id, alpaca_order_id)
                VALUES (?, ?)
                """,
                (client_order_id, alpaca_order_id),
            )
            con.commit()
