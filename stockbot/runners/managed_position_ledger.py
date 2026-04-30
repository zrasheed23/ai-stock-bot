"""
Step 7 Stage 1 — persistent managed-position ledger (SQLite).

Written after Step 6 for each successfully submitted or idempotently replayed opening buy.
Step 7 exit monitoring will read only these rows; no selling logic here.

``entry_timestamp_utc`` is set only from Alpaca fill/order timestamps or a reliable position
timestamp supplied by the caller. If it stays null, Step 7 must not treat the row as
eligible for stop-loss / take-profit (minimum-hold and return math require a known entry time).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

STOP_LOSS_PCT = -0.008
TAKE_PROFIT_PCT_STRONG = 0.012
TAKE_PROFIT_PCT_NORMAL = 0.008
# Applied to base normal/strong take-profit targets (exit only; stop unchanged).
TAKE_PROFIT_MULTIPLIER = 1.25

_STRONG_MIN_CONFIDENCE = 0.72
_STRONG_MIN_PM_RETURN = 0.02
_STRONG_MIN_PM_VOLUME = 1_000_000
_STRONG_MIN_PM_CLOSE_IN_RANGE = 0.50


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _normalize_and_validate_entry_timestamp_utc(raw: str) -> str | None:
    """
    Accept only strings that parse to a timezone-aware, non-naive datetime instant.
    Returns normalized ISO-8601 text for storage, or None if invalid (no guessing).
    """
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return None
    return dt.isoformat()


def strong_stock_deterministic(*, ai_confidence: float, step2_row: Mapping[str, Any] | None) -> bool:
    """
    Strong if ALL hold: confidence >= 0.72, Step 2 ok, pm return >= 2%, volume >= 1M,
    close-in-range >= 0.50. Missing/null metrics fail the check.
    """
    if not _finite(ai_confidence) or float(ai_confidence) < _STRONG_MIN_CONFIDENCE:
        return False
    if step2_row is None:
        return False
    if step2_row.get("status") != "ok":
        return False
    pm_ret = step2_row.get("pm_session_return_pct")
    if pm_ret is None or not _finite(pm_ret) or float(pm_ret) < _STRONG_MIN_PM_RETURN:
        return False
    vol = step2_row.get("pm_volume")
    if vol is None or not _finite(vol) or float(vol) < float(_STRONG_MIN_PM_VOLUME):
        return False
    pos = step2_row.get("pm_close_position_in_range")
    if pos is None or not _finite(pos) or float(pos) < _STRONG_MIN_PM_CLOSE_IN_RANGE:
        return False
    return True


def take_profit_pct_for(*, strong: bool) -> float:
    base = TAKE_PROFIT_PCT_STRONG if strong else TAKE_PROFIT_PCT_NORMAL
    return float(base) * float(TAKE_PROFIT_MULTIPLIER)


def step2_row_by_symbol(step2_packet: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Upper(symbol) -> Step 2 symbol row dict."""
    out: dict[str, dict[str, Any]] = {}
    if step2_packet is None:
        return out
    syms = step2_packet.get("symbols")
    if not isinstance(syms, list):
        return out
    for row in syms:
        if not isinstance(row, Mapping):
            continue
        s = row.get("symbol")
        if not isinstance(s, str) or not s.strip():
            continue
        out[s.strip().upper()] = {k: row[k] for k in row}
    return out


def _candidates_by_rank(validated_decision: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if validated_decision is None:
        return out
    cands = validated_decision.get("candidates")
    if not isinstance(cands, list):
        return out
    for c in cands:
        if not isinstance(c, Mapping):
            continue
        r = c.get("rank")
        if type(r) is not int:
            continue
        out[int(r)] = dict(c)
    return out


def default_managed_position_ledger_path() -> Path:
    raw = os.environ.get("STOCKBOT_MANAGED_POSITION_LEDGER_DB", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path("var/state/managed_positions.sqlite3").resolve()


class SqliteManagedPositionLedger:
    """SQLite ledger: one row per bot opening buy (buy_client_order_id primary key)."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS managed_positions (
                    buy_client_order_id TEXT PRIMARY KEY,
                    trade_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    buy_rank INTEGER NOT NULL,
                    ai_confidence REAL NOT NULL,
                    step2_row_snapshot TEXT NOT NULL,
                    strong_stock INTEGER NOT NULL,
                    take_profit_pct REAL NOT NULL,
                    stop_loss_pct REAL NOT NULL,
                    buy_alpaca_order_id TEXT,
                    entry_price REAL,
                    filled_qty REAL,
                    entry_timestamp_utc TEXT,
                    exit_client_order_id TEXT,
                    exit_alpaca_order_id TEXT,
                    status TEXT NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_managed_positions_trade_date "
                "ON managed_positions (trade_date)"
            )
            con.commit()

    @property
    def path(self) -> Path:
        return self._path

    def upsert_after_opening_buy(
        self,
        *,
        trade_date: str,
        symbol: str,
        buy_rank: int,
        ai_confidence: float,
        step2_row_snapshot: Mapping[str, Any],
        strong_stock: bool,
        take_profit_pct: float,
        stop_loss_pct: float,
        buy_client_order_id: str,
        buy_alpaca_order_id: str | None,
        status: str = "open",
    ) -> None:
        snap_json = json.dumps(dict(step2_row_snapshot), default=str, sort_keys=True)
        strong_i = 1 if strong_stock else 0
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                INSERT INTO managed_positions (
                    buy_client_order_id, trade_date, symbol, buy_rank, ai_confidence,
                    step2_row_snapshot, strong_stock, take_profit_pct, stop_loss_pct,
                    buy_alpaca_order_id, entry_price, filled_qty, entry_timestamp_utc,
                    exit_client_order_id, exit_alpaca_order_id, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?)
                ON CONFLICT(buy_client_order_id) DO UPDATE SET
                    trade_date = excluded.trade_date,
                    symbol = excluded.symbol,
                    buy_rank = excluded.buy_rank,
                    ai_confidence = excluded.ai_confidence,
                    step2_row_snapshot = excluded.step2_row_snapshot,
                    strong_stock = excluded.strong_stock,
                    take_profit_pct = excluded.take_profit_pct,
                    stop_loss_pct = excluded.stop_loss_pct,
                    buy_alpaca_order_id = excluded.buy_alpaca_order_id,
                    entry_price = COALESCE(managed_positions.entry_price, excluded.entry_price),
                    filled_qty = COALESCE(managed_positions.filled_qty, excluded.filled_qty),
                    entry_timestamp_utc = COALESCE(
                        managed_positions.entry_timestamp_utc, excluded.entry_timestamp_utc
                    ),
                    exit_client_order_id = COALESCE(
                        managed_positions.exit_client_order_id, excluded.exit_client_order_id
                    ),
                    exit_alpaca_order_id = COALESCE(
                        managed_positions.exit_alpaca_order_id, excluded.exit_alpaca_order_id
                    ),
                    status = CASE
                        WHEN managed_positions.status IN ('exited', 'exit_submitted', 'not_manageable')
                        THEN managed_positions.status
                        ELSE excluded.status
                    END
                """,
                (
                    buy_client_order_id,
                    trade_date,
                    symbol.upper(),
                    int(buy_rank),
                    float(ai_confidence),
                    snap_json,
                    strong_i,
                    float(take_profit_pct),
                    float(stop_loss_pct),
                    buy_alpaca_order_id,
                    status,
                ),
            )
            con.commit()


def record_submitted_opening_buys(
    ledger: SqliteManagedPositionLedger,
    *,
    trade_date: str,
    submission: Mapping[str, Any],
    validated_decision: Mapping[str, Any],
    step2_packet: Mapping[str, Any],
) -> list[str]:
    """
    For each Step 6 order with status ``submitted`` and a known Alpaca order id, upsert one ledger row.

    Returns ``buy_client_order_id`` values written.
    """
    written: list[str] = []
    by_rank = _candidates_by_rank(validated_decision)
    s2 = step2_row_by_symbol(step2_packet)

    orders = submission.get("orders")
    if not isinstance(orders, list):
        return written

    for o in orders:
        if not isinstance(o, Mapping):
            continue
        if o.get("status") != "submitted":
            continue
        cid = o.get("client_order_id")
        if not isinstance(cid, str) or not cid.strip():
            continue
        oid = o.get("alpaca_order_id")
        if oid is None:
            continue
        sym = str(o.get("symbol", "")).strip().upper()
        if not sym:
            continue
        rank_raw = o.get("rank")
        if type(rank_raw) is not int:
            continue
        rank = int(rank_raw)
        cand = by_rank.get(rank)
        if cand is None:
            continue
        if str(cand.get("symbol", "")).strip().upper() != sym:
            continue
        conf = cand.get("confidence")
        if type(conf) not in (int, float) or isinstance(conf, bool):
            continue
        ai_confidence = float(conf)

        row2 = s2.get(sym)
        strong = strong_stock_deterministic(ai_confidence=ai_confidence, step2_row=row2)
        tp = take_profit_pct_for(strong=strong)
        snap: dict[str, Any] = dict(row2) if row2 is not None else {}

        ledger.upsert_after_opening_buy(
            trade_date=trade_date,
            symbol=sym,
            buy_rank=rank,
            ai_confidence=ai_confidence,
            step2_row_snapshot=snap,
            strong_stock=strong,
            take_profit_pct=tp,
            stop_loss_pct=STOP_LOSS_PCT,
            buy_client_order_id=cid.strip(),
            buy_alpaca_order_id=str(oid),
            status="open",
        )
        written.append(cid.strip())

    return written


def update_entry_fill_data(
    ledger: SqliteManagedPositionLedger,
    symbol: str,
    buy_client_order_id: str,
    entry_price: float | None,
    filled_qty: float | None,
    entry_timestamp_utc: str | None,
) -> bool:
    """
    Update entry fill fields on a single ledger row (for Step 7 reconciliation).

    Matches ``buy_client_order_id`` and uppercased ``symbol``. Pass ``None`` for any field
    that is still unknown; those columns are left unchanged. Do not pass approximated times.

    ``entry_timestamp_utc`` must parse as a timezone-aware datetime (non-naive). Invalid or
    naive values are not written; existing ``entry_timestamp_utc`` is left unchanged.

    Returns True if exactly one row was updated.
    """
    sym_u = str(symbol).strip().upper()
    cid = str(buy_client_order_id).strip()
    if not sym_u or not cid:
        return False

    assignments: list[str] = []
    values: list[Any] = []

    if entry_price is not None:
        if _finite(entry_price) and float(entry_price) > 0.0:
            assignments.append("entry_price = ?")
            values.append(float(entry_price))

    if filled_qty is not None:
        if _finite(filled_qty) and float(filled_qty) > 0.0:
            assignments.append("filled_qty = ?")
            values.append(float(filled_qty))

    if entry_timestamp_utc is not None:
        canonical_ts = _normalize_and_validate_entry_timestamp_utc(str(entry_timestamp_utc))
        if canonical_ts is not None:
            assignments.append("entry_timestamp_utc = ?")
            values.append(canonical_ts)

    if not assignments:
        return False

    values.extend([cid, sym_u])
    set_clause = ", ".join(assignments)
    sql = f"UPDATE managed_positions SET {set_clause} WHERE buy_client_order_id = ? AND symbol = ?"

    with sqlite3.connect(ledger.path) as con:
        cur = con.execute(sql, values)
        con.commit()
        return bool(cur.rowcount == 1)


def count_managed_positions_for_date(ledger: SqliteManagedPositionLedger, trade_date: str) -> int:
    """Rows in the ledger for ``trade_date`` (any status)."""
    td = str(trade_date).strip()
    with sqlite3.connect(ledger.path) as con:
        row = con.execute(
            "SELECT COUNT(*) FROM managed_positions WHERE trade_date = ?",
            (td,),
        ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def load_open_positions_for_date(
    ledger: SqliteManagedPositionLedger,
    trade_date: str,
) -> list[dict[str, Any]]:
    """
    Return active ledger rows for ``trade_date`` (``status`` is ``open`` or ``exit_submitted``).

    ``step2_row_snapshot`` is a dict (empty if stored JSON is invalid). ``strong_stock`` is bool.
    """
    td = str(trade_date).strip()
    sql = """
        SELECT
            trade_date, symbol, buy_rank, ai_confidence, step2_row_snapshot, strong_stock,
            take_profit_pct, stop_loss_pct, buy_client_order_id, buy_alpaca_order_id,
            entry_price, filled_qty, entry_timestamp_utc, exit_client_order_id,
            exit_alpaca_order_id, status
        FROM managed_positions
        WHERE trade_date = ? AND status IN ('open', 'exit_submitted')
        ORDER BY symbol, buy_rank
    """
    out: list[dict[str, Any]] = []
    with sqlite3.connect(ledger.path) as con:
        cur = con.execute(sql, (td,))
        colnames = [d[0] for d in cur.description]
        rows = cur.fetchall()

    for row in rows:
        d = dict(zip(colnames, row))
        snap_raw = d.pop("step2_row_snapshot")
        strong_i = d.pop("strong_stock")
        step2: dict[str, Any] = {}
        if isinstance(snap_raw, str) and snap_raw.strip():
            try:
                parsed = json.loads(snap_raw)
                if isinstance(parsed, dict):
                    step2 = {str(k): v for k, v in parsed.items()}
            except (json.JSONDecodeError, TypeError, ValueError):
                step2 = {}
        if strong_i is None:
            d["strong_stock"] = False
        else:
            d["strong_stock"] = bool(int(strong_i))
        d["buy_rank"] = int(d["buy_rank"]) if d.get("buy_rank") is not None else 0
        d["ai_confidence"] = float(d["ai_confidence"]) if d["ai_confidence"] is not None else 0.0
        d["take_profit_pct"] = float(d["take_profit_pct"]) if d["take_profit_pct"] is not None else 0.0
        d["stop_loss_pct"] = float(d["stop_loss_pct"]) if d["stop_loss_pct"] is not None else 0.0
        for key in ("entry_price", "filled_qty"):
            v = d.get(key)
            if v is not None:
                d[key] = float(v)
        ordered: dict[str, Any] = {
            "trade_date": d["trade_date"],
            "symbol": d["symbol"],
            "buy_rank": d["buy_rank"],
            "ai_confidence": d["ai_confidence"],
            "step2_row_snapshot": step2,
            "strong_stock": d["strong_stock"],
            "take_profit_pct": d["take_profit_pct"],
            "stop_loss_pct": d["stop_loss_pct"],
            "buy_client_order_id": d["buy_client_order_id"],
            "buy_alpaca_order_id": d["buy_alpaca_order_id"],
            "entry_price": d["entry_price"],
            "filled_qty": d["filled_qty"],
            "entry_timestamp_utc": d["entry_timestamp_utc"],
            "exit_client_order_id": d["exit_client_order_id"],
            "exit_alpaca_order_id": d["exit_alpaca_order_id"],
            "status": d["status"],
        }
        out.append(ordered)

    return out


def build_exit_client_order_id(trade_date: str, symbol: str, buy_rank: int) -> str:
    return f"EXIT-{str(trade_date).strip()}-{str(symbol).strip().upper()}-{int(buy_rank)}"


def record_exit_submission(
    ledger: SqliteManagedPositionLedger,
    *,
    symbol: str,
    buy_client_order_id: str,
    exit_client_order_id: str,
    exit_alpaca_order_id: str,
) -> bool:
    """Persist exit order ids and set ``exit_submitted`` after Alpaca accepts the sell (not yet flat)."""
    sym_u = str(symbol).strip().upper()
    cid = str(buy_client_order_id).strip()
    if not sym_u or not cid:
        return False
    with sqlite3.connect(ledger.path) as con:
        cur = con.execute(
            """
            UPDATE managed_positions
            SET exit_client_order_id = ?,
                exit_alpaca_order_id = ?,
                status = 'exit_submitted'
            WHERE buy_client_order_id = ? AND symbol = ? AND status = 'open'
            """,
            (
                str(exit_client_order_id).strip(),
                str(exit_alpaca_order_id).strip(),
                cid,
                sym_u,
            ),
        )
        con.commit()
        return bool(cur.rowcount == 1)


def mark_position_closed_flat(
    ledger: SqliteManagedPositionLedger,
    *,
    symbol: str,
    buy_client_order_id: str,
) -> bool:
    """Broker position is flat; mark row ``exited`` (from ``open`` or after ``exit_submitted``)."""
    sym_u = str(symbol).strip().upper()
    cid = str(buy_client_order_id).strip()
    if not sym_u or not cid:
        return False
    with sqlite3.connect(ledger.path) as con:
        cur = con.execute(
            """
            UPDATE managed_positions
            SET status = 'exited'
            WHERE buy_client_order_id = ? AND symbol = ? AND status IN ('open', 'exit_submitted')
            """,
            (cid, sym_u),
        )
        con.commit()
        return bool(cur.rowcount == 1)


def ledger_row_exit_pending(
    ledger: SqliteManagedPositionLedger,
    *,
    symbol: str,
    buy_client_order_id: str,
) -> tuple[str | None, str | None]:
    """Return (exit_client_order_id, exit_alpaca_order_id) for idempotency checks."""
    sym_u = str(symbol).strip().upper()
    cid = str(buy_client_order_id).strip()
    with sqlite3.connect(ledger.path) as con:
        row = con.execute(
            """
            SELECT exit_client_order_id, exit_alpaca_order_id
            FROM managed_positions
            WHERE buy_client_order_id = ? AND symbol = ?
            """,
            (cid, sym_u),
        ).fetchone()
    if not row:
        return None, None
    ec, ea = row[0], row[1]
    return (
        str(ec) if ec is not None else None,
        str(ea) if ea is not None else None,
    )


def sync_buy_client_order_id(trade_date: str, symbol: str) -> str:
    """Deterministic client id for paper account→ledger sync rows (not a real Alpaca buy)."""
    td = str(trade_date).strip()
    sym_u = str(symbol).strip().upper()
    return f"SYNC-{td}-{sym_u}"


def ledger_has_active_row_for_symbol_date(
    ledger: SqliteManagedPositionLedger,
    trade_date: str,
    symbol: str,
) -> bool:
    """True if an ``open`` or ``exit_submitted`` row exists for this trade_date and symbol."""
    td = str(trade_date).strip()
    sym_u = str(symbol).strip().upper()
    if not sym_u:
        return False
    with sqlite3.connect(ledger.path) as con:
        row = con.execute(
            """
            SELECT 1 FROM managed_positions
            WHERE trade_date = ? AND symbol = ? AND status IN ('open', 'exit_submitted')
            LIMIT 1
            """,
            (td, sym_u),
        ).fetchone()
    return row is not None


def sync_alpaca_positions_to_managed_ledger(
    ledger: SqliteManagedPositionLedger,
    *,
    trade_date: str,
    positions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Create managed ledger rows for Alpaca positions missing from the ledger (paper ownership).

    Does not sell, does not overwrite bot rows, does not set ``entry_timestamp_utc``.
    """
    td = str(trade_date).strip()
    synced: list[str] = []
    skipped: list[dict[str, str]] = []

    for raw in positions:
        if not isinstance(raw, Mapping):
            skipped.append({"symbol": "", "reason": "invalid_position_row"})
            continue
        sym = str(raw.get("symbol", "")).strip().upper()
        if not sym:
            skipped.append({"symbol": "", "reason": "missing_symbol"})
            continue
        try:
            qty = float(raw.get("qty"))
        except (TypeError, ValueError):
            skipped.append({"symbol": sym, "reason": "invalid_qty"})
            continue
        if not math.isfinite(qty) or qty <= 0.0:
            skipped.append({"symbol": sym, "reason": "non_positive_qty"})
            continue

        if ledger_has_active_row_for_symbol_date(ledger, td, sym):
            skipped.append({"symbol": sym, "reason": "active_ledger_exists"})
            continue

        buy_cid = sync_buy_client_order_id(td, sym)
        filled_qty = abs(qty)
        entry_px: float | None = None
        aep = raw.get("avg_entry_price")
        if aep is not None and _finite(aep) and float(aep) > 0.0:
            entry_px = float(aep)

        snap_json = json.dumps({}, sort_keys=True)
        try:
            with sqlite3.connect(ledger.path) as con:
                con.execute(
                    """
                    INSERT INTO managed_positions (
                        buy_client_order_id, trade_date, symbol, buy_rank, ai_confidence,
                        step2_row_snapshot, strong_stock, take_profit_pct, stop_loss_pct,
                        buy_alpaca_order_id, entry_price, filled_qty, entry_timestamp_utc,
                        exit_client_order_id, exit_alpaca_order_id, status
                    ) VALUES (?, ?, ?, 99, 0.0, ?, 0, ?, ?, NULL, ?, ?, NULL, NULL, NULL, 'open')
                    """,
                    (
                        buy_cid,
                        td,
                        sym,
                        snap_json,
                        take_profit_pct_for(strong=False),
                        STOP_LOSS_PCT,
                        entry_px,
                        filled_qty,
                    ),
                )
                con.commit()
        except sqlite3.IntegrityError:
            skipped.append({"symbol": sym, "reason": "duplicate_buy_client_order_id"})
            continue
        synced.append(sym)

    return {"synced": synced, "skipped": skipped}
