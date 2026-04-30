"""
Paper exit engine runner: Step 7 loop for managed ledger positions only.

Requires Alpaca paper trading + market data URLs. Never submits buys.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from stockbot.config import Settings, _env_bool
from stockbot.execution.alpaca_exit_submit import AlpacaExitHttpClient
from stockbot.execution.exit_engine import run_exit_engine_loop
from stockbot.runners.managed_position_ledger import (
    SqliteManagedPositionLedger,
    default_managed_position_ledger_path,
    sync_alpaca_positions_to_managed_ledger,
)

_LOG = logging.getLogger("stockbot.runners.paper_exit_run")
_ET = ZoneInfo("America/New_York")


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _require_env() -> tuple[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "").strip()
    sec = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    trading = os.environ.get("ALPACA_TRADING_BASE_URL", "").strip().rstrip("/")
    data = os.environ.get("ALPACA_DATA_BASE_URL", "").strip().rstrip("/")
    missing = [
        n
        for n, v in (
            ("ALPACA_API_KEY", key),
            ("ALPACA_SECRET_KEY", sec),
            ("ALPACA_TRADING_BASE_URL", trading),
            ("ALPACA_DATA_BASE_URL", data),
        )
        if not v
    ]
    if missing:
        _die(f"Missing required environment variables: {', '.join(missing)}.")
    p = urlparse(trading)
    if p.scheme != "https" or p.hostname != "paper-api.alpaca.markets":
        _die(
            f"ALPACA_TRADING_BASE_URL must be https://paper-api.alpaca.markets (got host={p.hostname!r})."
        )
    d = urlparse(data)
    if d.scheme != "https" or d.hostname != "data.alpaca.markets":
        _die(f"ALPACA_DATA_BASE_URL must be https://data.alpaca.markets (got host={d.hostname!r}).")
    return trading, data


def _settings_for_paper(trading_base_url: str) -> Settings:
    os.environ["ALPACA_BASE_URL"] = trading_base_url
    base = Settings.from_env()
    return replace(base, dry_run=False)


def run_paper_exit(
    trade_date: date,
    *,
    settings: Settings | None = None,
    max_scans: int | None = None,
    sync_account_positions: bool = False,
) -> dict[str, Any]:
    trading_url, data_url = _require_env()
    if settings is None:
        settings = _settings_for_paper(trading_url)
    else:
        pu = urlparse(settings.alpaca_base_url.rstrip("/"))
        if pu.scheme != "https" or pu.hostname != "paper-api.alpaca.markets":
            _die("settings.alpaca_base_url must be https://paper-api.alpaca.markets when settings is passed in.")
        settings = replace(settings, dry_run=False)

    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }
    client = AlpacaExitHttpClient(
        trading_base_url=settings.alpaca_base_url,
        data_base_url=data_url,
        headers=headers,
        data_feed=settings.alpaca_data_feed,
    )
    ledger = SqliteManagedPositionLedger(default_managed_position_ledger_path())
    account_sync: dict[str, Any] | None = None
    if sync_account_positions:
        account_sync = sync_alpaca_positions_to_managed_ledger(
            ledger,
            trade_date=trade_date.isoformat(),
            positions=client.list_positions(),
        )
    eod = _env_bool("STOCKBOT_EXIT_EOD_FLATTEN", True)
    loop_result = run_exit_engine_loop(
        trade_date=trade_date.isoformat(),
        ledger=ledger,
        client=client,
        eod_flatten_enabled=eod,
        max_scans=max_scans,
    )
    if account_sync is not None:
        return {"account_sync": account_sync, **loop_result}
    return loop_result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Paper exit engine (Step 7; ledger positions only).")
    parser.add_argument(
        "--trade-date",
        type=str,
        default=None,
        help="YYYY-MM-DD (default: today in America/New_York)",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Stop after N scans (default: run until no open ledger rows)",
    )
    parser.add_argument(
        "--sync-account-positions",
        action="store_true",
        help=(
            "Before the exit loop: create managed ledger rows for Alpaca positions "
            "missing from the ledger (paper account ownership; no sells)."
        ),
    )
    args = parser.parse_args()
    if args.trade_date:
        td = date.fromisoformat(args.trade_date)
    else:
        td = datetime.now(_ET).date()
    try:
        result = run_paper_exit(
            td,
            max_scans=args.max_scans,
            sync_account_positions=args.sync_account_positions,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        _LOG.exception("paper_exit_run failed")
        _die(f"paper_exit_run failed: {exc!r}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
