#!/usr/bin/env python3
"""Daily entrypoint: scheduler runs this once before the open."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone

from stockbot.config import Settings
from stockbot.pipeline import run_daily_pipeline


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _parse_ymd(value: str) -> date:
    """Parse --date as YYYY-MM-DD; used by argparse for a clear error message."""
    try:
        return date.fromisoformat(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"not a valid date: {value!r} (use YYYY-MM-DD, e.g. 2026-04-17)"
        ) from err


def main() -> int:
    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Run the stockbot daily pipeline once.",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        type=_parse_ymd,
        default=None,
        help="Trade date to use (default: today, UTC calendar date)",
    )
    args = parser.parse_args()

    trade_date = args.date
    if trade_date is None:
        trade_date = datetime.now(timezone.utc).date()

    settings = Settings.from_env()
    result = run_daily_pipeline(trade_date=trade_date, settings=settings)
    print(result.plain_english)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
