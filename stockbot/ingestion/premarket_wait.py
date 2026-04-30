"""Live-run scheduling: wait until 09:25 ET (not part of Step 1 ingestion math)."""

from __future__ import annotations

import logging
import os
import time as time_module
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from stockbot.config import Settings

_LOG = logging.getLogger("stockbot.ingestion.premarket_wait")
_ET = ZoneInfo("America/New_York")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def wait_until_premarket_decision_et(trade_date: date, settings: Settings) -> None:
    """
    For *live* runs on ``trade_date`` (today in ET), sleep until 09:25 ET so the
    pre-market snapshot is not taken too early. Historical / backtest dates skip waiting.

    Set ``STOCKBOT_PREMARKET_NO_WAIT=1`` to disable sleeping (tests / CI).
    """
    if not settings.enable_premarket_signals:
        return
    if _env_bool("STOCKBOT_PREMARKET_NO_WAIT", False):
        return
    now_et = datetime.now(_ET)
    if now_et.date() != trade_date:
        return
    target = datetime.combine(trade_date, time(9, 25), tzinfo=_ET)
    if now_et >= target:
        return
    sleep_s = (target - now_et).total_seconds()
    if sleep_s > 0:
        _LOG.info(
            "[premarket] waiting %.0fs until 09:25 America/New_York (trade_date=%s)",
            sleep_s,
            trade_date.isoformat(),
        )
        time_module.sleep(sleep_s)
