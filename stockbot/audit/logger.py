"""Structured audit trail: reasoning JSON + Python logging for operational errors."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stockbot.models import DailyReasoningRecord

_LOG = logging.getLogger("stockbot")


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    raise TypeError(f"Not serializable: {type(o)}")


class AuditLogger:
    def __init__(self, audit_dir: Path):
        self._dir = audit_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write_reasoning(self, record: DailyReasoningRecord) -> Path:
        day = record.trade_date
        path = self._dir / f"reasoning_{day}.json"
        path.write_text(
            json.dumps(asdict(record), indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def log_error(msg: str, exc_info: bool = False) -> None:
        _LOG.error(msg, exc_info=exc_info)
