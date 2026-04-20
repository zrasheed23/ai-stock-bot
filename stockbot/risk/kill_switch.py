"""Filesystem kill-switch: create/touch file or write '1' to halt all execution."""

from __future__ import annotations

from pathlib import Path


class KillSwitch:
    def __init__(self, path: Path):
        self._path = path

    def is_active(self) -> bool:
        if not self._path.exists():
            return False
        try:
            text = self._path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            return True
        if text == "":
            return True
        return text in {"1", "true", "yes", "on", "halt", "stop"}
