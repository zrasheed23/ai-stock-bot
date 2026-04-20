"""Pre-trade guardrails: capital, sizing, frequency, global halt."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from stockbot.config import Settings
from stockbot.models import RiskVerdict, ScoredCandidate
from stockbot.risk.kill_switch import KillSwitch

# Live fills only (see pipeline): cap how many successful broker submits count against the day.
MAX_LIVE_TRADES_PER_DAY = 2


@dataclass
class AccountSummary:
    equity: float
    cash: float
    buying_power: float


class RiskEngine:
    def __init__(self, settings: Settings, kill_switch: KillSwitch | None = None):
        self._settings = settings
        self._kill = kill_switch or KillSwitch(settings.kill_switch_path)

    def evaluate(
        self,
        trade_date: date,
        candidate: ScoredCandidate | None,
        account: AccountSummary,
        daily_trades_executed: int,
        open_position_symbols: set[str],
        notional_fraction: float = 1.0,
    ) -> RiskVerdict:
        reasons: list[str] = []

        if self._kill.is_active():
            reasons.append("KILL_SWITCH_ACTIVE")

        if daily_trades_executed >= MAX_LIVE_TRADES_PER_DAY:
            reasons.append("DAILY_TRADE_LIMIT")

        if candidate is None:
            reasons.append("NO_CANDIDATE")
            return RiskVerdict(allowed=False, block_reasons=reasons)

        if candidate.symbol in open_position_symbols:
            reasons.append("ALREADY_HOLDING_SYMBOL")

        if account.equity <= 0:
            reasons.append("INVALID_EQUITY")

        # Scales the usual max sleeve (e.g. 0.7 / 0.3 when two trades selected — see pipeline).
        nf = max(0.0, float(notional_fraction))
        max_notional = account.equity * self._settings.max_position_fraction * nf
        if max_notional < 1.0:
            reasons.append("POSITION_SIZE_FLOOR")

        if account.buying_power < max_notional * 0.99:
            reasons.append("INSUFFICIENT_BUYING_POWER")

        # No-trade conditions from features (deterministic)
        vol = candidate.features.technical.get("volatility_ann", 0.0)
        if vol > 0.55:
            reasons.append("VOLATILITY_TOO_HIGH")

        if candidate.features.sentiment.get("has_high_risk", 0.0) >= 1.0:
            reasons.append("ELEVATED_LLM_RISK_FLAGS")

        if reasons:
            return RiskVerdict(allowed=False, block_reasons=reasons)

        # Whole-share qty for equities (extend for fractional if broker supports)
        last = candidate.features.technical.get("last_close", 0.0)
        if last <= 0:
            return RiskVerdict(allowed=False, block_reasons=["BAD_PRICE"])

        qty = int(max_notional // last)
        if qty < 1:
            return RiskVerdict(allowed=False, block_reasons=["QTY_ZERO"])

        notional = qty * last
        return RiskVerdict(
            allowed=True,
            block_reasons=[],
            position_qty=qty,
            notional_usd=round(notional, 2),
        )


def load_daily_trade_count(state_dir: Path, trade_date: date) -> int:
    """How many live executions were persisted for *trade_date* (0..MAX_LIVE_TRADES_PER_DAY)."""
    p = state_dir / "daily_state.json"
    if not p.exists():
        return 0
    import json

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    date_s = trade_date.isoformat()
    if data.get("last_trade_date") != date_s:
        return 0
    if "trades_on_date" in data:
        return min(MAX_LIVE_TRADES_PER_DAY, int(data["trades_on_date"]))
    # Legacy file: a single trade was recorded for last_trade_date.
    return 1


def record_trade_execution(state_dir: Path, trade_date: date) -> None:
    """Increment persisted live trade count for *trade_date* (used after each successful non-dry submit)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    import json

    p = state_dir / "daily_state.json"
    nxt = load_daily_trade_count(state_dir, trade_date) + 1
    payload = {"last_trade_date": trade_date.isoformat(), "trades_on_date": nxt}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
