"""
Paper-trading deployable sizing — orchestration only, outside Steps 1–6.

Use the returned value as ``deployable_usd`` for ``build_opening_execution_plan``.
"""

from __future__ import annotations

import logging

from stockbot.execution.broker import AlpacaBroker

_LOG = logging.getLogger("stockbot.execution.paper_deployable")

# Fixed fraction of account equity (paper / live account summary from Alpaca).
PAPER_DEPLOYABLE_EQUITY_FRACTION = 0.75


def deployable_usd_from_account_equity(account_equity: float) -> float:
    """deployable_usd = PAPER_DEPLOYABLE_EQUITY_FRACTION * account_value (equity). No caps or scaling."""
    eq = float(account_equity)
    dep = PAPER_DEPLOYABLE_EQUITY_FRACTION * eq
    _LOG.info(
        "DEPLOYABLE_EQUITY equity=%s deployable_fraction=%s deployable_usd=%s",
        eq,
        PAPER_DEPLOYABLE_EQUITY_FRACTION,
        dep,
    )
    return dep


def deployable_usd_from_alpaca_broker(broker: AlpacaBroker) -> float:
    """Read equity via ``broker.get_account()`` then apply the fixed fraction."""
    acct = broker.get_account()
    return deployable_usd_from_account_equity(acct.equity)
