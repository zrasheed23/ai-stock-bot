"""Load settings from environment. No secret files committed."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

_LOG = logging.getLogger("stockbot.config")

# -----------------------------------------------------------------------------
# Pre-market MVP-0 — thresholds (tune here without editing strategy math)
# -----------------------------------------------------------------------------
# Master switch default; override with env ``STOCKBOT_ENABLE_PREMARKET_SIGNALS=0``.
ENABLE_PREMARKET_SIGNALS = True

# Optional conservative behavior toggles (opening / mid-morning only; default off).
# STOCKBOT_SLOT2_RELAX_OPENING=1 — slightly easier slot-2 acceptance + smaller slot-2 weight when borderline.
# STOCKBOT_MIDMORNING_RELAX_FILTERS=1 — small tape-threshold deltas for sector-RS mid-morning path.

PM_GAP_ATR_HARD_SKIP = 2.5
PM_GAP_ATR_WARN = 1.0
PM_RVOL_MIN_ON_GAP = 0.3
PM_RVOL_STRONG = 2.0
SPY_GAP_ATR_REDUCE = -1.5
SPY_GAP_ATR_NO_TRADE = -2.5

# TEMPORARY: remove this block once you trust `.env` + shell loading (beginner debug aid).
_FINNHUB_ENV = "FINNHUB_API_KEY"


def _log_finnhub_key_fingerprint(phase: str) -> None:
    """Safe hint: length + last 4 chars only (never log the full key)."""
    raw = os.environ.get(_FINNHUB_ENV)
    if raw is None:
        _LOG.info("[env debug] %s value (%s): absent", _FINNHUB_ENV, phase)
        return
    v = raw.strip()
    if not v:
        _LOG.info("[env debug] %s value (%s): empty string", _FINNHUB_ENV, phase)
        return
    last4 = v[-4:] if len(v) >= 4 else v
    _LOG.info(
        "[env debug] %s fingerprint (%s): length=%d last4=%s",
        _FINNHUB_ENV,
        phase,
        len(v),
        last4,
    )


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_if_present() -> None:
    """
    Load a local `.env` file into os.environ (if it exists).

    Why:
      - In production you typically inject env vars via the process manager.
      - In local/dev, beginners often store keys in a `.env` file.

    This keeps architecture intact: Settings still reads from os.environ, we just
    populate it first.
    """
    # Project root = directory that contains the `stockbot/` package (where `.env` lives).
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    env_abs = str(env_path.resolve())
    env_exists = env_path.is_file()

    _LOG.info("[env debug] .env path (absolute): %s", env_abs)
    _LOG.info("[env debug] .env file exists: %s", env_exists)

    before_present = _FINNHUB_ENV in os.environ
    _LOG.info("[env debug] %s in os.environ before load_dotenv: %s", _FINNHUB_ENV, before_present)
    if before_present:
        _log_finnhub_key_fingerprint("before load_dotenv")

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        # python-dotenv is optional at runtime, but recommended for local usage.
        _LOG.warning("[env debug] python-dotenv not installed; cannot load .env file")
        after_present = _FINNHUB_ENV in os.environ
        _LOG.info("[env debug] %s in os.environ after load_dotenv: %s", _FINNHUB_ENV, after_present)
        _log_finnhub_key_fingerprint("after (no dotenv)")
        return

    if env_exists:
        # override=True: if FINNHUB_API_KEY (etc.) is already set in the shell from an old
        # session, we still want `.env` to win during local dev so edits to `.env` take effect.
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        _LOG.info("[env debug] load_dotenv skipped (no .env file at path above)")

    after_present = _FINNHUB_ENV in os.environ
    _LOG.info("[env debug] %s in os.environ after load_dotenv: %s", _FINNHUB_ENV, after_present)
    _log_finnhub_key_fingerprint("after load_dotenv")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Extend as needed; keep defaults safe for local dry-run."""

    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    anthropic_api_key: str
    finnhub_api_key: str
    kill_switch_path: Path
    state_dir: Path
    audit_dir: Path
    dry_run: bool
    max_position_fraction: float  # max fraction of equity for one position
    max_daily_trades: int
    enable_premarket_signals: bool
    alpaca_data_feed: str  # Alpaca stock bars feed, e.g. ``sip`` or ``iex``

    @staticmethod
    def from_env() -> "Settings":
        _load_dotenv_if_present()
        base = Path(os.environ.get("STOCKBOT_STATE_DIR", "./var/state")).resolve()
        audit = Path(os.environ.get("STOCKBOT_AUDIT_DIR", "./var/audit")).resolve()
        kill = Path(
            os.environ.get("STOCKBOT_KILL_SWITCH_PATH", "./var/kill_switch")
        ).resolve()
        return Settings(
            alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
            alpaca_base_url=os.environ.get(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
            ),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            finnhub_api_key=os.environ.get("FINNHUB_API_KEY", ""),
            kill_switch_path=kill,
            state_dir=base,
            audit_dir=audit,
            dry_run=_env_bool("STOCKBOT_DRY_RUN", True),
            max_position_fraction=float(os.environ.get("STOCKBOT_MAX_POSITION_FRAC", "0.1")),
            max_daily_trades=int(os.environ.get("STOCKBOT_MAX_DAILY_TRADES", "1")),
            enable_premarket_signals=_env_bool(
                "STOCKBOT_ENABLE_PREMARKET_SIGNALS", ENABLE_PREMARKET_SIGNALS
            ),
            alpaca_data_feed=os.environ.get("STOCKBOT_ALPACA_DATA_FEED", "iex").strip().lower()
            or "iex",
        )
