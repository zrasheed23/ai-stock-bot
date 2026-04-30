"""
Step 3 — locked opening-bell decision prompt (text only).

Must stay aligned with ``opening_decision_validate.validate_opening_decision_response`` /
``_validate_payload``: same keys, types, and constraints so model output passes Step 3.5 unchanged.
"""

from __future__ import annotations

# System prompt: authoritative contract for the opening JSON object (Step 3.5-compatible).
OPENING_DECISION_SYSTEM_PROMPT = """You are the opening-bell decision model for a trading research pipeline.

Output requirements (non-negotiable):
- Emit exactly ONE JSON object. No markdown code fences. No prose before or after the JSON.
- The JSON must parse with standard json.loads.

Top-level keys (exactly these four keys, no extras, no omissions):
1) "trade_date" — string, must equal the session trade_date supplied in the user message (YYYY-MM-DD).
2) "decision_status" — string, exactly "trade" OR "no_trade".
3) "market_read" — object with exactly one key: "summary". Value is a non-empty string.
4) "candidates" — JSON array.

If decision_status is "no_trade":
- "candidates" MUST be an empty array [].

If decision_status is "trade":
- "candidates" MUST be a non-empty array with at most 3 elements.
- Each element MUST be a JSON object with exactly these keys (no others): "rank", "symbol", "direction", "confidence", "reason".

Per-candidate rules (trade path):
- "rank": JSON integer (not a float). The first list element MUST have rank 1, the second rank 2, the third rank 3. Sequential 1-based only.
- "symbol": string; after stripping, uppercase must be one of the allowed symbols listed in the user message (and only those).
- "direction": string, exactly "long" or "short".
- "confidence": JSON number (integer or float), finite, in the closed interval [0, 1] (0 and 1 allowed).
- "reason": non-empty string; at most 40 words (words separated by whitespace).

General:
- Do not include NaN, Infinity, or null where a string or number is required.
- Do not use boolean where a number is required for confidence.

Your entire reply body must be parseable as that single JSON object."""


def build_opening_decision_user_content(
    *,
    packet_json: str,
    expected_trade_date: str,
    allowed_symbols: list[str],
) -> str:
    """User message for Step 3; keeps trade_date and symbol allowlist explicit for the model."""
    allowed = ", ".join(allowed_symbols)
    return (
        f"Session trade_date (the root JSON trade_date field MUST be this exact string): "
        f"{expected_trade_date}\n\n"
        f"Allowed candidate symbols (uppercase only in output): {allowed}\n\n"
        f"Premarket packet (JSON):\n{packet_json}\n\n"
        "Respond with the single JSON object only."
    )


def build_midmorning_decision_user_content(
    *,
    packet_json: str,
    expected_trade_date: str,
    allowed_symbols: list[str],
) -> str:
    """
    Mid-morning window (~10:30 America/New_York): same JSON schema as opening; independent decision.

    The packet JSON uses opening-field names but reflects fresh tape through the morning cutoff.
    """
    allowed = ", ".join(allowed_symbols)
    return (
        "Decision window: mid-morning (approximately 10:30 America/New_York). "
        "This is an independent decision — do not infer or reuse any earlier opening-bell decision.\n\n"
        f"Session trade_date (the root JSON trade_date field MUST be this exact string): "
        f"{expected_trade_date}\n\n"
        f"Allowed candidate symbols (uppercase only in output): {allowed}\n\n"
        f"Intraday-enriched market packet (JSON):\n{packet_json}\n\n"
        "Respond with the single JSON object only."
    )
