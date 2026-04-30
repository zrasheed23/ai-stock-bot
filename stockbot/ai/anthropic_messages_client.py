"""Anthropic Messages API via the official SDK (URL, versioning, and errors handled by the client)."""

from __future__ import annotations

import os
from typing import Iterable

import anthropic


def _join_text_blocks(content: Iterable[object]) -> str:
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def anthropic_messages_text(
    *,
    api_key: str,
    system: str,
    user_text: str,
    model: str,
    max_tokens: int,
) -> str:
    """
    One-shot user message with optional system prompt; returns concatenated text blocks.

    ``model`` must be supplied by the caller (runner / processor) so each call site can
    document its own env override without hidden defaults in shared code.
    """
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_text}],
    )
    return _join_text_blocks(message.content)


def resolve_opening_decision_model() -> str:
    """Opening-bell Step 3 model id: dedicated env, then shared Anthropic model env, then API default."""
    for name in ("STOCKBOT_ANTHROPIC_OPENING_MODEL", "STOCKBOT_ANTHROPIC_MODEL"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    # Default: current Sonnet family id (older ``claude-3-5-sonnet-20241022`` often 404s as retired).
    return "claude-sonnet-4-20250514"
