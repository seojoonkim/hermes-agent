"""Send the turn receipt (Simon's requirement #1) without blocking the turn.

Called from the gateway right after a message is accepted for an agent run
and before session hygiene / preflight compression, so the user sees within
seconds that the request was understood. Bounded and fail-open: a slow or
broken platform send can never delay or break the actual work.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Optional

from agent.turn_receipt import build_receipt, is_trivial_message

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 4.0


async def send_turn_receipt(
    adapter: Any,
    source: Any,
    message_text: str,
    *,
    estimate: Optional[Mapping[str, Any]],
    enabled: bool,
    metadata: Optional[dict] = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> bool:
    """Return True only when a receipt was actually delivered."""
    if not enabled or adapter is None or is_trivial_message(message_text):
        return False
    chat_id = getattr(source, "chat_id", None)
    if not chat_id:
        return False
    text = build_receipt(message_text, eta=estimate)
    try:
        result = await asyncio.wait_for(adapter.send(chat_id, text, metadata=metadata), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — receipt is best-effort, never blocks work
        logger.warning("turn receipt not delivered (%s: %s)", type(exc).__name__, exc)
        return False
    return bool(getattr(result, "success", False))
