"""Deterministic direct replay of the immediately preceding assistant answer."""

from __future__ import annotations

import re
from typing import Any

_EXACT_REPLAY_PATTERNS = (
    re.compile(r"^(?:전체|전부|원문|직전\s*답변|방금\s*답변)(?:을|를|은|는)?\s*다시\s*(?:전체\s*)?(?:출력|보여|보내|써)(?:\s*줘|\s*주세요|줘요|해|해줘|하세요)?[.!?]?$"),
    re.compile(
        r"^(?:again\s+)?(?:print|show|send)\s+(?:the\s+)?"
        r"(?:(?:whole|full)\s+)?(?:(?:previous|last)\s+)?"
        r"(?:answer|response)(?:\s+again)?[.!?]?$",
        re.IGNORECASE,
    ),
)


def is_exact_replay_request(text: str | None) -> bool:
    """Match only standalone replay requests, never requests with new edits."""
    normalized = " ".join(str(text or "").strip().split())
    if not normalized or len(normalized) > 80:
        return False
    return any(pattern.fullmatch(normalized) for pattern in _EXACT_REPLAY_PATTERNS)


async def resolve_direct_replay(
    *,
    text: str | None,
    current_session_id: str,
    previous_session_id: str | None,
    session_db: Any,
) -> str | None:
    """Return exact durable text from the current or auto-reset predecessor."""
    if not is_exact_replay_request(text) or session_db is None:
        return None
    # An auto-reset creates an empty current session. Prefer the explicitly
    # recorded predecessor; otherwise replay the current session's last answer.
    target = str(previous_session_id or current_session_id or "").strip()
    if not target:
        return None
    row = await session_db.get_latest_assistant_message(target)
    if not isinstance(row, dict):
        return None
    content = row.get("content")
    return content if isinstance(content, str) and content.strip() else None
