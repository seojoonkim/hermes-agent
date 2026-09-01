"""Human-facing fast replies for status questions during busy turns.

This module is intentionally pure: the gateway owns delivery and session state,
while these helpers only classify a narrow status-question surface and render a
short reply. Keeping the classifier narrow prevents ordinary work instructions
from being consumed by the fast lane.
"""

from __future__ import annotations

import re

_KOREAN_RE = re.compile(r"[가-힣]")
_KOREAN_STATUS_PATTERNS = (
    re.compile(r"왜\s*(?:이렇게\s*)?(?:오래|늦)"),
    re.compile(r"왜.*(?:답|응답).*(?:없|안)"),
    re.compile(r"(?:얼마나|몇\s*분).*(?:남|걸)"),
    re.compile(r"(?:지금\s*)?(?:무슨|어떤)\s*(?:상태|단계|작업)"),
    re.compile(r"(?:진행|상태|현황).*(?:어때|알려|뭐)"),
)
_ENGLISH_STATUS_PATTERNS = (
    re.compile(r"\bwhy\b.*\b(?:slow|long|late|respond|reply|answer)\b"),
    re.compile(r"\bhow\s+long\b"),
    re.compile(r"\b(?:status|progress|eta)\b"),
    re.compile(r"\b(?:still\s+(?:working|there)|what\s+(?:are\s+you|is)\s+doing)\b"),
)


def is_busy_status_question(text: str | None) -> bool:
    """Return True only for a direct status/latency check.

    A lone question mark is treated as a status check because this helper is
    called only while the same conversation already has an active turn.
    """

    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    if normalized in {"?", "??", "???", "왜?", "상태?", "진행?", "eta?"}:
        return True
    patterns = (
        _KOREAN_STATUS_PATTERNS if _KOREAN_RE.search(normalized) else _ENGLISH_STATUS_PATTERNS
    )
    return any(pattern.search(normalized) for pattern in patterns)


def render_compression_timeout_reply(text: str | None) -> str:
    """Explain a fail-open without exposing settings or destructive commands."""
    raw = str(text or "")
    if _KOREAN_RE.search(raw):
        return (
            "이전 대화 정리가 30초 안에 끝나지 않아 이번에는 건너뛰었어. "
            "기존 메시지는 그대로 보존돼 있고, 지금 요청을 계속 처리할게."
        )
    return (
        "Organizing the earlier conversation did not finish within 30 seconds, "
        "so I skipped it this time. Your messages are preserved and I am continuing "
        "with the current request."
    )


def render_compression_status_reply(text: str | None, elapsed_seconds: float = 0.0) -> str:
    """Render a truthful reply without exposing queue/compression internals."""

    elapsed = max(0, int(float(elapsed_seconds or 0.0)))
    if _KOREAN_RE.search(str(text or "")) or str(text or "").strip() in {"?", "??", "???"}:
        elapsed_note = f" 시작한 지 약 {max(1, elapsed // 60)}분 됐어." if elapsed >= 60 else ""
        return (
            "지금 이 대화의 이전 기록을 정리 중이라 원래 작업의 답변 시작이 늦어지고 있어."
            f"{elapsed_note} 기록은 보존돼 있고, 이런 상태 질문에는 지금 바로 답할 수 있어. "
            "원래 요청은 정리 단계가 끝나거나 제한 시간에 도달하면 이어서 처리할게."
        )
    elapsed_note = f" It started about {max(1, elapsed // 60)} minute(s) ago." if elapsed >= 60 else ""
    return (
        "I am organizing this conversation's earlier history, so the original task has not "
        f"started answering yet.{elapsed_note} The history is preserved, and I can answer "
        "status questions immediately while that finishes or reaches its time limit."
    )
