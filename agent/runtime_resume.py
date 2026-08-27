"""Bounded handoff payloads for resuming incomplete runtime turns."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

_MAX_GOAL_CHARS = 1000
_MAX_FAILURES = 10
_MAX_FAILURE_CHARS = 240
_COMMIT_RE = re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE)
_TEST_FAILURE_RE = re.compile(r"(?m)^FAILED\s+([^\s]+::[^\s]+)")


@dataclass(frozen=True)
class IncompleteTurnHandoff:
    """Small durable contract shared by the runtime, memory, and gateway."""

    resume_token: str
    session_id: str
    termination_reason: str
    incomplete_goal: str
    last_verified_commit: str
    failing_tests: tuple[str, ...]
    next_dispatch_intent: str = "resume_in_distinct_gateway_turn"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failing_tests"] = list(self.failing_tests)
        return payload
def _message_texts(messages: Iterable[Any]) -> Iterable[str]:
    def _content_texts(content: Any) -> Iterable[str]:
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    yield block
                elif isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str):
                        yield text

    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        yield from _content_texts(content)


def build_incomplete_handoff(
    *,
    session_id: str,
    turn_exit_reason: str,
    incomplete_goal: Any,
    messages: Iterable[Mapping[str, Any]],
) -> IncompleteTurnHandoff:
    """Build a deterministic, size-bounded resume handoff from verified output."""
    goal = str(incomplete_goal or "").strip()[:_MAX_GOAL_CHARS]
    commit = ""
    failures: list[str] = []
    for text in _message_texts(messages):
        for match in _COMMIT_RE.finditer(text):
            commit = match.group(0).lower()
        for match in _TEST_FAILURE_RE.finditer(text):
            node_id = match.group(1).rstrip(".,:;)]")[:_MAX_FAILURE_CHARS]
            if node_id not in failures:
                failures.append(node_id)
                if len(failures) >= _MAX_FAILURES:
                    break
        if len(failures) >= _MAX_FAILURES:
            break
    failures.sort()

    identity = {
        "session_id": str(session_id or "")[:200],
        "termination_reason": str(turn_exit_reason or "")[:200],
        "incomplete_goal": goal,
        "last_verified_commit": commit,
        "failing_tests": failures,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    token = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return IncompleteTurnHandoff(
        resume_token=token,
        session_id=identity["session_id"],
        termination_reason=identity["termination_reason"],
        incomplete_goal=goal,
        last_verified_commit=commit,
        failing_tests=tuple(failures),
    )


def build_resume_prompt(handoff: Mapping[str, Any]) -> str:
    """Render the bounded handoff as a synthetic next-turn instruction."""
    bounded = {
        "resume_token": str(handoff.get("resume_token") or "")[:64],
        "last_verified_commit": str(handoff.get("last_verified_commit") or "")[:40],
        "incomplete_goal": str(handoff.get("incomplete_goal") or "")[:_MAX_GOAL_CHARS],
        "failing_tests": [str(item)[:_MAX_FAILURE_CHARS] for item in (handoff.get("failing_tests") or [])[:_MAX_FAILURES]],
        "next_dispatch_intent": str(handoff.get("next_dispatch_intent") or "")[:100],
    }
    return (
        "[Hermes durable resume handoff — distinct turn]\n"
        "Continue the incomplete goal from the persisted checkpoint. Diagnose the prior stop from the evidence "
        "before acting: verify current state, identify the failed assumption or blocked dependency, then choose "
        "a smaller bounded step that produces a verifiable artifact. Address any listed failing tests and finish "
        "or report a concrete blocker. Do not repeat the same failed approach unchanged, merely restate the prior "
        "summary, or expand scope.\n"
        + json.dumps(bounded, ensure_ascii=False, sort_keys=True)
    )
