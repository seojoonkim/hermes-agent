"""Bounded, durable handoffs for resuming incomplete runtime turns.

The contract is deliberately independent of transcripts and prompt-cache state.
Only verified, bounded evidence is persisted in MemKraft's task namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

MAX_RESUME_DEPTH = 2
MAX_GOAL_CHARS = 1000
MAX_FAILING_TESTS = 10
MAX_TEST_ID_CHARS = 240
_MAX_SCOPE_CHARS = 300
_COMMIT_RE = re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE)
_TEST_FAILURE_RE = re.compile(r"(?m)^FAILED\s+([^\s]+::[^\s]+)")
_TOKEN_RE = re.compile(r"\A[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class IncompleteTurnHandoff:
    """Closed, size-bounded contract shared by runtime, memory, and gateway."""

    resume_token: str
    prior_token: str
    profile: str
    session_id: str
    channel_key: str
    termination_reason: str
    incomplete_goal: str
    last_verified_commit: str
    failing_tests: tuple[str, ...]
    chain_depth: int
    next_dispatch_intent: str = "resume_in_distinct_gateway_turn"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failing_tests"] = list(self.failing_tests)
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> IncompleteTurnHandoff | None:
        """Validate untrusted persisted data, failing closed on schema drift."""
        try:
            token = str(raw["resume_token"])
            commit = str(raw["last_verified_commit"])
            failures = tuple(str(item) for item in raw["failing_tests"])
            depth = int(raw["chain_depth"])
            if not _TOKEN_RE.fullmatch(token):
                return None
            if commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
                return None
            if depth < 0 or depth >= MAX_RESUME_DEPTH:
                return None
            if len(failures) > MAX_FAILING_TESTS or any(
                len(item) > MAX_TEST_ID_CHARS or "::" not in item for item in failures
            ):
                return None
            handoff = cls(
                resume_token=token,
                prior_token=str(raw.get("prior_token") or "")[:64],
                profile=str(raw["profile"])[:_MAX_SCOPE_CHARS],
                session_id=str(raw["session_id"])[:_MAX_SCOPE_CHARS],
                channel_key=str(raw["channel_key"])[:_MAX_SCOPE_CHARS],
                termination_reason=str(raw["termination_reason"])[:200],
                incomplete_goal=str(raw["incomplete_goal"])[:MAX_GOAL_CHARS],
                last_verified_commit=commit,
                failing_tests=failures,
                chain_depth=depth,
                next_dispatch_intent=str(raw["next_dispatch_intent"])[:100],
            )
        except (KeyError, TypeError, ValueError):
            return None
        return handoff if handoff.resume_token == _resume_token(handoff) else None


def _verified_tool_texts(messages: Iterable[Any]) -> Iterable[str]:
    """Yield evidence only from tool-result messages, never model assertions."""

    def content_texts(content: Any) -> Iterable[str]:
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
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if role in {"tool", "tool_result"}:
            yield from content_texts(content)


def _identity(handoff: IncompleteTurnHandoff) -> dict[str, Any]:
    """Progress identity excludes chain metadata, so unchanged work is detected."""
    return {
        "profile": handoff.profile,
        "session_id": handoff.session_id,
        "channel_key": handoff.channel_key,
        "termination_reason": handoff.termination_reason,
        "incomplete_goal": handoff.incomplete_goal,
        "last_verified_commit": handoff.last_verified_commit,
        "failing_tests": list(handoff.failing_tests),
    }


def _resume_token(handoff: IncompleteTurnHandoff) -> str:
    canonical = json.dumps(
        _identity(handoff), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def build_incomplete_handoff(
    *,
    profile: str,
    session_id: str,
    channel_key: str,
    turn_exit_reason: str,
    incomplete_goal: Any,
    messages: Iterable[Mapping[str, Any]],
    prior_token: str = "",
    chain_depth: int = 0,
) -> IncompleteTurnHandoff:
    """Build a deterministic handoff from the last verified tool output."""
    goal = str(incomplete_goal or "").strip()[:MAX_GOAL_CHARS]
    commit = ""
    failures: list[str] = []
    for text in _verified_tool_texts(messages):
        commits = _COMMIT_RE.findall(text)
        if commits:
            commit = commits[-1].lower()
        for match in _TEST_FAILURE_RE.finditer(text):
            node_id = match.group(1).rstrip(".,:;)]")
            if len(node_id) <= MAX_TEST_ID_CHARS and node_id not in failures:
                failures.append(node_id)
    failures = sorted(failures)[:MAX_FAILING_TESTS]
    handoff = IncompleteTurnHandoff(
        resume_token="0" * 32,
        prior_token=str(prior_token or "")[:64],
        profile=str(profile or "")[:_MAX_SCOPE_CHARS],
        session_id=str(session_id or "")[:_MAX_SCOPE_CHARS],
        channel_key=str(channel_key or "")[:_MAX_SCOPE_CHARS],
        termination_reason=str(turn_exit_reason or "")[:200],
        incomplete_goal=goal,
        last_verified_commit=commit,
        failing_tests=tuple(failures),
        chain_depth=max(0, int(chain_depth)),
    )
    return replace(handoff, resume_token=_resume_token(handoff))


def build_resume_prompt(handoff: Mapping[str, Any]) -> str:
    """Render only the bounded contract; never splice transcript or cache text."""
    bounded = {
        "resume_token": str(handoff.get("resume_token") or "")[:32],
        "last_verified_commit": str(handoff.get("last_verified_commit") or "")[:40],
        "incomplete_goal": str(handoff.get("incomplete_goal") or "")[:MAX_GOAL_CHARS],
        "failing_tests": [
            str(item)[:MAX_TEST_ID_CHARS]
            for item in (handoff.get("failing_tests") or [])[:MAX_FAILING_TESTS]
        ],
        "next_dispatch_intent": str(handoff.get("next_dispatch_intent") or "")[:100],
    }
    return (
        "[Hermes durable resume handoff — distinct internal turn]\n"
        "Verify current state, then choose a smaller bounded step that produces a "
        "verifiable artifact. Address listed failing tests and finish the goal or "
        "report a concrete blocker; do not repeat the same approach or expand scope.\n"
        + json.dumps(bounded, ensure_ascii=False, sort_keys=True)
    )


class MemKraftResumeStore:
    """Atomic task-file persistence compatible with a MemKraft data root."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._directory = Path(root) / "tasks" / "hermes-resume"

    def _path(self, token: str) -> Path:
        if not _TOKEN_RE.fullmatch(token):
            raise ValueError("invalid resume token")
        return self._directory / f"{token}.json"

    def persist(self, handoff: IncompleteTurnHandoff) -> bool:
        """Persist once by deterministic token; equal retries are idempotent."""
        path = self._path(handoff.resume_token)
        document = {"status": "incomplete", "handoff": handoff.to_dict()}
        encoded = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")) == document
            except (OSError, json.JSONDecodeError):
                return False
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(self._directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        except OSError:
            if temporary:
                Path(temporary).unlink(missing_ok=True)
            return False

    def load(
        self,
        token: str,
        *,
        profile: str | None = None,
        session_id: str | None = None,
        channel_key: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            document = json.loads(self._path(token).read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if document.get("status") not in {"incomplete", "dispatched"}:
            return None
        handoff = IncompleteTurnHandoff.from_dict(document.get("handoff") or {})
        if handoff is None:
            return None
        expected = (
            ("profile", profile),
            ("session_id", session_id),
            ("channel_key", channel_key),
        )
        if any(
            value is not None and getattr(handoff, key) != value
            for key, value in expected
        ):
            return None
        return handoff.to_dict()

    def mark_dispatched(self, token: str) -> bool:
        """Durably record dispatch, preventing duplicate restart delivery."""
        try:
            path = self._path(token)
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("status") == "dispatched":
                return False
            if document.get("status") != "incomplete":
                return False
            document["status"] = "dispatched"
            path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False


class ResumeCoordinator:
    """Persist and arm one token-bound internal turn after user delivery."""

    def __init__(
        self,
        *,
        store: MemKraftResumeStore,
        register_post_delivery: Callable[[Callable[[str], None]], Any],
        schedule_internal: Callable[[str, str], Any],
    ) -> None:
        self._store = store
        self._register = register_post_delivery
        self._schedule = schedule_internal
        self._seen: set[str] = set()

    def mark_seen(self, token: str) -> None:
        self._seen.add(token)

    def arm(self, handoff: IncompleteTurnHandoff) -> bool:
        if handoff.chain_depth >= MAX_RESUME_DEPTH:
            return False
        if (
            handoff.resume_token == handoff.prior_token
            or handoff.resume_token in self._seen
        ):
            return False
        if not handoff.termination_reason.startswith("max_iterations_reached"):
            return False
        if not self._store.persist(handoff):
            return False
        fired = False

        def after_delivery(delivered_token: str) -> None:
            nonlocal fired
            if fired or delivered_token != handoff.resume_token:
                return
            fired = True
            if not self._store.mark_dispatched(handoff.resume_token):
                return
            self._seen.add(handoff.resume_token)
            self._schedule(handoff.resume_token, build_resume_prompt(handoff.to_dict()))

        try:
            self._register(after_delivery)
        except Exception:
            return False
        return True

    def restore(self, token: str) -> IncompleteTurnHandoff | None:
        payload = self._store.load(token)
        return IncompleteTurnHandoff.from_dict(payload) if payload else None
