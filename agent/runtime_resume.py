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
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

MAX_RESUME_DEPTH = 2
RESUME_TURN_PREFIX = "[Hermes durable resume handoff — distinct internal turn]"
MAX_GOAL_CHARS = 1000
MAX_FAILING_TESTS = 10
MAX_TEST_ID_CHARS = 240
_MAX_SCOPE_CHARS = 300
_COMMIT_RE = re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE)
_TEST_FAILURE_RE = re.compile(r"(?m)^FAILED\s+([^\s]+::[^\s]+)")
_TOKEN_RE = re.compile(r"\A[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class RuntimeResumeScope:
    """Structured, delimiter-safe route identity for durable resume dispatch."""

    profile: str
    platform: str
    account_id: str
    chat_id: str
    thread_id: str
    chat_type: str
    scope_id: str
    user_id: str
    session_id: str

    def to_dict(self) -> dict[str, str]:
        return {key: str(value or "")[:_MAX_SCOPE_CHARS] for key, value in asdict(self).items()}

    def encode(self) -> str:
        return "json:" + json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def decode(cls, encoded: str) -> RuntimeResumeScope | None:
        if not isinstance(encoded, str) or not encoded.startswith("json:"):
            return None
        try:
            raw = json.loads(encoded[5:])
            if not isinstance(raw, Mapping):
                return None
            values = {
                key: str(raw.get(key) or "")[:_MAX_SCOPE_CHARS]
                for key in (
                    "profile",
                    "platform",
                    "account_id",
                    "chat_id",
                    "thread_id",
                    "chat_type",
                    "scope_id",
                    "user_id",
                    "session_id",
                )
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not values["profile"] or not values["platform"] or not values["chat_id"] or not values["session_id"]:
            return None
        return cls(**values)


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

    @property
    def goal(self) -> str:
        return self.incomplete_goal

    @property
    def failing_test_ids(self) -> tuple[str, ...]:
        return self.failing_tests

    @property
    def verified_sha(self) -> str:
        return self.last_verified_commit

    def to_payload(self) -> dict[str, Any]:
        return {
            "resume_token": self.resume_token,
            "prior_token": self.prior_token,
            "profile": self.profile,
            "session_id": self.session_id,
            "channel_key": self.channel_key,
            "termination_reason": self.termination_reason,
            "goal": self.goal,
            "failing_test_ids": list(self.failing_test_ids),
            "verified_sha": self.verified_sha,
            "chain_depth": self.chain_depth,
            "next_dispatch_intent": self.next_dispatch_intent,
        }

    def to_turn_text(self) -> str:
        return build_resume_prompt(self.to_dict())

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
        "chain_depth": max(0, int(handoff.get("chain_depth") or 0)),
    }
    return (
        f"{RESUME_TURN_PREFIX}\n"
        "Verify current state, then choose a smaller bounded step that produces a "
        "verifiable artifact. Address listed failing tests and finish the goal or "
        "report a concrete blocker; do not repeat the same approach or expand scope.\n"
        + json.dumps(bounded, ensure_ascii=False, sort_keys=True)
    )


def parse_resume_prompt(text: Any) -> dict[str, Any] | None:
    """Return the bounded internal-resume payload, or ``None``."""
    if not isinstance(text, str) or not text.startswith(RESUME_TURN_PREFIX):
        return None
    try:
        raw = json.loads(text.rsplit("\n", 1)[-1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping) or not _TOKEN_RE.fullmatch(
        str(raw.get("resume_token") or "")
    ):
        return None
    try:
        depth = int(raw.get("chain_depth") or 0)
    except (TypeError, ValueError):
        return None
    if depth < 0 or depth >= MAX_RESUME_DEPTH:
        return None
    return {
        "resume_token": str(raw["resume_token"]),
        "chain_depth": depth,
        "incomplete_goal": str(raw.get("incomplete_goal") or "")[:MAX_GOAL_CHARS],
    }


class MemKraftResumeStore:
    """Atomic task-file persistence compatible with a MemKraft data root."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._directory = Path(root) / "tasks" / "hermes-resume"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    @contextmanager
    def _mutation_lock(self, token: str):
        """Serialize checkpoint mutations across threads and processes."""
        lock_path = self._directory / f".{token}.lock"
        with self._thread_lock:
            handle = lock_path.open("a+b")
            try:
                if os.name == "posix":
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if os.name == "posix":
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                handle.close()

    def _path(self, token: str) -> Path:
        if not _TOKEN_RE.fullmatch(token):
            raise ValueError("invalid resume token")
        return self._directory / f"{token}.json"

    def _write_document(self, path: Path, document: Mapping[str, Any]) -> bool:
        encoded = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        self._directory.mkdir(parents=True, exist_ok=True)
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

    def persist(self, handoff: IncompleteTurnHandoff) -> bool:
        """Persist once by deterministic token; equal retries are idempotent."""
        path = self._path(handoff.resume_token)
        document = {"status": "incomplete", "handoff": handoff.to_dict()}
        self._directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")) == document
            except (OSError, json.JSONDecodeError):
                return False
        return self._write_document(path, document)

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

    def reserve_dispatch(self, token: str, *, lease_id: str) -> dict[str, Any] | None:
        """Atomically claim an incomplete checkpoint for one dispatcher."""
        with self._mutation_lock(token):
            try:
                path = self._path(token)
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return None
            if document.get("status") != "incomplete":
                return None
            handoff = IncompleteTurnHandoff.from_dict(document.get("handoff") or {})
            if handoff is None:
                return None
            document["status"] = "dispatching"
            document["lease_id"] = str(lease_id or "")[:128]
            if not document["lease_id"]:
                return None
            return handoff.to_dict() if self._write_document(path, document) else None

    def release_dispatch(self, token: str, *, lease_id: str) -> bool:
        """Release a claimed checkpoint if dispatch acceptance failed."""
        with self._mutation_lock(token):
            try:
                path = self._path(token)
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return False
            if document.get("status") != "dispatching" or document.get("lease_id") != lease_id:
                return False
            document.pop("lease_id", None)
            document["status"] = "incomplete"
            return self._write_document(path, document)

    def mark_dispatched(self, token: str, *, lease_id: str | None = None) -> bool:
        """Durably record dispatch, preventing duplicate restart delivery."""
        with self._mutation_lock(token):
            try:
                path = self._path(token)
                document = json.loads(path.read_text(encoding="utf-8"))
                if document.get("status") == "dispatched":
                    return False
                if lease_id is None:
                    if document.get("status") != "incomplete":
                        return False
                else:
                    if document.get("status") != "dispatching" or document.get("lease_id") != lease_id:
                        return False
                    document.pop("lease_id", None)
                document["status"] = "dispatched"
                return self._write_document(path, document)
            except (OSError, ValueError, json.JSONDecodeError):
                return False

    def reclaim_interrupted_dispatches(self) -> int:
        """Return leases left by a dead gateway process to retryable state."""
        reclaimed = 0
        for path in self._directory.glob("*.json"):
            token = path.stem
            with self._mutation_lock(token):
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                if document.get("status") != "dispatching":
                    continue
                document.pop("lease_id", None)
                document["status"] = "incomplete"
                if self._write_document(path, document):
                    reclaimed += 1
        return reclaimed


class ResumeCoordinator:
    """Persist and arm one token-bound internal turn after user delivery."""

    def __init__(self, *, store=None, register_post_delivery=None,
                 schedule_internal=None, memory_manager=None,
                 checkpoint_store=None, schedule_turn=None, session_key="",
                 profile="", depth=0) -> None:
        """Accept both the durable store seam and the legacy gateway seam."""
        self._store = store or checkpoint_store
        self._manager = memory_manager
        self._register = register_post_delivery
        self._schedule = schedule_internal or schedule_turn
        self._session_key = session_key
        self._profile = profile
        self._depth = int(depth or 0)
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

    def maybe_schedule(self, result: Mapping[str, Any], *, goal="",
                       failing_test_ids=(), verified_sha="") -> bool:
        if not is_resume_eligible(result, depth=self._depth):
            return False
        handoff = build_handoff(session_key=self._session_key,
                                profile=self._profile, goal=goal,
                                failing_test_ids=failing_test_ids,
                                verified_sha=verified_sha, depth=self._depth)
        payload = handoff.to_payload()
        # Legacy multiplex managers receive the notification at checkpoint
        # time. Provider failure is fail-open: durable scheduling still proceeds.
        provider = select_resume_provider(self._manager, self._profile) if self._manager is not None else None
        if provider is not None:
            try:
                provider.on_incomplete_turn(payload)
            except Exception:
                pass
        if hasattr(self._store, "save_resume_checkpoint"):
            try:
                if not self._store.save_resume_checkpoint(payload):
                    return False
            except Exception:
                return False
            fired = False
            def after_delivery(*_args):
                nonlocal fired
                if fired:
                    return
                fired = True
                try:
                    self._schedule(handoff.to_turn_text())
                except Exception:
                    pass
            try:
                self._register(after_delivery)
            except Exception:
                return False
            return True
        return self.arm(handoff)

    def restore(self, token: str) -> IncompleteTurnHandoff | None:
        payload = self._store.load(token)
        return IncompleteTurnHandoff.from_dict(payload) if payload else None


HANDOFF_FIELDS = frozenset({
    "resume_token", "prior_token", "profile", "session_id", "channel_key",
    "termination_reason", "goal", "failing_test_ids", "verified_sha",
    "chain_depth", "next_dispatch_intent",
})


def is_resume_eligible(result: Mapping[str, Any], *, depth: int = 0) -> bool:
    return bool(isinstance(result, Mapping)
        and not result.get("completed") and not result.get("failed")
        and not result.get("interrupted") and int(depth or 0) < MAX_RESUME_DEPTH
        and str(result.get("turn_exit_reason") or "").startswith("max_iterations_reached"))


def build_handoff(*, session_key: str, profile: str, goal: Any,
                  failing_test_ids=(), verified_sha="", depth: int = 0) -> IncompleteTurnHandoff:
    failures = []
    for item in failing_test_ids or ():
        text = str(item).splitlines()[0].strip()
        if "::" in text and len(text) <= MAX_TEST_ID_CHARS and text not in failures:
            failures.append(text)
    commit = str(verified_sha or "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit, re.IGNORECASE):
        commit = ""
    handoff = IncompleteTurnHandoff(
        resume_token="0" * 32, prior_token="", profile=str(profile or "")[:_MAX_SCOPE_CHARS],
        session_id=str(session_key or "")[:_MAX_SCOPE_CHARS],
        channel_key=str(session_key or "")[:_MAX_SCOPE_CHARS],
        termination_reason="max_iterations_reached",
        incomplete_goal=str(goal or "")[:MAX_GOAL_CHARS],
        last_verified_commit=commit,
        failing_tests=tuple(sorted(failures)[:MAX_FAILING_TESTS]),
        chain_depth=max(0, int(depth or 0)),
    )
    return replace(handoff, resume_token=_resume_token(handoff))


def select_resume_provider(memory_manager: Any, profile: str):
    providers = [p for p in list(getattr(memory_manager, "providers", ()) or ())
                 if callable(getattr(p, "on_incomplete_turn", None))]
    scoped = [p for p in providers if getattr(p, "profile", None) == profile]
    if scoped:
        return scoped[0]
    generic = [p for p in providers if not getattr(p, "profile", None)]
    return generic[0] if generic else None


def maybe_schedule_resume(agent: Any, result: Mapping[str, Any]) -> None:
    coordinator = getattr(agent, "resume_coordinator", None)
    if coordinator is not None:
        try:
            coordinator.maybe_schedule(result)
        except Exception:
            pass
