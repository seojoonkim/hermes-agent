"""Bounded, measurement-only task/phase/attempt event stream (Stage 1).

This is deliberately *not* a second TaskRun aggregator: Relay/shared-metrics
still owns task aggregation. This module only records small, typed lifecycle
facts in a process-local ring buffer so future retrospective code has something
to read. It performs no disk or network I/O, keeps steady-state append O(1),
and enforces a hard privacy boundary — no message/prompt/response/tool-result/
token/absolute-path payloads may ever enter an event.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from time import monotonic_ns, time
from types import MappingProxyType
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_CAPACITY = 512

MAX_IDENTIFIER_LENGTH = 128
MAX_METADATA_ENTRIES = 8
MAX_METADATA_VALUE_LENGTH = 64

#: Short allowlist of scalar *operational* metadata. Content-bearing names
#: (prompt/response/tool_result/tokens/paths/...) are absent by construction.
ALLOWED_METADATA_KEYS = frozenset({
    "platform",
    "entrypoint",
    "phase",
    "attempt",
    "outcome",
    "reason",
    "error_type",
    "retried",
    "retry_count",
    "duration_bucket",
})

_SCALAR_TYPES = (str, int, float, bool)
_OPERATIONAL_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]*")
_MAX_NUMERIC_MAGNITUDE = 10**12


class TaskEventKind(str, Enum):
    """Closed set of Stage 1 event kinds."""

    TASK_START = "task_start"
    TASK_SUCCESS = "task_success"
    TASK_FAILURE = "task_failure"
    TASK_CANCELLED = "task_cancelled"
    TASK_TIMEOUT = "task_timeout"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    ATTEMPT_START = "attempt_start"
    ATTEMPT_END = "attempt_end"


TERMINAL_TASK_KINDS = frozenset({
    TaskEventKind.TASK_SUCCESS,
    TaskEventKind.TASK_FAILURE,
    TaskEventKind.TASK_CANCELLED,
    TaskEventKind.TASK_TIMEOUT,
})

#: Maps the outcome vocabulary already used by the Relay turn lifecycle onto
#: terminal event kinds, so the integration never invents a second vocabulary.
TERMINAL_KIND_BY_OUTCOME: dict[str, TaskEventKind] = {
    "success": TaskEventKind.TASK_SUCCESS,
    "failed": TaskEventKind.TASK_FAILURE,
    "cancelled": TaskEventKind.TASK_CANCELLED,
    "timed_out": TaskEventKind.TASK_TIMEOUT,
}

_EMPTY_METADATA: Mapping[str, Any] = MappingProxyType({})


class MetadataRejected(ValueError):
    """Raised when an event field violates the privacy/size boundary."""


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """One immutable lifecycle fact."""

    schema_version: int
    sequence: int
    kind: TaskEventKind
    task_id: str
    session_id: str
    monotonic_ns: int
    wall_time: float
    metadata: Mapping[str, Any]


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise MetadataRejected(f"{name} must be a string, got {type(value).__name__}")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise MetadataRejected(f"{name} exceeds {MAX_IDENTIFIER_LENGTH} characters")
    if value == "" and name == "session_id":
        return value
    if not value or not _OPERATIONAL_TOKEN.fullmatch(value):
        raise MetadataRejected(f"{name} must be an opaque operational token")
    return value


def _validate_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return _EMPTY_METADATA
    if not isinstance(metadata, Mapping):
        raise MetadataRejected("metadata must be a mapping")
    if not metadata:
        return _EMPTY_METADATA
    if len(metadata) > MAX_METADATA_ENTRIES:
        raise MetadataRejected(f"metadata exceeds {MAX_METADATA_ENTRIES} entries")

    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in ALLOWED_METADATA_KEYS:
            raise MetadataRejected(f"metadata key {key!r} is not allowlisted")
        # bool is an int subclass, so it passes the scalar check intentionally.
        if isinstance(value, (bytes, bytearray, memoryview)) or not isinstance(
            value, _SCALAR_TYPES
        ):
            raise MetadataRejected(
                f"metadata value for {key!r} must be a scalar, "
                f"got {type(value).__name__}"
            )
        if isinstance(value, str) and len(value) > MAX_METADATA_VALUE_LENGTH:
            raise MetadataRejected(
                f"metadata value for {key!r} exceeds {MAX_METADATA_VALUE_LENGTH} chars"
            )
        if (
            isinstance(value, str)
            and value
            and not _OPERATIONAL_TOKEN.fullmatch(value)
        ):
            raise MetadataRejected(
                f"metadata value for {key!r} must be an operational token"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise MetadataRejected(f"metadata value for {key!r} must be finite")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and abs(value) > _MAX_NUMERIC_MAGNITUDE
        ):
            raise MetadataRejected(f"metadata value for {key!r} is out of range")
        clean[key] = value
    return MappingProxyType(clean)


class TaskEventStream:
    """Process-local, thread-safe, bounded ring buffer of :class:`TaskEvent`."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._events: deque[TaskEvent] = deque(maxlen=capacity)
        self._sequence = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def append(
        self,
        kind: TaskEventKind,
        *,
        task_id: str,
        session_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskEvent:
        """Record one event. Validation happens before the lock is taken."""
        kind = TaskEventKind(kind)
        task_id = _validate_identifier("task_id", task_id)
        session_id = _validate_identifier("session_id", session_id)
        clean_metadata = _validate_metadata(metadata)

        with self._lock:
            self._sequence += 1
            event = TaskEvent(
                schema_version=SCHEMA_VERSION,
                sequence=self._sequence,
                kind=kind,
                task_id=task_id,
                session_id=session_id,
                monotonic_ns=monotonic_ns(),
                wall_time=time(),
                metadata=clean_metadata,
            )
            self._events.append(event)
        return event

    def snapshot(self) -> tuple[TaskEvent, ...]:
        """Detached oldest-first copy of the buffered events."""
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        """Drop buffered events, keeping the sequence counter monotonic."""
        with self._lock:
            self._events.clear()

    def reset(self) -> None:
        """Drop buffered events *and* restart the sequence (tests only)."""
        with self._lock:
            self._events.clear()
            self._sequence = 0


_STREAM = TaskEventStream()


def get_task_event_stream() -> TaskEventStream:
    """Return the process-local stream."""
    return _STREAM


# ---------------------------------------------------------------------------
# Fail-open facade — the only surface production code should call.
# ---------------------------------------------------------------------------


def emit(
    kind: TaskEventKind,
    *,
    task_id: str,
    session_id: str = "",
    **metadata: Any,
) -> bool:
    """Append one event, swallowing every failure. Returns success as a bool."""
    try:
        _STREAM.append(
            kind,
            task_id=task_id,
            session_id=session_id,
            metadata=metadata or None,
        )
        return True
    except Exception:  # measurement must never change caller semantics
        # KeyboardInterrupt/SystemExit/CancelledError are BaseException and
        # deliberately propagate: process control is not ours to swallow.
        _safe_debug("Task event emit failed (kind=%r)", kind, exc_info=True)
        return False


def _safe_debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Best-effort diagnostics that cannot break the fail-open boundary."""
    try:
        logger.debug(message, *args, **kwargs)
    except Exception:
        pass


def emit_task_start(*, task_id: str, session_id: str = "", **metadata: Any) -> bool:
    return emit(
        TaskEventKind.TASK_START,
        task_id=task_id,
        session_id=session_id,
        **metadata,
    )


def emit_task_terminal(
    outcome: str, *, task_id: str, session_id: str = "", **metadata: Any
) -> bool:
    """Emit the terminal event matching a Relay turn ``outcome``."""
    try:
        kind = TERMINAL_KIND_BY_OUTCOME[outcome]
    except (KeyError, TypeError):
        _safe_debug("Unknown terminal task outcome %r", outcome, exc_info=True)
        return False
    return emit(kind, task_id=task_id, session_id=session_id, **metadata)
