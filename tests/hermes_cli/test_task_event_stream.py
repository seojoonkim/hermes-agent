"""Focused tests for the Stage 1 bounded task event stream.

Measurement-only: this stream records task/phase/attempt *facts* for future
retrospective consumers. It must never carry user content, never touch disk or
network, and never change caller semantics.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.observability import task_event_stream as tes
from hermes_cli.observability.task_event_stream import (
    ALLOWED_METADATA_KEYS,
    MAX_METADATA_ENTRIES,
    MAX_METADATA_VALUE_LENGTH,
    SCHEMA_VERSION,
    MetadataRejected,
    TaskEvent,
    TaskEventKind,
    TaskEventStream,
    TERMINAL_KIND_BY_OUTCOME,
    TERMINAL_TASK_KINDS,
)


@pytest.fixture()
def stream():
    return TaskEventStream(capacity=4)


# ---------------------------------------------------------------------------
# Contract 1/2 — typed immutable event, closed kinds
# ---------------------------------------------------------------------------


def test_event_carries_full_typed_contract(stream):
    event = stream.append(
        TaskEventKind.TASK_START,
        task_id="task-1",
        session_id="session-1",
        metadata={"platform": "cli"},
    )

    assert isinstance(event, TaskEvent)
    assert event.schema_version == SCHEMA_VERSION
    assert event.sequence == 1
    assert event.kind is TaskEventKind.TASK_START
    assert event.task_id == "task-1"
    assert event.session_id == "session-1"
    assert isinstance(event.monotonic_ns, int)
    assert isinstance(event.wall_time, float)
    assert dict(event.metadata) == {"platform": "cli"}


def test_event_is_immutable(stream):
    event = stream.append(TaskEventKind.TASK_START, task_id="t", session_id="s")

    with pytest.raises(Exception):
        event.task_id = "other"
    with pytest.raises(TypeError):
        event.metadata["platform"] = "cli"


def test_event_kinds_are_closed_and_cover_lifecycles():
    assert {k.value for k in TaskEventKind} == {
        "task_start",
        "task_success",
        "task_failure",
        "task_cancelled",
        "task_timeout",
        "phase_start",
        "phase_end",
        "attempt_start",
        "attempt_end",
    }
    assert TERMINAL_TASK_KINDS == frozenset({
        TaskEventKind.TASK_SUCCESS,
        TaskEventKind.TASK_FAILURE,
        TaskEventKind.TASK_CANCELLED,
        TaskEventKind.TASK_TIMEOUT,
    })
    assert TERMINAL_KIND_BY_OUTCOME == {
        "success": TaskEventKind.TASK_SUCCESS,
        "failed": TaskEventKind.TASK_FAILURE,
        "cancelled": TaskEventKind.TASK_CANCELLED,
        "timed_out": TaskEventKind.TASK_TIMEOUT,
    }


def test_append_rejects_unknown_kind(stream):
    with pytest.raises(ValueError):
        stream.append("task_exploded", task_id="t", session_id="s")


# ---------------------------------------------------------------------------
# Contract 3 — bounded ring buffer, ordering, eviction, reset
# ---------------------------------------------------------------------------


def test_sequence_is_monotonic_and_snapshot_is_oldest_first(stream):
    for i in range(3):
        stream.append(TaskEventKind.PHASE_START, task_id=f"t{i}", session_id="s")

    snapshot = stream.snapshot()
    assert isinstance(snapshot, tuple)
    assert [e.sequence for e in snapshot] == [1, 2, 3]
    assert [e.task_id for e in snapshot] == ["t0", "t1", "t2"]
    assert [e.monotonic_ns for e in snapshot] == sorted(e.monotonic_ns for e in snapshot)


def test_capacity_evicts_oldest_entries(stream):
    for i in range(6):
        stream.append(TaskEventKind.ATTEMPT_START, task_id=f"t{i}", session_id="s")

    snapshot = stream.snapshot()
    assert stream.capacity == 4
    assert len(snapshot) == 4
    assert len(stream) == 4
    # Oldest two evicted; sequence numbers keep counting past eviction.
    assert [e.task_id for e in snapshot] == ["t2", "t3", "t4", "t5"]
    assert [e.sequence for e in snapshot] == [3, 4, 5, 6]


def test_snapshot_is_a_detached_copy(stream):
    stream.append(TaskEventKind.TASK_START, task_id="t", session_id="s")
    snapshot = stream.snapshot()
    stream.append(TaskEventKind.TASK_SUCCESS, task_id="t", session_id="s")

    assert len(snapshot) == 1


def test_clear_drops_events_but_keeps_sequence_monotonic(stream):
    stream.append(TaskEventKind.TASK_START, task_id="t", session_id="s")
    stream.clear()

    assert stream.snapshot() == ()
    event = stream.append(TaskEventKind.TASK_SUCCESS, task_id="t", session_id="s")
    assert event.sequence == 2


def test_reset_restarts_the_sequence_for_tests(stream):
    stream.append(TaskEventKind.TASK_START, task_id="t", session_id="s")
    stream.reset()

    assert stream.snapshot() == ()
    event = stream.append(TaskEventKind.TASK_START, task_id="t", session_id="s")
    assert event.sequence == 1


def test_capacity_must_be_a_positive_int():
    with pytest.raises(ValueError):
        TaskEventStream(capacity=0)
    with pytest.raises(ValueError):
        TaskEventStream(capacity=-1)


def test_concurrent_appends_are_thread_safe_and_sequences_unique():
    stream = TaskEventStream(capacity=512)

    def _worker(n: int) -> None:
        for _ in range(50):
            stream.append(TaskEventKind.ATTEMPT_END, task_id=f"t{n}", session_id="s")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(8)))

    snapshot = stream.snapshot()
    assert len(snapshot) == 400
    assert len({e.sequence for e in snapshot}) == 400
    assert [e.sequence for e in snapshot] == sorted(e.sequence for e in snapshot)


# ---------------------------------------------------------------------------
# Contract 4 — privacy boundary
# ---------------------------------------------------------------------------


def test_metadata_allowlist_excludes_all_content_bearing_fields():
    forbidden = {
        "message",
        "user_message",
        "prompt",
        "response",
        "final_response",
        "tool_result",
        "tool_output",
        "content",
        "tokens",
        "total_tokens",
        "prompt_tokens",
        "path",
        "file_path",
        "cwd",
    }
    assert ALLOWED_METADATA_KEYS.isdisjoint(forbidden)


@pytest.mark.parametrize(
    "metadata",
    [
        {"prompt": "secret user text"},
        {"final_response": "model output"},
        {"tool_result": "grep output"},
        {"file_path": "/Users/someone/secret.py"},
        {"total_tokens": 42},
    ],
)
def test_content_bearing_metadata_is_rejected(stream, metadata):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.TASK_START,
            task_id="t",
            session_id="s",
            metadata=metadata,
        )
    assert stream.snapshot() == ()


@pytest.mark.parametrize(
    "value",
    [
        {"nested": "map"},
        ["nested", "list"],
        ("nested", "tuple"),
        b"raw bytes",
        object(),
    ],
)
def test_non_scalar_metadata_values_are_rejected(stream, value):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.PHASE_START,
            task_id="t",
            session_id="s",
            metadata={"phase": value},
        )


@pytest.mark.parametrize("metadata", [[], 0, ""])
def test_falsy_non_mapping_metadata_is_rejected(stream, metadata):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.TASK_START,
            task_id="t",
            session_id="s",
            metadata=metadata,
        )


def test_oversized_metadata_value_is_rejected(stream):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.PHASE_START,
            task_id="t",
            session_id="s",
            metadata={"phase": "x" * (MAX_METADATA_VALUE_LENGTH + 1)},
        )


def test_too_many_metadata_entries_rejected(stream):
    metadata = dict.fromkeys(sorted(ALLOWED_METADATA_KEYS), "v")
    assert len(metadata) > MAX_METADATA_ENTRIES
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.PHASE_START,
            task_id="t",
            session_id="s",
            metadata=metadata,
        )


def test_oversized_identifiers_are_rejected(stream):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.TASK_START,
            task_id="t" * 1024,
            session_id="s",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_id", "/Users/someone/private.txt"),
        ("task_id", "raw secret prompt"),
        ("session_id", "session\nforged"),
    ],
)
def test_identifiers_reject_content_and_absolute_paths(stream, field, value):
    kwargs = {"task_id": "t", "session_id": "s"}
    kwargs[field] = value
    with pytest.raises(MetadataRejected):
        stream.append(TaskEventKind.TASK_START, **kwargs)


@pytest.mark.parametrize(
    "value",
    ["raw secret prompt", "/Users/someone/private.txt", "line\nforged"],
)
def test_allowlisted_string_values_reject_content_shaped_text(stream, value):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.PHASE_START,
            task_id="t",
            session_id="s",
            metadata={"reason": value},
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numeric_metadata_is_rejected(stream, value):
    assert not math.isfinite(value)
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.ATTEMPT_END,
            task_id="t",
            session_id="s",
            metadata={"duration_bucket": value},
        )


def test_oversized_numeric_metadata_is_rejected(stream):
    with pytest.raises(MetadataRejected):
        stream.append(
            TaskEventKind.ATTEMPT_END,
            task_id="t",
            session_id="s",
            metadata={"attempt": 10**30},
        )


def test_allowed_scalar_metadata_round_trips(stream):
    event = stream.append(
        TaskEventKind.ATTEMPT_END,
        task_id="t",
        session_id="s",
        metadata={"attempt": 3, "outcome": "failed", "retried": True},
    )
    assert dict(event.metadata) == {
        "attempt": 3,
        "outcome": "failed",
        "retried": True,
    }


# ---------------------------------------------------------------------------
# Contract 5 — fail-open facade
# ---------------------------------------------------------------------------


def test_facade_swallows_failures_and_debug_logs(monkeypatch, caplog):
    boom = RuntimeError("ring buffer exploded")

    class _Broken:
        def append(self, *_args, **_kwargs):
            raise boom

    monkeypatch.setattr(tes, "_STREAM", _Broken())
    with caplog.at_level(logging.DEBUG, logger=tes.__name__):
        assert tes.emit_task_start(task_id="t", session_id="s") is False
        assert tes.emit_task_terminal("success", task_id="t", session_id="s") is False

    assert caplog.records
    assert all(r.levelno == logging.DEBUG for r in caplog.records)


@pytest.mark.parametrize("fatal", [KeyboardInterrupt(), SystemExit(2)])
def test_emit_does_not_swallow_process_control_exceptions(monkeypatch, fatal):
    class _Broken:
        def append(self, *_args, **_kwargs):
            raise fatal

    monkeypatch.setattr(tes, "_STREAM", _Broken())

    with pytest.raises(type(fatal)):
        tes.emit_task_start(task_id="t", session_id="s")


@pytest.mark.parametrize("fatal", [KeyboardInterrupt(), SystemExit(2)])
def test_safe_debug_does_not_swallow_process_control_exceptions(monkeypatch, fatal):
    monkeypatch.setattr(
        tes.logger,
        "debug",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(fatal),
    )

    with pytest.raises(type(fatal)):
        tes._safe_debug("diagnostic")


def test_facade_swallows_rejected_metadata(monkeypatch):
    stream = TaskEventStream(capacity=4)
    monkeypatch.setattr(tes, "_STREAM", stream)

    assert tes.emit_task_start(task_id="t", session_id="s", prompt="secret") is False
    assert stream.snapshot() == ()


def test_facade_swallows_unknown_terminal_outcome(monkeypatch):
    stream = TaskEventStream(capacity=4)
    monkeypatch.setattr(tes, "_STREAM", stream)

    assert tes.emit_task_terminal("weird", task_id="t", session_id="s") is False
    assert stream.snapshot() == ()


def test_facade_appends_to_the_process_local_stream(monkeypatch):
    stream = TaskEventStream(capacity=4)
    monkeypatch.setattr(tes, "_STREAM", stream)

    assert tes.emit_task_start(task_id="t", session_id="s", platform="cli") is True
    assert tes.emit_task_terminal("timed_out", task_id="t", session_id="s") is True

    kinds = [e.kind for e in tes.get_task_event_stream().snapshot()]
    assert kinds == [TaskEventKind.TASK_START, TaskEventKind.TASK_TIMEOUT]


# ---------------------------------------------------------------------------
# Contract 7 — no I/O on append
# ---------------------------------------------------------------------------


def test_append_performs_no_disk_or_network_io(stream, monkeypatch):
    import builtins
    import socket

    def _no_open(*_args, **_kwargs):
        raise AssertionError("append must not touch disk")

    def _no_socket(*_args, **_kwargs):
        raise AssertionError("append must not touch the network")

    monkeypatch.setattr(builtins, "open", _no_open)
    monkeypatch.setattr(socket, "socket", _no_socket)

    stream.append(TaskEventKind.TASK_START, task_id="t", session_id="s")
    assert len(stream.snapshot()) == 1
