"""Stage 1 integration: AIAgent.run_conversation emits bounded task events.

The new event stream is measurement-only and fail-open — it must never change
the conversation result/error semantics, and it must not disturb the existing
Relay / shared-metrics task lifecycle.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.observability import task_event_stream as tes
from hermes_cli.observability.task_event_stream import TaskEventKind
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


@pytest.fixture()
def events(monkeypatch):
    stream = tes.TaskEventStream(capacity=32)
    monkeypatch.setattr(tes, "_STREAM", stream)
    return stream


def _result(**overrides):
    base = {
        "final_response": "ok",
        "messages": [],
        "completed": True,
        "failed": False,
        "interrupted": False,
    }
    base.update(overrides)
    return base


def _patched_turn(loop_return=None, loop_side_effect=None):
    return (
        patch(
            "agent.conversation_loop.run_conversation",
            return_value=loop_return,
            side_effect=loop_side_effect,
        ),
        patch("hermes_cli.observability.relay_shared_metrics.start_task_run"),
        patch("hermes_cli.observability.relay_shared_metrics.finish_task_run"),
    )


def _kinds(stream):
    return [e.kind for e in stream.snapshot()]


def test_successful_turn_emits_start_and_success(agent, events):
    loop, start, finish = _patched_turn(loop_return=_result())
    with loop, start as start_task_run, finish as finish_task_run:
        result = agent.run_conversation("hello", task_id="task-1")

    assert result["completed"] is True
    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_SUCCESS]
    snapshot = events.snapshot()
    assert all(e.task_id == "task-1" for e in snapshot)
    assert all(e.session_id == str(agent.session_id or "") for e in snapshot)
    # Existing shared-metrics lifecycle is untouched.
    start_task_run.assert_called_once()
    finish_task_run.assert_called_once()


def test_failed_result_emits_failure_terminal(agent, events):
    loop, start, finish = _patched_turn(loop_return=_result(completed=False, failed=True))
    with loop, start, finish:
        agent.run_conversation("hello", task_id="task-2")

    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_FAILURE]


def test_interrupted_result_emits_cancelled_terminal(agent, events):
    loop, start, finish = _patched_turn(
        loop_return=_result(completed=False, interrupted=True)
    )
    with loop, start, finish:
        agent.run_conversation("hello", task_id="task-3")

    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_CANCELLED]


def test_keyboard_interrupt_emits_cancelled_terminal(agent, events):
    loop, start, finish = _patched_turn(loop_side_effect=KeyboardInterrupt())
    with loop, start, finish:
        with pytest.raises(KeyboardInterrupt):
            agent.run_conversation("hello", task_id="task-4")

    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_CANCELLED]


def test_timeout_error_emits_timeout_terminal(agent, events):
    loop, start, finish = _patched_turn(loop_side_effect=TimeoutError("slow"))
    with loop, start, finish:
        with pytest.raises(TimeoutError):
            agent.run_conversation("hello", task_id="task-5")

    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_TIMEOUT]


def test_raised_exception_emits_failure_terminal(agent, events):
    loop, start, finish = _patched_turn(loop_side_effect=RuntimeError("provider down"))
    with loop, start, finish:
        with pytest.raises(RuntimeError):
            agent.run_conversation("hello", task_id="task-6")

    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_FAILURE]


def test_collector_failure_cannot_break_the_conversation(agent, monkeypatch):
    class _Broken:
        def append(self, *_args, **_kwargs):
            raise RuntimeError("collector exploded")

    monkeypatch.setattr(tes, "_STREAM", _Broken())
    expected = _result()
    loop, start, finish = _patched_turn(loop_return=expected)
    with loop, start as start_task_run, finish as finish_task_run:
        result = agent.run_conversation("hello", task_id="task-7")

    assert result is expected
    start_task_run.assert_called_once()
    finish_task_run.assert_called_once()


def test_failed_start_does_not_create_an_orphan_terminal(agent, events, monkeypatch):
    original_append = events.append
    calls = 0

    def _fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient collector failure")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(events, "append", _fail_first)
    loop, start, finish = _patched_turn(loop_return=_result())
    with loop, start, finish:
        result = agent.run_conversation("hello", task_id="task-transient")

    assert result["completed"] is True
    assert events.snapshot() == ()


def test_shared_metrics_finalization_failure_is_measurement_failure_only(agent, events):
    loop, start, finish = _patched_turn(loop_return=_result())
    with (
        loop,
        start,
        finish as finish_task_run,
        patch(
            "agent.relay_runtime.SESSION_COORDINATOR.finish_logical_calls"
        ) as finish_logical_calls,
    ):
        finish_task_run.side_effect = RuntimeError("metrics finalization failed")
        with pytest.raises(RuntimeError, match="metrics finalization failed"):
            agent.run_conversation("hello", task_id="task-finalize")

    assert _kinds(events) == [TaskEventKind.TASK_START, TaskEventKind.TASK_FAILURE]
    finish_task_run.assert_called_once()
    assert {call.kwargs["outcome"] for call in finish_logical_calls.call_args_list} == {
        "success"
    }


def test_no_events_when_shared_metrics_start_fails(agent, events):
    relay_lease = SimpleNamespace(
        parent_session_id="",
        profile_key="/profile",
        session_id=agent.session_id or "",
    )
    coordinator = MagicMock()
    coordinator.acquire_conversation.return_value = relay_lease
    coordinator.begin_turn.return_value = object()

    with (
        patch("agent.relay_runtime.SESSION_COORDINATOR", coordinator),
        patch("agent.relay_runtime.current_profile_key", return_value="/profile"),
        patch(
            "hermes_cli.observability.relay_shared_metrics.start_task_run",
            side_effect=RuntimeError("task metrics start failed"),
        ),
        patch("hermes_cli.observability.relay_shared_metrics.finish_task_run"),
        patch("agent.conversation_loop.run_conversation"),
    ):
        with pytest.raises(RuntimeError):
            agent.run_conversation("hello", task_id="task-8")

    assert events.snapshot() == ()


def test_events_never_carry_user_content(agent, events):
    loop, start, finish = _patched_turn(loop_return=_result(final_response="model said"))
    with loop, start, finish:
        agent.run_conversation("very secret prompt", task_id="task-9")

    for event in events.snapshot():
        blob = repr(dict(event.metadata))
        assert "secret" not in blob
        assert "model said" not in blob
