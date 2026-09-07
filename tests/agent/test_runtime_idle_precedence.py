"""Opt-in inactivity policy must reach actual request watchdogs."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as helpers


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {
        "agent": {"stream_idle_timeout_seconds": 120, "nonstream_stale_timeout_seconds": 180}
    })


def test_stream_policy_bypasses_context_and_reasoning_floors():
    agent = SimpleNamespace(provider="custom", model="claude-opus-4-6")
    assert helpers._derive_stream_stale_timeout(agent, {
        "model": agent.model, "messages": [{"role": "user", "content": "x" * 500_000}]
    }) == 120


def test_inline_policy_reaches_actual_transport_and_timer(monkeypatch):
    timers = []
    class Timer:
        def __init__(self, interval, callback):
            self.interval = interval
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            pass
    monkeypatch.setattr(helpers.threading, "Timer", Timer)
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.provider = "custom"
    agent.model = "m"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._compute_non_stream_stale_timeout.return_value = 900
    client = agent._create_request_openai_client.return_value
    client.chat.completions.create.return_value = SimpleNamespace(id="ok")
    assert helpers.direct_api_call(agent, {"model": "m", "messages": []}).id == "ok"
    assert timers[0].interval == 180
    assert client.chat.completions.create.call_args.kwargs["timeout"].read == 180


@pytest.mark.parametrize("streaming,threshold", [(True, 120), (False, 180)])
def test_actual_worker_watchdog_uses_explicit_budget(monkeypatch, streaming, threshold):
    clock = [0.0]
    class BudgetReached(Exception):
        pass
    class PendingWorker:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        def is_alive(self):
            return True
        def join(self, timeout):
            clock[0] += threshold + 0.1
            assert clock[0] < threshold * 2, "explicit deadline did not fire"
    monkeypatch.setattr(helpers.threading, "Thread", PendingWorker)
    monkeypatch.setattr(helpers.time, "time", lambda: clock[0])
    monkeypatch.setattr(helpers.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda agent: False)
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.provider = "custom"
    agent.model = "m"
    agent.base_url = "http://localhost:8000/v1"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._compute_non_stream_stale_timeout.return_value = 900
    agent._buffer_status.side_effect = BudgetReached
    if not streaming:
        def reached(*args, **kwargs):
            raise BudgetReached
        monkeypatch.setattr(helpers, "_report_stale_nonstream_kill", reached)
    call = helpers.interruptible_streaming_api_call if streaming else helpers.interruptible_api_call
    with pytest.raises(BudgetReached):
        call(agent, {"model": "m", "messages": []})
    assert clock[0] == threshold + 0.1


def test_missing_policy_preserves_legacy_resolver(monkeypatch):
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: {})
    agent = SimpleNamespace(_compute_non_stream_stale_timeout=lambda payload: 900)
    assert helpers._resolve_direct_stale_timeout(agent, {}) == 900
    agent.provider, agent.model = "custom", "claude-opus-4-6"
    assert helpers._derive_stream_stale_timeout(agent, {"model": agent.model}) > 120
