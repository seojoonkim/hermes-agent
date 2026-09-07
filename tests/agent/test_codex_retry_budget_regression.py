"""Physical Codex retries must receive a fresh semantic-inactivity budget.

Reuse the clock regression's socket-denying, queue-synchronized real worker;
only the provider transport and clocks are fake, not Codex or Relay execution.
"""
from unittest.mock import MagicMock

import httpx

from agent import chat_completion_helpers as h
from tests.agent.test_codex_idle_clock_regression import runtime, terminal_event


def test_physical_connect_retry_gets_fresh_monotonic_budget(runtime):
    x = runtime
    stream = MagicMock()
    attempts = []

    def events():
        # No progress event arrives before the original deadline (110).
        # A watchdog poll at 118 must permit this still-live second attempt:
        # its reset at 109 gives it until 119, unlike the original request.
        x.pause(118)
        yield terminal_event()

    def create(**kwargs):
        control = x.agent._run_codex_stream.call_args.kwargs["request_control"]
        attempts.append((x.clock[0], control["progress"]["t"]))
        assert kwargs["stream"] is True
        if len(attempts) == 1:
            x.pause(100)
            x.pause(109)
            raise httpx.ConnectError("first physical connection failed near budget")
        assert len(attempts) == 2, "unexpected extra physical connection"
        x.pause(109)
        return stream

    stream.__iter__.side_effect = events
    x.client.responses.create.side_effect = create
    response = h.interruptible_api_call(x.agent, {"model": "m", "input": []})

    assert response is not None
    assert response.id == "r"
    assert attempts == [(100, 100), (109, 109)]
    assert x.clock[0] == 118
    assert x.client.responses.create.call_count == 2
    # One logical invocation, with the retry inside the real native runtime.
    assert x.agent._run_codex_stream.call_count == 1
    x.agent._fire_stream_delta.assert_not_called()
    x.agent._abort_request_openai_client.assert_not_called()
    assert x.agent._consecutive_stale_streams == 0
    assert not x.agent._interrupt_requested
    stream.close.assert_called()


def test_unset_idle_policy_allows_slow_success_without_semantic_abort(runtime, monkeypatch):
    x = runtime
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: None)
    stream = MagicMock()

    def events():
        # This exceeds the fixture's configured 10-second semantic budget,
        # but remains well below the legacy provider watchdog (1000 seconds).
        x.pause(100)
        yield terminal_event()

    stream.__iter__.side_effect = events
    x.client.responses.create.return_value = stream
    response = h.interruptible_api_call(x.agent, {"model": "m", "input": []})

    assert response is not None
    assert response.id == "r"
    # The contract is behavioral: unset means no semantic inactivity timeout.
    control = x.agent._run_codex_stream.call_args.kwargs["request_control"]
    assert control["semantic"] is False
    assert x.client.responses.create.call_count == 1
    x.agent._abort_request_openai_client.assert_not_called()
    stream.close.assert_called()
