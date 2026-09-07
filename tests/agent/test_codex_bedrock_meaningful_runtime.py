"""Real worker/Relay paths against socket-denied native stream fakes."""
import socket
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from agent import chat_completion_helpers as h
from agent.codex_runtime import run_codex_stream


@pytest.fixture
def harness(monkeypatch):
    def denied(*a, **kw):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(h, "should_use_direct_api_call", lambda a: False)
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: 10)
    monkeypatch.setattr(h, "get_provider_stale_timeout", lambda *a: 1000)
    clock = [0.0]
    monkeypatch.setattr(h, "time", NS(time=lambda: clock[0], monotonic=lambda: clock[0]))
    ready, permit = threading.Event(), threading.Event()
    real_thread = threading.Thread
    workers = []
    class Worker:
        def __init__(self, **kw):
            self.thread = real_thread(**kw)
            workers.append(self)
        def start(self):
            self.thread.start()
            assert ready.wait(2)
        def is_alive(self): return self.thread.is_alive()
        def join(self, timeout=None):
            if self.thread.is_alive():
                ready.clear()
                permit.set()
                assert ready.wait(2)
                self.thread.join(.001)
    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h, "_join_worker_for_relay_teardown", lambda *a, **kw: None)
    agent = MagicMock()
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._fallback_index = 0
    agent._compute_non_stream_stale_timeout.return_value = 1000
    agent._stream_diag_init.return_value = {}
    agent._base_url_lower = "https://example.invalid"
    agent.provider, agent.model = "test", "m"
    agent._interruptible_api_call.side_effect = lambda kw: h.interruptible_api_call(agent, kw)
    agent._run_codex_stream.side_effect = lambda kw, **k: run_codex_stream(agent, kw, **k)
    def events(values):
        try:
            for at, value in values:
                ready.set()
                assert permit.wait(2)
                permit.clear()
                clock[0] = at
                yield value
        finally:
            ready.set()
    yield NS(agent=agent, clock=clock, events=events, permit=permit, ready=ready)
    permit.set()
    for worker in workers:
        worker.thread.join(2)
        assert not worker.thread.is_alive()


def test_codex_heartbeats_do_not_extend_idle(harness):
    x = harness
    x.agent.api_mode = "codex_responses"
    stream = MagicMock()
    stream.__iter__.side_effect = lambda: x.events([
        (9, {"type": "response.created"}),
        (10, {"type": "response.in_progress"}),
        (20, {"type": "response.completed", "response": {"id": "r", "output": [], "status": "completed"}}),
    ])
    x.agent._create_request_openai_client.return_value.responses.create.return_value = stream
    with pytest.raises((TimeoutError, InterruptedError)):
        h.interruptible_api_call(x.agent, {"model": "m", "input": []})
    x.agent._abort_request_openai_client.assert_called()
    x.agent._fire_stream_delta.assert_not_called()
    stream.close.assert_called()


def test_bedrock_metadata_does_not_extend_idle(harness, monkeypatch):
    from agent import bedrock_adapter as b
    x = harness
    x.agent.api_mode = "bedrock_converse"
    client = MagicMock()
    client.converse_stream.return_value = {"stream": x.events([
        (9, {"metadata": {"usage": {}}}),
        (10, {"metadata": {"usage": {}}}),
        (20, {"messageStop": {"stopReason": "end_turn"}}),
    ])}
    monkeypatch.setattr(b, "_get_bedrock_runtime_client", lambda region: client)
    monkeypatch.setattr(b, "invalidate_runtime_client", lambda region: None)
    with pytest.raises((TimeoutError, InterruptedError)):
        h.interruptible_streaming_api_call(x.agent, {"modelId": "m", "messages": []})
    x.agent._fire_stream_delta.assert_not_called()
    assert x.agent._consecutive_stale_streams == 1
    x.permit.set()
    x.ready.wait(2)


@pytest.mark.parametrize("mode", ["codex", "bedrock"])
def test_productive_stream_outlives_idle_budget(harness, monkeypatch, mode):
    x = harness
    if mode == "codex":
        x.agent.api_mode = "codex_responses"
        values = [(at, {"type": "response.output_text.delta", "delta": "x"}) for at in [9, 18, 27]]
        values.append((28, {"type": "response.completed", "response": {"id": "r", "output": [], "status": "completed"}}))
        stream = MagicMock()
        stream.__iter__.side_effect = lambda: x.events(values)
        x.agent._create_request_openai_client.return_value.responses.create.return_value = stream
        response = h.interruptible_api_call(x.agent, {"model": "m", "input": []})
    else:
        from agent import bedrock_adapter as b
        x.agent.api_mode = "bedrock_converse"
        values = [(at, {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "x"}}}) for at in [9, 18, 27]]
        values.append((28, {"messageStop": {"stopReason": "end_turn"}}))
        client = MagicMock()
        client.converse_stream.return_value = {"stream": x.events(values)}
        monkeypatch.setattr(b, "_get_bedrock_runtime_client", lambda region: client)
        response = h.interruptible_streaming_api_call(x.agent, {"modelId": "m", "messages": []})
    assert response is not None
    assert [c.args[0] for c in x.agent._fire_stream_delta.call_args_list] == ["x"] * 3
    x.agent._abort_request_openai_client.assert_not_called()
    assert x.agent._consecutive_stale_streams == 0
