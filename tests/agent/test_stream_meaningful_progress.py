"""Real chat streaming worker/Relay with deterministic, socket-denied transport."""
import socket
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from agent import chat_completion_helpers as h


@pytest.mark.parametrize("kind,expected", [("empty", 10), ("text", 37), ("reasoning", 37), ("name", 37), ("arguments", 37), ("terminal", 37), ("retry", 19), ("legacy", None)])
def test_progress_clock(monkeypatch, kind, expected):
    def denied(*a, **kw):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: None if kind == "legacy" else 10)
    monkeypatch.setattr(h, "get_provider_stale_timeout", lambda *a: 10)
    monkeypatch.setattr(h, "get_provider_request_timeout", lambda *a: None)
    monkeypatch.setattr(h, "should_use_direct_api_call", lambda a: False)
    clock = [0.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(h.time, "time", lambda: clock[0] if kind == "legacy" else 1000.0 - clock[0])
    permit, ready = threading.Event(), threading.Event()
    real_thread = threading.Thread
    class Worker:
        def __init__(self, **kw):
            self.thread = real_thread(**kw)
        def start(self):
            self.thread.start()
            assert ready.wait(2)
        def is_alive(self):
            return self.thread.is_alive()
        def join(self, timeout=None):
            ready.clear()
            permit.set()
            assert ready.wait(2), "worker did not reach transport checkpoint"
            self.thread.join(0.001)
    monkeypatch.setattr(h.threading, "Thread", Worker)
    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.base_url = "chat_completions", "custom", "m", "https://example.invalid/v1"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._stream_diag_init.return_value = {}
    agent._fallback_index = 0
    aborts = []
    attempts = [0]
    class Stream:
        response = None
        def __iter__(self):
            attempts[0] += 1
            schedule = [18, 19, 20] if kind == "retry" and attempts[0] == 2 else [9, 10, 18, 27, 37, 38]
            for at in schedule:
                ready.set()
                assert permit.wait(2)
                permit.clear()
                if agent._interrupt_requested:
                    break
                clock[0] = at
                if kind == "retry" and attempts[0] == 1:
                    import httpx
                    raise httpx.ReadTimeout("retry transport")
                delta = NS(content=None, reasoning_content=None, tool_calls=None)
                finish = None
                if kind == "legacy":
                    delta.content = "ok" if at == 38 else None
                    finish = "stop" if at == 38 else None
                if at in [9, 18, 27]:
                    if kind == "text": delta.content = "x"
                    if kind == "reasoning": delta.reasoning_content = "thinking"
                    if kind in ("name", "arguments"):
                        delta.tool_calls = [NS(index=0, id="call", function=NS(name="tool" if kind == "name" else None, arguments=" " if kind == "arguments" else None))]
                    if kind == "terminal": finish = "stop"
                yield NS(choices=[NS(delta=delta, finish_reason=finish)], model="m", usage=None)
            ready.set()
        def close(self):
            pass
    stream = Stream()
    agent._create_request_openai_client.return_value.chat.completions.create.return_value = stream
    def abort(*a, **kw):
        aborts.append(clock[0])
        agent._interrupt_requested = True
        permit.set()
    agent._abort_request_openai_client.side_effect = abort
    monkeypatch.setattr(h, "_join_worker_for_relay_teardown", lambda worker, **kw: worker.thread.join(2))
    if kind == "legacy":
        response = h.interruptible_streaming_api_call(agent, {"model": "m", "messages": []})
        assert response.choices[0].message.content == "ok"
        assert not aborts
    else:
        with pytest.raises(InterruptedError):
            h.interruptible_streaming_api_call(agent, {"model": "m", "messages": []})
        assert aborts[0] == expected
    agent._close_request_openai_client.assert_called()
    assert agent._create_request_openai_client.call_count == (2 if kind == "retry" else 1)
