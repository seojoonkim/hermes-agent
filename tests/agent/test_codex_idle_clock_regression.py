"""Codex semantic-idle contracts through the real worker and Relay path."""
import queue
import socket
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as h
from agent import codex_runtime as codex


@pytest.fixture
def runtime(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(h, "should_use_direct_api_call", lambda agent: False)
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: 10)
    monkeypatch.setattr(h, "get_provider_stale_timeout", lambda *args: 1000)
    clock = [100.0]
    # The native module's fallback monotonic clock deliberately differs too:
    # semantic progress must use the exact clock supplied by the helper.
    monkeypatch.setattr(h, "time", NS(
        time=lambda: 1_800_000_000 + clock[0], monotonic=lambda: clock[0]))
    monkeypatch.setattr(codex, "time", NS(
        time=lambda: 1_800_000_000 + clock[0], monotonic=lambda: 50_000 + clock[0]))
    ready, permit = queue.Queue(), queue.Queue()
    stopping = threading.Event()
    real_thread = threading.Thread
    workers = []

    def pause(at):
        clock[0] = at
        ready.put("paused")
        if permit.get(timeout=2) == "stop":
            raise InterruptedError("test teardown")

    class Worker:
        def __init__(self, *, target, **kwargs):
            def run():
                try:
                    target()
                finally:
                    ready.put("done")
            self.thread = real_thread(target=run, **kwargs)
            workers.append(self)

        def start(self):
            self.thread.start()
            assert ready.get(timeout=2) == "paused"

        def is_alive(self):
            return self.thread.is_alive()

        def join(self, timeout=None):
            if self.thread.is_alive():
                permit.put("continue")
                state = ready.get(timeout=2)
                if state == "done":
                    self.thread.join(2)

    monkeypatch.setattr(h, "threading", NS(
        Thread=Worker, Lock=threading.Lock, get_ident=threading.get_ident))
    agent = MagicMock()
    agent.api_mode = "codex_responses"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._fallback_index = 0
    agent._compute_non_stream_stale_timeout.return_value = 1000
    agent._stream_diag_init.return_value = {}
    agent._base_url_lower = "https://example.invalid"
    agent.provider, agent.model = "test", "m"
    agent._run_codex_stream.side_effect = lambda kw, **k: codex.run_codex_stream(agent, kw, **k)
    client = agent._create_request_openai_client.return_value
    yield NS(agent=agent, client=client, pause=pause, clock=clock)
    stopping.set()
    for worker in workers:
        permit.put("stop")
        worker.thread.join(2)
        assert not worker.thread.is_alive(), "native worker leaked"


def terminal_event():
    return {"type": "response.completed", "response": {
        "id": "r", "output": [], "status": "completed"}}


def test_productive_then_silent_times_out_with_distinct_clocks(runtime):
    x = runtime
    stream = MagicMock()

    def events():
        x.pause(100)
        x.clock[0] = 109
        yield {"type": "response.output_text.delta", "delta": "useful"}
        x.pause(109)
        # No heartbeat or other event: transport is genuinely silent.
        x.pause(119)
        yield terminal_event()

    stream.__iter__.side_effect = events
    x.client.responses.create.return_value = stream
    with pytest.raises(TimeoutError, match="meaningful inactivity"):
        h.interruptible_api_call(x.agent, {"model": "m", "input": []})
    assert x.clock[0] == 119
    x.agent._fire_stream_delta.assert_called_once_with("useful")
    x.agent._abort_request_openai_client.assert_called_once()
    assert x.agent._consecutive_stale_streams == 1
    assert x.agent._codex_stream_last_event_ts >= 1_800_000_000
    stream.close.assert_called()
    assert x.client.responses.create.call_count == 1