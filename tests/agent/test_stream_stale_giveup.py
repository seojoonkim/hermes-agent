"""Bounded stale-abort failure, real worker with a deterministic poll clock."""
import socket
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from agent import chat_completion_helpers as h


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
@pytest.mark.parametrize("late", ["chunk", "error", "create"])
def test_stuck_abort_gives_up_and_fences_late_worker(monkeypatch, late, mode):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: 10)
    monkeypatch.setattr(h, "get_provider_stale_timeout", lambda *a: 10)
    monkeypatch.setattr(h, "get_provider_request_timeout", lambda *a: None)
    monkeypatch.setattr(h, "should_use_direct_api_call", lambda a: False)
    clock = [0.0]
    monkeypatch.setattr(h.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(h.time, "time", lambda: 1000.0 - clock[0])
    ready, release = threading.Event(), threading.Event()
    real_thread = threading.Thread
    workers = []
    class Worker:
        def __init__(self, **kwargs):
            self.thread = real_thread(**kwargs)
            workers.append(self)
        def start(self):
            self.thread.start()
            assert ready.wait(2)
        def is_alive(self):
            return self.thread.is_alive()
        def join(self, timeout=None):
            clock[0] += 1
            assert clock[0] <= 16, "stale abort never gave up within bounded grace"
    monkeypatch.setattr(h.threading, "Thread", Worker)
    teardown = MagicMock()
    monkeypatch.setattr(h, "_join_worker_for_relay_teardown", teardown)
    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.base_url = "chat_completions", "custom", "m", "https://example.invalid/v1"
    agent.api_mode = mode
    generation = [0]
    def claim():
        generation[0] += 1
        return generation[0]
    agent._claim_stream_writer.side_effect = claim
    agent._stream_writer_is_current.side_effect = lambda token: token == generation[0]
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._stream_diag_init.return_value = {}
    agent._fallback_index = 0
    close_threads = []
    agent._close_request_openai_client.side_effect = lambda *a, **kw: close_threads.append(threading.get_ident())
    agent._close_request_anthropic_client.side_effect = lambda *a, **kw: close_threads.append(threading.get_ident())
    class Stream:
        response = NS(headers={"x-test": "old-response"})
        def __enter__(self):
            if late == "create":
                ready.set()
                assert release.wait(3)
            return self
        def __exit__(self, *args): pass
        def get_final_message(self):
            return NS(content=[], stop_reason="end_turn")
        def __iter__(self):
            if late != "create":
                ready.set()
                assert release.wait(3)
            if late == "error":
                import httpx
                raise httpx.ReadTimeout("late transport failure")
            if mode == "anthropic_messages":
                yield NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking="LATE"))
            else:
                yield NS(choices=[NS(delta=NS(content="LATE", reasoning_content="LATE", tool_calls=[NS(index=0, id="call", function=NS(name="danger", arguments="{}"))]), finish_reason="tool_calls")], model="m", usage=None)
        def close(self):
            pass
    stream = Stream()
    def create(**kwargs):
        if late == "create":
            ready.set()
            assert release.wait(3)
        return stream
    agent._create_request_openai_client.return_value.chat.completions.create.side_effect = create
    agent._create_request_anthropic_client.return_value.messages.stream.return_value = stream
    first_delta = MagicMock()
    try:
        with pytest.raises(InterruptedError, match="stale.*abort.*did not stop"):
            h.interruptible_streaming_api_call(agent, {"model": "m", "messages": []}, on_first_delta=first_delta)
        assert 10 < clock[0] <= 15
        assert workers[0].is_alive()
        assert not close_threads, "watchdog must not close worker-owned client"
        teardown.assert_called_once()
        # Ignore pre-failure diagnostics; no late delivery/lifecycle callbacks.
        new_token = h.claim_stream_writer(agent)
        agent.reset_mock()
        # A later turn can reset the agent flag; the request-local fence must
        # still reject this old worker's output and prevent its retry.
        agent._interrupt_requested = False
        release.set()
        workers[0].thread.join(2)
        assert not workers[0].is_alive()
        first_delta.assert_not_called()
        agent._fire_stream_delta.assert_not_called()
        agent._fire_reasoning_delta.assert_not_called()
        agent._create_request_openai_client.assert_not_called()
        agent._create_request_anthropic_client.assert_not_called()
        agent._claim_stream_writer.assert_not_called()
        assert h.stream_writer_is_current(agent, new_token)
        assert generation[0] == new_token
        if late == "create":
            agent._capture_rate_limits.assert_not_called()
            agent._capture_credits.assert_not_called()
            agent._stream_diag_capture_response.assert_not_called()
            agent._check_openrouter_cache_status.assert_not_called()
        agent._emit_stream_drop.assert_not_called()
        assert close_threads == [workers[0].thread.ident]
    finally:
        release.set()
        workers[0].thread.join(2)
