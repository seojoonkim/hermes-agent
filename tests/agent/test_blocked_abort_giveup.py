"""Runtime watchdog regression: abort itself blocks independently of recv."""
import socket
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as h


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages", "codex_responses"])
@pytest.mark.parametrize("late", ["chunk", "create", "error", "interrupt"])
def test_blocked_abort_cannot_hold_giveup_caller(monkeypatch, mode, late):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: 10)
    monkeypatch.setattr(h, "get_provider_stale_timeout", lambda *a: 10)
    monkeypatch.setattr(h, "get_provider_request_timeout", lambda *a: None)
    monkeypatch.setattr(h, "should_use_direct_api_call", lambda a: False)
    clock = [0.0]
    monkeypatch.setattr(h, "time", NS(time=lambda: 1000 + clock[0], monotonic=lambda: clock[0]))
    ready, release_worker = threading.Event(), threading.Event()
    abort_entered, release_abort = threading.Event(), threading.Event()
    caller_done = threading.Event()
    threads, workers, close_threads, aborted = [], [], [], []

    class Thread:
        def __init__(self, **kwargs):
            self.thread = threading.Thread(**kwargs)
            self.worker = not kwargs.get("name", "").startswith("hermes-request-abort")
            threads.append(self)
            if self.worker:
                workers.append(self)
        def start(self):
            self.thread.start()
            if self.worker:
                assert ready.wait(2)
            else:
                assert abort_entered.wait(2)
        def is_alive(self):
            return self.thread.is_alive()
        def join(self, timeout=None):
            clock[0] += 1
            self.thread.join(0.001)

    monkeypatch.setattr(h, "threading", NS(Thread=Thread, Lock=threading.Lock, get_ident=threading.get_ident))
    monkeypatch.setattr(h, "_RequestAbortThread", Thread)
    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model = mode, "custom", "m"
    agent.base_url = "https://example.invalid/v1"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._fallback_index = 0
    agent._stream_diag_init.return_value = {}
    agent._compute_non_stream_stale_timeout.return_value = 1000
    agent._base_url_lower = "https://example.invalid"
    if mode == "codex_responses":
        from agent import codex_runtime
        agent._run_codex_stream.side_effect = lambda kw, **k: codex_runtime.run_codex_stream(agent, kw, **k)
    generation = [0]
    def claim():
        generation[0] += 1
        return generation[0]
    agent._claim_stream_writer.side_effect = claim
    agent._stream_writer_is_current.side_effect = lambda token: token == generation[0]
    def abort(client, **kwargs):
        if late == "interrupt":
            agent._interrupt_requested = True
        abort_entered.set()
        assert release_abort.wait(10), "test failed to release blocked abort"
        aborted.append(client)
    agent._abort_request_openai_client.side_effect = abort
    agent._abort_request_anthropic_client.side_effect = abort
    agent._close_request_openai_client.side_effect = lambda *a, **kw: close_threads.append(threading.get_ident())
    agent._close_request_anthropic_client.side_effect = lambda *a, **kw: close_threads.append(threading.get_ident())

    def block_worker():
        ready.set()
        assert release_worker.wait(10), "test failed to release worker"
    class Stream:
        response = NS(headers={})
        def __enter__(self):
            if late == "create":
                block_worker()
            return self
        def __exit__(self, *args): pass
        def close(self): pass
        def get_final_message(self):
            return NS(content=[], stop_reason="end_turn")
        def __iter__(self):
            if late != "create":
                block_worker()
            if late == "error":
                import httpx
                raise httpx.ReadTimeout("late transport failure")
            if mode == "codex_responses":
                yield {"type": "response.output_text.delta", "delta": "LATE"}
                yield {"type": "response.completed", "response": {"id": "r", "output": [], "status": "completed"}}
            elif mode == "anthropic_messages":
                yield NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking="LATE"))
            else:
                yield NS(choices=[NS(delta=NS(content="LATE", reasoning_content="LATE", tool_calls=[]), finish_reason="stop")], model="m", usage=None)
    stream = Stream()
    def create(**kwargs):
        if late == "create":
            block_worker()
        return stream
    agent._create_request_openai_client.return_value.chat.completions.create.side_effect = create
    agent._create_request_openai_client.return_value.responses.create.side_effect = create
    agent._create_request_anthropic_client.return_value.messages.stream.return_value = stream
    original_client = (agent._create_request_anthropic_client.return_value if mode == "anthropic_messages"
                       else agent._create_request_openai_client.return_value)
    outcome = []
    first_delta = MagicMock()
    def call():
        try:
            if mode == "codex_responses":
                h.interruptible_api_call(agent, {"model": "m", "input": []})
            else:
                h.interruptible_streaming_api_call(agent, {"model": "m", "messages": []}, on_first_delta=first_delta)
        except BaseException as exc:
            outcome.append(exc)
        finally:
            caller_done.set()
    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    try:
        assert abort_entered.wait(2), "watchdog never reached abort"
        # Decisive RED assertion happens BEFORE either blocked event is released.
        assert caller_done.wait(1), "blocked abort prevented bounded watchdog give-up"
        assert len(outcome) == 1 and isinstance(outcome[0], InterruptedError), outcome
        assert "stale" in str(outcome[0]).lower() or late == "interrupt"
        assert workers[0].is_alive()
        assert not close_threads
        assert agent._create_request_openai_client.call_count + agent._create_request_anthropic_client.call_count == 1
        new_token = h.claim_stream_writer(agent)
        agent.reset_mock()
        agent._interrupt_requested = False
        # Let old output arrive while abort is STILL blocked: it cannot publish,
        # close/cache its client, or start another checkout under the held lock.
        release_worker.set()
        workers[0].thread.join(0.05)
        assert workers[0].is_alive()
        assert not close_threads
        release_abort.set()
        for thread in threads:
            thread.thread.join(2)
            assert not thread.is_alive()
        assert aborted == [original_client]
        assert close_threads == [workers[0].thread.ident]
        first_delta.assert_not_called()
        for name in ("_fire_stream_delta", "_fire_reasoning_delta", "_fire_tool_gen_started",
                     "_create_request_openai_client", "_create_request_anthropic_client",
                     "_claim_stream_writer", "_emit_stream_drop"):
            getattr(agent, name).assert_not_called()
        assert h.stream_writer_is_current(agent, new_token)
        if late == "create":
            agent._capture_rate_limits.assert_not_called()
            agent._stream_diag_capture_response.assert_not_called()
    finally:
        release_abort.set()
        release_worker.set()
        caller.join(2)
        for thread in threads:
            thread.thread.join(2)
        assert not caller.is_alive()
        assert all(not thread.is_alive() for thread in threads)
        original_client.close.assert_not_called()


def test_import_is_exact_checkout():
    from pathlib import Path
    assert Path(h.__file__).resolve() == Path(__file__).resolve().parents[2] / "agent/chat_completion_helpers.py"
