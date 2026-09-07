"""Native worker/Relay progress tests; fake transport, no network."""
import socket
import threading
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from agent import chat_completion_helpers as h



@pytest.mark.parametrize("kind", ["ping", "signature", "thinking", "json", "legacy"])
def test_native_progress_budget(monkeypatch, kind):
    def denied(*a, **kw):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(h, "get_agent_stream_idle_timeout", lambda: None if kind == "legacy" else 10)
    monkeypatch.setattr(h, "get_provider_stale_timeout", lambda *a: 10)
    monkeypatch.setattr(h, "get_provider_request_timeout", lambda *a: None)
    monkeypatch.setattr(h, "should_use_direct_api_call", lambda a: False)
    clock = [0.0]

    # Replace the module clock, not the process-global time functions.
    monkeypatch.setattr(h, "time", NS(time=lambda: clock[0], monotonic=lambda: clock[0]))
    ready, permit = threading.Event(), threading.Event()
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
            assert ready.wait(2)
            self.thread.join(0.001)
    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h, "_join_worker_for_relay_teardown", lambda worker, **kw: worker.thread.join(2))
    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model = "anthropic_messages", "anthropic", "m"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._stream_diag_init.return_value = {}
    agent._fallback_index = 0
    signed = NS(type="thinking", thinking="xxx", signature="signed-unchanged")
    tool = NS(type="tool_use", id="call", name="tool", input={"x": 1})
    final = NS(content=[signed, tool], stop_reason="tool_use")
    aborts = []
    class Stream:
        response = None
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def close(self): pass
        def get_final_message(self): return final
        def __iter__(self):
            for at in [9, 10, 18, 27, 28]:
                ready.set()
                assert permit.wait(2)
                permit.clear()
                if agent._interrupt_requested: break
                clock[0] = at
                if at == 28:
                    yield NS(type="message_stop")
                elif kind in ("ping", "legacy"):
                    yield NS(type="ping")
                else:
                    delta = {"signature": NS(type="signature_delta", signature="signed-unchanged"),
                             "thinking": NS(type="thinking_delta", thinking="x"),
                             "json": NS(type="input_json_delta", partial_json=" ")}[kind]
                    yield NS(type="content_block_delta", index=0, delta=delta)
            ready.set()
    agent._create_request_anthropic_client.return_value.messages.stream.return_value = Stream()
    def abort(*a, **kw):
        aborts.append(clock[0])
        agent._interrupt_requested = True
        permit.set()
    agent._abort_request_anthropic_client.side_effect = abort
    if kind in ("ping", "signature"):
        with pytest.raises(InterruptedError):
            h.interruptible_streaming_api_call(agent, {"model": "m", "messages": []})
        assert aborts and set(aborts) == {10}
        assert not [c for c in agent._touch_activity.call_args_list if c.args == ("receiving stream response",)]
    else:
        result = h.interruptible_streaming_api_call(agent, {"model": "m", "messages": []})
        assert result is final
        assert result.content[0].signature == "signed-unchanged"
        assert result.content[1].input == {"x": 1}
        assert not aborts
    agent._fire_stream_delta.assert_not_called()

    if kind == "thinking":
        assert [c.args[0] for c in agent._fire_reasoning_delta.call_args_list] == ["x"] * 4
    if kind != "thinking": agent._fire_reasoning_delta.assert_not_called()
    agent._close_request_anthropic_client.assert_called()
    assert agent._create_request_anthropic_client.call_count == 1
