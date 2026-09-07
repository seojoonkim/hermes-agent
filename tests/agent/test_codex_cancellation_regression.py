import socket
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from agent import codex_runtime, relay_llm
from agent.stream_progress import _CodexProgressClient


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("network forbidden in cancellation regression")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


@pytest.mark.parametrize("boundary", ["created", "accept", "text", "reasoning", "commentary", "retry"])
def test_native_codex_cancellation_after_ingress(monkeypatch, boundary):
    """Exercise native runtime + real Relay with cancellation past the proxy.

    Relay hook interception covers the ownership/acceptance gaps. Parsing
    interception covers an event already accepted and interrupt-checked, before
    its user callbacks. Neither the consumer nor progress proxy is replaced.
    """
    cancelled = {"value": False}
    control = {"cancelled": cancelled, "progress": {"t": 0}, "clock": lambda: 7}
    agent = NS(
        _interrupt_requested=False, _fallback_index=0,
        _fire_stream_delta=Mock(), _fire_reasoning_delta=Mock(),
        _fire_streamed_codex_commentary=Mock(), _touch_activity=Mock(),
        _claim_stream_writer=Mock(return_value=123),
        _stream_writer_is_current=Mock(return_value=True),
        interim_assistant_callback=Mock(), _client_log_context=lambda: "test",
    )
    first_delta = Mock()
    event = {"type": "response.output_text.delta", "delta": "late"}
    if boundary == "reasoning":
        event = {"type": "response.reasoning_summary_text.delta", "delta": "late"}
    elif boundary == "commentary":
        event = {"type": "response.output_item.done", "item": {
            "type": "message", "phase": "commentary", "content": [
                {"type": "output_text", "text": "late"}],
        }}
    raw = Mock()
    raw.__iter__ = Mock(return_value=iter([event]))
    client = NS(responses=NS(create=Mock(return_value=raw)))
    monkeypatch.setattr(relay_llm.relay_runtime, "resolve_execution_context", lambda *_: (None, None, None))
    real_stream = relay_llm.stream
    real_field = codex_runtime._event_field
    accepted = []
    cancelled_at = []

    def cancel():
        cancelled["value"] = True
        # An abandoned request must remain fenced after a later turn reset.
        agent._interrupt_requested = False
        cancelled_at.append(boundary)

    def intercept_stream(*args, **kwargs):
        original_created = kwargs["on_stream_created"]
        original_accept = kwargs["accept_chunk"]

        def created(stream):
            if boundary == "created":
                cancel()
            original_created(stream)

        def accept(chunk):
            if boundary == "accept":
                cancel()
            result = original_accept(chunk)
            accepted.append(result)
            return result

        kwargs.update(on_stream_created=created, accept_chunk=accept)
        managed = real_stream(*args, **kwargs)
        if boundary == "retry":
            managed.close()
            cancel()
            raise ConnectionError("cancelled connection")
        return managed

    def field(value, name, default=None):
        result = real_field(value, name, default)
        trigger = "item" if boundary == "commentary" else "delta"
        if boundary in {"text", "reasoning", "commentary"} and value is event and name == trigger:
            # This read is downstream of ingress AND the consumer interrupt
            # gate; cancellation here must still fence each native callback.
            cancel()
        return result

    monkeypatch.setattr(relay_llm, "stream", intercept_stream)
    monkeypatch.setattr(codex_runtime, "_event_field", field)
    try:
        codex_runtime.run_codex_stream(
            agent, {"model": "test", "input": []}, client=client,
            on_first_delta=first_delta, request_control=control,
        )
    except (InterruptedError, RuntimeError):
        # Cancellation may terminate iteration without a terminal response.
        pass

    assert cancelled_at, "the intended post-ingress cancellation seam was not reached"
    assert cancelled["value"] and not agent._interrupt_requested
    assert {
        "text": agent._fire_stream_delta.call_count,
        "reasoning": agent._fire_reasoning_delta.call_count,
        "commentary": agent._fire_streamed_codex_commentary.call_count,
        "first_delta": first_delta.call_count,
        "stored_text": agent._codex_streamed_text_parts,
        "late_writer_claim": agent._claim_stream_writer.call_count if boundary == "created" else 0,
        "late_chunk_accepted": any(accepted) if boundary == "accept" else False,
    } == {
        "text": 0, "reasoning": 0, "commentary": 0, "first_delta": 0,
        "stored_text": [], "late_writer_claim": 0, "late_chunk_accepted": False,
    }
    assert client.responses.create.call_count == 1
    raw.close.assert_called()
    if boundary == "retry":
        assert len(cancelled_at) == 1

class SocketDenied:
    def __init__(self): self.calls=0
    def create(self, **kwargs):
        self.calls += 1
        raise OSError("socket denied")

class LateStream:
    def __init__(self): self.closed=False
    def __iter__(self): return iter([type("E", (), {"type":"response.output_text.delta", "delta":"late"})()])
    def close(self): self.closed=True

def test_cancelled_request_denies_late_sdk_creation_without_socket():
    cancelled={"value":True}; progress={"t":0}; sdk=SocketDenied()
    with pytest.raises(InterruptedError): _CodexProgressClient(sdk,cancelled,progress,lambda: 7).create(stream=True)
    assert sdk.calls == 0

def test_cancelled_request_fences_late_callbacks_and_first_delta():
    cancelled={"value":False}; progress={"t":0}; stream=LateStream()
    class SDK:
        responses=type("R", (), {"create":lambda self, **kw: stream})()
    proxy=_CodexProgressClient(SDK(),cancelled,progress,lambda: 7)
    wrapped=proxy.create(stream=True); cancelled["value"]=True
    with pytest.raises(InterruptedError): list(wrapped)
    assert not stream.closed

def test_cancellation_blocks_physical_retry_after_interrupt_reset():
    cancelled={"value":True}; calls=[]
    class SDK:
        class responses:
            @staticmethod
            def create(**kw): calls.append(1); return LateStream()
    proxy=_CodexProgressClient(SDK(),cancelled,{"t":0},lambda: 7)
    with pytest.raises(InterruptedError): proxy.create(stream=True)
    cancelled["value"]=True
    with pytest.raises(InterruptedError): proxy.create(stream=True)
    assert calls == []
