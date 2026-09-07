"""Bounded offline safety review: execute production setup block, no network."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import gateway.run
from gateway.platforms.base import SendResult
from gateway.stream_consumer import StreamConsumerConfig
from tests.gateway.test_ack_eta_delivery_integration import make_consumer, ACK
from tests.gateway.test_ack_final_boundary_integration import harness


def setup_callback(adapter, streaming=True, interim=True):
    # Execute the complete production consumer/callback setup block, including
    # its capability assignment (not an assignment supplied by the harness).
    tree = ast.parse(Path(gateway.run.__file__).read_text())
    owner = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and any(isinstance(c, ast.FunctionDef) and c.name == '_interim_assistant_cb' for c in n.body))
    start = next(i for i, n in enumerate(owner.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_stream_consumer' for t in n.targets))
    end = next(i for i, n in enumerate(owner.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == 'receipt_aware' for t in n.targets))
    ctx = SimpleNamespace(streaming_tts_consumer_holder=[None], user_config={},
        resolve_display_setting=lambda *a: None, interim_assistant_messages_enabled=interim,
        source=SimpleNamespace(chat_id='offline'), _status_thread_metadata={},
        progress_queue=None, event_message_id=None, _run_still_current=lambda: True,
        stream_consumer_holder=[None], _status_adapter=None)
    runner = SimpleNamespace(config=SimpleNamespace(streaming=SimpleNamespace(enabled=streaming, transport='auto')),
        _adapter_for_source=lambda source: adapter,
        _build_stream_consumer_config=lambda *a, **k: (StreamConsumerConfig(buffer_only=True), None))
    ns = dict(vars(gateway.run), ctx=ctx, self=SimpleNamespace(_runner=runner), platform_key='telegram')
    exec(compile(ast.Module(body=owner.body[start:end + 1], type_ignores=[]), '<production-setup>', 'exec'), ns)
    return ns['_interim_assistant_cb'], ctx.stream_consumer_holder[0]


@pytest.mark.parametrize('streaming,interim,present', [(True, True, True), (False, True, True), (False, False, False)])
def test_production_setup_assigns_receipt_capability(streaming, interim, present):
    _, adapter = make_consumer()
    cb, consumer = setup_callback(adapter, streaming, interim)
    assert cb.receipt_aware is present
    assert (consumer is not None) is present
    if present:
        assert cb(ACK) is False
        assert not consumer.has_delivered_text(ACK)
        assert consumer._queue.qsize() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('message_id','partial'), ('continuation_message_ids',['partial']), ('raw_response',{'accepted':True})])
@pytest.mark.parametrize('kind', ['forbidden','not_found','rate_limited','bad_format'])
async def test_partial_negative_never_releases_claim(field, value, kind):
    result = SendResult(success=False, error_kind=kind)
    setattr(result, field, value)
    consumer, adapter = make_consumer(result)
    consumer.on_commentary_receipt_aware(ACK)
    await consumer._send_commentary(ACK)
    consumer.on_commentary_receipt_aware(ACK)
    assert consumer._queue.qsize() == 1
    assert not consumer.has_delivered_text(ACK)
    assert not consumer.final_response_sent


def test_structured_already_streamed_preserves_segment_break():
    agent, _, _ = harness()
    cb = Mock(return_value=False)
    cb.receipt_aware = True
    agent.interim_assistant_callback = cb
    agent._current_streamed_assistant_text = 'part A\n\npart B'
    agent._emit_interim_assistant_message({'role':'assistant', 'content':'',
        'codex_message_items':[{'type':'message', 'phase':'commentary',
        'content':[{'type':'output_text','text':p}]} for p in ['part A','part B']]})
    assert any(c.kwargs.get('already_streamed') is True for c in cb.call_args_list), cb.call_args_list
