"""Offline acceptance tests for agent -> actual gateway callback -> consumer.

Only the external adapter is controlled. The nested gateway callback is compiled
from the current source AST (not reimplemented); full gateway construction and
provider generation are deliberately outside this test. No Telegram I/O occurs.
"""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from run_agent import AIAgent

ACK = "전달 경로를 확인할게. 수동 예상으로 3~5분이야."
FINAL = "검증을 마쳤어. 결과를 보고서에 저장했어."


def gateway_callback(consumer, current=lambda: True):
    source = Path(gateway.run.__file__)
    tree = ast.parse(source.read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef)
             and node.name == "_interim_assistant_cb"]
    assert len(nodes) == 1
    namespace = dict(vars(gateway.run))
    namespace.update(
        ctx=SimpleNamespace(_run_still_current=current),
        _stream_consumer=consumer,
    )
    module = ast.Module(body=[nodes[0]], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["_interim_assistant_cb"]


def harness(first_success=True, raises=False):
    attempts = []

    async def send(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1 and raises:
            raise RuntimeError("offline transport failure")
        success = first_success if len(attempts) == 1 else True
        return SimpleNamespace(success=success,
                               error_kind=None if success else "rate_limited",
                               message_id=f"offline-{len(attempts)}" if success else None)

    adapter = SimpleNamespace(MAX_MESSAGE_LENGTH=4096, REQUIRES_EDIT_FINALIZE=False,
                              send=send, edit_message=AsyncMock())
    consumer = GatewayStreamConsumer(adapter, "offline-chat",
                                     StreamConsumerConfig(buffer_only=True))
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "offline-session"
    agent.model = "offline-no-provider-call"
    agent.provider = "offline"
    agent.platform = "telegram"
    agent.interim_assistant_callback = gateway_callback(consumer)
    return agent, consumer, attempts


def emit(agent, text, route):
    if route == "live-codex":
        agent._fire_streamed_codex_commentary(text)
    elif route == "structured-codex":
        agent._emit_interim_assistant_message({
            "role": "assistant", "content": "",
            "codex_message_items": [{"type": "message", "phase": "commentary",
                                     "content": [{"type": "output_text", "text": text}]}],
        })
    else:
        agent._emit_interim_assistant_message({"role": "assistant", "content": text})


async def settled(consumer):
    # Flush marker is processed after commentary send/receipt handling, unlike
    # polling the adapter invocation, which can race delivery bookkeeping.
    assert await asyncio.to_thread(consumer.flush_pending_sync, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["plain", "structured-codex", "live-codex"])
@pytest.mark.parametrize("raises", [False, True], ids=["negative-receipt", "exception"])
async def test_failed_adapter_receipt_allows_same_agent_interim_retry(route, raises):
    agent, consumer, attempts = harness(first_success=False, raises=raises)
    task = asyncio.create_task(consumer.run())
    try:
        emit(agent, ACK, route)
        await settled(consumer)
        assert len(attempts) == 1
        assert not consumer.has_delivered_text(ACK)
        assert not consumer.final_response_sent
        emit(agent, ACK, route)
        await settled(consumer)
        # Acceptance, not characterization: a definitely failed send must not
        # poison the caller's dedup ledger and suppress the explicit retry.
        if raises:
            # Revised safety requirement: an exception is an UNKNOWN outcome,
            # not a definite rejection. Do not risk a second delivery.
            assert len(attempts) == 1
            assert not consumer.has_delivered_text(ACK)
        else:
            assert len(attempts) == 2, "AIAgent invocation dedup suppressed failed-delivery retry"
            assert consumer.has_delivered_text(ACK)
    finally:
        consumer.finish()
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["plain", "structured-codex", "live-codex"])
async def test_successful_ack_replay_once_and_distinct_final_delivered(route):
    agent, consumer, attempts = harness()
    task = asyncio.create_task(consumer.run())
    try:
        emit(agent, ACK, route)
        await settled(consumer)
        assert consumer.has_delivered_text(ACK)
        emit(agent, ACK, route)
        await settled(consumer)
        assert len(attempts) == 1
        consumer.on_delta(FINAL)
    finally:
        consumer.finish()
        await asyncio.wait_for(task, 2)
    assert [item["content"] for item in attempts] == [ACK, FINAL]
    assert attempts[0]["metadata"]["_interim_send"] is True
    assert not (attempts[-1].get("metadata") or {}).get("_interim_send", False)
    assert consumer.final_response_sent
    assert consumer.final_content_delivered
    assert consumer.delivered_final_matches(FINAL)


@pytest.mark.asyncio
async def test_failed_ack_does_not_block_distinct_final():
    agent, consumer, attempts = harness(first_success=False)
    emit(agent, ACK, "plain")
    consumer.on_delta(FINAL)
    consumer.finish()
    await asyncio.wait_for(consumer.run(), 2)
    assert [item["content"] for item in attempts] == [ACK, FINAL]
    assert not consumer.has_delivered_text(ACK)
    assert consumer.delivered_final_matches(FINAL)
    assert consumer.final_content_delivered
    assert consumer.final_response_sent


@pytest.mark.asyncio
async def test_in_flight_duplicate_is_suppressed_until_receipt():
    attempts = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def send(**kwargs):
        attempts.append(kwargs)
        started.set()
        await release.wait()
        return SimpleNamespace(success=True, message_id="offline-success")

    adapter = SimpleNamespace(MAX_MESSAGE_LENGTH=4096, REQUIRES_EDIT_FINALIZE=False,
                              send=send, edit_message=AsyncMock())
    consumer = GatewayStreamConsumer(adapter, "offline-chat",
                                     StreamConsumerConfig(buffer_only=True))
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "offline-session"
    agent.model = "offline-no-provider-call"
    agent.provider = "offline"
    agent.platform = "telegram"
    agent.interim_assistant_callback = gateway_callback(consumer)
    task = asyncio.create_task(consumer.run())
    try:
        emit(agent, ACK, "plain")
        await asyncio.wait_for(started.wait(), 2)
        emit(agent, ACK, "plain")
        assert len(attempts) == 1
        release.set()
        await settled(consumer)
        emit(agent, ACK, "plain")
        await settled(consumer)
        assert len(attempts) == 1
        assert consumer.has_delivered_text(ACK)
    finally:
        consumer.finish()
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_pending_receipt_dedups_concurrent_agent_emissions():
    agent, consumer, attempts = harness()
    started, release = asyncio.Event(), asyncio.Event()
    original = consumer.adapter.send

    async def blocked_send(**kwargs):
        started.set()
        await release.wait()
        return await original(**kwargs)

    consumer.adapter.send = blocked_send
    task = asyncio.create_task(consumer.run())
    try:
        emit(agent, ACK, "plain")
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.gather(*(asyncio.to_thread(emit, agent, ACK, "plain")
                               for _ in range(12)))
        assert not consumer.has_delivered_text(ACK)
        assert not agent._interim_text_was_delivered(ACK)
        release.set()
        await settled(consumer)
        assert len(attempts) == 1
        emit(agent, ACK, "plain")
        await settled(consumer)
        assert len(attempts) == 1
    finally:
        release.set()
        consumer.finish()
        await asyncio.wait_for(task, 2)


@pytest.mark.parametrize("route", ["plain", "structured-codex", "live-codex"])
def test_legacy_synchronous_callback_retains_dedup(route):
    agent, _, _ = harness()
    seen = []
    def callback(text, **kwargs):
        seen.append(text)
    agent.interim_assistant_callback = callback
    emit(agent, ACK, route)
    emit(agent, ACK, route)
    assert seen == [ACK]
    assert agent._interim_text_was_delivered(ACK)


@pytest.mark.asyncio
async def test_stale_callback_does_not_claim_delivery_or_poison_later_turn():
    agent, consumer, attempts = harness()
    agent.interim_assistant_callback = gateway_callback(consumer, lambda: False)
    emit(agent, ACK, "plain")
    assert not agent._interim_text_was_delivered(ACK)
    agent.interim_assistant_callback = gateway_callback(consumer)
    emit(agent, ACK, "plain")
    consumer.finish()
    await asyncio.wait_for(consumer.run(), 2)
    assert len(attempts) == 1
    assert consumer.has_delivered_text(ACK)
