"""Receipt-controlled real consumer-loop coverage; not a live Telegram canary.

The gateway disables token callbacks, not the consumer, when only interim
messages are enabled. These tests therefore enqueue commentary without deltas.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

ACK = "전달 경로부터 확인하고 회귀 테스트로 검증할게. 수동 예상으로 3~5분이야."
FINAL = "전달 경로 검증을 마쳤어. 결과는 보고서에 저장했어."


def make_consumer(*receipts):
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.send = AsyncMock(side_effect=list(receipts))
    adapter.edit_message = AsyncMock(return_value=receipt())
    consumer = GatewayStreamConsumer(
        adapter, "controlled-chat", StreamConsumerConfig(buffer_only=True),
        metadata={"thread_id": "controlled-thread"},
    )
    return consumer, adapter


def receipt(success=True):
    return SimpleNamespace(success=success, message_id="controlled-receipt" if success else None)


async def drain(consumer):
    consumer.finish()
    await asyncio.wait_for(consumer.run(), timeout=2)


@pytest.mark.asyncio
async def test_ack_eta_delivered_without_any_token_deltas():
    consumer, adapter = make_consumer(receipt())
    consumer.on_commentary(ACK)
    assert not consumer.has_delivered_text(ACK)  # queued/generated is not delivered
    await drain(consumer)
    adapter.send.assert_awaited_once_with(
        chat_id="controlled-chat", content=ACK,
        metadata={"thread_id": "controlled-thread", "_interim_send": True},
    )
    assert consumer.has_delivered_text(ACK)
    assert not consumer.final_response_sent
    assert not consumer.final_content_delivered
    assert consumer.metadata == {"thread_id": "controlled-thread"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [receipt(False), RuntimeError("controlled transport failure")], ids=["negative-receipt", "transport-exception"])
async def test_failed_ack_never_marked_delivered(failure):
    consumer, adapter = make_consumer(failure)
    consumer.on_commentary(ACK)
    await drain(consumer)
    assert adapter.send.await_count == 1
    assert not consumer.has_delivered_text(ACK)
    assert not consumer.final_response_sent
    assert not consumer.final_content_delivered


@pytest.mark.asyncio
async def test_failed_commentary_can_be_retried_then_recorded():
    consumer, adapter = make_consumer(receipt(False), receipt())
    consumer.on_commentary(ACK)
    consumer.on_commentary(ACK)
    await drain(consumer)
    assert adapter.send.await_count == 2
    assert consumer.has_delivered_text(ACK)
    assert not consumer.final_response_sent


@pytest.mark.asyncio
async def test_ack_does_not_suppress_distinct_final_stream():
    consumer, adapter = make_consumer(receipt(), receipt())
    consumer.on_commentary(ACK)
    consumer.on_delta(FINAL)
    await drain(consumer)
    assert [call.kwargs["content"] for call in adapter.send.await_args_list] == [ACK, FINAL]
    assert consumer.final_response_sent
    assert consumer.final_content_delivered
    assert consumer.has_delivered_text(FINAL)
    assert consumer.delivered_final_matches(FINAL) is True
    assert not adapter.send.await_args_list[-1].kwargs.get("metadata", {}).get("_interim_send", False)


@pytest.mark.asyncio
async def test_commentary_only_leaves_nonstream_final_for_gateway():
    consumer, adapter = make_consumer(receipt())
    consumer.on_commentary(ACK)
    consumer.finish(final_text=FINAL)
    await asyncio.wait_for(consumer.run(), timeout=2)
    assert adapter.send.await_count == 1
    assert consumer.has_delivered_text(ACK)
    assert not consumer.has_delivered_text(FINAL)
    assert not consumer.final_response_sent
    assert not consumer.final_content_delivered


@pytest.mark.parametrize("text", [ACK, "아직 확인 중이야. 2분 안에 확인된 내용과 남은 작업을 알려줄게."])
def test_noise_filters_preserve_human_eta(text):
    assert GatewayStreamConsumer._clean_for_display(text) == text
    assert _prepare_gateway_status_message(Platform.TELEGRAM, "info", text) == text
    assert _prepare_gateway_status_message(
        Platform.TELEGRAM, "info",
        "ETA p50 96.1s · p80 599.3s · confidence high · critical path operations · risk +503.2s",
    ) is None


@pytest.mark.asyncio
async def test_delivery_lookup_recognizes_exact_commentary_not_other_completion():
    consumer, adapter = make_consumer(receipt())
    consumer.on_commentary(FINAL)
    await drain(consumer)
    assert consumer.has_delivered_text(FINAL)
    assert not consumer.has_delivered_text(FINAL + " 추가 검증도 마쳤어.")
    assert adapter.send.await_count == 1
    assert not consumer.final_response_sent


@pytest.mark.asyncio
async def test_consumer_duplicate_commentary_current_contract_requires_caller_dedup():
    # Characterization, NOT a dedup acceptance test: on_commentary is a queue,
    # and neither it nor _send_commentary promises idempotency. The agent's
    # upstream callback layer owns invocation dedup today.
    consumer, adapter = make_consumer(receipt(), receipt())
    consumer.on_commentary(ACK)
    consumer.on_commentary(ACK)
    await drain(consumer)
    assert adapter.send.await_count == 2
    assert consumer.has_delivered_text(ACK)
    assert not consumer.final_response_sent
