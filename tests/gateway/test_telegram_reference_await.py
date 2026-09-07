"""Telegram: an instruction that points at a *following* message must wait
for that reference material instead of dispatching immediately.

Screenshot-proven failure (Sano · Personal & Dev, 2026-09-07): user sent
"아래 메시지 참고해서 1분짜리 16:9 영상 만들어" and ~1s later forwarded a
post (photo + caption with $BEN metrics).  The instruction was dispatched
alone after the 0.18s split-batch window, so the agent replied that the
"아래 정보" was missing.
"""

import asyncio

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from tests.gateway.test_telegram_text_batching import _make_adapter, _make_event


def _photo_event(caption: str, chat_id: str = "12345") -> MessageEvent:
    return MessageEvent(
        text=caption,
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
        media_urls=["/tmp/ben-chart.jpg"],
        media_types=["image/jpeg"],
    )


class TestReferenceAwait:
    @pytest.mark.asyncio
    async def test_instruction_pointing_below_waits_for_followup_text(self):
        adapter = _make_adapter()
        adapter._reference_await_delay_seconds = 1.0

        adapter._enqueue_text_event(_make_event("아래 메시지 참고해서 1분짜리 16:9 영상 만들어줘"))
        await asyncio.sleep(0.4)  # well past the 0.18s fast window
        adapter.handle_message.assert_not_called()

        adapter._enqueue_text_event(_make_event("$BEN 시총 $5.51M, 가격 $0.00551, 유동성 $264.3K"))
        await asyncio.sleep(0.5)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "16:9 영상" in text
        assert "$BEN" in text

    @pytest.mark.asyncio
    async def test_instruction_pointing_below_merges_forwarded_photo(self):
        adapter = _make_adapter()
        adapter._reference_await_delay_seconds = 1.0
        adapter._media_batch_delay_seconds = 0.05

        adapter._enqueue_text_event(_make_event("아래 내용 참고해서 영상 만들어"))
        await asyncio.sleep(0.4)
        photo = _photo_event("$BEN 밈코인 차트 · 시총 $5.51M")
        adapter._enqueue_photo_event(adapter._photo_batch_key_for_event(photo), photo)
        await asyncio.sleep(0.5)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "영상 만들어" in dispatched.text
        assert "$BEN" in dispatched.text
        assert dispatched.media_urls == ["/tmp/ben-chart.jpg"]

    @pytest.mark.asyncio
    async def test_instruction_without_reference_cue_is_not_delayed(self):
        adapter = _make_adapter()
        adapter._reference_await_delay_seconds = 1.0

        adapter._enqueue_text_event(_make_event("안녕 잘 지내?"))
        await asyncio.sleep(0.3)

        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_reference_wait_gives_up_after_window(self):
        adapter = _make_adapter()
        adapter._reference_await_delay_seconds = 0.3

        adapter._enqueue_text_event(_make_event("아래 자료 참고해서 정리해줘"))
        await asyncio.sleep(0.6)

        adapter.handle_message.assert_called_once()
