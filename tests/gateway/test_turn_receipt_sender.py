"""Gateway sends the turn receipt before any heavy turn preparation.

Measured 2026-10-05: user message -> first outbound text p50 99s / p90 384s,
because preflight compression (p50 31s, p90 61s) runs before the first model
call and the existing ETA status is suppressed as internal telemetry.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.turn_receipt_sender import send_turn_receipt


class _Adapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content))
        return SimpleNamespace(success=True, message_id="1")


def _source(chat_id="46291309"):
    return SimpleNamespace(chat_id=chat_id, thread_id=None)


@pytest.mark.asyncio
async def test_work_request_gets_receipt_with_summary():
    adapter = _Adapter()
    ok = await send_turn_receipt(adapter, _source(), "사노 덱 레이아웃 충돌 다 고쳐줘", estimate=None, enabled=True)
    assert ok is True
    assert len(adapter.sent) == 1
    assert "받았어" in adapter.sent[0][1]
    assert "사노 덱 레이아웃 충돌" in adapter.sent[0][1]


@pytest.mark.asyncio
async def test_trivial_message_gets_no_receipt():
    adapter = _Adapter()
    ok = await send_turn_receipt(adapter, _source(), "ㅇㅇ", estimate=None, enabled=True)
    assert ok is False and adapter.sent == []


@pytest.mark.asyncio
async def test_disabled_sends_nothing():
    adapter = _Adapter()
    ok = await send_turn_receipt(adapter, _source(), "덱 고쳐줘 지금", estimate=None, enabled=False)
    assert ok is False and adapter.sent == []


@pytest.mark.asyncio
async def test_receipt_is_bounded_and_never_raises_on_slow_or_broken_adapter():
    class Slow(_Adapter):
        async def send(self, chat_id, content, metadata=None):
            await asyncio.sleep(5)

    class Broken(_Adapter):
        async def send(self, chat_id, content, metadata=None):
            raise RuntimeError("telegram down")

    t0 = asyncio.get_event_loop().time()
    assert await send_turn_receipt(Slow(), _source(), "덱 다시 빌드해", estimate=None, enabled=True, timeout=0.2) is False
    assert asyncio.get_event_loop().time() - t0 < 1.0
    assert await send_turn_receipt(Broken(), _source(), "덱 다시 빌드해", estimate=None, enabled=True) is False


@pytest.mark.asyncio
async def test_estimate_is_rendered_in_plain_words():
    adapter = _Adapter()
    est = {"p50_ms": 6 * 60_000, "p80_ms": 11 * 60_000, "sample_count": 20}
    await send_turn_receipt(adapter, _source(), "투자 덱 19~20페이지 다시 빌드해", estimate=est, enabled=True)
    text = adapter.sent[0][1]
    assert "6~11분" in text
    assert "p50" not in text and "p80" not in text
