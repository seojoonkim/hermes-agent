from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.callbacks = {}
        self._active_sessions = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return SimpleNamespace(success=True)

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callbacks[session_key] = (generation, callback)


@pytest.mark.asyncio
async def test_requirements_notice_and_continuation_run_after_main_delivery():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)
    queued = []
    runner._enqueue_fifo = lambda key, event, selected_adapter: queued.append(
        (key, event, selected_adapter)
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="46291309",
        user_id="46291309",
        profile="sion",
    )
    result = {
        "completion_blocked": True,
        "failed": False,
        "interrupted": False,
        "pending_requirements": [
            {"id": "req:1", "content": "최종 파일을 전달한다."},
        ],
    }

    scheduled = await runner._post_turn_requirements_continuation(
        source=source,
        result=result,
    )

    assert scheduled is True
    assert adapter.calls == []
    assert queued == []
    assert len(adapter.callbacks) == 1

    _, callback = next(iter(adapter.callbacks.values()))
    await callback()

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"].startswith("⚠️ 아직 완료되지 않은 요구사항이 1개 있어.")
    assert len(queued) == 1
    _, event, selected_adapter = queued[0]
    assert selected_adapter is adapter
    assert event.source is source
    assert event.internal is True
    assert "[내부 자동 재개 지시]" in event.text
    assert "req:1" in event.text


@pytest.mark.asyncio
async def test_requirements_continuation_is_not_scheduled_for_user_blocker():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = FakeAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="46291309",
        profile="sion",
    )

    scheduled = await runner._post_turn_requirements_continuation(
        source=source,
        result={
            "completion_blocked": True,
            "guardrail": {"action": "require_approval"},
            "pending_requirements": [{"id": "req:1", "content": "위험 명령 실행"}],
        },
    )

    assert scheduled is False
    assert adapter.callbacks == {}
