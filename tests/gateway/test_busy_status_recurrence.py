"""Screenshot recurrence: status checks must not mutate a busy turn."""
import time
from unittest.mock import AsyncMock

import pytest

from gateway.busy_status import is_busy_status_question
from gateway.config import Platform
from tests.gateway.test_priority_path_compression_demotion_56391 import _make_event, _make_runner
from tests.run_agent.test_steer import _bare_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["하는중?", "왜 오래걸려?", "왜 대답 안해?"])
@pytest.mark.parametrize("mode", ["steer", "interrupt", "queue"])
@pytest.mark.parametrize("path", ["priority", "active"])
async def test_normal_busy_status_does_not_mutate_turn(text, mode, path):
    runner, _, sk = _make_runner(compression_in_flight=False, busy_input_mode=mode)
    agent = _bare_agent()
    agent.get_activity_summary = lambda: {"last_activity_desc": "api_call", "seconds_since_activity": 12}
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time()  # include startup grace window
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._send_with_retry = AsyncMock()
    event = _make_event(text)
    if path == "priority":
        reply = await runner._handle_message(event)
    else:
        assert await runner._handle_active_session_busy_message(event, sk)
        reply = adapter._send_with_retry.call_args.kwargs["content"]
    assert reply and "완료 시간" in reply
    assert "반영했어" not in reply
    assert "정리 중" not in reply
    assert agent._drain_pending_steer() is None
    assert not agent._interrupt_requested
    assert sk not in adapter._pending_messages


@pytest.mark.parametrize("text", ["왜 늦어? 버튼도 고쳐", "왜 오래걸려? 버튼도 고쳐", "/stop", "status: fix the button"])
def test_mixed_action_is_not_status(text):
    assert not is_busy_status_question(text)


@pytest.mark.asyncio
async def test_steer_ack_does_not_claim_applied(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", "true")
    runner, agent, sk = _make_runner(compression_in_flight=False, busy_input_mode="steer")
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._send_with_retry = AsyncMock()
    await runner._handle_active_session_busy_message(_make_event("버튼도 고쳐"), sk)
    reply = adapter._send_with_retry.call_args.kwargs["content"]
    assert "반영했어" not in reply
    assert "받았어" in reply and "기다" in reply
    agent.steer.assert_called_once()


@pytest.mark.asyncio
async def test_status_caption_preserves_attachment():
    runner, agent, sk = _make_runner(compression_in_flight=False, busy_input_mode="steer")
    event = _make_event("왜 오래걸려?")
    event.media_urls = ["https://example.invalid/photo.png"]
    event.media_types = ["image/png"]
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._send_with_retry = AsyncMock()
    await runner._handle_active_session_busy_message(event, sk)
    assert adapter._pending_messages[sk] is event
    agent.steer.assert_not_called()
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_status_is_not_answered():
    runner, agent, sk = _make_runner(compression_in_flight=False)
    runner._is_user_authorized = lambda source: False
    assert await runner._handle_message(_make_event("하는중?")) is None
    agent.steer.assert_not_called()
    agent.interrupt.assert_not_called()
    assert sk not in runner.adapters[Platform.TELEGRAM]._pending_messages
