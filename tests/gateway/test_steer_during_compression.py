"""Compression steering uses a separate slot, then the adopted transcript."""
import asyncio
import threading
from unittest.mock import patch

import pytest

from tests.gateway.test_priority_path_compression_demotion_56391 import _make_runner, _make_event
from tests.run_agent.test_steer import _bare_agent
from gateway.config import Platform


@pytest.mark.asyncio
@pytest.mark.parametrize("has_tool", [True, False])
@pytest.mark.parametrize("path", ["priority", "active"])
async def test_compression_steer_survives_snapshot_in_order(has_tool, path):
    runner, _, sk = _make_runner(compression_in_flight=True, busy_input_mode="steer")
    agent = _bare_agent()
    agent._supports_compression_steer = True  # synthetic opt-in; production is disabled
    agent.get_activity_summary = lambda: {"seconds_since_activity": 0.0}
    agent.session_id = "test"
    agent._conversation_root_id = lambda: None
    runner._running_agents[sk] = agent
    from unittest.mock import AsyncMock
    runner.adapters[Platform.TELEGRAM]._send_with_retry = AsyncMock()
    started, release = threading.Event(), threading.Event()
    original = [{"role": "user", "content": "original"}]
    adopted = [{"role": "user", "content": "summary"}]
    if has_tool:
        adopted.append({"role": "tool", "content": "result", "tool_call_id": "t"})

    def compress(_agent, snapshot, *args, **kwargs):
        assert snapshot is not original
        started.set()
        assert release.wait(5)
        assert snapshot == original
        return adopted, "sys"

    with patch("agent.conversation_compression.compress_context", side_effect=compress), patch(
        "agent.conversation_compression.resolve_context_compression_timeouts", return_value=(3, 4)
    ):
        work = asyncio.create_task(asyncio.to_thread(agent._compress_context, original, "sys"))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            for text in ("first correction", "second correction"):
                if path == "priority":
                    await runner._handle_message(_make_event(text))
                else:
                    await runner._handle_active_session_busy_message(_make_event(text), sk)
            assert sk not in runner.adapters[Platform.TELEGRAM]._pending_messages
            assert original == [{"role": "user", "content": "original"}]
        finally:
            release.set()
            result, _ = await work
    content = result[-1]["content"]
    assert content.count("first correction") == 1
    assert content.count("second correction") == 1
    assert content.index("first correction") < content.index("second correction")
    assert agent._drain_pending_steer() is None
    assert not agent._interrupt_requested


@pytest.mark.asyncio
async def test_unsupported_agent_still_queues():
    runner, agent, sk = _make_runner(compression_in_flight=True, busy_input_mode="steer")
    agent._supports_compression_steer = False
    await runner._handle_message(_make_event("keep this"))
    agent.steer.assert_not_called()
    assert runner.adapters[Platform.TELEGRAM]._pending_messages[sk].text == "keep this"
    agent.interrupt.assert_not_called()
