"""Routine restart admission must recover without killing or losing work."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_long_after_turn_budget_is_bounded(monkeypatch):
    runner, _ = make_restart_runner()
    runner._restart_after_turn_timeout = 1800
    runner._running_agents['busy'] = MagicMock()
    runner.stop = AsyncMock()
    # Advance only the clock consulted by the wait, not asyncio's scheduler.
    import gateway.run as module
    clock = MagicMock()
    import itertools
    clock.time.side_effect = itertools.chain([0, 0, 21], itertools.repeat(1801))
    monkeypatch.setattr(module.asyncio, 'get_running_loop', lambda: clock)
    assert runner.request_restart()
    await runner._restart_task
    assert not runner._draining
    assert not runner._restart_requested
    runner.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('before_start', [True, False])
async def test_cancelled_restart_reopens_admission(before_start):
    runner, _ = make_restart_runner()
    runner._running_agents['busy'] = MagicMock()
    assert runner.request_restart()
    if not before_start:
        await asyncio.sleep(0)
    runner._restart_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner._restart_task
    await asyncio.sleep(0)
    assert not runner._draining
    assert not runner._restart_task_started


@pytest.mark.asyncio
async def test_restart_wait_exception_reopens_admission():
    runner, _ = make_restart_runner()
    runner._await_active_work_before_restart = AsyncMock(side_effect=RuntimeError('probe'))
    assert runner.request_restart()
    with pytest.raises(RuntimeError):
        await runner._restart_task
    await asyncio.sleep(0)
    assert not runner._draining


@pytest.mark.asyncio
async def test_shutdown_owns_drain_when_routine_wait_finishes():
    runner, _ = make_restart_runner()
    runner._restart_after_turn_timeout = 0
    runner._running_agents['busy'] = MagicMock()
    assert runner.request_restart()
    runner._update_runtime_status.reset_mock()
    runner._draining = True  # explicit shutdown takes ownership
    runner._stop_task = asyncio.create_task(asyncio.sleep(0))
    await runner._restart_task
    assert runner._draining
    assert ('running',) not in [c.args for c in runner._update_runtime_status.call_args_list]
    await runner._stop_task


@pytest.mark.asyncio
async def test_aborted_restart_replays_idle_fifo_once_without_touching_busy():
    runner, adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 0
    source = make_restart_source()
    key = runner._session_key_for_source(source)
    events = [MessageEvent(text=str(i), message_type=MessageType.TEXT, source=source, message_id=str(i)) for i in range(3)]
    for event in events:
        runner._enqueue_fifo(key, event, adapter)
    busy_source = make_restart_source(chat_id='other')
    busy_key = runner._session_key_for_source(busy_source)
    busy_event = MessageEvent(text='busy', source=busy_source)
    runner._running_agents[busy_key] = MagicMock()
    adapter._pending_messages[busy_key] = busy_event
    seen = []
    async def handle(event):
        seen.append(event)
    adapter.handle_message = handle
    assert runner.request_restart()
    await runner._restart_task
    await asyncio.sleep(0)
    replay = getattr(runner, '_restart_replay_task', None)
    if replay:
        await replay
    assert seen == events
    assert adapter._pending_messages[busy_key] is busy_event
    assert key not in adapter._pending_messages
    assert not runner._session_state(key).conversation.queued_events


@pytest.mark.asyncio
async def test_restart_refused_during_external_drain_or_shutdown():
    runner, _ = make_restart_runner()
    runner._external_drain_active = True
    assert runner.request_restart() is False
    runner._external_drain_active = False
    runner._draining = True
    assert runner.request_restart() is False
    runner._draining = False
    runner._running = False
    runner._shutdown_event.set()
    assert runner.request_restart() is False


def test_routine_budget_preserves_short_and_zero_values():
    runner, _ = make_restart_runner()
    for configured, expected in [(0, 0), (0.2, 0.2), (1800, 20)]:
        runner._restart_after_turn_timeout = configured
        assert runner._routine_restart_admission_timeout() == expected


@pytest.mark.asyncio
async def test_external_drain_still_wins_on_routine_abort():
    runner, _ = make_restart_runner()
    runner._restart_after_turn_timeout = 0
    runner._running_agents['busy'] = MagicMock()
    assert runner.request_restart()
    runner._external_drain_active = True
    await runner._restart_task
    assert not runner._draining
    assert runner._external_drain_active
    runner._update_runtime_status.assert_called_with('draining')


@pytest.mark.asyncio
async def test_busy_message_is_queued_normally_while_restart_waits():
    runner, adapter = make_restart_runner()
    runner._busy_input_mode = 'queue'
    source = make_restart_source()
    key = runner._session_key_for_source(source)
    runner._running_agents[key] = MagicMock()
    runner._restart_after_turn_timeout = 0.2
    event = MessageEvent(text='new task during restart', source=source, message_id='new')
    assert runner.request_restart()
    await asyncio.sleep(0)
    response = await runner._handle_message(event)
    assert response is None
    assert adapter._pending_messages[key] is event
    assert not any('restarting' in text for text in adapter.sent)
    await runner._restart_task


@pytest.mark.asyncio
async def test_replay_profile_routing_and_failure_preserve_fifo():
    from tests.gateway.restart_test_helpers import RestartTestAdapter
    runner, root_adapter = make_restart_runner()
    profile_adapter = RestartTestAdapter()
    runner._profile_adapters = {'sion': {profile_adapter.platform: profile_adapter}}
    source = make_restart_source()
    source.profile = 'sion'
    key = runner._session_key_for_source(source)
    events = [MessageEvent(text=str(i), source=source, message_id=str(i), allow_gateway_control=False) for i in range(2)]
    for event in events:
        runner._enqueue_fifo(key, event, profile_adapter)
    profile_adapter.handle_message = AsyncMock(side_effect=RuntimeError('adapter unavailable'))
    root_adapter.handle_message = AsyncMock()
    with pytest.raises(RuntimeError):
        await runner._replay_idle_restart_pending()
    assert profile_adapter._pending_messages[key] is events[0]
    assert runner._session_state(key).conversation.queued_events == [events[1]]
    root_adapter.handle_message.assert_not_awaited()
    profile_adapter.handle_message = AsyncMock()
    await runner._replay_idle_restart_pending()
    await runner._replay_idle_restart_pending()
    assert [call.args[0] for call in profile_adapter.handle_message.await_args_list] == events
    assert all(not event.allow_gateway_control for event in events)
