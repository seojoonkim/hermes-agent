"""Regression contracts for stale restart replay and explicit startup resume."""

from datetime import datetime

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _pending_entry(runner, chat_id: str = "same-chat", *, handoff: dict | None = None) -> SessionEntry:
    source = make_restart_source(chat_id=chat_id)
    now = datetime.now()
    return SessionEntry(
        session_key=runner._session_key_for_source(source),
        session_id=f"sid-{chat_id}",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
        last_resume_marked_at=now,
        runtime_resume_handoff=handoff,
    )


@pytest.mark.asyncio
async def test_startup_does_not_replay_plain_pending_request():
    """A plain pending bit must not synthesize a fresh turn at startup."""
    runner, adapter = make_restart_runner()
    entry = _pending_entry(runner)
    runner.session_store._entries = {entry.session_key: entry}
    adapter.handle_message = AsyncMock(return_value=None)

    assert runner._schedule_resume_pending_sessions() == 0
    await asyncio.sleep(0)
    assert entry.resume_pending is True


@pytest.mark.asyncio
async def test_startup_replays_only_runtime_handoff_once():
    runner, adapter = make_restart_runner()
    handoff = {
        "resume_token": "resume-token-1",
        "incomplete_goal": "finish the interrupted turn",
        "next_dispatch_intent": "continue",
    }
    entry = _pending_entry(runner, handoff=handoff)
    entry.resume_reason = "runtime_max_iterations"
    runner.session_store._entries = {entry.session_key: entry}
    adapter.handle_message = AsyncMock(return_value=None)

    assert runner._schedule_resume_pending_sessions() == 1
    assert runner._schedule_resume_pending_sessions() == 0
    await asyncio.sleep(0)

    resumed_event: MessageEvent = adapter.handle_message.await_args_list[0].args[0]
    assert resumed_event.internal is True
    assert resumed_event.message_type == MessageType.TEXT
    assert resumed_event.text.startswith("[Hermes durable resume handoff")
    assert resumed_event.raw_message["hermes_runtime_resume"]["resume_token"] == "resume-token-1"
