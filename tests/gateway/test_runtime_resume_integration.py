from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from agent.runtime_resume import (
    MemKraftResumeStore,
    build_incomplete_handoff,
    build_resume_prompt,
)
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_verified_incomplete_turn_is_restored_once_in_exact_gateway_scope(tmp_path):
    """Delivery arms one durable, scope-bound internal turn for a fresh runner."""
    source = make_restart_source(chat_id="resume-chat", thread_id="topic-7")
    source.profile = "sano"
    session_key = "agent:sano:telegram:dm:resume-chat:thread:topic-7"
    session_id = "session-1"
    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    store = MemKraftResumeStore(tmp_path)

    first, first_adapter = make_restart_runner()
    first._runtime_resume_store = store
    first._active_profile_name = lambda: "sano"
    first._session_key_for_source = lambda _source: session_key
    first.session_store._entries = {session_key: entry}

    result = {
        "completed": False,
        "turn_exit_reason": "max_iterations_reached(90/90)",
        "final_response": "Verified progress; adapter remains incomplete.",
        "messages": [
            {"role": "user", "content": "finish adapter"},
            {"role": "tool", "content": "verified progress without a git commit"},
        ],
    }
    token = first._arm_runtime_resume_after_delivery(
        agent_result=result,
        source=source,
        session_entry=entry,
        session_key=session_key,
        run_generation=1,
        incomplete_goal="finish adapter",
    )
    assert token
    checkpoint = tmp_path / "tasks" / "hermes-resume" / f"{token}.json"
    assert not checkpoint.exists(), "checkpoint must not precede verified delivery"

    first._schedule_durable_resume_checkpoints = lambda: 0
    callback = first_adapter.pop_post_delivery_callback(session_key, generation=1)
    assert callback is not None
    callback()
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "incomplete"

    # Untrusted neighboring records must not escape their exact profile/session/channel.
    cross_scope = build_incomplete_handoff(
        profile="other",
        session_id=session_id,
        channel_key=session_key,
        turn_exit_reason="max_iterations_reached(90/90)",
        incomplete_goal="wrong profile",
        messages=[{"role": "tool", "content": "commit " + "b" * 40}],
    )
    assert store.persist(cross_scope)
    malformed = checkpoint.parent / ("f" * 32 + ".json")
    malformed.write_text("{broken", encoding="utf-8")

    fresh, fresh_adapter = make_restart_runner()
    fresh._runtime_resume_store = MemKraftResumeStore(tmp_path)
    fresh._active_profile_name = lambda: "sano"
    fresh._session_key_for_source = lambda _source: session_key
    fresh.session_store._entries = {session_key: entry}
    delivered = []

    async def capture(event):
        delivered.append(event)

    fresh_adapter.handle_message = capture

    assert fresh._schedule_durable_resume_checkpoints() == 1
    await asyncio.sleep(0)
    assert len(delivered) == 1
    event = delivered[0]
    assert event.internal is True
    assert event.source.profile == "sano"
    assert event.source.chat_id == "resume-chat"
    assert event.source.thread_id == "topic-7"
    assert token in event.text
    assert "finish adapter" in event.text

    assert fresh._schedule_durable_resume_checkpoints() == 0
    for _ in range(3):
        await asyncio.sleep(0)
    assert len(delivered) == 1
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "dispatched"
    assert json.loads(
        (checkpoint.parent / f"{cross_scope.resume_token}.json").read_text(encoding="utf-8")
    )["status"] == "incomplete"


def test_runtime_resume_scope_includes_account_route_and_validates_session_lineage(tmp_path):
    source = make_restart_source(chat_id="chat:colon", thread_id="topic/7")
    source.profile = "sano"
    source.scope_id = "workspace:blue"
    source.chat_type = "group"
    source.user_id = "user:42"
    session_key = "agent:sano:telegram:group:workspace:blue:chat:colon:topic/7"
    session_id = "session-1"
    now = datetime.now()
    entry = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner, adapter = make_restart_runner()
    runner._runtime_resume_store = MemKraftResumeStore(tmp_path)
    runner._active_profile_name = lambda: "sano"
    runner._session_key_for_source = lambda _source: session_key
    runner._runtime_resume_account_id = lambda _source, _adapter=None: "bot:prod/1"
    runner.session_store._entries = {session_key: entry}

    token = runner._arm_runtime_resume_after_delivery(
        agent_result={
            "completed": False,
            "turn_exit_reason": "max_iterations_reached(90/90)",
            "final_response": "partial",
            "messages": [{"role": "tool", "content": "commit " + "a" * 40}],
        },
        source=source,
        session_entry=entry,
        session_key=session_key,
        run_generation=1,
        incomplete_goal="finish adapter",
    )
    assert token
    callback = adapter.pop_post_delivery_callback(session_key, generation=1)
    assert callback is not None
    callback()

    stale_source = make_restart_source(chat_id="chat:colon", thread_id="topic/7")
    stale_source.profile = "sano"
    stale_source.scope_id = "workspace:blue"
    stale_entry = SessionEntry(
        session_key=session_key,
        session_id="different-session",
        created_at=now,
        updated_at=now,
        origin=stale_source,
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    fresh, _fresh_adapter = make_restart_runner()
    fresh._runtime_resume_store = MemKraftResumeStore(tmp_path)
    fresh._active_profile_name = lambda: "sano"
    fresh._session_key_for_source = lambda _source: session_key
    fresh._runtime_resume_account_id = lambda _source, _adapter=None: "bot:prod/1"
    fresh.session_store._entries = {session_key: stale_entry}
    assert fresh._schedule_durable_resume_checkpoints() == 0
    assert json.loads((tmp_path / "tasks" / "hermes-resume" / f"{token}.json").read_text())["status"] == "incomplete"


@pytest.mark.asyncio
async def test_completed_max_turn_delivery_arms_resume_from_production_turn_path():
    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="resume-chat")
    event = MessageEvent(text="finish adapter", message_type=MessageType.TEXT, source=source)
    entry = SessionEntry(
        session_key="session-key",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner._async_session_store = MagicMock()
    runner._async_session_store.get_or_create_session.return_value = entry
    runner._session_key_for_source = lambda _source: "session-key"
    runner._arm_runtime_resume_after_delivery = MagicMock(return_value="t" * 32)
    agent_result = {
        "completed": False,
        "turn_exit_reason": "max_iterations_reached(90/90)",
        "final_response": "partial",
    }

    await runner._run_post_turn_hooks(
        agent_result=agent_result,
        source=source,
        is_internal=False,
        event=event,
    )

    runner._arm_runtime_resume_after_delivery.assert_called_once()


@pytest.mark.asyncio
async def test_internal_resume_can_rearm_one_bounded_progress_turn():
    runner, _adapter = make_restart_runner()
    source = make_restart_source(chat_id="resume-chat")
    entry = SessionEntry(
        session_key="session-key",
        session_id="session-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner._async_session_store = MagicMock()
    runner._async_session_store.get_or_create_session.return_value = entry
    runner._session_key_for_source = lambda _source: "session-key"
    runner._arm_runtime_resume_after_delivery = MagicMock(return_value="n" * 32)
    original = build_incomplete_handoff(
        profile="sano",
        session_id="session-1",
        channel_key="json:{}",
        turn_exit_reason="max_iterations_reached(90/90)",
        incomplete_goal="finish adapter",
        messages=[],
    )
    event = MessageEvent(
        text=build_resume_prompt(original.to_dict()),
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
    )

    await runner._run_post_turn_hooks(
        agent_result={
            "completed": False,
            "turn_exit_reason": "max_iterations_reached(90/90)",
            "final_response": "more verified progress",
        },
        source=source,
        is_internal=True,
        event=event,
    )

    kwargs = runner._arm_runtime_resume_after_delivery.call_args.kwargs
    assert kwargs["prior_token"] == original.resume_token
    assert kwargs["chain_depth"] == 1
    assert kwargs["incomplete_goal"] == "finish adapter"
