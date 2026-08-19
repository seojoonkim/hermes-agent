from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from agent.runtime_resume import MemKraftResumeStore, build_incomplete_handoff
from gateway.config import Platform
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
            {"role": "tool", "content": "commit " + "a" * 40},
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
    await asyncio.sleep(0)
    assert len(delivered) == 1
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "dispatched"
    assert json.loads(
        (checkpoint.parent / f"{cross_scope.resume_token}.json").read_text(encoding="utf-8")
    )["status"] == "incomplete"
