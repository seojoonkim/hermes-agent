"""Reliability contracts for bounded, staggered gateway startup resume."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import (
    _startup_resume_concurrency,
    _startup_resume_jitter,
    _startup_resume_stagger,
)
from gateway.session import SessionEntry
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _pending_entry(runner, chat_id: str) -> SessionEntry:
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
        runtime_resume_handoff={"resume_token": "resume-token", "next_dispatch_intent": "resume_session"},
    )


def test_startup_resume_config_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("HERMES_STARTUP_RESUME_CONCURRENCY", raising=False)
    monkeypatch.delenv("HERMES_STARTUP_RESUME_STAGGER", raising=False)
    monkeypatch.delenv("HERMES_STARTUP_RESUME_JITTER", raising=False)
    assert _startup_resume_concurrency() == 2
    assert _startup_resume_stagger() == 0.25
    assert _startup_resume_jitter() == 0.25

    monkeypatch.setenv("HERMES_STARTUP_RESUME_CONCURRENCY", "999")
    monkeypatch.setenv("HERMES_STARTUP_RESUME_STAGGER", "-4")
    monkeypatch.setenv("HERMES_STARTUP_RESUME_JITTER", "bad")
    assert _startup_resume_concurrency() == 16
    assert _startup_resume_stagger() == 0.0
    assert _startup_resume_jitter() == 0.25


@pytest.mark.asyncio
async def test_startup_resume_is_nonblocking_bounded_and_staggered(monkeypatch):
    runner, adapter = make_restart_runner()
    entries = [_pending_entry(runner, str(index)) for index in range(4)]
    runner.session_store._entries = {entry.session_key: entry for entry in entries}

    active = 0
    peak = 0
    releases = [asyncio.Event() for _ in entries]
    started: list[str] = []

    async def handle(event):
        nonlocal active, peak
        index = int(event.source.chat_id)
        started.append(event.source.chat_id)
        active += 1
        peak = max(peak, active)
        try:
            await releases[index].wait()
        finally:
            active -= 1

    sleep_calls: list[float] = []

    async def controlled_sleep(delay: float):
        sleep_calls.append(delay)
        await asyncio.sleep(0)

    adapter.handle_message = AsyncMock(side_effect=handle)
    runner._startup_resume_concurrency_override = 2
    runner._startup_resume_stagger_override = 0.5
    runner._startup_resume_jitter_override = 0.25
    runner._startup_resume_random = lambda: 0.4  # delay = .5 + (.25 * .4) = .6
    runner._startup_resume_sleep = controlled_sleep

    assert runner._schedule_resume_pending_sessions() == 4
    assert started == []  # scheduling itself never enters adapter code

    for _ in range(8):
        await asyncio.sleep(0)
    assert started == ["0", "1"]
    assert peak == 2
    assert sleep_calls == [pytest.approx(0.6)]

    releases[0].set()
    releases[1].set()
    for _ in range(8):
        await asyncio.sleep(0)
    assert started == ["0", "1", "2", "3"]
    assert peak == 2
    releases[2].set()
    releases[3].set()
    await asyncio.gather(*list(runner._background_tasks))
    assert runner._startup_resume_inflight == set()


@pytest.mark.asyncio
async def test_startup_resume_failure_does_not_block_later_jobs():
    runner, adapter = make_restart_runner()
    entries = [_pending_entry(runner, str(index)) for index in range(3)]
    runner.session_store._entries = {entry.session_key: entry for entry in entries}
    seen: list[str] = []

    async def handle(event):
        seen.append(event.source.chat_id)
        if event.source.chat_id == "0":
            raise RuntimeError("one broken resume")

    adapter.handle_message = AsyncMock(side_effect=handle)
    runner._startup_resume_concurrency_override = 1
    runner._startup_resume_stagger_override = 0
    runner._startup_resume_jitter_override = 0

    assert runner._schedule_resume_pending_sessions() == 3
    await asyncio.gather(*list(runner._background_tasks))
    assert seen == ["0", "1", "2"]
    assert runner._startup_resume_inflight == set()


@pytest.mark.asyncio
async def test_startup_resume_supervisor_cancellation_cleans_workers_and_dedupe():
    runner, adapter = make_restart_runner()
    entries = [_pending_entry(runner, str(index)) for index in range(3)]
    runner.session_store._entries = {entry.session_key: entry for entry in entries}
    entered = asyncio.Event()

    async def handle(_event):
        entered.set()
        await asyncio.Event().wait()

    adapter.handle_message = AsyncMock(side_effect=handle)
    runner._startup_resume_concurrency_override = 1
    runner._startup_resume_stagger_override = 0
    runner._startup_resume_jitter_override = 0

    assert runner._schedule_resume_pending_sessions() == 3
    await entered.wait()
    tasks = list(runner._background_tasks)
    assert len(tasks) == 1
    tasks[0].cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)

    assert runner._startup_resume_inflight == set()
    assert not [task for task in asyncio.all_tasks() if not task.done() and "startup_resume" in task.get_name()]
