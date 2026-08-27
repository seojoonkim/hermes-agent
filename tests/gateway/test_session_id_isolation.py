"""Regression tests for gateway session-id isolation across concurrent turns."""

import asyncio
import os

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    set_current_session_id,
    set_session_vars,
)


def test_gateway_context_binds_session_id_without_mutating_process_env(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated-process-session")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="chat-a",
        session_key="agent:main:telegram:sano:group:chat-a:user",
        session_id="session-a",
    )
    try:
        assert get_session_env("HERMES_SESSION_ID") == "session-a"
        set_current_session_id("session-a-compressed")
        assert get_session_env("HERMES_SESSION_ID") == "session-a-compressed"
        assert os.environ["HERMES_SESSION_ID"] == "unrelated-process-session"
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_concurrent_gateway_turns_cannot_overwrite_each_others_session_id(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "process-fallback")

    async def rotate(session_key, initial, compressed):
        tokens = set_session_vars(
            platform="telegram",
            chat_id=session_key,
            session_key=session_key,
            session_id=initial,
        )
        try:
            await asyncio.sleep(0)

            def rotate_and_read():
                set_current_session_id(compressed)
                return get_session_env("HERMES_SESSION_ID")

            return await asyncio.to_thread(rotate_and_read)
        finally:
            clear_session_vars(tokens)

    a, b = await asyncio.gather(
        rotate("channel-a", "session-a", "session-a-compressed"),
        rotate("channel-b", "session-b", "session-b-compressed"),
    )

    assert (a, b) == ("session-a-compressed", "session-b-compressed")
    assert os.environ["HERMES_SESSION_ID"] == "process-fallback"


def test_runner_sets_composite_turn_session_id():
    runner = GatewayRunner.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="42",
        account_id="sano",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.TELEGRAM],
        home_channels={},
        session_key="agent:main:telegram:sano:group:-1001:42",
        session_id="session-owned-by-channel-a",
    )

    tokens = runner._set_session_env(context)
    try:
        assert get_session_env("HERMES_SESSION_KEY") == context.session_key
        assert get_session_env("HERMES_SESSION_ID") == context.session_id
    finally:
        runner._clear_session_env(tokens)
