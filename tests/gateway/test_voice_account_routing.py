"""Behavior contracts for account-scoped Discord voice sibling paths."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from gateway.config import Platform
from gateway.run import GatewayRunner, TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


def _source(account_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="text-channel",
        chat_type="channel",
        user_id="user",
        account_id=account_id,
    )


def test_voice_transcript_dedupe_is_isolated_by_account():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._recent_voice_transcripts = {}

    assert runner._is_duplicate_voice_transcript(10, 20, "same transcript", "bot-a") is False
    assert runner._is_duplicate_voice_transcript(10, 20, "same transcript", "bot-a") is True
    assert runner._is_duplicate_voice_transcript(10, 20, "same transcript", "bot-b") is False


def test_voice_channel_sidecar_uses_source_account_adapter():
    runner = GatewayRunner.__new__(GatewayRunner)
    source = _source("bot-b")
    expected = MagicMock()
    expected.get_voice_channel_context.return_value = "Simon speaking"
    primary = MagicMock()
    runner.adapters = {Platform.DISCORD: primary}
    runner._adapter_for_source = MagicMock(return_value=expected)
    runner._get_guild_id = MagicMock(return_value=42)
    state = SimpleNamespace(conversation=SimpleNamespace(vc_last=""))
    runner._session_state = MagicMock(return_value=state)

    note = runner._voice_channel_sidecar_note(object(), source, "session-key")

    assert note == "[Voice channel now: Simon speaking]"
    runner._adapter_for_source.assert_called_once_with(source)
    expected.get_voice_channel_context.assert_called_once_with(42)
    primary.get_voice_channel_context.assert_not_called()


def test_pre_tool_voice_ack_uses_source_account_adapter():
    runner = GatewayRunner.__new__(GatewayRunner)
    source = _source("bot-b")
    expected = MagicMock()
    acknowledgement = object()
    expected.play_ack_in_voice.return_value = acknowledgement
    primary = MagicMock()
    runner.adapters = {Platform.DISCORD: primary}
    runner._adapter_for_source = MagicMock(return_value=expected)
    ctx = SimpleNamespace(
        source=source,
        _voice_ack_fired=[False],
        _voice_ack_guild=[42],
        _voice_ack_loop=object(),
        _run_still_current=lambda: True,
    )

    with patch("gateway.run.safe_schedule_threadsafe") as schedule:
        TurnRunner(runner, cast(TurnContext, ctx)).voice_ack_callback(
            "call", "terminal", {}
        )

    assert ctx._voice_ack_fired == [True]
    runner._adapter_for_source.assert_called_once_with(source)
    expected.play_ack_in_voice.assert_called_once_with(42)
    schedule.assert_called_once()
    assert schedule.call_args.args[0] is acknowledgement
    primary.play_ack_in_voice.assert_not_called()
