"""Telegram connector-account identity discovery and inbound stamping tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", extra={}))
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._should_process_message = lambda _message, **_kwargs: True
    return adapter


def _message(text="hello"):
    return SimpleNamespace(
        chat=SimpleNamespace(id=100, type="private", title=None, full_name="User", is_forum=False),
        from_user=SimpleNamespace(id=7, full_name="User", username="user", is_bot=False),
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=11,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
    )


def test_account_id_is_normalized_bot_user_id_and_stamps_message_source():
    adapter = _adapter()
    adapter._note_bot_account_id(" 123456 ")

    event = adapter._build_message_event(_message(), MessageType.TEXT)

    assert adapter.account_id == "123456"
    assert event.source.account_id == "123456"


def test_media_and_topic_sources_share_the_bot_account_id():
    adapter = _adapter()
    adapter._note_bot_account_id(123456)
    msg = _message()
    msg.chat = SimpleNamespace(
        id=-100200, type="supergroup", title="Forum", full_name=None, is_forum=True
    )
    msg.message_thread_id = 44
    msg.is_topic_message = True

    event = adapter._build_message_event(msg, MessageType.PHOTO)

    assert event.source.thread_id == "44"
    assert event.source.account_id == "123456"


@pytest.mark.asyncio
async def test_account_identity_is_discovered_from_get_me_before_inbound_dispatch():
    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=None,
        username="not-an-account-id",
        get_me=AsyncMock(return_value=SimpleNamespace(id=987654, username="hermes_bot")),
    )
    captured = []
    adapter._enqueue_text_event = captured.append
    msg = _message()
    update = SimpleNamespace(effective_message=msg, message=msg, update_id=42)

    await adapter._handle_text_message(update, MagicMock())
    adapter._bot.get_me.assert_awaited_once()
    event = captured[0]
    assert adapter.account_id == "987654"
    assert event.source.account_id == "987654"


@pytest.mark.asyncio
async def test_unknown_account_identity_fails_closed_before_text_dispatch():
    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=None,
        username="hermes_bot",
        get_me=AsyncMock(return_value=SimpleNamespace(id="   ", username="hermes_bot")),
    )
    adapter._enqueue_text_event = MagicMock()
    msg = _message()
    update = SimpleNamespace(effective_message=msg, message=msg, update_id=42)

    await adapter._handle_text_message(update, MagicMock())

    adapter._enqueue_text_event.assert_not_called()
    assert adapter.account_id is None


def test_callback_authorization_source_is_stamped_with_account_id():
    adapter = _adapter()
    adapter._note_bot_account_id(24680)
    class Runner:
        def __init__(self):
            self.auth = MagicMock(return_value=True)

        async def on_message(self, _event):
            pass

        def _is_user_authorized(self, source):
            return self.auth(source)

    runner = Runner()
    adapter._message_handler = runner.on_message

    assert adapter._is_callback_user_authorized("7", chat_id="100") is True

    source = runner.auth.call_args.args[0]
    assert source.account_id == "24680"


@pytest.mark.parametrize("invalid_id", ["not-numeric", "123.0", "１２３", "", None])
def test_non_ascii_numeric_bot_id_is_rejected(invalid_id):
    adapter = _adapter()

    adapter._note_bot_account_id(invalid_id)

    assert adapter.account_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_account_id", [None, "123456"])
async def test_connected_adapter_without_bot_identity_fails_closed(cached_account_id):
    adapter = _adapter()
    adapter._running = True
    adapter._bot = None
    adapter._bot_account_id = cached_account_id

    assert await adapter._ensure_account_identity() is False


@pytest.mark.asyncio
async def test_preconnect_direct_builder_without_bot_remains_compatible():
    adapter = _adapter()
    adapter._bot = None

    assert await adapter._ensure_account_identity() is True


def test_bot_username_token_and_chat_are_never_used_as_account_id():
    adapter = _adapter()
    adapter._bot = SimpleNamespace(id=None, username="hermes_bot")

    assert adapter.account_id is None
    assert adapter.config.token == "fake-token"
    assert adapter._build_message_event(_message(), MessageType.TEXT).source.account_id is None
