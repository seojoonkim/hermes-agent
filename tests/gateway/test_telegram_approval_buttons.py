"""Tests for Telegram inline keyboard approval buttons."""

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram import adapter as telegram_adapter_module
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    """Create a TelegramAdapter with mocked internals."""
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._bot.get_me = AsyncMock(
        return_value=SimpleNamespace(id=123456789, username="hermes_test_bot")
    )
    adapter._app = MagicMock()
    return adapter


class _AuthRunner:
    """Minimal runner shim for callback auth tests."""

    def __init__(self, authorized: bool):
        self.authorized = authorized
        self.last_source = None

    async def _handle_message(self, event):
        return None

    def _is_user_authorized(self, source):
        self.last_source = source
        return self.authorized


# ===========================================================================
# send_exec_approval — inline keyboard buttons
# ===========================================================================

class TestTelegramExecApproval:
    """Test the send_exec_approval method sends InlineKeyboard buttons."""

    @pytest.mark.asyncio
    async def test_sends_inline_keyboard(self):
        adapter = _make_adapter()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        adapter._bot.send_message = AsyncMock(return_value=mock_msg)

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="rm -rf /important",
            session_key="agent:main:telegram:group:12345:99",
            description="dangerous deletion",
        )

        assert result.success is True
        assert result.message_id == "42"

        adapter._bot.send_message.assert_called_once()
        kwargs = adapter._bot.send_message.call_args[1]
        assert kwargs["chat_id"] == 12345
        assert "rm -rf /important" in kwargs["text"]
        assert "dangerous deletion" in kwargs["text"]
        assert kwargs["reply_markup"] is not None  # InlineKeyboardMarkup

    @pytest.mark.asyncio
    async def test_registers_approval_before_card_can_be_clicked(self):
        """An immediate Telegram tap must find its approval routing state."""
        adapter = _make_adapter()
        session_key = "agent:main:telegram:dm:12345"

        async def send_while_user_can_tap(**kwargs):
            assert list(adapter._approval_state.values()) == [session_key]
            return SimpleNamespace(message_id=42)

        adapter._send_message_with_thread_fallback = send_while_user_can_tap

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="<write to AGENTS.md>",
            session_key=session_key,
            description="protected instruction write",
            allow_permanent=False,
            allow_session=False,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_immediate_click_resolves_while_card_send_is_in_flight(self):
        """A tap arriving before send_exec_approval resumes must still route."""
        adapter = _make_adapter()
        session_key = "agent:main:telegram:dm:12345"
        send_started = asyncio.Event()
        finish_send = asyncio.Event()

        async def delayed_send(**kwargs):
            send_started.set()
            await finish_send.wait()
            return SimpleNamespace(message_id=42)

        adapter._send_message_with_thread_fallback = delayed_send
        send_task = asyncio.create_task(
            adapter.send_exec_approval(
                chat_id="12345",
                command="<write to AGENTS.md>",
                session_key=session_key,
                description="protected instruction write",
                allow_permanent=False,
                allow_session=False,
            )
        )
        await send_started.wait()

        # Reproduce the real race: Telegram delivers a callback while the
        # send coroutine has not yet resumed from send_message.
        approval_id = next(iter(adapter._approval_state))
        query = AsyncMock()
        query.data = f"ea:once:{approval_id}"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.message.message_thread_id = None
        query.from_user = MagicMock()
        query.from_user.id = "12345"
        query.from_user.first_name = "Simon"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock()
        update.callback_query = query

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
                await adapter._handle_callback_query(update, MagicMock())

        resolve.assert_called_once_with(session_key, "once")
        assert adapter._approval_state == {}
        assert "Approved once" in query.edit_message_text.call_args.kwargs["text"]

        finish_send.set()
        result = await send_task
        assert result.success is True
        assert adapter._approval_state == {}

    @pytest.mark.asyncio
    async def test_failed_approval_send_removes_preregistered_state(self):
        """A failed card send must not leave an unclickable approval route."""
        adapter = _make_adapter()

        async def fail_send(**kwargs):
            raise RuntimeError("telegram send failed")

        adapter._send_message_with_thread_fallback = fail_send

        result = await adapter.send_exec_approval(
            chat_id="12345",
            command="<write to AGENTS.md>",
            session_key="agent:main:telegram:dm:12345",
        )

        assert result.success is False
        assert adapter._approval_state == {}

    @pytest.mark.asyncio
    async def test_callback_survives_adapter_rebuild(self):
        """A replacement adapter must resolve a card sent by its predecessor."""
        sender = _make_adapter()
        sender._send_message_with_thread_fallback = AsyncMock(
            return_value=SimpleNamespace(message_id=42)
        )
        session_key = "agent:main:telegram:group:-100:7"
        await sender.send_exec_approval(
            chat_id="-100", command="<write to AGENTS.md>", session_key=session_key,
        )
        approval_id = next(iter(sender._approval_state))

        replacement = _make_adapter()
        query = AsyncMock()
        query.data = f"ea:once:{approval_id}"
        query.message = MagicMock()
        query.message.chat_id = -100
        query.message.chat.type = "group"
        query.message.message_thread_id = None
        query.from_user = MagicMock(id="7", first_name="Simon")
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        update = MagicMock(callback_query=query)

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as resolve:
                await replacement._handle_callback_query(update, MagicMock())

        resolve.assert_called_once_with(session_key, "once")
        assert approval_id not in telegram_adapter_module._TELEGRAM_APPROVAL_ROUTES


    @pytest.mark.asyncio
    async def test_non_smart_allow_permanent_false_keeps_session(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        buttons = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: buttons.append(text) or text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup", lambda rows: rows
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
            allow_permanent=False,
        )

        assert buttons == ["✅ Allow Once", "✅ Session", "❌ Deny"]

    @pytest.mark.asyncio
    async def test_full_approval_keyboard_is_two_by_two(self, monkeypatch):
        """Regression: d48bf743f flattened all buttons into one row (4x1)."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        captured_rows = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
        )

        assert captured_rows == [
            ["✅ Allow Once", "✅ Session"],
            ["✅ Always", "❌ Deny"],
        ]


    @pytest.mark.asyncio
    async def test_smart_deny_two_buttons_share_one_row(self, monkeypatch):
        """smart_deny yields 2 buttons — they pair into a single readable row."""
        adapter = _make_adapter()
        adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
        captured_rows = []
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            lambda text, callback_data: text,
        )
        monkeypatch.setattr(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            lambda rows: captured_rows.extend(rows) or rows,
        )

        await adapter.send_exec_approval(
            chat_id="12345", command="curl example.test", session_key="s",
            allow_permanent=False, smart_denied=True,
        )

        assert captured_rows == [
            ["✅ Allow Once", "❌ Deny"],
        ]


    @pytest.mark.asyncio
    async def test_send_update_prompt_escapes_dynamic_prompt(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=55)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_update_prompt(
            chat_id="12345",
            prompt="Fix [issue]_1 and verify *markdown*",
            default="alpha_beta",
            metadata={"thread_id": "999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "Fix \\[issue\\]\\_1" in sent["text"]
        assert "alpha\\_beta" in sent["text"]

# _handle_callback_query — approval button clicks
# ===========================================================================

class TestTelegramApprovalCallback:
    """Test the approval callback handling in _handle_callback_query."""


    @pytest.mark.asyncio
    async def test_resume_typing_after_inline_approval(self):
        """Clicking an inline approval button must un-pause the chat's typing.

        Regression for #27853: the text /approve path resumed typing, but the
        ea: callback path did not, so the typing indicator stayed gone for the
        rest of a long-running turn after a button click.
        """
        adapter = _make_adapter()
        adapter._approval_state[5] = "agent:main:telegram:group:12345:99"
        adapter.pause_typing_for_chat("12345")
        assert "12345" in adapter._typing_paused

        query = AsyncMock()
        query.data = "ea:once:5"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Norbert"
        query.from_user.id = "12345"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        assert "12345" not in adapter._typing_paused


    @pytest.mark.asyncio
    async def test_approval_callback_escapes_dynamic_user_name(self):
        adapter = _make_adapter()
        adapter._approval_state[3] = "agent:main:telegram:group:12345:99"

        query = AsyncMock()
        query.data = "ea:once:3"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.first_name = "Alice_Bob"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        query.from_user.id = "12345"

        with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
            with patch("tools.approval.resolve_gateway_approval", return_value=1):
                await adapter._handle_callback_query(update, context)

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "Alice\\_Bob" in edit_kwargs["text"]
        assert "Approved once" in edit_kwargs["text"]


    @pytest.mark.asyncio
    async def test_update_prompt_callback_not_affected(self, tmp_path):
        """Ensure update prompt callbacks still work."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 123
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
                # Allow the caller — the new fail-closed allowlist gate
                # (#24457) rejects empty TELEGRAM_ALLOWED_USERS, but this
                # test isn't exercising that gate; it's verifying the
                # update_prompt callback still writes the response.
                with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}):
                    await adapter._handle_callback_query(update, context)

        # Should NOT have triggered approval resolution
        mock_resolve.assert_not_called()
        assert (tmp_path / ".update_response").read_text() == "y"

    @pytest.mark.asyncio
    async def test_update_prompt_callback_rejects_unauthorized_user(self, tmp_path):
        """Update prompt buttons should honor TELEGRAM_ALLOWED_USERS."""
        adapter = _make_adapter()

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.from_user.id = 222
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "111"}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()

    @pytest.mark.asyncio
    async def test_update_prompt_callback_rejects_user_blocked_by_global_allowlist(self, tmp_path):
        adapter = _make_adapter()
        runner = _AuthRunner(authorized=False)
        adapter._message_handler = runner._handle_message

        query = AsyncMock()
        query.data = "update_prompt:y"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.message.chat.type = "private"
        query.from_user = MagicMock()
        query.from_user.id = 222
        query.from_user.first_name = "Mallory"
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": ""}):
                await adapter._handle_callback_query(update, context)

        query.answer.assert_called_once()
        assert "not authorized" in query.answer.call_args[1]["text"].lower()
        query.edit_message_text.assert_not_called()
        assert not (tmp_path / ".update_response").exists()
        assert runner.last_source is not None
        assert runner.last_source.platform == Platform.TELEGRAM
        assert runner.last_source.user_id == "222"

