"""Tests for preserving media from the message a Telegram turn replies to.

Replying to a photo with "put this in my calendar" must carry that image into
the agent turn exactly once, reuse a bounded in-process cache instead of
re-downloading, leave direct attachments alone, and fail open when Telegram's
file API errors.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


def _adapter_class():
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except ModuleNotFoundError:  # PR branch before Telegram plugin extraction
        from gateway.platforms.telegram import TelegramAdapter
    return TelegramAdapter


def _make_adapter():
    TelegramAdapter = _adapter_class()
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={"allow_from": ["111"], "allowed_chats": ["111"]},
    )
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._message_handler = None
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 10.0
    adapter._text_batch_split_delay_seconds = 10.0
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._max_doc_bytes = 20 * 1024 * 1024
    adapter._replied_media_cache = type(adapter)._new_replied_media_cache()
    return adapter


class _FakeFile:
    def __init__(self, data, file_path):
        self._data = data
        self.file_path = file_path

    async def download_as_bytearray(self):
        return bytearray(self._data)


class _FakePhotoSize:
    """Largest PhotoSize of a replied-to photo; counts its own downloads."""

    def __init__(self, *, file_unique_id="uniq-1", data=b"\x89PNG\r\n\x1a\nreply", fail=False):
        self.file_unique_id = file_unique_id
        self.file_id = f"fileid-{file_unique_id}"
        self.file_size = len(data)
        self._data = data
        self._fail = fail
        self.download_count = 0

    async def get_file(self):
        self.download_count += 1
        if self._fail:
            raise RuntimeError("telegram cdn unreachable: bot123456:SECRET-TOKEN")
        return _FakeFile(self._data, "photos/reply.png")


def _blank_message(**overrides):
    msg = SimpleNamespace(
        message_id=101,
        text="이거 캘린더에 넣어줘",
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=111, type="private", title=None, full_name="Simon", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Simon", first_name="Simon"),
        reply_to_message=None,
        quote=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )
    for key, value in overrides.items():
        setattr(msg, key, value)
    return msg


def _reply_to_photo(photo, *, message_id=101, text="이거 캘린더에 넣어줘"):
    replied = _blank_message(
        message_id=100,
        text=None,
        caption="boarding-pass screenshot",
        photo=[photo],
    )
    return _blank_message(message_id=message_id, text=text, reply_to_message=replied)


@pytest.fixture
def stub_media_cache(monkeypatch):
    """Stub cache_media_bytes so no real bytes hit the on-disk media cache."""
    import gateway.platforms.base as base

    calls = []

    class _Cached:
        kind = "image"
        display_name = "reply.png"
        path = "/tmp/hermes-media-cache/reply.png"
        media_type = "image/png"

        def context_note(self):
            return f"[Image saved at: {self.path}]"

    def fake_cache_media_bytes(data, *, filename="", mime_type="", default_kind=None):
        calls.append((bytes(data), filename, mime_type, default_kind))
        return _Cached()

    monkeypatch.setattr(base, "cache_media_bytes", fake_cache_media_bytes)
    return calls


async def _drain(adapter):
    for task in list(adapter._pending_text_batch_tasks.values()):
        task.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_text_reply_to_photo_attaches_replied_image_once(stub_media_cache):
    """A text reply to a photo carries that photo into the event exactly once."""
    adapter = _make_adapter()
    photo = _FakePhotoSize()
    msg = _reply_to_photo(photo)
    update = SimpleNamespace(update_id=9001, message=msg, effective_message=msg)

    await adapter._handle_text_message(update, SimpleNamespace())

    assert len(stub_media_cache) == 1
    assert photo.download_count == 1
    event = next(iter(adapter._pending_text_batches.values()))
    assert event.reply_to_message_id == "100"
    assert event.media_urls == ["/tmp/hermes-media-cache/reply.png"]
    assert event.media_types == ["image/png"]
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls.count("/tmp/hermes-media-cache/reply.png") == 1
    await _drain(adapter)


@pytest.mark.asyncio
async def test_second_reply_to_same_photo_reuses_cache_without_redownload(stub_media_cache):
    """Replying again to the same photo must not re-download or re-cache it."""
    adapter = _make_adapter()
    photo = _FakePhotoSize()

    for idx, update_id in enumerate((9001, 9002)):
        msg = _reply_to_photo(photo, message_id=101 + idx, text=f"chunk {idx}")
        await adapter._handle_text_message(
            SimpleNamespace(update_id=update_id, message=msg, effective_message=msg),
            SimpleNamespace(),
        )

    assert photo.download_count == 1, "replied-to media re-downloaded"
    assert len(stub_media_cache) == 1, "replied-to media re-cached to disk"
    await _drain(adapter)


@pytest.mark.asyncio
async def test_batched_replies_to_same_photo_attach_media_once(stub_media_cache):
    """Text-batch merging must not duplicate the replied-to media in one event."""
    adapter = _make_adapter()
    photo = _FakePhotoSize()

    for idx, update_id in enumerate((9001, 9002, 9003)):
        msg = _reply_to_photo(photo, message_id=101, text=f"chunk {idx}")
        await adapter._handle_text_message(
            SimpleNamespace(update_id=update_id, message=msg, effective_message=msg),
            SimpleNamespace(),
        )

    assert len(adapter._pending_text_batches) == 1
    event = next(iter(adapter._pending_text_batches.values()))
    assert event.media_urls == ["/tmp/hermes-media-cache/reply.png"]
    assert event.media_types == ["image/png"]
    assert event.text.count("/tmp/hermes-media-cache/reply.png") == 1
    await _drain(adapter)


@pytest.mark.asyncio
async def test_replied_media_cache_is_bounded(stub_media_cache):
    """The dedup cache evicts oldest entries instead of growing unbounded."""
    adapter = _make_adapter()
    TelegramAdapter = _adapter_class()
    limit = TelegramAdapter._REPLIED_MEDIA_CACHE_MAX

    for i in range(limit + 5):
        photo = _FakePhotoSize(file_unique_id=f"uniq-{i}")
        msg = _reply_to_photo(photo, message_id=200 + i)
        await adapter._handle_text_message(
            SimpleNamespace(update_id=i, message=msg, effective_message=msg),
            SimpleNamespace(),
        )

    assert len(adapter._replied_media_cache) == limit
    assert "uniq-0" not in adapter._replied_media_cache
    assert f"uniq-{limit + 4}" in adapter._replied_media_cache
    await _drain(adapter)


@pytest.mark.asyncio
async def test_replied_media_cache_keys_carry_no_token_or_path(stub_media_cache):
    """Cache keys and values must not carry bot tokens; keys carry no paths."""
    adapter = _make_adapter()
    photo = _FakePhotoSize()
    msg = _reply_to_photo(photo)
    await adapter._handle_text_message(
        SimpleNamespace(update_id=9001, message=msg, effective_message=msg),
        SimpleNamespace(),
    )

    token = adapter.config.token
    for key in adapter._replied_media_cache:
        assert token not in key
        assert "/" not in key
    await _drain(adapter)


@pytest.mark.asyncio
async def test_reply_media_download_failure_fails_open(stub_media_cache, caplog):
    """A failing replied-to download still dispatches the text turn, no token in logs."""
    adapter = _make_adapter()
    photo = _FakePhotoSize(fail=True)
    msg = _reply_to_photo(photo)

    with caplog.at_level("WARNING"):
        await adapter._handle_text_message(
            SimpleNamespace(update_id=9001, message=msg, effective_message=msg),
            SimpleNamespace(),
        )

    assert len(adapter._pending_text_batches) == 1
    event = next(iter(adapter._pending_text_batches.values()))
    assert event.text
    assert event.media_urls == []
    assert not adapter._replied_media_cache, "failed download must not be cached"
    assert "SECRET-TOKEN" not in caplog.text
    await _drain(adapter)


@pytest.mark.asyncio
async def test_plain_text_without_reply_touches_nothing(stub_media_cache):
    """No reply_to_message means no download, no cache entry, no media."""
    adapter = _make_adapter()
    msg = _blank_message()
    await adapter._handle_text_message(
        SimpleNamespace(update_id=9001, message=msg, effective_message=msg),
        SimpleNamespace(),
    )

    assert stub_media_cache == []
    assert not adapter._replied_media_cache
    event = next(iter(adapter._pending_text_batches.values()))
    assert event.media_urls == []
    await _drain(adapter)


@pytest.mark.asyncio
async def test_direct_photo_attachment_path_unaffected(stub_media_cache, monkeypatch):
    """A photo sent directly still goes through the photo cache/batch path."""
    try:
        import plugins.platforms.telegram.adapter as adapter_module
    except ModuleNotFoundError:  # pragma: no cover - pre-extraction layout
        import gateway.platforms.telegram as adapter_module

    adapter = _make_adapter()
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._media_batch_delay_seconds = 10.0

    monkeypatch.setattr(
        adapter_module, "cache_image_from_bytes", lambda data, ext=".jpg": f"/tmp/direct{ext}"
    )
    enqueued = []
    adapter._enqueue_photo_event = lambda key, event: enqueued.append(event)

    photo = _FakePhotoSize()
    msg = _blank_message(text=None, caption="look", photo=[photo])
    await adapter._handle_media_message(
        SimpleNamespace(update_id=9001, message=msg, effective_message=msg),
        SimpleNamespace(),
    )

    assert len(enqueued) == 1
    assert enqueued[0].media_urls == ["/tmp/direct.png"]
    assert stub_media_cache == [], "direct photos must not use the replied-media path"
