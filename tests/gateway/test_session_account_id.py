"""Focused coverage for account-aware gateway session sources and keys."""

from dataclasses import replace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _source(**overrides):
    values = {
        "platform": Platform.TELEGRAM,
        "chat_id": "chat-1",
        "chat_type": "group",
        "user_id": "user-1",
        "thread_id": "thread-1",
        "profile": "coder",
    }
    values.update(overrides)
    return SessionSource(**values)


def test_account_id_serialization_round_trips_separately_from_profile():
    source = _source(account_id="bot:primary")

    payload = source.to_dict()
    restored = SessionSource.from_dict(payload)

    assert payload["account_id"] == "bot:primary"
    assert payload["profile"] == "coder"
    assert restored.account_id == "bot:primary"
    assert restored.profile == "coder"


def test_legacy_payload_without_account_id_remains_supported():
    source = SessionSource.from_dict({"platform": "telegram", "chat_id": "chat-1"})

    assert source.account_id is None
    assert "account_id" not in source.to_dict()


@pytest.mark.parametrize(
    ("source", "kwargs", "legacy_key"),
    [
        (
            _source(chat_type="dm", user_id=None, thread_id=None, profile=None),
            {},
            "agent:main:telegram:dm:chat-1",
        ),
        (
            _source(profile=None),
            {},
            "agent:main:telegram:group:chat-1:thread-1",
        ),
        (
            _source(),
            {"profile": "coder"},
            "agent:coder:telegram:group:chat-1:thread-1",
        ),
    ],
)
def test_absent_account_id_preserves_legacy_key_bytes(source, kwargs, legacy_key):
    assert build_session_key(source, **kwargs) == legacy_key


def test_account_id_isolates_otherwise_identical_sources():
    source = _source()

    first = build_session_key(replace(source, account_id="bot-one"), profile="coder")
    second = build_session_key(replace(source, account_id="bot-two"), profile="coder")
    legacy = build_session_key(source, profile="coder")

    assert first != second
    assert first != legacy
    assert second != legacy


def test_account_id_is_unambiguous_and_separate_from_profile():
    source = _source(profile=None, user_id=None)

    account_with_delimiter = build_session_key(
        replace(source, account_id="coder:telegram:group")
    )
    account_without_delimiter = build_session_key(
        replace(source, account_id="codertelegramgroup")
    )
    same_text_as_profile = build_session_key(replace(source, account_id="coder"))
    named_profile = build_session_key(source, profile="coder")

    assert len(
        {
            account_with_delimiter,
            account_without_delimiter,
            same_text_as_profile,
            named_profile,
        }
    ) == 4
    assert "coder:telegram:group" not in account_with_delimiter
