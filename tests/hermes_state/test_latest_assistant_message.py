from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_latest_assistant_message_returns_exact_visible_text(db):
    db.create_session("s1", source="telegram")
    db.append_message("s1", role="assistant", content="older answer")
    db.append_message("s1", role="user", content="thanks")
    db.append_message("s1", role="assistant", content="exact latest answer")

    row = db.get_latest_assistant_message("s1")

    assert row is not None
    assert row["content"] == "exact latest answer"


def test_latest_assistant_message_skips_internal_status_rows(db):
    db.create_session("s1", source="telegram")
    db.append_message("s1", role="assistant", content="real answer")
    db.append_message(
        "s1",
        role="assistant",
        content="ETA p50 100s",
        display_kind="internal_notification",
    )

    row = db.get_latest_assistant_message("s1")

    assert row is not None
    assert row["content"] == "real answer"


def test_latest_assistant_message_returns_none_without_visible_answer(db):
    db.create_session("s1", source="telegram")
    db.append_message("s1", role="user", content="hello")
    assert db.get_latest_assistant_message("s1") is None
