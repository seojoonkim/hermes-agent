from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from hermes_state import SessionDB


def _agent(*, session_db=None):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            skip_context_files=True,
            skip_memory=True,
        )


def test_close_releases_lazily_created_session_db(tmp_path: Path, monkeypatch) -> None:
    db = SessionDB(tmp_path / "lazy.db")
    db.close()
    monkeypatch.setattr("hermes_state._default_db_path", lambda: tmp_path / "lazy.db")

    agent = _agent()
    lazy_db = agent._get_session_db_for_recall()

    assert lazy_db is not None
    assert lazy_db._conn is not None
    agent.close()
    assert lazy_db._conn is None
    assert agent._session_db is None
    agent.close()


def test_close_preserves_injected_shared_session_db(tmp_path: Path) -> None:
    shared_db = SessionDB(tmp_path / "shared.db")
    agent = _agent(session_db=shared_db)

    try:
        agent.close()
        assert shared_db._conn is not None
        shared_db.create_session("still-shared", source="test")
        assert shared_db.get_session("still-shared") is not None
    finally:
        shared_db.close()
