"""Post-commit steering must survive a real SQLite reopen."""
import copy
import pytest
from unittest.mock import patch

from hermes_state import SessionDB
from tests.agent.test_compression_worker_isolation_76354 import _build_agent_with_db


@pytest.mark.xfail(strict=True, reason="Compression steer disabled until durable cache-safe ingestion exists")
def test_post_commit_steer_survives_restart(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("steer", source="cli")
    agent = _build_agent_with_db(db, "steer")
    agent._supports_compression_steer = True  # expose the blocked implementation
    messages = [{"role": "user", "content": "tail", "api_content": "tail with plugin"}]

    def commit(*args, **kwargs):
        db.archive_and_compact("steer", messages)
        agent.steer("durable correction")
        return messages, "sys"

    with patch("agent.conversation_compression.compress_context", side_effect=commit), patch(
        "agent.conversation_compression.resolve_context_compression_timeouts", return_value=(0, 0)
    ):
        returned, _ = agent._compress_context(copy.deepcopy(messages), "sys")
    db.close()
    reopened = SessionDB(db_path=path)
    try:
        row = reopened.get_messages("steer")[-1]
        assert "durable correction" in row["content"]
        assert "durable correction" in row["api_content"]
        assert row["api_content"] == returned[-1]["api_content"]
        assert agent._drain_pending_steer() is None
        assert len(reopened.get_messages("steer")) == 1
    finally:
        reopened.close()


def test_stop_does_not_consume_post_compression_steer(tmp_path):
    db = SessionDB(db_path=tmp_path / "stop.db")
    db.create_session("stop", source="cli")
    agent = _build_agent_with_db(db, "stop")
    messages = [{"role": "user", "content": "tail"}]
    agent.steer("pending")
    agent._interrupt_requested = True
    with patch("agent.conversation_compression.compress_context", return_value=(messages, "sys")), patch(
        "agent.conversation_compression.resolve_context_compression_timeouts", return_value=(0, 0)
    ):
        agent._compress_context(messages, "sys")
    assert messages[0]["content"] == "tail"
    assert agent._pending_steer == "pending"
    db.close()
