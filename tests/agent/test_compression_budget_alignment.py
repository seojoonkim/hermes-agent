"""Compression's host budget must not preempt a valid auxiliary summary."""
from agent.conversation_compression import resolve_context_compression_timeouts
from agent.auxiliary_client import _COMPRESSION_TIMEOUT_FLOOR_SECONDS
from hermes_cli.config_defaults import DEFAULT_CONFIG

import pytest
from unittest.mock import MagicMock


@pytest.mark.parametrize("force", [False, True])
def test_timeout_notice_is_visible_for_manual_only(monkeypatch, force):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "compression-budget-test"
    agent._cached_system_prompt = "system"
    agent._emit_warning = MagicMock()
    agent._touch_activity = MagicMock()
    agent.context_compressor = MagicMock()
    monkeypatch.setattr("agent.portal_tags.get_conversation_context", lambda: object())

    def timed_out(**kwargs):
        kwargs["on_timeout"](90, 600, 0.1)
        return kwargs["messages"], "system"

    monkeypatch.setattr(
        "agent.conversation_compression.run_compress_context_with_progress_timeout",
        timed_out,
    )
    original = [{"role": "user", "content": "preserve original"}]
    messages, prompt = AIAgent._compress_context(agent, original, "system", force=force)
    assert messages is original
    assert prompt == "system"
    assert agent._emit_warning.call_count == int(force)
    agent.context_compressor.record_timeout_failure.assert_called_once()
    if force:
        assert "기존 메시지는 그대로 보존" in agent._emit_warning.call_args.args[0]


def test_timeout_ladder_survives_recreated_agents_and_expiry(tmp_path):
    from tests.agent.test_hygiene_timeout_cooldown_isolation import _bound_compressor
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("repeat", source="telegram")
    try:
        for expected in (60, 300, 900):
            compressor = _bound_compressor(db, "repeat")
            compressor.record_timeout_failure("host compress_context timeout")
            state = compressor.get_active_compression_failure_cooldown()
            assert expected - 2 <= state["remaining_seconds"] <= expected
            # Expired deadlines should allow retry, not erase failure history.
            row = db.get_compression_failure_cooldown_row("repeat")
            db.record_compression_failure_cooldown("repeat", 1, row["error"])
        compressor._clear_compression_failure_cooldown()
        recreated = _bound_compressor(db, "repeat")
        recreated.record_timeout_failure("host compress_context timeout")
        assert recreated.get_active_compression_failure_cooldown()["remaining_seconds"] <= 60
    finally:
        db.close()


def test_default_host_budget_remains_distinct_from_auxiliary_floor():
    for cfg in ({}, DEFAULT_CONFIG["compression"]):
        idle, ceiling = resolve_context_compression_timeouts(cfg)
        assert idle == 90
        assert ceiling == 180


def test_explicit_host_cap_remains_authoritative():
    assert resolve_context_compression_timeouts({
        "context_timeout_seconds": 90,
        "context_total_ceiling_seconds": 180,
    }) == (90, 180)
