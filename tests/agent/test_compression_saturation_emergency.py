"""Compression pool saturation must not strand an oversized session."""
import threading
import time

import pytest

from agent import conversation_compression as cc
from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_context_compression_timeouts,
    run_compress_context_with_progress_timeout,
)
from hermes_state import SessionDB
from tests.agent.test_hygiene_timeout_cooldown_isolation import _bound_compressor


def _drain():
    with cc._compress_admission_lock:
        cc._compress_admitted_count = 0


def test_pool_saturation_records_shared_budget_failure(tmp_path):
    _drain()
    release = threading.Event()
    started = threading.Event()

    def blocked(fence: CompressionCommitFence):
        started.set()
        assert release.wait(30)
        return ([], "done")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("victim", source="telegram")
        victim = _bound_compressor(db, "victim")

        class Agent:
            session_id = "victim"
            _compression_attempt_id = "attempt"
            context_compressor = victim

        host = threading.Thread(
            target=run_compress_context_with_progress_timeout,
            kwargs=dict(
                worker=blocked,
                messages=[{"role": "user", "content": "first"}],
                system_prompt_fallback="fb",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.1,
            ),
        )
        host.start()
        assert started.wait(5)
        host.join(5)
        try:
            result = run_compress_context_with_progress_timeout(
                worker=lambda fence: ([], "never"),
                messages=[{"role": "user", "content": "second"}],
                system_prompt_fallback="fb2",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.1,
                telemetry_agent=Agent(),
            )
            assert result[1] == "fb2"
            row = db.get_compression_failure_cooldown_row("victim")
            assert row and "total compression budget exhausted" in (row.get("error") or "")
            assert victim.get_active_compression_failure_cooldown() is not None
            assert victim.should_emergency_fallback(victim.context_length)
        finally:
            release.set()
            _drain()
    finally:
        db.close()


@pytest.mark.parametrize(
    "agent_cfg,expected_ceiling",
    [({"stream_idle_timeout_seconds": 120}, 120.0), ({}, 180.0)],
)
def test_summary_wait_ceiling_follows_agent_idle_policy(monkeypatch, agent_cfg, expected_ceiling):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"compression": {}, "agent": agent_cfg},
    )
    monkeypatch.setattr(
        hermes_cli.config,
        "load_config_readonly",
        lambda: {"compression": {}, "agent": agent_cfg},
    )
    idle, ceiling = resolve_context_compression_timeouts()
    assert idle <= ceiling
    assert ceiling == expected_ceiling
