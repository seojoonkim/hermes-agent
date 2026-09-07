"""Total wall-clock exhaustion must not buy a second summary wait."""
import time

from hermes_state import SessionDB
from tests.agent.test_hygiene_timeout_cooldown_isolation import _bound_compressor


def test_total_budget_exhaustion_is_shared_with_preflight(tmp_path):
    from gateway.run import _hygiene_timeout_error

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("budget", source="telegram")
        reason = _hygiene_timeout_error(elapsed=31, ceiling=30, idle=0.1)
        db.record_compression_failure_cooldown("budget", time.time() + 300, reason)
        compressor = _bound_compressor(db, "budget")
        assert compressor.get_active_compression_failure_cooldown() is not None
        assert not compressor.should_compress(compressor.threshold_tokens + 1)
        assert compressor.should_emergency_fallback(compressor.context_length)
    finally:
        db.close()


def test_idle_only_timeout_keeps_independent_preflight_budget():
    from gateway.run import _hygiene_timeout_error
    from agent.context_compressor import _is_hygiene_idle_timeout_error

    reason = _hygiene_timeout_error(elapsed=10, ceiling=30, idle=10)
    assert _is_hygiene_idle_timeout_error(reason)


def test_total_timeout_does_not_claim_no_summary_output():
    from gateway.run import _hygiene_timeout_error

    reason = _hygiene_timeout_error(elapsed=30, ceiling=30, idle=0.1)
    assert "no output" not in reason
    assert "total" in reason
