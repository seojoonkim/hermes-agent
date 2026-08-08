"""Restart recovery must not accept a promise as completed work.

Live dogfood regression: after a gateway restart the resumed turn replied
"I'll continue working on that now" and did nothing else.  That result is
syntactically successful (no ``interrupted``/``failed`` flag, non-empty
``final_response``), so ``resume_pending`` was cleared and the interrupted
task was silently abandoned — the very failure the marker exists to prevent.

A resumed turn may only clear the marker when it actually did work: it ran
tools, or it produced a substantive answer rather than a bare commitment to
continue later.  Non-resumed turns keep the pre-existing behaviour exactly.
"""

import pytest

from gateway.run import (
    _RESUME_COMMITMENT_MAX_TRACKED,
    _clear_resume_commitment_turn,
    _mark_resume_commitment_turn,
    _resumed_turn_completed_work,
    _should_clear_resume_pending_after_turn,
    _take_resume_commitment_turn,
)


class TestResumedTurnCompletedWork:
    def test_bare_commitment_is_not_completed_work(self):
        for text in (
            "I'll continue working on that now.",
            "Let me continue where I left off.",
            "Continuing now — I'll get back to you shortly.",
            "Working on it, one moment.",
            "I will resume the interrupted task.",
        ):
            assert _resumed_turn_completed_work({"final_response": text}) is False, text

    def test_substantive_answer_is_completed_work(self):
        result = {
            "final_response": (
                "The deploy finished. I re-ran the migration, confirmed the "
                "three pending rows landed, and the health check is green on "
                "both replicas, so nothing from the interrupted turn is left "
                "outstanding at this point in the rollout sequence."
            )
        }
        assert _resumed_turn_completed_work(result) is True

    def test_tool_work_counts_even_with_a_promise_shaped_reply(self):
        result = {
            "final_response": "I'll continue working on that now.",
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": "ok"},
            ],
        }
        assert _resumed_turn_completed_work(result) is True

    def test_empty_response_is_not_completed_work(self):
        assert _resumed_turn_completed_work({"final_response": ""}) is False

    def test_scan_is_bounded_to_the_response_prefix(self):
        """A promise followed by a huge tail is still only a promise."""
        result = {"final_response": "I'll continue working on that now. " + "x" * 100_000}
        assert _resumed_turn_completed_work(result) is False


class TestShouldClearResumePending:
    def test_non_resumed_turn_behaviour_is_unchanged(self):
        promise = {"final_response": "I'll continue working on that now."}
        assert _should_clear_resume_pending_after_turn(promise) is True
        assert _should_clear_resume_pending_after_turn({"interrupted": True}) is False

    def test_resumed_turn_keeps_marker_on_a_bare_promise(self):
        promise = {"final_response": "I'll continue working on that now."}
        assert _should_clear_resume_pending_after_turn(promise, resumed_turn=True) is False

    def test_resumed_turn_clears_marker_on_real_work(self):
        done = {
            "final_response": "Done — migration applied and verified.",
            "messages": [{"role": "tool", "tool_call_id": "1", "content": "ok"}],
        }
        assert _should_clear_resume_pending_after_turn(done, resumed_turn=True) is True

    def test_resumed_turn_still_respects_interrupt_flags(self):
        assert (
            _should_clear_resume_pending_after_turn(
                {"final_response": "Done, verified.", "interrupted": True},
                resumed_turn=True,
            )
            is False
        )


class TestCommitmentMarkerStore:
    def setup_method(self):
        _clear_resume_commitment_turn(None)

    def teardown_method(self):
        _clear_resume_commitment_turn(None)

    def test_mark_and_take_is_one_shot(self):
        _mark_resume_commitment_turn("tg:1:2")
        assert _take_resume_commitment_turn("tg:1:2") is True
        assert _take_resume_commitment_turn("tg:1:2") is False

    def test_unknown_session_is_not_resumed(self):
        assert _take_resume_commitment_turn("tg:9:9") is False

    def test_blank_keys_are_ignored(self):
        _mark_resume_commitment_turn("")
        _mark_resume_commitment_turn(None)
        assert _take_resume_commitment_turn("") is False

    def test_store_is_bounded(self):
        for i in range(_RESUME_COMMITMENT_MAX_TRACKED + 50):
            _mark_resume_commitment_turn(f"tg:{i}")
        # Oldest entries evicted; the newest are still tracked.
        assert _take_resume_commitment_turn("tg:0") is False
        assert _take_resume_commitment_turn(
            f"tg:{_RESUME_COMMITMENT_MAX_TRACKED + 49}"
        ) is True

    def test_only_session_keys_are_stored_never_payloads(self):
        """Privacy: the marker carries no message/response content."""
        from gateway.run import _RESUME_COMMITMENT_TURNS

        _mark_resume_commitment_turn("tg:1:2")
        assert list(_RESUME_COMMITMENT_TURNS) == ["tg:1:2"]
        assert set(_RESUME_COMMITMENT_TURNS.values()) == {True}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
