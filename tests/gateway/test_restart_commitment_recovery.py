"""Regression tests for restart recovery of promised-but-unfinished work."""

from gateway.restart import (
    RESTART_RESUME_COMPLETE_MARKER,
    build_restart_resume_note,
    consume_restart_resume_marker,
)


def test_restart_note_treats_progress_promises_as_unfinished_work():
    note = build_restart_resume_note("restart_timeout")

    assert "gateway restart" in note
    assert "promised actions or deliverables" in note
    assert "progress or intent" in note
    assert "not evidence of completion" in note
    assert "execute and verify" in note
    assert "promised post-restart check and final report" in note
    assert "concrete verification evidence" in note
    assert "related but narrower success" in note
    assert "original user request" in note
    assert "current chat/topic session lineage" in note
    assert "unrelated room, chat, channel, topic, thread, or session key" in note
    assert "conversation history supplied to this turn is the complete recovery scope" in note


def test_shutdown_note_keeps_reason_specific_wording():
    note = build_restart_resume_note("shutdown_timeout")

    assert "gateway shutdown" in note


def test_unknown_reason_uses_generic_interruption_wording():
    note = build_restart_resume_note("unexpected_exit")

    assert "gateway interruption" in note


def test_completion_marker_is_required_only_after_verified_recovery():
    note = build_restart_resume_note("restart_timeout")

    assert RESTART_RESUME_COMPLETE_MARKER in note
    assert "Only after every item is complete and verified" in note
    assert "Do not append it to a progress update" in note


def test_completion_marker_is_consumed_before_delivery():
    visible, completed = consume_restart_resume_marker(
        f"Recovered, deployed, and verified. {RESTART_RESUME_COMPLETE_MARKER}"
    )

    assert completed is True
    assert visible == "Recovered, deployed, and verified."


def test_progress_without_completion_marker_stays_pending():
    visible, completed = consume_restart_resume_marker("Still validating production.")

    assert completed is False
    assert visible == "Still validating production."
