from __future__ import annotations

from types import SimpleNamespace

from agent.conversation_loop import _requirements_completion_gate
from agent.requirements_ledger import TurnRequirementsLedger


def _agent_with(ledger=None):
    agent = SimpleNamespace()
    if ledger is not None:
        agent._requirements_ledger = ledger
    return agent


def test_pending_must_requirement_blocks_success_and_adds_metadata():
    ledger = TurnRequirementsLedger("turn-1")
    requirement = ledger.register_steer("also run the focused test")

    decision = _requirements_completion_gate(_agent_with(ledger), True)

    assert decision["completed"] is False
    assert decision["turn_exit_reason"] == "pending_must_requirements"
    assert decision["requirements"] == [requirement]
    assert decision["requirements_revision"] == 1
    assert decision["pending_requirements"] == [requirement]
    assert decision["completion_blocked"] is True
    assert decision["footer"] == "⚠️ Incomplete: 1 must requirement remains pending."


def test_completed_requirements_preserve_existing_completion_behavior():
    ledger = TurnRequirementsLedger("turn-2")
    requirement = ledger.register_steer("verify rollback")
    ledger.reconcile_todos([{**requirement, "status": "completed"}])

    decision = _requirements_completion_gate(_agent_with(ledger), True)

    assert decision == {
        "completed": True,
        "requirements": ledger.requirements_snapshot(),
        "requirements_revision": 1,
        "pending_requirements": [],
        "completion_blocked": False,
    }


def test_missing_ledger_is_a_strict_no_op():
    assert _requirements_completion_gate(_agent_with(), True) == {"completed": True}
    assert _requirements_completion_gate(_agent_with(), False) == {"completed": False}


def test_late_snapshot_can_block_an_earlier_allow_decision():
    ledger = TurnRequirementsLedger("turn-race")
    agent = _agent_with(ledger)

    early = _requirements_completion_gate(agent, True)
    ledger.register_steer("late steer must not be lost")
    late = _requirements_completion_gate(agent, early["completed"])

    assert early["completed"] is True
    assert late["completed"] is False
    assert late["turn_exit_reason"] == "pending_must_requirements"
    assert late["pending_requirements"][0]["content"] == "late steer must not be lost"


def test_pending_requirement_metadata_preserves_an_existing_failure_exit():
    ledger = TurnRequirementsLedger("turn-failed")
    requirement = ledger.register_steer("still pending")

    decision = _requirements_completion_gate(_agent_with(ledger), False)

    assert decision["completed"] is False
    assert decision["requirements"] == [requirement]
    assert decision["requirements_revision"] == 1
    assert decision["pending_requirements"] == [requirement]
    assert decision["completion_blocked"] is False
    assert "turn_exit_reason" not in decision
    assert "footer" not in decision


def test_footer_pluralizes_multiple_pending_requirements():
    ledger = TurnRequirementsLedger("turn-many")
    ledger.register_steer("one")
    ledger.register_steer("two")

    decision = _requirements_completion_gate(_agent_with(ledger), True)

    assert decision["footer"] == "⚠️ Incomplete: 2 must requirements remain pending."


def test_interrupt_is_never_reported_completed():
    from agent.conversation_loop import _base_turn_completed

    assert _base_turn_completed("partial response", 1, 10, False, True) is False
