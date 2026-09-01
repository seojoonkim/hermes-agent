from agent.requirements_ledger import TurnRequirementsLedger


def test_register_steer_creates_stable_must_requirement_and_revision():
    ledger = TurnRequirementsLedger("turn-7")

    first = ledger.register_steer("Run the focused tests")
    second = ledger.register_steer("Preserve existing changes")

    assert first["revision"] == 1
    assert second["revision"] == 2
    assert [item["id"] for item in ledger.pending_snapshot()] == [
        "req:turn-7:000001",
        "req:turn-7:000002",
    ]
    assert all(item["must"] for item in ledger.pending_snapshot())


def test_classification_is_deterministic_and_has_three_bounded_levels():
    assert TurnRequirementsLedger.classify("Reply yes") == "fast"
    assert TurnRequirementsLedger.classify("Update the parser and run its tests") == "standard"
    assert TurnRequirementsLedger.classify(
        "Refactor the authentication architecture, migrate storage, and run the full integration suite"
    ) == "deep"
    assert TurnRequirementsLedger.classify("Reply yes") == "fast"


def test_projection_and_reconcile_sync_completed_status():
    ledger = TurnRequirementsLedger("abc")
    ledger.register_steer("Keep this requirement")
    projected = ledger.project_todos()

    assert projected == [{
        "id": "req:abc:000001",
        "content": "Keep this requirement",
        "status": "pending",
    }]

    ledger.reconcile_todos([{**projected[0], "status": "completed"}])
    assert ledger.pending_snapshot() == []
    assert ledger.requirements_snapshot()[0]["status"] == "completed"


def test_requirement_todo_content_is_canonical_and_only_status_is_mutable():
    ledger = TurnRequirementsLedger("canonical")
    requirement = ledger.register_steer("Do not rewrite me")

    reconciled = ledger.reconcile_todos([{
        "id": requirement["id"],
        "content": "mutated by model",
        "status": "completed",
    }])

    assert reconciled == [{
        "id": requirement["id"],
        "content": "Do not rewrite me",
        "status": "completed",
    }]
    assert ledger.requirements_snapshot()[0]["content"] == "Do not rewrite me"


def test_completion_decision_accounts_for_existing_completed_work():
    ledger = TurnRequirementsLedger("done")
    ledger.register_steer("One")
    ledger.register_steer("Two")
    ledger.reconcile_todos([
        {"id": "req:done:000001", "content": "One", "status": "completed"},
        {"id": "req:done:000002", "content": "Two", "status": "pending"},
    ])

    blocked = ledger.completion_decision(existing_completed={"req:done:000001"})
    assert blocked["complete"] is False
    assert blocked["pending_ids"] == ["req:done:000002"]

    ledger.reconcile_todos([
        {"id": "req:done:000002", "content": "Two", "status": "completed"},
    ])
    allowed = ledger.completion_decision(existing_completed={"req:done:000001"})
    assert allowed["complete"] is True
    assert allowed["pending_ids"] == []
