from __future__ import annotations

import json

from agent.runtime_resume import (
    MAX_RESUME_DEPTH,
    MemKraftResumeStore,
    ResumeCoordinator,
    build_incomplete_handoff,
    build_resume_prompt,
)


def _handoff(*, goal="finish adapter", prior_token="", depth=0):
    return build_incomplete_handoff(
        profile="sano",
        session_id="session-1",
        channel_key="telegram:account-1:chat-1:topic-7",
        turn_exit_reason="max_iterations_reached(90/90)",
        incomplete_goal=goal,
        prior_token=prior_token,
        chain_depth=depth,
        messages=[
            {"role": "tool", "content": "commit " + "a" * 40},
            {
                "role": "tool",
                "content": "FAILED tests/z.py::test_z - nope\nFAILED tests/a.py::test_a - nope",
            },
        ],
    )


def test_handoff_extracts_verified_output_and_is_deterministic_bounded():
    first = _handoff(goal="G" * 2000)
    second = _handoff(goal="G" * 2000)

    assert first == second
    assert len(first.incomplete_goal) == 1000
    assert first.last_verified_commit == "a" * 40
    assert first.failing_tests == ("tests/a.py::test_a", "tests/z.py::test_z")
    assert len(first.resume_token) == 32
    assert first.profile == "sano"
    assert first.channel_key == "telegram:account-1:chat-1:topic-7"
    assert set(first.to_dict()) == {
        "resume_token",
        "prior_token",
        "profile",
        "session_id",
        "channel_key",
        "termination_reason",
        "incomplete_goal",
        "last_verified_commit",
        "failing_tests",
        "chain_depth",
        "next_dispatch_intent",
    }
    prompt = build_resume_prompt(first.to_dict())
    assert "smaller bounded step" in prompt
    assert "a" * 40 in prompt


def test_only_verified_tool_output_contributes_evidence():
    handoff = build_incomplete_handoff(
        profile="sano",
        session_id="session-1",
        channel_key="telegram:chat-1",
        turn_exit_reason="max_iterations_reached(90/90)",
        incomplete_goal="finish adapter",
        messages=[
            {
                "role": "assistant",
                "content": "commit "
                + "b" * 40
                + "\nFAILED tests/fake.py::test_fake - imagined",
            },
            {"role": "tool", "content": "commit " + "a" * 40},
        ],
    )
    assert handoff.last_verified_commit == "a" * 40
    assert handoff.failing_tests == ()


def test_memkraft_checkpoint_is_atomic_idempotent_and_restores_after_restart(tmp_path):
    store = MemKraftResumeStore(tmp_path)
    handoff = _handoff()

    assert store.persist(handoff) is True
    assert store.persist(handoff) is True
    files = list((tmp_path / "tasks" / "hermes-resume").glob("*.json"))
    assert len(files) == 1
    assert not list(files[0].parent.glob(".*.tmp"))

    restarted = MemKraftResumeStore(tmp_path)
    restored = restarted.load(handoff.resume_token)
    assert restored == handoff.to_dict()
    assert json.loads(files[0].read_text(encoding="utf-8"))["status"] == "incomplete"


def test_post_delivery_resume_is_token_bound_one_shot_and_restart_restorable(tmp_path):
    callbacks = []
    scheduled = []
    store = MemKraftResumeStore(tmp_path)
    handoff = _handoff()
    coordinator = ResumeCoordinator(
        store=store,
        register_post_delivery=callbacks.append,
        schedule_internal=lambda token, prompt: scheduled.append((token, prompt)),
    )

    assert coordinator.arm(handoff) is True
    assert scheduled == []
    callbacks[0]("wrong-token")
    assert scheduled == []
    callbacks[0](handoff.resume_token)
    callbacks[0](handoff.resume_token)
    assert len(scheduled) == 1
    assert scheduled[0][0] == handoff.resume_token

    restarted = ResumeCoordinator(
        store=MemKraftResumeStore(tmp_path),
        register_post_delivery=callbacks.append,
        schedule_internal=lambda token, prompt: scheduled.append((token, prompt)),
    )
    restored = restarted.restore(handoff.resume_token)
    assert restored == handoff


def test_cycle_progress_and_depth_guards_block_repeated_resume(tmp_path):
    callbacks = []
    coordinator = ResumeCoordinator(
        store=MemKraftResumeStore(tmp_path),
        register_post_delivery=callbacks.append,
        schedule_internal=lambda *_: None,
    )
    first = _handoff()
    assert coordinator.arm(first) is True

    no_progress = _handoff(prior_token=first.resume_token, depth=1)
    assert no_progress.resume_token == first.resume_token
    assert coordinator.arm(no_progress) is False

    cycle = _handoff(goal="changed", prior_token=first.resume_token, depth=1)
    coordinator.mark_seen(cycle.resume_token)
    assert coordinator.arm(cycle) is False

    too_deep = _handoff(goal="other", prior_token="different", depth=MAX_RESUME_DEPTH)
    assert coordinator.arm(too_deep) is False


def test_scope_mismatch_fails_closed_on_restore(tmp_path):
    handoff = _handoff()
    store = MemKraftResumeStore(tmp_path)
    store.persist(handoff)

    assert store.load(handoff.resume_token, profile="other") is None
    assert store.load(handoff.resume_token, session_id="other") is None
    assert store.load(handoff.resume_token, channel_key="other") is None


def test_malformed_checkpoint_fails_closed(tmp_path):
    handoff = _handoff()
    store = MemKraftResumeStore(tmp_path)
    store.persist(handoff)
    path = tmp_path / "tasks" / "hermes-resume" / f"{handoff.resume_token}.json"
    path.write_text("{}", encoding="utf-8")
    assert store.load(handoff.resume_token) is None
