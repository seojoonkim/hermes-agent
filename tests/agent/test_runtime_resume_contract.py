from __future__ import annotations

import json
import threading

from agent.runtime_resume import (
    MAX_RESUME_DEPTH,
    MemKraftResumeStore,
    ResumeCoordinator,
    RuntimeResumeScope,
    build_incomplete_handoff,
    build_resume_prompt,
    parse_resume_prompt,
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
    assert parse_resume_prompt(prompt) == {
        "resume_token": first.resume_token,
        "chain_depth": 0,
        "incomplete_goal": first.incomplete_goal,
    }


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


def test_structured_scope_round_trips_delimiter_bearing_route_components():
    scope = RuntimeResumeScope(
        profile="sano",
        platform="telegram",
        account_id="bot:prod/1",
        chat_id="chat:with:colon",
        thread_id="topic/7:alpha",
        chat_type="group",
        scope_id="workspace:prod/blue",
        user_id="user:42",
        session_id="session-1",
    )

    encoded = scope.encode()
    assert encoded.startswith("json:")
    assert RuntimeResumeScope.decode(encoded) == scope


def test_dispatch_reservation_is_atomic_and_failed_acceptance_retries(tmp_path):
    handoff = _handoff()
    store = MemKraftResumeStore(tmp_path)
    assert store.persist(handoff)

    assert store.reserve_dispatch(handoff.resume_token, lease_id="runner-a") == handoff.to_dict()
    assert store.reserve_dispatch(handoff.resume_token, lease_id="runner-b") is None

    store.release_dispatch(handoff.resume_token, lease_id="runner-a")
    assert store.reserve_dispatch(handoff.resume_token, lease_id="runner-b") == handoff.to_dict()
    assert store.mark_dispatched(handoff.resume_token, lease_id="runner-a") is False
    assert store.mark_dispatched(handoff.resume_token, lease_id="runner-b") is True
    assert store.reserve_dispatch(handoff.resume_token, lease_id="runner-c") is None


def test_dispatch_reservation_is_atomic_across_store_instances(tmp_path):
    handoff = _handoff()
    first = MemKraftResumeStore(tmp_path)
    second = MemKraftResumeStore(tmp_path)
    assert first.persist(handoff)
    barrier = threading.Barrier(2)
    claims = []

    def claim(store, lease):
        barrier.wait()
        claims.append(store.reserve_dispatch(handoff.resume_token, lease_id=lease))

    threads = [
        threading.Thread(target=claim, args=(first, "lease-a")),
        threading.Thread(target=claim, args=(second, "lease-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert sum(item is not None for item in claims) == 1


def test_startup_reclaims_interrupted_dispatch_lease(tmp_path):
    handoff = _handoff()
    store = MemKraftResumeStore(tmp_path)
    assert store.persist(handoff)
    assert store.reserve_dispatch(handoff.resume_token, lease_id="dead-gateway")

    assert store.reclaim_interrupted_dispatches() == 1
    assert store.reserve_dispatch(handoff.resume_token, lease_id="replacement")
