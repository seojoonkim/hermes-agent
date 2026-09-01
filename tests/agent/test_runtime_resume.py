"""Bounded checkpoint/resume of incomplete (iteration-capped) turns.

Ports the live MemKraft incomplete-turn resume flow onto the v2026.8.3
architecture: eligibility is decided from the finalized turn ``result`` dict
(``agent/turn_finalizer.py``), the memory adapter is selected per profile
(multiplex), and the follow-up turn is scheduled as a distinct *internal*
turn only after the user's response has actually been delivered
(``register_post_delivery_callback`` on the platform adapter).

The provider implementation itself is NOT ported — these tests drive the
runtime seam with a fake memory manager/provider.
"""

import pytest

from agent import runtime_resume as rr


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, name="fake", profile=None, boom=False):
        self._name = name
        self.profile = profile
        self.boom = boom
        self.calls = []

    @property
    def name(self):
        return self._name

    def on_incomplete_turn(self, payload):
        self.calls.append(payload)
        if self.boom:
            raise RuntimeError("provider exploded")
        return True


class PlainProvider:
    """A provider that does not implement the resume hook at all."""

    def __init__(self, name="plain"):
        self._name = name

    @property
    def name(self):
        return self._name


class FakeManager:
    def __init__(self, providers):
        self._providers = list(providers)

    @property
    def providers(self):
        return list(self._providers)


class FakeStore:
    def __init__(self, ok=True, boom=False):
        self.ok = ok
        self.boom = boom
        self.saved = []

    def save_resume_checkpoint(self, payload):
        if self.boom:
            raise RuntimeError("disk full")
        self.saved.append(payload)
        return self.ok


def make_result(**over):
    result = {
        "turn_exit_reason": "max_iterations_reached(60/60)",
        "completed": False,
        "failed": False,
        "interrupted": False,
        "final_response": "partial answer",
    }
    result.update(over)
    return result


def make_coordinator(**over):
    scheduled = []
    callbacks = []
    kwargs = dict(
        memory_manager=FakeManager([FakeProvider()]),
        checkpoint_store=FakeStore(),
        register_post_delivery=callbacks.append,
        schedule_turn=scheduled.append,
        session_key="sess:1",
        profile="",
        depth=0,
    )
    kwargs.update(over)
    coord = rr.ResumeCoordinator(**kwargs)
    coord._test_scheduled = scheduled
    coord._test_callbacks = callbacks
    return coord


def fire(coord):
    """Invoke every registered post-delivery callback once."""
    for cb in list(coord._test_callbacks):
        cb()


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------

def test_incomplete_iteration_capped_turn_is_eligible():
    assert rr.is_resume_eligible(make_result(), depth=0) is True


@pytest.mark.parametrize("over", [
    {"turn_exit_reason": "text_response(stop)"},
    {"turn_exit_reason": "interrupted_by_user"},
    {"turn_exit_reason": "budget_exhausted"},
    {"failed": True},
    {"interrupted": True},
    {"completed": True},
])
def test_other_turns_are_not_eligible(over):
    assert rr.is_resume_eligible(make_result(**over), depth=0) is False


def test_resume_chain_is_bounded():
    assert rr.is_resume_eligible(make_result(), depth=rr.MAX_RESUME_DEPTH) is False


# --------------------------------------------------------------------------
# bounded, deterministic handoff
# --------------------------------------------------------------------------

def test_handoff_is_bounded_and_deterministic():
    a = rr.build_handoff(
        session_key="s",
        profile="work",
        goal="G" * 5000,
        failing_test_ids=[f"tests/test_x.py::test_{i}" for i in range(50)],
        verified_sha="a" * 40,
        depth=0,
    )
    b = rr.build_handoff(
        session_key="s",
        profile="work",
        goal="G" * 5000,
        failing_test_ids=[f"tests/test_x.py::test_{i}" for i in range(50)],
        verified_sha="a" * 40,
        depth=0,
    )
    assert a.to_payload() == b.to_payload()
    assert len(a.goal) <= rr.MAX_GOAL_CHARS
    assert len(a.failing_test_ids) <= rr.MAX_FAILING_TESTS
    assert all(len(t) <= rr.MAX_TEST_ID_CHARS for t in a.failing_test_ids)
    assert a.verified_sha == "a" * 40


def test_handoff_rejects_non_sha_and_raw_tool_output():
    h = rr.build_handoff(
        session_key="s",
        profile="",
        goal="fix the parser",
        failing_test_ids=["tests/a.py::test_a\n<tool_result>SECRET DUMP</tool_result>"],
        verified_sha="not-a-sha; rm -rf /",
        depth=0,
    )
    assert h.verified_sha == ""
    blob = repr(h.to_payload()) + h.to_turn_text()
    assert "SECRET" not in blob
    assert "tool_result" not in blob


def test_turn_text_is_marked_internal_resume():
    h = rr.build_handoff(session_key="s", profile="", goal="fix parser",
                         failing_test_ids=[], verified_sha="", depth=0)
    assert h.to_turn_text().startswith(rr.RESUME_TURN_PREFIX)


# --------------------------------------------------------------------------
# profile-scoped adapter selection (multiplex)
# --------------------------------------------------------------------------

def test_profile_scoped_adapter_selection_under_multiplex():
    work = FakeProvider("mk-work", profile="work")
    home = FakeProvider("mk-home", profile="home")
    mgr = FakeManager([home, work])
    assert rr.select_resume_provider(mgr, "work") is work
    assert rr.select_resume_provider(mgr, "home") is home


def test_unscoped_adapter_used_only_when_no_profile_match():
    generic = FakeProvider("mk", profile=None)
    other = FakeProvider("mk-home", profile="home")
    mgr = FakeManager([other, generic])
    assert rr.select_resume_provider(mgr, "work") is generic


def test_providers_without_hook_are_ignored():
    mgr = FakeManager([PlainProvider()])
    assert rr.select_resume_provider(mgr, "") is None


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def test_schedules_internal_turn_only_after_delivery():
    coord = make_coordinator()
    assert coord.maybe_schedule(make_result(), goal="fix parser") is True
    assert coord._test_scheduled == []          # nothing before delivery
    fire(coord)
    assert len(coord._test_scheduled) == 1
    assert coord._test_scheduled[0].startswith(rr.RESUME_TURN_PREFIX)


def test_duplicate_delivery_callback_is_idempotent():
    coord = make_coordinator()
    coord.maybe_schedule(make_result(), goal="fix parser")
    fire(coord)
    fire(coord)
    fire(coord)
    assert len(coord._test_scheduled) == 1


def test_persist_failure_blocks_scheduling():
    coord = make_coordinator(checkpoint_store=FakeStore(ok=False))
    assert coord.maybe_schedule(make_result(), goal="g") is False
    fire(coord)
    assert coord._test_scheduled == []


def test_persist_exception_blocks_scheduling_and_does_not_raise():
    coord = make_coordinator(checkpoint_store=FakeStore(boom=True))
    assert coord.maybe_schedule(make_result(), goal="g") is False
    assert coord._test_scheduled == []


def test_provider_notification_is_fail_open():
    provider = FakeProvider(boom=True)
    coord = make_coordinator(memory_manager=FakeManager([provider]))
    assert coord.maybe_schedule(make_result(), goal="g") is True
    fire(coord)
    assert provider.calls, "provider hook must still be invoked"
    assert len(coord._test_scheduled) == 1


def test_provider_receives_only_bounded_payload():
    provider = FakeProvider()
    coord = make_coordinator(memory_manager=FakeManager([provider]))
    coord.maybe_schedule(
        make_result(),
        goal="fix parser",
        failing_test_ids=["tests/a.py::test_a"],
        verified_sha="b" * 40,
    )
    payload = provider.calls[0]
    assert payload["goal"] == "fix parser"
    assert payload["failing_test_ids"] == ["tests/a.py::test_a"]
    assert payload["verified_sha"] == "b" * 40
    assert set(payload) == set(rr.HANDOFF_FIELDS)


def test_missing_provider_still_persists_and_schedules():
    store = FakeStore()
    coord = make_coordinator(memory_manager=FakeManager([]), checkpoint_store=store)
    assert coord.maybe_schedule(make_result(), goal="g") is True
    assert store.saved
    fire(coord)
    assert len(coord._test_scheduled) == 1


def test_ineligible_turn_registers_nothing():
    coord = make_coordinator()
    assert coord.maybe_schedule(make_result(completed=True), goal="g") is False
    assert coord._test_callbacks == []


def test_scheduling_failure_never_propagates():
    def boom(_text):
        raise RuntimeError("gateway down")

    coord = make_coordinator(schedule_turn=boom)
    assert coord.maybe_schedule(make_result(), goal="g") is True
    fire(coord)  # must not raise


def test_registration_failure_never_propagates():
    def boom(_cb):
        raise RuntimeError("adapter gone")

    coord = make_coordinator(register_post_delivery=boom)
    assert coord.maybe_schedule(make_result(), goal="g") is False


def test_resumed_turn_cannot_chain_again():
    coord = make_coordinator(depth=rr.MAX_RESUME_DEPTH)
    assert coord.maybe_schedule(make_result(), goal="g") is False


# --------------------------------------------------------------------------
# turn_finalizer integration seam
# --------------------------------------------------------------------------

def test_finalizer_hook_invokes_coordinator_fail_open():
    class Agent:
        pass

    seen = []

    class Coord:
        def maybe_schedule(self, result, **kw):
            seen.append(result)
            raise RuntimeError("coordinator broke")

    agent = Agent()
    agent.resume_coordinator = Coord()
    rr.maybe_schedule_resume(agent, make_result())  # must not raise
    assert seen


def test_finalizer_hook_noop_without_coordinator():
    class Agent:
        pass

    rr.maybe_schedule_resume(Agent(), make_result())
