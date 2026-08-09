"""Gateway wiring for bounded runtime resume (agent.runtime_resume).

The coordinator owns the policy (eligibility, bounding, one-shot scheduling).
This module tests only the *seam*: the gateway picks the memory provider,
persists through it, registers a post-delivery callback tied to the run
generation, and — once the response is actually delivered — enqueues exactly
one internal follow-up turn on the session FIFO.

Everything here must fail open: a missing provider, a refused persistence, or
a missing adapter capability leaves the finished user turn untouched.
"""
from types import SimpleNamespace

import pytest

from agent.runtime_resume import RESUME_TURN_PREFIX
from gateway.run import GatewayRunner


CAPPED = {"turn_exit_reason": "max_iterations_reached", "completed": False}


class _FakeProvider:
    """Stands in for the MemKraft provider: on_incomplete_turn only."""

    def __init__(self, ok=True, profile="work"):
        self._ok = ok
        self.profile = profile
        self.payloads = []

    def on_incomplete_turn(self, payload):
        self.payloads.append(payload)
        return self._ok


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self.registered = []

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.registered.append((session_key, callback, generation))


def _agent(provider):
    providers = [provider] if provider is not None else []
    return SimpleNamespace(memory_manager=SimpleNamespace(providers=providers))


@pytest.fixture
def runner():
    return GatewayRunner.__new__(GatewayRunner)


@pytest.fixture
def wired(runner, monkeypatch):
    adapter = _FakeAdapter()
    monkeypatch.setattr(
        GatewayRunner, "_adapter_for_source", lambda self, source: adapter, raising=False
    )
    enqueued = []
    monkeypatch.setattr(
        GatewayRunner,
        "_enqueue_fifo",
        lambda self, key, event, adp: enqueued.append((key, event, adp)),
        raising=False,
    )
    return SimpleNamespace(adapter=adapter, enqueued=enqueued)


def _arm(runner, agent, *, text="finish the migration", generation=7):
    return runner._arm_runtime_resume(
        agent=agent,
        result=CAPPED,
        source=SimpleNamespace(platform="telegram", chat_id="c1"),
        session_key="telegram:c1",
        user_text=text,
        run_generation=generation,
    )


class TestArmRuntimeResume:
    def test_persists_and_registers_without_enqueuing_yet(self, runner, wired):
        provider = _FakeProvider()

        assert _arm(runner, _agent(provider)) is True

        # Durable persistence went through the provider exactly once.
        assert len(provider.payloads) == 1
        # ...and nothing is queued until the response is actually delivered.
        assert wired.enqueued == []
        assert len(wired.adapter.registered) == 1

    def test_generation_is_passed_through(self, runner, wired):
        _arm(runner, _agent(_FakeProvider()), generation=11)
        assert wired.adapter.registered[0][0] == "telegram:c1"
        assert wired.adapter.registered[0][2] == 11

    def test_callback_twice_enqueues_exactly_one_internal_turn(self, runner, wired):
        _arm(runner, _agent(_FakeProvider()))
        callback = wired.adapter.registered[0][1]

        callback()
        callback()

        assert len(wired.enqueued) == 1
        session_key, event, adapter = wired.enqueued[0]
        assert session_key == "telegram:c1"
        assert adapter is wired.adapter
        assert event.internal is True
        assert event.text.startswith(RESUME_TURN_PREFIX)
        # Bounded handoff: the goal comes from the user's own text, and no
        # transcript or tool output rides along.
        assert "finish the migration" in event.text
        assert not getattr(event, "raw_message", None)

    def test_internal_resume_input_does_not_arm_another_continuation(self, runner, wired):
        provider = _FakeProvider()
        armed = _arm(
            runner,
            _agent(provider),
            text=f"{RESUME_TURN_PREFIX} continue the previous turn.\ngoal: x",
        )
        assert armed is False
        assert provider.payloads == []
        assert wired.adapter.registered == []

    def test_no_provider_is_fail_open(self, runner, wired):
        assert _arm(runner, _agent(None)) is False
        assert wired.adapter.registered == []
        assert wired.enqueued == []

    def test_no_memory_manager_is_fail_open(self, runner, wired):
        assert _arm(runner, SimpleNamespace(memory_manager=None)) is False
        assert wired.adapter.registered == []

    def test_persistence_refusal_registers_nothing(self, runner, wired):
        provider = _FakeProvider(ok=False)
        assert _arm(runner, _agent(provider)) is False
        assert len(provider.payloads) == 1
        assert wired.adapter.registered == []
        assert wired.enqueued == []

    def test_provider_exception_is_fail_open(self, runner, wired):
        class _Boom:
            profile = "work"

            def on_incomplete_turn(self, payload):
                raise RuntimeError("disk full")

        assert _arm(runner, _agent(_Boom())) is False
        assert wired.adapter.registered == []

    def test_missing_adapter_capability_is_fail_open(self, runner, monkeypatch):
        provider = _FakeProvider()
        monkeypatch.setattr(
            GatewayRunner, "_adapter_for_source", lambda self, source: None, raising=False
        )
        assert _arm(runner, _agent(provider)) is False
        assert provider.payloads == []

    def test_completed_turn_is_not_armed(self, runner, wired):
        provider = _FakeProvider()
        armed = runner._arm_runtime_resume(
            agent=_agent(provider),
            result={"completed": True, "turn_exit_reason": "completed"},
            source=SimpleNamespace(platform="telegram", chat_id="c1"),
            session_key="telegram:c1",
            user_text="hi",
            run_generation=1,
        )
        assert armed is False
        assert provider.payloads == []

    def test_provider_is_notified_only_once(self, runner, wired):
        """The shim persists; the coordinator must not double-notify memory."""
        provider = _FakeProvider()
        _arm(runner, _agent(provider))
        wired.adapter.registered[0][1]()
        assert len(provider.payloads) == 1
