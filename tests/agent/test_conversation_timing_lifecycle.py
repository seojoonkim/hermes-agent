from types import SimpleNamespace

import pytest

import agent.conversation_loop as loop


class TimingManager:
    def __init__(self):
        self.events = []

    def on_turn_timing_start(self, turn, **kwargs):
        self.events.append(("timing_start", turn, kwargs))

    def estimate_turn(self, **kwargs):
        return None

    def on_turn_finish(self, turn, **kwargs):
        self.events.append(("finish", turn, kwargs))


def agent_with(manager):
    return SimpleNamespace(
        _memory_manager=manager,
        session_id="parent-session",
        _user_turn_count=4,
        platform="cli",
        _interrupt_requested=False,
        _emit_status=lambda _text: None,
    )


@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        ({"completed": True}, "completed"),
        ({"completed": False, "interrupted": True}, "interrupted"),
        ({"completed": False, "failed": True}, "failed"),
        ({"completed": False}, "partial"),
    ],
)
def test_lifecycle_guard_closes_representative_early_returns_once(monkeypatch, result, outcome):
    manager = TimingManager()
    agent = agent_with(manager)
    monkeypatch.setattr(loop, "_run_conversation_impl", lambda *_a, **_k: result)

    assert loop.run_conversation(agent, "code request") is result
    assert [event[0] for event in manager.events] == ["timing_start", "finish"]
    assert manager.events[-1][2]["outcome"] == outcome


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (RuntimeError("boom"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
        (InterruptedError(), "interrupted"),
        (type("CancelledError", (Exception,), {})(), "interrupted"),
    ],
)
def test_lifecycle_guard_closes_exception_and_baseexception(monkeypatch, error, outcome):
    manager = TimingManager()
    agent = agent_with(manager)

    def raise_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(loop, "_run_conversation_impl", raise_error)
    with pytest.raises(type(error)):
        loop.run_conversation(agent, "request")
    assert [event[0] for event in manager.events] == ["timing_start", "finish"]
    assert manager.events[-1][2]["outcome"] == outcome


def test_lifecycle_freezes_identity_before_impl_can_rotate_session(monkeypatch):
    manager = TimingManager()
    agent = agent_with(manager)

    def rotate_during_context(_agent, *_args, **_kwargs):
        _agent.session_id = "compacted-child"
        return {"completed": True}

    monkeypatch.setattr(loop, "_run_conversation_impl", rotate_during_context)
    loop.run_conversation(agent, "request")

    assert manager.events[0][2]["session_id"] == "parent-session"
    assert manager.events[-1][2]["session_id"] == "parent-session"


def test_resumed_history_allocates_distinct_turns_without_retaining_raw_data(monkeypatch):
    manager = TimingManager()
    agent = agent_with(manager)
    agent._user_turn_count = 0
    agent.session_id = "resume-private-session"
    history = [
        {"role": "user", "content": "RAW FIRST MESSAGE"},
        {"role": "assistant", "content": "RAW FIRST RESPONSE"},
        {"role": "user", "content": "RAW SECOND MESSAGE"},
        {"role": "assistant", "content": "RAW SECOND RESPONSE"},
    ]

    def hydrate_like_real_prologue(_agent, _message, _system, supplied_history, *_args):
        if supplied_history and _agent._user_turn_count == 0:
            _agent._user_turn_count = sum(
                item.get("role") == "user" for item in supplied_history
            )
        _agent._user_turn_count += 1
        return {"completed": True}

    monkeypatch.setattr(loop, "_run_conversation_impl", hydrate_like_real_prologue)
    loop.run_conversation(agent, "current request", conversation_history=history)
    loop.run_conversation(agent, "next request", conversation_history=history)

    starts = [event for event in manager.events if event[0] == "timing_start"]
    finishes = [event for event in manager.events if event[0] == "finish"]
    assert [event[1] for event in starts] == [3, 4]
    assert [event[1] for event in finishes] == [3, 4]
    assert len({(event[1], event[2]["session_id"]) for event in starts}) == 2
    rendered = repr(manager.events)
    assert "RAW FIRST" not in rendered
    assert "RAW SECOND" not in rendered


def test_generic_turn_start_remains_before_prefetch():
    import agent.turn_context as turn_context

    lines = open(turn_context.__file__, encoding="utf-8").read()
    start = lines.index("agent._memory_manager.on_turn_start")
    prefetch = lines.index("agent._memory_manager.prefetch_all", start)
    assert start < prefetch


def test_transport_active_transition_is_before_transport_specific_normalization():
    lines = open(loop._run_conversation_impl.__code__.co_filename, encoding="utf-8").read()
    common = lines.index("agent._turn_received_provider_response = True")
    branch = lines.index('if agent.api_mode == "codex_responses":', common)
    active = lines.index('phase="active"', common)
    assert common < active < branch
