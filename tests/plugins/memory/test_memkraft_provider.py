import builtins
from datetime import timezone
import json
import sys
import threading
import time

import pytest

from agent.runtime_resume import HANDOFF_FIELDS, build_handoff
from plugins.memory.memkraft import MemKraftMemoryProvider


class FakeMemKraft:
    def __init__(self, *, injection="", raise_injection=False):
        self.injection = injection
        self.raise_injection = raise_injection
        self.inject_calls = []

    def reasoning_inject_for_task(self, query):
        self.inject_calls.append(query)
        if self.raise_injection:
            raise RuntimeError("reasoning bank unavailable")
        return self.injection


def provider_with(fake_mk):
    provider = MemKraftMemoryProvider()
    provider._mk = fake_mk
    provider._prefetch_top_k = 3
    provider._prefetch_cache = {}
    provider._prefetch_threads = {}
    provider._lock = threading.RLock()
    return provider


def test_incomplete_turn_accepts_canonical_resume_handoff_and_writes_bounded_checkpoint(tmp_path):
    provider = provider_with(object())
    provider._base_dir = str(tmp_path)
    handoff = build_handoff(
        session_key="telegram:111",
        profile="work",
        goal="finish the migration",
        failing_test_ids=["tests/test_migration.py::test_upgrade"],
        verified_sha="a" * 40,
        depth=0,
    )

    assert set(handoff.to_payload()) == set(HANDOFF_FIELDS)
    assert provider.on_incomplete_turn(handoff.to_payload()) is True

    checkpoints = list((tmp_path / "tasks" / "hermes-resume").glob("*.json"))
    assert len(checkpoints) == 1
    persisted = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert persisted == {
        "status": "incomplete",
        "session_key": "telegram:111",
        "profile": "work",
        "goal": "finish the migration",
        "failing_test_ids": ["tests/test_migration.py::test_upgrade"],
        "verified_sha": "a" * 40,
        "depth": 0,
    }


def test_prefetch_combines_search_context_and_reasoning_injection(monkeypatch):
    fake_mk = FakeMemKraft(injection="## ReasoningBank\nfrontend design guidance")
    provider = provider_with(fake_mk)
    monkeypatch.setattr(provider, "_format_search_context", lambda query, top_k: "## MemKraft Memory\nprior memory")

    result = provider.prefetch("design a dashboard", session_id="sess-1")

    assert result == "## MemKraft Memory\nprior memory\n\n## ReasoningBank\nfrontend design guidance"
    assert fake_mk.inject_calls == ["design a dashboard"]


def test_prefetch_returns_reasoning_injection_when_search_context_empty(monkeypatch):
    provider = provider_with(FakeMemKraft(injection="## ReasoningBank\nuse strong visual hierarchy"))
    monkeypatch.setattr(provider, "_format_search_context", lambda query, top_k: "")

    result = provider.prefetch("frontend design task")

    assert result == "## ReasoningBank\nuse strong visual hierarchy"


def test_prefetch_preserves_search_context_when_reasoning_method_missing_or_empty_or_errors(monkeypatch):
    provider = provider_with(object())
    monkeypatch.setattr(provider, "_format_search_context", lambda query, top_k: "## MemKraft Memory\nexisting")
    assert provider.prefetch("task") == "## MemKraft Memory\nexisting"

    provider = provider_with(FakeMemKraft(injection="   "))
    monkeypatch.setattr(provider, "_format_search_context", lambda query, top_k: "## MemKraft Memory\nexisting")
    assert provider.prefetch("task") == "## MemKraft Memory\nexisting"

    provider = provider_with(FakeMemKraft(injection="ignored", raise_injection=True))
    monkeypatch.setattr(provider, "_format_search_context", lambda query, top_k: "## MemKraft Memory\nexisting")
    assert provider.prefetch("task") == "## MemKraft Memory\nexisting"


def test_queue_prefetch_reuses_cache_only_for_the_same_query(monkeypatch):
    fake_mk = FakeMemKraft(injection="## ReasoningBank\nqueued guidance")
    provider = provider_with(fake_mk)
    monkeypatch.setattr(
        provider,
        "_format_search_context",
        lambda query, top_k: f"## MemKraft Memory\n{query}",
    )

    provider.queue_prefetch("frontend task", session_id="sess-queue")
    deadline = time.time() + 2
    while time.time() < deadline:
        if ("sess-queue", "frontend task") in provider._prefetch_cache:
            break
        time.sleep(0.01)

    assert provider.prefetch("frontend task", session_id="sess-queue") == (
        "## MemKraft Memory\nfrontend task\n\n## ReasoningBank\nqueued guidance"
    )
    assert fake_mk.inject_calls == ["frontend task"]


def test_prefetch_does_not_inject_a_previous_turns_cached_query(monkeypatch):
    fake_mk = FakeMemKraft(injection="")
    provider = provider_with(fake_mk)
    monkeypatch.setattr(
        provider,
        "_format_search_context",
        lambda query, top_k: f"## MemKraft Memory\n{query}",
    )

    provider.queue_prefetch("memkraft product bug", session_id="sess-stale")
    deadline = time.time() + 2
    while time.time() < deadline:
        if ("sess-stale", "memkraft product bug") in provider._prefetch_cache:
            break
        time.sleep(0.01)

    assert provider.prefetch("골드트라이앵글 숙소", session_id="sess-stale") == (
        "## MemKraft Memory\n골드트라이앵글 숙소"
    )
    assert ("sess-stale", "memkraft product bug") not in provider._prefetch_cache


def test_initialize_uses_installed_memkraft_without_forcing_source_path(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "memory:\n  provider: memkraft\nplugins:\n  memkraft:\n    base_dir: ${HERMES_HOME}/memkraft-memory\n",
        encoding="utf-8",
    )
    original_path = list(sys.path)

    provider = MemKraftMemoryProvider()
    provider.initialize("sess", hermes_home=str(hermes_home), agent_identity="smoke")

    assert provider._source_path == ""
    assert sys.path == original_path
    assert provider._base_dir == str(hermes_home / "memkraft-memory")


def test_config_schema_declares_eval_bridge_as_disabled_boolean_opt_in():
    schema = {item["key"]: item for item in MemKraftMemoryProvider().get_config_schema()}

    assert schema["eval_bridge_enabled"]["default"] is False
    assert schema["eval_bridge_enabled"]["type"] == "boolean"


@pytest.mark.parametrize("disabled_value", [False, None, "true"])
def test_delegation_outcome_requires_literal_true(monkeypatch, disabled_value):
    provider = provider_with(object())
    provider._config = {"eval_bridge_enabled": disabled_value}
    calls = []
    monkeypatch.setattr(
        "plugins.evaluation.eval_bridge.register_failure_case",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    provider.on_delegation_outcome({"kind": "delegated_failed"})

    assert calls == []


def test_delegation_outcome_requires_existing_memkraft_sink(monkeypatch):
    provider = provider_with(None)
    provider._config = {"eval_bridge_enabled": True}
    calls = []
    monkeypatch.setattr(
        "plugins.evaluation.eval_bridge.register_failure_case",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    provider.on_delegation_outcome({"kind": "delegated_failed"})

    assert calls == []


@pytest.mark.parametrize(
    "kind",
    ["delegated_failed", "delegated_partial", "delegated_timed_out"],
)
def test_delegation_outcome_sends_exact_constant_allowlist_to_same_sink(monkeypatch, kind):
    sink = object()
    provider = provider_with(sink)
    provider._config = {"eval_bridge_enabled": True}
    provider._profile = "/Users/private/profiles/customer-9482"
    captured = []

    def capture(actual_sink, event, *, now):
        captured.append((actual_sink, event, now))

    monkeypatch.setattr("plugins.evaluation.eval_bridge.register_failure_case", capture)
    provider.on_delegation_outcome(
        {
            "kind": kind,
            "task": "RAW SECRET TASK",
            "result": "RAW PRIVATE RESULT",
            "trace_ref": "/private/path/trace.json",
            "session_id": "customer-9482",
            "expected_behavior": "attacker-controlled expected text",
            "forbidden_behavior": "attacker-controlled forbidden text",
            "source_profile": "attacker.profile",
            "privacy_scope": "public",
            "unknown": {"nested": "RAW NESTED DATA"},
        }
    )

    assert len(captured) == 1
    actual_sink, event, now = captured[0]
    assert actual_sink is sink
    assert event == {
        "kind": kind,
        "source_profile": "hermes.subagent",
        "expected_behavior": "Delegated work should complete successfully with a verified result.",
        "forbidden_behavior": "Failed, partial, or timed-out delegated work must not be treated as successful.",
        "privacy_scope": "local_private",
    }
    assert now.tzinfo is timezone.utc
    assert now.utcoffset().total_seconds() == 0
    assert "RAW" not in repr(captured)
    assert "/Users/" not in repr(captured)
    assert "customer-9482" not in repr(captured)


@pytest.mark.parametrize(
    "outcome",
    [
        {"kind": "delegated_succeeded"},
        {"kind": "user_correction"},
        {"kind": "unknown"},
        {"kind": None},
        {},
        "delegated_failed",
        None,
    ],
)
def test_delegation_outcome_ignores_success_unknown_and_malformed(monkeypatch, outcome):
    provider = provider_with(object())
    provider._config = {"eval_bridge_enabled": True}
    calls = []
    monkeypatch.setattr(
        "plugins.evaluation.eval_bridge.register_failure_case",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    provider.on_delegation_outcome(outcome)

    assert calls == []


def test_delegation_outcome_swallows_adapter_exception_without_raw_leakage(monkeypatch, caplog):
    provider = provider_with(object())
    provider._config = {"eval_bridge_enabled": True}
    secret = "DO-NOT-LEAK-adapter-secret"

    def fail(*args, **kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr("plugins.evaluation.eval_bridge.register_failure_case", fail)

    assert provider.on_delegation_outcome({"kind": "delegated_failed", "raw": secret}) is None
    assert secret not in caplog.text
    assert "delegated_failed" not in caplog.text


def test_delegation_outcome_swallows_real_sink_exception_without_raw_leakage(caplog):
    secret = "DO-NOT-LEAK-sink-secret"

    class FailingSink:
        def evaluation_register_case(self, **kwargs):
            raise RuntimeError(secret)

    sink = FailingSink()
    provider = provider_with(sink)
    provider._config = {"eval_bridge_enabled": True}

    assert provider.on_delegation_outcome({"kind": "delegated_partial", "raw": secret}) is None
    assert provider._mk is sink
    assert secret not in caplog.text
    assert "delegated_partial" not in caplog.text


def test_delegation_outcome_swallows_bridge_import_exception_without_raw_leakage(monkeypatch, caplog):
    provider = provider_with(object())
    provider._config = {"eval_bridge_enabled": True}
    secret = "DO-NOT-LEAK-import-secret"
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "plugins.evaluation.eval_bridge":
            raise ImportError(secret)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert provider.on_delegation_outcome({"kind": "delegated_failed", "raw": secret}) is None
    assert secret not in caplog.text
    assert "delegated_failed" not in caplog.text
