import builtins
from datetime import timezone
import json
from itertools import chain, count, repeat
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_constants
import plugins.memory.memkraft as memkraft_plugin
from plugins.memory.memkraft import MemKraftMemoryProvider


class FakeMemKraft:
    def __init__(self, *, injection="", raise_injection=False):
        self.injection = injection
        self.raise_injection = raise_injection
        self.inject_calls = []
        self.delay_starts = []
        self.delay_finishes = []
        self.delay_estimates = {}
        self.fail_start_ids = set()
        self.fail_finish_ids = set()
        self.open_runs = set()
        self.finished_runs = set()
        self.duplicate_starts = 0
        self.duplicate_finishes = 0

    def delay_run_start(self, run_id, kind, subject, **kwargs):
        if run_id in self.fail_start_ids:
            raise RuntimeError("start failed")
        if run_id in self.open_runs or run_id in self.finished_runs:
            self.duplicate_starts += 1
            raise RuntimeError("run_id already exists")
        self.delay_starts.append((run_id, kind, subject, kwargs))
        self.open_runs.add(run_id)
        return {"outcome": "applied"}

    def delay_run_finish(self, run_id, elapsed_ms, **kwargs):
        if run_id in self.fail_finish_ids:
            raise RuntimeError("finish failed")
        if run_id in self.finished_runs:
            self.duplicate_finishes += 1
            raise RuntimeError("run was already finished")
        if run_id not in self.open_runs:
            raise RuntimeError("run was never started")
        self.delay_finishes.append((run_id, elapsed_ms, kwargs))
        self.open_runs.remove(run_id)
        self.finished_runs.add(run_id)
        return {"outcome": "applied"}

    def delay_estimate(self, kind, subject, **kwargs):
        return self.delay_estimates.get((kind, subject), {
            "available": False,
            "reason": "insufficient_samples",
            "sample_count": 0,
        })

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


def test_turn_timing_uses_only_bounded_metadata_and_monotonic_clock(monkeypatch):
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    ticks = iter([10.0, 10.4, 10.9, 11.2])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))

    provider.on_turn_timing_start(7, session_id="session-private", platform="telegram", subject="development")
    provider.on_turn_progress(7, session_id="session-private", phase="wait", iteration=1)
    provider.on_turn_progress(7, session_id="session-private", phase="rework", iteration=2)
    provider.on_turn_finish(7, session_id="session-private", outcome="completed")

    rendered = repr(fake_mk.delay_starts) + repr(fake_mk.delay_finishes)
    assert "RAW SECRET" not in rendered
    assert "session-private" not in rendered
    assert [item[2] for item in fake_mk.delay_starts] == [
        "telegram:sano:development",
        "telegram:sano:development:active",
        "telegram:sano:development:wait",
        "telegram:sano:development:rework",
    ]
    assert [item[1] for item in fake_mk.delay_finishes] == [400, 500, 300, 1200]


def test_real_memkraft_adapter_round_trip_is_private_and_closes_task(tmp_path, monkeypatch):
    MemKraft = pytest.importorskip("memkraft").MemKraft

    mk = MemKraft(base_dir=str(tmp_path))
    mk.init(verbose=False)
    provider = provider_with(mk)
    provider._profile = "sano"
    ticks = chain([10.0, 10.5, 11.0, 11.5], repeat(11.5))
    monkeypatch.setattr(time, "monotonic", ticks.__next__)

    provider.on_turn_timing_start(
        9, session_id="private-session-id",
        platform="telegram", subject="development",
    )
    provider.on_turn_progress(9, session_id="private-session-id", phase="wait", iteration=1)
    provider.on_turn_finish(9, session_id="private-session-id", outcome="completed")

    path = Path(tmp_path) / ".memkraft" / "delay" / "events.jsonl"
    raw = path.read_text()
    records = [json.loads(line) for line in raw.splitlines()]
    assert "RAW SECRET" not in raw
    assert "private-session-id" not in raw
    task_ids = {
        record["run_id"] for record in records
        if record["record_type"] == "delay_run_start" and record["kind"] == "task"
    }
    task_finishes = [
        record for record in records
        if record["record_type"] == "delay_run_finish" and record["run_id"] in task_ids
    ]
    assert len(task_finishes) == 1
    assert task_finishes[0]["outcome"] == "completed"
    assert provider._turn_timing == {}


def test_real_memkraft_repairs_partial_phase_close_finish_only(tmp_path, monkeypatch):
    MemKraft = pytest.importorskip("memkraft").MemKraft

    mk = MemKraft(base_dir=str(tmp_path))
    mk.init(verbose=False)
    original_finish = mk.delay_run_finish
    failed_ids = set()

    def fail_first_phase_finish(run_id, elapsed_ms, **kwargs):
        if run_id.endswith("-active-aggregate") and run_id not in failed_ids:
            failed_ids.add(run_id)
            raise RuntimeError("injected phase finish failure")
        return original_finish(run_id, elapsed_ms, **kwargs)

    monkeypatch.setattr(mk, "delay_run_finish", fail_first_phase_finish)
    provider = provider_with(mk)
    provider._profile = "sano"
    ticks = count(10.0)
    monkeypatch.setattr(time, "monotonic", ticks.__next__)
    provider.on_turn_timing_start(10, session_id="real-repair", platform="telegram", subject="development")

    provider.on_turn_finish(10, session_id="real-repair", outcome="completed")
    assert len(provider._turn_timing) == 1
    repair_state = next(iter(provider._turn_timing.values()))
    assert repair_state["started_phases"] == {"active"}

    provider.on_turn_finish(10, session_id="real-repair", outcome="completed")
    assert provider._turn_timing == {}

    path = Path(tmp_path) / ".memkraft" / "delay" / "events.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    starts = [record for record in records if record["record_type"] == "delay_run_start"]
    finishes = [record for record in records if record["record_type"] == "delay_run_finish"]
    assert len(starts) == 2
    assert len({record["run_id"] for record in starts}) == 2
    assert len(finishes) == 2
    assert len({record["run_id"] for record in finishes}) == 2
    assert {record["run_id"] for record in starts} == {record["run_id"] for record in finishes}
    task_finish = next(record for record in finishes if not record["run_id"].endswith("-aggregate"))
    assert task_finish["outcome"] == "completed"


def test_turn_finish_preserves_repair_state_when_aggregate_phase_close_fails(monkeypatch):
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    monkeypatch.setattr(time, "monotonic", count(10.0).__next__)
    provider.on_turn_timing_start(3, session_id="s", platform="telegram", subject="development")
    state = next(iter(provider._turn_timing.values()))
    phase_id = f"{state['task_id']}-active-aggregate"
    task_id = state["task_id"]
    fake_mk.fail_finish_ids.add(phase_id)

    provider.on_turn_finish(3, session_id="s", outcome="completed")

    assert fake_mk.delay_finishes == []
    assert len(provider._turn_timing) == 1
    repair_state = next(iter(provider._turn_timing.values()))
    assert repair_state["started_phases"] == {"active"}

    fake_mk.fail_finish_ids.clear()
    provider.on_turn_finish(3, session_id="s", outcome="completed")
    assert [item[0] for item in fake_mk.delay_finishes] == [phase_id, task_id]
    assert provider._turn_timing == {}
    assert fake_mk.duplicate_starts == 0
    assert fake_mk.duplicate_finishes == 0
    assert fake_mk.open_runs == set()
    assert fake_mk.delay_finishes[-1][2]["outcome"] == "completed"


def test_turn_finish_repairs_task_close_without_restarting_or_changing_outcome(monkeypatch):
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    monkeypatch.setattr(time, "monotonic", count(10.0).__next__)
    provider.on_turn_timing_start(8, session_id="s", platform="telegram", subject="development")
    state = next(iter(provider._turn_timing.values()))
    task_id = state["task_id"]
    fake_mk.fail_finish_ids.add(task_id)

    provider.on_turn_finish(8, session_id="s", outcome="completed")

    assert len(provider._turn_timing) == 1
    assert fake_mk.open_runs == {task_id}
    fake_mk.fail_finish_ids.clear()
    provider.abort_open_turns()

    assert provider._turn_timing == {}
    assert fake_mk.duplicate_starts == 0
    assert fake_mk.duplicate_finishes == 0
    assert fake_mk.open_runs == set()
    assert fake_mk.delay_finishes[-1][0] == task_id
    assert fake_mk.delay_finishes[-1][2]["outcome"] == "completed"


def test_progress_transitions_are_in_memory_and_ledger_writes_stay_bounded(monkeypatch):
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    ticks = iter([10.0] + [10.0 + index / 10 for index in range(1, 31)] + [13.1])
    monkeypatch.setattr(time, "monotonic", ticks.__next__)
    provider.on_turn_timing_start(4, session_id="s", platform="telegram", subject="development")

    phases = ("wait", "active", "rework") * 10
    for index, phase in enumerate(phases, 1):
        provider.on_turn_progress(4, session_id="s", phase=phase, iteration=index)

    assert len(fake_mk.delay_starts) == 1
    assert fake_mk.delay_finishes == []
    provider.on_turn_finish(4, session_id="s", outcome="completed")
    assert len(fake_mk.delay_starts) <= 4
    assert len(fake_mk.delay_finishes) <= 4
    assert provider._turn_timing == {}


def test_abort_open_turns_and_shutdown_close_hashed_timing_keys(monkeypatch):
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    ticks = iter([10.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr(time, "monotonic", ticks.__next__)

    provider.on_turn_timing_start(6, session_id="session-a", platform="telegram", subject="development")
    hashed_key = next(iter(provider._turn_timing))
    assert ":" not in hashed_key
    provider.abort_open_turns()
    assert provider._turn_timing == {}
    assert fake_mk.delay_finishes[-1][2]["outcome"] == "aborted"

    provider.on_turn_timing_start(7, session_id="session-b", platform="telegram", subject="development")
    provider.shutdown()
    assert provider._turn_timing == {}
    assert provider._mk is None
    assert fake_mk.delay_finishes[-1][2]["outcome"] == "aborted"


def test_estimate_turn_exposes_p50_p80_confidence_and_risk_adjustment_only():
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    fake_mk.delay_estimates[("task", "telegram:sano:development")] = {
        "available": True,
        "sample_count": 8,
        "samples_ms": [1, 2, 3, 4, 5, 6, 7, 8],
        "p50_ms": 4000,
        "p80_ms": 7000,
        "mad_ms": 1000,
        "anomaly_threshold_ms": 8000,
        "through_seq": None,
    }

    estimate = provider.estimate_turn(platform="telegram", subject="development")

    assert estimate == {
        "p50_ms": 4000,
        "p80_ms": 7000,
        "confidence": "medium",
        "sample_count": 8,
        "risk_adjustment_ms": 3000,
        "critical_path": "development",
    }
    assert "samples_ms" not in estimate


def test_estimate_turn_is_unavailable_below_five_exact_subject_samples():
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    fake_mk.delay_estimates[("task", "telegram:sano:development")] = {
        "available": True,
        "sample_count": 4,
        "p50_ms": 1000,
        "p80_ms": 2000,
    }

    assert provider.estimate_turn(platform="telegram", subject="development") is None

    fake_mk.delay_estimates[("task", "telegram:sano:development")].update(
        sample_count=5, p50_ms=1500, p80_ms=2500
    )
    exact = provider.estimate_turn(platform="telegram", subject="development")
    assert exact is not None
    assert exact["sample_count"] == 5
    # Another exact cohort cannot borrow these five samples.
    assert provider.estimate_turn(platform="telegram", subject="research") is None


def test_turn_timing_survives_unrelated_session_rotation(monkeypatch):
    fake_mk = FakeMemKraft()
    provider = provider_with(fake_mk)
    provider._profile = "sano"
    monkeypatch.setattr(time, "monotonic", iter([10.0, 10.5, 11.0]).__next__)

    provider.on_turn_timing_start(5, session_id="frozen", platform="telegram", subject="development")
    provider.on_session_switch("compressed-child", parent_session_id="frozen")
    provider.on_turn_progress(5, session_id="frozen", phase="wait", iteration=1)
    provider.on_turn_finish(5, session_id="frozen", outcome="completed")

    assert provider._turn_timing == {}
    assert fake_mk.delay_finishes[-1][2]["outcome"] == "completed"


def test_installed_memkraft_35_is_available():
    pytest.importorskip("memkraft")
    assert MemKraftMemoryProvider().is_available() is True

def test_is_available_uses_context_local_hermes_home_for_templated_base_dir(tmp_path, monkeypatch):
    observed = {}

    class ProbeMemKraft:
        def __init__(self, base_dir):
            observed["base_dir"] = base_dir

        delay_run_start = delay_run_finish = delay_estimate = lambda *args, **kwargs: None
        delay_record_retrospective = delay_record_action = lambda *args, **kwargs: None
        delay_record_application = delay_record_verification = lambda *args, **kwargs: None

    fake_module = type("Module", (), {
        "__version__": "3.5.0",
        "MemKraft": ProbeMemKraft,
    })
    monkeypatch.setattr("importlib.import_module", lambda name: fake_module)
    monkeypatch.setattr(
        memkraft_plugin,
        "_load_config",
        lambda *_args, **_kwargs: {"base_dir": "$HERMES_HOME/memkraft-memory"},
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "profile-sano")

    assert MemKraftMemoryProvider().is_available() is True
    assert observed["base_dir"] == str(tmp_path / "profile-sano" / "memkraft-memory")

def test_initialize_rejects_memkraft_without_35_delay_api(tmp_path, monkeypatch):
    class LegacyMemKraft:
        def __init__(self, _base_dir):
            pass

    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: type("Module", (), {"MemKraft": LegacyMemKraft}) if name == "memkraft" else None,
    )
    provider = MemKraftMemoryProvider()

    with pytest.raises(RuntimeError, match=r"MemKraft 3\.5\.0\+ is required"):
        provider.initialize("sess", hermes_home=str(tmp_path), agent_identity="sano")
    assert provider._mk is None
