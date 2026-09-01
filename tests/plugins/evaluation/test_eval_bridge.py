"""Tests for the Hermes Eval Bridge v0 pure adapter.

The bridge turns a structured Hermes failure event into a MemKraft Evaluation
Corpus registration payload. It is a *pure adapter*: no MemKraft import, no
network, no promotion APIs, and it must never leak raw conversation content.
"""

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

eb = importlib.import_module("plugins.evaluation.eval_bridge")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _event(**overrides):
    base = {
        "kind": "delegated_failed",
        "source_profile": "hermes.subagent",
        "expected_behavior": "delegate returns a completed subtask result",
        "forbidden_behavior": "silently reporting success after a failed delegation",
        "anonymized_input_ref": "corpus://anon/abc123",
        "trace_ref": "trace://local/def456",
    }
    base.update(overrides)
    return base


NOW = "2026-08-08T12:00:00Z"
OTHER_NOW = "2020-01-01T00:00:00Z"

#: Privacy scopes accepted by the MemKraft Evaluation Corpus.
BRIDGE_PRIVACY_SCOPES = {"local_private", "private_pointer"}

CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")

ALLOWED_KEYS = {
    "case_id",
    "failure_type",
    "anonymized_input_ref",
    "expected_behavior",
    "forbidden_behavior",
    "trace_ref",
    "source_profile",
    "privacy_scope",
}


class RecordingSink:
    """Duck-typed MemKraft sink."""

    def __init__(self):
        self.calls = []
        self.nows = []

    def evaluation_register_case(self, *, now, **payload):
        self.calls.append(payload)
        self.nows.append(now)
        return {"ok": True}


class ExplodingSink:
    def __init__(self):
        self.calls = 0

    def evaluation_register_case(self, *, now, **payload):
        self.calls += 1
        raise RuntimeError("corpus unavailable")


# ---------------------------------------------------------------------------
# Candidate kinds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", [
    "user_correction",
    "delegated_failed",
    "delegated_partial",
    "delegated_timed_out",
    "tool_error",
    "routing_error",
    "unverified_completion",
])
def test_supported_kinds_produce_candidate(kind):
    payload = eb.build_registration(_event(kind=kind))
    assert payload is not None
    assert payload["failure_type"] == kind


@pytest.mark.parametrize("kind", [
    "task_completed",
    "tool_success",
    "heartbeat",
    "message",
    "",
])
def test_success_and_noise_events_produce_no_candidate(kind):
    assert eb.build_registration(_event(kind=kind)) is None


def test_non_dict_event_produces_no_candidate():
    for bad in [None, [], "delegated_failed", 42]:
        assert eb.build_registration(bad) is None


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------

def test_payload_contains_exactly_the_allowed_keys():
    payload = eb.build_registration(_event())
    assert set(payload) == ALLOWED_KEYS


def test_metadata_is_separate_bounded_and_non_sensitive():
    candidate = eb.build_candidate(_event())
    assert set(candidate.payload) == ALLOWED_KEYS
    md = candidate.metadata
    assert set(md) <= {"schema_version", "adapter", "failure_type", "privacy_scope"}
    assert all(isinstance(v, str) for v in md.values())
    assert all(len(v) <= 64 for v in md.values())


def test_default_privacy_scope_is_local_private():
    assert eb.build_registration(_event())["privacy_scope"] == "local_private"


@pytest.mark.parametrize("scope", sorted(BRIDGE_PRIVACY_SCOPES))
def test_local_bridge_privacy_scopes_are_accepted(scope):
    assert eb.build_registration(_event(privacy_scope=scope))["privacy_scope"] == scope


def test_bridge_privacy_scopes_are_local_only():
    assert set(eb.KNOWN_PRIVACY_SCOPES) == BRIDGE_PRIVACY_SCOPES


def test_untrusted_event_cannot_assert_public_safe():
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(privacy_scope="public_safe"))


def test_privacy_scope_must_be_known():
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(privacy_scope="local_shared"))


# ---------------------------------------------------------------------------
# Refs / profile validation — fail closed
# ---------------------------------------------------------------------------

def test_missing_refs_are_derived_deterministically_as_sha256_local_refs():
    ev = _event()
    ev.pop("anonymized_input_ref")
    ev.pop("trace_ref")
    a = eb.build_registration(ev)
    b = eb.build_registration(dict(ev))
    assert a["anonymized_input_ref"] == b["anonymized_input_ref"]
    assert a["anonymized_input_ref"].startswith("local:sha256:")
    assert a["trace_ref"].startswith("local:sha256:")
    assert len(a["anonymized_input_ref"].split(":")[-1]) == 64


def test_raw_looking_ref_is_rejected_not_passed_through():
    ev = _event(anonymized_input_ref="please fix my password hunter2 in /etc/shadow")
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(ev)


@pytest.mark.parametrize("ref", ["input:hunter2", "token:sk-live-CAFE",
                                  "/Users/private/secret", "trace://local/../x"])
def test_secret_bearing_or_path_like_refs_are_rejected(ref):
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(trace_ref=ref))


def test_generated_local_digest_refs_round_trip():
    first = eb.build_registration(_event(
        anonymized_input_ref=None, trace_ref=None))
    second = eb.build_registration(_event(
        anonymized_input_ref=first["anonymized_input_ref"],
        trace_ref=first["trace_ref"],
    ))
    assert second["anonymized_input_ref"] == first["anonymized_input_ref"]
    assert second["trace_ref"] == first["trace_ref"]


def test_unknown_source_profile_fails_closed():
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(source_profile="../../etc/passwd"))
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(source_profile=None))
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(source_profile="a" * 81))


def test_missing_expectations_fail_closed():
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(expected_behavior=""))
    with pytest.raises(eb.EvalBridgeError):
        eb.build_registration(_event(forbidden_behavior=None))


# ---------------------------------------------------------------------------
# Determinism of case_id
# ---------------------------------------------------------------------------

def test_case_id_matches_memkraft_grammar():
    case_id = eb.build_registration(_event())["case_id"]
    assert CASE_ID_RE.match(case_id)
    assert case_id.startswith("case.")
    suffix = case_id[len("case."):]
    assert len(suffix) == 64
    assert all(c in "0123456789abcdef" for c in suffix)


def test_case_id_is_deterministic_for_canonically_identical_events():
    a = eb.build_registration(_event())
    reordered = dict(reversed(list(_event().items())))
    b = eb.build_registration(reordered)
    assert a["case_id"] == b["case_id"]


def test_case_id_does_not_depend_on_time_or_volatile_fields():
    a = eb.build_registration(_event(timestamp="2020-01-01T00:00:00Z", session_id="s1"))
    b = eb.build_registration(_event(timestamp="2026-08-08T12:00:00Z", session_id="s2"))
    assert a["case_id"] == b["case_id"]


@pytest.mark.parametrize("delta", [
    {"kind": "tool_error"},
    {"trace_ref": "trace://local/other"},
    {"anonymized_input_ref": "corpus://anon/other"},
    {"expected_behavior": "something materially different"},
    {"forbidden_behavior": "a different forbidden thing"},
    {"source_profile": "hermes.router"},
])
def test_materially_distinct_events_get_distinct_case_ids(delta):
    base = eb.build_registration(_event())["case_id"]
    other = eb.build_registration(_event(**delta))["case_id"]
    assert base != other


# ---------------------------------------------------------------------------
# Security / privacy — adversarial event
# ---------------------------------------------------------------------------

ADVERSARIAL_SECRETS = [
    "hunter2",
    "sk-live-DEADBEEFCAFE",
    "Bearer abc.def.ghi",
    "/Users/sano/.ssh/id_rsa",
    "BEGIN RSA PRIVATE KEY",
    "my credit card is 4111111111111111",
    "SELECT * FROM users",
    "please delete prod",
]


def _adversarial_event():
    return _event(
        messages=[
            {"role": "user", "content": "please delete prod, my password is hunter2"},
            {"role": "assistant", "content": "SELECT * FROM users"},
        ],
        raw_text="my credit card is 4111111111111111",
        tool_call={
            "name": "shell",
            "arguments": {"cmd": "cat /Users/sano/.ssh/id_rsa", "token": "sk-live-DEADBEEFCAFE"},
            "result": {"stdout": "BEGIN RSA PRIVATE KEY", "path": "/Users/sano/.ssh/id_rsa"},
        },
        api_token="sk-live-DEADBEEFCAFE",
        password="hunter2",
        authorization="Bearer abc.def.ghi",
        cwd="/Users/sano/secret-project",
        unknown_future_field={"deeply": {"nested": "hunter2"}},
    )


def test_adversarial_event_leaks_nothing_into_payload():
    payload = eb.build_registration(_adversarial_event())
    blob = json.dumps(payload, sort_keys=True)
    for secret in ADVERSARIAL_SECRETS:
        assert secret not in blob
    for banned in ["messages", "tool_call", "arguments", "password", "api_token",
                   "authorization", "cwd", "unknown_future_field", "raw_text"]:
        assert banned not in blob


def test_adversarial_event_leaks_nothing_into_candidate_metadata():
    candidate = eb.build_candidate(_adversarial_event())
    blob = json.dumps({"p": candidate.payload, "m": candidate.metadata}, sort_keys=True)
    for secret in ADVERSARIAL_SECRETS:
        assert secret not in blob


def test_unknown_event_fields_are_dropped_not_forwarded():
    payload = eb.build_registration(_event(some_new_field="value-xyz"))
    assert "value-xyz" not in json.dumps(payload)
    assert set(payload) == ALLOWED_KEYS


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def test_writer_calls_sink_exactly_once_with_payload_only():
    sink = RecordingSink()
    ev = _adversarial_event()
    result = eb.register_failure_case(sink, ev, now=NOW)
    assert len(sink.calls) == 1
    assert set(sink.calls[0]) == ALLOWED_KEYS
    assert result == sink.calls[0]["case_id"]
    blob = json.dumps(sink.calls[0], sort_keys=True)
    for secret in ADVERSARIAL_SECRETS:
        assert secret not in blob


def test_writer_returns_none_and_does_not_write_for_non_candidate():
    sink = RecordingSink()
    assert eb.register_failure_case(sink, _event(kind="tool_success"), now=NOW) is None
    assert sink.calls == []


def test_writer_fails_closed_when_sink_raises():
    sink = ExplodingSink()
    with pytest.raises(RuntimeError):
        eb.register_failure_case(sink, _event(), now=NOW)
    assert sink.calls == 1


def test_writer_rejects_sink_without_register_method():
    class Bad:
        def promote(self, **kw):  # pragma: no cover - must never be called
            raise AssertionError("promotion API must not be used")

    with pytest.raises(eb.EvalBridgeError):
        eb.register_failure_case(Bad(), _event(), now=NOW)


def test_writer_never_touches_promotion_or_improvement_apis():
    calls = []

    class WideSink:
        def evaluation_register_case(self, *, now, **payload):
            calls.append("register")

        def __getattr__(self, name):  # any other attribute access is a violation
            calls.append(name)
            raise AssertionError(f"unexpected sink API used: {name}")

    eb.register_failure_case(WideSink(), _event(), now=NOW)
    assert calls == ["register"]


def test_writer_is_idempotent_in_identity_across_calls():
    sink = RecordingSink()
    eb.register_failure_case(sink, _event(), now=NOW)
    eb.register_failure_case(sink, _event(), now=NOW)
    assert sink.calls[0]["case_id"] == sink.calls[1]["case_id"]


def test_writer_forwards_now_to_sink_as_call_context():
    sink = RecordingSink()
    eb.register_failure_case(sink, _event(), now=NOW)
    assert len(sink.calls) == 1
    assert sink.nows == [NOW]
    assert "now" not in sink.calls[0]


def test_now_is_not_part_of_the_built_payload():
    assert "now" not in eb.build_registration(_event())


def test_different_now_values_yield_the_same_case_id():
    sink = RecordingSink()
    a = eb.register_failure_case(sink, _event(), now=NOW)
    b = eb.register_failure_case(sink, _event(), now=OTHER_NOW)
    assert a == b
    assert sink.nows == [NOW, OTHER_NOW]
    assert sink.calls[0] == sink.calls[1]


def test_now_is_required_and_keyword_only():
    sink = RecordingSink()
    with pytest.raises(TypeError):
        eb.register_failure_case(sink, _event())
    with pytest.raises(TypeError):
        eb.register_failure_case(sink, _event(), NOW)
    assert sink.calls == []


@pytest.mark.parametrize("kind", sorted(eb.SUPPORTED_KINDS))
def test_failure_kind_is_preserved_exactly_through_the_sink(kind):
    sink = RecordingSink()
    eb.register_failure_case(sink, _event(kind=kind), now=NOW)
    assert sink.calls[0]["failure_type"] == kind


def test_nul_in_behavior_text_cannot_shift_the_field_boundary():
    """Field boundaries must survive NUL bytes inside behavior text."""
    a = eb.build_registration(
        _event(expected_behavior="a\x00b", forbidden_behavior="c")
    )
    b = eb.build_registration(
        _event(expected_behavior="a", forbidden_behavior="b\x00c")
    )
    assert a["case_id"] != b["case_id"]
