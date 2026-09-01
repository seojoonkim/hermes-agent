"""Hermes Eval Bridge v0 — pure adapter from failure events to corpus payloads.

Turns a structured Hermes failure event into a MemKraft Evaluation Corpus
registration payload. This module is deliberately a *pure adapter*:

* stdlib only — no MemKraft import, no network, no core loop integration;
* the emitted payload contains exactly the allowlisted keys, nothing else;
* raw / unknown event fields are dropped rather than forwarded;
* identity (``case_id``) is a deterministic sha256 over canonical fields and
  never depends on wall-clock time or other volatile fields;
* the writer only ever calls the duck-typed ``evaluation_register_case`` sink
  method — no promotion or self-improvement APIs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

SCHEMA_VERSION = "0"
ADAPTER_NAME = "hermes.eval_bridge"

#: Event kinds that are considered evaluation candidates.
SUPPORTED_KINDS = frozenset({
    "user_correction",
    "delegated_failed",
    "delegated_partial",
    "delegated_timed_out",
    "tool_error",
    "routing_error",
    "unverified_completion",
})

#: The exact set of keys emitted in a registration payload.
PAYLOAD_KEYS = (
    "case_id",
    "failure_type",
    "anonymized_input_ref",
    "expected_behavior",
    "forbidden_behavior",
    "trace_ref",
    "source_profile",
    "privacy_scope",
)

KNOWN_PRIVACY_SCOPES = frozenset({"local_private", "private_pointer"})
DEFAULT_PRIVACY_SCOPE = "local_private"

_REF_RE = re.compile(
    r"^(?:(?:corpus|trace|note|report|artifact|run|case|eval|label|bundle)://"
    r"[a-z0-9][a-z0-9._-]{0,63}(?:/[A-Za-z0-9._-]{1,64}){1,8}"
    r"|local:sha256:[0-9a-f]{64})$"
)
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
#: MemKraft corpus case_id grammar.
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")

_MAX_BEHAVIOR_LEN = 512


class EvalBridgeError(ValueError):
    """Raised when an event cannot be adapted safely; the bridge fails closed."""


@dataclass(frozen=True)
class EvalCandidate:
    """A registration payload plus bounded, non-sensitive metadata."""

    payload: Dict[str, str]
    metadata: Dict[str, str]


def _behavior(event: Dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalBridgeError(f"{key} is required and must be a non-empty string")
    text = value.strip()
    if len(text) > _MAX_BEHAVIOR_LEN:
        raise EvalBridgeError(f"{key} exceeds {_MAX_BEHAVIOR_LEN} characters")
    return text


def _source_profile(event: Dict[str, Any]) -> str:
    value = event.get("source_profile")
    if not isinstance(value, str) or not _PROFILE_RE.match(value.strip()):
        raise EvalBridgeError("source_profile is unknown or malformed")
    return value.strip()


def _privacy_scope(event: Dict[str, Any]) -> str:
    value = event.get("privacy_scope", DEFAULT_PRIVACY_SCOPE)
    if value is None:
        value = DEFAULT_PRIVACY_SCOPE
    if value not in KNOWN_PRIVACY_SCOPES:
        raise EvalBridgeError(f"unknown privacy_scope: {value!r}")
    return value


def _digest(*parts: str) -> str:
    # Canonical JSON of the ordered parts: unambiguous regardless of which
    # characters (NUL included) appear inside a part.
    canonical = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ref(event: Dict[str, Any], key: str, seed: str) -> str:
    value = event.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return "local:sha256:" + _digest(key, seed)
    if not isinstance(value, str) or not _REF_RE.match(value.strip()):
        raise EvalBridgeError(
            f"{key} must be an opaque reference, not raw content"
        )
    normalized = value.strip()
    if "://" in normalized:
        segments = normalized.split("://", 1)[1].split("/")[1:]
        if any(segment in (".", "..") for segment in segments):
            raise EvalBridgeError(f"{key} contains a forbidden path segment")
    return normalized


def build_candidate(event: Any) -> Optional[EvalCandidate]:
    """Build an :class:`EvalCandidate` from ``event``, or ``None`` if it is not one."""
    if not isinstance(event, dict):
        return None
    failure_type = event.get("kind")
    if not isinstance(failure_type, str) or failure_type not in SUPPORTED_KINDS:
        return None

    expected = _behavior(event, "expected_behavior")
    forbidden = _behavior(event, "forbidden_behavior")
    source_profile = _source_profile(event)
    privacy_scope = _privacy_scope(event)

    seed = _digest(failure_type, source_profile, expected, forbidden)
    anonymized_input_ref = _ref(event, "anonymized_input_ref", seed)
    trace_ref = _ref(event, "trace_ref", seed)

    case_id = "case." + _digest(
        SCHEMA_VERSION,
        failure_type,
        source_profile,
        expected,
        forbidden,
        anonymized_input_ref,
        trace_ref,
        privacy_scope,
    )
    if not _CASE_ID_RE.match(case_id):  # pragma: no cover - defensive
        raise EvalBridgeError(f"generated case_id violates corpus grammar: {case_id!r}")

    payload = {
        "case_id": case_id,
        "failure_type": failure_type,
        "anonymized_input_ref": anonymized_input_ref,
        "expected_behavior": expected,
        "forbidden_behavior": forbidden,
        "trace_ref": trace_ref,
        "source_profile": source_profile,
        "privacy_scope": privacy_scope,
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "adapter": ADAPTER_NAME,
        "failure_type": failure_type,
        "privacy_scope": privacy_scope,
    }
    return EvalCandidate(payload=payload, metadata=metadata)


def build_registration(event: Any) -> Optional[Dict[str, str]]:
    """Return the allowlisted registration payload, or ``None`` for non-candidates."""
    candidate = build_candidate(event)
    return None if candidate is None else candidate.payload


def register_failure_case(sink: Any, event: Any, *, now: Any) -> Optional[str]:
    """Register ``event`` with a duck-typed sink; returns the ``case_id`` or ``None``.

    ``now`` is call context required by the corpus API: it is forwarded to the
    sink as an explicit keyword-only argument and is never part of the payload,
    so it cannot influence the deterministic ``case_id``.

    Errors raised by the sink propagate unchanged — the bridge never swallows a
    failed write.
    """
    payload = build_registration(event)
    if payload is None:
        return None
    register = getattr(sink, "evaluation_register_case", None)
    if not callable(register):
        raise EvalBridgeError("sink does not implement evaluation_register_case")
    register(**payload, now=now)
    return payload["case_id"]


__all__ = [
    "ADAPTER_NAME",
    "DEFAULT_PRIVACY_SCOPE",
    "EvalBridgeError",
    "EvalCandidate",
    "KNOWN_PRIVACY_SCOPES",
    "PAYLOAD_KEYS",
    "SCHEMA_VERSION",
    "SUPPORTED_KINDS",
    "build_candidate",
    "build_registration",
    "register_failure_case",
]
