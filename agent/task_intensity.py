"""Runtime-neutral task intensity classification and verification policy.

The classifier is intentionally pure and standard-library-only.  It describes
an appropriate verification budget; it never invokes tools or changes runtime
orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

_LEVELS = ("fast", "standard", "deep")


@dataclass(frozen=True)
class VerificationBudget:
    """Declarative verification policy for one intensity level."""

    targeted_check: bool = False
    focused_tests: bool = False
    diff_check: bool = False
    proportional_review: bool = False
    plan_task_list: bool = False
    delegation_when_feasible: bool = False
    broader_integration: bool = False
    security_review: bool = False
    force_delegation: bool = False
    force_browser: bool = False
    force_full_suite: bool = False

    @classmethod
    def for_level(cls, level: str) -> "VerificationBudget":
        if level == "fast":
            return cls(targeted_check=True)
        if level == "standard":
            return cls(
                focused_tests=True,
                diff_check=True,
                proportional_review=True,
            )
        if level == "deep":
            return cls(
                focused_tests=True,
                diff_check=True,
                proportional_review=True,
                plan_task_list=True,
                delegation_when_feasible=True,
                broader_integration=True,
                security_review=True,
            )
        raise ValueError(f"unknown task intensity level: {level!r}")

    def as_dict(self) -> dict[str, bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


_GUIDANCE = {
    "fast": (
        "Use a minimal verification budget: perform one targeted check. "
        "Do not force delegation, browser use, or a full test suite."
    ),
    "standard": (
        "Use a proportional verification budget: run focused tests, inspect "
        "the diff, and perform a review proportional to the change."
    ),
    "deep": (
        "Use a broad verification budget: keep a plan/task list, delegate when "
        "feasible, and include broader integration, security, and final review."
    ),
}


@dataclass(frozen=True)
class TaskIntensityDecision:
    """Immutable classification result safe to expose as turn metadata."""

    level: str
    reasons: Tuple[str, ...]
    verification: VerificationBudget

    @property
    def prompt_guidance(self) -> str:
        return _GUIDANCE[self.level]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reasons": list(self.reasons),
            "verification_budget": self.verification.as_dict(),
        }


def _contains(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def extract_request_text(request: Any) -> str:
    """Return classifier-safe text from a string or multimodal content parts.

    Only explicit text parts are read. URLs, base64 payloads, image metadata,
    and other media fields are deliberately ignored so incidental words in
    media data cannot affect lexical classification.
    """
    if isinstance(request, str):
        return request
    if not isinstance(request, (list, tuple)):
        return ""

    text_parts: list[str] = []
    for part in request:
        if not isinstance(part, Mapping):
            continue
        part_type = str(part.get("type", "")).lower()
        if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return "\n".join(text_parts)


def classify_task_intensity(
    request: Any,
    *,
    override: Optional[str] = None,
    signals: Optional[Mapping[str, Any]] = None,
) -> TaskIntensityDecision:
    """Classify a request using conservative deterministic heuristics.

    ``override`` is the sole authority when supplied.  Otherwise only the
    current request and a copied view of structured signals are considered.
    Request length is deliberately not a feature.
    """
    if override is not None:
        normalized_override = str(override).strip().lower()
        if normalized_override not in _LEVELS:
            raise ValueError("override must be one of: fast, standard, deep")
        return TaskIntensityDecision(
            normalized_override,
            (f"explicit override: {normalized_override}",),
            VerificationBudget.for_level(normalized_override),
        )

    text = extract_request_text(request)
    facts = dict(signals or {})
    deep_reasons: list[str] = []

    artifact_count = facts.get("artifact_count", 0)
    try:
        if int(artifact_count) > 1:
            deep_reasons.append("multiple artifacts")
    except (TypeError, ValueError):
        pass
    if facts.get("multi_artifact"):
        deep_reasons.append("multiple artifacts")
    if facts.get("high_risk"):
        deep_reasons.append("high-risk structured signal")
    if facts.get("security_sensitive"):
        deep_reasons.append("security-sensitive structured signal")
    if facts.get("deployment"):
        deep_reasons.append("deployment structured signal")
    if facts.get("research_heavy"):
        deep_reasons.append("research-heavy structured signal")

    # These markers express work/depth intent in their own right. Risk nouns
    # such as "security", "auth", and "research" are handled separately so a
    # read-only definition or ordinary question is not escalated by subject.
    marker_groups = (
        ("explicit depth request", (r"\bthorough(?:ly)?\b", r"\bin[- ]depth\b", r"\bdeep(?:ly)?\b", r"꼼꼼", r"철저")),
        ("migration work", (r"\bmigrat(?:e|ion|ing)\b", r"마이그레이션")),
        ("deployment work", (r"\bdeploy(?:ment|ing)?\b", r"\brelease to production\b", r"배포")),
        ("audit work", (r"\baudit(?:ing)?\b", r"감사")),
        ("research-heavy work", (r"\binvestigat(?:e|ion)\b", r"\bresearch and (?:compare|evaluate|build)\b", r"조사(?:해|하고|하여)")),
        ("multi-artifact work", (r"\bmulti[- ]artifact\b", r"\bacross (?:multiple|several) (?:files|services|packages)\b")),
    )
    for reason, patterns in marker_groups:
        if _contains(text, patterns):
            deep_reasons.append(reason)

    risk_subject = _contains(
        text,
        (r"\bsecurity\b", r"\bsecure(?:ly)?\b", r"\bauth(?:entication|orization)?\b", r"보안"),
    )
    action_intent = _contains(
        text,
        (
            r"\b(?:implement|change|modify|fix|build|configure|rotate|replace|harden|patch)\b",
            r"\b(?:migrat(?:e|ion|ing)|deploy(?:ment|ing)?|audit(?:ing)?)\b",
            r"(?:구현|변경|수정|교체|강화|패치|마이그레이션|배포|감사)",
        ),
    )
    if risk_subject and action_intent:
        deep_reasons.append("security-sensitive action")

    if deep_reasons:
        reasons = tuple(dict.fromkeys(deep_reasons))
        return TaskIntensityDecision("deep", reasons, VerificationBudget.for_level("deep"))

    fast_reasons: list[str] = []
    if facts.get("single_small_edit"):
        fast_reasons.append("single small edit")
    if facts.get("read_only") and facts.get("single_lookup"):
        fast_reasons.append("single read-only lookup")
    if facts.get("narrow") and facts.get("low_risk", True):
        fast_reasons.append("narrow low-risk task")

    # Lexical fast paths require strong, narrow intent rather than mere brevity.
    if _contains(text, (r"\b(?:fix|correct) (?:this |the )?typo\b", r"오타(?:를)? (?:고쳐|수정)")):
        fast_reasons.append("single small edit")
    if _contains(text, (r"\b(?:look up|find) (?:the )?(?:value|definition|documentation)\b", r"값(?:을)? 찾아")):
        fast_reasons.append("narrow lookup")

    if fast_reasons:
        reasons = tuple(dict.fromkeys(fast_reasons))
        return TaskIntensityDecision("fast", reasons, VerificationBudget.for_level("fast"))

    return TaskIntensityDecision(
        "standard",
        ("conservative default; no decisive fast or deep signal",),
        VerificationBudget.for_level("standard"),
    )
