from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent.conversation_loop import (
    _build_task_intensity_system_context,
    _initialize_task_intensity,
)
from agent.task_intensity import (
    TaskIntensityDecision,
    VerificationBudget,
    classify_task_intensity,
    extract_request_text,
)


def test_standard_is_the_conservative_default_and_not_based_on_length():
    short = classify_task_intensity("Help me with this")
    long_but_simple = classify_task_intensity(
        "Look up the documented value for this option and explain it to me " * 20,
        signals={"read_only": True, "single_lookup": True},
    )

    assert short.level == "standard"
    assert long_but_simple.level == "fast"
    assert "request length" not in " ".join(long_but_simple.reasons).lower()


def test_explicit_override_wins_over_conflicting_high_risk_signals():
    decision = classify_task_intensity(
        "Deploy a security migration thoroughly",
        override="fast",
        signals={"high_risk": True, "artifact_count": 8},
    )

    assert decision.level == "fast"
    assert decision.reasons == ("explicit override: fast",)


def test_invalid_override_is_rejected():
    with pytest.raises(ValueError, match="override"):
        classify_task_intensity("do it", override="turbo")


@pytest.mark.parametrize(
    "user_text,signals,reason_fragment",
    [
        ("Please do a thorough investigation", None, "depth"),
        ("Migrate the authentication system", None, "migration"),
        ("Ship this change", {"high_risk": True}, "high-risk"),
        ("Update these outputs", {"artifact_count": 3}, "multiple artifacts"),
        ("Research and compare the available approaches", None, "research"),
    ],
)
def test_deep_indicators_are_deterministic(user_text, signals, reason_fragment):
    first = classify_task_intensity(user_text, signals=signals)
    second = classify_task_intensity(user_text, signals=signals)

    assert first == second
    assert first.level == "deep"
    assert any(reason_fragment in reason for reason in first.reasons)


def test_fast_requires_narrow_low_risk_evidence():
    lookup = classify_task_intensity(
        "Find the value of this setting",
        signals={"read_only": True, "single_lookup": True},
    )
    edit = classify_task_intensity(
        "Correct this typo",
        signals={"single_small_edit": True},
    )

    assert lookup.level == "fast"
    assert edit.level == "fast"
    assert lookup.verification == VerificationBudget.for_level("fast")
    assert lookup.verification.force_delegation is False
    assert lookup.verification.force_browser is False
    assert lookup.verification.force_full_suite is False


def test_decision_and_nested_budget_are_immutable():
    decision = classify_task_intensity("Help me with this")

    with pytest.raises(FrozenInstanceError):
        decision.level = "deep"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.verification.focused_tests = False  # type: ignore[misc]


def test_each_level_has_prompt_guidance_and_serializable_metadata():
    decisions = [
        classify_task_intensity("lookup", override="fast"),
        classify_task_intensity("normal", override="standard"),
        classify_task_intensity("audit", override="deep"),
    ]

    assert "targeted check" in decisions[0].prompt_guidance
    assert "focused tests" in decisions[1].prompt_guidance
    assert "plan/task list" in decisions[2].prompt_guidance
    assert decisions[2].as_metadata() == {
        "level": "deep",
        "reasons": ["explicit override: deep"],
        "verification_budget": decisions[2].verification.as_dict(),
    }


def test_structured_signals_are_not_mutated():
    signals = {"artifact_count": 1, "read_only": True, "single_lookup": True}
    before = dict(signals)

    classify_task_intensity("look it up", signals=signals)

    assert signals == before
    assert classify_task_intensity("look it up", signals=signals).level == "fast"


def test_verification_budgets_match_level_policy():
    standard = VerificationBudget.for_level("standard")
    deep = VerificationBudget.for_level("deep")

    assert standard.focused_tests and standard.diff_check and standard.proportional_review
    assert deep.plan_task_list and deep.broader_integration
    assert deep.security_review and deep.delegation_when_feasible
    assert VerificationBudget.for_level("fast").targeted_check
    with pytest.raises(ValueError, match="level"):
        VerificationBudget.for_level("unknown")


def test_turn_initialization_exposes_metadata_and_prompt_guidance():
    agent = SimpleNamespace(
        _task_intensity_override=None,
        _task_intensity_signals={"artifact_count": 3},
    )

    decision = _initialize_task_intensity(agent, "Update the generated outputs")

    assert isinstance(decision, TaskIntensityDecision)
    assert agent._task_intensity_metadata["level"] == "deep"
    assert "broader integration" in agent._task_intensity_prompt_guidance


def test_turn_initialization_honors_agent_explicit_override():
    agent = SimpleNamespace(
        _task_intensity_override="fast",
        _task_intensity_signals={"high_risk": True},
    )

    _initialize_task_intensity(agent, "Deploy the security migration")

    assert agent._task_intensity_metadata["level"] == "fast"
    assert agent._task_intensity_metadata["reasons"] == ["explicit override: fast"]


def test_multimodal_extraction_uses_only_text_parts_not_media_payloads_or_urls():
    request = [
        {"type": "text", "text": "What is this image?"},
        {"type": "image_url", "image_url": {"url": "https://example/security-migration-audit.png"}},
        {"type": "image", "source": {"data": "deploy-security-research"}},
    ]

    assert extract_request_text(request) == "What is this image?"
    assert classify_task_intensity(request).level == "standard"


@pytest.mark.parametrize(
    "user_text",
    [
        "What is security?",
        "Define security research",
        "Explain authentication security",
        "보안의 정의가 뭐야?",
    ],
)
def test_read_only_security_or_research_questions_are_not_deep_from_risk_nouns(user_text):
    assert classify_task_intensity(user_text).level in {"fast", "standard"}


def test_security_migration_action_is_deep():
    decision = classify_task_intensity("Migrate the authentication system securely")

    assert decision.level == "deep"
    assert any("migration" in reason for reason in decision.reasons)


def test_task_intensity_guidance_is_ephemeral_system_context_for_model_request():
    decision = classify_task_intensity("Migrate the authentication system")
    persisted_messages = [{"role": "user", "content": "Migrate the authentication system"}]

    effective = _build_task_intensity_system_context("stable system", decision)

    assert effective.startswith("stable system")
    assert "[TASK INTENSITY: deep]" in effective
    assert decision.prompt_guidance in effective
    assert persisted_messages == [{"role": "user", "content": "Migrate the authentication system"}]


def test_steer_guidance_is_recomputed_and_included_with_steer_delivery():
    turn = classify_task_intensity("Fix this typo", override="fast")

    effective = _build_task_intensity_system_context(
        "stable system", turn, steer_text="thoroughly audit the deployment"
    )

    assert "[TASK INTENSITY: fast]" in effective
    assert "[STEER TASK INTENSITY: deep]" in effective
    assert "broader integration" in effective
