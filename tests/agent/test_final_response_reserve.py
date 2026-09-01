"""Contracts for the final-response iteration reserve."""

from agent.conversation_loop import (
    _resolve_final_response_reserve,
    _should_enter_final_response_reserve,
)


def test_default_reserve_is_three(monkeypatch):
    monkeypatch.delenv("HERMES_FINAL_RESPONSE_RESERVE", raising=False)
    assert _resolve_final_response_reserve(90) == 3


def test_invalid_reserve_uses_default(monkeypatch):
    monkeypatch.setenv("HERMES_FINAL_RESPONSE_RESERVE", "not-a-number")
    assert _resolve_final_response_reserve(90) == 3


def test_small_budget_still_allows_one_normal_call(monkeypatch):
    monkeypatch.setenv("HERMES_FINAL_RESPONSE_RESERVE", "3")
    assert _resolve_final_response_reserve(2) == 1


def test_enters_reserve_at_87_of_90():
    assert _should_enter_final_response_reserve(
        api_call_count=87, remaining=3, reserve=3, grace_call=False
    ) is True


def test_does_not_reserve_before_first_call_or_during_grace():
    assert _should_enter_final_response_reserve(
        api_call_count=0, remaining=3, reserve=3, grace_call=False
    ) is False
    assert _should_enter_final_response_reserve(
        api_call_count=87, remaining=3, reserve=3, grace_call=True
    ) is False
