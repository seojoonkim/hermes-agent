"""Receipt gating regressions found by the fleet test sweep (2026-10-05).

1. Greetings / chit-chat ("hello", "안녕") are not work requests — no receipt.
2. The receipt is a Telegram-first feature (Simon's surface); other platforms
   (Slack threads, Discord) keep their existing UX unless opted in, so a
   receipt never collides with platform-specific first-message flows.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.turn_receipt import is_trivial_message
from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.mark.parametrize("text", ["hello", "hi", "안녕", "안녕하세요", "hello there", "고마워!", "ㅎㅇ"])
def test_greetings_are_trivial(text):
    assert is_trivial_message(text)


@pytest.mark.parametrize("text", ["사노 덱 레이아웃 충돌 다 고쳐줘", "PR 머지해줘", "fix the login bug"])
def test_work_requests_are_not_trivial(text):
    assert not is_trivial_message(text)


def _runner_with_cfg(cfg):
    r = object.__new__(GatewayRunner)
    import gateway.run as gr
    r._cfg_for_test = cfg
    return r, gr


def test_receipt_defaults_on_for_telegram_off_elsewhere(monkeypatch):
    import gateway.run as gr
    monkeypatch.setattr(gr, "_load_gateway_config", lambda: {})
    r = object.__new__(GatewayRunner)
    tg = SimpleNamespace(platform=Platform.TELEGRAM)
    sl = SimpleNamespace(platform=Platform.SLACK)
    assert r._turn_receipt_enabled(tg) is True
    assert r._turn_receipt_enabled(sl) is False


def test_receipt_can_be_opted_in_per_platform(monkeypatch):
    import gateway.run as gr
    monkeypatch.setattr(gr, "_load_gateway_config", lambda: {
        "display": {"platforms": {"slack": {"turn_receipt": True}}}})
    r = object.__new__(GatewayRunner)
    assert r._turn_receipt_enabled(SimpleNamespace(platform=Platform.SLACK)) is True
