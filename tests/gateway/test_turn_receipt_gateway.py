"""Integration: the receipt is scheduled before transcript load / hygiene.

Uses the real GatewayRunner._schedule_turn_receipt against a fake adapter and
a temp HERMES_HOME (no MemKraft ledger -> no ETA), and asserts:
  * a work request produces exactly one receipt with the restated request;
  * trivial and internal events produce none;
  * the hook sits BEFORE session hygiene in _handle_message_with_agent
    (source-order contract — compression must never delay the receipt).
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


class _Adapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="7")


def _runner(adapter):
    r = object.__new__(GatewayRunner)
    r._adapter_for_source = lambda source: adapter
    r._thread_metadata_for_source = lambda source: None
    r._turn_receipt_enabled = lambda source: True
    return r


def _source():
    return SimpleNamespace(platform=Platform.TELEGRAM, chat_id="46291309", thread_id=None)


@pytest.mark.asyncio
async def test_schedule_turn_receipt_delivers_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _Adapter()
    _runner(adapter)._schedule_turn_receipt(_source(), "사노 덱 레이아웃 충돌 다 고쳐줘")
    for _ in range(50):
        if adapter.sent:
            break
        await asyncio.sleep(0.02)
    assert len(adapter.sent) == 1
    assert "사노 덱 레이아웃 충돌" in adapter.sent[0]


@pytest.mark.asyncio
async def test_trivial_message_schedules_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _Adapter()
    _runner(adapter)._schedule_turn_receipt(_source(), "ㅇㅇ")
    await asyncio.sleep(0.2)
    assert adapter.sent == []


def test_receipt_hook_runs_before_session_hygiene():
    src = inspect.getsource(GatewayRunner._handle_message_with_agent)
    hook = src.index("self._schedule_turn_receipt(")
    hygiene = src.index("Session hygiene: auto-compress")
    transcript = src.index("history = await self.async_session_store.load_transcript")
    assert hook < transcript < hygiene
    # internal wake events (cron/kanban/process notifications) never get receipts
    assert 'if not getattr(event, "internal", False):' in src[hook - 300:hook]
