"""A fire-and-forget receipt must survive GC and actually send under the real loop.

Regression for 2026-10-05 16:09: in production the receipt task was created with
a bare ``loop.create_task(...)`` (asyncio keeps only a weak reference), and no
receipt reached the user even though the same code path delivers in isolation.
The runner must keep a strong reference until the task finishes.
"""
from __future__ import annotations

import asyncio
import gc
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


class _Adapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None, **kw):
        await asyncio.sleep(0.05)  # yield, like a real network send
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="1")


@pytest.mark.asyncio
async def test_receipt_task_is_strongly_held_and_delivers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda s: adapter
    runner._thread_metadata_for_source = lambda s, *a: None
    runner._background_tasks = set()
    runner._turn_receipt_enabled = lambda s: True
    src = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="42", thread_id=None,
                          chat_type="dm", message_id=None, user_id="1")

    runner._schedule_turn_receipt(src, "배포 스크립트 고쳐줘", session_key="k")
    assert runner._background_tasks, "receipt task must be strongly referenced"
    gc.collect()
    for _ in range(60):
        await asyncio.sleep(0.05)
        if adapter.sent:
            break
    assert adapter.sent and adapter.sent[0].startswith("받았어")
    await asyncio.sleep(0.05)
    assert not runner._background_tasks, "task must be released after it finishes"
