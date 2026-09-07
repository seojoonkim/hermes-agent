"""Restart replay discovers only the multiplexer's served profile stores."""
import json
import time
from collections import Counter
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_constants import get_hermes_home
from tools import async_delegation as durable


@pytest.mark.asyncio
@pytest.mark.parametrize("multiplex", [True, False])
@pytest.mark.parametrize("active", ["default", "sano"])
async def test_restart_watcher_restores_served_profile_databases(
    tmp_path, monkeypatch, multiplex, active,
):
    from hermes_cli import profiles
    from tools import process_registry

    root = tmp_path / "hermes"
    homes = {"default": root, "sano": root / "profiles" / "sano",
             "excluded": root / "profiles" / "excluded"}
    monkeypatch.setenv("HERMES_HOME", str(homes[active]))
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: root / "profiles")
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: active)
    monkeypatch.setattr(durable, "_records", {})
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        event = {"type": "async_delegation", "delegation_id": name,
                 "parent_session_id": "parent-" + name,
                 "session_key": f"agent:{'main' if name == 'default' else name}:telegram:dm:1"}
        with _profile_runtime_scope(home), durable._transaction() as conn:
            conn.execute(
                "INSERT INTO async_delegations "
                "(delegation_id, origin_session, parent_session_id, state, "
                "dispatched_at, completed_at, updated_at, event_json) "
                "VALUES (?, ?, ?, 'completed', ?, ?, ?, ?)",
                (name, event["session_key"], event["parent_session_id"],
                 time.time(), time.time(), time.time(), json.dumps(event)),
            )

    # A fresh registry simulates process startup: no old queue or memory survives.
    registry = process_registry.ProcessRegistry()
    assert registry.completion_queue.qsize() == 1  # current home only
    monkeypatch.setattr(process_registry, "process_registry", registry)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        multiplex_profiles=multiplex, multiplex_profile_allowlist=["sano"],
    )
    runner._running = True
    runner._deliver_async_delegation_group = AsyncMock(return_value=True)

    ticks = 0
    async def two_ticks(delay):
        nonlocal ticks
        if delay != 3:
            ticks += 1
            runner._running = ticks < 2
    monkeypatch.setattr("gateway.run.asyncio.sleep", two_ticks)
    await runner._async_delegation_watcher(interval=0)

    events = [event for call in runner._deliver_async_delegation_group.await_args_list
              for event in call.args[0]]
    assert Counter(event["delegation_id"] for event in events) == Counter(
        ["default", "sano"] if multiplex else [active]
    )
    assert all(event["restored"] is True for event in events)
    assert registry.completion_queue.empty()
    assert get_hermes_home() == homes[active]
    # Loading does not acknowledge delivery, and excluded stores stay untouched.
    for name, home in homes.items():
        with _profile_runtime_scope(home), durable._transaction() as conn:
            assert conn.execute(
                "SELECT delivery_state, delivery_attempts FROM async_delegations "
                "WHERE delegation_id=?", (name,),
            ).fetchone() == ("pending", 0)
