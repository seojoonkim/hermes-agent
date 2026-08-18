"""Incident regression: delegated approvals stay with their gateway owner."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os

import pytest

from agent.delegation_context import delegated_child_context, is_delegated_child_context
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
from tools.thread_context import propagate_context_to_thread


@pytest.mark.asyncio
async def test_concurrent_zeon_sion_delegated_approvals_keep_ownership(monkeypatch):
    """A Zeon child prompt must never surface to or resolve Sion, and vice versa."""
    from gateway.run import GatewayRunner
    from tools import approval
    from tools.approval import (
        check_all_command_guards,
        register_gateway_notify,
        reset_current_session_key,
        resolve_gateway_approval,
        set_current_session_key,
        unregister_gateway_notify,
    )

    sessions = {
        "zeon": {
            "profile": "zeon-profile",
            "chat_id": "zeon-chat",
            "thread_id": "zeon-thread",
            "session_key": "telegram:zeon-chat:zeon-thread",
            "session_id": "zeon-parent-session",
            "child_id": "zeon-delegated-child",
            "command": "rm -rf /zeon-incident-data",
        },
        "sion": {
            "profile": "sion-profile",
            "chat_id": "sion-chat",
            "thread_id": "sion-thread",
            "session_key": "telegram:sion-chat:sion-thread",
            "session_id": "sion-parent-session",
            "child_id": "sion-delegated-child",
            "command": "rm -rf /sion-incident-data",
        },
    }
    poisoned = {
        "HERMES_SESSION_KEY": "poison-global-session-key",
        "HERMES_SESSION_ID": "poison-global-session-id",
    }
    for name, value in poisoned.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")

    approval._gateway_queues.clear()
    approval._gateway_notify_cbs.clear()
    approval._session_approved.clear()
    approval._permanent_approved.clear()
    approval._pending.clear()

    runner = object.__new__(GatewayRunner)

    notifications = {name: [] for name in sessions}
    observations = {}
    results = {}
    for name, spec in sessions.items():
        register_gateway_notify(
            spec["session_key"],
            lambda data, owner=name: notifications[owner].append(data),
        )

    def delegated_agent_step(owner: str):
        spec = sessions[owner]

        def tool_worker():
            observations[owner] = {
                "profile": get_session_env("HERMES_SESSION_PROFILE"),
                "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
                "thread_id": get_session_env("HERMES_SESSION_THREAD_ID"),
                "session_key": approval.get_current_session_key(),
                "session_id": get_session_env("HERMES_SESSION_ID"),
                "delegated": is_delegated_child_context(),
            }
            return check_all_command_guards(spec["command"], "local")

        with delegated_child_context(spec["child_id"]):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                # Production tool dispatch captures context on the submitting
                # agent thread; the worker itself never sets an approval key.
                return pool.submit(propagate_context_to_thread(tool_worker)).result()

    async def run_session(owner: str):
        spec = sessions[owner]
        tokens = set_session_vars(
            platform="telegram",
            chat_id=spec["chat_id"],
            thread_id=spec["thread_id"],
            session_key=spec["session_key"],
            session_id=spec["session_id"],
            profile=spec["profile"],
            cron_session="",
        )
        approval_token = set_current_session_key(spec["session_key"])
        try:
            results[owner] = await runner._run_in_executor_with_context(
                delegated_agent_step, owner
            )
        finally:
            reset_current_session_key(approval_token)
            clear_session_vars(tokens)

    tasks = {name: asyncio.create_task(run_session(name)) for name in sessions}
    try:
        async def both_notified():
            for _ in range(500):
                if all(len(items) == 1 for items in notifications.values()):
                    return
                await asyncio.sleep(0.01)
            pytest.fail("both delegated workers did not reach approval")

        await both_notified()

        for owner, spec in sessions.items():
            other = "sion" if owner == "zeon" else "zeon"
            assert spec["command"] in notifications[owner][0]["command"]
            assert spec["command"] not in notifications[other][0]["command"]
            assert observations[owner] == {
                "profile": spec["profile"],
                "chat_id": spec["chat_id"],
                "thread_id": spec["thread_id"],
                "session_key": spec["session_key"],
                "session_id": spec["child_id"],
                "delegated": True,
            }

        assert resolve_gateway_approval(sessions["zeon"]["session_key"], "once") == 1
        await asyncio.wait_for(tasks["zeon"], timeout=5)
        await asyncio.sleep(0)
        assert results["zeon"]["approved"] is True
        assert not tasks["sion"].done(), "Zeon approval released Sion"
        assert len(approval._gateway_queues[sessions["sion"]["session_key"]]) == 1

        assert resolve_gateway_approval(sessions["sion"]["session_key"], "deny") == 1
        await asyncio.wait_for(tasks["sion"], timeout=5)
        assert results["sion"]["approved"] is False
        assert results["sion"]["outcome"] == "denied"

        assert {name: os.environ[name] for name in poisoned} == poisoned
    finally:
        for spec in sessions.values():
            resolve_gateway_approval(spec["session_key"], "deny", resolve_all=True)
            unregister_gateway_notify(spec["session_key"])
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        runner._shutdown_executor()
        approval._gateway_queues.clear()
        approval._gateway_notify_cbs.clear()
        approval._session_approved.clear()
        approval._permanent_approved.clear()
        approval._pending.clear()
