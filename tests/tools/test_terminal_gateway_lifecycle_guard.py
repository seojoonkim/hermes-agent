"""Shared gateway lifecycle commands must not run inside its process tree."""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import tools.terminal_tool as terminal_tool


@pytest.fixture(autouse=True)
def isolated_terminal_execution(monkeypatch, tmp_path):
    """Never acquire a real backend, even if the lifecycle guard regresses."""
    execute = Mock(side_effect=pytest.fail.Exception("unexpected terminal execution"))
    env = SimpleNamespace(cwd=str(tmp_path), execute=execute)
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_create_environment", Mock(return_value=env))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 5,
    })
    # Avoid approval helpers invoking external security scanners.
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **kw: {
        "approved": True,
    })
    return execute


def test_bypassed_lifecycle_guard_cannot_execute(monkeypatch, isolated_terminal_execution):
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)

    with pytest.raises(pytest.fail.Exception, match="unexpected terminal execution"):
        _result("printf hermes-isolation-probe")

    isolated_terminal_execution.assert_called_once()
    assert isolated_terminal_execution.call_args.args[0] == "printf hermes-isolation-probe"


@pytest.fixture(autouse=True)
def supervised_gateway_tree(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", "1")
    # Emulate the owning process, not merely its inherited environment.
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **kwargs: os.getpid())


def _result(command: str) -> dict:
    return json.loads(terminal_tool.terminal_tool(command=command, timeout=5))


@pytest.mark.parametrize(
    "command",
    [
        "launchctl kickstart -k gui/501/ai.hermes.gateway",
        "launchctl bootout gui/501/ai.hermes.gateway",
        "hermes gateway restart",
        "hermes gateway stop",
        "hermes gateway start",
        "hermes gateway install",
    ],
)
def test_gateway_tree_blocks_every_shared_lifecycle_entrypoint(command):
    blocked = _result(command)

    assert blocked["exit_code"] != 0
    assert "Blocked:" in blocked["error"]


def test_launchctl_submit_guidance_never_recommends_gateway_owned_cron():
    blocked = _result("launchctl submit -l hermes-test -- /usr/bin/true")

    assert blocked["exit_code"] != 0
    assert "Use Hermes cron for one-shot delayed work" not in blocked["error"]
    assert "separate shell outside the gateway process" in blocked["error"]
    assert "Hermes cron runs inside the gateway" in blocked["error"]


def test_separate_shell_without_gateway_marker_keeps_non_destructive_start_available(monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY")

    from cron.lifecycle_guard import (
        contains_gateway_lifecycle_command,
        contains_unqualified_gateway_start_or_install,
    )

    assert contains_gateway_lifecycle_command("hermes gateway start") is False
    assert contains_unqualified_gateway_start_or_install("hermes gateway start") is True
