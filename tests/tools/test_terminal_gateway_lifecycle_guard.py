"""Shared gateway lifecycle commands must not run inside its process tree."""

import json
import os

import pytest

import tools.terminal_tool as terminal_tool


@pytest.fixture(autouse=True)
def supervised_gateway_tree(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", "1")


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


def test_separate_shell_without_gateway_marker_keeps_non_destructive_start_available(monkeypatch):
    monkeypatch.delenv("_HERMES_GATEWAY")

    from cron.lifecycle_guard import (
        contains_gateway_lifecycle_command,
        contains_unqualified_gateway_start_or_install,
    )

    assert contains_gateway_lifecycle_command("hermes gateway start") is False
    assert contains_unqualified_gateway_start_or_install("hermes gateway start") is True
