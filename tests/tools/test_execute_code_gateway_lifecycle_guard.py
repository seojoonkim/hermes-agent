"""Gateway tool execution cannot bypass lifecycle guards via execute_code."""

import json

import tools.code_execution_tool as code_execution_tool


def test_supervised_gateway_blocks_lifecycle_python(monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(
        "tools.process_registry._is_supervised_gateway_process",
        lambda: True,
    )
    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        lambda *args, **kwargs: {"approved": True},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        lambda: {"env_type": "ssh"},
    )
    monkeypatch.setattr(
        "tools.terminal_tool._docker_has_host_access",
        lambda config: False,
    )
    monkeypatch.setattr(
        code_execution_tool,
        "_execute_remote",
        lambda *args, **kwargs: json.dumps({"status": "escaped"}),
    )

    result = json.loads(
        code_execution_tool.execute_code(
            "import os\nos.system('hermes gateway start')",
            enabled_tools=[],
        )
    )

    assert result.get("status") != "escaped"
    assert "Blocked:" in result["error"]
