"""Real timeout methods and config, without importing agent startup side effects."""
import ast
import os
from pathlib import Path
import time
from typing import Any

import pytest
import yaml


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    from agent.model_metadata import is_local_endpoint
    from hermes_cli.timeouts import get_provider_stale_timeout

    source = Path(__file__).resolve().parents[2] / "run_agent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AIAgent")
    names = {"_resolved_api_call_stale_timeout_base", "_compute_non_stream_stale_timeout", "_stale_timeout_is_explicit"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    namespace = {"Any": Any, "os": os, "time": time,
                 "get_provider_stale_timeout": get_provider_stale_timeout,
                 "is_local_endpoint": is_local_endpoint}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    harness = type("TimeoutHarness", (), {name: namespace[name] for name in names})()
    harness.provider = "custom"
    harness.model = "ordinary-model"
    harness.base_url = "https://example.invalid/v1"
    harness.run_budget_seconds = None
    return harness


def configure(tmp_path, value=None, **extra):
    config = {"agent": {} if value is None else {"nonstream_stale_timeout_seconds": value}, **extra}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_policy_wins_over_legacy_floor_context_local_and_budget(agent, tmp_path, monkeypatch):
    configure(tmp_path, 12.5, providers={"custom": {"stale_timeout_seconds": 800}})
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "900")
    agent.model = "deepseek-r1"
    agent.base_url = "http://localhost:8000/v1"
    agent.run_budget_seconds = 1
    agent._run_budget_started_at = time.time()
    payload = {"messages": [{"role": "user", "content": "x" * 600_000}]}
    assert agent._compute_non_stream_stale_timeout(payload) == 12.5


def test_policy_alone_is_explicit_and_not_budget_capped(agent, tmp_path):
    configure(tmp_path, 180)
    agent.run_budget_seconds = 1
    agent._run_budget_started_at = time.time()
    assert agent._stale_timeout_is_explicit() is True
    assert agent._compute_non_stream_stale_timeout([]) == 180


@pytest.mark.parametrize("value", [None, 0, -1, True, "180", float("inf"), float("nan")])
def test_absent_or_invalid_policy_preserves_default(agent, tmp_path, value):
    configure(tmp_path, value)
    assert agent._stale_timeout_is_explicit() is False
    assert agent._compute_non_stream_stale_timeout([]) == 90


def test_absent_policy_preserves_reasoning_floor_and_budget(agent, tmp_path):
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    configure(tmp_path)
    agent.model = "deepseek-r1"
    floor = get_reasoning_stale_timeout_floor(agent.model)
    assert floor is not None and floor > 60
    assert agent._compute_non_stream_stale_timeout([]) == floor
    assert agent._stale_timeout_is_explicit() is False
    agent.run_budget_seconds = 1
    agent._run_budget_started_at = time.time()
    assert agent._compute_non_stream_stale_timeout([]) == 60


def test_absent_policy_preserves_local_exemption_and_context_scaling(agent, tmp_path):
    configure(tmp_path)
    agent.base_url = "http://localhost:8000/v1"
    assert agent._compute_non_stream_stale_timeout([]) == float("inf")
    agent.base_url = "https://example.invalid/v1"
    assert agent._compute_non_stream_stale_timeout({"messages": [{"role": "user", "content": "x" * 600_000}]}) == 240


@pytest.mark.parametrize("via_env", [False, True])
def test_absent_policy_preserves_legacy_explicit_timeout(agent, tmp_path, monkeypatch, via_env):
    configure(tmp_path, providers={} if via_env else {"custom": {"stale_timeout_seconds": 300}})
    if via_env:
        monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "300")
    agent.run_budget_seconds = 1
    agent._run_budget_started_at = time.time()
    assert agent._stale_timeout_is_explicit() is True
    assert agent._compute_non_stream_stale_timeout([]) == 300


if __name__ == "__main__":
    import socket
    import tempfile

    def deny_network(*args, **kwargs):
        raise AssertionError("Network denied in nonstream inactivity tests")

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="hermes-nonstream-") as home:
        os.environ["HERMES_HOME"] = home
        raise SystemExit(pytest.main([__file__, "--noconftest", "-o", "addopts=", "-p", "no:cacheprovider", "-q"]))
