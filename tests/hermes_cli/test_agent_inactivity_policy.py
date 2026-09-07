"""Preparatory inactivity-policy accessors; no runtime watchdog wiring."""

import pytest
import yaml

from hermes_cli import timeouts


@pytest.fixture
def policy_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write(config):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    return write


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("get_agent_stream_idle_timeout", "stream_idle_timeout_seconds", 120),
        ("get_agent_nonstream_stale_timeout", "nonstream_stale_timeout_seconds", 180),
    ],
)
def test_explicit_policy_seconds(policy_config, name, key, value):
    policy_config({"agent": {key: value}})
    accessor = getattr(timeouts, name, None)
    assert callable(accessor), f"Missing policy accessor: {name}"
    assert accessor() == float(value)
    assert isinstance(accessor(), float)


@pytest.fixture(params=[
    ("get_agent_stream_idle_timeout", "stream_idle_timeout_seconds"),
    ("get_agent_nonstream_stale_timeout", "nonstream_stale_timeout_seconds"),
])
def policy(request):
    name, key = request.param
    return getattr(timeouts, name), key


@pytest.mark.parametrize("config", [{}, {"agent": {}}, {"agent": None}, {"agent": []}, {"agent": "invalid"}])
def test_absent_or_malformed_agent_preserves_legacy(policy_config, policy, config):
    policy_config(config)
    accessor, _ = policy
    assert accessor() is None


@pytest.mark.parametrize("value", [
    None, 0, -1, -0.5, True, False, "120", "invalid", "", [], {},
    float("nan"), float("inf"), float("-inf"), 10**400,
])
def test_invalid_policy_is_unset(policy_config, policy, value):
    accessor, key = policy
    policy_config({"agent": {key: value}})
    assert accessor() is None


@pytest.mark.parametrize("value", [0.001, 120.5, 180.0])
def test_positive_finite_fractions(policy_config, policy, value):
    accessor, key = policy
    policy_config({"agent": {key: value}})
    assert accessor() == value


def test_independent_policies_do_not_change_provider_timeouts(policy_config):
    policy_config({
        "agent": {"stream_idle_timeout_seconds": 120, "nonstream_stale_timeout_seconds": 180},
        "providers": {"custom": {"request_timeout_seconds": 900, "stale_timeout_seconds": 240}},
    })
    assert timeouts.get_agent_stream_idle_timeout() == 120
    assert timeouts.get_agent_nonstream_stale_timeout() == 180
    assert timeouts.get_provider_request_timeout("custom") == 900
    assert timeouts.get_provider_stale_timeout("custom") == 240
    assert timeouts.get_provider_stale_timeout("custom:codex-lb") is None


def test_config_read_failure_is_unset(monkeypatch, policy):
    from hermes_cli import config

    def fail():
        raise OSError("unreadable test config")

    monkeypatch.setattr(config, "load_config_readonly", fail)
    accessor, _ = policy
    assert accessor() is None


if __name__ == "__main__":
    # Run with python -m: deny sockets before collection/config imports, avoid
    # unrelated repository conftest hooks, and isolate collection-time home.
    import os
    import socket
    import tempfile

    def deny_network(*args, **kwargs):
        raise AssertionError("Network denied in inactivity policy tests")

    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="hermes-idle-policy-") as home:
        os.environ["HERMES_HOME"] = home
        raise SystemExit(pytest.main([__file__, "--noconftest", "-o", "addopts=", "-p", "no:cacheprovider", "-q"]))
