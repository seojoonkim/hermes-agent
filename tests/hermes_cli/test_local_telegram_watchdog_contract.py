"""Regression tests for the installed multi-profile Telegram watchdog contract."""

from __future__ import annotations

import importlib.util
import plistlib
from pathlib import Path


SCRIPT = Path.home() / ".hermes" / "scripts" / "hermes-telegram-healthcheck.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("hermes_telegram_healthcheck", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_profile(root: Path, name: str, token: str = "token") -> None:
    path = root / "profiles" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(
        "telegram:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    (path / ".env").write_text(f"TELEGRAM_BOT_TOKEN={token}\n", encoding="utf-8")


def test_configured_accounts_accept_profile_scoped_bots(tmp_path, monkeypatch):
    health = _load_module()
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n", encoding="utf-8")
    for name in sorted(health.EXPECTED_PROFILE_ACCOUNTS):
        _write_profile(root, name, token=f"{name}-token")

    monkeypatch.setattr(health, "ROOT_CONFIG", root / "config.yaml")
    monkeypatch.setattr(health, "PROFILES_DIR", root / "profiles")

    assert health.configured_profile_accounts() == health.EXPECTED_PROFILE_ACCOUNTS


def test_validate_gateway_plist_accepts_official_supervisor_wrapper(tmp_path, monkeypatch):
    health = _load_module()
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": health.GATEWAY_LABEL,
                "EnvironmentVariables": {"HERMES_HOME": str(tmp_path / ".hermes")},
                "ProgramArguments": [
                    "/venv/bin/python",
                    "-m",
                    "hermes_cli.stderr_timestamp",
                    "--error-log",
                    str(tmp_path / "gateway.error.log"),
                    "--",
                    "/venv/bin/python",
                    "-m",
                    "hermes_cli.main",
                    "gateway",
                    "run",
                    "--replace",
                    "--external-supervisor",
                ],
            }
        )
    )
    monkeypatch.setattr(health, "GATEWAY_PLIST", plist)
    monkeypatch.setattr(health, "HERMES_ROOT", tmp_path / ".hermes")

    assert health.validate_gateway_plist() is True


def test_watchdog_root_gateway_disables_direct_kickstart():
    health = _load_module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "allow_direct_kick=False" in source
    assert "hermes-gateway-restart.py" in source


def test_gateway_account_evidence_accepts_profile_connection_log(monkeypatch):
    health = _load_module()
    text = "Starting Hermes Gateway...\n" + "\n".join(
        f"✓ telegram connected (profile: {name})"
        for name in sorted(health.EXPECTED_PROFILE_ACCOUNTS)
    )
    monkeypatch.setattr(health, "recent_gateway_log_text", lambda: text)

    assert health.gateway_accounts_connected(health.EXPECTED_PROFILE_ACCOUNTS) is True


def test_gateway_state_accepts_gateway_child_of_launchd_supervisor(tmp_path, monkeypatch):
    health = _load_module()
    monkeypatch.setattr(health, "service_pid", lambda _label: 100)
    monkeypatch.setattr(health, "pid_descends_from", lambda child, parent: (child, parent) == (101, 100))
    monkeypatch.setattr(
        health,
        "load_gateway_state",
        lambda: {
            "pid": 101,
            "gateway_state": "running",
            "served_profiles": ["default", *sorted(health.EXPECTED_PROFILE_ACCOUNTS)],
            "platforms": {
                "telegram": {"state": "connected"},
                **{
                    f"{name}:telegram": {"state": "connected"}
                    for name in health.EXPECTED_PROFILE_ACCOUNTS
                },
            },
        },
    )
    monkeypatch.setattr(health, "gateway_accounts_connected", lambda expected: True)

    assert health.gateway_state_healthy(health.EXPECTED_PROFILE_ACCOUNTS) is True
