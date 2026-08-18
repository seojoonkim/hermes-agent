from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from ruamel.yaml import YAML

SCRIPT = Path(__file__).parents[2] / "scripts" / "migrate_telegram_accounts.py"
spec = importlib.util.spec_from_file_location("migrate_telegram_accounts", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

EXPECTED = ("zeon", "sion", "mion", "sano", "raon", "kara")
TOKENS = {name: f"100{name}:SECRET-{name}" for name in EXPECTED}


def dump_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(value, handle)


def load_yaml(path: Path) -> dict:
    yaml = YAML(typ="safe")
    return yaml.load(path.read_text(encoding="utf-8"))


def make_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    accounts = []
    for index, name in enumerate(EXPECTED):
        account = {
            ("name" if index % 2 else "profile"): name,
            "token": TOKENS[name],
            "allowed_users": [index],
            "reply_mode": "quote",
        }
        accounts.append(account)
        profile = home / "profiles" / name
        dump_yaml(
            profile / "config.yaml",
            {
                "model": {"default": f"model-{name}"},
                "memory": {"provider": "local"},
                "approvals": {"required": True},
                "telegram": [] if name == "kara" else {"reply_mode": "old", "keep": name},
            },
        )
        (profile / ".env").write_text(
            f"# {name} secrets\nOTHER_SECRET=keep-{name}\nTELEGRAM_BOT_TOKEN=old-{name}\n",
            encoding="utf-8",
        )
    dump_yaml(
        home / "config.yaml",
        {
            "agent": {"name": "root"},
            "telegram": {"accounts": accounts, "root_keep": True, "enabled": True},
        },
    )
    (home / ".env").write_text(
        f"ROOT_SECRET=keep\n{module.TOKEN_KEY}={TOKENS['zeon']}\n",
        encoding="utf-8",
    )
    return home


def snapshot(home: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(home)): path.read_bytes()
        for path in home.rglob("*")
        if path.is_file()
    }


def test_dry_run_redacts_tokens_and_writes_nothing(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    before = snapshot(home)

    plan = module.migrate(home, apply=False)
    rendered = json.dumps(plan, sort_keys=True)

    assert snapshot(home) == before
    assert all(token not in rendered for token in TOKENS.values())
    assert all(module.token_fingerprint(token) in rendered for token in TOKENS.values())


def test_apply_migrates_preserves_settings_and_is_idempotent(tmp_path: Path) -> None:
    home = make_home(tmp_path)

    result = module.migrate(home, apply=True)

    root = load_yaml(home / "config.yaml")
    assert root["agent"] == {"name": "root"}
    assert root["telegram"] == {"root_keep": True, "enabled": False}
    assert root["gateway"]["multiplex_profiles"] is True
    assert root["gateway"]["multiplex_profile_allowlist"] == list(EXPECTED)
    root_env = (home / ".env").read_text(encoding="utf-8")
    assert root_env == "ROOT_SECRET=keep\n"
    assert "TELEGRAM_BOT_TOKEN" not in root_env
    for index, name in enumerate(EXPECTED):
        config = load_yaml(home / "profiles" / name / "config.yaml")
        assert config["model"] == {"default": f"model-{name}"}
        assert config["memory"] == {"provider": "local"}
        assert config["approvals"] == {"required": True}
        assert config["telegram"]["allowed_users"] == [index]
        assert config["telegram"]["reply_mode"] == "quote"
        if name != "kara":
            assert config["telegram"]["keep"] == name
        env = (home / "profiles" / name / ".env").read_text(encoding="utf-8")
        assert f"OTHER_SECRET=keep-{name}" in env
        assert env.count("TELEGRAM_BOT_TOKEN=") == 1
        assert f"TELEGRAM_BOT_TOKEN={TOKENS[name]}" in env
    manifest = Path(result["manifest"])
    assert manifest.is_file()
    manifest_text = manifest.read_text(encoding="utf-8")
    assert all(token not in manifest_text for token in TOKENS.values())

    after = snapshot(home)
    second = module.migrate(home, apply=True)
    assert second["status"] == "already_migrated"
    assert snapshot(home) == after


def test_rollback_restores_exact_original_files(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    before = snapshot(home)
    result = module.migrate(home, apply=True)

    module.rollback(Path(result["manifest"]))

    restored = snapshot(home)
    for relative, content in before.items():
        assert restored[relative] == content
    assert not any((home / "backups" / "telegram-accounts").glob("*/manifest.json"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda accounts: accounts.pop(),
        lambda accounts: accounts.append(dict(accounts[0])),
        lambda accounts: accounts.__setitem__(1, {**accounts[1], "profile": "zeon", "name": "zeon"}),
        lambda accounts: accounts.__setitem__(1, {**accounts[1], "token": accounts[0]["token"]}),
        lambda accounts: accounts.__setitem__(0, {**accounts[0], "token": ""}),
        lambda accounts: accounts.__setitem__(0, "not-a-mapping"),
    ],
)
def test_invalid_accounts_fail_closed_without_writes(tmp_path: Path, mutate) -> None:
    home = make_home(tmp_path)
    root = load_yaml(home / "config.yaml")
    mutate(root["telegram"]["accounts"])
    dump_yaml(home / "config.yaml", root)
    before = snapshot(home)

    with pytest.raises(module.MigrationError):
        module.migrate(home, apply=True)

    assert snapshot(home) == before


def test_conflicting_profile_and_name_fails_closed(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    root = load_yaml(home / "config.yaml")
    root["telegram"]["accounts"][0]["name"] = "sion"
    dump_yaml(home / "config.yaml", root)
    before = snapshot(home)

    with pytest.raises(module.MigrationError):
        module.migrate(home, apply=True)

    assert snapshot(home) == before


def test_bad_root_or_profile_structure_fails_closed(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    dump_yaml(home / "profiles" / "zeon" / "config.yaml", ["bad"])
    before = snapshot(home)

    with pytest.raises(module.MigrationError):
        module.migrate(home, apply=True)

    assert snapshot(home) == before


def test_legacy_telegram_list_is_accepted_only_for_kara(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    dump_yaml(home / "profiles" / "zeon" / "config.yaml", {"telegram": []})
    before = snapshot(home)

    with pytest.raises(module.MigrationError):
        module.migrate(home, apply=True)

    assert snapshot(home) == before


def test_duplicate_env_token_assignment_fails_closed(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    env_path = home / "profiles" / "zeon" / ".env"
    env_path.write_text(
        f"{module.TOKEN_KEY}=old-one\n{module.TOKEN_KEY}=old-two\n",
        encoding="utf-8",
    )
    before = snapshot(home)

    with pytest.raises(module.MigrationError):
        module.migrate(home, apply=True)

    assert snapshot(home) == before


def test_mismatched_root_token_fails_closed(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    (home / ".env").write_text(
        "ROOT_SECRET=keep\nTELEGRAM_BOT_TOKEN=wrong-token\n",
        encoding="utf-8",
    )
    before = snapshot(home)

    with pytest.raises(module.MigrationError, match="does not match"):
        module.migrate(home, apply=True)

    assert snapshot(home) == before


def test_env_without_token_gets_appended_preserving_final_newline(tmp_path: Path) -> None:
    home = make_home(tmp_path)
    env_path = home / "profiles" / "zeon" / ".env"
    env_path.write_text("A=1\n# keep\n", encoding="utf-8")

    module.migrate(home, apply=True)

    assert env_path.read_text(encoding="utf-8") == f"A=1\n# keep\nTELEGRAM_BOT_TOKEN={TOKENS['zeon']}\n"


def test_atomic_write_failure_rolls_back_all_touched_files(tmp_path: Path, monkeypatch) -> None:
    home = make_home(tmp_path)
    before = snapshot(home)
    original = module.atomic_write
    calls = 0

    def fail_midway(path: Path, content: bytes, mode: int | None = None) -> None:
        nonlocal calls
        calls += 1
        # Thirteen backups plus the manifest are written before target writes.
        # Fail during target mutation to exercise transactional restoration.
        if calls == 18:
            raise OSError("injected write failure")
        original(path, content, mode)

    monkeypatch.setattr(module, "atomic_write", fail_midway)
    with pytest.raises(OSError, match="injected"):
        module.migrate(home, apply=True)

    restored = snapshot(home)
    for relative, content in before.items():
        assert restored[relative] == content
