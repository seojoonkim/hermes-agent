#!/usr/bin/env python3
"""Safely migrate legacy root ``telegram.accounts`` into profile-local files.

The command is dry-run by default and requires an explicit Hermes home.  It is
intentionally scoped to the six profiles used by this one-time migration.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

EXPECTED_PROFILES = ("zeon", "sion", "mion", "sano", "raon", "kara")
TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
TOKEN_LINE = re.compile(r"^\s*(?:export\s+)?TELEGRAM_BOT_TOKEN\s*=")


class MigrationError(RuntimeError):
    """Raised before mutation when the legacy input is not exactly expected."""


def token_fingerprint(token: str) -> str:
    """Return a non-reversible, short identifier suitable for plans/logs."""
    return f"sha256:{hashlib.sha256(token.encode()).hexdigest()[:12]}"


def _yaml() -> YAML:
    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


def _load_mapping(path: Path) -> CommentedMap:
    if not path.is_file():
        raise MigrationError(f"required file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = _yaml().load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise MigrationError(f"cannot read YAML: {path}") from exc
    if not isinstance(value, CommentedMap):
        raise MigrationError(f"YAML root must be a mapping: {path}")
    return value


def _dump_yaml(value: CommentedMap) -> bytes:
    stream = io.StringIO()
    _yaml().dump(value, stream)
    return stream.getvalue().encode("utf-8")


def _profile_name(account: CommentedMap) -> str:
    profile = account.get("profile")
    name = account.get("name")
    if profile is not None and name is not None and profile != name:
        raise MigrationError("account profile and name disagree")
    target = profile if profile is not None else name
    if not isinstance(target, str) or target not in EXPECTED_PROFILES:
        raise MigrationError("account must name one expected profile")
    return target


def _validated_accounts(root: CommentedMap) -> dict[str, tuple[str, dict[str, Any]]]:
    telegram = root.get("telegram")
    if not isinstance(telegram, CommentedMap):
        raise MigrationError("root telegram must be a mapping")
    accounts = telegram.get("accounts")
    if not isinstance(accounts, list) or len(accounts) != len(EXPECTED_PROFILES):
        raise MigrationError("telegram.accounts must contain exactly six entries")

    result: dict[str, tuple[str, dict[str, Any]]] = {}
    tokens: set[str] = set()
    for account in accounts:
        if not isinstance(account, CommentedMap):
            raise MigrationError("every account must be a mapping")
        target = _profile_name(account)
        token = account.get("token")
        if not isinstance(token, str) or not token or token != token.strip():
            raise MigrationError(f"{target}: token must be a non-empty string")
        if target in result:
            raise MigrationError(f"duplicate profile: {target}")
        if token in tokens:
            raise MigrationError("duplicate token")
        options = {
            key: value
            for key, value in account.items()
            if key not in {"name", "profile", "token"}
        }
        if any(not isinstance(key, str) for key in options):
            raise MigrationError(f"{target}: Telegram option keys must be strings")
        result[target] = (token, options)
        tokens.add(token)
    if set(result) != set(EXPECTED_PROFILES):
        raise MigrationError("accounts do not match the expected profile set")
    return result


def _updated_env(original: bytes, token: str, path: Path) -> bytes:
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MigrationError(f".env is not UTF-8: {path}") from exc
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if TOKEN_LINE.match(line)]
    if len(matches) > 1:
        raise MigrationError(f"duplicate {TOKEN_KEY} lines: {path}")
    replacement = f"{TOKEN_KEY}={token}"
    if matches:
        index = matches[0]
        ending = "\r\n" if lines[index].endswith("\r\n") else "\n" if lines[index].endswith("\n") else ""
        lines[index] = replacement + ending
    else:
        if text and not text.endswith(("\n", "\r")):
            lines.append("\n")
        lines.append(replacement + "\n")
    return "".join(lines).encode("utf-8")


def _root_env_without_telegram_token(
    original: bytes, expected_token: str, path: Path
) -> bytes:
    """Remove the legacy process-global Telegram token after profile migration."""
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MigrationError(f".env is not UTF-8: {path}") from exc
    lines = text.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if TOKEN_LINE.match(line)]
    if len(matches) > 1:
        raise MigrationError(f"duplicate {TOKEN_KEY} lines: {path}")
    if not matches:
        return original
    assignment = lines[matches[0]].split("=", 1)[1].strip().strip("\"'")
    if assignment != expected_token:
        raise MigrationError(
            "root TELEGRAM_BOT_TOKEN does not match the zeon account token"
        )
    del lines[matches[0]]
    return "".join(lines).encode("utf-8")


def atomic_write(path: Path, content: bytes, mode: int | None = None) -> None:
    """Write bytes using a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _mode(path: Path, default: int) -> int:
    return path.stat().st_mode & 0o777 if path.exists() else default


def _already_migrated(root: CommentedMap) -> bool:
    telegram = root.get("telegram")
    gateway = root.get("gateway")
    return (
        isinstance(telegram, CommentedMap)
        and "accounts" not in telegram
        and isinstance(gateway, CommentedMap)
        and gateway.get("multiplex_profiles") is True
        and list(gateway.get("multiplex_profile_allowlist") or []) == list(EXPECTED_PROFILES)
    )


def _build_changes(home: Path) -> tuple[dict[Path, bytes], dict[str, Any]]:
    root_path = home / "config.yaml"
    root = _load_mapping(root_path)
    if _already_migrated(root):
        return {}, {"status": "already_migrated", "profiles": list(EXPECTED_PROFILES)}
    accounts = _validated_accounts(root)
    changes: dict[Path, bytes] = {}
    plan_profiles = []

    for target in EXPECTED_PROFILES:
        token, options = accounts[target]
        profile_dir = home / "profiles" / target
        config_path = profile_dir / "config.yaml"
        env_path = profile_dir / ".env"
        config = _load_mapping(config_path)
        telegram = config.get("telegram")
        if telegram is None or (target == "kara" and isinstance(telegram, list)):
            telegram = CommentedMap()
            config["telegram"] = telegram
        elif not isinstance(telegram, CommentedMap):
            raise MigrationError(f"profile telegram must be a mapping or legacy list: {config_path}")
        for key, value in options.items():
            telegram[key] = value
        try:
            env_original = env_path.read_bytes() if env_path.exists() else b""
        except OSError as exc:
            raise MigrationError(f"cannot read .env: {env_path}") from exc
        changes[config_path] = _dump_yaml(config)
        changes[env_path] = _updated_env(env_original, token, env_path)
        plan_profiles.append({
            "profile": target,
            "token": token_fingerprint(token),
            "telegram_option_keys": sorted(str(key) for key in options),
        })

    telegram = root["telegram"]
    del telegram["accounts"]
    # The official multiplexer always serves the default profile as well as the
    # named allowlist. Keep default from polling Zeon's migrated credential.
    telegram["enabled"] = False
    gateway = root.get("gateway")
    if gateway is None:
        gateway = CommentedMap()
        root["gateway"] = gateway
    elif not isinstance(gateway, CommentedMap):
        raise MigrationError("root gateway must be a mapping")
    gateway["multiplex_profiles"] = True
    gateway["multiplex_profile_allowlist"] = list(EXPECTED_PROFILES)
    changes[root_path] = _dump_yaml(root)

    root_env_path = home / ".env"
    try:
        root_env = root_env_path.read_bytes() if root_env_path.exists() else b""
    except OSError as exc:
        raise MigrationError(f"cannot read .env: {root_env_path}") from exc
    changes[root_env_path] = _root_env_without_telegram_token(
        root_env, accounts["zeon"][0], root_env_path
    )
    return changes, {"status": "planned", "profiles": plan_profiles}


def _backup_and_manifest(home: Path, changes: dict[Path, bytes]) -> tuple[Path, dict[str, Any]]:
    backup_root = home / "backups" / "telegram-accounts" / f"{time.time_ns()}-{os.getpid()}"
    try:
        backup_root.mkdir(parents=True, mode=0o700)
        os.chmod(backup_root, 0o700)
        files = []
        for index, path in enumerate(changes):
            relative = path.relative_to(home)
            existed = path.exists()
            entry: dict[str, Any] = {"path": str(relative), "existed": existed}
            if existed:
                content = path.read_bytes()
                backup_name = f"{index:02d}.bak"
                backup_path = backup_root / backup_name
                atomic_write(backup_path, content, 0o600)
                entry.update({
                    "backup": backup_name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "mode": _mode(path, 0o600),
                })
            files.append(entry)
        manifest = {"version": 1, "home": str(home.resolve()), "files": files}
        manifest_path = backup_root / "manifest.json"
        atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode(), 0o600)
        return manifest_path, manifest
    except Exception:
        shutil.rmtree(backup_root, ignore_errors=True)
        raise


def _restore(manifest_path: Path, manifest: dict[str, Any], *, remove_backup: bool) -> None:
    home = Path(manifest["home"])
    backup_root = manifest_path.parent
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise MigrationError("unsafe path in rollback manifest")
        path = home / relative
        if entry["existed"]:
            content = (backup_root / entry["backup"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise MigrationError(f"backup checksum mismatch: {entry['path']}")
            atomic_write(path, content, int(entry["mode"]))
        else:
            path.unlink(missing_ok=True)
    if remove_backup:
        shutil.rmtree(backup_root)


def rollback(manifest_path: Path) -> None:
    """Restore every migrated file from a generated manifest and remove backup."""
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MigrationError(f"invalid rollback manifest: {manifest_path}") from exc
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), list):
        raise MigrationError("unsupported rollback manifest")
    home = Path(manifest.get("home", "")).resolve()
    if manifest_path.parent.parent != (home / "backups" / "telegram-accounts").resolve():
        raise MigrationError("manifest is outside its Hermes backup directory")
    _restore(manifest_path, manifest, remove_backup=True)


def migrate(home: Path, *, apply: bool = False) -> dict[str, Any]:
    """Plan or apply the migration under the explicitly supplied Hermes home."""
    home = Path(home).resolve()
    changes, plan = _build_changes(home)
    if not changes:
        return plan
    if not apply:
        return plan

    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    try:
        manifest_path, manifest = _backup_and_manifest(home, changes)
        for path, content in changes.items():
            atomic_write(path, content, _mode(path, 0o600))
    except Exception:
        if manifest_path is not None and manifest is not None:
            _restore(manifest_path, manifest, remove_backup=True)
        elif manifest_path is not None:
            shutil.rmtree(manifest_path.parent, ignore_errors=True)
        raise
    return {**plan, "status": "applied", "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, help="Hermes home to migrate (required except rollback)")
    parser.add_argument("--apply", action="store_true", help="apply the plan; default is dry-run")
    parser.add_argument("--rollback", type=Path, metavar="MANIFEST", help="restore a prior migration")
    args = parser.parse_args()
    try:
        if args.rollback:
            rollback(args.rollback)
            result = {"status": "rolled_back", "manifest": str(args.rollback)}
        else:
            if args.home is None:
                parser.error("--home is required")
            result = migrate(args.home, apply=args.apply)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except MigrationError as exc:
        parser.exit(2, f"migration refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
