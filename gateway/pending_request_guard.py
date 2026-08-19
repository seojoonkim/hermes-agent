"""Fail-closed guard against false completion claims for profile pending ledgers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

OPEN_STATUSES = {"pending", "in_progress"}
_CLOSURE_PATTERNS = (
    re.compile(r"남은\s*(?:작업|일|것)(?:은|이)?\s*(?:없|0)"),
    re.compile(r"(?:모두|전부)\s*(?:완료|끝)"),
    re.compile(r"(?:작업|요청)(?:은|이)?\s*(?:완전히\s*)?(?:완료|종료)"),
    re.compile(r"(?:nothing|no work|no tasks?)\s+(?:remains?|left)", re.IGNORECASE),
    re.compile(r"(?:all|everything)\s+(?:is\s+)?(?:done|complete(?:d)?)", re.IGNORECASE),
)


def _is_closure_claim(response: str) -> bool:
    return any(pattern.search(response) for pattern in _CLOSURE_PATTERNS)


def _without_closure_claims(response: str) -> str:
    """Keep useful progress text while removing contradicted closure sentences."""
    kept: list[str] = []
    for line in response.splitlines():
        parts = re.split(r"(?<=[.!?。！？])\s+", line)
        safe = [part for part in parts if part and not _is_closure_claim(part)]
        if safe:
            kept.append(" ".join(safe))
    return "\n".join(kept).strip()


def _blocked_response(response: str, reason: str) -> str:
    safe = _without_closure_claims(response)
    prefix = f"완료 판정을 보류했어. {reason}"
    return f"{safe}\n\n{prefix}" if safe else prefix


def _scope_from_source(source: Any) -> dict[str, str]:
    platform = getattr(source, "platform", "")
    platform_value = getattr(platform, "value", platform)
    account_id = getattr(source, "account_id", None) or getattr(source, "profile", None)
    return {
        "platform": str(platform_value or "").strip().casefold(),
        "chat_id": str(getattr(source, "chat_id", "") or "").strip(),
        "thread_id": str(getattr(source, "thread_id", "") or "").strip(),
        "account_id": str(account_id or "").strip(),
    }


def _scope_matches(item: dict, scope: dict[str, str]) -> bool:
    stored = item.get("scope")
    if not isinstance(stored, dict):
        return False

    for key in ("platform", "chat_id", "thread_id"):
        if str(stored.get(key, "") or "").strip().casefold() != str(scope.get(key, "") or "").strip().casefold():
            return False

    stored_account = str(stored.get("account_id", "") or "").strip()
    if stored_account:
        return stored_account.casefold() == str(scope.get("account_id", "") or "").strip().casefold()
    return True


def _ledger_path(hermes_home: str | Path, account_id: str) -> Path:
    """Resolve a profile ledger from either shared-root or profile-root home."""
    home = Path(hermes_home)
    shared_root = home.parent.parent if home.parent.name == "profiles" else home
    return shared_root / "profiles" / account_id / "state" / "pending-requests.json"


def guard_false_closure(
    response: str,
    *,
    source: Any,
    hermes_home: str | Path,
) -> tuple[str, bool]:
    """Reject only closure claims contradicted by the current channel's ledger."""
    if not response or not _is_closure_claim(response):
        return response, False

    scope = _scope_from_source(source)
    if not scope["platform"] or not scope["chat_id"] or not scope["account_id"]:
        return (
            _blocked_response(
                response,
                "현재 채널 식별자를 확인하지 못해 남은 작업을 검증할 수 없어.",
            ),
            True,
        )

    ledger = _ledger_path(hermes_home, scope["account_id"])
    if not ledger.exists():
        return (
            _blocked_response(
                response,
                "현재 채널의 작업 ledger를 찾지 못해 남은 작업을 검증할 수 없어.",
            ),
            True,
        )
    try:
        data = json.loads(ledger.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("requests"), list):
            raise ValueError("invalid pending ledger schema")
        open_items = [
            item for item in data["requests"]
            if isinstance(item, dict)
            and item.get("status") in OPEN_STATUSES
            and _scope_matches(item, scope)
        ]
    except Exception:
        return (
            _blocked_response(
                response,
                "작업 ledger를 읽지 못해 남은 작업을 검증할 수 없어.",
            ),
            True,
        )

    if not open_items:
        return response, False
    ids = ", ".join(str(item.get("id") or "unknown") for item in open_items[:8])
    suffix = "" if len(open_items) <= 8 else f" 외 {len(open_items) - 8}개"
    return (
        _blocked_response(
            response,
            f"현재 채널에 검증되지 않은 open 요청이 남아 있어: `{ids}`{suffix}",
        ),
        True,
    )
