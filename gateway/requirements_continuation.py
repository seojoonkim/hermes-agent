"""Build user-visible and internal follow-ups for unfinished turn requirements."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence


CONTINUATION_MARKER = "hermes-requirements-continuation"
MAX_CONTINUATION_ATTEMPTS = 8
_MARKER_RE = re.compile(rf"<!--\s*{CONTINUATION_MARKER}:(.*?)-->", re.DOTALL)


@dataclass(frozen=True)
class RequirementsContinuation:
    notice: str
    prompt: str


def parse_continuation_payload(message: Any) -> dict[str, Any] | None:
    """Read continuation bookkeeping embedded by the trusted gateway."""
    if not isinstance(message, str):
        return None
    match = _MARKER_RE.search(message)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("requirements"), list):
        return None
    cleaned = []
    for item in payload["requirements"]:
        if not isinstance(item, dict):
            continue
        content = " ".join(str(item.get("content") or "").split())
        if content:
            cleaned.append({"content": content})
    if not cleaned:
        return None
    try:
        attempt = max(1, int(payload.get("attempt", 1)))
    except (TypeError, ValueError):
        attempt = 1
    return {"attempt": attempt, "requirements": cleaned}


def _pending_items(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw_items = result.get("pending_requirements")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return []
    items: list[tuple[str, str]] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("id") or "").strip()
        content = " ".join(str(raw.get("content") or "").split())
        if requirement_id and content:
            items.append((requirement_id, content))
    return items


def build_requirements_continuation(result: Mapping[str, Any]) -> RequirementsContinuation | None:
    """Return a bounded continuation only for a clean, blocked completion."""
    if (not result.get("completion_blocked") or result.get("failed")
            or result.get("interrupted") or result.get("guardrail")
            or result.get("requires_user_input") or result.get("approval_required")):
        return None
    items = _pending_items(result)
    if not items:
        return None
    try:
        attempt = max(0, int(result.get("requirements_continuation_attempt", 0))) + 1
    except (TypeError, ValueError):
        attempt = 1
    if attempt > MAX_CONTINUATION_ATTEMPTS:
        bullets = "\n".join(f"- {content}" for _, content in items)
        return RequirementsContinuation(
            notice=(
                f"⚠️ 자동 재개를 {MAX_CONTINUATION_ATTEMPTS}회 시도했지만 같은 필수 요구사항이 "
                f"남아 있어.\n\n{bullets}\n\n"
                "이유: 반복 실행에서 완료 상태로 바뀐 항목이 확인되지 않아 무한 반복을 "
                "막았어. 계속하려면 막힌 조건이나 필요한 입력을 확인해줘."
            ),
            prompt="",
        )

    count = len(items)
    bullets = "\n".join(f"- {content}" for _, content in items)
    notice = (
        f"⚠️ 아직 완료되지 않은 요구사항이 {count}개 있어.\n\n{bullets}\n\n"
        "이유: 위 요구사항이 완료 처리되기 전에 최종 응답이 먼저 생성돼 "
        "완료 판정이 막혔어. 사용자 확인이 필요한 안전 중단은 아니어서 "
        "남은 작업은 자동으로 이어서 처리할게."
    )
    checklist = "\n".join(f"- [{requirement_id}] {content}" for requirement_id, content in items)
    payload = json.dumps(
        {"attempt": attempt, "requirements": [{"content": content} for _, content in items]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = (
        f"<!-- {CONTINUATION_MARKER}:{payload} -->\n"
        "[내부 자동 재개 지시]\n"
        "직전 턴은 아래 필수 요구사항이 남아 완료되지 않았다. 사용자의 새 요청으로 "
        "취급하지 말고, 현재 대화와 도구 상태를 이어받아 즉시 실행하라. 각 요구사항을 "
        "실제로 수행하고 검증한 뒤 완료 상태로 갱신하라. 중간 결과를 최종 완료처럼 "
        "보고하지 마라. 안전 승인이나 사용자만 제공할 수 있는 정보가 필요하면 그 정확한 "
        "이유와 필요한 조치만 별도 메시지로 요청하라.\n\n"
        f"남은 필수 요구사항:\n{checklist}"
    )
    return RequirementsContinuation(notice=notice, prompt=prompt)
