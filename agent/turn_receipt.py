"""Turn contract helpers: instant receipt + remaining-time heartbeat text.

Pure functions (no I/O) so the gateway can call them before any compression
or model call, and tests can pin the wording contract.

Simon's requirements (2026-10-05):
 1. Before real work, confirm the command and say how work will start.
 2. A mid-task instruction is heard first and can change direction.
 3. Remaining time is communicated continuously.

ETA inputs come from MemKraft's ``estimate_turn`` (learned p50/p80 of past
similar turns). With fewer than ``MIN_ETA_SAMPLES`` samples no number is
shown, so the receipt never invents an estimate.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional

MIN_ETA_SAMPLES = 5
_SHORT_QUESTION_MAX_LETTERS = 10
_SUMMARY_MAX = 60

_REPLY_WRAPPER = re.compile(r"^\s*\[Replying to:.*?\]\s*", re.S)
_TRIVIAL_EXACT = {
    "ㅇㅇ", "ㅇ", "엉", "응", "웅", "넹", "네", "예", "고마워", "ㄳ", "땡큐",
    "ok", "okay", "ㅇㅋ", "오케이", "좋아", "굿", "?", "??", "!",
}
_STATUS_ONLY = re.compile(
    r"^(다\s*됐어|됐어|어디까지\s*했어|뭐\s*해|진행\s*상황|상태\s*어때|끝났어|어떻게\s*돼\s*가)\??$"
)
# Greetings / small talk: answered directly, never a work receipt.
_GREETING = re.compile(
    r"^(hi|hello|hey|yo|good\s+(morning|afternoon|evening|night))(\s+there)?$"
    r"|^(안녕(하세요|히)?|ㅎㅇ|하이|반가워|좋은\s*(아침|밤))$",
    re.I,
)


def summarize_request(text: str) -> str:
    """One bounded line restating the user's request (no reply quote)."""
    body = _REPLY_WRAPPER.sub("", str(text or "")).strip()
    for line in body.splitlines():
        line = line.strip()
        if line:
            body = line
            break
    body = re.sub(r"\s+", " ", body)
    if len(body) > _SUMMARY_MAX:
        body = body[: _SUMMARY_MAX - 1].rstrip() + "…"
    return body


def is_trivial_message(text: str) -> bool:
    """True for acks, thanks, emoji, and pure status questions.

    These get a direct answer from the model; a receipt would just be noise.
    """
    body = summarize_request(text)
    if not body:
        return True
    low = body.lower().rstrip(".!~ ")
    if low in _TRIVIAL_EXACT or _STATUS_ONLY.match(low) or _GREETING.match(low):
        return True
    # Short questions ("이제 들려?", "됐어?", "이거 맞아?") are check-ins the
    # model answers in one line. A work receipt with "하던 작업 이어서 … N분"
    # is wrong there (2026-10-09 Ethval Dev: "이제 들려?" got an 11-min ETA).
    if body.rstrip().endswith("?"):
        q_letters = re.sub(r"[\W_]+", "", body, flags=re.U)
        if len(q_letters) <= _SHORT_QUESTION_MAX_LETTERS:
            return True
    # Terse imperatives ("다 고쳐", "계속해", "ㄱㄱ") are real work orders for
    # Simon, so length is NOT a triviality signal. Only messages with no
    # letters at all (emoji / punctuation) are noise.
    letters = re.sub(r"[\W_]+", "", body, flags=re.U)
    return not letters


def _minutes(ms: float) -> int:
    return max(1, round(float(ms) / 60_000))


def _grounded(eta: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(eta, Mapping):
        return False
    try:
        return int(eta.get("sample_count") or 0) >= MIN_ETA_SAMPLES and float(eta.get("p80_ms") or 0) > 0
    except (TypeError, ValueError):
        return False


def _eta_phrase(eta: Mapping[str, Any]) -> str:
    """'비슷한 요청 N건 기준 보통 X~Y분' — the number always carries its basis."""
    lo = _minutes(eta.get("p50_ms") or eta["p80_ms"])
    hi = _minutes(eta["p80_ms"])
    span = f"{hi}분 안쪽" if lo >= hi else f"{lo}~{hi}분"
    return f"비슷한 요청 {int(eta.get('sample_count') or 0)}건 기준 보통 {span}"


def build_receipt(request_text: str, *, eta: Optional[Mapping[str, Any]]) -> str:
    """Instant receipt sent the moment a work request is accepted."""
    summary = summarize_request(request_text)
    letters = re.sub(r"[\W_]+", "", summary, flags=re.U)
    if len(letters) <= 6:
        # Short go-orders ("ㄱㄱ", "다 고쳐"): quoting them back adds nothing.
        head = "받았어. 바로 진행할게"
        if _grounded(eta):
            head += f" ({_eta_phrase(eta)})"
        return head + "."
    tail = "계획 정리해서 바로 알려줄게"
    if _grounded(eta):
        tail += f" ({_eta_phrase(eta)})"
    return f"받았어 — “{summary}”. {tail}."


def build_heartbeat(*, elapsed_s: float, phase: str, eta: Optional[Mapping[str, Any]]) -> str:
    """Periodic progress line: elapsed, current phase, and honest remaining time."""
    elapsed_min = max(1, round(float(elapsed_s) / 60))
    phase = (phase or "작업").strip()
    head = f"⏳ {elapsed_min}분째 진행 중 · 지금: {phase}"
    if not _grounded(eta):
        return head + " · 끝나면 바로 알려줄게."
    remaining_ms = float(eta["p80_ms"]) - float(elapsed_s) * 1000
    if remaining_ms <= 0:
        return head + " · 예상보다 길어지고 있어, 이 단계 끝나는 대로 다시 알려줄게."
    return head + f" · 남은 예상: 약 {_minutes(remaining_ms)}분."


# Tool name -> plain Korean phrase for the heartbeat "지금:" slot. Raw tool
# names / arguments never reach the user (no paths, commands, or secrets).
_PHASE_BY_TOOL = {
    "terminal": "명령 실행",
    "execute_code": "코드 실행",
    "process": "백그라운드 작업 확인",
    "read_file": "파일 확인",
    "search_files": "파일 검색",
    "write_file": "파일 작성",
    "patch": "코드 수정",
    "web_search": "자료 검색",
    "web_extract": "자료 읽기",
    "browser_exec": "브라우저 작업",
    "delegate_task": "하위 작업 진행",
    "skill_view": "작업 가이드 확인",
    "session_search": "이전 대화 확인",
}


def describe_phase(activity: Optional[Mapping[str, Any]]) -> str:
    """Map an agent activity summary to a short, non-sensitive phase label."""
    if not isinstance(activity, Mapping):
        return "작업"
    tool = str(activity.get("current_tool") or "").strip()
    if tool:
        return _PHASE_BY_TOOL.get(tool, "작업 진행")
    desc = str(activity.get("last_activity_desc") or "").lower()
    if "api call" in desc or "waiting" in desc:
        return "다음 단계 정리"
    if "compress" in desc:
        return "대화 정리"
    return "작업"
