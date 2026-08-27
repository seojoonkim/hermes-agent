from __future__ import annotations

import json
from pathlib import Path

from gateway.config import Platform
from gateway.pending_request_guard import guard_false_closure
from gateway.session import SessionSource


def write_ledger(home: Path, account: str, requests: list[dict]) -> None:
    path = home / "profiles" / account / "state" / "pending-requests.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "requests": requests}), encoding="utf-8")


def source(*, chat_id: str = "-5100649508", chat_name: str = "New Room Name", thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_name=chat_name,
        chat_type="group",
        user_id="46291309",
        thread_id=thread_id,
        account_id="sano",
    )


def request(request_id: str, chat_id: str, *, status: str = "pending", thread_id: str = "") -> dict:
    return {
        "id": request_id,
        "request": f"work {request_id}",
        "source": "telegram:Old Room Name",
        "status": status,
        "criteria": ["verified"],
        "evidence": [],
        "scope": {
            "platform": "telegram",
            "chat_id": chat_id,
            "thread_id": thread_id,
            "account_id": "sano",
        },
    }


def test_room_rename_does_not_hide_open_request(tmp_path):
    write_ledger(tmp_path, "sano", [request("plan", "-5100649508")])

    response, blocked = guard_false_closure(
        "전부 완료했고 남은 작업은 없어.", source=source(chat_name="Renamed Again"), hermes_home=tmp_path
    )

    assert blocked is True
    assert "plan" in response
    assert "완료 판정을 보류" in response
    assert "전부 완료" not in response
    assert "남은 작업은 없어" not in response


def test_other_chat_open_request_is_not_exposed(tmp_path):
    write_ledger(tmp_path, "sano", [request("private-other-room", "-999")])
    report = (
        "완료 여부: 정상 완료\n"
        "실행 내용: 현재 채팅 범위를 점검했어.\n"
        "검증 결과: 현재 범위 open 요청 0건이야.\n"
        "남은 문제: 없어. 모두 완료했어."
    )

    response, blocked = guard_false_closure(
        report, source=source(), hermes_home=tmp_path
    )

    assert blocked is False
    assert response == report
    assert "private-other-room" not in response


def test_topics_match_exactly(tmp_path):
    write_ledger(tmp_path, "sano", [
        request("topic-7", "-5100649508", thread_id="7"),
        request("topic-70", "-5100649508", thread_id="70"),
    ])

    response, blocked = guard_false_closure(
        "모두 완료했어.", source=source(thread_id="7"), hermes_home=tmp_path
    )

    assert blocked is True
    assert "topic-7" in response
    assert "topic-70" not in response


def test_progress_response_is_not_rewritten(tmp_path):
    write_ledger(tmp_path, "sano", [request("plan", "-5100649508")])

    response, blocked = guard_false_closure(
        "지금 독립 모델 산출물을 수집 중이야.", source=source(), hermes_home=tmp_path
    )

    assert blocked is False
    assert response == "지금 독립 모델 산출물을 수집 중이야."


def test_malformed_ledger_fails_closed_without_leaking_path(tmp_path):
    path = tmp_path / "profiles" / "sano" / "state" / "pending-requests.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")

    response, blocked = guard_false_closure(
        "남은 작업은 없어.", source=source(), hermes_home=tmp_path
    )

    assert blocked is True
    assert "완료 판정을 보류" in response
    assert "작업 ledger를 읽지 못해" in response
    assert str(path) not in response


def test_missing_ledger_fails_closed(tmp_path):
    response, blocked = guard_false_closure(
        "검증 로그는 저장했어. 모두 완료했어.",
        source=source(),
        hermes_home=tmp_path,
    )

    assert blocked is True
    assert "검증 로그는 저장했어." in response
    assert "모두 완료" not in response
    assert "ledger를 찾지 못해" in response


def test_completed_current_request_requires_structured_completion_report(tmp_path):
    write_ledger(tmp_path, "sano", [request("done", "-5100649508", status="completed")])

    response, blocked = guard_false_closure(
        "모두 완료했어.", source=source(), hermes_home=tmp_path
    )

    assert blocked is True
    assert "모두 완료" not in response
    assert "완료 보고에 실행 내용, 검증 결과, 남은 문제" in response


def test_structured_completion_report_passes_when_current_scope_is_clear(tmp_path):
    write_ledger(tmp_path, "sano", [request("done", "-5100649508", status="completed")])
    report = (
        "완료 여부: 정상 완료\n"
        "실행 내용: 현재 채팅의 복구 범위를 점검했어.\n"
        "검증 결과: open 요청 0건을 확인했어.\n"
        "남은 문제: 없음. 모두 완료했어."
    )

    response, blocked = guard_false_closure(report, source=source(), hermes_home=tmp_path)

    assert blocked is False
    assert response == report


def test_english_structured_completion_report_passes(tmp_path):
    write_ledger(tmp_path, "sano", [])
    report = (
        "Completion status: completed successfully\n"
        "Actions taken: inspected the current recovery scope\n"
        "Verification: confirmed zero open requests\n"
        "Remaining issues: none; all tasks are complete"
    )

    response, blocked = guard_false_closure(report, source=source(), hermes_home=tmp_path)

    assert blocked is False
    assert response == report


def test_profile_scoped_hermes_home_resolves_shared_profile_ledger(tmp_path):
    profile_home = tmp_path / "profiles" / "sion"
    profile_home.mkdir(parents=True)
    write_ledger(tmp_path, "sano", [request("sano-current-room", "-5100649508")])

    response, blocked = guard_false_closure(
        "남은 작업은 없어.", source=source(), hermes_home=profile_home
    )

    assert blocked is True
    assert "sano-current-room" in response
    assert "남은 작업은 없어" not in response
