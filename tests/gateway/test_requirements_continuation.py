from __future__ import annotations

from gateway.requirements_continuation import (
    MAX_CONTINUATION_ATTEMPTS,
    build_requirements_continuation,
    parse_continuation_payload,
)


def _result(*, failed=False, interrupted=False, completion_blocked=True):
    return {
        "completion_blocked": completion_blocked,
        "failed": failed,
        "interrupted": interrupted,
        "pending_requirements": [
            {"id": "req:1", "content": "표지 제목을 최종 확정한다."},
            {"id": "req:2", "content": "완성 파일을 사용자에게 전달한다."},
        ],
    }


def test_pending_requirements_produce_explained_notice_and_internal_continuation():
    continuation = build_requirements_continuation(_result())

    assert continuation is not None
    assert continuation.notice.startswith("⚠️ 아직 완료되지 않은 요구사항이 2개 있어.")
    assert "표지 제목을 최종 확정한다." in continuation.notice
    assert "완성 파일을 사용자에게 전달한다." in continuation.notice
    assert "최종 응답이 먼저 생성돼 완료 판정이 막혔어" in continuation.notice
    assert "자동으로 이어서 처리할게" in continuation.notice
    assert "내부 자동 재개 지시" in continuation.prompt
    assert "중간 결과를 최종 완료처럼 보고하지 마라" in continuation.prompt
    assert "req:1" in continuation.prompt
    assert "req:2" in continuation.prompt
    payload = parse_continuation_payload(continuation.prompt)
    assert payload == {
        "attempt": 1,
        "requirements": [
            {"content": "표지 제목을 최종 확정한다."},
            {"content": "완성 파일을 사용자에게 전달한다."},
        ],
    }


def test_failed_interrupted_or_unblocked_turn_does_not_auto_continue():
    assert build_requirements_continuation(_result(failed=True)) is None
    assert build_requirements_continuation(_result(interrupted=True)) is None
    assert build_requirements_continuation(_result(completion_blocked=False)) is None
    user_blocked = _result()
    user_blocked["requires_user_input"] = True
    assert build_requirements_continuation(user_blocked) is None


def test_missing_or_malformed_pending_items_do_not_create_a_loop():
    assert build_requirements_continuation({"completion_blocked": True}) is None
    assert build_requirements_continuation({
        "completion_blocked": True,
        "pending_requirements": [{"id": "req:empty", "content": "   "}],
    }) is None


def test_continuation_attempt_limit_breaks_a_repeated_no_progress_loop():
    result = _result()
    result["requirements_continuation_attempt"] = MAX_CONTINUATION_ATTEMPTS

    continuation = build_requirements_continuation(result)

    assert continuation is not None
    assert "무한 반복을 막았어" in continuation.notice
    assert continuation.prompt == ""
