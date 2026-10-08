"""Turn contract: instant receipt, remaining-time heartbeat, steer preempt bound.

Simon's requirements (2026-10-05):
 1. Before real work, the agent confirms the command and says how it will start.
 2. A mid-task instruction is heard first and can change direction.
 3. Remaining time is communicated continuously.
"""
from __future__ import annotations

import pytest

from agent.turn_receipt import (
    build_heartbeat,
    build_receipt,
    is_trivial_message,
    summarize_request,
)


class TestTrivialDetection:
    @pytest.mark.parametrize("text", ["ㅇㅇ", "엉", "고마워", "ok", "?", "👍", "다 됐어?"])
    def test_short_or_status_messages_are_trivial(self, text):
        assert is_trivial_message(text) is True

    @pytest.mark.parametrize("text", [
        "사노 덱 레이아웃 충돌 다 고쳐줘",
        "에이전트 속도 병목 찾아서 개선해",
        "Fix the failing CI on the memkraft PR and merge it",
    ])
    def test_work_requests_are_not_trivial(self, text):
        assert is_trivial_message(text) is False


class TestSummarize:
    def test_summary_is_single_line_and_bounded(self):
        s = summarize_request("첫 줄 요청이야\n두번째 줄 상세 설명 " + "가" * 300)
        assert "\n" not in s
        assert len(s) <= 60

    def test_reply_wrapper_is_stripped(self):
        s = summarize_request('[Replying to: "옛날 메시지"]\n\n이거 고쳐줘')
        assert "Replying to" not in s
        assert "이거 고쳐줘" in s


class TestReceipt:
    def test_receipt_confirms_request_and_promises_plan(self):
        r = build_receipt("에이전트 속도 병목 찾아서 개선해", eta=None)
        assert "받았어" in r
        assert "에이전트 속도 병목" in r
        assert "계획" in r

    def test_receipt_includes_grounded_eta_when_available(self):
        r = build_receipt("덱 다시 빌드해줘", eta={"p50_ms": 8 * 60_000, "p80_ms": 14 * 60_000, "sample_count": 12})
        assert "8~14분" in r

    def test_receipt_omits_eta_with_too_few_samples(self):
        r = build_receipt("덱 다시 빌드해줘", eta={"p50_ms": 8 * 60_000, "p80_ms": 14 * 60_000, "sample_count": 2})
        assert "분" not in r.split("—")[-1] or "예상" not in r


class TestHeartbeat:
    def test_heartbeat_reports_elapsed_phase_and_remaining(self):
        hb = build_heartbeat(elapsed_s=6 * 60, phase="터미널 명령 실행 중", eta={"p80_ms": 14 * 60_000, "sample_count": 12})
        assert "6분" in hb
        assert "터미널 명령 실행 중" in hb
        assert "8분" in hb  # 14 - 6 remaining

    def test_heartbeat_admits_overrun_instead_of_fake_number(self):
        hb = build_heartbeat(elapsed_s=20 * 60, phase="테스트 실행 중", eta={"p80_ms": 14 * 60_000, "sample_count": 12})
        assert "예상보다 길어지고 있어" in hb
        assert "남은 예상" not in hb

    def test_heartbeat_without_eta_still_reports_progress(self):
        hb = build_heartbeat(elapsed_s=5 * 60, phase="파일 수정 중", eta=None)
        assert "5분" in hb and "파일 수정 중" in hb


def test_describe_phase_is_plain_and_never_leaks_args():
    from agent.turn_receipt import describe_phase
    assert describe_phase({"current_tool": "terminal"}) == "명령 실행"
    assert describe_phase({"current_tool": "some_new_tool"}) == "작업 진행"
    assert describe_phase({"last_activity_desc": "starting API call #4"}) == "다음 단계 정리"
    assert describe_phase(None) == "작업"
    assert "/" not in describe_phase({"current_tool": "terminal", "last_activity_desc": "rm -rf /tmp/x"})

