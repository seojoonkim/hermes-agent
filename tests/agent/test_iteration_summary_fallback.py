"""Focused tests for bounded iteration-summary recovery."""

from agent.chat_completion_helpers import (
    _bounded_iteration_summary_messages,
    _local_iteration_summary,
)


def test_bounded_summary_keeps_latest_real_user_and_limits_tail():
    request = "summary-now"
    messages = [{"role": "user", "content": "original request"}]
    for i in range(40):
        messages.append({"role": "assistant", "content": f"step {i}"})
    messages.append({"role": "user", "content": request})
    bounded = _bounded_iteration_summary_messages(
        messages, request, max_messages=8, max_content_chars=100
    )
    assert len(bounded) == 9
    assert bounded[0]["content"] == "original request"
    assert bounded[-1]["content"] == request


def test_bounded_summary_truncates_large_content_without_mutating_source():
    long_text = "a" * 1000
    messages = [{"role": "tool", "content": long_text}]
    bounded = _bounded_iteration_summary_messages(
        messages, "summary", max_content_chars=100
    )
    assert "중간 내용 생략" in bounded[0]["content"]
    assert len(bounded[0]["content"]) < len(long_text)
    assert messages[0]["content"] == long_text


def test_local_summary_uses_recent_assistant_progress_and_hides_error():
    messages = [
        {"role": "assistant", "content": "배포를 완료했어."},
        {"role": "assistant", "content": "라이브 검증도 통과했어."},
    ]
    result = _local_iteration_summary(messages, 90)
    assert "배포를 완료했어" in result
    assert "라이브 검증도 통과했어" in result
    assert "Error:" not in result
    assert "보존" in result
