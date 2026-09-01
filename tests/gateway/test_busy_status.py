from gateway.busy_status import (
    is_busy_status_question,
    render_compression_timeout_reply,
)


def test_compression_timeout_reply_is_human_and_non_destructive_korean():
    text = render_compression_timeout_reply("왜 이렇게 오래 걸려?")
    assert "30초" in text
    assert "메시지는 그대로 보존" in text
    assert "/reset" not in text
    assert "/compress" not in text
    assert "auxiliary.compression" not in text


def test_status_classifier_does_not_consume_actionable_correction():
    assert is_busy_status_question("화면 폰트 크기 변화를 고쳐줘") is False
