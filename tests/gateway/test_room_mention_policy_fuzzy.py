import pytest

from gateway.room_mention_policy import (
    CURRENT_ROOM_MENTION_OPTIONAL_PHRASES,
    CURRENT_ROOM_MENTION_REQUIRED_PHRASES,
    current_room_mention_policy,
)


@pytest.mark.parametrize(
    "text",
    [
        "너 안불러도 일할 수 있게 세팅해",
        "너 안 불러도 일할 수 있게 세팅해",
        "너 안불러도 대답하고 일하게 세팅해",
        "안불러도 일하게 해줘",
        "이 방에서는 태그 안해도 일해",
        "멘션 없이도 반응하게 설정해",
        "너를 태그 안해도 내 말 알아듣게 해줘",
        "호명 없이 답하게 세팅해",
    ],
)
def test_optional_paraphrases(text):
    assert current_room_mention_policy(text) == "optional"


@pytest.mark.parametrize(
    "text",
    [
        "이 방에선 태그할 때만 답하게 세팅해",
        "멘션했을 때만 대답해",
        "태그하기 전까지 말하지 마",
    ],
)
def test_required_paraphrases(text):
    assert current_room_mention_policy(text) == "required"


@pytest.mark.parametrize(
    "text",
    [
        "civstat 기획안 살펴보고 개선할 방법 리서치해보자",
        "시온이는 그냥 바로 세팅하는데 너는 왜 못해? 안불러도 일하게 된거 어떻게 된건지 확인해",
        "왜 안불러?",
        "일해",
        "",
    ],
)
def test_non_policy_messages_pass_through(text):
    assert current_room_mention_policy(text) is None


def test_exact_phrases_still_win():
    for p in CURRENT_ROOM_MENTION_OPTIONAL_PHRASES:
        assert current_room_mention_policy(p) == "optional"
    for p in CURRENT_ROOM_MENTION_REQUIRED_PHRASES:
        assert current_room_mention_policy(p) == "required"
