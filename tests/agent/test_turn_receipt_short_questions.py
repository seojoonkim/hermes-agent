"""Short check-in questions get a direct answer, never a work receipt/ETA."""

import pytest

from agent.turn_receipt import is_trivial_message


@pytest.mark.parametrize("text", ["이제 들려?", "들려?", "됐어?", "이거 맞아?", "지금 돼?"])
def test_short_question_is_not_a_work_order(text):
    assert is_trivial_message(text)


@pytest.mark.parametrize(
    "text",
    [
        "다 고쳐",
        "계속해",
        "civstat 기획안이랑 개발현황 살펴보고 개선방법 리서치해볼래?",
    ],
)
def test_work_orders_still_get_receipt(text):
    assert not is_trivial_message(text)
