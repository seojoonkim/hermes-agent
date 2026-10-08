"""Short imperative commands are real work requests (2026-10-05 regression).

Simon writes terse orders ("다 고쳐", "계속해", "ㄱㄱ", "다 해"). The first receipt
gate treated anything with <=3 letters as trivial, so "다 고쳐" got no receipt.
Only thanks / greetings / pure status questions are trivial.
"""
import pytest

from agent.turn_receipt import build_receipt, is_trivial_message


@pytest.mark.parametrize("text", ["다 고쳐", "계속해", "ㄱㄱ", "다 해", "고쳐", "엉 다 해", "남은거 다 해"])
def test_short_commands_get_a_receipt(text):
    assert is_trivial_message(text) is False


@pytest.mark.parametrize("text", ["고마워", "ㄳ", "땡큐", "안녕", "hello", "다 됐어?", "어디까지 했어", "?", "👍"])
def test_thanks_greetings_status_stay_trivial(text):
    assert is_trivial_message(text) is True


def test_short_command_receipt_does_not_parrot_the_command():
    text = build_receipt("ㄱㄱ", eta=None)
    assert text.startswith("받았어")
    assert "“ㄱㄱ”" not in text
