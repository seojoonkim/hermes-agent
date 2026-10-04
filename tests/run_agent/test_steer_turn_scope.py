"""Regression: a mid-turn user message (steer) must never be buried in an
older turn's tool output.

Incident 2026-10-05 (sion, Telegram DM): the user sent an extra instruction
while a text-only turn was running. The pre-API drain scanned the whole
history, appended it to a tool result from many turns earlier, and the model
never acted on it. The fix restricts the target to tool results after the
current turn's user message; otherwise the steer stays pending and is
delivered as the next user turn.
"""

from agent.conversation_loop import _pre_api_steer_target_index


def _history_then_text_only_turn():
    return [
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": "old tool output"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},  # index 4
    ]


def test_steer_not_injected_into_previous_turn_tool_result():
    messages = _history_then_text_only_turn()
    assert _pre_api_steer_target_index(messages, 4) is None


def test_steer_uses_current_turn_tool_result():
    messages = _history_then_text_only_turn() + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
        {"role": "tool", "tool_call_id": "t2", "content": "new tool output"},
    ]
    assert _pre_api_steer_target_index(messages, 4) == 6


def test_unknown_turn_index_keeps_legacy_scan():
    messages = _history_then_text_only_turn()
    assert _pre_api_steer_target_index(messages, None) == 2
