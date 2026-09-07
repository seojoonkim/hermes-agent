"""Native event classification, without SDK or runtime dependencies."""
from types import SimpleNamespace

import pytest


def progress(event):
    from agent.stream_progress import _anthropic_event_has_meaningful_progress

    return _anthropic_event_has_meaningful_progress(event)


@pytest.mark.parametrize("kind,field", [
    ("text_delta", "text"),
    ("thinking_delta", "thinking"),
    ("input_json_delta", "partial_json"),
])
@pytest.mark.parametrize("value", ["x", " ", "\n", '{"arg":'])
def test_nonempty_delta_is_progress(kind, field, value):
    event = SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(
        type=kind, **{field: value}))
    assert progress(event) is True


@pytest.mark.parametrize("value", [None, "", 1, True, [], {}, b"text"])
def test_invalid_or_empty_delta_is_not_progress(value):
    assert progress(SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(
        type="text_delta", text=value))) is False


@pytest.mark.parametrize("kind", ["tool_use", "server_tool_use"])
@pytest.mark.parametrize("identity,expected", [
    ({"id": "tool_1"}, True), ({"name": "search"}, True),
    ({"id": "", "name": "search"}, True), ({}, False),
    ({"id": 1}, False), ({"name": True}, False),
    ({"id": "", "name": ""}, False),
])
def test_tool_start_requires_nonempty_string_identity(kind, identity, expected):
    assert progress(SimpleNamespace(type="content_block_start", content_block=
        SimpleNamespace(type=kind, input={}, **identity))) is expected


@pytest.mark.parametrize("kind", ["text", "thinking"])
@pytest.mark.parametrize("value,expected", [("", False), (" ", True),
                                          ("content", True), (None, False)])
def test_content_block_start(kind, value, expected):
    assert progress(SimpleNamespace(type="content_block_start", content_block=
        SimpleNamespace(type=kind, **{kind: value}))) is expected


@pytest.mark.parametrize("block", [None, {}, [], SimpleNamespace(type=[]),
                                    SimpleNamespace(type="unknown")])
def test_malformed_block_start(block):
    assert progress(SimpleNamespace(type="content_block_start", content_block=block)) is False


def test_message_stop_is_terminal_progress():
    assert progress(SimpleNamespace(type="message_stop")) is True


@pytest.mark.parametrize("kind", ["ping", "message_start", "message_delta",
                                  "content_block_stop", "unknown"])
def test_structural_and_usage_events_are_not_progress(kind):
    assert progress(SimpleNamespace(type=kind, usage={"output_tokens": 100},
                                    delta=SimpleNamespace(stop_reason="end_turn"))) is False


def test_signature_is_not_reasoning_progress():
    assert progress(SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(
        type="signature_delta", signature="signed", thinking="not content"))) is False


@pytest.mark.parametrize("event", [None, {}, [], 42, SimpleNamespace(),
    SimpleNamespace(type=[]), SimpleNamespace(type="content_block_delta"),
    SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type=[]))])
def test_malformed_events_fail_closed(event):
    assert progress(event) is False
