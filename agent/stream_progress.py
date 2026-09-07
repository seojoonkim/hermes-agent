"""Pure provider event predicates; classification does not filter events."""
from typing import Any


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _codex_event_has_meaningful_progress(event: Any) -> bool:
    kind = _field(event, "type")
    if kind in ("response.completed", "response.failed", "response.incomplete"):
        return True
    if kind in ("response.output_text.delta", "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta", "response.function_call_arguments.delta"):
        value = _field(event, "delta")
        return isinstance(value, str) and bool(value)
    if kind == "response.output_item.added":
        item = _field(event, "item")
        return _field(item, "type") == "function_call" and bool(_field(item, "name"))
    return False


def _bedrock_event_has_meaningful_progress(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    if "messageStop" in event:
        return True
    delta = event.get("contentBlockDelta", {}).get("delta", {})
    start = event.get("contentBlockStart", {}).get("start", {})
    return any(isinstance(value, str) and bool(value) for value in (
        delta.get("text"), delta.get("reasoningContent", {}).get("text"),
        delta.get("toolUse", {}).get("input"), start.get("toolUse", {}).get("name"),
    ))


class _CodexProgressClient:
    """Request-local proxy: fence native ingress before Relay can claim a writer.

    The original client remains the holder's transport owner for abort/close.
    No shared SDK methods or agent callbacks are mutated.
    """
    def __init__(self, client, cancelled, progress, clock):
        self._client = client
        self._cancelled = cancelled
        self._progress = progress
        self._clock = clock
        self.responses = self

    def __getattr__(self, name):
        return getattr(self._client, name)

    def create(self, **kwargs):
        self._check()
        self._progress["t"] = self._clock()  # fresh physical retry budget
        stream = self._client.responses.create(**kwargs)
        if self._cancelled["value"]:
            stream.close()  # late creation: worker owns this FD
            self._check()
        return _CodexProgressStream(stream, self)

    def _check(self):
        if self._cancelled["value"]:
            raise InterruptedError("Codex request cancelled")


class _CodexProgressStream:
    def __init__(self, stream, owner):
        self._stream, self._owner = stream, owner

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __iter__(self):
        for event in self._stream:
            self._owner._check()
            if _codex_event_has_meaningful_progress(event):
                self._owner._progress["t"] = self._owner._clock()
            yield event

    def close(self):
        self._stream.close()


def _anthropic_event_has_meaningful_progress(event: Any) -> bool:
    """Recognize content or completion, not transport/usage/signature metadata.

    Whitespace and incomplete JSON fragments are productive. A message stop
    grants a terminal progress boundary, not evidence of visible output.
    """
    event_type = getattr(event, "type", None)
    if not isinstance(event_type, str):
        return False
    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        kind = getattr(delta, "type", None)
        if not isinstance(kind, str):
            return False
        field = {"text_delta": "text", "thinking_delta": "thinking",
                 "input_json_delta": "partial_json"}.get(kind)
        value = getattr(delta, field, None) if field else None
        return isinstance(value, str) and bool(value)
    if event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        kind = getattr(block, "type", None)
        if not isinstance(kind, str):
            return False
        if kind in ("tool_use", "server_tool_use"):
            return any(isinstance(value, str) and bool(value) for value in (
                getattr(block, "id", None), getattr(block, "name", None)))
        field = {"text": "text", "thinking": "thinking"}.get(kind)
        value = getattr(block, field, None) if field else None
        return isinstance(value, str) and bool(value)
    return event_type == "message_stop"
