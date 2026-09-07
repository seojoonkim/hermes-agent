"""Keep quiet delegation on the streaming-capable path, including schema retry.

No provider request is made: exercise the real child runner with a recording child.
"""
import pytest

from tools.delegate_tool import _run_single_child


class Child:
    tool_progress_callback = None
    _delegate_saved_tool_names = []
    _credential_pool = None
    _subagent_id = None
    _delegate_depth = 1
    _parent_subagent_id = None
    model = "test-model"
    quiet_mode = True
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_reasoning_tokens = 0
    session_estimated_cost_usd = 0.0

    def __init__(self, retry):
        self.callbacks = []
        self._delegate_output_schema = {"type": "object"} if retry else None
        self.responses = ["invalid", "{}"] if retry else ["done"]

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        self.callbacks.append(stream_callback)
        assert callable(stream_callback)
        stream_callback("delta")
        return {"final_response": self.responses.pop(0), "completed": True,
                "api_calls": 1, "messages": []}

    def close(self):
        pass


@pytest.mark.parametrize("error", [None, "TimeoutError"])
def test_api_failure_text_is_not_success(error, monkeypatch):
    from tools import delegate_tool
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    child = Child(False)
    child.run_conversation = lambda **kw: {
        "final_response": "API call failed after 1 retries: Non-streaming API call timed out",
        "completed": False, "error": error, "api_calls": 1, "messages": [],
    }
    result = _run_single_child(0, "test", child, Parent())
    assert result["status"] == "failed"


class Parent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, description):
        pass


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("consumer", ["absent", "recording", "raising"])
def test_quiet_child_always_receives_callback(retry, consumer, monkeypatch):
    from tools import delegate_tool
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    child = Child(retry)
    events = []
    if consumer != "absent":
        def progress(event, *args, **kwargs):
            if event == "subagent.text":
                if consumer == "raising":
                    raise RuntimeError("display disconnected")
                events.append(kwargs["preview"])
        child.tool_progress_callback = progress
    result = _run_single_child(0, "test", child, Parent())
    expected = 2 if retry else 1
    assert result["status"] == "completed"
    assert len(child.callbacks) == expected
    assert all(cb is child.callbacks[0] for cb in child.callbacks)
    if retry:
        assert result["schema_valid"] is True
    if consumer == "recording":
        assert events == ["delta"] * expected
    else:
        assert events == []


def test_lifecycle_adapter_uses_shared_child_runner():
    # Exercise the lifecycle adapter with an inert worker function; full async
    # dispatch is covered separately by test_async_delegation.py.
    from tools import delegate_tool
    from unittest.mock import patch
    child = Child(False)
    parent = Parent()
    with patch.object(delegate_tool, "_run_single_child", return_value={
        "status": "completed", "summary": "done", "task_index": 0,
    }) as run:
        delegate_tool._run_child_lifecycle(0, "test", child, parent)
    run.assert_called_once_with(0, "test", child, parent)
