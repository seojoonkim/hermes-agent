"""Regression: a window with many tool calls must not exceed the scorer request budget.

Sano 2026-10-06: 1,977-row middle, Jev raised "request exceeds 30000-token input
budget" and the whole relevance stage fell through to a 260 s summary.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agent.model_metadata import estimate_tokens_rough
from agent.relevance_scorer import (
    ScoreResult, ScorerUnavailable, batch_items_by_budget, run_relevance_stage,
    score_item_request_tokens,
)


def _jev_like_request_tokens(state, items):
    # Mirror of the Jev plugin body construction (plugins/jev-compaction).
    questions = {}
    for i, item in enumerate(items):
        d = (f"call_id={item.call_id}; tool={item.tool_name}; "
             f"input={item.args_preview}; result={item.result_note}")
        questions[f"call_t{i}"] = {"type": "noul", "instructions": f"Tool call {d} should stay in history: knowing the call and input still matters for what the assistant does next."}
        questions[f"result_t{i}"] = {"type": "noul", "instructions": f"The full output of {d} should stay verbatim: its contents are still needed for the current unfinished work, and re-running the tool would not do."}
    body = {"model": "jev-latest", "state": state, "questions": questions}
    return estimate_tokens_rough(json.dumps(body, ensure_ascii=False))


class BudgetedScorer:
    name = "budgeted"

    def __init__(self, limit=30000):
        self.limit = limit
        self.calls = []

    def score(self, state, items):
        size = _jev_like_request_tokens(state, items)
        self.calls.append(size)
        if size > self.limit:
            raise ScorerUnavailable(f"request {size} exceeds {self.limit}")
        return [ScoreResult(i.call_id, 0.1, 0.1) for i in items]


def _transcript(n_calls, arg_chars=600, out_chars=3000):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
    for k in range(n_calls):
        cid = f"c{k}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": "terminal", "arguments": json.dumps({"command": "x" * arg_chars})}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": f"out{k} " + "y" * out_chars})
    msgs.append({"role": "user", "content": "now"})
    return msgs


def test_many_tool_calls_stay_under_request_budget():
    msgs = _transcript(400)
    scorer = BudgetedScorer()
    out, report, tel = run_relevance_stage(
        msgs, head_end=2, tail_start=len(msgs) - 1, scorer=scorer,
        config={"max_state_tokens": 25000, "max_request_tokens": 28000}, redact=lambda s: s,
    )
    assert scorer.calls, "scorer never called"
    assert max(scorer.calls) <= 30000
    assert tel["requests"] >= 2
    assert report.scored == 400


def test_batching_preserves_order_and_items():
    msgs = _transcript(50)
    from agent.relevance_scorer import collect_score_items
    items = collect_score_items(msgs, 2, len(msgs) - 1)
    batches = batch_items_by_budget(items, 3 * score_item_request_tokens(items[0]))
    assert [i.call_id for b in batches for i in b] == [i.call_id for i in items]
    assert all(len(b) <= 3 for b in batches)


def test_real_jev_plugin_body_under_budget(monkeypatch):
    """Drive the real Jev plugin up to its own budget check (no network)."""
    path = Path.home() / ".hermes/plugins/jev-compaction/__init__.py"
    if not path.exists():
        pytest.skip("jev plugin not installed")
    spec = importlib.util.spec_from_file_location("jev_plugin_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setattr("agent.relevance_scorer.register_relevance_scorer", lambda *a, **k: None, raising=False)
    spec.loader.exec_module(mod)

    class Sentinel(Exception):
        pass

    scorer = mod.JevScorer({"timeout_seconds": 20})
    monkeypatch.setattr(scorer, "_key", lambda: (_ for _ in ()).throw(Sentinel()))
    msgs = _transcript(400)

    seen = []

    class Wrap:
        name = "jev"

        def score(self, state, items):
            try:
                scorer.score(state, items)
            except Sentinel:  # passed the 30k budget check, stopped before network
                seen.append(len(items))
                return [ScoreResult(i.call_id, 0.9, 0.9) for i in items]

    run_relevance_stage(msgs, head_end=2, tail_start=len(msgs) - 1, scorer=Wrap(),
                        config={"max_state_tokens": 25000}, redact=lambda s: s)
    assert sum(seen) == 400
