"""Relevance-scored tool-result pruning for context compaction.

Position-based pruning (Phase 1 of ``ContextCompressor.compress``) demotes
every old tool result regardless of whether it still matters.  This module
adds a *relevance* stage: each tool call inside the compressible region is
scored with two yes/no probabilities —

* ``keep_call``   – is it still useful to know this call happened?
* ``keep_result`` – is the full result body still needed?

— by a pluggable :class:`RelevanceScorer`.  Decisions are applied by the pure
function :func:`apply_relevance_decisions`, which never touches the protected
head or tail, never breaks ``tool_call`` ↔ ``tool`` pairing, and never leaves
two same-role rows adjacent.

Backends shipped in-tree:

* ``NullScorer`` – keeps everything (stage becomes a no-op).
* ``AuxLLMScorer`` – asks the configured auxiliary ``compression`` model for
  JSON probabilities (provider-agnostic default).

Third-party probability models (e.g. TypeSafe Jev) plug in through
``register_relevance_scorer`` and live outside the core tree.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough

logger = logging.getLogger(__name__)

# Keep hung plugin/SDK calls bounded process-wide. A timed-out owner does not
# release the permit: only the actual worker exiting can do that.
_STAGE_WORKERS = threading.BoundedSemaphore(4)
_STAGE_BUDGET: contextvars.ContextVar[Any] = contextvars.ContextVar("relevance_stage_budget", default=None)


def _check_stage_budget() -> Optional[float]:
    budget = _STAGE_BUDGET.get()
    if budget is None:
        return None
    deadline, cancel_check = budget
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    if cancel_check():
        raise AuxiliaryExplicitCancellation()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ScorerUnavailable("relevance stage deadline exceeded")
    return remaining


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ScoreItem:
    """One tool call to be scored (call + paired result)."""

    call_id: str
    tool_name: str
    args_preview: str
    result_note: str          # "ok, 4213 chars (omitted)" — never the body
    assistant_idx: int
    tool_idx: int
    result_chars: int


@dataclass(frozen=True)
class ScoreResult:
    call_id: str
    keep_call: float
    keep_result: float

    def __post_init__(self) -> None:
        for name in ("keep_call", "keep_result"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v != v or not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be a probability in [0, 1], got {v!r}")


@dataclass
class ApplyReport:
    kept: int = 0
    truncated: int = 0
    dropped: int = 0
    removed_assistant_rows: int = 0
    scored: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    decisions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def reduction_ratio(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return max(0.0, (self.tokens_before - self.tokens_after) / self.tokens_before)


class ScorerUnavailable(RuntimeError):
    """The scorer could not produce decisions; caller must fall through."""


class RelevanceScorer(Protocol):
    name: str

    def score(self, state: str, items: Sequence[ScoreItem]) -> List[ScoreResult]:
        ...


# --------------------------------------------------------------------------- #
# Registry (plugins register third-party backends here)
# --------------------------------------------------------------------------- #

_SCORER_FACTORIES: Dict[str, Callable[[Dict[str, Any]], RelevanceScorer]] = {}


def register_relevance_scorer(name: str, factory: Callable[[Dict[str, Any]], RelevanceScorer]) -> None:
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("scorer name required")
    _SCORER_FACTORIES[key] = factory


def get_relevance_scorer(name: str, config: Dict[str, Any]) -> Optional[RelevanceScorer]:
    key = (name or "off").strip().lower()
    if key in ("off", "none", ""):
        return None
    if key == "null":
        return NullScorer()
    if key == "aux":
        return AuxLLMScorer(config)
    factory = _SCORER_FACTORIES.get(key)
    if factory is None:
        logger.warning("relevance_prune: unknown backend %r — stage disabled", name)
        return None
    return factory(config)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class NullScorer:
    name = "null"

    def score(self, state: str, items: Sequence[ScoreItem]) -> List[ScoreResult]:
        return [ScoreResult(i.call_id, 1.0, 1.0) for i in items]


_AUX_PROMPT = """You are pruning an AI agent's conversation history before it is compacted.
Below is the conversation state (tool result bodies are replaced by short notes) followed by a list of tool calls.

For EACH tool call answer two probabilities in [0,1]:
- keep_call: probability that it is still useful for the agent to know this call happened (its arguments / that it ran).
- keep_result: probability that the FULL result body is still needed to continue the current work.

Guidance: results of files that were later rewritten, searches that were superseded, successful builds/tests already acted upon, and exploratory reads are usually NOT needed. Error messages the user is still debugging, file paths/config values referenced later, and the most recent evidence for pending work ARE needed.

Return ONLY a JSON object: {{"<call_id>": {{"keep_call": p, "keep_result": p}}, ...}} covering every call_id.

=== CONVERSATION STATE ===
{state}

=== TOOL CALLS TO SCORE ===
{items}
"""


class AuxLLMScorer:
    """Provider-agnostic backend: the auxiliary ``compression`` model returns JSON probabilities."""

    name = "aux"

    def __init__(self, config: Dict[str, Any]):
        self.timeout = float(config.get("timeout_seconds", 20) or 20)
        self.model_override = str(config.get("model") or "")
        self._call_llm = config.get("_call_llm")  # test seam

    def score(self, state: str, items: Sequence[ScoreItem]) -> List[ScoreResult]:
        if not items:
            return []
        listing = "\n".join(
            f"- call_id={i.call_id} tool={i.tool_name} args={i.args_preview} result={i.result_note}"
            for i in items
        )
        prompt = _AUX_PROMPT.format(state=state, items=listing)
        call_llm = self._call_llm
        if call_llm is None:
            from agent.auxiliary_client import call_llm as _real_call_llm
            call_llm = _real_call_llm
        kwargs: Dict[str, Any] = {
            "task": "compression",
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        remaining = _check_stage_budget()
        if remaining is not None:
            kwargs["timeout"] = min(self.timeout, remaining)
            kwargs["deadline"] = _STAGE_BUDGET.get()[0]
        if self.model_override:
            kwargs["model"] = self.model_override
        try:
            response = call_llm(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ScorerUnavailable(f"aux scorer call failed: {exc}") from exc
        text = _response_text(response)
        parsed = _extract_json_object(text)
        if not isinstance(parsed, dict):
            raise ScorerUnavailable("aux scorer returned non-JSON output")
        results: List[ScoreResult] = []
        for item in items:
            entry = parsed.get(item.call_id)
            if not isinstance(entry, dict):
                raise ScorerUnavailable(f"aux scorer omitted call_id {item.call_id}")
            try:
                results.append(ScoreResult(
                    item.call_id,
                    float(entry.get("keep_call", 1.0)),
                    float(entry.get("keep_result", 1.0)),
                ))
            except (TypeError, ValueError) as exc:
                raise ScorerUnavailable(f"aux scorer bad probability for {item.call_id}: {exc}") from exc
        return results


def _response_text(response: Any) -> str:
    try:
        if isinstance(response, dict):
            choices = response.get("choices") or [{}]
            msg = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
        else:
            msg = response.choices[0].message
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", msg)
        return content if isinstance(content, str) else (str(content) if content else "")
    except Exception:  # noqa: BLE001
        return ""


def _extract_json_object(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Item collection + state rendering
# --------------------------------------------------------------------------- #

_ALREADY_PRUNED_MARKERS = (
    "[Old tool output cleared to save context space]",
    "chars omitted",
    "[digest unavailable",
    "— recover via session_search",
    "lines output",            # Phase 1 one-line demotion: "[tool] ran `…` -> exit 0, N lines output"
    "[Duplicate tool output",
    "[screenshot removed",
)


def _is_already_pruned(content: Any) -> bool:
    """True when a tool row is a Phase 1 stub, a dedupe back-reference, or a
    relevance truncation (head + pointer). Checked on the edges only so a
    genuine long body that merely *mentions* a marker is not misclassified."""
    if not isinstance(content, str):
        return False
    edges = content[:200] + "\n" + content[-260:]
    if len(content) <= 400 or content.startswith("[") or content.rstrip().endswith("]"):
        return any(m in edges for m in _ALREADY_PRUNED_MARKERS)
    return False


def _content_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, ensure_ascii=False))
    except Exception:  # noqa: BLE001
        return 0


def collect_score_items(
    messages: List[Dict[str, Any]],
    start: int,
    end: int,
    *,
    min_result_chars: int = 200,
    pristine: Optional[Dict[str, str]] = None,
    redact: Callable[[str], str] | None = None,
) -> List[ScoreItem]:
    """Return scoreable (assistant tool_call, tool result) pairs within ``[start, end)``.

    ``pristine`` maps ``tool_call_id`` → verbatim result captured *before*
    position-based demotion. When a row was already demoted but a pristine
    copy exists, the pristine body is scored (and can be restored on keep).
    """
    pristine = pristine or {}
    tool_rows: Dict[str, int] = {}
    for idx in range(start, end):
        m = messages[idx]
        if m.get("role") == "tool" and m.get("tool_call_id"):
            tool_rows[str(m["tool_call_id"])] = idx
    items: List[ScoreItem] = []
    for idx in range(start, end):
        m = messages[idx]
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            cid = str(tc.get("id") or "")
            tidx = tool_rows.get(cid)
            if not cid or tidx is None:
                continue
            content = messages[tidx].get("content")
            orig = pristine.get(cid)
            if orig and not _is_already_pruned(orig) and (
                _is_already_pruned(content) or _content_len(content) < len(orig)
            ):
                # Phase 1 (or a prior pass) already shrank this row; score the
                # verbatim body so relevance can restore or re-truncate it.
                content = orig
            elif _is_already_pruned(content):
                continue
            n = _content_len(content)
            if n < min_result_chars:
                continue
            fn = tc.get("function") or {}
            args = fn.get("arguments") or ""
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            # A truncated credential may no longer match the redactor (PEM,
            # quoted secrets, etc.). Redact the full value before taking a preview.
            if redact:
                args = redact(args)
            status = "error" if isinstance(content, str) and re.search(r'"exit_code"\s*:\s*[1-9]|Error|Traceback', content[:400]) else "ok"
            items.append(ScoreItem(
                call_id=cid,
                tool_name=str(fn.get("name") or "tool"),
                args_preview=args[:200].replace("\n", " "),
                result_note=f"{status}, {n:,} chars (omitted)",
                assistant_idx=idx,
                tool_idx=tidx,
                result_chars=n,
            ))
    return items


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(parts)
    return str(content) if content else ""


def render_state(
    messages: List[Dict[str, Any]],
    head_end: int,
    window: range,
    *,
    max_tokens: int,
    redact: Callable[[str], str] | None = None,
) -> str:
    """Render pinned head + window as text; tool bodies replaced by notes; abridged to ``max_tokens``."""
    redact = redact or (lambda s: s)

    def row(idx: int, cap: int) -> str:
        m = messages[idx]
        role = m.get("role")
        if role == "tool":
            n = _content_len(m.get("content"))
            return redact(f"[{idx}] tool({m.get('tool_call_id', '?')}): {'ok' if n else 'empty'}, {n:,} chars (omitted)")
        text = redact(_text_of(m.get("content")))
        if role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(
                f"{redact(str((tc.get('function') or {}).get('name', 'tool')))}({redact(str((tc.get('function') or {}).get('arguments') or ''))[:120]}) id={redact(str(tc.get('id')))}"
                for tc in m["tool_calls"]
            )
            text = (text + "\n" if text else "") + f"→ calls: {calls}"
        if len(text) > cap:
            text = text[: cap // 2] + f" …[{len(text) - cap:,} chars abridged]… " + text[-cap // 2:]
        return f"[{idx}] {role}: {redact(text)}"

    head_idx = [i for i in range(0, head_end) if messages[i].get("role") != "system"]
    body_idx = list(window)
    for cap in (4000, 1500, 600, 250, 100):
        lines = ["=== PINNED HEAD ==="] + [row(i, cap) for i in head_idx]
        lines += ["=== WINDOW ==="] + [row(i, cap) for i in body_idx]
        state = "\n".join(lines)
        if estimate_tokens_rough(state) <= max_tokens:
            return state
    # Last resort: drop oldest window rows until it fits.
    while len(body_idx) > 1:
        body_idx = body_idx[len(body_idx) // 4 or 1:]
        lines = ["=== PINNED HEAD ==="] + [row(i, 100) for i in head_idx]
        lines += [f"=== WINDOW (oldest {len(window) - len(body_idx)} rows collapsed) ==="] + [row(i, 100) for i in body_idx]
        state = "\n".join(lines)
        if estimate_tokens_rough(state) <= max_tokens:
            return state
    return state


# Per-question boilerplate (two instruction sentences + JSON keys) in tokens.
_QUESTION_OVERHEAD_TOKENS = 90


def _json_tokens(text: str) -> int:
    """Token estimate for ``text`` as it will appear JSON-encoded in a request."""
    return estimate_tokens_rough(json.dumps(text, ensure_ascii=False))


def score_item_request_tokens(item: "ScoreItem") -> int:
    """Request tokens one item costs: its description appears in two questions."""
    desc = f"{item.call_id} {item.tool_name} {item.args_preview} {item.result_note}"
    return 2 * _json_tokens(desc) + _QUESTION_OVERHEAD_TOKENS


def batch_items_by_budget(items: List["ScoreItem"], budget_tokens: int) -> List[List["ScoreItem"]]:
    """Greedy, order-preserving split so each batch's question tokens fit ``budget_tokens``.

    A single oversized item still gets its own batch (the backend decides).
    """
    batches: List[List[ScoreItem]] = []
    cur: List[ScoreItem] = []
    cur_tokens = 0
    for item in items:
        cost = score_item_request_tokens(item)
        if cur and cur_tokens + cost > budget_tokens:
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(item)
        cur_tokens += cost
    if cur:
        batches.append(cur)
    return batches


def split_windows(
    messages: List[Dict[str, Any]],
    start: int,
    end: int,
    *,
    max_window_tokens: int,
) -> List[range]:
    """Split ``[start, end)`` into windows whose rendered size stays under budget, never mid tool-group."""
    windows: List[range] = []
    cur_start = start
    cur_tokens = 0
    idx = start
    while idx < end:
        m = messages[idx]
        # A tool row belongs to the preceding assistant; never open a window on it.
        est = 40 if m.get("role") == "tool" else min(1200, estimate_tokens_rough(_text_of(m.get("content"))) + 40)
        if cur_tokens + est > max_window_tokens and idx > cur_start and m.get("role") != "tool":
            windows.append(range(cur_start, idx))
            cur_start, cur_tokens = idx, 0
        cur_tokens += est
        idx += 1
    if cur_start < end:
        windows.append(range(cur_start, end))
    return windows


# --------------------------------------------------------------------------- #
# Apply decisions (pure)
# --------------------------------------------------------------------------- #

_SIGNAL_LINE_RE = re.compile(
    r"(?i)(\berror\b|\bexception\b|traceback|\bfailed\b|\bfail:|\bfatal\b|\bpanic\b|"
    r"exit[ _]?code[:= ]+\s*[1-9]|assert(ion)?error|^E\s{2,})"
)
_MAX_SIGNAL_LINES = 8
_SIGNAL_LINE_MAX_CHARS = 160


def _truncate_result(content: Any, head_chars: int, pointer: str) -> str:
    """Head + failure-signal lines + tail digest.

    Tool output (tests, builds, commands) usually puts the verdict at the END
    and the cause in ERROR/Traceback lines in the middle, so a head-only cut
    throws away the most decision-relevant evidence. Budget: head_chars for the
    head, head_chars for the tail, and at most ``_MAX_SIGNAL_LINES`` short
    signal lines from the omitted middle. The full original stays recoverable
    via ``pointer``.
    """
    text = _text_of(content)
    if len(text) <= head_chars:
        return text
    tail_chars = head_chars
    head = text[:head_chars]
    tail_start = max(head_chars, len(text) - tail_chars)
    # Snap the tail to a line boundary so it starts on a whole line.
    nl = text.find("\n", tail_start)
    if nl != -1 and nl < len(text) - 1:
        tail_start = nl + 1
    middle = text[head_chars:tail_start]
    tail = text[tail_start:]
    signals: List[str] = []
    signal_budget = head_chars  # total chars for signal lines == one head
    for line in middle.splitlines():
        if _SIGNAL_LINE_RE.search(line):
            piece = line.strip()[:_SIGNAL_LINE_MAX_CHARS]
            if signal_budget - len(piece) < 0:
                break
            signals.append(piece)
            signal_budget -= len(piece) + 1
            if len(signals) >= _MAX_SIGNAL_LINES:
                break
    omitted = len(middle)
    parts = [head, f"\n[... {omitted:,} chars omitted — {pointer}]"]
    if signals:
        parts.append("\n[signal lines from omitted part]\n" + "\n".join(signals))
    if tail:
        parts.append("\n[tail]\n" + tail)
        # Close with the pointer marker so edge-based _is_already_pruned()
        # still recognizes this row on a second pass (idempotency).
        parts.append(f"\n[... end of digest, {omitted:,} chars omitted — {pointer}]")
    return "".join(parts)


def apply_relevance_decisions(
    messages: List[Dict[str, Any]],
    items: Sequence[ScoreItem],
    results: Sequence[ScoreResult],
    *,
    head_end: int,
    tail_start: int,
    keep_threshold: float = 0.5,
    truncate_head_chars: int = 300,
    session_id: str = "",
    pristine: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict[str, Any]], ApplyReport]:
    """Apply scores to a copy of ``messages``.

    Invariants (tested):
    * rows ``< head_end`` and ``>= tail_start`` are byte-identical;
    * every remaining ``tool`` row has a live ``tool_call`` owner;
    * no two adjacent rows share a role;
    * idempotent — a second pass with the same decisions changes nothing.
    """
    pristine = pristine or {}
    report = ApplyReport(scored=len(items), tokens_before=estimate_messages_tokens_rough(messages))
    by_id = {r.call_id: r for r in results}
    out = [dict(m) for m in messages]
    drop_tool_idx: set[int] = set()
    drop_calls: Dict[int, set[str]] = {}

    for item in items:
        if not (head_end <= item.assistant_idx < tail_start and head_end <= item.tool_idx < tail_start):
            continue
        res = by_id.get(item.call_id)
        if res is None:
            report.kept += 1
            continue
        action = "keep"
        # Source body: the verbatim copy captured before position-based
        # demotion when available, else the current row content.
        source = pristine.get(item.call_id)
        if not source or _is_already_pruned(source):
            source = out[item.tool_idx].get("content")
        if res.keep_call < keep_threshold:
            action = "drop"
            drop_tool_idx.add(item.tool_idx)
            drop_calls.setdefault(item.assistant_idx, set()).add(item.call_id)
            report.dropped += 1
        elif res.keep_result < keep_threshold:
            action = "truncate"
            pointer = f"session_search(query='{item.tool_name} {item.args_preview[:60]}'"
            pointer += f", session_id='{session_id}')" if session_id else ")"
            truncated = _truncate_result(source, truncate_head_chars, pointer)
            current = out[item.tool_idx].get("content")
            # Never inflate: if Phase 1 already left a smaller stub, keep it.
            if not (isinstance(current, str) and len(current) < len(truncated)):
                out[item.tool_idx]["content"] = truncated
            report.truncated += 1
        else:
            # Relevance says keep → restore verbatim body even if Phase 1
            # had already demoted it by position.
            if isinstance(source, str) and source != out[item.tool_idx].get("content"):
                out[item.tool_idx]["content"] = source
            report.kept += 1
        report.decisions.append({
            "call_id": item.call_id, "tool": item.tool_name,
            "keep_call": round(res.keep_call, 3), "keep_result": round(res.keep_result, 3),
            "action": action, "result_chars": item.result_chars,
        })

    # Remove dropped tool_calls from their assistant rows.
    remove_rows: set[int] = set(drop_tool_idx)
    for aidx, ids in drop_calls.items():
        row = out[aidx]
        remaining = [tc for tc in (row.get("tool_calls") or []) if str(tc.get("id")) not in ids]
        if remaining:
            row["tool_calls"] = remaining
        else:
            row.pop("tool_calls", None)
            if not _text_of(row.get("content")).strip():
                remove_rows.add(aidx)
                report.removed_assistant_rows += 1

    if remove_rows:
        out = [m for i, m in enumerate(out) if i not in remove_rows]
        # Fix same-role adjacency created inside the mutable region only.
        out = _repair_alternation(out, head_end, tail_start - len(remove_rows))

    report.tokens_after = estimate_messages_tokens_rough(out)
    return out, report


def _repair_alternation(messages: List[Dict[str, Any]], lo: int, hi: int) -> List[Dict[str, Any]]:
    """Merge adjacent same-role assistant rows within ``[lo, hi)`` (tool rows are exempt)."""
    out: List[Dict[str, Any]] = []
    for idx, m in enumerate(messages):
        if out and lo < idx < hi and m.get("role") == "assistant" and out[-1].get("role") == "assistant" \
                and not out[-1].get("tool_calls") and not m.get("tool_calls"):
            merged = dict(out[-1])
            merged["content"] = (_text_of(merged.get("content")) + "\n\n" + _text_of(m.get("content"))).strip()
            out[-1] = merged
            continue
        out.append(m)
    return out


def verify_invariants(messages: List[Dict[str, Any]]) -> List[str]:
    """Return a list of invariant violations (empty == healthy). Used by tests and shadow telemetry."""
    problems: List[str] = []
    live_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                live_ids.add(str(tc.get("id")))
    prev_role = None
    for idx, m in enumerate(messages):
        role = m.get("role")
        if role == "tool" and str(m.get("tool_call_id")) not in live_ids:
            problems.append(f"orphan tool row at {idx}")
        if role in ("user", "assistant") and role == prev_role:
            problems.append(f"same-role adjacency ({role}) at {idx}")
        if role == "assistant" and not m.get("tool_calls") and not _text_of(m.get("content")).strip():
            problems.append(f"empty assistant row at {idx}")
        # A tool row legitimately separates ``assistant(tool_calls) → tool →
        # assistant(text)``; only literal neighbours count as adjacency.
        prev_role = role
    return problems


# --------------------------------------------------------------------------- #
# Orchestration entry used by ContextCompressor
# --------------------------------------------------------------------------- #

def _run_relevance_stage(
    messages: List[Dict[str, Any]],
    *,
    head_end: int,
    tail_start: int,
    scorer: RelevanceScorer,
    config: Dict[str, Any],
    session_id: str = "",
    redact: Callable[[str], str] | None = None,
    pristine: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict[str, Any]], ApplyReport, Dict[str, Any]]:
    """Score + apply over ``[head_end, tail_start)``. Raises ScorerUnavailable on backend failure."""
    t0 = time.monotonic()
    max_state = int(config.get("max_state_tokens", 25000) or 25000)
    max_request = int(config.get("max_request_tokens", 28000) or 28000)
    # Leave at least ~40% of each request for questions.
    state_cap = max(2000, min(max_state, int(max_request * 0.6)))
    items = collect_score_items(messages, head_end, tail_start, pristine=pristine, redact=redact)
    telemetry: Dict[str, Any] = {"backend": getattr(scorer, "name", "?"), "items": len(items), "windows": 0}
    if not items:
        report = ApplyReport(tokens_before=estimate_messages_tokens_rough(messages))
        report.tokens_after = report.tokens_before
        telemetry["duration_ms"] = int((time.monotonic() - t0) * 1000)
        return messages, report, telemetry

    windows = split_windows(messages, head_end, tail_start, max_window_tokens=max(2000, max_state // 2))
    telemetry["windows"] = len(windows)
    results: List[ScoreResult] = []
    for w in windows:
        w_items = [i for i in items if i.assistant_idx in w]
        if not w_items:
            continue
        state = render_state(messages, head_end, w, max_tokens=state_cap, redact=redact)
        safe_items = [replace(i, args_preview=redact(i.args_preview),
                              result_note=redact(i.result_note)) for i in w_items] if redact else w_items
        # The state is bounded, but the questions grow with the number of tool
        # calls in the window. Batch them so every request (state + questions)
        # stays under the backend's request budget instead of failing whole.
        question_budget = max(1000, max_request - _json_tokens(state) - 500)
        batches = batch_items_by_budget(safe_items, question_budget)
        telemetry["requests"] = telemetry.get("requests", 0) + len(batches)
        for batch in batches:
            _check_stage_budget()
            scores = scorer.score(state, batch)
            _check_stage_budget()
            results.extend(scores)

    _check_stage_budget()
    out, report = apply_relevance_decisions(
        messages, items, results,
        head_end=head_end, tail_start=tail_start,
        keep_threshold=float(config.get("keep_threshold", 0.5) or 0.5),
        truncate_head_chars=int(config.get("truncate_head_chars", 300) or 300),
        session_id=session_id,
        pristine=pristine,
    )
    telemetry.update({
        "kept": report.kept, "truncated": report.truncated, "dropped": report.dropped,
        "tokens_before": report.tokens_before, "tokens_after": report.tokens_after,
        "reduction": round(report.reduction_ratio, 3),
        "duration_ms": int((time.monotonic() - t0) * 1000),
    })
    return out, report, telemetry


def run_relevance_stage(
    messages: List[Dict[str, Any]],
    *,
    head_end: int,
    tail_start: int,
    scorer: RelevanceScorer,
    config: Dict[str, Any],
    session_id: str = "",
    redact: Callable[[str], str] | None = None,
    pristine: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict[str, Any]], ApplyReport, Dict[str, Any]]:
    """Bound the entire synchronous stage, even for uncooperative plugins.

    Late workers own only local scores/results; the owner alone can publish.
    Python cannot stop an in-flight network call. Keep its slot occupied until
    it finishes, and never close a process-shared auxiliary client.
    """
    deadline = time.monotonic() + float(config.get("timeout_seconds", 20) or 20)
    from agent.auxiliary_client import (
        AuxiliaryExplicitCancellation, _AuxiliaryCancellationDecision,
        _capture_aux_cancel_check, aux_interrupt_protection,
    )
    cancel_check = _AuxiliaryCancellationDecision(_capture_aux_cancel_check() or (lambda: False))

    def check() -> float:
        if cancel_check():
            raise AuxiliaryExplicitCancellation()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if not cancel_check.begin_timeout_cleanup():
                raise AuxiliaryExplicitCancellation()
            raise ScorerUnavailable("relevance stage deadline exceeded")
        return remaining

    check()
    if not _STAGE_WORKERS.acquire(blocking=False):
        raise ScorerUnavailable("relevance stage workers busy")
    done = threading.Event()
    outcome: Dict[str, Any] = {}
    context = contextvars.copy_context()

    def worker() -> None:
        token = _STAGE_BUDGET.set((deadline, cancel_check))
        try:
            # This stage already isolates blocking work. Avoid nested protected
            # provider workers releasing our slot while their network call lives.
            with aux_interrupt_protection(active=False, cancel_check=cancel_check):
                check()
                outcome["result"] = _run_relevance_stage(
                    messages, head_end=head_end, tail_start=tail_start,
                    scorer=scorer, config=config, session_id=session_id,
                    redact=redact, pristine=pristine,
                )
        except BaseException as exc:
            outcome["exception"] = exc
        finally:
            _STAGE_BUDGET.reset(token)
            _STAGE_WORKERS.release()
            done.set()

    try:
        threading.Thread(target=context.run, args=(worker,),
                         name="hermes-relevance-stage", daemon=True).start()
    except BaseException:
        _STAGE_WORKERS.release()
        raise
    while True:
        remaining = check()
        if done.wait(min(.01, remaining)):
            check()
            if "exception" in outcome:
                raise outcome["exception"]
            return outcome["result"]
