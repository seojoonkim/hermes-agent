"""MemKraft memory provider for Hermes.

Local-first bridge from Hermes MemoryProvider to Simon's MemKraft repo.

Config in $HERMES_HOME/config.yaml::

  memory:
    provider: memkraft
  plugins:
    memkraft:
      base_dir: /path/to/memory
      # Optional for editable/source checkouts; normally MemKraft is imported from the active Python env.
      source_path: /path/to/memcraft/src
      prefetch_top_k: 5
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_PATH = ""


def _load_config(hermes_home: str | None = None) -> dict:
    """Return the ``plugins.memkraft`` config block for a profile.

    With no explicit ``hermes_home`` this defers to the cached, profile-aware
    ``load_config_readonly()`` (read-only: the returned dict is never mutated
    here).  When a caller passes an explicit home — ``initialize()`` does, so
    each profile's provider instance stays pinned to its own directory — the
    file under that home is read directly instead.
    """
    if hermes_home is None:
        try:
            from hermes_cli.config import load_config_readonly

            raw = load_config_readonly() or {}
            return ((raw.get("plugins") or {}).get("memkraft") or {})
        except Exception as e:
            logger.debug("MemKraft config load failed: %s", e)
        try:
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
        except Exception:
            hermes_home = os.environ.get("HERMES_HOME", "")
    cfg_path = Path(hermes_home) / "config.yaml" if hermes_home else None
    if not cfg_path or not cfg_path.exists():
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8-sig") as f:
            raw = yaml.safe_load(f) or {}
        return ((raw.get("plugins") or {}).get("memkraft") or {})
    except Exception as e:
        logger.debug("MemKraft config load failed: %s", e)
        return {}


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


SEARCH_SCHEMA = _schema(
    "memkraft_search",
    "Search this profile's local MemKraft memory/archive. Use for project history, prior decisions, old OpenClaw/MemKraft context, and durable profile-specific recall.",
    {
        "query": {"type": "string", "description": "Natural-language query or keywords."},
        "top_k": {"type": "integer", "description": "Maximum hits, default 5."},
        "expand_query": {"type": "boolean", "description": "Use MemKraft keyword expansion, default true."},
        "fuzzy": {"type": "boolean", "description": "Enable fuzzy matching, default false."},
    },
    ["query"],
)

SEARCH_TYPED_SCHEMA = _schema(
    "memkraft_search_typed",
    "Typed MemKraft search with optional entity_type and fact_key post-filters.",
    {
        "query": {"type": "string"},
        "entity_type": {"type": "string", "description": "Optional entity type filter, e.g. person/project."},
        "fact_key": {"type": "string", "description": "Optional fact key filter."},
        "top_k": {"type": "integer", "description": "Maximum hits, default 5."},
        "fuzzy": {"type": "boolean", "description": "Enable fuzzy matching, default false."},
    },
    ["query"],
)

REMEMBER_SCHEMA = _schema(
    "memkraft_remember",
    "Store a durable fact in this profile's MemKraft memory. Use for explicit user preferences, stable project facts, decisions, and reusable lessons. Do not store temporary task progress or secrets.",
    {
        "content": {"type": "string", "description": "Declarative fact to remember."},
        "entity": {"type": "string", "description": "Entity name, default Simon."},
        "key": {"type": "string", "description": "Fact key/category, default note."},
        "fact_type": {"type": "string", "enum": ["semantic", "episodic", "procedural"], "description": "Cognitive type, default semantic."},
        "source": {"type": "string", "description": "Source label, default Hermes."},
    },
    ["content"],
)

STATUS_SCHEMA = _schema(
    "memkraft_status",
    "Inspect this profile's MemKraft provider status: base_dir, counts, cache stats, and version.",
    {},
)

REASONING_SEARCH_SCHEMA = _schema(
    "memkraft_reasoning_search",
    "Recall similar reasoning trajectories before starting a task. Use to avoid repeating past failures and reuse successful patterns.",
    {
        "task": {"type": "string", "description": "Task description to match against past reasoning."},
        "top_k": {"type": "integer", "description": "Maximum trajectories, default 5."},
    },
    ["task"],
)

LOG_REASONING_SCHEMA = _schema(
    "memkraft_log_reasoning",
    "Record a reasoning episode or task outcome in MemKraft ReasoningBank. Use after complex tasks, failures, partial successes, or reusable discoveries.",
    {
        "task": {"type": "string", "description": "Short task description."},
        "outcome": {"type": "string", "enum": ["success", "failure", "partial"], "description": "Outcome."},
        "steps": {"type": "array", "items": {"type": "string"}, "description": "Optional reasoning/action steps."},
        "error": {"type": "string", "description": "Optional error or lesson."},
        "tags": {"type": "string", "description": "Comma-separated tags."},
    },
    ["task", "outcome"],
)

REASONING_STATS_SCHEMA = _schema(
    "memkraft_reasoning_stats",
    "Return aggregate ReasoningBank stats: success/failure counts and top failure patterns.",
    {},
)

DECISION_SCHEMA = _schema(
    "memkraft_decision_record",
    "Record a durable decision in MemKraft's decision store with rationale and rollout notes.",
    {
        "what": {"type": "string", "description": "One-line decision."},
        "why": {"type": "string", "description": "Rationale."},
        "how": {"type": "string", "description": "Implementation or rollout notes."},
        "outcome": {"type": "string", "description": "Optional observed outcome."},
        "tags": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": ["proposed", "accepted", "superseded", "rejected"], "description": "Decision status, default accepted."},
        "source": {"type": "string", "description": "Source label, default Hermes."},
    },
    ["what", "why", "how"],
)

EXTRACT_SCHEMA = _schema(
    "memkraft_extract_structured",
    "Extract dates, URLs, emails, money, percentages, versions, and phones from text; optionally auto-save extracted fields to an entity.",
    {
        "text": {"type": "string"},
        "entity_hint": {"type": "string", "description": "Optional entity name for auto_save."},
        "auto_save": {"type": "boolean", "description": "If true and entity_hint is set, save extracted facts."},
    },
    ["text"],
)

CACHE_STATS_SCHEMA = _schema(
    "memkraft_cache_stats",
    "Return MemKraft search cache statistics.",
    {},
)


class MemKraftMemoryProvider(MemoryProvider):
    """Hermes memory-provider adapter for MemKraft."""

    def __init__(self):
        self._config: dict = {}
        self._mk = None
        self._base_dir = ""
        self._profile = ""
        self._source_path = DEFAULT_SOURCE_PATH
        self._prefetch_top_k = 5
        self._prefetch_cache: Dict[tuple[str, str], str] = {}
        self._prefetch_threads: Dict[tuple[str, str], threading.Thread] = {}
        self._prefetch_expected: Dict[str, str] = {}
        self._turn_timing: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "memkraft"

    def is_available(self) -> bool:
        cfg = _load_config()
        source_path = cfg.get("source_path") or DEFAULT_SOURCE_PATH
        if source_path and source_path not in sys.path:
            sys.path.insert(0, source_path)
        try:
            import importlib
            from packaging.version import Version

            module = importlib.import_module("memkraft")
            if Version(str(getattr(module, "__version__", "0"))) < Version("3.5.0"):
                return False
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
            raw_probe_dir = cfg.get("base_dir") or str(Path(hermes_home) / "memkraft-memory")
            probe_dir = str(raw_probe_dir).replace("$HERMES_HOME", hermes_home).replace(
                "${HERMES_HOME}", hermes_home
            )
            probe = getattr(module, "MemKraft")(str(Path(probe_dir).expanduser()))
            required = (
                "delay_run_start", "delay_run_finish", "delay_estimate",
                "delay_record_retrospective", "delay_record_action",
                "delay_record_application", "delay_record_verification",
            )
            return all(callable(getattr(probe, name, None)) for name in required)
        except Exception:
            return False

    def get_config_schema(self):
        return [
            {"key": "base_dir", "description": "MemKraft memory base_dir for this profile", "default": "$HERMES_HOME/memkraft-memory"},
            {"key": "source_path", "description": "Optional path containing a source checkout of the memkraft package; leave empty to use the installed package", "default": ""},
            {"key": "prefetch_top_k", "description": "Hits injected before a turn", "default": "5"},
            {"key": "eval_bridge_enabled", "description": "Register allowlisted delegation failures in this profile's local evaluation corpus", "default": False, "type": "boolean"},
        ]

    def save_config(self, values, hermes_home):
        import yaml

        cfg_path = Path(hermes_home) / "config.yaml"
        raw = {}
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8-sig") as f:
                raw = yaml.safe_load(f) or {}
        raw.setdefault("plugins", {})["memkraft"] = values
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME") or ""
        if not hermes_home:
            try:
                from hermes_constants import get_hermes_home

                hermes_home = str(get_hermes_home())
            except Exception:
                hermes_home = ""
        self._config = _load_config(hermes_home)
        self._profile = kwargs.get("agent_identity") or Path(hermes_home).name or "hermes"
        self._source_path = self._config.get("source_path") or DEFAULT_SOURCE_PATH
        if self._source_path and self._source_path not in sys.path:
            sys.path.insert(0, self._source_path)
        import importlib

        MemKraft = getattr(importlib.import_module("memkraft"), "MemKraft")

        base_dir = self._config.get("base_dir") or str(Path(hermes_home) / "memkraft-memory")
        base_dir = str(base_dir).replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)
        self._base_dir = str(Path(base_dir).expanduser())
        self._prefetch_top_k = int(self._config.get("prefetch_top_k", 5) or 5)
        self._mk = MemKraft(self._base_dir)
        required_delay_api = (
            "delay_run_start", "delay_run_finish", "delay_estimate",
            "delay_record_retrospective", "delay_record_action",
            "delay_record_application", "delay_record_verification",
        )
        missing = [name for name in required_delay_api if not callable(getattr(self._mk, name, None))]
        if missing:
            self._mk = None
            raise RuntimeError(
                "MemKraft 3.5.0+ is required; missing delay API: " + ", ".join(missing)
            )
        try:
            self._mk.init(verbose=False)
        except TypeError:
            self._mk.init()

    def system_prompt_block(self) -> str:
        if not self._mk:
            return ""
        return (
            "# MemKraft Memory\n"
            f"Active for profile `{self._profile}` at `{self._base_dir}`.\n"
            "Use `memkraft_search` for project/history recall, `memkraft_reasoning_search` before complex tasks, "
            "`memkraft_log_reasoning` after reusable successes/failures, `memkraft_decision_record` for stable decisions, "
            "and `memkraft_remember` only for durable facts. Do not store secrets or temporary task progress."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        session_key = session_id or "default"
        normalized_query = query.strip()
        key = (session_key, normalized_query)
        with self._lock:
            self._prefetch_expected[session_key] = normalized_query
            for stale_key in [k for k in self._prefetch_cache if k[0] == session_key and k != key]:
                self._prefetch_cache.pop(stale_key, None)
            cached = self._prefetch_cache.pop(key, "")
        if cached:
            return cached
        # First turn/cache miss fallback stays synchronous so recall works immediately.
        return self._format_prefetch_context(query, self._prefetch_top_k)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not query.strip() or not self._mk:
            return
        session_key = session_id or "default"
        normalized_query = query.strip()
        key = (session_key, normalized_query)
        with self._lock:
            self._prefetch_expected[session_key] = normalized_query
            if key in self._prefetch_threads and self._prefetch_threads[key].is_alive():
                return

        def _worker():
            try:
                text = self._format_prefetch_context(query, self._prefetch_top_k)
                with self._lock:
                    if self._prefetch_expected.get(session_key) == normalized_query:
                        self._prefetch_cache[key] = text
            except Exception as e:
                logger.debug("MemKraft background prefetch failed: %s", e)
            finally:
                with self._lock:
                    self._prefetch_threads.pop(key, None)

        t = threading.Thread(target=_worker, daemon=True, name=f"memkraft-prefetch-{key[0][:12]}")
        with self._lock:
            self._prefetch_threads[key] = t
        t.start()

    @staticmethod
    def _timing_subject(platform: Any, profile: Any, subject: Any) -> str:
        platforms = {"telegram", "discord", "slack", "cli", "cron", "gateway"}
        subjects = {"development", "research", "operations", "scheduling", "health", "general"}
        safe_platform = platform if platform in platforms else "gateway"
        safe_subject = subject if subject in subjects else "general"
        safe_profile = profile if isinstance(profile, str) and re.fullmatch(r"[a-z0-9_-]{1,32}", profile) else "hermes"
        return f"{safe_platform}:{safe_profile}:{safe_subject}"


    @staticmethod
    def _timing_key(session_id: Any, turn_number: int) -> str:
        raw = f"{session_id if isinstance(session_id, str) else ''}:{turn_number}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


    @staticmethod
    def _timing_now() -> datetime:
        return datetime.now(timezone.utc)


    def on_turn_timing_start(self, turn_number: int, **kwargs) -> None:
        if not self._mk:
            return
        key = self._timing_key(kwargs.get("session_id"), turn_number)
        subject = self._timing_subject(kwargs.get("platform"), self._profile, kwargs.get("subject"))
        task_id = f"hermes-turn-{key}"
        started = time.monotonic()
        try:
            task_result = self._mk.delay_run_start(task_id, "task", subject, now=self._timing_now())
            if not isinstance(task_result, dict) or task_result.get("outcome") not in {"applied", "already_applied"}:
                return
        except Exception:
            return
        with self._lock:
            self._turn_timing[key] = {
                "task_id": task_id,
                "subject": subject,
                "task_started": started,
                "phase": "active",
                "phase_started": started,
                "phase_totals_ms": {"active": 0, "wait": 0, "rework": 0},
            }


    def on_turn_progress(self, turn_number: int, **kwargs) -> None:
        phase = kwargs.get("phase")
        if phase not in {"active", "wait", "rework"} or not self._mk:
            return
        key = self._timing_key(kwargs.get("session_id"), turn_number)
        now_tick = time.monotonic()
        with self._lock:
            state = self._turn_timing.get(key)
            if not state or state["phase"] == phase:
                return
            elapsed_ms = max(0, round((now_tick - state["phase_started"]) * 1000))
            state["phase_totals_ms"][state["phase"]] += elapsed_ms
            state["phase"] = phase
            state["phase_started"] = now_tick


    def on_turn_finish(self, turn_number: int, **kwargs) -> None:
        key = self._timing_key(kwargs.get("session_id"), turn_number)
        self._finish_timing_key(key, kwargs.get("outcome"))


    def _finish_timing_key(self, key: str, outcome: Optional[str]) -> None:
        if not self._mk:
            return
        now_tick = time.monotonic()
        with self._lock:
            state = self._turn_timing.get(key)
            if not state:
                return
            if not state.get("totals_finalized"):
                elapsed_ms = max(0, round((now_tick - state["phase_started"]) * 1000))
                state["phase_totals_ms"][state["phase"]] += elapsed_ms
                state["totals_finalized"] = True
                state["task_ms"] = max(
                    0, round((now_tick - state["task_started"]) * 1000)
                )
            if outcome not in {"completed", "failed", "interrupted", "partial", "aborted"}:
                outcome = "failed"
            # The first close request owns the durable outcome. A later repair
            # (including shutdown's abort sweep) must not rewrite it.
            outcome = state.setdefault("outcome", outcome)

            # Flush one aggregate child per observed phase. The number of
            # ledger writes is bounded by the closed phase set, not API calls.
            flushed = state.setdefault("flushed_phases", set())
            started = state.setdefault("started_phases", set())
            for phase_name in ("active", "wait", "rework"):
                phase_ms = state["phase_totals_ms"][phase_name]
                if phase_ms <= 0 or phase_name in flushed:
                    continue
                phase_id = f"{state['task_id']}-{phase_name}-aggregate"
                try:
                    if phase_name not in started:
                        start_result = self._mk.delay_run_start(
                            phase_id, "phase", f"{state['subject']}:{phase_name}",
                            parent_run_id=state["task_id"], now=self._timing_now(),
                        )
                        if (not isinstance(start_result, dict)
                                or start_result.get("outcome") not in {"applied", "already_applied"}):
                            return
                        # MemKraft 3.5 rejects a second start for the same ID.
                        started.add(phase_name)
                    finish_result = self._mk.delay_run_finish(
                        phase_id, phase_ms, now=self._timing_now()
                    )
                    if (not isinstance(finish_result, dict)
                            or finish_result.get("outcome") not in {"applied", "already_applied"}):
                        return
                except Exception:
                    return
                flushed.add(phase_name)
            try:
                task_result = self._mk.delay_run_finish(
                    state["task_id"], state["task_ms"],
                    outcome=outcome, now=self._timing_now(),
                )
            except Exception:
                return
            if (not isinstance(task_result, dict)
                    or task_result.get("outcome") not in {"applied", "already_applied"}):
                return
            self._turn_timing.pop(key, None)


    def estimate_turn(self, **kwargs) -> Optional[Dict[str, Any]]:
        if not self._mk or not hasattr(self._mk, "delay_estimate"):
            return None
        subject = self._timing_subject(kwargs.get("platform"), self._profile, kwargs.get("subject"))
        try:
            estimate = self._mk.delay_estimate("task", subject, now=self._timing_now())
        except Exception:
            return None
        if not isinstance(estimate, dict) or estimate.get("available") is not True:
            return None
        count = estimate.get("sample_count", 0)
        # Enforce the privacy/quality floor at the adapter boundary too. Do not
        # rely solely on an upstream implementation's ``available`` flag.
        if not isinstance(count, int) or isinstance(count, bool) or count < 5:
            return None
        confidence = "high" if count >= 20 else "medium" if count >= 8 else "low"
        return {
            "p50_ms": estimate["p50_ms"],
            "p80_ms": estimate["p80_ms"],
            "confidence": confidence,
            "sample_count": count,
            "risk_adjustment_ms": max(0, estimate["p80_ms"] - estimate["p50_ms"]),
            "critical_path": subject.rsplit(":", 1)[-1],
        }


    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Avoid noisy full-transcript storage. Durable writes happen through explicit tools or built-in memory mirror.
        return None

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Give the compressor compact MemKraft recall for the latest user intent."""
        if not self._mk or not messages:
            return ""
        latest_user = ""
        for m in reversed(messages):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                latest_user = m.get("content", "")[:1000]
                break
        if not latest_user.strip():
            return ""
        ctx = self._format_search_context(latest_user, min(self._prefetch_top_k, 5))
        if not ctx:
            return ""
        return "MemKraft recall to preserve during compression:\n" + ctx

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        if reset:
            with self._lock:
                self._prefetch_cache.clear()
                self._prefetch_expected.clear()

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if action != "add" or not content or not self._mk:
            return
        entity = "Simon" if target == "user" else f"{self._profile} memory"
        key = "user_profile" if target == "user" else "memory"
        self._store_fact(entity=entity, key=key, content=content, fact_type="semantic", source="Hermes built-in memory")

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        # Do not auto-store delegated results; many are temporary task progress.
        return None

    def on_delegation_outcome(self, outcome: Any) -> None:
        """Opt-in, fail-closed sink for privacy-safe delegation failure cases."""
        try:
            if self._config.get("eval_bridge_enabled") is not True or self._mk is None:
                return None
            if not isinstance(outcome, dict):
                return None
            kind = outcome.get("kind")
            if kind not in {
                "delegated_failed",
                "delegated_partial",
                "delegated_timed_out",
            }:
                return None

            # Intentionally do not forward any outcome content or references.
            # The bridge derives opaque local digests from this fixed allowlist.
            event = {
                "kind": kind,
                "source_profile": "hermes.subagent",
                "expected_behavior": "Delegated work should complete successfully with a verified result.",
                "forbidden_behavior": "Failed, partial, or timed-out delegated work must not be treated as successful.",
                "privacy_scope": "local_private",
            }
            from plugins.evaluation.eval_bridge import register_failure_case

            register_failure_case(self._mk, event, now=datetime.now(timezone.utc))
        except Exception:
            # This callback must never expose adapter, sink, path, event, or raw
            # outcome details through logs or disrupt the calling agent.
            return None
        return None

    def on_incomplete_turn(self, handoff: Dict[str, Any]) -> bool:
        """Persist a Hermes handoff; scheduling remains gateway-owned."""
        if not self._mk or not self._base_dir or not isinstance(handoff, dict):
            return False
        try:
            from agent.runtime_resume import HANDOFF_FIELDS

            if set(handoff) != set(HANDOFF_FIELDS):
                return False
            payload = {
                "session_key": str(handoff.get("session_key") or "")[:200],
                "profile": str(handoff.get("profile") or ""),
                "goal": str(handoff.get("goal") or "")[:1000],
                "failing_test_ids": [
                    str(item)[:240] for item in (handoff.get("failing_test_ids") or [])[:10]
                ],
                "verified_sha": str(handoff.get("verified_sha") or "")[:40],
                "depth": int(handoff.get("depth") or 0),
                "status": "incomplete",
            }
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            token = hashlib.sha256(canonical).hexdigest()[:32]
        except (TypeError, ValueError):
            return False
        checkpoint_dir = Path(self._base_dir) / "tasks" / "hermes-resume"
        checkpoint_path = checkpoint_dir / f"{token}.json"
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if checkpoint_path.exists():
                return True
            fd, tmp_name = tempfile.mkstemp(prefix=f".{token}.", dir=str(checkpoint_dir))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp_name, checkpoint_path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            return True
        except Exception as exc:
            logger.debug("MemKraft incomplete-turn checkpoint failed: %s", exc)
            return False

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            SEARCH_SCHEMA,
            SEARCH_TYPED_SCHEMA,
            REMEMBER_SCHEMA,
            STATUS_SCHEMA,
            REASONING_SEARCH_SCHEMA,
            LOG_REASONING_SCHEMA,
            REASONING_STATS_SCHEMA,
            DECISION_SCHEMA,
            EXTRACT_SCHEMA,
            CACHE_STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._mk:
            return tool_error("MemKraft 3.5.0+ provider is unavailable or not initialized")
        try:
            if tool_name == "memkraft_search":
                q = str(args.get("query") or "").strip()
                if not q:
                    return tool_error("query is required")
                results = self._search(q, top_k=int(args.get("top_k") or 5), expand_query=bool(args.get("expand_query", True)), fuzzy=bool(args.get("fuzzy", False)))
                return self._json({"success": True, "base_dir": self._base_dir, "count": len(results), "results": [self._format_hit(r) for r in results]})

            if tool_name == "memkraft_search_typed":
                q = str(args.get("query") or "").strip()
                if not q:
                    return tool_error("query is required")
                results = self._search_typed(q, entity_type=str(args.get("entity_type") or ""), fact_key=str(args.get("fact_key") or ""), top_k=int(args.get("top_k") or 5), fuzzy=bool(args.get("fuzzy", False)))
                return self._json({"success": True, "base_dir": self._base_dir, "count": len(results), "results": [self._format_hit(r) for r in results]})

            if tool_name == "memkraft_remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return tool_error("content is required")
                stored = self._store_fact(
                    entity=str(args.get("entity") or "Simon").strip() or "Simon",
                    key=str(args.get("key") or "note").strip() or "note",
                    content=content,
                    fact_type=str(args.get("fact_type") or "semantic").strip() or "semantic",
                    source=str(args.get("source") or "Hermes").strip() or "Hermes",
                )
                return self._json({"success": True, "stored": stored, "base_dir": self._base_dir})

            if tool_name == "memkraft_status":
                return self._json(self._status())

            if tool_name == "memkraft_reasoning_search":
                task = str(args.get("task") or "").strip()
                if not task:
                    return tool_error("task is required")
                results = self._reasoning_search(task, int(args.get("top_k") or 5))
                return self._json({"success": True, "count": len(results), "results": results})

            if tool_name == "memkraft_log_reasoning":
                task = str(args.get("task") or "").strip()
                outcome = str(args.get("outcome") or "partial").strip() or "partial"
                if not task:
                    return tool_error("task is required")
                result = self._call("log_reasoning", task, outcome, steps=args.get("steps") or [], error=args.get("error"), tags=args.get("tags") or "")
                return self._json({"success": True, "result": result})

            if tool_name == "memkraft_reasoning_stats":
                result = self._call("reasoning_stats")
                return self._json({"success": True, "stats": result})

            if tool_name == "memkraft_decision_record":
                for k in ("what", "why", "how"):
                    if not str(args.get(k) or "").strip():
                        return tool_error(f"{k} is required")
                decision_id = self._call(
                    "decision_record",
                    str(args.get("what")).strip(),
                    str(args.get("why")).strip(),
                    str(args.get("how")).strip(),
                    outcome=(str(args.get("outcome") or "").strip() or None),
                    tags=args.get("tags") or [],
                    status=str(args.get("status") or "accepted").strip() or "accepted",
                    source=str(args.get("source") or "Hermes").strip() or "Hermes",
                )
                return self._json({"success": True, "decision_id": decision_id})

            if tool_name == "memkraft_extract_structured":
                text = str(args.get("text") or "")
                if not text.strip():
                    return tool_error("text is required")
                result = self._call("extract_structured", text, entity_hint=(str(args.get("entity_hint") or "").strip() or None), auto_save=bool(args.get("auto_save", False)))
                return self._json({"success": True, "result": result})

            if tool_name == "memkraft_cache_stats":
                result = self._call("cache_stats")
                return self._json({"success": True, "stats": result})

            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(f"MemKraft tool failed: {e}")

    def abort_open_turns(self) -> None:
        if not self._mk:
            return
        with self._lock:
            keys = list(self._turn_timing)
        for key in keys:
            self._finish_timing_key(key, "aborted")


    def shutdown(self) -> None:
        self.abort_open_turns()
        if self._turn_timing:
            logger.warning(
                "MemKraft shutdown left %d timing run(s) open for repair",
                len(self._turn_timing),
            )
        self._mk = None

    def _json(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _call(self, method: str, *args, **kwargs):
        if not self._mk:
            raise RuntimeError("MemKraft not initialized")
        fn = getattr(self._mk, method)
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def _search(self, query: str, top_k: int = 5, expand_query: bool = True, fuzzy: bool = False) -> list[dict]:
        if not self._mk:
            return []
        top_k = max(1, min(int(top_k or 5), 25))
        with contextlib.redirect_stdout(io.StringIO()):
            if hasattr(self._mk, "search_v2"):
                return self._mk.search_v2(query, top_k=top_k, expand_query=expand_query, fuzzy=fuzzy) or []
            out = self._mk.search(query, fuzzy=fuzzy) or []
            return out[:top_k]

    def _search_typed(self, query: str, *, entity_type: str = "", fact_key: str = "", top_k: int = 5, fuzzy: bool = False) -> list[dict]:
        if not self._mk:
            return []
        top_k = max(1, min(int(top_k or 5), 25))
        if hasattr(self._mk, "search_typed"):
            with contextlib.redirect_stdout(io.StringIO()):
                return self._mk.search_typed(query, entity_type=entity_type, fact_key=fact_key, top_k=top_k, fuzzy=fuzzy) or []
        return self._search(query, top_k=top_k, expand_query=True, fuzzy=fuzzy)

    def _reasoning_search(self, task: str, top_k: int = 5) -> list[dict]:
        if not self._mk:
            return []
        top_k = max(1, min(int(top_k or 5), 25))
        if hasattr(self._mk, "get_similar_reasoning"):
            return self._call("get_similar_reasoning", task, top_k=top_k) or []
        if hasattr(self._mk, "reasoning_recall"):
            return self._call("reasoning_recall", task, top_k=top_k) or []
        return []

    def _store_fact(self, entity: str, key: str, content: str, fact_type: str = "semantic", source: str = "Hermes") -> dict:
        if not self._mk:
            raise RuntimeError("MemKraft not initialized")
        entity = entity.strip() or "Simon"
        key = key.strip() or "note"
        content = content.strip()
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                self._mk.track(entity, entity_type="person" if entity.lower() == "simon" else "project", source=source)
            except Exception:
                pass
            try:
                fact = self._mk.fact_add(entity, key, content, fact_type=fact_type)
            except Exception:
                fact = {"entity": entity, "key": key, "value": content, "fact_type": fact_type}
            try:
                self._mk.update(entity, f"[{key}] {content}", source=source)
            except Exception:
                pass
        return fact

    def _format_prefetch_context(self, query: str, top_k: int) -> str:
        parts = []
        search_context = self._format_search_context(query, top_k)
        if search_context:
            parts.append(search_context)
        reasoning_context = self._format_reasoning_injection(query)
        if reasoning_context:
            parts.append(reasoning_context)
        return "\n\n".join(parts)

    def _format_reasoning_injection(self, query: str) -> str:
        if not self._mk or not hasattr(self._mk, "reasoning_inject_for_task"):
            return ""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                text = self._mk.reasoning_inject_for_task(query)
        except Exception as e:
            logger.debug("MemKraft ReasoningBank injection failed: %s", e)
            return ""
        if not isinstance(text, str) or not text.strip():
            return ""
        return text.strip()

    def _format_search_context(self, query: str, top_k: int) -> str:
        results = self._search(query, top_k=top_k, expand_query=True, fuzzy=False)
        if not results:
            return ""
        lines = ["## MemKraft Memory"]
        for r in results[:top_k]:
            file = r.get("file") or ""
            score = r.get("score", 0)
            snippet = self._snippet_for_hit(r)
            lines.append(f"- ({score:.3f}) {file}: {snippet}")
        return "\n".join(lines)

    def _snippet_for_hit(self, hit: dict) -> str:
        for key in ("snippet", "context", "content", "match"):
            val = hit.get(key)
            if isinstance(val, str) and val.strip():
                return " ".join(val.strip().split())[:300]
        fp = hit.get("file")
        if fp:
            try:
                p = Path(fp)
                if not p.is_absolute():
                    p = Path(self._base_dir) / p
                text = p.read_text(encoding="utf-8", errors="replace")
                return " ".join(text.split())[:300]
            except Exception:
                pass
        return ""

    def _format_hit(self, hit: dict) -> dict:
        return {
            "file": str(hit.get("file") or ""),
            "score": hit.get("score", 0),
            "match": str(hit.get("match") or "")[:200],
            "snippet": self._snippet_for_hit(hit),
        }

    def _status(self) -> dict:
        base = Path(self._base_dir) if self._base_dir else None
        counts = {}
        if base and base.exists():
            for sub in ["live-notes", "entities", "decisions", "originals", "inbox", "tasks", "sessions"]:
                d = base / sub
                counts[sub] = len(list(d.glob("*.md"))) if d.exists() else 0
        version = None
        cache_stats = None
        try:
            import memkraft

            version = getattr(memkraft, "__version__", None)
        except Exception:
            pass
        try:
            cache_stats = self._call("cache_stats") if self._mk and hasattr(self._mk, "cache_stats") else None
        except Exception:
            cache_stats = None
        return {
            "success": True,
            "provider": "memkraft",
            "profile": self._profile,
            "base_dir": self._base_dir,
            "source_path": self._source_path,
            "version": version,
            "counts": counts,
            "cache_stats": cache_stats,
        }
