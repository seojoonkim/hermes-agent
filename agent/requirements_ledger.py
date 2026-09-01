"""Turn-scoped requirements tracking and todo projection.

This module deliberately depends only on the standard library so the ledger can
be reused by transports and runtimes without importing agent infrastructure.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, Iterable, List, Optional, Set


class TurnRequirementsLedger:
    """Record must-satisfy guidance received during one agent turn."""

    def __init__(self, turn_id: str, lock: Optional[threading.RLock] = None):
        cleaned_turn_id = str(turn_id).strip()
        if not cleaned_turn_id:
            raise ValueError("turn_id must not be empty")
        self.turn_id = cleaned_turn_id
        self._lock = lock or threading.RLock()
        self._revision = 0
        self._requirements: List[Dict[str, Any]] = []

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @staticmethod
    def classify(text: str) -> str:
        """Classify work using fixed, deterministic lexical complexity rules."""
        normalized = " ".join(str(text).split()).lower()
        words = re.findall(r"[\w'-]+", normalized)
        deep_markers = (
            "architecture", "refactor", "migrate", "integration", "end-to-end",
            "security", "concurrent", "repository", "full suite",
        )
        action_markers = (
            " and ", " then ", "test", "verify", "update", "change", "implement",
            "create", "fix", "run ",
        )
        deep_score = sum(marker in normalized for marker in deep_markers)
        action_score = sum(marker in normalized for marker in action_markers)
        if len(words) >= 18 or deep_score >= 2 or (deep_score and action_score >= 3):
            return "deep"
        if len(words) >= 7 or action_score >= 2 or deep_score:
            return "standard"
        return "fast"

    def register_steer(self, text: str) -> Dict[str, Any]:
        content = " ".join(str(text).split())
        if not content:
            raise ValueError("requirement text must not be empty")
        with self._lock:
            self._revision += 1
            requirement = {
                "id": f"req:{self.turn_id}:{self._revision:06d}",
                "revision": self._revision,
                "content": content,
                "must": True,
                "classification": self.classify(content),
                "status": "pending",
            }
            self._requirements.append(requirement)
            return requirement.copy()

    def rollback_registration(self, requirement_id: str) -> None:
        """Undo the newest registration when its atomic todo projection fails."""
        with self._lock:
            if not self._requirements or self._requirements[-1]["id"] != requirement_id:
                raise RuntimeError("can only roll back the newest requirement")
            self._requirements.pop()
            self._revision -= 1

    def requirements_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [item.copy() for item in self._requirements]

    def pending_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                item.copy() for item in self._requirements
                if item["must"] and item["status"] != "completed"
            ]

    def project_todos(self) -> List[Dict[str, str]]:
        with self._lock:
            return [self._as_todo(item) for item in self._requirements]

    def reconcile_todos(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Sync known statuses and restore requirements omitted by replace writes."""
        with self._lock:
            supplied = {str(item.get("id", "")): item for item in todos}
            known = {item["id"]: item for item in self._requirements}
            for req_id, requirement in known.items():
                incoming = supplied.get(req_id)
                if incoming is not None:
                    status = str(incoming.get("status", requirement["status"])).lower()
                    if status in {"pending", "in_progress", "completed", "cancelled"}:
                        requirement["status"] = status

            result = [dict(item) for item in todos]
            # Requirement identity and text are ledger-owned; only status is mutable.
            for item in result:
                canonical = known.get(str(item.get("id", "")))
                if canonical is not None:
                    item["content"] = canonical["content"]
                    item["status"] = canonical["status"]
            present_ids = {str(item.get("id", "")) for item in result}
            for requirement in self._requirements:
                if requirement["id"] not in present_ids:
                    result.append(self._as_todo(requirement))
            return result

    def completion_decision(
        self, existing_completed: Optional[Iterable[str]] = None
    ) -> Dict[str, Any]:
        """Return whether every must requirement is completed.

        ``existing_completed`` lets a caller include completion evidence that
        predates the latest todo reconciliation without mutating the ledger.
        """
        completed: Set[str] = {str(item) for item in (existing_completed or ())}
        with self._lock:
            pending_ids = [
                item["id"] for item in self._requirements
                if item["must"]
                and item["status"] != "completed"
                and item["id"] not in completed
            ]
            return {
                "complete": not pending_ids,
                "pending_ids": pending_ids,
                "revision": self._revision,
            }

    @staticmethod
    def _as_todo(requirement: Dict[str, Any]) -> Dict[str, str]:
        return {
            "id": requirement["id"],
            "content": requirement["content"],
            "status": requirement["status"],
        }
