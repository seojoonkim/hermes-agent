"""Bounded checkpoint/resume of incomplete (iteration-capped) turns.

When a turn stops because it hit the iteration cap, the user still gets the
partial answer — but the work itself is unfinished, and nothing in the runtime
picks it back up. This module carries the unfinished work into a *distinct
follow-up turn* instead of dropping it.

Three properties matter more than completeness of the handoff:

1. **Bounded.** The handoff is a fixed set of small fields (goal, failing test
   ids, a verified commit sha). It never carries transcript text, so a resume
   cannot grow with the conversation it came from.
2. **Sanitized.** Everything crossing into the handoff is re-derived from a
   strict pattern (a pytest node id, a 40-hex sha) rather than trusted. Raw
   tool output pasted into a "test id" is dropped, not truncated — truncation
   still leaks the head of a secret.
3. **Fail-open.** A resume is an optimization on top of a turn that already
   finished. Persisting, notifying memory, or scheduling can all fail; none of
   them may surface as an error on the user's completed turn. The single
   exception is *persistence*: if the checkpoint did not durably land we do not
   schedule, because a resume turn without its checkpoint would restate the
   prior summary instead of continuing.

The follow-up turn is scheduled only after the user's response has actually
been delivered (via the platform adapter's post-delivery callback), so the
resume never races ahead of the answer it follows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: How many times a turn may be resumed before the chain is cut. A resumed turn
#: that caps out again is itself resumable, but only up to this depth.
MAX_RESUME_DEPTH = 1

MAX_GOAL_CHARS = 1000
MAX_FAILING_TESTS = 10
MAX_TEST_ID_CHARS = 240

#: Marks the synthetic turn as internal so downstream turn handling (and the
#: user-facing transcript) can tell it apart from something the user typed.
RESUME_TURN_PREFIX = "[hermes:internal-resume]"

#: The complete, closed set of fields that may cross the handoff boundary.
HANDOFF_FIELDS = (
    "session_key",
    "profile",
    "goal",
    "failing_test_ids",
    "verified_sha",
    "depth",
)

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z", re.IGNORECASE)
_TEST_ID_RE = re.compile(r"\A[\w./\\-]+::[\w./\\\[\]-]+\Z")

#: Exit reasons that mean "ran out of iterations", not "finished" or "stopped".
_ELIGIBLE_EXIT_PREFIX = "max_iterations_reached"


def is_resume_eligible(result: Mapping[str, Any], depth: int) -> bool:
    """Decide from a finalized turn ``result`` whether it should be resumed.

    Only an iteration-capped turn qualifies: a completed, failed, interrupted,
    or budget-exhausted turn has either finished its work or been deliberately
    stopped, and resuming those would fight the user or burn budget twice.
    """
    if depth >= MAX_RESUME_DEPTH:
        return False
    if result.get("completed") or result.get("failed") or result.get("interrupted"):
        return False
    reason = str(result.get("turn_exit_reason") or "")
    return reason.startswith(_ELIGIBLE_EXIT_PREFIX)


def _clean_test_id(raw: Any) -> str:
    """Return ``raw`` only if it is *entirely* a pytest node id, else ``""``.

    Anything else — a node id with tool output appended, a shell fragment — is
    dropped whole. Salvaging the matching prefix would be worse than useless:
    it would let an attacker-controlled string decide where the cut lands.
    """
    text = str(raw or "").strip()
    if len(text) > MAX_TEST_ID_CHARS or not _TEST_ID_RE.match(text):
        return ""
    return text


def _clean_sha(raw: Any) -> str:
    text = str(raw or "").strip()
    return text.lower() if _SHA_RE.match(text) else ""


@dataclass(frozen=True)
class ResumeHandoff:
    """The whole durable contract between the capped turn and its successor."""

    session_key: str
    profile: str
    goal: str
    failing_test_ids: tuple[str, ...]
    verified_sha: str
    depth: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "profile": self.profile,
            "goal": self.goal,
            "failing_test_ids": list(self.failing_test_ids),
            "verified_sha": self.verified_sha,
            "depth": self.depth,
        }

    def to_turn_text(self) -> str:
        lines = [
            f"{RESUME_TURN_PREFIX} continue the previous turn, which stopped at the "
            "iteration cap before finishing.",
            f"goal: {self.goal}",
        ]
        if self.verified_sha:
            lines.append(f"last verified commit: {self.verified_sha}")
        if self.failing_test_ids:
            lines.append("failing tests: " + ", ".join(self.failing_test_ids))
        lines.append(
            "Verify current state before changing anything, then finish the goal or "
            "report a concrete blocker. Do not restate the previous summary."
        )
        return "\n".join(lines)


def build_handoff(
    *,
    session_key: str,
    profile: str,
    goal: Any,
    failing_test_ids: Iterable[Any],
    verified_sha: Any,
    depth: int,
) -> ResumeHandoff:
    """Build a bounded, sanitized handoff. Deterministic for equal inputs."""
    cleaned: list[str] = []
    for raw in failing_test_ids or ():
        test_id = _clean_test_id(raw)
        if test_id and test_id not in cleaned:
            cleaned.append(test_id)
            if len(cleaned) >= MAX_FAILING_TESTS:
                break
    return ResumeHandoff(
        session_key=str(session_key or "")[:200],
        profile=str(profile or ""),
        goal=str(goal or "").strip()[:MAX_GOAL_CHARS],
        failing_test_ids=tuple(cleaned),
        verified_sha=_clean_sha(verified_sha),
        depth=int(depth),
    )


def select_resume_provider(memory_manager: Any, profile: str) -> Any:
    """Pick the memory provider that should hear about a resume.

    Under multiplex each profile has its own adapter, so a profile-scoped match
    wins; an unscoped adapter is the fallback for profiles with no dedicated
    one. Providers that do not implement the hook are not candidates at all.
    """
    fallback = None
    try:
        providers = list(getattr(memory_manager, "providers", ()) or ())
    except Exception:
        logger.debug("resume: could not list memory providers", exc_info=True)
        return None
    for provider in providers:
        if not callable(getattr(provider, "on_incomplete_turn", None)):
            continue
        scope = getattr(provider, "profile", None)
        if scope and scope == profile:
            return provider
        if not scope and fallback is None:
            fallback = provider
    return fallback


class ResumeCoordinator:
    """Persists a checkpoint, then schedules one follow-up turn after delivery."""

    def __init__(
        self,
        *,
        memory_manager: Any,
        checkpoint_store: Any,
        register_post_delivery: Callable[[Callable[[], None]], Any],
        schedule_turn: Callable[[str], Any],
        session_key: str,
        profile: str = "",
        depth: int = 0,
    ) -> None:
        self._memory_manager = memory_manager
        self._checkpoint_store = checkpoint_store
        self._register_post_delivery = register_post_delivery
        self._schedule_turn = schedule_turn
        self._session_key = session_key
        self._profile = profile
        self._depth = depth

    def maybe_schedule(
        self,
        result: Mapping[str, Any],
        *,
        goal: Any = "",
        failing_test_ids: Sequence[Any] = (),
        verified_sha: Any = "",
    ) -> bool:
        """Arm a resume for this turn. Returns whether one was armed."""
        if not is_resume_eligible(result, self._depth):
            return False

        handoff = build_handoff(
            session_key=self._session_key,
            profile=self._profile,
            goal=goal,
            failing_test_ids=failing_test_ids,
            verified_sha=verified_sha,
            depth=self._depth,
        )

        if not self._persist(handoff):
            return False

        # Memory hears about the incomplete turn as soon as the checkpoint is
        # durable — it is scoped to this turn, not to the follow-up one.
        self._notify_memory(handoff)

        fired = False

        def on_delivered() -> None:
            # The adapter may invoke callbacks more than once (retried delivery,
            # multiple sinks); a resume must be scheduled exactly once.
            nonlocal fired
            if fired:
                return
            fired = True
            try:
                self._schedule_turn(handoff.to_turn_text())
            except Exception:
                logger.warning("resume: scheduling follow-up turn failed", exc_info=True)

        try:
            self._register_post_delivery(on_delivered)
        except Exception:
            logger.warning("resume: post-delivery registration failed", exc_info=True)
            return False
        return True

    def _persist(self, handoff: ResumeHandoff) -> bool:
        try:
            return bool(self._checkpoint_store.save_resume_checkpoint(handoff.to_payload()))
        except Exception:
            logger.warning("resume: persisting checkpoint failed", exc_info=True)
            return False

    def _notify_memory(self, handoff: ResumeHandoff) -> None:
        provider = select_resume_provider(self._memory_manager, self._profile)
        if provider is None:
            return
        try:
            provider.on_incomplete_turn(handoff.to_payload())
        except Exception:
            logger.warning("resume: memory provider hook failed", exc_info=True)


def maybe_schedule_resume(agent: Any, result: Mapping[str, Any], **kwargs: Any) -> None:
    """Turn-finalizer seam: never raises, never blocks the finished turn."""
    coordinator = getattr(agent, "resume_coordinator", None)
    if coordinator is None:
        return
    try:
        coordinator.maybe_schedule(result, **kwargs)
    except Exception:
        logger.warning("resume: coordinator failed", exc_info=True)
