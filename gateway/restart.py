"""Shared gateway restart constants and parsing helpers."""

from hermes_cli.config import DEFAULT_CONFIG

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart
# the gateway after a graceful drain/reload path completes.
GATEWAY_SERVICE_RESTART_EXIT_CODE = 75

# The model emits this only after it has completed and verified every item from
# the interrupted request. The gateway consumes it before user delivery and
# uses it as the durable resume_pending clear condition.
RESTART_RESUME_COMPLETE_MARKER = "[GATEWAY_RESUME_COMPLETE]"

DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT = float(
    DEFAULT_CONFIG["agent"]["restart_drain_timeout"]
)


def parse_restart_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to the shared default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    return max(0.0, value)


def build_restart_resume_note(reason: str | None) -> str:
    """Build the durable recovery instruction for an interrupted gateway turn.

    A persisted assistant progress update can describe actions that were still
    pending when the process stopped. Recovery must therefore audit the whole
    original request and transcript, not only an unfinished tool-result tail.
    """
    reason_phrase = (
        "the agent reaching its bounded tool-iteration limit"
        if reason == "runtime_max_iterations"
        else "a gateway restart"
        if reason == "restart_timeout"
        else "a gateway shutdown"
        if reason == "shutdown_timeout"
        else "a gateway interruption"
    )
    return (
        "[System note: Your previous turn in this session was interrupted "
        f"by {reason_phrase}. The conversation history below is intact. "
        "Limit recovery to the current chat/topic session lineage. Never audit, "
        "resume, or report unfinished work from an unrelated room, chat, channel, "
        "topic, thread, or session key. The conversation history supplied to this "
        "turn is the complete recovery scope. Resume the original user request "
        "before treating any newer message as "
        "a replacement. Audit the transcript for unfinished tool results and "
        "for promised actions or deliverables in assistant progress updates. "
        "Statements of progress or intent are not evidence of completion. "
        "For every still-pending item, execute and verify it, then report what was "
        "actually completed. Treat each promised post-restart check and final report "
        "as pending until the transcript contains its concrete verification evidence; "
        "a related but narrower success (for example, saving a preference) does not "
        "complete the promised runtime checks. Only after every item is complete and "
        "verified, append "
        f"{RESTART_RESUME_COMPLETE_MARKER} to your final response. Do not append it "
        "to a progress update, partial result, blocker report, or plan. Address the "
        "user's new message only after recovering the original request.]"
    )


def consume_restart_resume_marker(text: object) -> tuple[str, bool]:
    """Strip the internal completion marker and return whether it was present."""
    value = str(text or "")
    completed = RESTART_RESUME_COMPLETE_MARKER in value
    return value.replace(RESTART_RESUME_COMPLETE_MARKER, "").rstrip(), completed
