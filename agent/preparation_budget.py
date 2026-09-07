"""Turn-local deadline for optional pre-answer maintenance, not user work."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
_deadline = ContextVar('preparation_deadline', default=None)

@contextmanager
def preparation_budget(seconds):
    previous = _deadline.get()
    deadline = time.monotonic() + seconds
    token = _deadline.set(min(previous, deadline) if previous is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)

def remaining_preparation_seconds(limit):
    deadline = _deadline.get()
    return limit if deadline is None else max(0.001, min(limit, deadline-time.monotonic()))
