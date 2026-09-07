"""Content-free lifecycle timing, including cancelled and failed turns."""
import asyncio
import logging
import time
from contextlib import contextmanager
logger = logging.getLogger(__name__)

@contextmanager
def measure_agent_stage(session_id, generation, preparation_started):
    started = time.monotonic()
    outcome = 'ok'
    try:
        yield
    except asyncio.CancelledError:
        outcome = 'cancelled'
        raise
    except BaseException:
        outcome = 'error'
        raise
    finally:
        logger.info(
            'gateway_latency: session=%s generation=%s prep_seconds=%.3f agent_seconds=%.3f outcome=%s',
            session_id, generation, started-preparation_started, time.monotonic()-started, outcome,
        )
