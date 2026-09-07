"""Static-only admission stays occupied until detached workers really finish."""

import concurrent.futures
import socket
import threading

import pytest

from agent import conversation_compression as cc
from tools.daemon_pool import DaemonThreadPoolExecutor


@pytest.fixture
def static_pool(monkeypatch):
    """Own every worker; restore globals only after all callbacks have drained."""
    slot = threading.BoundedSemaphore(1)
    pool = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="test-static")
    futures = []
    releases = []

    class RecordingExecutor:
        def submit(self, *args, **kwargs):
            future = pool.submit(*args, **kwargs)
            futures.append(future)
            return future

    executor = RecordingExecutor()
    monkeypatch.setattr(cc, "_static_compress_slot", slot)
    monkeypatch.setattr(cc, "_static_compress_executor", executor)

    def no_network(*args, **kwargs):
        raise AssertionError("pool lifecycle tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    yield executor, slot, futures, releases
    for release in releases:
        release.set()
    for future in futures:
        try:
            future.result(timeout=3)
        except (ValueError, concurrent.futures.CancelledError):
            pass
    # Future.result can return before done callbacks: synchronize on admission.
    assert slot.acquire(timeout=3), "static admission was leaked"
    slot.release()
    pool.shutdown(wait=True)


def run(worker, **kwargs):
    return cc.run_compress_context_with_progress_timeout(
        worker=worker,
        messages=[{"role": "user", "content": "original"}],
        system_prompt_fallback="original prompt",
        idle_timeout_seconds=kwargs.pop("idle_timeout_seconds", 1),
        total_ceiling_seconds=kwargs.pop("total_ceiling_seconds", 2),
        static_only=kwargs.pop("static_only", True),
        **kwargs,
    )


def assert_reusable(slot):
    assert slot.acquire(timeout=3), "finished worker did not release admission"
    slot.release()
    expected = ([{"role": "assistant", "content": "reused"}], "new prompt")
    assert run(lambda fence: expected) == expected


@pytest.mark.parametrize("exit_kind", ["timeout", "cancel"])
def test_blocked_worker_retains_slot_until_completion(static_pool, monkeypatch, exit_kind):
    executor, slot, futures, releases = static_pool
    entered = threading.Event()
    release = threading.Event()
    releases.append(release)
    commit_attempts = []
    fence = cc.CompressionCommitFence()

    def blocked(worker_fence):
        entered.set()
        assert release.wait(3), "test failed to release blocked worker"
        admitted = worker_fence.begin_commit()
        commit_attempts.append(admitted)
        if admitted:
            worker_fence.finish_commit()
        return [], "late result must not publish"

    if exit_kind == "cancel":
        original_submit = executor.submit

        def submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            original_result = future.result

            def cancelled_result(timeout=None):
                assert entered.wait(1), "real worker never started"
                # Only the host wait is interrupted; the real worker stays alive.
                monkeypatch.setattr(future, "result", original_result)
                raise concurrent.futures.CancelledError()

            monkeypatch.setattr(future, "result", cancelled_result)
            return future

        monkeypatch.setattr(executor, "submit", submit)
        with pytest.raises(concurrent.futures.CancelledError):
            run(blocked, fence=fence)
        monkeypatch.setattr(executor, "submit", original_submit)
    else:
        timed_out = []
        result = run(
            blocked, fence=fence, idle_timeout_seconds=0.1,
            total_ceiling_seconds=0.2,
            on_timeout=lambda *args: timed_out.append(args),
        )
        assert result == ([{"role": "user", "content": "original"}], "original prompt")
        assert len(timed_out) == 1

    assert entered.is_set()
    assert fence.is_cancelled
    assert not futures[0].done()
    assert not slot.acquire(blocking=False)
    refused = threading.Event()
    assert run(lambda fence: (refused.set(), "unexpected")) == (
        [{"role": "user", "content": "original"}], "original prompt"
    )
    assert not refused.is_set()
    assert len(futures) == 1, "saturated work must not be queued"
    release.set()
    futures[0].result(timeout=3)
    assert commit_attempts == [False]
    assert_reusable(slot)


def test_submit_failure_releases_static_slot(static_pool, monkeypatch):
    executor, slot, futures, _ = static_pool
    original_submit = executor.submit

    def fail_submit(*args, **kwargs):
        raise RuntimeError("submit failed")

    monkeypatch.setattr(executor, "submit", fail_submit)
    with pytest.raises(RuntimeError, match="submit failed"):
        run(lambda fence: ([], "unused"))
    assert futures == []
    monkeypatch.setattr(executor, "submit", original_submit)
    assert_reusable(slot)


def test_worker_failure_releases_static_slot(static_pool):
    _, slot, _, _ = static_pool

    def fail_worker(fence):
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        run(fail_worker)
    assert_reusable(slot)


def test_ordinary_summary_unchanged_while_static_worker_detached(static_pool):
    _, slot, futures, releases = static_pool
    entered = threading.Event()
    release = threading.Event()
    releases.append(release)

    def blocked(fence):
        entered.set()
        assert release.wait(3)
        return [], "discarded"

    run(blocked, idle_timeout_seconds=0.1, total_ceiling_seconds=0.2)
    assert entered.is_set()
    assert not futures[0].done()
    assert not slot.acquire(blocking=False)
    with cc._compress_admission_lock:
        before = cc._compress_admitted_count
    summary = ([{"role": "assistant", "content": "ordinary summary"}], "summary prompt")
    summary_started = threading.Event()

    def ordinary(fence):
        summary_started.set()
        assert fence.begin_commit()
        fence.finish_commit()
        return summary

    assert run(ordinary, static_only=False) == summary
    assert summary_started.is_set()
    # An executor barrier ensures the ordinary worker's release callback ran.
    cc._get_compress_timeout_executor().submit(lambda: None).result(timeout=3)
    with cc._compress_admission_lock:
        assert cc._compress_admitted_count == before
    assert len(futures) == 1, "ordinary summary must not use static executor"
    release.set()
    futures[0].result(timeout=3)
    assert_reusable(slot)
