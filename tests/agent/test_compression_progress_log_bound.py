"""Actual wait loop must not emit a status at every 50ms poll."""
import concurrent.futures
from types import SimpleNamespace
from agent import conversation_compression as cc

def test_progress_logs_bounded_without_changing_wait(monkeypatch, caplog):
    clock=[0.0]
    class Future:
        def add_done_callback(self, cb): pass
        def result(self, timeout):
            clock[0]+=timeout
            if clock[0]>=65: return ([], 'done')
            raise concurrent.futures.TimeoutError()
    monkeypatch.setattr(cc, '_get_compress_timeout_executor',lambda:SimpleNamespace(submit=lambda *a:Future()))
    monkeypatch.setattr(cc, '_try_admit_compression_job',lambda:True)
    monkeypatch.setattr(cc.time,'monotonic',lambda:clock[0])
    fence=cc.CompressionCommitFence()
    monkeypatch.setattr(fence,'seconds_since_progress',lambda:0.0)
    with caplog.at_level('INFO',logger=cc.__name__):
        result=cc.run_compress_context_with_progress_timeout(worker=lambda f:([],''),messages=[],system_prompt_fallback='',idle_timeout_seconds=90,total_ceiling_seconds=120,fence=fence)
    assert result==([], 'done')
    lines=[r for r in caplog.records if 'still streaming after' in r.message]
    assert len(lines)<=2
