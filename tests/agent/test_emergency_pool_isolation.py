"""Built-in static compaction remains possible with summary capacity full."""
from agent import conversation_compression as cc
from hermes_state import SessionDB
from tests.agent.test_compression_concurrent_fork import _build_agent_with_db


def test_real_static_compaction_with_summary_slot_full(tmp_path,monkeypatch):
    db=SessionDB(db_path=tmp_path/'state.db')
    db.create_session('static-test',source='test')
    agent=_build_agent_with_db(db,'static-test',stub_compressor=False)
    agent.compression_in_place=False
    agent._compression_feasibility_checked=True
    comp=agent.context_compressor
    comp.record_timeout_failure('summary pool saturated')
    comp.protect_first_n=1
    comp.protect_last_n=2
    messages=[{'role':'user' if i%2==0 else 'assistant','content':f'MARKER_{i} '+('detail '*1000)} for i in range(30)]
    messages.append({'role':'user','content':'LATEST_REQUEST_KEEP'})
    monkeypatch.setattr(cc,'_try_admit_compression_job',lambda:False)
    for message in messages:
        db.append_message('static-test',message['role'],message['content'])
    from agent.context_compressor import estimate_messages_tokens_rough
    import socket
    def no_network(*args, **kwargs):
        raise AssertionError('static compaction must not call a provider')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    before_tokens=estimate_messages_tokens_rough(messages)
    before=repr(messages)
    try:
        result,_=agent._compress_context(messages,'sys',approx_tokens=300000,emergency_fallback=True)
        assert len(repr(result))<len(before)
        assert estimate_messages_tokens_rough(result) < before_tokens
        assert 'LATEST_REQUEST_KEEP' in repr(result)
        assert repr(messages)==before
        assert db.get_session('static-test') is not None
        archived=db.get_messages('static-test',include_inactive=True,include_compacted=True)
        # Rotation may persist a final snapshot as well; the original rows
        # must remain intact and in order, not merely leave a parent shell.
        assert [x['content'] for x in archived[:len(messages)]] == [x['content'] for x in messages]
    finally:
        agent.close()
        db.close()
