import json
from unittest.mock import Mock
from tests.tools.test_delegate_apiserver_background import _patch_delegate, _fake_parent, _clean_queue_and_context
from gateway.session_context import set_session_vars

def test_capacity_never_runs_background_request_inline(monkeypatch):
 dt=_patch_delegate(monkeypatch)
 children=[]
 original=dt._build_child_agent
 def build(**kwargs):
  child=original(**kwargs); children.append(child); return child
 monkeypatch.setattr(dt,'_build_child_agent',build)
 run=Mock(side_effect=AssertionError('background request ran inline'))
 monkeypatch.setattr(dt,'_run_single_child',run)
 monkeypatch.setattr('tools.async_delegation.dispatch_async_delegation_batch',lambda **kw: {'status':'rejected','error':'capacity reached'})
 set_session_vars(platform='telegram',chat_id='123',session_key='test',session_id='test',async_delivery=True)
 result=json.loads(dt.delegate_task(goal='task',context='ctx',background=True,parent_agent=_fake_parent()))
 assert result.get('status')=='rejected'
 run.assert_not_called()
 assert result['started'] is False
 assert children
 for child in children:
  child.close.assert_called_once()
