from types import SimpleNamespace as N
import pytest
from agent import auxiliary_client as aux

@pytest.mark.parametrize('delta,finish,expected', [
    (None,None,0), (N(content='',role='assistant'),None,0),
    (N(content='hello'),None,1), (N(reasoning_content='thinking'),None,1),
    (N(tool_calls=[N(index=0,function=N(name='tool',arguments=''))]),None,1),
    (None,'stop',1),
])
def test_aux_progress_requires_payload(monkeypatch, delta, finish, expected):
    calls=[]
    monkeypatch.setattr(aux,'_notify_aux_progress',lambda: calls.append(1))
    acc=aux._ChatStreamAccumulator()
    acc.feed(N(choices=[N(delta=delta,finish_reason=finish)]))
    assert len(calls)==expected

def test_usage_only_does_not_refresh(monkeypatch):
    calls=[]
    monkeypatch.setattr(aux,'_notify_aux_progress',lambda: calls.append(1))
    acc=aux._ChatStreamAccumulator()
    acc.feed(N(choices=[],usage=N(total_tokens=5)))
    assert not calls
    assert acc.finish().usage.total_tokens==5
