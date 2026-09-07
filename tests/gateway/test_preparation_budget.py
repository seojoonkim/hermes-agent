from agent.preparation_budget import preparation_budget, remaining_preparation_seconds

def test_budget_shared_and_scope_restored(monkeypatch):
 import agent.preparation_budget as b
 clock=[10.0];monkeypatch.setattr(b.time,'monotonic',lambda:clock[0])
 assert remaining_preparation_seconds(30)==30
 with preparation_budget(60):
  assert remaining_preparation_seconds(120)==60
  clock[0]+=40
  assert remaining_preparation_seconds(30)==20
  clock[0]+=21
  assert remaining_preparation_seconds(30)==0.001
 assert remaining_preparation_seconds(30)==30
