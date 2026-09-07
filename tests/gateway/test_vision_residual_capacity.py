import asyncio
import threading
import pytest
from gateway.run import GatewayRunner

@pytest.mark.asyncio
async def test_timed_out_analysis_retains_capacity_until_thread_finishes(monkeypatch):
 import gateway.run as g
 import tools.vision_tools as v
 monkeypatch.setattr(g,'_VISION_BATCH_BUDGET_SECONDS',0.01)
 monkeypatch.setattr(g,'_vision_enrichment_slots',threading.BoundedSemaphore(1),raising=False)
 release=threading.Event(); entered=threading.Event(); calls=[]
 def blocking():
  entered.set(); release.wait(2)
 async def vision(**kw):
  calls.append(kw['image_url'])
  await asyncio.to_thread(blocking)
  return '{"success":true,"analysis":"done"}'
 monkeypatch.setattr(v,'vision_analyze_tool',vision)
 runner=object.__new__(GatewayRunner)
 try:
  a=await runner._enrich_message_with_vision('A',['first'])
  assert entered.is_set()
  b=await runner._enrich_message_with_vision('B',['second'])
  assert calls==['first']
  assert 'first' in a and 'second' in b
 finally:
  release.set()
  await asyncio.sleep(0.05)
 assert g._vision_enrichment_slots.acquire(blocking=False)
 g._vision_enrichment_slots.release()
