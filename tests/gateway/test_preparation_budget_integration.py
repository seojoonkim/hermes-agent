import asyncio
import pytest
from agent.preparation_budget import preparation_budget
from gateway.run import GatewayRunner

@pytest.mark.asyncio
async def test_prior_preparation_consumes_vision_budget(monkeypatch):
 import tools.vision_tools as v
 calls=[]
 async def slow(**kw):
  calls.append(kw['image_url'])
  await asyncio.sleep(1)
 monkeypatch.setattr(v,'vision_analyze_tool',slow)
 with preparation_budget(0.02):
  await asyncio.sleep(0.03)
  result=await asyncio.wait_for(GatewayRunner._enrich_message_with_vision(object.__new__(GatewayRunner),'caption',['a','b']),0.1)
 assert 'caption' in result and 'a' in result and 'b' in result
 assert len(calls)<=1

def test_budget_reaches_compression_worker_context():
 from tools.thread_context import propagate_context_to_thread
 from concurrent.futures import ThreadPoolExecutor
 from agent.preparation_budget import remaining_preparation_seconds
 with preparation_budget(0.2):
  with ThreadPoolExecutor(max_workers=1) as pool:
   value=pool.submit(propagate_context_to_thread(lambda:remaining_preparation_seconds(120))).result(timeout=1)
 assert 0<value<=0.2
