import asyncio
import pytest
from tests.gateway.test_ack_final_boundary_integration import harness,emit,settled

@pytest.mark.asyncio
@pytest.mark.parametrize('reverse',[False,True])
async def test_live_and_structured_parts_share_receipts(reverse):
 a,c,attempts=harness()
 # Gateway advertises its per-part asynchronous receipt ownership.
 a.interim_assistant_callback.receipt_aware=True
 task=asyncio.create_task(c.run())
 def structured():
  a._emit_interim_assistant_message({'role':'assistant','content':'','codex_message_items':[{'type':'message','phase':'commentary','content':[{'type':'output_text','text':s}]} for s in ['part A','part B']]})
 try:
  if reverse:structured()
  emit(a,'part A','live-codex');emit(a,'part B','live-codex')
  if not reverse:structured()
  await settled(c)
  assert [x['content'] for x in attempts]==['part A','part B']
 finally:
  c.finish();await task
