import logging
import asyncio
import pytest
from gateway.latency import measure_agent_stage

@pytest.mark.parametrize('error',[None,ValueError,asyncio.CancelledError])
def test_all_outcomes_have_receipt(caplog,error):
 with caplog.at_level(logging.INFO,logger='gateway.latency'):
  try:
   with measure_agent_stage('test',1,0):
    if error: raise error()
  except BaseException: pass
 assert len(caplog.records)==1
 text=caplog.records[0].getMessage()
 assert ('outcome=ok' if error is None else 'outcome=cancelled' if error is asyncio.CancelledError else 'outcome=error') in text
