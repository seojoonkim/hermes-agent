import pytest
from gateway.platforms.base import SendResult
from tests.gateway.test_ack_eta_delivery_integration import make_consumer,drain,ACK,receipt

@pytest.mark.asyncio
@pytest.mark.parametrize('kind',[None,'unknown','transient','too_long'])
async def test_unknown_negative_does_not_release(kind):
 c,a=make_consumer(SendResult(success=False,error='uncertain',error_kind=kind),receipt())
 c.on_commentary_receipt_aware(ACK)
 await c._send_commentary(ACK)
 c.on_commentary_receipt_aware(ACK)
 assert c._queue.qsize()==1

@pytest.mark.asyncio
async def test_definite_rejection_allows_explicit_retry():
 c,a=make_consumer(SendResult(success=False,error_kind='rate_limited'))
 c.on_commentary_receipt_aware(ACK)
 await c._send_commentary(ACK)
 c.on_commentary_receipt_aware(ACK)
 assert c._queue.qsize()==2
