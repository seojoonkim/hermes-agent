import asyncio
import pytest
from gateway.run import GatewayRunner

@pytest.mark.asyncio
async def test_slow_vision_batch_preserves_all_paths_and_caption(monkeypatch):
    import gateway.run as g
    import tools.vision_tools as v
    calls=[]
    async def slow(**kw):
        calls.append(kw['image_url'])
        await asyncio.sleep(1)
        return '{"success":true,"analysis":"late"}'
    monkeypatch.setattr(v,'vision_analyze_tool',slow)
    monkeypatch.setattr(g,'_VISION_BATCH_BUDGET_SECONDS',0.01,raising=False)
    runner=object.__new__(GatewayRunner)
    result=await asyncio.wait_for(runner._enrich_message_with_vision('KEEP CAPTION',['/tmp/a.jpg','/tmp/b.jpg']),0.2)
    assert 'KEEP CAPTION' in result
    assert '/tmp/a.jpg' in result and '/tmp/b.jpg' in result
    assert 'late' not in result
    assert calls==['/tmp/a.jpg']
