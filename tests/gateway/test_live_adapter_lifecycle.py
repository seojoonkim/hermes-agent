from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platform_registry import platform_registry
from gateway.run import GatewayRunner


class _Adapter:
    platform = Platform.TELEGRAM

    def __init__(self, account_id=None):
        self.account_id = account_id
        self.cancel_background_tasks = AsyncMock()
        self.disconnect = AsyncMock()


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._active_profile_name = lambda: "default"
    runner._adapter_disconnect_timeout_secs = lambda: 0
    return runner


def test_publish_requires_connected_telegram_numeric_identity():
    runner = _runner()

    with pytest.raises(ValueError, match="numeric account identity"):
        runner._publish_live_adapter("default", Platform.TELEGRAM, _Adapter())


@pytest.mark.parametrize("bad", ["", "bot-name", "１２３", "123a"])
def test_publish_rejects_non_ascii_decimal_telegram_identity(bad):
    runner = _runner()

    with pytest.raises(ValueError, match="numeric account identity"):
        runner._publish_live_adapter("default", Platform.TELEGRAM, _Adapter(bad))


def test_publish_and_unpublish_are_profile_scoped():
    runner = _runner()
    default = _Adapter("123456")
    work = _Adapter("123456")
    runner._publish_live_adapter("default", Platform.TELEGRAM, default)
    runner._publish_live_adapter("work", Platform.TELEGRAM, work)
    try:
        assert platform_registry.resolve_adapter("default", "telegram", "123456") is default
        assert platform_registry.resolve_adapter("work", "telegram", "123456") is work

        assert runner._unpublish_live_adapter(default) is True
        assert platform_registry.resolve_adapter("default", "telegram", "123456") is None
        assert platform_registry.resolve_adapter("work", "telegram", "123456") is work
    finally:
        runner._unpublish_live_adapter(default)
        runner._unpublish_live_adapter(work)


@pytest.mark.asyncio
async def test_bounded_teardown_unpublishes_owner():
    runner = _runner()
    adapter = _Adapter("123456")
    runner._publish_live_adapter("default", Platform.TELEGRAM, adapter)

    await runner._bounded_adapter_teardown(adapter, Platform.TELEGRAM)

    assert platform_registry.resolve_adapter("default", "telegram", "123456") is None
    adapter.cancel_background_tasks.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()
