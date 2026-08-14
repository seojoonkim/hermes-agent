import weakref
from types import SimpleNamespace
from typing import Any, cast

from gateway.config import Platform
from gateway.platform_registry import PlatformIdentity, platform_registry
from gateway.run import GatewayRunner


class _WeakrefableTransport:
    pass


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {
        Platform.TELEGRAM: cast(Any, _WeakrefableTransport())
    }
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    return runner


def _source(account_id: str, *, platform=Platform.TELEGRAM, transport=None):
    return SimpleNamespace(
        platform=platform,
        profile=None,
        account_id=account_id,
        delivered_via_upstream_relay=False,
        _transport_adapter_ref=weakref.ref(transport) if transport is not None else None,
    )


def test_exact_account_miss_does_not_fall_back_to_platform_adapter():
    runner = _runner()
    registered = object()
    identity = PlatformIdentity("telegram", "123456")
    platform_registry.register_live_adapter("default", identity, registered)
    try:
        assert runner._adapter_for_source(_source("999999")) is None
    finally:
        platform_registry.unregister_live_adapter("default", identity)


def test_exact_account_resolves_registered_adapter():
    runner = _runner()
    expected = object()
    identity = PlatformIdentity("telegram", "123456")
    platform_registry.register_live_adapter("default", identity, expected)
    try:
        assert runner._adapter_for_source(_source("123456")) is expected
    finally:
        platform_registry.unregister_live_adapter("default", identity)


def test_telegram_transport_reference_cannot_bypass_exact_account_identity():
    runner = _runner()
    transport = runner.adapters[Platform.TELEGRAM]

    assert runner._adapter_for_source(_source("999999", transport=transport)) is None


def test_non_telegram_account_id_keeps_existing_transport_routing():
    runner = _runner()
    transport = _WeakrefableTransport()
    runner.adapters[Platform.DISCORD] = cast(Any, transport)

    assert runner._adapter_for_source(
        _source("work-bot", platform=Platform.DISCORD, transport=transport)
    ) is transport
