import ast
import inspect
import textwrap
import weakref
from types import SimpleNamespace
from typing import Any, cast

from gateway.config import Platform
from gateway.platform_registry import PlatformIdentity, platform_registry
from gateway.run import GatewayRunner
from gateway.slash_commands import GatewaySlashCommandsMixin


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


def _source(
    account_id: str | None, *, platform=Platform.TELEGRAM, transport=None
) -> Any:
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


def test_missing_telegram_account_does_not_fall_back_to_platform_adapter():
    runner = _runner()

    assert runner._adapter_for_source(_source(None)) is None


def test_invalid_telegram_account_does_not_fall_back_to_platform_adapter():
    runner = _runner()

    assert runner._adapter_for_source(_source("１２３")) is None


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


def test_runtime_routed_profile_can_use_registered_transport_with_exact_account():
    runner = _runner()
    transport = runner.adapters[Platform.TELEGRAM]
    identity = PlatformIdentity("telegram", "123456")
    platform_registry.register_live_adapter("default", identity, transport)
    setattr(transport, "_platform_registry_binding", ("default", identity))
    source = _source("123456", transport=transport)
    source.profile = "runtime-route"
    try:
        assert runner._adapter_for_source(source) is transport
    finally:
        platform_registry.unregister_live_adapter("default", identity)


def test_runtime_routed_transport_still_fails_closed_on_account_mismatch():
    runner = _runner()
    transport = runner.adapters[Platform.TELEGRAM]
    identity = PlatformIdentity("telegram", "123456")
    platform_registry.register_live_adapter("default", identity, transport)
    setattr(transport, "_platform_registry_binding", ("default", identity))
    source = _source("999999", transport=transport)
    source.profile = "runtime-route"
    try:
        assert runner._adapter_for_source(source) is None
    finally:
        platform_registry.unregister_live_adapter("default", identity)


def test_non_telegram_account_id_keeps_existing_transport_routing():
    runner = _runner()
    transport = _WeakrefableTransport()
    runner.adapters[Platform.DISCORD] = cast(Any, transport)

    assert runner._adapter_for_source(
        _source("work-bot", platform=Platform.DISCORD, transport=transport)
    ) is transport


def test_source_sensitive_slash_handlers_do_not_read_primary_adapter_map():
    """Account-scoped command paths must use the authoritative source resolver."""
    for method_name in (
        "_handle_status_command",
        "_handle_goal_command",
        "_handle_voice_command",
        "_handle_approve_command",
        "_handle_deny_command",
    ):
        method = getattr(GatewaySlashCommandsMixin, method_name)
        source = textwrap.dedent(inspect.getsource(method))
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_adapter_for_source"
        ]
        assert calls, f"{method_name} must resolve adapters from the message source"
        assert "self.adapters.get(source.platform)" not in source
        assert "self.adapters.get(event.source.platform)" not in source


def test_session_env_uses_source_adapter_async_delivery_capability(monkeypatch):
    runner = _runner()
    source = _source("123456")
    for name, value in {
        "chat_id": "chat",
        "chat_type": None,
        "chat_name": None,
        "thread_id": None,
        "user_id": None,
        "user_name": None,
        "message_id": None,
        "profile": "default",
    }.items():
        setattr(source, name, value)
    adapter = SimpleNamespace(supports_async_delivery=False)
    monkeypatch.setattr(runner, "_adapter_for_source", lambda value: adapter)

    captured = {}
    monkeypatch.setattr(
        "gateway.session_context.set_session_vars",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    runner._set_session_env(cast(Any, SimpleNamespace(source=source, session_key="key")))

    assert captured["async_delivery"] is False


def test_voice_reply_uses_source_adapter_auto_tts(monkeypatch):
    from gateway.platforms.base import MessageType

    runner = _runner()
    runner._voice_mode = {}
    runner._voice_key = (
        lambda platform, chat_id, account_id=None:
        f"{platform.value}:{account_id}:{chat_id}"
        if account_id else f"{platform.value}:{chat_id}"
    )
    adapter = SimpleNamespace(_should_auto_tts_for_chat=lambda chat_id: True)
    monkeypatch.setattr(runner, "_adapter_for_source", lambda source: adapter)
    source = _source("123456")
    source.chat_id = "chat"
    event = SimpleNamespace(source=source, message_type=MessageType.TEXT)

    assert runner._should_send_voice_reply(cast(Any, event), "reply", []) is True
