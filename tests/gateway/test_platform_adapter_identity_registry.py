"""Focused tests for profile-scoped live platform adapter resolution."""

from dataclasses import FrozenInstanceError

import pytest

from gateway.platform_registry import PlatformIdentity, PlatformRegistry


class TestPlatformIdentity:
    def test_is_immutable_and_hashable(self):
        identity = PlatformIdentity(platform="discord", account_id="work-bot")

        assert {identity: "adapter"}[identity] == "adapter"
        with pytest.raises(FrozenInstanceError):
            identity.account_id = "other-bot"

    @pytest.mark.parametrize("account_id", ["", "   ", None])
    def test_rejects_empty_account_id(self, account_id):
        with pytest.raises(ValueError, match="account_id"):
            PlatformIdentity(platform="discord", account_id=account_id)

    @pytest.mark.parametrize("platform", ["", "   ", None])
    def test_rejects_empty_platform(self, platform):
        with pytest.raises(ValueError, match="platform"):
            PlatformIdentity(platform=platform, account_id="work-bot")


class TestLivePlatformAdapterRegistry:
    def test_register_and_resolve_exact_identity(self):
        registry = PlatformRegistry()
        adapter = object()
        identity = PlatformIdentity("discord", "work-bot")

        registry.register_live_adapter("default", identity, adapter)

        assert registry.resolve_adapter("default", "discord", "work-bot") is adapter
        assert registry.resolve_adapter("default", "discord", "unknown-bot") is None
        assert registry.resolve_adapter("default", "slack", "work-bot") is None

    def test_profiles_are_separate_scopes(self):
        registry = PlatformRegistry()
        default_adapter = object()
        work_adapter = object()
        identity = PlatformIdentity("discord", "shared-account")
        registry.register_live_adapter("default", identity, default_adapter)
        registry.register_live_adapter("work", identity, work_adapter)

        assert registry.resolve_adapter("default", "discord", "shared-account") is default_adapter
        assert registry.resolve_adapter("work", "discord", "shared-account") is work_adapter
        assert registry.resolve_adapter("unknown", "discord", "shared-account") is None

    def test_accountless_resolution_requires_exactly_one_match(self):
        registry = PlatformRegistry()
        only_adapter = object()
        registry.register_live_adapter(
            "default", PlatformIdentity("discord", "only-bot"), only_adapter
        )

        assert registry.resolve_adapter("default", "discord") is only_adapter

        registry.register_live_adapter(
            "default", PlatformIdentity("discord", "second-bot"), object()
        )
        assert registry.resolve_adapter("default", "discord") is None

    def test_accountless_resolution_ignores_other_platforms_and_profiles(self):
        registry = PlatformRegistry()
        expected = object()
        registry.register_live_adapter(
            "default", PlatformIdentity("discord", "discord-bot"), expected
        )
        registry.register_live_adapter(
            "default", PlatformIdentity("slack", "slack-bot"), object()
        )
        registry.register_live_adapter(
            "work", PlatformIdentity("discord", "other-bot"), object()
        )

        assert registry.resolve_adapter("default", "discord") is expected

    def test_duplicate_live_identity_fails_without_replacing_adapter(self):
        registry = PlatformRegistry()
        identity = PlatformIdentity("discord", "work-bot")
        original = object()
        registry.register_live_adapter("default", identity, original)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_live_adapter("default", identity, object())

        assert registry.resolve_adapter("default", "discord", "work-bot") is original

    def test_unregister_is_profile_scoped_and_allows_reregistration(self):
        registry = PlatformRegistry()
        identity = PlatformIdentity("discord", "work-bot")
        default_adapter = object()
        work_adapter = object()
        registry.register_live_adapter("default", identity, default_adapter)
        registry.register_live_adapter("work", identity, work_adapter)

        assert registry.unregister_live_adapter("default", identity) is True
        assert registry.unregister_live_adapter("default", identity) is False
        assert registry.resolve_adapter("default", "discord", "work-bot") is None
        assert registry.resolve_adapter("work", "discord", "work-bot") is work_adapter

        replacement = object()
        registry.register_live_adapter("default", identity, replacement)
        assert registry.resolve_adapter("default", "discord", "work-bot") is replacement

    def test_stale_adapter_cannot_unregister_replacement(self):
        registry = PlatformRegistry()
        identity = PlatformIdentity("telegram", "123456")
        stale = object()
        replacement = object()
        registry.register_live_adapter("default", identity, stale)
        assert registry.unregister_live_adapter(
            "default", identity, expected_adapter=stale
        ) is True
        registry.register_live_adapter("default", identity, replacement)

        assert registry.unregister_live_adapter(
            "default", identity, expected_adapter=stale
        ) is False
        assert registry.resolve_adapter("default", "telegram", "123456") is replacement

    def test_factory_registry_api_remains_independent(self):
        registry = PlatformRegistry()
        live_adapter = object()
        identity = PlatformIdentity("example", "account")
        registry.register_live_adapter("default", identity, live_adapter)

        assert registry.get("example") is None
        assert registry.unregister("example") is False
        assert registry.resolve_adapter("default", "example", "account") is live_adapter
