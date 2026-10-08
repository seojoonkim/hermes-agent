from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _event(text: str, *, allow_gateway_control: bool = True, chat_type: str = "group"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-10042",
            chat_type=chat_type,
            user_id="owner-1",
        ),
        allow_gateway_control=allow_gateway_control,
    )


def _runner():
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        config=PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "require_mention_chats": ["-999"],
                "free_response_chats": ["-10042", "-777"],
            },
        ),
        send=AsyncMock(),
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    return runner, adapter


@pytest.mark.asyncio
async def test_exact_korean_policy_persists_and_updates_live_adapter(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "telegram:\n"
        "  require_mention_chats:\n"
        "    - '-999'\n"
        "  free_response_chats:\n"
        "    - '-10042'\n"
        "    - '-777'\n"
        "display:\n"
        "  quiet: true\n",
        encoding="utf-8",
    )
    runner, adapter = _runner()
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(
            _event("이 방에서는 멘션할 때만 답해")
        )
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ 이 방에서는 이제 멘션할 때만 답할게요."
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["telegram"]["require_mention_chats"] == ["-999", "-10042"]
    assert saved["telegram"]["free_response_chats"] == ["-777"]
    assert saved["display"] == {"quiet": True}
    assert adapter.config.extra["require_mention_chats"] == ["-999", "-10042"]
    assert adapter.config.extra["free_response_chats"] == ["-777"]


@pytest.mark.asyncio
async def test_exact_english_inverse_policy_persists_and_updates_live_adapter(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "telegram:\n"
        "  require_mention: true\n"
        "  require_mention_chats: ['-10042', '-999']\n",
        encoding="utf-8",
    )
    runner, adapter = _runner()
    adapter.config.extra["require_mention_chats"] = ["-10042", "-999"]
    adapter.config.extra["free_response_chats"] = []
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(
            _event("In this room, respond without mentions")
        )
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ I’ll now respond in this room without requiring a mention."
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["telegram"]["require_mention_chats"] == ["-999"]
    assert saved["telegram"]["free_response_chats"] == ["-10042"]
    assert adapter.config.extra["require_mention_chats"] == ["-999"]
    assert adapter.config.extra["free_response_chats"] == ["-10042"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "너 안불러도 작동하게해",
        "너 안 불러도 작동하게 해",
        "너 안불러도 일하게 세팅해",
        "이 방에서 너 안불러도 일하게 세팅해",
        "에이전트 태그 안해도 응답해",
        "에이전트 태그 안 해도 응답해",
        "이 방에서는 에이전트 태그 안 해도 응답해",
        "이 채널에서는 에이전트 태그 안 해도 응답해",
        "태그 안 해도 응답해",
        "멘션 안 해도 응답해",
    ],
)
async def test_korean_no_call_wording_hot_applies_all_telegram_room_gates(
    tmp_path, phrase
):
    (tmp_path / "config.yaml").write_text(
        "telegram:\n"
        "  allowed_chats: '-999'\n"
        "  group_allowed_chats: ['-999']\n"
        "  require_mention_chats: ['-10042', '-999']\n"
        "  free_response_chats: ['-999']\n",
        encoding="utf-8",
    )
    runner, adapter = _runner()
    adapter.config.extra.update(
        {
            "allowed_chats": "-999",
            "group_allowed_chats": ["-999"],
            "require_mention_chats": ["-10042", "-999"],
            "free_response_chats": ["-999"],
        }
    )
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(
            _event(phrase)
        )
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ 이 방에서는 이제 멘션 없이도 답할게요."
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["telegram"]["allowed_chats"] == "-999,-10042"
    assert saved["telegram"]["group_allowed_chats"] == ["-999"]
    assert saved["telegram"]["require_mention_chats"] == ["-999"]
    assert saved["telegram"]["free_response_chats"] == ["-999", "-10042"]
    assert adapter.config.extra["allowed_chats"] == "-999,-10042"
    assert adapter.config.extra["group_allowed_chats"] == ["-999"]
    assert adapter.config.extra["require_mention_chats"] == ["-999"]
    assert adapter.config.extra["free_response_chats"] == ["-999", "-10042"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_allowed,live_allowed,expected",
    [
        (None, "-999", "-999,-10042"),
        ([], "-999", "-999,-10042"),
        ("", "-999", "-999,-10042"),
        (-999, -999, "-999,-10042"),
        (["-999"], ["-999"], ["-999", "-10042"]),
    ],
)
async def test_optional_policy_uses_live_allowed_gate_and_normalizes_scalar(
    tmp_path, stored_allowed, live_allowed, expected
):
    telegram = {
        "group_allowed_chats": ["-999"],
        "require_mention_chats": [],
        "free_response_chats": [],
    }
    if stored_allowed is not None:
        telegram["allowed_chats"] = stored_allowed
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"telegram": telegram}, sort_keys=False),
        encoding="utf-8",
    )
    runner, adapter = _runner()
    adapter.config.extra.update(
        {
            "allowed_chats": live_allowed,
            "group_allowed_chats": ["-999"],
            "require_mention_chats": [],
            "free_response_chats": [],
        }
    )
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(
            _event("너 안불러도 작동하게해")
        )
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ 이 방에서는 이제 멘션 없이도 답할게요."
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["telegram"]["allowed_chats"] == expected
    assert saved["telegram"]["group_allowed_chats"] == ["-999"]
    assert adapter.config.extra["allowed_chats"] == expected
    assert adapter.config.extra["group_allowed_chats"] == ["-999"]


@pytest.mark.asyncio
async def test_slack_policy_persists_and_updates_live_adapter(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "slack:\n"
        "  require_mention_channels: ['C999']\n"
        "  free_response_channels: ['C123', 'C777']\n",
        encoding="utf-8",
    )
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        config=PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "require_mention_channels": ["C999"],
                "free_response_channels": ["C123", "C777"],
            },
        )
    )
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    event = MessageEvent(
        text="Only respond when mentioned in this room",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="group",
            user_id="UOWNER",
        ),
    )
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(event)
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ I’ll now respond in this room only when mentioned."
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["slack"]["require_mention_channels"] == ["C999", "C123"]
    assert saved["slack"]["free_response_channels"] == ["C777"]
    assert adapter.config.extra["require_mention_channels"] == ["C999", "C123"]
    assert adapter.config.extra["free_response_channels"] == ["C777"]


@pytest.mark.asyncio
async def test_profile_routed_policy_writes_profile_home_not_root(tmp_path):
    root = tmp_path / "root"
    profile = tmp_path / "profiles" / "zeon"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("telegram: {}\n", encoding="utf-8")
    (profile / "config.yaml").write_text("telegram: {}\n", encoding="utf-8")
    runner, adapter = _runner()
    runner._profile_adapters = {"zeon": {Platform.TELEGRAM: adapter}}
    runner._resolve_profile_home_for_source = lambda source: profile
    event = _event("내가 태그할 때만 답해")
    event.source.profile = "zeon"
    token = set_hermes_home_override(str(root))
    try:
        reply = await runner._handle_current_room_mention_policy_command(event)
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ 이 방에서는 이제 멘션할 때만 답할게요."
    assert yaml.safe_load((root / "config.yaml").read_text()) == {"telegram": {}}
    saved = yaml.safe_load((profile / "config.yaml").read_text())
    assert saved["telegram"]["require_mention_chats"] == ["-10042"]
    assert adapter.config.extra["require_mention_chats"] == ["-10042"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform,section,require_key,free_key,chat_id",
    [
        (Platform.DISCORD, "discord", "require_mention_channels", "free_response_channels", "C-DISCORD"),
        (Platform.MATTERMOST, "mattermost", "require_mention_channels", "free_response_channels", "C-MM"),
        (Platform.MATRIX, "matrix", "require_mention_rooms", "free_response_rooms", "!room:example.org"),
    ],
)
async def test_other_shared_platform_policy_persists_and_updates_live_adapter(
    tmp_path, platform, section, require_key, free_key, chat_id
):
    (tmp_path / "config.yaml").write_text(
        f"{section}:\n  {require_key}: []\n  {free_key}: ['{chat_id}']\n",
        encoding="utf-8",
    )
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        config=PlatformConfig(
            enabled=True,
            token="***",
            extra={require_key: [], free_key: [chat_id]},
        )
    )
    runner.adapters = {platform: adapter}
    runner._profile_adapters = {}
    event = MessageEvent(
        text="내가 태그하기 전에는 말하지 마",
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type="group",
            user_id="owner-1",
        ),
    )
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(event)
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ 이 방에서는 이제 멘션할 때만 답할게요."
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved[section][require_key] == [chat_id]
    assert saved[section][free_key] == []
    assert adapter.config.extra[require_key] == [chat_id]
    assert adapter.config.extra[free_key] == []
    if platform == Platform.MATRIX:
        assert adapter._require_mention_rooms == {chat_id}
        assert adapter._free_rooms == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "내가 태그하기 전에는 말하지 마",
        "내가 태그하기 전까지 말하지 마",
        "이 방에서는 내가 태그할 때만 답해",
    ],
)
async def test_tag_wording_enables_required_policy(tmp_path, phrase):
    (tmp_path / "config.yaml").write_text("telegram: {}\n", encoding="utf-8")
    runner, adapter = _runner()
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(
            _event(phrase)
        )
    finally:
        reset_hermes_home_override(token)

    assert reply == "✓ 이 방에서는 이제 멘션할 때만 답할게요."
    assert adapter.config.extra["require_mention_chats"] == ["-10042"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        # Short Korean paraphrases are now accepted (fuzzy matcher); long
        # task messages that merely mention the topic must still pass through.
        _event("멘션할 때만 답하게 바꾼 거 어떻게 된 건지 로그 확인하고 원인 정리해줘"),
        _event("Only respond when mentioned in this room please"),
        _event("이 방에서는 멘션할 때만 답해", allow_gateway_control=False),
        _event("이 방에서는 멘션할 때만 답해", chat_type="dm"),
        MessageEvent(
            text="이 방에서는 멘션할 때만 답해",
            source=SessionSource(
                platform=Platform.WHATSAPP,
                chat_id="room-1",
                chat_type="group",
                user_id="owner-1",
            ),
        ),
    ],
)
async def test_policy_command_rejects_near_matches_and_ineligible_events(tmp_path, event):
    (tmp_path / "config.yaml").write_text("telegram: {}\n", encoding="utf-8")
    runner, adapter = _runner()
    before = dict(adapter.config.extra)
    token = set_hermes_home_override(str(tmp_path))
    try:
        assert await runner._handle_current_room_mention_policy_command(event) is None
    finally:
        reset_hermes_home_override(token)

    assert adapter.config.extra == before
    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) == {
        "telegram": {}
    }
