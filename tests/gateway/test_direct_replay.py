from __future__ import annotations

import inspect
from unittest.mock import AsyncMock

import pytest

from gateway.direct_replay import is_exact_replay_request, resolve_direct_replay


def test_gateway_direct_replay_runs_inside_session_turn_lease():
    """Alias routing keys must serialize before replay reads or transcript writes."""
    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner._handle_message_with_agent)
    acquire_at = source.index("_lease_registry.acquire")
    replay_at = source.index("resolve_direct_replay", acquire_at)
    assert acquire_at < replay_at


@pytest.mark.parametrize(
    "text",
    [
        "전체 다시 출력해",
        "전체 다시 출력해줘",
        "직전 답변 다시 보내줘",
        "방금 답변 다시 보여줘",
        "show the full previous answer again",
    ],
)
def test_exact_replay_request_matches_standalone_requests(text):
    assert is_exact_replay_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "전체 다시 출력하되 문체를 바꿔줘",
        "이 문서 전체를 다시 출력해",
        "왜 전체 출력이 오래 걸려?",
        "",
    ],
)
def test_exact_replay_request_rejects_new_work(text):
    assert is_exact_replay_request(text) is False


@pytest.mark.asyncio
async def test_auto_reset_replay_uses_recorded_predecessor_without_search():
    db = AsyncMock()
    db.get_latest_assistant_message.return_value = {"content": "exact prior answer"}

    result = await resolve_direct_replay(
        text="전체 다시 출력해",
        current_session_id="fresh",
        previous_session_id="previous",
        session_db=db,
    )

    assert result == "exact prior answer"
    db.get_latest_assistant_message.assert_awaited_once_with("previous")


@pytest.mark.asyncio
async def test_non_replay_request_does_not_touch_database():
    db = AsyncMock()
    result = await resolve_direct_replay(
        text="전체 다시 출력하되 문체를 바꿔줘",
        current_session_id="fresh",
        previous_session_id="previous",
        session_db=db,
    )
    assert result is None
    db.get_latest_assistant_message.assert_not_awaited()
