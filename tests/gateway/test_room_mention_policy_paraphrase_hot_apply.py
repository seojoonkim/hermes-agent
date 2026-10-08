"""Paraphrased owner requests hot-apply the room policy without a restart."""

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.gateway.test_current_room_mention_policy_command import _event, _runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phrase",
    [
        "너 안불러도 일할 수 있게 세팅해",
        "너 안불러도 대답하고 일하게 세팅해",
        "태그 안해도 내 말 알아듣게 해줘",
    ],
)
async def test_paraphrase_hot_applies_free_response(tmp_path, phrase):
    (tmp_path / "config.yaml").write_text("telegram: {}\n", encoding="utf-8")
    runner, adapter = _runner()
    token = set_hermes_home_override(str(tmp_path))
    try:
        reply = await runner._handle_current_room_mention_policy_command(
            _event(phrase)
        )
    finally:
        reset_hermes_home_override(token)

    assert reply is not None and "멘션 없이" in reply
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert "-10042" in saved["telegram"]["free_response_chats"]
    # Live adapter updated in-process: no restart required.
    assert "-10042" in adapter.config.extra["free_response_chats"]
