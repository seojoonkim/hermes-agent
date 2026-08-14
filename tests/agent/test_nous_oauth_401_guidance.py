"""Runtime contracts for Nous Portal OAuth recovery after HTTP 401."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


PORTAL_URL = "https://inference-api.nousresearch.com/v1"


def _unauthorized() -> Exception:
    error = Exception("Error code: 401 - Portal token rejected")
    error.status_code = 401
    error.body = {"error": "invalid agent key"}
    return error


def _response(content: str = "Recovered") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="nous/test-model", usage=None)


@pytest.fixture
def nous_agent() -> AIAgent:
    """A real conversation-loop agent with only its network boundary mocked."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="portal-agent-key",
            base_url=PORTAL_URL,
            provider="nous",
            api_mode="chat_completions",
            model="nous/test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run(agent: AIAgent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_recover_with_credential_pool", return_value=(False, False)),
        patch("agent.conversation_loop._print_nous_entitlement_guidance", return_value=False),
    ):
        return agent.run_conversation("hello")


def test_nous_401_forces_refresh_then_retries_request_once(nous_agent, capsys):
    nous_agent.client.chat.completions.create.side_effect = [
        _unauthorized(),
        _response(),
    ]
    nous_agent._try_refresh_nous_client_credentials = MagicMock(return_value=True)

    result = _run(nous_agent)

    nous_agent._try_refresh_nous_client_credentials.assert_called_once_with(force=True)
    assert nous_agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "Recovered"
    assert "hermes auth add nous" not in capsys.readouterr().out


def test_persistent_nous_401_does_not_refresh_or_retry_more_than_once(
    nous_agent, capsys
):
    nous_agent.client.chat.completions.create.side_effect = [
        _unauthorized(),
        _unauthorized(),
    ]
    nous_agent._try_refresh_nous_client_credentials = MagicMock(return_value=True)

    result = _run(nous_agent)

    nous_agent._try_refresh_nous_client_credentials.assert_called_once_with(force=True)
    assert nous_agent.client.chat.completions.create.call_count == 2
    assert result["failed"] is True
    output = capsys.readouterr().out
    assert "Nous 401" in output
    assert "portal.nousresearch.com" in output
    assert "hermes auth add nous" in output


def test_failed_nous_refresh_emits_actionable_portal_guidance(nous_agent, capsys):
    nous_agent.client.chat.completions.create.side_effect = _unauthorized()
    nous_agent._try_refresh_nous_client_credentials = MagicMock(return_value=False)

    result = _run(nous_agent)

    nous_agent._try_refresh_nous_client_credentials.assert_called_once_with(force=True)
    assert nous_agent.client.chat.completions.create.call_count == 1
    assert result["failed"] is True
    output = capsys.readouterr().out
    assert "Nous 401" in output
    assert "portal.nousresearch.com" in output
    assert "hermes auth add nous" in output
    assert "hermes setup" not in output
