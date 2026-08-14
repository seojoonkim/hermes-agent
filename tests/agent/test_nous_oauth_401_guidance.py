"""Tests for the Nous OAuth 401 actionable-guidance branch in
``agent.conversation_loop.run_conversation``.

Source-inspection style (matches ``test_gemini_fast_fallback.py``): we assert
that the guidance strings exist in the function body so that the user-facing
hint cannot be silently removed by a future refactor.

Regression context: ashh hit a Nous 401 (OAuth token expired / portal said
account out of credits) plus a model slug ``deepseek/deepseek-v4-flash:free``
that's OpenRouter syntax, not a Nous catalog name. The previous guidance
branch only covered ``openai-codex`` and ``xai-oauth``; ``nous`` fell through
to a generic "Your API key was rejected... run hermes setup" message, which is
the wrong advice for a pure-OAuth provider.
"""
from __future__ import annotations

import inspect

from agent import conversation_loop


def test_nous_provider_has_dedicated_401_recovery_gate():
    """Nous 401s must use the dedicated Portal credential refresh path."""
    source = inspect.getsource(conversation_loop._run_conversation_impl)

    assert 'agent.provider == "nous"' in source
    assert "status_code == 401" in source
    assert "_retry.nous_auth_retry_attempted" in source
    assert "agent._try_refresh_nous_client_credentials(force=True)" in source


def test_nous_401_guidance_strings_present():
    """User-facing remediation strings for Nous OAuth 401s must exist."""
    source = inspect.getsource(conversation_loop._run_conversation_impl)

    # Must tell the user it's an OAuth token problem, NOT an API key problem
    # (Nous Portal has no API key path — auth_type=oauth_device_code only).
    assert "Nous 401 — Portal authentication failed." in source

    # Must give a concrete re-auth command, not a generic "hermes setup".
    assert "hermes auth add nous" in source

    # Must point at the portal so users can check account/credit status.
    assert "portal.nousresearch.com" in source


