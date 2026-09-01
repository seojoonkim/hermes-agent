from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_PATH = (
    Path(__file__).parents[3]
    / "plugins"
    / "image_gen"
    / "active-model"
    / "__init__.py"
)
PNG_BYTES = b"\x89PNG\r\n\x1a\nvalid-test-image"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("active_model_image_gen", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin():
    return _load_plugin()


def test_reuses_active_model_endpoint_and_api_key_and_caches_b64(
    plugin, monkeypatch, tmp_path
):
    secret = "super-secret-active-model-key"
    config = {
        "model": {
            "provider": "custom:codex-lb",
            "base_url": "https://model.example.test/v1/",
            "api_key": secret,
        },
        "image_gen": {"provider": "active-model"},
    }
    monkeypatch.setattr(plugin, "_load_config", lambda: config)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    class FakeResponse:
        content = json.dumps(
            {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}]}
        ).encode()

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"b64_json": base64.b64encode(PNG_BYTES).decode()}]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(plugin.requests, "post", fake_post)

    result = plugin.ActiveModelImageGenProvider().generate("draw a cat", "portrait")

    assert result["success"] is True
    assert result["provider"] == "active-model"
    assert result["model"] == "gpt-image-2"
    assert Path(result["image"]).read_bytes() == PNG_BYTES
    assert captured["url"] == "https://model.example.test/v1/images/generations"
    assert captured["headers"] == {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    assert captured["json"] == {
        "model": "gpt-image-2",
        "prompt": "draw a cat",
        "size": "1024x1536",
        "n": 1,
        "quality": "medium",
    }
    assert captured["timeout"] == 300
    assert secret not in repr(result)


def test_missing_active_model_credentials_fails_closed_without_http(plugin, monkeypatch):
    monkeypatch.setattr(
        plugin,
        "_load_config",
        lambda: {
            "model": {
                "provider": "custom:codex-lb",
                "base_url": "https://model.example.test/v1",
                "api_key": "",
            }
        },
    )
    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("HTTP must not be called without active-model credentials")

    monkeypatch.setattr(plugin.requests, "post", fake_post)

    provider = plugin.ActiveModelImageGenProvider()
    result = provider.generate("draw a cat")

    assert provider.is_available() is False
    assert result["success"] is False
    assert result["error_type"] == "auth_required"
    assert called is False


def test_http_error_does_not_expose_active_model_api_key(plugin, monkeypatch):
    secret = "never-print-this-key"
    monkeypatch.setattr(
        plugin,
        "_load_config",
        lambda: {
            "model": {
                "provider": "custom:codex-lb",
                "base_url": "https://model.example.test/v1",
                "api_key": secret,
            }
        },
    )

    def fake_post(*args, **kwargs):
        raise RuntimeError(f"request failed with Authorization: Bearer {secret}")

    monkeypatch.setattr(plugin.requests, "post", fake_post)
    result = plugin.ActiveModelImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert secret not in result["error"]
    assert secret not in repr(result)


def test_rejects_response_json_over_limit_without_saving(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_MAX_RESPONSE_JSON_BYTES", 32)
    monkeypatch.setattr(
        plugin,
        "_active_model_credentials",
        lambda: ("https://model.example.test/v1", "secret-key"),
    )

    class FakeResponse:
        content = b"{" + (b" " * 32) + b"}"

        def raise_for_status(self):
            return None

        def json(self):
            raise AssertionError("oversized JSON must be rejected before parsing")

    monkeypatch.setattr(plugin.requests, "post", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        plugin,
        "save_b64_image",
        lambda *args, **kwargs: pytest.fail("oversized response must not be saved"),
    )

    result = plugin.ActiveModelImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "invalid_response"
    assert "secret-key" not in repr(result)


@pytest.mark.parametrize(
    ("b64_data", "encoded_limit", "decoded_limit"),
    [
        ("%%%not-base64%%%", 128, 128),
        (base64.b64encode(b"not-a-png").decode(), 128, 128),
        (base64.b64encode(PNG_BYTES).decode(), 8, 128),
        (base64.b64encode(PNG_BYTES).decode(), 128, 8),
    ],
    ids=["non-strict-base64", "non-png", "encoded-over-limit", "decoded-over-limit"],
)
def test_rejects_invalid_or_oversized_b64_without_saving(
    plugin, monkeypatch, b64_data, encoded_limit, decoded_limit
):
    monkeypatch.setattr(plugin, "_MAX_B64_IMAGE_BYTES", encoded_limit)
    monkeypatch.setattr(plugin, "_MAX_DECODED_IMAGE_BYTES", decoded_limit)
    monkeypatch.setattr(
        plugin,
        "_active_model_credentials",
        lambda: ("https://model.example.test/v1", "secret-key"),
    )
    body = {"data": [{"b64_json": b64_data}]}

    class FakeResponse:
        content = json.dumps(body).encode()

        def raise_for_status(self):
            return None

        def json(self):
            return body

    monkeypatch.setattr(plugin.requests, "post", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        plugin,
        "save_b64_image",
        lambda *args, **kwargs: pytest.fail("invalid image must not be saved"),
    )

    result = plugin.ActiveModelImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "invalid_response"
    assert "secret-key" not in repr(result)


def test_registers_under_explicit_provider_name(plugin):
    registered = []
    plugin.register(SimpleNamespace(register_image_gen_provider=registered.append))

    assert len(registered) == 1
    assert registered[0].name == "active-model"
    assert registered[0].default_model() == "gpt-image-2"
