"""Image generation through the active conversation model endpoint."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

API_MODEL = "gpt-image-2"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_DECODED_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_B64_IMAGE_BYTES = 44 * 1024 * 1024
_MAX_RESPONSE_JSON_BYTES = 45 * 1024 * 1024
_REQUEST_TIMEOUT_SECONDS = 300
_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        return config if isinstance(config, dict) else {}
    except Exception:
        logger.debug("Could not load active model config", exc_info=True)
        return {}


def _active_model_credentials() -> Tuple[str, str]:
    config = _load_config()
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, dict):
        return "", ""

    provider = model.get("provider")
    base_url = model.get("base_url")
    api_key = model.get("api_key")
    if not isinstance(provider, str) or not provider.strip():
        return "", ""
    if not isinstance(base_url, str) or not isinstance(api_key, str):
        return "", ""
    return base_url.strip().rstrip("/"), api_key.strip()


class ActiveModelImageGenProvider(ImageGenProvider):
    """Call ``/images/generations`` with the active model's endpoint and key."""

    @property
    def name(self) -> str:
        return "active-model"

    @property
    def display_name(self) -> str:
        return "Active conversation model endpoint"

    def is_available(self) -> bool:
        base_url, api_key = _active_model_credentials()
        return bool(base_url and api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": API_MODEL, "display": "GPT Image 2"}]

    def default_model(self) -> Optional[str]:
        return API_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "reuse",
            "tag": "Reuse model.base_url and model.api_key",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                model=API_MODEL,
                aspect_ratio=aspect,
            )

        base_url, api_key = _active_model_credentials()
        if not base_url or not api_key:
            return error_response(
                error=(
                    "The active model must configure non-empty model.provider, "
                    "model.base_url, and model.api_key."
                ),
                error_type="auth_required",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        size = _SIZES[aspect]
        payload = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": "medium",
        }
        try:
            response = requests.post(
                f"{base_url}/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except Exception as exc:
            message = str(exc).replace(api_key, "[REDACTED]")
            return error_response(
                error=f"Active model image generation failed: {message}",
                error_type="api_error",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if len(response.content) > _MAX_RESPONSE_JSON_BYTES:
            return error_response(
                error="Active model endpoint returned an oversized JSON response",
                error_type="invalid_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            body = response.json()
        except Exception:
            return error_response(
                error="Active model endpoint returned invalid JSON",
                error_type="invalid_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = body.get("data") if isinstance(body, dict) else None
        first = data[0] if isinstance(data, list) and data else None
        b64_data = first.get("b64_json") if isinstance(first, dict) else None
        if not isinstance(b64_data, str) or not b64_data:
            return error_response(
                error="Active model endpoint returned no b64_json image data",
                error_type="empty_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if len(b64_data) > _MAX_B64_IMAGE_BYTES:
            return error_response(
                error="Active model endpoint returned oversized base64 image data",
                error_type="invalid_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            image_bytes = base64.b64decode(b64_data.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError):
            return error_response(
                error="Active model endpoint returned invalid base64 image data",
                error_type="invalid_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if len(image_bytes) > _MAX_DECODED_IMAGE_BYTES:
            return error_response(
                error="Active model endpoint returned an oversized decoded image",
                error_type="invalid_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        if not image_bytes.startswith(_PNG_SIGNATURE):
            return error_response(
                error="Active model endpoint returned image data without a PNG signature",
                error_type="invalid_response",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            image_path = save_b64_image(
                base64.b64encode(image_bytes).decode("ascii"),
                prefix="active_model_gpt_image_2",
            )
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider=self.name,
                model=API_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(image_path),
            model=API_MODEL,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={"size": size, "quality": "medium"},
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ActiveModelImageGenProvider())
