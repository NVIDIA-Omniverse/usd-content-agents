# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anthropic backend for chat and VLM models."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
    register_vlm_backend,
)
from world_understanding.functions.models.request_timeout import (
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
)

_DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
# Shared hosted-model deadline; WU_MODEL_REQUEST_TIMEOUT_SECONDS.
_DEFAULT_TIMEOUT_SECONDS = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS


def create_anthropic_chat(
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    timeout: float | None = _DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Anthropic chat model."""
    from langchain_anthropic import ChatAnthropic

    if not api_key:
        raise ValueError("API key is required for Anthropic backend")

    # Remove kwargs not applicable to Anthropic
    kwargs.pop("api_version", None)

    chat_kwargs: dict[str, Any] = {}
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    if top_p is not None:
        chat_kwargs["top_p"] = top_p
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens
    chat_kwargs.update(kwargs)

    return ChatAnthropic(
        model_name=model or _DEFAULT_ANTHROPIC_MODEL,
        api_key=api_key,  # type: ignore[arg-type]
        streaming=streaming,
        timeout=timeout,
        **chat_kwargs,
    )


def create_anthropic_vlm(api_key: str | None = None, **kwargs: Any) -> Any:
    """Create Anthropic VLM."""
    from world_understanding.functions.models.vision_language_models import (
        AnthropicVLM,
    )

    if not api_key:
        raise ValueError("API key is required for Anthropic backend")
    return AnthropicVLM(api_key=api_key, **kwargs)


register_chat_backend("anthropic", create_anthropic_chat)
register_vlm_backend("anthropic", create_anthropic_vlm)
