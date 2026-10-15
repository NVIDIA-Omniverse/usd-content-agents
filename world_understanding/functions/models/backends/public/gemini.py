# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemini backend for chat, VLM, and image generation."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
    register_image_gen_backend,
    register_vlm_backend,
)
from world_understanding.functions.models.request_timeout import (
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
)
from world_understanding.utils.credentials import get_env_api_key_for_backend

_DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
# Shared hosted-model deadline; WU_MODEL_REQUEST_TIMEOUT_SECONDS.
_DEFAULT_TIMEOUT_SECONDS = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS


def create_gemini_chat(
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    timeout: float | None = _DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> BaseChatModel:
    """Create Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Remove kwargs not applicable to Gemini
    kwargs.pop("api_version", None)

    api_key = get_env_api_key_for_backend("gemini", api_key)
    if not api_key:
        raise ValueError(
            "API key is required. Provide via api_key parameter or "
            "GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )

    chat_kwargs: dict[str, Any] = {}
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    if top_p is not None:
        chat_kwargs["top_p"] = top_p
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens
    chat_kwargs.update(kwargs)

    return ChatGoogleGenerativeAI(
        model=model or _DEFAULT_GEMINI_MODEL,
        google_api_key=api_key,
        streaming=streaming,
        timeout=timeout,
        **chat_kwargs,
    )


def create_gemini_vlm(api_key: str | None = None, **kwargs: Any) -> Any:
    """Create Gemini VLM."""
    from world_understanding.functions.models.vision_language_models import GeminiVLM

    return GeminiVLM(api_key=api_key, **kwargs)


def create_gemini_image_gen(**kwargs: Any) -> Any:
    """Create Gemini image generation model."""
    from world_understanding.functions.models.image_generation_models import (
        GeminiImageGenerationModel,
    )

    return GeminiImageGenerationModel(**kwargs)


register_chat_backend("gemini", create_gemini_chat)
register_vlm_backend("gemini", create_gemini_vlm)
register_image_gen_backend("gemini", create_gemini_image_gen)
