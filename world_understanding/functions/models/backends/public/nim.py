# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NIM backend for chat, VLM, and image generation models."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
    register_image_gen_backend,
    register_vlm_backend,
)
from world_understanding.functions.models.nim_timeout import _apply_nim_chat_timeout
from world_understanding.utils.credentials import get_nim_api_key_for_base_url

_DEFAULT_NIM_MODEL = "google/gemma-4-31b-it"
_DEFAULT_TIMEOUT_SECONDS = 120.0


def create_nim_chat(
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    timeout: float | None = _DEFAULT_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> BaseChatModel:
    """Create NVIDIA NIM chat model."""
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    api_key = get_nim_api_key_for_base_url(kwargs.get("base_url"), api_key)
    if not api_key:
        raise ValueError("API key is required for NIM backend")

    chat_kwargs: dict[str, Any] = {}
    if temperature is not None:
        chat_kwargs["temperature"] = temperature
    if top_p is not None:
        chat_kwargs["top_p"] = top_p
    if max_tokens is not None:
        chat_kwargs["max_tokens"] = max_tokens
    # api_version and other stray keys are not valid ChatNVIDIA ctor params;
    # langchain would otherwise push them into model_kwargs and they would
    # be serialized as body fields. Strict NIM serving (e.g. Nemotron Nano
    # 8B) rejects unknown body fields with 400 extra_forbidden. Drop them.
    kwargs.pop("api_version", None)
    chat_kwargs.update(kwargs)

    # `timeout` and `streaming` are intentionally omitted from the
    # ChatNVIDIA constructor: they are not declared ctor fields in the
    # installed langchain_nvidia_ai_endpoints version, so langchain pushes
    # them to model_kwargs which are serialized as body fields. Strict NIM
    # serving (e.g. Nemotron Nano 8B) rejects unknown body fields with
    # "400 extra_forbidden". Timeout is applied to the HTTP clients after
    # construction. Streaming is only needed when the caller asks for it;
    # pass it through only then.
    ctor_kwargs: dict[str, Any] = {}
    if streaming:
        ctor_kwargs["streaming"] = True
    chat_model = ChatNVIDIA(
        model=model or _DEFAULT_NIM_MODEL,
        nvidia_api_key=api_key,
        **ctor_kwargs,
        **chat_kwargs,
    )
    _apply_nim_chat_timeout(chat_model, timeout, label="create_nim_chat")
    return chat_model


def create_nim_vlm(api_key: str | None = None, **kwargs: Any) -> Any:
    """Create NVIDIA NIM VLM."""
    from world_understanding.functions.models.vision_language_models import (
        NvidiaNIMVLM,
    )

    api_key = get_nim_api_key_for_base_url(kwargs.get("base_url"), api_key)
    if not api_key:
        raise ValueError("API key is required for NIM backend")
    return NvidiaNIMVLM(api_key=api_key, **kwargs)


def create_nim_image_gen(api_key: str | None = None, **kwargs: Any) -> Any:
    """Create NIM image generation model."""
    from world_understanding.functions.models.image_generation_models import (
        NIMImageGenerationModel,
    )

    api_key = get_nim_api_key_for_base_url(kwargs.get("base_url"), api_key)
    if not api_key:
        raise ValueError(
            "API key is required. Provide via api_key parameter or "
            "NVIDIA_API_KEY environment variable."
        )
    return NIMImageGenerationModel(api_key=api_key, **kwargs)


register_chat_backend("nim", create_nim_chat)
register_vlm_backend("nim", create_nim_vlm)
register_image_gen_backend("nim", create_nim_image_gen)
