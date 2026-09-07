# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NIM backend for chat, VLM, and image generation models."""

from typing import Any, ClassVar

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
    register_image_gen_backend,
    register_vlm_backend,
)
from world_understanding.functions.models.nim_timeout import _apply_nim_chat_timeout
from world_understanding.functions.models.token_limits import (
    ensure_model_output_token_budget,
    model_output_token_floor,
)
from world_understanding.utils.credentials import get_nim_api_key_for_base_url

_DEFAULT_NIM_MODEL = "moonshotai/kimi-k3"
_DEFAULT_TIMEOUT_SECONDS = 120.0


def _normalize_nim_output_token_kwargs(
    model_name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Normalize per-call NIM output tokens before payload construction."""
    options = dict(kwargs)
    max_completion_tokens = options.pop("max_completion_tokens", None)
    max_tokens = options.pop("max_tokens", None)
    requested = (
        max_completion_tokens if max_completion_tokens is not None else max_tokens
    )
    if requested is not None:
        options["max_tokens"] = ensure_model_output_token_budget(
            model_name,
            requested,
        )
    return options


class _NIMOutputTokenBudgetMixin:
    """Apply model output-token policy to every ChatNVIDIA request path."""

    _nim_output_token_model: ClassVar[str]

    def _prepare_inputs_and_payload(
        self,
        messages: Any,
        stop: Any = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        parent: Any = super()
        return parent._prepare_inputs_and_payload(
            messages,
            stop=stop,
            stream=stream,
            **_normalize_nim_output_token_kwargs(
                self._nim_output_token_model,
                kwargs,
            ),
        )


def _request_budgeted_chat_type(chat_type: Any, model_name: str) -> Any:
    """Compose request normalization with the lazily imported SDK class."""
    if not isinstance(chat_type, type):
        # Unit-test mocks are callable instances rather than SDK classes.
        return chat_type
    return type(
        f"_RequestBudgeted{chat_type.__name__}",
        (_NIMOutputTokenBudgetMixin, chat_type),
        {"_nim_output_token_model": model_name},
    )


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
    effective_model = model or _DEFAULT_NIM_MODEL
    max_completion_tokens = kwargs.pop("max_completion_tokens", None)
    requested_max_tokens = (
        max_completion_tokens if max_completion_tokens is not None else max_tokens
    )
    if requested_max_tokens is None:
        requested_max_tokens = model_output_token_floor(effective_model)
    if requested_max_tokens is not None:
        chat_kwargs["max_tokens"] = ensure_model_output_token_budget(
            effective_model,
            requested_max_tokens,
        )
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
    request_budgeted_chat = _request_budgeted_chat_type(ChatNVIDIA, effective_model)
    chat_model = request_budgeted_chat(
        model=effective_model,
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
