# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chat model implementations using LangChain."""

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from world_understanding.telemetry import traced_llm
from world_understanding.utils.credentials import (
    apply_llm_nim_env_override,
    get_env_api_key_for_backend,
    get_llm_nim_env_base_url_override,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    resolve_endpoint_api_key,
)

# Default configurations
_DEFAULT_NIM_MODEL = "qwen/qwen3.5-397b-a17b"


class EchoChatModel(BaseChatModel):
    """Simple echo chat model for testing.

    This model simply echoes back the user's input with an optional prefix.
    """

    prefix: str = "Echo: "
    last_request: dict[str, Any] | None = None

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        """Generate response by echoing the last message."""
        # Store request for testing
        self.last_request = {"messages": messages, "stop": stop, **kwargs}

        # Get the last message content
        last_message = messages[-1] if messages else None
        content = ""

        if last_message:
            if hasattr(last_message, "content"):
                content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                content = last_message["content"]

        response = self.prefix + content

        # Return in the format expected by LangChain
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> Any:
        """Async version of generate (just calls sync version)."""
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return the type of language model."""
        return "echo"


def create_echo_chat_model(prefix: str = "Echo: ", **kwargs: Any) -> EchoChatModel:
    """Create echo chat model for testing."""
    return EchoChatModel(prefix=prefix)


def create_nim_chat_model(
    api_key: str,
    model: str = _DEFAULT_NIM_MODEL,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Create NVIDIA NIM chat model.

    Compatibility wrapper around the canonical public NIM implementation.

    Runtime backend selection belongs to :func:`create_chat_model`. Delegating
    directly here keeps this historically registrable wrapper from resolving
    itself recursively when installed as the ``nim`` factory.
    """
    from world_understanding.functions.models.backends.public.nim import (
        create_nim_chat,
    )

    return create_nim_chat(
        api_key=api_key,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        streaming=streaming,
        **kwargs,
    )


@traced_llm(name="chat_model.create", system="langchain", operation="create")
def create_chat_model(
    backend: str,
    api_key: str | None = None,
    api_version: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model for the specified backend.

    Available backends depend on the installation. Public providers are always
    available and optional packages contribute factories through entry points.

    Args:
        backend: Backend name (use ``list_chat_backends()`` to see available)
        api_key: API key for the backend
        api_version: API version (backend-specific)
        model: Model ID (optional, defaults used if not specified)
        temperature: Controls randomness (0.0-1.0)
        top_p: Controls diversity via nucleus sampling (0.0-1.0)
        max_tokens: Maximum tokens in the generated response
        streaming: Whether to stream responses
        **kwargs: Additional backend-specific arguments, such as timeout

    Returns:
        Configured chat model instance

    Raises:
        ValueError: If backend is not supported
    """
    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.functions.models.backends.registry import (
        get_chat_factory,
    )

    factory = get_chat_factory(backend)
    return factory(
        api_key=api_key,
        api_version=api_version,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        streaming=streaming,
        **kwargs,
    )


_logger = logging.getLogger(__name__)


def create_chat_model_from_config(
    llm_config: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> BaseChatModel | None:
    """Create a chat model from a config dict.

    Centralises the boilerplate of extracting backend/model/temperature/etc.
    from an ``llm_config`` dict and calling :func:`create_chat_model`.

    Args:
        llm_config: Dict with keys ``backend``, ``model``, ``temperature``,
            ``max_tokens``, and optionally ``api_key``.
        defaults: Optional fallback values for missing keys.

    Returns:
        A ``BaseChatModel`` instance, or ``None`` when the API key cannot
        be resolved.
    """
    d = defaults or {}
    original_backend = llm_config.get("backend", d.get("backend", "nim"))

    # Air-gapped override: *_LLM_NIM_BASE_URL (preferred) or the VLM variant
    # *_VLM_NIM_BASE_URL (fallback, routes both VLM and LLM through one local
    # NIM endpoint). When set, force backend=nim + base_url and drop any stale
    # endpoint-scoped fields from the prior backend so a hosted provider key
    # cannot be forwarded to the local sidecar.
    if get_llm_nim_env_base_url_override() and original_backend != "nim":
        _logger.info(
            "*_LLM_NIM_BASE_URL/*_VLM_NIM_BASE_URL set — overriding LLM "
            "backend from '%s' to 'nim'",
            original_backend,
        )
    llm_config = apply_llm_nim_env_override(llm_config)

    backend = llm_config.get("backend", d.get("backend", "nim"))
    model = llm_config.get("model", d.get("model"))
    temperature = llm_config.get("temperature", d.get("temperature", 0.1))
    max_tokens = llm_config.get("max_tokens", d.get("max_tokens", 1024))
    timeout = llm_config.get("timeout", d.get("timeout"))
    base_url = llm_config.get("base_url")

    kwargs: dict[str, Any] = {
        "backend": backend,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    reserved_keys = {
        "api_key",
        "api_key_env",
        "backend",
        "base_url",
        "max_tokens",
        "model",
        "temperature",
        "timeout",
    }
    kwargs.update(
        {key: value for key, value in llm_config.items() if key not in reserved_keys}
    )

    import world_understanding.functions.models.backends  # noqa: F401
    from world_understanding.functions.models.backends.registry import (
        chat_backend_requires_api_key,
    )

    if chat_backend_requires_api_key(backend):
        explicit_api_key = resolve_endpoint_api_key(
            llm_config.get("api_key"),
            llm_config.get("api_key_env"),
        )
        if backend == "nim":
            api_key = get_nim_api_key_for_base_url(base_url, explicit_api_key)
        elif backend == "openai":
            api_key = get_openai_api_key_for_base_url(base_url, explicit_api_key)
        else:
            api_key = get_env_api_key_for_backend(
                backend,
                explicit_api_key,
            )
        if not api_key:
            _logger.warning("No API key available for LLM (backend=%s)", backend)
            return None
        kwargs["api_key"] = api_key

    # Pass through base_url for custom OpenAI-compatible endpoints
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout

    return create_chat_model(**kwargs)
