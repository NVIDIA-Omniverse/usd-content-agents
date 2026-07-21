# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration utility functions.

This module provides common configuration utilities used across all agents
for path resolution, API key management, and more.
"""

from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import (
    API_KEY_ENV_VAR_MAP,
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    get_env_api_key_for_backend,
    get_nim_api_key_for_base_url,
    get_openai_api_key_for_base_url,
    is_local_base_url,
    is_local_nim_api_key_placeholder,
    is_placeholder_api_key,
    openai_missing_credential_message,
    resolve_effective_openai_base_url,
    resolve_endpoint_api_key,
)

__all__ = [
    "API_KEY_ENV_VAR_MAP",
    "LOCAL_NIM_API_KEY_PLACEHOLDER",
    "ensure_tuple",
    "get_api_key_for_backend",
    "get_api_key_for_model_config",
    "get_openai_api_key_for_base_url",
    "is_local_base_url",
    "is_local_nim_api_key_placeholder",
    "is_placeholder_api_key",
    "resolve_path_from_config",
    "safe_divide",
]


def get_api_key_for_model_config(
    backend: str, model_config: dict[str, Any], model_type: str = "model"
) -> str:
    """Resolve a configured model API key using runtime precedence."""
    api_key = resolve_endpoint_api_key(
        model_config.get("api_key"),
        model_config.get("api_key_env"),
        require_env=model_config.get("api_key") is None,
    )
    if backend == "nim":
        base_url = model_config.get("base_url")
        resolved_nim_key = get_nim_api_key_for_base_url(
            base_url,
            str(api_key) if api_key is not None else None,
        )
        if resolved_nim_key:
            return resolved_nim_key
        if base_url:
            raise ValueError(f"NVIDIA_API_KEY not set for {backend} {model_type}")
        return get_api_key_for_backend(backend, model_type)

    if backend == "openai":
        base_url = model_config.get("base_url")
        # Effective base URL includes ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE``
        # because the OpenAI SDK falls back to those when the constructor
        # gets no explicit ``base_url``. The endpoint check must reflect
        # where the request will actually go, or the hosted ``OPENAI_API_KEY``
        # would be forwarded to an env-redirected custom endpoint.
        effective_base_url = resolve_effective_openai_base_url(base_url)
        resolved_openai_key = get_openai_api_key_for_base_url(
            base_url,
            str(api_key) if api_key is not None else None,
        )
        if resolved_openai_key:
            return resolved_openai_key
        if effective_base_url:
            raise ValueError(openai_missing_credential_message(base_url, model_type))
        return get_api_key_for_backend(backend, model_type)

    if api_key and not is_placeholder_api_key(api_key):
        return str(api_key)

    return get_api_key_for_backend(backend, model_type)


def resolve_path_from_config(
    path_key: str,
    config: dict,
    config_path: Path,
    override_key: str | None = None,
    context: dict | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a path from configuration with override support.

    This handles the common pattern of:
    1. Check for override in context
    2. Use config value if available
    3. Make relative paths absolute relative to config file
    4. Optionally validate existence

    Args:
        path_key: Key name in config dict (e.g., "usd_path", "output_dir")
        config: Configuration dictionary
        config_path: Path to the configuration file (for relative path resolution)
        override_key: Optional key name in context for override value
        context: Optional context dictionary containing overrides
        must_exist: If True, raise FileNotFoundError if path doesn't exist

    Returns:
        Resolved absolute Path object

    Raises:
        ValueError: If neither override nor config provides a value
        FileNotFoundError: If must_exist=True and path doesn't exist

    Example:
        ```python
        # In a config task
        usd_path = resolve_path_from_config(
            path_key="usd_path",
            config=config,
            config_path=config_path,
            override_key="source_override",
            context=context,
            must_exist=True
        )
        ```
    """
    context = context or {}
    override_key = override_key or f"{path_key}_override"

    # Try override first
    path_value = context.get(override_key)

    # Fall back to config
    if path_value is None:
        path_value = config.get(path_key)

    # Validate we got a value
    if path_value is None:
        raise ValueError(
            f"'{path_key}' not specified. Provide it via config or context['{override_key}']"
        )

    # Convert to Path and resolve
    path_obj = Path(path_value)

    # Make relative paths absolute relative to config file directory
    if not path_obj.is_absolute():
        if config_path.is_file():
            path_obj = config_path.parent / path_obj
        else:
            path_obj = config_path / path_obj

    # Resolve to absolute path
    path_obj = path_obj.resolve()

    # Validate existence if required
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"{path_key} not found: {path_obj}")

    return path_obj


def get_api_key_for_backend(backend: str, model_type: str = "model") -> str:
    """Get API key for a backend with validation.

    Retrieves API keys from environment variables with backend-specific
    error messages.

    Args:
        backend: Backend name (e.g., "nim", "openai")
        model_type: Type of model for error message (e.g., "VLM", "LLM")

    Returns:
        API key string

    Raises:
        ValueError: If required API key is not set

    Example:
        ```python
        api_key = get_api_key_for_backend("nim", "VLM")
        # Retrieves NVIDIA_API_KEY or raises error
        ```
    """
    env_vars = API_KEY_ENV_VAR_MAP.get(backend)

    if env_vars is None:
        # Backend doesn't require API key or uses default mechanism
        return ""

    api_key = get_env_api_key_for_backend(backend)
    if not api_key:
        # Create helpful error message with all attempted env vars
        env_vars_str = " or ".join(env_vars)
        raise ValueError(f"{env_vars_str} not set for {backend} {model_type}")

    return api_key


def ensure_tuple(value: list | tuple | None, default: list | tuple) -> tuple:
    """Ensure value is a tuple, converting from list if needed.

    Args:
        value: Value to convert (list, tuple, or None)
        default: Default value to use if value is None

    Returns:
        Tuple version of the value

    Example:
        ```python
        color = ensure_tuple([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        # Returns (1.0, 0.0, 0.0)

        color = ensure_tuple(None, [0.5, 0.5, 0.5])
        # Returns (0.5, 0.5, 0.5)
        ```
    """
    if value is None:
        value = default
    return tuple(value) if isinstance(value, list) else value


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value

    Example:
        ```python
        ratio = safe_divide(10, 5)     # Returns 2.0
        ratio = safe_divide(10, 0)     # Returns 0.0
        ratio = safe_divide(10, 0, 1)  # Returns 1.0
        ```
    """
    return numerator / denominator if denominator > 0 else default
