# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration models and utilities for World Understanding agents."""

from .base_path_resolver import BasePathResolver
from .context_loader import (
    ConfigEmptyError,
    ConfigLoadError,
    ConfigParseError,
    ConfigSourceError,
    ConfigStructureError,
    config_source_name,
    load_config_mapping_from_context,
    log_config_source,
)
from .isolation import (
    UNSUPPORTED_YAML_CONFIG_MESSAGE,
    clone_config_containers,
    normalize_yaml_config_value,
)
from .loader import ConfigError, ConfigLoader, load_config
from .model_credentials import validate_selected_model_credentials
from .usd_dataset import (
    PrimFilters,
    RendererConfig,
    RenderingModeConfig,
    USDDatasetConfig,
)
from .utils import (
    API_KEY_ENV_VAR_MAP,
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    ensure_tuple,
    get_api_key_for_backend,
    get_api_key_for_model_config,
    get_openai_api_key_for_base_url,
    is_local_base_url,
    is_local_nim_api_key_placeholder,
    is_placeholder_api_key,
    resolve_path_from_config,
    safe_divide,
)

__all__ = [
    # Models
    "USDDatasetConfig",
    "RendererConfig",
    "RenderingModeConfig",
    "PrimFilters",
    # Path Resolvers
    "BasePathResolver",
    # Utilities
    "API_KEY_ENV_VAR_MAP",
    "LOCAL_NIM_API_KEY_PLACEHOLDER",
    "resolve_path_from_config",
    "get_api_key_for_backend",
    "get_api_key_for_model_config",
    "get_openai_api_key_for_base_url",
    "is_local_nim_api_key_placeholder",
    "is_placeholder_api_key",
    "is_local_base_url",
    "ensure_tuple",
    "safe_divide",
    # Loaders
    "ConfigLoader",
    "ConfigError",
    "load_config",
    "validate_selected_model_credentials",
    "load_config_mapping_from_context",
    "clone_config_containers",
    "normalize_yaml_config_value",
    "UNSUPPORTED_YAML_CONFIG_MESSAGE",
    "config_source_name",
    "log_config_source",
    "ConfigLoadError",
    "ConfigSourceError",
    "ConfigParseError",
    "ConfigStructureError",
    "ConfigEmptyError",
]
