# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration loading for Material Agent workflow tasks."""

from pathlib import Path
from typing import Any

from world_understanding.agentic.config import load_config_mapping_from_context
from world_understanding.utils.credentials import resolve_path_with_safe_diagnostics


def resolve_config_relative_path(value: str | Path, config_path: Path) -> str:
    """Resolve one config-owned path against its source-file anchor.

    Dictionary configuration is transported in memory, so ``config_path`` may
    be a non-existent source anchor.  Never use existence checks to choose the
    base: relative values always belong to the anchor's parent, while absolute
    values retain their caller-provided spelling.
    """
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(
        resolve_path_with_safe_diagnostics(
            config_path.parent / path,
            label="configuration-relative path",
        )
    )


def load_config_from_context(
    context: dict[str, Any],
    *,
    missing_path_message: str = "config_path not provided in context",
    missing_file_message: str = "Configuration file not found: {config_path}",
    empty_message: str = "Configuration file is empty",
    non_mapping_message: str | None = None,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Load an isolated config dictionary and its relative-path anchor.

    ``config_dict`` is authoritative when present. ``config_path`` remains useful
    in that mode as the anchor for relative paths, but the path is never opened
    or required to exist. File-based callers retain the legacy YAML behavior.
    Custom diagnostics may use ``{config_path}`` and ``{type_name}``; all other
    braces remain literal.
    """
    return load_config_mapping_from_context(
        context,
        default_config_path=Path.cwd() / "material_agent_config.yaml",
        allow_empty=allow_empty,
        missing_path_message=missing_path_message,
        missing_file_message=missing_file_message,
        parse_error_message="Unable to parse configuration file: {config_path}",
        empty_message=empty_message,
        config_dict_non_mapping_message=(
            non_mapping_message or "config_dict must be a mapping, got {type_name}"
        ),
        file_non_mapping_message=(
            non_mapping_message or "Configuration must be a mapping, got {type_name}"
        ),
    )
