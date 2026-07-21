# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-safe diagnostic helpers for Material Agent API boundaries."""

from pathlib import Path
from typing import Any

from world_understanding.utils.credentials import redact_sensitive_path

_CONFIG_FILE_NOT_FOUND_MESSAGE = "Config file not found"
_CONFIG_FILE_INSPECTION_FAILURE_MESSAGE = "Unable to inspect config file"


def normalize_required_config(
    config: Path | dict[str, Any],
) -> Path | dict[str, Any]:
    """Normalize an API config without reflecting its path in failures."""
    if isinstance(config, dict):
        if not config:
            raise ValueError("Config dictionary cannot be empty")
        return config

    config_path = Path(config)
    inspection_failed = False
    try:
        config_exists = config_path.exists()
    except OSError:
        inspection_failed = True
        config_exists = False
    if inspection_failed:
        raise OSError(_CONFIG_FILE_INSPECTION_FAILURE_MESSAGE)
    if not config_exists:
        raise FileNotFoundError(_CONFIG_FILE_NOT_FOUND_MESSAGE)
    return config_path


def diagnostic_path(value: str | Path | None) -> str:
    """Project a runtime path into a credential-safe diagnostic string."""
    return redact_sensitive_path(value)
