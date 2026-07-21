# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def derive_completed_step_names(
    preferred_names: Any,
    completed_steps: Any,
) -> list[str]:
    """Return explicit step names or derive them from progress snapshots.

    An explicit list is authoritative even when empty. This lets regenerated
    runs clear names from an older EventBus snapshot. Callers that want an
    empty explicit list to fall back must pass ``None`` instead.
    """
    if isinstance(preferred_names, list):
        return [name for name in preferred_names if isinstance(name, str)]
    if not isinstance(completed_steps, list):
        return []
    return [
        step["name"]
        for step in completed_steps
        if isinstance(step, dict) and isinstance(step.get("name"), str)
    ]


def get_version() -> str:
    try:
        return version("physics-agent-service")
    except PackageNotFoundError:
        return "0.0.1-dev"


class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Exclude healthchecks from access logs
        return (
            record.getMessage().find("/health") == -1
            and record.getMessage().find("/metrics") == -1
        )
