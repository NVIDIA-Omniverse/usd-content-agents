# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional compatibility bridge for material assignment USD authoring."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any


class MaterialApplyUnavailableError(RuntimeError):
    """Raised when the optional material-agent authoring dependency is absent."""


def _load_material_apply_runner() -> Callable[[dict[str, Any]], dict[str, Any]]:
    try:
        from material_agent.tasks.apply_materials_to_usd import (
            ApplyMaterialsToUSDTask,
        )
    except ImportError as exc:
        raise MaterialApplyUnavailableError(
            "Material assignment apply requires the optional material-agent package. "
            "Install apps/material_agent or route material authoring through the "
            "material assignment workflow package before calling this compatibility "
            "endpoint."
        ) from exc

    task = ApplyMaterialsToUSDTask()

    def run(context: dict[str, Any]) -> dict[str, Any]:
        result = task.run(context)
        if not isinstance(result, dict):
            raise RuntimeError("Material apply task returned a non-dict result")
        return result

    return run


def run_material_apply_task(
    context: dict[str, Any],
    *,
    executor: ThreadPoolExecutor | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the optional material-agent USD authoring task."""
    run_task = _load_material_apply_runner()

    if executor is None or timeout_seconds is None:
        return run_task(context)

    future: Future[dict[str, Any]] = executor.submit(run_task, context)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            "Timed out waiting for material apply task after "
            f"{timeout_seconds:g} seconds"
        ) from exc
