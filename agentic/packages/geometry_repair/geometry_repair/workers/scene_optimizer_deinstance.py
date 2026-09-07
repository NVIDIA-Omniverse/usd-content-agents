# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed NVIDIA Scene Optimizer prerequisite for editable instance geometry."""

from __future__ import annotations

from pathlib import Path

from ..mesh_io import USD_SUFFIXES
from ..models import RepairOperation
from .base import WorkerResult


class SceneOptimizerDeinstanceWorker:
    """Materialize USD instances without splitting, merging, or deduplicating."""

    name = "scene_optimizer_deinstance"
    operations = frozenset({"deinstance_without_split_merge_or_deduplicate"})

    def available(self) -> tuple[bool, str | None]:
        try:
            from world_understanding.functions.graphics.scene_optimizer_local import (
                _resolve_so_package_dir,
            )

            _resolve_so_package_dir()
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        if source.suffix.lower() not in USD_SUFFIXES:
            return WorkerResult(
                status="unavailable",
                failures=["Scene Optimizer deinstance requires a prepared USD source"],
            )
        from world_understanding.functions.graphics.scene_optimizer_local import (
            optimize_usd_local,
        )

        approved_dependency_roots = operation.parameters.get("approved_dependency_roots")
        if approved_dependency_roots is not None and (
            not isinstance(approved_dependency_roots, list)
            or not approved_dependency_roots
            or not all(isinstance(root, str) and root for root in approved_dependency_roots)
        ):
            return WorkerResult(
                status="failed",
                failures=["approved_dependency_roots must be a non-empty list of paths"],
            )

        result = optimize_usd_local(
            input_path=source,
            output_path=output,
            approved_dependency_roots=approved_dependency_roots,
            optimization_config={
                "timeout": float(operation.parameters.get("timeout_s", 300.0)),
                "scene_optimizer_settings": {
                    "enable_deinstance": True,
                    "enable_split_meshes": False,
                    "enable_deduplicate": False,
                    "deinstance": {"prim_paths": []},
                    "generate_report": True,
                    "capture_stats": True,
                    "verbose": False,
                },
            },
        )
        status = str(result.get("status") or "").lower()
        if status not in {"completed", "success"} or not output.is_file():
            return WorkerResult(
                status="failed",
                failures=[
                    str(result.get("error") or "Scene Optimizer did not produce deinstanced USD")
                ],
                metadata={"scene_optimizer_status": status},
            )
        correspondence = result.get("correspondence_map")
        correspondence_count = len(correspondence) if isinstance(correspondence, dict) else 0
        return WorkerResult(
            status="completed",
            output_path=str(output),
            changed=True,
            operations=["scene_optimizer_deinstance"],
            metadata={
                "scene_optimizer_status": status,
                "optimization_time_s": result.get("optimization_time"),
                "stage_size_bytes": result.get("stage_size_bytes"),
                "operations_executed": list(result.get("operations_executed") or []),
                "correspondence_entry_count": correspondence_count,
                "report": result.get("report"),
            },
        )
