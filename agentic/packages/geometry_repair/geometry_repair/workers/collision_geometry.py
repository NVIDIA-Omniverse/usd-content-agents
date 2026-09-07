# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated worker adapter for the complete collision-geometry pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..artifacts import file_sha256
from ..models import (
    CollisionReport,
    ProtectedFeature,
    RepairBudgets,
    RepairOperation,
    RepairProfile,
    StrictModel,
)
from .base import WorkerResult


class CollisionGeometryParameters(StrictModel):
    """Strict wire contract for one complete collision-geometry build."""

    operation: Literal["build_collision_geometry"]
    profile: RepairProfile
    budgets: RepairBudgets
    protected_features: list[ProtectedFeature] = Field(default_factory=list)
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    coacd_enabled: bool = True
    sdf_collision_rebuild_enabled: bool = True
    timeout_s: float = Field(gt=0.0)
    runtime_engine: Literal["skip", "fake", "ovphysx"] = "skip"
    report_path: Path
    source_collision_audit_path: Path

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_sdf_toggle(cls, value: object) -> object:
        if not isinstance(value, dict) or "openvdb_collision_rebuild_enabled" not in value:
            return value
        normalized = dict(value)
        legacy = normalized.pop("openvdb_collision_rebuild_enabled")
        if (
            "sdf_collision_rebuild_enabled" in normalized
            and normalized["sdf_collision_rebuild_enabled"] != legacy
        ):
            raise ValueError("conflicting canonical and legacy SDF collision toggles")
        normalized["sdf_collision_rebuild_enabled"] = legacy
        return normalized


class CollisionGeometryWorker:
    """Run collision planning and construction behind the generic worker boundary."""

    name = "collision_geometry_pipeline"
    operations = frozenset({"build_collision_geometry"})

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        parameters = CollisionGeometryParameters.model_validate(operation.parameters)
        source = source.expanduser().resolve()
        output = output.expanduser().resolve()
        checkpoint = Path(operation.source_checkpoint).expanduser().resolve()
        report_path = parameters.report_path.expanduser().resolve()
        source_collision_audit_path = parameters.source_collision_audit_path.expanduser().resolve()

        if checkpoint != source:
            raise ValueError("collision worker source does not match its source checkpoint")
        if output == source:
            raise ValueError("collision worker output must not overwrite the render source")
        if (
            report_path.parent != output.parent
            or source_collision_audit_path.parent != output.parent
        ):
            raise ValueError("collision worker evidence paths must remain beside its output")
        if output.name != "collision.usda":
            raise ValueError("collision worker output must use the collision.usda artifact name")
        if report_path.name != "collision_report.json":
            raise ValueError(
                "collision worker report must use the collision_report.json artifact name"
            )
        if source_collision_audit_path.name != "source_collision_audit.json":
            raise ValueError(
                "collision worker audit must use the source_collision_audit.json artifact name"
            )
        if len({source, output, report_path, source_collision_audit_path}) != 4:
            raise ValueError("collision worker source, output, and evidence paths must be distinct")

        source_collision_audit_path.unlink(missing_ok=True)
        from ..collision import build_collision_geometry

        report = build_collision_geometry(
            source,
            output,
            profile=parameters.profile,
            budgets=parameters.budgets,
            protected_features=parameters.protected_features,
            deterministic_seed=parameters.deterministic_seed,
            coacd_enabled=parameters.coacd_enabled,
            sdf_collision_rebuild_enabled=parameters.sdf_collision_rebuild_enabled,
            timeout_s=parameters.timeout_s,
            runtime_engine=parameters.runtime_engine,
            report_path=report_path,
            source_collision_audit_path=source_collision_audit_path,
        )
        if not report_path.is_file():
            raise RuntimeError("collision pipeline did not write its required report")
        persisted_report = CollisionReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        if persisted_report != report:
            raise RuntimeError("persisted collision report does not match the returned report")
        if report.report_path is not None and Path(report.report_path).resolve() != report_path:
            raise RuntimeError("collision report points outside its assigned report artifact")
        metadata = {
            "collision_report_path": str(report_path),
            "collision_report_sha256": file_sha256(report_path),
            "collision_status": report.status,
        }
        if report.source_collision_audit_path is not None:
            claimed_audit = Path(report.source_collision_audit_path).resolve()
            if claimed_audit != source_collision_audit_path or not claimed_audit.is_file():
                raise RuntimeError("collision report contains invalid source-audit evidence")
            metadata.update(
                {
                    "source_collision_audit_path": str(claimed_audit),
                    "source_collision_audit_sha256": file_sha256(claimed_audit),
                }
            )
        elif source_collision_audit_path.exists():
            source_collision_audit_path.unlink(missing_ok=True)
            raise RuntimeError("collision pipeline left an unreported source-audit artifact")

        collision_path: Path | None = None
        if report.collision_path is not None:
            collision_path = Path(report.collision_path).resolve()
            if collision_path != output or not output.is_file():
                raise RuntimeError("collision report contains an invalid collision artifact path")
            if report.collision_sha256 != file_sha256(output):
                raise RuntimeError("collision report digest does not match its collision artifact")
        elif output.exists():
            raise RuntimeError("collision pipeline left an unreported collision artifact")

        return WorkerResult(
            status="completed",
            output_path=str(collision_path) if collision_path is not None else None,
            changed=collision_path is not None,
            operations=[parameters.operation],
            metadata=metadata,
        )
