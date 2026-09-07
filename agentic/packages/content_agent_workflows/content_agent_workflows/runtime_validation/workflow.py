# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared authored-physics and temporary-proxy runtime validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from world_understanding.agentic.usd_tasks.optimize_usd import OptimizeUSDTask

from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.common.optimizer import (
    OptimizerBackend,
    OptimizerRequestOptions,
)
from content_agent_workflows.physics.scene_ops import (
    apply_schema,
    inspect_authored_physics,
    inspect_mesh_candidates,
    validate_runtime,
)

RUNTIME_VALIDATION_SCHEMA_VERSION = "content-agent-workflows.runtime-validation.v1"
RuntimeValidationMode = Literal[
    "skip",
    "authored_physics",
    "temporary_loadability_proxy",
]
RuntimeValidationEngine = Literal["ovphysx", "fake", "none"]
RuntimeValidationStatus = Literal["pass", "fail", "warning", "not_evaluated"]


class RuntimeValidationRequest(BaseModel):
    """Explicit runtime-validation request shared by domain workflows."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    asset_path: Path
    output_dir: Path
    mode: RuntimeValidationMode = "skip"
    engine: RuntimeValidationEngine = "ovphysx"
    duration_s: float = Field(default=3.0, gt=0.0, le=10.0)
    dt: float = Field(default=1.0 / 120.0, gt=0.0, le=0.1)
    sample_fps: int = Field(default=30, ge=1, le=120)
    drop_height_m: float | None = None


class RuntimeValidationResult(BaseModel):
    """Durable runtime result with explicit claim scope."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = RUNTIME_VALIDATION_SCHEMA_VERSION
    status: RuntimeValidationStatus
    mode: RuntimeValidationMode
    claim_scope: str
    asset_path: str
    runtime_validation_usd: str | None = None
    temporary_proxy_used: bool = False
    upstream_runtime_report: str | None = None
    evidence_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None


def _has_authored_rigid_body(asset_path: Path) -> bool:
    report = inspect_authored_physics(asset_path)
    return int(report.get("rigid_body_count") or 0) > 0


def _guide_mesh_paths(asset_path: Path) -> set[str]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        return set()
    return {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
        and UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide
    }


def _leaf_mesh_candidates(
    candidates: list[dict[str, Any]],
    *,
    excluded_prim_paths: set[str],
) -> list[dict[str, Any]]:
    meshes = [
        item
        for item in candidates
        if item.get("prim_path")
        and str(item.get("type_name") or "") == "Mesh"
        and str(item["prim_path"]) not in excluded_prim_paths
    ]
    paths = [str(item["prim_path"]) for item in meshes]
    return [
        item
        for item in meshes
        if not any(
            other != str(item["prim_path"])
            and other.startswith(f"{item['prim_path']}/")
            for other in paths
        )
    ]


def _write_proxy_predictions(
    *, inspection: dict[str, Any], predictions_path: Path, excluded: set[str]
) -> int:
    candidates = _leaf_mesh_candidates(
        list(inspection.get("candidates") or []), excluded_prim_paths=excluded
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as stream:
        for candidate in candidates:
            stream.write(
                json.dumps(
                    {
                        "id": str(candidate["prim_path"]),
                        "classification": {
                            "component": candidate.get("prim_name"),
                            "physical_properties": {
                                "density": 500.0,
                                "static_friction": 0.8,
                                "dynamic_friction": 0.7,
                                "restitution": 0.0,
                            },
                            "collision_approximation": "convexHull",
                            "confidence": 0.5,
                            "reasoning": (
                                "Temporary loadability proxy; not final physics "
                                "authoring or a SimReady claim."
                            ),
                        },
                        "source": (
                            "content_agent_workflows.runtime_validation."
                            "temporary_loadability_proxy"
                        ),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return len(candidates)


def _prepare_temporary_proxy(
    asset_path: Path, output_dir: Path
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    proxy_dir = output_dir / "temporary_proxy"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    deinstanced = proxy_dir / f"{asset_path.stem}.deinstanced{asset_path.suffix}"
    proxy_usd = proxy_dir / f"{asset_path.stem}.runtime-proxy{asset_path.suffix}"
    predictions = proxy_dir / "runtime_proxy_predictions.jsonl"
    for stale_path in (deinstanced, proxy_usd, predictions):
        stale_path.unlink(missing_ok=True)

    options = OptimizerRequestOptions(
        optimize=True,
        optimizer_backend=OptimizerBackend.LOCAL,
        flatten_prototypes=False,
        enable_deinstance=True,
        enable_split=False,
        enable_deduplicate=False,
    )
    optimizer_context = OptimizeUSDTask().run(
        {
            "input_usd_path": str(asset_path),
            "output_usd_path": str(deinstanced),
            "optimization_config": options.resolved_optimization_config(),
        }
    )
    if not optimizer_context.get("optimization_success") or not deinstanced.exists():
        raise RuntimeError("Temporary runtime proxy could not deinstance the asset.")

    inspection = inspect_mesh_candidates(deinstanced)
    excluded = _guide_mesh_paths(deinstanced)
    prediction_count = _write_proxy_predictions(
        inspection=inspection,
        predictions_path=predictions,
        excluded=excluded,
    )
    if prediction_count <= 0:
        raise RuntimeError("Temporary runtime proxy found no eligible mesh candidates.")
    apply_report = apply_schema(
        usd_path=deinstanced,
        predictions_jsonl_path=predictions,
        output_usd_path=proxy_usd,
        collision_approximation="convexHull",
    )
    artifacts = [
        {
            "kind": "runtime_proxy_deinstanced_usd",
            "path": str(deinstanced),
            "description": "Temporary deinstanced geometry used for proxy authoring.",
        },
        {
            "kind": "runtime_proxy_predictions",
            "path": str(predictions),
            "description": "Temporary neutral proxy predictions.",
        },
        {
            "kind": "runtime_validation_usd",
            "path": str(proxy_usd),
            "description": "Temporary proxy USD used only for loadability validation.",
        },
    ]
    return (
        proxy_usd,
        {
            "prediction_count": prediction_count,
            "guide_mesh_count": len(excluded),
            "rigid_body_count": apply_report.get("rigid_body_count"),
            "collision_count": apply_report.get("collision_count"),
            "optimizer_metadata": optimizer_context.get("optimization_metadata") or {},
        },
        artifacts,
    )


def _write_result(
    result: RuntimeValidationResult, output_dir: Path
) -> RuntimeValidationResult:
    report_path = output_dir / "runtime_validation_workflow.json"
    resolved = result.model_copy(update={"report_path": str(report_path)})
    atomic_write_json(report_path, resolved)
    return resolved


def run_runtime_validation(
    params: RuntimeValidationRequest,
) -> RuntimeValidationResult:
    """Run authored-physics or explicit temporary-proxy validation."""

    asset_path = params.asset_path.resolve()
    output_dir = params.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not asset_path.exists():
        return _write_result(
            RuntimeValidationResult(
                status="fail",
                mode=params.mode,
                claim_scope="runtime_loadability",
                asset_path=str(asset_path),
                failures=[f"Runtime validation asset does not exist: {asset_path}"],
            ),
            output_dir,
        )
    if params.mode == "skip" or params.engine == "none":
        return _write_result(
            RuntimeValidationResult(
                status="not_evaluated",
                mode=params.mode,
                claim_scope="none",
                asset_path=str(asset_path),
                metadata={"reason": "Runtime validation was explicitly skipped."},
            ),
            output_dir,
        )

    runtime_usd = asset_path
    temporary_proxy_used = False
    metadata: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    try:
        if params.mode == "authored_physics":
            if not _has_authored_rigid_body(asset_path):
                raise RuntimeError(
                    "authored_physics mode requires an authored rigid body; "
                    "Geometry did not create temporary physics."
                )
            claim_scope = "authored_physics_runtime"
        else:
            runtime_usd, proxy_metadata, proxy_artifacts = _prepare_temporary_proxy(
                asset_path, output_dir
            )
            metadata["temporary_proxy"] = proxy_metadata
            artifacts.extend(proxy_artifacts)
            temporary_proxy_used = True
            claim_scope = "temporary_geometry_loadability_only"

        upstream = validate_runtime(
            physics_usd=runtime_usd,
            output_dir=output_dir / "scene",
            engine=params.engine,
            duration_s=params.duration_s,
            dt=params.dt,
            sample_fps=params.sample_fps,
            drop_height_m=params.drop_height_m,
        )
        failures = [str(item) for item in upstream.get("failures") or []]
        warnings = [str(item) for item in upstream.get("warnings") or []]
        artifacts.extend(
            dict(item)
            for item in upstream.get("evidence_artifacts") or []
            if isinstance(item, dict) and item.get("path")
        )
        upstream_report = upstream.get("runtime_report")
        if temporary_proxy_used and not failures:
            warnings.append(
                "Temporary proxy passed loadability checks; this is not authored "
                "physics, behavioral validation, or SimReady evidence."
            )
        status: RuntimeValidationStatus = (
            "fail" if failures else "warning" if warnings else "pass"
        )
        metadata["engine"] = params.engine
        metadata["upstream_result"] = upstream
        return _write_result(
            RuntimeValidationResult(
                status=status,
                mode=params.mode,
                claim_scope=claim_scope,
                asset_path=str(asset_path),
                runtime_validation_usd=str(runtime_usd),
                temporary_proxy_used=temporary_proxy_used,
                upstream_runtime_report=str(upstream_report)
                if upstream_report
                else None,
                evidence_artifacts=artifacts,
                failures=failures,
                warnings=warnings,
                metadata=metadata,
            ),
            output_dir,
        )
    except Exception as exc:
        return _write_result(
            RuntimeValidationResult(
                status="fail",
                mode=params.mode,
                claim_scope=(
                    "temporary_geometry_loadability_only"
                    if params.mode == "temporary_loadability_proxy"
                    else "authored_physics_runtime"
                ),
                asset_path=str(asset_path),
                runtime_validation_usd=str(runtime_usd),
                temporary_proxy_used=temporary_proxy_used,
                evidence_artifacts=artifacts,
                failures=[f"Runtime validation failed: {type(exc).__name__}: {exc}"],
                metadata=metadata,
            ),
            output_dir,
        )
