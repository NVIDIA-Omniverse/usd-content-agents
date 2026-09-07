# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ownership-safe USD structure and coherent stage-metric repair."""

from __future__ import annotations

from pathlib import Path

from ..mesh_io import USD_SUFFIXES, copy_usd_stage
from ..models import RepairOperation
from .base import WorkerResult


def repair_usd_structure(source: Path, output: Path) -> dict:
    """Assign an unambiguous default prim without authoring domain schemas."""

    from pxr import Usd

    copy_usd_stage(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"Could not open USD structure repair target {output}")
    changes: list[str] = []
    if not stage.GetDefaultPrim():
        roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
        if len(roots) != 1:
            raise RuntimeError(f"Cannot infer default prim from {len(roots)} root prims")
        stage.SetDefaultPrim(roots[0])
        changes.append(f"set_default_prim:{roots[0].GetPath()}")
    stage.GetRootLayer().Save()
    return {"changes": changes, "output": str(output)}


def normalize_usd_stage_metrics(source: Path, output: Path) -> dict:
    """Normalize to meter/Z-up using a coherent reference root transform."""

    from pxr import Gf, Usd, UsdGeom

    source_stage = Usd.Stage.Open(str(source))
    if source_stage is None:
        raise RuntimeError(f"Could not open USD metric source {source}")
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim:
        roots = [prim for prim in source_stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
        if len(roots) != 1:
            raise RuntimeError(f"Cannot normalize stage metrics with {len(roots)} ambiguous roots")
        default_prim = roots[0]
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    up_axis = str(UsdGeom.GetStageUpAxis(source_stage)).upper()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/RepairedAsset")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(str(source), default_prim.GetPath())
    xform = UsdGeom.Xformable(root)
    if up_axis == "Y":
        xform.AddRotateXOp().Set(90.0)
    elif up_axis == "X":
        xform.AddRotateYOp().Set(-90.0)
    elif up_axis != "Z":
        raise RuntimeError(f"Unsupported USD up axis {up_axis!r}")
    if abs(meters_per_unit - 1.0) > 1e-12:
        xform.AddScaleOp().Set(Gf.Vec3d(meters_per_unit))
    root.GetPrim().SetCustomDataByKey("sourceDefaultPrim", str(default_prim.GetPath()))
    root.GetPrim().SetCustomDataByKey("sourceMetersPerUnit", meters_per_unit)
    root.GetPrim().SetCustomDataByKey("sourceUpAxis", up_axis)
    stage.GetRootLayer().Save()
    return {
        "changes": [
            f"normalize_up_axis:{up_axis}->Z",
            f"normalize_meters_per_unit:{meters_per_unit}->1.0",
        ],
        "output": str(output),
    }


class UsdStructureRepairWorker:
    """Repair only stage structure and metric ownership boundaries."""

    name = "usd_structure_repair"
    operations = frozenset({"set_default_prim", "coherent_stage_metric_normalization"})

    def available(self) -> tuple[bool, str | None]:
        try:
            import pxr  # noqa: F401

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
                failures=["USD structure repair requires a prepared USD source"],
            )
        try:
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(str(source))
            if stage is None:
                raise RuntimeError(f"Could not open {source}")
            needs_metrics = (
                str(UsdGeom.GetStageUpAxis(stage)).upper() != "Z"
                or abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1e-12
            )
            report = (
                normalize_usd_stage_metrics(source, output)
                if needs_metrics
                else repair_usd_structure(source, output)
            )
        except Exception as exc:
            return WorkerResult(
                status="failed",
                failures=[f"{type(exc).__name__}: {exc}"],
            )
        return WorkerResult(
            status="completed",
            output_path=str(output),
            changed=bool(report["changes"]),
            operations=list(report["changes"]),
            metadata=report,
        )
