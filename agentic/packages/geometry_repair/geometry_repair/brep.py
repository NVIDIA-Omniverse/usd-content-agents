# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCascade-native CAD inspection and bounded shape healing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .models import BRepMetrics

BREP_SUFFIXES = {".step", ".stp", ".iges", ".igs", ".brep"}


def ocp_available() -> tuple[bool, str | None]:
    """Report whether the optional OCP/OCCT worker can be imported."""

    try:
        import OCP  # noqa: F401

        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _read_shape(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".step", ".stp"}:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise RuntimeError(f"OpenCascade could not read STEP file {path}")
        reader.TransferRoots()
        return reader.OneShape()
    if suffix in {".iges", ".igs"}:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.IGESControl import IGESControl_Reader

        reader = IGESControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise RuntimeError(f"OpenCascade could not read IGES file {path}")
        reader.TransferRoots()
        return reader.OneShape()
    if suffix == ".brep":
        from OCP.BRep import BRep_Builder
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape

        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
            raise RuntimeError(f"OpenCascade could not read BREP file {path}")
        return shape
    raise ValueError(f"Unsupported B-rep format: {suffix}")


def _count(shape: Any, kind: Any) -> int:
    from OCP.TopExp import TopExp_Explorer

    count = 0
    explorer = TopExp_Explorer(shape, kind)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _shape_metrics(shape: Any) -> BRepMetrics:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer

    area_properties = GProp_GProps()
    volume_properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, area_properties)
    BRepGProp.VolumeProperties_s(shape, volume_properties)
    bounds = Bnd_Box()
    BRepBndLib.Add_s(shape, bounds)
    minimum_x, minimum_y, minimum_z, maximum_x, maximum_y, maximum_z = bounds.Get()
    center = volume_properties.CentreOfMass()
    maximum_tolerance = float(ShapeAnalysis_ShapeTolerance().Tolerance(shape, 1))
    diagonal = math.sqrt(
        (maximum_x - minimum_x) ** 2 + (maximum_y - minimum_y) ** 2 + (maximum_z - minimum_z) ** 2
    )
    tiny_feature_threshold = max(maximum_tolerance * 2.0, diagonal * 1e-9)
    analyzer = BRepCheck_Analyzer(shape)
    status_counts: dict[str, int] = {}
    subshape_status_counts: dict[str, dict[str, int]] = {}

    def record_statuses(label: str, subshape: Any) -> None:
        try:
            statuses = list(analyzer.Result(subshape).Status())
        except Exception as exc:
            statuses = [f"CheckFailure:{type(exc).__name__}"]
        target = subshape_status_counts.setdefault(label, {})
        for status in statuses:
            name = getattr(status, "name", str(status)).removeprefix("BRepCheck_")
            target[name] = target.get(name, 0) + 1
            status_counts[name] = status_counts.get(name, 0) + 1

    record_statuses("shape", shape)
    edge_lengths: list[float] = []
    face_areas: list[float] = []
    for label, kind in (
        ("solid", TopAbs_SOLID),
        ("shell", TopAbs_SHELL),
        ("face", TopAbs_FACE),
        ("wire", TopAbs_WIRE),
        ("edge", TopAbs_EDGE),
    ):
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            subshape = explorer.Current()
            record_statuses(label, subshape)
            if kind == TopAbs_EDGE:
                properties = GProp_GProps()
                BRepGProp.LinearProperties_s(subshape, properties)
                edge_lengths.append(max(0.0, float(properties.Mass())))
            elif kind == TopAbs_FACE:
                properties = GProp_GProps()
                BRepGProp.SurfaceProperties_s(subshape, properties)
                face_areas.append(max(0.0, float(properties.Mass())))
            explorer.Next()
    return BRepMetrics(
        evaluated=True,
        valid=bool(analyzer.IsValid()),
        solid_count=_count(shape, TopAbs_SOLID),
        shell_count=_count(shape, TopAbs_SHELL),
        face_count=_count(shape, TopAbs_FACE),
        wire_count=_count(shape, TopAbs_WIRE),
        edge_count=_count(shape, TopAbs_EDGE),
        surface_area_source_units2=float(area_properties.Mass()),
        volume_source_units3=float(volume_properties.Mass()),
        center_of_mass_source_units=[float(center.X()), float(center.Y()), float(center.Z())],
        bbox_min_source_units=[minimum_x, minimum_y, minimum_z],
        bbox_max_source_units=[maximum_x, maximum_y, maximum_z],
        maximum_tolerance_source_units=maximum_tolerance,
        validity_status_counts=dict(sorted(status_counts.items())),
        subshape_status_counts={
            label: dict(sorted(counts.items()))
            for label, counts in sorted(subshape_status_counts.items())
        },
        minimum_edge_length_source_units=min(edge_lengths, default=None),
        minimum_face_area_source_units2=min(face_areas, default=None),
        tiny_edge_count=sum(length <= tiny_feature_threshold for length in edge_lengths),
        tiny_face_count=sum(
            area <= tiny_feature_threshold * tiny_feature_threshold for area in face_areas
        ),
        tiny_feature_threshold_source_units=tiny_feature_threshold,
    )


def inspect_brep(path: str | Path) -> tuple[Any | None, BRepMetrics]:
    """Read and inspect a native B-rep without modifying it."""

    source = Path(path).expanduser().resolve()
    try:
        shape = _read_shape(source)
        return shape, _shape_metrics(shape)
    except Exception as exc:
        return None, BRepMetrics(
            evaluated=True,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_shape(shape: Any, target: Path) -> None:
    suffix = target.suffix.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".step", ".stp"}:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        if writer.Write(str(target)) != IFSelect_RetDone:
            raise RuntimeError(f"OpenCascade could not write STEP file {target}")
        return
    if suffix in {".iges", ".igs"}:
        from OCP.IGESControl import IGESControl_Writer

        writer = IGESControl_Writer()
        writer.AddShape(shape)
        if not writer.Write(str(target)):
            raise RuntimeError(f"OpenCascade could not write IGES file {target}")
        return
    if suffix == ".brep":
        from OCP.BRepTools import BRepTools

        if not BRepTools.Write_s(shape, str(target)):
            raise RuntimeError(f"OpenCascade could not write BREP file {target}")
        return
    raise ValueError(f"Unsupported B-rep output format: {suffix}")


def heal_brep(
    source: str | Path,
    target: str | Path,
    *,
    precision: float,
    maximum_tolerance: float,
    sew: bool = True,
) -> dict[str, Any]:
    """Run bounded sewing plus ShapeFix and compare native CAD mass properties."""

    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.ShapeExtend import ShapeExtend_DONE, ShapeExtend_FAIL
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer

    source_path = Path(source).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    shape = _read_shape(source_path)
    before = _shape_metrics(shape)
    sewing_report: dict[str, Any] = {"requested": sew, "executed": False}
    repair_input = shape
    if sew and before.solid_count == 0:
        sewing = BRepBuilderAPI_Sewing(float(precision), True, True, True, False)
        sewing.SetMinTolerance(float(precision))
        sewing.SetMaxTolerance(float(maximum_tolerance))
        sewing.SetNonManifoldMode(False)
        sewing.SetFloatingEdgesMode(False)
        sewing.Add(shape)
        sewing.Perform()
        sewed = sewing.SewedShape()
        if sewed.IsNull():
            raise RuntimeError("OpenCascade sewing returned a null shape")
        repair_input = sewed
        sewing_report.update(
            {
                "executed": True,
                "free_edge_count": int(sewing.NbFreeEdges()),
                "contiguous_edge_count": int(sewing.NbContigousEdges()),
                "multiple_edge_count": int(sewing.NbMultipleEdges()),
                "degenerated_shape_count": int(sewing.NbDegeneratedShapes()),
                "deleted_face_count": int(sewing.NbDeletedFaces()),
                "tolerance": float(sewing.Tolerance()),
                "minimum_tolerance": float(sewing.MinTolerance()),
                "maximum_tolerance": float(sewing.MaxTolerance()),
            }
        )
        if sewing.NbDeletedFaces() > 0:
            raise RuntimeError(
                "OpenCascade sewing deleted faces; automatic B-rep repair refuses this semantic change"
            )
    elif sew:
        sewing_report["skip_reason"] = (
            "source already contains solids; whole-shape sewing would discard solid identity"
        )
    fixer = ShapeFix_Shape(repair_input)
    fixer.SetPrecision(float(precision))
    fixer.SetMinTolerance(float(precision))
    fixer.SetMaxTolerance(float(maximum_tolerance))
    performed = bool(fixer.Perform())
    healed = fixer.Shape()
    after = _shape_metrics(healed)
    if not after.valid:
        raise RuntimeError("OpenCascade ShapeFix output remains invalid")
    _write_shape(healed, target_path)
    context = fixer.Context()
    recorded_subshapes: dict[str, int] = {}
    if context is not None:
        for label, kind in (
            ("solid", TopAbs_SOLID),
            ("shell", TopAbs_SHELL),
            ("face", TopAbs_FACE),
            ("wire", TopAbs_WIRE),
            ("edge", TopAbs_EDGE),
        ):
            count = 0
            explorer = TopExp_Explorer(repair_input, kind)
            while explorer.More():
                if context.IsRecorded(explorer.Current()):
                    count += 1
                explorer.Next()
            recorded_subshapes[label] = count
    topology_deltas = {
        field: getattr(after, field) - getattr(before, field)
        for field in ("solid_count", "shell_count", "face_count", "wire_count", "edge_count")
    }
    return {
        "status": "completed",
        "source": str(source_path),
        "output": str(target_path),
        "precision": precision,
        "maximum_tolerance": maximum_tolerance,
        "shape_fix": {
            "performed": performed,
            "status_done": bool(fixer.Status(ShapeExtend_DONE)),
            "status_fail": bool(fixer.Status(ShapeExtend_FAIL)),
            "recorded_subshape_counts": recorded_subshapes,
            "topology_deltas": topology_deltas,
            "maximum_tolerance_delta_source_units": (
                (after.maximum_tolerance_source_units or 0.0)
                - (before.maximum_tolerance_source_units or 0.0)
            ),
        },
        "sewing": sewing_report,
        "before": before.model_dump(mode="json"),
        "after": after.model_dump(mode="json"),
    }
