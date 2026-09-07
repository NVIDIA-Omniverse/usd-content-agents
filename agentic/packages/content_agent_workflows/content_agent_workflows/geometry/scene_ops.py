# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workflow-owned geometry operations composed with usd-cli primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from world_understanding.agentic.usd_tasks.optimize_usd import OptimizeUSDTask
from world_understanding.utils.artifacts import fsync_directory

from content_agent_workflows.common.optimizer import (
    OptimizerBackend,
    OptimizerRequestOptions,
)

GeometryOptimizationPolicy = Literal[
    "skip",
    "preserve_correspondence",
    "runtime_efficiency",
]

_UNSUPPORTED_LEGACY_OPERATIONS = {
    "computeExtents",
    "decimateMeshes",
    "generateNormals",
    "manifoldMeshes",
    "merge",
    "meshCleanup",
    "triangulateMeshes",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FLOAT32_PACKAGING_REL_TOLERANCE = 1e-7
_MIN_SEMANTIC_GEOMETRY_TOLERANCE = 1e-12
_NON_BLOCKING_MISSING_ASSET_SUFFIXES = frozenset(
    {
        ".bmp",
        ".exr",
        ".hdr",
        ".jpeg",
        ".jpg",
        ".png",
        ".tga",
        ".tif",
        ".tiff",
        ".tx",
        ".webp",
    }
)


class _SourceDigestMismatch(ValueError):
    """Raised when a digest-bound optimizer input changed after validation."""


@dataclass(frozen=True)
class _SemanticMeshSnapshot:
    source_ids: tuple[int, ...]
    face_geometry: dict[int, tuple[tuple[float, float, float], ...]]
    subdivision_scheme: str
    orientation: str
    double_sided: bool
    hole_indices: tuple[int, ...]
    visibility: str
    purpose: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_source_digest(
    source: Path,
    expected_sha256: str | None,
    *,
    phase: str,
) -> None:
    if expected_sha256 is None:
        return
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected_source_sha256 must be a lowercase SHA-256 digest")
    observed = _file_sha256(source)
    if observed != expected_sha256:
        raise _SourceDigestMismatch(
            f"Digest-bound Geometry source changed {phase}; "
            f"expected={expected_sha256}, observed={observed}"
        )


def _copy_digest_bound_source(
    source: Path,
    output: Path,
    expected_sha256: str | None,
) -> None:
    _assert_source_digest(source, expected_sha256, phase="before copy")
    if source != output:
        output_suffix = output.suffix.lower()
        requires_composed_export = _usd_requires_composed_export(source, output)
        if output_suffix == ".usdc" and (
            not _has_binary_usdc_header(source) or requires_composed_export
        ):
            _export_composed_usd(source, output, file_format="usdc")
        elif output_suffix == ".usda" and (
            _has_binary_usdc_header(source) or requires_composed_export
        ):
            _export_composed_usd(source, output, file_format="usda")
        elif output_suffix == ".usd" and requires_composed_export:
            _export_composed_usd(
                source,
                output,
                file_format="usdc" if _has_binary_usdc_header(source) else "usda",
            )
        else:
            shutil.copy2(source, output)
    try:
        _assert_source_digest(source, expected_sha256, phase="during copy")
        if source != output and _file_sha256(output) == _file_sha256(source):
            _assert_source_digest(output, expected_sha256, phase="while copying output")
    except _SourceDigestMismatch:
        if source != output:
            output.unlink(missing_ok=True)
        raise


def _has_binary_usdc_header(path: Path) -> bool:
    """Inspect only the crate header, including for large source layers."""

    with path.open("rb") as stream:
        return stream.read(8) == b"PXR-USDC"


def _unique_external_writer_path(output: Path, *, suffix: str) -> Path:
    """Reserve a unique same-directory name for an API that creates its file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _fsync_file(path: Path) -> None:
    # Windows' CRT rejects ``fsync`` on a read-only descriptor with EBADF.
    # The file was just created by the workflow, so open it write-capable on
    # every host and keep the durability boundary portable.
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    fsync_directory(path)


def _usd_requires_composed_export(source: Path, output: Path) -> bool:
    """Return whether relocating the source bytes would break composition."""

    if source.parent == output.parent:
        return False
    layers, assets, unresolved = _usd_dependencies(source)
    if unresolved or assets:
        return True
    source_resolved = source.resolve()
    return any(
        Path(layer.realPath).resolve() != source_resolved
        for layer in layers
        if not layer.anonymous and layer.realPath
    )


def _usd_dependencies(path: Path) -> tuple[list[Any], list[Any], list[Any]]:
    from pxr import UsdUtils

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not inspect USD dependencies: {path}") from exc
    return list(layers), list(assets), list(unresolved)


def _dependency_identifier(value: Any) -> str:
    path = getattr(value, "path", None)
    return str(path if path else value)


def _missing_appearance_dependencies(path: Path) -> tuple[str, ...]:
    """Return unresolved texture-like dependencies for explicit handoff evidence."""

    _layers, _assets, unresolved = _usd_dependencies(path)
    return tuple(
        sorted(
            {
                _dependency_identifier(item)
                for item in unresolved
                if Path(_dependency_identifier(item)).suffix.lower()
                in _NON_BLOCKING_MISSING_ASSET_SUFFIXES
            }
        )
    )


def _export_composed_usd(
    source: Path,
    output: Path,
    *,
    file_format: Literal["usda", "usdc"],
) -> None:
    """Export a composed stage while retaining visible missing texture evidence."""

    from pxr import Sdf, Usd

    _layers, _assets, unresolved = _usd_dependencies(source)
    blocking_unresolved = [
        item
        for item in unresolved
        if Path(_dependency_identifier(item)).suffix.lower()
        not in _NON_BLOCKING_MISSING_ASSET_SUFFIXES
    ]
    if blocking_unresolved:
        names = ", ".join(
            sorted(_dependency_identifier(item) for item in blocking_unresolved)
        )
        raise RuntimeError(
            "USD source has unresolved dependencies in its composition and cannot be "
            f"flattened: {names}"
        )
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open USD source for USDC export: {source}")
    flattened = stage.Flatten(addSourceFileComment=False)
    if flattened is None:
        raise RuntimeError(f"Could not flatten USD source for USDC export: {source}")
    cached_output_layer = Sdf.Layer.Find(str(output))
    temporary = _unique_external_writer_path(
        output,
        suffix=f".tmp.{file_format}",
    )
    try:
        if not flattened.Export(str(temporary), args={"format": file_format}):
            raise RuntimeError(
                f"OpenUSD could not export composed {file_format.upper()}: {output}"
            )
        if file_format == "usdc" and not _has_binary_usdc_header(temporary):
            raise RuntimeError(f"OpenUSD export was not binary USDC: {temporary}")
        if file_format == "usda" and _has_binary_usdc_header(temporary):
            raise RuntimeError(f"OpenUSD export was not ASCII USDA: {temporary}")
        if Sdf.Layer.OpenAsAnonymous(str(temporary)) is None:
            format_label = "binary USDC" if file_format == "usdc" else "ASCII USDA"
            raise RuntimeError(f"OpenUSD could not reopen {format_label}: {temporary}")
        _fsync_file(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
        if Sdf.Layer.OpenAsAnonymous(str(output)) is None:
            if source != output:
                output.unlink(missing_ok=True)
            raise RuntimeError(f"OpenUSD could not reopen exported USD: {output}")
        if cached_output_layer is not None:
            cached_output_layer.Reload(force=True)
    finally:
        temporary.unlink(missing_ok=True)


def _export_binary_usdc(source: Path, output: Path) -> None:
    """Compatibility wrapper for the composed binary export boundary."""

    _export_composed_usd(source, output, file_format="usdc")


def _ensure_binary_usdc(path: Path) -> None:
    if path.suffix.lower() != ".usdc" or _has_binary_usdc_header(path):
        return
    _export_composed_usd(path, path, file_format="usdc")


def _normalized_coordinate_system(
    value: Mapping[str, Any],
    *,
    subject: str,
) -> dict[str, Any]:
    try:
        meters_per_unit = float(value["meters_per_unit"])
        up_axis = str(value["up_axis"]).upper()
        forward_axis = str(value["forward_axis"]).upper()
        handedness = str(value["handedness"]).lower()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{subject} is incomplete or invalid") from exc
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError(f"{subject} meters_per_unit must be finite and positive")
    if up_axis not in {"X", "Y", "Z"}:
        raise ValueError(f"{subject} up_axis is invalid: {up_axis!r}")
    if forward_axis not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
        raise ValueError(f"{subject} forward_axis is invalid: {forward_axis!r}")
    if forward_axis[-1] == up_axis:
        raise ValueError(f"{subject} forward_axis cannot equal its up_axis")
    if handedness not in {"left", "right"}:
        raise ValueError(f"{subject} handedness is invalid: {handedness!r}")
    return {
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "forward_axis": forward_axis,
        "handedness": handedness,
    }


def retain_geometry_source_coordinate_system(
    usd_path: Path | str,
    report_path: Path | str,
    *,
    coordinate_system: Mapping[str, Any],
    immutable_source_usd: Path | str,
    expected_immutable_source_sha256: str,
) -> dict[str, Any]:
    """Bind provider-declared axes to a workflow-owned USD without touching source."""

    path = Path(usd_path).resolve(strict=True)
    source = Path(immutable_source_usd).resolve(strict=True)
    report = Path(report_path).resolve()
    if path == source or (
        path.stat().st_dev == source.stat().st_dev
        and path.stat().st_ino == source.stat().st_ino
    ):
        raise ValueError(
            "Geometry source coordinates require a distinct workflow-owned USD"
        )
    _assert_source_digest(
        source,
        expected_immutable_source_sha256,
        phase="before retaining provider coordinates on workflow output",
    )
    declared = _normalized_coordinate_system(
        coordinate_system,
        subject="Declared Geometry source coordinates",
    )

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Could not open workflow-owned Geometry stage: {path}")
    if not stage.HasAuthoredMetadata("upAxis") or not stage.HasAuthoredMetadata(
        "metersPerUnit"
    ):
        raise ValueError(
            "Workflow-owned provider USD must retain authored upAxis and metersPerUnit"
        )
    observed_up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    observed_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if observed_up_axis != declared["up_axis"]:
        raise ValueError(
            "Workflow-owned provider USD upAxis contradicts its source bundle"
        )
    if not math.isclose(
        observed_meters_per_unit,
        declared["meters_per_unit"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Workflow-owned provider USD metersPerUnit contradicts its source bundle"
        )

    custom_data = dict(stage.GetRootLayer().customLayerData)
    retained = custom_data.get("geometrySourceCoordinateSystem")
    if retained is not None:
        if not isinstance(retained, Mapping):
            raise ValueError(
                "Provider USD geometrySourceCoordinateSystem must be a dictionary"
            )
        observed_retained = _normalized_coordinate_system(
            retained,
            subject="Retained Geometry source coordinates",
        )
        if observed_retained != declared:
            raise ValueError(
                "Provider USD geometrySourceCoordinateSystem contradicts its source bundle"
            )
    removed_stale_canonical = "geometryCanonicalCoordinateSystem" in custom_data
    custom_data.pop("geometryCanonicalCoordinateSystem", None)
    custom_data["geometrySourceCoordinateSystem"] = declared
    stage.GetRootLayer().customLayerData = custom_data
    if not stage.GetRootLayer().Save():
        raise RuntimeError(
            f"Could not persist provider coordinates on Geometry stage: {path}"
        )
    _ensure_binary_usdc(path)
    _assert_source_digest(
        source,
        expected_immutable_source_sha256,
        phase="while retaining provider coordinates on workflow output",
    )
    payload = {
        "schema_version": "content-agent-workflows.geometry-source-coordinates.v1",
        "status": "pass",
        "asset_path": str(path),
        "immutable_source_usd": str(source),
        "immutable_source_sha256": expected_immutable_source_sha256,
        "output_sha256": _file_sha256(path),
        "coordinate_system": declared,
        "removed_stale_canonical_coordinate_system": removed_stale_canonical,
    }
    _write_json(report, payload)
    return {**payload, "report_path": str(report)}


def canonicalize_usd_stage_metrics(
    usd_path: Path | str,
    report_path: Path | str,
) -> dict[str, Any]:
    """Author a coherent Z-up, meter-native transform on a Geometry handoff."""

    path = Path(usd_path).resolve()
    report = Path(report_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open Geometry stage for normalization: {path}")
    roots = [prim for prim in stage.GetPseudoRoot().GetChildren() if prim.IsValid()]
    if not roots:
        raise RuntimeError("Geometry metric normalization requires a root prim")

    default_prim = stage.GetDefaultPrim()
    changes: list[str] = []
    if not default_prim or not default_prim.IsValid():
        if len(roots) == 1:
            stage.SetDefaultPrim(roots[0])
            default_prim = roots[0]
            changes.append(f"set_default_prim:{default_prim.GetPath()}")

    missing_metric_metadata = [
        name
        for name in ("upAxis", "metersPerUnit")
        if not stage.HasAuthoredMetadata(name)
    ]
    if missing_metric_metadata:
        raise RuntimeError(
            "Geometry cannot guess unauthored stage metrics; missing metadata: "
            f"{missing_metric_metadata}"
        )

    source_up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    source_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if source_up_axis not in {"X", "Y", "Z"}:
        raise RuntimeError(f"Unsupported Geometry source up axis: {source_up_axis!r}")
    if not math.isfinite(source_meters_per_unit) or source_meters_per_unit <= 0.0:
        raise RuntimeError(
            "Geometry source metersPerUnit must be finite and positive; observed "
            f"{source_meters_per_unit}"
        )

    needs_axis = source_up_axis != "Z"
    needs_units = not math.isclose(
        source_meters_per_unit,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    transform_targets: list[Any] = []
    normalization_matrix = Gf.Matrix4d(1.0)
    if needs_axis or needs_units:
        authored_physics = sorted(
            str(attribute.GetPath())
            for prim in stage.TraverseAll()
            for attribute in prim.GetAttributes()
            if attribute.HasAuthoredValueOpinion()
            and (
                attribute.GetName().startswith("physics:")
                or attribute.GetName().startswith("physx")
                or ":physics:" in attribute.GetName()
            )
        )
        if authored_physics:
            raise RuntimeError(
                "Geometry cannot safely normalize stage metrics after physics "
                "authoring; normalize before Physics or remove authored physics: "
                f"{authored_physics[:8]}"
            )
        pending = list(reversed(roots))
        while pending:
            prim = pending.pop()
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                transform_targets.append(xformable)
                continue
            pending.extend(
                reversed([child for child in prim.GetChildren() if child.IsValid()])
            )
        if not transform_targets:
            raise RuntimeError(
                "Geometry stage has no xformable root branch to normalize"
            )
        if source_up_axis == "Y":
            normalization_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0))
            changes.append("normalize_up_axis:Y->Z")
        elif source_up_axis == "X":
            normalization_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), -90.0))
            changes.append("normalize_up_axis:X->Z")
        if needs_units:
            scale = Gf.Matrix4d(1.0)
            scale.SetScale(source_meters_per_unit)
            normalization_matrix = normalization_matrix * scale
            changes.append(
                f"normalize_meters_per_unit:{source_meters_per_unit:.17g}->1"
            )
        for xformable in transform_targets:
            prim = xformable.GetPrim()
            attribute = prim.GetAttribute("xformOp:transform:geometryStageMetrics")
            op = (
                UsdGeom.XformOp(attribute)
                if attribute and attribute.IsValid()
                else xformable.AddTransformOp(
                    UsdGeom.XformOp.PrecisionDouble,
                    "geometryStageMetrics",
                )
            )
            op.Set(normalization_matrix)
            order_attr = xformable.GetXformOpOrderAttr()
            op_name = op.GetOpName()
            op_order = [token for token in order_attr.Get() or [] if token != op_name]
            insert_at = 1 if op_order and str(op_order[0]) == "!resetXformStack!" else 0
            op_order.insert(insert_at, op_name)
            order_attr.Set(op_order)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    custom_data = dict(stage.GetRootLayer().customLayerData)
    source_coordinates = custom_data.get("geometrySourceCoordinateSystem")
    canonical_coordinates: dict[str, Any] | None = None
    if isinstance(source_coordinates, dict):
        signed_axes = {
            "+X": Gf.Vec3d(1.0, 0.0, 0.0),
            "-X": Gf.Vec3d(-1.0, 0.0, 0.0),
            "+Y": Gf.Vec3d(0.0, 1.0, 0.0),
            "-Y": Gf.Vec3d(0.0, -1.0, 0.0),
            "+Z": Gf.Vec3d(0.0, 0.0, 1.0),
            "-Z": Gf.Vec3d(0.0, 0.0, -1.0),
        }
        source_forward = str(source_coordinates.get("forward_axis") or "").upper()
        if source_forward in signed_axes:
            transformed = normalization_matrix.TransformDir(signed_axes[source_forward])
            components = [float(transformed[index]) for index in range(3)]
            dominant_index = max(range(3), key=lambda index: abs(components[index]))
            if abs(components[dominant_index]) > 1e-12 and all(
                abs(component) <= 1e-9
                for index, component in enumerate(components)
                if index != dominant_index
            ):
                canonical_forward = (
                    f"{'+' if components[dominant_index] > 0.0 else '-'}"
                    f"{'XYZ'[dominant_index]}"
                )
                canonical_coordinates = {
                    "meters_per_unit": 1.0,
                    "up_axis": "Z",
                    "forward_axis": canonical_forward,
                    "handedness": str(
                        source_coordinates.get("handedness") or "right"
                    ).lower(),
                }
                custom_data["geometryCanonicalCoordinateSystem"] = canonical_coordinates
                stage.GetRootLayer().customLayerData = custom_data

    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"OpenUSD could not save normalized Geometry stage: {path}")
    _ensure_binary_usdc(path)
    payload = {
        "schema_version": "content-agent-workflows.geometry-stage-metrics.v1",
        "asset_path": str(path),
        "status": "pass",
        "source_up_axis": source_up_axis,
        "source_meters_per_unit": source_meters_per_unit,
        "target_up_axis": "Z",
        "target_meters_per_unit": 1.0,
        "canonical_coordinate_system": canonical_coordinates,
        "default_prim": (
            str(default_prim.GetPath())
            if default_prim and default_prim.IsValid()
            else None
        ),
        "normalized_root_branches": [
            str(xformable.GetPrim().GetPath()) for xformable in transform_targets
        ]
        if needs_axis or needs_units
        else [],
        "changes": changes,
    }
    _write_json(report, payload)
    return {**payload, "report_path": str(report)}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _renderable_bounds_m(path: Path) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
        world_range = cache.ComputeWorldBound(root).ComputeAlignedRange()
        if world_range.IsEmpty():
            return {"status": "not_evaluated", "reason": "renderable bounds are empty"}
        minimum = world_range.GetMin()
        maximum = world_range.GetMax()
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        minimum_m = [float(minimum[index]) * meters_per_unit for index in range(3)]
        maximum_m = [float(maximum[index]) * meters_per_unit for index in range(3)]
        extent_m = [maximum_m[index] - minimum_m[index] for index in range(3)]
        center_m = [(maximum_m[index] + minimum_m[index]) * 0.5 for index in range(3)]
        return {
            "status": "measured",
            "minimum_m": minimum_m,
            "maximum_m": maximum_m,
            "extent_m": extent_m,
            "center_m": center_m,
        }
    except Exception as exc:
        return {
            "status": "not_evaluated",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _geometry_fidelity_check(source: Path, output: Path) -> dict[str, Any]:
    from geometry_repair.fidelity import compare_geometry
    from geometry_repair.models import RepairBudgets

    report_path = output.with_suffix(output.suffix + ".fidelity.json")
    try:
        report = compare_geometry(
            source,
            output,
            drift_band="conservative",
            budgets=RepairBudgets(),
            output_path=report_path,
        )
    except Exception as exc:
        return {
            "status": "not_evaluated",
            "passed": None,
            "reason": f"{type(exc).__name__}: {exc}",
            "report_path": str(report_path),
            "source": _renderable_bounds_m(source),
            "output": _renderable_bounds_m(output),
        }
    payload = report.model_dump(mode="json")
    if report.exact_world_geometry_match:
        source_bounds = output_bounds = {
            "status": "proven_equal",
            "reason": (
                "Exact world-space vertices, triangles, semantic paths, transforms, and "
                "material bindings were preserved."
            ),
        }
    else:
        source_bounds = _renderable_bounds_m(source)
        output_bounds = _renderable_bounds_m(output)
    return {
        **payload,
        "passed": report.status == "pass",
        "report_path": str(report_path),
        "source": source_bounds,
        "output": output_bounds,
    }


def _prim_and_ancestors(prim: Any) -> list[Any]:
    chain: list[Any] = []
    current = prim
    while current and not current.IsPseudoRoot():
        chain.append(current)
        current = current.GetParent()
    return chain


def _has_authored_composition_arc(prim: Any) -> bool:
    """Return whether a protected prim or ancestor owns a composition arc."""

    return bool(
        prim.HasAuthoredReferences()
        or prim.HasAuthoredPayloads()
        or prim.HasAuthoredInherits()
        or prim.HasAuthoredSpecializes()
        or prim.HasVariantSets()
    )


def _semantic_face_geometry_matches(
    source: dict[int, tuple[tuple[float, float, float], ...]],
    output: dict[int, tuple[tuple[float, float, float], ...]],
) -> bool:
    """Compare ordered face geometry within float32 packaging precision."""

    if source.keys() != output.keys():
        return False
    source_points = [point for triangle in source.values() for point in triangle]
    if not source_points:
        return False
    minimum = [min(point[axis] for point in source_points) for axis in range(3)]
    maximum = [max(point[axis] for point in source_points) for axis in range(3)]
    diagonal = math.dist(minimum, maximum)
    tolerance = max(
        diagonal * _FLOAT32_PACKAGING_REL_TOLERANCE,
        _MIN_SEMANTIC_GEOMETRY_TOLERANCE,
    )
    return all(
        len(source[source_id]) == len(output[source_id])
        and all(
            abs(source_value - output_value) <= tolerance
            for source_point, output_point in zip(
                source[source_id],
                output[source_id],
                strict=True,
            )
            for source_value, output_value in zip(
                source_point,
                output_point,
                strict=True,
            )
        )
        for source_id in source
    )


def _boundable_prim_inventory(
    path: Path,
    *,
    role: Literal["source", "optimized"],
) -> tuple[dict[str, str], list[str]]:
    """Inventory every boundable prim and reject schemas fidelity cannot measure."""

    inventory: dict[str, str] = {}
    failures: list[str] = []
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise RuntimeError(f"Could not open {role} semantic USD stage")
        supported_schemas = (
            UsdGeom.Mesh,
            UsdGeom.Cube,
            UsdGeom.Sphere,
            UsdGeom.Cylinder,
            UsdGeom.Cone,
            UsdGeom.Capsule,
        )
        unsupported: list[str] = []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            prim_path = str(prim.GetPath())
            type_name = str(prim.GetTypeName())
            inventory[prim_path] = type_name
            if not any(prim.IsA(schema) for schema in supported_schemas):
                unsupported.append(f"{prim_path} ({type_name})")
        if unsupported:
            failures.append(
                f"{role} semantic stage contains boundable geometry unsupported by "
                f"the fidelity authority: {sorted(unsupported)}"
            )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    return inventory, failures


def _semantic_stage_snapshots(
    path: Path,
    protected_prim_paths: list[str],
    *,
    role: Literal["source", "optimized"],
) -> tuple[dict[str, _SemanticMeshSnapshot], list[str]]:
    """Read protected meshes whose relevant ancestry is root-authored."""

    snapshots: dict[str, _SemanticMeshSnapshot] = {}
    failures: list[str] = []
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt

        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise RuntimeError(f"Could not open {role} protected semantic USD stage")
        root_layer = stage.GetRootLayer()

        for prim_path in protected_prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsA(UsdGeom.Mesh):
                failures.append(f"{role} semantic mesh is missing: {prim_path}")
                continue
            time_sampled_attribute = next(
                (
                    str(attribute.GetPath())
                    for ancestor in _prim_and_ancestors(prim)
                    for attribute in ancestor.GetAttributes()
                    if attribute.GetNumTimeSamples() != 0
                ),
                None,
            )
            if time_sampled_attribute is not None:
                failures.append(
                    f"{role} semantic mesh must be static; time-sampled attribute "
                    f"found at {time_sampled_attribute}"
                )
                continue
            composed_prim = prim
            directly_authored = True
            while composed_prim and not composed_prim.IsPseudoRoot():
                prim_stack = composed_prim.GetPrimStack()
                if (
                    _has_authored_composition_arc(composed_prim)
                    or not prim_stack
                    or any(
                        spec.layer.identifier != root_layer.identifier
                        or spec.path != composed_prim.GetPath()
                        for spec in prim_stack
                    )
                ):
                    directly_authored = False
                    break
                composed_prim = composed_prim.GetParent()
            if not directly_authored:
                failures.append(
                    f"{role} semantic mesh is composed instead of root-authored: "
                    f"{prim_path}"
                )
                continue

            try:
                source_ids_attr = prim.GetAttribute("meshSegmentation:sourceFaceIds")
                if not source_ids_attr:
                    raise ValueError(
                        f"{role} semantic mesh lacks source-face provenance: "
                        f"{prim_path}"
                    )
                if source_ids_attr.GetTypeName() != Sdf.ValueTypeNames.UIntArray:
                    raise ValueError(
                        f"{role} semantic mesh source-face provenance must be exact "
                        "UIntArray at "
                        f"{prim_path}; got {source_ids_attr.GetTypeName()}"
                    )
                if source_ids_attr.GetNumTimeSamples() != 0:
                    raise ValueError(
                        f"{role} semantic mesh source-face provenance must be "
                        f"default-only at {prim_path}"
                    )
                source_ids = source_ids_attr.Get()
                if type(source_ids) is not Vt.UIntArray:
                    raise ValueError(
                        f"{role} semantic mesh source-face provenance must resolve to "
                        f"exact Vt.UIntArray at {prim_path}; got "
                        f"{type(source_ids).__name__}"
                    )

                mesh = UsdGeom.Mesh(prim)
                counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
                indices = [
                    int(value)
                    for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])
                ]
                raw_points = list(mesh.GetPointsAttr().Get() or [])
                if len(counts) != len(source_ids) or any(
                    int(count) != 3 for count in counts
                ):
                    raise ValueError(
                        f"semantic mesh is not one triangle per source face: {prim_path}"
                    )
                if len(indices) != len(counts) * 3 or not raw_points:
                    raise ValueError(f"semantic mesh topology is invalid: {prim_path}")
                matrix = UsdGeom.XformCache(
                    Usd.TimeCode.Default()
                ).GetLocalToWorldTransform(prim)
                points: list[tuple[float, float, float]] = []
                for raw_point in raw_points:
                    world_point = matrix.Transform(
                        Gf.Vec3d(*(float(value) for value in raw_point))
                    )
                    float32_point = Gf.Vec3f(world_point)
                    point = tuple(float(value) for value in float32_point)
                    if not all(math.isfinite(value) for value in point):
                        raise ValueError(
                            f"semantic mesh has non-finite world geometry: {prim_path}"
                        )
                    points.append(point)
                source_id_values = tuple(int(value) for value in source_ids)
                face_map: dict[int, tuple[tuple[float, float, float], ...]] = {}
                for face_index, source_id in enumerate(source_ids):
                    source_id_value = int(source_id)
                    offset = face_index * 3
                    triangle = indices[offset : offset + 3]
                    if source_id_value in face_map:
                        raise ValueError(
                            f"semantic mesh has duplicate source face ID at {prim_path}"
                        )
                    if any(index < 0 or index >= len(points) for index in triangle):
                        raise ValueError(
                            f"semantic mesh has an invalid point index: {prim_path}"
                        )
                    face_map[source_id_value] = tuple(
                        points[index] for index in triangle
                    )
                imageable = UsdGeom.Imageable(prim)
                snapshots[prim_path] = _SemanticMeshSnapshot(
                    source_ids=source_id_values,
                    face_geometry=face_map,
                    subdivision_scheme=str(mesh.GetSubdivisionSchemeAttr().Get()),
                    orientation=str(mesh.GetOrientationAttr().Get()),
                    double_sided=bool(mesh.GetDoubleSidedAttr().Get()),
                    hole_indices=tuple(
                        int(value) for value in (mesh.GetHoleIndicesAttr().Get() or [])
                    ),
                    visibility=str(imageable.ComputeVisibility()),
                    purpose=str(imageable.ComputePurpose()),
                )
            except ValueError as exc:
                failures.append(str(exc))
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    return snapshots, failures


def _semantic_prim_boundary_check(
    source: Path,
    output: Path,
    protected_prim_paths: list[str],
) -> dict[str, Any]:
    """Prove that locked segmentation membership stayed at the same prim paths."""

    if not protected_prim_paths:
        return {"status": "not_requested", "passed": True, "prim_paths": []}
    source_snapshots, source_failures = _semantic_stage_snapshots(
        source,
        protected_prim_paths,
        role="source",
    )
    output_snapshots, output_failures = _semantic_stage_snapshots(
        output,
        protected_prim_paths,
        role="optimized",
    )
    failures = [*source_failures, *output_failures]
    source_inventory, source_inventory_failures = _boundable_prim_inventory(
        source,
        role="source",
    )
    output_inventory, output_inventory_failures = _boundable_prim_inventory(
        output,
        role="optimized",
    )
    failures.extend(source_inventory_failures)
    failures.extend(output_inventory_failures)
    if source_inventory != output_inventory:
        added = sorted(set(output_inventory).difference(source_inventory))
        removed = sorted(set(source_inventory).difference(output_inventory))
        changed = sorted(
            path
            for path in set(source_inventory).intersection(output_inventory)
            if source_inventory[path] != output_inventory[path]
        )
        failures.append(
            "optimized boundable prim inventory changed; "
            f"added={added}, removed={removed}, type_changed={changed}"
        )
    for prim_path in protected_prim_paths:
        source_snapshot = source_snapshots.get(prim_path)
        output_snapshot = output_snapshots.get(prim_path)
        if source_snapshot is None or output_snapshot is None:
            continue
        if source_snapshot.source_ids != output_snapshot.source_ids:
            failures.append(
                "optimized meshSegmentation:sourceFaceIds values changed at prim: "
                f"{prim_path}"
            )
        if not _semantic_face_geometry_matches(
            source_snapshot.face_geometry,
            output_snapshot.face_geometry,
        ):
            failures.append(
                "optimized semantic source-face membership or ordered face geometry "
                f"changed at prim: {prim_path}"
            )
        source_render_state = (
            source_snapshot.subdivision_scheme,
            source_snapshot.orientation,
            source_snapshot.double_sided,
            source_snapshot.hole_indices,
            source_snapshot.visibility,
            source_snapshot.purpose,
        )
        output_render_state = (
            output_snapshot.subdivision_scheme,
            output_snapshot.orientation,
            output_snapshot.double_sided,
            output_snapshot.hole_indices,
            output_snapshot.visibility,
            output_snapshot.purpose,
        )
        if source_render_state != output_render_state:
            failures.append(
                "optimized protected semantic mesh render state changed at prim: "
                f"{prim_path}; source={source_render_state}, "
                f"output={output_render_state}"
            )
    return {
        "status": "pass" if not failures else "fail",
        "passed": not failures,
        "prim_paths": list(protected_prim_paths),
        "failures": failures,
    }


def _quarantine_optimizer_output(
    output: Path,
    *,
    suffix: str,
) -> tuple[Path, Path | None]:
    """Move a rejected USD and its shared optimizer sidecar out of active paths."""

    rejected_output = output.with_name(f"{output.stem}.{suffix}{output.suffix}")
    rejected_output.unlink(missing_ok=True)
    output.replace(rejected_output)

    shared_metadata = output.with_suffix(".metadata.json")
    rejected_shared_metadata: Path | None = None
    if shared_metadata.exists():
        rejected_shared_metadata = rejected_output.with_suffix(".metadata.json")
        rejected_shared_metadata.unlink(missing_ok=True)
        shared_metadata.replace(rejected_shared_metadata)
    return rejected_output, rejected_shared_metadata


def optimize_geometry(
    *,
    source_usd: Path | str,
    output_usd: Path | str,
    policy: GeometryOptimizationPolicy = "preserve_correspondence",
    backend: Literal["local", "remote"] = "local",
    optimization_config: dict[str, Any] | None = None,
    protected_semantic_prim_paths: list[str] | None = None,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Normalize geometry through the optimizer contract used by usd-cli."""

    source = Path(source_usd).resolve()
    output = Path(output_usd).resolve()
    protected_paths = list(dict.fromkeys(protected_semantic_prim_paths or []))
    if not source.exists():
        raise FileNotFoundError(f"Geometry source USD not found: {source}")
    _assert_source_digest(source, expected_source_sha256, phase="before optimization")
    missing_asset_dependencies = _missing_appearance_dependencies(source)
    if protected_paths:
        _source_snapshots, source_lock_failures = _semantic_stage_snapshots(
            source,
            protected_paths,
            role="source",
        )
        _source_inventory, source_inventory_failures = _boundable_prim_inventory(
            source,
            role="source",
        )
        source_lock_failures.extend(source_inventory_failures)
        if source_lock_failures:
            raise ValueError(
                "Protected semantic source failed the optimizer lock contract: "
                + "; ".join(source_lock_failures)
            )
    metadata_path = output.with_suffix(output.suffix + ".optimization.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    advanced_config = dict(optimization_config or {})
    unsupported = sorted(_UNSUPPORTED_LEGACY_OPERATIONS.intersection(advanced_config))
    settings = advanced_config.get("scene_optimizer_settings")
    if isinstance(settings, dict):
        unsupported.extend(
            sorted(_UNSUPPORTED_LEGACY_OPERATIONS.intersection(settings))
        )
    if unsupported:
        raise ValueError(
            "Unsupported Scene Optimizer operation names: "
            + ", ".join(sorted(set(unsupported)))
        )

    metadata: dict[str, Any]
    if policy == "skip":
        _copy_digest_bound_source(source, output, expected_source_sha256)
        metadata = {
            "source_usd": str(source),
            "output_usd": str(output),
            "backend": "none",
            "policy": policy,
            "status": "skipped",
            "artifact_role": "normalized_copy",
            "optimization_config": {},
            "protected_semantic_prim_paths": protected_paths,
            "expected_source_sha256": expected_source_sha256,
            "missing_asset_dependencies": list(missing_asset_dependencies),
        }
        _write_json(metadata_path, metadata)
        return {**metadata, "metadata_path": str(metadata_path)}

    if protected_paths and policy == "preserve_correspondence":
        _copy_digest_bound_source(source, output, expected_source_sha256)
        metadata = {
            "source_usd": str(source),
            "output_usd": str(output),
            "backend": "none",
            "policy": policy,
            "status": "semantic_lock_skip",
            "artifact_role": "normalized_copy",
            "optimization_config": {},
            "protected_semantic_prim_paths": protected_paths,
            "expected_source_sha256": expected_source_sha256,
            "missing_asset_dependencies": list(missing_asset_dependencies),
            "degraded_reason": (
                "Scene Optimizer split does not preserve the locked "
                "meshSegmentation:sourceFaceIds contract, and no other operation is "
                "enabled by preserve_correspondence."
            ),
        }
        _write_json(metadata_path, metadata)
        return {**metadata, "metadata_path": str(metadata_path)}

    if source == output:
        raise ValueError(
            "Shared optimization requires a distinct output path so stale or "
            "partially written data cannot replace the source-of-record."
        )
    output.unlink(missing_ok=True)
    output.with_suffix(".metadata.json").unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)

    preserve_correspondence = policy == "preserve_correspondence"
    options = OptimizerRequestOptions(
        optimize=True,
        optimizer_backend=OptimizerBackend(backend),
        flatten_prototypes=not preserve_correspondence,
        enable_deinstance=not preserve_correspondence,
        enable_split=not protected_paths,
        enable_deduplicate=not preserve_correspondence,
        optimization_config=advanced_config,
    )
    resolved_config = options.resolved_optimization_config()

    try:
        context = OptimizeUSDTask().run(
            {
                "input_usd_path": str(source),
                "output_usd_path": str(output),
                "optimization_config": resolved_config,
            }
        )
        _assert_source_digest(
            source,
            expected_source_sha256,
            phase="during optimization",
        )
        task_metadata = dict(context.get("optimization_metadata") or {})
        requested_backend = str(task_metadata.get("requested_backend") or backend)
        actual_backend = str(task_metadata.get("actual_backend") or backend)
        metadata = {
            "source_usd": str(source),
            "output_usd": str(output),
            "backend": f"world_understanding.OptimizeUSDTask.{actual_backend}",
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "fallback_used": bool(task_metadata.get("fallback_used")),
            "fallback_reason": task_metadata.get("fallback_reason"),
            "policy": policy,
            "status": "completed",
            "artifact_role": "optimized_geometry",
            "optimization_config": resolved_config,
            "shared_optimizer_metadata": task_metadata,
            "protected_semantic_prim_paths": protected_paths,
            "expected_source_sha256": expected_source_sha256,
            "missing_asset_dependencies": list(missing_asset_dependencies),
        }
        if output.exists() and bool(context.get("optimization_success")):
            _ensure_binary_usdc(output)
            fidelity = _geometry_fidelity_check(source, output)
            if fidelity.get("passed") is not True:
                rejected_output, rejected_shared_metadata = (
                    _quarantine_optimizer_output(output, suffix="fidelity-rejected")
                )
                _copy_digest_bound_source(source, output, expected_source_sha256)
                metadata.update(
                    {
                        "status": "fidelity_fallback",
                        "artifact_role": "normalized_copy",
                        "degraded_reason": (
                            "Scene Optimizer output did not pass the complete source-relative "
                            "geometry fidelity contract."
                        ),
                        "geometry_fidelity": fidelity,
                        "rejected_output_usd": str(rejected_output),
                        "rejected_shared_metadata_path": (
                            str(rejected_shared_metadata)
                            if rejected_shared_metadata is not None
                            else None
                        ),
                    }
                )
                _write_json(metadata_path, metadata)
                return {**metadata, "metadata_path": str(metadata_path)}
            metadata["geometry_fidelity"] = fidelity
            semantic_boundaries = _semantic_prim_boundary_check(
                source,
                output,
                protected_paths,
            )
            metadata["semantic_prim_boundaries"] = semantic_boundaries
            if semantic_boundaries.get("passed") is not True:
                rejected_output, rejected_shared_metadata = (
                    _quarantine_optimizer_output(
                        output,
                        suffix="semantic-boundary-rejected",
                    )
                )
                _copy_digest_bound_source(source, output, expected_source_sha256)
                metadata.update(
                    {
                        "status": "semantic_fidelity_fallback",
                        "artifact_role": "normalized_copy",
                        "degraded_reason": (
                            "Scene Optimizer output changed locked semantic prim paths "
                            "or source-face membership."
                        ),
                        "rejected_output_usd": str(rejected_output),
                        "rejected_shared_metadata_path": (
                            str(rejected_shared_metadata)
                            if rejected_shared_metadata is not None
                            else None
                        ),
                    }
                )
                _write_json(metadata_path, metadata)
                return {**metadata, "metadata_path": str(metadata_path)}
            _assert_source_digest(
                source,
                expected_source_sha256,
                phase="while validating optimizer output",
            )
            shared_metadata_path = output.with_suffix(".metadata.json")
            if shared_metadata_path.exists():
                metadata["shared_metadata_path"] = str(shared_metadata_path)
            _write_json(metadata_path, metadata)
            return {**metadata, "metadata_path": str(metadata_path)}
        raise RuntimeError("OptimizeUSDTask completed without writing output USD.")
    except _SourceDigestMismatch:
        # The runtime-optimization path currently requires distinct paths, but
        # preserve the source-of-record if that invariant is ever relaxed.
        if source != output:
            output.unlink(missing_ok=True)
            output.with_suffix(".metadata.json").unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        metadata = {
            "source_usd": str(source),
            "output_usd": str(output),
            "backend": f"world_understanding.OptimizeUSDTask.{backend}",
            "policy": policy,
            "status": "optimization_unavailable",
            "artifact_role": "normalized_copy",
            "optimization_config": resolved_config,
            "degraded_reason": f"{type(exc).__name__}: {exc}",
            "protected_semantic_prim_paths": protected_paths,
            "expected_source_sha256": expected_source_sha256,
            "missing_asset_dependencies": list(missing_asset_dependencies),
        }

    _copy_digest_bound_source(source, output, expected_source_sha256)
    _write_json(metadata_path, metadata)
    return {**metadata, "metadata_path": str(metadata_path)}


def author_missing_mesh_normals(
    usd_path: Path | str,
    report_path: Path | str,
    *,
    crease_angle_deg: float = 40.0,
) -> dict[str, Any]:
    """Author crease-aware normals without changing mesh points or topology."""

    path = Path(usd_path).resolve()
    report = Path(report_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"USD file not found: {path}")

    modified: list[str] = []
    skipped: list[dict[str, str]] = []
    unchanged = 0
    try:
        from pxr import Gf, Usd, UsdGeom, Vt

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {path}")
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            authored = mesh.GetNormalsAttr().Get() or []
            primvar_normals_attr = prim.GetAttribute("primvars:normals")
            primvar_normals = (
                primvar_normals_attr.Get()
                if primvar_normals_attr
                and primvar_normals_attr.HasAuthoredValueOpinion()
                else None
            )
            if authored or primvar_normals:
                unchanged += 1
                continue
            points: list[tuple[float, float, float]] = [
                (float(point[0]), float(point[1]), float(point[2]))
                for point in (mesh.GetPointsAttr().Get() or [])
            ]
            counts = [
                int(value) for value in (mesh.GetFaceVertexCountsAttr().Get() or [])
            ]
            indices = [
                int(value) for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])
            ]
            normals = _crease_aware_face_varying_normals(
                points,
                counts,
                indices,
                crease_angle_deg=crease_angle_deg,
                left_handed=(
                    mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded
                ),
            )
            if len(normals) != len(indices):
                skipped.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "reason": "invalid or degenerate mesh topology",
                    }
                )
                continue
            mesh.CreateNormalsAttr(
                Vt.Vec3fArray([Gf.Vec3f(*normal) for normal in normals])
            )
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            modified.append(str(prim.GetPath()))
        stage.GetRootLayer().Save()
        status = "warning" if skipped else "pass"
        error = None
    except Exception as exc:
        status = "warning"
        error = f"{type(exc).__name__}: {exc}"
        skipped.append({"prim_path": "", "reason": error})

    payload = {
        "schema_version": "content-agent-workflows.geometry-mesh-normalization.v1",
        "asset_path": str(path),
        "status": status,
        "crease_angle_deg": float(crease_angle_deg),
        "modified_mesh_count": len(modified),
        "modified_meshes": modified,
        "unchanged_mesh_count": unchanged,
        "skipped": skipped,
        "points_or_topology_changed": False,
        "error": error,
    }
    _write_json(report, payload)
    return {**payload, "report_path": str(report)}


def _crease_aware_face_varying_normals(
    points: list[tuple[float, float, float]],
    counts: list[int],
    indices: list[int],
    *,
    crease_angle_deg: float,
    left_handed: bool,
) -> list[tuple[float, float, float]]:
    if not points or not counts or sum(counts) != len(indices) or min(counts) < 3:
        return []
    if any(index < 0 or index >= len(points) for index in indices):
        return []

    faces: list[list[int]] = []
    face_normals: list[tuple[float, float, float] | None] = []
    face_area_vectors: list[tuple[float, float, float]] = []
    offset = 0
    sign = -1.0 if left_handed else 1.0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        faces.append(face)
        origin = points[face[0]]
        nx = ny = nz = 0.0
        for corner in range(1, len(face) - 1):
            first = points[face[corner]]
            second = points[face[corner + 1]]
            ux, uy, uz = (first[axis] - origin[axis] for axis in range(3))
            vx, vy, vz = (second[axis] - origin[axis] for axis in range(3))
            nx += uy * vz - uz * vy
            ny += uz * vx - ux * vz
            nz += ux * vy - uy * vx
        nx, ny, nz = sign * nx, sign * ny, sign * nz
        magnitude = math.sqrt(nx * nx + ny * ny + nz * nz)
        face_area_vectors.append((nx, ny, nz))
        face_normals.append(
            None
            if magnitude <= 1e-18
            else (nx / magnitude, ny / magnitude, nz / magnitude)
        )

    incident: list[list[int]] = [[] for _ in points]
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            incident[vertex_index].append(face_index)

    # Zero-area faces still need one normal per face vertex for a valid USD
    # contract. Borrow a stable direction from adjacent valid faces; an
    # isolated zero-area face has no visible surface, so a fixed fallback is
    # sufficient without changing source topology.
    for face_index, normal in enumerate(face_normals):
        if normal is not None:
            continue
        adjacent = {
            adjacent_index
            for vertex_index in faces[face_index]
            for adjacent_index in incident[vertex_index]
            if face_normals[adjacent_index] is not None
        }
        sx = sum(face_area_vectors[index][0] for index in adjacent)
        sy = sum(face_area_vectors[index][1] for index in adjacent)
        sz = sum(face_area_vectors[index][2] for index in adjacent)
        magnitude = math.sqrt(sx * sx + sy * sy + sz * sz)
        face_normals[face_index] = (
            (0.0, 0.0, 1.0)
            if magnitude <= 1e-18
            else (sx / magnitude, sy / magnitude, sz / magnitude)
        )

    resolved_normals = [normal for normal in face_normals if normal is not None]
    if len(resolved_normals) != len(faces):
        return []
    cosine_limit = math.cos(math.radians(float(crease_angle_deg)))
    result: list[tuple[float, float, float]] = []
    for face_index, face in enumerate(faces):
        current = resolved_normals[face_index]
        for vertex_index in face:
            compatible = [
                adjacent
                for adjacent in incident[vertex_index]
                if sum(
                    current[axis] * resolved_normals[adjacent][axis]
                    for axis in range(3)
                )
                >= cosine_limit
            ]
            sx = sum(face_area_vectors[index][0] for index in compatible)
            sy = sum(face_area_vectors[index][1] for index in compatible)
            sz = sum(face_area_vectors[index][2] for index in compatible)
            magnitude = math.sqrt(sx * sx + sy * sy + sz * sz)
            result.append(
                current
                if magnitude <= 1e-18
                else (sx / magnitude, sy / magnitude, sz / magnitude)
            )
    return result


def inspect_usd_geometry(usd_path: Path | str) -> dict[str, Any]:
    """Return a lightweight usd-cli/USD inspection summary."""

    path = Path(usd_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"USD file not found: {path}")
    try:
        from world_understanding.utils.usd.stage import get_stage_info, load_stage

        from content_agent_workflows.physics.scene_ops import inspect_mesh_candidates

        inspection: dict[str, Any] = dict(inspect_mesh_candidates(path))
        stage_info = get_stage_info(load_stage(path))
        inspection["stage_info"] = {
            "up_axis": str(stage_info.get("up_axis") or "") or None,
            "meters_per_unit": float(stage_info["meters_per_unit"])
            if stage_info.get("meters_per_unit") is not None
            else None,
            "default_prim": stage_info.get("default_prim"),
            "prim_count": stage_info.get("prim_count"),
        }
        return inspection
    except Exception as exc:
        return {
            "asset": str(path),
            "candidate_count": 0,
            "candidates": [],
            "status": "inspection_unavailable",
            "error": str(exc),
        }
