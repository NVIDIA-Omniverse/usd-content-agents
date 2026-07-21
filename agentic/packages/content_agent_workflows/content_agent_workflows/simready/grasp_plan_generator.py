# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic, geometry-proven SimReady GSP.001 plans.

The generator does not infer grasp semantics. It proves only that a nonzero line
lies on one exact composed surface triangle under the default prim. Mesh
polygons are triangulated as the fan ``(v0, vi, vi+1)``. Analytic
``UsdGeomCube`` faces use a fixed six-face, two-triangle topology derived from
the composed ``size`` value. Candidate triangles are ranked by descending
finite area in default-prim-local coordinates, then by surface prim path,
geometry kind, face index, and triangle index. The two line endpoints use the
fixed interior barycentric coordinates ``(1/2, 1/4, 1/4)`` and
``(1/4, 1/2, 1/4)``.

Source files and their complete local dependency closure are hashed before and
after inspection and publication. Dependencies must be regular, non-symlink
files beneath the source asset's directory. There are deliberately no model
calls, asset-name rules, geometry-size limits, bounding-box approximations, or
default widths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .models import (
    SIMREADY_GRASP_PLAN_ANALYTIC_GENERATOR_VERSION,
    SIMREADY_GRASP_PLAN_ANALYTIC_SCHEMA_VERSION,
    SIMREADY_GRASP_PLAN_COMPOSED_GENERATOR_VERSION,
    SIMREADY_GRASP_PLAN_COMPOSED_SCHEMA_VERSION,
    SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION,
    SIMREADY_GRASP_PLAN_GENERATOR_VERSION,
    SIMREADY_GRASP_PLAN_SCHEMA_VERSION,
    SimReadyGraspLinePlan,
    SimReadyGraspPlan,
    SimReadyGraspPlanAnalyticMachineProvenance,
    SimReadyGraspPlanAnalyticProofChecks,
    SimReadyGraspPlanAnalyticSurfaceProof,
    SimReadyGraspPlanComposedMachineProvenance,
    SimReadyGraspPlanComposedProofChecks,
    SimReadyGraspPlanDependencyProof,
    SimReadyGraspPlanMachineProofChecks,
    SimReadyGraspPlanMachineProvenance,
    SimReadyGraspPlanTriangleProof,
)

_FLOAT32_MAX = 3.4028234663852886e38
_FLOAT32_MIN_POSITIVE = 1.401298464324817e-45
_HASH_CHUNK_SIZE = 1024 * 1024
_GRASP_LINE_NAME = "grasp_identifier_machine_geometry"
_BARYCENTRIC_COORDINATES = (
    ("1/2", "1/4", "1/4"),
    ("1/4", "1/2", "1/4"),
)
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class GraspPlanGenerationError(ValueError):
    """Raised when a source cannot support a fail-closed geometry proof."""


@dataclass(frozen=True)
class SimReadyGraspPlanGenerationResult:
    """Published plan and deterministic publication evidence."""

    plan: SimReadyGraspPlan
    output_path: Path
    plan_sha256: str
    reused_output: bool
    reused_existing_grasp_line: bool = False


@dataclass(frozen=True)
class _DependencyRecord:
    path: Path
    relative_path: str
    role: Literal["source_asset", "dependency"]
    sha256: str

    def proof(self) -> SimReadyGraspPlanDependencyProof:
        return SimReadyGraspPlanDependencyProof(
            role=self.role,
            relative_path=self.relative_path,
            sha256=self.sha256,
        )


@dataclass(frozen=True)
class _DependencySnapshot:
    source_path: Path
    records: tuple[_DependencyRecord, ...]
    bundle_sha256: str

    @property
    def source_sha256(self) -> str:
        matches = [
            record.sha256 for record in self.records if record.role == "source_asset"
        ]
        if len(matches) != 1:  # pragma: no cover - internal invariant
            raise GraspPlanGenerationError(
                "Dependency snapshot does not contain exactly one source asset."
            )
        return matches[0]


@dataclass(frozen=True)
class _TriangleCandidate:
    surface_kind: Literal["mesh", "cube"]
    surface_prim_path: str
    face_index: int
    triangle_index: int
    point_indices: tuple[int, int, int]
    surface_local_points: tuple[tuple[float, float, float], ...]
    default_prim_local_points: tuple[tuple[float, float, float], ...]
    area_squared: float
    cube_size: float | None = None


def generate_simready_grasp_plan(
    asset_path: str | Path,
    output_path: str | Path,
    *,
    width: float,
) -> SimReadyGraspPlanGenerationResult:
    """Generate and atomically publish one machine geometry-proof grasp plan.

    ``width`` is mandatory and is interpreted in the stage's authored units.
    Existing output is reused only when its bytes exactly match the canonical
    plan. A different existing file is never overwritten.
    """

    explicit_width = _validated_width(width)
    source_path = _absolute_path(Path(asset_path).expanduser())
    destination = _absolute_path(Path(output_path).expanduser())
    if source_path == destination:
        raise GraspPlanGenerationError(
            "Grasp plan output path must differ from the source asset."
        )

    snapshot = _capture_dependency_snapshot(source_path)
    if destination in {record.path for record in snapshot.records}:
        raise GraspPlanGenerationError(
            "Grasp plan output path collides with the source dependency closure."
        )

    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except ImportError as exc:  # pragma: no cover - environment failure
        raise GraspPlanGenerationError(
            f"OpenUSD Python APIs are unavailable: {exc}"
        ) from exc

    try:
        stage = Usd.Stage.Open(str(source_path), load=Usd.Stage.LoadAll)
    except Exception as exc:  # pragma: no cover - OpenUSD exceptions vary
        raise GraspPlanGenerationError(
            f"Could not open source USD stage {source_path}: {exc}"
        ) from exc
    if stage is None:
        raise GraspPlanGenerationError(
            f"Could not open source USD stage: {source_path}"
        )

    _validate_layers_current(stage.GetUsedLayers(), Sdf=Sdf)
    default_prim = _validated_default_prim(stage)
    has_composed_instances = _validate_composed_instance_state(
        stage,
        default_prim,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    candidate = _select_surface_triangle(
        stage,
        default_prim,
        traverse_instance_proxies=has_composed_instances,
        Gf=Gf,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    line_points = _line_points(candidate.default_prim_local_points, Gf=Gf)
    grasp_line_path = default_prim.GetPath().AppendChild(_GRASP_LINE_NAME)
    reused_existing_grasp_line = False
    existing_grasp_line = stage.GetPrimAtPath(grasp_line_path)
    if existing_grasp_line:
        _verify_existing_machine_grasp_line(
            existing_grasp_line,
            expected_points=line_points,
            expected_width=explicit_width,
            Gf=Gf,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            Vt=Vt,
        )
        reused_existing_grasp_line = True

    _require_snapshot_unchanged(snapshot)
    plan = _build_plan(
        snapshot=snapshot,
        default_prim_path=str(default_prim.GetPath()),
        grasp_line_path=str(grasp_line_path),
        candidate=candidate,
        line_points=line_points,
        width=explicit_width,
        composed_instances=has_composed_instances,
    )
    plan_bytes = _canonical_plan_bytes(plan)
    _require_snapshot_unchanged(snapshot)

    reused_output = False
    try:
        reused_output = _publish_canonical_bytes(destination, plan_bytes)
        _validate_exact_readback(
            destination, expected_bytes=plan_bytes, expected_plan=plan
        )
        _require_snapshot_unchanged(snapshot)
    except BaseException:
        if not reused_output:
            _remove_matching_output(destination, plan_bytes)
        raise

    return SimReadyGraspPlanGenerationResult(
        plan=plan,
        output_path=destination,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        reused_output=reused_output,
        reused_existing_grasp_line=reused_existing_grasp_line,
    )


def _validated_width(value: float) -> float:
    if isinstance(value, bool):
        raise GraspPlanGenerationError("width must be an explicit positive number.")
    try:
        width = float(value)
    except (TypeError, ValueError) as exc:
        raise GraspPlanGenerationError(
            "width must be an explicit positive number."
        ) from exc
    if (
        not math.isfinite(width)
        or width < _FLOAT32_MIN_POSITIVE
        or width > _FLOAT32_MAX
    ):
        raise GraspPlanGenerationError(
            "width must be positive, finite, and representable as a USD float."
        )
    return width


def _capture_dependency_snapshot(source_path: Path) -> _DependencySnapshot:
    try:
        from pxr import Ar, Sdf, UsdUtils
    except ImportError as exc:  # pragma: no cover - environment failure
        raise GraspPlanGenerationError(
            f"OpenUSD Python APIs are unavailable: {exc}"
        ) from exc

    package_root = source_path.parent
    _require_regular_local_file(
        source_path, package_root=package_root, label="source asset"
    )
    source_sha256_before = _file_sha256(source_path)
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source_path))
    except Exception as exc:  # pragma: no cover - OpenUSD exceptions vary
        raise GraspPlanGenerationError(
            f"Could not inspect the USD dependency closure: {exc}"
        ) from exc
    unresolved_paths = sorted({str(item) for item in unresolved})
    if unresolved_paths:
        raise GraspPlanGenerationError(
            "USD dependency closure contains unresolved paths: "
            + ", ".join(unresolved_paths)
        )
    _validate_layers_current(layers, Sdf=Sdf)

    paths = {source_path}
    for layer in layers:
        identifier = str(
            getattr(layer, "resolvedPath", "")
            or getattr(layer, "realPath", "")
            or getattr(layer, "identifier", "")
        )
        paths.add(
            _physical_dependency_path(
                identifier,
                source_path=source_path,
                package_root=package_root,
                Ar=Ar,
                Sdf=Sdf,
                label="USD layer",
            )
        )
    for asset in assets:
        identifier = str(
            getattr(asset, "resolvedPath", "") or getattr(asset, "path", "") or asset
        )
        paths.add(
            _physical_dependency_path(
                identifier,
                source_path=source_path,
                package_root=package_root,
                Ar=Ar,
                Sdf=Sdf,
                label="USD asset",
            )
        )

    records = []
    for path in sorted(
        paths, key=lambda item: item.relative_to(package_root).as_posix()
    ):
        _require_regular_local_file(
            path, package_root=package_root, label="USD dependency"
        )
        records.append(
            _DependencyRecord(
                path=path,
                relative_path=path.relative_to(package_root).as_posix(),
                role="source_asset" if path == source_path else "dependency",
                sha256=_file_sha256(path),
            )
        )
    if _file_sha256(source_path) != source_sha256_before:
        raise GraspPlanGenerationError(
            "Source asset changed while its dependency closure was inspected."
        )

    bundle_payload = [
        {
            "relative_path": record.relative_path,
            "role": record.role,
            "sha256": record.sha256,
        }
        for record in records
    ]
    bundle_bytes = json.dumps(
        bundle_payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _DependencySnapshot(
        source_path=source_path,
        records=tuple(records),
        bundle_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
    )


def _physical_dependency_path(
    identifier: str,
    *,
    source_path: Path,
    package_root: Path,
    Ar: Any,
    Sdf: Any,
    label: str,
) -> Path:
    if not identifier:
        raise GraspPlanGenerationError(f"{label} has no stable local identifier.")
    path_identifier, _args = Sdf.Layer.SplitIdentifier(identifier)
    outer = str(path_identifier)
    while Ar.IsPackageRelativePath(outer):
        outer, _inner = Ar.SplitPackageRelativePathOuter(outer)
    if (
        not outer
        or "://" in outer
        or outer.startswith(("anon:", "file:"))
        or _WINDOWS_DRIVE_PATH.match(outer)
    ):
        raise GraspPlanGenerationError(
            f"{label} is not a supported local filesystem dependency: {identifier}"
        )
    candidate = Path(outer)
    if not candidate.is_absolute():
        candidate = source_path.parent / candidate
    candidate = _absolute_path(candidate)
    _require_regular_local_file(candidate, package_root=package_root, label=label)
    return candidate


def _require_regular_local_file(path: Path, *, package_root: Path, label: str) -> None:
    path = _absolute_path(path)
    package_root = _absolute_path(package_root)
    try:
        path.relative_to(package_root)
    except ValueError as exc:
        raise GraspPlanGenerationError(
            f"{label} resolves outside the source package: {path}"
        ) from exc
    _reject_symlink_components(path)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise GraspPlanGenerationError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise GraspPlanGenerationError(f"{label} is not a regular file: {path}")
    resolved_root = package_root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise GraspPlanGenerationError(
            f"{label} resolves outside the source package: {path}"
        ) from exc


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise GraspPlanGenerationError(
                f"Source or dependency path contains a symlink: {current}"
            )


def _validate_layers_current(layers: Sequence[Any], *, Sdf: Any) -> None:
    for layer in sorted(
        (item for item in layers if not bool(getattr(item, "anonymous", False))),
        key=lambda item: str(getattr(item, "identifier", "")),
    ):
        identifier = str(getattr(layer, "identifier", ""))
        if not identifier:
            raise GraspPlanGenerationError(
                "A composed USD dependency layer has no stable identifier."
            )
        if bool(getattr(layer, "dirty", False)):
            raise GraspPlanGenerationError(
                f"A composed USD dependency layer has unsaved edits: {identifier}"
            )
        try:
            fresh = Sdf.Layer.OpenAsAnonymous(identifier)
        except Exception as exc:  # pragma: no cover - OpenUSD exceptions vary
            raise GraspPlanGenerationError(
                f"Could not read a fresh USD dependency layer {identifier}: {exc}"
            ) from exc
        if fresh is None:
            raise GraspPlanGenerationError(
                f"Could not read a fresh USD dependency layer: {identifier}"
            )
        if layer.ExportToString() != fresh.ExportToString():
            raise GraspPlanGenerationError(
                f"A cached USD dependency layer differs from source bytes: {identifier}"
            )
        if bool(getattr(layer, "dirty", False)):
            raise GraspPlanGenerationError(
                f"A composed USD dependency layer changed during inspection: {identifier}"
            )


def _validated_default_prim(stage: Any) -> Any:
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise GraspPlanGenerationError("Source stage has no valid default prim.")
    if default_prim.GetParent() != stage.GetPseudoRoot():
        raise GraspPlanGenerationError("Source default prim must be a top-level prim.")
    if (
        not default_prim.IsValid()
        or not default_prim.IsActive()
        or not default_prim.IsDefined()
        or default_prim.IsAbstract()
    ):
        raise GraspPlanGenerationError(
            f"Source default prim is not active and defined: {default_prim.GetPath()}"
        )
    return default_prim


def _composed_prim_range(default_prim: Any, *, Usd: Any) -> Any:
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    return Usd.PrimRange(default_prim, predicate)


def _validate_composed_instance_state(
    stage: Any,
    default_prim: Any,
    *,
    Usd: Any,
    UsdGeom: Any,
) -> bool:
    if (
        default_prim.IsInstance()
        or default_prim.IsInstanceProxy()
        or default_prim.IsInstanceable()
    ):
        raise GraspPlanGenerationError(
            "Source default prim cannot be an instance because the deterministic "
            "grasp-line target must have an editable parent."
        )
    prims = list(_composed_prim_range(default_prim, Usd=Usd))
    has_composed_instances = bool(stage.GetPrototypes()) or any(
        prim.IsInstance() or prim.IsInstanceProxy() or prim.IsInstanceable()
        for prim in prims
    )
    for prim in prims:
        if prim.IsPrototype():
            raise GraspPlanGenerationError(
                f"Default-prim traversal escaped into a prototype at {prim.GetPath()}."
            )
        if not prim.IsValid() or not prim.IsActive() or not prim.IsDefined():
            raise GraspPlanGenerationError(
                f"Composed prim is not active and defined: {prim.GetPath()}."
            )
        if prim.IsA(UsdGeom.PointInstancer):
            raise GraspPlanGenerationError(
                f"Source stage contains a PointInstancer at {prim.GetPath()}."
            )
    return has_composed_instances


def _verify_existing_machine_grasp_line(
    prim: Any,
    *,
    expected_points: Sequence[tuple[float, float, float]],
    expected_width: float,
    Gf: Any,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    Vt: Any,
) -> None:
    curve = UsdGeom.BasisCurves(prim)
    if (
        not prim.IsValid()
        or not prim.IsActive()
        or not prim.IsDefined()
        or prim.IsInstance()
        or prim.IsInstanceProxy()
        or not curve
    ):
        raise GraspPlanGenerationError(
            f"Existing machine grasp line is not an editable BasisCurves prim: "
            f"{prim.GetPath()}"
        )
    prim_stack = prim.GetPrimStack()
    if (
        len(prim_stack) != 1
        or prim_stack[0].specifier != Sdf.SpecifierDef
        or prim_stack[0].typeName != "BasisCurves"
        or set(prim_stack[0].ListInfoKeys()) != {"specifier", "typeName"}
        or prim.GetAppliedSchemas()
        or prim.GetChildren()
    ):
        raise GraspPlanGenerationError(
            f"Existing machine grasp line has ambiguous or extra prim state: "
            f"{prim.GetPath()}"
        )
    expected_properties = {
        "curveVertexCounts",
        "extent",
        "points",
        "type",
        "widths",
        "wrap",
    }
    properties = {prop.GetName(): prop for prop in prim.GetAuthoredProperties()}
    if set(properties) != expected_properties:
        raise GraspPlanGenerationError(
            f"Existing machine grasp line has unexpected properties: {prim.GetPath()}"
        )
    for prop in properties.values():
        if (
            prop.IsCustom()
            or prop.GetNumTimeSamples() != 0
            or len(prop.GetPropertyStack()) != 1
        ):
            raise GraspPlanGenerationError(
                f"Existing machine grasp line has unsafe authored state: "
                f"{prop.GetPath()}"
            )
    points = Vt.Vec3fArray([Gf.Vec3f(*point) for point in expected_points])
    widths = Vt.FloatArray([expected_width])
    expected_extent = UsdGeom.Boundable(curve).ComputeExtent(Usd.TimeCode.Default())
    if (
        curve.GetTypeAttr().Get() != UsdGeom.Tokens.linear
        or curve.GetWrapAttr().Get() != UsdGeom.Tokens.nonperiodic
        or list(curve.GetCurveVertexCountsAttr().Get() or []) != [len(points)]
        or curve.GetPointsAttr().Get() != points
        or curve.GetWidthsAttr().Get() != widths
        or curve.GetWidthsInterpolation() != UsdGeom.Tokens.constant
        or expected_extent is None
        or curve.GetExtentAttr().Get() != expected_extent
    ):
        raise GraspPlanGenerationError(
            f"Existing machine grasp line differs from the deterministic plan: "
            f"{prim.GetPath()}"
        )


def _select_surface_triangle(
    stage: Any,
    default_prim: Any,
    *,
    traverse_instance_proxies: bool,
    Gf: Any,
    Usd: Any,
    UsdGeom: Any,
) -> _TriangleCandidate:
    prims = list(
        _composed_prim_range(default_prim, Usd=Usd)
        if traverse_instance_proxies
        else Usd.PrimRange(default_prim)
    )
    for prim in prims:
        _validate_static_finite_transform(prim, Usd=Usd, UsdGeom=UsdGeom)

    mesh_prims = sorted(
        (prim for prim in prims if prim.IsA(UsdGeom.Mesh)),
        key=lambda prim: str(prim.GetPath()),
    )
    cube_prims = sorted(
        (prim for prim in prims if prim.IsA(UsdGeom.Cube)),
        key=lambda prim: str(prim.GetPath()),
    )
    if not mesh_prims and not cube_prims:
        raise GraspPlanGenerationError(
            "Default prim contains no supported explicit Mesh or Cube surface geometry."
        )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    candidates: list[_TriangleCandidate] = []
    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        relative_transform = _surface_to_default_prim_transform(
            prim,
            default_prim,
            xform_cache=xform_cache,
        )
        _require_finite_nonsingular_matrix(
            relative_transform,
            label=f"Mesh-to-default-prim transform at {prim.GetPath()}",
        )
        candidates.extend(
            _mesh_triangle_candidates(
                mesh,
                relative_transform=relative_transform,
                Gf=Gf,
                Usd=Usd,
            )
        )
    for prim in cube_prims:
        cube = UsdGeom.Cube(prim)
        relative_transform = _surface_to_default_prim_transform(
            prim,
            default_prim,
            xform_cache=xform_cache,
        )
        _require_finite_nonsingular_matrix(
            relative_transform,
            label=f"Cube-to-default-prim transform at {prim.GetPath()}",
        )
        candidates.extend(
            _cube_triangle_candidates(
                cube,
                relative_transform=relative_transform,
                Gf=Gf,
                Usd=Usd,
            )
        )
    if not candidates:
        raise GraspPlanGenerationError(
            "Default prim contains no usable finite nondegenerate supported "
            "surface triangle."
        )
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.area_squared,
            candidate.surface_prim_path,
            candidate.surface_kind,
            candidate.face_index,
            candidate.triangle_index,
        ),
    )


def _surface_to_default_prim_transform(
    prim: Any,
    default_prim: Any,
    *,
    xform_cache: Any,
) -> Any:
    """Return an exact surface-to-default transform across reset stacks."""

    relative_transform, reset_stack = xform_cache.ComputeRelativeTransform(
        prim, default_prim
    )
    if not reset_stack:
        return relative_transform

    default_world = xform_cache.GetLocalToWorldTransform(default_prim)
    _require_finite_nonsingular_matrix(
        default_world,
        label=f"Default-prim world transform at {default_prim.GetPath()}",
    )
    surface_world = xform_cache.GetLocalToWorldTransform(prim)
    return surface_world * default_world.GetInverse()


def _validate_static_finite_transform(prim: Any, *, Usd: Any, UsdGeom: Any) -> None:
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return
    if xformable.TransformMightBeTimeVarying():
        raise GraspPlanGenerationError(
            f"Time-varying transform is not supported at {prim.GetPath()}."
        )
    attributes = [xformable.GetXformOpOrderAttr()]
    attributes.extend(op.GetAttr() for op in xformable.GetOrderedXformOps())
    for attribute in attributes:
        if attribute and (
            attribute.GetNumTimeSamples() != 0 or attribute.ValueMightBeTimeVarying()
        ):
            raise GraspPlanGenerationError(
                f"Time-sampled transform is not supported at {prim.GetPath()}."
            )
    try:
        matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    except Exception as exc:  # pragma: no cover - OpenUSD exceptions vary
        raise GraspPlanGenerationError(
            f"Could not evaluate transform at {prim.GetPath()}: {exc}"
        ) from exc
    _require_finite_nonsingular_matrix(
        matrix,
        label=f"Local transform at {prim.GetPath()}",
    )


def _require_finite_nonsingular_matrix(matrix: Any, *, label: str) -> None:
    values = [float(value) for row in matrix for value in row]
    determinant = float(matrix.GetDeterminant())
    if not all(math.isfinite(value) for value in values) or not math.isfinite(
        determinant
    ):
        raise GraspPlanGenerationError(f"{label} is nonfinite.")
    if determinant == 0.0:
        raise GraspPlanGenerationError(f"{label} is singular.")


def _mesh_triangle_candidates(
    mesh: Any,
    *,
    relative_transform: Any,
    Gf: Any,
    Usd: Any,
) -> list[_TriangleCandidate]:
    prim_path = str(mesh.GetPrim().GetPath())
    points_attr = mesh.GetPointsAttr()
    counts_attr = mesh.GetFaceVertexCountsAttr()
    indices_attr = mesh.GetFaceVertexIndicesAttr()
    for attribute, label in (
        (points_attr, "points"),
        (counts_attr, "faceVertexCounts"),
        (indices_attr, "faceVertexIndices"),
    ):
        if not attribute.HasAuthoredValueOpinion():
            raise GraspPlanGenerationError(
                f"Mesh {prim_path} lacks explicit composed {label}."
            )
        if attribute.GetNumTimeSamples() != 0 or attribute.ValueMightBeTimeVarying():
            raise GraspPlanGenerationError(
                f"Mesh {prim_path} has time-varying {label}."
            )

    raw_points = points_attr.Get(Usd.TimeCode.Default())
    raw_counts = counts_attr.Get(Usd.TimeCode.Default())
    raw_indices = indices_attr.Get(Usd.TimeCode.Default())
    if raw_points is None or raw_counts is None or raw_indices is None:
        raise GraspPlanGenerationError(
            f"Mesh {prim_path} has blocked or unreadable composed topology."
        )
    points = tuple(
        _point3(point, label=f"Mesh point at {prim_path}") for point in raw_points
    )
    counts = tuple(int(value) for value in raw_counts)
    indices = tuple(int(value) for value in raw_indices)
    if not points:
        raise GraspPlanGenerationError(f"Mesh {prim_path} has no points.")
    if not counts or any(count < 3 for count in counts):
        raise GraspPlanGenerationError(
            f"Mesh {prim_path} has malformed face vertex counts."
        )
    if sum(counts) != len(indices):
        raise GraspPlanGenerationError(
            f"Mesh {prim_path} face counts do not match its index count."
        )
    if any(index < 0 or index >= len(points) for index in indices):
        raise GraspPlanGenerationError(
            f"Mesh {prim_path} contains an out-of-range topology index."
        )

    hole_indices = _mesh_hole_indices(
        mesh,
        face_count=len(counts),
        prim_path=prim_path,
        Usd=Usd,
    )
    transformed_points = tuple(
        _point3(
            relative_transform.Transform(Gf.Vec3d(*point)),
            label=f"Transformed Mesh point at {prim_path}",
        )
        for point in points
    )

    candidates = []
    offset = 0
    for face_index, count in enumerate(counts):
        face_indices = indices[offset : offset + count]
        offset += count
        if face_index in hole_indices:
            continue
        for triangle_index in range(count - 2):
            point_indices = (
                face_indices[0],
                face_indices[triangle_index + 1],
                face_indices[triangle_index + 2],
            )
            local_triangle = tuple(points[index] for index in point_indices)
            transformed_triangle = tuple(
                transformed_points[index] for index in point_indices
            )
            area_squared = _triangle_area_squared(transformed_triangle)
            if area_squared == 0.0:
                continue
            candidates.append(
                _TriangleCandidate(
                    surface_kind="mesh",
                    surface_prim_path=prim_path,
                    face_index=face_index,
                    triangle_index=triangle_index,
                    point_indices=point_indices,
                    surface_local_points=local_triangle,
                    default_prim_local_points=transformed_triangle,
                    area_squared=area_squared,
                )
            )
    return candidates


def _cube_triangle_candidates(
    cube: Any,
    *,
    relative_transform: Any,
    Gf: Any,
    Usd: Any,
) -> list[_TriangleCandidate]:
    """Return the fixed canonical face triangulation for one composed Cube."""

    prim_path = str(cube.GetPrim().GetPath())
    size_attr = cube.GetSizeAttr()
    if size_attr.GetNumTimeSamples() != 0 or size_attr.ValueMightBeTimeVarying():
        raise GraspPlanGenerationError(f"Cube {prim_path} has time-varying size.")
    if size_attr.HasAuthoredConnections() or size_attr.GetConnections():
        raise GraspPlanGenerationError(f"Cube {prim_path} has connected size.")
    raw_size = size_attr.Get(Usd.TimeCode.Default())
    if isinstance(raw_size, bool):
        raise GraspPlanGenerationError(f"Cube {prim_path} has invalid size.")
    try:
        size = float(raw_size)
    except (TypeError, ValueError) as exc:
        raise GraspPlanGenerationError(f"Cube {prim_path} has invalid size.") from exc
    if not math.isfinite(size) or size <= 0.0 or size > _FLOAT32_MAX:
        raise GraspPlanGenerationError(
            f"Cube {prim_path} size must be positive, finite, and representable."
        )

    half = size / 2.0
    points = (
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    )
    faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    transformed_points = tuple(
        _point3(
            relative_transform.Transform(Gf.Vec3d(*point)),
            label=f"Transformed Cube point at {prim_path}",
        )
        for point in points
    )
    candidates: list[_TriangleCandidate] = []
    for face_index, face_indices in enumerate(faces):
        triangles = (
            (face_indices[0], face_indices[1], face_indices[2]),
            (face_indices[0], face_indices[2], face_indices[3]),
        )
        for triangle_index, point_indices in enumerate(triangles):
            local_triangle = tuple(points[index] for index in point_indices)
            transformed_triangle = tuple(
                transformed_points[index] for index in point_indices
            )
            area_squared = _triangle_area_squared(transformed_triangle)
            if area_squared == 0.0:  # pragma: no cover - nonsingular transform guard
                continue
            candidates.append(
                _TriangleCandidate(
                    surface_kind="cube",
                    surface_prim_path=prim_path,
                    face_index=face_index,
                    triangle_index=triangle_index,
                    point_indices=point_indices,
                    surface_local_points=local_triangle,
                    default_prim_local_points=transformed_triangle,
                    area_squared=area_squared,
                    cube_size=size,
                )
            )
    return candidates


def _mesh_hole_indices(
    mesh: Any,
    *,
    face_count: int,
    prim_path: str,
    Usd: Any,
) -> set[int]:
    attribute = mesh.GetHoleIndicesAttr()
    if not attribute.HasAuthoredValueOpinion():
        return set()
    if attribute.GetNumTimeSamples() != 0 or attribute.ValueMightBeTimeVarying():
        raise GraspPlanGenerationError(
            f"Mesh {prim_path} has time-varying holeIndices."
        )
    value = attribute.Get(Usd.TimeCode.Default())
    if value is None:
        raise GraspPlanGenerationError(
            f"Mesh {prim_path} has blocked or unreadable holeIndices."
        )
    holes = [int(item) for item in value]
    if len(holes) != len(set(holes)) or any(
        item < 0 or item >= face_count for item in holes
    ):
        raise GraspPlanGenerationError(f"Mesh {prim_path} has malformed holeIndices.")
    return set(holes)


def _point3(value: Any, *, label: str) -> tuple[float, float, float]:
    try:
        point = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise GraspPlanGenerationError(f"{label} is not a numeric point.") from exc
    if len(point) != 3 or not all(math.isfinite(component) for component in point):
        raise GraspPlanGenerationError(f"{label} is not a finite 3D point.")
    if any(abs(component) > _FLOAT32_MAX for component in point):
        raise GraspPlanGenerationError(
            f"{label} is not representable by the grasp-plan schema."
        )
    return point


def _triangle_area_squared(
    points: Sequence[tuple[float, float, float]],
) -> float:
    p0, p1, p2 = points
    edge1 = tuple(p1[index] - p0[index] for index in range(3))
    edge2 = tuple(p2[index] - p0[index] for index in range(3))
    cross = (
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    )
    area_squared = sum(component * component for component in cross)
    if not math.isfinite(area_squared):
        raise GraspPlanGenerationError(
            "A transformed surface triangle has nonfinite area."
        )
    return area_squared


def _line_points(
    triangle: Sequence[tuple[float, float, float]],
    *,
    Gf: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    p0, p1, p2 = triangle
    first = tuple((2.0 * p0[index] + p1[index] + p2[index]) / 4.0 for index in range(3))
    second = tuple(
        (p0[index] + 2.0 * p1[index] + p2[index]) / 4.0 for index in range(3)
    )
    line = (
        _point3(first, label="First grasp-line endpoint"),
        _point3(second, label="Second grasp-line endpoint"),
    )
    if line[0] == line[1]:
        raise GraspPlanGenerationError(
            "Fixed interior barycentric endpoints collapse to a zero-length line."
        )
    authored = tuple(
        tuple(float(value) for value in Gf.Vec3f(*point)) for point in line
    )
    if authored[0] == authored[1] or not all(
        math.isfinite(value) for point in authored for value in point
    ):
        raise GraspPlanGenerationError(
            "Fixed interior endpoints do not form a finite nonzero USD float line."
        )
    return line


def _build_plan(
    *,
    snapshot: _DependencySnapshot,
    default_prim_path: str,
    grasp_line_path: str,
    candidate: _TriangleCandidate,
    line_points: Sequence[tuple[float, float, float]],
    width: float,
    composed_instances: bool,
) -> SimReadyGraspPlan:
    dependencies = [record.proof() for record in snapshot.records]
    schema_version: Literal[
        "content-agent-workflows.simready-grasp-plan.v1",
        "content-agent-workflows.simready-grasp-plan.v2",
        "content-agent-workflows.simready-grasp-plan.v3",
    ]
    provenance: (
        SimReadyGraspPlanMachineProvenance
        | SimReadyGraspPlanAnalyticMachineProvenance
        | SimReadyGraspPlanComposedMachineProvenance
    )
    if composed_instances:
        selected_triangle = None
        selected_surface = None
        if candidate.surface_kind == "mesh":
            selected_triangle = SimReadyGraspPlanTriangleProof(
                mesh_prim_path=candidate.surface_prim_path,
                face_index=candidate.face_index,
                triangle_index=candidate.triangle_index,
                point_indices=list(candidate.point_indices),
                mesh_local_points=[
                    list(point) for point in candidate.surface_local_points
                ],
                default_prim_local_points=[
                    list(point) for point in candidate.default_prim_local_points
                ],
            )
        else:
            if candidate.cube_size is None:  # pragma: no cover - candidate invariant
                raise GraspPlanGenerationError("Cube surface candidate has no size.")
            selected_surface = SimReadyGraspPlanAnalyticSurfaceProof(
                primitive_type="Cube",
                prim_path=candidate.surface_prim_path,
                size=candidate.cube_size,
                face_index=candidate.face_index,
                triangle_index=candidate.triangle_index,
                corner_indices=list(candidate.point_indices),
                primitive_local_points=[
                    list(point) for point in candidate.surface_local_points
                ],
                default_prim_local_points=[
                    list(point) for point in candidate.default_prim_local_points
                ],
            )
        provenance = SimReadyGraspPlanComposedMachineProvenance(
            source="machine_composed_geometry_proof",
            implementation=SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION,
            implementation_version=SIMREADY_GRASP_PLAN_COMPOSED_GENERATOR_VERSION,
            source_asset_sha256=snapshot.source_sha256,
            dependency_bundle_sha256=snapshot.bundle_sha256,
            dependencies=dependencies,
            selected_triangle=selected_triangle,
            selected_surface=selected_surface,
            barycentric_coordinates=[list(row) for row in _BARYCENTRIC_COORDINATES],
            line_points_default_prim_local=[list(point) for point in line_points],
            width_stage_units=width,
            proof_checks=SimReadyGraspPlanComposedProofChecks(
                dependency_closure_complete=True,
                source_bytes_preserved=True,
                composed_instance_proxies_resolved=True,
                no_point_instancers=True,
                static_geometry_and_transforms=True,
                surface_schema_and_topology_valid=True,
                transforms_finite_and_nonsingular=True,
                triangle_finite_and_nondegenerate=True,
                endpoints_strictly_inside_triangle=True,
                line_nonzero=True,
                width_explicit_positive_stage_units=True,
            ),
        )
        schema_version = SIMREADY_GRASP_PLAN_COMPOSED_SCHEMA_VERSION
    elif candidate.surface_kind == "mesh":
        provenance = SimReadyGraspPlanMachineProvenance(
            source="machine_geometry_proof",
            implementation=SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION,
            implementation_version=SIMREADY_GRASP_PLAN_GENERATOR_VERSION,
            source_asset_sha256=snapshot.source_sha256,
            dependency_bundle_sha256=snapshot.bundle_sha256,
            dependencies=dependencies,
            selected_triangle=SimReadyGraspPlanTriangleProof(
                mesh_prim_path=candidate.surface_prim_path,
                face_index=candidate.face_index,
                triangle_index=candidate.triangle_index,
                point_indices=list(candidate.point_indices),
                mesh_local_points=[
                    list(point) for point in candidate.surface_local_points
                ],
                default_prim_local_points=[
                    list(point) for point in candidate.default_prim_local_points
                ],
            ),
            barycentric_coordinates=[list(row) for row in _BARYCENTRIC_COORDINATES],
            line_points_default_prim_local=[list(point) for point in line_points],
            width_stage_units=width,
            proof_checks=SimReadyGraspPlanMachineProofChecks(
                dependency_closure_complete=True,
                source_bytes_preserved=True,
                no_instances_or_prototypes=True,
                static_geometry_and_transforms=True,
                topology_valid=True,
                transforms_finite_and_nonsingular=True,
                triangle_finite_and_nondegenerate=True,
                endpoints_strictly_inside_triangle=True,
                line_nonzero=True,
                width_explicit_positive_stage_units=True,
            ),
        )
        schema_version = SIMREADY_GRASP_PLAN_SCHEMA_VERSION
    else:
        if candidate.cube_size is None:  # pragma: no cover - candidate invariant
            raise GraspPlanGenerationError("Cube surface candidate has no size.")
        provenance = SimReadyGraspPlanAnalyticMachineProvenance(
            source="machine_analytic_geometry_proof",
            implementation=SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION,
            implementation_version=SIMREADY_GRASP_PLAN_ANALYTIC_GENERATOR_VERSION,
            source_asset_sha256=snapshot.source_sha256,
            dependency_bundle_sha256=snapshot.bundle_sha256,
            dependencies=dependencies,
            selected_surface=SimReadyGraspPlanAnalyticSurfaceProof(
                primitive_type="Cube",
                prim_path=candidate.surface_prim_path,
                size=candidate.cube_size,
                face_index=candidate.face_index,
                triangle_index=candidate.triangle_index,
                corner_indices=list(candidate.point_indices),
                primitive_local_points=[
                    list(point) for point in candidate.surface_local_points
                ],
                default_prim_local_points=[
                    list(point) for point in candidate.default_prim_local_points
                ],
            ),
            barycentric_coordinates=[list(row) for row in _BARYCENTRIC_COORDINATES],
            line_points_default_prim_local=[list(point) for point in line_points],
            width_stage_units=width,
            proof_checks=SimReadyGraspPlanAnalyticProofChecks(
                dependency_closure_complete=True,
                source_bytes_preserved=True,
                no_instances_or_prototypes=True,
                static_geometry_and_transforms=True,
                schema_parameters_valid=True,
                transforms_finite_and_nonsingular=True,
                triangle_finite_and_nondegenerate=True,
                endpoints_strictly_inside_triangle=True,
                line_nonzero=True,
                width_explicit_positive_stage_units=True,
            ),
        )
        schema_version = SIMREADY_GRASP_PLAN_ANALYTIC_SCHEMA_VERSION
    return SimReadyGraspPlan(
        schema_version=schema_version,
        source_asset_sha256=snapshot.source_sha256,
        default_prim_path=default_prim_path,
        provenance=provenance,
        grasp_lines=[
            SimReadyGraspLinePlan(
                prim_path=grasp_line_path,
                coordinate_space="local",
                points=[list(point) for point in line_points],
                widths=[width],
            )
        ],
    )


def _canonical_plan_bytes(plan: SimReadyGraspPlan) -> bytes:
    payload = plan.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _publish_canonical_bytes(path: Path, payload: bytes) -> bool:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GraspPlanGenerationError(
            f"Could not create grasp-plan output directory {parent}: {exc}"
        ) from exc
    _reject_symlink_components(parent)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise GraspPlanGenerationError(
            f"Grasp plan output is not a regular non-symlink file: {path}"
        )
    if path.is_file():
        if path.read_bytes() != payload:
            raise GraspPlanGenerationError(
                f"Different grasp plan output already exists: {path}"
            )
        return True

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise GraspPlanGenerationError(
                    f"Different grasp plan output appeared during publication: {path}"
                )
            return True
        _fsync_directory(parent)
        return False
    finally:
        temporary.unlink(missing_ok=True)


def _validate_exact_readback(
    path: Path,
    *,
    expected_bytes: bytes,
    expected_plan: SimReadyGraspPlan,
) -> None:
    payload = path.read_bytes()
    if payload != expected_bytes:
        raise GraspPlanGenerationError(
            f"Published grasp plan failed exact byte readback: {path}"
        )
    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
        readback = SimReadyGraspPlan.model_validate(parsed)
    except (UnicodeError, ValueError) as exc:
        raise GraspPlanGenerationError(
            f"Published grasp plan failed schema readback: {exc}"
        ) from exc
    if readback != expected_plan:
        raise GraspPlanGenerationError(
            f"Published grasp plan failed exact schema readback: {path}"
        )


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraspPlanGenerationError(
                f"Grasp plan contains duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise GraspPlanGenerationError(
        f"Grasp plan contains non-finite JSON number: {value}"
    )


def _require_snapshot_unchanged(expected: _DependencySnapshot) -> None:
    current = _capture_dependency_snapshot(expected.source_path)
    if current != expected:
        raise GraspPlanGenerationError(
            "Source asset or dependency closure changed during grasp-plan generation."
        )


def _remove_matching_output(path: Path, payload: bytes) -> None:
    try:
        if not path.is_symlink() and path.is_file() and path.read_bytes() == payload:
            path.unlink()
    except OSError:
        pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GraspPlanGenerationError(
            f"Could not hash regular file {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GraspPlanGenerationError(
            f"Could not open grasp-plan output directory for sync {path}: {exc}"
        ) from exc
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic machine geometry-proof GSP.001 plan."
    )
    parser.add_argument("asset", type=Path, help="Local USD/USDZ source asset.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Canonical JSON plan path; a different existing file is never replaced.",
    )
    parser.add_argument(
        "--width",
        type=float,
        required=True,
        help="Explicit positive grasp-line width in authored stage units.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grasp-plan generator CLI."""

    args = _build_parser().parse_args(argv)
    try:
        result = generate_simready_grasp_plan(
            args.asset,
            args.output,
            width=args.width,
        )
    except (GraspPlanGenerationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"error": str(exc), "status": "BLOCKED"},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "plan_sha256": result.plan_sha256,
                "reused_existing_grasp_line": result.reused_existing_grasp_line,
                "reused_output": result.reused_output,
                "source_asset_sha256": result.plan.source_asset_sha256,
                "status": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "GraspPlanGenerationError",
    "SimReadyGraspPlanGenerationResult",
    "generate_simready_grasp_plan",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
