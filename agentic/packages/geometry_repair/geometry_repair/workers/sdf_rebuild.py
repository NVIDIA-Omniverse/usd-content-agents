# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded per-part SDF reconstruction for explicitly reconstructive jobs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import sdf_tools
import trimesh
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts import atomic_write_json, file_sha256
from ..mesh_io import USD_SUFFIXES, MeshData, load_meshes, update_usd_triangle_meshes
from ..models import RepairOperation
from ..process_limits import temporary_cpu_affinity
from ..sdf_backend_qualification import (
    DEFAULT_SDF_BACKEND_ID,
    SDF_REBUILD_BUILD_ID,
    SDF_REBUILD_IMPLEMENTATION_VERSION,
    get_sdf_backend_qualification,
    inspect_qualified_sdf_backend,
)
from ..worker_ids import SDF_REBUILD_WORKER
from . import sdf_reconstruction
from .base import WorkerResult
from .sdf_reconstruction import (
    SdfReconstructionControls,
    SdfReconstructionResult,
    reconstruct_sdf_mesh,
)

_SDF_OPERATIONS = frozenset(
    {
        "per_part_signed_level_set_reconstruction",
        "per_part_unsigned_offset_reconstruction",
    }
)
_SDF_BACKEND_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")

_CONDITIONAL_WARNING = (
    "SDF reconstruction is generated geometry and remains conditional until independent "
    "known-ground-truth or recorded human visual/task review accepts the surface."
)


@dataclass(frozen=True)
class SdfCollisionReconstructionResult:
    """Owned collision-only reconstruction result; never authors source USD."""

    status: Literal["success", "unavailable", "refused", "failed"]
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    evidence_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class _SdfParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    backend_id: str = DEFAULT_SDF_BACKEND_ID
    mode: Literal["signed", "unsigned_offset"] = "signed"
    max_grid_dimension: int = Field(default=256, ge=32, le=512)
    max_output_faces: int = Field(default=1_000_000, ge=1_000, le=5_000_000)
    max_input_vertices: int = Field(default=5_000_000, ge=4, le=10_000_000)
    max_input_faces: int = Field(default=5_000_000, ge=4, le=10_000_000)
    max_active_voxels: int = Field(default=50_000_000, ge=1_000, le=50_000_000)
    half_width: float = Field(default=3.0, ge=2.0, le=16.0)
    offset_voxels: float = Field(default=1.5, gt=0.0, le=15.0)
    adaptivity: float = Field(default=0.0, ge=0.0, le=1.0)
    closing_steps: int = Field(default=2, ge=0, le=2)
    smoothing_steps: Literal[1] = 1
    feature_voxels: int = Field(default=6, ge=2, le=32)
    minimum_feature_m: float | None = Field(default=None, gt=0.0)
    deterministic_seed: int = Field(default=0, ge=0, le=2**31 - 1)
    # The outer worker runner owns execution deadlines; this is provenance only.
    timeout_s: float = Field(default=300.0, gt=0.0)

    @field_validator("smoothing_steps", mode="before")
    @classmethod
    def _validate_smoothing_steps_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("smoothing_steps must be the integer 1")
        return value

    @field_validator("backend_id")
    @classmethod
    def _validate_backend_id_syntax(cls, value: str) -> str:
        if _SDF_BACKEND_ID.fullmatch(value) is None:
            raise ValueError("backend_id must be a lowercase hyphen-separated identifier")
        return value

    @model_validator(mode="after")
    def _validate_band(self) -> _SdfParameters:
        if self.offset_voxels >= self.half_width:
            raise ValueError("offset_voxels must be below half_width")
        return self


def _refused(message: str, *, metadata: dict[str, object] | None = None) -> WorkerResult:
    return WorkerResult(
        status="unavailable",
        failures=[message],
        metadata={"sdf_status": "refused", **(metadata or {})},
    )


def _failed(message: str, *, metadata: dict[str, object] | None = None) -> WorkerResult:
    return WorkerResult(
        status="failed",
        failures=[message],
        metadata={"sdf_status": "failed", **(metadata or {})},
    )


def _qualified_sdf_backend_identity(backend_id: str) -> dict[str, Any]:
    return inspect_qualified_sdf_backend(backend_id)


def _failure_execution_evidence_payload(
    error: BaseException,
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    controls: SdfReconstructionControls,
    backend_call_status: Literal["failed", "unavailable"],
    backend_identity: dict[str, Any] | None = None,
    session: sdf_tools.SdfSession | None = None,
) -> dict[str, Any]:
    evidence = sdf_reconstruction.sdf_execution_evidence_from_exception(error)
    if evidence is None:
        evidence = sdf_reconstruction.sdf_execution_failure_evidence(
            vertices,
            triangles,
            controls,
            backend_call_status=backend_call_status,
            selected_backend_identity=backend_identity,
            selection_rejections=(
                sdf_reconstruction._selection_rejection_evidence(session)
                if session is not None
                else ()
            ),
        )
    return evidence.model_dump(mode="json")


def _sdf_controls(
    parameters: _SdfParameters,
    *,
    voxel_size: float,
) -> SdfReconstructionControls:
    return SdfReconstructionControls(
        backend_id=parameters.backend_id,
        mode=parameters.mode,
        voxel_size=voxel_size,
        half_width=parameters.half_width,
        offset_voxels=parameters.offset_voxels,
        adaptivity=parameters.adaptivity,
        closing_steps=parameters.closing_steps,
        smoothing_steps=parameters.smoothing_steps,
        deterministic_seed=parameters.deterministic_seed,
        max_grid_dimension=parameters.max_grid_dimension,
        max_input_vertices=parameters.max_input_vertices,
        max_input_faces=parameters.max_input_faces,
        max_active_voxels=parameters.max_active_voxels,
        max_output_faces=parameters.max_output_faces,
    )


def _canonical_array_record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.name,
        "shape": [int(value) for value in contiguous.shape],
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
    }


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return actual.keys() == expected.keys() and all(
            _strict_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            _strict_json_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return actual == expected


def _expected_grid_dimensions(
    vertices: np.ndarray,
    controls: SdfReconstructionControls,
    *,
    topology_closing: bool,
) -> list[int]:
    return list(
        sdf_reconstruction._estimated_grid_dimensions(
            np.asarray(vertices, dtype=np.float32),
            controls,
            topology_closing=topology_closing,
        )
    )


def _requires_topology_closing(
    mode: Literal["signed", "unsigned_offset"],
    triangles: np.ndarray,
) -> bool:
    if mode != "signed":
        return False
    return (
        sum(
            sdf_reconstruction._topology_defect_counts(
                np.asarray(triangles, dtype=np.int32),
            )
        )
        > 0
    )


def _bounded_voxel_size(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    diagonal: float,
    max_grid_dimension: int,
    mode: Literal["signed", "unsigned_offset"],
    half_width: float,
    offset_voxels: float,
    closing_steps: int,
) -> tuple[float, list[int]]:
    """Keep the legacy resolution when it fits the facade's exact mesh preflight."""

    legacy_padding = 2 * math.ceil(half_width) + 2
    legacy_usable_cells = max_grid_dimension - legacy_padding
    if legacy_usable_cells < 8:
        raise ValueError("SDF grid leaves fewer than eight usable interior cells")
    voxel_size = diagonal / legacy_usable_cells
    provisional = SdfReconstructionControls(
        mode=mode,
        voxel_size=voxel_size,
        half_width=half_width,
        offset_voxels=offset_voxels,
        closing_steps=closing_steps,
        max_grid_dimension=max_grid_dimension,
    )
    topology_closing = _requires_topology_closing(mode, triangles)
    try:
        return voxel_size, _expected_grid_dimensions(
            vertices,
            provisional,
            topology_closing=topology_closing,
        )
    except sdf_tools.ResourceLimitError:
        margin = sdf_reconstruction._grid_margin(
            provisional,
            topology_closing=topology_closing,
        )
        safe_usable_cells = max_grid_dimension - 2 * margin - 2
        if safe_usable_cells < 8:
            raise ValueError("SDF grid leaves fewer than eight facade-safe cells") from None
        maximum_extent = float(np.max(np.ptp(vertices, axis=0)))
        voxel_size = max(voxel_size, maximum_extent / safe_usable_cells)
        controls = provisional.model_copy(update={"voxel_size": voxel_size})
        return voxel_size, _expected_grid_dimensions(
            vertices,
            controls,
            topology_closing=topology_closing,
        )


def _validate_sdf_result(
    result: SdfReconstructionResult,
    *,
    controls: SdfReconstructionControls,
    input_vertices: np.ndarray,
    input_triangles: np.ndarray,
    backend_identity: dict[str, Any],
) -> str | None:
    if not isinstance(result, SdfReconstructionResult):
        return "SDF backend returned an unexpected result type"
    vertices = np.asarray(result.vertices)
    triangles = np.asarray(result.triangles)
    if (
        vertices.dtype != np.float32
        or vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not vertices.flags.c_contiguous
        or not vertices.flags.owndata
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        return "SDF backend returned invalid or unowned vertex arrays"
    if (
        triangles.dtype != np.int32
        or triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or not triangles.flags.c_contiguous
        or not triangles.flags.owndata
        or not len(triangles)
        or np.any(triangles < 0)
        or np.any(triangles >= len(vertices))
        or len(triangles) > controls.max_output_faces
        or len(vertices) > sdf_reconstruction._max_output_vertices(controls)
        or np.any(triangles[:, 0] == triangles[:, 1])
        or np.any(triangles[:, 1] == triangles[:, 2])
        or np.any(triangles[:, 2] == triangles[:, 0])
    ):
        return "SDF backend returned invalid or unowned triangle arrays"

    evidence_model = result.evidence
    if not isinstance(evidence_model, sdf_reconstruction.SdfExecutionEvidence):
        return "SDF backend evidence must use the strict execution-evidence model"
    evidence = evidence_model.model_dump(mode="json", warnings=False)
    try:
        json.dumps(evidence, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return "SDF backend evidence is not canonical JSON data"
    expected_sections = {
        "algorithm",
        "backend_call_status",
        "backend_qualification_id",
        "candidate_validation_status",
        "determinism",
        "geometry",
        "limits",
        "output_digests",
        "requested_backend_id",
        "required_operations",
        "resource_usage",
        "schema_version",
        "selected_backend_identity",
        "selection_rejections",
        "source_digests",
    }
    if set(evidence) != expected_sections:
        return "SDF backend evidence has an unexpected structure"
    if evidence.get("schema_version") != sdf_reconstruction.SDF_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        return "SDF backend evidence schema is invalid"
    if not _strict_json_equal(evidence.get("selected_backend_identity"), backend_identity):
        return "SDF driver identity changed during reconstruction"
    qualification = get_sdf_backend_qualification(controls.backend_id)
    expected_required_operations = sorted(
        operation.value for operation in qualification.required_operations
    )
    if (
        evidence.get("requested_backend_id") != controls.backend_id
        or evidence.get("backend_qualification_id") != qualification.qualification_id
        or evidence.get("required_operations") != expected_required_operations
        or evidence.get("selection_rejections") != []
    ):
        return "SDF backend selection evidence does not match its qualified request"
    if (
        evidence.get("backend_call_status") != "succeeded"
        or evidence.get("candidate_validation_status") != "accepted"
    ):
        return "SDF backend execution and candidate validation did not both succeed"
    algorithm = evidence.get("algorithm")
    resource_bounds = evidence.get("limits")
    resource_usage = evidence.get("resource_usage")
    geometry = evidence.get("geometry")
    source_digests = evidence.get("source_digests")
    output_digests = evidence.get("output_digests")
    if not all(
        isinstance(value, dict)
        for value in (
            algorithm,
            resource_bounds,
            resource_usage,
            geometry,
            source_digests,
            output_digests,
        )
    ):
        return "SDF backend evidence is incomplete"
    assert isinstance(algorithm, dict)
    assert isinstance(resource_bounds, dict)
    assert isinstance(resource_usage, dict)
    assert isinstance(geometry, dict)
    assert isinstance(source_digests, dict)
    assert isinstance(output_digests, dict)

    canonical_input_vertices = np.array(input_vertices, dtype=np.float32, order="C", copy=True)
    canonical_input_vertices[canonical_input_vertices == 0.0] = np.float32(0.0)
    canonical_input_triangles = np.array(input_triangles, dtype=np.int32, order="C", copy=True)
    boundary_edges, non_manifold_edges, duplicate_faces = (
        sdf_reconstruction._topology_defect_counts(canonical_input_triangles)
    )
    topology_defects = boundary_edges + non_manifold_edges + duplicate_faces
    topology_closing = controls.mode == "signed" and topology_defects > 0
    if controls.mode == "signed":
        route = "signed_topology_closing" if topology_closing else "signed_level_set"
        filter_operator = sdf_tools.Operation.SMOOTH_SDF.value
        isovalue = -controls.offset_voxels * controls.voxel_size if topology_closing else 0.0
    else:
        route = "unsigned_offset"
        filter_operator = sdf_tools.Operation.SMOOTH_SCALAR.value
        isovalue = controls.offset_voxels * controls.voxel_size
    expected_algorithm = {
        "route": route,
        "signed_level_set": controls.mode == "signed",
        "fallback_used": False,
        "explicit_closing_operator_applied": topology_closing,
        "unsigned_distance_sampling": topology_closing,
        "unsigned_offset_surface": controls.mode == "unsigned_offset",
        "gap_closing_budget_voxels": controls.closing_steps,
        "smoothing_steps": controls.smoothing_steps,
        "filter_operator": filter_operator,
        "adaptivity": controls.adaptivity,
        "half_width": controls.half_width,
        "offset_voxels": controls.offset_voxels,
        "voxel_size": controls.voxel_size,
        "isovalue": isovalue,
        "repair_orientation": True,
        "deterministic_seed": controls.deterministic_seed,
    }
    if not _strict_json_equal(algorithm, expected_algorithm):
        return "SDF backend algorithm evidence does not match its controls"

    expected_bounds = {
        "max_active_voxels": controls.max_active_voxels,
        "max_grid_dimension": controls.max_grid_dimension,
        "max_input_faces": controls.max_input_faces,
        "max_input_vertices": controls.max_input_vertices,
        "max_output_faces": controls.max_output_faces,
        "max_output_vertices": sdf_reconstruction._max_output_vertices(controls),
        "execution_limits": asdict(sdf_reconstruction._execution_limits(controls)),
    }
    if not _strict_json_equal(resource_bounds, expected_bounds):
        return "SDF backend resource evidence does not match its controls"
    expected_usage = {
        "estimated_grid_dimensions": _expected_grid_dimensions(
            canonical_input_vertices,
            controls,
            topology_closing=topology_closing,
        ),
        "input_array_bytes": canonical_input_vertices.nbytes + canonical_input_triangles.nbytes,
        "input_boundary_edges": boundary_edges,
        "input_duplicate_faces": duplicate_faces,
        "input_faces": len(input_triangles),
        "input_non_manifold_edges": non_manifold_edges,
        "input_topology_defects": topology_defects,
        "input_vertices": len(input_vertices),
        "output_array_bytes": vertices.nbytes + triangles.nbytes,
        "output_faces": len(triangles),
        "output_vertices": len(vertices),
        "grid_limits_enforced_by_driver": True,
    }
    if not _strict_json_equal(resource_usage, expected_usage):
        return "SDF backend resource usage does not match its arrays"

    determinism = evidence.get("determinism")
    expected_determinism = {
        "canonical_vertex_order": True,
        "canonical_face_order": True,
        "normalized_signed_zero": True,
        "shortest_quad_diagonal": True,
        "single_thread": True,
    }
    if not _strict_json_equal(determinism, expected_determinism):
        return "SDF backend determinism evidence is invalid"

    if set(geometry) != {"output_surface_area", "output_signed_volume"} or any(
        type(value) is not float or not math.isfinite(value) or value <= 0.0
        for value in geometry.values()
    ):
        return "SDF backend geometry evidence is invalid"
    try:
        recomputed_geometry = {
            "output_surface_area": sdf_reconstruction._surface_area(vertices, triangles),
            "output_signed_volume": sdf_reconstruction._signed_volume(vertices, triangles),
        }
    except sdf_tools.InvalidGeometryError:
        return "SDF backend geometry evidence is invalid"
    if any(
        not math.isclose(
            geometry[key],
            recomputed,
            rel_tol=1e-9,
            abs_tol=0.0,
        )
        for key, recomputed in recomputed_geometry.items()
    ):
        return "SDF backend geometry evidence does not match its arrays"

    expected_source_digests = {
        "vertices": _canonical_array_record(canonical_input_vertices),
        "triangles": _canonical_array_record(canonical_input_triangles),
    }
    expected_output_digests = {
        "vertices": _canonical_array_record(vertices),
        "triangles": _canonical_array_record(triangles),
    }
    if not _strict_json_equal(source_digests, expected_source_digests) or not _strict_json_equal(
        output_digests, expected_output_digests
    ):
        return "SDF backend source or output digest evidence does not match its arrays"
    return None


def _topology_attribute_refusals(source: Path, meshes: list[MeshData]) -> dict[str, list[str]]:
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        return {"<stage>": ["USD stage could not be reopened for attribute inspection"]}
    refusals: dict[str, list[str]] = {}
    topology_dependent_attributes = {
        "accelerations",
        "cornerIndices",
        "cornerSharpnesses",
        "creaseIndices",
        "creaseLengths",
        "creaseSharpnesses",
        "holeIndices",
        "normals",
        "velocities",
    }
    for mesh in meshes:
        reasons: list[str] = []
        if mesh.is_instance_proxy:
            reasons.append("instance proxy")
        if mesh.material_subset_count:
            reasons.append("face material subsets")
        if mesh.authored_uv_count:
            reasons.append("authored UVs")
        if mesh.transform_non_finite or mesh.transform_singular:
            reasons.append("invalid source transform")
        if mesh.transform_non_uniform or mesh.transform_sheared or mesh.transform_reflected:
            reasons.append("non-rigid or reflected source transform")
        prim = stage.GetPrimAtPath(mesh.path)
        if prim and prim.IsA(UsdGeom.Mesh):
            usd_mesh = UsdGeom.Mesh(prim)
            if any(
                attribute.GetNumTimeSamples() > 0
                for attribute in (
                    usd_mesh.GetPointsAttr(),
                    usd_mesh.GetFaceVertexCountsAttr(),
                    usd_mesh.GetFaceVertexIndicesAttr(),
                )
            ):
                reasons.append("time-sampled mesh topology")
            primvars = UsdGeom.PrimvarsAPI(prim).GetPrimvars()
            attributed_primvars = sorted(
                str(primvar.GetPrimvarName())
                for primvar in primvars
                if primvar.HasAuthoredValue()
                and primvar.GetInterpolation() != UsdGeom.Tokens.constant
                and not (
                    primvar.GetPrimvarName() == "normals"
                    and primvar.GetTypeName() == Sdf.ValueTypeNames.Normal3fArray
                    and primvar.GetInterpolation()
                    in {
                        UsdGeom.Tokens.faceVarying,
                        UsdGeom.Tokens.varying,
                        UsdGeom.Tokens.vertex,
                    }
                )
            )
            if attributed_primvars:
                reasons.append("topology-dependent primvars: " + ", ".join(attributed_primvars))
            authored_attributes = sorted(
                attribute.GetName()
                for attribute in prim.GetAttributes()
                if attribute.GetName() in topology_dependent_attributes
                and attribute.GetName() != "normals"
                and attribute.HasAuthoredValueOpinion()
            )
            if authored_attributes:
                reasons.append(
                    "topology-dependent mesh attributes: " + ", ".join(authored_attributes)
                )
        if reasons:
            refusals[mesh.path] = sorted(set(reasons))
    return refusals


def _normal_regeneration_plan(
    source: Path,
    meshes: list[MeshData],
) -> dict[str, list[str]]:
    """Inventory only normal properties that may be deterministically regenerated."""

    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        return {}
    plan: dict[str, list[str]] = {}
    for mesh in meshes:
        prim = stage.GetPrimAtPath(mesh.path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            continue
        properties: list[str] = []
        normals = UsdGeom.Mesh(prim).GetNormalsAttr()
        if normals and normals.HasAuthoredValueOpinion():
            properties.append("normals")
        for primvar in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if (
                primvar.GetPrimvarName() == "normals"
                and primvar.GetTypeName() == Sdf.ValueTypeNames.Normal3fArray
                and primvar.GetInterpolation()
                in {
                    UsdGeom.Tokens.faceVarying,
                    UsdGeom.Tokens.varying,
                    UsdGeom.Tokens.vertex,
                }
                and primvar.HasAuthoredValue()
            ):
                properties.append(str(primvar.GetAttr().GetName()))
                indices = primvar.GetIndicesAttr()
                if indices and indices.HasAuthoredValueOpinion():
                    properties.append(str(indices.GetName()))
        if properties:
            plan[mesh.path] = sorted(set(properties))
    return plan


def inspect_sdf_reconstruction_eligibility(source_path: str | Path) -> dict[str, Any]:
    """Report whether topology-changing SDF reconstruction may inspect a source safely."""

    source = Path(source_path).expanduser().resolve()
    try:
        meshes, _metadata = load_meshes(source)
    except Exception as exc:
        return {
            "schema_version": "geometry-repair.sdf-eligibility.v1",
            "source_path": str(source),
            "source_sha256": file_sha256(source) if source.is_file() else None,
            "eligible": False,
            "render_mesh_count": 0,
            "attribute_refusals": {},
            "normal_regeneration_plan": {},
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    render_meshes = sorted(
        (mesh for mesh in meshes if mesh.role == "render"),
        key=lambda mesh: mesh.path,
    )
    failures = [] if render_meshes else ["SDF reconstruction found no render meshes"]
    refusals = _topology_attribute_refusals(source, render_meshes)
    return {
        "schema_version": "geometry-repair.sdf-eligibility.v1",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "eligible": bool(render_meshes) and not refusals and not failures,
        "render_mesh_count": len(render_meshes),
        "attribute_refusals": refusals,
        "normal_regeneration_plan": _normal_regeneration_plan(source, render_meshes),
        "failures": failures,
    }


def _block_reconstructed_normals(
    output: Path,
    plan: dict[str, list[str]],
) -> None:
    """Block stale authored normals so renderers derive them from new topology."""

    from pxr import Usd

    if not plan:
        return
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError("could not reopen SDF candidate to regenerate normals")
    for prim_path, property_names in sorted(plan.items()):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise RuntimeError(f"normal-regeneration target no longer exists: {prim_path}")
        if prim.IsInstanceProxy():
            raise RuntimeError(
                f"normal-regeneration target is a read-only instance proxy: {prim_path}"
            )
        for property_name in property_names:
            attribute = prim.GetAttribute(property_name)
            if attribute:
                attribute.Block()
        prim.SetCustomDataByKey(
            "geometryRepairNormalPolicy",
            "derive_from_reconstructed_topology",
        )
    stage.GetRootLayer().Save()


def _reauthor_extents(
    output: Path,
    updates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError("could not reopen SDF USD candidate to author extents")
    for prim_path, (vertices, _triangles) in updates.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"SDF extent target no longer exists: {prim_path}")
        if prim.IsInstanceProxy():
            raise RuntimeError(f"SDF extent target is a read-only instance proxy: {prim_path}")
        points = np.asarray(vertices, dtype=np.float64).reshape((-1, 3))
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        UsdGeom.Mesh(prim).CreateExtentAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*(float(value) for value in minimum)),
                    Gf.Vec3f(*(float(value) for value in maximum)),
                ]
            )
        )
    stage.GetRootLayer().Save()


def _validate_source_arrays(mesh: MeshData, parameters: _SdfParameters) -> str | None:
    vertices = np.asarray(mesh.local_vertices, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape((-1, 3))
    if not len(vertices) or not len(triangles):
        return "source arrays are empty"
    if len(vertices) > parameters.max_input_vertices:
        return f"source vertex count exceeds max_input_vertices={parameters.max_input_vertices}"
    if len(triangles) > parameters.max_input_faces:
        return f"source face count exceeds max_input_faces={parameters.max_input_faces}"
    if not np.isfinite(vertices).all():
        return "source vertices are non-finite"
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        return "source triangles contain out-of-range indices"
    counts = np.asarray(mesh.source_face_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(mesh.source_face_indices, dtype=np.int64).reshape(-1)
    if (
        not len(counts)
        or np.any(counts < 3)
        or int(np.sum(counts)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(vertices))
    ):
        return "source face topology cannot be losslessly triangulated for reconstruction"
    return None


def _bounded_surface_points(mesh: trimesh.Trimesh, limit: int = 4096) -> np.ndarray:
    """Return deterministic vertex/centroid samples for reconstruction drift."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(mesh.triangles, dtype=np.float64).reshape((-1, 3, 3))
    centroids = triangles.mean(axis=1) if len(triangles) else np.empty((0, 3))
    points = np.vstack((vertices, centroids))
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
    return points[indices]


def _closest_surface_distance(
    source: trimesh.Trimesh,
    points: np.ndarray,
) -> np.ndarray:
    try:
        _closest, distances, _triangle_ids = trimesh.proximity.closest_point(source, points)
        return np.asarray(distances, dtype=np.float64)
    except (ImportError, ModuleNotFoundError):
        chunks = []
        for start in range(0, len(points), 32):
            _closest, distances, _triangle_ids = trimesh.proximity.closest_point_naive(
                source,
                points[start : start + 32],
            )
            chunks.append(np.asarray(distances, dtype=np.float64))
        return np.concatenate(chunks, axis=0)


def reconstruct_collision_mesh_sdf(
    *,
    source_render: Path,
    mesh: MeshData,
    work_dir: Path,
    request_id: str,
    max_grid_dimension: int,
    max_output_faces: int,
    feature_voxels: int,
    minimum_feature_m: float | None,
    deterministic_seed: int,
    timeout_s: float,
    max_surface_p99_ratio: float,
    backend_id: str = DEFAULT_SDF_BACKEND_ID,
) -> SdfCollisionReconstructionResult:
    """Reconstruct one collision working copy without touching attributed render USD."""

    render_path = source_render.expanduser().resolve()
    expanded_work_dir = work_dir.expanduser()
    render_sha256_before = file_sha256(render_path)
    if expanded_work_dir.is_symlink():
        return SdfCollisionReconstructionResult(
            status="refused",
            failures=["SDF collision work directory must not be a symbolic link"],
        )
    root = expanded_work_dir.resolve()
    if root.exists():
        if not root.is_dir():
            return SdfCollisionReconstructionResult(
                status="refused",
                failures=["SDF collision work path must be a directory"],
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)
    evidence_path = root / "collision_reconstruction.json"

    def finish(
        status: Literal["success", "unavailable", "refused", "failed"],
        *,
        vertices: np.ndarray | None = None,
        triangles: np.ndarray | None = None,
        warnings: list[str] | None = None,
        failures: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SdfCollisionReconstructionResult:
        render_sha256_after = file_sha256(render_path)
        unchanged = render_sha256_after == render_sha256_before
        resolved_failures = list(failures or [])
        if not unchanged:
            status = "failed"
            resolved_failures.append("collision reconstruction altered immutable render USD")
            vertices = None
            triangles = None
        payload = {
            "schema_version": "geometry-repair.sdf-collision-reconstruction.v1",
            "status": status,
            "target_role": "collision",
            "source_render_path": str(render_path),
            "source_render_sha256_before": render_sha256_before,
            "source_render_sha256_after": render_sha256_after,
            "source_render_unchanged": unchanged,
            "source_mesh_path": mesh.path,
            "warnings": list(warnings or []),
            "failures": resolved_failures,
            "metadata": metadata or {},
        }
        atomic_write_json(evidence_path, payload)
        return SdfCollisionReconstructionResult(
            status=status,
            vertices=vertices,
            triangles=triangles,
            evidence_path=str(evidence_path),
            warnings=list(warnings or []),
            failures=resolved_failures,
            metadata=metadata or {},
        )

    try:
        parameters = _SdfParameters(
            backend_id=backend_id,
            mode="signed",
            max_grid_dimension=max_grid_dimension,
            max_output_faces=max_output_faces,
            feature_voxels=feature_voxels,
            minimum_feature_m=minimum_feature_m,
            deterministic_seed=deterministic_seed,
            timeout_s=timeout_s,
            adaptivity=0.05,
            closing_steps=2,
        )
    except ValueError as exc:
        return finish("refused", failures=[f"invalid collision reconstruction bounds: {exc}"])

    vertices = np.asarray(mesh.world_vertices_m, dtype=np.float64).reshape((-1, 3))
    triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape((-1, 3))
    if not len(vertices) or not len(triangles):
        return finish("refused", failures=["collision reconstruction source arrays are empty"])
    if len(vertices) > parameters.max_input_vertices:
        return finish(
            "refused",
            failures=["collision reconstruction source exceeds max_input_vertices"],
        )
    if len(triangles) > parameters.max_input_faces:
        return finish(
            "refused",
            failures=["collision reconstruction source exceeds max_input_faces"],
        )
    if not np.isfinite(vertices).all():
        return finish("refused", failures=["collision reconstruction vertices are non-finite"])
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        return finish("refused", failures=["collision reconstruction indices are out of range"])

    extent = np.ptp(vertices, axis=0)
    diagonal_m = float(np.linalg.norm(extent))
    if not math.isfinite(diagonal_m) or diagonal_m <= 1e-12:
        return finish("refused", failures=["collision reconstruction diagonal is degenerate"])
    try:
        voxel_size_m, grid_dimensions = _bounded_voxel_size(
            vertices,
            triangles,
            diagonal=diagonal_m,
            max_grid_dimension=parameters.max_grid_dimension,
            mode=parameters.mode,
            half_width=parameters.half_width,
            offset_voxels=parameters.offset_voxels,
            closing_steps=parameters.closing_steps,
        )
    except (ValueError, sdf_tools.ResourceLimitError) as exc:
        return finish("refused", failures=[str(exc)])
    controls = _sdf_controls(parameters, voxel_size=voxel_size_m)
    if parameters.minimum_feature_m is not None:
        feature_limit_m = parameters.minimum_feature_m / parameters.feature_voxels
        if voxel_size_m > feature_limit_m * (1.0 + 1e-9):
            margin = sdf_reconstruction._grid_margin(
                controls,
                topology_closing=_requires_topology_closing(parameters.mode, triangles),
            )
            return finish(
                "refused",
                failures=[
                    f"collision voxel size {voxel_size_m:.6g} m exceeds protected-feature "
                    f"limit {feature_limit_m:.6g} m"
                ],
                metadata={
                    "required_grid_dimension": (
                        math.ceil(diagonal_m / feature_limit_m) + 2 * margin + 2
                    )
                },
            )

    try:
        qualification = get_sdf_backend_qualification(parameters.backend_id)
        backend_identity = _qualified_sdf_backend_identity(parameters.backend_id)
    except (
        sdf_tools.BackendUnavailableError,
        sdf_tools.CapabilityUnavailableError,
    ) as exc:
        execution_evidence = sdf_reconstruction.unavailable_sdf_execution_evidence(
            vertices,
            triangles,
            controls,
        )
        return finish(
            "unavailable",
            failures=[
                f"SDF backend {parameters.backend_id!r} is unavailable: {type(exc).__name__}: {exc}"
            ],
            metadata={
                "sdf_execution_evidence": execution_evidence.model_dump(mode="json"),
            },
        )

    operation_name = "per_part_signed_level_set_reconstruction"
    backend_metadata: dict[str, Any] = {
        "backend": "sdf_tools.in_process",
        "estimated_grid_dimensions": grid_dimensions,
        "backend_qualification_id": qualification.qualification_id,
        "implementation_build_id": SDF_REBUILD_BUILD_ID,
        "implementation_version": SDF_REBUILD_IMPLEMENTATION_VERSION,
        "operation": operation_name,
        "request_id": request_id,
        "sdf_backend": backend_identity,
        "requested_outer_timeout_s": parameters.timeout_s,
        "voxel_size_m": voxel_size_m,
    }
    try:
        with temporary_cpu_affinity(1):
            backend_result = reconstruct_sdf_mesh(vertices, triangles, controls)
    except (
        sdf_tools.BackendUnavailableError,
        sdf_tools.CapabilityUnavailableError,
    ) as exc:
        backend_metadata["sdf_execution_evidence"] = _failure_execution_evidence_payload(
            exc,
            vertices=vertices,
            triangles=triangles,
            controls=controls,
            backend_call_status="unavailable",
            backend_identity=backend_identity,
        )
        return finish(
            "unavailable",
            failures=[
                f"SDF backend {parameters.backend_id!r} became unavailable: "
                f"{type(exc).__name__}: {exc}"
            ],
            metadata=backend_metadata,
        )
    except Exception as exc:
        backend_metadata["sdf_execution_evidence"] = _failure_execution_evidence_payload(
            exc,
            vertices=vertices,
            triangles=triangles,
            controls=controls,
            backend_call_status="failed",
            backend_identity=backend_identity,
        )
        return finish(
            "failed",
            failures=[
                f"SDF collision reconstruction with backend {parameters.backend_id!r} failed: "
                f"{type(exc).__name__}: {exc}"
            ],
            metadata=backend_metadata,
        )
    failure = _validate_sdf_result(
        backend_result,
        controls=controls,
        input_vertices=vertices,
        input_triangles=triangles,
        backend_identity=backend_identity,
    )
    if failure:
        return finish(
            "failed",
            failures=[failure],
            metadata=backend_metadata,
        )
    rebuilt_vertices = np.array(backend_result.vertices, dtype=np.float32, order="C", copy=True)
    rebuilt_faces = np.array(backend_result.triangles, dtype=np.int32, order="C", copy=True)
    execution_evidence_payload = backend_result.evidence.model_dump(mode="json")
    backend_metadata["backend_evidence"] = execution_evidence_payload
    backend_metadata["sdf_execution_evidence"] = execution_evidence_payload
    rebuilt = trimesh.Trimesh(
        vertices=rebuilt_vertices,
        faces=rebuilt_faces,
        process=False,
    )
    source_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    source_area = float(source_mesh.area)
    area_ratio = float(rebuilt.area) / source_area if source_area > 0.0 else math.inf
    if (
        not rebuilt.is_watertight
        or not rebuilt.is_winding_consistent
        or not math.isfinite(float(rebuilt.volume))
        or float(rebuilt.volume) <= 0.0
    ):
        return finish(
            "failed",
            failures=["SDF collision surface is not a positive, consistently wound solid"],
            metadata=backend_metadata,
        )
    if not math.isfinite(area_ratio) or not 0.5 <= area_ratio <= 1.5:
        return finish(
            "failed",
            failures=[
                f"collision reconstruction area ratio {area_ratio:.6g} is outside [0.5, 1.5]"
            ],
            metadata=backend_metadata,
        )
    source_to_rebuilt = _closest_surface_distance(
        rebuilt,
        _bounded_surface_points(source_mesh),
    )
    rebuilt_to_source = _closest_surface_distance(
        source_mesh,
        _bounded_surface_points(rebuilt),
    )
    p99_m = float(np.percentile(np.concatenate((source_to_rebuilt, rebuilt_to_source)), 99.0))
    p99_ratio = p99_m / diagonal_m
    backend_metadata.update(
        {
            "generated_surface_area_ratio": area_ratio,
            "output_faces": len(rebuilt_faces),
            "output_vertices": len(rebuilt_vertices),
            "surface_p99_m": p99_m,
            "surface_p99_ratio": p99_ratio,
        }
    )
    if not math.isfinite(p99_ratio) or p99_ratio > max_surface_p99_ratio:
        return finish(
            "failed",
            failures=[
                f"collision reconstruction p99 surface drift {p99_ratio:.6g} exceeds "
                f"{max_surface_p99_ratio:.6g} of the source diagonal"
            ],
            metadata=backend_metadata,
        )
    return finish(
        "success",
        vertices=rebuilt_vertices,
        triangles=rebuilt_faces,
        warnings=[
            "Collision-only SDF geometry is generated and requires independent task or "
            "recorded human review before certification."
        ],
        metadata=backend_metadata,
    )


class SdfRebuildWorker:
    """Reconstruct eligible semantic meshes through one admitted SDF backend."""

    name = SDF_REBUILD_WORKER
    operations = _SDF_OPERATIONS

    def available(self) -> tuple[bool, str | None]:
        return self.available_for_backend(DEFAULT_SDF_BACKEND_ID)

    def available_for_backend(self, backend_id: str) -> tuple[bool, str | None]:
        """Check the qualified backend requested by one repair operation."""

        try:
            _qualified_sdf_backend_identity(backend_id)
        except sdf_tools.SdfToolsError as exc:
            return False, f"{backend_id}: {type(exc).__name__}: {exc}"
        return True, None

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        if operation.worker != self.name:
            return _refused(f"SDF operation targets unexpected worker {operation.worker!r}")
        if operation.drift_band != "reconstructive":
            return _refused("SDF reconstruction requires the reconstructive drift band")
        try:
            parameters = _SdfParameters.model_validate(operation.parameters)
        except Exception as exc:
            return _refused(f"invalid SDF reconstruction parameters: {exc}")
        expanded_source = source.expanduser()
        expanded_output = output.expanduser()
        if expanded_source.is_symlink():
            return _refused("SDF source must not be a symbolic link")
        if expanded_output.is_symlink():
            return _refused("SDF output must not be a symbolic link")
        source_path = expanded_source.resolve()
        output_path = expanded_output.resolve()
        if source_path.suffix.lower() not in USD_SUFFIXES:
            return _refused("SDF reconstruction requires a canonical USD working copy")
        if source_path == output_path:
            return _refused("SDF output must not overwrite its immutable source checkpoint")
        if not source_path.is_file() or source_path.is_symlink():
            return _refused("SDF source must be one regular canonical USD file")
        if output_path.exists() and not output_path.is_file():
            return _refused("SDF output path must not name an existing directory")
        try:
            source_sha256 = file_sha256(source_path)
            meshes, _metadata = load_meshes(source_path)
        except Exception as exc:
            return _failed(f"SDF worker could not inspect the source: {type(exc).__name__}: {exc}")
        render_meshes = sorted(
            (mesh for mesh in meshes if mesh.role == "render"), key=lambda mesh: mesh.path
        )
        if not render_meshes:
            return _refused("SDF reconstruction found no render meshes")
        if len({mesh.path for mesh in render_meshes}) != len(render_meshes):
            return _refused("SDF reconstruction render-mesh paths are not unique")
        attribute_refusals = _topology_attribute_refusals(source_path, render_meshes)
        if attribute_refusals:
            details = "; ".join(
                f"{path}: {', '.join(reasons)}"
                for path, reasons in sorted(attribute_refusals.items())
            )
            return _refused(
                "SDF reconstruction refused attributed or instanced topology without an "
                f"exact remapper: {details}",
                metadata={"attribute_guard": attribute_refusals},
            )
        normal_regeneration = _normal_regeneration_plan(source_path, render_meshes)
        for mesh in render_meshes:
            failure = _validate_source_arrays(mesh, parameters)
            if failure:
                return _refused(f"{mesh.path}: {failure}")

        updates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        invocations: list[dict[str, object]] = []
        session: sdf_tools.SdfSession | None = None
        backend_identity: dict[str, Any] | None = None
        qualification = None
        operation_name = (
            "per_part_signed_level_set_reconstruction"
            if parameters.mode == "signed"
            else "per_part_unsigned_offset_reconstruction"
        )
        for index, mesh in enumerate(render_meshes):
            vertices = np.asarray(mesh.local_vertices, dtype=np.float64).reshape((-1, 3))
            triangles = np.asarray(mesh.triangles, dtype=np.int64).reshape((-1, 3))
            local_extent = np.ptp(vertices, axis=0)
            world_extent = np.ptp(
                np.asarray(mesh.world_vertices_m, dtype=np.float64).reshape((-1, 3)), axis=0
            )
            local_diagonal = float(np.linalg.norm(local_extent))
            world_diagonal_m = float(np.linalg.norm(world_extent))
            if (
                not math.isfinite(local_diagonal)
                or local_diagonal <= 1e-12
                or not math.isfinite(world_diagonal_m)
                or world_diagonal_m <= 1e-12
            ):
                return _refused(f"{mesh.path}: source diagonal is not reconstructible")
            try:
                voxel_size, estimated_grid_dimensions = _bounded_voxel_size(
                    vertices,
                    triangles,
                    diagonal=local_diagonal,
                    max_grid_dimension=parameters.max_grid_dimension,
                    mode=parameters.mode,
                    half_width=parameters.half_width,
                    offset_voxels=parameters.offset_voxels,
                    closing_steps=parameters.closing_steps,
                )
            except (ValueError, sdf_tools.ResourceLimitError) as exc:
                return _refused(f"{mesh.path}: {exc}")
            voxel_size_m = voxel_size * (world_diagonal_m / local_diagonal)
            controls = _sdf_controls(parameters, voxel_size=voxel_size)
            if parameters.minimum_feature_m is not None:
                feature_limit_m = parameters.minimum_feature_m / parameters.feature_voxels
                if voxel_size_m > feature_limit_m * (1.0 + 1e-9):
                    margin = sdf_reconstruction._grid_margin(
                        controls,
                        topology_closing=_requires_topology_closing(parameters.mode, triangles),
                    )
                    required_grid = math.ceil(world_diagonal_m / feature_limit_m) + 2 * margin + 2
                    return _refused(
                        f"{mesh.path}: grid budget yields {voxel_size_m:.6g} m voxels, above "
                        f"the protected-feature limit {feature_limit_m:.6g} m",
                        metadata={
                            "required_grid_dimension": required_grid,
                            "max_grid_dimension": parameters.max_grid_dimension,
                        },
                    )

            if session is None:
                try:
                    qualification = get_sdf_backend_qualification(parameters.backend_id)
                    session = sdf_tools.create_session(
                        backend=parameters.backend_id,
                        require=qualification.required_operations,
                        limits=sdf_reconstruction._execution_limits(controls),
                    )
                    backend_identity = session.backend_info.as_dict()
                    qualification.validate(backend_identity)
                except sdf_tools.SdfToolsError as exc:
                    execution_evidence = sdf_reconstruction.unavailable_sdf_execution_evidence(
                        vertices,
                        triangles,
                        controls,
                    ).model_dump(mode="json")
                    invocations.append(
                        {
                            "backend": "sdf_tools.in_process",
                            "backend_id": parameters.backend_id,
                            "backend_qualification_id": execution_evidence[
                                "backend_qualification_id"
                            ],
                            "estimated_grid_dimensions": estimated_grid_dimensions,
                            "mesh_path": mesh.path,
                            "request_id": f"{operation.operation_id}:{index:04d}",
                            "sdf_execution_evidence": execution_evidence,
                            "sdf_status": "unavailable",
                            "voxel_size_local": voxel_size,
                            "voxel_size_m": voxel_size_m,
                        }
                    )
                    return WorkerResult(
                        status="unavailable",
                        failures=[
                            f"SDF backend {parameters.backend_id!r} is unavailable: "
                            f"{type(exc).__name__}: {exc}"
                        ],
                        metadata={
                            "backend": "sdf_tools.in_process",
                            "sdf_backend_id": parameters.backend_id,
                            "sdf_execution_evidence": execution_evidence,
                            "invocations": invocations,
                            "sdf_status": "unavailable",
                        },
                    )
                output_path.unlink(missing_ok=True)
            assert backend_identity is not None
            assert qualification is not None
            invocation: dict[str, object] = {
                "backend": "sdf_tools.in_process",
                "backend_id": parameters.backend_id,
                "backend_qualification_id": qualification.qualification_id,
                "estimated_grid_dimensions": estimated_grid_dimensions,
                "mesh_path": mesh.path,
                "sdf_status": "running",
                "request_id": f"{operation.operation_id}:{index:04d}",
                "voxel_size_local": voxel_size,
                "voxel_size_m": voxel_size_m,
            }
            invocations.append(invocation)
            try:
                backend_result = reconstruct_sdf_mesh(
                    vertices,
                    triangles,
                    controls,
                    session=session,
                )
            except (
                sdf_tools.BackendUnavailableError,
                sdf_tools.CapabilityUnavailableError,
            ) as exc:
                invocation["sdf_status"] = "unavailable"
                execution_evidence_payload = _failure_execution_evidence_payload(
                    exc,
                    vertices=vertices,
                    triangles=triangles,
                    controls=controls,
                    backend_call_status="unavailable",
                    backend_identity=backend_identity,
                    session=session,
                )
                invocation["sdf_execution_evidence"] = execution_evidence_payload
                return WorkerResult(
                    status="unavailable",
                    failures=[
                        f"{mesh.path}: SDF backend {parameters.backend_id!r} became unavailable: "
                        f"{type(exc).__name__}: {exc}"
                    ],
                    metadata={
                        "backend": "sdf_tools.in_process",
                        "invocations": invocations,
                        "sdf_execution_evidence": execution_evidence_payload,
                        "sdf_status": "unavailable",
                    },
                )
            except Exception as exc:
                invocation["sdf_status"] = "failed"
                invocation["sdf_execution_evidence"] = _failure_execution_evidence_payload(
                    exc,
                    vertices=vertices,
                    triangles=triangles,
                    controls=controls,
                    backend_call_status="failed",
                    backend_identity=backend_identity,
                    session=session,
                )
                return _failed(
                    f"{mesh.path}: SDF reconstruction with backend "
                    f"{parameters.backend_id!r} failed: {type(exc).__name__}: {exc}",
                    metadata={"invocations": invocations},
                )
            failure = _validate_sdf_result(
                backend_result,
                controls=controls,
                input_vertices=vertices,
                input_triangles=triangles,
                backend_identity=backend_identity,
            )
            if failure:
                invocation["sdf_status"] = "failed"
                return _failed(
                    f"{mesh.path}: {failure}",
                    metadata={"invocations": invocations},
                )
            evidence = backend_result.evidence
            algorithm_evidence = evidence["algorithm"]
            resource_evidence = evidence["resource_usage"]
            execution_evidence_payload = evidence.model_dump(mode="json")
            invocation["backend_evidence"] = execution_evidence_payload
            invocation["sdf_execution_evidence"] = execution_evidence_payload
            invocation["backend_report"] = {
                "backend_id": backend_identity["backend_id"],
                "implementation_version": backend_identity["implementation_version"],
                "fallback_used": algorithm_evidence["fallback_used"],
                "mode": parameters.mode,
                "output_faces": resource_evidence["output_faces"],
                "output_vertices": resource_evidence["output_vertices"],
                "route": algorithm_evidence["route"],
                "smoothing_steps": algorithm_evidence["smoothing_steps"],
                "voxel_size": algorithm_evidence["voxel_size"],
            }
            invocation["sdf_status"] = "backend_returned"
            rebuilt_vertices = np.array(
                backend_result.vertices, dtype=np.float32, order="C", copy=True
            )
            rebuilt_faces = np.array(backend_result.triangles, dtype=np.int32, order="C", copy=True)
            rebuilt = trimesh.Trimesh(
                vertices=rebuilt_vertices,
                faces=rebuilt_faces,
                process=False,
            )
            source_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            source_area = float(source_mesh.area)
            rebuilt_area = float(rebuilt.area)
            area_ratio = rebuilt_area / source_area if source_area > 0.0 else math.inf
            invocation["generated_surface_area_ratio"] = area_ratio
            if not math.isfinite(area_ratio) or not 0.5 <= area_ratio <= 1.5:
                return _failed(
                    f"{mesh.path}: generated surface area ratio {area_ratio:.6g} is outside "
                    "the coarse [0.5, 1.5] reconstruction sanity band",
                    metadata={"invocations": invocations},
                )
            if not rebuilt.is_watertight or not rebuilt.is_winding_consistent:
                return _failed(
                    f"{mesh.path}: signed reconstruction did not produce a consistently wound "
                    "watertight surface",
                    metadata={"invocations": invocations},
                )
            if not math.isfinite(float(rebuilt.volume)) or float(rebuilt.volume) <= 0.0:
                return _failed(
                    f"{mesh.path}: reconstructed surface has no positive enclosed volume",
                    metadata={"invocations": invocations},
                )
            invocation.update(
                {
                    "input_faces": len(triangles),
                    "input_vertices": len(vertices),
                    "output_faces": len(rebuilt_faces),
                    "output_vertices": len(rebuilt_vertices),
                }
            )
            updates[mesh.path] = (rebuilt_vertices, rebuilt_faces)
            if file_sha256(source_path) != source_sha256:
                return _failed(
                    "SDF reconstruction altered the immutable USD source",
                    metadata={"invocations": invocations},
                )

        try:
            update_usd_triangle_meshes(source_path, output_path, updates)
            _block_reconstructed_normals(output_path, normal_regeneration)
            _reauthor_extents(output_path, updates)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            return _failed(
                f"SDF USD candidate authoring failed: {type(exc).__name__}: {exc}",
                metadata={"invocations": invocations},
            )
        source_sha256_after = file_sha256(source_path)
        if source_sha256_after != source_sha256:
            output_path.unlink(missing_ok=True)
            return _failed(
                "SDF candidate authoring altered the immutable USD source",
                metadata={"invocations": invocations},
            )
        for invocation in invocations:
            invocation["sdf_status"] = "success"
        output_sha256 = file_sha256(output_path)
        return WorkerResult(
            status="completed",
            output_path=str(output_path),
            output_sha256=output_sha256,
            changed=True,
            operations=[operation_name],
            warnings=[
                _CONDITIONAL_WARNING,
                *(
                    [
                        "Topology-dependent authored normals were blocked; renderers must "
                        "derive normals from the reconstructed surface."
                    ]
                    if normal_regeneration
                    else []
                ),
            ],
            metadata={
                "acceptance_semantics": "conditional_generated_geometry",
                "attribute_policy": (
                    "whole_mesh_binding_only_refuse_topology_attributes_allow_normal_regeneration"
                ),
                "backend": "sdf_tools.in_process",
                "backend_success_is_acceptance": False,
                "deterministic": True,
                "deterministic_seed": parameters.deterministic_seed,
                "execution_isolation": "geometry_repair.worker_runner",
                "fallback_used": False,
                "backend_qualification_id": qualification.qualification_id,
                "implementation_build_id": SDF_REBUILD_BUILD_ID,
                "implementation_version": SDF_REBUILD_IMPLEMENTATION_VERSION,
                "invocations": invocations,
                "mode": parameters.mode,
                "regenerated_attributes": normal_regeneration,
                "sdf_status": "success",
                "sdf_backend": backend_identity,
                "part_identity_strategy": "one_source_mesh_to_same_usd_prim",
                "requires_ground_truth_or_human_review": True,
                "resource_bounds": {
                    "max_active_voxels": parameters.max_active_voxels,
                    "max_grid_dimension": parameters.max_grid_dimension,
                    "max_input_faces": parameters.max_input_faces,
                    "max_input_vertices": parameters.max_input_vertices,
                    "max_output_faces": parameters.max_output_faces,
                    "requested_worker_timeout_s": parameters.timeout_s,
                },
                "signed_reconstruction": parameters.mode == "signed",
                "source_sha256_after": source_sha256_after,
                "source_sha256_before": source_sha256,
                "source_unchanged": True,
            },
        )


# Import-only compatibility aliases. Canonical routing and new evidence use the
# SDF-qualified names above.
_OPENVDB_QUALIFICATION = get_sdf_backend_qualification("openvdb")
OPENVDB_BUILD_ID = SDF_REBUILD_BUILD_ID
OPENVDB_IMPLEMENTATION_VERSION = SDF_REBUILD_IMPLEMENTATION_VERSION
OPENVDB_BACKEND_SOURCE_ID = _OPENVDB_QUALIFICATION.expected_provenance["source_commit"]
OPENVDB_DISTRIBUTION_VERSION = _OPENVDB_QUALIFICATION.expected_provenance["distribution_version"]
_OPENVDB13_REQUIRED_OPERATIONS = _OPENVDB_QUALIFICATION.required_operations
OpenVdbCollisionReconstructionResult = SdfCollisionReconstructionResult
OpenVdbRebuildWorker = SdfRebuildWorker
inspect_openvdb_reconstruction_eligibility = inspect_sdf_reconstruction_eligibility
reconstruct_collision_mesh_openvdb = reconstruct_collision_mesh_sdf
_validate_openvdb13_result = _validate_sdf_result


def _source_locked_openvdb13_identity() -> dict[str, Any]:
    """Compatibility wrapper for tests and historical promotion tooling."""

    return _qualified_sdf_backend_identity("openvdb")
