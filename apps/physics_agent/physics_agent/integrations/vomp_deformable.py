# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author a Newton-compatible volume deformable from a complete VoMP field.

The adapter maps each occupied VoMP cube to the globally conforming six-tet
Freudenthal subdivision. Density is preserved as lumped per-point mass. VoMP's
spatial elastic field cannot be represented by Newton 1.4's single effective
volume material, so heterogeneous Young's modulus or Poisson ratio is rejected
unless the caller explicitly selects the conditional homogeneous reduction.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from world_understanding.functions.graphics.so_export import portable_sidecar_name

from physics_agent.integrations.vomp import (
    VompIntegrationError,
    VompVoxelField,
    _json_custom_data,
    _json_mapping,
    _normalize_composition_hashes,
    _publish_artifact_pair,
    _reload_and_hash_composition_layers,
    _require_finite_positive,
    _require_sha256,
    _serialize_json,
    _serialize_stage,
    _sha256,
    _stats,
    _target_world_frame,
    _validate_geometry_association,
    _validate_vomp_input_layer,
    _verify_composition_hashes,
    _verify_npz_digest,
    load_vomp_voxel_field,
)

AOUSD_DEFORMABLE_PROPOSAL_URL = (
    "https://github.com/PixarAnimationStudios/OpenUSD-proposals/pull/111"
)
AOUSD_DEFORMABLE_SCHEMA_COMMIT = "61d83b54b7efbe97ad2f480de885255cd1e593be"
NEWTON_1_4_AOUSD_BASELINE_COMMIT = "5d89c0ed46a26de92f4d3fefef3bfad6500c07ce"
NEWTON_DEFORMABLE_PROFILE = "newton-1.4-aousd-volume-deformable-draft"
DEFAULT_MAX_DEFORMABLE_VOXELS = 65_536

_OWNERSHIP_MARKER = "physics_agent.vomp_volume_deformable"
_SIMULATION_PRIM_NAME = "PhysicsAgentVompSimulation"
_LOOKS_PRIM_NAME = "PhysicsAgentVompLooks"
_MATERIAL_PRIM_NAME = "VolumeMaterial"
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_ELASTIC_UNIFORM_RTOL = 1.0e-6
_POISSON_UNIFORM_ATOL = 1.0e-7
_MASS_FLOAT32_RTOL = 2.0e-6
_NEWTON_POISSON_MIN = -0.999
_NEWTON_POISSON_MAX = 0.499
# Float32 endpoint subtraction through a 64-cell local offset peaks at 2.25e-5
# relative volume error for otherwise exact native-grid tetrahedra.
_GEOMETRY_FLOAT32_VOLUME_RTOL = 2.5e-5
_GEOMETRY_FLOAT32_CENTER_ATOL_VOXELS = 1.0e-5
_DEFORMABLE_SIM_SCHEMAS = frozenset(
    {
        "PhysicsCurvesDeformableSimAPI",
        "PhysicsSurfaceDeformableSimAPI",
        "PhysicsVolumeDeformableSimAPI",
    }
)
_OWNED_SIM_STALE_SCHEMAS = frozenset(
    {
        "PhysicsCurvesDeformableSimAPI",
        "PhysicsSurfaceDeformableSimAPI",
    }
)
_OWNED_SIM_STALE_PROPERTIES = frozenset(
    {
        "physics:restShapePoints",
        "physics:restTetVertexIndices",
        "surfaceFaceVertexIndices",
    }
)

type MaterialReductionPolicy = Literal["reject", "homogeneous-volume-average"]
type _Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class VompTetMesh:
    """Deterministic tetrahedralization and exact lumped mass in world SI."""

    points_world_m: np.ndarray
    tet_vertex_indices: np.ndarray
    point_masses_kg: np.ndarray
    voxel_count: int
    integrated_volume_m3: float
    total_mass_kg: float
    center_of_mass_world_m: _Vector3
    maximum_voxel_center_snap_error_m: float
    mass_center_snap_error_m: float
    topology_sha256: str

    @property
    def point_count(self) -> int:
        return int(self.points_world_m.shape[0])

    @property
    def tet_count(self) -> int:
        return int(self.tet_vertex_indices.shape[0])


@dataclass(frozen=True)
class VompElasticReduction:
    """One homogeneous elastic material and its approximation evidence."""

    policy: MaterialReductionPolicy
    youngs_modulus_pa: float
    poisson_ratio: float
    heterogeneous: bool
    validation_status: Literal["supported", "conditional"]
    statistics: dict[str, Any]


@dataclass(frozen=True)
class VompDeformableApplyResult:
    """Artifacts and authored contract from deformable VoMP authoring."""

    output_usd_path: Path
    provenance_path: Path
    target_prim_path: str
    simulation_prim_path: str
    material_prim_path: str
    sample_count: int
    point_count: int
    tet_count: int
    mass_kg: float
    center_of_mass_world_m: _Vector3
    material_reduction: VompElasticReduction


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VompIntegrationError(f"{label} must be a positive integer")
    return value


def _stable_mean(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    result = 0.0 if scale == 0.0 else scale * float(np.mean(values / scale))
    if not math.isfinite(result):
        raise VompIntegrationError("VoMP material reduction produced a non-finite mean")
    return result


def _stable_standard_deviation(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    result = 0.0 if scale == 0.0 else scale * float(np.std(values / scale))
    if not math.isfinite(result):
        raise VompIntegrationError(
            "VoMP material reduction produced a non-finite standard deviation"
        )
    return result


def reduce_vomp_elastic_field(
    field: VompVoxelField,
    *,
    policy: MaterialReductionPolicy = "reject",
) -> VompElasticReduction:
    """Reduce VoMP elasticity to Newton 1.4's one-material volume contract."""

    if policy not in {"reject", "homogeneous-volume-average"}:
        raise VompIntegrationError(
            "material_reduction must be 'reject' or 'homogeneous-volume-average'"
        )
    youngs = field.youngs_modulus_pa
    poisson = field.poisson_ratio
    youngs_mean = _stable_mean(youngs)
    poisson_mean = _stable_mean(poisson)
    youngs_std = _stable_standard_deviation(youngs)
    poisson_std = _stable_standard_deviation(poisson)
    youngs_min = float(np.min(youngs))
    youngs_max = float(np.max(youngs))
    poisson_min = float(np.min(poisson))
    poisson_max = float(np.max(poisson))
    youngs_uniform = bool(
        np.allclose(
            youngs,
            youngs[0],
            rtol=_ELASTIC_UNIFORM_RTOL,
            atol=0.0,
        )
    )
    poisson_uniform = bool(
        np.allclose(
            poisson,
            poisson[0],
            rtol=0.0,
            atol=_POISSON_UNIFORM_ATOL,
        )
    )
    heterogeneous = not (youngs_uniform and poisson_uniform)
    if poisson_min < _NEWTON_POISSON_MIN or poisson_max > _NEWTON_POISSON_MAX:
        raise VompIntegrationError(
            "VoMP Poisson ratio field is outside Newton 1.4's unclamped range "
            f"[{_NEWTON_POISSON_MIN}, {_NEWTON_POISSON_MAX}]; refusing to "
            "silently normalize invalid source material evidence"
        )
    minimum_scaled_reciprocal = float(np.mean(youngs_min / youngs))
    youngs_reuss = youngs_min / minimum_scaled_reciprocal
    statistics: dict[str, Any] = {
        "youngsModulusPa": {
            **_stats(youngs),
            "standardDeviation": youngs_std,
            "coefficientOfVariation": youngs_std / youngs_mean,
            "voigtUpperBound": youngs_mean,
            "reussLowerBound": youngs_reuss,
            "maxToMinRatio": youngs_max / youngs_min,
        },
        "poissonRatio": {
            **_stats(poisson),
            "standardDeviation": poisson_std,
            "span": poisson_max - poisson_min,
        },
        "uniformityTolerance": {
            "youngsModulusRelative": _ELASTIC_UNIFORM_RTOL,
            "poissonRatioAbsolute": _POISSON_UNIFORM_ATOL,
        },
        "validatedNewtonPoissonRange": {
            "minimum": _NEWTON_POISSON_MIN,
            "maximum": _NEWTON_POISSON_MAX,
        },
    }
    if heterogeneous and policy == "reject":
        raise VompIntegrationError(
            "VoMP Young's modulus or Poisson ratio is spatially heterogeneous, "
            "but Newton 1.4 imports one effective material for a volume deformable. "
            "Select material_reduction='homogeneous-volume-average' to publish a "
            "conditional approximation with reduction evidence."
        )
    return VompElasticReduction(
        policy=policy,
        youngs_modulus_pa=youngs_mean,
        poisson_ratio=poisson_mean,
        heterogeneous=heterogeneous,
        validation_status="conditional" if heterogeneous else "supported",
        statistics=statistics,
    )


def _topology_digest(
    points_world_m: np.ndarray,
    tet_vertex_indices: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for array, dtype in (
        (points_world_m, "<f8"),
        (tet_vertex_indices, "<i4"),
    ):
        canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
        digest.update(canonical.shape.__repr__().encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _point_mass_digest(point_masses_kg: np.ndarray) -> str:
    canonical = np.ascontiguousarray(point_masses_kg, dtype=np.dtype("<f4"))
    digest = hashlib.sha256()
    digest.update(canonical.shape.__repr__().encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def build_vomp_tet_mesh(
    field: VompVoxelField,
    *,
    max_voxels: int = DEFAULT_MAX_DEFORMABLE_VOXELS,
) -> VompTetMesh:
    """Build six positive, conforming tetrahedra for every occupied VoMP cube."""

    limit = _positive_integer(max_voxels, label="max_deformable_voxels")
    if field.sample_count > limit:
        raise VompIntegrationError(
            f"VoMP field has {field.sample_count} voxels, above the deformable "
            f"topology limit {limit}; refusing to subsample or coarsen"
        )

    pitch = field.voxel_size_m
    origin_center = np.min(field.coordinates_world_m, axis=0)
    lattice = np.rint((field.coordinates_world_m - origin_center) / pitch).astype(
        np.int64
    )
    order = np.lexsort((lattice[:, 2], lattice[:, 1], lattice[:, 0]))
    lattice = lattice[order]
    density = field.density_kg_m3[order]
    source_centers_world_m = field.coordinates_world_m[order]
    snapped_centers_world_m = origin_center + lattice.astype(np.float64) * pitch
    voxel_center_snap_errors_m = np.linalg.norm(
        snapped_centers_world_m - source_centers_world_m,
        axis=1,
    )

    corner_offsets = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ],
        dtype=np.int64,
    )
    voxel_corners = lattice[:, np.newaxis, :] + corner_offsets[np.newaxis, :, :]
    unique_corners, inverse = np.unique(
        voxel_corners.reshape(-1, 3),
        axis=0,
        return_inverse=True,
    )
    if len(unique_corners) > int(np.iinfo(np.int32).max):
        raise VompIntegrationError("VoMP deformable topology exceeds USD index limits")
    points_world_m = origin_center + (unique_corners.astype(np.float64) - 0.5) * pitch
    local_vertex_indices = inverse.reshape(field.sample_count, 8)
    six_tets = np.asarray(
        [
            (0, 1, 3, 7),
            (0, 3, 2, 7),
            (0, 2, 6, 7),
            (0, 6, 4, 7),
            (0, 4, 5, 7),
            (0, 5, 1, 7),
        ],
        dtype=np.int64,
    )
    tet_vertex_indices = local_vertex_indices[:, six_tets].reshape(-1, 4)
    tet_vertex_indices = np.asarray(tet_vertex_indices, dtype=np.int32)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        voxel_masses = density * field.voxel_volume_m3
    if np.any(~np.isfinite(voxel_masses)) or np.any(voxel_masses <= 0.0):
        raise VompIntegrationError(
            "VoMP density and voxel size produce invalid deformable point masses"
        )
    point_masses: np.ndarray = np.zeros(len(unique_corners), dtype=np.float64)
    np.add.at(
        point_masses,
        tet_vertex_indices.reshape(-1),
        np.repeat(voxel_masses / 24.0, 24),
    )
    total_mass = float(np.sum(point_masses, dtype=np.float64))
    center = (
        np.sum(
            points_world_m * point_masses[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        / total_mass
    )
    raw_source_center = np.sum(
        source_centers_world_m * voxel_masses[:, np.newaxis],
        axis=0,
        dtype=np.float64,
    ) / float(np.sum(voxel_masses, dtype=np.float64))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise VompIntegrationError("VoMP deformable mass reduction is invalid")
    # Check conservation in origin-relative coordinates so a large world-space
    # translation does not condition either reduction.  The two weighted means
    # have different reduction lengths, so their forward-error budget grows with
    # both the number of terms and the coordinate scale even when the exact CoM
    # is near zero (where a relative tolerance contributes nothing).
    point_center_from_origin = (
        np.sum(
            (points_world_m - origin_center) * point_masses[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        / total_mass
    )
    snapped_center_from_origin = np.sum(
        (snapped_centers_world_m - origin_center) * voxel_masses[:, np.newaxis],
        axis=0,
        dtype=np.float64,
    ) / float(np.sum(voxel_masses, dtype=np.float64))
    reduction_size = max(len(point_masses), len(voxel_masses), 24)
    reduction_eps = reduction_size * float(np.finfo(np.float64).eps)
    reduction_gamma = reduction_eps / (1.0 - reduction_eps)
    coordinate_scale_m = max(
        pitch,
        float(np.max(np.abs(points_world_m - origin_center))),
        float(np.max(np.abs(snapped_centers_world_m - origin_center))),
    )
    center_atol_m = max(
        pitch * 1.0e-12,
        4.0 * reduction_gamma * coordinate_scale_m,
    )
    if not np.allclose(
        point_center_from_origin,
        snapped_center_from_origin,
        rtol=0.0,
        atol=center_atol_m,
    ):
        raise VompIntegrationError(
            "VoMP voxel-to-tet conversion failed to conserve center of mass"
        )

    points_world_m = np.asarray(points_world_m, dtype=np.float64)
    points_world_m.setflags(write=False)
    tet_vertex_indices.setflags(write=False)
    point_masses.setflags(write=False)
    integrated_volume = field.sample_count * field.voxel_volume_m3
    return VompTetMesh(
        points_world_m=points_world_m,
        tet_vertex_indices=tet_vertex_indices,
        point_masses_kg=point_masses,
        voxel_count=field.sample_count,
        integrated_volume_m3=integrated_volume,
        total_mass_kg=total_mass,
        center_of_mass_world_m=cast(
            _Vector3,
            tuple(float(np.asarray(center)[index]) for index in range(3)),
        ),
        maximum_voxel_center_snap_error_m=float(np.max(voxel_center_snap_errors_m)),
        mass_center_snap_error_m=float(np.linalg.norm(center - raw_source_center)),
        topology_sha256=_topology_digest(
            points_world_m,
            tet_vertex_indices,
        ),
    )


def _applied_schemas(prim: Any) -> set[str]:
    return {str(value) for value in prim.GetPrimTypeInfo().GetAppliedAPISchemas()}


def _is_owned(prim: Any) -> bool:
    return bool(prim.GetCustomDataByKey("physicsAgentVompOwner") == _OWNERSHIP_MARKER)


def _validate_deformable_hierarchy(
    target: Any,
    *,
    sim_path: Any,
    looks_path: Any,
    material_path: Any,
    Usd: Any,
    UsdPhysics: Any,
) -> None:
    target_provenance = target.GetCustomDataByKey("physicsAgentVompDeformable")
    target_owned = (
        isinstance(target_provenance, Mapping)
        and target_provenance.get("adapter") == _OWNERSHIP_MARKER
    )
    target_schemas = _applied_schemas(target)
    body_enabled = target.GetAttribute("physics:bodyEnabled")
    kinematic_enabled = target.GetAttribute("physics:kinematicEnabled")
    starts_asleep = target.GetAttribute("physics:startsAsleep")
    for state_attribute in (body_enabled, kinematic_enabled, starts_asleep):
        if state_attribute and state_attribute.GetNumTimeSamples() > 0:
            raise VompIntegrationError(
                f"target prim {target.GetPath()} attribute "
                f"{state_attribute.GetBaseName()} is time-sampled; "
                "VoMP deformable v1 requires static body state"
            )
    if body_enabled and body_enabled.Get() is False:
        raise VompIntegrationError(
            f"target prim {target.GetPath()} has physics:bodyEnabled=false; "
            "Newton 1.4 would skip the volume deformable"
        )
    if kinematic_enabled and bool(kinematic_enabled.Get()):
        raise VompIntegrationError(
            f"target prim {target.GetPath()} has physics:kinematicEnabled=true; "
            "Newton 1.4 skips kinematic volume deformables"
        )
    if starts_asleep and bool(starts_asleep.Get()):
        raise VompIntegrationError(
            f"target prim {target.GetPath()} has physics:startsAsleep=true; "
            "VoMP deformable v1 requires an active initial body state"
        )
    simulation_owner = target.GetRelationship("physics:simulationOwner")
    if simulation_owner and simulation_owner.GetTargets():
        raise VompIntegrationError(
            f"target prim {target.GetPath()} has physics:simulationOwner targets; "
            "Newton 1.4 does not honor that scene-routing contract"
        )
    if "PhysicsDeformableBodyAPI" in target_schemas and not target_owned:
        raise VompIntegrationError(
            f"target prim {target.GetPath()} already has a non-VoMP deformable body contract"
        )

    current = target.GetParent()
    while current and current.IsValid() and not current.IsPseudoRoot():
        schemas = _applied_schemas(current)
        if "PhysicsRigidBodyAPI" in schemas or "PhysicsDeformableBodyAPI" in schemas:
            raise VompIntegrationError(
                f"target prim {target.GetPath()} is nested below physics body "
                f"{current.GetPath()}"
            )
        current = current.GetParent()

    # PrimDefaultPredicate deliberately omits inactive and undefined (over-only)
    # prims. Check reserved paths directly so those authored opinions cannot be
    # mistaken for empty space and replaced by generated definitions below.
    for reserved_path in (sim_path, looks_path, material_path):
        reserved_prim = target.GetStage().GetPrimAtPath(reserved_path)
        if reserved_prim and not _is_owned(reserved_prim):
            raise VompIntegrationError(
                "reserved Physics Agent prim path is already user-owned: "
                f"{reserved_path}"
            )
        if reserved_prim and (
            not reserved_prim.IsActive() or not reserved_prim.IsDefined()
        ):
            state = "inactive" if not reserved_prim.IsActive() else "undefined"
            raise VompIntegrationError(
                "owned generated Physics Agent prim is "
                f"{state}; refusing to redefine it: {reserved_path}"
            )

    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange(target, predicate):
        schemas = _applied_schemas(prim)
        if prim != target and schemas & {
            "PhysicsBodyAPI",
            "PhysicsDeformableBodyAPI",
        }:
            raise VompIntegrationError(
                f"deformable target {target.GetPath()} contains a nested physics "
                f"body at {prim.GetPath()}"
            )
        if schemas & {"PhysicsRigidBodyAPI", "PhysicsMassAPI"}:
            raise VompIntegrationError(
                f"deformable target {target.GetPath()} contains rigid mass semantics "
                f"at {prim.GetPath()}"
            )
        if "PhysicsCollisionAPI" in schemas and prim.GetPath() != sim_path:
            raise VompIntegrationError(
                f"deformable target {target.GetPath()} contains an existing collision "
                f"contract at {prim.GetPath()}"
            )
        deformable_schemas = schemas & _DEFORMABLE_SIM_SCHEMAS
        if deformable_schemas and prim.GetPath() != sim_path:
            raise VompIntegrationError(
                f"deformable target {target.GetPath()} already contains simulation "
                f"geometry at {prim.GetPath()}"
            )
        if prim.GetPath() == sim_path and _is_owned(prim):
            stale_schemas = sorted(schemas & _OWNED_SIM_STALE_SCHEMAS)
            stale_properties = sorted(
                name
                for name in _OWNED_SIM_STALE_PROPERTIES
                if prim.GetProperty(name).IsAuthored()
            )
            if stale_schemas or stale_properties:
                details = ", ".join([*stale_schemas, *stale_properties])
                raise VompIntegrationError(
                    "owned generated simulation prim contains unsupported stale "
                    f"deformable state: {details}"
                )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.MassAPI):
            raise VompIntegrationError(
                f"deformable target {target.GetPath()} contains rigid mass semantics "
                f"at {prim.GetPath()}"
            )


def _require_si_stage(
    stage: Any, *, UsdGeom: Any, UsdPhysics: Any
) -> tuple[float, float]:
    if not UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        raise VompIntegrationError(
            "USD stage must author metersPerUnit for deformable coordinate alignment"
        )
    if not UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
        raise VompIntegrationError(
            "USD stage must author kilogramsPerUnit for deformable mass alignment"
        )
    meters_per_unit = _require_finite_positive(
        UsdGeom.GetStageMetersPerUnit(stage),
        label="USD metersPerUnit",
    )
    kilograms_per_unit = _require_finite_positive(
        UsdPhysics.GetStageKilogramsPerUnit(stage),
        label="USD kilogramsPerUnit",
    )
    if not math.isclose(meters_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise VompIntegrationError(
            "Newton 1.4's USD importer does not convert non-SI length units; "
            "VoMP deformable v1 requires metersPerUnit = 1"
        )
    if not math.isclose(kilograms_per_unit, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise VompIntegrationError(
            "Newton 1.4's USD importer does not convert non-SI mass units; "
            "VoMP deformable v1 requires kilogramsPerUnit = 1"
        )
    return meters_per_unit, kilograms_per_unit


def _require_float32_array(
    values: np.ndarray,
    *,
    label: str,
    positive: bool = False,
) -> np.ndarray:
    if np.any(~np.isfinite(values)) or np.any(np.abs(values) > _FLOAT32_MAX):
        raise VompIntegrationError(f"{label} cannot be represented by USD float fields")
    converted = np.asarray(values, dtype=np.float32)
    if positive and (np.any(values <= 0.0) or np.any(converted <= 0.0)):
        raise VompIntegrationError(f"{label} underflows USD float fields")
    return cast(np.ndarray, converted)


def _preflight_vomp_deformable_target(
    stage: Any,
    target: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate the complete target contract before inference or authoring."""

    from pxr import Usd, UsdGeom, UsdPhysics

    if target.IsInstanceProxy() or target.IsInstanceable():
        raise VompIntegrationError(
            f"target prim {target.GetPath()} is instance-backed; deinstance it first"
        )
    if not target.IsA(UsdGeom.Xform):
        raise VompIntegrationError(
            f"target prim {target.GetPath()} must be a UsdGeom.Xform so visual "
            "and generated simulation geometry remain separate"
        )
    meters_per_unit, _ = _require_si_stage(
        stage,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )
    rotation_local_to_world, translation_world_m = _target_world_frame(
        target,
        meters_per_unit=meters_per_unit,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    target_path = target.GetPath()
    sim_path = target_path.AppendChild(_SIMULATION_PRIM_NAME)
    looks_path = target_path.AppendChild(_LOOKS_PRIM_NAME)
    material_path = looks_path.AppendChild(_MATERIAL_PRIM_NAME)
    _validate_deformable_hierarchy(
        target,
        sim_path=sim_path,
        looks_path=looks_path,
        material_path=material_path,
        Usd=Usd,
        UsdPhysics=UsdPhysics,
    )
    return meters_per_unit, rotation_local_to_world, translation_world_m


def apply_vomp_volume_deformable(
    usd_path: str | Path,
    vomp_npz_path: str | Path,
    output_usd_path: str | Path,
    *,
    target_prim_path: str,
    voxel_size_m: float,
    coordinate_unit_meters: float,
    complete_voxel_field: bool,
    coordinate_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material_reduction: MaterialReductionPolicy = "reject",
    max_deformable_voxels: int = DEFAULT_MAX_DEFORMABLE_VOXELS,
    provenance_path: str | Path | None = None,
    evidence_provenance: Mapping[str, Any] | None = None,
    expected_npz_sha256: str | None = None,
    expected_composition_hashes: Mapping[str | Path, str] | None = None,
) -> VompDeformableApplyResult:
    """Author one AOUSD-draft volume deformable from complete VoMP output.

    The v1 runtime profile is intentionally narrow: one deinstanced Xform body,
    one generated TetMesh, one Newton 1.4 material, and SI stage units. Existing
    rigid-body or independent deformable contracts fail before publication.
    """

    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    source = Path(usd_path).expanduser().resolve()
    npz_path = Path(vomp_npz_path).expanduser().resolve()
    output = Path(output_usd_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input USD not found: {source}")
    if source == output:
        raise VompIntegrationError("output_usd_path must differ from usd_path")
    _validate_vomp_input_layer(source)
    if output.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise VompIntegrationError(
            "VoMP deformable output must use .usd, .usda, or .usdc"
        )
    report = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else output.with_name(f"{output.stem}.vomp_deformable.json")
    )
    if report in {source, npz_path, output}:
        raise VompIntegrationError(
            "provenance_path must differ from the USD and NPZ input/output paths"
        )
    if complete_voxel_field is not True:
        raise VompIntegrationError(
            "complete_voxel_field=True is required; deformable topology cannot "
            "be authored from capped or subsampled VoMP output"
        )
    if expected_npz_sha256 is not None:
        expected_npz_sha256 = _require_sha256(
            expected_npz_sha256,
            label="expected_npz_sha256",
        )
    normalized_expected_composition = (
        _normalize_composition_hashes(expected_composition_hashes, source=source)
        if expected_composition_hashes is not None
        else None
    )

    field = load_vomp_voxel_field(
        npz_path,
        voxel_size_m=voxel_size_m,
        coordinate_unit_meters=coordinate_unit_meters,
        coordinate_offset_m=coordinate_offset_m,
    )
    if expected_npz_sha256 is not None and field.source_sha256 != expected_npz_sha256:
        raise VompIntegrationError(
            "VoMP NPZ does not match the worker-reported SHA-256 digest"
        )
    reduction = reduce_vomp_elastic_field(field, policy=material_reduction)
    topology = build_vomp_tet_mesh(field, max_voxels=max_deformable_voxels)

    root_layer = Sdf.Layer.FindOrOpen(str(source))
    if root_layer is None:
        raise VompIntegrationError(f"Failed to open USD layer: {source}")
    if root_layer.dirty:
        raise VompIntegrationError(
            "input USD root layer has unsaved in-memory edits; refusing to mix "
            "them into VoMP output"
        )
    inspection_stage = Usd.Stage.Open(root_layer)
    if not inspection_stage:
        raise VompIntegrationError(f"Failed to open USD stage: {source}")
    composition_hashes = _reload_and_hash_composition_layers(inspection_stage)
    if (
        normalized_expected_composition is not None
        and composition_hashes != normalized_expected_composition
    ):
        raise VompIntegrationError(
            "input USD composition changed after OVRTX evidence capture"
        )
    source_usd_sha256 = composition_hashes.get(source) or _sha256(source)
    composition_paths = set(composition_hashes)
    if output in composition_paths or report in composition_paths:
        raise VompIntegrationError(
            "output and provenance paths must not overwrite any composed input layer"
        )

    session_layer = Sdf.Layer.CreateAnonymous("physics_agent_vomp_deformable.usda")
    stage = Usd.Stage.Open(root_layer, session_layer)
    if not stage:
        raise VompIntegrationError(f"Failed to open USD stage: {source}")
    stage.SetEditTarget(session_layer)
    try:
        target_path = Sdf.Path(target_prim_path)
    except Exception as exc:
        raise VompIntegrationError(
            f"target_prim_path is not a valid USD path: {target_prim_path!r}"
        ) from exc
    if (
        not target_path.IsAbsolutePath()
        or not target_path.IsPrimPath()
        or target_path == Sdf.Path.absoluteRootPath
    ):
        raise VompIntegrationError("target_prim_path must be an absolute prim path")
    target = stage.GetPrimAtPath(target_path)
    if not target or not target.IsValid():
        raise VompIntegrationError(f"Target prim not found: {target_prim_path}")
    (
        meters_per_unit,
        rotation_local_to_world,
        translation_world_m,
    ) = _preflight_vomp_deformable_target(
        stage,
        target,
    )
    sim_path = target_path.AppendChild(_SIMULATION_PRIM_NAME)
    looks_path = target_path.AppendChild(_LOOKS_PRIM_NAME)
    material_path = looks_path.AppendChild(_MATERIAL_PRIM_NAME)
    world_bound_m = _validate_geometry_association(
        target,
        field,
        meters_per_unit=meters_per_unit,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )

    points_local_stage = (
        (topology.points_world_m - translation_world_m) @ rotation_local_to_world
    ) / meters_per_unit
    points_local_m = points_local_stage * meters_per_unit
    points_f32 = _require_float32_array(
        points_local_stage,
        label="generated deformable points",
    )
    if np.unique(points_f32, axis=0).shape[0] != points_f32.shape[0]:
        raise VompIntegrationError(
            "generated deformable points collapse when represented as USD point3f"
        )
    tet_points = points_f32[topology.tet_vertex_indices].astype(np.float64)
    signed_six_volumes = np.einsum(
        "ij,ij->i",
        np.cross(
            tet_points[:, 1] - tet_points[:, 0],
            tet_points[:, 2] - tet_points[:, 0],
        ),
        tet_points[:, 3] - tet_points[:, 0],
    )
    if np.any(~np.isfinite(signed_six_volumes)) or np.any(signed_six_volumes <= 0.0):
        raise VompIntegrationError(
            "generated deformable tetrahedra collapse when represented as USD point3f"
        )
    authored_tet_volumes_m3 = signed_six_volumes * (meters_per_unit**3) / 6.0
    expected_tet_volume_m3 = field.voxel_volume_m3 / 6.0
    tet_volume_relative_errors = (
        np.abs(authored_tet_volumes_m3 - expected_tet_volume_m3)
        / expected_tet_volume_m3
    )
    maximum_tet_volume_relative_error = float(np.max(tet_volume_relative_errors))
    authored_volume_m3 = float(np.sum(authored_tet_volumes_m3, dtype=np.float64))
    volume_relative_error = (
        abs(authored_volume_m3 - topology.integrated_volume_m3)
        / topology.integrated_volume_m3
    )
    point_masses_f32 = _require_float32_array(
        topology.point_masses_kg,
        label="generated deformable point masses",
        positive=True,
    )
    authored_mass_kg = float(np.sum(point_masses_f32, dtype=np.float64))
    mass_absolute_error_kg = abs(authored_mass_kg - topology.total_mass_kg)
    mass_relative_error = mass_absolute_error_kg / topology.total_mass_kg
    if mass_relative_error > _MASS_FLOAT32_RTOL:
        raise VompIntegrationError(
            "generated deformable point masses lose too much precision when "
            "represented as USD floats"
        )
    # Compare mass centers before restoring the target's world translation.
    # Independently reducing two centers near a large world origin can add an
    # error of several float64 ULPs even when their local values agree.  A
    # shared local origin also conditions reductions for geometry that is far
    # from the target origin.
    center_origin_local_m = np.min(points_local_m, axis=0)
    source_center_from_origin_m = (
        np.sum(
            (points_local_m - center_origin_local_m)
            * topology.point_masses_kg[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        / topology.total_mass_kg
    )
    authored_center_from_origin_m = (
        np.sum(
            (points_f32.astype(np.float64) * meters_per_unit - center_origin_local_m)
            * point_masses_f32[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        / authored_mass_kg
    )
    source_center_local_m = center_origin_local_m + source_center_from_origin_m
    authored_center_local_m = center_origin_local_m + authored_center_from_origin_m
    authored_center_world_m = (
        authored_center_local_m @ rotation_local_to_world.T + translation_world_m
    )
    source_center_world_m = (
        source_center_local_m @ rotation_local_to_world.T + translation_world_m
    )
    authored_center_world = cast(
        _Vector3,
        tuple(float(authored_center_world_m[index]) for index in range(3)),
    )
    center_error_m = float(
        np.linalg.norm(authored_center_from_origin_m - source_center_from_origin_m)
    )

    # Newton 1.4 stores the USD world affine in a default ``wp.mat44`` (float32)
    # before baking authored point3f positions into world-space float32 particles.
    # Quantize the affine first as Newton does: rounding only the final points can
    # miss collapse at a float32 bin boundary.
    newton_rotation_f32 = _require_float32_array(
        rotation_local_to_world,
        label="Newton 1.4 world rotation",
    )
    newton_translation_stage_f32 = _require_float32_array(
        translation_world_m / meters_per_unit,
        label="Newton 1.4 world translation",
    )
    points_world_from_authored_stage = points_f32.astype(
        np.float64
    ) @ newton_rotation_f32.astype(np.float64).T + newton_translation_stage_f32.astype(
        np.float64
    )
    newton_points_world_f32 = _require_float32_array(
        points_world_from_authored_stage,
        label="Newton 1.4 world-space deformable points",
    )
    if (
        np.unique(newton_points_world_f32, axis=0).shape[0]
        != newton_points_world_f32.shape[0]
    ):
        raise VompIntegrationError(
            "Newton 1.4 world-space deformable points collapse when represented "
            "as float32 particles"
        )
    newton_tet_points_stage = newton_points_world_f32[
        topology.tet_vertex_indices
    ].astype(np.float64)
    newton_signed_six_volumes_stage3 = np.einsum(
        "ij,ij->i",
        np.cross(
            newton_tet_points_stage[:, 1] - newton_tet_points_stage[:, 0],
            newton_tet_points_stage[:, 2] - newton_tet_points_stage[:, 0],
        ),
        newton_tet_points_stage[:, 3] - newton_tet_points_stage[:, 0],
    )
    if np.any(~np.isfinite(newton_signed_six_volumes_stage3)) or np.any(
        newton_signed_six_volumes_stage3 <= 0.0
    ):
        raise VompIntegrationError(
            "Newton 1.4 world-space deformable tetrahedra collapse when represented "
            "as float32 particles"
        )
    newton_tet_volumes_m3 = newton_signed_six_volumes_stage3 * meters_per_unit**3 / 6.0
    newton_tet_volume_relative_errors = (
        np.abs(newton_tet_volumes_m3 - expected_tet_volume_m3) / expected_tet_volume_m3
    )
    newton_maximum_tet_volume_relative_error = float(
        np.max(newton_tet_volume_relative_errors)
    )
    newton_authored_volume_m3 = float(np.sum(newton_tet_volumes_m3, dtype=np.float64))
    newton_volume_relative_error = (
        abs(newton_authored_volume_m3 - topology.integrated_volume_m3)
        / topology.integrated_volume_m3
    )
    newton_center_offset_from_source_m = (
        np.sum(
            (
                newton_points_world_f32.astype(np.float64) * meters_per_unit
                - source_center_world_m
            )
            * point_masses_f32[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        / authored_mass_kg
    )
    newton_center_error_m = float(np.linalg.norm(newton_center_offset_from_source_m))
    center_tolerance_m = field.voxel_size_m * _GEOMETRY_FLOAT32_CENTER_ATOL_VOXELS
    if newton_center_error_m > center_tolerance_m:
        raise VompIntegrationError(
            "Newton 1.4 world-space deformable mass center is materially displaced "
            "after float32 affine quantization; rebase the target closer to the "
            "world origin or use a larger voxel size"
        )
    if (
        maximum_tet_volume_relative_error > _GEOMETRY_FLOAT32_VOLUME_RTOL
        or volume_relative_error > _GEOMETRY_FLOAT32_VOLUME_RTOL
        or newton_maximum_tet_volume_relative_error > _GEOMETRY_FLOAT32_VOLUME_RTOL
        or newton_volume_relative_error > _GEOMETRY_FLOAT32_VOLUME_RTOL
        or center_error_m > center_tolerance_m
    ):
        raise VompIntegrationError(
            "generated deformable geometry or mass center is materially distorted "
            "when represented as USD point3f/float or Newton 1.4 world-space "
            "float32 particles; rebase the target closer to the world origin, use "
            "a larger voxel size, or rescale the density field"
        )
    elastic_values = np.asarray(
        [
            reduction.youngs_modulus_pa,
            reduction.poisson_ratio,
            _stable_mean(field.density_kg_m3),
        ],
        dtype=np.float64,
    )
    elastic_f32 = _require_float32_array(
        elastic_values,
        label="generated deformable material values",
    )
    if elastic_f32[0] <= 0.0 or elastic_f32[2] <= 0.0:
        raise VompIntegrationError(
            "generated deformable material underflows USD floats"
        )

    # Draft schemas are unregistered raw tokens in OpenUSD 25.5, so USD cannot
    # expand PhysicsDeformableBodyAPI's pseudo-base dependency for us.
    target.AddAppliedSchema("PhysicsBodyAPI")
    target.AddAppliedSchema("PhysicsDeformableBodyAPI")
    target.CreateAttribute(
        "physics:bodyEnabled",
        Sdf.ValueTypeNames.Bool,
        custom=False,
    ).Set(True)
    tet_mesh = UsdGeom.TetMesh.Define(stage, sim_path)
    tet_prim = tet_mesh.GetPrim()
    tet_prim.SetCustomDataByKey("physicsAgentVompOwner", _OWNERSHIP_MARKER)
    tet_mesh.ClearXformOpOrder()
    tet_mesh.SetResetXformStack(False)
    tet_mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded).Set(
        UsdGeom.Tokens.rightHanded
    )
    if tet_mesh.GetOrderedXformOps() or tet_mesh.GetResetXformStack():
        raise VompIntegrationError(
            "Unable to canonicalize generated deformable simulation transform"
        )
    tet_prim.AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
    tet_prim.AddAppliedSchema("PhysicsCollisionAPI")
    tet_prim.CreateAttribute(
        "physics:collisionEnabled",
        Sdf.ValueTypeNames.Bool,
        custom=False,
    ).Set(True)
    tet_mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    tet_mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points_f32))
    tet_mesh.CreateTetVertexIndicesAttr().Set(
        Vt.Vec4iArray.FromNumpy(topology.tet_vertex_indices)
    )
    extent = np.asarray(
        [np.min(points_f32, axis=0), np.max(points_f32, axis=0)],
        dtype=np.float32,
    )
    tet_mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent))
    tet_prim.CreateAttribute(
        "physics:masses",
        Sdf.ValueTypeNames.FloatArray,
        custom=False,
    ).Set(Vt.FloatArray.FromNumpy(point_masses_f32))

    looks = UsdGeom.Scope.Define(stage, looks_path).GetPrim()
    looks.SetCustomDataByKey("physicsAgentVompOwner", _OWNERSHIP_MARKER)
    material = UsdShade.Material.Define(stage, material_path)
    material_prim = material.GetPrim()
    material_prim.SetCustomDataByKey("physicsAgentVompOwner", _OWNERSHIP_MARKER)
    material_prim.AddAppliedSchema("PhysicsMaterialAPI")
    material_prim.AddAppliedSchema("PhysicsVolumeDeformableMaterialAPI")
    for name, value in (
        ("density", elastic_f32[2]),
        ("youngsModulus", elastic_f32[0]),
        ("poissonsRatio", elastic_f32[1]),
    ):
        material_prim.CreateAttribute(
            f"physics:{name}",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(float(value))
    binding = UsdShade.MaterialBindingAPI.Apply(tet_prim)
    if not binding.Bind(material, materialPurpose="physics"):
        raise VompIntegrationError("Failed to bind the generated deformable material")
    bound_material, _ = UsdShade.MaterialBindingAPI(tet_prim).ComputeBoundMaterial(
        "physics"
    )
    if not bound_material or bound_material.GetPath() != material_path:
        raise VompIntegrationError(
            "generated deformable physics material binding is overridden by a "
            "stronger authored binding"
        )

    authored_poisson_ratio = float(elastic_f32[1])
    newton_effective_poisson_ratio = max(
        _NEWTON_POISSON_MIN,
        min(authored_poisson_ratio, _NEWTON_POISSON_MAX),
    )

    provenance: dict[str, Any] = {
        "schemaVersion": 1,
        "adapter": _OWNERSHIP_MARKER,
        "schemaProfile": {
            "name": NEWTON_DEFORMABLE_PROFILE,
            "status": "draft",
            "authoringProposal": AOUSD_DEFORMABLE_PROPOSAL_URL,
            "authoringProposalCommit": AOUSD_DEFORMABLE_SCHEMA_COMMIT,
            "newtonRuntimeProposalBaseline": NEWTON_1_4_AOUSD_BASELINE_COMMIT,
            "validatedRuntime": "newton==1.4.0",
            "runtimeNewtonSchemaPlugin": "newton-usd-schemas==0.4.1",
            "draftSchemaRegistration": "unregistered_raw_api_tokens",
        },
        "source": {
            "npzSha256": field.source_sha256,
            "npzSchema": field.source_schema,
            "usdSha256": source_usd_sha256,
        },
        "association": {
            "coordinateSpace": "world",
            "coordinateUnitMeters": float(coordinate_unit_meters),
            "coordinateOffsetM": [float(value) for value in coordinate_offset_m],
            "targetPrimPath": str(target_path),
            "simulationPrimPath": str(sim_path),
            "materialPrimPath": str(material_path),
            "targetWorldBoundMinM": list(world_bound_m[0]),
            "targetWorldBoundMaxM": list(world_bound_m[1]),
        },
        "voxelField": {
            "declaredComplete": True,
            "sampleCount": field.sample_count,
            "voxelSizeM": field.voxel_size_m,
            "voxelVolumeM3": field.voxel_volume_m3,
            "integratedVolumeM3": topology.integrated_volume_m3,
            "densityKgM3": _stats(field.density_kg_m3),
            "youngsModulusPa": _stats(field.youngs_modulus_pa),
            "poissonRatio": _stats(field.poisson_ratio),
        },
        "topology": {
            "generator": "global-freudenthal-six-tet-per-voxel-v1",
            "pointCount": topology.point_count,
            "tetCount": topology.tet_count,
            "sha256": topology.topology_sha256,
            "maxDeformableVoxels": int(max_deformable_voxels),
            "maximumVoxelCenterSnapErrorM": (
                topology.maximum_voxel_center_snap_error_m
            ),
            "massCenterSnapErrorM": topology.mass_center_snap_error_m,
        },
        "geometryEncoding": {
            "pointType": "point3f",
            "sourceVolumeM3": topology.integrated_volume_m3,
            "authoredVolumeM3": authored_volume_m3,
            "volumeRelativeError": volume_relative_error,
            "maximumTetVolumeRelativeError": maximum_tet_volume_relative_error,
            "newtonWorldParticleType": "float32",
            "newtonWorldAuthoredVolumeM3": newton_authored_volume_m3,
            "newtonWorldVolumeRelativeError": newton_volume_relative_error,
            "newtonWorldMaximumTetVolumeRelativeError": (
                newton_maximum_tet_volume_relative_error
            ),
            "newtonWorldCenterOfMassErrorM": newton_center_error_m,
            "centerOfMassErrorM": center_error_m,
            "centerComparisonSpace": "target-local-origin-relative",
            "volumeRelativeTolerance": _GEOMETRY_FLOAT32_VOLUME_RTOL,
            "centerToleranceVoxels": _GEOMETRY_FLOAT32_CENTER_ATOL_VOXELS,
        },
        "massMapping": {
            "mode": "exact-piecewise-voxel-density-lumped-from-six-tet-fem",
            "massKg": authored_mass_kg,
            "sourceMassKg": topology.total_mass_kg,
            "absoluteFloat32ErrorKg": mass_absolute_error_kg,
            "relativeFloat32Error": mass_relative_error,
            "float32RelativeTolerance": _MASS_FLOAT32_RTOL,
            "centerOfMassWorldM": list(authored_center_world),
            "sourceCenterOfMassWorldM": source_center_world_m.tolist(),
            "centerOfMassTargetLocalM": authored_center_local_m.tolist(),
            "sourceCenterOfMassTargetLocalM": source_center_local_m.tolist(),
            "perPointAttribute": "physics:masses",
            "pointMassSha256": _point_mass_digest(point_masses_f32),
        },
        "elasticReduction": {
            "policy": reduction.policy,
            "heterogeneous": reduction.heterogeneous,
            "validationStatus": reduction.validation_status,
            "youngsModulusPa": reduction.youngs_modulus_pa,
            "poissonRatio": reduction.poisson_ratio,
            "statistics": reduction.statistics,
        },
        "authoredMaterial": {
            "densityKgM3": float(elastic_f32[2]),
            "youngsModulusPa": float(elastic_f32[0]),
            "poissonRatio": authored_poisson_ratio,
            "newtonEffectivePoissonRatio": newton_effective_poisson_ratio,
        },
        "fieldUse": {
            "density": "spatially_preserved_as_lumped_point_mass",
            "youngsModulus": (
                "homogeneous_volume_average"
                if reduction.heterogeneous
                else "uniform_direct"
            ),
            "poissonRatio": (
                "homogeneous_volume_average"
                if reduction.heterogeneous
                else "uniform_direct"
            ),
        },
        "limitations": [
            "one_volume_deformable_body",
            "si_stage_units_only",
            "generated_voxel_conforming_simulation_mesh",
            "visual_deformation_embedding_not_authored",
            "physx_omniphysics_schema_not_authored",
            "runtime_trajectory_validation_not_performed",
        ],
    }
    if evidence_provenance is not None:
        provenance["evidence"] = _json_mapping(evidence_provenance)
    exact_renderer_metadata: Any = None
    provenance_evidence = provenance.get("evidence")
    if isinstance(provenance_evidence, dict):
        provenance_rendering = provenance_evidence.get("rendering")
        if isinstance(provenance_rendering, dict):
            exact_renderer_metadata = provenance_rendering.get("rendererMetadata")
    usd_provenance = _json_custom_data(provenance)
    usd_evidence = usd_provenance.get("evidence")
    if isinstance(usd_evidence, dict):
        usd_rendering = usd_evidence.get("rendering")
        if isinstance(usd_rendering, dict):
            usd_rendering.pop("rendererMetadata", None)
            if exact_renderer_metadata is not None:
                usd_rendering["rendererMetadataJson"] = json.dumps(
                    exact_renderer_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
    target.SetCustomDataByKey(
        "physicsAgentVompDeformable",
        usd_provenance,
    )

    _verify_composition_hashes(composition_hashes)
    serialized_stage = None
    temporary_report: Path | None = None
    try:
        serialized_stage = _serialize_stage(stage, output)
        temporary_report = _serialize_json(report, provenance)
        _verify_composition_hashes(composition_hashes)
        _verify_npz_digest(npz_path, field.source_sha256)
        _publish_artifact_pair(
            temporary_report=temporary_report,
            report=report,
            temporary_output=serialized_stage.root,
            output=output,
            temporary_sidecar=serialized_stage.sidecar,
            sidecar=output.parent / portable_sidecar_name(output),
        )
    except VompIntegrationError:
        raise
    except Exception as exc:
        raise VompIntegrationError(
            "Unable to publish VoMP deformable USD and provenance artifacts"
        ) from exc
    finally:
        if serialized_stage is not None:
            shutil.rmtree(serialized_stage.staging_dir, ignore_errors=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)

    return VompDeformableApplyResult(
        output_usd_path=output,
        provenance_path=report,
        target_prim_path=str(target_path),
        simulation_prim_path=str(sim_path),
        material_prim_path=str(material_path),
        sample_count=field.sample_count,
        point_count=topology.point_count,
        tet_count=topology.tet_count,
        mass_kg=authored_mass_kg,
        center_of_mass_world_m=authored_center_world,
        material_reduction=reduction,
    )


__all__ = [
    "AOUSD_DEFORMABLE_PROPOSAL_URL",
    "AOUSD_DEFORMABLE_SCHEMA_COMMIT",
    "DEFAULT_MAX_DEFORMABLE_VOXELS",
    "NEWTON_1_4_AOUSD_BASELINE_COMMIT",
    "NEWTON_DEFORMABLE_PROFILE",
    "MaterialReductionPolicy",
    "VompDeformableApplyResult",
    "VompElasticReduction",
    "VompTetMesh",
    "apply_vomp_volume_deformable",
    "build_vomp_tet_mesh",
    "reduce_vomp_elastic_field",
]
