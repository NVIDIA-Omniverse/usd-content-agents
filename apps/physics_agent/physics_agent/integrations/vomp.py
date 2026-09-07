# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author rigid-body mass properties from a precomputed VoMP voxel field.

This adapter intentionally consumes an NPZ artifact instead of importing VoMP.
VoMP's model, CUDA, rendering, and simulation stacks therefore remain external
to Physics Agent. Only density participates in rigid-body derivation; Young's
modulus and Poisson ratio are retained as provenance statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from world_understanding.functions.graphics.so_export import (
    PORTABLE_SIDECAR_MARKER_BYTES,
    PORTABLE_SIDECAR_MARKER_NAME,
    export_stage_portably,
    portable_sidecar_name,
)

_MAX_ARCHIVE_ENTRIES = 64
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1 << 30
_MAX_VOXEL_SAMPLES = 10_000_000
_USD_LAYER_EXTENSIONS = {".usd", ".usda", ".usdc"}
_LATTICE_RELATIVE_TOLERANCE = 2e-4
_RIGID_TRANSFORM_TOLERANCE = 1e-8
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_DEFORMABLE_API_SCHEMAS = frozenset(
    {
        "PhysicsCurvesDeformableSimAPI",
        "PhysicsDeformableBodyAPI",
        "PhysicsSurfaceDeformableSimAPI",
        "PhysicsVolumeDeformableSimAPI",
    }
)
_VOXEL_CENTER_SEGMENT_IDS = frozenset(
    {frozenset({"voxel_material"}), frozenset({"voxel_center"})}
)

type _Vector3 = tuple[float, float, float]
type _Matrix3 = tuple[_Vector3, _Vector3, _Vector3]
type _Quaternion = tuple[float, float, float, float]


class VompIntegrationError(ValueError):
    """Raised when VoMP evidence cannot be mapped to mass properties safely."""


@dataclass(frozen=True)
class _VompVoxelField:
    """Validated VoMP values at complete, axis-aligned cubic voxel centers."""

    coordinates_world_m: np.ndarray
    density_kg_m3: np.ndarray
    youngs_modulus_pa: np.ndarray
    poisson_ratio: np.ndarray
    voxel_size_m: float
    source_schema: str
    source_sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.density_kg_m3.shape[0])

    @property
    def voxel_volume_m3(self) -> float:
        return self.voxel_size_m * self.voxel_size_m * self.voxel_size_m


# Public additive facade for consumers that need the complete validated VoMP
# field rather than its rigid-body reduction. Keep the private name as an alias
# so existing internal imports and tests remain source compatible.
VompVoxelField = _VompVoxelField


@dataclass(frozen=True)
class _VoxelMassProperties:
    """Density-integrated mass properties in the VoMP world frame and SI."""

    mass_kg: float
    center_of_mass_world_m: _Vector3
    inertia_tensor_world_kg_m2: _Matrix3
    integrated_volume_m3: float


@dataclass(frozen=True)
class AuthoredMassProperties:
    """Mass properties represented in the target rigid body's local frame."""

    mass_kg: float
    center_of_mass_local_m: _Vector3
    diagonal_inertia_kg_m2: _Vector3
    principal_axes_wxyz: _Quaternion


@dataclass(frozen=True)
class VompApplyResult:
    """Paths and values emitted by :func:`apply_vomp_mass_properties`."""

    output_usd_path: Path
    provenance_path: Path
    mass_properties: AuthoredMassProperties
    sample_count: int


@dataclass(frozen=True)
class _SerializedStageArtifacts:
    """Portable USD root and sidecar staged for transactional publication."""

    root: Path
    sidecar: Path | None
    staging_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_positive(value: object, *, label: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise VompIntegrationError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise VompIntegrationError(
            f"{label} must be finite and positive; got {value!r}"
        )
    return result


def _validate_vomp_input_layer(source: Path) -> None:
    if source.suffix.lower() not in _USD_LAYER_EXTENSIONS:
        raise VompIntegrationError(
            "VoMP mass authoring accepts .usd, .usda, or .usdc input layers; "
            "USDZ packages must be unpacked first"
        )


def _require_numeric_array(
    value: object,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise VompIntegrationError(
            f"{label} must use a real numeric dtype; got {array.dtype}"
        )
    if shape is not None and array.shape != shape:
        raise VompIntegrationError(
            f"{label} must have shape {shape}; got {array.shape}"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise VompIntegrationError(f"{label} contains non-finite values")
    result.setflags(write=False)
    return cast(np.ndarray, result)


def _inspect_npz_archive(path: Path) -> None:
    if path.suffix.lower() != ".npz":
        raise VompIntegrationError(f"VoMP input must be an .npz archive: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"VoMP NPZ not found: {path}")
    if not zipfile.is_zipfile(path):
        raise VompIntegrationError(f"VoMP input is not a valid NPZ archive: {path}")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > _MAX_ARCHIVE_ENTRIES:
            raise VompIntegrationError(
                "VoMP NPZ must contain between 1 and "
                f"{_MAX_ARCHIVE_ENTRIES} entries; got {len(entries)}"
            )
        if len({info.filename for info in entries}) != len(entries):
            raise VompIntegrationError("VoMP NPZ contains duplicate archive entries")
        if any(info.flag_bits & 0x1 for info in entries):
            raise VompIntegrationError("Encrypted NPZ entries are not supported")
        uncompressed_size = sum(info.file_size for info in entries)
        if uncompressed_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise VompIntegrationError(
                "VoMP NPZ expands beyond the 1 GiB safety limit: "
                f"{uncompressed_size} bytes"
            )


def _structured_voxel_arrays(
    voxel_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    names = set(voxel_data.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "density",
        "youngs_modulus",
        "poissons_ratio",
        "segment_id",
    }
    missing = sorted(required - names)
    if missing:
        raise VompIntegrationError(
            f"VoMP voxel_data is missing required field(s): {', '.join(missing)}"
        )
    segments = np.asarray(voxel_data["segment_id"])
    if segments.dtype.kind not in {"S", "U"}:
        raise VompIntegrationError("VoMP voxel_data.segment_id must be a string field")
    normalized_segments = (
        np.char.decode(segments, "utf-8") if segments.dtype.kind == "S" else segments
    )
    unique_segments = {str(value) for value in np.unique(normalized_segments)}
    if frozenset(unique_segments) not in _VOXEL_CENTER_SEGMENT_IDS:
        raise VompIntegrationError(
            "Rigid-body integration requires VoMP voxel-center output; "
            f"segment_id values were {sorted(unique_segments)!r}. Re-run VoMP "
            "with query_points='voxel_centers'."
        )
    coordinates = np.column_stack((voxel_data["x"], voxel_data["y"], voxel_data["z"]))
    return (
        coordinates,
        voxel_data["density"],
        voxel_data["youngs_modulus"],
        voxel_data["poissons_ratio"],
        "vomp.voxel_data.v1",
    )


def _direct_voxel_arrays(
    archive: Any,
    names: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if "query_coords_world" in names:
        raise VompIntegrationError(
            "NPZ contains query_coords_world. Density at arbitrary query points "
            "has no defined voxel volume; use VoMP query_points='voxel_centers'."
        )
    required = {
        "voxel_coords_world",
        "density",
        "youngs_modulus",
        "poisson_ratio",
    }
    missing = sorted(required - names)
    if missing:
        raise VompIntegrationError(
            f"VoMP NPZ is missing required array(s): {', '.join(missing)}"
        )
    if "segment_id" in names:
        segments = np.asarray(archive["segment_id"])
        if segments.dtype.kind not in {"S", "U"}:
            raise VompIntegrationError("segment_id must be a string array")
        if segments.dtype.kind == "S":
            segments = np.char.decode(segments, "utf-8")
        unique_segments = {str(value) for value in np.unique(segments)}
        if frozenset(unique_segments) not in _VOXEL_CENTER_SEGMENT_IDS:
            raise VompIntegrationError(
                "Rigid-body integration requires voxel-center samples; "
                f"segment_id values were {sorted(unique_segments)!r}"
            )
    return (
        archive["voxel_coords_world"],
        archive["density"],
        archive["youngs_modulus"],
        archive["poisson_ratio"],
        "vomp.direct_arrays.v1",
    )


def _validate_voxel_lattice(
    coordinates_world_m: np.ndarray, voxel_size_m: float
) -> None:
    origin = np.min(coordinates_world_m, axis=0)
    normalized = (coordinates_world_m - origin) / voxel_size_m
    if np.max(np.abs(normalized)) > float(np.iinfo(np.int64).max) / 2.0:
        raise VompIntegrationError("VoMP voxel coordinates exceed lattice index limits")
    lattice_indices = np.rint(normalized)
    error_m = np.abs(normalized - lattice_indices) * voxel_size_m
    coordinate_scale_m = max(
        float(np.finfo(np.float32).tiny),
        float(np.max(np.abs(coordinates_world_m))),
    )
    float32_roundoff_m = 4.0 * float(np.finfo(np.float32).eps) * coordinate_scale_m
    tolerance_m = max(
        voxel_size_m * _LATTICE_RELATIVE_TOLERANCE,
        float32_roundoff_m,
    )
    max_error_m = float(np.max(error_m))
    if max_error_m > tolerance_m:
        raise VompIntegrationError(
            "voxel_coords_world are not aligned to the declared cubic lattice: "
            f"maximum alignment error {max_error_m:.6g} m exceeds "
            f"{tolerance_m:.6g} m"
        )
    integer_indices = np.asarray(lattice_indices, dtype=np.int64)
    minimum = integer_indices.min(axis=0)
    maximum = integer_indices.max(axis=0)
    extents = tuple(int(maximum[axis]) - int(minimum[axis]) + 1 for axis in range(3))
    if math.prod(extents) <= int(np.iinfo(np.int64).max):
        shifted = integer_indices - minimum
        scalar_keys = (shifted[:, 0] * extents[1] + shifted[:, 1]) * extents[
            2
        ] + shifted[:, 2]
        unique_count = int(np.unique(scalar_keys).shape[0])
    else:
        unique_count = int(np.unique(integer_indices, axis=0).shape[0])
    if unique_count != coordinates_world_m.shape[0]:
        raise VompIntegrationError(
            "VoMP NPZ maps multiple samples to the same declared voxel cell"
        )


def _load_vomp_voxel_field(
    npz_path: str | Path,
    *,
    voxel_size_m: float,
    coordinate_unit_meters: float,
    coordinate_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> _VompVoxelField:
    """Load and validate a precomputed VoMP voxel-center NPZ artifact.

    ``coordinate_unit_meters`` and ``coordinate_offset_m`` explicitly map the
    archive's ``voxel_coords_world`` values into SI world coordinates. VoMP's
    standard NPZ does not retain voxel size, so ``voxel_size_m`` is mandatory.
    """

    path = Path(npz_path).expanduser().resolve()
    _inspect_npz_archive(path)
    source_sha256 = _sha256(path)
    size_m = _require_finite_positive(voxel_size_m, label="voxel_size_m")
    unit_m = _require_finite_positive(
        coordinate_unit_meters,
        label="coordinate_unit_meters",
    )
    if len(coordinate_offset_m) != 3:
        raise VompIntegrationError("coordinate_offset_m must have three values")
    offset_m = _require_numeric_array(
        coordinate_offset_m,
        label="coordinate_offset_m",
        shape=(3,),
    )

    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "voxel_data" in names:
                ambiguous = names & {
                    "voxel_coords_world",
                    "query_coords_world",
                    "density",
                    "youngs_modulus",
                    "poisson_ratio",
                    "segment_id",
                }
                if ambiguous:
                    raise VompIntegrationError(
                        "NPZ mixes structured voxel_data with direct material arrays: "
                        f"{sorted(ambiguous)!r}"
                    )
                raw = np.asarray(archive["voxel_data"])
                arrays = _structured_voxel_arrays(raw)
            else:
                arrays = _direct_voxel_arrays(archive, names)
            raw_coordinates, raw_density, raw_youngs, raw_poisson, schema = arrays
            coordinates = _require_numeric_array(
                raw_coordinates,
                label="voxel_coords_world",
            )
            if coordinates.ndim != 2 or coordinates.shape[1] != 3:
                raise VompIntegrationError(
                    "voxel_coords_world must have shape (N, 3); "
                    f"got {coordinates.shape}"
                )
            sample_count = int(coordinates.shape[0])
            if sample_count == 0 or sample_count > _MAX_VOXEL_SAMPLES:
                raise VompIntegrationError(
                    "VoMP NPZ must contain between 1 and "
                    f"{_MAX_VOXEL_SAMPLES} voxel samples; got {sample_count}"
                )
            expected_shape = (sample_count,)
            density = _require_numeric_array(
                raw_density,
                label="density",
                shape=expected_shape,
            )
            youngs = _require_numeric_array(
                raw_youngs,
                label="youngs_modulus",
                shape=expected_shape,
            )
            poisson = _require_numeric_array(
                raw_poisson,
                label="poisson_ratio",
                shape=expected_shape,
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, VompIntegrationError):
            raise
        raise VompIntegrationError(f"Unable to read VoMP NPZ: {exc}") from exc

    if np.any(density <= 0.0):
        raise VompIntegrationError("density values must all be positive kg/m^3")
    if np.any(youngs <= 0.0):
        raise VompIntegrationError("youngs_modulus values must all be positive Pa")
    if np.any((poisson <= -1.0) | (poisson >= 0.5)):
        raise VompIntegrationError("poisson_ratio values must satisfy -1 < nu < 0.5")

    coordinates_world_m = coordinates * unit_m + offset_m
    if not np.all(np.isfinite(coordinates_world_m)):
        raise VompIntegrationError(
            "coordinate unit/offset conversion produced non-finite world coordinates"
        )
    coordinates_world_m.setflags(write=False)
    _validate_voxel_lattice(coordinates_world_m, size_m)

    voxel_volume = size_m * size_m * size_m
    total_volume = sample_count * voxel_volume
    if not math.isfinite(total_volume) or total_volume <= 0.0:
        raise VompIntegrationError(
            "declared voxel size and sample count produce invalid total volume"
        )
    if _sha256(path) != source_sha256:
        raise VompIntegrationError("VoMP NPZ changed while it was being loaded")
    return _VompVoxelField(
        coordinates_world_m=coordinates_world_m,
        density_kg_m3=density,
        youngs_modulus_pa=youngs,
        poisson_ratio=poisson,
        voxel_size_m=size_m,
        source_schema=schema,
        source_sha256=source_sha256,
    )


def load_vomp_voxel_field(
    npz_path: str | Path,
    *,
    voxel_size_m: float,
    coordinate_unit_meters: float,
    coordinate_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> VompVoxelField:
    """Return the complete validated VoMP material field in world-space SI."""

    return _load_vomp_voxel_field(
        npz_path,
        voxel_size_m=voxel_size_m,
        coordinate_unit_meters=coordinate_unit_meters,
        coordinate_offset_m=coordinate_offset_m,
    )


def _derive_voxel_mass_properties(field: _VompVoxelField) -> _VoxelMassProperties:
    """Integrate mass, center of mass, and inertia over cubic VoMP voxels."""

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        voxel_masses = field.density_kg_m3 * field.voxel_volume_m3
    if np.any(~np.isfinite(voxel_masses)) or np.any(voxel_masses <= 0.0):
        raise VompIntegrationError(
            "density and voxel volume produce invalid per-voxel masses"
        )
    total_mass = float(np.sum(voxel_masses, dtype=np.float64))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise VompIntegrationError("density integration produced invalid total mass")
    center = np.asarray(
        np.sum(
            field.coordinates_world_m * voxel_masses[:, np.newaxis],
            axis=0,
            dtype=np.float64,
        )
        / total_mass,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(center)):
        raise VompIntegrationError("density integration produced a non-finite center")
    offsets = field.coordinates_world_m - center
    squared_radius = np.einsum("ij,ij->i", offsets, offsets)
    inertia = np.eye(3, dtype=np.float64) * float(
        np.sum(voxel_masses * squared_radius, dtype=np.float64)
    )
    inertia -= np.einsum(
        "i,ij,ik->jk",
        voxel_masses,
        offsets,
        offsets,
        dtype=np.float64,
    )
    # Each density sample represents a finite cube, not a point mass.
    cube_center_inertia = float(
        np.sum(voxel_masses, dtype=np.float64)
        * field.voxel_size_m
        * field.voxel_size_m
        / 6.0
    )
    inertia += np.eye(3, dtype=np.float64) * cube_center_inertia
    inertia = 0.5 * (inertia + inertia.T)
    if not np.all(np.isfinite(inertia)):
        raise VompIntegrationError("density integration produced non-finite inertia")
    try:
        eigenvalues = np.linalg.eigvalsh(inertia)
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise VompIntegrationError(
            "Unable to diagonalize the integrated inertia tensor"
        ) from exc
    if np.any(eigenvalues <= 0.0):
        raise VompIntegrationError(
            "density integration produced a non-positive inertia tensor"
        )
    return _VoxelMassProperties(
        mass_kg=total_mass,
        center_of_mass_world_m=cast(
            _Vector3,
            tuple(float(value) for value in center),
        ),
        inertia_tensor_world_kg_m2=cast(
            _Matrix3,
            tuple(tuple(float(value) for value in row) for row in inertia),
        ),
        integrated_volume_m3=field.sample_count * field.voxel_volume_m3,
    )


def _canonical_quaternion(
    quaternion: _Quaternion,
) -> _Quaternion:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 1e-15:
        raise VompIntegrationError("principal-axis rotation produced no quaternion")
    values /= norm
    for component in values:
        if abs(float(component)) <= 1e-15:
            continue
        if component < 0.0:
            values *= -1.0
        break
    values[np.abs(values) <= 1e-15] = 0.0
    return cast(_Quaternion, tuple(float(value) for value in values))


def _rotation_matrix_to_quaternion(
    matrix: np.ndarray,
) -> _Quaternion:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    return _canonical_quaternion(quaternion)


def _principal_mass_frame(
    inertia_local: np.ndarray,
) -> tuple[_Vector3, _Quaternion]:
    scale = max(float(np.max(np.abs(inertia_local))), float(np.finfo(np.float64).tiny))
    off_diagonal = inertia_local - np.diag(np.diag(inertia_local))
    if float(np.max(np.abs(off_diagonal))) <= scale * 1e-12:
        diagonal = cast(
            _Vector3,
            tuple(float(value) for value in np.diag(inertia_local)),
        )
        return diagonal, (1.0, 0.0, 0.0, 0.0)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_local)
    except (np.linalg.LinAlgError, RuntimeError) as exc:
        raise VompIntegrationError(
            "Unable to compute the principal mass frame"
        ) from exc
    for column in range(3):
        vector = eigenvectors[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0.0:
            eigenvectors[:, column] *= -1.0
    if np.linalg.det(eigenvectors) < 0.0:
        eigenvectors[:, 2] *= -1.0
    quaternion = _rotation_matrix_to_quaternion(eigenvectors)
    diagonal = cast(_Vector3, tuple(float(value) for value in eigenvalues))
    return diagonal, quaternion


def _enabled_rigid_body(prim: Any, UsdPhysics: Any) -> bool:
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    enabled_attr = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr()
    if enabled_attr.GetNumTimeSamples() > 0:
        raise VompIntegrationError(
            f"rigidBodyEnabled is time-sampled at {prim.GetPath()}"
        )
    enabled = enabled_attr.Get()
    return enabled is not False


def _has_deformable_semantics(prim: Any) -> bool:
    applied_schemas = {
        str(value) for value in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    }
    return (
        not applied_schemas.isdisjoint(_DEFORMABLE_API_SCHEMAS)
        or prim.GetCustomDataByKey("physicsAgentVompDeformable") is not None
    )


def _validate_rigid_body_hierarchy(prim: Any, UsdPhysics: Any) -> None:
    from pxr import Usd

    if _has_deformable_semantics(prim):
        raise VompIntegrationError(
            f"target prim {prim.GetPath()} already has deformable semantics; "
            "rigid VoMP mass authoring cannot replace them"
        )

    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if _has_deformable_semantics(parent):
            raise VompIntegrationError(
                f"target prim {prim.GetPath()} is below deformable prim "
                f"{parent.GetPath()}; rigid VoMP mass authoring cannot overlap it"
            )
        if _enabled_rigid_body(parent, UsdPhysics):
            raise VompIntegrationError(
                f"target prim {prim.GetPath()} is below enabled rigid body "
                f"{parent.GetPath()}; author mass on that body instead"
            )
        parent = parent.GetParent()
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for descendant in Usd.PrimRange(prim, predicate):
        if descendant == prim:
            continue
        if _has_deformable_semantics(descendant):
            raise VompIntegrationError(
                f"target prim {prim.GetPath()} contains deformable prim "
                f"{descendant.GetPath()}; rigid VoMP mass authoring cannot overlap it"
            )
        if _enabled_rigid_body(descendant, UsdPhysics):
            raise VompIntegrationError(
                f"target prim {prim.GetPath()} contains enabled rigid body "
                f"{descendant.GetPath()}; choose one unambiguous body"
            )
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        if not _enabled_rigid_body(prim, UsdPhysics):
            raise VompIntegrationError(
                f"target prim {prim.GetPath()} has explicitly disabled RigidBodyAPI"
            )


def _matrix4_array(matrix: Any) -> np.ndarray:
    return np.asarray(
        [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
        dtype=np.float64,
    )


def _transform_points(matrix: Any, points: np.ndarray) -> np.ndarray:
    """Apply a USD row-vector transform to an ``(N, 3)`` point array."""

    point_values = np.asarray(points, dtype=np.float64)
    if point_values.ndim != 2 or point_values.shape[1:] != (3,):
        raise VompIntegrationError("USD geometry points must have shape (N, 3)")
    homogeneous = np.ones((len(point_values), 4), dtype=np.float64)
    homogeneous[:, :3] = point_values
    transformed = homogeneous @ _matrix4_array(matrix)
    weights = transformed[:, 3]
    if not np.all(np.isfinite(transformed)) or np.any(
        np.abs(weights) <= np.finfo(np.float64).tiny
    ):
        raise VompIntegrationError("USD geometry has a non-finite world transform")
    return cast(np.ndarray, transformed[:, :3] / weights[:, np.newaxis])


def _validate_static_transform_chain(
    prim: Any,
    UsdGeom: Any,
    *,
    subject: str = "VoMP target geometry",
) -> None:
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        has_time_samples = xformable and any(
            operation.GetAttr().GetNumTimeSamples() > 0
            for operation in xformable.GetOrderedXformOps()
        )
        if xformable and (xformable.TransformMightBeTimeVarying() or has_time_samples):
            raise VompIntegrationError(
                f"{subject} transform is time-varying at {current.GetPath()}"
            )
        if xformable and xformable.GetResetXformStack():
            break
        current = current.GetParent()


def _validate_static_mesh_geometry(mesh: Any) -> None:
    geometry_attributes = (
        mesh.GetPointsAttr(),
        mesh.GetFaceVertexCountsAttr(),
        mesh.GetFaceVertexIndicesAttr(),
        mesh.GetHoleIndicesAttr(),
        mesh.GetSubdivisionSchemeAttr(),
        mesh.GetOrientationAttr(),
    )
    if any(attribute.GetNumTimeSamples() > 0 for attribute in geometry_attributes):
        raise VompIntegrationError(
            f"VoMP target mesh {mesh.GetPrim().GetPath()} has time-sampled geometry"
        )


def _target_world_frame(
    prim: Any,
    *,
    meters_per_unit: float,
    Usd: Any,
    UsdGeom: Any,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_static_transform_chain(prim, UsdGeom, subject="target world")

    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    origin = np.asarray(tuple(matrix.Transform((0.0, 0.0, 0.0))), dtype=np.float64)
    directions = [
        np.asarray(tuple(matrix.TransformDir(axis)), dtype=np.float64)
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ]
    rotation = np.column_stack(directions)
    gram = rotation.T @ rotation
    if not np.allclose(
        gram,
        np.eye(3),
        rtol=0.0,
        atol=_RIGID_TRANSFORM_TOLERANCE,
    ) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=_RIGID_TRANSFORM_TOLERANCE,
    ):
        raise VompIntegrationError(
            f"target prim {prim.GetPath()} has scale, shear, or reflection; "
            "VoMP mass authoring requires a rigid world transform"
        )
    return rotation, origin * meters_per_unit


def _preflight_vomp_mass_target(
    stage: Any,
    target: Any,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Validate all USD prerequisites needed by final mass authoring."""

    from pxr import Usd, UsdGeom, UsdPhysics

    if not UsdGeom.StageHasAuthoredMetersPerUnit(stage):
        raise VompIntegrationError(
            "USD stage must author metersPerUnit for coordinate alignment"
        )
    if not UsdPhysics.StageHasAuthoredKilogramsPerUnit(stage):
        raise VompIntegrationError(
            "USD stage must author kilogramsPerUnit for mass/inertia units"
        )
    meters_per_unit = _require_finite_positive(
        UsdGeom.GetStageMetersPerUnit(stage),
        label="USD metersPerUnit",
    )
    kilograms_per_unit = _require_finite_positive(
        UsdPhysics.GetStageKilogramsPerUnit(stage),
        label="USD kilogramsPerUnit",
    )
    if target.IsInstanceProxy() or target.IsInstanceable():
        raise VompIntegrationError(
            f"target prim {target.GetPath()} is instance-backed; deinstance it first"
        )
    if not target.IsA(UsdGeom.Xformable):
        raise VompIntegrationError(
            f"target prim {target.GetPath()} is not Xformable and cannot own a rigid body"
        )
    _validate_rigid_body_hierarchy(target, UsdPhysics)
    rotation_local_to_world, translation_world_m = _target_world_frame(
        target,
        meters_per_unit=meters_per_unit,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    return (
        meters_per_unit,
        kilograms_per_unit,
        rotation_local_to_world,
        translation_world_m,
    )


def _validate_geometry_association(
    prim: Any,
    field: _VompVoxelField,
    *,
    meters_per_unit: float,
    Usd: Any,
    UsdGeom: Any,
) -> tuple[_Vector3, _Vector3]:
    render_time = Usd.TimeCode.Default()
    purposes = {
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
    }
    xform_cache = UsdGeom.XformCache(render_time)
    point_chunks: list[np.ndarray] = []
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for geometry_prim in Usd.PrimRange(prim, predicate):
        if not geometry_prim.IsA(UsdGeom.Boundable):
            continue
        imageable = UsdGeom.Imageable(geometry_prim)
        if imageable.ComputeVisibility(render_time) == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() not in purposes:
            continue
        _validate_static_transform_chain(geometry_prim, UsdGeom)

        if geometry_prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(geometry_prim)
            _validate_static_mesh_geometry(mesh)
            points_attr = mesh.GetPointsAttr()
            local_points = np.asarray(points_attr.Get(render_time), dtype=np.float64)
            if local_points.ndim != 2 or local_points.shape[1:] != (3,):
                raise VompIntegrationError(
                    f"target geometry has invalid points at {geometry_prim.GetPath()}"
                )
        else:
            boundable = UsdGeom.Boundable(geometry_prim)
            extent = np.asarray(
                UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, render_time),
                dtype=np.float64,
            )
            if extent.shape != (2, 3):
                continue
            minimum_local, maximum_local = extent
            local_points = np.asarray(
                [
                    (x, y, z)
                    for x in (minimum_local[0], maximum_local[0])
                    for y in (minimum_local[1], maximum_local[1])
                    for z in (minimum_local[2], maximum_local[2])
                ],
                dtype=np.float64,
            )
        if not len(local_points) or not np.all(np.isfinite(local_points)):
            raise VompIntegrationError(
                f"target geometry has no finite points at {geometry_prim.GetPath()}"
            )
        world_points = _transform_points(
            xform_cache.GetLocalToWorldTransform(geometry_prim),
            local_points,
        )
        point_chunks.append(world_points * meters_per_unit)

    if not point_chunks:
        raise VompIntegrationError(
            f"target prim {prim.GetPath()} has no geometry bound for association"
        )
    world_points = np.concatenate(point_chunks, axis=0)
    minimum = np.min(world_points, axis=0)
    maximum = np.max(world_points, axis=0)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise VompIntegrationError(
            f"target prim {prim.GetPath()} has a non-finite geometry bound"
        )
    # VoMP snaps trimesh voxel centers to sparse-grid indices with floor-style
    # quantization. The reported center can therefore move by almost one pitch
    # from the surface-intersecting source cell. One full cell is the strict
    # association envelope for this pinned upstream representation.
    tolerance = field.voxel_size_m + max(1e-8, field.voxel_size_m * 1e-4)
    below = field.coordinates_world_m < minimum - tolerance
    above = field.coordinates_world_m > maximum + tolerance
    outside_rows = np.flatnonzero(np.any(below | above, axis=1))
    if outside_rows.size:
        first = int(outside_rows[0])
        coordinate = tuple(float(v) for v in field.coordinates_world_m[first])
        raise VompIntegrationError(
            f"VoMP voxel {first} at {coordinate} m lies outside the world bound "
            f"of target prim {prim.GetPath()}; verify coordinate scale, offset, "
            "voxel size, and geometry association"
        )
    bound_size = maximum - minimum
    association_volume = float(np.prod(bound_size + field.voxel_size_m))
    integrated_volume = field.sample_count * field.voxel_volume_m3
    if not math.isfinite(association_volume) or association_volume <= 0.0:
        raise VompIntegrationError(
            f"target prim {prim.GetPath()} has a degenerate geometry bound"
        )
    if integrated_volume > association_volume * (1.0 + 1e-6):
        raise VompIntegrationError(
            "declared non-overlapping voxel volume exceeds the target world-bound "
            "volume; verify voxel_size_m"
        )
    return (
        cast(_Vector3, tuple(float(value) for value in minimum)),
        cast(_Vector3, tuple(float(value) for value in maximum)),
    )


def _stats(values: np.ndarray) -> dict[str, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    scale = max(abs(minimum), abs(maximum))
    mean = 0.0 if scale == 0.0 else scale * float(np.mean(values / scale))
    if not all(math.isfinite(value) for value in (minimum, maximum, mean)):
        raise VompIntegrationError("VoMP material statistics are not finite")
    return {"min": minimum, "max": maximum, "mean": mean}


def _serialize_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}_",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _serialize_stage(stage: Any, output: Path) -> _SerializedStageArtifacts:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output.stem}_",
            dir=output.parent,
        )
    )
    temporary = staging_dir / output.name
    sidecar = staging_dir / portable_sidecar_name(temporary)
    try:
        dependency_roots = {
            Path(real_path).expanduser().resolve().parent
            for layer in stage.GetUsedLayers()
            if (real_path := str(getattr(layer, "realPath", "") or ""))
        }
        flattened = stage.Flatten()
        exported = export_stage_portably(
            stage,
            temporary,
            approved_dependency_roots=dependency_roots,
            export_layer=flattened,
        )
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise VompIntegrationError("Failed to export Physics Agent USD") from exc
    if not exported or not temporary.is_file():
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise VompIntegrationError("Failed to export Physics Agent USD")
    serialized_sidecar: Path | None = None
    if sidecar.exists():
        marker = sidecar / PORTABLE_SIDECAR_MARKER_NAME
        if (
            not sidecar.is_dir()
            or not marker.is_file()
            or marker.read_bytes() != PORTABLE_SIDECAR_MARKER_BYTES
        ):
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise VompIntegrationError("Failed to export Physics Agent USD")
        serialized_sidecar = sidecar
    return _SerializedStageArtifacts(
        root=temporary,
        sidecar=serialized_sidecar,
        staging_dir=staging_dir,
    )


def _remove_artifact(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _require_owned_portable_sidecar(sidecar: Path) -> None:
    if not sidecar.exists() and not sidecar.is_symlink():
        return
    marker = sidecar / PORTABLE_SIDECAR_MARKER_NAME
    if (
        sidecar.is_symlink()
        or not sidecar.is_dir()
        or marker.is_symlink()
        or not marker.is_file()
        or marker.read_bytes() != PORTABLE_SIDECAR_MARKER_BYTES
    ):
        raise VompIntegrationError(
            f"Refusing to replace unowned portable USD sidecar: {sidecar}"
        )


def _publish_artifact_pair(
    *,
    temporary_report: Path,
    report: Path,
    temporary_output: Path,
    output: Path,
    temporary_sidecar: Path | None = None,
    sidecar: Path | None = None,
) -> None:
    """Publish the provenance and portable USD bundle or restore prior artifacts."""

    if temporary_sidecar is not None and sidecar is None:
        raise ValueError("sidecar is required when temporary_sidecar is provided")
    destinations: list[tuple[Path | None, Path]] = [(temporary_report, report)]
    if sidecar is not None:
        _require_owned_portable_sidecar(sidecar)
        destinations.append((temporary_sidecar, sidecar))
    destinations.append((temporary_output, output))
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    retained_backups: set[Path] = set()
    try:
        for _, destination in destinations:
            if not destination.exists() and not destination.is_symlink():
                continue
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.stem}_backup_",
                suffix=destination.suffix,
                dir=destination.parent,
                delete=False,
            ) as stream:
                backup = Path(stream.name)
            backup.unlink(missing_ok=True)
            os.replace(destination, backup)
            backups[destination] = backup
        for temporary_artifact, destination in destinations:
            if temporary_artifact is None:
                continue
            os.replace(temporary_artifact, destination)
            published.append(destination)
    except Exception as publication_error:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            try:
                _remove_artifact(destination)
            except Exception as exc:
                rollback_errors.append(f"remove {destination}: {exc}")
        for destination, backup in reversed(tuple(backups.items())):
            if not backup.exists() and not backup.is_symlink():
                continue
            try:
                os.replace(backup, destination)
            except Exception as exc:
                retained_backups.add(backup)
                rollback_errors.append(f"restore {destination}: {exc}")
        if rollback_errors:
            retained = ", ".join(str(path) for path in sorted(retained_backups))
            message = "VoMP artifact publication failed and rollback was incomplete"
            if retained:
                message += f"; retained backups: {retained}"
            error = VompIntegrationError(message)
            for detail in rollback_errors:
                error.add_note(detail)
            raise error from publication_error
        raise
    finally:
        for temporary_artifact, _ in destinations:
            if temporary_artifact is not None:
                try:
                    _remove_artifact(temporary_artifact)
                except Exception:
                    pass
        for backup in backups.values():
            if backup in retained_backups:
                continue
            try:
                _remove_artifact(backup)
            except Exception:
                pass


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise VompIntegrationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _normalize_composition_hashes(
    expected_hashes: Mapping[str | Path, str],
    *,
    source: Path,
) -> dict[Path, str]:
    if not expected_hashes:
        raise VompIntegrationError("expected_composition_hashes must not be empty")
    normalized: dict[Path, str] = {}
    for index, (layer_path, digest) in enumerate(expected_hashes.items()):
        try:
            path = Path(layer_path).expanduser().resolve()
        except (OSError, TypeError, ValueError) as exc:
            raise VompIntegrationError(
                f"expected_composition_hashes entry {index} has an invalid path"
            ) from exc
        if path in normalized:
            raise VompIntegrationError(
                "expected_composition_hashes contains duplicate resolved paths"
            )
        normalized[path] = _require_sha256(
            digest,
            label=f"expected_composition_hashes entry {index}",
        )
    if source not in normalized:
        raise VompIntegrationError(
            "expected_composition_hashes must include the input USD root layer"
        )
    return normalized


def _verify_npz_digest(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise VompIntegrationError(
            "VoMP NPZ changed after worker verification and before USD publication"
        )


def _reload_and_hash_composition_layers(stage: Any) -> dict[Path, str]:
    """Reload clean local layers and attest their on-disk bytes."""

    local_layers: dict[Path, Any] = {}
    for layer in stage.GetUsedLayers():
        real_path = str(getattr(layer, "realPath", "") or "")
        if not real_path:
            continue
        path = Path(real_path).expanduser().resolve()
        if layer.dirty:
            raise VompIntegrationError(
                "input USD composition has unsaved in-memory layer edits"
            )
        local_layers[path] = layer
    for path, layer in local_layers.items():
        if not path.is_file():
            raise VompIntegrationError("input USD composition contains a missing layer")
        try:
            layer.Reload()
        except Exception as exc:
            raise VompIntegrationError(
                "Unable to reload an input USD composition layer"
            ) from exc
    return {path: _sha256(path) for path in local_layers}


def _verify_composition_hashes(expected_hashes: Mapping[Path, str]) -> None:
    for layer_path, expected_hash in expected_hashes.items():
        if not layer_path.is_file() or _sha256(layer_path) != expected_hash:
            raise VompIntegrationError(
                "input USD composition changed during VoMP mass-property authoring"
            )


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize caller provenance to a finite, detached JSON mapping."""
    try:
        encoded = json.dumps(value, allow_nan=False)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise VompIntegrationError(
            "evidence_provenance must contain finite JSON values"
        ) from exc
    if not isinstance(normalized, dict):
        raise VompIntegrationError("evidence_provenance must be a mapping")
    return cast(dict[str, Any], normalized)


def _json_custom_data(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize JSON provenance to values accepted by USD custom data."""

    normalized = _json_mapping(value)

    def remove_nulls(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): remove_nulls(child)
                for key, child in item.items()
                if child is not None
            }
        if isinstance(item, list):
            return [remove_nulls(child) for child in item if child is not None]
        return item

    return cast(dict[str, Any], remove_nulls(normalized))


def apply_vomp_mass_properties(
    usd_path: str | Path,
    vomp_npz_path: str | Path,
    output_usd_path: str | Path,
    *,
    target_prim_path: str,
    voxel_size_m: float,
    coordinate_unit_meters: float,
    complete_voxel_field: bool,
    coordinate_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    provenance_path: str | Path | None = None,
    evidence_provenance: Mapping[str, Any] | None = None,
    expected_npz_sha256: str | None = None,
    expected_composition_hashes: Mapping[str | Path, str] | None = None,
) -> VompApplyResult:
    """Derive and author strict rigid-body mass properties from VoMP density.

    ``complete_voxel_field`` must be explicitly true because upstream VoMP NPZ
    artifacts do not retain enough metadata to detect ``max_voxels`` capping.
    When supplied, ``expected_composition_hashes`` binds authoring to the exact
    local USD layer set used to render the evidence.
    No collision shape, physics material, friction, restitution, damping,
    joint, or trajectory property is created by this function.
    """

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    source = Path(usd_path).expanduser().resolve()
    npz_path = Path(vomp_npz_path).expanduser().resolve()
    output = Path(output_usd_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input USD not found: {source}")
    if source == output:
        raise VompIntegrationError("output_usd_path must differ from usd_path")
    _validate_vomp_input_layer(source)
    if output.suffix.lower() not in _USD_LAYER_EXTENSIONS:
        raise VompIntegrationError(
            "VoMP mass authoring output must use .usd, .usda, or .usdc"
        )
    report = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else output.with_name(f"{output.stem}.vomp_mass_properties.json")
    )
    if report in {source, npz_path, output}:
        raise VompIntegrationError(
            "provenance_path must differ from the USD and NPZ input/output paths"
        )
    if complete_voxel_field is not True:
        raise VompIntegrationError(
            "complete_voxel_field=True is required: upstream VoMP NPZ output "
            "does not record whether max_voxels truncated or subsampled the field"
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

    field = _load_vomp_voxel_field(
        npz_path,
        voxel_size_m=voxel_size_m,
        coordinate_unit_meters=coordinate_unit_meters,
        coordinate_offset_m=coordinate_offset_m,
    )
    if expected_npz_sha256 is not None and field.source_sha256 != expected_npz_sha256:
        raise VompIntegrationError(
            "VoMP NPZ does not match the worker-reported SHA-256 digest"
        )
    world_properties = _derive_voxel_mass_properties(field)
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
    session_layer = Sdf.Layer.CreateAnonymous("physics_agent_vomp_session.usda")
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
        kilograms_per_unit,
        rotation_local_to_world,
        translation_world_m,
    ) = _preflight_vomp_mass_target(
        stage,
        target,
    )
    world_bound_m = _validate_geometry_association(
        target,
        field,
        meters_per_unit=meters_per_unit,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )

    center_world_m = np.asarray(
        world_properties.center_of_mass_world_m,
        dtype=np.float64,
    )
    center_local_m = rotation_local_to_world.T @ (center_world_m - translation_world_m)
    inertia_world = np.asarray(
        world_properties.inertia_tensor_world_kg_m2,
        dtype=np.float64,
    )
    inertia_local = rotation_local_to_world.T @ inertia_world @ rotation_local_to_world
    inertia_local = 0.5 * (inertia_local + inertia_local.T)
    diagonal_inertia, principal_axes = _principal_mass_frame(inertia_local)
    authored = AuthoredMassProperties(
        mass_kg=world_properties.mass_kg,
        center_of_mass_local_m=cast(
            _Vector3,
            tuple(float(value) for value in center_local_m),
        ),
        diagonal_inertia_kg_m2=diagonal_inertia,
        principal_axes_wxyz=principal_axes,
    )

    mass_stage = authored.mass_kg / kilograms_per_unit
    center_stage = tuple(
        value / meters_per_unit for value in authored.center_of_mass_local_m
    )
    inertia_divisor = kilograms_per_unit * meters_per_unit * meters_per_unit
    if not math.isfinite(inertia_divisor) or inertia_divisor <= 0.0:
        raise VompIntegrationError(
            "USD stage units produce an invalid inertia-unit conversion"
        )
    inertia_stage = tuple(
        value / inertia_divisor for value in authored.diagonal_inertia_kg_m2
    )
    representable = (mass_stage, *center_stage, *inertia_stage, *principal_axes)
    if any(
        not math.isfinite(value) or abs(value) > _FLOAT32_MAX for value in representable
    ):
        raise VompIntegrationError(
            "derived mass properties cannot be represented by USD MassAPI float fields"
        )
    if any(
        value <= 0.0 or float(np.float32(value)) <= 0.0
        for value in (mass_stage, *inertia_stage)
    ):
        raise VompIntegrationError(
            "derived mass or inertia underflows USD MassAPI float fields"
        )

    provenance: dict[str, Any] = {
        "schemaVersion": 1,
        "adapter": "physics_agent.vomp_precomputed_npz",
        "source": {
            "npzSha256": field.source_sha256,
            "npzSchema": field.source_schema,
            "usdSha256": source_usd_sha256,
        },
        "association": {
            "coordinateSpace": "world",
            "coordinateUnitMeters": float(coordinate_unit_meters),
            "coordinateOffsetM": [float(value) for value in coordinate_offset_m],
            "targetPrimPath": str(target.GetPath()),
            "targetWorldBoundMinM": list(world_bound_m[0]),
            "targetWorldBoundMaxM": list(world_bound_m[1]),
            "voxelCenterBoundToleranceM": field.voxel_size_m
            + max(1e-8, field.voxel_size_m * 1e-4),
        },
        "voxelField": {
            "declaredComplete": True,
            "sampleCount": field.sample_count,
            "voxelSizeM": field.voxel_size_m,
            "voxelVolumeM3": field.voxel_volume_m3,
            "integratedVolumeM3": world_properties.integrated_volume_m3,
            "densityKgM3": _stats(field.density_kg_m3),
            "youngsModulusPa": _stats(field.youngs_modulus_pa),
            "poissonRatio": _stats(field.poisson_ratio),
        },
        "rigidBodyMassProperties": {
            "massKg": authored.mass_kg,
            "centerOfMassLocalM": list(authored.center_of_mass_local_m),
            "diagonalInertiaKgM2": list(authored.diagonal_inertia_kg_m2),
            "principalAxesWxyz": list(authored.principal_axes_wxyz),
        },
        "fieldUse": {
            "density": "mass_center_of_mass_inertia",
            "youngsModulus": "evidence_only_not_mapped_to_rigid_body",
            "poissonRatio": "evidence_only_not_mapped_to_rigid_body",
        },
        "notDerived": [
            "collision_geometry",
            "damping",
            "friction",
            "joints",
            "restitution",
            "trajectories",
        ],
    }
    if evidence_provenance is not None:
        provenance["evidence"] = _json_mapping(evidence_provenance)

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(target)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(target)
    mass_api.CreateMassAttr(float(mass_stage))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_stage))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia_stage))
    mass_api.CreatePrincipalAxesAttr(
        Gf.Quatf(principal_axes[0], Gf.Vec3f(*principal_axes[1:]))
    )
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
    target.SetCustomDataByKey("physicsAgentVomp", usd_provenance)

    _verify_composition_hashes(composition_hashes)
    serialized_stage: _SerializedStageArtifacts | None = None
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
            "Unable to publish VoMP USD and provenance artifacts"
        ) from exc
    finally:
        if serialized_stage is not None:
            shutil.rmtree(serialized_stage.staging_dir, ignore_errors=True)
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)

    return VompApplyResult(
        output_usd_path=output,
        provenance_path=report,
        mass_properties=authored,
        sample_count=field.sample_count,
    )


__all__ = [
    "AuthoredMassProperties",
    "VompApplyResult",
    "VompIntegrationError",
    "VompVoxelField",
    "apply_vomp_mass_properties",
    "load_vomp_voxel_field",
]
