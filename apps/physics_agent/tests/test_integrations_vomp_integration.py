# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic and contract tests for the precomputed VoMP NPZ adapter."""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import cast

import click
import numpy as np
import pytest
from typer.main import get_command
from typer.testing import CliRunner

pxr = pytest.importorskip("pxr", reason="USD (pxr) not available in this env")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils  # noqa: E402
from world_understanding.functions.graphics.so_export import (  # noqa: E402
    PORTABLE_SIDECAR_MARKER_BYTES,
    PORTABLE_SIDECAR_MARKER_NAME,
    portable_sidecar_name,
)

from physics_agent.cli import app  # noqa: E402
from physics_agent.integrations import vomp as vomp_module  # noqa: E402
from physics_agent.integrations.vomp import (  # noqa: E402
    VompIntegrationError,
    _canonical_quaternion,
    _derive_voxel_mass_properties,
    _enabled_rigid_body,
    _inspect_npz_archive,
    _load_vomp_voxel_field,
    _normalize_composition_hashes,
    _principal_mass_frame,
    _publish_artifact_pair,
    _rotation_matrix_to_quaternion,
    _serialize_json,
    _serialize_stage,
    _target_world_frame,
    _validate_rigid_body_hierarchy,
    _validate_voxel_lattice,
    _VompVoxelField,
    apply_vomp_mass_properties,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_cli_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "physics_agent.cli.setup_logging",
        lambda **_kwargs: logging.getLogger("physics_agent.tests.vomp"),
    )


def _field(
    coordinates: list[tuple[float, float, float]],
    density: list[float],
    *,
    voxel_size_m: float,
) -> _VompVoxelField:
    return _VompVoxelField(
        coordinates_world_m=np.asarray(coordinates, dtype=np.float64),
        density_kg_m3=np.asarray(density, dtype=np.float64),
        youngs_modulus_pa=np.full(len(density), 1.0e6, dtype=np.float64),
        poisson_ratio=np.full(len(density), 0.3, dtype=np.float64),
        voxel_size_m=voxel_size_m,
        source_schema="test",
        source_sha256="0" * 64,
    )


def _write_structured_npz(
    path: Path,
    coordinates: np.ndarray,
    density: np.ndarray,
    *,
    segment_id: str = "voxel_material",
) -> None:
    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("youngs_modulus", "<f4"),
        ("poissons_ratio", "<f4"),
        ("density", "<f4"),
        ("segment_id", "<U32"),
    ]
    voxel_data = np.zeros(len(coordinates), dtype=dtype)
    voxel_data["x"] = coordinates[:, 0]
    voxel_data["y"] = coordinates[:, 1]
    voxel_data["z"] = coordinates[:, 2]
    voxel_data["youngs_modulus"] = np.linspace(
        1.0e6,
        2.0e6,
        len(coordinates),
    )
    voxel_data["poissons_ratio"] = 0.3
    voxel_data["density"] = density
    voxel_data["segment_id"] = segment_id
    np.savez_compressed(path, voxel_data=voxel_data)


def _write_cube_stage(
    path: Path,
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_degrees: float = 0.0,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    author_units: bool = True,
    meters_per_unit: float = 1.0,
    kilograms_per_unit: float = 1.0,
) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    cube = UsdGeom.Cube.Define(stage, "/Cube")
    cube.CreateSizeAttr(2.0)
    transform = UsdGeom.Xformable(cube)
    if rotation_degrees:
        transform.AddRotateZOp().Set(rotation_degrees)
    if scale != (1.0, 1.0, 1.0):
        transform.AddScaleOp().Set(Gf.Vec3d(*scale))
    if translation != (0.0, 0.0, 0.0):
        transform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    stage.SetDefaultPrim(cube.GetPrim())
    if author_units:
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)
    stage.GetRootLayer().Save()


def test_uniform_density_matches_solid_cube_analytics() -> None:
    coordinates = [
        (x, y, z) for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)
    ]
    result = _derive_voxel_mass_properties(
        _field(coordinates, [1000.0] * 8, voxel_size_m=0.5)
    )

    assert result.integrated_volume_m3 == pytest.approx(1.0)
    assert result.mass_kg == pytest.approx(1000.0)
    assert result.center_of_mass_world_m == pytest.approx((0.0, 0.0, 0.0))
    assert np.asarray(result.inertia_tensor_world_kg_m2) == pytest.approx(
        np.eye(3) * (1000.0 / 6.0)
    )


def test_spatially_varying_density_shifts_center_and_inertia() -> None:
    result = _derive_voxel_mass_properties(
        _field(
            [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
            [1.0, 3.0],
            voxel_size_m=1.0,
        )
    )

    assert result.mass_kg == pytest.approx(4.0)
    assert result.center_of_mass_world_m == pytest.approx((0.25, 0.0, 0.0))
    assert np.asarray(result.inertia_tensor_world_kg_m2) == pytest.approx(
        np.diag([2.0 / 3.0, 17.0 / 12.0, 17.0 / 12.0])
    )


def test_loader_accepts_upstream_structured_voxel_archive(tmp_path: Path) -> None:
    npz = tmp_path / "materials.npz"
    coordinates = np.asarray([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]])
    _write_structured_npz(npz, coordinates, np.asarray([500.0, 750.0]))

    field = _load_vomp_voxel_field(
        npz,
        voxel_size_m=0.5,
        coordinate_unit_meters=2.0,
        coordinate_offset_m=(1.0, 0.0, 0.0),
    )

    assert field.source_schema == "vomp.voxel_data.v1"
    assert field.sample_count == 2
    assert field.voxel_volume_m3 == pytest.approx(0.125)
    assert field.coordinates_world_m == pytest.approx(
        np.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    )
    assert len(field.source_sha256) == 64


def test_loader_accepts_upstream_voxel_center_segment_name(tmp_path: Path) -> None:
    npz = tmp_path / "materials.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([500.0]),
        segment_id="voxel_center",
    )

    field = _load_vomp_voxel_field(
        npz,
        voxel_size_m=0.5,
        coordinate_unit_meters=1.0,
    )

    assert field.sample_count == 1


def test_loader_rejects_query_point_archive(tmp_path: Path) -> None:
    npz = tmp_path / "query_materials.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
        segment_id="query_point_material",
    )

    with pytest.raises(VompIntegrationError, match="voxel-center output"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.1,
            coordinate_unit_meters=1.0,
        )


def test_loader_accepts_direct_vomp_arrays(tmp_path: Path) -> None:
    npz = tmp_path / "direct_materials.npz"
    np.savez_compressed(
        npz,
        voxel_coords_world=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        density=np.asarray([900.0, 1100.0]),
        youngs_modulus=np.asarray([1.0e6, 2.0e6]),
        poisson_ratio=np.asarray([0.2, 0.3]),
    )

    field = _load_vomp_voxel_field(
        npz,
        voxel_size_m=0.5,
        coordinate_unit_meters=1.0,
    )

    assert field.source_schema == "vomp.direct_arrays.v1"
    assert field.sample_count == 2


def test_loader_accepts_direct_bytes_voxel_segment_id(tmp_path: Path) -> None:
    npz = tmp_path / "direct_materials.npz"
    np.savez_compressed(
        npz,
        voxel_coords_world=np.asarray([[0.0, 0.0, 0.0]]),
        density=np.asarray([900.0]),
        youngs_modulus=np.asarray([1.0e6]),
        poisson_ratio=np.asarray([0.3]),
        segment_id=np.asarray([b"voxel_material"]),
    )

    field = _load_vomp_voxel_field(
        npz,
        voxel_size_m=0.5,
        coordinate_unit_meters=1.0,
    )

    assert field.sample_count == 1


@pytest.mark.parametrize(
    ("segment_id", "message"),
    [
        (np.asarray([1]), "string array"),
        (np.asarray(["query_point_material"]), "voxel-center samples"),
    ],
)
def test_loader_rejects_invalid_direct_segment_id(
    tmp_path: Path,
    segment_id: np.ndarray,
    message: str,
) -> None:
    npz = tmp_path / "direct_materials.npz"
    np.savez_compressed(
        npz,
        voxel_coords_world=np.asarray([[0.0, 0.0, 0.0]]),
        density=np.asarray([900.0]),
        youngs_modulus=np.asarray([1.0e6]),
        poisson_ratio=np.asarray([0.3]),
        segment_id=segment_id,
    )

    with pytest.raises(VompIntegrationError, match=message):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
        )


def test_loader_rejects_mixed_structured_and_direct_schema(tmp_path: Path) -> None:
    npz = tmp_path / "ambiguous_materials.npz"
    structured = tmp_path / "structured_materials.npz"
    _write_structured_npz(
        structured,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([900.0]),
    )
    with np.load(structured, allow_pickle=False) as archive:
        np.savez_compressed(
            npz,
            voxel_data=archive["voxel_data"],
            youngs_modulus=np.asarray([2.0e6]),
        )

    with pytest.raises(VompIntegrationError, match="mixes structured"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
        )


@pytest.mark.parametrize("voxel_size_m", [1.0e308, 1.0e-300])
def test_loader_rejects_voxel_volume_overflow_or_underflow(
    tmp_path: Path,
    voxel_size_m: float,
) -> None:
    npz = tmp_path / "materials.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([900.0]),
    )

    with pytest.raises(VompIntegrationError, match="invalid total volume"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=voxel_size_m,
            coordinate_unit_meters=1.0,
        )


def test_loader_rejects_npz_changed_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npz = tmp_path / "materials.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([900.0]),
    )
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(vomp_module, "_sha256", lambda _path: next(digests))

    with pytest.raises(VompIntegrationError, match="changed while it was being loaded"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
        )


def test_npz_archive_limits_reject_excess_entries_and_uncompressed_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_many = tmp_path / "too_many.npz"
    with zipfile.ZipFile(too_many, "w") as archive:
        for index in range(65):
            archive.writestr(f"entry_{index}.npy", b"value")
    with pytest.raises(VompIntegrationError, match="between 1 and 64 entries"):
        _inspect_npz_archive(too_many)

    too_large = tmp_path / "too_large.npz"
    with zipfile.ZipFile(too_large, "w") as archive:
        archive.writestr("entry.npy", b"value")
    monkeypatch.setattr(vomp_module, "_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(VompIntegrationError, match="1 GiB safety limit"):
        _inspect_npz_archive(too_large)


def test_loader_rejects_direct_query_point_results(tmp_path: Path) -> None:
    npz = tmp_path / "query_materials.npz"
    np.savez_compressed(
        npz,
        voxel_coords_world=np.asarray([[0.0, 0.0, 0.0]]),
        query_coords_world=np.asarray([[0.0, 0.0, 0.0]]),
        density=np.asarray([900.0]),
        youngs_modulus=np.asarray([1.0e6]),
        poisson_ratio=np.asarray([0.3]),
    )

    with pytest.raises(VompIntegrationError, match="query_coords_world"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
        )


@pytest.mark.parametrize("density", [0.0, -1.0, np.nan, np.inf])
def test_loader_rejects_non_positive_or_non_finite_density(
    tmp_path: Path,
    density: float,
) -> None:
    npz = tmp_path / "bad_density.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([density]),
    )

    with pytest.raises(VompIntegrationError, match="density"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.1,
            coordinate_unit_meters=1.0,
        )


def test_loader_rejects_declared_lattice_mismatch(tmp_path: Path) -> None:
    npz = tmp_path / "off_lattice.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        np.asarray([1000.0, 1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="not aligned"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
        )


def test_loader_uses_voxel_relative_lattice_tolerance(tmp_path: Path) -> None:
    npz = tmp_path / "tiny_off_lattice.npz"
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        np.asarray([1000.0, 1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="not aligned"):
        _load_vomp_voxel_field(
            npz,
            voxel_size_m=1.0e-12,
            coordinate_unit_meters=1.0e-12,
        )


def test_loader_accounts_for_float32_roundoff_at_large_coordinates(
    tmp_path: Path,
) -> None:
    npz = tmp_path / "large_coordinates.npz"
    _write_structured_npz(
        npz,
        np.asarray([[1000.0, 0.0, 0.0], [1000.1, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1000.0, 1000.0]),
    )

    field = _load_vomp_voxel_field(
        npz,
        voxel_size_m=0.1,
        coordinate_unit_meters=1.0,
    )

    assert field.sample_count == 2


def test_voxel_lattice_duplicate_check_handles_large_index_space() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [3_000_000.0, 0.0, 0.0],
            [0.0, 3_000_000.0, 0.0],
            [0.0, 0.0, 3_000_000.0],
        ],
        dtype=np.float64,
    )

    _validate_voxel_lattice(coordinates, 1.0)

    with pytest.raises(VompIntegrationError, match="same declared voxel cell"):
        _validate_voxel_lattice(np.vstack((coordinates, coordinates[0])), 1.0)


def test_canonical_quaternion_uses_first_nonzero_component_sign() -> None:
    assert _canonical_quaternion((0.0, -2.0, 0.0, 0.0)) == (
        0.0,
        1.0,
        0.0,
        0.0,
    )


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (np.diag([1.0, -1.0, -1.0]), (0.0, 1.0, 0.0, 0.0)),
        (np.diag([-1.0, 1.0, -1.0]), (0.0, 0.0, 1.0, 0.0)),
        (np.diag([-1.0, -1.0, 1.0]), (0.0, 0.0, 0.0, 1.0)),
    ],
)
def test_rotation_matrix_to_quaternion_handles_half_turns(
    matrix: np.ndarray,
    expected: tuple[float, float, float, float],
) -> None:
    assert _rotation_matrix_to_quaternion(matrix) == pytest.approx(expected)


def test_principal_mass_frame_repairs_left_handed_eigenbasis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_handed = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    monkeypatch.setattr(
        vomp_module.np.linalg,
        "eigh",
        lambda _matrix: (np.asarray([1.0, 2.0, 3.0]), left_handed.copy()),
    )

    diagonal, quaternion = _principal_mass_frame(
        np.asarray(
            [
                [2.0, 0.25, 0.0],
                [0.25, 3.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
    )

    assert diagonal == pytest.approx((1.0, 2.0, 3.0))
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    repaired_axes = left_handed.copy()
    repaired_axes[:, 2] *= -1.0
    authored_quaternion = Gf.Quatd(
        quaternion[0],
        Gf.Vec3d(*quaternion[1:]),
    )
    rotation = Gf.Matrix3d(Gf.Rotation(authored_quaternion))
    rotation_array = np.asarray(
        [[float(rotation[row][column]) for column in range(3)] for row in range(3)]
    )
    reconstructed = rotation_array.T @ np.diag(diagonal) @ rotation_array
    expected = repaired_axes @ np.diag(diagonal) @ repaired_axes.T
    assert reconstructed == pytest.approx(expected)


def test_rigid_body_hierarchy_rejects_parent_descendant_and_disabled_body() -> None:
    stage = Usd.Stage.CreateInMemory()
    plain = UsdGeom.Xform.Define(stage, "/Plain").GetPrim()
    assert not _enabled_rigid_body(plain, UsdPhysics)

    parent = UsdGeom.Xform.Define(stage, "/Parent").GetPrim()
    parent_api = UsdPhysics.RigidBodyAPI.Apply(parent)
    parent_api.CreateRigidBodyEnabledAttr(True)
    UsdGeom.Xform.Define(stage, "/Parent/Group")
    child = UsdGeom.Cube.Define(stage, "/Parent/Group/Child").GetPrim()
    assert _enabled_rigid_body(parent, UsdPhysics)
    with pytest.raises(VompIntegrationError, match="below enabled rigid body"):
        _validate_rigid_body_hierarchy(child, UsdPhysics)

    container = UsdGeom.Xform.Define(stage, "/Container").GetPrim()
    nested = UsdGeom.Cube.Define(stage, "/Container/Nested").GetPrim()
    nested_api = UsdPhysics.RigidBodyAPI.Apply(nested)
    nested_api.CreateRigidBodyEnabledAttr(True)
    with pytest.raises(VompIntegrationError, match="contains enabled rigid body"):
        _validate_rigid_body_hierarchy(container, UsdPhysics)

    filtered_container = UsdGeom.Xform.Define(stage, "/FilteredContainer").GetPrim()
    inactive = UsdGeom.Cube.Define(stage, "/FilteredContainer/Inactive").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(inactive).CreateRigidBodyEnabledAttr(True)
    inactive.SetActive(False)
    abstract = stage.CreateClassPrim("/FilteredContainer/Abstract")
    abstract.SetTypeName("Cube")
    UsdPhysics.RigidBodyAPI.Apply(abstract).CreateRigidBodyEnabledAttr(True)
    _validate_rigid_body_hierarchy(filtered_container, UsdPhysics)

    disabled = UsdGeom.Cube.Define(stage, "/Disabled").GetPrim()
    disabled_api = UsdPhysics.RigidBodyAPI.Apply(disabled)
    disabled_api.CreateRigidBodyEnabledAttr(False)
    assert not _enabled_rigid_body(disabled, UsdPhysics)
    with pytest.raises(VompIntegrationError, match="explicitly disabled"):
        _validate_rigid_body_hierarchy(disabled, UsdPhysics)


@pytest.mark.parametrize(
    ("location", "relation"),
    [("ancestor", "below"), ("descendant", "contains")],
)
@pytest.mark.parametrize(
    "semantics",
    [
        "schema",
        "owned-marker",
        "PhysicsCurvesDeformableSimAPI",
        "PhysicsSurfaceDeformableSimAPI",
        "PhysicsVolumeDeformableSimAPI",
    ],
)
def test_rigid_body_hierarchy_rejects_deformable_semantics_across_hierarchy(
    location: str,
    relation: str,
    semantics: str,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    ancestor = UsdGeom.Xform.Define(stage, "/Parent").GetPrim()
    target = UsdGeom.Xform.Define(stage, "/Parent/Target").GetPrim()
    descendant = UsdGeom.Cube.Define(stage, "/Parent/Target/Child").GetPrim()
    conflicting_prim = {
        "ancestor": ancestor,
        "descendant": descendant,
    }[location]
    if semantics == "schema":
        conflicting_prim.AddAppliedSchema("PhysicsDeformableBodyAPI")
    elif semantics == "owned-marker":
        conflicting_prim.SetCustomDataByKey(
            "physicsAgentVompDeformable",
            {"adapter": "physics_agent.vomp_volume_deformable"},
        )
    else:
        conflicting_prim.AddAppliedSchema(semantics)

    with pytest.raises(
        VompIntegrationError,
        match=rf"{relation}.*deformable",
    ):
        _validate_rigid_body_hierarchy(target, UsdPhysics)


@pytest.mark.parametrize("sampled_location", ["target", "ancestor", "descendant"])
def test_rigid_body_hierarchy_rejects_time_sampled_enabled_state(
    sampled_location: str,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Parent").GetPrim()
    target = UsdGeom.Xform.Define(stage, "/Parent/Target").GetPrim()
    descendant = UsdGeom.Cube.Define(stage, "/Parent/Target/Child").GetPrim()
    sampled_prim = {
        "target": target,
        "ancestor": parent,
        "descendant": descendant,
    }[sampled_location]
    enabled = UsdPhysics.RigidBodyAPI.Apply(sampled_prim).CreateRigidBodyEnabledAttr()
    enabled.Set(False, Usd.TimeCode(0.0))
    enabled.Set(True, Usd.TimeCode(1.0))

    with pytest.raises(VompIntegrationError, match="time-sampled"):
        _validate_rigid_body_hierarchy(target, UsdPhysics)


def test_target_world_frame_stops_at_reset_xform_stack() -> None:
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Parent")
    animated = parent.AddTranslateOp()
    animated.Set(Gf.Vec3d(0.0), Usd.TimeCode(0.0))
    animated.Set(Gf.Vec3d(1.0, 0.0, 0.0), Usd.TimeCode(1.0))
    child = UsdGeom.Xform.Define(stage, "/Parent/Child")
    child.SetResetXformStack(True)

    rotation, origin = _target_world_frame(
        child.GetPrim(),
        meters_per_unit=1.0,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )

    assert rotation == pytest.approx(np.eye(3))
    assert origin == pytest.approx(np.zeros(3))


def test_expected_composition_hash_manifest_validation(tmp_path: Path) -> None:
    source = (tmp_path / "source.usda").resolve()
    other = (tmp_path / "other.usda").resolve()
    digest = "0" * 64

    assert _normalize_composition_hashes({source: digest}, source=source) == {
        source: digest
    }
    with pytest.raises(VompIntegrationError, match="must not be empty"):
        _normalize_composition_hashes({}, source=source)
    with pytest.raises(VompIntegrationError, match="invalid path"):
        _normalize_composition_hashes(
            cast(dict[str | Path, str], {object(): digest}),
            source=source,
        )
    with pytest.raises(VompIntegrationError, match="duplicate resolved paths"):
        _normalize_composition_hashes(
            {source: digest, str(source): digest},
            source=source,
        )
    with pytest.raises(VompIntegrationError, match="lowercase SHA-256"):
        _normalize_composition_hashes(
            cast(dict[str | Path, str], {source: 1}),
            source=source,
        )
    with pytest.raises(VompIntegrationError, match="input USD root layer"):
        _normalize_composition_hashes({other: digest}, source=source)


def test_artifact_serialization_and_pair_publication_are_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        _serialize_json(tmp_path / "invalid.json", {"value": object()})
    assert not list(tmp_path.glob(".invalid_*"))

    class _FailedLayer:
        def Export(self, _path: str) -> bool:
            return False

    class _FailedStage:
        def Flatten(self) -> _FailedLayer:
            return _FailedLayer()

    with pytest.raises(VompIntegrationError, match="Failed to export"):
        _serialize_stage(_FailedStage(), tmp_path / "failed.usda")

    class _ExportableStage:
        def GetUsedLayers(self) -> list[object]:
            return []

        def Flatten(self) -> object:
            return object()

    monkeypatch.setattr(
        vomp_module, "export_stage_portably", lambda *_args, **_kwargs: False
    )
    with pytest.raises(VompIntegrationError, match="Failed to export"):
        _serialize_stage(_ExportableStage(), tmp_path / "missing.usda")

    def export_with_invalid_sidecar(
        _stage: object,
        path: Path,
        **_kwargs: object,
    ) -> bool:
        path.write_text("#usda 1.0\n", encoding="ascii")
        (path.parent / portable_sidecar_name(path)).mkdir()
        return True

    monkeypatch.setattr(
        vomp_module,
        "export_stage_portably",
        export_with_invalid_sidecar,
    )
    with pytest.raises(VompIntegrationError, match="Failed to export"):
        _serialize_stage(_ExportableStage(), tmp_path / "invalid-sidecar.usda")

    report = tmp_path / "report.json"
    output = tmp_path / "output.usda"
    report.write_text("old report", encoding="ascii")
    output.write_text("old output", encoding="ascii")
    temporary_report = tmp_path / "new-report.tmp"
    temporary_output = tmp_path / "new-output.tmp"
    temporary_report.write_text("new report", encoding="ascii")
    temporary_output.write_text("new output", encoding="ascii")
    _publish_artifact_pair(
        temporary_report=temporary_report,
        report=report,
        temporary_output=temporary_output,
        output=output,
    )
    assert report.read_text(encoding="ascii") == "new report"
    assert output.read_text(encoding="ascii") == "new output"

    report.write_text("rollback report", encoding="ascii")
    output.write_text("rollback output", encoding="ascii")
    temporary_report.write_text("bad report", encoding="ascii")
    temporary_output.write_text("bad output", encoding="ascii")
    sidecar = tmp_path / portable_sidecar_name(output)
    sidecar.mkdir()
    (sidecar / PORTABLE_SIDECAR_MARKER_NAME).write_bytes(PORTABLE_SIDECAR_MARKER_BYTES)
    (sidecar / "asset.txt").write_text("rollback asset", encoding="ascii")
    temporary_sidecar = tmp_path / "new-sidecar"
    temporary_sidecar.mkdir()
    (temporary_sidecar / PORTABLE_SIDECAR_MARKER_NAME).write_bytes(
        PORTABLE_SIDECAR_MARKER_BYTES
    )
    (temporary_sidecar / "asset.txt").write_text("bad asset", encoding="ascii")
    real_replace = vomp_module.os.replace

    def fail_output_publish(source: Path, destination: Path) -> None:
        if Path(source) == temporary_output and Path(destination) == output:
            raise OSError("synthetic publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(vomp_module.os, "replace", fail_output_publish)
    with pytest.raises(OSError, match="synthetic publish failure"):
        _publish_artifact_pair(
            temporary_report=temporary_report,
            report=report,
            temporary_output=temporary_output,
            output=output,
            temporary_sidecar=temporary_sidecar,
            sidecar=sidecar,
        )
    assert report.read_text(encoding="ascii") == "rollback report"
    assert output.read_text(encoding="ascii") == "rollback output"
    assert (sidecar / "asset.txt").read_text(encoding="ascii") == "rollback asset"

    with pytest.raises(ValueError, match="sidecar is required"):
        _publish_artifact_pair(
            temporary_report=temporary_report,
            report=report,
            temporary_output=temporary_output,
            output=output,
            temporary_sidecar=temporary_sidecar,
        )


def test_publication_rollback_continues_and_retains_unrestored_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "output.usda"
    temporary_report = tmp_path / "new-report.tmp"
    temporary_output = tmp_path / "new-output.tmp"
    report.write_text("old report", encoding="ascii")
    output.write_text("old output", encoding="ascii")
    temporary_report.write_text("new report", encoding="ascii")
    temporary_output.write_text("new output", encoding="ascii")
    real_replace = vomp_module.os.replace
    real_remove = vomp_module._remove_artifact

    def fail_publish_and_output_restore(source: Path, destination: Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == temporary_output and destination_path == output:
            raise OSError("synthetic publish failure")
        if "_backup_" in source_path.name and destination_path == output:
            raise OSError("synthetic restore failure")
        real_replace(source, destination)

    def fail_one_removal(path: Path) -> None:
        if Path(path) in {report, temporary_output}:
            raise OSError("synthetic removal failure")
        real_remove(path)

    monkeypatch.setattr(vomp_module.os, "replace", fail_publish_and_output_restore)
    monkeypatch.setattr(vomp_module, "_remove_artifact", fail_one_removal)

    with pytest.raises(VompIntegrationError, match="rollback was incomplete") as exc:
        _publish_artifact_pair(
            temporary_report=temporary_report,
            report=report,
            temporary_output=temporary_output,
            output=output,
        )

    assert report.read_text(encoding="ascii") == "old report"
    assert not output.exists()
    retained = list(tmp_path.glob(".output_backup_*.usda"))
    assert len(retained) == 1
    assert retained[0].read_text(encoding="ascii") == "old output"
    assert "retained backups" in str(exc.value)
    assert len(exc.value.__notes__) == 2
    assert not temporary_report.exists()
    assert temporary_output.read_text(encoding="ascii") == "new output"


def test_successful_publication_ignores_backup_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "output.usda"
    temporary_report = tmp_path / "new-report.tmp"
    temporary_output = tmp_path / "new-output.tmp"
    report.write_text("old report", encoding="ascii")
    output.write_text("old output", encoding="ascii")
    temporary_report.write_text("new report", encoding="ascii")
    temporary_output.write_text("new output", encoding="ascii")
    real_remove = vomp_module._remove_artifact

    def retain_backups(path: Path) -> None:
        if "_backup_" in Path(path).name:
            raise OSError("synthetic cleanup failure")
        real_remove(path)

    monkeypatch.setattr(vomp_module, "_remove_artifact", retain_backups)
    _publish_artifact_pair(
        temporary_report=temporary_report,
        report=report,
        temporary_output=temporary_output,
        output=output,
    )

    assert report.read_text(encoding="ascii") == "new report"
    assert output.read_text(encoding="ascii") == "new output"
    assert len(list(tmp_path.glob(".*_backup_*"))) == 2


def test_apply_authors_only_rigid_body_mass_properties_and_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source, translation=(10.0, 0.0, 0.0))
    _write_structured_npz(
        npz,
        np.asarray([[9.5, 0.0, 0.0], [10.5, 0.0, 0.0]]),
        np.asarray([1.0, 3.0]),
    )

    result = apply_vomp_mass_properties(
        source,
        npz,
        output,
        target_prim_path="/Cube",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    assert result.mass_properties.mass_kg == pytest.approx(4.0)
    assert result.mass_properties.center_of_mass_local_m == pytest.approx(
        (0.25, 0.0, 0.0)
    )
    assert result.provenance_path.is_file()
    report = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert report["voxelField"]["declaredComplete"] is True
    assert report["fieldUse"]["density"] == "mass_center_of_mass_inertia"
    assert "not_mapped_to_rigid_body" in report["fieldUse"]["youngsModulus"]
    assert "friction" in report["notDerived"]

    source_stage = Usd.Stage.Open(str(source))
    source_prim = source_stage.GetPrimAtPath("/Cube")
    assert not source_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not source_prim.HasAPI(UsdPhysics.MassAPI)
    assert source_prim.GetCustomDataByKey("physicsAgentVomp") is None

    stage = Usd.Stage.Open(str(output))
    assert stage
    prim = stage.GetPrimAtPath("/Cube")
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert prim.HasAPI(UsdPhysics.MassAPI)
    assert not prim.HasAPI(UsdPhysics.CollisionAPI)
    mass_api = UsdPhysics.MassAPI(prim)
    assert mass_api.GetMassAttr().Get() == pytest.approx(4.0)
    assert tuple(mass_api.GetCenterOfMassAttr().Get()) == pytest.approx(
        (0.25, 0.0, 0.0)
    )
    assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        (2.0 / 3.0, 17.0 / 12.0, 17.0 / 12.0)
    )
    assert mass_api.GetPrincipalAxesAttr().Get() == Gf.Quatf(1.0)
    assert not prim.GetAttribute("physics:staticFriction").IsValid()
    assert not prim.GetAttribute("physics:restitution").IsValid()
    assert (
        prim.GetCustomDataByKey("physicsAgentVomp")["source"]["npzSha256"]
        == report["source"]["npzSha256"]
    )


@pytest.mark.parametrize(
    ("deformable_schema", "add_owned_marker"),
    [
        ("PhysicsDeformableBodyAPI", False),
        (None, True),
        ("PhysicsDeformableBodyAPI", True),
        ("PhysicsCurvesDeformableSimAPI", False),
        ("PhysicsSurfaceDeformableSimAPI", False),
        ("PhysicsVolumeDeformableSimAPI", False),
    ],
    ids=(
        "body-schema",
        "owned-marker",
        "body-schema-and-owned-marker",
        "curves-sim-schema",
        "surface-sim-schema",
        "volume-sim-schema",
    ),
)
def test_apply_rejects_existing_deformable_semantics(
    tmp_path: Path,
    *,
    deformable_schema: str | None,
    add_owned_marker: bool,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    stage = Usd.Stage.Open(str(source))
    target = stage.GetPrimAtPath("/Cube")
    if deformable_schema is not None:
        target.AddAppliedSchema(deformable_schema)
    if add_owned_marker:
        target.SetCustomDataByKey(
            "physicsAgentVompDeformable",
            {"adapter": "physics_agent.vomp_volume_deformable"},
        )
    stage.GetRootLayer().Save()
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="already has deformable semantics"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()
    assert not output.with_name("physics.vomp_mass_properties.json").exists()


def test_apply_localizes_asset_dependencies_for_relocatable_output(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    texture = source_dir / "textures" / "albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"portable-texture")
    source = source_dir / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    stage = Usd.Stage.Open(str(source))
    shader = UsdShade.Shader.Define(stage, "/Cube/Texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.png")
    )
    stage.GetRootLayer().Save()
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    apply_vomp_mass_properties(
        source,
        npz,
        output,
        target_prim_path="/Cube",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    sidecar = output.parent / portable_sidecar_name(output)
    assert (sidecar / PORTABLE_SIDECAR_MARKER_NAME).read_bytes() == (
        PORTABLE_SIDECAR_MARKER_BYTES
    )
    output_stage = Usd.Stage.Open(str(output))
    authored_asset = (
        UsdShade.Shader(output_stage.GetPrimAtPath("/Cube/Texture"))
        .GetInput("file")
        .Get()
    )
    assert not Path(authored_asset.path).is_absolute()
    assert Path(authored_asset.resolvedPath).is_relative_to(sidecar)

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    moved_output = Path(shutil.move(output, delivery / output.name))
    moved_sidecar = Path(shutil.move(sidecar, delivery / sidecar.name))
    _layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(str(moved_output))
    )
    assert not unresolved
    assert any(
        Path(str(asset)).resolve().is_relative_to(moved_sidecar.resolve())
        for asset in assets
    )


def test_apply_rotates_world_inertia_into_body_local_frame(tmp_path: Path) -> None:
    source = tmp_path / "rotated.usda"
    output = tmp_path / "rotated_physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source, rotation_degrees=45.0)
    _write_structured_npz(
        npz,
        np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.asarray([1.0, 3.0]),
    )

    result = apply_vomp_mass_properties(
        source,
        npz,
        output,
        target_prim_path="/Cube",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    assert result.mass_properties.center_of_mass_local_m == pytest.approx(
        (2**-0.5 * 0.25, -(2**-0.5) * 0.25, 0.0)
    )
    stage = Usd.Stage.Open(str(output))
    mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Cube"))
    diagonal = np.asarray(mass_api.GetDiagonalInertiaAttr().Get(), dtype=np.float64)
    quaternion = mass_api.GetPrincipalAxesAttr().Get()
    rotation = Gf.Matrix3d(Gf.Rotation(Gf.Quatd(quaternion)))
    rotation_array = np.asarray(
        [[float(rotation[row][column]) for column in range(3)] for row in range(3)]
    )
    # Gf matrices act on row vectors, so convert the authored principal frame
    # to the column-vector tensor convention used by NumPy here.
    reconstructed = rotation_array.T @ np.diag(diagonal) @ rotation_array
    angle = np.deg2rad(45.0)
    body_to_world = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_inertia = np.diag([2.0 / 3.0, 17.0 / 12.0, 17.0 / 12.0])
    expected_local = body_to_world.T @ world_inertia @ body_to_world
    assert reconstructed == pytest.approx(expected_local, abs=1e-6)


def test_apply_converts_si_mass_properties_to_authored_stage_units(
    tmp_path: Path,
) -> None:
    source = tmp_path / "centimeter_tonne.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(
        source,
        meters_per_unit=0.01,
        kilograms_per_unit=1000.0,
    )
    _write_structured_npz(
        npz,
        np.asarray([[-0.005, 0.0, 0.0], [0.005, 0.0, 0.0]]),
        np.asarray([1000.0, 3000.0]),
    )

    result = apply_vomp_mass_properties(
        source,
        npz,
        output,
        target_prim_path="/Cube",
        voxel_size_m=0.01,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    assert result.mass_properties.mass_kg == pytest.approx(0.004)
    assert result.mass_properties.center_of_mass_local_m == pytest.approx(
        (0.0025, 0.0, 0.0)
    )
    stage = Usd.Stage.Open(str(output))
    mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/Cube"))
    assert mass_api.GetMassAttr().Get() == pytest.approx(4.0e-6)
    assert tuple(mass_api.GetCenterOfMassAttr().Get()) == pytest.approx(
        (0.25, 0.0, 0.0)
    )
    assert tuple(mass_api.GetDiagonalInertiaAttr().Get()) == pytest.approx(
        (6.6666667e-7, 1.4166667e-6, 1.4166667e-6)
    )


def test_apply_requires_complete_voxel_field_attestation(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="does not record whether"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=False,
        )
    assert not output.exists()


def test_apply_requires_authored_stage_units(tmp_path: Path) -> None:
    source = tmp_path / "unitless.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source, author_units=False)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="metersPerUnit"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_voxels_outside_target_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[5.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="outside the world bound"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_derives_mesh_bounds_from_points_instead_of_authored_extent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stale_extent.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    stage = Usd.Stage.CreateNew(str(source))
    body = UsdGeom.Xform.Define(stage, "/Body")
    mesh = UsdGeom.Mesh.Define(stage, "/Body/Geometry")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1.0, -1.0, 0.0),
            Gf.Vec3f(1.0, -1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr([Gf.Vec3f(-100.0), Gf.Vec3f(100.0)])
    invisible = UsdGeom.Cube.Define(stage, "/Body/Invisible")
    invisible.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    guide = UsdGeom.Cube.Define(stage, "/Body/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()
    _write_structured_npz(
        npz,
        np.asarray([[50.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="outside the world bound"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Body",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()


def test_apply_rejects_boundable_without_computable_extent(tmp_path: Path) -> None:
    source = tmp_path / "missing_extent.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    stage = Usd.Stage.CreateNew(str(source))
    body = UsdGeom.Xform.Define(stage, "/Body")
    UsdGeom.Points.Define(stage, "/Body/Points")
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="no geometry bound"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Body",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )


@pytest.mark.parametrize(
    ("sampled_kind", "message"),
    [
        ("transform", "transform is time-varying"),
        ("topology", "time-sampled geometry"),
    ],
)
def test_apply_rejects_animated_descendant_geometry(
    tmp_path: Path,
    sampled_kind: str,
    message: str,
) -> None:
    source = tmp_path / f"animated_{sampled_kind}.usda"
    output = tmp_path / f"animated_{sampled_kind}_physics.usda"
    npz = tmp_path / f"animated_{sampled_kind}.npz"
    stage = Usd.Stage.CreateNew(str(source))
    body = UsdGeom.Xform.Define(stage, "/Body")
    mesh = UsdGeom.Mesh.Define(stage, "/Body/Geometry")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1.0, -1.0, 0.0),
            Gf.Vec3f(1.0, -1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    if sampled_kind == "transform":
        translate = UsdGeom.Xformable(mesh).AddTranslateOp()
        translate.Set(Gf.Vec3d(0.0))
        translate.Set(Gf.Vec3d(0.1, 0.0, 0.0), Usd.TimeCode(1.0))
    else:
        mesh.GetFaceVertexCountsAttr().Set([3], Usd.TimeCode(1.0))
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match=message):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Body",
            voxel_size_m=0.5,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()


def test_apply_binds_worker_npz_digest_through_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="lowercase SHA-256"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
            expected_npz_sha256="A" * 64,
        )
    with pytest.raises(VompIntegrationError, match="worker-reported"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
            expected_npz_sha256="0" * 64,
        )

    expected_digest = vomp_module._sha256(npz)
    real_serialize_json = vomp_module._serialize_json

    def replace_npz_before_publication(path: Path, value: dict[str, object]) -> Path:
        temporary = real_serialize_json(path, value)
        npz.write_bytes(b"replaced after worker verification")
        return temporary

    monkeypatch.setattr(vomp_module, "_serialize_json", replace_npz_before_publication)
    with pytest.raises(VompIntegrationError, match="before USD publication"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
            expected_npz_sha256=expected_digest,
        )

    assert not output.exists()
    assert not output.with_name("physics.vomp_mass_properties.json").exists()


def test_apply_rejects_scaled_rigid_body_frame(tmp_path: Path) -> None:
    source = tmp_path / "scaled.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source, scale=(2.0, 1.0, 1.0))
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="scale, shear, or reflection"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_does_not_overwrite_unsaved_source_layer_edits(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )
    live_stage = Usd.Stage.Open(str(source))
    live_stage.GetPrimAtPath("/Cube").SetCustomDataByKey("unsaved", True)
    try:
        with pytest.raises(VompIntegrationError, match="unsaved in-memory edits"):
            apply_vomp_mass_properties(
                source,
                npz,
                output,
                target_prim_path="/Cube",
                voxel_size_m=1.0,
                coordinate_unit_meters=1.0,
                complete_voxel_field=True,
            )
        assert live_stage.GetPrimAtPath("/Cube").GetCustomDataByKey("unsaved") is True
        assert not output.exists()
    finally:
        live_stage.GetRootLayer().Reload()


def test_apply_rejects_output_that_aliases_a_composed_input_layer(
    tmp_path: Path,
) -> None:
    component = tmp_path / "component.usda"
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(component)
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.subLayerPaths.append(component.name)
    root_layer.Save()
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="composed input layer"):
        apply_vomp_mass_properties(
            source,
            npz,
            component,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )


def test_apply_does_not_publish_usd_when_provenance_serialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    def fail_serialization(_path: Path, _value: dict) -> Path:
        raise RuntimeError("synthetic JSON failure")

    monkeypatch.setattr(vomp_module, "_serialize_json", fail_serialization)
    with pytest.raises(VompIntegrationError, match="publish"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()
    assert not output.with_name("physics.vomp_mass_properties.json").exists()


def test_apply_rejects_nonfinite_evidence_without_publishing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    with pytest.raises(VompIntegrationError, match="finite JSON values"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
            evidence_provenance={"score": float("nan")},
        )

    assert not output.exists()


def test_apply_rejects_voxel_sample_limit_without_publishing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.asarray([1000.0, 1000.0]),
    )
    monkeypatch.setattr(vomp_module, "_MAX_VOXEL_SAMPLES", 1)

    with pytest.raises(VompIntegrationError, match="between 1 and 1 voxel samples"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()


def test_apply_preserves_integration_error_from_stage_serialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1000.0]),
    )

    def fail_stage(_stage: object, _path: Path) -> Path:
        raise VompIntegrationError("synthetic stage failure")

    monkeypatch.setattr(vomp_module, "_serialize_stage", fail_stage)
    with pytest.raises(VompIntegrationError, match="synthetic stage failure"):
        apply_vomp_mass_properties(
            source,
            npz,
            output,
            target_prim_path="/Cube",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()


def test_apply_vomp_cli_help_and_smoke(tmp_path: Path) -> None:
    help_result = runner.invoke(app, ["apply-vomp", "--help"])
    assert help_result.exit_code == 0
    command = get_command(app)
    assert isinstance(command, click.Group)
    options = {
        option
        for parameter in command.commands["apply-vomp"].params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }
    assert {
        "--voxel-size-m",
        "--coordinate-unit-meters",
        "--complete-voxel-field",
    } <= options

    source = tmp_path / "source.usda"
    output = tmp_path / "physics.usda"
    npz = tmp_path / "materials.npz"
    _write_cube_stage(source)
    _write_structured_npz(
        npz,
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([800.0]),
    )
    result = runner.invoke(
        app,
        [
            "apply-vomp",
            str(source),
            str(npz),
            str(output),
            "--target-prim",
            "/Cube",
            "--voxel-size-m",
            "1.0",
            "--coordinate-unit-meters",
            "1.0",
            "--complete-voxel-field",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output.is_file()
    assert "800 kg" in result.stdout
