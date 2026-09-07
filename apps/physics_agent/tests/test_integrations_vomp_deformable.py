# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for VoMP volume-deformable authoring."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import pytest
from typer.main import get_command
from typer.testing import CliRunner

pxr = pytest.importorskip("pxr", reason="USD (pxr) not available in this env")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from physics_agent.cli import app  # noqa: E402
from physics_agent.integrations import (  # noqa: E402
    vomp_deformable as vomp_deformable_module,
)
from physics_agent.integrations.vomp import (  # noqa: E402
    VompIntegrationError,
    VompVoxelField,
)
from physics_agent.integrations.vomp_deformable import (  # noqa: E402
    AOUSD_DEFORMABLE_SCHEMA_COMMIT,
    NEWTON_1_4_AOUSD_BASELINE_COMMIT,
    NEWTON_DEFORMABLE_PROFILE,
    apply_vomp_volume_deformable,
    build_vomp_tet_mesh,
    reduce_vomp_elastic_field,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_cli_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "physics_agent.cli.setup_logging",
        lambda **_kwargs: logging.getLogger("physics_agent.tests.vomp_deformable"),
    )


def _field(
    coordinates: list[tuple[float, float, float]],
    density: list[float],
    *,
    youngs: list[float] | None = None,
    poisson: list[float] | None = None,
    voxel_size_m: float = 1.0,
) -> VompVoxelField:
    count = len(coordinates)
    return VompVoxelField(
        coordinates_world_m=np.asarray(coordinates, dtype=np.float64),
        density_kg_m3=np.asarray(density, dtype=np.float64),
        youngs_modulus_pa=np.asarray(
            youngs if youngs is not None else [3.0e5] * count,
            dtype=np.float64,
        ),
        poisson_ratio=np.asarray(
            poisson if poisson is not None else [0.3] * count,
            dtype=np.float64,
        ),
        voxel_size_m=voxel_size_m,
        source_schema="test",
        source_sha256="0" * 64,
    )


def _write_npz(
    path: Path,
    coordinates: list[tuple[float, float, float]],
    density: list[float],
    *,
    youngs: list[float] | None = None,
    poisson: list[float] | None = None,
) -> None:
    count = len(coordinates)
    np.savez_compressed(
        path,
        voxel_coords_world=np.asarray(coordinates, dtype=np.float64),
        density=np.asarray(density, dtype=np.float64),
        youngs_modulus=np.asarray(
            youngs if youngs is not None else [3.0e5] * count,
            dtype=np.float64,
        ),
        poisson_ratio=np.asarray(
            poisson if poisson is not None else [0.3] * count,
            dtype=np.float64,
        ),
    )


def _write_body_stage(
    path: Path,
    *,
    meters_per_unit: float = 1.0,
    kilograms_per_unit: float = 1.0,
) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    body = UsdGeom.Xform.Define(stage, "/World/SoftBody")
    visual = UsdGeom.Cube.Define(stage, "/World/SoftBody/Visual")
    visual.CreateSizeAttr(2.0)
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)
    assert body.GetPrim().IsValid()
    assert stage.GetRootLayer().Save()


def _signed_tet_volumes(mesh: object) -> np.ndarray:
    points = mesh.points_world_m
    indices = mesh.tet_vertex_indices
    p0 = points[indices[:, 0]]
    return (
        np.einsum(
            "ij,ij->i",
            np.cross(points[indices[:, 1]] - p0, points[indices[:, 2]] - p0),
            points[indices[:, 3]] - p0,
        )
        / 6.0
    )


def _applied_schemas(prim: object) -> set[str]:
    # GetAppliedSchemas() intentionally hides unregistered draft schema tokens.
    return {str(value) for value in prim.GetPrimTypeInfo().GetAppliedAPISchemas()}


def test_one_voxel_builds_six_positive_tetrahedra() -> None:
    mesh = build_vomp_tet_mesh(_field([(0.0, 0.0, 0.0)], [8.0]))

    assert mesh.point_count == 8
    assert mesh.tet_count == 6
    assert mesh.tet_vertex_indices.tolist() == [
        [0, 4, 6, 7],
        [0, 6, 2, 7],
        [0, 2, 3, 7],
        [0, 3, 1, 7],
        [0, 1, 5, 7],
        [0, 5, 4, 7],
    ]
    assert _signed_tet_volumes(mesh) == pytest.approx(np.full(6, 1.0 / 6.0))
    assert mesh.integrated_volume_m3 == pytest.approx(1.0)
    assert mesh.total_mass_kg == pytest.approx(8.0)
    assert sum(mesh.point_masses_kg) == pytest.approx(8.0)
    assert mesh.center_of_mass_world_m == pytest.approx((0.0, 0.0, 0.0))


def test_neighbor_voxels_share_vertices_and_conserve_density_center() -> None:
    field = _field(
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
        [1.0, 3.0],
    )
    mesh = build_vomp_tet_mesh(field)
    reversed_mesh = build_vomp_tet_mesh(
        _field(
            [(0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)],
            [3.0, 1.0],
        )
    )

    assert mesh.point_count == 12
    assert mesh.tet_count == 12
    assert np.all(_signed_tet_volumes(mesh) > 0.0)
    assert sum(_signed_tet_volumes(mesh)) == pytest.approx(2.0)
    assert mesh.total_mass_kg == pytest.approx(4.0)
    assert mesh.center_of_mass_world_m == pytest.approx((0.25, 0.0, 0.0))
    assert mesh.point_masses_kg == pytest.approx(
        [
            1.0 / 4.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            5.0 / 6.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 2.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            3.0 / 4.0,
        ]
    )
    assert mesh.topology_sha256 == reversed_mesh.topology_sha256
    assert mesh.points_world_m == pytest.approx(reversed_mesh.points_world_m)
    assert mesh.point_masses_kg == pytest.approx(reversed_mesh.point_masses_kg)


def test_large_near_zero_center_uses_scale_aware_reduction_tolerance() -> None:
    pitch = 1.0e-3
    axis = (np.arange(40, dtype=np.float64) - 19.5) * pitch
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    coordinates = grid.reshape(-1, 3)
    coordinates[:, 0] += 1.0e-14
    count = len(coordinates)
    field = VompVoxelField(
        coordinates_world_m=coordinates,
        density_kg_m3=np.ones(count, dtype=np.float64),
        youngs_modulus_pa=np.full(count, 3.0e5, dtype=np.float64),
        poisson_ratio=np.full(count, 0.3, dtype=np.float64),
        voxel_size_m=pitch,
        source_schema="test",
        source_sha256="0" * 64,
    )

    mesh = build_vomp_tet_mesh(field)

    assert mesh.voxel_count == 40**3
    assert mesh.center_of_mass_world_m == pytest.approx(
        (1.0e-14, 0.0, 0.0),
        abs=1.0e-13,
    )


def test_topology_limit_fails_without_coarsening() -> None:
    with pytest.raises(VompIntegrationError, match="refusing to subsample or coarsen"):
        build_vomp_tet_mesh(
            _field([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], [1.0, 1.0]),
            max_voxels=1,
        )


def test_elastic_reduction_requires_explicit_heterogeneity_policy() -> None:
    field = _field(
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
        [1.0, 1.0],
        youngs=[1.0e5, 3.0e5],
        poisson=[0.2, 0.4],
    )

    with pytest.raises(VompIntegrationError, match="spatially heterogeneous"):
        reduce_vomp_elastic_field(field)

    result = reduce_vomp_elastic_field(
        field,
        policy="homogeneous-volume-average",
    )
    assert result.heterogeneous is True
    assert result.validation_status == "conditional"
    assert result.youngs_modulus_pa == pytest.approx(2.0e5)
    assert result.poisson_ratio == pytest.approx(0.3)
    assert result.statistics["youngsModulusPa"]["voigtUpperBound"] == pytest.approx(
        2.0e5
    )
    assert result.statistics["youngsModulusPa"]["reussLowerBound"] == pytest.approx(
        1.5e5
    )


@pytest.mark.parametrize("poisson", [-0.9991, 0.4991])
def test_elastic_reduction_rejects_values_newton_would_clamp(
    poisson: float,
) -> None:
    with pytest.raises(VompIntegrationError, match="unclamped range"):
        reduce_vomp_elastic_field(_field([(0.0, 0.0, 0.0)], [1.0], poisson=[poisson]))


def test_elastic_reduction_rejects_out_of_range_source_before_averaging() -> None:
    with pytest.raises(VompIntegrationError, match="source material evidence"):
        reduce_vomp_elastic_field(
            _field(
                [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
                [1.0, 1.0],
                poisson=[-0.9991, 0.3],
            ),
            policy="homogeneous-volume-average",
        )


def test_apply_authors_newton_draft_contract_without_rigid_apis(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    result = apply_vomp_volume_deformable(
        source,
        npz,
        output,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    assert result.sample_count == 1
    assert result.point_count == 8
    assert result.tet_count == 6
    assert result.mass_kg == pytest.approx(800.0)
    assert result.material_reduction.validation_status == "supported"
    stage = Usd.Stage.Open(str(output))
    body = stage.GetPrimAtPath("/World/SoftBody")
    sim = stage.GetPrimAtPath(result.simulation_prim_path)
    material = stage.GetPrimAtPath(result.material_prim_path)
    assert {
        "PhysicsBodyAPI",
        "PhysicsDeformableBodyAPI",
    } <= _applied_schemas(body)
    assert "PhysicsRigidBodyAPI" not in _applied_schemas(body)
    assert "PhysicsMassAPI" not in _applied_schemas(body)
    body_enabled = body.GetAttribute("physics:bodyEnabled")
    assert body_enabled.Get() is True
    assert body_enabled.IsCustom() is False
    assert {
        "PhysicsVolumeDeformableSimAPI",
        "PhysicsCollisionAPI",
        "MaterialBindingAPI",
    } <= _applied_schemas(sim)
    assert {
        "PhysicsMaterialAPI",
        "PhysicsVolumeDeformableMaterialAPI",
    } <= _applied_schemas(material)
    assert stage.GetPrimAtPath("/World/SoftBody/Visual").IsValid()
    assert len(UsdGeom.TetMesh(sim).GetPointsAttr().Get()) == 8
    assert len(UsdGeom.TetMesh(sim).GetTetVertexIndicesAttr().Get()) == 6
    masses = list(sim.GetAttribute("physics:masses").Get())
    assert sim.GetAttribute("physics:collisionEnabled").IsCustom() is False
    assert sim.GetAttribute("physics:masses").IsCustom() is False
    assert len(masses) == 8
    assert sum(masses) == pytest.approx(800.0)
    bound_material = UsdShade.MaterialBindingAPI(sim).ComputeBoundMaterial("physics")[0]
    assert str(bound_material.GetPath()) == result.material_prim_path
    assert material.GetAttribute("physics:density").Get() == pytest.approx(800.0)
    assert material.GetAttribute("physics:youngsModulus").Get() == pytest.approx(3.0e5)
    assert material.GetAttribute("physics:poissonsRatio").Get() == pytest.approx(0.3)
    assert material.GetAttribute("physics:density").IsCustom() is False
    assert material.GetAttribute("physics:youngsModulus").IsCustom() is False
    assert material.GetAttribute("physics:poissonsRatio").IsCustom() is False

    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["schemaProfile"] == {
        "name": NEWTON_DEFORMABLE_PROFILE,
        "status": "draft",
        "authoringProposal": (
            "https://github.com/PixarAnimationStudios/OpenUSD-proposals/pull/111"
        ),
        "authoringProposalCommit": AOUSD_DEFORMABLE_SCHEMA_COMMIT,
        "newtonRuntimeProposalBaseline": NEWTON_1_4_AOUSD_BASELINE_COMMIT,
        "validatedRuntime": "newton==1.4.0",
        "runtimeNewtonSchemaPlugin": "newton-usd-schemas==0.4.1",
        "draftSchemaRegistration": "unregistered_raw_api_tokens",
    }
    assert provenance["massMapping"]["massKg"] == pytest.approx(800.0)
    assert provenance["massMapping"]["sourceMassKg"] == pytest.approx(800.0)
    assert provenance["massMapping"]["relativeFloat32Error"] <= 2.0e-6
    assert provenance["massMapping"]["centerOfMassWorldM"] == pytest.approx(
        result.center_of_mass_world_m
    )
    assert provenance["massMapping"]["sourceCenterOfMassWorldM"] == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert provenance["massMapping"]["centerOfMassTargetLocalM"] == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert provenance["massMapping"]["sourceCenterOfMassTargetLocalM"] == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    geometry_encoding = provenance["geometryEncoding"]
    assert geometry_encoding["centerComparisonSpace"] == (
        "target-local-origin-relative"
    )
    assert geometry_encoding["newtonWorldParticleType"] == "float32"
    assert geometry_encoding["newtonWorldVolumeRelativeError"] <= 2.5e-5
    assert geometry_encoding["newtonWorldMaximumTetVolumeRelativeError"] <= 2.5e-5
    assert provenance["elasticReduction"]["validationStatus"] == "supported"
    assert provenance["authoredMaterial"][
        "newtonEffectivePoissonRatio"
    ] == pytest.approx(0.3)
    usd_provenance = body.GetCustomDataByKey("physicsAgentVompDeformable")
    assert usd_provenance["source"]["npzSha256"] == provenance["source"]["npzSha256"]


def test_apply_rejects_heterogeneous_elasticity_transactionally(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    _write_npz(
        npz,
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)],
        [800.0, 1200.0],
        youngs=[1.0e5, 3.0e5],
        poisson=[0.2, 0.4],
    )

    with pytest.raises(VompIntegrationError, match="spatially heterogeneous"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()

    result = apply_vomp_volume_deformable(
        source,
        npz,
        output,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
        material_reduction="homogeneous-volume-average",
    )
    assert result.material_reduction.validation_status == "conditional"
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["fieldUse"]["density"] == (
        "spatially_preserved_as_lumped_point_mass"
    )
    assert provenance["fieldUse"]["youngsModulus"] == "homogeneous_volume_average"


@pytest.mark.parametrize(
    ("meters_per_unit", "kilograms_per_unit", "message"),
    [
        (0.01, 1.0, "requires metersPerUnit = 1"),
        (1.0, 1000.0, "requires kilogramsPerUnit = 1"),
    ],
)
def test_apply_rejects_units_newton_does_not_convert(
    tmp_path: Path,
    meters_per_unit: float,
    kilograms_per_unit: float,
    message: str,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(
        source,
        meters_per_unit=meters_per_unit,
        kilograms_per_unit=kilograms_per_unit,
    )
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match=message):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_existing_collision_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    visual = stage.GetPrimAtPath("/World/SoftBody/Visual")
    UsdPhysics.CollisionAPI.Apply(visual)
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match="existing collision contract"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_overridden_physics_material_binding(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = stage.GetPrimAtPath("/World/SoftBody")
    existing = UsdShade.Material.Define(stage, "/World/ExistingPhysicsMaterial")
    binding = UsdShade.MaterialBindingAPI.Apply(body)
    assert binding.Bind(
        existing,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match="overridden by a stronger"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_stale_kinematic_body_state(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = stage.GetPrimAtPath("/World/SoftBody")
    body.CreateAttribute(
        "physics:kinematicEnabled",
        Sdf.ValueTypeNames.Bool,
        custom=False,
    ).Set(True)
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match="skips kinematic"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("attribute_name", "sample_values"),
    [
        ("physics:bodyEnabled", (False, True)),
        ("physics:kinematicEnabled", (True, False)),
        ("physics:startsAsleep", (True, False)),
    ],
)
def test_apply_rejects_time_sampled_body_state(
    tmp_path: Path,
    attribute_name: str,
    sample_values: tuple[bool, bool],
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = stage.GetPrimAtPath("/World/SoftBody")
    attribute = body.CreateAttribute(
        attribute_name,
        Sdf.ValueTypeNames.Bool,
        custom=False,
    )
    attribute.Set(sample_values[0], Usd.TimeCode(0.0))
    attribute.Set(sample_values[1], Usd.TimeCode(1.0))
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(
        VompIntegrationError,
        match=rf"{attribute_name.removeprefix('physics:')} is time-sampled",
    ):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("body_state", "error"),
    [
        ("disabled", "bodyEnabled=false"),
        ("asleep", "startsAsleep=true"),
        ("simulation-owner", "simulationOwner targets"),
    ],
)
def test_apply_rejects_unsupported_existing_body_state(
    tmp_path: Path,
    body_state: str,
    error: str,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = stage.GetPrimAtPath("/World/SoftBody")
    if body_state == "disabled":
        body.CreateAttribute(
            "physics:bodyEnabled",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(False)
    elif body_state == "asleep":
        body.CreateAttribute(
            "physics:startsAsleep",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(True)
    else:
        UsdGeom.Scope.Define(stage, "/World/PhysicsScene")
        body.CreateRelationship(
            "physics:simulationOwner",
            custom=False,
        ).SetTargets([Sdf.Path("/World/PhysicsScene")])
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match=error):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_does_not_trust_foreign_provenance_as_ownership(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = stage.GetPrimAtPath("/World/SoftBody")
    body.AddAppliedSchema("PhysicsDeformableBodyAPI")
    body.SetCustomDataByKey(
        "physicsAgentVompDeformable",
        {"adapter": "foreign.adapter"},
    )
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match="non-VoMP deformable"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_nested_deformable_body_root(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    nested = UsdGeom.Xform.Define(stage, "/World/SoftBody/NestedBody").GetPrim()
    nested.AddAppliedSchema("PhysicsDeformableBodyAPI")
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match="nested physics body"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("reserved_suffix", "prim_state"),
    [
        ("PhysicsAgentVompSimulation", "inactive"),
        ("PhysicsAgentVompSimulation", "over"),
        ("PhysicsAgentVompLooks", "inactive"),
        ("PhysicsAgentVompLooks", "over"),
        ("PhysicsAgentVompLooks/VolumeMaterial", "inactive"),
        ("PhysicsAgentVompLooks/VolumeMaterial", "over"),
    ],
)
def test_apply_rejects_reserved_user_prims_excluded_from_default_traversal(
    tmp_path: Path,
    reserved_suffix: str,
    prim_state: str,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    looks_path = "/World/SoftBody/PhysicsAgentVompLooks"
    if reserved_suffix.endswith("VolumeMaterial"):
        looks = UsdGeom.Scope.Define(stage, looks_path).GetPrim()
        looks.SetCustomDataByKey(
            "physicsAgentVompOwner",
            "physics_agent.vomp_volume_deformable",
        )
    reserved_path = f"/World/SoftBody/{reserved_suffix}"
    if prim_state == "inactive":
        reserved = stage.DefinePrim(reserved_path, "Scope")
        reserved.SetActive(False)
        assert reserved.IsActive() is False
    else:
        reserved = stage.OverridePrim(reserved_path)
        assert reserved.IsDefined() is False
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    with pytest.raises(VompIntegrationError, match="already user-owned"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()


def test_apply_rejects_point3f_geometry_distortion(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    visual = UsdGeom.Cube(stage.GetPrimAtPath("/World/SoftBody/Visual"))
    visual.CreateSizeAttr(10.0)
    UsdGeom.Xformable(visual).AddTranslateOp().Set(Gf.Vec3d(1.0e8, 1.0e8, 1.0e8))
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(1.0e8, 1.0e8, 1.0e8)], [800.0])

    with pytest.raises(VompIntegrationError, match="materially distorted"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=10.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_newton_world_space_float32_geometry_collapse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = UsdGeom.Xformable(stage.GetPrimAtPath("/World/SoftBody"))
    translation = 1.0e12
    body.AddTranslateOp().Set(Gf.Vec3d(translation, translation, translation))
    assert stage.GetRootLayer().Save()
    # The asymmetric masses make independently reduced world-space centers
    # differ by several float64 ULPs.  The local CoM check must pass before the
    # Newton-facing float32 particle conversion detects the real collapse.
    _write_npz(
        npz,
        [
            (translation - 0.5, translation, translation),
            (translation + 0.5, translation, translation),
        ],
        [800.0, 1200.0],
    )

    with pytest.raises(
        VompIntegrationError,
        match="Newton 1.4 world-space deformable points collapse",
    ):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_newton_float32_world_matrix_quantization_collapse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = UsdGeom.Xformable(stage.GetPrimAtPath("/World/SoftBody"))
    # The double-precision transform puts the two x corners in adjacent float32
    # bins, but Newton first rounds this translation to 1e8. Both +/-4 corners
    # then land on the same tie-to-even value and the tetrahedra collapse.
    translation = 1.0e8 + 3.9
    body.AddTranslateOp().Set(Gf.Vec3d(translation, 0.0, 0.0))
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(translation, 0.0, 0.0)], [800.0])

    with pytest.raises(
        VompIntegrationError,
        match="Newton 1.4 world-space deformable points collapse",
    ):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=8.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_rejects_newton_float32_affine_mass_center_displacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    body = UsdGeom.Xformable(stage.GetPrimAtPath("/World/SoftBody"))
    # The 16 m voxel remains non-degenerate after Newton quantizes this affine,
    # but the half-bin translation rounds down and displaces every particle 4 m.
    translation = 100_000_004.0
    body.AddTranslateOp().Set(Gf.Vec3d(translation, 0.0, 0.0))
    assert stage.GetRootLayer().Save()
    _write_npz(npz, [(translation, 0.0, 0.0)], [800.0])

    with pytest.raises(
        VompIntegrationError,
        match="Newton 1.4 world-space deformable mass center is materially displaced",
    ):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=16.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_accepts_native_grid_float32_roundoff_at_64_voxel_offset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    voxel_size_m = 0.001
    coordinate = 64.0 * voxel_size_m
    _write_npz(npz, [(coordinate, coordinate, coordinate)], [800.0])

    result = apply_vomp_volume_deformable(
        source,
        npz,
        output,
        target_prim_path="/World/SoftBody",
        voxel_size_m=voxel_size_m,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    geometry = provenance["geometryEncoding"]
    assert geometry["maximumTetVolumeRelativeError"] > 1.0e-5
    assert (
        geometry["maximumTetVolumeRelativeError"] <= geometry["volumeRelativeTolerance"]
    )
    assert geometry["volumeRelativeTolerance"] == pytest.approx(2.5e-5)
    assert output.is_file()


def test_apply_rejects_float32_point_mass_center_distortion(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    stage = Usd.Stage.Open(str(source))
    visual = UsdGeom.Cube(stage.GetPrimAtPath("/World/SoftBody/Visual"))
    visual.CreateSizeAttr(20_002.0)
    assert stage.GetRootLayer().Save()
    _write_npz(
        npz,
        [(-10_000.0, 0.0, 0.0), (10_000.0, 0.0, 0.0)],
        [0.0009942697954362768, 0.0007971527215165565],
    )

    with pytest.raises(VompIntegrationError, match="mass center is materially"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not output.exists()


def test_apply_is_idempotent_for_owned_generated_prims(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    first_result = apply_vomp_volume_deformable(
        source,
        npz,
        first,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )
    second_result = apply_vomp_volume_deformable(
        first,
        npz,
        second,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    first_provenance = json.loads(
        first_result.provenance_path.read_text(encoding="utf-8")
    )
    second_provenance = json.loads(
        second_result.provenance_path.read_text(encoding="utf-8")
    )
    assert first_provenance["topology"] == second_provenance["topology"]
    stage = Usd.Stage.Open(str(second))
    sim_prims = [
        prim
        for prim in stage.Traverse()
        if "PhysicsVolumeDeformableSimAPI" in _applied_schemas(prim)
    ]
    assert [str(prim.GetPath()) for prim in sim_prims] == [
        second_result.simulation_prim_path
    ]


def test_reapply_canonicalizes_owned_simulation_transform_and_orientation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])
    first_result = apply_vomp_volume_deformable(
        source,
        npz,
        first,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )
    stage = Usd.Stage.Open(str(first))
    sim = UsdGeom.TetMesh(stage.GetPrimAtPath(first_result.simulation_prim_path))
    sim.AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
    sim.CreateOrientationAttr(UsdGeom.Tokens.leftHanded).Set(UsdGeom.Tokens.leftHanded)
    assert stage.GetRootLayer().Save()

    second_result = apply_vomp_volume_deformable(
        first,
        npz,
        second,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    stage = Usd.Stage.Open(str(second))
    sim = UsdGeom.TetMesh(stage.GetPrimAtPath(second_result.simulation_prim_path))
    assert sim.GetOrderedXformOps() == []
    assert sim.GetResetXformStack() is False
    assert sim.GetOrientationAttr().Get() == UsdGeom.Tokens.rightHanded
    points = np.asarray(sim.GetPointsAttr().Get(), dtype=np.float64)
    assert np.min(points[:, 0]) == pytest.approx(-0.5)
    assert np.max(points[:, 0]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "stale_state",
    ["surface-schema", "rest-shape", "surface-face-indices"],
)
def test_reapply_rejects_unsupported_stale_owned_simulation_state(
    tmp_path: Path,
    stale_state: str,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])
    first_result = apply_vomp_volume_deformable(
        source,
        npz,
        first,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )
    stage = Usd.Stage.Open(str(first))
    sim = stage.GetPrimAtPath(first_result.simulation_prim_path)
    if stale_state == "surface-schema":
        sim.AddAppliedSchema("PhysicsSurfaceDeformableSimAPI")
    elif stale_state == "rest-shape":
        sim.CreateAttribute(
            "physics:restShapePoints",
            Sdf.ValueTypeNames.Point3fArray,
            custom=False,
        ).Set([Gf.Vec3f(123.0, 0.0, 0.0)])
    else:
        UsdGeom.TetMesh(sim).CreateSurfaceFaceVertexIndicesAttr().Set(
            [Gf.Vec3i(0, 1, 999)]
        )
    assert stage.GetRootLayer().Save()

    with pytest.raises(VompIntegrationError, match="unsupported stale"):
        apply_vomp_volume_deformable(
            first,
            npz,
            second,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not second.exists()


@pytest.mark.parametrize("reserved_prim", ["simulation", "looks", "material"])
@pytest.mark.parametrize("prim_state", ["inactive", "undefined"])
def test_reapply_rejects_non_authorable_owned_reserved_prim(
    tmp_path: Path,
    reserved_prim: str,
    prim_state: str,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    first = tmp_path / "first.usda"
    second = tmp_path / "second.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])
    first_result = apply_vomp_volume_deformable(
        source,
        npz,
        first,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )
    stage = Usd.Stage.Open(str(first))
    material_path = Sdf.Path(first_result.material_prim_path)
    reserved_path = {
        "simulation": Sdf.Path(first_result.simulation_prim_path),
        "looks": material_path.GetParentPath(),
        "material": material_path,
    }[reserved_prim]
    reserved = stage.GetPrimAtPath(reserved_path)
    if prim_state == "inactive":
        reserved.SetActive(False)
        assert reserved.IsActive() is False
    else:
        reserved.SetSpecifier(Sdf.SpecifierOver)
        assert reserved.IsDefined() is False
    assert stage.GetRootLayer().Save()

    with pytest.raises(
        VompIntegrationError,
        match=rf"owned generated Physics Agent prim is {prim_state}",
    ):
        apply_vomp_volume_deformable(
            first,
            npz,
            second,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )
    assert not second.exists()


def test_apply_preserves_integration_error_from_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])

    def fail_publication(**_kwargs: object) -> None:
        raise VompIntegrationError("synthetic publication failure")

    monkeypatch.setattr(
        vomp_deformable_module,
        "_publish_artifact_pair",
        fail_publication,
    )

    with pytest.raises(VompIntegrationError, match="synthetic publication failure"):
        apply_vomp_volume_deformable(
            source,
            npz,
            output,
            target_prim_path="/World/SoftBody",
            voxel_size_m=1.0,
            coordinate_unit_meters=1.0,
            complete_voxel_field=True,
        )

    assert not output.exists()


def test_cli_help_and_smoke(tmp_path: Path) -> None:
    help_result = runner.invoke(app, ["apply-vomp-deformable", "--help"])
    assert help_result.exit_code == 0
    command = get_command(app)
    assert isinstance(command, click.Group)
    options = {
        option
        for parameter in command.commands["apply-vomp-deformable"].params
        if isinstance(parameter, click.Option)
        for option in parameter.opts
    }
    assert {
        "--voxel-size-m",
        "--coordinate-unit-meters",
        "--complete-voxel-field",
        "--material-reduction",
        "--max-deformable-voxels",
    } <= options

    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    _write_npz(npz, [(0.0, 0.0, 0.0)], [800.0])
    result = runner.invoke(
        app,
        [
            "apply-vomp-deformable",
            str(source),
            str(npz),
            str(output),
            "--target-prim",
            "/World/SoftBody",
            "--voxel-size-m",
            "1.0",
            "--coordinate-unit-meters",
            "1.0",
            "--complete-voxel-field",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert output.is_file()
    assert "Mass" in result.stdout
    assert "kg" in result.stdout
    assert "6" in result.stdout


def test_newton_1_4_imports_authored_volume_contract(tmp_path: Path) -> None:
    if os.environ.get("WU_UNIT_TEST_GROUP") == "apps-platform":
        try:
            __import__("newton_usd_schemas")
            newton = __import__("newton")
        except ModuleNotFoundError as exc:
            pytest.fail(
                "apps-platform must install Newton USD conformance dependencies: "
                f"{exc}",
                pytrace=False,
            )
    else:
        pytest.importorskip(
            "newton_usd_schemas",
            reason="Newton USD importer schemas are not installed",
        )
        newton = pytest.importorskip(
            "newton", reason="Newton importer extra not installed"
        )
    version = importlib.metadata.version("newton")
    assert version == "1.4.0", f"Newton 1.4.0 conformance test, found {version}"
    schema_version = importlib.metadata.version("newton-usd-schemas")
    assert schema_version == "0.4.1", (
        f"newton-usd-schemas 0.4.1 conformance test, found {schema_version}"
    )

    source = tmp_path / "source.usda"
    npz = tmp_path / "materials.npz"
    output = tmp_path / "deformable.usda"
    _write_body_stage(source)
    _write_npz(npz, [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)], [1.0, 3.0])
    authored = apply_vomp_volume_deformable(
        source,
        npz,
        output,
        target_prim_path="/World/SoftBody",
        voxel_size_m=1.0,
        coordinate_unit_meters=1.0,
        complete_voxel_field=True,
    )

    builder = newton.ModelBuilder()
    imported = builder.add_usd(str(output), return_deformable_results=True)
    assert authored.simulation_prim_path in imported["path_soft_map"]
    ranges = imported["path_soft_map"][authored.simulation_prim_path]
    assert tuple(ranges["particle"]) == (0, 12)
    assert tuple(ranges["tet"]) == (0, 12)
    assert np.asarray(builder.particle_mass, dtype=np.float64) == pytest.approx(
        [
            1.0 / 4.0,
            1.0 / 12.0,
            1.0 / 12.0,
            1.0 / 12.0,
            5.0 / 6.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 2.0,
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 4.0,
            3.0 / 4.0,
        ]
    )
    assert sum(builder.particle_mass) == pytest.approx(4.0)
    shear_modulus, lame_first_parameter, _ = builder.tet_materials[0]
    assert shear_modulus == pytest.approx(3.0e5 / 2.6, rel=1.0e-5)
    assert lame_first_parameter == pytest.approx(
        3.0e5 * 0.3 / (1.3 * 0.4),
        rel=1.0e-5,
    )
