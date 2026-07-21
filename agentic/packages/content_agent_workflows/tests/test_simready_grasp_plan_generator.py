# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic machine geometry-proof SimReady grasp plans."""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import content_agent_workflows.simready.conform_profile as conform_profile_module
import content_agent_workflows.simready.grasp_plan_generator as generator_module
from content_agent_workflows.simready import (
    SimReadyConformanceInput,
    generate_simready_grasp_plan,
    run_simready_profile_conformance,
)
from content_agent_workflows.simready.grasp_plan_generator import (
    GraspPlanGenerationError,
)
from content_agent_workflows.simready.models import (
    SIMREADY_GRASP_PLAN_ANALYTIC_GENERATOR_VERSION,
    SIMREADY_GRASP_PLAN_ANALYTIC_SCHEMA_VERSION,
    SIMREADY_GRASP_PLAN_COMPOSED_GENERATOR_VERSION,
    SIMREADY_GRASP_PLAN_COMPOSED_SCHEMA_VERSION,
    SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION,
    SIMREADY_GRASP_PLAN_GENERATOR_VERSION,
    SimReadyGraspPlan,
    SimReadyGraspPlanAnalyticMachineProvenance,
    SimReadyGraspPlanComposedMachineProvenance,
    SimReadyGraspPlanMachineProvenance,
    SimReadyValidationInput,
)
from content_agent_workflows.simready.validate_profile import (
    run_simready_profile_validation,
)


@pytest.fixture(autouse=True)
def _stub_conformance_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: SimpleNamespace(
            foundation_root=None,
            foundation_commit=None,
            foundation_spec_root=None,
            warnings=[],
        ),
    )


def _new_stage(path: Path):
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(root.GetPrim())
    return stage, root


def _add_mesh(
    stage,
    path: str,
    *,
    points: list[tuple[float, float, float]] | None = None,
    counts: list[int] | None = None,
    indices: list[int] | None = None,
):
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        points if points is not None else [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
    )
    mesh.CreateFaceVertexCountsAttr(counts if counts is not None else [3])
    mesh.CreateFaceVertexIndicesAttr(indices if indices is not None else [0, 1, 2])
    return mesh


def _add_cube(stage, path: str, *, size: float = 2.0):
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(size)
    return cube


def _write_triangle_asset(path: Path) -> None:
    stage, _root = _new_stage(path)
    _add_mesh(stage, "/robot/body")
    assert stage.GetRootLayer().Save()


def _machine_provenance(result) -> SimReadyGraspPlanMachineProvenance:
    provenance = result.plan.provenance
    assert isinstance(provenance, SimReadyGraspPlanMachineProvenance)
    return provenance


def test_generator_is_deterministic_across_traversal_order_and_reuses_exact_bytes(
    tmp_path: Path,
) -> None:
    selections = []
    for ordinal, paths in enumerate(
        (("/robot/z_mesh", "/robot/a_mesh"), ("/robot/a_mesh", "/robot/z_mesh"))
    ):
        asset = tmp_path / f"asset-{ordinal}.usda"
        stage, _root = _new_stage(asset)
        for path in paths:
            _add_mesh(stage, path)
        assert stage.GetRootLayer().Save()
        source_bytes = asset.read_bytes()

        output = tmp_path / f"plan-{ordinal}.json"
        first = generate_simready_grasp_plan(asset, output, width=0.02)
        first_bytes = output.read_bytes()
        second = generate_simready_grasp_plan(asset, output, width=0.02)

        assert not first.reused_output
        assert second.reused_output
        assert output.read_bytes() == first_bytes
        assert first.plan_sha256 == second.plan_sha256
        assert asset.read_bytes() == source_bytes
        assert first_bytes.endswith(b"\n")
        assert b'": ' not in first_bytes
        selections.append(_machine_provenance(first).selected_triangle.mesh_prim_path)

    assert selections == ["/robot/a_mesh", "/robot/a_mesh"]


def test_generator_triangulates_ngon_and_transforms_to_default_prim_local_space(
    tmp_path: Path,
) -> None:
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "transformed.usda"
    stage, root = _new_stage(asset)
    root.AddTranslateOp().Set((100, 200, 300))
    part = UsdGeom.Xform.Define(stage, "/robot/part")
    part.AddTranslateOp().Set((10, 0, 0))
    part.AddScaleOp().Set((2, 3, 1))
    _add_mesh(
        stage,
        "/robot/part/quad",
        points=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        counts=[4],
        indices=[0, 1, 2, 3],
    )
    assert stage.GetRootLayer().Save()

    result = generate_simready_grasp_plan(
        asset,
        tmp_path / "plan.json",
        width=0.125,
    )
    provenance = _machine_provenance(result)

    assert provenance.selected_triangle.face_index == 0
    assert provenance.selected_triangle.triangle_index == 0
    assert provenance.selected_triangle.point_indices == [0, 1, 2]
    assert provenance.selected_triangle.default_prim_local_points == [
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [12.0, 3.0, 0.0],
    ]
    assert result.plan.grasp_lines[0].points == [
        [11.0, 0.75, 0.0],
        [11.5, 0.75, 0.0],
    ]
    assert provenance.barycentric_coordinates == [
        ["1/2", "1/4", "1/4"],
        ["1/4", "1/2", "1/4"],
    ]
    assert provenance.width_stage_units == 0.125


def test_generator_proves_canonical_cube_surface_and_emits_v2_plan(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "cube.usda"
    stage, _root = _new_stage(asset)
    _add_cube(stage, "/robot/body", size=2.0)
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()

    result = generate_simready_grasp_plan(
        asset,
        tmp_path / "cube-plan.json",
        width=0.04,
    )

    provenance = result.plan.provenance
    assert isinstance(provenance, SimReadyGraspPlanAnalyticMachineProvenance)
    assert result.plan.schema_version == SIMREADY_GRASP_PLAN_ANALYTIC_SCHEMA_VERSION
    assert provenance.source == "machine_analytic_geometry_proof"
    assert (
        provenance.implementation_version
        == SIMREADY_GRASP_PLAN_ANALYTIC_GENERATOR_VERSION
    )
    assert provenance.selected_surface.primitive_type == "Cube"
    assert provenance.selected_surface.prim_path == "/robot/body"
    assert provenance.selected_surface.size == 2.0
    assert provenance.selected_surface.face_index == 0
    assert provenance.selected_surface.triangle_index == 0
    assert provenance.selected_surface.corner_indices == [0, 1, 2]
    assert provenance.selected_surface.primitive_local_points == [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
    ]
    assert result.plan.grasp_lines[0].points == [
        [0.0, -0.5, -1.0],
        [0.5, -0.5, -1.0],
    ]
    assert asset.read_bytes() == source_bytes


@pytest.mark.parametrize("geometry_kind", ["mesh", "cube"])
def test_generator_maps_reset_xform_surface_into_default_prim_space(
    tmp_path: Path,
    geometry_kind: str,
) -> None:
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / f"reset-{geometry_kind}.usda"
    stage, root = _new_stage(asset)
    root.AddTranslateOp().Set((100.0, 0.0, 0.0))
    reset = UsdGeom.Xform.Define(stage, "/robot/reset")
    reset.AddTranslateOp().Set((10.0, 0.0, 0.0))
    reset.SetResetXformStack(True)
    if geometry_kind == "mesh":
        _add_mesh(stage, "/robot/reset/surface")
        expected_points = [[-89.5, 0.5, 0.0], [-89.0, 0.5, 0.0]]
    else:
        _add_cube(stage, "/robot/reset/surface", size=2.0)
        expected_points = [[-90.0, -0.5, -1.0], [-89.5, -0.5, -1.0]]
    assert stage.GetRootLayer().Save()

    result = generate_simready_grasp_plan(
        asset,
        tmp_path / f"reset-{geometry_kind}-plan.json",
        width=0.04,
    )

    assert result.plan.grasp_lines[0].points == expected_points


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("wrong_schema", "requires the v2"),
        ("barycentrics", "fixed rational coordinates"),
        ("duplicate_dependency", "unique and sorted"),
        ("missing_source_dependency", "exactly one source asset"),
        ("source_dependency_hash", "dependency SHA-256 does not match"),
        ("corner_indices", "canonical Cube face triangle"),
        ("primitive_points", "composed Cube size"),
        ("zero_line", "line must be nonzero"),
        ("proof_line_mismatch", "selected surface triangle"),
        ("plan_source_hash", "source SHA-256 must match"),
        ("multiple_lines", "exactly one grasp line"),
        ("line_mismatch", "line points must match"),
        ("width_mismatch", "width must match"),
    ],
)
def test_analytic_plan_contract_fails_closed(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    asset = tmp_path / f"cube-{case}.usda"
    stage, _root = _new_stage(asset)
    _add_cube(stage, "/robot/body", size=2.0)
    assert stage.GetRootLayer().Save()
    result = generate_simready_grasp_plan(
        asset,
        tmp_path / f"cube-{case}.json",
        width=0.04,
    )
    payload = result.plan.model_dump(mode="json")
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    dependencies = provenance["dependencies"]
    assert isinstance(dependencies, list)
    grasp_lines = payload["grasp_lines"]
    assert isinstance(grasp_lines, list)

    if case == "wrong_schema":
        payload["schema_version"] = "content-agent-workflows.simready-grasp-plan.v1"
    elif case == "barycentrics":
        provenance["barycentric_coordinates"] = [["1", "0", "0"]] * 2
    elif case == "duplicate_dependency":
        dependencies.append(copy.deepcopy(dependencies[0]))
    elif case == "missing_source_dependency":
        dependencies[0]["role"] = "dependency"
    elif case == "source_dependency_hash":
        dependencies[0]["sha256"] = "f" * 64
    elif case == "corner_indices":
        provenance["selected_surface"]["corner_indices"] = [0, 1, 3]
    elif case == "primitive_points":
        provenance["selected_surface"]["primitive_local_points"][0][0] = 0.0
    elif case == "zero_line":
        points = provenance["line_points_default_prim_local"]
        points[1] = copy.deepcopy(points[0])
    elif case == "proof_line_mismatch":
        provenance["line_points_default_prim_local"][0][0] = 0.25
    elif case == "plan_source_hash":
        payload["source_asset_sha256"] = "f" * 64
    elif case == "multiple_lines":
        grasp_lines.append(copy.deepcopy(grasp_lines[0]))
        grasp_lines[1]["prim_path"] = "/robot/grasp_identifier_second"
    elif case == "line_mismatch":
        grasp_lines[0]["points"][0][0] = 0.25
    else:
        grasp_lines[0]["widths"] = [0.05]

    with pytest.raises(ValueError, match=reason):
        SimReadyGraspPlan.model_validate(payload, strict=True)


def test_generator_ranks_mesh_and_cube_by_transformed_surface_area(
    tmp_path: Path,
) -> None:
    small_cube_asset = tmp_path / "mesh-wins.usda"
    stage, _root = _new_stage(small_cube_asset)
    _add_mesh(stage, "/robot/mesh")
    _add_cube(stage, "/robot/cube", size=1.0)
    assert stage.GetRootLayer().Save()

    mesh_result = generate_simready_grasp_plan(
        small_cube_asset,
        tmp_path / "mesh-plan.json",
        width=0.01,
    )
    mesh_provenance = mesh_result.plan.provenance
    assert isinstance(mesh_provenance, SimReadyGraspPlanMachineProvenance)
    assert mesh_provenance.selected_triangle.mesh_prim_path == "/robot/mesh"

    large_cube_asset = tmp_path / "cube-wins.usda"
    stage, _root = _new_stage(large_cube_asset)
    _add_mesh(stage, "/robot/mesh")
    _add_cube(stage, "/robot/cube", size=10.0)
    assert stage.GetRootLayer().Save()

    cube_result = generate_simready_grasp_plan(
        large_cube_asset,
        tmp_path / "large-cube-plan.json",
        width=0.01,
    )
    cube_provenance = cube_result.plan.provenance
    assert isinstance(cube_provenance, SimReadyGraspPlanAnalyticMachineProvenance)
    assert cube_provenance.selected_surface.prim_path == "/robot/cube"


def test_generator_proves_valid_nested_composed_instance_and_gsp_consumes_plan(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "composed-instance.usda"
    stage, _root = _new_stage(asset)
    model = UsdGeom.Xform.Define(stage, "/Model")
    _add_mesh(stage, "/Model/body")
    instance = UsdGeom.Xform.Define(stage, "/robot/instance")
    instance.GetPrim().GetReferences().AddInternalReference(model.GetPath())
    instance.GetPrim().SetInstanceable(True)
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()

    plan_path = tmp_path / "composed-plan.json"
    result = generate_simready_grasp_plan(asset, plan_path, width=0.04)

    provenance = result.plan.provenance
    assert isinstance(provenance, SimReadyGraspPlanComposedMachineProvenance)
    assert result.plan.schema_version == SIMREADY_GRASP_PLAN_COMPOSED_SCHEMA_VERSION
    assert (
        provenance.implementation_version
        == SIMREADY_GRASP_PLAN_COMPOSED_GENERATOR_VERSION
    )
    assert provenance.selected_triangle is not None
    assert provenance.selected_triangle.mesh_prim_path == "/robot/instance/body"
    assert provenance.selected_surface is None
    assert provenance.proof_checks.composed_instance_proxies_resolved

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001"],
            grasp_plan_path=str(plan_path),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["GSP.001"]
    assert asset.read_bytes() == source_bytes
    output_stage = Usd.Stage.Open(report.output_usd_path, load=Usd.Stage.LoadAll)
    assert output_stage is not None
    assert output_stage.GetPrimAtPath("/robot/grasp_identifier_machine_geometry")


def test_generator_reuses_exact_existing_machine_target_and_rejects_drift(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "asset.usda"
    _write_triangle_asset(asset)
    initial_plan = tmp_path / "initial-plan.json"
    generate_simready_grasp_plan(asset, initial_plan, width=0.04)
    authored = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "initial-conform"),
            repair_requirements=["GSP.001"],
            grasp_plan_path=str(initial_plan),
            force=True,
        )
    )
    assert authored.passed
    authored_path = Path(authored.output_usd_path)
    authored_bytes = authored_path.read_bytes()

    reused = generate_simready_grasp_plan(
        authored_path,
        tmp_path / "reused-plan.json",
        width=0.04,
    )

    assert reused.reused_existing_grasp_line
    assert authored_path.read_bytes() == authored_bytes
    reuse_report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(authored_path),
            output_dir=str(tmp_path / "reuse-conform"),
            repair_requirements=["GSP.001"],
            grasp_plan_path=None,
            force=True,
        )
    )
    assert reuse_report.passed

    stage = Usd.Stage.Open(str(authored_path), load=Usd.Stage.LoadAll)
    assert stage is not None
    curve = UsdGeom.BasisCurves(
        stage.GetPrimAtPath("/robot/grasp_identifier_machine_geometry")
    )
    assert curve.GetWidthsAttr().Set([0.05])
    assert stage.GetRootLayer().Save()
    with pytest.raises(GraspPlanGenerationError, match="differs"):
        generate_simready_grasp_plan(
            authored_path,
            tmp_path / "drifted-plan.json",
            width=0.04,
        )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("no_mesh", "no supported explicit Mesh or Cube surface"),
        ("degenerate", "no usable finite nondegenerate supported surface"),
        ("malformed_counts", "face counts do not match"),
        ("bad_index", "out-of-range topology index"),
        ("nonfinite", "not a finite 3D point"),
        ("nonfinite_transform", "is nonfinite"),
        ("singular_transform", "is singular"),
        ("time_sampled_points", "time-varying points"),
        ("time_sampled_cube_size", "time-varying size"),
        ("invalid_cube_size", "size must be positive"),
        ("time_sampled_transform", "Time-varying transform"),
        ("default_instance", "default prim cannot be an instance"),
    ],
)
def test_generator_blocks_invalid_or_unprovable_geometry(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / f"{case}.usda"
    stage, root = _new_stage(asset)
    if case == "no_mesh":
        pass
    elif case == "degenerate":
        _add_mesh(
            stage,
            "/robot/body",
            points=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        )
    elif case == "malformed_counts":
        _add_mesh(stage, "/robot/body", counts=[4], indices=[0, 1, 2])
    elif case == "bad_index":
        _add_mesh(stage, "/robot/body", indices=[0, 1, 9])
    elif case == "nonfinite":
        _add_mesh(
            stage,
            "/robot/body",
            points=[(0, 0, 0), (math.nan, 0, 0), (0, 1, 0)],
        )
    elif case == "nonfinite_transform":
        root.AddTranslateOp().Set((math.nan, 0, 0))
        _add_mesh(stage, "/robot/body")
    elif case == "singular_transform":
        root.AddScaleOp().Set((1, 0, 1))
        _add_mesh(stage, "/robot/body")
    elif case == "time_sampled_points":
        mesh = _add_mesh(stage, "/robot/body")
        mesh.GetPointsAttr().Set(
            [(0, 0, 0), (3, 0, 0), (0, 3, 0)],
            Usd.TimeCode(1),
        )
    elif case == "time_sampled_cube_size":
        cube = _add_cube(stage, "/robot/body")
        cube.GetSizeAttr().Set(3.0, Usd.TimeCode(1))
    elif case == "invalid_cube_size":
        _add_cube(stage, "/robot/body", size=0.0)
    elif case == "time_sampled_transform":
        translate = root.AddTranslateOp()
        translate.Set((0, 0, 0), Usd.TimeCode(0))
        translate.Set((1, 0, 0), Usd.TimeCode(1))
        _add_mesh(stage, "/robot/body")
    elif case == "default_instance":
        model = UsdGeom.Xform.Define(stage, "/model")
        _add_mesh(stage, "/model/body")
        root.GetPrim().GetReferences().AddInternalReference(model.GetPath())
        root.GetPrim().SetInstanceable(True)
    assert stage.GetRootLayer().Save()

    with pytest.raises(GraspPlanGenerationError, match=reason):
        generate_simready_grasp_plan(
            asset,
            tmp_path / "plan.json",
            width=0.01,
        )
    assert not (tmp_path / "plan.json").exists()


def _write_dependency_root(path: Path, dependency: str) -> None:
    path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "robot"
    subLayers = [@{dependency}@]
)

def Xform "robot"
{{
    def Mesh "body"
    {{
        point3f[] points = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }}
}}
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing", "unresolved paths"),
        ("outside", "outside the source package"),
        ("symlink", "contains a symlink"),
    ],
)
def test_generator_blocks_missing_outside_and_symlink_dependencies(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    asset = package / "root.usda"
    if case == "missing":
        dependency = "missing.usda"
    elif case == "outside":
        outside = tmp_path / "outside.usda"
        outside.write_text("#usda 1.0\n", encoding="utf-8")
        dependency = "../outside.usda"
    else:
        real = package / "real.usda"
        real.write_text("#usda 1.0\n", encoding="utf-8")
        (package / "dependency.usda").symlink_to(real.name)
        dependency = "dependency.usda"
    _write_dependency_root(asset, dependency)

    with pytest.raises(GraspPlanGenerationError, match=reason):
        generate_simready_grasp_plan(
            asset,
            tmp_path / "plan.json",
            width=0.01,
        )


def test_generator_binds_complete_nested_local_dependency_closure(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    layer_dir = package / "layers"
    texture_dir = package / "textures"
    layer_dir.mkdir(parents=True)
    texture_dir.mkdir()
    (texture_dir / "proof.bin").write_bytes(b"geometry-proof-dependency")
    (layer_dir / "dependency.usda").write_text(
        """#usda 1.0

over "robot"
{
    def Scope "evidence"
    {
        asset proof:file = @../textures/proof.bin@
    }
}
""",
        encoding="utf-8",
    )
    asset = package / "root.usda"
    _write_dependency_root(asset, "layers/dependency.usda")

    result = generate_simready_grasp_plan(
        asset,
        tmp_path / "plan.json",
        width=0.01,
    )
    dependencies = _machine_provenance(result).dependencies

    assert [(item.relative_path, item.role) for item in dependencies] == [
        ("layers/dependency.usda", "dependency"),
        ("root.usda", "source_asset"),
        ("textures/proof.bin", "dependency"),
    ]


def test_generator_binds_usdz_to_exact_archive_bytes(tmp_path: Path) -> None:
    UsdUtils = pytest.importorskip("pxr.UsdUtils")

    source = tmp_path / "source.usda"
    _write_triangle_asset(source)
    package = tmp_path / "asset.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(source), str(package))
    source.unlink()
    package_bytes = package.read_bytes()

    result = generate_simready_grasp_plan(
        package,
        tmp_path / "plan.json",
        width=0.01,
    )
    provenance = _machine_provenance(result)

    assert package.read_bytes() == package_bytes
    assert [(item.relative_path, item.role) for item in provenance.dependencies] == [
        ("asset.usdz", "source_asset")
    ]
    assert result.plan.source_asset_sha256 == generator_module._file_sha256(package)


def test_generator_rejects_source_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_triangle_asset(asset)
    original_select = generator_module._select_surface_triangle

    def mutate_source(*args, **kwargs):
        selected = original_select(*args, **kwargs)
        asset.write_bytes(asset.read_bytes() + b"\n")
        return selected

    monkeypatch.setattr(generator_module, "_select_surface_triangle", mutate_source)

    with pytest.raises(GraspPlanGenerationError, match="changed|differs"):
        generate_simready_grasp_plan(
            asset,
            tmp_path / "plan.json",
            width=0.01,
        )
    assert not (tmp_path / "plan.json").exists()


def test_generator_rejects_dependency_mutation_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    dependency = package / "dependency.usda"
    dependency.write_text("#usda 1.0\n", encoding="utf-8")
    asset = package / "root.usda"
    _write_dependency_root(asset, "dependency.usda")
    original_select = generator_module._select_surface_triangle

    def mutate_dependency(*args, **kwargs):
        selected = original_select(*args, **kwargs)
        dependency.write_bytes(dependency.read_bytes() + b"\n")
        return selected

    monkeypatch.setattr(generator_module, "_select_surface_triangle", mutate_dependency)

    with pytest.raises(GraspPlanGenerationError, match="changed|differs"):
        generate_simready_grasp_plan(
            asset,
            tmp_path / "plan.json",
            width=0.01,
        )
    assert not (tmp_path / "plan.json").exists()


def test_generator_never_overwrites_a_different_output(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_triangle_asset(asset)
    output = tmp_path / "plan.json"
    output.write_bytes(b"preserve-existing-output\n")

    with pytest.raises(GraspPlanGenerationError, match="already exists"):
        generate_simready_grasp_plan(asset, output, width=0.01)

    assert output.read_bytes() == b"preserve-existing-output\n"


def test_generator_requires_explicit_positive_width_and_cli_plumbs_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset = tmp_path / "asset.usda"
    _write_triangle_asset(asset)
    width_parameter = inspect.signature(generate_simready_grasp_plan).parameters[
        "width"
    ]
    assert width_parameter.default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        generate_simready_grasp_plan(asset, tmp_path / "missing-width.json")
    for invalid in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(GraspPlanGenerationError, match="width"):
            generate_simready_grasp_plan(
                asset,
                tmp_path / f"invalid-{invalid}.json",
                width=invalid,
            )
    with pytest.raises(SystemExit):
        generator_module.main([str(asset), "--output", str(tmp_path / "cli.json")])

    assert (
        generator_module.main(
            [
                str(asset),
                "--output",
                str(tmp_path / "cli.json"),
                "--width",
                "0.03",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    plan = SimReadyGraspPlan.model_validate_json(
        (tmp_path / "cli.json").read_text(encoding="ascii")
    )
    assert plan.grasp_lines[0].widths == [0.03]


def test_generated_plan_has_exact_schema_and_is_consumed_by_pr643_authorer(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "asset.usda"
    _write_triangle_asset(asset)
    source_bytes = asset.read_bytes()
    plan_path = tmp_path / "plan.json"
    generated = generate_simready_grasp_plan(asset, plan_path, width=0.04)

    parsed = SimReadyGraspPlan.model_validate_json(
        plan_path.read_text(encoding="ascii")
    )
    provenance = parsed.provenance
    assert parsed == generated.plan
    assert isinstance(provenance, SimReadyGraspPlanMachineProvenance)
    assert provenance.source == "machine_geometry_proof"
    assert provenance.implementation == SIMREADY_GRASP_PLAN_GENERATOR_IMPLEMENTATION
    assert provenance.implementation_version == SIMREADY_GRASP_PLAN_GENERATOR_VERSION
    assert not hasattr(provenance, "approved_by")
    assert not hasattr(provenance, "evidence")

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001"],
            grasp_plan_path=str(plan_path),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["GSP.001"]
    assert asset.read_bytes() == source_bytes
    stage = Usd.Stage.Open(report.output_usd_path)
    assert stage is not None
    curve = UsdGeom.BasisCurves(
        stage.GetPrimAtPath("/robot/grasp_identifier_machine_geometry")
    )
    assert curve
    assert len(curve.GetPointsAttr().Get()) == 2
    assert list(curve.GetWidthsAttr().Get()) == pytest.approx([0.04])
    author_receipt = json.loads(
        Path(report.reports["GSP.001"]).read_text(encoding="utf-8")
    )
    assert author_receipt["provenance"]["source"] == "machine_geometry_proof"
    assert author_receipt["readback_verified"] is True


def test_generated_cube_plan_is_consumed_by_gsp001_authorer(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "cube.usda"
    stage, _root = _new_stage(asset)
    _add_cube(stage, "/robot/body", size=2.0)
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()
    plan_path = tmp_path / "cube-plan.json"
    generate_simready_grasp_plan(asset, plan_path, width=0.04)

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform-cube"),
            repair_requirements=["GSP.001"],
            grasp_plan_path=str(plan_path),
            force=True,
        )
    )

    assert report.passed
    assert report.requirements_repaired == ["GSP.001"]
    assert asset.read_bytes() == source_bytes
    output_stage = Usd.Stage.Open(report.output_usd_path)
    assert output_stage is not None
    curve = UsdGeom.BasisCurves(
        output_stage.GetPrimAtPath("/robot/grasp_identifier_machine_geometry")
    )
    assert curve
    assert list(curve.GetWidthsAttr().Get()) == pytest.approx([0.04])


@pytest.mark.skipif(
    not (
        os.getenv("SIMREADY_FOUNDATION_ROOT")
        and os.getenv("CONTENT_WORKFLOW_SIMREADY_VENV")
    ),
    reason="official SimReady Foundation checkout and venv are not configured",
)
def test_official_foundation_gsp001_clears_after_generation_and_authoring(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_triangle_asset(asset)
    plan_path = tmp_path / "plan.json"
    generate_simready_grasp_plan(asset, plan_path, width=0.01)
    authored = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "conform"),
            repair_requirements=["GSP.001"],
            grasp_plan_path=str(plan_path),
            force=True,
        )
    )
    assert authored.passed

    foundation = run_simready_profile_validation(
        SimReadyValidationInput(
            asset_path=authored.output_usd_path,
            profile="Prop-Robotics-Isaac",
            profile_version="1.0.0",
            report_path=str(tmp_path / "foundation.json"),
            foundation_root=os.environ["SIMREADY_FOUNDATION_ROOT"],
            venv_path=os.environ["CONTENT_WORKFLOW_SIMREADY_VENV"],
            install_missing=False,
        )
    )

    assert foundation.status != "BLOCKED"
    assert foundation.status != "ERROR"
    assert "GSP.001" not in foundation.rerun_reasons
    assert not any(
        str(issue.get("requirement", issue.get("requirement_id", ""))) == "GSP.001"
        for issue in foundation.issues
    )
