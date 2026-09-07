# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration and executable-identity tests for the native PMP hole worker."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from geometry_repair.artifacts import file_sha256
from geometry_repair.hard_mesh_policy import (
    HARD_MESH_CAPABILITIES_SCHEMA,
    HARD_MESH_PROTOCOL,
    PMP_BUILD_ID,
    PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
    PMP_SOURCE_COMMIT,
    PMP_SOURCE_TREE_SHA256,
)
from geometry_repair.mesh_io import load_meshes
from geometry_repair.models import (
    ClassifiedHoleIntent,
    RepairBudgets,
    RepairOperation,
    RepairRequest,
)
from geometry_repair.orchestrator import run_geometry_repair
from geometry_repair.workers.pmp_patch import _PMP_SPEC, PmpPatchWorker


def _operation() -> RepairOperation:
    loop = list(range(5))
    return RepairOperation(
        operation_id="native-pmp-pentagonal-hole",
        worker="pmp_patch",
        implementation="pmp-native-integration-test",
        parameters={
            "operation": "pmp_fill_classified_hole",
            "target_mesh_path": "/Asset/PartA",
            "region_intent": "classified_accidental_hole",
            "intent_evidence_id": "mesh:pentagonal-hole",
            "boundary_loop_vertex_ids": loop,
            "frozen_boundary_vertex_ids": loop,
            "protected_edge_vertex_pairs": [
                [loop[index], loop[(index + 1) % len(loop)]] for index in range(len(loop))
            ],
            "max_loop_perimeter_ratio": 0.25,
            "max_patch_area_ratio": 0.05,
            "max_nonplanarity_ratio": 0.005,
            "max_boundary_turn_radians": math.pi,
            "max_envelope_ratio": 0.001,
            "max_new_vertices": 100,
            "deterministic_seed": 17,
            "timeout_s": 30.0,
        },
        issue_ids=["mesh:pentagonal-hole"],
        drift_band="conservative",
        source_checkpoint="source.usda",
    )


def _write_open_pentagonal_prism(path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    radius = 0.1
    top = [
        Gf.Vec3f(
            radius * math.cos(2.0 * math.pi * index / 5.0),
            radius * math.sin(2.0 * math.pi * index / 5.0),
            10.0,
        )
        for index in range(5)
    ]
    bottom = [Gf.Vec3f(point[0], point[1], 0.0) for point in top]
    points = top + bottom + [Gf.Vec3f(0.0, 0.0, 0.0)]
    faces: list[tuple[int, int, int]] = []
    for index in range(5):
        following = (index + 1) % 5
        bottom_index = 5 + index
        bottom_following = 5 + following
        faces.extend(
            [
                (bottom_index, bottom_following, following),
                (bottom_index, following, index),
                (10, bottom_following, bottom_index),
            ]
        )

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    root.GetPrim().SetCustomDataByKey("phase2Marker", "preserve-me")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/PartA")
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([value for face in faces for value in face]))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    marker = mesh.GetPrim().CreateAttribute("repair:testMarker", Sdf.ValueTypeNames.String)
    marker.Set("source-identity")
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 1.0)] * (len(faces) * 3)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0.25, 0.75)]))
    st.SetIndices(Vt.IntArray([0] * (len(faces) * 3)))
    face_weight = primvars.CreatePrimvar(
        "faceWeight",
        Sdf.ValueTypeNames.FloatArray,
        UsdGeom.Tokens.uniform,
    )
    face_weight.Set(Vt.FloatArray([1.0] * len(faces)))
    subset = UsdGeom.Subset.Define(stage, "/Asset/PartA/SteelFaces")
    subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
    subset.CreateIndicesAttr(Vt.IntArray(range(len(faces))))
    material = UsdShade.Material.Define(stage, "/Asset/Looks/Steel")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    untouched = UsdGeom.Mesh.Define(stage, "/Asset/Untouched")
    untouched.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(2.0, 0.0, 0.0), Gf.Vec3f(3.0, 0.0, 0.0), Gf.Vec3f(2.0, 1.0, 0.0)])
    )
    untouched.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    untouched.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    untouched.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path


def _installed_executable() -> Path:
    configured = os.environ.get("GEOMETRY_REPAIR_PMP_EXECUTABLE")
    discovered = configured or shutil.which("geometry_repair_pmp_patch")
    if not discovered or not Path(discovered).is_file():
        pytest.skip(
            "native PMP helper is not built; run scripts/build_pmp_patch.sh and source pmp_patch.env"
        )
    return Path(discovered).expanduser().resolve()


@pytest.mark.parametrize(
    ("approved_digest", "expected_failure"),
    [
        (None, "approved executable digest is unavailable"),
        ("f" * 64, "digest mismatch"),
    ],
)
def test_environment_override_requires_matching_independent_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approved_digest: str | None,
    expected_failure: str,
) -> None:
    executable = tmp_path / "fake-pmp"
    capability = {
        "schema_version": HARD_MESH_CAPABILITIES_SCHEMA,
        "protocol_version": HARD_MESH_PROTOCOL,
        "worker": "pmp_patch",
        "implementation_version": "forged-self-report",
        "build_id": PMP_BUILD_ID,
        "operations": sorted(_PMP_SPEC.operations),
        "capabilities": sorted(_PMP_SPEC.required_capabilities),
        "deterministic": True,
        "backend_source_commit": PMP_SOURCE_COMMIT,
        "backend_source_tree_sha256": PMP_SOURCE_TREE_SHA256,
        "executable_sha256": "0" * 64,
    }
    executable.write_text(
        "#!/bin/sh\nprintf '%s\\n' '" + json.dumps(capability, sort_keys=True) + "'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("GEOMETRY_REPAIR_PMP_EXECUTABLE", str(executable))
    if approved_digest is None:
        monkeypatch.delenv(PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE, approved_digest)

    available, reason = PmpPatchWorker().available()

    assert available is False
    assert expected_failure in (reason or "")


def test_native_pmp_fills_only_named_five_vertex_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    executable = _installed_executable()
    executable_sha256 = file_sha256(executable)
    monkeypatch.setenv("GEOMETRY_REPAIR_PMP_EXECUTABLE", str(executable))
    monkeypatch.setenv(PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE, executable_sha256)
    source = _write_open_pentagonal_prism(tmp_path / "source.usda")
    output = tmp_path / "repaired.usda"
    source_meshes, _ = load_meshes(source)
    source_target = next(mesh for mesh in source_meshes if mesh.path == "/Asset/PartA")
    source_untouched = next(mesh for mesh in source_meshes if mesh.path == "/Asset/Untouched")

    result = PmpPatchWorker().execute_typed(
        source=source,
        output=output,
        operation=_operation(),
    )

    assert result.status == "success", result.failures
    assert result.output_path == str(output.resolve())
    assert result.output_sha256 == file_sha256(output)
    assert result.report is not None
    assert result.report.executable_sha256 == executable_sha256
    assert result.report.backend_source_commit == PMP_SOURCE_COMMIT
    assert result.report.backend_source_tree_sha256 == PMP_SOURCE_TREE_SHA256
    assert result.metadata["approved_executable_sha256"] == executable_sha256
    assert result.report.correspondence_coverage_ratio == 1.0
    assert result.report.preserved_frozen_vertices is True
    assert result.report.preserved_protected_edges is True

    output_meshes, _ = load_meshes(output)
    output_target = next(mesh for mesh in output_meshes if mesh.path == "/Asset/PartA")
    output_untouched = next(mesh for mesh in output_meshes if mesh.path == "/Asset/Untouched")
    assert len(output_target.triangles) > len(source_target.triangles)
    np.testing.assert_allclose(
        output_target.local_vertices[: len(source_target.local_vertices)],
        source_target.local_vertices,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        output_target.triangles[: len(source_target.triangles)], source_target.triangles
    )
    assert result.report.changed_face_ids == list(
        range(len(source_target.triangles), len(output_target.triangles))
    )
    assert result.report.generated_face_ids == result.report.changed_face_ids
    np.testing.assert_array_equal(output_untouched.local_vertices, source_untouched.local_vertices)
    np.testing.assert_array_equal(output_untouched.triangles, source_untouched.triangles)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath().pathString == "/Asset"
    assert stage.GetPrimAtPath("/Asset").GetCustomDataByKey("phase2Marker") == "preserve-me"
    target_prim = stage.GetPrimAtPath("/Asset/PartA")
    assert target_prim.GetAttribute("repair:testMarker").Get() == "source-identity"
    output_mesh = UsdGeom.Mesh(target_prim)
    output_face_count = len(output_mesh.GetFaceVertexCountsAttr().Get())
    assert len(output_mesh.GetNormalsAttr().Get()) == output_face_count * 3
    output_primvars = UsdGeom.PrimvarsAPI(target_prim)
    assert len(output_primvars.GetPrimvar("st").GetIndices()) == output_face_count * 3
    assert len(output_primvars.GetPrimvar("faceWeight").Get()) == output_face_count
    output_subset = UsdGeom.Subset.GetAllGeomSubsets(output_mesh)[0]
    assert list(output_subset.GetIndicesAttr().Get()) == list(range(output_face_count))
    bound_material, _relationship = UsdShade.MaterialBindingAPI(target_prim).ComputeBoundMaterial()
    assert bound_material.GetPath().pathString == "/Asset/Looks/Steel"

    correspondence = json.loads(
        Path(result.report.correspondence_path or "").read_text(encoding="utf-8")
    )
    assert correspondence["source_faces"] == [
        {"output_face_id": index, "source_face_id": index}
        for index in range(len(source_target.triangles))
    ]


def test_native_pmp_does_not_follow_predictable_temporary_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _installed_executable()
    executable_sha256 = file_sha256(executable)
    monkeypatch.setenv("GEOMETRY_REPAIR_PMP_EXECUTABLE", str(executable))
    monkeypatch.setenv(PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE, executable_sha256)
    source = _write_open_pentagonal_prism(tmp_path / "source.usda")
    output = tmp_path / "repaired.usda"
    bridge_dir = output.parent / f"{output.name}.pmp_bridge"
    bridge_dir.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("preserve-me", encoding="utf-8")
    (bridge_dir / "candidate.obj.tmp").symlink_to(victim)

    result = PmpPatchWorker().execute_typed(
        source=source,
        output=output,
        operation=_operation(),
    )

    assert result.status == "success", result.failures
    assert victim.read_text(encoding="utf-8") == "preserve-me"
    assert (bridge_dir / "candidate.obj.tmp").is_symlink()


def test_orchestrator_routes_explicit_five_vertex_hole_to_native_pmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pxr import Usd

    executable = _installed_executable()
    monkeypatch.setenv("GEOMETRY_REPAIR_PMP_EXECUTABLE", str(executable))
    monkeypatch.setenv(
        PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE,
        file_sha256(executable),
    )
    source = _write_open_pentagonal_prism(tmp_path / "orchestrated_source.usda")
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    stage.RemovePrim("/Asset/Untouched")
    stage.GetRootLayer().Save()

    result = run_geometry_repair(
        RepairRequest(
            source_path=source,
            output_dir=tmp_path / "repair",
            profile="rigid_pick_place",
            mode="auto",
            classified_holes=[
                ClassifiedHoleIntent(
                    target_mesh_path="/Asset/PartA",
                    intent_evidence_id="operator:classified-top-cap",
                    confirmed_accidental_hole=True,
                    boundary_loop_vertex_ids=list(range(5)),
                )
            ],
            budgets=RepairBudgets(
                max_attempts=4,
                timeout_s=180.0,
                audit_wall_time_s=20.0,
            ),
            enabled_workers=[
                "manifold_restore_merge_vectors",
                "pmp_patch",
                "coacd_collision",
            ],
        )
    )

    pmp_attempt = next(
        attempt for attempt in result.attempts if attempt.operation.worker == "pmp_patch"
    )
    assert pmp_attempt.status == "accepted", pmp_attempt.reasons
    assert pmp_attempt.correspondence_path is not None
    correspondence = json.loads(Path(pmp_attempt.correspondence_path).read_text(encoding="utf-8"))
    assert correspondence["schema_version"] == "geometry-repair.correspondence.v2"
    assert correspondence["status"] in {"pass", "conditional"}
    assert correspondence["generated_output_regions"]["faces"]
    assert result.outcome in {"certified", "conditional"}
    assert result.render_usd_path is not None
    assert result.collision_usd_path is not None
