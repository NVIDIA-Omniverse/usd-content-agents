# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact-plan Gate 3A primvar and mesh-topology authoring."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt  # noqa: E402

import content_agent_workflows.simready.gate3a_mesh_topology as topology_module  # noqa: E402
from content_agent_workflows.simready.gate3a_mesh_topology import (  # noqa: E402
    GATE3A_MESH_TOPOLOGY_PLAN_SCHEMA_VERSION,
    Gate3AMeshTopologyPlan,
    author_gate3a_mesh_topology_derivative,
)

_MESH_PATH = "/World/Body/Collider"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_usdz(root_layer_path: Path, output_path: Path) -> Path:
    os.utime(root_layer_path, (315532800, 315532800))
    assert UsdUtils.CreateNewUsdzPackage(str(root_layer_path), str(output_path))
    return output_path.resolve()


def _base_stage(path: Path) -> tuple[Any, Any]:
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.ArticulationRootAPI.Apply(world)
    body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body).CreateMassAttr(2.0)
    other = UsdGeom.Xform.Define(stage, "/World/Other").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(other)
    UsdPhysics.MassAPI.Apply(other).CreateMassAttr(1.0)
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Fixed")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Body")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Other")])
    return stage, body


def _bind_material(stage: Any, prim: Any) -> str:
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/TestMaterial/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material)
    assert looks
    return str(material.GetPath())


def _author_common_mesh_data(
    mesh: Any,
    *,
    points: Any,
    counts: list[int],
    indices: list[int],
) -> None:
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(Vt.Vec3fArray(points)))
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(Vt.Vec3fArray([(0.0, 0.0, 1.0)] * len(indices)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    primvars = UsdGeom.PrimvarsAPI(mesh)
    color = primvars.CreatePrimvar(
        "displayColor",
        Sdf.ValueTypeNames.Color3fArray,
        UsdGeom.Tokens.constant,
    )
    color.Set(Vt.Vec3fArray([(0.25, 0.5, 0.75)]))
    st = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set(Vt.Vec2fArray([(float(index), 0.5) for index in range(len(indices))]))
    st.SetIndices(Vt.IntArray(range(len(indices))))


def _write_topology_asset(
    tmp_path: Path,
    *,
    rigid_on_mesh: bool = False,
    joint_targets_mesh: bool = False,
    point_domain_conflict: bool = False,
    time_sampled: bool = False,
    ambiguous_custom_data: bool = False,
    variant_on_parent: bool = False,
) -> Path:
    root = tmp_path / "topology.usda"
    stage, _body = _base_stage(root)
    if variant_on_parent:
        stage.GetPrimAtPath("/World/Body").GetVariantSets().AddVariantSet("bad")
    mesh = UsdGeom.Mesh.Define(stage, _MESH_PATH)
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (9.0, 9.0, 9.0),
    ]
    counts = [3, 3, 3]
    indices = [0, 1, 2, 1, 5, 3, 5, 1, 4]
    _author_common_mesh_data(
        mesh,
        points=points,
        counts=counts,
        indices=indices,
    )
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
    if rigid_on_mesh:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)
    if joint_targets_mesh:
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/World/Fixed"))
        joint.GetBody0Rel().SetTargets([Sdf.Path(_MESH_PATH)])
    mesh.AddTranslateOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.inherited)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.default_)
    _bind_material(stage, prim)
    prim.CreateAttribute(
        "smoothgroups3DSMax",
        Sdf.ValueTypeNames.UIntArray,
        custom=True,
    ).Set(Vt.UIntArray([11, 22, 33]))
    prim.CreateAttribute("test:scalar", Sdf.ValueTypeNames.String, custom=True).Set(
        "sentinel"
    )
    if point_domain_conflict:
        velocities = mesh.CreateVelocitiesAttr(
            Vt.Vec3fArray([(0.0, 0.0, 0.0)] * len(points))
        )
        values = list(velocities.Get())
        values[5] = Gf.Vec3f(1.0, 0.0, 0.0)
        velocities.Set(Vt.Vec3fArray(values))
    if time_sampled:
        mesh.GetPointsAttr().Set(mesh.GetPointsAttr().Get(), Usd.TimeCode(1.0))
    if ambiguous_custom_data:
        prim.CreateAttribute(
            "test:ambiguous",
            Sdf.ValueTypeNames.IntArray,
            custom=True,
        ).Set(Vt.IntArray([1, 2]))
    assert stage.GetRootLayer().Save()
    stage = None
    gc.collect()
    return _write_usdz(root, tmp_path / "topology.usdz")


def _write_primvar_asset(tmp_path: Path) -> tuple[Path, list[list[float]]]:
    root = tmp_path / "primvar.usda"
    stage, _body = _base_stage(root)
    mesh = UsdGeom.Mesh.Define(stage, _MESH_PATH)
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    counts = [3, 3]
    indices = [0, 1, 2, 0, 2, 3]
    _author_common_mesh_data(mesh, points=points, counts=counts, indices=indices)
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    values = Vt.Vec2fArray([(0.0, 0.0), (1.0, 0.0), (9.0, 9.0), (1.0, 1.0)])
    primvar.Set(values)
    primvar.SetIndices(Vt.IntArray([0, 1, 3, 0, 3, 1]))
    flattened = [
        [float(value[0]), float(value[1])] for value in primvar.ComputeFlattened()
    ]
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    assert stage.GetRootLayer().Save()
    stage = None
    gc.collect()
    return _write_usdz(root, tmp_path / "primvar.usdz"), flattened


def _write_evidence(
    tmp_path: Path, *, source_sha256: str, include_subject: bool = True
) -> Path:
    evidence = tmp_path / "gate3a-results.json"
    evidence.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "artifact_sha256": (
                            source_sha256 if include_subject else "0" * 64
                        ),
                        "issues": [
                            {
                                "rule": "MeshTopologyChecker",
                                "severity": "WARNING",
                            }
                        ],
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return evidence.resolve()


def _plan_payload(
    *,
    asset_path: Path,
    evidence_path: Path,
    primvar_compactions: list[dict[str, Any]] | None = None,
    mesh_normalizations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_sha256 = _sha256(asset_path)
    return {
        "schema_version": GATE3A_MESH_TOPOLOGY_PLAN_SCHEMA_VERSION,
        "source_asset_path": str(asset_path.resolve()),
        "source_asset_sha256": source_sha256,
        "provenance": {
            "approved_by": "asset-owner",
            "approval_reference": "https://example.com/approvals/issue-529",
            "evidence": [
                {
                    "kind": "gate3a_validation",
                    "artifact_path": str(evidence_path.resolve()),
                    "artifact_sha256": _sha256(evidence_path),
                    "subject_asset_sha256": source_sha256,
                }
            ],
        },
        "primvar_compactions": primvar_compactions or [],
        "mesh_normalizations": mesh_normalizations or [],
    }


def _topology_operation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prim_path": _MESH_PATH,
        "expected_point_count": 7,
        "expected_exact_unique_point_count": 6,
        "expected_unused_point_count": 1,
        "expected_face_count": 3,
        "expected_face_vertex_index_count": 9,
        "expected_nonmanifold_edge_count": 1,
        "expected_output_part_count": 3,
    }
    payload.update(overrides)
    return payload


def _primvar_operation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prim_path": _MESH_PATH,
        "attribute_name": "primvars:st",
        "expected_value_count": 4,
        "expected_index_count": 6,
        "expected_referenced_value_count": 3,
    }
    payload.update(overrides)
    return payload


def _write_plan(
    tmp_path: Path,
    *,
    payload: dict[str, Any],
    name: str = "plan.json",
) -> Path:
    plan = tmp_path / name
    plan.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan.resolve()


def _topology_plan(
    tmp_path: Path, asset: Path, **operation_overrides: Any
) -> tuple[Path, Path]:
    evidence = _write_evidence(tmp_path, source_sha256=_sha256(asset))
    payload = _plan_payload(
        asset_path=asset,
        evidence_path=evidence,
        mesh_normalizations=[_topology_operation(**operation_overrides)],
    )
    return _write_plan(tmp_path, payload=payload), evidence


def _primvar_plan(
    tmp_path: Path, asset: Path, **operation_overrides: Any
) -> tuple[Path, Path]:
    evidence = _write_evidence(tmp_path, source_sha256=_sha256(asset))
    payload = _plan_payload(
        asset_path=asset,
        evidence_path=evidence,
        primvar_compactions=[_primvar_operation(**operation_overrides)],
    )
    return _write_plan(tmp_path, payload=payload), evidence


def test_compacts_only_referenced_primvar_values_and_is_deterministic(
    tmp_path: Path,
) -> None:
    asset, expected_flattened = _write_primvar_asset(tmp_path)
    source_bytes = asset.read_bytes()
    plan, _evidence = _primvar_plan(tmp_path, asset)

    first = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "first",
    )
    second = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "second",
    )

    assert first.passed and second.passed
    assert _sha256(first.output_path) == _sha256(second.output_path)
    assert asset.read_bytes() == source_bytes
    stage = Usd.Stage.Open(str(first.output_path))
    primvar = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(_MESH_PATH)).GetPrimvar("st")
    assert len(primvar.Get()) == 3
    assert list(primvar.GetIndices()) == [0, 1, 2, 0, 2, 1]
    assert [
        [float(v[0]), float(v[1])] for v in primvar.ComputeFlattened()
    ] == expected_flattened
    receipt = json.loads(first.receipt_path.read_text(encoding="utf-8"))
    assert receipt["invariants"]["source_bytes_preserved"] is True
    assert receipt["primvar_proofs"][0]["output_value_count"] == 3


def test_normalizes_collision_mesh_into_ordered_manifold_parts(tmp_path: Path) -> None:
    asset = _write_topology_asset(tmp_path)
    source_bytes = asset.read_bytes()
    plan, _evidence = _topology_plan(tmp_path, asset)

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert result.passed, result.reason
    assert asset.read_bytes() == source_bytes
    stage = Usd.Stage.Open(str(result.output_path))
    parent = stage.GetPrimAtPath(_MESH_PATH)
    assert parent.IsA(UsdGeom.Xform)
    assert not parent.HasAPI(UsdPhysics.CollisionAPI)
    assert parent.GetAttribute("test:scalar").Get() == "sentinel"
    children = parent.GetChildren()
    assert [child.GetName() for child in children] == [
        "MeshPart_0000",
        "MeshPart_0001",
        "MeshPart_0002",
    ]
    for index, child in enumerate(children):
        assert child.IsA(UsdGeom.Mesh)
        assert child.HasAPI(UsdPhysics.CollisionAPI)
        assert child.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert not child.HasAPI(UsdPhysics.RigidBodyAPI)
        mesh = UsdGeom.Mesh(child)
        assert len(mesh.GetPointsAttr().Get()) == 3
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3]
        assert set(mesh.GetFaceVertexIndicesAttr().Get()) == {0, 1, 2}
        assert list(child.GetAttribute("smoothgroups3DSMax").Get()) == [
            [11, 22, 33][index]
        ]
        material = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial()[0]
        assert str(material.GetPath()) == "/World/Looks/TestMaterial"
        st = UsdGeom.PrimvarsAPI(child).GetPrimvar("st")
        assert len(st.Get()) == 3
        assert len(st.ComputeFlattened()) == 3
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    proof = receipt["mesh_proofs"][0]
    assert proof["source_nonmanifold_edge_count"] == 1
    assert proof["output_part_count"] == 3
    assert receipt["invariants"]["world_triangle_multisets_preserved"] is True


def test_split_component_does_not_retain_winding_inconsistent_normals(
    tmp_path: Path,
) -> None:
    asset = _write_topology_asset(tmp_path)
    plan, _evidence = _topology_plan(tmp_path, asset)

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert result.passed, result.reason
    stage = Usd.Stage.Open(str(result.output_path))
    first = UsdGeom.Mesh(stage.GetPrimAtPath(f"{_MESH_PATH}/MeshPart_0000"))
    second = UsdGeom.Mesh(stage.GetPrimAtPath(f"{_MESH_PATH}/MeshPart_0001"))
    inconsistent = UsdGeom.Mesh(stage.GetPrimAtPath(f"{_MESH_PATH}/MeshPart_0002"))
    assert first.GetNormalsAttr().HasAuthoredValueOpinion()
    assert second.GetNormalsAttr().HasAuthoredValueOpinion()
    assert inconsistent.GetNormalsAttr().HasAuthoredValueOpinion()
    assert inconsistent.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    assert {
        tuple(round(float(component), 6) for component in normal)
        for normal in inconsistent.GetNormalsAttr().Get()
    } == {(0.0, -1.0, 0.0)}
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    mesh_proof = receipt["mesh_proofs"][0]
    assert mesh_proof["preserved_normal_part_paths"] == [
        f"{_MESH_PATH}/MeshPart_0000",
        f"{_MESH_PATH}/MeshPart_0001",
    ]
    assert mesh_proof["omitted_normal_part_paths"] == [f"{_MESH_PATH}/MeshPart_0002"]
    assert mesh_proof["derived_normal_part_paths"] == [f"{_MESH_PATH}/MeshPart_0002"]


def test_reuses_identical_atomic_bundle(tmp_path: Path) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    plan, _evidence = _primvar_plan(tmp_path, asset)

    first = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )
    second = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert first.passed and second.passed
    assert first.output_path == second.output_path
    assert second.report["reused_output"] is True
    assert sorted(path.name for path in second.output_path.parent.iterdir()) == sorted(
        ["primvar.gate3a-mesh-topology.usdz", "receipt.json"]
    )


def test_plan_requires_owner_evidence_and_operations(tmp_path: Path) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    evidence = _write_evidence(tmp_path, source_sha256=_sha256(asset))
    payload = _plan_payload(asset_path=asset, evidence_path=evidence)
    payload["provenance"]["evidence"] = []

    with pytest.raises(ValidationError):
        Gate3AMeshTopologyPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_point_count", 8),
        ("expected_exact_unique_point_count", 5),
        ("expected_unused_point_count", 0),
        ("expected_face_count", 4),
        ("expected_face_vertex_index_count", 10),
        ("expected_nonmanifold_edge_count", 0),
        ("expected_output_part_count", 2),
    ],
)
def test_blocks_stale_mesh_characterization(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    asset = _write_topology_asset(tmp_path)
    plan, _evidence = _topology_plan(tmp_path, asset, **{field: value})

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "stale_mesh_plan" in result.reason


def test_blocks_stale_primvar_characterization(tmp_path: Path) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    plan, _evidence = _primvar_plan(tmp_path, asset, expected_value_count=5)

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "stale_primvar_plan" in result.reason


def test_blocks_evidence_without_exact_source_result(tmp_path: Path) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    evidence = _write_evidence(
        tmp_path,
        source_sha256=_sha256(asset),
        include_subject=False,
    )
    payload = _plan_payload(
        asset_path=asset,
        evidence_path=evidence,
        primvar_compactions=[_primvar_operation()],
    )
    plan = _write_plan(tmp_path, payload=payload)

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "invalid_gate3a_evidence" in result.reason


def test_blocks_stale_source_sha256(tmp_path: Path) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    plan, _evidence = _primvar_plan(tmp_path, asset)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["source_asset_sha256"] = "0" * 64
    payload["provenance"]["evidence"][0]["subject_asset_sha256"] = "0" * 64
    plan.write_text(json.dumps(payload), encoding="utf-8")

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "stale_source_sha256" in result.reason


@pytest.mark.parametrize(
    ("asset_options", "reason"),
    [
        ({"rigid_on_mesh": True}, "forbidden_physics_ownership"),
        ({"joint_targets_mesh": True}, "joint_endpoint_target"),
        ({"time_sampled": True}, "time_varying_input"),
        ({"point_domain_conflict": True}, "point_domain_conflict"),
        ({"ambiguous_custom_data": True}, "ambiguous_custom_data"),
        ({"variant_on_parent": True}, "unsupported_variants"),
    ],
)
def test_blocks_unsafe_mesh_semantics(
    tmp_path: Path,
    asset_options: dict[str, bool],
    reason: str,
) -> None:
    asset = _write_topology_asset(tmp_path, **asset_options)
    plan, _evidence = _topology_plan(tmp_path, asset)

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert reason in result.reason


def test_blocks_compressed_usdz_before_extracting(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    stage, _body = _base_stage(source)
    assert stage.GetRootLayer().Save()
    stage = None
    gc.collect()
    package = tmp_path / "compressed.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname="source.usda")
    evidence = _write_evidence(tmp_path, source_sha256=_sha256(package))
    payload = _plan_payload(
        asset_path=package,
        evidence_path=evidence,
        primvar_compactions=[_primvar_operation()],
    )
    plan = _write_plan(tmp_path, payload=payload)

    result = author_gate3a_mesh_topology_derivative(
        asset_path=package,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "compressed USDZ entry" in result.reason


def test_detects_source_mutation_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    plan, _evidence = _primvar_plan(tmp_path, asset)
    original_publish = topology_module._publish_bundle

    def mutate_then_publish(**kwargs: Any):
        asset.write_bytes(asset.read_bytes() + b"changed")
        return original_publish(**kwargs)

    monkeypatch.setattr(topology_module, "_publish_bundle", mutate_then_publish)
    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "input_changed" in result.reason
    publish_root = tmp_path / "output" / "gate3a-mesh-topology"
    assert not list(publish_root.iterdir())


def test_rejects_duplicate_plan_json_key(tmp_path: Path) -> None:
    asset, _flattened = _write_primvar_asset(tmp_path)
    plan, _evidence = _primvar_plan(tmp_path, asset)
    raw = plan.read_text(encoding="utf-8")
    plan.write_text(
        raw.replace("{", '{"schema_version":"duplicate",', 1), encoding="utf-8"
    )

    result = author_gate3a_mesh_topology_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path / "output",
    )

    assert not result.passed
    assert "duplicate JSON key" in result.reason
