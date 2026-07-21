# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial tests for exact physics-profile plan authoring."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import content_agent_workflows.simready.physics_profile_plan as plan_module
from content_agent_workflows.simready.physics_profile_plan import (
    PHYSICS_PROFILE_PLAN_SCHEMA_VERSION,
    PhysicsProfilePlanError,
    author_physics_profile_plan,
    inspect_physics_profile_source,
    load_physics_profile_plan,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_asset(
    path: Path,
    *,
    collider_paths: tuple[str, ...] = ("/Root/Collider",),
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
    for collider_path in collider_paths:
        mesh = UsdGeom.Mesh.Define(stage, collider_path)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        assert UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    assert stage.GetRootLayer().Save()
    del stage


def _write_mixed_gprim_asset(path: Path) -> tuple[str, ...]:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
    cube = UsdGeom.Cube.Define(stage, "/Root/Cube")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    assert UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    assert UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    assert stage.GetRootLayer().Save()
    del stage
    return ("/Root/Cube", "/Root/Mesh")


def _write_instance_proxy_asset(path: Path) -> Path:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    prototype_path = path.with_name(f"{path.stem}-prototype.usda")
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    prototype_root = UsdGeom.Xform.Define(prototype_stage, "/Prototype")
    prototype_stage.SetDefaultPrim(prototype_root.GetPrim())
    prototype_mesh = UsdGeom.Mesh.Define(prototype_stage, "/Prototype/Collider")
    prototype_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    prototype_mesh.CreateFaceVertexCountsAttr([3])
    prototype_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    assert UsdPhysics.CollisionAPI.Apply(prototype_mesh.GetPrim())
    assert prototype_stage.GetRootLayer().Save()
    del prototype_stage

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
    ordinary = UsdGeom.Mesh.Define(stage, "/Root/Ordinary")
    ordinary.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    ordinary.CreateFaceVertexCountsAttr([3])
    ordinary.CreateFaceVertexIndicesAttr([0, 1, 2])
    assert UsdPhysics.CollisionAPI.Apply(ordinary.GetPrim())
    instance = UsdGeom.Xform.Define(stage, "/Root/Instance").GetPrim()
    assert instance.GetReferences().AddReference(prototype_path.name)
    assert instance.SetInstanceable(True)
    assert stage.GetRootLayer().Save()
    del stage
    return prototype_path


def _apply_proxy_inventory_state(stage: Any, conflict: str) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    host = UsdGeom.Xform.Define(stage, "/Prototype/BindingHost").GetPrim()
    if conflict == "physics_material":
        material = UsdShade.Material.Define(
            stage,
            "/Prototype/ExistingPhysicsMaterial",
        )
        assert UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    elif conflict == "direct_binding":
        relationship = host.CreateRelationship(
            "material:binding:physics",
            custom=False,
        )
        assert relationship.SetTargets([Sdf.Path("/Prototype/BindingTarget")])
    elif conflict == "collection_binding":
        relationship = host.CreateRelationship(
            "material:binding:collection:physics:existing",
            custom=False,
        )
        assert relationship.SetTargets(
            [
                Sdf.Path("/Prototype/BindingHost"),
                Sdf.Path("/Prototype/BindingTarget"),
            ]
        )
    else:  # pragma: no cover - test helper invariant
        raise AssertionError(f"Unsupported proxy inventory conflict: {conflict}")


def _write_instance_proxy_inventory_asset(
    path: Path,
    *,
    conflict: str | None = None,
) -> Path:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    prototype_path = path.with_name(f"{path.stem}-inventory-prototype.usda")
    prototype_stage = Usd.Stage.CreateNew(str(prototype_path))
    prototype_root = UsdGeom.Xform.Define(prototype_stage, "/Prototype")
    prototype_stage.SetDefaultPrim(prototype_root.GetPrim())
    UsdGeom.Xform.Define(prototype_stage, "/Prototype/BindingHost")
    if conflict is not None:
        _apply_proxy_inventory_state(prototype_stage, conflict)
    assert prototype_stage.GetRootLayer().Save()
    del prototype_stage

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
    ordinary = UsdGeom.Mesh.Define(stage, "/Root/Ordinary")
    ordinary.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    ordinary.CreateFaceVertexCountsAttr([3])
    ordinary.CreateFaceVertexIndicesAttr([0, 1, 2])
    assert UsdPhysics.CollisionAPI.Apply(ordinary.GetPrim())
    instance = UsdGeom.Xform.Define(stage, "/Root/Instance").GetPrim()
    assert instance.GetReferences().AddReference(prototype_path.name)
    assert instance.SetInstanceable(True)
    assert stage.GetRootLayer().Save()
    del stage
    return prototype_path


def _add_proxy_inventory_state(path: Path, conflict: str) -> None:
    Usd = pytest.importorskip("pxr.Usd")

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    assert stage is not None
    _apply_proxy_inventory_state(stage, conflict)
    assert stage.GetRootLayer().Save()
    del stage


def _plan_payload(
    asset: Path,
    *,
    collider_paths: tuple[str, ...] = ("/Root/Collider",),
    density: float | None = 875.0,
    source_dependency_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    if source_dependency_bundle_sha256 is None:
        source_dependency_bundle_sha256 = inspect_physics_profile_source(
            asset
        ).source_dependency_bundle_sha256
    return {
        "schema_version": PHYSICS_PROFILE_PLAN_SCHEMA_VERSION,
        "source_asset_sha256": _file_sha256(asset),
        "source_dependency_bundle_sha256": source_dependency_bundle_sha256,
        "collider_prim_paths": list(collider_paths),
        "mesh_approximation": "sdf",
        "physics_material": {
            "prim_path": "/Root/PhysicsMaterials/OwnerApproved",
            "static_friction": 0.75,
            "dynamic_friction": 0.5,
            "restitution": 0.125,
            "density": density,
        },
        "approval": {
            "approved": True,
            "owner_identity": "asset-owner@example.test",
            "evidence": "approval-record-sha256:0123456789abcdef",
        },
    }


def _write_plan(
    path: Path,
    asset: Path,
    *,
    collider_paths: tuple[str, ...] = ("/Root/Collider",),
    density: float | None = 875.0,
    source_dependency_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    payload = _plan_payload(
        asset,
        collider_paths=collider_paths,
        density=density,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_policy_asset(
    path: Path,
    *,
    collider_paths: tuple[str, ...],
    approximation_tokens: dict[str, str] | None = None,
    with_existing_material: bool = False,
    binding_host_paths: tuple[str, ...] = (),
    direct_bound_paths: tuple[str, ...] = (),
    material_values: dict[str, float | None] | None = None,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    UsdUtils = pytest.importorskip("pxr.UsdUtils")

    source = path
    if path.suffix == ".usdz":
        source_tree = path.parent / f"{path.stem}-policy-source"
        source_tree.mkdir()
        source = source_tree / "root.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
    for host_path in binding_host_paths:
        UsdGeom.Xform.Define(stage, host_path)
    for collider_path in collider_paths:
        mesh = UsdGeom.Mesh.Define(stage, collider_path)
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        assert UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        token = (approximation_tokens or {}).get(collider_path)
        if token is not None:
            approximation = UsdPhysics.MeshCollisionAPI.Apply(
                mesh.GetPrim()
            ).CreateApproximationAttr()
            assert approximation.Set(token)

    if with_existing_material:
        material = UsdShade.Material.Define(
            stage,
            "/Root/PhysicsMaterials/Existing",
        )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        values = material_values or {
            "static_friction": 0.5,
            "dynamic_friction": 0.4,
            "restitution": 0.15,
            "density": None,
        }
        assert material_api.CreateStaticFrictionAttr().Set(values["static_friction"])
        assert material_api.CreateDynamicFrictionAttr().Set(values["dynamic_friction"])
        assert material_api.CreateRestitutionAttr().Set(values["restitution"])
        if values["density"] is not None:
            assert material_api.CreateDensityAttr().Set(values["density"])
        for host_path in binding_host_paths:
            assert UsdShade.MaterialBindingAPI.Apply(
                stage.GetPrimAtPath(host_path)
            ).Bind(material, materialPurpose="physics")
        for collider_path in direct_bound_paths:
            assert UsdShade.MaterialBindingAPI.Apply(
                stage.GetPrimAtPath(collider_path)
            ).Bind(material, materialPurpose="physics")
    assert stage.GetRootLayer().Save()
    del stage
    if path.suffix == ".usdz":
        assert UsdUtils.CreateNewUsdzPackage(str(source), str(path))


def _policy_plan_payload(
    asset: Path,
    *,
    collider_paths: tuple[str, ...],
    material_operation: str = "create",
    bind_missing_collider_paths: tuple[str, ...] = (),
    material_values: dict[str, float | None] | None = None,
    source_authored_opinions: dict[str, bool] | None = None,
    collider_approximations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = material_values or {
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.15,
        "density": None,
    }
    material_path = (
        "/Root/PhysicsMaterials/Existing"
        if material_operation == "reuse_existing"
        else "/Root/PhysicsMaterials/OwnerApproved"
    )
    material: dict[str, Any] = {
        "prim_path": material_path,
        **values,
    }
    if material_operation != "create":
        material.update(
            {
                "operation": material_operation,
                "source_authored_opinions": source_authored_opinions
                or {
                    "static_friction": True,
                    "dynamic_friction": True,
                    "restitution": True,
                    "density": values["density"] is not None,
                },
                "bind_missing_collider_paths": list(bind_missing_collider_paths),
            }
        )
    identity = inspect_physics_profile_source(asset)
    payload: dict[str, Any] = {
        "schema_version": PHYSICS_PROFILE_PLAN_SCHEMA_VERSION,
        "source_asset_sha256": identity.source_asset_sha256,
        "source_dependency_bundle_sha256": identity.source_dependency_bundle_sha256,
        "collider_prim_paths": list(collider_paths),
        "mesh_approximation": "sdf",
        "physics_material": material,
        "approval": {
            "approved": True,
            "owner_identity": "asset-owner@example.test",
            "evidence": "approval-record-sha256:reuse-existing-policy",
        },
    }
    if collider_approximations is not None:
        payload["collider_approximations"] = collider_approximations
    return payload


def _write_policy_plan(
    path: Path,
    asset: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = _policy_plan_payload(asset, **kwargs)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _open_stage(path: Path) -> Any:
    Usd = pytest.importorskip("pxr.Usd")
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    assert stage is not None
    return stage


def test_authors_exact_profile_receipt_and_preserves_source(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    _write_plan(plan_path, asset)
    source_bytes = asset.read_bytes()

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    assert asset.read_bytes() == source_bytes
    assert result.output_asset_path.parent.name == "artifact"
    assert result.output_asset_path.parent.parent.name == (
        f"physics-profile-{result.receipt.output_artifact_sha256}"
    )
    assert (
        result.receipt.source_asset_sha256 == hashlib.sha256(source_bytes).hexdigest()
    )
    assert result.receipt.output_asset_sha256 == _file_sha256(result.output_asset_path)
    _plan, plan_sha256 = load_physics_profile_plan(plan_path)
    assert result.receipt.plan_sha256 == plan_sha256
    assert (
        result.receipt.source_dependency_bundle_sha256
        == _plan.source_dependency_bundle_sha256
    )
    assert result.receipt_path.read_bytes().endswith(b"\n")
    assert b'": ' not in result.receipt_path.read_bytes()

    stage = _open_stage(result.output_asset_path)
    UsdShade = pytest.importorskip("pxr.UsdShade")
    material = stage.GetPrimAtPath("/Root/PhysicsMaterials/OwnerApproved")
    collider = stage.GetPrimAtPath("/Root/Collider")
    assert material
    assert collider.HasAPI(UsdShade.MaterialBindingAPI)
    assert collider.GetAttribute("physics:approximation").Get() == "sdf"
    assert collider.GetRelationship("material:binding:physics").GetTargets() == [
        material.GetPath()
    ]
    applied = collider.GetMetadata("apiSchemas").GetAppliedItems()
    assert "PhysxCollisionAPI" in applied
    assert "PhysxSDFMeshCollisionAPI" in applied


def test_authors_exact_profile_for_mesh_and_analytic_gprim_colliders(
    tmp_path: Path,
) -> None:
    UsdShade = pytest.importorskip("pxr.UsdShade")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    collider_paths = _write_mixed_gprim_asset(asset)
    _write_plan(plan_path, asset, collider_paths=collider_paths)
    source_bytes = asset.read_bytes()

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    assert asset.read_bytes() == source_bytes
    stage = _open_stage(result.output_asset_path)
    material = stage.GetPrimAtPath("/Root/PhysicsMaterials/OwnerApproved")
    cube = stage.GetPrimAtPath("/Root/Cube")
    mesh = stage.GetPrimAtPath("/Root/Mesh")
    for collider in (cube, mesh):
        assert collider.HasAPI(UsdShade.MaterialBindingAPI)
        assert collider.GetRelationship("material:binding:physics").GetTargets() == [
            material.GetPath()
        ]
        assert (
            "PhysxCollisionAPI" in collider.GetMetadata("apiSchemas").GetAppliedItems()
        )
    cube_schemas = cube.GetMetadata("apiSchemas").GetAppliedItems()
    assert "PhysicsMeshCollisionAPI" not in cube_schemas
    assert "PhysxSDFMeshCollisionAPI" not in cube_schemas
    assert not cube.GetAttribute("physics:approximation")
    mesh_schemas = mesh.GetMetadata("apiSchemas").GetAppliedItems()
    assert "PhysicsMeshCollisionAPI" in mesh_schemas
    assert "PhysxSDFMeshCollisionAPI" in mesh_schemas
    assert mesh.GetAttribute("physics:approximation").Get() == "sdf"


@pytest.mark.parametrize("suffix", [".usda", ".usdz"])
def test_reuses_exact_material_and_binds_only_approved_missing_colliders(
    tmp_path: Path,
    suffix: str,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    asset = tmp_path / f"asset{suffix}"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/Bound/Collider", "/Root/Missing")
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        with_existing_material=True,
        binding_host_paths=("/Root/Bound",),
    )
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=colliders,
        material_operation="reuse_existing",
        bind_missing_collider_paths=colliders,
    )
    source_bytes = asset.read_bytes()
    source_stage = _open_stage(asset)
    source_material_fingerprint = plan_module._spec_stack_fingerprints(
        source_stage,
        Sdf.Path("/Root/PhysicsMaterials/Existing"),
        Sdf=Sdf,
    )

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    assert asset.read_bytes() == source_bytes
    assert result.receipt.physics_material_operation == "reuse_existing"
    assert (
        result.receipt.reused_physics_material_path == "/Root/PhysicsMaterials/Existing"
    )
    assert result.receipt.bound_missing_collider_paths == colliders
    assert result.receipt.preserved_approximations == ()
    output_stage = _open_stage(result.output_asset_path)
    assert (
        plan_module._spec_stack_fingerprints(
            output_stage,
            Sdf.Path("/Root/PhysicsMaterials/Existing"),
            Sdf=Sdf,
        )
        == source_material_fingerprint
    )
    material_path = output_stage.GetPrimAtPath(
        "/Root/PhysicsMaterials/Existing"
    ).GetPath()
    inherited = output_stage.GetPrimAtPath("/Root/Bound/Collider")
    missing = output_stage.GetPrimAtPath("/Root/Missing")
    inherited_material, _relationship = UsdShade.MaterialBindingAPI(
        inherited
    ).ComputeBoundMaterial("physics")
    missing_material, _relationship = UsdShade.MaterialBindingAPI(
        missing
    ).ComputeBoundMaterial("physics")
    assert inherited_material.GetPath() == material_path
    assert missing_material.GetPath() == material_path
    assert inherited.GetRelationship("material:binding:physics").GetTargets() == [
        material_path
    ]
    assert missing.GetRelationship("material:binding:physics").GetTargets() == [
        material_path
    ]
    assert (
        not output_stage.GetPrimAtPath("/Root/PhysicsMaterials/Existing")
        .GetAttribute("physics:density")
        .HasAuthoredValueOpinion()
    )


def test_reuse_receipt_and_output_are_deterministic(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/A", "/Root/B")
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        with_existing_material=True,
        direct_bound_paths=("/Root/A",),
    )
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=colliders,
        material_operation="reuse_existing",
        bind_missing_collider_paths=("/Root/B",),
    )

    first = author_physics_profile_plan(asset, plan_path, tmp_path / "output-a")
    second = author_physics_profile_plan(asset, plan_path, tmp_path / "output-b")
    reused = author_physics_profile_plan(asset, plan_path, tmp_path / "output-a")

    assert first.output_asset_path.read_bytes() == second.output_asset_path.read_bytes()
    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    assert first.receipt == second.receipt
    assert reused.reused_output


def test_preserves_convex_hull_and_supports_mixed_per_collider_policy(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/Convex", "/Root/Sdf")
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        approximation_tokens={"/Root/Convex": "convexHull"},
    )
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=colliders,
        collider_approximations=[
            {
                "prim_path": "/Root/Convex",
                "operation": "preserve_existing",
                "source_token": "convexHull",
            },
            {
                "prim_path": "/Root/Sdf",
                "operation": "author_sdf",
            },
        ],
    )
    source_stage = _open_stage(asset)
    source_fingerprint = plan_module._spec_stack_fingerprints(
        source_stage,
        Sdf.Path("/Root/Convex.physics:approximation"),
        Sdf=Sdf,
    )

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    stage = _open_stage(result.output_asset_path)
    convex = stage.GetPrimAtPath("/Root/Convex")
    sdf = stage.GetPrimAtPath("/Root/Sdf")
    assert convex.GetAttribute("physics:approximation").Get() == "convexHull"
    assert sdf.GetAttribute("physics:approximation").Get() == "sdf"
    assert (
        "PhysxSDFMeshCollisionAPI"
        not in convex.GetMetadata("apiSchemas").GetAppliedItems()
    )
    assert (
        "PhysxConvexHullCollisionAPI"
        in convex.GetMetadata("apiSchemas").GetAppliedItems()
    )
    assert "PhysxSDFMeshCollisionAPI" in sdf.GetMetadata("apiSchemas").GetAppliedItems()
    assert (
        plan_module._spec_stack_fingerprints(
            stage,
            Sdf.Path("/Root/Convex.physics:approximation"),
            Sdf=Sdf,
        )
        == source_fingerprint
    )
    assert [
        (item.prim_path, item.source_token)
        for item in result.receipt.preserved_approximations
    ] == [("/Root/Convex", "convexHull")]


def test_source_bound_author_sdf_transitions_convex_hull_without_source_mutation(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/A", "/Root/B")
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        approximation_tokens={path: "convexHull" for path in colliders},
    )
    source_bytes = asset.read_bytes()
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=colliders,
        collider_approximations=[
            {
                "prim_path": path,
                "operation": "author_sdf",
                "source_token": "convexHull",
            }
            for path in colliders
        ],
    )

    first = author_physics_profile_plan(asset, plan_path, tmp_path / "output-a")
    second = author_physics_profile_plan(asset, plan_path, tmp_path / "output-b")

    assert asset.read_bytes() == source_bytes
    assert first.output_asset_path.read_bytes() == second.output_asset_path.read_bytes()
    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    stage = _open_stage(first.output_asset_path)
    for path in colliders:
        prim = stage.GetPrimAtPath(path)
        assert prim.GetAttribute("physics:approximation").Get() == "sdf"
        applied = prim.GetMetadata("apiSchemas").GetAppliedItems()
        assert "PhysxSDFMeshCollisionAPI" in applied
        assert "PhysxConvexHullCollisionAPI" not in applied
    assert [
        (item.prim_path, item.source_token, item.output_token)
        for item in first.receipt.authored_sdf_transitions
    ] == [
        ("/Root/A", "convexHull", "sdf"),
        ("/Root/B", "convexHull", "sdf"),
    ]


def test_source_bound_author_sdf_rejects_stale_or_incompatible_source(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/A",)
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        approximation_tokens={"/Root/A": "sdf"},
    )
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=colliders,
        collider_approximations=[
            {
                "prim_path": "/Root/A",
                "operation": "author_sdf",
                "source_token": "convexHull",
            }
        ],
    )

    with pytest.raises(PhysicsProfilePlanError, match="Conflicting"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


def test_explicit_create_defaults_preserve_legacy_plan_and_artifact_bytes(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    legacy_plan_path = tmp_path / "legacy.json"
    explicit_plan_path = tmp_path / "explicit.json"
    _write_asset(asset)
    legacy = _plan_payload(asset)
    explicit = json.loads(json.dumps(legacy))
    explicit["physics_material"].update(
        {
            "operation": "create",
            "source_authored_opinions": None,
            "bind_missing_collider_paths": [],
        }
    )
    explicit["collider_approximations"] = []
    legacy_plan_path.write_text(json.dumps(legacy), encoding="utf-8")
    explicit_plan_path.write_text(json.dumps(explicit), encoding="utf-8")

    _legacy_plan, legacy_sha = load_physics_profile_plan(legacy_plan_path)
    _explicit_plan, explicit_sha = load_physics_profile_plan(explicit_plan_path)
    legacy_result = author_physics_profile_plan(
        asset, legacy_plan_path, tmp_path / "legacy-output"
    )
    explicit_result = author_physics_profile_plan(
        asset, explicit_plan_path, tmp_path / "explicit-output"
    )

    assert explicit_sha == legacy_sha
    assert (
        explicit_result.output_asset_path.read_bytes()
        == legacy_result.output_asset_path.read_bytes()
    )
    assert (
        explicit_result.receipt_path.read_bytes()
        == legacy_result.receipt_path.read_bytes()
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong_value", "effective value drifted"),
        ("wrong_opinion", "authored-opinion presence drifted"),
        ("stale_source", "source_asset_sha256 does not match"),
    ],
)
def test_reuse_rejects_wrong_value_opinion_and_stale_source(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_policy_asset(
        asset,
        collider_paths=("/Root/Collider",),
        with_existing_material=True,
        direct_bound_paths=("/Root/Collider",),
    )
    payload = _policy_plan_payload(
        asset,
        collider_paths=("/Root/Collider",),
        material_operation="reuse_existing",
    )
    if case == "wrong_value":
        payload["physics_material"]["dynamic_friction"] = 0.6
    elif case == "wrong_opinion":
        payload["physics_material"]["source_authored_opinions"]["dynamic_friction"] = (
            False
        )
        payload["physics_material"]["dynamic_friction"] = 0.0
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    if case == "stale_source":
        stage = _open_stage(asset)
        material = UsdPhysics.MaterialAPI(
            stage.GetPrimAtPath("/Root/PhysicsMaterials/Existing")
        )
        assert material.GetDynamicFrictionAttr().Set(0.6)
        assert stage.GetRootLayer().Save()
        del stage

    with pytest.raises(PhysicsProfilePlanError, match=message):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


@pytest.mark.parametrize("case", ["extra_material", "ambiguous_opinions"])
def test_reuse_rejects_extra_or_ambiguous_material(
    tmp_path: Path,
    case: str,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_policy_asset(
        asset,
        collider_paths=("/Root/Collider",),
        with_existing_material=True,
        direct_bound_paths=("/Root/Collider",),
    )
    if case == "extra_material":
        stage = _open_stage(asset)
        extra = UsdShade.Material.Define(stage, "/Root/PhysicsMaterials/Extra")
        assert UsdPhysics.MaterialAPI.Apply(extra.GetPrim())
        assert stage.GetRootLayer().Save()
        del stage
    else:
        dependency = tmp_path / "material-opinion.usda"
        dependency_stage = Usd.Stage.CreateNew(str(dependency))
        UsdGeom.Xform.Define(dependency_stage, "/Root")
        material = UsdShade.Material.Define(
            dependency_stage,
            "/Root/PhysicsMaterials/Existing",
        )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        assert material_api.CreateStaticFrictionAttr().Set(0.5)
        assert dependency_stage.GetRootLayer().Save()
        del dependency_stage
        stage = _open_stage(asset)
        stage.GetRootLayer().subLayerPaths = [dependency.name]
        assert stage.GetRootLayer().Save()
        del stage
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=("/Root/Collider",),
        material_operation="reuse_existing",
    )

    with pytest.raises(
        PhysicsProfilePlanError,
        match="exactly the approved|ambiguous authored opinions",
    ):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


@pytest.mark.parametrize("case", ["wrong_target", "collection"])
def test_reuse_rejects_incompatible_or_ambiguous_bindings(
    tmp_path: Path,
    case: str,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_policy_asset(
        asset,
        collider_paths=("/Root/Collider",),
        with_existing_material=True,
    )
    stage = _open_stage(asset)
    collider = stage.GetPrimAtPath("/Root/Collider")
    if case == "wrong_target":
        other = UsdShade.Material.Define(stage, "/Root/PhysicsMaterials/Other")
        relationship = collider.CreateRelationship(
            "material:binding:physics", custom=False
        )
        assert relationship.SetTargets([other.GetPath()])
    else:
        material = UsdShade.Material(
            stage.GetPrimAtPath("/Root/PhysicsMaterials/Existing")
        )
        collection = Usd.CollectionAPI.Apply(stage.GetDefaultPrim(), "colliders")
        assert collection.CreateIncludesRel().AddTarget(collider.GetPath())
        assert UsdShade.MaterialBindingAPI.Apply(stage.GetDefaultPrim()).Bind(
            collection,
            material,
            "existing",
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    assert stage.GetRootLayer().Save()
    del stage
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=("/Root/Collider",),
        material_operation="reuse_existing",
        bind_missing_collider_paths=("/Root/Collider",),
    )

    with pytest.raises(
        PhysicsProfilePlanError,
        match="incompatible physics binding|collection-based",
    ):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


@pytest.mark.parametrize(
    "approved",
    [("/Root/A",), ("/Root/A", "/Root/B", "/Root/C")],
)
def test_reuse_requires_exact_missing_binding_approval(
    tmp_path: Path,
    approved: tuple[str, ...],
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/A", "/Root/B", "/Root/C")
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        with_existing_material=True,
        direct_bound_paths=("/Root/C",),
    )
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=colliders,
        material_operation="reuse_existing",
        bind_missing_collider_paths=approved,
    )

    with pytest.raises(PhysicsProfilePlanError, match="do not exactly match"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


def test_reuse_source_mutation_aborts_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    _write_policy_asset(
        asset,
        collider_paths=("/Root/Collider",),
        with_existing_material=True,
    )
    _write_policy_plan(
        plan_path,
        asset,
        collider_paths=("/Root/Collider",),
        material_operation="reuse_existing",
        bind_missing_collider_paths=("/Root/Collider",),
    )
    original_check = plan_module._require_snapshot_unchanged
    calls = 0

    def mutate_during_authoring(snapshot: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            with asset.open("ab") as stream:
                stream.write(b"\n# concurrent reuse source mutation\n")
                stream.flush()
                os.fsync(stream.fileno())
        original_check(snapshot)

    monkeypatch.setattr(
        plan_module,
        "_require_snapshot_unchanged",
        mutate_during_authoring,
    )

    with pytest.raises(PhysicsProfilePlanError, match="changed during authoring"):
        author_physics_profile_plan(asset, plan_path, output)
    assert not list(output.glob("physics-profile-*"))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing", "missing, stale, or ambiguous"),
        ("stale", "missing, stale, or ambiguous"),
        ("partial", "exactly cover planned Mesh"),
        ("unsupported", "Invalid physics profile plan"),
        ("ambiguous", "ambiguous authored opinions"),
        ("sdf_schema", "incompatible SDF schema"),
    ],
)
def test_preserve_existing_approximation_rejects_unsafe_state(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    colliders = ("/Root/A", "/Root/B")
    source_tokens = (
        {} if case == "missing" else {path: "convexHull" for path in colliders}
    )
    if case == "unsupported":
        source_tokens = {path: "meshSimplification" for path in colliders}
    _write_policy_asset(
        asset,
        collider_paths=colliders,
        approximation_tokens=source_tokens,
    )
    if case == "ambiguous":
        dependency = tmp_path / "approximation-opinion.usda"
        dependency_stage = Usd.Stage.CreateNew(str(dependency))
        UsdGeom.Xform.Define(dependency_stage, "/Root")
        for path in colliders:
            mesh = UsdGeom.Mesh.Define(dependency_stage, path)
            assert (
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
                .CreateApproximationAttr()
                .Set("convexHull")
            )
        assert dependency_stage.GetRootLayer().Save()
        del dependency_stage
        stage = _open_stage(asset)
        stage.GetRootLayer().subLayerPaths = [dependency.name]
        assert stage.GetRootLayer().Save()
        del stage
    elif case == "sdf_schema":
        stage = _open_stage(asset)
        assert stage.GetPrimAtPath("/Root/A").AddAppliedSchema(
            "PhysxSDFMeshCollisionAPI"
        )
        assert stage.GetRootLayer().Save()
        del stage
    token = (
        "sdf"
        if case == "stale"
        else ("meshSimplification" if case == "unsupported" else "convexHull")
    )
    entries = [
        {
            "prim_path": path,
            "operation": "preserve_existing",
            "source_token": token,
        }
        for path in colliders
    ]
    if case == "partial":
        entries.pop()
    payload = _policy_plan_payload(
        asset,
        collider_paths=colliders,
        collider_approximations=entries,
    )
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PhysicsProfilePlanError, match=message):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


def test_rejects_collision_api_on_non_gprim(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/Root/PhysicsMaterials")
    collider = UsdGeom.Xform.Define(stage, "/Root/Collider")
    assert UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    assert stage.GetRootLayer().Save()
    del stage
    _write_plan(plan_path, asset)

    with pytest.raises(PhysicsProfilePlanError, match="editable active Gprim"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")
    assert not list((tmp_path / "output").glob("physics-profile-*"))


def test_rejects_mesh_only_opinion_on_analytic_gprim(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    collider_paths = _write_mixed_gprim_asset(asset)
    stage = _open_stage(asset)
    stage.GetPrimAtPath("/Root/Cube").CreateAttribute(
        "physics:approximation",
        Sdf.ValueTypeNames.Token,
        custom=False,
    ).Set("sdf")
    assert stage.GetRootLayer().Save()
    del stage
    _write_plan(plan_path, asset, collider_paths=collider_paths)

    with pytest.raises(PhysicsProfilePlanError, match="Non-Mesh collider"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")
    assert not list((tmp_path / "output").glob("physics-profile-*"))


def test_instance_proxy_colliders_are_inventoried_and_fail_closed(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    prototype = _write_instance_proxy_asset(asset)
    source_bytes = asset.read_bytes()
    prototype_bytes = prototype.read_bytes()
    _write_plan(plan_path, asset, collider_paths=("/Root/Ordinary",))

    with pytest.raises(PhysicsProfilePlanError, match="exactly match"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    _write_plan(
        plan_path,
        asset,
        collider_paths=("/Root/Instance/Collider", "/Root/Ordinary"),
    )
    with pytest.raises(PhysicsProfilePlanError, match="editable active Gprim"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")
    assert asset.read_bytes() == source_bytes
    assert prototype.read_bytes() == prototype_bytes
    assert not list((tmp_path / "output").glob("physics-profile-*"))


@pytest.mark.parametrize(
    ("conflict", "message"),
    [
        ("physics_material", "contains PhysicsMaterialAPI"),
        ("direct_binding", "Conflicting material:binding:physics"),
        ("collection_binding", "Conflicting collection-based"),
    ],
)
def test_instance_proxy_material_and_binding_source_inventories_fail_closed(
    tmp_path: Path,
    conflict: str,
    message: str,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    prototype = _write_instance_proxy_inventory_asset(asset, conflict=conflict)
    source_bytes = asset.read_bytes()
    prototype_bytes = prototype.read_bytes()
    _write_plan(plan_path, asset, collider_paths=("/Root/Ordinary",))

    with pytest.raises(PhysicsProfilePlanError, match=message):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    assert asset.read_bytes() == source_bytes
    assert prototype.read_bytes() == prototype_bytes
    assert not list((tmp_path / "output").glob("physics-profile-*"))


@pytest.mark.parametrize(
    ("conflict", "message"),
    [
        ("physics_material", "exactly the planned PhysicsMaterialAPI"),
        ("direct_binding", "inexact physics material binding"),
        ("collection_binding", "retained a collection-based"),
    ],
)
def test_instance_proxy_material_and_binding_reopen_inventories_fail_closed(
    tmp_path: Path,
    conflict: str,
    message: str,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    prototype = _write_instance_proxy_inventory_asset(asset)
    _write_plan(plan_path, asset, collider_paths=("/Root/Ordinary",))
    plan, _plan_sha256 = load_physics_profile_plan(plan_path)
    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    stage = _open_stage(result.output_asset_path)
    expected_state = plan_module._root_api_list_op_state(
        stage.GetRootLayer(),
        Sdf.Path("/Root/Ordinary"),
    )
    assert expected_state is not None
    del stage
    published_prototype = result.output_asset_path.with_name(prototype.name)
    _add_proxy_inventory_state(published_prototype, conflict)

    with pytest.raises(PhysicsProfilePlanError, match=message):
        plan_module._verify_profile(
            result.output_asset_path,
            plan,
            expected_state=plan_module._AuthoredProfileState(
                list_ops={"/Root/Ordinary": expected_state},
                material_spec_fingerprints=None,
                binding_states=(),
                preserved_approximation_fingerprints=(),
            ),
        )


@pytest.mark.parametrize(
    "case",
    [
        "extra_field",
        "wrong_schema",
        "wrong_approximation",
        "missing_density",
        "missing_dependency_bundle",
        "approval_false",
        "approval_identity_whitespace",
        "unsorted_colliders",
        "duplicate_colliders",
        "property_path",
        "nonfinite_value",
        "overflow_value",
    ],
)
def test_plan_schema_is_strict_and_frozen(
    tmp_path: Path,
    case: str,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset, collider_paths=("/Root/A", "/Root/B"))
    payload = _plan_payload(asset, collider_paths=("/Root/A", "/Root/B"))
    if case == "extra_field":
        payload["asset_id"] = "forbidden-inference-key"
    elif case == "wrong_schema":
        payload["schema_version"] = "content-agent-workflows.physics-profile-plan.v2"
    elif case == "wrong_approximation":
        payload["mesh_approximation"] = "convexHull"
    elif case == "missing_density":
        del payload["physics_material"]["density"]
    elif case == "missing_dependency_bundle":
        del payload["source_dependency_bundle_sha256"]
    elif case == "approval_false":
        payload["approval"]["approved"] = False
    elif case == "approval_identity_whitespace":
        payload["approval"]["owner_identity"] = " owner@example.test "
    elif case == "unsorted_colliders":
        payload["collider_prim_paths"] = ["/Root/B", "/Root/A"]
    elif case == "duplicate_colliders":
        payload["collider_prim_paths"] = ["/Root/A", "/Root/A"]
    elif case == "property_path":
        payload["collider_prim_paths"] = ["/Root/A.physics:mass"]
    elif case == "nonfinite_value":
        payload["physics_material"]["static_friction"] = float("nan")
    elif case == "overflow_value":
        payload["physics_material"]["static_friction"] = 10**4000
    plan_path = tmp_path / f"{case}.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PhysicsProfilePlanError):
        load_physics_profile_plan(plan_path)


def test_plan_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    payload = _plan_payload(asset)
    encoded = json.dumps(payload)
    encoded = encoded[:-1] + ',"mesh_approximation":"sdf"}'
    path = tmp_path / "duplicate.json"
    path.write_text(encoded, encoding="utf-8")

    with pytest.raises(PhysicsProfilePlanError, match="duplicate JSON key"):
        load_physics_profile_plan(path)


@pytest.mark.parametrize(
    "field",
    ["static_friction", "dynamic_friction", "restitution", "density"],
)
def test_plan_rejects_physics_value_float32_underflow(
    tmp_path: Path,
    field: str,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    payload = _plan_payload(asset)
    payload["physics_material"][field] = 1e-50
    path = tmp_path / f"underflow-{field}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PhysicsProfilePlanError, match="underflow USD floats"):
        load_physics_profile_plan(path)


@pytest.mark.parametrize(
    "field",
    ["static_friction", "dynamic_friction", "restitution"],
)
@pytest.mark.parametrize("token", ["1e-400", "-1e-400"])
def test_plan_rejects_lexically_nonzero_json_float_that_parses_to_zero(
    tmp_path: Path,
    field: str,
    token: str,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    payload = _plan_payload(asset)
    encoded = json.dumps(payload)
    original = json.dumps(payload["physics_material"][field])
    marker = f'"{field}": {original}'
    assert marker in encoded
    path = tmp_path / f"parser-underflow-{field}.json"
    path.write_text(
        encoded.replace(marker, f'"{field}": {token}', 1),
        encoding="utf-8",
    )

    with pytest.raises(PhysicsProfilePlanError, match="lexically nonzero"):
        load_physics_profile_plan(path)


def test_plan_wraps_json_integer_digit_limit_as_domain_error(tmp_path: Path) -> None:
    path = tmp_path / "huge-integer.json"
    path.write_text('{"value":' + ("9" * 5000) + "}", encoding="ascii")

    with pytest.raises(PhysicsProfilePlanError, match="Invalid physics profile plan"):
        load_physics_profile_plan(path)


def test_source_digest_must_match_exact_bytes(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    payload = _write_plan(plan_path, asset)
    payload["source_asset_sha256"] = "0" * 64
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PhysicsProfilePlanError, match="does not match"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")


def test_dependency_bundle_digest_must_match_owner_approved_bytes(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    dependency = package / "geometry.usda"
    root = package / "root.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(dependency)
    root.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Root"\n'
        "    subLayers = [@geometry.usda@]\n)\n"
        'over "Root" {}\n',
        encoding="utf-8",
    )
    _write_plan(plan_path, root)
    with dependency.open("ab") as stream:
        stream.write(b"\n# changed after owner approval\n")

    with pytest.raises(
        PhysicsProfilePlanError,
        match="source_dependency_bundle_sha256 does not match",
    ):
        author_physics_profile_plan(root, plan_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_preserves_existing_api_schema_list_op_categories(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    layer = Sdf.Layer.FindOrOpen(str(asset))
    assert layer is not None
    spec = layer.GetPrimAtPath("/Root/Collider")
    operation = Sdf.TokenListOp()
    operation.addedItems = ["ExistingAddedAPI"]
    operation.prependedItems = ["PhysicsCollisionAPI", "ExistingPreAPI"]
    operation.appendedItems = ["ExistingAppendAPI"]
    operation.deletedItems = ["RetiredAPI"]
    operation.orderedItems = ["ExistingAppendAPI", "ExistingPreAPI"]
    spec.SetInfo("apiSchemas", operation)
    assert layer.Save()
    _write_plan(plan_path, asset)

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    output_layer = Sdf.Layer.FindOrOpen(str(result.output_asset_path))
    assert output_layer is not None
    output_spec = output_layer.GetPrimAtPath("/Root/Collider")
    output_operation = output_spec.GetInfo("apiSchemas")
    assert output_operation.prependedItems == [
        "PhysicsCollisionAPI",
        "ExistingPreAPI",
        "PhysicsMeshCollisionAPI",
        "MaterialBindingAPI",
        "PhysxCollisionAPI",
        "PhysxSDFMeshCollisionAPI",
    ]
    assert output_operation.addedItems == ["ExistingAddedAPI"]
    assert output_operation.appendedItems == ["ExistingAppendAPI"]
    assert output_operation.deletedItems == ["RetiredAPI"]
    assert output_operation.orderedItems == [
        "ExistingAppendAPI",
        "ExistingPreAPI",
    ]


def test_preserves_explicit_api_schema_list_op(tmp_path: Path) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    layer = Sdf.Layer.FindOrOpen(str(asset))
    assert layer is not None
    spec = layer.GetPrimAtPath("/Root/Collider")
    operation = Sdf.TokenListOp()
    operation.explicitItems = ["PhysicsCollisionAPI", "ExistingAPI"]
    spec.SetInfo("apiSchemas", operation)
    assert layer.Save()
    _write_plan(plan_path, asset)

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    output_layer = Sdf.Layer.FindOrOpen(str(result.output_asset_path))
    assert output_layer is not None
    output_operation = output_layer.GetPrimAtPath("/Root/Collider").GetInfo(
        "apiSchemas"
    )
    assert output_operation.isExplicit
    assert output_operation.explicitItems == [
        "PhysicsCollisionAPI",
        "ExistingAPI",
        "PhysicsMeshCollisionAPI",
        "MaterialBindingAPI",
        "PhysxCollisionAPI",
        "PhysxSDFMeshCollisionAPI",
    ]


def test_list_op_guard_rejects_added_item_drift() -> None:
    before = plan_module._TokenListOpState(
        False,
        (),
        ("ExistingAddedAPI",),
        ("PhysicsMeshCollisionAPI",),
        (),
        (),
        (),
    )
    after = plan_module._TokenListOpState(
        False,
        (),
        ("ChangedAddedAPI",),
        ("PhysicsMeshCollisionAPI",),
        (),
        (),
        (),
    )

    with pytest.raises(PhysicsProfilePlanError, match="added items changed"):
        plan_module._require_list_op_preserved(
            before,
            after,
            required=("PhysicsMeshCollisionAPI",),
            path="/Root/Collider",
        )


@pytest.mark.parametrize("coverage_case", ["missing", "extra"])
def test_requires_exact_collision_api_coverage(
    tmp_path: Path,
    coverage_case: str,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset, collider_paths=("/Root/A", "/Root/B"))
    planned = (
        ("/Root/A",)
        if coverage_case == "missing"
        else (
            "/Root/A",
            "/Root/B",
            "/Root/C",
        )
    )
    _write_plan(plan_path, asset, collider_paths=planned)

    with pytest.raises(PhysicsProfilePlanError, match="exactly match"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")
    assert not list((tmp_path / "output").glob("physics-profile-*"))


@pytest.mark.parametrize(
    "conflict",
    [
        "approximation",
        "binding",
        "binding_strength",
        "collection_binding",
        "physics_material",
        "deleted_schema",
    ],
)
def test_existing_conflicts_fail_closed(
    tmp_path: Path,
    conflict: str,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    stage = _open_stage(asset)
    collider = stage.GetPrimAtPath("/Root/Collider")
    if conflict == "approximation":
        collider.CreateAttribute(
            "physics:approximation",
            Sdf.ValueTypeNames.Token,
            custom=False,
        ).Set("convexHull")
    elif conflict == "binding":
        collider.CreateRelationship(
            "material:binding:physics",
            custom=False,
        ).SetTargets([Sdf.Path("/Root/PhysicsMaterials/Other")])
    elif conflict == "binding_strength":
        relationship = collider.CreateRelationship(
            "material:binding:physics",
            custom=False,
        )
        assert relationship.SetTargets(
            [Sdf.Path("/Root/PhysicsMaterials/OwnerApproved")]
        )
        assert relationship.SetMetadata(
            "bindMaterialAs",
            UsdShade.Tokens.strongerThanDescendants,
        )
    elif conflict == "collection_binding":
        material = UsdShade.Material.Define(
            stage,
            "/Root/PhysicsMaterials/Existing",
        )
        collection = Usd.CollectionAPI.Apply(stage.GetDefaultPrim(), "colliders")
        assert collection.CreateIncludesRel().AddTarget(collider.GetPath())
        binding_api = UsdShade.MaterialBindingAPI.Apply(stage.GetDefaultPrim())
        assert binding_api.Bind(
            collection,
            material,
            "existing",
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )
    elif conflict == "physics_material":
        material = UsdShade.Material.Define(
            stage,
            "/Root/PhysicsMaterials/Existing",
        )
        assert UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    elif conflict == "deleted_schema":
        spec = stage.GetRootLayer().GetPrimAtPath("/Root/Collider")
        operation = spec.GetInfo("apiSchemas")
        operation.deletedItems = ["PhysxCollisionAPI"]
        spec.SetInfo("apiSchemas", operation)
    assert stage.GetRootLayer().Save()
    del stage
    _write_plan(plan_path, asset)

    with pytest.raises(PhysicsProfilePlanError, match="Conflicting|deleted|contains"):
        author_physics_profile_plan(asset, plan_path, tmp_path / "output")
    assert not list((tmp_path / "output").glob("physics-profile-*"))


def test_explicit_null_density_authors_no_density_opinion(tmp_path: Path) -> None:
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    _write_plan(plan_path, asset, density=None)

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    stage = _open_stage(result.output_asset_path)
    material = UsdPhysics.MaterialAPI(
        stage.GetPrimAtPath("/Root/PhysicsMaterials/OwnerApproved")
    )
    assert not material.GetDensityAttr().HasAuthoredValueOpinion()


def test_source_mutation_aborts_before_publication_and_cleans_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    _write_asset(asset)
    _write_plan(plan_path, asset)
    original_check = plan_module._require_snapshot_unchanged
    calls = 0

    def mutate_before_publication(snapshot: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            with asset.open("ab") as stream:
                stream.write(b"\n# concurrent source mutation\n")
                stream.flush()
                os.fsync(stream.fileno())
        original_check(snapshot)

    monkeypatch.setattr(
        plan_module,
        "_require_snapshot_unchanged",
        mutate_before_publication,
    )

    with pytest.raises(PhysicsProfilePlanError, match="changed during authoring"):
        author_physics_profile_plan(asset, plan_path, output)
    assert not list(output.glob("physics-profile-*"))
    assert not list(output.glob(".physics-profile-work-*"))


def test_post_rename_failure_rolls_back_complete_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    _write_asset(asset)
    _write_plan(plan_path, asset)
    original_fsync = plan_module._fsync_directory

    def fail_output_directory_sync(path: Path) -> None:
        if path == output:
            raise OSError("injected directory fsync failure")
        original_fsync(path)

    monkeypatch.setattr(
        plan_module,
        "_fsync_directory",
        fail_output_directory_sync,
    )

    with pytest.raises(OSError, match="injected"):
        author_physics_profile_plan(asset, plan_path, output)
    assert not list(output.glob("physics-profile-*"))
    assert not list(output.glob(".physics-profile-work-*"))


def test_post_publication_ordinary_failure_rolls_back_complete_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    _write_asset(asset)
    _write_plan(plan_path, asset)

    def fail_receipt_readback(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected receipt readback failure")

    monkeypatch.setattr(
        plan_module,
        "_verify_published_receipt",
        fail_receipt_readback,
    )

    with pytest.raises(RuntimeError, match="injected"):
        author_physics_profile_plan(asset, plan_path, output)
    assert not list(output.glob("physics-profile-*"))
    assert not list(output.glob(".physics-profile-work-*"))


def test_late_rollback_cannot_delete_concurrently_returned_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    _write_asset(asset)
    _write_plan(plan_path, asset)

    first_at_verify = threading.Event()
    second_waiting_for_lock = threading.Event()
    original_verify = plan_module._verify_published_receipt
    original_file_lock = plan_module.FileLock
    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}

    class ObservedFileLock:
        def __init__(self, path: str) -> None:
            self._lock = original_file_lock(path)

        def __enter__(self) -> Any:
            if threading.current_thread().name == "second-author":
                second_waiting_for_lock.set()
            return self._lock.__enter__()

        def __exit__(self, *args: Any) -> Any:
            return self._lock.__exit__(*args)

    def fail_first_receipt_readback(*args: Any, **kwargs: Any) -> None:
        if threading.current_thread().name == "first-author":
            first_at_verify.set()
            assert second_waiting_for_lock.wait(30)
            raise RuntimeError("injected late first-author failure")
        original_verify(*args, **kwargs)

    def run_author(name: str) -> None:
        try:
            results[name] = author_physics_profile_plan(asset, plan_path, output)
        except BaseException as exc:
            errors[name] = exc

    monkeypatch.setattr(plan_module, "FileLock", ObservedFileLock)
    monkeypatch.setattr(
        plan_module,
        "_verify_published_receipt",
        fail_first_receipt_readback,
    )

    first = threading.Thread(target=run_author, args=("first",), name="first-author")
    first.start()
    assert first_at_verify.wait(30)
    second = threading.Thread(
        target=run_author,
        args=("second",),
        name="second-author",
    )
    second.start()
    first.join(30)
    second.join(30)

    assert not first.is_alive()
    assert not second.is_alive()
    assert isinstance(errors.get("first"), RuntimeError)
    assert "second" not in errors
    assert not results["second"].reused_output
    assert results["second"].output_asset_path.is_file()


def _write_usdz_asset(path: Path, *, second_point_x: float = 1.0) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdUtils = pytest.importorskip("pxr.UsdUtils")

    source_tree = path.parent / f"{path.stem}-source"
    source_tree.mkdir()
    dependency = source_tree / "geometry.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency))
    UsdGeom.Xform.Define(dependency_stage, "/Root")
    mesh = UsdGeom.Mesh.Define(dependency_stage, "/Root/Collider")
    mesh.CreatePointsAttr([(0, 0, 0), (second_point_x, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    assert UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    assert dependency_stage.GetRootLayer().Save()
    del dependency_stage

    root_layer = source_tree / "root.usda"
    root_stage = Usd.Stage.CreateNew(str(root_layer))
    root = UsdGeom.Xform.Define(root_stage, "/Root")
    root_stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(root_stage, "/Root/PhysicsMaterials")
    root_stage.GetRootLayer().subLayerPaths = ["geometry.usda"]
    assert root_stage.GetRootLayer().Save()
    del root_stage
    assert UsdUtils.CreateNewUsdzPackage(str(root_layer), str(path))


def test_usdz_derivative_is_self_contained_and_preserves_dependencies(
    tmp_path: Path,
) -> None:
    UsdUtils = pytest.importorskip("pxr.UsdUtils")

    asset = tmp_path / "asset.usdz"
    plan_path = tmp_path / "plan.json"
    _write_usdz_asset(asset)
    _write_plan(plan_path, asset)
    source_bytes = asset.read_bytes()

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    assert asset.read_bytes() == source_bytes
    assert zipfile.is_zipfile(result.output_asset_path)
    with zipfile.ZipFile(result.output_asset_path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        assert len(files) >= 2
        assert all(info.compress_type == zipfile.ZIP_STORED for info in files)
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(
        str(result.output_asset_path)
    )
    assert not unresolved
    stage = _open_stage(result.output_asset_path)
    assert stage.GetPrimAtPath("/Root/Collider")


def test_usdz_extraction_uses_verified_private_copy_during_aba_source_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "asset.usdz"
    alternate = tmp_path / "alternate.usdz"
    plan_path = tmp_path / "plan.json"
    _write_usdz_asset(asset, second_point_x=1.0)
    _write_usdz_asset(alternate, second_point_x=2.0)
    _write_plan(plan_path, asset)
    approved_bytes = asset.read_bytes()
    alternate_bytes = alternate.read_bytes()
    approved_sha256 = _file_sha256(asset)
    extracted_sources: list[tuple[Path, str]] = []
    original_extract = plan_module._extract_usdz

    def swap_source_around_extraction(source: Path, destination: Path) -> str:
        extracted_sources.append((source, _file_sha256(source)))
        asset.write_bytes(alternate_bytes)
        try:
            return original_extract(source, destination)
        finally:
            asset.write_bytes(approved_bytes)

    monkeypatch.setattr(
        plan_module,
        "_extract_usdz",
        swap_source_around_extraction,
    )

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    assert asset.read_bytes() == approved_bytes
    assert len(extracted_sources) == 1
    extracted_source, extracted_sha256 = extracted_sources[0]
    assert extracted_source != asset
    assert extracted_sha256 == approved_sha256
    stage = _open_stage(result.output_asset_path)
    points = UsdGeom.Mesh(stage.GetPrimAtPath("/Root/Collider")).GetPointsAttr().Get()
    assert tuple(points[1]) == pytest.approx((1.0, 0.0, 0.0))


def test_usdz_dependency_inventory_rejects_extra_packaged_bytes(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "inventory.usdz"
    dependency_bytes = b"approved dependency"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("root.usda", b'#usda 1.0\ndef Xform "Root" {}\n')
        archive.writestr("geometry.bin", dependency_bytes)
        archive.writestr("unplanned.bin", b"unplanned dependency")

    with pytest.raises(PhysicsProfilePlanError, match="inventory is not an exact"):
        plan_module._require_usdz_dependency_bytes(
            archive_path,
            (("geometry.bin", hashlib.sha256(dependency_bytes).hexdigest()),),
        )


def test_usdz_and_receipt_bytes_are_deterministic_across_clean_runs(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usdz"
    plan_path = tmp_path / "plan.json"
    _write_usdz_asset(asset)
    _write_plan(plan_path, asset)

    first = author_physics_profile_plan(asset, plan_path, tmp_path / "output-a")
    second = author_physics_profile_plan(asset, plan_path, tmp_path / "output-b")
    reused = author_physics_profile_plan(asset, plan_path, tmp_path / "output-a")

    assert first.output_asset_path.read_bytes() == second.output_asset_path.read_bytes()
    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    assert first.receipt == second.receipt
    assert reused.reused_output
    assert reused.output_asset_path == first.output_asset_path


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is unavailable")
def test_usdz_and_receipt_bytes_are_deterministic_across_timezones(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usdz"
    plan_path = tmp_path / "plan.json"
    _write_usdz_asset(asset)
    _write_plan(plan_path, asset)
    original_timezone = os.environ.get("TZ")
    outputs: list[tuple[bytes, bytes, tuple[tuple[int, ...], ...]]] = []
    mtimes: list[float] = []

    try:
        for index, timezone in enumerate(("UTC", "America/Los_Angeles", "Asia/Tokyo")):
            os.environ["TZ"] = timezone
            time.tzset()
            mtimes.append(plan_module._deterministic_zip_mtime())
            result = author_physics_profile_plan(
                asset,
                plan_path,
                tmp_path / f"output-{index}",
            )
            with zipfile.ZipFile(result.output_asset_path) as archive:
                timestamps = tuple(
                    info.date_time for info in archive.infolist() if not info.is_dir()
                )
            outputs.append(
                (
                    result.output_asset_path.read_bytes(),
                    result.receipt_path.read_bytes(),
                    timestamps,
                )
            )
    finally:
        if original_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_timezone
        time.tzset()

    assert len({output for output, _receipt, _timestamps in outputs}) == 1
    assert len({receipt for _output, receipt, _timestamps in outputs}) == 1
    assert set(mtimes) == {315532800.0}
    assert {
        timestamp
        for _output, _receipt, timestamps in outputs
        for timestamp in timestamps
    } == {(1980, 1, 1, 0, 0, 0)}


def test_source_and_archive_symlink_or_traversal_paths_are_rejected(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    link = tmp_path / "linked.usda"
    link.symlink_to(asset)
    link_plan = tmp_path / "linked-plan.json"
    _write_plan(link_plan, asset)

    with pytest.raises(PhysicsProfilePlanError, match="symlink"):
        author_physics_profile_plan(link, link_plan, tmp_path / "link-output")

    unsafe = tmp_path / "unsafe.usdz"
    with zipfile.ZipFile(unsafe, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("root.usda", asset.read_bytes())
        archive.writestr("../escape.bin", b"escape")
    unsafe_plan = tmp_path / "unsafe-plan.json"
    _write_plan(
        unsafe_plan,
        unsafe,
        source_dependency_bundle_sha256="0" * 64,
    )
    with pytest.raises(PhysicsProfilePlanError, match="unsafe member path"):
        author_physics_profile_plan(
            unsafe,
            unsafe_plan,
            tmp_path / "unsafe-output",
        )


def test_absolute_dependency_that_escapes_private_package_is_rejected(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    dependency = tmp_path / "outside.usda"
    _write_asset(dependency)
    root = package / "root.usda"
    root.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Root"\n'
        f"    subLayers = [@{dependency}@]\n)\n"
        'over "Root" {}\n',
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, root)

    with pytest.raises(PhysicsProfilePlanError, match="outside its private package"):
        author_physics_profile_plan(root, plan_path, tmp_path / "output")


def test_relative_parent_dependency_is_preserved_without_package_size_policy(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    dependency = tmp_path / "outside.usda"
    _write_asset(dependency)
    root = package / "root.usda"
    root.write_text(
        '#usda 1.0\n(\n    defaultPrim = "Root"\n'
        "    subLayers = [@../outside.usda@]\n)\n"
        'over "Root" {}\n',
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, root)

    result = author_physics_profile_plan(root, plan_path, tmp_path / "output")

    published_dependency = result.output_asset_path.parent.parent / dependency.name
    assert result.output_asset_path.name == root.name
    assert published_dependency.read_bytes() == dependency.read_bytes()


def test_dependency_larger_than_legacy_snapshot_limit_is_streamed(
    tmp_path: Path,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    dependency = tmp_path / "large-dependency.bin"
    _write_asset(asset)
    dependency.write_bytes(b"L" * (16 * 1024 * 1024 + 1))
    stage = _open_stage(asset)
    stage.GetDefaultPrim().CreateAttribute(
        "validation:largeDependency",
        Sdf.ValueTypeNames.Asset,
        custom=True,
    ).Set(Sdf.AssetPath(dependency.name))
    assert stage.GetRootLayer().Save()
    del stage
    _write_plan(plan_path, asset)

    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    published_dependency = result.output_asset_path.parent / dependency.name
    assert published_dependency.stat().st_size == dependency.stat().st_size
    assert _file_sha256(published_dependency) == _file_sha256(dependency)


def test_raw_tokens_resolve_as_real_physx_apis_when_runtime_is_available(
    tmp_path: Path,
) -> None:
    try:
        from pxr import PhysxSchema
    except ImportError:
        pytest.skip("PhysxSchema runtime is not installed")
    collision_api = getattr(PhysxSchema, "PhysxCollisionAPI", None)
    sdf_api = getattr(PhysxSchema, "PhysxSDFMeshCollisionAPI", None)
    if collision_api is None or sdf_api is None:
        pytest.skip("Installed PhysxSchema runtime lacks required APIs")

    asset = tmp_path / "asset.usda"
    plan_path = tmp_path / "plan.json"
    _write_asset(asset)
    _write_plan(plan_path, asset)
    result = author_physics_profile_plan(asset, plan_path, tmp_path / "output")

    stage = _open_stage(result.output_asset_path)
    collider = stage.GetPrimAtPath("/Root/Collider")
    assert collider.HasAPI(collision_api)
    assert collider.HasAPI(sdf_api)


def test_main_parses_arguments_and_emits_stable_pass_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.usda"
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "output"
    output_asset = output / "physics-profile-digest" / "artifact" / source.name
    receipt_path = output_asset.parent.parent / "physics-profile-receipt.json"
    receipt = SimpleNamespace(
        output_asset_sha256="1" * 64,
        output_artifact_sha256="2" * 64,
        plan_sha256="3" * 64,
        source_asset_sha256="4" * 64,
    )
    result = SimpleNamespace(
        output_asset_path=output_asset,
        receipt_path=receipt_path,
        receipt=receipt,
        reused_output=False,
    )
    received: list[tuple[Path, Path, Path]] = []

    def fake_author(
        source_asset: str | Path,
        requested_plan: str | Path,
        output_dir: str | Path,
    ) -> SimpleNamespace:
        received.append((Path(source_asset), Path(requested_plan), Path(output_dir)))
        return result

    monkeypatch.setattr(plan_module, "author_physics_profile_plan", fake_author)

    exit_code = plan_module.main(
        [str(source), str(plan_path), "--output-dir", str(output)]
    )

    assert exit_code == 0
    assert received == [(source, plan_path, output)]
    expected = {
        "output_asset_path": str(output_asset),
        "output_asset_sha256": "1" * 64,
        "output_artifact_sha256": "2" * 64,
        "plan_sha256": "3" * 64,
        "receipt_path": str(receipt_path),
        "reused_output": False,
        "source_asset_sha256": "4" * 64,
        "status": "PASS",
    }
    captured = capsys.readouterr()
    assert captured.out == json.dumps(expected, indent=2, sort_keys=True) + "\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "error",
    [
        PhysicsProfilePlanError("owner approval is invalid"),
        RuntimeError("unexpected authoring failure"),
    ],
    ids=["expected", "unexpected"],
)
def test_main_emits_stable_blocked_json_for_ordinary_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail_author(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(plan_module, "author_physics_profile_plan", fail_author)

    exit_code = plan_module.main(
        [
            str(tmp_path / "source.usda"),
            str(tmp_path / "plan.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 1
    expected = {"error": str(error), "status": "BLOCKED"}
    captured = capsys.readouterr()
    assert captured.out == json.dumps(expected, sort_keys=True) + "\n"
    assert captured.err == ""


@pytest.mark.parametrize("blocked", [False, True], ids=["pass", "blocked"])
def test_main_flushes_machine_json_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked: bool,
) -> None:
    receipt = SimpleNamespace(
        output_asset_sha256="1" * 64,
        output_artifact_sha256="2" * 64,
        plan_sha256="3" * 64,
        source_asset_sha256="4" * 64,
    )
    result = SimpleNamespace(
        output_asset_path=tmp_path / "output.usda",
        receipt_path=tmp_path / "receipt.json",
        receipt=receipt,
        reused_output=False,
    )

    def fake_author(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        if blocked:
            raise PhysicsProfilePlanError("blocked")
        return result

    print_calls: list[dict[str, Any]] = []

    def capture_print(*_args: Any, **kwargs: Any) -> None:
        print_calls.append(kwargs)

    monkeypatch.setattr(plan_module, "author_physics_profile_plan", fake_author)
    monkeypatch.setattr("builtins.print", capture_print)

    exit_code = plan_module.main(
        [
            str(tmp_path / "source.usda"),
            str(tmp_path / "plan.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == (1 if blocked else 0)
    assert print_calls == [{"flush": True}]


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt(), SystemExit(23)],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_main_does_not_swallow_base_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    def stop_author(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(plan_module, "author_physics_profile_plan", stop_author)

    with pytest.raises(type(error)) as raised:
        plan_module.main(
            [
                str(tmp_path / "source.usda"),
                str(tmp_path / "plan.json"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )

    if isinstance(error, SystemExit):
        raised_error = raised.value
        assert isinstance(raised_error, SystemExit)
        assert raised_error.code == error.code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_requires_output_directory_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def author_must_not_run(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("authoring ran before CLI argument validation")

    monkeypatch.setattr(
        plan_module,
        "author_physics_profile_plan",
        author_must_not_run,
    )

    with pytest.raises(SystemExit) as raised:
        plan_module.main([str(tmp_path / "source.usda"), str(tmp_path / "plan.json")])

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--output-dir" in captured.err
