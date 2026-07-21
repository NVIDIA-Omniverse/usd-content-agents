# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused adversarial tests for Gate 3A stage-metadata plans."""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pxr")
from pxr import Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import content_agent_workflows.simready.gate3a_stage_metadata as stage_metadata  # noqa: E402
from content_agent_workflows.simready import (  # noqa: E402
    GATE3A_STAGE_METADATA_PLAN_SCHEMA_VERSION,
    author_gate3a_stage_metadata_derivative,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(
    tmp_path: Path,
    *,
    explicit_units: bool = True,
    reset_body: bool = False,
    translated_root: bool = False,
    extra_top_root: bool = False,
    unsafe_transform: str | None = None,
    transform_time_sample: bool = False,
    linear_time_sample: bool = False,
    variant: bool = False,
    instanceable: bool = False,
    composition: bool = False,
    multiple_endpoint: bool = False,
    distance_joint: bool = False,
    nonfinite_linear: bool = False,
    extra_bytes: int = 0,
) -> Path:
    layer = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(layer))
    if explicit_units:
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    root.SetMetadata("kind", Kind.Tokens.component)
    root_xform = UsdGeom.Xformable(root)
    if translated_root:
        root_xform.AddTranslateOp().Set(Gf.Vec3d(50.0, 0.0, 0.0))
    if unsafe_transform is not None:
        unsafe = UsdGeom.Xform.Define(stage, "/Asset/Unsafe").GetPrim()
        matrix = Gf.Matrix4d(1.0)
        if unsafe_transform == "reflection":
            matrix.SetRow(0, Gf.Vec4d(-1.0, 0.0, 0.0, 0.0))
        elif unsafe_transform == "shear":
            matrix.SetRow(0, Gf.Vec4d(1.0, 0.25, 0.0, 0.0))
        else:  # pragma: no cover - helper contract
            raise ValueError(unsafe_transform)
        UsdGeom.Xformable(unsafe).AddTransformOp().Set(matrix)
    if transform_time_sample:
        root_xform.AddRotateXOp().Set(10.0, 1.0)
    if variant:
        variants = root.GetVariantSets().AddVariantSet("model")
        variants.AddVariant("a")
        variants.SetVariantSelection("a")
        with variants.GetVariantEditContext():
            stage.DefinePrim("/Asset/VariantChild", "Xform")
    if instanceable:
        root.SetInstanceable(True)
    dependency: Path | None = None
    if composition:
        dependency = tmp_path / "dependency.usda"
        dependency_stage = Usd.Stage.CreateNew(str(dependency))
        dependency_stage.DefinePrim("/Referenced", "Xform")
        assert dependency_stage.GetRootLayer().Save()
        root.GetReferences().AddReference(dependency.name, "/Referenced")
    nested = UsdGeom.Xform.Define(stage, "/Asset/Nested").GetPrim()
    nested.SetMetadata("kind", Kind.Tokens.group)
    body0 = UsdGeom.Xform.Define(stage, "/Asset/Body0").GetPrim()
    body1 = UsdGeom.Xform.Define(stage, "/Asset/Body1").GetPrim()
    for body in (body0, body1):
        UsdPhysics.RigidBodyAPI.Apply(body)
    mesh = UsdGeom.Cube.Define(stage, "/Asset/Body1/Shape")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    body1_xform = UsdGeom.Xformable(body1)
    if reset_body:
        body1_xform.SetResetXformStack(True)
    body1_xform.AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Asset/Joint")
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets(
        [body1.GetPath(), body0.GetPath()] if multiple_endpoint else [body1.GetPath()]
    )
    joint.CreateLocalPos0Attr(Gf.Vec3f(100.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(float("nan") if nonfinite_linear else 50.0)
    if linear_time_sample:
        joint.GetUpperLimitAttr().Set(60.0, 1.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateMaxForceAttr(40.0)
    drive.CreateTargetPositionAttr(25.0)
    drive.CreateTargetVelocityAttr(10.0)
    joint.GetPrim().CreateAttribute(
        "state:linear:physics:position", Sdf.ValueTypeNames.Float
    ).Set(20.0)
    joint.GetPrim().CreateAttribute(
        "state:linear:physics:velocity", Sdf.ValueTypeNames.Float
    ).Set(5.0)
    if distance_joint:
        distance = UsdPhysics.DistanceJoint.Define(stage, "/Asset/Distance")
        distance.CreateBody0Rel().SetTargets([body0.GetPath()])
        distance.CreateBody1Rel().SetTargets([body1.GetPath()])
        distance.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        distance.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        distance.CreateMinDistanceAttr(10.0)
        distance.CreateMaxDistanceAttr(20.0)
    UsdPhysics.FilteredPairsAPI.Apply(body0).CreateFilteredPairsRel().SetTargets(
        [body1.GetPath()]
    )
    if extra_top_root:
        accessory = UsdGeom.Xform.Define(stage, "/Accessory")
        accessory.AddTranslateOp().Set(Gf.Vec3d(200.0, 0.0, 0.0))
        UsdGeom.Cube.Define(stage, "/Accessory/Shape")
    stage.SetDefaultPrim(root)
    assert stage.GetRootLayer().Save()
    package = tmp_path / "asset.usdz"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_STORED) as archive:
        archive.write(layer, layer.name)
        if dependency is not None:
            archive.write(dependency, dependency.name)
        if extra_bytes:
            archive.writestr("payload.bin", b"x" * extra_bytes)
    return package.resolve()


def _plan(tmp_path: Path, asset: Path, *, mutate: Any | None = None) -> Path:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"owner_finding":"stage-metadata"}\n', encoding="utf-8")
    stage = Usd.Stage.Open(str(asset), load=Usd.Stage.LoadAll)
    explicit_units = stage.HasAuthoredMetadata("metersPerUnit")
    default_root = stage.GetDefaultPrim()
    scale_targets: dict[str, Any] = {}
    for prim in stage.GetPseudoRoot().GetChildren():
        if prim != default_root:
            scale_targets[str(prim.GetPath())] = prim

    def add_first_xformable_descendants(parent: Any) -> None:
        for child in parent.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                scale_targets[str(child.GetPath())] = child
            else:
                add_first_xformable_descendants(child)

    add_first_xformable_descendants(default_root)
    for prim in stage.TraverseAll():
        if (
            prim != default_root
            and prim.IsA(UsdGeom.Xformable)
            and UsdGeom.Xformable(prim).GetResetXformStack()
        ):
            scale_targets[str(prim.GetPath())] = prim
    quantities = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdPhysics.PrismaticJoint):
            names = (
                "physics:lowerLimit",
                "physics:upperLimit",
                "drive:linear:physics:maxForce",
                "drive:linear:physics:targetPosition",
                "drive:linear:physics:targetVelocity",
                "state:linear:physics:position",
                "state:linear:physics:velocity",
            )
        elif prim.IsA(UsdPhysics.DistanceJoint):
            names = ("physics:minDistance", "physics:maxDistance")
        else:
            continue
        for name in names:
            value = prim.GetAttribute(name).Get()
            source = (
                float(value)
                if isinstance(value, int | float)
                else [float(v) for v in value]
            )
            values = source if isinstance(source, list) else [source]
            if not all(math.isfinite(item) for item in values):
                continue
            target = (
                source * 0.01
                if isinstance(source, float)
                else [v * 0.01 for v in source]
            )
            quantities.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "attribute_name": name,
                    "source_value": source,
                    "target_value": target,
                }
            )
    payload: dict[str, Any] = {
        "schema_version": GATE3A_STAGE_METADATA_PLAN_SCHEMA_VERSION,
        "source_asset_path": str(asset),
        "source_asset_sha256": _sha(asset),
        "provenance": {
            "approved_by": "asset-owner",
            "approval_reference": "issue-529",
            "evidence": [
                {"artifact_path": str(evidence), "artifact_sha256": _sha(evidence)}
            ],
        },
        "source_units_mode": (
            "authored_stage_metadata"
            if explicit_units
            else "owner_approved_missing_metadata"
        ),
        "source_meters_per_unit": 0.01,
        "target_meters_per_unit": 1.0,
        "default_root_path": str(default_root.GetPath()),
        "transform_scales": [
            {"prim_path": prim_path, "scale": 0.01}
            for prim_path in sorted(scale_targets)
        ],
        "linear_quantities": quantities,
        "clear_nested_kinds": [{"prim_path": "/Asset/Nested", "source_kind": "group"}],
    }
    if mutate:
        mutate(payload)
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path.resolve()


def test_authors_exact_plan_and_preserves_physical_and_physics_proofs(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path)
    source = asset.read_bytes()
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert result.passed, result.reason
    assert asset.read_bytes() == source
    assert result.receipt_path and result.receipt_path.is_file()
    receipt = json.loads(result.receipt_path.read_text())
    assert all(
        receipt[key]
        for key in (
            "physical_world_transforms_preserved",
            "physical_geometry_preserved",
            "physical_joint_anchors_preserved",
            "physics_inventory_preserved",
            "joint_graph_preserved",
            "filtered_pairs_preserved",
            "dependencies_preserved",
            "source_bytes_preserved",
        )
    )
    stage = Usd.Stage.Open(str(result.output_path), load=Usd.Stage.LoadAll)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    default_root = stage.GetDefaultPrim()
    assert Gf.IsClose(
        UsdGeom.Xformable(default_root).GetLocalTransformation(),
        Gf.Matrix4d(1.0),
        1e-12,
    )
    assert not default_root.GetAttribute("xformOp:scale:gate3aStageMetadata")
    assert stage.GetPrimAtPath("/Asset/Body0").GetAttribute(
        "xformOp:scale:gate3aStageMetadata"
    ).Get() == Gf.Vec3d(0.01)
    assert not stage.GetPrimAtPath("/Asset/Nested").HasAuthoredMetadata("kind")
    assert stage.GetPrimAtPath("/Asset/Joint").GetAttribute(
        "physics:upperLimit"
    ).Get() == pytest.approx(0.5)
    assert stage.GetPrimAtPath("/Asset/Joint").GetAttribute(
        "drive:linear:physics:maxForce"
    ).Get() == pytest.approx(0.4)
    assert receipt["default_root_identity_preserved"]
    assert (
        receipt["source_physical_state_sha256"]
        == receipt["output_physical_state_sha256"]
    )


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda plan: plan["transform_scales"].clear(), "transform_scales"),
        (
            lambda plan: plan["transform_scales"].append(
                {"prim_path": "/Asset/Body0/Extra", "scale": 0.01}
            ),
            "extra=",
        ),
        (lambda plan: plan["linear_quantities"].pop(), "linear_quantities"),
        (lambda plan: plan["clear_nested_kinds"].clear(), "clear_nested_kinds"),
    ],
)
def test_blocks_incomplete_or_fallback_based_plan(
    tmp_path: Path, mutation: Any, reason: str
) -> None:
    asset = _asset(tmp_path)
    plan = _plan(tmp_path, asset, mutate=mutation)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert reason in result.reason


def test_owner_evidence_normalizes_missing_source_metadata(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path, explicit_units=False)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert result.passed, result.reason
    stage = Usd.Stage.Open(str(result.output_path))
    assert stage.HasAuthoredMetadata("metersPerUnit")
    assert float(stage.GetMetadata("metersPerUnit")) == 1.0
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["source_units_mode"] == "owner_approved_missing_metadata"


@pytest.mark.parametrize(
    "explicit_units,mode",
    [
        (False, "authored_stage_metadata"),
        (True, "owner_approved_missing_metadata"),
    ],
)
def test_blocks_source_unit_mode_mismatch(
    tmp_path: Path, explicit_units: bool, mode: str
) -> None:
    asset = _asset(tmp_path, explicit_units=explicit_units)
    plan = _plan(
        tmp_path,
        asset,
        mutate=lambda payload: payload.update(source_units_mode=mode),
    )
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert "metersPerUnit" in result.reason


def test_reset_and_multiple_roots_are_complete_and_scale_is_outermost(
    tmp_path: Path,
) -> None:
    asset = _asset(tmp_path, reset_body=True, extra_top_root=True)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert result.passed, result.reason
    stage = Usd.Stage.Open(str(result.output_path), load=Usd.Stage.LoadAll)
    default_root = stage.GetDefaultPrim()
    assert Gf.IsClose(
        UsdGeom.Xformable(default_root).GetLocalTransformation(),
        Gf.Matrix4d(1.0),
        1e-12,
    )
    assert not default_root.GetAttribute("xformOp:scale:gate3aStageMetadata")
    top_order = [
        str(value)
        for value in stage.GetPrimAtPath("/Accessory")
        .GetAttribute("xformOpOrder")
        .Get()
    ]
    reset_order = [
        str(value)
        for value in stage.GetPrimAtPath("/Asset/Body1")
        .GetAttribute("xformOpOrder")
        .Get()
    ]
    assert top_order[:2] == [
        "xformOp:scale:gate3aStageMetadata",
        "xformOp:translate",
    ]
    assert reset_order[:3] == [
        "!resetXformStack!",
        "xformOp:scale:gate3aStageMetadata",
        "xformOp:translate",
    ]


def test_blocks_non_identity_default_root(tmp_path: Path) -> None:
    asset = _asset(tmp_path, translated_root=True)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert "default root local transform must be identity" in result.reason


def test_default_root_remains_identity_for_pinned_origin_rule(tmp_path: Path) -> None:
    asset = _asset(tmp_path, reset_body=True, extra_top_root=True)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert result.passed, result.reason
    stage = Usd.Stage.Open(str(result.output_path), load=Usd.Stage.LoadAll)
    local = UsdGeom.Xformable(stage.GetDefaultPrim()).GetLocalTransformation()
    assert Gf.IsClose(local, Gf.Matrix4d(1.0), 1e-12)
    assert all(
        local.GetRow(index).GetLength() == pytest.approx(1.0) for index in range(3)
    )


def test_distance_joint_limits_are_scaled(tmp_path: Path) -> None:
    asset = _asset(tmp_path, distance_joint=True)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert result.passed, result.reason
    stage = Usd.Stage.Open(str(result.output_path), load=Usd.Stage.LoadAll)
    distance = stage.GetPrimAtPath("/Asset/Distance")
    assert distance.GetAttribute("physics:minDistance").Get() == pytest.approx(0.1)
    assert distance.GetAttribute("physics:maxDistance").Get() == pytest.approx(0.2)


@pytest.mark.parametrize(
    "unsafe_transform,reason",
    [("reflection", "reflect"), ("shear", "shear")],
)
def test_blocks_unsafe_root_transforms(
    tmp_path: Path, unsafe_transform: str, reason: str
) -> None:
    asset = _asset(tmp_path, unsafe_transform=unsafe_transform)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert reason in result.reason


@pytest.mark.parametrize(
    "asset_kwargs,reason",
    [
        ({"instanceable": True}, "instances or variants"),
        ({"variant": True}, "instances or variants"),
        ({"composition": True}, "composition ambiguity"),
        ({"transform_time_sample": True}, "time samples"),
        ({"linear_time_sample": True}, "time samples"),
        ({"multiple_endpoint": True}, "endpoint"),
    ],
)
def test_blocks_ambiguous_or_time_varying_sources(
    tmp_path: Path, asset_kwargs: dict[str, Any], reason: str
) -> None:
    asset = _asset(tmp_path, **asset_kwargs)
    plan = _plan(tmp_path, asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert reason in result.reason


def test_blocks_stale_source_and_evidence(tmp_path: Path) -> None:
    asset = _asset(tmp_path)
    plan = _plan(tmp_path, asset)
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"owner_finding":"changed"}\n', encoding="utf-8")
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-evidence-out",
    )
    assert not result.passed
    assert "evidence SHA-256 is stale" in result.reason

    plan = _plan(tmp_path, asset)
    with asset.open("ab") as stream:
        stream.write(b"changed")
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-source-out",
    )
    assert not result.passed
    assert "source_asset_sha256 is stale" in result.reason


def test_detects_source_a_b_a_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset(tmp_path)
    original_bytes = asset.read_bytes()
    plan = _plan(tmp_path, asset)
    original_apply = stage_metadata._apply_plan

    def mutate_source(*args: Any, **kwargs: Any) -> None:
        original_apply(*args, **kwargs)
        asset.write_bytes(original_bytes + b"temporary mutation")
        asset.write_bytes(original_bytes)

    monkeypatch.setattr(stage_metadata, "_apply_plan", mutate_source)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert "source asset" in result.reason and "during authoring" in result.reason


def test_reverifies_inputs_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset(tmp_path)
    original_bytes = asset.read_bytes()
    plan = _plan(tmp_path, asset)
    output_dir = tmp_path.parent / f"{tmp_path.name}-out"
    original_publish = stage_metadata._publish_bundle

    def mutate_after_publish(*args: Any, **kwargs: Any) -> Any:
        published = original_publish(*args, **kwargs)
        asset.write_bytes(original_bytes + b"temporary mutation")
        asset.write_bytes(original_bytes)
        return published

    monkeypatch.setattr(stage_metadata, "_publish_bundle", mutate_after_publish)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset, plan_path=plan, output_dir=output_dir
    )
    assert not result.passed
    assert "source asset" in result.reason and "during authoring" in result.reason
    publish_root = output_dir / "stage-metadata"
    assert publish_root.is_dir()
    assert not list(publish_root.iterdir())


def test_rejects_source_reached_through_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    asset = _asset(real)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    linked_asset = (linked / asset.name).absolute()
    plan = _plan(tmp_path, linked_asset)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=linked_asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )
    assert not result.passed
    assert "path contains a symlink" in result.reason


def test_repeat_is_deterministic_and_streams_unbounded_members(tmp_path: Path) -> None:
    asset = _asset(tmp_path, extra_bytes=3 * 1024 * 1024)
    plan = _plan(tmp_path, asset)
    output_dir = tmp_path.parent / f"{tmp_path.name}-out"
    first = author_gate3a_stage_metadata_derivative(
        asset_path=asset, plan_path=plan, output_dir=output_dir
    )
    second = author_gate3a_stage_metadata_derivative(
        asset_path=asset, plan_path=plan, output_dir=output_dir
    )
    assert first.passed and second.passed
    assert _sha(first.output_path) == _sha(second.output_path)
    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    assert second.report["publication_outcome"] == "reused"
    with zipfile.ZipFile(first.output_path) as archive:
        assert archive.getinfo("payload.bin").file_size == 3 * 1024 * 1024


def test_publication_is_atomic_on_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset(tmp_path)
    plan = _plan(tmp_path, asset)
    output_dir = tmp_path.parent / f"{tmp_path.name}-out"
    original_rename = os.rename

    def fail_bundle_rename(source: Any, target: Any, *args: Any, **kwargs: Any) -> None:
        if ".stage-metadata-bundle-" in str(source):
            raise OSError("injected publication failure")
        original_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(stage_metadata.os, "rename", fail_bundle_rename)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset, plan_path=plan, output_dir=output_dir
    )
    assert not result.passed
    assert "injected publication failure" in result.reason
    publish_root = output_dir / "stage-metadata"
    assert not publish_root.exists() or not list(publish_root.iterdir())


def test_publication_rolls_back_on_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset(tmp_path)
    plan = _plan(tmp_path, asset)
    output_dir = tmp_path.parent / f"{tmp_path.name}-out"
    original_fsync_directory = stage_metadata._fsync_directory
    failed = False

    def fail_first_publish_root_fsync(path: Path) -> None:
        nonlocal failed
        if path.name == "stage-metadata" and not failed:
            failed = True
            raise OSError("injected publication fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        stage_metadata, "_fsync_directory", fail_first_publish_root_fsync
    )
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset, plan_path=plan, output_dir=output_dir
    )
    assert not result.passed
    assert "injected publication fsync failure" in result.reason
    publish_root = output_dir / "stage-metadata"
    assert publish_root.is_dir()
    assert not list(publish_root.iterdir())


def test_missing_openusd_returns_blocked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def fail_openusd_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pxr":
            raise ImportError("injected missing OpenUSD")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_openusd_import)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=tmp_path / "asset.usdz",
        plan_path=tmp_path / "plan.json",
        output_dir=tmp_path / "out",
    )
    assert result.status == "BLOCKED"
    assert not result.passed
    assert "injected missing OpenUSD" in result.reason


def test_blocks_nonfinite_authored_linear_quantity(tmp_path: Path) -> None:
    asset = _asset(tmp_path, nonfinite_linear=True)
    plan = _plan(tmp_path, asset)

    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )

    assert result.status == "BLOCKED"
    assert not result.passed
    assert "non-finite authored linear quantity" in result.reason
    assert "physics:upperLimit" in result.reason


def test_openusd_error_returns_blocked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset(tmp_path)
    plan = _plan(tmp_path, asset)

    def fail_inspection(**_kwargs: Any) -> dict[str, Any]:
        raise Tf.ErrorException("injected OpenUSD error")

    monkeypatch.setattr(stage_metadata, "_inspect_and_validate", fail_inspection)
    result = author_gate3a_stage_metadata_derivative(
        asset_path=asset,
        plan_path=plan,
        output_dir=tmp_path.parent / f"{tmp_path.name}-out",
    )

    assert result.status == "BLOCKED"
    assert not result.passed
    assert "injected OpenUSD error" in result.reason
