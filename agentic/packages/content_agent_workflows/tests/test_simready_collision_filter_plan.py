# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact-plan OpenUSD collision-filter derivatives."""

from __future__ import annotations

import builtins
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

pytest.importorskip("pxr")
from pxr import Sdf, Tf, Usd, UsdGeom, UsdPhysics  # noqa: E402

import content_agent_workflows.simready.collision_filter_plan as filter_module  # noqa: E402
from content_agent_workflows.simready import (  # noqa: E402
    COLLISION_FILTER_PLAN_SCHEMA_VERSION,
    CollisionFilterPlan,
    author_collision_filter_derivative,
    filtered_pair_is_authored,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_body(
    stage: Any,
    path: str,
    *,
    collision: bool = True,
    rigid_body: bool = True,
) -> Any:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    if rigid_body:
        UsdPhysics.RigidBodyAPI.Apply(prim)
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    return prim


def _write_asset(
    tmp_path: Path,
    *,
    body_options: dict[str, dict[str, bool]] | None = None,
) -> tuple[Path, Path]:
    package_root = tmp_path / "package"
    package_root.mkdir()
    asset_path = package_root / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset_path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    world.CreateAttribute("test:unchanged", Sdf.ValueTypeNames.String).Set("sentinel")
    body_options = body_options or {"A": {}, "B": {}, "C": {}}
    for name, options in body_options.items():
        _add_body(stage, f"/World/{name}", **options)
    assert stage.GetRootLayer().Save()
    return package_root, asset_path


def _write_evidence(tmp_path: Path) -> Path:
    evidence_path = tmp_path / "machine-collision-results.json"
    evidence_path.write_text(
        json.dumps(
            {
                "validator": "Gate3A.NonAdjacentCollisionMeshesDoNotClash",
                "pairs": [["/World/A", "/World/B"]],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return evidence_path.resolve()


def _plan_payload(
    *,
    asset_path: Path,
    evidence_path: Path,
    pairs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": COLLISION_FILTER_PLAN_SCHEMA_VERSION,
        "source_asset_path": str(asset_path.resolve()),
        "source_asset_sha256": _sha256(asset_path),
        "provenance": {
            "approved_by": "asset-owner",
            "approval_reference": "issue-529-machine-review",
            "evidence": [
                {
                    "kind": "gate3a_validation",
                    "artifact_path": str(evidence_path.resolve()),
                    "artifact_sha256": _sha256(evidence_path),
                }
            ],
        },
        "pairs": pairs or [{"body_a_path": "/World/B", "body_b_path": "/World/A"}],
    }


def _write_plan(
    tmp_path: Path,
    *,
    asset_path: Path,
    pairs: list[dict[str, str]] | None = None,
    mutate: Any | None = None,
) -> tuple[Path, Path]:
    evidence_path = _write_evidence(tmp_path)
    payload = _plan_payload(
        asset_path=asset_path,
        evidence_path=evidence_path,
        pairs=pairs,
    )
    if mutate is not None:
        mutate(payload)
    plan_path = tmp_path / "collision-filter-plan.json"
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan_path.resolve(), evidence_path


def _author(
    tmp_path: Path,
    *,
    asset_path: Path,
    package_root: Path,
    plan_path: Path,
):
    return author_collision_filter_derivative(
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
        output_dir=tmp_path / "output",
    )


def test_authors_canonical_one_way_pair_and_receipt(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    source_bytes = asset_path.read_bytes()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    assert result.status == "AUTHORED"
    assert result.receipt_path is not None and result.receipt_path.is_file()
    assert asset_path.read_bytes() == source_bytes
    stage = Usd.Stage.Open(str(result.output_path))
    assert stage
    body_a = stage.GetPrimAtPath("/World/A")
    body_b = stage.GetPrimAtPath("/World/B")
    assert body_a.HasAPI(UsdPhysics.FilteredPairsAPI)
    assert body_a.GetRelationship("physics:filteredPairs").GetTargets() == [
        Sdf.Path("/World/B")
    ]
    assert not body_b.GetRelationship("physics:filteredPairs")
    assert filtered_pair_is_authored(stage, "/World/A", "/World/B")
    assert filtered_pair_is_authored(stage, "/World/B", "/World/A")
    assert stage.GetPrimAtPath("/World").GetAttribute("test:unchanged").Get() == (
        "sentinel"
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["canonical_representation"] == "one_way_lexicographic"
    assert receipt["canonical_pairs"] == [
        {"source_body_path": "/World/A", "target_body_path": "/World/B"}
    ]
    assert receipt["evidence_artifact_integrity_verified"] is True
    assert "machine_evidence_verified" not in receipt
    assert receipt["geometry_and_topology_preserved"] is True
    assert receipt["dependencies_preserved"] is True


def test_canonicalizes_reverse_pair_and_preserves_other_targets(
    tmp_path: Path,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    stage = Usd.Stage.Open(str(asset_path))
    body_b = stage.GetPrimAtPath("/World/B")
    api = UsdPhysics.FilteredPairsAPI.Apply(body_b)
    assert api.CreateFilteredPairsRel().SetTargets(
        [Sdf.Path("/World/A"), Sdf.Path("/World/C")]
    )
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    output = Usd.Stage.Open(str(result.output_path))
    assert output.GetPrimAtPath("/World/A").GetRelationship(
        "physics:filteredPairs"
    ).GetTargets() == [Sdf.Path("/World/B")]
    assert output.GetPrimAtPath("/World/B").GetRelationship(
        "physics:filteredPairs"
    ).GetTargets() == [Sdf.Path("/World/C")]


def test_preserves_source_authored_filter_and_api_list_ops(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    stage = Usd.Stage.Open(str(asset_path))
    body_b = stage.GetPrimAtPath("/World/B")
    api = UsdPhysics.FilteredPairsAPI.Apply(body_b)
    relationship = api.CreateFilteredPairsRel()
    assert relationship.SetTargets([Sdf.Path("/World/A"), Sdf.Path("/World/C")])
    root_layer = stage.GetRootLayer()
    body_b_spec = root_layer.GetPrimAtPath(Sdf.Path("/World/B"))
    relationship_spec = body_b_spec.GetPropertyAtPath(
        Sdf.Path("/World/B.physics:filteredPairs")
    )
    relationship_spec.targetPathList.ClearEdits()
    relationship_spec.targetPathList.Prepend(Sdf.Path("/World/A"))
    relationship_spec.targetPathList.Append(Sdf.Path("/World/C"))
    source_api_schemas = body_b_spec.GetInfo("apiSchemas")
    source_non_filter_apis = [
        str(item)
        for item in source_api_schemas.prependedItems
        if str(item) != "PhysicsFilteredPairsAPI"
    ]
    assert root_layer.Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    output_layer = Sdf.Layer.FindOrOpen(str(result.output_path))
    output_body_b = output_layer.GetPrimAtPath(Sdf.Path("/World/B"))
    output_targets = output_body_b.GetPropertyAtPath(
        Sdf.Path("/World/B.physics:filteredPairs")
    ).GetInfo("targetPaths")
    assert list(output_targets.prependedItems) == []
    assert list(output_targets.appendedItems) == [Sdf.Path("/World/C")]
    assert list(output_targets.deletedItems) == [Sdf.Path("/World/A")]
    output_api_schemas = output_body_b.GetInfo("apiSchemas")
    assert [
        str(item)
        for item in output_api_schemas.prependedItems
        if str(item) != "PhysicsFilteredPairsAPI"
    ] == source_non_filter_apis


def test_already_canonical_pair_records_no_changes(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    stage = Usd.Stage.Open(str(asset_path))
    api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/World/A"))
    assert api.CreateFilteredPairsRel().AddTarget(Sdf.Path("/World/B"))
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    assert result.report["changes"] == []


def test_authors_pair_with_unicode_prim_paths(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(
        tmp_path,
        body_options={"Café": {}, "箱": {}},
    )
    plan_path, _evidence_path = _write_plan(
        tmp_path,
        asset_path=asset_path,
        pairs=[
            {
                "body_a_path": "/World/箱",
                "body_b_path": "/World/Café",
            }
        ],
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    output = Usd.Stage.Open(str(result.output_path))
    assert filtered_pair_is_authored(output, "/World/Café", "/World/箱")


def test_preserves_nested_package_dependency_bytes(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    dependency_dir = package_root / "deps"
    dependency_dir.mkdir()
    dependency_path = dependency_dir / "marker.usda"
    dependency = Usd.Stage.CreateNew(str(dependency_path))
    UsdGeom.Xform.Define(dependency, "/DependencyMarker")
    assert dependency.GetRootLayer().Save()
    dependency_bytes = dependency_path.read_bytes()
    root_layer = Sdf.Layer.FindOrOpen(str(asset_path))
    root_layer.subLayerPaths.append("deps/marker.usda")
    assert root_layer.Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    output_dependency = result.output_path.parent / "deps" / "marker.usda"
    assert output_dependency.read_bytes() == dependency_bytes
    output = Usd.Stage.Open(str(result.output_path))
    assert output.GetPrimAtPath("/DependencyMarker")


def test_streams_and_preserves_usdz_without_member_size_guard(tmp_path: Path) -> None:
    package_root, root_path = _write_asset(tmp_path)
    dependency = package_root / "dependency.usda"
    dependency.write_text('#usda 1.0\ndef Xform "Dependency" {}\n', encoding="utf-8")
    root_layer = Sdf.Layer.FindOrOpen(str(root_path))
    root_layer.subLayerPaths.append("dependency.usda")
    assert root_layer.Save()
    blob = package_root / "large-evidence.bin"
    blob.write_bytes(b"x" * (3 * 1024 * 1024))
    usdz_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(root_path, "asset.usda")
        archive.write(dependency, "dependency.usda")
        archive.write(blob, "large-evidence.bin")
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=usdz_path)

    result = author_collision_filter_derivative(
        asset_path=usdz_path,
        plan_path=plan_path,
        output_dir=tmp_path / "output",
    )
    replay = author_collision_filter_derivative(
        asset_path=usdz_path,
        plan_path=plan_path,
        output_dir=tmp_path / "output",
    )
    clean_run = author_collision_filter_derivative(
        asset_path=usdz_path,
        plan_path=plan_path,
        output_dir=tmp_path / "other-output",
    )

    assert result.passed
    assert replay.passed
    assert clean_run.passed
    assert replay.report["publication_outcome"] == "cache_hit"
    assert result.receipt_path is not None
    assert replay.receipt_path is not None
    assert clean_run.receipt_path is not None
    assert result.receipt_path.read_bytes() == replay.receipt_path.read_bytes()
    assert result.receipt_path.read_bytes() == clean_run.receipt_path.read_bytes()
    assert ".collision-filter-source-" not in result.receipt_path.read_text(
        encoding="utf-8"
    )
    assert (result.output_path.parent / "dependency.usda").read_bytes() == (
        dependency.read_bytes()
    )
    assert (result.output_path.parent / "large-evidence.bin").stat().st_size == (
        3 * 1024 * 1024
    )


@pytest.mark.parametrize(
    "members",
    [
        ("textures", "textures/albedo.png"),
        ("textures/albedo.png", "textures"),
    ],
)
def test_rejects_usdz_file_ancestor_collision_before_extraction(
    tmp_path: Path,
    members: tuple[str, str],
) -> None:
    usdz_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("asset.usda", "#usda 1.0\n")
        for member in members:
            archive.writestr(member, b"payload")
    extraction_dir = tmp_path / "extract"
    extraction_dir.mkdir()

    with pytest.raises(
        ValueError,
        match=(
            "USDZ contains a file/member ancestor collision: "
            "textures and textures/albedo.png"
        ),
    ):
        filter_module._extract_usdz_without_size_limit(
            asset_path=usdz_path,
            extraction_dir=extraction_dir,
        )

    assert list(extraction_dir.iterdir()) == []


def test_rejects_usdz_file_directory_collision_before_extraction(
    tmp_path: Path,
) -> None:
    usdz_path = tmp_path / "asset.usdz"
    with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("asset.usda", "#usda 1.0\n")
        archive.writestr("textures/", b"")
        archive.writestr("textures", b"payload")
    extraction_dir = tmp_path / "extract"
    extraction_dir.mkdir()

    with pytest.raises(ValueError, match="USDZ contains a file/directory collision"):
        filter_module._extract_usdz_without_size_limit(
            asset_path=usdz_path,
            extraction_dir=extraction_dir,
        )

    assert list(extraction_dir.iterdir()) == []


def test_rejects_package_paths_before_recursive_io_can_overflow(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "deep-package"
    package_root.mkdir()
    deepest = package_root
    for _index in range(filter_module._MAX_PACKAGE_PATH_DEPTH + 1):
        deepest /= "d"
        deepest.mkdir()

    with pytest.raises(ValueError, match="maximum path depth"):
        filter_module._validate_tree_layout(package_root)

    usdz_path = tmp_path / "deep.usdz"
    deep_member = "/".join(
        ["d"] * filter_module._MAX_PACKAGE_PATH_DEPTH + ["payload.bin"]
    )
    with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("asset.usda", "#usda 1.0\n")
        archive.writestr(deep_member, b"payload")
    extraction_dir = tmp_path / "extract-deep"
    extraction_dir.mkdir()

    with pytest.raises(ValueError, match="maximum package path depth"):
        filter_module._extract_usdz_without_size_limit(
            asset_path=usdz_path,
            extraction_dir=extraction_dir,
        )
    assert list(extraction_dir.iterdir()) == []


def test_source_package_propagates_interrupt_and_cleans_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    def interrupt_extraction(**_kwargs: Any) -> Path:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        filter_module,
        "_extract_usdz_without_size_limit",
        interrupt_extraction,
    )

    with pytest.raises(KeyboardInterrupt):
        filter_module._source_package(
            asset_path=tmp_path / "asset.usdz",
            package_root=None,
            output_dir=output_dir,
        )

    assert list(output_dir.glob(".collision-filter-source-*")) == []


@pytest.mark.parametrize(
    ("body_options", "pairs", "message"),
    [
        (
            {"A": {}, "B": {}},
            [{"body_a_path": "/World/A", "body_b_path": "/World/Missing"}],
            "does not exist",
        ),
        (
            {"A": {"collision": False}, "B": {}},
            [{"body_a_path": "/World/A", "body_b_path": "/World/B"}],
            "owns no active PhysicsCollisionAPI prim",
        ),
        (
            {"A": {"rigid_body": False}, "B": {}},
            [{"body_a_path": "/World/A", "body_b_path": "/World/B"}],
            "lacks PhysicsRigidBodyAPI",
        ),
    ],
)
def test_rejects_invalid_body_prims(
    tmp_path: Path,
    body_options: dict[str, dict[str, bool]],
    pairs: list[dict[str, str]],
    message: str,
) -> None:
    package_root, asset_path = _write_asset(
        tmp_path,
        body_options=body_options,
    )
    plan_path, _evidence_path = _write_plan(
        tmp_path,
        asset_path=asset_path,
        pairs=pairs,
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert message in result.reason


def test_accepts_rigid_bodies_with_owned_collider_children(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(
        tmp_path,
        body_options={
            "A": {"collision": False},
            "B": {"collision": False},
        },
    )
    stage = Usd.Stage.Open(str(asset_path))
    for name in ("A", "B"):
        collider = UsdGeom.Cube.Define(stage, f"/World/{name}/Collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    output = Usd.Stage.Open(str(result.output_path))
    assert filtered_pair_is_authored(output, "/World/A", "/World/B")
    assert not output.GetPrimAtPath("/World/A/Collider").HasAPI(
        UsdPhysics.FilteredPairsAPI
    )


def test_nested_rigid_body_does_not_supply_parent_collider_ownership(
    tmp_path: Path,
) -> None:
    package_root, asset_path = _write_asset(
        tmp_path,
        body_options={"A": {"collision": False}, "B": {}},
    )
    stage = Usd.Stage.Open(str(asset_path))
    _add_body(stage, "/World/A/Nested")
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "owns no active PhysicsCollisionAPI prim: /World/A" in result.reason


@pytest.mark.parametrize("missing_endpoint", ["/World/A", "/World/B"])
def test_canonical_target_map_rejects_missing_endpoints(
    missing_endpoint: str,
) -> None:
    initial_targets = {
        endpoint: ()
        for endpoint in ("/World/A", "/World/B")
        if endpoint != missing_endpoint
    }

    with pytest.raises(
        ValueError, match="initial targets are missing endpoints"
    ) as exc:
        filter_module._canonical_target_map(
            initial_targets=initial_targets,
            canonical_pairs=(("/World/A", "/World/B"),),
        )

    assert str(exc.value).endswith(missing_endpoint)


def test_rejects_directly_joint_adjacent_pair(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    stage = Usd.Stage.Open(str(asset_path))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
    assert joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/A")])
    assert joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/B")])
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "directly joint-adjacent" in result.reason


def test_rejects_instance_proxy_edit(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    prototype_path = package_root / "prototype.usda"
    prototype = Usd.Stage.CreateNew(str(prototype_path))
    model = UsdGeom.Xform.Define(prototype, "/Model").GetPrim()
    prototype.SetDefaultPrim(model)
    _add_body(prototype, "/Model/Body")
    assert prototype.GetRootLayer().Save()

    asset_path = package_root / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset_path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    for name in ("A", "B"):
        instance = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        instance.GetReferences().AddReference("prototype.usda", "/Model")
        instance.SetInstanceable(True)
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(
        tmp_path,
        asset_path=asset_path,
        pairs=[
            {
                "body_a_path": "/World/A/Body",
                "body_b_path": "/World/B/Body",
            }
        ],
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "instance or prototype" in result.reason


@pytest.mark.parametrize(
    ("pairs", "message"),
    [
        (
            [{"body_a_path": "/World/A", "body_b_path": "/World/A"}],
            "cannot reference one body twice",
        ),
        (
            [
                {"body_a_path": "/World/A", "body_b_path": "/World/B"},
                {"body_a_path": "/World/B", "body_b_path": "/World/A"},
            ],
            "duplicate unordered body pair",
        ),
    ],
)
def test_rejects_self_and_duplicate_unordered_pairs(
    tmp_path: Path,
    pairs: list[dict[str, str]],
    message: str,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(
        tmp_path,
        asset_path=asset_path,
        pairs=pairs,
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert message in result.reason


def test_strict_model_rejects_extra_fields(tmp_path: Path) -> None:
    _package_root, asset_path = _write_asset(tmp_path)
    evidence_path = _write_evidence(tmp_path)
    payload = _plan_payload(asset_path=asset_path, evidence_path=evidence_path)
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        CollisionFilterPlan.model_validate(payload)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    evidence_path = _write_evidence(tmp_path)
    payload = _plan_payload(asset_path=asset_path, evidence_path=evidence_path)
    valid = json.dumps(payload)
    duplicate = valid[:-1] + ', "pairs": []}'
    plan_path = tmp_path / "collision-filter-plan.json"
    plan_path.write_text(duplicate, encoding="utf-8")

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "duplicate JSON key: pairs" in result.reason


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(source_asset_sha256="0" * 64),
            "source_asset_sha256 is stale",
        ),
        (
            lambda payload: payload.update(source_asset_path="/not/the/source.usda"),
            "source_asset_path does not match",
        ),
        (
            lambda payload: payload["provenance"]["evidence"][0].update(
                artifact_sha256="0" * 64
            ),
            "Evidence artifact_sha256 is stale",
        ),
    ],
)
def test_rejects_stale_source_and_evidence_identity(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(
        tmp_path,
        asset_path=asset_path,
        mutate=mutation,
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert message in result.reason


def test_rejects_source_symlink(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    symlink = tmp_path / "asset-link.usda"
    symlink.symlink_to(asset_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=symlink)

    result = _author(
        tmp_path,
        asset_path=symlink,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "Source asset cannot be a symlink" in result.reason


def test_missing_deferred_pxr_dependency_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_pxr_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pxr":
            raise ModuleNotFoundError("No module named 'pxr'", name="pxr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_pxr_import)

    with pytest.raises(ModuleNotFoundError, match="No module named 'pxr'"):
        author_collision_filter_derivative(
            asset_path=tmp_path / "asset.usda",
            plan_path=tmp_path / "plan.json",
            output_dir=tmp_path / "output",
        )


def test_fails_closed_for_openusd_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    def fail_authoring(**_kwargs: Any) -> list[dict[str, Any]]:
        raise Tf.ErrorException("simulated OpenUSD runtime failure")

    monkeypatch.setattr(
        filter_module,
        "_author_filtered_pairs",
        fail_authoring,
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert result.status == "BLOCKED"
    assert "simulated OpenUSD runtime failure" in result.reason
    assert list((tmp_path / "output").rglob(".collision-filter-build-*")) == []


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_does_not_mask_programming_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    def fail_authoring(**_kwargs: Any) -> list[dict[str, Any]]:
        raise error_type("simulated programming error")

    monkeypatch.setattr(
        filter_module,
        "_author_filtered_pairs",
        fail_authoring,
    )

    with pytest.raises(error_type, match="simulated programming error"):
        _author(
            tmp_path,
            asset_path=asset_path,
            package_root=package_root,
            plan_path=plan_path,
        )
    assert list((tmp_path / "output").rglob(".collision-filter-build-*")) == []


@pytest.mark.parametrize("mutated_input", ["source", "plan", "evidence"])
def test_fails_closed_when_bound_input_changes_during_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_input: str,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    original_author = filter_module._author_filtered_pairs

    def mutate_after_authoring(*args: Any, **kwargs: Any) -> Any:
        changes = original_author(*args, **kwargs)
        path = {
            "source": asset_path,
            "plan": plan_path,
            "evidence": evidence_path,
        }[mutated_input]
        path.write_bytes(path.read_bytes() + b"\n")
        return changes

    monkeypatch.setattr(
        filter_module,
        "_author_filtered_pairs",
        mutate_after_authoring,
    )

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    expected = {
        "source": "Source asset changed",
        "plan": "plan changed",
        "evidence": "evidence changed",
    }[mutated_input]
    assert expected in result.reason


def test_rejects_dependency_outside_package_root(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    outside = tmp_path / "outside.usda"
    outside.write_text('#usda 1.0\ndef Xform "Outside" {}\n', encoding="utf-8")
    root_layer = Sdf.Layer.FindOrOpen(str(asset_path))
    root_layer.subLayerPaths.append("../outside.usda")
    assert root_layer.Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "outside package" in result.reason


def test_rejects_unresolved_asset_array_that_matches_asset_identifier(
    tmp_path: Path,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    stage = Usd.Stage.Open(str(asset_path))
    world = stage.GetPrimAtPath("/World")
    world.SetAssetInfoByKey("identifier", Sdf.AssetPath("missing.bin"))
    assert world.CreateAttribute(
        "test:dependencies",
        Sdf.ValueTypeNames.AssetArray,
        custom=True,
    ).Set([Sdf.AssetPath("missing.bin")])
    assert stage.GetRootLayer().Save()
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "USD dependency closure is unresolved: missing.bin" in result.reason


def test_rejects_unplanned_root_layer_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    original_author = filter_module._author_filtered_pairs

    def mutate_unrelated_attribute(*args: Any, **kwargs: Any) -> Any:
        changes = original_author(*args, **kwargs)
        stage = kwargs["stage"]
        stage.GetPrimAtPath("/World").GetAttribute("test:unchanged").Set("changed")
        return changes

    monkeypatch.setattr(
        filter_module,
        "_author_filtered_pairs",
        mutate_unrelated_attribute,
    )
    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "changed root-layer content outside" in result.reason


def test_rejects_unplanned_api_schema_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    original_author = filter_module._author_filtered_pairs

    def remove_rigid_body_api(*args: Any, **kwargs: Any) -> Any:
        result = original_author(*args, **kwargs)
        prim = kwargs["stage"].GetPrimAtPath("/World/A")
        assert prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        return result

    monkeypatch.setattr(
        filter_module,
        "_author_filtered_pairs",
        remove_rigid_body_api,
    )
    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "apiSchemas list-op changed" in result.reason


def test_second_run_reuses_identical_tree_and_receipt(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    first = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )
    assert first.passed and first.receipt_path is not None
    receipt_bytes = first.receipt_path.read_bytes()

    second = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert second.passed
    assert second.output_path == first.output_path
    assert second.report["publication_outcome"] == "cache_hit"
    assert second.receipt_path == first.receipt_path
    assert second.receipt_path.read_bytes() == receipt_bytes


def test_distinct_plans_for_identical_output_have_plan_scoped_receipts(
    tmp_path: Path,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    first_plan = tmp_path / "plan-a.json"
    plan_path.rename(first_plan)
    second_payload = json.loads(first_plan.read_text(encoding="utf-8"))
    second_payload["provenance"]["approval_reference"] = "second-owner-review"
    second_plan = tmp_path / "plan-b.json"
    second_plan.write_text(
        json.dumps(second_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=first_plan,
    )
    second = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=second_plan,
    )

    assert first.passed and first.receipt_path is not None
    assert second.passed and second.receipt_path is not None
    assert second.report["publication_outcome"] == "cache_hit"
    assert second.output_path == first.output_path
    assert second.receipt_path != first.receipt_path
    assert first.report["plan_sha256"] in first.receipt_path.name
    assert second.report["plan_sha256"] in second.receipt_path.name


def test_verifies_concurrent_winner_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    def publish_concurrent_copy(source: Path, target: Path) -> None:
        shutil.copytree(source, target)
        raise OSError("simulated concurrent winner")

    monkeypatch.setattr(filter_module, "_atomic_rename", publish_concurrent_copy)
    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    assert result.report["publication_outcome"] == "concurrent_reuse"
    assert result.output_path.is_file()


def test_preserves_publication_oserror_when_no_concurrent_winner_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    def fail_rename(_source: Path, _target: Path) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(filter_module, "_atomic_rename", fail_rename)
    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not result.passed
    assert "No space left on device" in result.reason
    assert list((tmp_path / "output").rglob(".collision-filter-build-*")) == []


def test_rejects_mutated_content_addressed_winner(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    first = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )
    assert first.passed
    first.output_path.write_bytes(first.output_path.read_bytes() + b"\n# mutation\n")

    second = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not second.passed
    assert "Existing collision-filter output failed identity check" in second.reason


def test_rejects_conflicting_receipt_bytes(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    first = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )
    assert first.passed and first.receipt_path is not None
    first.receipt_path.write_text("{}\n", encoding="utf-8")

    second = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert not second.passed
    assert "receipt has conflicting bytes" in second.reason


def test_can_copy_read_only_prior_derivative(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)
    asset_path.chmod(0o444)
    package_root.chmod(0o555)

    result = _author(
        tmp_path,
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
    )

    assert result.passed
    assert result.output_path.is_file()


def test_rejects_output_inside_source_package(tmp_path: Path) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    result = author_collision_filter_derivative(
        asset_path=asset_path,
        package_root=package_root,
        plan_path=plan_path,
        output_dir=package_root / "generated",
    )

    assert not result.passed
    assert "output cannot be located inside the source package" in result.reason
    assert not (package_root / "generated").exists()


def test_collision_filter_cli_authors_and_reports_blocked_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_root, asset_path = _write_asset(tmp_path)
    plan_path, _evidence_path = _write_plan(tmp_path, asset_path=asset_path)

    assert (
        filter_module.main(
            [
                str(asset_path),
                str(plan_path),
                "--package-root",
                str(package_root),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )
    authored = json.loads(capsys.readouterr().out)
    assert authored["status"] == "AUTHORED"
    assert authored["passed"] is True
    assert Path(authored["output_path"]).is_file()
    assert Path(authored["receipt_path"]).is_file()

    assert (
        filter_module.main(
            [
                str(asset_path),
                str(tmp_path / "missing-plan.json"),
                "--package-root",
                str(package_root),
                "--output-dir",
                str(tmp_path / "blocked-output"),
            ]
        )
        == 1
    )
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "BLOCKED"
    assert blocked["passed"] is False
    assert blocked["receipt_path"] is None
