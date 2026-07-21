# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict SDF collision-filter family completion."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402

import content_agent_workflows.simready.collision_filter_completion as completion_module  # noqa: E402
from content_agent_workflows.simready import (  # noqa: E402
    COLLISION_FILTER_COMPLETION_SCHEMA_VERSION,
    CollisionFilterCompletionError,
    CollisionFilterCompletionRequestV1,
    PhysicsColliderApproximationPlanV1,
    PhysicsMaterialPlanV1,
    PhysicsProfileApprovalV1,
    PhysicsProfilePlanError,
    PhysicsProfilePlanV1,
    author_collision_filter_derivative,
    author_physics_profile_plan,
    filtered_pair_is_authored,
    generate_collision_filter_completion_plan,
    inspect_physics_profile_source,
    verify_physics_profile_receipt,
)

_OWNER = "/RootNode/Geometry/seat_01_obj_00/seat_01_mesh_00"
_BASES = tuple(
    f"/RootNode/Geometry/tireBase_0{index}_obj_00/tireBase_0{index}_mesh_00"
    for index in range(1, 5)
)
_TIRES = tuple(
    f"/RootNode/Geometry/tire_0{index}_obj_00/tire_0{index}_mesh_00"
    for index in range(1, 5)
)
_CLASH_MESSAGE = f"Colliding meshes {_OWNER} and {_TIRES[0]} are not adjacent"
_PROFILE_CATEGORIES = (
    "Basic",
    "Omni:Basic",
    "Usd:Performance",
    "Usd:Schema",
    "Omni:Geometry",
    "Omni:Material",
    "Omni:Layout",
    "Omni:Skel",
    "Usd:Physics",
    "Omni:SimReady",
    "IsaacSim.PhysicsRules",
    "IsaacSim.SimReadyAssetRules",
)
_REGISTERED_RULES = {
    "AtomicAsset": [
        "AnchoredAssetPathsChecker",
        "SupportedFileTypesChecker",
        "UsdzUdimLimitationChecker",
    ],
    "Basic": [
        "UsdzPackageValidator",
        "MissingReferenceChecker",
        "PortableAssetPathChecker",
        "StageMetadataChecker",
        "TextureChecker",
        "PrimEncapsulationChecker",
        "NormalMapTextureChecker",
    ],
    "IsaacSim.PhysicsRules": [
        "PhysicsJointHasDriveOrMimicAPI",
        "PhysicsJointMaxVelocity",
        "PhysicsDriveAndJointState",
        "DriveJointValueReasonable",
        "JointHasCorrectTransformAndState",
        "JointHasJointStateAPI",
        "MimicAPICheck",
        "RigidBodyHasMassAPI",
        "RigidBodyHasCollider",
        "NonAdjacentCollisionMeshesDoNotClash",
        "InvisibleCollisionMeshHasPurposeGuide",
        "HasArticulationRoot",
    ],
    "IsaacSim.RobotRules": [
        "RobotNaming",
        "CleanFolder",
        "NoOverrides",
        "RobotSchema",
        "JointsExist",
        "LinksExist",
        "ThumbnailExists",
        "CheckRobotRelationships",
        "VerifyRobotPhysicsAttributesSourceLayer",
        "VerifyRobotPhysicsSchemaSourceLayer",
    ],
    "IsaacSim.SimReadyAssetRules": [
        "NoNestedMaterials",
        "MaterialsOnTopLevelOnly",
    ],
    "Omni:Basic": [
        "KindChecker",
        "ExtentsChecker",
        "TypeChecker",
        "LayerSpecChecker",
        "UnicodeNameChecker",
    ],
    "Omni:Geometry": [
        "IndexedPrimvarChecker",
        "ManifoldChecker",
        "NormalsExistChecker",
        "NormalsValidChecker",
        "NormalsWindingsChecker",
        "GaussianSplatSchemaChecker",
        "SubdivisionSchemeChecker",
        "UnusedMeshTopologyChecker",
        "UnusedPrimvarChecker",
        "ValidateTopologyChecker",
        "WeldChecker",
        "ZeroAreaFaceChecker",
    ],
    "Omni:Layout": [
        "OmniDefaultPrimChecker",
        "OmniOrphanedPrimChecker",
    ],
    "Omni:Material": [
        "MaterialPathChecker",
        "MaterialOutOfScopeChecker",
        "MaterialOldMdlSchemaChecker",
        "ShaderImplementationSourceChecker",
        "OmniMaterialUsdPreviewSurfaceChecker",
    ],
    "Omni:SimReady": [
        "GroundTruthCapabilityChecker",
        "NonVisualSensorCapabilityChecker",
        "VisualSensorCapabilityChecker",
        "PhysxRigidBodyChecker",
        "PhysxArticulationChecker",
        "AssetOriginPositioningChecker",
        "ContainsMeshChecker",
        "HierarchyHasRootChecker",
        "RootPrimXformableChecker",
        "UpAxisZChecker",
        "MetersPerUnit1Checker",
    ],
    "Omni:Skel": ["OmniSkelUpgradeChecker"],
    "Usd:Performance": [
        "AlmostExtremeExtentChecker",
        "PointsPrecisionErrorChecker",
        "PointsPrecisionWarningChecker",
        "UsdAsciiPerformanceChecker",
    ],
    "Usd:Physics": [
        "RigidBodyChecker",
        "ColliderChecker",
        "PhysicsJointChecker",
        "ArticulationChecker",
        "MassChecker",
    ],
    "Usd:Schema": [
        "UsdDanglingMaterialBinding",
        "UsdGeomSubsetChecker",
        "UsdLuxSchemaChecker",
        "UsdMaterialBindingApi",
        "SkelBindingAPIAppliedChecker",
    ],
}
_NOT_APPLICABLE_RULES = {
    "GroundTruthCapabilityChecker",
    "NonVisualSensorCapabilityChecker",
    "VisualSensorCapabilityChecker",
}
_ENABLED_RULES = {
    category: [
        rule
        for rule in _REGISTERED_RULES[category]
        if rule not in _NOT_APPLICABLE_RULES
    ]
    for category in _PROFILE_CATEGORIES
}


@dataclass(frozen=True)
class _Evidence:
    request: CollisionFilterCompletionRequestV1
    baseline_asset: Path
    post_asset: Path
    baseline_results: Path
    post_results: Path
    physics_plan: Path
    physics_receipt: Path
    source_dependency: Path | None
    output_dependency: Path | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + suffix
    ).encode("ascii")


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _artifact_dependency_fields(asset: Path) -> dict[str, Any]:
    if asset.suffix.lower() != ".usdz":
        return {
            "artifact_dependency_bundle_schema_version": None,
            "artifact_dependency_bundle_sha256": None,
            "artifact_dependency_bundle_entry_count": None,
        }
    with zipfile.ZipFile(asset) as archive:
        infos = archive.infolist()
        assert infos
        entries: list[dict[str, Any]] = []
        for info in infos:
            digest = hashlib.sha256()
            if not info.is_dir():
                with archive.open(info) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            entries.append(
                {
                    "kind": "package_directory" if info.is_dir() else "package_entry",
                    "path": info.filename,
                    "size": 0 if info.is_dir() else info.file_size,
                    "sha256": digest.hexdigest(),
                }
            )
    manifest = {
        "schema_version": "joint-agent-usd-artifact-dependency-bundle-v3",
        "container": "usdz",
        "root_entry": infos[0].filename,
        "entries": sorted(entries, key=lambda item: str(item["path"])),
    }
    return {
        "artifact_dependency_bundle_schema_version": manifest["schema_version"],
        "artifact_dependency_bundle_sha256": _canonical_json_sha256(manifest),
        "artifact_dependency_bundle_entry_count": len(entries),
    }


def _define_collider(stage: Any, path: str, *, approximation: str) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    prim = mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)


def _define_joint(
    stage: Any,
    path: str,
    body0: str,
    body1: str,
    *,
    fixed: bool = False,
    enabled: bool = True,
    disabled_at_time: bool = False,
) -> None:
    if fixed:
        joint = UsdPhysics.FixedJoint.Define(stage, path)
    else:
        joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    assert joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    assert joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    if not enabled:
        joint.CreateJointEnabledAttr(False)
    if disabled_at_time:
        assert joint.CreateJointEnabledAttr().Set(False, 1.0)


def _write_loose_chair(
    path: Path,
    *,
    approximation: str,
    peer_count: int = 3,
    candidate_filtered: bool = False,
    candidate_is_owner: bool = False,
    mismatched_branch: bool = False,
    inactive_peer: bool = False,
    abstract_peer: bool = False,
    disabled_peer_joint: bool = False,
    time_sampled_disabled_peer_rigid: bool = False,
    time_sampled_disabled_peer_collision: bool = False,
    time_sampled_disabled_peer_joint: bool = False,
    time_sampled_disabled_intermediate_rigid: bool = False,
    time_sampled_disabled_intermediate_collision: bool = False,
    dependency_name: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/RootNode").GetPrim()
    stage.SetDefaultPrim(root)
    for collider in (_OWNER, *_BASES, *_TIRES):
        _define_collider(stage, collider, approximation=approximation)
    sampled_peer = stage.GetPrimAtPath(_TIRES[1])
    if time_sampled_disabled_peer_rigid:
        assert (
            UsdPhysics.RigidBodyAPI(sampled_peer)
            .CreateRigidBodyEnabledAttr()
            .Set(False, 1.0)
        )
    if time_sampled_disabled_peer_collision:
        assert (
            UsdPhysics.CollisionAPI(sampled_peer)
            .CreateCollisionEnabledAttr()
            .Set(False, 1.0)
        )
    sampled_intermediate = stage.GetPrimAtPath(_BASES[1])
    if time_sampled_disabled_intermediate_rigid:
        assert (
            UsdPhysics.RigidBodyAPI(sampled_intermediate)
            .CreateRigidBodyEnabledAttr()
            .Set(False, 1.0)
        )
    if time_sampled_disabled_intermediate_collision:
        assert (
            UsdPhysics.CollisionAPI(sampled_intermediate)
            .CreateCollisionEnabledAttr()
            .Set(False, 1.0)
        )
    owner_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(_OWNER))
    peer_targets = list(_TIRES[1 : 1 + peer_count])
    if candidate_filtered:
        peer_targets.insert(0, _TIRES[0])
    assert owner_api.CreateFilteredPairsRel().SetTargets(
        [Sdf.Path(item) for item in peer_targets]
    )
    if candidate_is_owner:
        candidate_api = UsdPhysics.FilteredPairsAPI.Apply(
            stage.GetPrimAtPath(_TIRES[0])
        )
        assert candidate_api.CreateFilteredPairsRel().SetTargets(
            [Sdf.Path(_BASES[1]), Sdf.Path(_BASES[2])]
        )
    for index, (base, tire) in enumerate(zip(_BASES, _TIRES, strict=True), 1):
        _define_joint(
            stage,
            f"/RootNode/Joints/seat_to_base_{index}",
            _OWNER,
            base,
        )
        _define_joint(
            stage,
            f"/RootNode/Joints/base_to_tire_{index}",
            base,
            tire,
            fixed=mismatched_branch and index == 4,
            enabled=not (disabled_peer_joint and index == 2),
            disabled_at_time=time_sampled_disabled_peer_joint and index == 2,
        )
    if inactive_peer:
        stage.GetPrimAtPath(_TIRES[1]).SetActive(False)
    if abstract_peer:
        stage.GetPrimAtPath(_TIRES[1]).SetSpecifier(Sdf.SpecifierClass)
    if dependency_name is not None:
        dependency = path.parent / dependency_name
        dependency_layer = Sdf.Layer.CreateNew(str(dependency))
        assert dependency_layer
        dependency_layer.customLayerData = {"identity": dependency_name}
        assert dependency_layer.Save()
        stage.GetRootLayer().subLayerPaths.append(dependency.name)
    assert stage.GetRootLayer().Save()


def _package_usdz(loose: Path, package: Path) -> None:
    package.parent.mkdir(parents=True, exist_ok=True)
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(loose)), str(package))


def _issues_row(asset: Path, *, status: str, issues: list[dict[str, Any]]) -> dict:
    counts = Counter(item["severity"] for item in issues)
    return {
        **_artifact_dependency_fields(asset),
        "artifact_sha256": _sha256(asset),
        "hard_failure_count": sum(
            counts.get(severity, 0) for severity in ("ERROR", "FAILURE")
        ),
        "issue_count": len(issues),
        "issues": issues,
        "original_path": str(asset.resolve()),
        "path": str(asset.resolve()),
        "severity_counts": dict(sorted(counts.items())),
        "status": status,
    }


def _write_results(
    path: Path,
    *,
    asset: Path,
    status: str,
    issues: list[dict[str, Any]],
) -> None:
    row = _issues_row(asset, status=status, issues=issues)
    closeout = {
        "schema_version": "test-gate3a-closeout-v1",
        "assets": [
            {
                "asset_id": "test_asset",
                "generated_usd_path": str(asset.resolve()),
                "generated_usd_sha256": _sha256(asset),
                "reference_usd_path": None,
            }
        ],
    }
    row["asset_id"] = "test_asset"
    closeout_path = path.with_name(f"{path.stem}-closeout.json").resolve()
    closeout_path.write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    extension_records = [
        {
            "name": "isaacsim.asset.validation",
            "id": "isaacsim.asset.validation-1.3.5",
            "path": "exts/isaacsim.asset.validation",
        },
        {
            "name": "omni.asset_validator.core",
            "id": "omni.asset_validator.core-1.19.3",
            "path": "extscache/omni.asset_validator.core-1.19.3",
        },
    ]
    results = [row]
    payload = {
        "schema_version": "joint-agent-isaac-sim-asset-validator-v2",
        "created_at": "2026-07-12T00:00:00+00:00",
        "runtime": {
            "python": "/test/isaacsim/bin/python",
            "omni_kit_accept_eula": "YES",
            "kit_launch": {
                "profile": ("joint-rigger-gate3a-standalone-default-experience-v1"),
                "experience": "simulation_app_default",
            },
            "kit_runtime_attestation": None,
            "kit_enabled_extension_inventory": {
                "schema_version": (
                    "joint-rigger-gate3a-enabled-extension-inventory-v1"
                ),
                "extension_count": len(extension_records),
                "extensions_sha256": _canonical_json_sha256(
                    {"extensions": extension_records}
                ),
                "extensions": extension_records,
            },
            "package_versions": dict.fromkeys(
                (
                    "isaacsim",
                    "isaacsim-app",
                    "isaacsim-asset",
                    "isaacsim-extscache-kit",
                    "isaacsim-extscache-physics",
                ),
                "6.0.0.1",
            ),
            "extensions_enabled": {
                "omni.asset_validator.core": True,
                "isaacsim.asset.validation": True,
            },
            "extension_identities": {
                item["name"]: {
                    "id": item["id"],
                    "path": f"/test/{item['path']}",
                }
                for item in extension_records
            },
        },
        "closeout_json": str(closeout_path),
        "closeout_sha256": _canonical_json_sha256(closeout),
        "repo_root": str(path.parent.resolve()),
        "path_prefix_maps": [],
        "selection": {"asset_ids": "all", "kinds": ["generated"]},
        "validation_profile": {
            "name": "articulated-prop-v1",
            "description": "strict articulated prop test profile",
            "status_basis": (
                "ERROR and FAILURE issues from applicable categories fail the "
                "target; WARNING issues produce warning status."
            ),
            "applicable_categories": list(_PROFILE_CATEGORIES),
            "not_applicable_categories": ["AtomicAsset", "IsaacSim.RobotRules"],
            "not_applicable_rules": [
                "GroundTruthCapabilityChecker",
                "NonVisualSensorCapabilityChecker",
                "VisualSensorCapabilityChecker",
            ],
            "missing_categories": [],
        },
        "registered_categories": [
            "AtomicAsset",
            *_PROFILE_CATEGORIES,
            "IsaacSim.RobotRules",
        ],
        "registered_rules_by_category": _REGISTERED_RULES,
        "default_enabled_categories": list(_PROFILE_CATEGORIES),
        "enabled_categories": list(_PROFILE_CATEGORIES),
        "enabled_rules_by_category": _ENABLED_RULES,
        "result_count": len(results),
        "aggregate": {
            "status_counts": {status: 1},
            "severity_counts": dict(
                sorted(Counter(item["severity"] for item in issues).items())
            ),
        },
        "results": results,
        "results_sha256": _canonical_json_sha256({"results": results}),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_profile_receipt(
    profile_root: Path,
    *,
    baseline_asset: Path,
    transition_paths: tuple[str, ...] = (_OWNER, *_BASES, *_TIRES),
) -> tuple[Path, Path, Path]:
    source_identity = inspect_physics_profile_source(baseline_asset)
    collider_paths = tuple(sorted((_OWNER, *_BASES, *_TIRES)))
    transitioned = set(transition_paths)
    plan = PhysicsProfilePlanV1(
        schema_version="content-agent-workflows.physics-profile-plan.v1",
        source_asset_sha256=source_identity.source_asset_sha256,
        source_dependency_bundle_sha256=(
            source_identity.source_dependency_bundle_sha256
        ),
        collider_prim_paths=collider_paths,
        mesh_approximation="sdf",
        physics_material=PhysicsMaterialPlanV1(
            prim_path="/RootNode/TestPhysicsMaterial",
            static_friction=0.5,
            dynamic_friction=0.4,
            restitution=0.15,
            density=1000.0,
        ),
        approval=PhysicsProfileApprovalV1(
            approved=True,
            owner_identity="asset-owner",
            evidence="issue-529-cross-gate-review",
        ),
        collider_approximations=tuple(
            PhysicsColliderApproximationPlanV1(
                prim_path=prim_path,
                operation=(
                    "author_sdf" if prim_path in transitioned else "preserve_existing"
                ),
                source_token="convexHull",
            )
            for prim_path in collider_paths
        ),
    )
    plan_path = profile_root.parent / "physics-profile-plan.json"
    plan_path.write_bytes(
        _canonical_json_bytes(
            plan.model_dump(mode="json", exclude_defaults=True),
            newline=True,
        )
    )
    result = author_physics_profile_plan(baseline_asset, plan_path, profile_root)
    return (
        plan_path.resolve(),
        result.receipt_path.resolve(),
        result.output_asset_path.resolve(),
    )


def _make_evidence(
    tmp_path: Path,
    *,
    use_usdz: bool = True,
    peer_count: int = 3,
    candidate_filtered: bool = False,
    candidate_is_owner: bool = False,
    mismatched_post_branch: bool = False,
    inactive_peer: bool = False,
    abstract_peer: bool = False,
    disabled_peer_joint: bool = False,
    time_sampled_disabled_peer_rigid: bool = False,
    time_sampled_disabled_peer_collision: bool = False,
    time_sampled_disabled_peer_joint: bool = False,
    time_sampled_disabled_intermediate_rigid: bool = False,
    time_sampled_disabled_intermediate_collision: bool = False,
    source_dependency: bool = False,
    output_dependency: bool = False,
    transition_paths: tuple[str, ...] = (_OWNER, *_BASES, *_TIRES),
) -> _Evidence:
    baseline_loose = tmp_path / "source" / "baseline.usda"
    baseline_dependency_name = (
        "baseline-support.usda" if source_dependency or output_dependency else None
    )
    _write_loose_chair(
        baseline_loose,
        approximation="convexHull",
        peer_count=peer_count,
        candidate_filtered=candidate_filtered,
        candidate_is_owner=candidate_is_owner,
        inactive_peer=inactive_peer,
        abstract_peer=abstract_peer,
        disabled_peer_joint=disabled_peer_joint,
        time_sampled_disabled_peer_rigid=time_sampled_disabled_peer_rigid,
        time_sampled_disabled_peer_collision=time_sampled_disabled_peer_collision,
        time_sampled_disabled_peer_joint=time_sampled_disabled_peer_joint,
        time_sampled_disabled_intermediate_rigid=(
            time_sampled_disabled_intermediate_rigid
        ),
        time_sampled_disabled_intermediate_collision=(
            time_sampled_disabled_intermediate_collision
        ),
        mismatched_branch=mismatched_post_branch,
        dependency_name=baseline_dependency_name,
    )
    if use_usdz:
        baseline_asset = baseline_loose.with_suffix(".usdz")
        _package_usdz(baseline_loose, baseline_asset)
    else:
        baseline_asset = baseline_loose

    physics_plan, physics_receipt, post_asset = _publish_profile_receipt(
        tmp_path / "profile",
        baseline_asset=baseline_asset,
        transition_paths=transition_paths,
    )

    baseline_results = tmp_path / "baseline-gate3a.json"
    post_results = tmp_path / "post-gate3a.json"
    _write_results(baseline_results, asset=baseline_asset, status="pass", issues=[])
    _write_results(
        post_results,
        asset=post_asset,
        status="fail",
        issues=[
            {
                "at": _OWNER,
                "message": _CLASH_MESSAGE,
                "requirement": None,
                "rule": "NonAdjacentCollisionMeshesDoNotClash",
                "severity": "ERROR",
            }
        ],
    )
    request = CollisionFilterCompletionRequestV1(
        schema_version=COLLISION_FILTER_COMPLETION_SCHEMA_VERSION,
        post_sdf_asset_path=str(post_asset.resolve()),
        post_sdf_asset_sha256=_sha256(post_asset),
        baseline_gate3a_results_path=str(baseline_results.resolve()),
        baseline_gate3a_results_sha256=_sha256(baseline_results),
        post_sdf_gate3a_results_path=str(post_results.resolve()),
        post_sdf_gate3a_results_sha256=_sha256(post_results),
        physics_profile_plan_path=str(physics_plan.resolve()),
        physics_profile_plan_sha256=_sha256(physics_plan),
        physics_profile_receipt_path=str(physics_receipt.resolve()),
        physics_profile_receipt_sha256=_sha256(physics_receipt),
        approved_by="asset-owner",
        approval_reference="issue-529-cross-gate-review",
    )
    return _Evidence(
        request=request,
        baseline_asset=baseline_asset.resolve(),
        post_asset=post_asset.resolve(),
        baseline_results=baseline_results.resolve(),
        post_results=post_results.resolve(),
        physics_plan=physics_plan.resolve(),
        physics_receipt=physics_receipt.resolve(),
        source_dependency=(
            (baseline_loose.parent / baseline_dependency_name).resolve()
            if baseline_dependency_name is not None
            else None
        ),
        output_dependency=(
            (post_asset.parent / baseline_dependency_name).resolve()
            if baseline_dependency_name is not None
            and output_dependency
            and not use_usdz
            else None
        ),
    )


def _refresh_request(evidence: _Evidence, **changes: Any):
    identities = {
        "post_sdf_asset_sha256": _sha256(evidence.post_asset),
        "baseline_gate3a_results_sha256": _sha256(evidence.baseline_results),
        "post_sdf_gate3a_results_sha256": _sha256(evidence.post_results),
        "physics_profile_plan_sha256": _sha256(evidence.physics_plan),
        "physics_profile_receipt_sha256": _sha256(evidence.physics_receipt),
        **changes,
    }
    return evidence.request.model_copy(update=identities)


def test_real_chair_topology_generates_plan_and_existing_authorer_applies_it(
    tmp_path: Path,
) -> None:
    evidence = _make_evidence(tmp_path)

    generated = generate_collision_filter_completion_plan(
        request=evidence.request,
        output_dir=tmp_path / "plans",
    )
    authored = author_collision_filter_derivative(
        asset_path=evidence.post_asset,
        package_root=evidence.post_asset.parent,
        plan_path=generated.plan_path,
        output_dir=tmp_path / "authored",
    )

    assert generated.plan.pairs[0].canonical_paths() == tuple(
        sorted((_OWNER, _TIRES[0]))
    )
    assert authored.passed, authored.reason
    output = Usd.Stage.Open(str(authored.output_path))
    assert output
    assert filtered_pair_is_authored(output, _OWNER, _TIRES[0])
    targets = (
        output.GetPrimAtPath(_OWNER)
        .GetRelationship("physics:filteredPairs")
        .GetTargets()
    )
    assert targets[:3] == [Sdf.Path(item) for item in _TIRES[1:]]
    assert set(targets) == {Sdf.Path(item) for item in _TIRES}
    assert authored.receipt_path is not None
    receipt = json.loads(authored.receipt_path.read_text(encoding="utf-8"))
    assert receipt["source_asset_sha256"] == _sha256(evidence.post_asset)
    assert receipt["evidence_artifact_integrity_verified"] is True
    assert "machine_evidence_verified" not in receipt


def test_plan_bytes_are_deterministic_and_reused(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    first = generate_collision_filter_completion_plan(
        request=evidence.request,
        output_dir=tmp_path / "plans",
    )
    second = generate_collision_filter_completion_plan(
        request=evidence.request,
        output_dir=tmp_path / "plans",
    )

    assert not first.reused_plan
    assert second.reused_plan
    assert first.plan_path == second.plan_path
    assert _sha256(first.plan_path) == first.plan_sha256 == second.plan_sha256


def test_accepts_exact_usdz_evidence_without_extracting_in_generator(
    tmp_path: Path,
) -> None:
    evidence = _make_evidence(tmp_path, use_usdz=True)

    result = generate_collision_filter_completion_plan(
        request=evidence.request,
        output_dir=tmp_path / "plans",
    )

    assert result.plan.source_asset_path == str(evidence.post_asset)
    assert result.plan.source_asset_sha256 == _sha256(evidence.post_asset)


def test_rejects_minimal_synthetic_gate3a_document(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    original = json.loads(evidence.baseline_results.read_text(encoding="utf-8"))
    results = original["results"]
    minimal = {
        "schema_version": "joint-agent-isaac-sim-asset-validator-v2",
        "result_count": len(results),
        "aggregate": original["aggregate"],
        "results": results,
        "results_sha256": _canonical_json_sha256({"results": results}),
    }
    evidence.baseline_results.write_text(
        json.dumps(minimal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CollisionFilterCompletionError, match="validation profile"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_tampered_gate3a_result_digest(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.baseline_results.read_text(encoding="utf-8"))
    payload["results_sha256"] = "0" * 64
    evidence.baseline_results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CollisionFilterCompletionError, match="results_sha256"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_stale_gate3a_dependency_bundle_digest(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.baseline_results.read_text(encoding="utf-8"))
    payload["results"][0]["artifact_dependency_bundle_sha256"] = "0" * 64
    payload["results_sha256"] = _canonical_json_sha256({"results": payload["results"]})
    evidence.baseline_results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CollisionFilterCompletionError, match="bundle digest is stale"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_unpinned_gate3a_registered_rule_inventory(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.baseline_results.read_text(encoding="utf-8"))
    payload["registered_rules_by_category"]["Basic"].append("InventedRule")
    payload["enabled_rules_by_category"]["Basic"].append("InventedRule")
    evidence.baseline_results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CollisionFilterCompletionError, match="inventory is not pinned"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "isaacsim.asset.validation-9.9.9"),
        ("path", "/test/exts/not-the-recorded-extension"),
    ],
)
def test_rejects_stale_gate3a_extension_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.baseline_results.read_text(encoding="utf-8"))
    payload["runtime"]["extension_identities"]["isaacsim.asset.validation"][field] = (
        value
    )
    evidence.baseline_results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CollisionFilterCompletionError, match="identity is stale"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_tampered_gate3a_closeout(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.baseline_results.read_text(encoding="utf-8"))
    closeout_path = Path(payload["closeout_json"])
    closeout = json.loads(closeout_path.read_text(encoding="utf-8"))
    closeout["assets"][0]["generated_usd_sha256"] = "0" * 64
    closeout_path.write_text(
        json.dumps(closeout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CollisionFilterCompletionError, match="closeout_sha256"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_receipt_not_bound_to_approved_physics_plan(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.physics_receipt.read_text(encoding="utf-8"))
    payload["plan_sha256"] = "f" * 64
    evidence.physics_receipt.write_bytes(_canonical_json_bytes(payload, newline=True))

    with pytest.raises(CollisionFilterCompletionError, match="exact approved plan"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_modified_approved_physics_plan(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    payload = json.loads(evidence.physics_plan.read_text(encoding="utf-8"))
    payload["approval"]["evidence"] = "claimant-replaced-approval"
    evidence.physics_plan.write_bytes(_canonical_json_bytes(payload, newline=True))

    with pytest.raises(CollisionFilterCompletionError, match="exact approved plan"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_completion_approval_from_different_plan_owner(
    tmp_path: Path,
) -> None:
    evidence = _make_evidence(tmp_path)

    with pytest.raises(CollisionFilterCompletionError, match="plan owner"):
        generate_collision_filter_completion_plan(
            request=evidence.request.model_copy(
                update={"approved_by": "different-owner"}
            ),
            output_dir=tmp_path / "plans",
        )


def test_rejects_post_usdz_changed_after_topology_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _make_evidence(tmp_path)
    original = completion_module._validate_completion_families

    def validate_then_mutate(**kwargs: Any) -> None:
        original(**kwargs)
        evidence.post_asset.write_bytes(evidence.post_asset.read_bytes() + b"changed")

    monkeypatch.setattr(
        completion_module,
        "_validate_completion_families",
        validate_then_mutate,
    )
    with pytest.raises(CollisionFilterCompletionError, match="input changed"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_changed_loose_source_dependency(tmp_path: Path) -> None:
    evidence = _make_evidence(
        tmp_path,
        use_usdz=False,
        source_dependency=True,
    )
    assert evidence.source_dependency is not None
    evidence.source_dependency.write_bytes(
        evidence.source_dependency.read_bytes() + b"# changed\n"
    )

    with pytest.raises(PhysicsProfilePlanError, match="dependency bundle"):
        verify_physics_profile_receipt(
            evidence.physics_receipt,
            plan_path=evidence.physics_plan,
            source_asset=evidence.baseline_asset,
            output_asset=evidence.post_asset,
        )


def test_rejects_changed_loose_output_dependency(tmp_path: Path) -> None:
    evidence = _make_evidence(
        tmp_path,
        use_usdz=False,
        output_dependency=True,
    )
    assert evidence.output_dependency is not None
    evidence.output_dependency.write_bytes(
        evidence.output_dependency.read_bytes() + b"# changed\n"
    )

    with pytest.raises(PhysicsProfilePlanError, match="artifact inventory"):
        verify_physics_profile_receipt(
            evidence.physics_receipt,
            plan_path=evidence.physics_plan,
            source_asset=evidence.baseline_asset,
            output_asset=evidence.post_asset,
        )


def test_rejects_loose_gate3a_subject_for_machine_approval(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path, use_usdz=False)

    with pytest.raises(CollisionFilterCompletionError, match="self-contained USDZ"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_physics_plan_rejects_inactive_filter_family_member(tmp_path: Path) -> None:
    with pytest.raises(PhysicsProfilePlanError, match="editable active Gprim"):
        _make_evidence(tmp_path, inactive_peer=True)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("abstract_peer", "active, defined, and non-abstract"),
        ("disabled_peer_joint", "not enabled at default time"),
        ("time_sampled_disabled_peer_rigid", "not enabled at time"),
        ("time_sampled_disabled_peer_collision", "not enabled at time"),
        ("time_sampled_disabled_peer_joint", "not enabled at time"),
        ("time_sampled_disabled_intermediate_rigid", "not enabled at time"),
        ("time_sampled_disabled_intermediate_collision", "not enabled at time"),
    ],
)
def test_rejects_non_simulating_filter_family_proof(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    evidence = _make_evidence(tmp_path, **{mode: True})

    with pytest.raises(CollisionFilterCompletionError, match=message):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("post_asset", "post-SDF asset SHA-256 is stale"),
        ("baseline_results", "baseline Gate 3A results SHA-256 is stale"),
        ("post_results", "post-SDF Gate 3A results SHA-256 is stale"),
        ("physics_receipt", "physics-profile receipt SHA-256 is stale"),
    ],
)
def test_rejects_stale_exact_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    evidence = _make_evidence(tmp_path)
    path = getattr(evidence, mutation)
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(CollisionFilterCompletionError, match=message):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_nonpassing_baseline(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path)
    _write_results(
        evidence.baseline_results,
        asset=evidence.baseline_asset,
        status="fail",
        issues=[
            {
                "message": "baseline failure",
                "rule": "OtherRule",
                "severity": "ERROR",
            }
        ],
    )

    with pytest.raises(CollisionFilterCompletionError, match="exact pass"):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


@pytest.mark.parametrize(
    ("rule", "message"),
    [
        ("OtherRule", _CLASH_MESSAGE),
        ("NonAdjacentCollisionMeshesDoNotClash", "unstructured clash"),
    ],
)
def test_rejects_unapproved_post_sdf_failure_shape(
    tmp_path: Path,
    rule: str,
    message: str,
) -> None:
    evidence = _make_evidence(tmp_path)
    _write_results(
        evidence.post_results,
        asset=evidence.post_asset,
        status="fail",
        issues=[{"message": message, "rule": rule, "severity": "ERROR"}],
    )

    with pytest.raises(CollisionFilterCompletionError):
        generate_collision_filter_completion_plan(
            request=_refresh_request(evidence),
            output_dir=tmp_path / "plans",
        )


def test_rejects_family_with_fewer_than_two_existing_peers(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path, peer_count=1)

    with pytest.raises(CollisionFilterCompletionError, match="family owner"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_ambiguous_family_owners(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path, candidate_is_owner=True)

    with pytest.raises(CollisionFilterCompletionError, match="unambiguous"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_pair_already_filtered_in_source(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path, candidate_filtered=True)

    with pytest.raises(CollisionFilterCompletionError, match="already authored"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_mixed_family_joint_signatures(tmp_path: Path) -> None:
    evidence = _make_evidence(tmp_path, mismatched_post_branch=True)

    with pytest.raises(CollisionFilterCompletionError, match="family joint topology"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )


def test_rejects_family_member_without_sealed_sdf_transition(tmp_path: Path) -> None:
    evidence = _make_evidence(
        tmp_path,
        transition_paths=(_OWNER, *_BASES, *_TIRES[:-1]),
    )

    with pytest.raises(CollisionFilterCompletionError, match="sealed SDF transition"):
        generate_collision_filter_completion_plan(
            request=evidence.request,
            output_dir=tmp_path / "plans",
        )
