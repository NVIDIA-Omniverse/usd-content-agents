# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate strict collision-filter plans for SDF-introduced clash regressions.

This module only decides whether exact validation evidence completes an already
authored filtered-pair topology family. Publication, package handling, OpenUSD
list-op preservation, and receipts remain owned by :mod:`collision_filter_plan`.
Validator JSON is trusted-host evidence, not a signed execution attestation;
this module verifies its exact-byte and semantic consistency but does not claim
external producer authenticity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)

from .collision_filter_plan import (
    CollisionFilterEvidence,
    CollisionFilterPair,
    CollisionFilterPlan,
    CollisionFilterPlanProvenance,
    filtered_pair_is_authored,
)
from .physics_profile_plan import (
    PhysicsProfilePlanError,
    PhysicsProfileReceiptV1,
    verify_physics_profile_receipt,
)

COLLISION_FILTER_COMPLETION_SCHEMA_VERSION = (
    "content-agent-workflows.simready-collision-filter-completion.v1"
)

_GATE3A_RESULTS_SCHEMA_VERSION = "joint-agent-isaac-sim-asset-validator-v2"
_GATE3A_DEPENDENCY_BUNDLE_SCHEMA_VERSION = (
    "joint-agent-usd-artifact-dependency-bundle-v3"
)
_GATE3A_PROFILE_NAME = "articulated-prop-v1"
_GATE3A_PROFILE_CATEGORIES = (
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
_GATE3A_NOT_APPLICABLE_CATEGORIES = ("AtomicAsset", "IsaacSim.RobotRules")
_GATE3A_NOT_APPLICABLE_RULES = (
    "GroundTruthCapabilityChecker",
    "NonVisualSensorCapabilityChecker",
    "VisualSensorCapabilityChecker",
)
_GATE3A_PINNED_PACKAGE_VERSIONS = {
    name: "6.0.0.1"
    for name in (
        "isaacsim",
        "isaacsim-app",
        "isaacsim-asset",
        "isaacsim-extscache-kit",
        "isaacsim-extscache-physics",
    )
}
_GATE3A_REQUIRED_EXTENSIONS = (
    "omni.asset_validator.core",
    "isaacsim.asset.validation",
)
_GATE3A_REQUIRED_EXTENSION_IDENTITIES = {
    "omni.asset_validator.core": {
        "id": "omni.asset_validator.core-1.19.3",
        "path": "extscache/omni.asset_validator.core-1.19.3",
    },
    "isaacsim.asset.validation": {
        "id": "isaacsim.asset.validation-1.3.5",
        "path": "exts/isaacsim.asset.validation",
    },
}
_GATE3A_ALLOWED_LAUNCH_PROFILES = frozenset(
    {
        "joint-rigger-gate3a-minimal-kit-v1",
        "joint-rigger-gate3a-standalone-default-experience-v1",
    }
)
_GATE3A_EXTENSION_INVENTORY_SCHEMA = (
    "joint-rigger-gate3a-enabled-extension-inventory-v1"
)
_GATE3A_REGISTERED_RULES_SHA256 = (
    "3abb83daeca7e04cba4172e6f7d66e272ea6b7e10376c85e3b3ac973013ee104"
)
_GATE3A_REQUIRED_RULES = {
    "Basic": frozenset(
        {"UsdzPackageValidator", "MissingReferenceChecker", "StageMetadataChecker"}
    ),
    "Usd:Physics": frozenset(
        {
            "RigidBodyChecker",
            "ColliderChecker",
            "PhysicsJointChecker",
            "ArticulationChecker",
            "MassChecker",
        }
    ),
    "Omni:SimReady": frozenset(
        {
            "ContainsMeshChecker",
            "HierarchyHasRootChecker",
            "UpAxisZChecker",
            "MetersPerUnit1Checker",
        }
    ),
    "IsaacSim.PhysicsRules": frozenset(
        {
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
        }
    ),
    "IsaacSim.SimReadyAssetRules": frozenset(
        {"NoNestedMaterials", "MaterialsOnTopLevelOnly"}
    ),
}
_CLASH_RULE = "NonAdjacentCollisionMeshesDoNotClash"
_CLASH_MESSAGE = re.compile(
    r"^Colliding meshes (?P<body_a>/\S+) and (?P<body_b>/\S+) are not adjacent$"
)
_HARD_SEVERITIES = frozenset({"ERROR", "FAILURE", "WARNING"})
_MIN_EXISTING_FAMILY_PEERS = 2
_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})

_StrictNonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
_ReadableStrictString = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
_Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]


class CollisionFilterCompletionError(ValueError):
    """Raised when exact evidence cannot safely complete a filter family."""


class CollisionFilterCompletionRequestV1(BaseModel):
    """Exact evidence needed to generate one collision-filter plan."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[
        "content-agent-workflows.simready-collision-filter-completion.v1"
    ]
    post_sdf_asset_path: _StrictNonEmptyString
    post_sdf_asset_sha256: _Sha256
    baseline_gate3a_results_path: _StrictNonEmptyString
    baseline_gate3a_results_sha256: _Sha256
    post_sdf_gate3a_results_path: _StrictNonEmptyString
    post_sdf_gate3a_results_sha256: _Sha256
    physics_profile_plan_path: _StrictNonEmptyString
    physics_profile_plan_sha256: _Sha256
    physics_profile_receipt_path: _StrictNonEmptyString
    physics_profile_receipt_sha256: _Sha256
    approved_by: _ReadableStrictString
    approval_reference: _ReadableStrictString

    @field_validator(
        "post_sdf_asset_path",
        "baseline_gate3a_results_path",
        "post_sdf_gate3a_results_path",
        "physics_profile_plan_path",
        "physics_profile_receipt_path",
    )
    @classmethod
    def validate_absolute_paths(cls, value: str) -> str:
        """Require every request identity to use one exact absolute path."""

        if not Path(value).is_absolute():
            raise ValueError("completion request paths must be absolute")
        return value


@dataclass(frozen=True)
class CollisionFilterCompletionResult:
    """One deterministic plan generated from exact cross-gate evidence."""

    plan_path: Path
    plan_sha256: str
    plan: CollisionFilterPlan
    reused_plan: bool


@dataclass(frozen=True)
class _FileIdentity:
    path: Path
    sha256: str
    stat_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _Gate3ARow:
    row: dict[str, Any]
    asset_identity: _FileIdentity


@dataclass(frozen=True)
class _Gate3AEvidence:
    payload: dict[str, Any]
    closeout_identity: _FileIdentity


@dataclass(frozen=True)
class _JointEdge:
    neighbor: str
    joint_type: str
    direction: Literal["body0_to_body1", "body1_to_body0"]


def generate_collision_filter_completion_plan(
    *,
    request: CollisionFilterCompletionRequestV1,
    output_dir: Path,
) -> CollisionFilterCompletionResult:
    """Generate a plan only for a proven missing member of a filter family.

    The baseline must pass Gate 3A before the exact physics-profile derivative,
    whose only hard post-SDF findings must be non-adjacent collider clashes. Each
    clash must complete a pre-existing filtered topology family with at least two
    peers and one unique, matching directed joint-path signature.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    post_asset = _capture_expected_file(
        Path(request.post_sdf_asset_path),
        expected_sha256=request.post_sdf_asset_sha256,
        label="post-SDF asset",
        suffixes=_USD_SUFFIXES,
    )
    baseline_results = _capture_expected_file(
        Path(request.baseline_gate3a_results_path),
        expected_sha256=request.baseline_gate3a_results_sha256,
        label="baseline Gate 3A results",
        suffixes=frozenset({".json"}),
    )
    post_results = _capture_expected_file(
        Path(request.post_sdf_gate3a_results_path),
        expected_sha256=request.post_sdf_gate3a_results_sha256,
        label="post-SDF Gate 3A results",
        suffixes=frozenset({".json"}),
    )
    physics_receipt_identity = _capture_expected_file(
        Path(request.physics_profile_receipt_path),
        expected_sha256=request.physics_profile_receipt_sha256,
        label="physics-profile receipt",
        suffixes=frozenset({".json"}),
    )
    physics_plan_identity = _capture_expected_file(
        Path(request.physics_profile_plan_path),
        expected_sha256=request.physics_profile_plan_sha256,
        label="physics-profile plan",
        suffixes=frozenset({".json"}),
    )
    _load_json_object(physics_receipt_identity.path, label="physics-profile receipt")
    try:
        receipt = PhysicsProfileReceiptV1.model_validate_json(
            physics_receipt_identity.path.read_bytes(),
            strict=True,
        )
    except (OSError, ValidationError) as exc:
        raise CollisionFilterCompletionError(
            f"Invalid physics-profile receipt: {physics_receipt_identity.path}"
        ) from exc

    baseline_evidence = _load_gate3a_results(baseline_results.path)
    baseline_row = _select_gate3a_row(
        baseline_evidence.payload,
        artifact_sha256=receipt.source_asset_sha256,
        exact_asset_path=None,
        label="baseline",
    )
    _require_baseline_pass(baseline_row.row)
    try:
        verified_receipt = verify_physics_profile_receipt(
            physics_receipt_identity.path,
            plan_path=physics_plan_identity.path,
            source_asset=baseline_row.asset_identity.path,
            output_asset=post_asset.path,
            expected_owner_identity=request.approved_by,
        )
    except PhysicsProfilePlanError as exc:
        raise CollisionFilterCompletionError(
            f"Invalid physics-profile receipt evidence: {exc}"
        ) from exc
    if verified_receipt != receipt:
        _fail("Physics-profile receipt changed during strict verification.")

    post_evidence = _load_gate3a_results(post_results.path)
    post_row = _select_gate3a_row(
        post_evidence.payload,
        artifact_sha256=post_asset.sha256,
        exact_asset_path=post_asset.path,
        label="post-SDF",
    )
    pairs = _post_sdf_clash_pairs(post_row.row)

    baseline_stage = Usd.Stage.Open(str(baseline_row.asset_identity.path))
    if baseline_stage is None:
        _fail(f"Unable to open baseline asset: {baseline_row.asset_identity.path}")
    post_stage = Usd.Stage.Open(str(post_asset.path))
    if post_stage is None:
        _fail(f"Unable to open post-SDF asset: {post_asset.path}")

    transitions = _sdf_transitions(receipt)
    _validate_completion_families(
        baseline_stage=baseline_stage,
        post_stage=post_stage,
        pairs=pairs,
        transitions=transitions,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )

    plan = CollisionFilterPlan(
        schema_version="content-agent-workflows.simready-collision-filter-plan.v1",
        source_asset_path=str(post_asset.path),
        source_asset_sha256=post_asset.sha256,
        provenance=CollisionFilterPlanProvenance(
            approved_by=request.approved_by,
            approval_reference=request.approval_reference,
            evidence=[
                CollisionFilterEvidence(
                    kind="gate3a_validation",
                    artifact_path=str(baseline_results.path),
                    artifact_sha256=baseline_results.sha256,
                ),
                CollisionFilterEvidence(
                    kind="gate3a_validation",
                    artifact_path=str(post_results.path),
                    artifact_sha256=post_results.sha256,
                ),
                CollisionFilterEvidence(
                    kind="machine_collision_preflight",
                    artifact_path=str(physics_plan_identity.path),
                    artifact_sha256=physics_plan_identity.sha256,
                ),
                CollisionFilterEvidence(
                    kind="machine_collision_preflight",
                    artifact_path=str(physics_receipt_identity.path),
                    artifact_sha256=physics_receipt_identity.sha256,
                ),
            ],
        ),
        pairs=[
            CollisionFilterPair(body_a_path=first, body_b_path=second)
            for first, second in pairs
        ],
    )
    plan_bytes = _canonical_model_bytes(plan)
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()

    identities = (
        post_asset,
        baseline_results,
        post_results,
        physics_plan_identity,
        physics_receipt_identity,
        baseline_row.asset_identity,
        baseline_evidence.closeout_identity,
        post_evidence.closeout_identity,
    )
    _require_inputs_unchanged(identities)
    plan_path, reused = _publish_content_addressed_plan(
        output_dir=output_dir,
        plan_sha256=plan_sha256,
        payload=plan_bytes,
    )
    _require_inputs_unchanged(identities)
    return CollisionFilterCompletionResult(
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        plan=plan,
        reused_plan=reused,
    )


def _capture_expected_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    suffixes: frozenset[str],
) -> _FileIdentity:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise CollisionFilterCompletionError(
            f"{label} does not exist: {absolute}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular non-symlink: {absolute}")
    if absolute.suffix.lower() not in suffixes:
        _fail(f"Unsupported {label} suffix: {absolute.suffix}")
    identity = _capture_file_identity(absolute)
    if identity.sha256 != expected_sha256:
        _fail(
            f"{label} SHA-256 is stale: expected {identity.sha256}, "
            f"received {expected_sha256}."
        )
    return identity


def _capture_file_identity(path: Path) -> _FileIdentity:
    metadata = path.stat(follow_symlinks=False)
    return _FileIdentity(
        path=path,
        sha256=_file_sha256(path),
        stat_identity=(
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ),
    )


def _load_gate3a_results(path: Path) -> _Gate3AEvidence:
    payload = _load_json_object(path, label="Gate 3A results")
    if payload.get("schema_version") != _GATE3A_RESULTS_SCHEMA_VERSION:
        _fail("Unsupported Gate 3A results schema_version.")
    results = payload.get("results")
    if not isinstance(results, list) or not all(
        isinstance(item, dict) for item in results
    ):
        _fail("Gate 3A results must contain a results list.")
    if payload.get("result_count") != len(results):
        _fail("Gate 3A result_count is inconsistent.")
    _require_gate3a_attestation(payload)

    status_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    for row in results:
        _require_consistent_issue_counts(row, label="aggregate")
        status = row.get("status")
        if not isinstance(status, str) or not status:
            _fail("Gate 3A result status is invalid.")
        status_counts[status] += 1
        severity_counts.update(item["severity"] for item in row["issues"])
    expected_aggregate = {
        "status_counts": dict(sorted(status_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
    }
    if payload.get("aggregate") != expected_aggregate:
        _fail("Gate 3A aggregate counts are inconsistent.")
    if payload.get("results_sha256") != _canonical_json_sha256({"results": results}):
        _fail("Gate 3A results_sha256 is stale.")

    closeout_identity, closeout = _load_gate3a_closeout(payload)
    _require_gate3a_closeout_rows(results, closeout)
    return _Gate3AEvidence(
        payload=payload,
        closeout_identity=closeout_identity,
    )


def _require_gate3a_attestation(payload: dict[str, Any]) -> None:
    profile = payload.get("validation_profile")
    if not isinstance(profile, dict) or profile.get("name") != _GATE3A_PROFILE_NAME:
        _fail("Gate 3A validation profile is not articulated-prop-v1.")
    categories = list(_GATE3A_PROFILE_CATEGORIES)
    if profile.get("applicable_categories") != categories:
        _fail("Gate 3A validation profile categories are incomplete.")
    if profile.get("not_applicable_categories") != list(
        _GATE3A_NOT_APPLICABLE_CATEGORIES
    ) or profile.get("not_applicable_rules") != list(_GATE3A_NOT_APPLICABLE_RULES):
        _fail("Gate 3A not-applicable profile identity drifted.")
    if profile.get("missing_categories") != []:
        _fail("Gate 3A validation profile has missing categories.")
    if payload.get("enabled_categories") != categories:
        _fail("Gate 3A enabled categories do not match the strict profile.")

    registered = payload.get("registered_rules_by_category")
    enabled = payload.get("enabled_rules_by_category")
    if not isinstance(registered, dict) or not isinstance(enabled, dict):
        _fail("Gate 3A rule inventories are missing.")
    if _canonical_json_sha256(registered) != _GATE3A_REGISTERED_RULES_SHA256:
        _fail("Gate 3A registered rule inventory is not pinned.")
    if set(enabled) != set(categories):
        _fail("Gate 3A enabled-rule categories do not match the strict profile.")
    not_applicable = set(_GATE3A_NOT_APPLICABLE_RULES)
    for category in categories:
        registered_rules = registered.get(category)
        enabled_rules = enabled.get(category)
        if (
            not isinstance(registered_rules, list)
            or not registered_rules
            or any(
                not isinstance(rule, str) or not rule.strip()
                for rule in registered_rules
            )
            or len(registered_rules) != len(set(registered_rules))
        ):
            _fail(f"Gate 3A registered rules are invalid for {category}.")
        expected_enabled = [
            rule for rule in registered_rules if rule not in not_applicable
        ]
        if enabled_rules != expected_enabled or len(expected_enabled) != len(
            set(expected_enabled)
        ):
            _fail(f"Gate 3A enabled rules are incomplete for {category}.")
        required = _GATE3A_REQUIRED_RULES.get(category, frozenset())
        if not required.issubset(expected_enabled):
            _fail(f"Gate 3A required rules are missing for {category}.")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        _fail("Gate 3A runtime attestation is missing.")
    if runtime.get("package_versions") != _GATE3A_PINNED_PACKAGE_VERSIONS:
        _fail("Gate 3A runtime package versions are not pinned.")
    if runtime.get("extensions_enabled") != dict.fromkeys(
        _GATE3A_REQUIRED_EXTENSIONS,
        True,
    ):
        _fail("Gate 3A required validator extensions are not enabled.")
    launch = runtime.get("kit_launch")
    if (
        not isinstance(launch, dict)
        or launch.get("profile") not in _GATE3A_ALLOWED_LAUNCH_PROFILES
    ):
        _fail("Gate 3A Kit launch profile is not pinned.")
    if "kit_runtime_attestation" not in runtime:
        _fail("Gate 3A Kit runtime attestation field is missing.")

    inventory = runtime.get("kit_enabled_extension_inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema_version") != _GATE3A_EXTENSION_INVENTORY_SCHEMA
    ):
        _fail("Gate 3A enabled-extension inventory is invalid.")
    extension_records = inventory.get("extensions")
    if not isinstance(extension_records, list) or not all(
        isinstance(item, dict) for item in extension_records
    ):
        _fail("Gate 3A enabled-extension records are invalid.")
    if inventory.get("extension_count") != len(extension_records):
        _fail("Gate 3A enabled-extension count is inconsistent.")
    if inventory.get("extensions_sha256") != _canonical_json_sha256(
        {"extensions": extension_records}
    ):
        _fail("Gate 3A enabled-extension inventory digest is stale.")
    records_by_name: dict[str, dict[str, Any]] = {}
    for record in extension_records:
        name = record.get("name")
        extension_id = record.get("id")
        extension_path = record.get("path")
        if (
            not isinstance(name, str)
            or not name
            or name in records_by_name
            or not isinstance(extension_id, str)
            or not extension_id
            or not isinstance(extension_path, str)
            or not extension_path
            or extension_path.startswith("/")
            or "\\" in extension_path
            or any(part in {"", ".", ".."} for part in extension_path.split("/"))
        ):
            _fail("Gate 3A enabled-extension record identity is invalid.")
        records_by_name[name] = record
    identities = runtime.get("extension_identities")
    if not isinstance(identities, dict):
        _fail("Gate 3A validator extension identities are missing.")
    for name in _GATE3A_REQUIRED_EXTENSIONS:
        record = records_by_name.get(name)
        identity = identities.get(name)
        expected = _GATE3A_REQUIRED_EXTENSION_IDENTITIES[name]
        identity_path = identity.get("path") if isinstance(identity, dict) else None
        if (
            record is None
            or record.get("id") != expected["id"]
            or record.get("path") != expected["path"]
            or not isinstance(identity, dict)
            or identity.get("id") != expected["id"]
            or not isinstance(identity_path, str)
            or not Path(identity_path).is_absolute()
            or "\\" in identity_path
            or ".." in Path(identity_path).parts
            or tuple(Path(identity_path).parts[-len(expected["path"].split("/")) :])
            != tuple(expected["path"].split("/"))
        ):
            _fail(f"Gate 3A validator extension identity is stale for {name}.")
    selection = payload.get("selection")
    if not isinstance(selection, dict) or selection.get("kinds") != ["generated"]:
        _fail("Gate 3A evidence is not scoped to generated assets.")


def _load_gate3a_closeout(
    payload: dict[str, Any],
) -> tuple[_FileIdentity, dict[str, Any]]:
    raw_path = payload.get("closeout_json")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        _fail("Gate 3A closeout path must be absolute.")
    path = Path(os.path.abspath(Path(raw_path).expanduser()))
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise CollisionFilterCompletionError(
            f"Gate 3A closeout does not exist: {path}"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or path.suffix.lower() != ".json"
    ):
        _fail(f"Gate 3A closeout must be a regular JSON file: {path}")
    identity = _capture_file_identity(path)
    closeout = _load_json_object(path, label="Gate 3A closeout")
    expected_sha256 = payload.get("closeout_sha256")
    if (
        not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or expected_sha256 != _canonical_json_sha256(closeout)
    ):
        _fail("Gate 3A closeout_sha256 is stale.")
    return identity, closeout


def _require_gate3a_closeout_rows(
    results: list[dict[str, Any]],
    closeout: dict[str, Any],
) -> None:
    assets = closeout.get("assets")
    if not isinstance(assets, list) or not all(
        isinstance(item, dict) for item in assets
    ):
        _fail("Gate 3A closeout assets are invalid.")
    closeout_by_id: dict[str, dict[str, Any]] = {}
    for item in assets:
        asset_id = item.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id or asset_id in closeout_by_id:
            _fail("Gate 3A closeout asset identity is invalid or duplicated.")
        closeout_by_id[asset_id] = item
    result_ids: set[str] = set()
    for row in results:
        asset_id = row.get("asset_id")
        path = row.get("path")
        artifact_sha256 = row.get("artifact_sha256")
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or asset_id in result_ids
            or not isinstance(path, str)
            or not Path(path).is_absolute()
            or row.get("original_path") != path
            or not isinstance(artifact_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256)
        ):
            _fail("Gate 3A result identity is invalid or duplicated.")
        result_ids.add(asset_id)
        closeout_row = closeout_by_id.get(asset_id)
        if (
            closeout_row is None
            or closeout_row.get("generated_usd_path") != path
            or closeout_row.get("generated_usd_sha256") != artifact_sha256
        ):
            _fail(f"Gate 3A closeout does not bind result asset {asset_id}.")
    if result_ids != set(closeout_by_id):
        _fail("Gate 3A closeout and result asset rosters differ.")


def _select_gate3a_row(
    payload: dict[str, Any],
    *,
    artifact_sha256: str,
    exact_asset_path: Path | None,
    label: str,
) -> _Gate3ARow:
    matches = [
        item
        for item in payload["results"]
        if isinstance(item, dict) and item.get("artifact_sha256") == artifact_sha256
    ]
    if len(matches) != 1:
        _fail(
            f"{label} Gate 3A evidence must contain exactly one result for "
            f"{artifact_sha256}."
        )
    row = matches[0]
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        _fail(f"{label} Gate 3A result path must be absolute.")
    row_path = Path(os.path.abspath(Path(raw_path).expanduser()))
    if row.get("original_path") != raw_path:
        _fail(f"{label} Gate 3A result path identity is ambiguous.")
    if exact_asset_path is not None and row_path != exact_asset_path:
        _fail(f"{label} Gate 3A result does not name the exact requested asset path.")
    identity = _capture_expected_file(
        row_path,
        expected_sha256=artifact_sha256,
        label=f"{label} Gate 3A subject",
        suffixes=_USD_SUFFIXES,
    )
    if identity.path.suffix.lower() != ".usdz":
        _fail(f"{label} Gate 3 completion requires one self-contained USDZ.")
    _require_gate3a_dependency_bundle(row, asset_path=identity.path, label=label)
    _require_consistent_issue_counts(row, label=label)
    return _Gate3ARow(row=row, asset_identity=identity)


def _require_gate3a_dependency_bundle(
    row: dict[str, Any],
    *,
    asset_path: Path,
    label: str,
) -> None:
    try:
        with zipfile.ZipFile(asset_path) as archive:
            infos = archive.infolist()
            if not infos:
                _fail(f"{label} Gate 3A USDZ package is empty.")
            entries: list[dict[str, Any]] = []
            for info in infos:
                digest = hashlib.sha256()
                if not info.is_dir():
                    with archive.open(info) as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                entries.append(
                    {
                        "kind": (
                            "package_directory" if info.is_dir() else "package_entry"
                        ),
                        "path": info.filename,
                        "size": 0 if info.is_dir() else info.file_size,
                        "sha256": digest.hexdigest(),
                    }
                )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CollisionFilterCompletionError(
            f"Invalid {label} Gate 3A USDZ dependency bundle: {asset_path}"
        ) from exc
    manifest = {
        "schema_version": _GATE3A_DEPENDENCY_BUNDLE_SCHEMA_VERSION,
        "container": "usdz",
        "root_entry": infos[0].filename,
        "entries": sorted(entries, key=lambda item: str(item["path"])),
    }
    if row.get("artifact_dependency_bundle_schema_version") != (
        _GATE3A_DEPENDENCY_BUNDLE_SCHEMA_VERSION
    ):
        _fail(f"{label} Gate 3A dependency-bundle schema is not pinned.")
    if row.get("artifact_dependency_bundle_entry_count") != len(entries):
        _fail(f"{label} Gate 3A dependency-bundle entry count is stale.")
    if row.get("artifact_dependency_bundle_sha256") != _canonical_json_sha256(manifest):
        _fail(f"{label} Gate 3A dependency-bundle digest is stale.")


def _require_consistent_issue_counts(row: dict[str, Any], *, label: str) -> None:
    issues = row.get("issues")
    if not isinstance(issues, list) or not all(
        isinstance(item, dict) for item in issues
    ):
        _fail(f"{label} Gate 3A result has an invalid issues list.")
    if row.get("issue_count") != len(issues):
        _fail(f"{label} Gate 3A issue_count is inconsistent.")
    counts = Counter(item.get("severity") for item in issues)
    if any(not isinstance(key, str) for key in counts):
        _fail(f"{label} Gate 3A issue severity is invalid.")
    if row.get("severity_counts") != dict(sorted(counts.items())):
        _fail(f"{label} Gate 3A severity_counts are inconsistent.")
    hard_count = sum(counts.get(severity, 0) for severity in ("ERROR", "FAILURE"))
    if row.get("hard_failure_count") != hard_count:
        _fail(f"{label} Gate 3A hard_failure_count is inconsistent.")


def _require_baseline_pass(row: dict[str, Any]) -> None:
    if row.get("status") != "pass" or row.get("hard_failure_count") != 0:
        _fail("Baseline Gate 3A result must be an exact pass.")
    counts = row.get("severity_counts", {})
    if any(counts.get(severity, 0) for severity in _HARD_SEVERITIES):
        _fail("Baseline Gate 3A result contains a hard severity.")


def _post_sdf_clash_pairs(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    if row.get("status") != "fail":
        _fail("Post-SDF Gate 3A result must expose a remediation failure.")
    issues = row["issues"]
    hard_issues = [item for item in issues if item.get("severity") in _HARD_SEVERITIES]
    if not hard_issues:
        _fail("Post-SDF Gate 3A result contains no hard clash evidence.")
    pairs: list[tuple[str, str]] = []
    for item in hard_issues:
        if item.get("severity") != "ERROR" or item.get("rule") != _CLASH_RULE:
            _fail("Post-SDF Gate 3A contains a hard issue outside the clash rule.")
        message = item.get("message")
        match = _CLASH_MESSAGE.fullmatch(message) if isinstance(message, str) else None
        if match is None:
            _fail("Post-SDF clash message does not contain one exact collider pair.")
        first, second = sorted((match["body_a"], match["body_b"]))
        if first == second:
            _fail("Post-SDF clash evidence contains a self-pair.")
        pairs.append((first, second))
    result = tuple(sorted(pairs))
    if len(result) != len(set(result)):
        _fail("Post-SDF clash evidence contains duplicate unordered pairs.")
    return result


def _sdf_transitions(receipt: PhysicsProfileReceiptV1) -> dict[str, str]:
    transitions: dict[str, str] = {}
    for item in receipt.authored_sdf_transitions:
        if item.prim_path in transitions:
            _fail("Physics-profile receipt contains duplicate SDF transitions.")
        if item.source_token != "convexHull" or item.output_token != "sdf":
            _fail("Collision-filter completion requires convexHull-to-SDF transitions.")
        transitions[item.prim_path] = item.source_token
    return transitions


def _validate_completion_families(
    *,
    baseline_stage: Any,
    post_stage: Any,
    pairs: tuple[tuple[str, str], ...],
    transitions: dict[str, str],
    UsdGeom: Any,
    UsdPhysics: Any,
) -> None:
    baseline_graph = _joint_graph(baseline_stage, UsdPhysics=UsdPhysics)
    post_graph = _joint_graph(post_stage, UsdPhysics=UsdPhysics)
    for first, second in pairs:
        if filtered_pair_is_authored(post_stage, first, second):
            _fail("Post-SDF clash pair is already authored as a filtered pair.")
        owner_candidates: list[tuple[str, str, tuple[str, ...]]] = []
        for owner, candidate in ((first, second), (second, first)):
            targets = _filtered_targets(post_stage, owner, UsdPhysics=UsdPhysics)
            if len(targets) >= _MIN_EXISTING_FAMILY_PEERS:
                owner_candidates.append((owner, candidate, targets))
        if len(owner_candidates) != 1:
            _fail("Clash pair does not identify one unambiguous filtered family owner.")
        owner, candidate, peers = owner_candidates[0]
        baseline_peers = _filtered_targets(baseline_stage, owner, UsdPhysics=UsdPhysics)
        if baseline_peers != peers:
            _fail("Physics-profile derivative changed the existing filter family.")
        if filtered_pair_is_authored(baseline_stage, owner, candidate):
            _fail("Baseline already contains the proposed filtered pair.")
        if candidate in peers or owner in peers:
            _fail("Existing filter family contains the proposed endpoint or itself.")

        family = (owner, candidate, *peers)
        for path in family:
            if path not in transitions:
                _fail(f"Filter family member lacks a sealed SDF transition: {path}")
            _require_transitioned_collider(
                baseline_stage=baseline_stage,
                post_stage=post_stage,
                path=path,
                UsdGeom=UsdGeom,
                UsdPhysics=UsdPhysics,
            )
        for peer in peers:
            if not _pair_is_authored_from(
                post_stage,
                owner=owner,
                target=peer,
                UsdPhysics=UsdPhysics,
            ):
                _fail("Existing filter family has ambiguous relationship direction.")

        candidate_signature = _unique_joint_signature(
            post_graph, source=owner, target=candidate
        )
        if len(candidate_signature) < 2:
            _fail("Collision-filter completion cannot suppress a direct joint pair.")
        baseline_candidate_signature = _unique_joint_signature(
            baseline_graph, source=owner, target=candidate
        )
        if baseline_candidate_signature != candidate_signature:
            _fail("Physics-profile derivative changed the candidate joint topology.")
        for peer in peers:
            post_signature = _unique_joint_signature(
                post_graph, source=owner, target=peer
            )
            baseline_signature = _unique_joint_signature(
                baseline_graph, source=owner, target=peer
            )
            if post_signature != candidate_signature:
                _fail("Candidate does not match the existing family joint topology.")
            if baseline_signature != post_signature:
                _fail("Physics-profile derivative changed the family joint topology.")


def _filtered_targets(stage: Any, path: str, *, UsdPhysics: Any) -> tuple[str, ...]:
    prim = _require_active_defined_prim(stage, path, label="filtered family endpoint")
    has_api = prim.HasAPI(UsdPhysics.FilteredPairsAPI)
    relationship = prim.GetRelationship("physics:filteredPairs")
    if relationship and not has_api:
        _fail(f"physics:filteredPairs exists without its API at {path}.")
    if not relationship:
        return ()
    targets: list[str] = []
    for target in relationship.GetTargets():
        if not target.IsAbsolutePath() or not target.IsPrimPath():
            _fail(f"Existing filtered target is not an absolute prim path: {target}")
        _require_active_defined_prim(
            stage,
            str(target),
            label="existing filtered target",
        )
        targets.append(str(target))
    if len(targets) != len(set(targets)):
        _fail(f"Existing filter family contains duplicate targets at {path}.")
    return tuple(sorted(targets))


def _pair_is_authored_from(
    stage: Any,
    *,
    owner: str,
    target: str,
    UsdPhysics: Any,
) -> bool:
    if target not in _filtered_targets(stage, owner, UsdPhysics=UsdPhysics):
        return False
    return owner not in _filtered_targets(stage, target, UsdPhysics=UsdPhysics)


def _require_transitioned_collider(
    *,
    baseline_stage: Any,
    post_stage: Any,
    path: str,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> None:
    expected = ((baseline_stage, "convexHull"), (post_stage, "sdf"))
    for stage, approximation in expected:
        prim = _require_active_defined_prim(
            stage,
            path,
            label="filter family collider",
        )
        if (
            not prim.IsA(UsdGeom.Mesh)
            or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
            or not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        ):
            _fail(f"Filter family member is not a rigid Mesh collider: {path}")
        _require_enabled_for_all_time(
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr(),
            label=f"rigid body {path}",
        )
        _require_enabled_for_all_time(
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr(),
            label=f"collider {path}",
        )
        attribute = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if (
            not attribute
            or not attribute.HasAuthoredValueOpinion()
            or attribute.Get() != approximation
        ):
            _fail(f"Filter family member has stale physics:approximation at {path}.")


def _joint_graph(stage: Any, *, UsdPhysics: Any) -> dict[str, tuple[_JointEdge, ...]]:
    adjacency: dict[str, list[_JointEdge]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        _require_enabled_for_all_time(
            joint.GetJointEnabledAttr(),
            label=f"joint {prim.GetPath()}",
        )
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue
        if len(body0) != 1 or len(body1) != 1:
            _fail(f"Joint has ambiguous body targets: {prim.GetPath()}")
        first, second = body0[0], body1[0]
        if (
            not first.IsAbsolutePath()
            or not second.IsAbsolutePath()
            or not first.IsPrimPath()
            or not second.IsPrimPath()
            or first == second
        ):
            _fail(f"Joint has invalid body targets: {prim.GetPath()}")
        first_text, second_text = str(first), str(second)
        first_prim = _require_active_defined_prim(
            stage,
            first_text,
            label="joint body target",
        )
        second_prim = _require_active_defined_prim(
            stage,
            second_text,
            label="joint body target",
        )
        _require_enabled_joint_body(
            first_prim,
            path=first_text,
            UsdPhysics=UsdPhysics,
        )
        _require_enabled_joint_body(
            second_prim,
            path=second_text,
            UsdPhysics=UsdPhysics,
        )
        joint_type = str(prim.GetTypeName())
        adjacency.setdefault(first_text, []).append(
            _JointEdge(second_text, joint_type, "body0_to_body1")
        )
        adjacency.setdefault(second_text, []).append(
            _JointEdge(first_text, joint_type, "body1_to_body0")
        )
    return {
        path: tuple(
            sorted(
                edges,
                key=lambda edge: (edge.neighbor, edge.joint_type, edge.direction),
            )
        )
        for path, edges in adjacency.items()
    }


def _require_active_defined_prim(stage: Any, path: str, *, label: str) -> Any:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsActive() or not prim.IsDefined() or prim.IsAbstract():
        _fail(f"{label} is not active, defined, and non-abstract: {path}")
    return prim


def _require_enabled_for_all_time(attribute: Any, *, label: str) -> None:
    if not attribute or attribute.Get() is not True:
        _fail(f"{label} is not enabled at default time.")
    for time_code in attribute.GetTimeSamples():
        if attribute.Get(time_code) is not True:
            _fail(f"{label} is not enabled at time {time_code}.")


def _require_enabled_joint_body(prim: Any, *, path: str, UsdPhysics: Any) -> None:
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        _fail(f"joint body target is not a rigid body: {path}")
    _require_enabled_for_all_time(
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr(),
        label=f"joint rigid body {path}",
    )
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        _require_enabled_for_all_time(
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr(),
            label=f"joint collider {path}",
        )


def _unique_joint_signature(
    graph: dict[str, tuple[_JointEdge, ...]],
    *,
    source: str,
    target: str,
) -> tuple[tuple[str, str], ...]:
    distances = {source: 0}
    path_counts = {source: 1}
    signatures: dict[str, tuple[tuple[str, str], ...] | None] = {source: ()}
    queue: deque[str] = deque([source])
    while queue:
        current = queue.popleft()
        for edge in graph.get(current, ()):
            distance = distances[current] + 1
            signature = signatures[current]
            candidate = (
                None
                if signature is None
                else (*signature, (edge.joint_type, edge.direction))
            )
            if edge.neighbor not in distances:
                distances[edge.neighbor] = distance
                path_counts[edge.neighbor] = path_counts[current]
                signatures[edge.neighbor] = candidate
                queue.append(edge.neighbor)
            elif distances[edge.neighbor] == distance:
                path_counts[edge.neighbor] = min(
                    2, path_counts[edge.neighbor] + path_counts[current]
                )
                signatures[edge.neighbor] = None
    if target not in distances:
        _fail(f"Filtered family target is disconnected from its owner: {target}")
    signature = signatures[target]
    if path_counts[target] != 1 or signature is None:
        _fail(f"Filtered family target has an ambiguous shortest joint path: {target}")
    return signature


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise CollisionFilterCompletionError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} must be a JSON object.")
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"Evidence contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_number(value: str) -> None:
    _fail(f"Evidence contains non-finite JSON number: {value}")


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _publish_content_addressed_plan(
    *,
    output_dir: Path,
    plan_sha256: str,
    payload: bytes,
) -> tuple[Path, bool]:
    output_dir = Path(os.path.abspath(output_dir.expanduser()))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = output_dir.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _fail(f"Plan output must be a regular directory: {output_dir}")
    plan_path = output_dir / f"collision-filter-completion-{plan_sha256}.json"
    if plan_path.exists():
        if plan_path.is_symlink() or not plan_path.is_file():
            _fail(f"Existing plan path is unsafe: {plan_path}")
        if plan_path.read_bytes() != payload:
            _fail(f"Existing content-addressed plan has conflicting bytes: {plan_path}")
        return plan_path, True

    descriptor, temporary_name = tempfile.mkstemp(prefix=".completion-", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, plan_path)
        except FileExistsError:
            if plan_path.is_symlink() or plan_path.read_bytes() != payload:
                _fail(
                    f"Concurrent plan publication produced conflicting bytes: {plan_path}"
                )
            return plan_path, True
        return plan_path, False
    finally:
        temporary.unlink(missing_ok=True)


def _require_inputs_unchanged(identities: tuple[_FileIdentity, ...]) -> None:
    for expected in identities:
        current = _capture_file_identity(expected.path)
        if current != expected:
            _fail(f"Completion input changed during plan generation: {expected.path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> NoReturn:
    raise CollisionFilterCompletionError(message)
