# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply an owner-approved, source-bound physics validation plan.

This module deliberately does not infer physics values or collider scope. The
caller supplies a frozen v1 JSON plan that binds the root asset and complete
dependency bundle, names every composed CollisionAPI prim, and names every
value to author. Work happens in a private package tree and is published only
after save/reopen validation succeeds.
"""

from __future__ import annotations

import argparse
import calendar
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from filelock import FileLock
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

PHYSICS_PROFILE_PLAN_SCHEMA_VERSION: Literal[
    "content-agent-workflows.physics-profile-plan.v1"
] = "content-agent-workflows.physics-profile-plan.v1"
PHYSICS_PROFILE_RECEIPT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.physics-profile-receipt.v1"
] = "content-agent-workflows.physics-profile-receipt.v1"

_ARTIFACT_DIGEST_SCHEMA = "content-agent-workflows.physics-profile-artifact.v1"
_RECEIPT_NAME = "physics-profile-receipt.json"
_ARTIFACT_DIR_NAME = "artifact"
_HASH_CHUNK_SIZE = 1024 * 1024
_DETERMINISTIC_ZIP_UTC_TIME = (1980, 1, 1, 0, 0, 0, -1, -1, -1)
_DETERMINISTIC_ZIP_DOS_TIME = 0
_DETERMINISTIC_ZIP_DOS_DATE = 33
_ZIP_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
_ZIP_CENTRAL_HEADER_SIGNATURE = b"PK\x01\x02"
_ZIP_CENTRAL_HEADER_SIZE = 46
_USD_LAYER_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
_USD_SUFFIXES = _USD_LAYER_SUFFIXES | {".usdz"}
_PHYSX_COLLISION_SCHEMA_TOKEN = "PhysxCollisionAPI"
_PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN = "PhysxSDFMeshCollisionAPI"
_PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN = "PhysxConvexHullCollisionAPI"
_GPRIM_COLLIDER_SCHEMA_TOKENS = (
    "MaterialBindingAPI",
    _PHYSX_COLLISION_SCHEMA_TOKEN,
)
_MESH_COLLIDER_SCHEMA_TOKENS = (
    "PhysicsMeshCollisionAPI",
    "MaterialBindingAPI",
    _PHYSX_COLLISION_SCHEMA_TOKEN,
    _PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN,
)
_PHYSICS_BINDING_RELATIONSHIP = "material:binding:physics"
_PHYSICS_COLLECTION_BINDING_PREFIX = "material:binding:collection:physics:"
_FOUNDATION_PRESERVED_APPROXIMATION_TOKENS = frozenset({"convexHull", "sdf"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIM_COMPONENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*$")
_WINDOWS_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


class PhysicsProfilePlanError(ValueError):
    """Raised when a plan cannot be applied without inference or mutation."""


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PhysicsProfileApprovalV1(_FrozenStrictModel):
    """Explicit evidence that the exact plan values were owner approved."""

    approved: Literal[True]
    owner_identity: str
    evidence: str

    @field_validator("owner_identity", "evidence")  # type: ignore[misc]
    @classmethod
    def _nonempty_exact_text(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("approval text must be nonempty and already trimmed")
        return value


class PhysicsMaterialAuthoredOpinionsV1(_FrozenStrictModel):
    """Exact source authored-opinion inventory for material reuse."""

    static_friction: bool
    dynamic_friction: bool
    restitution: bool
    density: bool


class PhysicsMaterialPlanV1(_FrozenStrictModel):
    """One exact physics material and its owner-approved values."""

    prim_path: str
    static_friction: float
    dynamic_friction: float
    restitution: float
    density: float | None
    operation: Literal["create", "reuse_existing"] = "create"
    source_authored_opinions: PhysicsMaterialAuthoredOpinionsV1 | None = None
    bind_missing_collider_paths: tuple[str, ...] = ()

    @field_validator("prim_path")  # type: ignore[misc]
    @classmethod
    def _valid_material_path(cls, value: str) -> str:
        _require_prim_path_text(value, field="physics_material.prim_path")
        return value

    @field_validator(  # type: ignore[misc]
        "static_friction",
        "dynamic_friction",
        "restitution",
        "density",
        mode="before",
    )
    @classmethod
    def _explicit_number(cls, value: Any, info: Any) -> float | None:
        if value is None and info.field_name == "density":
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("physics values must be explicit JSON numbers")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("physics values must be finite USD floats") from exc
        if not math.isfinite(number) or abs(number) > _float32_max():
            raise ValueError("physics values must be finite USD floats")
        usd_number = _as_float32(number)
        if number != 0.0 and usd_number == 0.0:
            raise ValueError("physics values must not underflow USD floats")
        return usd_number

    @model_validator(mode="after")  # type: ignore[misc]
    def _physical_ranges(self) -> PhysicsMaterialPlanV1:
        if self.static_friction < 0.0 or self.dynamic_friction < 0.0:
            raise ValueError("friction values must be nonnegative")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be between zero and one")
        if self.density is not None and self.density <= 0.0:
            raise ValueError("density must be positive when supplied")
        if tuple(sorted(self.bind_missing_collider_paths)) != (
            self.bind_missing_collider_paths
        ):
            raise ValueError("bind_missing_collider_paths must be sorted")
        if len(set(self.bind_missing_collider_paths)) != len(
            self.bind_missing_collider_paths
        ):
            raise ValueError("bind_missing_collider_paths must be unique")
        for path in self.bind_missing_collider_paths:
            _require_prim_path_text(
                path,
                field="physics_material.bind_missing_collider_paths",
            )
        if self.operation == "create":
            if self.source_authored_opinions is not None:
                raise ValueError(
                    "create material operation cannot declare source opinions"
                )
            if self.bind_missing_collider_paths:
                raise ValueError(
                    "create material operation cannot declare missing bindings"
                )
            return self
        if self.source_authored_opinions is None:
            raise ValueError(
                "reuse_existing material operation requires source opinions"
            )
        expected = self.source_authored_opinions
        for field, authored in (
            ("static_friction", expected.static_friction),
            ("dynamic_friction", expected.dynamic_friction),
            ("restitution", expected.restitution),
        ):
            if not authored and getattr(self, field) != 0.0:
                raise ValueError(
                    f"unauthored reuse value {field} must equal its zero fallback"
                )
        if expected.density != (self.density is not None):
            raise ValueError(
                "reuse_existing density value and authored-opinion presence disagree"
            )
        return self


class PhysicsColliderApproximationPlanV1(_FrozenStrictModel):
    """Exact operation for one composed Mesh collider approximation."""

    prim_path: str
    operation: Literal["author_sdf", "preserve_existing"] = "author_sdf"
    source_token: Literal["convexHull", "sdf"] | None = None

    @field_validator("prim_path")  # type: ignore[misc]
    @classmethod
    def _valid_prim_path(cls, value: str) -> str:
        _require_prim_path_text(value, field="collider_approximations.prim_path")
        return value

    @model_validator(mode="after")  # type: ignore[misc]
    def _operation_contract(self) -> PhysicsColliderApproximationPlanV1:
        if self.operation == "preserve_existing" and self.source_token is None:
            raise ValueError("preserve_existing requires an exact source token")
        return self


class PhysicsProfilePlanV1(_FrozenStrictModel):
    """Frozen v1 plan for exact validation-preparation authoring."""

    schema_version: Literal["content-agent-workflows.physics-profile-plan.v1"]
    source_asset_sha256: str
    source_dependency_bundle_sha256: str
    collider_prim_paths: tuple[str, ...]
    mesh_approximation: Literal["sdf"]
    physics_material: PhysicsMaterialPlanV1
    approval: PhysicsProfileApprovalV1
    collider_approximations: tuple[PhysicsColliderApproximationPlanV1, ...] = ()

    @field_validator(  # type: ignore[misc]
        "source_asset_sha256", "source_dependency_bundle_sha256"
    )
    @classmethod
    def _valid_source_digest(cls, value: str, info: Any) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256")
        return value

    @field_validator("collider_prim_paths")  # type: ignore[misc]
    @classmethod
    def _valid_collider_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("collider_prim_paths must not be empty")
        for path in value:
            _require_prim_path_text(path, field="collider_prim_paths")
        if tuple(sorted(value)) != value:
            raise ValueError("collider_prim_paths must be sorted")
        if len(set(value)) != len(value):
            raise ValueError("collider_prim_paths must be unique")
        return value

    @model_validator(mode="after")  # type: ignore[misc]
    def _separate_material_namespace(self) -> PhysicsProfilePlanV1:
        material = self.physics_material.prim_path
        for collider in self.collider_prim_paths:
            if _path_is_prefix(material, collider) or _path_is_prefix(
                collider, material
            ):
                raise ValueError(
                    "physics material and collider paths must use disjoint namespaces"
                )
        approved_bindings = set(self.physics_material.bind_missing_collider_paths)
        unplanned_bindings = approved_bindings.difference(self.collider_prim_paths)
        if unplanned_bindings:
            raise ValueError(
                "missing-binding approvals must name planned colliders: "
                f"{sorted(unplanned_bindings)}"
            )
        approximation_paths = tuple(
            item.prim_path for item in self.collider_approximations
        )
        if tuple(sorted(approximation_paths)) != approximation_paths:
            raise ValueError("collider_approximations must be sorted by prim_path")
        if len(set(approximation_paths)) != len(approximation_paths):
            raise ValueError("collider_approximations must have unique prim paths")
        unplanned_approximations = set(approximation_paths).difference(
            self.collider_prim_paths
        )
        if unplanned_approximations:
            raise ValueError(
                "collider approximation entries must name planned colliders: "
                f"{sorted(unplanned_approximations)}"
            )
        return self


class PhysicsProfileArtifactFileV1(_FrozenStrictModel):
    relative_path: str
    sha256: str


class PhysicsProfilePreservedApproximationV1(_FrozenStrictModel):
    """Receipt evidence for one preserved source approximation opinion."""

    prim_path: str
    source_token: Literal["convexHull", "sdf"]


class PhysicsProfileAuthoredSdfTransitionV1(_FrozenStrictModel):
    """Receipt evidence for an exact source-bound transition to SDF."""

    prim_path: str
    source_token: Literal["convexHull", "sdf"]
    output_token: Literal["sdf"] = "sdf"


class PhysicsProfileReceiptV1(_FrozenStrictModel):
    """Deterministic machine receipt for a published derivative."""

    schema_version: Literal["content-agent-workflows.physics-profile-receipt.v1"]
    source_asset_sha256: str
    source_dependency_bundle_sha256: str
    plan_sha256: str
    output_asset_sha256: str
    output_artifact_sha256: str
    output_asset_relative_path: str
    artifact_files: tuple[PhysicsProfileArtifactFileV1, ...]
    physics_material_operation: Literal["create", "reuse_existing"] = "create"
    reused_physics_material_path: str | None = None
    bound_missing_collider_paths: tuple[str, ...] = ()
    preserved_approximations: tuple[PhysicsProfilePreservedApproximationV1, ...] = ()
    authored_sdf_transitions: tuple[PhysicsProfileAuthoredSdfTransitionV1, ...] = ()


@dataclass(frozen=True)
class PhysicsProfilePlanResult:
    """Published derivative, receipt, and reuse status."""

    output_asset_path: Path
    receipt_path: Path
    receipt: PhysicsProfileReceiptV1
    reused_output: bool


@dataclass(frozen=True)
class PhysicsProfileSourceIdentity:
    """Exact source identity fields required by a v1 profile plan."""

    source_asset_sha256: str
    source_dependency_bundle_sha256: str


@dataclass(frozen=True)
class _FileRecord:
    path: Path
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class _SourceSnapshot:
    source_path: Path
    package_root: Path
    source_relative_path: str
    source_asset_sha256: str
    files: tuple[_FileRecord, ...]
    dependency_bundle_sha256: str


@dataclass(frozen=True)
class _StagedSource:
    tree_root: Path
    root_asset: Path
    root_relative_path: str
    dependency_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _TokenListOpState:
    is_explicit: bool
    explicit: tuple[str, ...]
    added: tuple[str, ...]
    prepended: tuple[str, ...]
    appended: tuple[str, ...]
    deleted: tuple[str, ...]
    ordered: tuple[str, ...]


@dataclass(frozen=True)
class _PhysicsBindingState:
    prim_path: str
    custom: bool
    targets: tuple[str, ...]
    binding_strength: str | None
    spec_fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class _AuthoredProfileState:
    list_ops: dict[str, _TokenListOpState]
    material_spec_fingerprints: tuple[str, ...] | None
    binding_states: tuple[_PhysicsBindingState, ...]
    preserved_approximation_fingerprints: tuple[tuple[str, tuple[str, ...]], ...]


def inspect_physics_profile_source(
    source_asset: str | Path,
) -> PhysicsProfileSourceIdentity:
    """Compute the root and dependency-bundle identities required by a plan."""

    source = _absolute_path(Path(source_asset).expanduser())
    snapshot = _capture_source_snapshot(source)
    return PhysicsProfileSourceIdentity(
        source_asset_sha256=snapshot.source_asset_sha256,
        source_dependency_bundle_sha256=snapshot.dependency_bundle_sha256,
    )


def verify_physics_profile_receipt(
    receipt_path: str | Path,
    *,
    plan_path: str | Path,
    source_asset: str | Path,
    output_asset: str | Path,
    expected_owner_identity: str | None = None,
) -> PhysicsProfileReceiptV1:
    """Replay an exact plan and verify its published source/output receipt."""

    receipt_file = _absolute_path(Path(receipt_path).expanduser())
    plan_file = _absolute_path(Path(plan_path).expanduser())
    source = _absolute_path(Path(source_asset).expanduser())
    output = _absolute_path(Path(output_asset).expanduser())
    _require_regular_nonsymlink_file(receipt_file, label="physics-profile receipt")
    try:
        receipt_bytes = receipt_file.read_bytes()
        receipt = PhysicsProfileReceiptV1.model_validate_json(
            receipt_bytes,
            strict=True,
        )
    except (OSError, ValidationError) as exc:
        raise PhysicsProfilePlanError(
            f"Invalid physics-profile receipt: {receipt_file}"
        ) from exc
    if receipt_bytes != _canonical_model_bytes(receipt):
        raise PhysicsProfilePlanError(
            "Physics-profile receipt is not the canonical published document."
        )
    plan, plan_sha256 = load_physics_profile_plan(plan_file)
    plan_bytes = plan_file.read_bytes()
    if plan_sha256 != receipt.plan_sha256:
        raise PhysicsProfilePlanError(
            "Physics-profile receipt does not bind the exact approved plan."
        )
    if (
        expected_owner_identity is not None
        and plan.approval.owner_identity != expected_owner_identity
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile plan owner does not match the completion approval."
        )

    source_snapshot = _capture_source_snapshot(source)
    if source_snapshot.source_asset_sha256 != receipt.source_asset_sha256:
        raise PhysicsProfilePlanError(
            "Physics-profile receipt source_asset_sha256 is stale."
        )
    if source_snapshot.dependency_bundle_sha256 != (
        receipt.source_dependency_bundle_sha256
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile receipt source dependency bundle is stale."
        )
    if (
        plan.source_asset_sha256 != receipt.source_asset_sha256
        or plan.source_dependency_bundle_sha256
        != receipt.source_dependency_bundle_sha256
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile plan and receipt source identities differ."
        )
    _require_receipt_matches_plan(receipt, plan=plan)

    publication_root = receipt_file.parent
    _reject_symlink_components(publication_root, allow_missing=False)
    if not publication_root.is_dir():
        raise PhysicsProfilePlanError(
            f"Physics-profile publication is not a directory: {publication_root}"
        )
    expected_publication_name = f"physics-profile-{receipt.output_artifact_sha256}"
    if publication_root.name != expected_publication_name:
        raise PhysicsProfilePlanError(
            "Physics-profile publication path does not match its artifact digest."
        )

    artifact_root = publication_root / _ARTIFACT_DIR_NAME
    _reject_symlink_components(artifact_root, allow_missing=False)
    if not artifact_root.is_dir():
        raise PhysicsProfilePlanError(
            f"Physics-profile artifact directory is missing: {artifact_root}"
        )
    actual_records = _directory_file_records(
        artifact_root,
        relative_to=publication_root,
    )
    expected_records: list[tuple[str, str]] = []
    for item in receipt.artifact_files:
        parts = item.relative_path.split("/")
        if (
            not item.relative_path
            or "\\" in item.relative_path
            or item.relative_path.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or not _SHA256_RE.fullmatch(item.sha256)
        ):
            raise PhysicsProfilePlanError(
                "Physics-profile receipt contains an unsafe artifact record."
            )
        expected_records.append((item.relative_path, item.sha256))
    if expected_records != sorted(expected_records) or len(expected_records) != len(
        set(expected_records)
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile receipt artifact inventory is not canonical."
        )
    actual_record_identities = [
        (record.relative_path, record.sha256) for record in actual_records
    ]
    if actual_record_identities != expected_records:
        raise PhysicsProfilePlanError(
            "Physics-profile receipt artifact inventory does not match the "
            "published tree."
        )

    output_relative = _relative_to(output, publication_root)
    if (
        output_relative is None
        or output_relative.as_posix() != receipt.output_asset_relative_path
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile receipt does not bind the exact output asset path."
        )
    _require_regular_nonsymlink_file(output, label="physics-profile output asset")
    output_sha256 = _file_sha256(output)
    if output_sha256 != receipt.output_asset_sha256:
        raise PhysicsProfilePlanError(
            "Physics-profile receipt output_asset_sha256 is stale."
        )
    if (receipt.output_asset_relative_path, output_sha256) not in expected_records:
        raise PhysicsProfilePlanError(
            "Physics-profile artifact inventory omits the output asset."
        )

    output_snapshot = _capture_source_snapshot(output)
    for record in output_snapshot.files:
        if _relative_to(record.path, artifact_root) is None:
            raise PhysicsProfilePlanError(
                "Physics-profile output has a dependency outside its artifact tree."
            )
    artifact_sha256 = _artifact_digest(
        actual_records,
        plan_sha256=receipt.plan_sha256,
        source_asset_sha256=receipt.source_asset_sha256,
    )
    if artifact_sha256 != receipt.output_artifact_sha256:
        raise PhysicsProfilePlanError(
            "Physics-profile receipt output artifact digest is stale."
        )

    temporary_root = Path(tempfile.mkdtemp(prefix=".physics-profile-verify-"))
    try:
        staged = _stage_source(source_snapshot, temporary_root / "source-tree")
        expected_state = _author_private_tree(staged, plan)
        _verify_derivative(
            output,
            plan=plan,
            expected_state=expected_state,
            expected_dependency_hashes=staged.dependency_hashes,
            package_root=artifact_root,
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    if _capture_source_snapshot(source) != source_snapshot:
        raise PhysicsProfilePlanError(
            "Physics-profile source changed during receipt verification."
        )
    if _capture_source_snapshot(output) != output_snapshot:
        raise PhysicsProfilePlanError(
            "Physics-profile output changed during receipt verification."
        )
    if (
        _directory_file_records(artifact_root, relative_to=publication_root)
        != actual_records
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile artifact tree changed during receipt verification."
        )
    if receipt_file.read_bytes() != receipt_bytes:
        raise PhysicsProfilePlanError(
            "Physics-profile receipt changed during verification."
        )
    if plan_file.read_bytes() != plan_bytes:
        raise PhysicsProfilePlanError(
            "Physics-profile plan changed during verification."
        )
    return receipt


def _require_receipt_matches_plan(
    receipt: PhysicsProfileReceiptV1,
    *,
    plan: PhysicsProfilePlanV1,
) -> None:
    expected_reused_path = (
        plan.physics_material.prim_path
        if plan.physics_material.operation == "reuse_existing"
        else None
    )
    expected_preserved = tuple(
        PhysicsProfilePreservedApproximationV1(
            prim_path=item.prim_path,
            source_token=item.source_token,
        )
        for item in plan.collider_approximations
        if item.operation == "preserve_existing" and item.source_token is not None
    )
    expected_transitions = tuple(
        PhysicsProfileAuthoredSdfTransitionV1(
            prim_path=item.prim_path,
            source_token=item.source_token,
        )
        for item in plan.collider_approximations
        if item.operation == "author_sdf" and item.source_token is not None
    )
    if (
        receipt.physics_material_operation != plan.physics_material.operation
        or receipt.reused_physics_material_path != expected_reused_path
        or receipt.bound_missing_collider_paths
        != plan.physics_material.bind_missing_collider_paths
        or receipt.preserved_approximations != expected_preserved
        or receipt.authored_sdf_transitions != expected_transitions
    ):
        raise PhysicsProfilePlanError(
            "Physics-profile receipt operations do not match the approved plan."
        )


def author_physics_profile_plan(
    source_asset: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
) -> PhysicsProfilePlanResult:
    """Apply one strict plan and atomically publish a content-addressed result."""

    source = _absolute_path(Path(source_asset).expanduser())
    plan_file = _absolute_path(Path(plan_path).expanduser())
    output_root = _absolute_path(Path(output_dir).expanduser())
    plan, plan_sha256 = load_physics_profile_plan(plan_file)
    snapshot = _capture_source_snapshot(source)
    if snapshot.source_asset_sha256 != plan.source_asset_sha256:
        raise PhysicsProfilePlanError(
            "Plan source_asset_sha256 does not match the exact source bytes."
        )
    if snapshot.dependency_bundle_sha256 != plan.source_dependency_bundle_sha256:
        raise PhysicsProfilePlanError(
            "Plan source_dependency_bundle_sha256 does not match the exact "
            "source dependency bundle."
        )

    _prepare_output_root(output_root, source=source)
    temporary_root = Path(
        tempfile.mkdtemp(dir=output_root, prefix=".physics-profile-work-")
    )
    try:
        staged = _stage_source(snapshot, temporary_root / "source-tree")
        expected_state = _author_private_tree(staged, plan)
        _require_snapshot_unchanged(snapshot)

        publication = temporary_root / "publication"
        artifact_dir = publication / _ARTIFACT_DIR_NAME
        artifact_dir.parent.mkdir(parents=True, exist_ok=False)
        if source.suffix.lower() == ".usdz":
            artifact_dir.mkdir()
            output_asset = artifact_dir / source.name
            _create_deterministic_usdz(
                staged.root_asset,
                output_asset,
                package_root=staged.tree_root,
            )
        else:
            staged.tree_root.rename(artifact_dir)
            output_asset = artifact_dir / staged.root_relative_path

        _verify_derivative(
            output_asset,
            plan=plan,
            expected_state=expected_state,
            expected_dependency_hashes=staged.dependency_hashes,
            package_root=artifact_dir,
        )
        artifact_files = _directory_file_records(
            artifact_dir,
            relative_to=publication,
        )
        artifact_sha256 = _artifact_digest(
            artifact_files,
            plan_sha256=plan_sha256,
            source_asset_sha256=snapshot.source_asset_sha256,
        )
        output_relative = output_asset.relative_to(publication).as_posix()
        output_sha256 = _file_sha256(output_asset)
        receipt = PhysicsProfileReceiptV1(
            schema_version=PHYSICS_PROFILE_RECEIPT_SCHEMA_VERSION,
            source_asset_sha256=snapshot.source_asset_sha256,
            source_dependency_bundle_sha256=snapshot.dependency_bundle_sha256,
            plan_sha256=plan_sha256,
            output_asset_sha256=output_sha256,
            output_artifact_sha256=artifact_sha256,
            output_asset_relative_path=output_relative,
            artifact_files=tuple(
                PhysicsProfileArtifactFileV1(
                    relative_path=record.relative_path,
                    sha256=record.sha256,
                )
                for record in artifact_files
            ),
            physics_material_operation=plan.physics_material.operation,
            reused_physics_material_path=(
                plan.physics_material.prim_path
                if plan.physics_material.operation == "reuse_existing"
                else None
            ),
            bound_missing_collider_paths=(
                plan.physics_material.bind_missing_collider_paths
            ),
            preserved_approximations=tuple(
                PhysicsProfilePreservedApproximationV1(
                    prim_path=item.prim_path,
                    source_token=item.source_token,
                )
                for item in plan.collider_approximations
                if item.operation == "preserve_existing"
                and item.source_token is not None
            ),
            authored_sdf_transitions=tuple(
                PhysicsProfileAuthoredSdfTransitionV1(
                    prim_path=item.prim_path,
                    source_token=item.source_token,
                )
                for item in plan.collider_approximations
                if item.operation == "author_sdf" and item.source_token is not None
            ),
        )
        receipt_path = publication / _RECEIPT_NAME
        _write_new_file(receipt_path, _canonical_model_bytes(receipt))
        _fsync_tree(publication)
        final_expected_files = _directory_file_records(
            publication,
            relative_to=publication,
        )
        final_path = output_root / f"physics-profile-{artifact_sha256}"
        lock_path = final_path.with_name(f".{final_path.name}.lock")
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise PhysicsProfilePlanError(
                f"Publication lock path is not a regular file: {lock_path}"
            )

        with FileLock(str(lock_path)):
            created_publication = False
            try:
                _require_snapshot_unchanged(snapshot)
                reused_output = _publish_directory(
                    publication,
                    final_path,
                    expected_files=final_expected_files,
                )
                created_publication = not reused_output
                published_asset = final_path / output_relative
                published_receipt = final_path / _RECEIPT_NAME
                _verify_published_receipt(published_receipt, receipt)
                _verify_derivative(
                    published_asset,
                    plan=plan,
                    expected_state=expected_state,
                    expected_dependency_hashes=staged.dependency_hashes,
                    package_root=final_path / _ARTIFACT_DIR_NAME,
                )
                _require_snapshot_unchanged(snapshot)
                return PhysicsProfilePlanResult(
                    output_asset_path=published_asset,
                    receipt_path=published_receipt,
                    receipt=receipt,
                    reused_output=reused_output,
                )
            except Exception:
                if created_publication:
                    _remove_directory_if_exact(final_path, final_expected_files)
                raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def load_physics_profile_plan(
    plan_path: str | Path,
) -> tuple[PhysicsProfilePlanV1, str]:
    """Load strict JSON and return the plan plus its canonical SHA-256."""

    path = _absolute_path(Path(plan_path).expanduser())
    _require_regular_nonsymlink_file(path, label="physics profile plan")
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_json_float,
        )
        plan = PhysicsProfilePlanV1.model_validate_json(payload, strict=True)
    except PhysicsProfilePlanError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise PhysicsProfilePlanError(f"Invalid physics profile plan: {exc}") from exc
    canonical = _canonical_model_bytes(plan)
    return plan, hashlib.sha256(canonical).hexdigest()


def _capture_source_snapshot(source: Path) -> _SourceSnapshot:
    if source.suffix.lower() not in _USD_SUFFIXES:
        raise PhysicsProfilePlanError("Source must be a USD, USDA, USDC, or USDZ file.")
    _require_regular_nonsymlink_file(source, label="source asset")
    if source.suffix.lower() == ".usdz":
        _inspect_usdz(source)

    paths = _dependency_files(source)
    if source.suffix.lower() == ".usdz":
        package_root = source.parent
    else:
        package_root = _common_dependency_root(paths)
    records: list[_FileRecord] = []
    for path in sorted(paths, key=str):
        _require_regular_nonsymlink_file(path, label="source dependency")
        if source.suffix.lower() == ".usdz":
            if path != source:
                raise PhysicsProfilePlanError(
                    "USDZ source has an external filesystem dependency."
                )
            relative = source.name
        else:
            relative_path = _relative_to(path, package_root)
            if relative_path is None:  # pragma: no cover - common-root invariant
                raise PhysicsProfilePlanError(
                    f"USD dependency resolves outside the source package: {path}"
                )
            relative = relative_path.as_posix()
        records.append(
            _FileRecord(
                path=path,
                relative_path=relative,
                sha256=_file_sha256(path),
            )
        )
    source_digest = _file_sha256(source)
    if not any(record.path == source for record in records):
        raise PhysicsProfilePlanError("Dependency closure omitted the source asset.")
    # The second read is intentional: reject mutation after selecting the source
    # digest but before finalizing the dependency snapshot.
    if _file_sha256(source) != source_digest:
        raise PhysicsProfilePlanError(
            "Source asset changed while its dependency closure was inspected."
        )
    return _SourceSnapshot(
        source_path=source,
        package_root=package_root,
        source_relative_path=source.relative_to(package_root).as_posix(),
        source_asset_sha256=source_digest,
        files=tuple(records),
        dependency_bundle_sha256=_file_record_digest(records),
    )


def _dependency_files(asset_path: Path) -> set[Path]:
    try:
        from pxr import Ar, Sdf, UsdUtils
    except ImportError as exc:  # pragma: no cover - environment failure
        raise PhysicsProfilePlanError(
            f"OpenUSD Python APIs are unavailable: {exc}"
        ) from exc
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(asset_path))
    except Exception as exc:  # pragma: no cover - OpenUSD failures vary
        raise PhysicsProfilePlanError(
            f"Could not inspect USD dependency closure: {exc}"
        ) from exc
    unresolved_paths = sorted({str(value) for value in unresolved})
    if unresolved_paths:
        raise PhysicsProfilePlanError(
            "USD dependency closure contains unresolved paths: "
            + ", ".join(unresolved_paths)
        )

    identifiers = [str(asset_path)]
    identifiers.extend(
        str(
            getattr(layer, "resolvedPath", "")
            or getattr(layer, "realPath", "")
            or getattr(layer, "identifier", "")
        )
        for layer in layers
    )
    identifiers.extend(
        str(getattr(asset, "resolvedPath", "") or getattr(asset, "path", "") or asset)
        for asset in assets
    )
    return {
        _physical_outer_path(
            identifier,
            base_dir=asset_path.parent,
            Ar=Ar,
            Sdf=Sdf,
        )
        for identifier in identifiers
    }


def _physical_outer_path(
    identifier: str,
    *,
    base_dir: Path,
    Ar: Any,
    Sdf: Any,
) -> Path:
    text = identifier.strip().strip("@")
    if not text:
        raise PhysicsProfilePlanError("USD dependency has no stable identifier.")
    try:
        outer, _arguments = Sdf.Layer.SplitIdentifier(text)
    except Exception as exc:
        raise PhysicsProfilePlanError(
            f"USD dependency identifier is invalid: {identifier}"
        ) from exc
    while Ar.IsPackageRelativePath(outer):
        outer, _member = Ar.SplitPackageRelativePathOuter(outer)
    if (
        not outer
        or "://" in outer
        or outer.startswith(("anon:", "file:"))
        or _WINDOWS_DRIVE_PATH_RE.match(outer)
    ):
        raise PhysicsProfilePlanError(
            f"USD dependency is not a supported local path: {identifier}"
        )
    path = Path(outer)
    if not path.is_absolute():
        path = base_dir / path
    return _absolute_path(path)


def _common_dependency_root(paths: set[Path]) -> Path:
    if not paths:  # pragma: no cover - source is always included
        raise PhysicsProfilePlanError("USD dependency closure is empty.")
    try:
        common = Path(os.path.commonpath([str(path.parent) for path in paths]))
    except ValueError as exc:
        raise PhysicsProfilePlanError(
            "USD dependency closure spans incompatible filesystem roots."
        ) from exc
    common = _absolute_path(common)
    if common == Path(common.anchor):
        raise PhysicsProfilePlanError(
            "USD dependency closure has no bounded local package root."
        )
    return common


def _stage_source(snapshot: _SourceSnapshot, tree_root: Path) -> _StagedSource:
    tree_root.mkdir(parents=True, exist_ok=False)
    if snapshot.source_path.suffix.lower() == ".usdz":
        private_archive = tree_root.parent / ".source-archive.usdz"
        _copy_regular_file(snapshot.source_path, private_archive)
        if _file_sha256(private_archive) != snapshot.source_asset_sha256:
            raise PhysicsProfilePlanError(
                "Private USDZ source copy failed exact readback."
            )
        root_relative_path = _extract_usdz(private_archive, tree_root)
        if _file_sha256(private_archive) != snapshot.source_asset_sha256:
            raise PhysicsProfilePlanError(
                "Private USDZ source copy changed during extraction."
            )
        root_asset = tree_root / root_relative_path
    else:
        for record in snapshot.files:
            destination = tree_root / record.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_regular_file(record.path, destination)
            if _file_sha256(destination) != record.sha256:
                raise PhysicsProfilePlanError(
                    f"Private source copy failed exact readback: {record.relative_path}"
                )
        root_relative_path = snapshot.source_relative_path
        root_asset = tree_root / root_relative_path

    closure = _local_dependency_hashes(root_asset, tree_root)
    root_relative = root_asset.relative_to(tree_root).as_posix()
    if root_relative not in closure:
        raise PhysicsProfilePlanError(
            "Private dependency closure omitted its root layer."
        )
    return _StagedSource(
        tree_root=tree_root,
        root_asset=root_asset,
        root_relative_path=root_relative,
        dependency_hashes=tuple(
            sorted(
                (relative, digest)
                for relative, digest in closure.items()
                if relative != root_relative
            )
        ),
    )


def _author_private_tree(
    staged: _StagedSource,
    plan: PhysicsProfilePlanV1,
) -> _AuthoredProfileState:
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - environment failure
        raise PhysicsProfilePlanError(
            f"OpenUSD Python APIs are unavailable: {exc}"
        ) from exc
    try:
        stage = Usd.Stage.Open(str(staged.root_asset), load=Usd.Stage.LoadAll)
    except Exception as exc:  # pragma: no cover - OpenUSD failures vary
        raise PhysicsProfilePlanError(
            f"Could not open private USD stage: {exc}"
        ) from exc
    if stage is None:
        raise PhysicsProfilePlanError("Could not open private USD stage.")
    root_layer = stage.GetRootLayer()
    if bool(getattr(root_layer, "dirty", False)):
        raise PhysicsProfilePlanError("Private root layer opened with unsaved edits.")
    stage.SetEditTarget(root_layer)

    collider_paths = _collision_api_paths(stage, Usd=Usd, UsdPhysics=UsdPhysics)
    if collider_paths != plan.collider_prim_paths:
        missing = sorted(set(plan.collider_prim_paths) - set(collider_paths))
        extra = sorted(set(collider_paths) - set(plan.collider_prim_paths))
        raise PhysicsProfilePlanError(
            "Plan collider coverage does not exactly match composed CollisionAPI "
            f"prims; missing={missing}, extra={extra}."
        )

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsActive() or not default_prim.IsDefined():
        raise PhysicsProfilePlanError(
            "Source stage has no active, defined default prim."
        )
    material_path = Sdf.Path(plan.physics_material.prim_path)
    if not material_path.IsAbsolutePath() or not material_path.IsPrimPath():
        raise PhysicsProfilePlanError(
            "Physics material path is not an absolute prim path."
        )
    if not material_path.HasPrefix(default_prim.GetPath()):
        raise PhysicsProfilePlanError(
            "Physics material path must be beneath the source default prim."
        )
    parent = stage.GetPrimAtPath(material_path.GetParentPath())
    if not parent or not parent.IsActive() or not parent.IsDefined():
        raise PhysicsProfilePlanError(
            f"Physics material parent must already be active and defined: "
            f"{material_path.GetParentPath()}"
        )
    values = plan.physics_material
    material_spec_fingerprints: tuple[str, ...] | None = None
    if values.operation == "create":
        if stage.GetPrimAtPath(material_path):
            raise PhysicsProfilePlanError(
                f"Planned physics material path already exists: {material_path}"
            )
        existing_materials = sorted(
            str(prim.GetPath())
            for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd)
            if prim.HasAPI(UsdPhysics.MaterialAPI)
        )
        if existing_materials:
            raise PhysicsProfilePlanError(
                "Source already contains PhysicsMaterialAPI prims: "
                + ", ".join(existing_materials)
            )
        _validate_existing_physics_bindings(
            stage,
            Usd=Usd,
            collider_paths=plan.collider_prim_paths,
            material_path=material_path,
        )
        material = UsdShade.Material.Define(stage, material_path)
        if not material:
            raise PhysicsProfilePlanError(
                f"Could not define physics material at {material_path}."
            )
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        if not material_api:
            raise PhysicsProfilePlanError(
                f"Could not apply PhysicsMaterialAPI at {material_path}."
            )
        _set_required_attribute(
            material_api.CreateStaticFrictionAttr(values.static_friction),
            values.static_friction,
            name="physics:staticFriction",
        )
        _set_required_attribute(
            material_api.CreateDynamicFrictionAttr(values.dynamic_friction),
            values.dynamic_friction,
            name="physics:dynamicFriction",
        )
        _set_required_attribute(
            material_api.CreateRestitutionAttr(values.restitution),
            values.restitution,
            name="physics:restitution",
        )
        if values.density is not None:
            _set_required_attribute(
                material_api.CreateDensityAttr(values.density),
                values.density,
                name="physics:density",
            )
    else:
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        material_spec_fingerprints = _validate_reused_material(
            stage,
            values,
            Sdf=Sdf,
            Usd=Usd,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        missing_bindings = _reuse_missing_binding_paths(
            stage,
            collider_paths=plan.collider_prim_paths,
            material_path=material_path,
            Usd=Usd,
            UsdShade=UsdShade,
        )
        if missing_bindings != values.bind_missing_collider_paths:
            raise PhysicsProfilePlanError(
                "reuse_existing missing-binding approvals do not exactly match "
                f"source state; expected={list(missing_bindings)}, "
                f"approved={list(values.bind_missing_collider_paths)}."
            )

    approximation_plans = _resolved_approximation_plans(
        stage,
        plan,
        UsdGeom=UsdGeom,
    )

    before_list_ops: dict[str, _TokenListOpState | None] = {}
    required_schemas: dict[str, tuple[str, ...]] = {}
    preserved_approximation_fingerprints: dict[str, tuple[str, ...]] = {}
    for path_text in plan.collider_prim_paths:
        path = Sdf.Path(path_text)
        prim = stage.GetPrimAtPath(path)
        if (
            not prim
            or not prim.IsActive()
            or not prim.IsDefined()
            or prim.IsInstance()
            or prim.IsInstanceProxy()
            or not prim.IsA(UsdGeom.Gprim)
        ):
            raise PhysicsProfilePlanError(
                f"Planned collider is not an editable active Gprim: {path_text}"
            )
        is_mesh = prim.IsA(UsdGeom.Mesh)
        approximation_plan = approximation_plans.get(path_text)
        needs_direct_binding = (
            values.operation == "create"
            or path_text in values.bind_missing_collider_paths
        )
        has_direct_binding_api = prim.HasAPI(UsdShade.MaterialBindingAPI)
        uses_sdf = bool(
            is_mesh
            and (
                approximation_plan is None
                or approximation_plan.operation == "author_sdf"
                or approximation_plan.source_token == "sdf"
            )
        )
        uses_convex_hull = bool(
            is_mesh
            and approximation_plan is not None
            and approximation_plan.operation == "preserve_existing"
            and approximation_plan.source_token == "convexHull"
        )
        required = _required_collider_schema_tokens(
            is_mesh=is_mesh,
            uses_sdf=uses_sdf,
            uses_convex_hull=uses_convex_hull,
            requires_material_binding=(needs_direct_binding or has_direct_binding_api),
        )
        required_schemas[path_text] = required
        approximation_fingerprint = _validate_collider_conflicts(
            stage,
            prim,
            material_path=material_path,
            is_mesh=is_mesh,
            approximation_plan=approximation_plan,
            source_state=True,
            Sdf=Sdf,
        )
        if approximation_fingerprint is not None:
            preserved_approximation_fingerprints[path_text] = approximation_fingerprint
        _reject_deleted_required_schemas(
            prim,
            required=required,
        )
        before_list_ops[path_text] = _root_api_list_op_state(root_layer, path)

        if is_mesh:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            if not mesh_collision:
                raise PhysicsProfilePlanError(
                    f"Could not apply PhysicsMeshCollisionAPI at {path_text}."
                )
            if (
                approximation_plan is None
                or approximation_plan.operation == "author_sdf"
            ):
                approximation = mesh_collision.CreateApproximationAttr(
                    plan.mesh_approximation,
                    writeSparsely=False,
                )
                if not approximation or not approximation.Set(plan.mesh_approximation):
                    raise PhysicsProfilePlanError(
                        f"Could not author physics:approximation at {path_text}."
                    )
        if needs_direct_binding:
            material_binding = UsdShade.MaterialBindingAPI.Apply(prim)
            if not material_binding or not material_binding.Bind(
                material,
                materialPurpose="physics",
            ):
                raise PhysicsProfilePlanError(
                    f"Could not bind physics material at {path_text}."
                )
        raw_physx_tokens = [_PHYSX_COLLISION_SCHEMA_TOKEN]
        if uses_sdf:
            raw_physx_tokens.append(_PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN)
        elif uses_convex_hull:
            raw_physx_tokens.append(_PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN)
        for token in raw_physx_tokens:
            if not prim.AddAppliedSchema(token):
                raise PhysicsProfilePlanError(
                    f"Could not author raw applied-schema token {token} at {path_text}."
                )
    after_list_ops: dict[str, _TokenListOpState] = {}
    for path_text in plan.collider_prim_paths:
        after = _root_api_list_op_state(root_layer, Sdf.Path(path_text))
        if after is None:
            raise PhysicsProfilePlanError(
                f"Collider has no authored apiSchemas list-op: {path_text}"
            )
        _require_list_op_preserved(
            before_list_ops[path_text],
            after,
            required=required_schemas[path_text],
            path=path_text,
        )
        after_list_ops[path_text] = after

    try:
        root_layer.Save()
    except Exception as exc:  # pragma: no cover - OpenUSD failures vary
        raise PhysicsProfilePlanError(
            f"Could not save private USD layer: {exc}"
        ) from exc
    if bool(getattr(root_layer, "dirty", False)):
        raise PhysicsProfilePlanError(
            "Private USD root layer remained dirty after save."
        )
    if material_spec_fingerprints is not None:
        actual_material_fingerprints = _spec_stack_fingerprints(
            stage,
            material_path,
            Sdf=Sdf,
        )
        if actual_material_fingerprints != material_spec_fingerprints:
            raise PhysicsProfilePlanError(
                "reuse_existing changed the existing material prim opinions."
            )
    for (
        path_text,
        expected_fingerprints,
    ) in preserved_approximation_fingerprints.items():
        actual_fingerprints = _spec_stack_fingerprints(
            stage,
            Sdf.Path(path_text).AppendProperty("physics:approximation"),
            Sdf=Sdf,
        )
        if actual_fingerprints != expected_fingerprints:
            raise PhysicsProfilePlanError(
                f"preserve_existing changed physics:approximation at {path_text}."
            )
    if values.operation == "reuse_existing":
        remaining_missing = _reuse_missing_binding_paths(
            stage,
            collider_paths=plan.collider_prim_paths,
            material_path=material_path,
            Usd=Usd,
            UsdShade=UsdShade,
        )
        if remaining_missing:
            raise PhysicsProfilePlanError(
                "reuse_existing left colliders without the approved material: "
                f"{list(remaining_missing)}."
            )
        binding_states = _physics_binding_states(stage, Sdf=Sdf, Usd=Usd)
    else:
        binding_states = ()
    expected_state = _AuthoredProfileState(
        list_ops=after_list_ops,
        material_spec_fingerprints=material_spec_fingerprints,
        binding_states=binding_states,
        preserved_approximation_fingerprints=tuple(
            sorted(preserved_approximation_fingerprints.items())
        ),
    )
    del material, stage

    current_dependencies = _local_dependency_hashes(
        staged.root_asset,
        staged.tree_root,
    )
    for relative, expected_sha in staged.dependency_hashes:
        if current_dependencies.get(relative) != expected_sha:
            raise PhysicsProfilePlanError(
                f"Authoring changed a source dependency: {relative}"
            )
    _verify_profile(staged.root_asset, plan, expected_state=expected_state)
    return expected_state


def _validate_existing_physics_bindings(
    stage: Any,
    *,
    Usd: Any,
    collider_paths: tuple[str, ...],
    material_path: Any,
) -> None:
    planned = set(collider_paths)
    for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd):
        collection_bindings = _authored_collection_physics_bindings(prim)
        if collection_bindings:
            raise PhysicsProfilePlanError(
                f"Conflicting collection-based physics material binding exists at "
                f"{prim.GetPath()}: {list(collection_bindings)}."
            )
        relationship = prim.GetRelationship(_PHYSICS_BINDING_RELATIONSHIP)
        if not relationship or not relationship.GetPropertyStack():
            continue
        path = str(prim.GetPath())
        if path not in planned or relationship.GetTargets() != [material_path]:
            raise PhysicsProfilePlanError(
                f"Conflicting material:binding:physics exists at {path}."
            )


def _validate_reused_material(
    stage: Any,
    plan: PhysicsMaterialPlanV1,
    *,
    Sdf: Any,
    Usd: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> tuple[str, ...]:
    material_paths = tuple(
        sorted(
            str(prim.GetPath())
            for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd)
            if prim.HasAPI(UsdPhysics.MaterialAPI)
        )
    )
    if material_paths != (plan.prim_path,):
        raise PhysicsProfilePlanError(
            "reuse_existing requires exactly the approved PhysicsMaterialAPI "
            f"prim; found={list(material_paths)}."
        )
    prim = stage.GetPrimAtPath(plan.prim_path)
    if (
        not prim
        or not prim.IsActive()
        or not prim.IsDefined()
        or prim.IsInstance()
        or prim.IsInstanceProxy()
        or not prim.IsA(UsdShade.Material)
    ):
        raise PhysicsProfilePlanError(
            "reuse_existing material must be an editable active UsdShade Material."
        )
    fingerprints = _spec_stack_fingerprints(stage, prim.GetPath(), Sdf=Sdf)
    if len(fingerprints) != 1:
        raise PhysicsProfilePlanError(
            "reuse_existing material prim has ambiguous authored opinions."
        )
    opinions = plan.source_authored_opinions
    if opinions is None:  # pragma: no cover - schema invariant
        raise PhysicsProfilePlanError(
            "reuse_existing plan omitted source authored opinions."
        )
    material_api = UsdPhysics.MaterialAPI(prim)
    _verify_reused_float_attribute(
        material_api.GetStaticFrictionAttr(),
        expected=plan.static_friction,
        expected_authored=opinions.static_friction,
        name="physics:staticFriction",
    )
    _verify_reused_float_attribute(
        material_api.GetDynamicFrictionAttr(),
        expected=plan.dynamic_friction,
        expected_authored=opinions.dynamic_friction,
        name="physics:dynamicFriction",
    )
    _verify_reused_float_attribute(
        material_api.GetRestitutionAttr(),
        expected=plan.restitution,
        expected_authored=opinions.restitution,
        name="physics:restitution",
    )
    _verify_reused_float_attribute(
        material_api.GetDensityAttr(),
        expected=plan.density if plan.density is not None else 0.0,
        expected_authored=opinions.density,
        name="physics:density",
    )
    return fingerprints


def _verify_reused_float_attribute(
    attribute: Any,
    *,
    expected: float,
    expected_authored: bool,
    name: str,
) -> None:
    if not attribute or attribute.IsCustom() or attribute.GetNumTimeSamples() != 0:
        raise PhysicsProfilePlanError(
            f"reuse_existing has unsafe or time-varying {name}."
        )
    property_stack = attribute.GetPropertyStack()
    if attribute.HasAuthoredValueOpinion() != expected_authored:
        raise PhysicsProfilePlanError(
            f"reuse_existing authored-opinion presence drifted for {name}."
        )
    expected_stack_size = 1 if expected_authored else 0
    if len(property_stack) != expected_stack_size:
        raise PhysicsProfilePlanError(
            f"reuse_existing has ambiguous authored opinions for {name}."
        )
    actual = attribute.Get()
    if actual is None or float(actual) != _as_float32(expected):
        raise PhysicsProfilePlanError(
            f"reuse_existing effective value drifted for {name}."
        )


def _reuse_missing_binding_paths(
    stage: Any,
    *,
    collider_paths: tuple[str, ...],
    material_path: Any,
    Usd: Any,
    UsdShade: Any,
) -> tuple[str, ...]:
    for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd):
        collection_bindings = _authored_collection_physics_bindings(prim)
        if collection_bindings:
            raise PhysicsProfilePlanError(
                "reuse_existing rejects collection-based physics material "
                f"bindings at {prim.GetPath()}: {list(collection_bindings)}."
            )
        relationship = prim.GetRelationship(_PHYSICS_BINDING_RELATIONSHIP)
        if not relationship or not relationship.GetPropertyStack():
            continue
        path_text = str(prim.GetPath())
        binding_strength = relationship.GetMetadata("bindMaterialAs")
        if (
            relationship.IsCustom()
            or relationship.GetTargets() != [material_path]
            or binding_strength not in (None, "weakerThanDescendants")
            or not prim.HasAPI(UsdShade.MaterialBindingAPI)
        ):
            raise PhysicsProfilePlanError(
                f"reuse_existing has an incompatible physics binding at {path_text}."
            )
        if not any(_path_is_prefix(path_text, path) for path in collider_paths):
            raise PhysicsProfilePlanError(
                f"reuse_existing found an unrelated physics binding at {path_text}."
            )

    missing: list[str] = []
    for path_text in collider_paths:
        prim = stage.GetPrimAtPath(path_text)
        direct_relationship = prim.GetRelationship(_PHYSICS_BINDING_RELATIONSHIP)
        if direct_relationship and direct_relationship.GetPropertyStack():
            continue
        bound_material, bound_relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial("physics")
        if bound_material:
            if bound_material.GetPath() != material_path:
                raise PhysicsProfilePlanError(
                    "reuse_existing collider resolves to an incompatible material "
                    f"at {path_text}: {bound_material.GetPath()}."
                )
        elif bound_relationship:
            raise PhysicsProfilePlanError(
                f"reuse_existing has an ambiguous binding at {path_text}."
            )
        missing.append(path_text)
    return tuple(sorted(missing))


def _physics_binding_states(
    stage: Any,
    *,
    Sdf: Any,
    Usd: Any,
) -> tuple[_PhysicsBindingState, ...]:
    states: list[_PhysicsBindingState] = []
    for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd):
        relationship = prim.GetRelationship(_PHYSICS_BINDING_RELATIONSHIP)
        if not relationship or not relationship.GetPropertyStack():
            continue
        strength = relationship.GetMetadata("bindMaterialAs")
        states.append(
            _PhysicsBindingState(
                prim_path=str(prim.GetPath()),
                custom=bool(relationship.IsCustom()),
                targets=tuple(str(path) for path in relationship.GetTargets()),
                binding_strength=str(strength) if strength is not None else None,
                spec_fingerprints=_spec_stack_fingerprints(
                    stage,
                    relationship.GetPath(),
                    Sdf=Sdf,
                ),
            )
        )
    return tuple(sorted(states, key=lambda state: state.prim_path))


def _spec_stack_fingerprints(
    stage: Any,
    path: Any,
    *,
    Sdf: Any,
) -> tuple[str, ...]:
    fingerprints: list[str] = []
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.GetObjectAtPath(path) is None:
            continue
        temporary = Sdf.Layer.CreateAnonymous()
        if path.IsPrimPath():
            destination = Sdf.Path("/Fingerprint")
        elif path.IsPropertyPath():
            Sdf.CreatePrimInLayer(temporary, Sdf.Path("/Fingerprint"))
            destination = Sdf.Path("/Fingerprint").AppendProperty(path.name)
        else:  # pragma: no cover - internal invariant
            raise PhysicsProfilePlanError(
                f"Cannot fingerprint unsupported USD spec path: {path}"
            )
        if not Sdf.CopySpec(layer, path, temporary, destination):
            raise PhysicsProfilePlanError(f"Could not fingerprint USD spec at {path}.")
        fingerprints.append(
            hashlib.sha256(temporary.ExportToString().encode("utf-8")).hexdigest()
        )
    return tuple(fingerprints)


def _resolved_approximation_plans(
    stage: Any,
    plan: PhysicsProfilePlanV1,
    *,
    UsdGeom: Any,
) -> dict[str, PhysicsColliderApproximationPlanV1]:
    if not plan.collider_approximations:
        return {}
    mesh_paths = tuple(
        path
        for path in plan.collider_prim_paths
        if stage.GetPrimAtPath(path).IsA(UsdGeom.Mesh)
    )
    approximation_paths = tuple(item.prim_path for item in plan.collider_approximations)
    if approximation_paths != mesh_paths:
        raise PhysicsProfilePlanError(
            "Explicit collider_approximations must exactly cover planned Mesh "
            f"colliders; expected={list(mesh_paths)}, "
            f"received={list(approximation_paths)}."
        )
    return {item.prim_path: item for item in plan.collider_approximations}


def _required_collider_schema_tokens(
    *,
    is_mesh: bool,
    uses_sdf: bool,
    uses_convex_hull: bool,
    requires_material_binding: bool,
) -> tuple[str, ...]:
    if is_mesh and uses_sdf and requires_material_binding:
        return _MESH_COLLIDER_SCHEMA_TOKENS
    if not is_mesh and requires_material_binding:
        return _GPRIM_COLLIDER_SCHEMA_TOKENS
    required: list[str] = []
    if is_mesh:
        required.append("PhysicsMeshCollisionAPI")
    if requires_material_binding:
        required.append("MaterialBindingAPI")
    required.append(_PHYSX_COLLISION_SCHEMA_TOKEN)
    if is_mesh and uses_sdf:
        required.append(_PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN)
    elif is_mesh and uses_convex_hull:
        required.append(_PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN)
    return tuple(required)


def _validate_collider_conflicts(
    stage: Any,
    prim: Any,
    *,
    material_path: Any,
    is_mesh: bool,
    approximation_plan: PhysicsColliderApproximationPlanV1 | None,
    source_state: bool,
    Sdf: Any,
) -> tuple[str, ...] | None:
    approximation = prim.GetAttribute("physics:approximation")
    preserved_fingerprints: tuple[str, ...] | None = None
    schema_operation = prim.GetMetadata("apiSchemas")
    applied = (
        tuple(str(value) for value in schema_operation.GetAppliedItems())
        if schema_operation is not None
        else ()
    )
    if approximation and approximation.GetPropertyStack():
        if not is_mesh:
            raise PhysicsProfilePlanError(
                f"Non-Mesh collider has mesh-only physics:approximation at "
                f"{prim.GetPath()}."
            )
    if is_mesh and approximation_plan is not None:
        if approximation_plan.operation == "preserve_existing":
            token = approximation_plan.source_token
            if token not in _FOUNDATION_PRESERVED_APPROXIMATION_TOKENS:
                raise PhysicsProfilePlanError(
                    f"Unsupported preserved approximation at {prim.GetPath()}: {token}."
                )
            if (
                not approximation
                or not approximation.HasAuthoredValueOpinion()
                or approximation.IsCustom()
                or approximation.GetNumTimeSamples() != 0
                or approximation.Get() != token
            ):
                raise PhysicsProfilePlanError(
                    "preserve_existing approximation state is missing, stale, or "
                    f"ambiguous at {prim.GetPath()}."
                )
            fingerprints = _spec_stack_fingerprints(
                stage,
                approximation.GetPath(),
                Sdf=Sdf,
            )
            if len(fingerprints) != 1:
                raise PhysicsProfilePlanError(
                    "preserve_existing approximation has ambiguous authored "
                    f"opinions at {prim.GetPath()}."
                )
            if token == "convexHull" and (
                _PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN in applied
            ):
                raise PhysicsProfilePlanError(
                    "preserve_existing non-SDF collider has an incompatible SDF "
                    f"schema at {prim.GetPath()}."
                )
            if token == "sdf" and (
                _PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN in applied
            ):
                raise PhysicsProfilePlanError(
                    "preserve_existing SDF collider has an incompatible convex-hull "
                    f"schema at {prim.GetPath()}."
                )
            preserved_fingerprints = fingerprints
        else:
            expected_token = (
                approximation_plan.source_token
                if source_state and approximation_plan.source_token is not None
                else "sdf"
            )
            has_opinion = bool(
                approximation and approximation.HasAuthoredValueOpinion()
            )
            if approximation_plan.source_token is not None and source_state:
                if not has_opinion:
                    raise PhysicsProfilePlanError(
                        "source-bound author_sdf approximation is missing at "
                        f"{prim.GetPath()}."
                    )
                fingerprints = _spec_stack_fingerprints(
                    stage,
                    approximation.GetPath(),
                    Sdf=Sdf,
                )
                if len(fingerprints) != 1:
                    raise PhysicsProfilePlanError(
                        "source-bound author_sdf approximation has ambiguous "
                        f"opinions at {prim.GetPath()}."
                    )
            if has_opinion and (
                approximation.IsCustom()
                or approximation.GetNumTimeSamples() != 0
                or approximation.Get() != expected_token
            ):
                raise PhysicsProfilePlanError(
                    f"Conflicting physics:approximation exists at {prim.GetPath()}."
                )
            if _PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN in applied:
                raise PhysicsProfilePlanError(
                    "author_sdf cannot preserve an incompatible convex-hull schema "
                    f"at {prim.GetPath()}."
                )
    elif is_mesh and approximation and approximation.GetPropertyStack():
        if (
            approximation.IsCustom()
            or approximation.GetNumTimeSamples() != 0
            or approximation.Get() != "sdf"
        ):
            raise PhysicsProfilePlanError(
                f"Conflicting physics:approximation exists at {prim.GetPath()}."
            )
    if (
        is_mesh
        and approximation_plan is None
        and _PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN in applied
    ):
        raise PhysicsProfilePlanError(
            "author_sdf cannot preserve an incompatible convex-hull schema at "
            f"{prim.GetPath()}."
        )
    if not is_mesh:
        invalid = [
            token
            for token in (
                "PhysicsMeshCollisionAPI",
                _PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN,
                _PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN,
            )
            if token in applied
        ]
        if invalid:
            raise PhysicsProfilePlanError(
                f"Non-Mesh collider has mesh-only applied schemas at "
                f"{prim.GetPath()}: {invalid}."
            )
    relationship = prim.GetRelationship(_PHYSICS_BINDING_RELATIONSHIP)
    if relationship and relationship.GetPropertyStack():
        binding_strength = relationship.GetMetadata("bindMaterialAs")
        if (
            relationship.IsCustom()
            or relationship.GetTargets() != [material_path]
            or binding_strength not in (None, "weakerThanDescendants")
        ):
            raise PhysicsProfilePlanError(
                f"Conflicting physics material binding exists at {prim.GetPath()}."
            )
    return preserved_fingerprints


def _reject_deleted_required_schemas(prim: Any, *, required: Sequence[str]) -> None:
    required_set = set(required)
    for spec in prim.GetPrimStack():
        if not spec.HasInfo("apiSchemas"):
            continue
        operation = spec.GetInfo("apiSchemas")
        deleted = required_set.intersection(
            str(value) for value in operation.deletedItems
        )
        if deleted:
            raise PhysicsProfilePlanError(
                f"Required applied schema is explicitly deleted at {prim.GetPath()}: "
                + ", ".join(sorted(deleted))
            )


def _require_list_op_preserved(
    before: _TokenListOpState | None,
    after: _TokenListOpState,
    *,
    required: Sequence[str],
    path: str,
) -> None:
    initial = before or _TokenListOpState(False, (), (), (), (), (), ())
    if initial.is_explicit != after.is_explicit and before is not None:
        raise PhysicsProfilePlanError(f"apiSchemas list-op mode changed at {path}.")
    if after.appended != initial.appended:
        raise PhysicsProfilePlanError(f"apiSchemas appended items changed at {path}.")
    if after.added != initial.added:
        raise PhysicsProfilePlanError(f"apiSchemas added items changed at {path}.")
    if after.deleted != initial.deleted:
        raise PhysicsProfilePlanError(f"apiSchemas deleted items changed at {path}.")
    if after.ordered != initial.ordered:
        raise PhysicsProfilePlanError(f"apiSchemas ordered items changed at {path}.")

    before_target = initial.explicit if initial.is_explicit else initial.prepended
    after_target = after.explicit if after.is_explicit else after.prepended
    if after_target[: len(before_target)] != before_target:
        raise PhysicsProfilePlanError(
            f"apiSchemas existing items were removed or reordered at {path}."
        )
    if any(value not in required for value in after_target[len(before_target) :]):
        raise PhysicsProfilePlanError(
            f"apiSchemas gained an unplanned token while authoring {path}."
        )
    applied = _apply_token_list_op(after)
    missing = [token for token in required if token not in applied]
    if missing:
        raise PhysicsProfilePlanError(
            f"apiSchemas lacks required tokens at {path}: {missing}"
        )


def _root_api_list_op_state(root_layer: Any, path: Any) -> _TokenListOpState | None:
    spec = root_layer.GetPrimAtPath(path)
    if spec is None or not spec.HasInfo("apiSchemas"):
        return None
    operation = spec.GetInfo("apiSchemas")
    return _TokenListOpState(
        is_explicit=bool(operation.isExplicit),
        explicit=tuple(str(value) for value in operation.explicitItems),
        added=tuple(str(value) for value in operation.addedItems),
        prepended=tuple(str(value) for value in operation.prependedItems),
        appended=tuple(str(value) for value in operation.appendedItems),
        deleted=tuple(str(value) for value in operation.deletedItems),
        ordered=tuple(str(value) for value in operation.orderedItems),
    )


def _apply_token_list_op(state: _TokenListOpState) -> tuple[str, ...]:
    if state.is_explicit:
        return state.explicit
    values = list(state.prepended)
    values.extend(value for value in state.added if value not in values)
    values.extend(value for value in state.appended if value not in values)
    values = [value for value in values if value not in set(state.deleted)]
    return tuple(values)


def _set_required_attribute(attribute: Any, value: float, *, name: str) -> None:
    if not attribute or not attribute.Set(value):
        raise PhysicsProfilePlanError(f"Could not author {name}.")


def _verify_profile(
    asset_path: Path,
    plan: PhysicsProfilePlanV1,
    *,
    expected_state: _AuthoredProfileState,
) -> None:
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:  # pragma: no cover - environment failure
        raise PhysicsProfilePlanError(
            f"OpenUSD Python APIs are unavailable: {exc}"
        ) from exc
    try:
        stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadAll)
    except Exception as exc:  # pragma: no cover - OpenUSD failures vary
        raise PhysicsProfilePlanError(f"Could not reopen derivative: {exc}") from exc
    if stage is None:
        raise PhysicsProfilePlanError("Could not reopen derivative.")
    root_layer = stage.GetRootLayer()
    if (
        _collision_api_paths(stage, Usd=Usd, UsdPhysics=UsdPhysics)
        != plan.collider_prim_paths
    ):
        raise PhysicsProfilePlanError("Derivative collider coverage changed on reopen.")

    material_paths = tuple(
        sorted(
            str(prim.GetPath())
            for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd)
            if prim.HasAPI(UsdPhysics.MaterialAPI)
        )
    )
    if material_paths != (plan.physics_material.prim_path,):
        raise PhysicsProfilePlanError(
            "Derivative must contain exactly the planned PhysicsMaterialAPI prim."
        )
    material_prim = stage.GetPrimAtPath(plan.physics_material.prim_path)
    material_api = UsdPhysics.MaterialAPI(material_prim)
    values = plan.physics_material
    if values.operation == "create":
        _verify_float_attribute(
            material_api.GetStaticFrictionAttr(),
            values.static_friction,
            name="physics:staticFriction",
        )
        _verify_float_attribute(
            material_api.GetDynamicFrictionAttr(),
            values.dynamic_friction,
            name="physics:dynamicFriction",
        )
        _verify_float_attribute(
            material_api.GetRestitutionAttr(),
            values.restitution,
            name="physics:restitution",
        )
        density = material_api.GetDensityAttr()
        if values.density is None:
            if density and density.HasAuthoredValueOpinion():
                raise PhysicsProfilePlanError(
                    "Derivative authored unplanned physics:density."
                )
        else:
            _verify_float_attribute(density, values.density, name="physics:density")
    else:
        material_fingerprints = _validate_reused_material(
            stage,
            values,
            Sdf=Sdf,
            Usd=Usd,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        if material_fingerprints != expected_state.material_spec_fingerprints:
            raise PhysicsProfilePlanError(
                "Derivative changed reuse_existing material prim opinions."
            )
        missing_bindings = _reuse_missing_binding_paths(
            stage,
            collider_paths=plan.collider_prim_paths,
            material_path=material_prim.GetPath(),
            Usd=Usd,
            UsdShade=UsdShade,
        )
        if missing_bindings:
            raise PhysicsProfilePlanError(
                "Derivative reuse_existing binding coverage is incomplete: "
                f"{list(missing_bindings)}."
            )
        if (
            _physics_binding_states(stage, Sdf=Sdf, Usd=Usd)
            != expected_state.binding_states
        ):
            raise PhysicsProfilePlanError(
                "Derivative reuse_existing physics bindings changed on readback."
            )

    if values.operation == "create":
        bound_paths: list[str] = []
        for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd):
            collection_bindings = _authored_collection_physics_bindings(prim)
            if collection_bindings:
                raise PhysicsProfilePlanError(
                    "Derivative retained a collection-based physics material "
                    f"binding at {prim.GetPath()}: {list(collection_bindings)}."
                )
            relationship = prim.GetRelationship(_PHYSICS_BINDING_RELATIONSHIP)
            if not relationship or not relationship.GetPropertyStack():
                continue
            path_text = str(prim.GetPath())
            bound_paths.append(path_text)
            binding_strength = relationship.GetMetadata("bindMaterialAs")
            if (
                relationship.IsCustom()
                or relationship.GetTargets() != [material_prim.GetPath()]
                or binding_strength not in (None, "weakerThanDescendants")
            ):
                raise PhysicsProfilePlanError(
                    "Derivative has an inexact physics material binding at "
                    f"{path_text}."
                )
        if tuple(sorted(bound_paths)) != plan.collider_prim_paths:
            raise PhysicsProfilePlanError(
                "Derivative physics material bindings do not exactly cover "
                "planned colliders."
            )

    approximation_plans = _resolved_approximation_plans(
        stage,
        plan,
        UsdGeom=UsdGeom,
    )
    expected_preserved = dict(expected_state.preserved_approximation_fingerprints)
    actual_preserved: dict[str, tuple[str, ...]] = {}
    for path_text in plan.collider_prim_paths:
        prim = stage.GetPrimAtPath(path_text)
        if (
            not prim
            or not prim.IsActive()
            or not prim.IsDefined()
            or prim.IsInstance()
            or prim.IsInstanceProxy()
            or not prim.IsA(UsdGeom.Gprim)
        ):
            raise PhysicsProfilePlanError(
                f"Derivative collider is not an editable active Gprim: {path_text}."
            )
        if (
            values.operation == "create"
            or path_text in values.bind_missing_collider_paths
        ) and not prim.HasAPI(UsdShade.MaterialBindingAPI):
            raise PhysicsProfilePlanError(
                f"Derivative lacks MaterialBindingAPI at {path_text}."
            )
        is_mesh = prim.IsA(UsdGeom.Mesh)
        approximation_plan = approximation_plans.get(path_text)
        preserved = _validate_collider_conflicts(
            stage,
            prim,
            material_path=material_prim.GetPath(),
            is_mesh=is_mesh,
            approximation_plan=approximation_plan,
            source_state=False,
            Sdf=Sdf,
        )
        if preserved is not None:
            actual_preserved[path_text] = preserved
        approximation = prim.GetAttribute("physics:approximation")
        if is_mesh and (
            approximation_plan is None or approximation_plan.operation == "author_sdf"
        ):
            if (
                not approximation
                or not approximation.HasAuthoredValueOpinion()
                or approximation.IsCustom()
                or approximation.GetNumTimeSamples() != 0
                or approximation.Get() != plan.mesh_approximation
            ):
                raise PhysicsProfilePlanError(
                    f"Derivative has an inexact physics:approximation at {path_text}."
                )
        elif not is_mesh and approximation and approximation.GetPropertyStack():
            raise PhysicsProfilePlanError(
                f"Derivative Non-Mesh collider has mesh-only physics:approximation "
                f"at {path_text}."
            )
        operation = prim.GetMetadata("apiSchemas")
        applied = tuple(str(value) for value in operation.GetAppliedItems())
        uses_sdf = bool(
            is_mesh
            and (
                approximation_plan is None
                or approximation_plan.operation == "author_sdf"
                or approximation_plan.source_token == "sdf"
            )
        )
        uses_convex_hull = bool(
            is_mesh
            and approximation_plan is not None
            and approximation_plan.operation == "preserve_existing"
            and approximation_plan.source_token == "convexHull"
        )
        required = _required_collider_schema_tokens(
            is_mesh=is_mesh,
            uses_sdf=uses_sdf,
            uses_convex_hull=uses_convex_hull,
            requires_material_binding=prim.HasAPI(UsdShade.MaterialBindingAPI),
        )
        missing = [token for token in required if token not in applied]
        if missing:
            raise PhysicsProfilePlanError(
                f"Derivative lacks required collider schemas at {path_text}: {missing}"
            )
        if not is_mesh:
            invalid = [
                token
                for token in (
                    "PhysicsMeshCollisionAPI",
                    _PHYSX_SDF_MESH_COLLISION_SCHEMA_TOKEN,
                    _PHYSX_CONVEX_HULL_COLLISION_SCHEMA_TOKEN,
                )
                if token in applied
            ]
            if invalid:
                raise PhysicsProfilePlanError(
                    f"Derivative Non-Mesh collider has mesh-only applied schemas "
                    f"at {path_text}: {invalid}."
                )
        actual_state = _root_api_list_op_state(root_layer, prim.GetPath())
        if actual_state != expected_state.list_ops[path_text]:
            raise PhysicsProfilePlanError(
                f"Derivative apiSchemas list-op changed on reopen at {path_text}."
            )
    if actual_preserved != expected_preserved:
        raise PhysicsProfilePlanError(
            "Derivative preserved approximation opinions changed on readback."
        )


def _verify_float_attribute(attribute: Any, expected: float, *, name: str) -> None:
    if (
        not attribute
        or not attribute.HasAuthoredValueOpinion()
        or attribute.GetNumTimeSamples() != 0
        or float(attribute.Get()) != _as_float32(expected)
    ):
        raise PhysicsProfilePlanError(f"Derivative failed exact readback for {name}.")


def _authored_collection_physics_bindings(prim: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(relationship.GetName())
            for relationship in prim.GetRelationships()
            if str(relationship.GetName()).startswith(
                _PHYSICS_COLLECTION_BINDING_PREFIX
            )
            and relationship.GetPropertyStack()
        )
    )


def _collision_api_paths(
    stage: Any,
    *,
    Usd: Any,
    UsdPhysics: Any,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(prim.GetPath())
            for prim in _stage_prims_including_instance_proxies(stage, Usd=Usd)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        )
    )


def _stage_prims_including_instance_proxies(stage: Any, *, Usd: Any) -> Any:
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    return Usd.PrimRange.Stage(stage, predicate)


def _verify_derivative(
    asset_path: Path,
    *,
    plan: PhysicsProfilePlanV1,
    expected_state: _AuthoredProfileState,
    expected_dependency_hashes: tuple[tuple[str, str], ...],
    package_root: Path,
) -> None:
    _require_regular_nonsymlink_file(asset_path, label="derivative asset")
    if asset_path.suffix.lower() == ".usdz":
        _inspect_usdz(asset_path)
        _require_usdz_self_contained(asset_path)
        _require_usdz_dependency_bytes(asset_path, expected_dependency_hashes)
    else:
        closure = _local_dependency_hashes(asset_path, package_root)
        for relative, digest in expected_dependency_hashes:
            if closure.get(relative) != digest:
                raise PhysicsProfilePlanError(
                    f"Derivative changed or omitted dependency: {relative}"
                )
    _verify_profile(asset_path, plan, expected_state=expected_state)


def _local_dependency_hashes(asset_path: Path, package_root: Path) -> dict[str, str]:
    paths = _dependency_files(asset_path)
    result: dict[str, str] = {}
    for path in paths:
        relative = _relative_to(path, package_root)
        if relative is None:
            raise PhysicsProfilePlanError(
                f"Derivative dependency resolves outside its private package: {path}"
            )
        _require_regular_nonsymlink_file(path, label="private USD dependency")
        result[relative.as_posix()] = _file_sha256(path)
    return result


def _require_usdz_self_contained(asset_path: Path) -> None:
    for dependency in _dependency_files(asset_path):
        if dependency != asset_path:
            raise PhysicsProfilePlanError(
                f"USDZ derivative has an external dependency: {dependency}"
            )


def _require_usdz_dependency_bytes(
    asset_path: Path,
    expected: tuple[tuple[str, str], ...],
) -> None:
    expected_counts = Counter(digest for _relative, digest in expected)
    with zipfile.ZipFile(asset_path) as archive:
        root_name = _usdz_root_member(archive.infolist())
        actual_counts = Counter(
            _zip_member_sha256(archive, info)
            for info in archive.infolist()
            if not info.is_dir() and _normalized_member_name(info.filename) != root_name
        )
    missing = expected_counts - actual_counts
    extra = actual_counts - expected_counts
    if missing or extra:
        raise PhysicsProfilePlanError(
            "USDZ derivative dependency inventory is not an exact match."
        )


def _create_deterministic_usdz(
    root_asset: Path,
    output_path: Path,
    *,
    package_root: Path,
) -> None:
    mtime = _deterministic_zip_mtime()
    for path in sorted(package_root.rglob("*")):
        if path.is_file():
            os.utime(
                path,
                (mtime, mtime),
                follow_symlinks=False,
            )
    try:
        from pxr import UsdUtils
    except ImportError as exc:  # pragma: no cover - environment failure
        raise PhysicsProfilePlanError(
            f"OpenUSD Python APIs are unavailable: {exc}"
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        success = UsdUtils.CreateNewUsdzPackage(str(root_asset), str(output_path))
    except Exception as exc:  # pragma: no cover - OpenUSD failures vary
        raise PhysicsProfilePlanError(
            f"Could not create USDZ derivative: {exc}"
        ) from exc
    if not success or not output_path.is_file():
        raise PhysicsProfilePlanError("Could not create USDZ derivative.")
    _normalize_usdz_timestamps(output_path)
    _fsync_file(output_path)


def _deterministic_zip_mtime() -> float:
    try:
        return float(calendar.timegm(_DETERMINISTIC_ZIP_UTC_TIME))
    except (OSError, OverflowError, ValueError) as exc:
        raise PhysicsProfilePlanError(
            f"Could not represent deterministic ZIP UTC time: {exc}"
        ) from exc


def _normalize_usdz_timestamps(path: Path) -> None:
    timestamp = struct.pack(
        "<HH",
        _DETERMINISTIC_ZIP_DOS_TIME,
        _DETERMINISTIC_ZIP_DOS_DATE,
    )
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            central_offset = archive.start_dir
        with path.open("r+b") as stream:
            for info in infos:
                stream.seek(info.header_offset)
                if stream.read(4) != _ZIP_LOCAL_HEADER_SIGNATURE:
                    raise PhysicsProfilePlanError(
                        "USDZ local file header is invalid during normalization."
                    )
                stream.seek(info.header_offset + 10)
                stream.write(timestamp)

            cursor = central_offset
            for _info in infos:
                stream.seek(cursor)
                header = stream.read(_ZIP_CENTRAL_HEADER_SIZE)
                if (
                    len(header) != _ZIP_CENTRAL_HEADER_SIZE
                    or header[:4] != _ZIP_CENTRAL_HEADER_SIGNATURE
                ):
                    raise PhysicsProfilePlanError(
                        "USDZ central directory is invalid during normalization."
                    )
                name_length = int.from_bytes(header[28:30], "little")
                extra_length = int.from_bytes(header[30:32], "little")
                comment_length = int.from_bytes(header[32:34], "little")
                stream.seek(cursor + 12)
                stream.write(timestamp)
                cursor += (
                    _ZIP_CENTRAL_HEADER_SIZE
                    + name_length
                    + extra_length
                    + comment_length
                )
            stream.flush()
            os.fsync(stream.fileno())
    except PhysicsProfilePlanError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PhysicsProfilePlanError(
            f"Could not normalize deterministic USDZ timestamps: {exc}"
        ) from exc


def _extract_usdz(source: Path, destination: Path) -> str:
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            root_name = _usdz_root_member(infos)
            _validate_usdz_member_layout(infos)
            for info in infos:
                parts = _safe_member_parts(info.filename)
                assert parts is not None
                target = destination.joinpath(*parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.open(info) as source_stream,
                    target.open("xb") as target_stream,
                ):
                    shutil.copyfileobj(source_stream, target_stream, _HASH_CHUNK_SIZE)
                    target_stream.flush()
                    os.fsync(target_stream.fileno())
            return root_name
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PhysicsProfilePlanError(f"Could not extract USDZ source: {exc}") from exc


def _inspect_usdz(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            _usdz_root_member(infos)
            _validate_usdz_member_layout(infos)
            for info in infos:
                if info.is_dir():
                    continue
                with archive.open(info) as stream:
                    while stream.read(_HASH_CHUNK_SIZE):
                        pass
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PhysicsProfilePlanError(f"Invalid USDZ package: {exc}") from exc


def _usdz_root_member(infos: Sequence[zipfile.ZipInfo]) -> str:
    first_file = next((info for info in infos if not info.is_dir()), None)
    if first_file is None:
        raise PhysicsProfilePlanError("USDZ package contains no files.")
    normalized = _normalized_member_name(first_file.filename)
    if Path(normalized).suffix.lower() not in _USD_LAYER_SUFFIXES:
        raise PhysicsProfilePlanError("USDZ first file is not its root USD layer.")
    return normalized


def _validate_usdz_member_layout(infos: Sequence[zipfile.ZipInfo]) -> None:
    entries: dict[tuple[str, ...], zipfile.ZipInfo] = {}
    for info in infos:
        parts = _safe_member_parts(info.filename)
        if parts is None:
            raise PhysicsProfilePlanError(
                f"USDZ package contains an unsafe member path: {info.filename}"
            )
        if parts in entries:
            raise PhysicsProfilePlanError(
                "USDZ package contains duplicate normalized member paths."
            )
        if info.flag_bits & 0x1:
            raise PhysicsProfilePlanError("USDZ package contains an encrypted member.")
        if _zip_info_is_symlink(info):
            raise PhysicsProfilePlanError("USDZ package contains a symlink member.")
        if not info.is_dir() and info.compress_type != zipfile.ZIP_STORED:
            raise PhysicsProfilePlanError(
                "USDZ package member is compressed instead of stored."
            )
        entries[parts] = info
    files = {parts for parts, info in entries.items() if not info.is_dir()}
    for parts in entries:
        if any(parts[:depth] in files for depth in range(1, len(parts))):
            raise PhysicsProfilePlanError(
                "USDZ package contains a file/member ancestor collision."
            )


def _safe_member_parts(name: str) -> tuple[str, ...] | None:
    if "\\" in name:
        return None
    normalized = unquote(name)
    if normalized.startswith("/"):
        return None
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        return None
    return parts


def _normalized_member_name(name: str) -> str:
    parts = _safe_member_parts(name)
    if parts is None:
        raise PhysicsProfilePlanError(f"Unsafe USDZ member path: {name}")
    return "/".join(parts)


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _zip_member_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info) as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_root(output_root: Path, *, source: Path) -> None:
    if output_root == source:
        raise PhysicsProfilePlanError(
            "Output directory must differ from the source asset path."
        )
    _reject_symlink_components(output_root, allow_missing=True)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PhysicsProfilePlanError(
            f"Could not create output directory {output_root}: {exc}"
        ) from exc
    _reject_symlink_components(output_root, allow_missing=False)
    if not output_root.is_dir():
        raise PhysicsProfilePlanError(f"Output path is not a directory: {output_root}")


def _publish_directory(
    publication: Path,
    final_path: Path,
    *,
    expected_files: tuple[_FileRecord, ...],
) -> bool:
    if final_path.is_symlink():
        raise PhysicsProfilePlanError(
            f"Content-addressed publication path is a symlink: {final_path}"
        )
    if final_path.exists():
        _require_directory_exact(final_path, expected_files)
        return True
    renamed = False
    try:
        publication.rename(final_path)
        renamed = True
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not final_path.exists():
            raise PhysicsProfilePlanError(
                f"Could not atomically publish derivative: {exc}"
            ) from exc
        _require_directory_exact(final_path, expected_files)
        return True
    try:
        _fsync_directory(final_path.parent)
        _require_directory_exact(final_path, expected_files)
    except Exception:
        if renamed:
            _remove_directory_if_exact(final_path, expected_files)
        raise
    return False


def _require_directory_exact(
    path: Path,
    expected_files: tuple[_FileRecord, ...],
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PhysicsProfilePlanError(
            f"Content-addressed publication is not a regular directory: {path}"
        )
    actual = _directory_file_records(path, relative_to=path)
    expected = tuple(
        _FileRecord(
            path=path / record.relative_path,
            relative_path=record.relative_path,
            sha256=record.sha256,
        )
        for record in expected_files
    )
    if tuple((r.relative_path, r.sha256) for r in actual) != tuple(
        (r.relative_path, r.sha256) for r in expected
    ):
        raise PhysicsProfilePlanError(
            f"Conflicting content-addressed publication already exists: {path}"
        )


def _remove_directory_if_exact(
    path: Path,
    expected_files: tuple[_FileRecord, ...],
) -> None:
    try:
        _require_directory_exact(path, expected_files)
    except (OSError, PhysicsProfilePlanError):
        return
    shutil.rmtree(path, ignore_errors=True)


def _verify_published_receipt(
    path: Path,
    expected: PhysicsProfileReceiptV1,
) -> None:
    expected_bytes = _canonical_model_bytes(expected)
    if path.read_bytes() != expected_bytes:
        raise PhysicsProfilePlanError(
            "Published machine receipt failed exact readback."
        )
    try:
        actual = PhysicsProfileReceiptV1.model_validate_json(
            expected_bytes,
            strict=True,
        )
    except ValidationError as exc:  # pragma: no cover - internal invariant
        raise PhysicsProfilePlanError(
            f"Published machine receipt failed schema readback: {exc}"
        ) from exc
    if actual != expected:
        raise PhysicsProfilePlanError("Published machine receipt changed on readback.")


def _directory_file_records(
    root: Path,
    *,
    relative_to: Path,
) -> tuple[_FileRecord, ...]:
    records: list[_FileRecord] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise PhysicsProfilePlanError(f"Artifact tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PhysicsProfilePlanError(f"Artifact tree entry is not regular: {path}")
        records.append(
            _FileRecord(
                path=path,
                relative_path=path.relative_to(relative_to).as_posix(),
                sha256=_file_sha256(path),
            )
        )
    return tuple(records)


def _artifact_digest(
    records: Sequence[_FileRecord],
    *,
    plan_sha256: str,
    source_asset_sha256: str,
) -> str:
    payload = {
        "schema": _ARTIFACT_DIGEST_SCHEMA,
        "plan_sha256": plan_sha256,
        "source_asset_sha256": source_asset_sha256,
        "files": [
            {"relative_path": record.relative_path, "sha256": record.sha256}
            for record in records
        ],
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _file_record_digest(records: Sequence[_FileRecord]) -> str:
    payload = [
        {"relative_path": record.relative_path, "sha256": record.sha256}
        for record in records
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return _canonical_json_bytes(model.model_dump(mode="json", exclude_defaults=True))


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise PhysicsProfilePlanError(
            f"Could not write machine receipt: {exc}"
        ) from exc


def _copy_regular_file(source: Path, destination: Path) -> None:
    try:
        with (
            source.open("rb") as source_stream,
            destination.open("xb") as target_stream,
        ):
            shutil.copyfileobj(source_stream, target_stream, _HASH_CHUNK_SIZE)
            target_stream.flush()
            os.fsync(target_stream.fileno())
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise PhysicsProfilePlanError(
            f"Could not copy source dependency {source}: {exc}"
        ) from exc


def _require_snapshot_unchanged(expected: _SourceSnapshot) -> None:
    current = _capture_source_snapshot(expected.source_path)
    if current != expected:
        raise PhysicsProfilePlanError(
            "Source asset or dependency closure changed during authoring."
        )


def _require_regular_nonsymlink_file(path: Path, *, label: str) -> None:
    _reject_symlink_components(path, allow_missing=False)
    try:
        metadata = path.stat()
    except OSError as exc:
        raise PhysicsProfilePlanError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise PhysicsProfilePlanError(f"{label} is not a regular file: {path}")


def _reject_symlink_components(path: Path, *, allow_missing: bool) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    missing_seen = False
    for part in absolute.parts[1:]:
        current /= part
        if missing_seen:
            continue
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not allow_missing:
                raise PhysicsProfilePlanError(f"Path component is missing: {current}")
            missing_seen = True
            continue
        except OSError as exc:
            raise PhysicsProfilePlanError(
                f"Could not inspect path component {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PhysicsProfilePlanError(f"Path contains a symlink: {current}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as exc:
        raise PhysicsProfilePlanError(f"Could not hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicsProfilePlanError(f"Plan contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise PhysicsProfilePlanError(f"Plan contains non-finite JSON number: {value}")


def _parse_json_float(value: str) -> float:
    number = float(value)
    if number == 0.0 and not Decimal(value).is_zero():
        raise PhysicsProfilePlanError(
            "Plan contains a lexically nonzero JSON number that underflows during "
            "parsing."
        )
    return number


def _require_prim_path_text(value: str, *, field: str) -> None:
    if not isinstance(value, str) or value == "/" or not value.startswith("/"):
        raise ValueError(f"{field} must contain absolute prim paths")
    parts = value[1:].split("/")
    if any(not _PRIM_COMPONENT_RE.fullmatch(part) for part in parts):
        raise ValueError(f"{field} contains a noncanonical prim path: {value}")


def _path_is_prefix(parent: str, child: str) -> bool:
    return child == parent or child.startswith(parent + "/")


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _as_float32(value: float) -> float:
    return float(struct.unpack("!f", struct.pack("!f", value))[0])


def _float32_max() -> float:
    return float(struct.unpack("!f", bytes.fromhex("7f7fffff"))[0])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply an exact owner-approved physics validation plan."
    )
    parser.add_argument("source_asset", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact physics-profile authorer CLI."""

    args = _build_parser().parse_args(argv)
    try:
        result = author_physics_profile_plan(
            args.source_asset,
            args.plan,
            args.output_dir,
        )
    except Exception as exc:
        print(
            json.dumps({"error": str(exc), "status": "BLOCKED"}, sort_keys=True),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "output_asset_path": str(result.output_asset_path),
                "output_asset_sha256": result.receipt.output_asset_sha256,
                "output_artifact_sha256": result.receipt.output_artifact_sha256,
                "plan_sha256": result.receipt.plan_sha256,
                "receipt_path": str(result.receipt_path),
                "reused_output": result.reused_output,
                "source_asset_sha256": result.receipt.source_asset_sha256,
                "status": "PASS",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "PHYSICS_PROFILE_PLAN_SCHEMA_VERSION",
    "PHYSICS_PROFILE_RECEIPT_SCHEMA_VERSION",
    "PhysicsProfileAuthoredSdfTransitionV1",
    "PhysicsColliderApproximationPlanV1",
    "PhysicsMaterialAuthoredOpinionsV1",
    "PhysicsMaterialPlanV1",
    "PhysicsProfileApprovalV1",
    "PhysicsProfilePlanError",
    "PhysicsProfilePlanResult",
    "PhysicsProfilePlanV1",
    "PhysicsProfilePreservedApproximationV1",
    "PhysicsProfileReceiptV1",
    "PhysicsProfileSourceIdentity",
    "author_physics_profile_plan",
    "inspect_physics_profile_source",
    "load_physics_profile_plan",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
