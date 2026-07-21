# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-plan Gate 3A primvar and collision-mesh topology authoring.

The authorer accepts one owner-approved plan bound to exact source and machine-
evidence bytes. It edits a private source snapshot, proves readback invariants,
and atomically publishes a deterministic, self-contained USDZ plus receipt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, NoReturn, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from world_understanding.utils.usd.package import safe_usdz_member_parts

GATE3A_MESH_TOPOLOGY_PLAN_SCHEMA_VERSION = (
    "content-agent-workflows.simready-gate3a-mesh-topology-plan.v1"
)
GATE3A_MESH_TOPOLOGY_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.simready-gate3a-mesh-topology-receipt.v1"
)
GATE3A_MESH_TOPOLOGY_REQUIREMENT = "G3A.MESH.001"
GATE3A_MESH_TOPOLOGY_OUTPUT_DIR = "gate3a-mesh-topology"

_USD_LAYER_SUFFIXES = {".usd", ".usda", ".usdc"}
_USD_SUFFIXES = _USD_LAYER_SUFFIXES | {".usdz"}
_FIXED_PACKAGE_MTIME = 315532800
_ZIP_LOCAL_HEADER_SIZE = 30
_ZIP_ALIGNMENT = 64
_MAX_PACKAGE_PATH_DEPTH = 256
_PART_PREFIX = "MeshPart_"
_NORMAL_WINDING_COSINE_EPSILON = 1.0e-6

_COLLISION_API_BASES = {
    "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxCollisionAPI",
    "PhysxSDFMeshCollisionAPI",
}
_FORBIDDEN_PHYSICS_API_BASES = {
    "PhysicsArticulationRootAPI",
    "PhysicsFilteredPairsAPI",
    "PhysicsMassAPI",
    "PhysicsRigidBodyAPI",
    "PhysxRigidBodyAPI",
}
_ALLOWED_PARENT_API_BASES = {"MaterialBindingAPI"}
_COLLISION_PROPERTY_NAMES = {"physics:approximation", "physics:collisionEnabled"}
_RETAINED_PROPERTY_NAMES = {"purpose", "visibility", "proxyPrim"}
_STATIC_MESH_PROPERTY_NAMES = {
    "doubleSided",
    "faceVaryingLinearInterpolation",
    "interpolateBoundary",
    "orientation",
    "subdivisionScheme",
    "triangleSubdivisionRule",
}
_UNSUPPORTED_MESH_PROPERTY_NAMES = {
    "cornerIndices",
    "cornerSharpnesses",
    "creaseIndices",
    "creaseLengths",
    "creaseSharpnesses",
    "holeIndices",
}
_CORE_MESH_PROPERTY_NAMES = {
    "extent",
    "faceVertexCounts",
    "faceVertexIndices",
    "normals",
    "points",
}
_POINT_DOMAIN_PROPERTY_NAMES = {"accelerations", "velocities"}


class _Blocked(ValueError):
    """Expected fail-closed plan, package, or OpenUSD rejection."""


_StrictNonEmptyString = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
]
_ReadableStrictString = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1),
]
_Sha256 = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
_PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
_NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]


class Gate3AMeshTopologyEvidence(BaseModel):
    """Immutable machine evidence bound to one exact source artifact."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: Literal["gate3a_validation", "machine_mesh_preflight"]
    artifact_path: _StrictNonEmptyString
    artifact_sha256: _Sha256
    subject_asset_sha256: _Sha256

    @field_validator("artifact_path")
    @classmethod
    def validate_absolute_artifact_path(cls, value: str) -> str:
        """Require a stable, explicit local evidence path."""

        if not Path(value).is_absolute():
            raise ValueError("artifact_path must be absolute")
        return value


class Gate3AMeshTopologyProvenance(BaseModel):
    """Owner approval and machine evidence for an exact topology plan."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    approved_by: _ReadableStrictString
    approval_reference: _ReadableStrictString
    evidence: Annotated[list[Gate3AMeshTopologyEvidence], Field(min_length=1)]


class IndexedPrimvarCompaction(BaseModel):
    """One exact indexed primvar table compaction."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    prim_path: _StrictNonEmptyString
    attribute_name: _StrictNonEmptyString
    expected_value_count: _PositiveStrictInt
    expected_index_count: _PositiveStrictInt
    expected_referenced_value_count: _PositiveStrictInt

    @field_validator("prim_path")
    @classmethod
    def validate_prim_path(cls, value: str) -> str:
        """Reject relative, property, and root paths before opening a stage."""

        if not value.startswith("/") or value == "/" or "." in value.rsplit("/", 1)[-1]:
            raise ValueError("prim_path must be an absolute prim path")
        return value

    @field_validator("attribute_name")
    @classmethod
    def validate_attribute_name(cls, value: str) -> str:
        """Limit this operation to authored primvars."""

        if not value.startswith("primvars:") or value.endswith(":indices"):
            raise ValueError("attribute_name must name a primvar value attribute")
        return value

    @model_validator(mode="after")
    def validate_referenced_count(self) -> IndexedPrimvarCompaction:
        """A compaction must remove at least one unreferenced table value."""

        if self.expected_referenced_value_count >= self.expected_value_count:
            raise ValueError(
                "expected_referenced_value_count must be smaller than "
                "expected_value_count"
            )
        return self


class CollisionMeshNormalization(BaseModel):
    """One exact collision-only Mesh normalization and partition."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    prim_path: _StrictNonEmptyString
    expected_point_count: _PositiveStrictInt
    expected_exact_unique_point_count: _PositiveStrictInt
    expected_unused_point_count: _NonNegativeStrictInt
    expected_face_count: _PositiveStrictInt
    expected_face_vertex_index_count: _PositiveStrictInt
    expected_nonmanifold_edge_count: _NonNegativeStrictInt
    expected_output_part_count: _PositiveStrictInt

    @field_validator("prim_path")
    @classmethod
    def validate_prim_path(cls, value: str) -> str:
        """Reject relative, property, and root paths before opening a stage."""

        if not value.startswith("/") or value == "/" or "." in value.rsplit("/", 1)[-1]:
            raise ValueError("prim_path must be an absolute prim path")
        return value

    @model_validator(mode="after")
    def validate_point_counts(self) -> CollisionMeshNormalization:
        """Reject internally inconsistent owner expectations."""

        if self.expected_exact_unique_point_count > self.expected_point_count:
            raise ValueError(
                "expected_exact_unique_point_count cannot exceed expected_point_count"
            )
        if self.expected_unused_point_count >= self.expected_point_count:
            raise ValueError(
                "expected_unused_point_count must be smaller than expected_point_count"
            )
        return self


class Gate3AMeshTopologyPlan(BaseModel):
    """Strict source/evidence-bound owner plan for Gate 3A mesh authoring."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[
        "content-agent-workflows.simready-gate3a-mesh-topology-plan.v1"
    ]
    source_asset_path: _StrictNonEmptyString
    source_asset_sha256: _Sha256
    provenance: Gate3AMeshTopologyProvenance
    primvar_compactions: list[IndexedPrimvarCompaction]
    mesh_normalizations: list[CollisionMeshNormalization]

    @field_validator("source_asset_path")
    @classmethod
    def validate_absolute_source_path(cls, value: str) -> str:
        """Require plans to identify one exact local source path."""

        if not Path(value).is_absolute():
            raise ValueError("source_asset_path must be absolute")
        return value

    @model_validator(mode="after")
    def validate_operations(self) -> Gate3AMeshTopologyPlan:
        """Require unique, disjoint, non-empty operation lists and evidence."""

        primvar_keys = [
            (operation.prim_path, operation.attribute_name)
            for operation in self.primvar_compactions
        ]
        if len(primvar_keys) != len(set(primvar_keys)):
            raise ValueError("primvar_compactions contains a duplicate operation")
        mesh_paths = [operation.prim_path for operation in self.mesh_normalizations]
        if len(mesh_paths) != len(set(mesh_paths)):
            raise ValueError("mesh_normalizations contains a duplicate prim path")
        if not primvar_keys and not mesh_paths:
            raise ValueError("the plan must contain at least one operation")
        if {path for path, _name in primvar_keys}.intersection(mesh_paths):
            raise ValueError(
                "a prim cannot have both standalone primvar and mesh operations"
            )
        if any(
            evidence.subject_asset_sha256 != self.source_asset_sha256
            for evidence in self.provenance.evidence
        ):
            raise ValueError("all evidence must bind the plan source_asset_sha256")
        evidence_paths = [item.artifact_path for item in self.provenance.evidence]
        if len(evidence_paths) != len(set(evidence_paths)):
            raise ValueError("provenance.evidence contains a duplicate path")
        return self


@dataclass(frozen=True)
class Gate3AMeshTopologyResult:
    """Result of applying an exact Gate 3A mesh-topology plan."""

    status: str
    passed: bool
    reason: str
    output_path: Path
    receipt_path: Path | None
    report: dict[str, Any]


@dataclass(frozen=True)
class _FileIdentity:
    path: Path
    sha256: str
    stat_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _SourceCapture:
    package_root: Path
    root_layer_path: Path
    root_entry: str
    source_container: str
    source_manifest: dict[str, Any]


@dataclass(frozen=True)
class _PrimvarPlan:
    operation: IndexedPrimvarCompaction
    type_name: Any
    interpolation: Any
    element_size: int
    value_metadata: dict[str, Any]
    indices_metadata: dict[str, Any]
    compact_values: Any
    compact_indices: tuple[int, ...]
    flattened_values: Any


@dataclass(frozen=True)
class _DomainAttribute:
    name: str
    type_name: Any
    custom: bool
    variability: Any
    metadata: dict[str, Any]
    domain: Literal["constant", "uniform", "faceVarying", "vertex"]
    values: Any


@dataclass(frozen=True)
class _TopologyPrimvar:
    name: str
    type_name: Any
    interpolation: Any
    element_size: int
    value_metadata: dict[str, Any]
    indices_metadata: dict[str, Any] | None
    values: Any
    indices: tuple[int, ...] | None
    flattened_values: Any


@dataclass(frozen=True)
class _MeshPart:
    path: str
    source_face_indices: tuple[int, ...]
    canonical_point_ids: tuple[int, ...]
    points: Any
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]


@dataclass(frozen=True)
class _MeshPlan:
    operation: CollisionMeshNormalization
    parent_api_schemas: tuple[str, ...]
    collider_api_schemas: tuple[str, ...]
    retained_parent_properties: dict[str, dict[str, Any]]
    retained_parent_metadata: dict[str, Any]
    collision_properties: dict[str, dict[str, Any]]
    core_attribute_metadata: dict[str, dict[str, Any]]
    static_mesh_attributes: tuple[_DomainAttribute, ...]
    domain_attributes: tuple[_DomainAttribute, ...]
    authored_normals: _DomainAttribute | None
    preserved_normal_part_paths: tuple[str, ...]
    omitted_normal_part_paths: tuple[str, ...]
    primvars: tuple[_TopologyPrimvar, ...]
    removed_property_names: tuple[str, ...]
    source_points: Any
    source_faces: tuple[tuple[int, ...], ...]
    canonical_members: dict[int, tuple[int, ...]]
    canonical_representatives: dict[int, int]
    face_corner_offsets: tuple[int, ...]
    parts: tuple[_MeshPart, ...]
    source_world_faces: Counter[tuple[tuple[float, float, float], ...]]
    source_world_triangles: Counter[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ]
    material_bindings: dict[str, str | None]
    effective_visibility: str
    effective_purpose: str


@dataclass(frozen=True)
class _StageProof:
    default_prim_path: str
    root_metadata: dict[str, Any]
    prim_inventory: tuple[tuple[str, str], ...]
    physics_inventory: dict[str, Any]
    primvar_plans: tuple[_PrimvarPlan, ...]
    mesh_plans: tuple[_MeshPlan, ...]


def author_gate3a_mesh_topology_derivative(
    *,
    asset_path: Path,
    plan_path: Path,
    output_dir: Path,
) -> Gate3AMeshTopologyResult:
    """Apply one exact plan and atomically publish a deterministic USDZ bundle."""

    report: dict[str, Any] = {
        "schema_version": GATE3A_MESH_TOPOLOGY_RECEIPT_SCHEMA_VERSION,
        "requirement": GATE3A_MESH_TOPOLOGY_REQUIREMENT,
        "asset_path": str(asset_path),
        "plan_path": str(plan_path),
        "changes": [],
    }
    workspace: Path | None = None
    stage: Any | None = None
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

        source_identity = _capture_file_identity(
            _regular_file(asset_path, label="source asset", suffixes=_USD_SUFFIXES)
        )
        plan_identity = _capture_file_identity(
            _regular_file(plan_path, label="mesh-topology plan", suffixes={".json"})
        )
        plan = _load_plan(plan_identity.path)
        if str(source_identity.path) != plan.source_asset_path:
            _fail(
                "stale_source_path",
                "plan source_asset_path does not match the requested source: "
                f"expected {source_identity.path}, received {plan.source_asset_path}",
            )
        if source_identity.sha256 != plan.source_asset_sha256:
            _fail(
                "stale_source_sha256",
                "plan source_asset_sha256 is stale: expected "
                f"{source_identity.sha256}, received {plan.source_asset_sha256}",
            )
        evidence_identities = _validate_evidence(
            plan=plan,
            source_identity=source_identity,
        )
        output_root = _prepare_output_root(output_dir)
        workspace = Path(
            tempfile.mkdtemp(prefix=".gate3a-mesh-topology-", dir=output_root)
        )
        os.chmod(workspace, stat.S_IRWXU)
        capture = _capture_source(
            source=source_identity.path,
            source_sha256=source_identity.sha256,
            workspace=workspace,
        )
        stage = Usd.Stage.Open(str(capture.root_layer_path), load=Usd.Stage.LoadAll)
        if stage is None:
            _fail("invalid_source_stage", f"could not open {source_identity.path}")
        _validate_dependency_closure(
            root_layer_path=capture.root_layer_path,
            package_root=capture.package_root,
            expected_files=set(capture.source_manifest["entry_paths"]),
            UsdUtils=UsdUtils,
        )
        proof = _preflight(
            stage=stage,
            plan=plan,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        changes = _apply_plan(
            stage=stage,
            proof=proof,
            Sdf=Sdf,
            UsdGeom=UsdGeom,
            Vt=Vt,
        )
        _validate_authored_stage(
            stage=stage,
            proof=proof,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        if not stage.GetRootLayer().Save():
            _fail("output_write_failed", "OpenUSD did not save the authored root")
        stage = None
        gc.collect()

        bundle_dir = workspace / "bundle"
        bundle_dir.mkdir(mode=stat.S_IRWXU)
        output_name = f"{source_identity.path.stem}.gate3a-mesh-topology.usdz"
        staged_output = bundle_dir / output_name
        _write_usdz(
            root_layer_path=capture.root_layer_path,
            package_root=capture.package_root,
            output_path=staged_output,
            UsdUtils=UsdUtils,
        )
        output_manifest = _asset_manifest(staged_output)
        _validate_output_manifest(
            source_manifest=capture.source_manifest,
            output_manifest=output_manifest,
        )
        output_sha256 = _file_sha256(staged_output)

        readback = Usd.Stage.Open(str(staged_output), load=Usd.Stage.LoadAll)
        if readback is None:
            _fail("output_readback_failed", f"could not reopen {staged_output}")
        _validate_authored_stage(
            stage=readback,
            proof=proof,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        readback = None
        gc.collect()

        _verify_all_inputs_unchanged(
            source=source_identity,
            plan=plan_identity,
            evidence=evidence_identities,
        )
        receipt = _receipt_payload(
            plan=plan,
            plan_sha256=plan_identity.sha256,
            source=source_identity,
            source_capture=capture,
            output_sha256=output_sha256,
            output_manifest=output_manifest,
            proof=proof,
            changes=changes,
        )
        staged_receipt = bundle_dir / "receipt.json"
        receipt_bytes = _canonical_json_bytes(receipt)
        _write_exclusive(staged_receipt, receipt_bytes)
        bundle_sha256 = _canonical_json_sha256(
            {
                "output_asset_sha256": output_sha256,
                "plan_sha256": plan_identity.sha256,
                "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "source_asset_sha256": source_identity.sha256,
            }
        )

        def verify_before_publish() -> None:
            _verify_all_inputs_unchanged(
                source=source_identity,
                plan=plan_identity,
                evidence=evidence_identities,
            )
            if _file_sha256(staged_output) != output_sha256:
                _fail("staged_output_changed", "staged output bytes changed")
            if staged_receipt.read_bytes() != receipt_bytes:
                _fail("staged_receipt_changed", "staged receipt bytes changed")

        verify_before_publish()
        final_bundle, reused = _publish_bundle(
            bundle_dir=bundle_dir,
            output_root=output_root,
            bundle_sha256=bundle_sha256,
            output_name=output_name,
            expected_output_sha256=output_sha256,
            expected_receipt=receipt_bytes,
            precommit_validator=verify_before_publish,
        )
        final_output = final_bundle / output_name
        final_receipt = final_bundle / "receipt.json"
        runtime_report = {
            **receipt,
            "asset_path": str(source_identity.path),
            "plan_path": str(plan_identity.path),
            "output_path": str(final_output),
            "receipt_path": str(final_receipt),
            "bundle_sha256": bundle_sha256,
            "reused_output": reused,
            "status": "AUTHORED",
            "passed": True,
            "reason": "Published an exact evidence-backed Gate 3A mesh derivative.",
        }
        return Gate3AMeshTopologyResult(
            status="AUTHORED",
            passed=True,
            reason=runtime_report["reason"],
            output_path=final_output,
            receipt_path=final_receipt,
            report=runtime_report,
        )
    except (
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        ValidationError,
        _Blocked,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        report.update(
            {
                "status": "BLOCKED",
                "passed": False,
                "reason": str(exc),
                "failure": str(exc),
                "changes": [],
            }
        )
        return Gate3AMeshTopologyResult(
            status="BLOCKED",
            passed=False,
            reason=str(exc),
            output_path=Path(asset_path),
            receipt_path=None,
            report=report,
        )
    finally:
        stage = None
        gc.collect()
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)


def _load_plan(path: Path) -> Gate3AMeshTopologyPlan:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_number,
    )
    return Gate3AMeshTopologyPlan.model_validate(payload)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            _fail("invalid_plan", f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json_number(value: str) -> None:
    _fail("invalid_plan", f"non-finite JSON number: {value}")


def _regular_file(path: Path, *, label: str, suffixes: set[str]) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        _fail("missing_input", f"{label} does not exist: {absolute}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("unsafe_input", f"{label} must be a regular non-symlink: {absolute}")
    if absolute.suffix.lower() not in suffixes:
        _fail("unsupported_input", f"unsupported {label} suffix: {absolute.suffix}")
    return absolute


def _capture_file_identity(path: Path) -> _FileIdentity:
    metadata = path.stat(follow_symlinks=False)
    return _FileIdentity(
        path=path,
        sha256=_file_sha256(path),
        stat_identity=_stat_identity(metadata),
    )


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_evidence(
    *,
    plan: Gate3AMeshTopologyPlan,
    source_identity: _FileIdentity,
) -> tuple[_FileIdentity, ...]:
    identities: list[_FileIdentity] = []
    for item in plan.provenance.evidence:
        path = _regular_file(
            Path(item.artifact_path),
            label="machine evidence",
            suffixes={Path(item.artifact_path).suffix.lower()},
        )
        identity = _capture_file_identity(path)
        if identity.sha256 != item.artifact_sha256:
            _fail(
                "stale_evidence_sha256",
                f"evidence SHA-256 is stale for {path}: expected "
                f"{identity.sha256}, received {item.artifact_sha256}",
            )
        if item.subject_asset_sha256 != source_identity.sha256:
            _fail("stale_evidence_subject", f"evidence subject is stale for {path}")
        if item.kind == "gate3a_validation":
            _validate_gate3a_evidence_subject(
                path=path,
                subject_sha256=source_identity.sha256,
            )
        identities.append(identity)
    return tuple(identities)


def _validate_gate3a_evidence_subject(*, path: Path, subject_sha256: str) -> None:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_number,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        _fail("invalid_gate3a_evidence", f"evidence has no results list: {path}")
    matches = [
        item
        for item in payload["results"]
        if isinstance(item, dict) and item.get("artifact_sha256") == subject_sha256
    ]
    if len(matches) != 1:
        _fail(
            "invalid_gate3a_evidence",
            f"evidence must contain exactly one result for {subject_sha256}: {path}",
        )


def _prepare_output_root(output_dir: Path) -> Path:
    output_dir = Path(os.path.abspath(output_dir.expanduser()))
    output_dir.mkdir(parents=True, exist_ok=True, mode=stat.S_IRWXU)
    if output_dir.is_symlink() or not output_dir.is_dir():
        _fail("unsafe_output", f"output directory is not regular: {output_dir}")
    output_root = output_dir / GATE3A_MESH_TOPOLOGY_OUTPUT_DIR
    output_root.mkdir(mode=stat.S_IRWXU, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        _fail("unsafe_output", f"publish root is not regular: {output_root}")
    return output_root


def _capture_source(
    *,
    source: Path,
    source_sha256: str,
    workspace: Path,
) -> _SourceCapture:
    package_root = workspace / "source-package"
    package_root.mkdir(mode=stat.S_IRWXU)
    if source.suffix.lower() == ".usdz":
        captured = workspace / "source.usdz"
        shutil.copyfile(source, captured)
        if _file_sha256(captured) != source_sha256:
            _fail("source_capture_mismatch", "captured USDZ bytes differ from source")
        manifest = _extract_usdz_without_size_limit(captured, package_root)
        root_entry = manifest["root_entry"]
        return _SourceCapture(
            package_root=package_root,
            root_layer_path=package_root.joinpath(*PurePosixPath(root_entry).parts),
            root_entry=root_entry,
            source_container="usdz",
            source_manifest=manifest,
        )

    captured_root = package_root / source.name
    shutil.copyfile(source, captured_root)
    if _file_sha256(captured_root) != source_sha256:
        _fail("source_capture_mismatch", "captured USD bytes differ from source")
    entry = _manifest_entry(captured_root, captured_root.name)
    return _SourceCapture(
        package_root=package_root,
        root_layer_path=captured_root,
        root_entry=captured_root.name,
        source_container="raw_usd",
        source_manifest={
            "container": "raw_usd",
            "root_entry": captured_root.name,
            "entry_paths": [captured_root.name],
            "entries": [entry],
            "dependency_bundle_sha256": _canonical_json_sha256([entry]),
        },
    )


def _extract_usdz_without_size_limit(
    package_path: Path,
    destination: Path,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    normalized_names: set[str] = set()
    file_names: set[str] = set()
    with zipfile.ZipFile(package_path) as archive:
        infos = archive.infolist()
        if not infos:
            _fail("invalid_usdz", "USDZ package is empty")
        root_parts = _validate_usdz_info(infos[0], require_file=True)
        root_entry = "/".join(root_parts)
        if Path(root_entry).suffix.lower() not in _USD_LAYER_SUFFIXES:
            _fail("invalid_usdz", "USDZ first entry must be a USD root layer")
        for info in infos:
            parts = _validate_usdz_info(info, require_file=False)
            normalized = "/".join(parts)
            if normalized in normalized_names:
                _fail("invalid_usdz", f"duplicate normalized entry: {normalized}")
            normalized_names.add(normalized)
            for depth in range(1, len(parts)):
                ancestor = "/".join(parts[:depth])
                if ancestor in file_names:
                    _fail("invalid_usdz", f"file/member collision: {ancestor}")
            target = destination.joinpath(*parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True, mode=stat.S_IRWXU)
                continue
            file_names.add(normalized)
            target.parent.mkdir(parents=True, exist_ok=True, mode=stat.S_IRWXU)
            digest = hashlib.sha256()
            with (
                archive.open(info) as source_stream,
                target.open("xb") as output_stream,
            ):
                for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                    output_stream.write(chunk)
                    digest.update(chunk)
            entries.append(
                {
                    "path": normalized,
                    "size": target.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    ordered = sorted(entries, key=lambda item: item["path"])
    return {
        "container": "usdz",
        "root_entry": root_entry,
        "entry_paths": [entry["path"] for entry in ordered],
        "entries": ordered,
        "dependency_bundle_sha256": _canonical_json_sha256(ordered),
    }


def _validate_usdz_info(
    info: zipfile.ZipInfo,
    *,
    require_file: bool,
) -> tuple[str, ...]:
    parts = safe_usdz_member_parts(info.filename)
    if (
        parts is None
        or not parts
        or len(parts) > _MAX_PACKAGE_PATH_DEPTH
        or "\\" in info.filename
        or PurePosixPath(info.filename).is_absolute()
    ):
        _fail("invalid_usdz", f"unsafe USDZ entry path: {info.filename!r}")
    if require_file and info.is_dir():
        _fail("invalid_usdz", "USDZ first entry cannot be a directory")
    if info.flag_bits & 0x1:
        _fail("invalid_usdz", f"encrypted USDZ entry: {info.filename}")
    mode = (info.external_attr >> 16) & 0o170000
    if mode == stat.S_IFLNK:
        _fail("invalid_usdz", f"symbolic-link USDZ entry: {info.filename}")
    if not info.is_dir():
        if info.compress_type != zipfile.ZIP_STORED:
            _fail("invalid_usdz", f"compressed USDZ entry: {info.filename}")
        data_offset = (
            info.header_offset
            + _ZIP_LOCAL_HEADER_SIZE
            + len(info.filename.encode("utf-8"))
            + len(info.extra)
        )
        if data_offset % _ZIP_ALIGNMENT:
            _fail("invalid_usdz", f"unaligned USDZ entry: {info.filename}")
    return parts


def _validate_dependency_closure(
    *,
    root_layer_path: Path,
    package_root: Path,
    expected_files: set[str],
    UsdUtils: Any,
) -> None:
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            str(root_layer_path)
        )
    except Exception as exc:
        _fail(
            "dependency_inventory_failed",
            f"{type(exc).__name__}: {exc}",
        )
    if unresolved:
        _fail(
            "unresolved_dependency",
            f"unresolved dependencies: {sorted(map(str, unresolved))}",
        )
    package_root_resolved = package_root.resolve(strict=True)
    resolved_files: set[str] = set()
    for item in (*layers, *assets):
        candidate = _resolved_dependency_path(item, base=root_layer_path.parent)
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(package_root_resolved)
        except (FileNotFoundError, ValueError):
            _fail(
                "outside_dependency",
                f"dependency is not a regular in-package file: {candidate}",
            )
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail("outside_dependency", f"dependency is not regular: {resolved}")
        resolved_files.add(relative.as_posix())
    if resolved_files != expected_files:
        _fail(
            "unsupported_package_contents",
            "package files must exactly equal the resolved dependency closure: "
            f"resolved={sorted(resolved_files)}, package={sorted(expected_files)}",
        )


def _resolved_dependency_path(item: Any, *, base: Path) -> Path:
    candidate_text = (
        str(getattr(item, "realPath", ""))
        or str(getattr(item, "resolvedPath", ""))
        or str(getattr(item, "path", ""))
        or str(getattr(item, "identifier", item))
    )
    candidate = Path(candidate_text)
    return candidate if candidate.is_absolute() else base / candidate


def _preflight(
    *,
    stage: Any,
    plan: Gate3AMeshTopologyPlan,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> _StageProof:
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        _fail("invalid_default_prim", "source stage must have a valid default prim")
    root_layer = stage.GetRootLayer()
    if root_layer.HasRelocates():
        _fail("unsupported_composition", "root-layer relocates are not supported")
    primvar_plans = tuple(
        _preflight_primvar(
            stage=stage,
            operation=operation,
            Usd=Usd,
            UsdGeom=UsdGeom,
        )
        for operation in sorted(
            plan.primvar_compactions,
            key=lambda item: (item.prim_path, item.attribute_name),
        )
    )
    mesh_plans = tuple(
        _preflight_mesh(
            stage=stage,
            operation=operation,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )
        for operation in sorted(
            plan.mesh_normalizations,
            key=lambda item: item.prim_path,
        )
    )
    return _StageProof(
        default_prim_path=str(default_prim.GetPath()),
        root_metadata=_root_metadata(stage),
        prim_inventory=_prim_inventory(stage),
        physics_inventory=_physics_inventory(stage, UsdPhysics=UsdPhysics),
        primvar_plans=primvar_plans,
        mesh_plans=mesh_plans,
    )


def _preflight_primvar(
    *,
    stage: Any,
    operation: IndexedPrimvarCompaction,
    Usd: Any,
    UsdGeom: Any,
) -> _PrimvarPlan:
    prim = stage.GetPrimAtPath(operation.prim_path)
    if not prim or not prim.IsValid() or not prim.IsActive() or not prim.IsDefined():
        _fail(
            "missing_prim",
            f"primvar target is not active and defined: {operation.prim_path}",
        )
    _reject_candidate_composition(prim)
    primvar = UsdGeom.Primvar(prim.GetAttribute(operation.attribute_name))
    if not primvar or not primvar.IsDefined():
        _fail(
            "missing_primvar",
            f"planned primvar does not exist: {operation.prim_path}.{operation.attribute_name}",
        )
    if primvar.GetElementSize() != 1:
        _fail(
            "unsupported_primvar",
            f"primvar elementSize must be one: {primvar.GetAttr().GetPath()}",
        )
    type_name = primvar.GetTypeName()
    if not type_name.isArray or not primvar.IsIndexed():
        _fail(
            "unsupported_primvar",
            f"planned primvar must be an indexed array: {primvar.GetAttr().GetPath()}",
        )
    value_attr = primvar.GetAttr()
    indices_attr = primvar.GetIndicesAttr()
    _reject_dynamic_attribute(value_attr, label="primvar value")
    if not indices_attr:
        _fail("missing_primvar_indices", f"indices are missing: {value_attr.GetPath()}")
    _reject_dynamic_attribute(indices_attr, label="primvar indices")
    values = value_attr.Get(Usd.TimeCode.Default())
    raw_indices = primvar.GetIndices(Usd.TimeCode.Default())
    if values is None or raw_indices is None:
        _fail("unreadable_primvar", f"could not read {value_attr.GetPath()}")
    indices = _validated_indices(
        raw_indices,
        upper_bound=len(values),
        label=f"primvar indices at {value_attr.GetPath()}",
    )
    used = set(indices)
    actual = (len(values), len(indices), len(used))
    expected = (
        operation.expected_value_count,
        operation.expected_index_count,
        operation.expected_referenced_value_count,
    )
    if actual != expected:
        _fail(
            "stale_primvar_plan",
            f"planned counts changed at {value_attr.GetPath()}: expected {expected}, received {actual}",
        )
    if len(used) == len(values):
        _fail(
            "no_primvar_change", f"primvar has no unused values: {value_attr.GetPath()}"
        )
    old_to_new: dict[int, int] = {}
    compact_values_list: list[Any] = []
    for old_index, value in enumerate(values):
        if old_index not in used:
            continue
        old_to_new[old_index] = len(compact_values_list)
        compact_values_list.append(value)
    compact_values = type(values)(compact_values_list)
    compact_indices = tuple(old_to_new[index] for index in indices)
    flattened = primvar.ComputeFlattened(Usd.TimeCode.Default())
    projected = type(flattened)([compact_values[index] for index in compact_indices])
    if not _exact_array_equal(flattened, projected):
        _fail(
            "primvar_invariant_failed",
            f"compaction does not preserve flattened values: {value_attr.GetPath()}",
        )
    return _PrimvarPlan(
        operation=operation,
        type_name=type_name,
        interpolation=primvar.GetInterpolation(),
        element_size=primvar.GetElementSize(),
        value_metadata=_attribute_metadata(value_attr),
        indices_metadata=_attribute_metadata(indices_attr),
        compact_values=compact_values,
        compact_indices=compact_indices,
        flattened_values=flattened,
    )


def _preflight_mesh(
    *,
    stage: Any,
    operation: CollisionMeshNormalization,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> _MeshPlan:
    prim = stage.GetPrimAtPath(operation.prim_path)
    if not prim or not prim.IsValid() or not prim.IsActive() or not prim.IsDefined():
        _fail(
            "missing_prim",
            f"mesh target is not active and defined: {operation.prim_path}",
        )
    if not prim.IsA(UsdGeom.Mesh):
        _fail("not_a_mesh", f"planned target is not a Mesh: {operation.prim_path}")
    if prim.GetChildren():
        _fail(
            "unsupported_children",
            f"planned Mesh already has children: {operation.prim_path}",
        )
    _reject_candidate_composition(prim)
    _reject_dynamic_candidate(prim, UsdGeom=UsdGeom)
    _reject_physics_ownership(
        stage=stage,
        prim=prim,
        UsdPhysics=UsdPhysics,
    )

    source_schemas = tuple(str(token) for token in prim.GetAppliedSchemas())
    parent_schemas, collider_schemas = _classify_api_schemas(
        operation.prim_path,
        source_schemas,
    )
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    counts_value = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    indices_value = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if points is None or counts_value is None or indices_value is None:
        _fail("invalid_mesh", f"mesh topology is unreadable: {operation.prim_path}")
    _reject_nonfinite_points(points, path=operation.prim_path)
    counts = tuple(int(value) for value in counts_value)
    indices = _validated_indices(
        indices_value,
        upper_bound=len(points),
        label=f"faceVertexIndices at {operation.prim_path}",
    )
    if not counts or any(count < 3 for count in counts):
        _fail(
            "degenerate_mesh",
            f"mesh contains a face with fewer than three vertices: {operation.prim_path}",
        )
    if sum(counts) != len(indices):
        _fail(
            "invalid_mesh",
            f"face counts do not match index count: {operation.prim_path}",
        )
    faces: list[tuple[int, ...]] = []
    offsets = [0]
    offset = 0
    for count in counts:
        face = tuple(indices[offset : offset + count])
        faces.append(face)
        offset += count
        offsets.append(offset)

    canonical_ids, canonical_members, canonical_representatives = _canonical_points(
        points
    )
    canonical_faces = tuple(
        tuple(canonical_ids[index] for index in face) for face in faces
    )
    for face_index, face in enumerate(canonical_faces):
        if len(set(face)) != len(face):
            _fail(
                "degenerate_mesh",
                f"exact welding degenerates face {face_index} at {operation.prim_path}",
            )
    edge_users = _edge_users(canonical_faces)
    nonmanifold_edges = sum(1 for users in edge_users.values() if len(users) > 2)
    referenced_points = set(indices)
    actual_counts = {
        "point_count": len(points),
        "exact_unique_point_count": len(canonical_members),
        "unused_point_count": len(set(range(len(points))) - referenced_points),
        "face_count": len(faces),
        "face_vertex_index_count": len(indices),
        "nonmanifold_edge_count": nonmanifold_edges,
    }
    expected_counts = {
        "point_count": operation.expected_point_count,
        "exact_unique_point_count": operation.expected_exact_unique_point_count,
        "unused_point_count": operation.expected_unused_point_count,
        "face_count": operation.expected_face_count,
        "face_vertex_index_count": operation.expected_face_vertex_index_count,
        "nonmanifold_edge_count": operation.expected_nonmanifold_edge_count,
    }
    if actual_counts != expected_counts:
        _fail(
            "stale_mesh_plan",
            f"planned mesh counts changed at {operation.prim_path}: expected "
            f"{expected_counts}, received {actual_counts}",
        )
    components = _face_components(
        face_count=len(canonical_faces),
        edge_users=edge_users,
    )
    if len(components) != operation.expected_output_part_count:
        _fail(
            "stale_mesh_plan",
            f"planned output part count changed at {operation.prim_path}: expected "
            f"{operation.expected_output_part_count}, received {len(components)}",
        )
    parts = _build_mesh_parts(
        prim_path=operation.prim_path,
        points=points,
        canonical_faces=canonical_faces,
        canonical_representatives=canonical_representatives,
        components=components,
    )
    for part in parts:
        _validate_part_topology(part, label=part.path)

    property_names = {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}
    primvars = _topology_primvars(
        prim=prim,
        point_count=len(points),
        face_count=len(faces),
        corner_count=len(indices),
        canonical_members=canonical_members,
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    primvar_property_names = {item.name for item in primvars} | {
        f"{item.name}:indices" for item in primvars if item.indices is not None
    }
    collision_properties: dict[str, dict[str, Any]] = {}
    static_attributes: list[_DomainAttribute] = []
    domain_attributes: list[_DomainAttribute] = []
    authored_normals: _DomainAttribute | None = None
    normals_attr = mesh.GetNormalsAttr()
    if normals_attr and normals_attr.HasAuthoredValueOpinion():
        normals_domain = _primvar_domain(str(mesh.GetNormalsInterpolation()))
        expected_normal_count = {
            "constant": 1,
            "uniform": len(faces),
            "faceVarying": len(indices),
            "vertex": len(points),
        }[normals_domain]
        authored_normals = _validated_domain_attribute(
            attr=normals_attr,
            domain=normals_domain,
            expected_count=expected_normal_count,
            label=f"{operation.prim_path}.normals",
        )
    removed_property_names = set(_CORE_MESH_PROPERTY_NAMES) | primvar_property_names
    retained_property_names: set[str] = set()

    for name in sorted(property_names):
        prop = prim.GetProperty(name)
        if name in primvar_property_names or name in _CORE_MESH_PROPERTY_NAMES:
            continue
        if name.startswith("xformOp:") or name == "xformOpOrder":
            retained_property_names.add(name)
            continue
        if name in _RETAINED_PROPERTY_NAMES or name.startswith("material:binding"):
            retained_property_names.add(name)
            continue
        if name in _COLLISION_PROPERTY_NAMES or name.startswith("physxCollision:"):
            collision_properties[name] = _copyable_property_snapshot(prop)
            removed_property_names.add(name)
            continue
        if name.startswith("physics:") or name.startswith("physx"):
            _fail(
                "unsupported_physics_property",
                f"cannot migrate non-collision physics property at {operation.prim_path}: {name}",
            )
        if name in _STATIC_MESH_PROPERTY_NAMES:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion():
                static_attributes.append(_domain_attribute(attr, domain="constant"))
            removed_property_names.add(name)
            continue
        if name in _UNSUPPORTED_MESH_PROPERTY_NAMES:
            attr = prim.GetAttribute(name)
            value = attr.Get() if attr else None
            if value:
                _fail(
                    "unsupported_mesh_data",
                    f"non-empty {name} is outside the partition contract at {operation.prim_path}",
                )
            removed_property_names.add(name)
            continue
        if name in _POINT_DOMAIN_PROPERTY_NAMES:
            attr = prim.GetAttribute(name)
            domain_attributes.append(
                _validated_domain_attribute(
                    attr=attr,
                    domain="vertex",
                    expected_count=len(points),
                    label=f"{operation.prim_path}.{name}",
                )
            )
            removed_property_names.add(name)
            continue
        if prop.IsCustom() and hasattr(prop, "GetTypeName"):
            attr = prim.GetAttribute(name)
            domain = _infer_custom_attribute_domain(
                attr=attr,
                point_count=len(points),
                face_count=len(faces),
                corner_count=len(indices),
            )
            if domain is None:
                retained_property_names.add(name)
            else:
                domain_attributes.append(_domain_attribute(attr, domain=domain))
                removed_property_names.add(name)
            continue
        _fail(
            "unsupported_mesh_property",
            f"cannot remap property exactly at {operation.prim_path}: {name}",
        )

    _validate_vertex_domain_equality(
        canonical_members=canonical_members,
        domain_attributes=[
            *domain_attributes,
            *([authored_normals] if authored_normals is not None else []),
        ],
        primvars=primvars,
        path=operation.prim_path,
    )
    retained_parent_properties = {
        name: _property_snapshot(prim.GetProperty(name))
        for name in sorted(retained_property_names)
    }
    retained_parent_metadata = _prim_metadata_without_type_and_apis(prim)
    core_attribute_metadata = {
        name: _attribute_metadata(prim.GetAttribute(name))
        for name in sorted(_CORE_MESH_PROPERTY_NAMES)
        if name != "normals"
        and prim.GetAttribute(name)
        and prim.GetAttribute(name).HasAuthoredValueOpinion()
    }
    orientation = str(mesh.GetOrientationAttr().Get())
    preserved_normal_part_paths, omitted_normal_part_paths = _normal_part_policy(
        authored_normals=authored_normals,
        parts=parts,
        face_corner_offsets=tuple(offsets),
        canonical_representatives=canonical_representatives,
        orientation=orientation,
    )
    material_bindings = _material_bindings(prim, UsdShade=UsdShade)
    imageable = UsdGeom.Imageable(prim)
    source_world_faces, source_world_triangles = _world_geometry(
        prim=prim,
        points=points,
        faces=tuple(faces),
        Usd=Usd,
        UsdGeom=UsdGeom,
    )
    return _MeshPlan(
        operation=operation,
        parent_api_schemas=parent_schemas,
        collider_api_schemas=collider_schemas,
        retained_parent_properties=retained_parent_properties,
        retained_parent_metadata=retained_parent_metadata,
        collision_properties=collision_properties,
        core_attribute_metadata=core_attribute_metadata,
        static_mesh_attributes=tuple(static_attributes),
        domain_attributes=tuple(domain_attributes),
        authored_normals=authored_normals,
        preserved_normal_part_paths=preserved_normal_part_paths,
        omitted_normal_part_paths=omitted_normal_part_paths,
        primvars=primvars,
        removed_property_names=tuple(sorted(removed_property_names)),
        source_points=points,
        source_faces=tuple(faces),
        canonical_members=canonical_members,
        canonical_representatives=canonical_representatives,
        face_corner_offsets=tuple(offsets),
        parts=parts,
        source_world_faces=source_world_faces,
        source_world_triangles=source_world_triangles,
        material_bindings=material_bindings,
        effective_visibility=str(imageable.ComputeEffectiveVisibility()),
        effective_purpose=str(imageable.ComputePurpose()),
    )


def _reject_candidate_composition(prim: Any) -> None:
    root_layer = prim.GetStage().GetRootLayer()
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        if (
            current.IsInstance()
            or current.IsInstanceProxy()
            or current.IsInstanceable()
        ):
            _fail(
                "unsupported_instance", f"instance composition at {current.GetPath()}"
            )
        if current.HasVariantSets() or current.GetVariantSets().GetNames():
            _fail("unsupported_variants", f"variant composition at {current.GetPath()}")
        if (
            current.HasAuthoredReferences()
            or current.HasAuthoredPayloads()
            or current.HasAuthoredInherits()
            or current.HasAuthoredSpecializes()
        ):
            _fail("unsupported_composition", f"composition arc at {current.GetPath()}")
        if any(
            str(key).startswith("clips") for key in current.GetAllAuthoredMetadata()
        ):
            _fail("unsupported_composition", f"value clips at {current.GetPath()}")
        if any(spec.layer != root_layer for spec in current.GetPrimStack()):
            _fail("unsupported_composition", f"multi-layer prim at {current.GetPath()}")
        current = current.GetParent()


def _reject_dynamic_candidate(prim: Any, *, UsdGeom: Any) -> None:
    for prop in prim.GetAuthoredProperties():
        if hasattr(prop, "GetTimeSamples"):
            _reject_dynamic_attribute(prop, label="mesh attribute")
    current = prim
    while current and current.IsValid() and not current.IsPseudoRoot():
        xformable = UsdGeom.Xformable(current)
        if xformable:
            attributes = [
                xformable.GetXformOpOrderAttr(),
                *(operation.GetAttr() for operation in xformable.GetOrderedXformOps()),
            ]
            for attr in attributes:
                _reject_dynamic_attribute(attr, label="ancestor transform")
        imageable = UsdGeom.Imageable(current)
        for attr in (imageable.GetVisibilityAttr(), imageable.GetPurposeAttr()):
            if attr:
                _reject_dynamic_attribute(attr, label="ancestor imageable attribute")
        current = current.GetParent()


def _reject_dynamic_attribute(attr: Any, *, label: str) -> None:
    if attr.GetTimeSamples() or attr.ValueMightBeTimeVarying():
        _fail("time_varying_input", f"{label} is time-varying: {attr.GetPath()}")
    if attr.HasAuthoredConnections():
        _fail("connected_input", f"{label} has connections: {attr.GetPath()}")


def _reject_physics_ownership(*, stage: Any, prim: Any, UsdPhysics: Any) -> None:
    path = str(prim.GetPath())
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        _fail("collision_api_required", f"Mesh is not collision-owned: {path}")
    forbidden = [
        schema
        for schema in prim.GetAppliedSchemas()
        if str(schema).split(":", maxsplit=1)[0] in _FORBIDDEN_PHYSICS_API_BASES
    ]
    if forbidden:
        _fail(
            "forbidden_physics_ownership",
            f"Mesh owns {list(map(str, forbidden))}: {path}",
        )
    for candidate in stage.TraverseAll():
        if not candidate.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(candidate)
        targets = {
            str(target.GetPrimPath())
            for target in (
                *joint.GetBody0Rel().GetTargets(),
                *joint.GetBody1Rel().GetTargets(),
            )
        }
        if path in targets:
            _fail("joint_endpoint_target", f"planned Mesh is a joint endpoint: {path}")


def _classify_api_schemas(
    path: str,
    schemas: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parent: list[str] = []
    collider: list[str] = []
    for schema in schemas:
        base = schema.split(":", maxsplit=1)[0]
        if base in _COLLISION_API_BASES:
            collider.append(schema)
        elif base in _FORBIDDEN_PHYSICS_API_BASES:
            _fail("forbidden_physics_ownership", f"unsupported API at {path}: {schema}")
        elif base in _ALLOWED_PARENT_API_BASES:
            parent.append(schema)
        else:
            _fail("unsupported_api_schema", f"cannot migrate API at {path}: {schema}")
    if "PhysicsCollisionAPI" not in {
        item.split(":", maxsplit=1)[0] for item in collider
    }:
        _fail("collision_api_required", f"PhysicsCollisionAPI is missing at {path}")
    return tuple(parent), tuple(collider)


def _validated_indices(
    values: Any,
    *,
    upper_bound: int,
    label: str,
) -> tuple[int, ...]:
    validated: list[int] = []
    for raw_value in values:
        if isinstance(raw_value, bool):
            _fail("invalid_indices", f"{label} contains a boolean index")
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            _fail("invalid_indices", f"{label} contains a non-integer index")
        if value < 0 or value >= upper_bound:
            _fail("invalid_indices", f"{label} contains an out-of-bounds index")
        validated.append(value)
    if not validated:
        _fail("invalid_indices", f"{label} is empty")
    return tuple(validated)


def _reject_nonfinite_points(points: Any, *, path: str) -> None:
    if not points:
        _fail("invalid_mesh", f"Mesh has no points: {path}")
    for point in points:
        if len(point) != 3 or any(not math.isfinite(float(value)) for value in point):
            _fail("invalid_mesh", f"Mesh contains a non-finite point: {path}")


def _canonical_points(
    points: Any,
) -> tuple[list[int], dict[int, tuple[int, ...]], dict[int, int]]:
    coordinate_to_id: dict[tuple[float, float, float], int] = {}
    point_ids: list[int] = []
    members: defaultdict[int, list[int]] = defaultdict(list)
    representatives: dict[int, int] = {}
    for point_index, point in enumerate(points):
        key = (float(point[0]), float(point[1]), float(point[2]))
        point_id = coordinate_to_id.get(key)
        if point_id is None:
            point_id = len(coordinate_to_id)
            coordinate_to_id[key] = point_id
            representatives[point_id] = point_index
        point_ids.append(point_id)
        members[point_id].append(point_index)
    return (
        point_ids,
        {key: tuple(value) for key, value in members.items()},
        representatives,
    )


def _edge_users(
    faces: Sequence[Sequence[int]],
) -> dict[tuple[int, int], tuple[int, ...]]:
    users: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for corner, first in enumerate(face):
            second = face[(corner + 1) % len(face)]
            if first == second:
                _fail("degenerate_mesh", f"face {face_index} contains a zero edge")
            users[(min(first, second), max(first, second))].append(face_index)
    return {edge: tuple(face_users) for edge, face_users in users.items()}


def _face_components(
    *,
    face_count: int,
    edge_users: Mapping[tuple[int, int], Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    parents = list(range(face_count))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        lower, upper = sorted((first_root, second_root))
        parents[upper] = lower

    for users in edge_users.values():
        if len(users) == 2:
            union(int(users[0]), int(users[1]))
    components: defaultdict[int, list[int]] = defaultdict(list)
    for face_index in range(face_count):
        components[find(face_index)].append(face_index)
    return tuple(
        tuple(component)
        for component in sorted(components.values(), key=lambda item: min(item))
    )


def _build_mesh_parts(
    *,
    prim_path: str,
    points: Any,
    canonical_faces: Sequence[Sequence[int]],
    canonical_representatives: Mapping[int, int],
    components: Sequence[Sequence[int]],
) -> tuple[_MeshPart, ...]:
    parts: list[_MeshPart] = []
    for part_index, face_indices in enumerate(components):
        canonical_point_ids: list[int] = []
        canonical_to_local: dict[int, int] = {}
        local_counts: list[int] = []
        local_indices: list[int] = []
        for face_index in face_indices:
            face = canonical_faces[face_index]
            local_counts.append(len(face))
            for point_id in face:
                local_index = canonical_to_local.get(point_id)
                if local_index is None:
                    local_index = len(canonical_point_ids)
                    canonical_to_local[point_id] = local_index
                    canonical_point_ids.append(point_id)
                local_indices.append(local_index)
        part_points = type(points)(
            [
                points[canonical_representatives[point_id]]
                for point_id in canonical_point_ids
            ]
        )
        parts.append(
            _MeshPart(
                path=f"{prim_path}/{_PART_PREFIX}{part_index:04d}",
                source_face_indices=tuple(int(value) for value in face_indices),
                canonical_point_ids=tuple(canonical_point_ids),
                points=part_points,
                face_vertex_counts=tuple(local_counts),
                face_vertex_indices=tuple(local_indices),
            )
        )
    return tuple(parts)


def _validate_part_topology(part: _MeshPart, *, label: str) -> None:
    point_keys = [tuple(float(value) for value in point) for point in part.points]
    if len(point_keys) != len(set(point_keys)):
        _fail("duplicate_output_points", f"part contains duplicate points: {label}")
    if sum(part.face_vertex_counts) != len(part.face_vertex_indices):
        _fail("invalid_output_topology", f"face counts do not match indices: {label}")
    if set(part.face_vertex_indices) != set(range(len(part.points))):
        _fail("unused_output_points", f"part contains unused points: {label}")
    faces: list[tuple[int, ...]] = []
    offset = 0
    for count in part.face_vertex_counts:
        face = tuple(part.face_vertex_indices[offset : offset + count])
        offset += count
        if len(face) < 3 or len(set(face)) != len(face):
            _fail("degenerate_output", f"part contains a degenerate face: {label}")
        faces.append(face)
    edges = _edge_users(faces)
    if any(len(users) > 2 for users in edges.values()):
        _fail("nonmanifold_output_edge", f"part has a non-manifold edge: {label}")
    _validate_vertex_fans(faces=faces, edge_users=edges, label=label)


def _validate_vertex_fans(
    *,
    faces: Sequence[Sequence[int]],
    edge_users: Mapping[tuple[int, int], Sequence[int]],
    label: str,
) -> None:
    incident_faces: defaultdict[int, set[int]] = defaultdict(set)
    vertex_edges: defaultdict[int, list[tuple[tuple[int, int], Sequence[int]]]] = (
        defaultdict(list)
    )
    for face_index, face in enumerate(faces):
        for point_index in face:
            incident_faces[point_index].add(face_index)
    for edge, users in edge_users.items():
        for point_index in edge:
            vertex_edges[point_index].append((edge, users))
    for point_index, face_indices in incident_faces.items():
        adjacency: dict[int, set[int]] = {
            face_index: set() for face_index in face_indices
        }
        boundary_edge_count = 0
        for _edge, users in vertex_edges[point_index]:
            if len(users) == 1:
                boundary_edge_count += 1
            elif len(users) == 2:
                first, second = int(users[0]), int(users[1])
                adjacency[first].add(second)
                adjacency[second].add(first)
        pending = [min(face_indices)]
        visited: set[int] = set()
        while pending:
            face_index = pending.pop()
            if face_index in visited:
                continue
            visited.add(face_index)
            pending.extend(sorted(adjacency[face_index] - visited, reverse=True))
        if visited != face_indices or boundary_edge_count not in {0, 2}:
            _fail(
                "nonmanifold_output_vertex_fan",
                f"part has a non-manifold vertex fan at point {point_index}: {label}",
            )


def _topology_primvars(
    *,
    prim: Any,
    point_count: int,
    face_count: int,
    corner_count: int,
    canonical_members: Mapping[int, Sequence[int]],
    Usd: Any,
    UsdGeom: Any,
) -> tuple[_TopologyPrimvar, ...]:
    result: list[_TopologyPrimvar] = []
    local_names: set[str] = set()
    for primvar in sorted(
        UsdGeom.PrimvarsAPI(prim).GetPrimvarsWithAuthoredValues(),
        key=lambda item: str(item.GetName()),
    ):
        attr = primvar.GetAttr()
        if attr.GetPrim() != prim:
            continue
        name = str(primvar.GetName())
        local_names.add(name)
        if primvar.GetElementSize() != 1:
            _fail("unsupported_primvar", f"elementSize must be one: {attr.GetPath()}")
        type_name = primvar.GetTypeName()
        if not type_name.isArray:
            _fail("unsupported_primvar", f"value must be an array: {attr.GetPath()}")
        _reject_dynamic_attribute(attr, label="mesh primvar")
        values = attr.Get(Usd.TimeCode.Default())
        if values is None:
            _fail("unreadable_primvar", f"could not read {attr.GetPath()}")
        interpolation = str(primvar.GetInterpolation())
        domain_count = _domain_count(
            interpolation=interpolation,
            point_count=point_count,
            face_count=face_count,
            corner_count=corner_count,
            path=str(attr.GetPath()),
        )
        indices_attr = primvar.GetIndicesAttr()
        indices: tuple[int, ...] | None = None
        indices_metadata: dict[str, Any] | None = None
        if primvar.IsIndexed():
            if not indices_attr:
                _fail(
                    "missing_primvar_indices", f"indices are missing: {attr.GetPath()}"
                )
            _reject_dynamic_attribute(indices_attr, label="mesh primvar indices")
            indices = _validated_indices(
                primvar.GetIndices(Usd.TimeCode.Default()),
                upper_bound=len(values),
                label=f"primvar indices at {attr.GetPath()}",
            )
            if len(indices) != domain_count:
                _fail(
                    "invalid_primvar_domain",
                    f"index count does not match {interpolation}: {attr.GetPath()}",
                )
            indices_metadata = _attribute_metadata(indices_attr)
        elif len(values) != domain_count:
            _fail(
                "invalid_primvar_domain",
                f"value count does not match {interpolation}: {attr.GetPath()}",
            )
        flattened = primvar.ComputeFlattened(Usd.TimeCode.Default())
        if flattened is None or len(flattened) != domain_count:
            _fail("invalid_primvar", f"could not flatten {attr.GetPath()}")
        result.append(
            _TopologyPrimvar(
                name=name,
                type_name=type_name,
                interpolation=primvar.GetInterpolation(),
                element_size=primvar.GetElementSize(),
                value_metadata=_attribute_metadata(attr),
                indices_metadata=indices_metadata,
                values=values,
                indices=indices,
                flattened_values=flattened,
            )
        )
    for inherited in UsdGeom.PrimvarsAPI(prim).FindPrimvarsWithInheritance():
        attr = inherited.GetAttr()
        if attr.GetPrim() == prim or str(inherited.GetName()) in local_names:
            continue
        if str(inherited.GetInterpolation()) != str(UsdGeom.Tokens.constant):
            _fail(
                "unsupported_inherited_primvar",
                f"non-constant inherited primvar at {prim.GetPath()}: {inherited.GetName()}",
            )
    _validate_vertex_domain_equality(
        canonical_members=canonical_members,
        domain_attributes=(),
        primvars=result,
        path=str(prim.GetPath()),
    )
    return tuple(result)


def _domain_count(
    *,
    interpolation: str,
    point_count: int,
    face_count: int,
    corner_count: int,
    path: str,
) -> int:
    if interpolation == "constant":
        return 1
    if interpolation == "uniform":
        return face_count
    if interpolation == "faceVarying":
        return corner_count
    if interpolation in {"vertex", "varying"}:
        return point_count
    _fail(
        "unsupported_interpolation",
        f"unsupported interpolation at {path}: {interpolation}",
    )


def _domain_attribute(
    attr: Any,
    *,
    domain: Literal["constant", "uniform", "faceVarying", "vertex"],
) -> _DomainAttribute:
    _reject_dynamic_attribute(attr, label="mesh attribute")
    values = attr.Get()
    if values is None:
        _fail("unreadable_mesh_data", f"could not read {attr.GetPath()}")
    return _DomainAttribute(
        name=str(attr.GetName()),
        type_name=attr.GetTypeName(),
        custom=bool(attr.IsCustom()),
        variability=attr.GetVariability(),
        metadata=_attribute_metadata(attr),
        domain=domain,
        values=values,
    )


def _validated_domain_attribute(
    *,
    attr: Any,
    domain: Literal["constant", "uniform", "faceVarying", "vertex"],
    expected_count: int,
    label: str,
) -> _DomainAttribute:
    data = _domain_attribute(attr, domain=domain)
    try:
        actual_count = len(data.values)
    except TypeError:
        _fail("invalid_mesh_data", f"{label} is not an array")
    if actual_count != expected_count:
        _fail(
            "invalid_mesh_data",
            f"{label} has {actual_count} values; expected {expected_count}",
        )
    return data


def _infer_custom_attribute_domain(
    *,
    attr: Any,
    point_count: int,
    face_count: int,
    corner_count: int,
) -> Literal["constant", "uniform", "faceVarying", "vertex"] | None:
    _reject_dynamic_attribute(attr, label="custom mesh attribute")
    value = attr.Get()
    if value is None:
        _fail("unreadable_custom_data", f"could not read {attr.GetPath()}")
    if not attr.GetTypeName().isArray:
        return None
    try:
        value_count = len(value)
    except TypeError:
        return None
    interpolation = str(attr.GetMetadata("interpolation") or "")
    explicit_domains = {
        "constant": ("constant", 1),
        "uniform": ("uniform", face_count),
        "faceVarying": ("faceVarying", corner_count),
        "vertex": ("vertex", point_count),
        "varying": ("vertex", point_count),
    }
    if interpolation:
        item = explicit_domains.get(interpolation)
        if item is None or value_count != item[1]:
            _fail(
                "ambiguous_custom_data",
                f"custom attribute interpolation cannot be remapped: {attr.GetPath()}",
            )
        return item[0]  # type: ignore[return-value]
    candidates: list[Literal["uniform", "faceVarying", "vertex"]] = []
    if value_count == face_count:
        candidates.append("uniform")
    if value_count == corner_count:
        candidates.append("faceVarying")
    if value_count == point_count:
        candidates.append("vertex")
    if len(candidates) != 1:
        _fail(
            "ambiguous_custom_data",
            f"custom array domain cannot be proven exactly: {attr.GetPath()}",
        )
    return candidates[0]


def _validate_vertex_domain_equality(
    *,
    canonical_members: Mapping[int, Sequence[int]],
    domain_attributes: Sequence[_DomainAttribute],
    primvars: Sequence[_TopologyPrimvar],
    path: str,
) -> None:
    arrays: list[tuple[str, Any]] = [
        (item.name, item.values)
        for item in domain_attributes
        if item.domain == "vertex"
    ]
    arrays.extend(
        (item.name, item.flattened_values)
        for item in primvars
        if str(item.interpolation) in {"vertex", "varying"}
    )
    for name, values in arrays:
        for members in canonical_members.values():
            reference = values[members[0]]
            if any(
                not _exact_value_equal(reference, values[index])
                for index in members[1:]
            ):
                _fail(
                    "point_domain_conflict",
                    f"exact-equal points disagree in {name} at {path}",
                )


def _attribute_metadata(attr: Any) -> dict[str, Any]:
    metadata = dict(attr.GetAllAuthoredMetadata())
    for key in (
        "connectionPaths",
        "custom",
        "default",
        "timeSamples",
        "typeName",
        "variability",
    ):
        metadata.pop(key, None)
    return metadata


def _property_snapshot(prop: Any) -> dict[str, Any]:
    if not prop or not prop.IsValid():
        _fail("missing_property", "planned property disappeared during preflight")
    if hasattr(prop, "GetTypeName"):
        return {
            "kind": "attribute",
            "type_name": str(prop.GetTypeName()),
            "custom": bool(prop.IsCustom()),
            "variability": str(prop.GetVariability()),
            "value": _usd_value(prop.Get()),
            "metadata": _usd_value(_attribute_metadata(prop)),
        }
    return {
        "kind": "relationship",
        "custom": bool(prop.IsCustom()),
        "targets": [str(target) for target in prop.GetTargets()],
        "metadata": _usd_value(_relationship_metadata(prop)),
    }


def _relationship_metadata(relationship: Any) -> dict[str, Any]:
    metadata = dict(relationship.GetAllAuthoredMetadata())
    for key in ("custom", "targetPaths"):
        metadata.pop(key, None)
    return metadata


def _prim_metadata_without_type_and_apis(prim: Any) -> dict[str, Any]:
    metadata = dict(prim.GetAllAuthoredMetadata())
    metadata.pop("apiSchemas", None)
    metadata.pop("typeName", None)
    return cast(dict[str, Any], _usd_value(metadata))


def _material_bindings(prim: Any, *, UsdShade: Any) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    binding_api = UsdShade.MaterialBindingAPI(prim)
    purposes = (
        UsdShade.Tokens.allPurpose,
        UsdShade.Tokens.preview,
        UsdShade.Tokens.full,
    )
    for purpose in purposes:
        material, _relationship = binding_api.ComputeBoundMaterial(purpose)
        result[str(purpose)] = str(material.GetPath()) if material else None
    return result


def _world_geometry(
    *,
    prim: Any,
    points: Any,
    faces: Sequence[Sequence[int]],
    Usd: Any,
    UsdGeom: Any,
) -> tuple[
    Counter[tuple[tuple[float, float, float], ...]],
    Counter[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
]:
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    transformed: list[tuple[float, float, float]] = []
    for point in points:
        value = matrix.Transform(point)
        transformed.append((float(value[0]), float(value[1]), float(value[2])))
    world_faces: Counter[tuple[tuple[float, float, float], ...]] = Counter()
    world_triangles: Counter[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = Counter()
    for face in faces:
        world_face = tuple(transformed[index] for index in face)
        world_faces[world_face] += 1
        for corner in range(1, len(world_face) - 1):
            world_triangles[
                (world_face[0], world_face[corner], world_face[corner + 1])
            ] += 1
    return world_faces, world_triangles


def _root_metadata(stage: Any) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _usd_value(dict(stage.GetPseudoRoot().GetAllAuthoredMetadata())),
    )


def _prim_inventory(stage: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(prim.GetPath()), str(prim.GetTypeName()))
            for prim in stage.TraverseAll()
        )
    )


def _physics_inventory(stage: Any, *, UsdPhysics: Any) -> dict[str, Any]:
    rigid_bodies: list[str] = []
    colliders: list[str] = []
    masses: list[str] = []
    articulation_roots: list[str] = []
    joints: list[dict[str, Any]] = []
    filtered_pairs: list[dict[str, Any]] = []
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(path)
        if prim.HasAPI(UsdPhysics.MassAPI):
            masses.append(path)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append(
                {
                    "path": path,
                    "type_name": str(prim.GetTypeName()),
                    "body0_targets": sorted(map(str, joint.GetBody0Rel().GetTargets())),
                    "body1_targets": sorted(map(str, joint.GetBody1Rel().GetTargets())),
                }
            )
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            targets = sorted(
                str(target)
                for target in UsdPhysics.FilteredPairsAPI(prim)
                .GetFilteredPairsRel()
                .GetTargets()
            )
            filtered_pairs.append({"path": path, "targets": targets})
    return {
        "rigid_bodies": sorted(rigid_bodies),
        "colliders": sorted(colliders),
        "masses": sorted(masses),
        "articulation_roots": sorted(articulation_roots),
        "joints": sorted(joints, key=lambda item: item["path"]),
        "filtered_pairs": sorted(filtered_pairs, key=lambda item: item["path"]),
    }


def _apply_plan(
    *,
    stage: Any,
    proof: _StageProof,
    Sdf: Any,
    UsdGeom: Any,
    Vt: Any,
) -> list[dict[str, Any]]:
    stage.SetEditTarget(stage.GetRootLayer())
    changes: list[dict[str, Any]] = []
    for primvar_plan in proof.primvar_plans:
        primvar_operation = primvar_plan.operation
        prim = stage.GetPrimAtPath(primvar_operation.prim_path)
        primvar = UsdGeom.Primvar(prim.GetAttribute(primvar_operation.attribute_name))
        if not primvar or not primvar.Set(primvar_plan.compact_values):
            _fail(
                "authoring_failed",
                "could not author compact values: "
                f"{primvar_operation.prim_path}.{primvar_operation.attribute_name}",
            )
        if not primvar.SetIndices(Vt.IntArray(primvar_plan.compact_indices)):
            _fail(
                "authoring_failed",
                "could not author compact indices: "
                f"{primvar_operation.prim_path}.{primvar_operation.attribute_name}",
            )
        changes.append(
            {
                "kind": "compact_indexed_primvar",
                "prim_path": primvar_operation.prim_path,
                "attribute_name": primvar_operation.attribute_name,
                "source_value_count": primvar_operation.expected_value_count,
                "output_value_count": len(primvar_plan.compact_values),
                "index_count": primvar_operation.expected_index_count,
            }
        )

    for mesh_plan in proof.mesh_plans:
        mesh_operation = mesh_plan.operation
        parent = stage.GetPrimAtPath(mesh_operation.prim_path)
        if not parent.SetTypeName("Xform"):
            _fail(
                "authoring_failed",
                f"could not retype {mesh_operation.prim_path} to Xform",
            )
        _set_applied_schemas(parent, mesh_plan.parent_api_schemas, Sdf=Sdf)
        for name in mesh_plan.removed_property_names:
            if parent.HasProperty(name) and not parent.RemoveProperty(name):
                _fail(
                    "authoring_failed",
                    f"could not remove {mesh_operation.prim_path}.{name}",
                )
        for part in mesh_plan.parts:
            mesh = UsdGeom.Mesh.Define(stage, part.path)
            child = mesh.GetPrim()
            _set_applied_schemas(child, mesh_plan.collider_api_schemas, Sdf=Sdf)
            points_attr = mesh.CreatePointsAttr(part.points)
            counts_attr = mesh.CreateFaceVertexCountsAttr(
                Vt.IntArray(part.face_vertex_counts)
            )
            indices_attr = mesh.CreateFaceVertexIndicesAttr(
                Vt.IntArray(part.face_vertex_indices)
            )
            _set_attribute_metadata(
                points_attr,
                mesh_plan.core_attribute_metadata.get("points", {}),
            )
            _set_attribute_metadata(
                counts_attr,
                mesh_plan.core_attribute_metadata.get("faceVertexCounts", {}),
            )
            _set_attribute_metadata(
                indices_attr,
                mesh_plan.core_attribute_metadata.get("faceVertexIndices", {}),
            )
            extent = _extent_for_points(part.points, Vt=Vt)
            extent_attr = mesh.CreateExtentAttr(extent)
            _set_attribute_metadata(
                extent_attr,
                mesh_plan.core_attribute_metadata.get("extent", {}),
            )
            for attribute in mesh_plan.static_mesh_attributes:
                _restore_attribute(
                    child,
                    name=attribute.name,
                    type_name=attribute.type_name,
                    custom=attribute.custom,
                    variability=attribute.variability,
                    value=attribute.values,
                    metadata=attribute.metadata,
                )
            if (
                mesh_plan.authored_normals is not None
                and part.path in mesh_plan.preserved_normal_part_paths
            ):
                normal_values = _project_domain_values(
                    values=mesh_plan.authored_normals.values,
                    domain=mesh_plan.authored_normals.domain,
                    part=part,
                    mesh_plan=mesh_plan,
                )
                _restore_attribute(
                    child,
                    name=mesh_plan.authored_normals.name,
                    type_name=mesh_plan.authored_normals.type_name,
                    custom=mesh_plan.authored_normals.custom,
                    variability=mesh_plan.authored_normals.variability,
                    value=normal_values,
                    metadata=mesh_plan.authored_normals.metadata,
                )
            elif (
                mesh_plan.authored_normals is not None
                and part.path in mesh_plan.omitted_normal_part_paths
            ):
                derived_normals = _derived_face_varying_normals(
                    part=part,
                    orientation=str(mesh.GetOrientationAttr().Get()),
                    template=mesh_plan.authored_normals.values,
                )
                _restore_attribute(
                    child,
                    name=mesh_plan.authored_normals.name,
                    type_name=mesh_plan.authored_normals.type_name,
                    custom=mesh_plan.authored_normals.custom,
                    variability=mesh_plan.authored_normals.variability,
                    value=derived_normals,
                    metadata={
                        key: value
                        for key, value in mesh_plan.authored_normals.metadata.items()
                        if key != "interpolation"
                    },
                )
                if not UsdGeom.Mesh(child).SetNormalsInterpolation(
                    UsdGeom.Tokens.faceVarying
                ):
                    _fail(
                        "authoring_failed",
                        f"could not set derived-normal interpolation: {part.path}",
                    )
            for attribute in mesh_plan.domain_attributes:
                value = _project_domain_values(
                    values=attribute.values,
                    domain=attribute.domain,
                    part=part,
                    mesh_plan=mesh_plan,
                )
                _restore_attribute(
                    child,
                    name=attribute.name,
                    type_name=attribute.type_name,
                    custom=attribute.custom,
                    variability=attribute.variability,
                    value=value,
                    metadata=attribute.metadata,
                )
            for primvar in mesh_plan.primvars:
                _author_part_primvar(
                    child=child,
                    primvar=primvar,
                    part=part,
                    mesh_plan=mesh_plan,
                    Sdf=Sdf,
                    Vt=Vt,
                )
            for name, snapshot in sorted(mesh_plan.collision_properties.items()):
                _restore_property(child, name=name, snapshot=snapshot)
        changes.append(
            {
                "kind": "normalize_collision_mesh",
                "source_mesh_path": mesh_operation.prim_path,
                "output_part_paths": [part.path for part in mesh_plan.parts],
                "source_point_count": mesh_operation.expected_point_count,
                "source_face_count": mesh_operation.expected_face_count,
                "output_part_count": len(mesh_plan.parts),
                "preserved_normal_part_paths": list(
                    mesh_plan.preserved_normal_part_paths
                ),
                "omitted_normal_part_paths": list(mesh_plan.omitted_normal_part_paths),
                "derived_normal_part_paths": list(mesh_plan.omitted_normal_part_paths),
            }
        )
    return changes


def _set_applied_schemas(prim: Any, schemas: Sequence[str], *, Sdf: Any) -> None:
    if not prim.SetMetadata(
        "apiSchemas", Sdf.TokenListOp.CreateExplicit(list(schemas))
    ):
        _fail("authoring_failed", f"could not set API schemas at {prim.GetPath()}")


def _extent_for_points(points: Any, *, Vt: Any) -> Any:
    minimum = [min(float(point[axis]) for point in points) for axis in range(3)]
    maximum = [max(float(point[axis]) for point in points) for axis in range(3)]
    return Vt.Vec3fArray((tuple(minimum), tuple(maximum)))


def _normal_part_policy(
    *,
    authored_normals: _DomainAttribute | None,
    parts: Sequence[_MeshPart],
    face_corner_offsets: tuple[int, ...],
    canonical_representatives: Mapping[int, int],
    orientation: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if authored_normals is None:
        return (), ()
    if orientation not in {"leftHanded", "rightHanded"}:
        _fail(
            "unsupported_orientation",
            f"cannot prove authored-normal winding for orientation {orientation!r}",
        )
    preserved: list[str] = []
    omitted: list[str] = []
    for part in parts:
        projected = _project_domain_values_for_part(
            values=authored_normals.values,
            domain=authored_normals.domain,
            part=part,
            face_corner_offsets=face_corner_offsets,
            canonical_representatives=canonical_representatives,
        )
        if _normals_are_winding_consistent(
            part=part,
            normals=projected,
            domain=authored_normals.domain,
            orientation=orientation,
        ):
            preserved.append(part.path)
        else:
            omitted.append(part.path)
    return tuple(preserved), tuple(omitted)


def _normals_are_winding_consistent(
    *,
    part: _MeshPart,
    normals: Any,
    domain: Literal["constant", "uniform", "faceVarying", "vertex"],
    orientation: str,
) -> bool:
    expected_count = {
        "constant": 1,
        "uniform": len(part.face_vertex_counts),
        "faceVarying": len(part.face_vertex_indices),
        "vertex": len(part.points),
    }[domain]
    if normals is None or len(normals) != expected_count:
        return False
    face_offset = 0
    for face_index, count in enumerate(part.face_vertex_counts):
        face = part.face_vertex_indices[face_offset : face_offset + count]
        geometric = _oriented_face_normal(
            points=part.points,
            face=face,
            orientation=orientation,
        )
        if geometric is None:
            return False
        normal_indices: Sequence[int]
        if domain == "constant":
            normal_indices = (0,)
        elif domain == "uniform":
            normal_indices = (face_index,)
        elif domain == "faceVarying":
            normal_indices = tuple(range(face_offset, face_offset + count))
        else:
            normal_indices = face
        for normal_index in normal_indices:
            try:
                normal = normals[normal_index]
                values = (
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                )
            except (IndexError, TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in values):
                return False
            normal_length = math.sqrt(sum(value * value for value in values))
            if not math.isfinite(normal_length) or normal_length <= 0.0:
                return False
            cosine = (
                sum(geometric[axis] * values[axis] for axis in range(3)) / normal_length
            )
            if not math.isfinite(cosine) or cosine <= _NORMAL_WINDING_COSINE_EPSILON:
                return False
        face_offset += count
    return face_offset == len(part.face_vertex_indices)


def _oriented_face_normal(
    *,
    points: Any,
    face: Sequence[int],
    orientation: str,
) -> tuple[float, float, float] | None:
    if len(face) < 3:
        return None
    origin = points[face[0]]
    normal = [0.0, 0.0, 0.0]
    for corner in range(1, len(face) - 1):
        first = points[face[corner]]
        second = points[face[corner + 1]]
        first_vector = [float(first[axis] - origin[axis]) for axis in range(3)]
        second_vector = [float(second[axis] - origin[axis]) for axis in range(3)]
        normal[0] += (
            first_vector[1] * second_vector[2] - first_vector[2] * second_vector[1]
        )
        normal[1] += (
            first_vector[2] * second_vector[0] - first_vector[0] * second_vector[2]
        )
        normal[2] += (
            first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
        )
    if orientation == "leftHanded":
        normal = [-value for value in normal]
    length = math.sqrt(sum(value * value for value in normal))
    if not math.isfinite(length) or length <= 0.0:
        return None
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def _derived_face_varying_normals(
    *,
    part: _MeshPart,
    orientation: str,
    template: Any,
) -> Any:
    derived: list[tuple[float, float, float]] = []
    face_offset = 0
    for count in part.face_vertex_counts:
        face = part.face_vertex_indices[face_offset : face_offset + count]
        normal = _oriented_face_normal(
            points=part.points,
            face=face,
            orientation=orientation,
        )
        if normal is None:
            _fail(
                "degenerate_output",
                f"cannot derive a face normal for {part.path}",
            )
        derived.extend([normal] * count)
        face_offset += count
    if face_offset != len(part.face_vertex_indices):
        _fail(
            "invalid_output_topology",
            f"cannot derive normals from mismatched topology: {part.path}",
        )
    return type(template)(derived)


def _project_domain_values(
    *,
    values: Any,
    domain: Literal["constant", "uniform", "faceVarying", "vertex"],
    part: _MeshPart,
    mesh_plan: _MeshPlan,
) -> Any:
    return _project_domain_values_for_part(
        values=values,
        domain=domain,
        part=part,
        face_corner_offsets=mesh_plan.face_corner_offsets,
        canonical_representatives=mesh_plan.canonical_representatives,
    )


def _project_domain_values_for_part(
    *,
    values: Any,
    domain: Literal["constant", "uniform", "faceVarying", "vertex"],
    part: _MeshPart,
    face_corner_offsets: Sequence[int],
    canonical_representatives: Mapping[int, int],
) -> Any:
    if domain == "constant":
        return values
    if domain == "uniform":
        return type(values)([values[index] for index in part.source_face_indices])
    if domain == "faceVarying":
        selected: list[Any] = []
        for face_index in part.source_face_indices:
            start = face_corner_offsets[face_index]
            end = face_corner_offsets[face_index + 1]
            selected.extend(values[start:end])
        return type(values)(selected)
    return type(values)(
        [
            values[canonical_representatives[point_id]]
            for point_id in part.canonical_point_ids
        ]
    )


def _author_part_primvar(
    *,
    child: Any,
    primvar: _TopologyPrimvar,
    part: _MeshPart,
    mesh_plan: _MeshPlan,
    Sdf: Any,
    Vt: Any,
) -> None:
    domain = _primvar_domain(str(primvar.interpolation))
    if primvar.indices is None:
        value = _project_domain_values(
            values=primvar.values,
            domain=domain,
            part=part,
            mesh_plan=mesh_plan,
        )
        indices: tuple[int, ...] | None = None
    else:
        selected_indices = _project_domain_values(
            values=primvar.indices,
            domain=domain,
            part=part,
            mesh_plan=mesh_plan,
        )
        used = set(int(value) for value in selected_indices)
        old_to_new: dict[int, int] = {}
        compact_values_list: list[Any] = []
        for old_index, value_item in enumerate(primvar.values):
            if old_index not in used:
                continue
            old_to_new[old_index] = len(compact_values_list)
            compact_values_list.append(value_item)
        value = type(primvar.values)(compact_values_list)
        indices = tuple(old_to_new[int(index)] for index in selected_indices)
    attr = _restore_attribute(
        child,
        name=primvar.name,
        type_name=primvar.type_name,
        custom=False,
        variability=Sdf.VariabilityVarying,
        value=value,
        metadata=primvar.value_metadata,
    )
    if indices is not None:
        indices_attr = _restore_attribute(
            child,
            name=f"{primvar.name}:indices",
            type_name=Sdf.ValueTypeNames.IntArray,
            custom=False,
            variability=Sdf.VariabilityVarying,
            value=Vt.IntArray(indices),
            metadata=primvar.indices_metadata or {},
        )
        if not indices_attr:
            _fail("authoring_failed", f"could not author indices at {child.GetPath()}")
    authored = type(primvar.flattened_values)(
        [value[index] for index in indices] if indices is not None else list(value)
    )
    expected = _project_domain_values(
        values=primvar.flattened_values,
        domain=domain,
        part=part,
        mesh_plan=mesh_plan,
    )
    if not _exact_array_equal(authored, expected):
        _fail("authoring_failed", f"primvar projection changed at {attr.GetPath()}")


def _primvar_domain(
    interpolation: str,
) -> Literal["constant", "uniform", "faceVarying", "vertex"]:
    if interpolation in {"vertex", "varying"}:
        return "vertex"
    if interpolation in {"constant", "uniform", "faceVarying"}:
        return interpolation  # type: ignore[return-value]
    _fail("unsupported_interpolation", f"unsupported interpolation: {interpolation}")


def _restore_attribute(
    prim: Any,
    *,
    name: str,
    type_name: Any,
    custom: bool,
    variability: Any,
    value: Any,
    metadata: Mapping[str, Any],
) -> Any:
    attr = prim.CreateAttribute(
        name,
        type_name,
        custom=custom,
        variability=variability,
    )
    if not attr.Set(value):
        _fail("authoring_failed", f"could not set {prim.GetPath()}.{name}")
    _set_attribute_metadata(attr, metadata)
    return attr


def _set_attribute_metadata(attr: Any, metadata: Mapping[str, Any]) -> None:
    for key, value in sorted(metadata.items()):
        if not attr.SetMetadata(key, value):
            _fail("authoring_failed", f"could not set metadata {attr.GetPath()}.{key}")


def _copyable_property_snapshot(prop: Any) -> dict[str, Any]:
    if not prop or not prop.IsValid():
        _fail("missing_property", "planned property disappeared before copying")
    if hasattr(prop, "GetTypeName"):
        _reject_dynamic_attribute(prop, label="collision attribute")
        return {
            "kind": "attribute",
            "type_name": prop.GetTypeName(),
            "custom": bool(prop.IsCustom()),
            "variability": prop.GetVariability(),
            "value": prop.Get(),
            "metadata": _attribute_metadata(prop),
        }
    return {
        "kind": "relationship",
        "custom": bool(prop.IsCustom()),
        "targets": tuple(prop.GetTargets()),
        "metadata": _relationship_metadata(prop),
    }


def _restore_property(prim: Any, *, name: str, snapshot: Mapping[str, Any]) -> None:
    if snapshot["kind"] == "attribute":
        _restore_attribute(
            prim,
            name=name,
            type_name=snapshot["type_name"],
            custom=bool(snapshot["custom"]),
            variability=snapshot["variability"],
            value=snapshot["value"],
            metadata=snapshot["metadata"],
        )
        return
    relationship = prim.CreateRelationship(name, custom=bool(snapshot["custom"]))
    if not relationship.SetTargets(snapshot["targets"]):
        _fail(
            "authoring_failed", f"could not restore targets at {prim.GetPath()}.{name}"
        )
    for key, value in sorted(snapshot["metadata"].items()):
        if not relationship.SetMetadata(key, value):
            _fail(
                "authoring_failed",
                f"could not restore metadata at {relationship.GetPath()}.{key}",
            )


def _copyable_property_semantics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if snapshot["kind"] == "attribute":
        return {
            "kind": "attribute",
            "type_name": str(snapshot["type_name"]),
            "custom": bool(snapshot["custom"]),
            "variability": str(snapshot["variability"]),
            "value": _usd_value(snapshot["value"]),
            "metadata": _usd_value(snapshot["metadata"]),
        }
    return {
        "kind": "relationship",
        "custom": bool(snapshot["custom"]),
        "targets": [str(target) for target in snapshot["targets"]],
        "metadata": _usd_value(snapshot["metadata"]),
    }


def _validate_authored_stage(
    *,
    stage: Any,
    proof: _StageProof,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    default_prim = stage.GetDefaultPrim()
    if not default_prim or str(default_prim.GetPath()) != proof.default_prim_path:
        _fail("invariant_failed", "default prim changed")
    if _root_metadata(stage) != proof.root_metadata:
        _fail("invariant_failed", "root-layer metadata changed")
    expected_inventory = dict(proof.prim_inventory)
    for mesh_plan in proof.mesh_plans:
        expected_inventory[mesh_plan.operation.prim_path] = "Xform"
        for part in mesh_plan.parts:
            expected_inventory[part.path] = "Mesh"
    if dict(_prim_inventory(stage)) != expected_inventory:
        _fail("invariant_failed", "unexpected prim paths or types changed")
    expected_physics = json.loads(json.dumps(proof.physics_inventory))
    expected_colliders = set(expected_physics["colliders"])
    for mesh_plan in proof.mesh_plans:
        expected_colliders.remove(mesh_plan.operation.prim_path)
        expected_colliders.update(part.path for part in mesh_plan.parts)
    expected_physics["colliders"] = sorted(expected_colliders)
    if _physics_inventory(stage, UsdPhysics=UsdPhysics) != expected_physics:
        _fail("invariant_failed", "physics ownership or joint graph changed")
    for primvar_plan in proof.primvar_plans:
        _validate_primvar_readback(
            stage=stage,
            primvar_plan=primvar_plan,
            Usd=Usd,
            UsdGeom=UsdGeom,
        )
    for mesh_plan in proof.mesh_plans:
        _validate_mesh_readback(
            stage=stage,
            mesh_plan=mesh_plan,
            Sdf=Sdf,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
            UsdShade=UsdShade,
        )


def _validate_primvar_readback(
    *,
    stage: Any,
    primvar_plan: _PrimvarPlan,
    Usd: Any,
    UsdGeom: Any,
) -> None:
    operation = primvar_plan.operation
    prim = stage.GetPrimAtPath(operation.prim_path)
    primvar = UsdGeom.Primvar(prim.GetAttribute(operation.attribute_name))
    if not primvar:
        _fail(
            "invariant_failed", f"compacted primvar is missing: {operation.prim_path}"
        )
    value_attr = primvar.GetAttr()
    indices_attr = primvar.GetIndicesAttr()
    if (
        primvar.GetTypeName() != primvar_plan.type_name
        or primvar.GetInterpolation() != primvar_plan.interpolation
        or primvar.GetElementSize() != primvar_plan.element_size
        or _attribute_metadata(value_attr) != primvar_plan.value_metadata
        or not indices_attr
        or _attribute_metadata(indices_attr) != primvar_plan.indices_metadata
    ):
        _fail("invariant_failed", f"primvar metadata changed: {value_attr.GetPath()}")
    if not _exact_array_equal(value_attr.Get(), primvar_plan.compact_values):
        _fail("invariant_failed", f"primvar values differ: {value_attr.GetPath()}")
    if (
        tuple(int(value) for value in primvar.GetIndices())
        != primvar_plan.compact_indices
    ):
        _fail("invariant_failed", f"primvar indices differ: {value_attr.GetPath()}")
    flattened = primvar.ComputeFlattened(Usd.TimeCode.Default())
    if not _exact_array_equal(flattened, primvar_plan.flattened_values):
        _fail("invariant_failed", f"flattened primvar changed: {value_attr.GetPath()}")


def _validate_mesh_readback(
    *,
    stage: Any,
    mesh_plan: _MeshPlan,
    Sdf: Any,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
    UsdShade: Any,
) -> None:
    path = mesh_plan.operation.prim_path
    parent = stage.GetPrimAtPath(path)
    if not parent.IsA(UsdGeom.Xform) or parent.IsA(UsdGeom.Gprim):
        _fail("invariant_failed", f"source Mesh was not replaced by Xform: {path}")
    if (
        tuple(str(token) for token in parent.GetAppliedSchemas())
        != mesh_plan.parent_api_schemas
    ):
        _fail("invariant_failed", f"parent API schemas changed: {path}")
    if (
        _prim_metadata_without_type_and_apis(parent)
        != mesh_plan.retained_parent_metadata
    ):
        _fail("invariant_failed", f"parent metadata changed: {path}")
    actual_retained = {
        name: _property_snapshot(parent.GetProperty(name))
        for name in sorted(mesh_plan.retained_parent_properties)
    }
    if actual_retained != mesh_plan.retained_parent_properties:
        _fail("invariant_failed", f"retained parent properties changed: {path}")
    if any(parent.HasProperty(name) for name in mesh_plan.removed_property_names):
        _fail("invariant_failed", f"migrated Mesh property remains on parent: {path}")
    if parent.HasAPI(UsdPhysics.CollisionAPI):
        _fail("invariant_failed", f"parent still owns CollisionAPI: {path}")
    actual_children = tuple(str(child.GetPath()) for child in parent.GetChildren())
    expected_children = tuple(part.path for part in mesh_plan.parts)
    if actual_children != expected_children:
        _fail("invariant_failed", f"deterministic child paths changed: {path}")

    output_faces: Counter[tuple[tuple[float, float, float], ...]] = Counter()
    output_triangles: Counter[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = Counter()
    for part in mesh_plan.parts:
        child = stage.GetPrimAtPath(part.path)
        if not child.IsA(UsdGeom.Mesh):
            _fail("invariant_failed", f"output part is not a Mesh: {part.path}")
        if (
            tuple(str(token) for token in child.GetAppliedSchemas())
            != mesh_plan.collider_api_schemas
        ):
            _fail("invariant_failed", f"collider APIs changed: {part.path}")
        if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(UsdPhysics.MassAPI):
            _fail("invariant_failed", f"part acquired body ownership: {part.path}")
        for name, snapshot in mesh_plan.collision_properties.items():
            if _property_snapshot(
                child.GetProperty(name)
            ) != _copyable_property_semantics(snapshot):
                _fail(
                    "invariant_failed",
                    f"collision property changed: {part.path}.{name}",
                )
        if _material_bindings(child, UsdShade=UsdShade) != mesh_plan.material_bindings:
            _fail("invariant_failed", f"material binding changed: {part.path}")
        imageable = UsdGeom.Imageable(child)
        if (
            str(imageable.ComputeEffectiveVisibility())
            != mesh_plan.effective_visibility
            or str(imageable.ComputePurpose()) != mesh_plan.effective_purpose
        ):
            _fail("invariant_failed", f"visibility or purpose changed: {part.path}")
        _validate_part_payload(
            child=child,
            part=part,
            mesh_plan=mesh_plan,
            Usd=Usd,
            UsdGeom=UsdGeom,
        )
        mesh = UsdGeom.Mesh(child)
        points = mesh.GetPointsAttr().Get()
        counts = tuple(int(value) for value in mesh.GetFaceVertexCountsAttr().Get())
        indices = tuple(int(value) for value in mesh.GetFaceVertexIndicesAttr().Get())
        faces: list[tuple[int, ...]] = []
        offset = 0
        for count in counts:
            faces.append(tuple(indices[offset : offset + count]))
            offset += count
        faces_counter, triangle_counter = _world_geometry(
            prim=child,
            points=points,
            faces=faces,
            Usd=Usd,
            UsdGeom=UsdGeom,
        )
        output_faces.update(faces_counter)
        output_triangles.update(triangle_counter)
    if output_faces != mesh_plan.source_world_faces:
        _fail("invariant_failed", f"world-space face multiset changed: {path}")
    if output_triangles != mesh_plan.source_world_triangles:
        _fail("invariant_failed", f"world-space triangle multiset changed: {path}")


def _validate_part_payload(
    *,
    child: Any,
    part: _MeshPart,
    mesh_plan: _MeshPlan,
    Usd: Any,
    UsdGeom: Any,
) -> None:
    from pxr import Vt

    mesh = UsdGeom.Mesh(child)
    actual_part = _MeshPart(
        path=part.path,
        source_face_indices=part.source_face_indices,
        canonical_point_ids=part.canonical_point_ids,
        points=mesh.GetPointsAttr().Get(),
        face_vertex_counts=tuple(
            int(value) for value in mesh.GetFaceVertexCountsAttr().Get()
        ),
        face_vertex_indices=tuple(
            int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()
        ),
    )
    if (
        not _exact_array_equal(actual_part.points, part.points)
        or actual_part.face_vertex_counts != part.face_vertex_counts
        or actual_part.face_vertex_indices != part.face_vertex_indices
    ):
        _fail("invariant_failed", f"part topology changed: {part.path}")
    _validate_part_topology(actual_part, label=part.path)
    expected_extent = _usd_value(_extent_for_points(part.points, Vt=Vt))
    if _usd_value(mesh.GetExtentAttr().Get()) != expected_extent:
        _fail("invariant_failed", f"part extent changed: {part.path}")
    for name, metadata in mesh_plan.core_attribute_metadata.items():
        attr = child.GetAttribute(name)
        if not attr or _attribute_metadata(attr) != metadata:
            _fail("invariant_failed", f"core Mesh metadata changed: {part.path}.{name}")
    for attribute in mesh_plan.static_mesh_attributes:
        attr = child.GetAttribute(attribute.name)
        if not attr or not _exact_value_equal(attr.Get(), attribute.values):
            _fail(
                "invariant_failed",
                f"static Mesh data changed: {part.path}.{attribute.name}",
            )
        if _attribute_metadata(attr) != attribute.metadata:
            _fail(
                "invariant_failed",
                f"static Mesh metadata changed: {part.path}.{attribute.name}",
            )
    for attribute in mesh_plan.domain_attributes:
        expected = _project_domain_values(
            values=attribute.values,
            domain=attribute.domain,
            part=part,
            mesh_plan=mesh_plan,
        )
        attr = child.GetAttribute(attribute.name)
        if not attr or not _exact_array_equal(attr.Get(), expected):
            _fail(
                "invariant_failed", f"domain data changed: {part.path}.{attribute.name}"
            )
        if _attribute_metadata(attr) != attribute.metadata:
            _fail(
                "invariant_failed",
                f"domain metadata changed: {part.path}.{attribute.name}",
            )
    _validate_part_normals(
        child=child,
        part=part,
        mesh_plan=mesh_plan,
        UsdGeom=UsdGeom,
    )
    for primvar in mesh_plan.primvars:
        authored = UsdGeom.Primvar(child.GetAttribute(primvar.name))
        if not authored:
            _fail("invariant_failed", f"primvar is missing: {part.path}.{primvar.name}")
        expected = _project_domain_values(
            values=primvar.flattened_values,
            domain=_primvar_domain(str(primvar.interpolation)),
            part=part,
            mesh_plan=mesh_plan,
        )
        flattened = authored.ComputeFlattened(Usd.TimeCode.Default())
        if not _exact_array_equal(flattened, expected):
            _fail("invariant_failed", f"primvar changed: {part.path}.{primvar.name}")
        if _attribute_metadata(authored.GetAttr()) != primvar.value_metadata:
            _fail(
                "invariant_failed",
                f"primvar metadata changed: {part.path}.{primvar.name}",
            )
        if authored.IsIndexed():
            indices_attr = authored.GetIndicesAttr()
            if not indices_attr or _attribute_metadata(indices_attr) != (
                primvar.indices_metadata or {}
            ):
                _fail(
                    "invariant_failed",
                    f"primvar index metadata changed: {part.path}.{primvar.name}",
                )


def _validate_part_normals(
    *,
    child: Any,
    part: _MeshPart,
    mesh_plan: _MeshPlan,
    UsdGeom: Any,
) -> None:
    normal_attr = UsdGeom.Mesh(child).GetNormalsAttr()
    is_preserved = part.path in mesh_plan.preserved_normal_part_paths
    is_omitted = part.path in mesh_plan.omitted_normal_part_paths
    if mesh_plan.authored_normals is None:
        if is_preserved or is_omitted or normal_attr.HasAuthoredValueOpinion():
            _fail("invariant_failed", f"part acquired authored normals: {part.path}")
        return
    if is_preserved == is_omitted:
        _fail("invariant_failed", f"part has ambiguous normal policy: {part.path}")
    if is_omitted:
        orientation = str(UsdGeom.Mesh(child).GetOrientationAttr().Get())
        expected = _derived_face_varying_normals(
            part=part,
            orientation=orientation,
            template=mesh_plan.authored_normals.values,
        )
        expected_metadata = {
            key: value
            for key, value in mesh_plan.authored_normals.metadata.items()
            if key != "interpolation"
        }
        actual_metadata = _attribute_metadata(normal_attr)
        actual_metadata.pop("interpolation", None)
        if (
            not normal_attr.HasAuthoredValueOpinion()
            or not _exact_array_equal(normal_attr.Get(), expected)
            or actual_metadata != expected_metadata
            or str(UsdGeom.Mesh(child).GetNormalsInterpolation()) != "faceVarying"
        ):
            _fail(
                "invariant_failed",
                f"winding-derived normals changed: {part.path}",
            )
        if not _normals_are_winding_consistent(
            part=part,
            normals=normal_attr.Get(),
            domain="faceVarying",
            orientation=orientation,
        ):
            _fail(
                "invariant_failed",
                f"winding-derived normals are invalid: {part.path}",
            )
        return
    expected = _project_domain_values(
        values=mesh_plan.authored_normals.values,
        domain=mesh_plan.authored_normals.domain,
        part=part,
        mesh_plan=mesh_plan,
    )
    if (
        not normal_attr.HasAuthoredValueOpinion()
        or not _exact_array_equal(normal_attr.Get(), expected)
        or _attribute_metadata(normal_attr) != mesh_plan.authored_normals.metadata
    ):
        _fail("invariant_failed", f"preserved normals changed: {part.path}")
    if not _normals_are_winding_consistent(
        part=part,
        normals=normal_attr.Get(),
        domain=mesh_plan.authored_normals.domain,
        orientation=str(UsdGeom.Mesh(child).GetOrientationAttr().Get()),
    ):
        _fail("invariant_failed", f"preserved normals are invalid: {part.path}")


def _write_usdz(
    *,
    root_layer_path: Path,
    package_root: Path,
    output_path: Path,
    UsdUtils: Any,
) -> None:
    for path in sorted(package_root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            _fail("unsafe_package", f"package contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            os.utime(path, (_FIXED_PACKAGE_MTIME, _FIXED_PACKAGE_MTIME))
        elif not stat.S_ISDIR(metadata.st_mode):
            _fail("unsafe_package", f"package contains a non-file entry: {path}")
    if not UsdUtils.CreateNewUsdzPackage(str(root_layer_path), str(output_path)):
        _fail("output_write_failed", "OpenUSD could not create the USDZ package")
    _asset_manifest(output_path)


def _asset_manifest(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos:
            _fail("output_readback_failed", "output USDZ is empty")
        root_parts = _validate_usdz_info(infos[0], require_file=True)
        root_entry = "/".join(root_parts)
        if Path(root_entry).suffix.lower() not in _USD_LAYER_SUFFIXES:
            _fail("output_readback_failed", "output USDZ first entry is not USD")
        seen: set[str] = set()
        for info in infos:
            parts = _validate_usdz_info(info, require_file=False)
            name = "/".join(parts)
            if name in seen:
                _fail("output_readback_failed", f"duplicate output entry: {name}")
            seen.add(name)
            if info.is_dir():
                continue
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append(
                {
                    "path": name,
                    "size": info.file_size,
                    "sha256": digest.hexdigest(),
                }
            )
    ordered = sorted(entries, key=lambda item: item["path"])
    return {
        "container": "usdz",
        "root_entry": root_entry,
        "entry_paths": [entry["path"] for entry in ordered],
        "entries": ordered,
        "dependency_bundle_sha256": _canonical_json_sha256(ordered),
    }


def _validate_output_manifest(
    *,
    source_manifest: Mapping[str, Any],
    output_manifest: Mapping[str, Any],
) -> None:
    if output_manifest["root_entry"] != source_manifest["root_entry"]:
        _fail("package_invariant_failed", "output root entry changed")
    if output_manifest["entry_paths"] != source_manifest["entry_paths"]:
        _fail("package_invariant_failed", "output package file inventory changed")
    source_entries = {item["path"]: item for item in source_manifest["entries"]}
    output_entries = {item["path"]: item for item in output_manifest["entries"]}
    root_entry = str(source_manifest["root_entry"])
    changed_dependencies = [
        path
        for path in sorted(source_entries)
        if path != root_entry
        and (
            source_entries[path]["sha256"] != output_entries[path]["sha256"]
            or source_entries[path]["size"] != output_entries[path]["size"]
        )
    ]
    if changed_dependencies:
        _fail(
            "package_invariant_failed",
            "output changed dependency bytes: " + ", ".join(changed_dependencies[:5]),
        )


def _manifest_entry(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "size": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _receipt_payload(
    *,
    plan: Gate3AMeshTopologyPlan,
    plan_sha256: str,
    source: _FileIdentity,
    source_capture: _SourceCapture,
    output_sha256: str,
    output_manifest: Mapping[str, Any],
    proof: _StageProof,
    changes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mesh_proofs = []
    for mesh_plan in proof.mesh_plans:
        mesh_proofs.append(
            {
                "source_mesh_path": mesh_plan.operation.prim_path,
                "output_part_paths": [part.path for part in mesh_plan.parts],
                "source_point_count": mesh_plan.operation.expected_point_count,
                "source_exact_unique_point_count": (
                    mesh_plan.operation.expected_exact_unique_point_count
                ),
                "source_unused_point_count": (
                    mesh_plan.operation.expected_unused_point_count
                ),
                "source_face_count": mesh_plan.operation.expected_face_count,
                "source_face_vertex_index_count": (
                    mesh_plan.operation.expected_face_vertex_index_count
                ),
                "source_nonmanifold_edge_count": (
                    mesh_plan.operation.expected_nonmanifold_edge_count
                ),
                "output_part_count": len(mesh_plan.parts),
                "authored_normal_policy": (
                    "preserve_per_part_only_when_exact_domain_remap_and_"
                    "positive_winding_alignment_are_proven_otherwise_"
                    "replace_with_face_winding_derived_normals"
                ),
                "source_had_authored_normals": (mesh_plan.authored_normals is not None),
                "preserved_normal_part_paths": list(
                    mesh_plan.preserved_normal_part_paths
                ),
                "omitted_normal_part_paths": list(mesh_plan.omitted_normal_part_paths),
                "omitted_source_normal_part_paths": list(
                    mesh_plan.omitted_normal_part_paths
                ),
                "derived_normal_part_paths": list(mesh_plan.omitted_normal_part_paths),
                "world_face_multiset_sha256": _counter_sha256(
                    mesh_plan.source_world_faces
                ),
                "world_triangle_multiset_sha256": _counter_sha256(
                    mesh_plan.source_world_triangles
                ),
                "collision_property_sha256": _canonical_json_sha256(
                    {
                        name: _copyable_property_semantics(snapshot)
                        for name, snapshot in sorted(
                            mesh_plan.collision_properties.items()
                        )
                    }
                ),
                "material_binding_sha256": _canonical_json_sha256(
                    mesh_plan.material_bindings
                ),
            }
        )
    return {
        "schema_version": GATE3A_MESH_TOPOLOGY_RECEIPT_SCHEMA_VERSION,
        "requirement": GATE3A_MESH_TOPOLOGY_REQUIREMENT,
        "plan_schema_version": plan.schema_version,
        "plan_sha256": plan_sha256,
        "source_identity": {
            "asset_path": str(source.path),
            "asset_sha256": source.sha256,
            "container": source_capture.source_container,
            "root_entry": source_capture.root_entry,
            "dependency_bundle_sha256": source_capture.source_manifest[
                "dependency_bundle_sha256"
            ],
        },
        "output_identity": {
            "asset_sha256": output_sha256,
            "container": "usdz",
            "root_entry": output_manifest["root_entry"],
            "dependency_bundle_sha256": output_manifest["dependency_bundle_sha256"],
        },
        "provenance": plan.provenance.model_dump(mode="json"),
        "changes": list(changes),
        "primvar_proofs": [
            {
                "prim_path": item.operation.prim_path,
                "attribute_name": item.operation.attribute_name,
                "source_value_count": item.operation.expected_value_count,
                "output_value_count": len(item.compact_values),
                "index_count": item.operation.expected_index_count,
                "flattened_value_sha256": _canonical_json_sha256(
                    _usd_value(item.flattened_values)
                ),
            }
            for item in proof.primvar_plans
        ],
        "mesh_proofs": mesh_proofs,
        "invariants": {
            "source_bytes_preserved": True,
            "evidence_bytes_preserved": True,
            "source_prim_paths_preserved": True,
            "source_transforms_preserved": True,
            "material_bindings_preserved": True,
            "visibility_and_purpose_preserved": True,
            "joint_graph_preserved": True,
            "rigid_mass_articulation_ownership_preserved": True,
            "collision_ownership_migrated_to_parts": True,
            "world_face_multisets_preserved": True,
            "world_triangle_multisets_preserved": True,
            "output_has_no_duplicate_or_unused_points": True,
            "output_has_no_nonmanifold_edges_or_vertex_fans": True,
            "authored_normals_preserved_only_when_winding_consistent": True,
            "invalid_source_normals_replaced_with_winding_derived_normals": True,
            "dependencies_preserved": True,
            "self_contained_usdz": True,
            "private_snapshot_used": True,
            "atomic_bundle_publication": True,
            "readback_verified": True,
        },
        "status": "AUTHORED",
        "passed": True,
        "reason": "Published an exact evidence-backed Gate 3A mesh derivative.",
    }


def _counter_sha256(counter: Counter[Any]) -> str:
    payload = [
        {"item": _usd_value(item), "count": count}
        for item, count in sorted(counter.items(), key=lambda pair: repr(pair[0]))
    ]
    return _canonical_json_sha256(payload)


def _verify_all_inputs_unchanged(
    *,
    source: _FileIdentity,
    plan: _FileIdentity,
    evidence: Sequence[_FileIdentity],
) -> None:
    _verify_file_identity(source, label="source asset")
    _verify_file_identity(plan, label="mesh-topology plan")
    for item in evidence:
        _verify_file_identity(item, label="machine evidence")


def _verify_file_identity(identity: _FileIdentity, *, label: str) -> None:
    try:
        metadata = identity.path.lstat()
    except FileNotFoundError:
        _fail("input_changed", f"{label} disappeared: {identity.path}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        _fail("input_changed", f"{label} is no longer regular: {identity.path}")
    if _stat_identity(metadata) != identity.stat_identity:
        _fail("input_changed", f"{label} filesystem identity changed: {identity.path}")
    if _file_sha256(identity.path) != identity.sha256:
        _fail("input_changed", f"{label} bytes changed: {identity.path}")


def _publish_bundle(
    *,
    bundle_dir: Path,
    output_root: Path,
    bundle_sha256: str,
    output_name: str,
    expected_output_sha256: str,
    expected_receipt: bytes,
    precommit_validator: Any,
) -> tuple[Path, bool]:
    final_bundle = output_root / bundle_sha256
    precommit_validator()
    try:
        os.rename(bundle_dir, final_bundle)
        return final_bundle, False
    except FileExistsError:
        pass
    except OSError as exc:
        if exc.errno not in {17, 39}:
            raise
    _verify_existing_bundle(
        bundle=final_bundle,
        output_name=output_name,
        expected_output_sha256=expected_output_sha256,
        expected_receipt=expected_receipt,
    )
    return final_bundle, True


def _verify_existing_bundle(
    *,
    bundle: Path,
    output_name: str,
    expected_output_sha256: str,
    expected_receipt: bytes,
) -> None:
    if bundle.is_symlink() or not bundle.is_dir():
        _fail("publication_collision", f"existing bundle is unsafe: {bundle}")
    entries = sorted(path.name for path in bundle.iterdir())
    if entries != sorted(["receipt.json", output_name]):
        _fail("publication_collision", f"existing bundle inventory differs: {bundle}")
    output = bundle / output_name
    receipt = bundle / "receipt.json"
    for path in (output, receipt):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail("publication_collision", f"existing bundle file is unsafe: {path}")
    if _file_sha256(output) != expected_output_sha256:
        _fail("publication_collision", f"existing output bytes differ: {output}")
    if receipt.read_bytes() != expected_receipt:
        _fail("publication_collision", f"existing receipt bytes differ: {receipt}")


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_value_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _exact_array_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        if len(left) != len(right):
            return False
    except TypeError:
        return _exact_value_equal(left, right)
    return all(
        _exact_value_equal(first, second)
        for first, second in zip(left, right, strict=True)
    )


def _usd_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _usd_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_usd_value(item) for item in value]
    if hasattr(value, "path") and hasattr(value, "resolvedPath"):
        return {
            "path": str(value.path),
            "resolved_path": str(value.resolvedPath),
        }
    try:
        return [_usd_value(item) for item in value]
    except TypeError:
        return str(value)


def _fail(code: str, detail: str) -> NoReturn:
    raise _Blocked(f"{code}: {detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Author an exact-plan Gate 3A mesh-topology USDZ derivative."
    )
    parser.add_argument("asset_path", type=Path)
    parser.add_argument("plan_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = author_gate3a_mesh_topology_derivative(
        asset_path=args.asset_path,
        plan_path=args.plan_path,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            result.report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0 if result.passed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
