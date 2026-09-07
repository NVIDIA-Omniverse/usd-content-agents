# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic state transitions for the composed single-asset workflow."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import shlex
import signal
import stat
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from filelock import FileLock
from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryAuthoringParameterValue,
    GeometryAuthoringReference,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    GeometryRepresentationBinding,
    GeometrySourceBundle,
    geometry_authoring_request_digest,
    resolve_semantic_parameter_overrides,
    validate_returned_parameter_state,
)
from pydantic import ValidationError
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_regular_file_no_follow,
)

from content_agent_workflows.common.artifacts import (
    _directory_chain_matches,
    _open_directory_no_symlinks,
    _stable_ctime_ns,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    load_json,
)
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION,
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
    DomainExecutionContext,
    DomainName,
    EmbeddedStageBinding,
    ExecutionArtifactBinding,
    domain_execution_context_from_metadata,
)
from content_agent_workflows.common.embedded_domain_decision import (
    ContractArtifactReference,
    EmbeddedDecisionIdentity,
    NamedDecisionDigests,
)
from content_agent_workflows.common.validation_evidence import (
    VALIDATION_EVIDENCE_SCHEMA_VERSION,
    ValidationEvidence,
)

from .authoring_evidence import (
    load_external_source_bundle,
    select_source_usd,
    source_artifact_path,
    source_representation_path,
    validate_external_source_manifest,
)
from .catalog import (
    AssetLeafRuntimeBinding,
    resolve_repository_asset_leaf_catalog,
)
from .models import (
    ASSET_CAD_MODELING_STAGE_RESULT_SCHEMA_VERSION,
    ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
    ASSET_EXECUTION_GRAPH_SCHEMA_VERSION,
    ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION,
    ASSET_LEAF_CATALOG_SCHEMA_VERSION,
    ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
    ASSET_REQUEST_SCHEMA_VERSION,
    ASSET_TERMINAL_VALIDATION_SCHEMA_VERSION,
    CAD_MODELING_STAGE_ORDER,
    COMPATIBILITY_FIXED_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
    GEOMETRY_STAGE_ORDER,
    LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
    LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION,
    LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION,
    LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION,
    LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
    LEGACY_STAGE_ORDER,
    ArtifactBinding,
    AssetCadModelingStageResult,
    AssetCombinedReport,
    AssetCompositionRun,
    AssetCoordinatorEvidenceReview,
    AssetCoordinatorPlan,
    AssetCoordinatorPlanDraft,
    AssetCoordinatorReviewDraft,
    AssetCoordinatorState,
    AssetCrossStageValidation,
    AssetExecutionGraph,
    AssetGeometryStageResult,
    AssetGraphTerminalReceipt,
    AssetLeafCatalog,
    AssetLeafProjection,
    AssetLeafReceipt,
    AssetLeafState,
    AssetLeafTransition,
    AssetRunRequest,
    AssetSegmentationRunBinding,
    AssetStageHandoff,
    AssetTerminalValidation,
    CoordinatorMode,
    CrossStageHandoffName,
    LegacyAssetCadModelingRequest,
    PhysicsValidationMode,
    StageName,
    StageState,
    StageStatus,
    StageTransition,
    SupersededArticulationReview,
    SupersededStageAttempt,
)

if TYPE_CHECKING:
    from content_agent_workflows.articulation import (
        EmbeddedArticulationGraphRevision,
        EmbeddedArticulationHumanAcceptance,
    )
    from content_agent_workflows.geometry.rendering import GeometryRenderEvidence


class AssetCompositionStateError(RuntimeError):
    """Raised when coordinator state or a requested transition is invalid."""


class _CadProductJobInterrupted(SystemExit):
    """Exit after a termination signal once the owned CAD tree is gone."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        super().__init__(128 + signal_number)


class _AttemptLeaseBusyError(RuntimeError):
    """The immutable stage attempt is already locked by another executor."""


class _PersistentAttemptFileLock:
    """Hold a POSIX advisory lock without unlinking its stable lock inode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    @property
    def is_locked(self) -> bool:
        return self._descriptor is not None

    def acquire(self, *, timeout: int = 0) -> None:
        if timeout != 0:
            raise ValueError(
                "attempt execution leases only support non-blocking acquisition"
            )
        if self.is_locked:
            raise RuntimeError("attempt execution lease is already acquired")
        if os.name != "posix":
            raise OSError(
                errno.ENOTSUP, "persistent attempt leases require POSIX flock"
            )

        import fcntl

        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise OSError(errno.ENOTSUP, "persistent attempt leases require O_NOFOLLOW")
        flags = os.O_RDWR | os.O_CREAT | no_follow
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = os.stat(self.path, follow_symlinks=False)
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or descriptor_stat.st_nlink != 1
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise OSError(
                    errno.EINVAL, "attempt lease path is not a stable regular file"
                )
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise _AttemptLeaseBusyError from exc
                raise
            locked_path_stat = os.stat(self.path, follow_symlinks=False)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
                locked_path_stat.st_dev,
                locked_path_stat.st_ino,
            ):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                raise OSError(
                    errno.EINVAL, "attempt lease path changed during acquisition"
                )
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass
class _CadAttemptExecutionLease:
    """OS-backed ownership for one deterministic stage attempt."""

    lock: _PersistentAttemptFileLock
    record_path: Path
    token: str
    run_id: str
    stage_attempt: int
    stage: Literal["cad_modeling", "geometry"]

    def require_owner(
        self,
        *,
        run_id: str,
        stage_attempt: int,
        stage: Literal["cad_modeling", "geometry"] | None = None,
    ) -> None:
        expected_stage = stage or self.stage
        if (
            not self.lock.is_locked
            or run_id != self.run_id
            or stage_attempt != self.stage_attempt
            or expected_stage != self.stage
        ):
            raise AssetCompositionStateError(
                f"{self.stage.replace('_', ' ').title()} execution lease does not "
                "own this stage attempt"
            )
        try:
            record = load_json(self.record_path)
        except (OSError, ValueError) as exc:
            raise AssetCompositionStateError(
                f"{self.stage.replace('_', ' ').title()} execution lease record "
                "is unavailable"
            ) from exc
        if not isinstance(record, dict) or (
            record.get("token") != self.token
            or record.get("run_id") != self.run_id
            or record.get("stage_attempt") != self.stage_attempt
            or record.get("stage") != self.stage
        ):
            raise AssetCompositionStateError(
                f"{self.stage.replace('_', ' ').title()} execution lease ownership "
                "changed"
            )

    def release(self) -> None:
        try:
            try:
                record = load_json(self.record_path)
            except (OSError, ValueError):
                record = None
            if isinstance(record, dict) and record.get("token") == self.token:
                try:
                    self.record_path.unlink(missing_ok=True)
                except OSError:
                    pass
        finally:
            self.lock.release()


class _BindingLike(Protocol):
    path: str
    sha256: str


def _same_binding(left: _BindingLike, right: _BindingLike) -> bool:
    return left.path == right.path and left.sha256 == right.sha256


@dataclass(frozen=True)
class _EmbeddedDomainContextSnapshot:
    context: DomainExecutionContext
    plan_binding: ArtifactBinding
    plan: AssetCoordinatorPlan


@dataclass(frozen=True)
class _VerifiedRunSnapshot:
    run: AssetCompositionRun
    coordinator_plans: Mapping[str, AssetCoordinatorPlan]


_VERIFIED_BINDINGS: ContextVar[dict[tuple[str, str | None], ArtifactBinding] | None] = (
    ContextVar("asset_composition_verified_bindings", default=None)
)
_VERIFIED_DEPENDENCIES: ContextVar[
    dict[tuple[str, str | None], list[ArtifactBinding]] | None
] = ContextVar("asset_composition_verified_dependencies", default=None)
_VERIFIED_COORDINATOR_PLANS: ContextVar[dict[str, AssetCoordinatorPlan] | None] = (
    ContextVar("asset_composition_verified_coordinator_plans", default=None)
)
_ACCEPTANCE_VALIDATED_STAGES = frozenset(
    {
        "cad_modeling",
        "geometry",
        "articulation",
        "material",
        "texture",
        "physics",
        "validation",
        "finalization",
    }
)
_CAD_SOURCE_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
_GEOMETRY_AUTHORING_COMMAND_ENV = "CONTENT_AGENT_GEOMETRY_AUTHORING_COMMAND"
_CAD_PROCESS_TERMINATION_GRACE_SECONDS = 5.0
_CAD_PROCESS_KILL_GRACE_SECONDS = 5.0
_CAD_PROCESS_GROUP_POLL_SECONDS = 0.02


def _final_usdz_validator() -> Any:
    """Resolve the required package validator at run admission or finalization."""

    try:
        from world_understanding.utils.usd.package import validate_usdz_package_layout
    except ImportError as exc:
        raise AssetCompositionStateError(
            "The required shared USDZ package validator is unavailable"
        ) from exc
    return validate_usdz_package_layout


def _validate_final_usdz(path: Path) -> None:
    """Validate the final package through the required shared USD boundary."""

    try:
        _final_usdz_validator()(path)
    except Exception as exc:  # noqa: BLE001 - normalize the package boundary
        raise AssetCompositionStateError(
            f"Final asset is not a canonical USDZ package: {exc}"
        ) from exc


def _usd_dependency_paths(path: Path) -> list[Path]:
    """Return external files in a resolved USD dependency closure."""

    if path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        return []

    from pxr import Ar, UsdUtils

    try:
        _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001 - normalize the USD boundary
        raise AssetCompositionStateError(
            f"Could not inspect {path.name} dependency closure: {exc}"
        ) from exc
    unresolved_paths = sorted({str(item) for item in unresolved})
    if unresolved_paths:
        sample = ", ".join(unresolved_paths[:8])
        suffix = "" if len(unresolved_paths) <= 8 else ", ..."
        raise AssetCompositionStateError(
            f"{path.name} has unresolved USD dependencies: {sample}{suffix}"
        )

    root = path.resolve()
    dependencies: set[Path] = set()

    def add_dependency(raw: object) -> None:
        text = str(raw)
        if not text:
            return
        package_root = text
        while Ar.IsPackageRelativePath(package_root):
            package_root, _member = Ar.SplitPackageRelativePathOuter(package_root)
        candidate = Path(package_root).expanduser()
        if not candidate.is_absolute():
            raise AssetCompositionStateError(
                f"{path.name} has a non-resolved USD dependency: {text}"
            )
        resolved = candidate.resolve()
        if resolved != root:
            dependencies.add(resolved)

    for layer in _layers:
        if layer.anonymous:
            continue
        add_dependency(layer.resolvedPath or layer.realPath or layer.identifier)
    for asset in _assets:
        add_dependency(asset)
    return sorted(dependencies, key=str)


def _validate_physics_runtime_evidence(
    evidence: Sequence[ArtifactBinding],
    *,
    output: ArtifactBinding,
    validation_mode: PhysicsValidationMode,
) -> None:
    """Enforce the frozen Physics evidence policy before handoff."""

    native_evidence: list[tuple[Path, ValidationEvidence]] = []
    for binding in evidence:
        evidence_path = Path(binding.path)
        if evidence_path.suffix.lower() != ".json":
            continue
        payload = _json_object_from_binding(
            binding,
            label=f"Physics evidence {evidence_path.name}",
        )
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == VALIDATION_EVIDENCE_SCHEMA_VERSION
        ):
            try:
                parsed = ValidationEvidence.model_validate(payload)
            except ValidationError as exc:
                raise AssetCompositionStateError(
                    f"Physics native validation evidence is invalid: {exc}"
                ) from exc
            native_evidence.append((evidence_path, parsed))

    if not native_evidence:
        raise AssetCompositionStateError(
            "Physics completion requires native validation evidence with schema "
            f"{VALIDATION_EVIDENCE_SCHEMA_VERSION}"
        )

    for evidence_path, native_payload in native_evidence:
        if native_payload.workflow != "physics_authoring":
            raise AssetCompositionStateError(
                "Physics native validation evidence has the wrong workflow: "
                f"{native_payload.workflow}"
            )
        if Path(native_payload.asset).expanduser().resolve() != Path(output.path):
            raise AssetCompositionStateError(
                "Physics native validation evidence is not bound to the accepted "
                f"output: {native_payload.asset}"
            )
        if native_payload.metadata.get("asset_sha256") != output.sha256:
            raise AssetCompositionStateError(
                "Physics native validation evidence has the wrong accepted-output "
                "digest"
            )
        if not native_payload.checks:
            raise AssetCompositionStateError(
                f"Physics native validation evidence has no checks: {evidence_path.name}"
            )
        if validation_mode == "runtime_required":
            incomplete = [
                check.name for check in native_payload.checks if check.status != "pass"
            ]
        else:
            properties = [
                check
                for check in native_payload.checks
                if check.name == "physics_properties"
            ]
            if len(properties) != 1 or properties[0].status != "pass":
                raise AssetCompositionStateError(
                    "Physics schema-readback acceptance requires one passing "
                    "physics_properties check"
                )
            incomplete = [
                check.name for check in native_payload.checks if check.status == "fail"
            ]
        status = native_payload.sim_ready_status
        if status == "fail" or (
            validation_mode == "runtime_required" and status != "pass"
        ):
            raise AssetCompositionStateError(
                "Physics native runtime validation must pass; "
                f"{evidence_path.name} reported {status!r}"
            )
        if incomplete:
            raise AssetCompositionStateError(
                "Physics native validation checks violate the frozen acceptance "
                "mode; incomplete checks: " + ", ".join(incomplete)
            )


def _validate_articulated_physics_output(path: Path) -> None:
    """Reject ambiguous or non-functional moving-joint rigid-body ownership."""

    from pxr import Usd
    from world_understanding.functions.physics.physics_topology import (
        inspect_physics_topology,
    )

    try:
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {path}")
        instance_proxies = sorted(
            str(prim.GetPath())
            for prim in stage.Traverse(Usd.TraverseInstanceProxies())
            if prim.IsInstanceProxy()
        )
        if instance_proxies:
            raise AssetCompositionStateError(
                "Physics output contains instance proxies that cannot be authored; "
                "de-instance the source offline before composition: "
                + ", ".join(instance_proxies[:8])
            )
        topology = inspect_physics_topology(path)
    except AssetCompositionStateError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize the USD boundary
        raise AssetCompositionStateError(
            f"Could not inspect Physics topology in {path.name}: {exc}"
        ) from exc

    nested = [
        str(finding.get("prim_path") or "<unknown>")
        for finding in topology.get("findings", [])
        if finding.get("code") == "nested_enabled_rigid_body"
    ]
    if nested:
        raise AssetCompositionStateError(
            "Physics output contains nested enabled rigid bodies: "
            + ", ".join(sorted(nested))
        )

    rigid_bodies = set(topology.get("rigid_body_paths", []))
    for joint in topology.get("joints", []):
        if not joint.get("enabled", True) or joint.get("is_fixed_joint"):
            continue
        joint_path = str(joint.get("prim_path") or "<unknown>")
        endpoints: list[str] = []
        for endpoint_name in ("body0_targets", "body1_targets"):
            targets = joint.get(endpoint_name) or []
            if not isinstance(targets, list) or len(targets) > 1:
                raise AssetCompositionStateError(
                    f"Physics joint {joint_path} has ambiguous {endpoint_name}"
                )
            if targets:
                target = str(targets[0])
                if target not in rigid_bodies:
                    raise AssetCompositionStateError(
                        f"Physics joint {joint_path} {endpoint_name} must target "
                        f"an exact enabled rigid-body prim; got {target}"
                    )
                endpoints.append(target)
        if not endpoints:
            raise AssetCompositionStateError(
                f"Physics joint {joint_path} has no rigid-body endpoint"
            )
        if len(endpoints) == 2 and endpoints[0] == endpoints[1]:
            raise AssetCompositionStateError(
                f"Physics joint {joint_path} connects the same rigid body twice: "
                f"{endpoints[0]}"
            )


def _joint_graph_signature(path: Path, *, label: str) -> tuple[str, ...]:
    """Read back the exact authored joint graph needed for preservation checks."""

    from pxr import Usd, UsdPhysics

    try:
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {path}")
        joint_records: list[dict[str, object]] = []
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            attributes = []
            for attribute in prim.GetAttributes():
                time_samples = attribute.GetTimeSamples()
                attributes.append(
                    {
                        "name": attribute.GetName(),
                        "type": str(attribute.GetTypeName()),
                        "default": _canonical_joint_graph_value(attribute.Get()),
                        "time_samples": [
                            [
                                _canonical_joint_graph_value(float(time)),
                                _canonical_joint_graph_value(
                                    attribute.Get(Usd.TimeCode(time))
                                ),
                            ]
                            for time in time_samples
                        ],
                        "connections": sorted(
                            str(item) for item in attribute.GetConnections()
                        ),
                    }
                )
            relationships = [
                {
                    "name": relationship.GetName(),
                    "targets": sorted(str(item) for item in relationship.GetTargets()),
                }
                for relationship in prim.GetRelationships()
            ]
            joint_records.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "joint_type": prim.GetTypeName(),
                    "applied_schemas": sorted(prim.GetAppliedSchemas()),
                    "attributes": sorted(
                        attributes, key=lambda item: str(item["name"])
                    ),
                    "relationships": sorted(
                        relationships,
                        key=lambda item: str(item["name"]),
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001 - normalize the USD boundary
        raise AssetCompositionStateError(
            f"Could not inspect {label} Joint graph in {path.name}: {exc}"
        ) from exc
    return tuple(
        sorted(
            json.dumps(
                joint,
                sort_keys=True,
                separators=(",", ":"),
            )
            for joint in joint_records
        )
    )


def _canonical_joint_graph_value(value: Any) -> object:
    """Project one authored USD value into stable, exact JSON data."""

    from pxr import Gf, Sdf

    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, Sdf.AssetPath):
        # Only the authored path is part of the USD value. The resolved path is
        # process- and resolver-context-dependent and must not enter identity.
        return ["asset_path", value.path]
    if isinstance(value, Sdf.Path):
        return ["path", str(value)]
    if isinstance(value, Sdf.TimeCode):
        return [
            "time_code",
            _canonical_joint_graph_value(float(cast(Any, value))),
        ]
    if isinstance(value, Gf.Quatd | Gf.Quatf | Gf.Quath):
        quaternion = cast(Any, value)
        return [
            "quaternion",
            _canonical_joint_graph_value(float(quaternion.GetReal())),
            _canonical_joint_graph_value(tuple(quaternion.GetImaginary())),
        ]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Mapping):
        entries = [
            [
                _canonical_joint_graph_value(key),
                _canonical_joint_graph_value(item),
            ]
            for key, item in value.items()
        ]
        return [
            "mapping",
            sorted(
                entries,
                key=lambda entry: json.dumps(
                    entry[0],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        ]
    try:
        items = list(value)
    except TypeError as exc:
        raise AssetCompositionStateError(
            "Joint graph contains an unsupported authored USD value type: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        ) from exc
    return ["sequence", [_canonical_joint_graph_value(item) for item in items]]


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _run_root(state_path: Path) -> Path:
    return state_path.parent.resolve()


def _attempt_lease_path(
    state_path: Path,
    *,
    run_id: str,
    stage: Literal["cad_modeling", "geometry"],
    stage_attempt: int,
) -> Path:
    """Keep executor ownership outside the child-writable composed run."""

    # There is one bounded file per immutable attempt. Retention tooling may
    # remove it only after the corresponding run can no longer execute.
    resolved = state_path.expanduser().resolve()
    run_dir = resolved.parent
    identity = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    return run_dir.parent / (
        f".{run_dir.name}.{identity}.{stage.replace('_', '-')}-attempt-"
        f"{stage_attempt}.lock"
    )


def _cad_attempt_lease_path(
    state_path: Path,
    *,
    stage_attempt: int,
    run_id: str | None = None,
) -> Path:
    """Return the stable CAD lease path; retained for operational tooling."""

    if run_id is None:
        payload = load_json(state_path)
        candidate = payload.get("run_id") if isinstance(payload, dict) else None
        if not isinstance(candidate, str) or not candidate:
            raise AssetCompositionStateError("CAD lease path requires a valid run ID")
        run_id = candidate
    return _attempt_lease_path(
        state_path,
        run_id=run_id,
        stage="cad_modeling",
        stage_attempt=stage_attempt,
    )


@contextmanager
def _cad_attempt_execution_lease(
    state_path: Path,
    *,
    run_id: str,
    stage_attempt: int,
) -> Iterator[_CadAttemptExecutionLease]:
    lease_path = _cad_attempt_lease_path(
        state_path,
        stage_attempt=stage_attempt,
        run_id=run_id,
    )
    lock = _PersistentAttemptFileLock(lease_path)
    try:
        lock.acquire(timeout=0)
    except _AttemptLeaseBusyError as exc:
        raise AssetCompositionStateError(
            "Another executor already owns this CAD modeling stage attempt"
        ) from exc
    except OSError as exc:
        raise AssetCompositionStateError(
            f"CAD modeling execution lease is unavailable: {lease_path}"
        ) from exc

    lease = _CadAttemptExecutionLease(
        lock=lock,
        record_path=lease_path.with_suffix(".json"),
        token=secrets.token_hex(32),
        run_id=run_id,
        stage_attempt=stage_attempt,
        stage="cad_modeling",
    )
    try:
        atomic_write_json(
            lease.record_path,
            {
                "schema_version": (
                    "content-agent-workflows.cad-attempt-execution-lease.v1"
                ),
                "run_id": run_id,
                "stage": lease.stage,
                "stage_attempt": stage_attempt,
                "token": lease.token,
                "pid": os.getpid(),
                "acquired_at": _timestamp(),
            },
        )
        lease.require_owner(run_id=run_id, stage_attempt=stage_attempt)
        yield lease
    finally:
        lease.release()


@contextmanager
def _geometry_attempt_execution_lease(
    state_path: Path,
    *,
    run_id: str,
    stage_attempt: int,
) -> Iterator[_CadAttemptExecutionLease]:
    lease_path = _attempt_lease_path(
        state_path,
        run_id=run_id,
        stage="geometry",
        stage_attempt=stage_attempt,
    )
    lock = _PersistentAttemptFileLock(lease_path)
    try:
        lock.acquire(timeout=0)
    except _AttemptLeaseBusyError as exc:
        raise AssetCompositionStateError(
            "Another executor already owns this Geometry stage attempt"
        ) from exc
    except OSError as exc:
        raise AssetCompositionStateError(
            f"Geometry execution lease is unavailable: {lease_path}"
        ) from exc
    lease = _CadAttemptExecutionLease(
        lock=lock,
        record_path=lease_path.with_suffix(".json"),
        token=secrets.token_hex(32),
        run_id=run_id,
        stage_attempt=stage_attempt,
        stage="geometry",
    )
    try:
        atomic_write_json(
            lease.record_path,
            {
                "schema_version": (
                    "content-agent-workflows.geometry-attempt-execution-lease.v1"
                ),
                "run_id": run_id,
                "stage": lease.stage,
                "stage_attempt": stage_attempt,
                "token": lease.token,
                "pid": os.getpid(),
                "acquired_at": _timestamp(),
            },
        )
        lease.require_owner(
            run_id=run_id,
            stage_attempt=stage_attempt,
            stage="geometry",
        )
        yield lease
    finally:
        lease.release()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_name(f".{path.name}.lock"))):
        yield


@contextmanager
def _open_stable_regular_file(
    path: str | Path,
    *,
    label: str,
) -> Iterator[tuple[Path, int, os.stat_result]]:
    """Pin one existing regular file through a no-symlink directory walk."""

    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if os.name == "nt":
        try:
            with open_regular_file_no_follow(candidate) as (stream, before):
                current = candidate.lstat()
                identity = (before.st_dev, before.st_ino)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise AssetCompositionStateError(
                        f"{label} must be a regular non-symlink file: {candidate}"
                    )

                yield candidate, stream.fileno(), before

                after = os.fstat(stream.fileno())
                current = candidate.lstat()
                if (
                    not stat.S_ISREG(after.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or (after.st_dev, after.st_ino) != identity
                    or (current.st_dev, current.st_ino) != identity
                    or after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns
                    or _stable_ctime_ns(after) != _stable_ctime_ns(before)
                ):
                    raise AssetCompositionStateError(
                        f"{label} changed while being read"
                    )
        except AssetCompositionStateError:
            raise
        except (ArtifactPathError, OSError) as exc:
            raise AssetCompositionStateError(
                f"Could not safely read {label}: {candidate}"
            ) from exc
        return

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise AssetCompositionStateError(
            f"Cannot safely open {label}: O_NOFOLLOW is unavailable"
        )
    try:
        parent_fd, parent_chain = _open_directory_no_symlinks(
            candidate.parent,
            create_missing=False,
        )
    except OSError as exc:
        raise AssetCompositionStateError(
            f"{label} parent is unavailable or uses a symlink: {candidate.parent}"
        ) from exc
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(candidate.name, flags, dir_fd=parent_fd)
        before = os.fstat(descriptor)
        current = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
        ):
            raise AssetCompositionStateError(
                f"{label} must be a regular non-symlink file: {candidate}"
            )

        yield candidate, descriptor, before

        after = os.fstat(descriptor)
        current = os.stat(candidate.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not _directory_chain_matches(candidate.parent, parent_chain)
            or not stat.S_ISREG(after.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (after.st_dev, after.st_ino) != identity
            or (current.st_dev, current.st_ino) != identity
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or _stable_ctime_ns(after) != _stable_ctime_ns(before)
        ):
            raise AssetCompositionStateError(f"{label} changed while being read")
    except AssetCompositionStateError:
        raise
    except OSError as exc:
        raise AssetCompositionStateError(
            f"Could not safely read {label}: {candidate}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _binding(
    path: str | Path,
    *,
    label: str,
    required_root: Path | None = None,
) -> ArtifactBinding:
    digest = hashlib.sha256()
    size_bytes = 0
    with _open_stable_regular_file(path, label=label) as (
        resolved,
        descriptor,
        metadata,
    ):
        if required_root is not None:
            root = required_root.resolve()
            if not resolved.is_relative_to(root):
                raise AssetCompositionStateError(
                    f"{label} must be inside the composed run directory: {resolved}"
                )
            if metadata.st_nlink != 1:
                raise AssetCompositionStateError(
                    f"{label} must not be hard-linked: {resolved}"
                )
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
        if size_bytes != metadata.st_size:
            raise AssetCompositionStateError(f"{label} changed while being read")
    return ArtifactBinding(
        path=str(resolved),
        sha256=digest.hexdigest(),
        size_bytes=size_bytes,
    )


def _verify_binding(
    binding: ArtifactBinding,
    *,
    label: str,
    required_root: Path | None = None,
) -> None:
    cache = _VERIFIED_BINDINGS.get()
    cache_key = (
        binding.path,
        str(required_root.resolve()) if required_root is not None else None,
    )
    current = cache.get(cache_key) if cache is not None else None
    if current is None:
        current = _binding(
            binding.path,
            label=label,
            required_root=required_root,
        )
        if cache is not None:
            cache[cache_key] = current
    if current != binding:
        raise AssetCompositionStateError(
            f"{label} identity changed: expected {binding.sha256}, got {current.sha256}"
        )


def _dependency_bindings(
    path: Path,
    *,
    label: str,
    required_root: Path | None = None,
) -> list[ArtifactBinding]:
    return [
        _binding(
            dependency,
            label=f"{label} dependency {index + 1}",
            required_root=required_root,
        )
        for index, dependency in enumerate(_usd_dependency_paths(path))
    ]


def _verify_dependency_bindings(
    path: Path,
    dependencies: Sequence[ArtifactBinding],
    *,
    label: str,
    required_root: Path | None = None,
) -> None:
    cache = _VERIFIED_DEPENDENCIES.get()
    cache_key = (
        str(path.expanduser().resolve()),
        str(required_root.resolve()) if required_root is not None else None,
    )
    current = cache.get(cache_key) if cache is not None else None
    if current is None:
        if path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
            dependency_paths = tuple(str(item) for item in _usd_dependency_paths(path))
            if dependency_paths != tuple(binding.path for binding in dependencies):
                raise AssetCompositionStateError(
                    f"{label} dependency closure identity changed"
                )
        elif len({binding.path for binding in dependencies}) != len(dependencies):
            raise AssetCompositionStateError(
                f"{label} dependency closure contains duplicate paths"
            )
        current = list(dependencies)
        for index, binding in enumerate(current, start=1):
            try:
                _verify_binding(
                    binding,
                    label=f"{label} dependency {index}",
                    required_root=required_root,
                )
            except AssetCompositionStateError as exc:
                raise AssetCompositionStateError(
                    f"{label} dependency closure identity changed: {exc}"
                ) from exc
        if cache is not None:
            cache[cache_key] = current
    if current != list(dependencies):
        raise AssetCompositionStateError(f"{label} dependency closure identity changed")


def bind_usd_dependency_closure(path: str | Path) -> list[ArtifactBinding]:
    """Bind every external file in one resolved USD dependency closure."""

    resolved = Path(path).expanduser().resolve()
    return _dependency_bindings(resolved, label=f"{resolved.name} USD asset")


def verify_usd_dependency_closure(
    path: str | Path,
    dependencies: Sequence[ArtifactBinding],
) -> None:
    """Fail closed when a previously bound USD dependency closure changes."""

    resolved = Path(path).expanduser().resolve()
    _verify_dependency_bindings(
        resolved,
        dependencies,
        label=f"{resolved.name} USD asset",
    )


def _verify_frozen_input_binding(
    path: str,
    binding: ArtifactBinding,
    *,
    label: str,
) -> None:
    if str(Path(path).expanduser().resolve()) != binding.path:
        raise AssetCompositionStateError(
            f"{label} path does not match its frozen binding"
        )
    _verify_binding(binding, label=label)


def verify_frozen_asset_inputs(
    request: AssetRunRequest,
    *,
    run_source: ArtifactBinding,
    run_source_dependencies: Sequence[ArtifactBinding] | None = None,
) -> None:
    """Revalidate all external inputs named by one frozen asset request."""

    if request.source_asset != run_source.path:
        raise AssetCompositionStateError(
            "Frozen request source asset does not match durable state"
        )
    if request.source_staging is not None:
        staging = request.source_staging
        run_root = Path(request.run_dir).expanduser().resolve()
        if staging.staged_source.path != request.source_asset:
            raise AssetCompositionStateError(
                "Frozen request source does not match its staged source identity"
            )
        staged_paths = [binding.path for binding in staging.staged_dependencies]
        if len(staged_paths) != len(set(staged_paths)):
            raise AssetCompositionStateError(
                "Frozen staged source dependency paths are not unique"
            )
        if run_source_dependencies is None:
            # Direct callers that have not just verified durable run state must
            # still hash the complete staged closure themselves.
            for index, binding in enumerate(
                staging.staged_dependencies,
                start=1,
            ):
                _verify_binding(
                    binding,
                    label=f"staged source artifact {index}",
                    required_root=run_root,
                )
        elif staging.staged_source != run_source or sorted(
            staging.staged_dependencies,
            key=lambda binding: binding.path,
        ) != sorted(
            [run_source, *run_source_dependencies],
            key=lambda binding: binding.path,
        ):
            # ``load_verified_run`` has already re-hashed this exact source and
            # dependency closure. Its USD traversal is root-first while the
            # staging manifest is path-sorted, so compare the exact identities
            # independent of those two legitimate orderings. Reuse that attestation
            # instead of reading a potentially multi-gigabyte closure a second time
            # on every coordinator tool call.
            raise AssetCompositionStateError(
                "Frozen staged source closure differs from durable state"
            )
        _verify_binding(
            staging.manifest,
            label="staged source manifest",
            required_root=run_root,
        )
        digest_payload = json.dumps(
            sorted(binding.sha256 for binding in staging.staged_dependencies),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        observed_digest_set = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
        if observed_digest_set != staging.dependency_digest_set_sha256:
            raise AssetCompositionStateError(
                "Frozen staged source dependency digest set changed"
            )
        try:
            manifest_payload = load_json(Path(staging.manifest.path))
        except Exception as exc:  # noqa: BLE001 - normalize the custody boundary
            raise AssetCompositionStateError(
                f"Frozen staged source manifest is invalid: {exc}"
            ) from exc
        if not isinstance(manifest_payload, dict):
            raise AssetCompositionStateError(
                "Frozen staged source manifest is not a JSON object"
            )
        manifest_files = manifest_payload.get("files")
        if not isinstance(manifest_files, list):
            raise AssetCompositionStateError(
                "Frozen staged source manifest omitted dependency files"
            )
        expected_original = [
            (binding.path, binding.sha256, binding.size_bytes)
            for binding in staging.original_dependencies
        ]
        expected_staged = [
            (binding.path, binding.sha256, binding.size_bytes)
            for binding in staging.staged_dependencies
        ]
        observed_original: list[tuple[str, str, int]] = []
        observed_staged: list[tuple[str, str, int]] = []
        observed_relative: list[str | None] = []
        try:
            for item in manifest_files:
                if not isinstance(item, dict):
                    raise TypeError("manifest file entry is not an object")
                digest = str(item["sha256"])
                size_bytes = int(item["size_bytes"])
                observed_original.append((str(item["source_path"]), digest, size_bytes))
                observed_staged.append((str(item["staged_path"]), digest, size_bytes))
                relative_value = item.get("relative_path")
                observed_relative.append(
                    str(relative_value) if relative_value is not None else None
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise AssetCompositionStateError(
                "Frozen staged source manifest has malformed dependency identity"
            ) from exc
        manifest_source_path = manifest_payload.get(
            "source_path",
            manifest_payload.get("source_usd_path"),
        )
        manifest_staged_path = manifest_payload.get(
            "staged_path",
            manifest_payload.get("staged_usd_path"),
        )
        closure_metadata_changed = False
        if (
            manifest_payload.get("schema_version")
            == "content-workflow-cli.immutable-source-closure.v1"
        ):
            staged_root = run_root / "inputs" / "asset_source"
            try:
                expected_relative = [
                    Path(binding.path).relative_to(staged_root).as_posix()
                    for binding in staging.staged_dependencies
                ]
            except ValueError:
                closure_metadata_changed = True
            else:
                self_containment = manifest_payload.get("self_containment")
                closure_metadata_changed = (
                    observed_relative != expected_relative
                    or manifest_payload.get("file_count") != len(expected_staged)
                    or manifest_payload.get("total_size_bytes")
                    != sum(
                        binding.size_bytes for binding in staging.staged_dependencies
                    )
                    or not isinstance(
                        manifest_payload.get("dependency_discovery_strategy"),
                        str,
                    )
                    or not str(
                        manifest_payload.get("dependency_discovery_strategy", "")
                    ).strip()
                    or self_containment != {"status": "verified", "escaped_paths": []}
                )
        if (
            observed_original != expected_original
            or observed_staged != expected_staged
            or manifest_source_path != staging.original_source.path
            or manifest_payload.get("source_sha256") != staging.original_source.sha256
            or manifest_staged_path != staging.staged_source.path
            or manifest_payload.get("dependency_digest_set_sha256")
            != staging.dependency_digest_set_sha256
            or manifest_payload.get("unresolved_dependencies") != []
            or closure_metadata_changed
        ):
            raise AssetCompositionStateError(
                "Frozen staged source manifest identity changed"
            )
    reference_paths = [*request.reference_images, *request.reference_files]
    if reference_paths != [binding.path for binding in request.reference_bindings]:
        raise AssetCompositionStateError(
            "Reference paths do not match their frozen bindings"
        )
    for index, binding in enumerate(request.reference_bindings, start=1):
        _verify_frozen_input_binding(
            binding.path,
            binding,
            label=f"reference {index}",
        )
    if request.selected_mode == "agentic":
        return
    if (
        request.joint_config is None
        or request.joint_config_binding is None
        or request.materials_yaml is None
        or request.materials_yaml_binding is None
    ):
        raise AssetCompositionStateError(
            "Fixed compatibility request lacks Joint or Material identity"
        )
    expected: list[tuple[str, ArtifactBinding, str]] = [
        (request.joint_config, request.joint_config_binding, "Joint config"),
        (
            request.materials_yaml,
            request.materials_yaml_binding,
            "materials manifest",
        ),
    ]
    if (request.materials_usd is None) != (request.materials_usd_binding is None):
        raise AssetCompositionStateError(
            "Frozen request must provide both materials_usd and its binding"
        )
    if request.materials_usd is None:
        raise AssetCompositionStateError(
            "Frozen request predates the required materials library identity; "
            "start a new asset run so the library bytes can be bound before "
            "execution"
        )
    if request.materials_usd_binding is None:
        raise AssetCompositionStateError(
            "Frozen request materials library binding is missing"
        )
    expected.append(
        (
            request.materials_usd,
            request.materials_usd_binding,
            "materials library",
        )
    )
    verify_usd_dependency_closure(
        request.materials_usd,
        request.materials_usd_dependencies,
    )
    for input_path, binding, label in expected:
        _verify_frozen_input_binding(input_path, binding, label=label)

    if request.source_images != [
        binding.path for binding in request.source_image_bindings
    ]:
        raise AssetCompositionStateError(
            "CAD source image paths do not match their frozen bindings"
        )
    for index, binding in enumerate(request.source_image_bindings, start=1):
        _verify_frozen_input_binding(
            binding.path,
            binding,
            label=f"CAD source image {index}",
        )
    if request.geometry is not None and request.geometry.segmentation_run is not None:
        _verify_staged_segmentation_run(
            request.geometry.segmentation_run,
            run_root=Path(request.run_dir).expanduser().resolve(),
        )


def _verify_staged_segmentation_run(
    binding: AssetSegmentationRunBinding,
    *,
    run_root: Path,
) -> None:
    root = Path(binding.run_dir)
    expected_parent = (run_root / "inputs" / "geometry-segmentation").resolve()
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise AssetCompositionStateError(
            f"Frozen segmentation run is unavailable: {root}"
        ) from exc
    if resolved_root != root:
        raise AssetCompositionStateError(
            "Frozen segmentation run path must be canonical and must not traverse links"
        )
    try:
        resolved_root.relative_to(expected_parent)
    except ValueError as exc:
        raise AssetCompositionStateError(
            "Frozen segmentation run is not staged below the composed run inputs"
        ) from exc
    try:
        root_metadata = resolved_root.lstat()
    except OSError as exc:
        raise AssetCompositionStateError(
            f"Frozen segmentation run is unavailable: {root}"
        ) from exc
    if resolved_root.is_symlink() or not stat.S_ISDIR(root_metadata.st_mode):
        raise AssetCompositionStateError(
            f"Frozen segmentation run must be a regular directory: {resolved_root}"
        )

    observed: list[str] = []
    for current_text, directory_names, file_names in os.walk(
        resolved_root,
        followlinks=False,
    ):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            candidate = current / name
            metadata = candidate.lstat()
            if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise AssetCompositionStateError(
                    f"Frozen segmentation run contains an unsafe directory: {candidate}"
                )
        for name in file_names:
            candidate = current / name
            metadata = candidate.lstat()
            if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise AssetCompositionStateError(
                    f"Frozen segmentation run contains a non-regular file: {candidate}"
                )
            observed.append(str(candidate))

    observed.sort()
    expected = [item.path for item in binding.artifacts]
    if observed != expected:
        raise AssetCompositionStateError("Frozen segmentation run file closure changed")
    for index, artifact in enumerate(binding.artifacts, start=1):
        _verify_binding(
            artifact,
            label=f"frozen segmentation artifact {index}",
            required_root=run_root,
        )


def _normalize_historical_asset_request_payload(
    request_payload: dict[str, object],
) -> dict[str, object]:
    """Normalize one decoded pre-usd-cli request.v1 payload for validation."""

    request_payload.setdefault("coordinator_mode", "single_reasoning_loop")
    request_payload.setdefault("physics_validation_mode", "runtime_required")
    request_payload.setdefault("selected_mode", "compatibility_fixed")
    runtime = request_payload.get("runtime")
    if (
        isinstance(runtime, dict)
        and "scene_tool_timeout_seconds" not in runtime
        and "workbench_timeout_seconds" in runtime
    ):
        # Compatibility is read/migration-only. New requests never persist
        # these historical Content Workbench fields.
        runtime["scene_tool_timeout_seconds"] = runtime.pop("workbench_timeout_seconds")
        runtime.pop("workbench_url", None)
        runtime.pop("start_workbench", None)
        runtime.pop("keep_workbench", None)
    return request_payload


def load_verified_asset_request(
    path: str | Path,
    *,
    run: AssetCompositionRun | None = None,
) -> AssetRunRequest:
    """Load the exact frozen request and revalidate every external input."""

    state_path = _resolved(path)
    current = run or load_verified_run(state_path)
    raw = _read_stable_regular_bytes(
        Path(current.request.path),
        label="composed asset request",
    )
    if (
        len(raw) != current.request.size_bytes
        or hashlib.sha256(raw).hexdigest() != current.request.sha256
    ):
        raise AssetCompositionStateError("Composed asset request identity changed")
    try:
        request_payload = json.loads(raw.decode("utf-8"))
        request = AssetRunRequest.model_validate(
            _normalize_historical_asset_request_payload(request_payload)
            if isinstance(request_payload, dict)
            else request_payload
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Invalid frozen composed asset request: {exc}"
        ) from exc

    root = state_path.parent.resolve()
    expected_request_path = root / "request.json"
    if Path(current.request.path) != expected_request_path:
        raise AssetCompositionStateError(
            "Run state is not bound to the expected frozen request"
        )
    if Path(request.run_dir).expanduser().resolve() != root:
        raise AssetCompositionStateError(
            "Frozen request run_dir does not match durable state"
        )
    if Path(request.run_state).expanduser().resolve() != state_path:
        raise AssetCompositionStateError(
            "Frozen request run_state does not match durable state"
        )
    if request.run_id != current.run_id:
        raise AssetCompositionStateError(
            "Frozen request run_id does not match durable state"
        )
    if request.selected_mode != current.selected_mode:
        raise AssetCompositionStateError(
            "Frozen request selected_mode does not match durable state"
        )
    if request.source_asset != current.source_asset.path:
        raise AssetCompositionStateError(
            "Frozen request source asset does not match durable state"
        )
    if current.selected_mode == "compatibility_fixed":
        if (request.geometry is not None) != ("geometry" in current.stage_order):
            raise AssetCompositionStateError(
                "Frozen request Geometry policy does not match durable stage order"
            )
        if (request.cad_modeling is not None) != (
            "cad_modeling" in current.stage_order
        ):
            raise AssetCompositionStateError(
                "Frozen request CAD modeling policy does not match durable stage order"
            )
    verify_frozen_asset_inputs(
        request,
        run_source=current.source_asset,
        run_source_dependencies=current.source_dependencies,
    )
    return request


def _physics_validation_mode_from_run(
    run: AssetCompositionRun,
) -> PhysicsValidationMode:
    """Read the immutable Physics acceptance policy without recursing on state."""

    if run.coordinator.mode != "single_reasoning_loop":
        return "runtime_required"
    try:
        request = AssetRunRequest.model_validate(
            _normalize_historical_asset_request_payload(
                _json_object_from_binding(
                    run.request,
                    label="frozen asset request",
                )
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid frozen composed asset request: {exc}"
        ) from exc
    if request.coordinator_mode != run.coordinator.mode:
        raise AssetCompositionStateError(
            "Frozen request coordinator mode differs from durable state"
        )
    return request.physics_validation_mode


def _load_verified_transition_run(state_path: Path) -> AssetCompositionRun:
    """Verify state and frozen inputs at a single-loop mutation boundary."""

    run = load_verified_run(state_path)
    if run.coordinator.mode == "single_reasoning_loop":
        load_verified_asset_request(state_path, run=run)
    else:
        request_payload = _json_object_from_binding(
            run.request,
            label="composed asset request",
        )
        # Requests created before coordinator mode existed intentionally omit
        # this field; activate_single_reasoning_loop validates and injects it.
        requested_mode = request_payload.get("coordinator_mode")
        if requested_mode is not None and requested_mode != run.coordinator.mode:
            raise AssetCompositionStateError(
                "Frozen request coordinator mode differs from durable state"
            )
    return run


def _write_run(path: Path, run: AssetCompositionRun) -> AssetCompositionRun:
    payload = run.model_dump(mode="python")
    payload["revision"] = run.revision + 1
    try:
        updated = AssetCompositionRun.model_validate(payload)
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Refusing to persist an invalid composed asset transition: {exc}"
        ) from exc
    atomic_write_json(path, updated)
    return updated


def _read_stable_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one descriptor-pinned regular-file version."""

    with _open_stable_regular_file(path, label=label) as (
        _resolved,
        descriptor,
        metadata,
    ):
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        if len(payload) != metadata.st_size:
            raise AssetCompositionStateError(f"{label} changed while being read")
        return payload


def _read_stable_regular_prefix(path: Path, *, label: str, size: int) -> bytes:
    """Read at most ``size`` bytes while retaining descriptor identity checks."""

    if size < 0:
        raise ValueError("stable regular-file prefix size must be non-negative")
    with _open_stable_regular_file(path, label=label) as (
        _resolved_path,
        descriptor,
        _metadata,
    ):
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def load_run_state(path: str | Path) -> AssetCompositionRun:
    """Load typed coordinator state without accepting any artifact claims."""

    resolved = _resolved(path)
    try:
        return AssetCompositionRun.model_validate(load_json(resolved))
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Invalid composed asset run state at {resolved}: {exc}"
        ) from exc


def _load_coordinator_plan(binding: ArtifactBinding) -> AssetCoordinatorPlan:
    try:
        return AssetCoordinatorPlan.model_validate(
            _json_object_from_binding(binding, label="coordinator plan")
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid coordinator plan at {binding.path}: {exc}"
        ) from exc


def _load_coordinator_review(
    binding: ArtifactBinding,
) -> AssetCoordinatorEvidenceReview:
    try:
        return AssetCoordinatorEvidenceReview.model_validate(
            _json_object_from_binding(binding, label="coordinator evidence review")
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid coordinator evidence review at {binding.path}: {exc}"
        ) from exc


def _verify_coordinator_state(run: AssetCompositionRun, *, root: Path) -> None:
    coordinator = run.coordinator
    if coordinator.mode == "legacy":
        if coordinator.plan_revisions or coordinator.evidence_reviews:
            raise AssetCompositionStateError(
                "Legacy coordinator state cannot contain reasoning artifacts"
            )
        return

    prior_plan: ArtifactBinding | None = None
    plans: dict[str, AssetCoordinatorPlan] = {}
    for index, binding in enumerate(coordinator.plan_revisions, start=1):
        _verify_binding(
            binding,
            label=f"coordinator plan {index}",
            required_root=root,
        )
        plan = _load_coordinator_plan(binding)
        if (
            plan.plan_revision != index
            or plan.run_id != run.run_id
            or plan.request != run.request
            or plan.source_asset != run.source_asset
            or plan.prior_plan != prior_plan
        ):
            raise AssetCompositionStateError(
                f"Coordinator plan {index} is not bound to the expected plan chain"
            )
        for evidence_index, evidence in enumerate(plan.evidence, start=1):
            _verify_binding(
                evidence,
                label=f"coordinator plan {index} evidence {evidence_index}",
                required_root=root,
            )
        plans[binding.sha256] = plan
        verified_plans = _VERIFIED_COORDINATOR_PLANS.get()
        if verified_plans is not None:
            verified_plans[binding.sha256] = plan
        prior_plan = binding

    prior_review: ArtifactBinding | None = None
    for index, binding in enumerate(coordinator.evidence_reviews, start=1):
        _verify_binding(
            binding,
            label=f"coordinator evidence review {index}",
            required_root=root,
        )
        review = _load_coordinator_review(binding)
        if (
            review.review_index != index
            or review.run_id != run.run_id
            or review.request != run.request
            or review.prior_review != prior_review
            or review.plan.sha256 not in plans
        ):
            raise AssetCompositionStateError(
                f"Coordinator evidence review {index} is not bound to the expected chain"
            )
        _verify_binding(
            review.plan,
            label=f"coordinator evidence review {index} plan",
            required_root=root,
        )
        _verify_binding(review.input_asset, label=f"{review.stage} reviewed input")
        if review.output_asset is not None:
            _verify_binding(
                review.output_asset,
                label=f"{review.stage} reviewed output",
                required_root=root,
            )
            _verify_dependency_bindings(
                Path(review.output_asset.path),
                review.output_dependencies,
                label=f"{review.stage} reviewed output",
                required_root=root,
            )
        for evidence_index, evidence in enumerate(review.evidence, start=1):
            _verify_binding(
                evidence,
                label=f"coordinator review {index} evidence {evidence_index}",
                required_root=root,
            )
        if review.articulation_review_decisions is not None:
            _verify_binding(
                review.articulation_review_decisions,
                label=f"coordinator review {index} articulation decisions",
                required_root=root,
            )
        prior_review = binding


def _verify_superseded_attempt(
    attempt: SupersededStageAttempt,
    *,
    stage: StageName,
    index: int,
    root: Path,
) -> None:
    prefix = f"{stage} superseded attempt {index}"
    _verify_binding(
        attempt.reason_review,
        label=f"{prefix} reason review",
        required_root=root,
    )
    if attempt.input_asset is not None:
        _verify_binding(attempt.input_asset, label=f"{prefix} input")
        _verify_dependency_bindings(
            Path(attempt.input_asset.path),
            attempt.input_dependencies,
            label=f"{prefix} input",
        )
    if attempt.output_asset is not None:
        _verify_binding(
            attempt.output_asset,
            label=f"{prefix} output",
            required_root=root,
        )
        _verify_dependency_bindings(
            Path(attempt.output_asset.path),
            attempt.output_dependencies,
            label=f"{prefix} output",
            required_root=root,
        )
    for evidence_index, evidence in enumerate(attempt.evidence, start=1):
        _verify_binding(
            evidence,
            label=f"{prefix} evidence {evidence_index}",
            required_root=root,
        )
    for label, binding in (
        ("handoff", attempt.handoff),
        ("review candidates", attempt.review_candidates),
        ("review decisions", attempt.review_decisions),
    ):
        if binding is not None:
            _verify_binding(
                binding,
                label=f"{prefix} {label}",
                required_root=root,
            )
    if stage != "articulation" and attempt.superseded_reviews:
        raise AssetCompositionStateError(
            f"{prefix} cannot retain Articulation review revisions"
        )
    _verify_articulation_review_history(
        attempt.superseded_reviews,
        current_candidates=attempt.review_candidates,
        root=root,
        prefix=prefix,
    )


def _verify_superseded_articulation_review(
    review: SupersededArticulationReview,
    *,
    index: int,
    root: Path,
) -> EmbeddedArticulationGraphRevision:
    from content_agent_workflows.articulation import (
        EmbeddedArticulationCanonicalGraph,
        EmbeddedArticulationGraphRevision,
        EmbeddedArticulationGraphRevisionPatch,
        EmbeddedArticulationOuterReview,
        canonical_articulation_graph_changes,
        load_articulation_revision_human_decisions,
    )
    from content_agent_workflows.common.embedded_domain_decision import (
        AcceptedSemanticDecision,
        EmbeddedCoordinatorDecision,
        EmbeddedHumanDecision,
        accepted_semantic_decision_digest,
        artifact_reference,
        canonical_json_digest,
    )

    prefix = f"articulation superseded review {index}"
    for label, binding in (
        ("revision receipt", review.revision_receipt),
        ("review candidates", review.review_candidates),
        ("review decisions", review.review_decisions),
    ):
        _verify_binding(
            binding,
            label=f"{prefix} {label}",
            required_root=root,
        )
    try:
        record = EmbeddedArticulationGraphRevision.model_validate_json(
            Path(review.revision_receipt.path).read_bytes()
        )
    except (OSError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"{prefix} revision receipt is invalid: {exc}"
        ) from exc
    if not _same_binding(
        record.parent_canonical_graph,
        review.review_candidates,
    ) or (
        record.human_decisions.path != review.review_decisions.path
        or record.human_decisions.sha256 != review.review_decisions.sha256
        or record.human_decisions.size_bytes != review.review_decisions.size_bytes
    ):
        raise AssetCompositionStateError(
            f"{prefix} does not bind its exact prior human review"
        )
    record_bindings: list[tuple[str, _BindingLike]] = [
        ("parent graph", record.parent_canonical_graph),
        ("parent outer review", record.parent_outer_review),
        ("parent coordinator decision", record.parent_coordinator_decision),
        ("human decision", record.human_decision),
        ("revised graph", record.revised_canonical_graph),
        ("revised outer review", record.revised_outer_review),
        ("revised coordinator decision", record.revised_coordinator_decision),
    ]
    if record.parent_revision is not None:
        record_bindings.append(("parent revision", record.parent_revision))
    for label, record_binding in record_bindings:
        actual = _binding(
            record_binding.path,
            label=f"{prefix} {label}",
            required_root=root,
        )
        if actual.path != record_binding.path or actual.sha256 != record_binding.sha256:
            raise AssetCompositionStateError(f"{prefix} {label} identity changed")
    try:
        parent_graph = EmbeddedArticulationCanonicalGraph.model_validate_json(
            Path(record.parent_canonical_graph.path).read_bytes()
        )
        revised_graph = EmbeddedArticulationCanonicalGraph.model_validate_json(
            Path(record.revised_canonical_graph.path).read_bytes()
        )
        parent_outer = EmbeddedArticulationOuterReview.model_validate_json(
            Path(record.parent_outer_review.path).read_bytes()
        )
        revised_outer = EmbeddedArticulationOuterReview.model_validate_json(
            Path(record.revised_outer_review.path).read_bytes()
        )
        parent_decision = EmbeddedCoordinatorDecision.model_validate_json(
            Path(record.parent_coordinator_decision.path).read_bytes()
        )
        revised_decision = EmbeddedCoordinatorDecision.model_validate_json(
            Path(record.revised_coordinator_decision.path).read_bytes()
        )
        human = EmbeddedHumanDecision.model_validate_json(
            Path(record.human_decision.path).read_bytes()
        )
        changes = canonical_articulation_graph_changes(parent_graph, revised_graph)
        decisions = load_articulation_revision_human_decisions(
            record.human_decisions,
            graph=parent_graph,
        )
        reconstructed_patch = EmbeddedArticulationGraphRevisionPatch(
            expected_state_revision=record.parent_state_revision,
            identity_digest=record.identity_digest,
            evidence_digest=record.evidence_digest,
            proposal_digest=record.proposal_digest,
            parent_canonical_graph_sha256=record.parent_canonical_graph.sha256,
            parent_canonical_graph_digest=record.parent_canonical_graph_digest,
            human_decision_sha256=record.human_decision.sha256,
            human_decisions=record.human_decisions,
            reviewer=record.reviewer,
            revision_reason=record.revision_reason,
            requested_at=record.requested_at,
            changes=record.changes,
            revised_graph=revised_graph,
            revised_graph_digest=record.revised_canonical_graph_digest,
        )
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        raise AssetCompositionStateError(
            f"{prefix} provenance is invalid: {exc}"
        ) from exc
    revised_ids = {
        candidate_id
        for candidate_id, disposition in decisions.items()
        if disposition == "revise"
    }
    parent_semantic = AcceptedSemanticDecision(
        schema_version=parent_graph.schema_version,
        values={
            "canonical_graph": parent_graph.model_dump(mode="json"),
            "human_review": parent_outer.human_review.model_dump(mode="json"),
        },
    )
    revised_semantic = AcceptedSemanticDecision(
        schema_version=revised_graph.schema_version,
        values={
            "canonical_graph": revised_graph.model_dump(mode="json"),
            "human_review": revised_outer.human_review.model_dump(mode="json"),
        },
    )
    parent_proposal_digest = (
        parent_decision.proposal_artifacts[-1].sha256
        if parent_decision.proposal_artifacts
        else None
    )
    graph_provenance_invalid = (
        record.parent_canonical_graph_digest != canonical_json_digest(parent_graph)
        or record.revised_canonical_graph_digest != canonical_json_digest(revised_graph)
        or changes != record.changes
        or {item.candidate_id for item in changes} != revised_ids
        or "reject" in decisions.values()
        or record.revision_patch_digest != canonical_json_digest(reconstructed_patch)
    )
    if graph_provenance_invalid:
        raise AssetCompositionStateError(
            f"{prefix} graph digests or exact delta are stale"
        )
    human_decision_invalid = (
        record.requested_at < human.created_at
        or record.created_at < record.requested_at
        or human.disposition != "revise"
        or human.producer.producer_id != record.reviewer
        or human.coordinator_decision != artifact_reference(parent_decision)
        or human.reviewed_decision_digest != parent_decision.accepted_decision_digest
    )
    if human_decision_invalid:
        raise AssetCompositionStateError(
            f"{prefix} human revise decision provenance is stale"
        )
    coordinator_decisions_invalid = (
        parent_decision.identity != revised_decision.identity
        or parent_decision.producer != revised_decision.producer
        or record.identity_digest != canonical_json_digest(revised_decision.identity)
        or parent_decision.evidence_artifacts != revised_decision.evidence_artifacts
        or len(revised_decision.evidence_artifacts) != 1
        or record.evidence_digest != revised_decision.evidence_artifacts[0].sha256
        or parent_decision.proposal_artifacts != revised_decision.proposal_artifacts
        or record.proposal_digest != parent_proposal_digest
        or parent_decision.disposition != "accept"
        or revised_decision.disposition != "accept"
        or revised_decision.created_at != record.created_at
        or not parent_decision.human_decision_required
        or not revised_decision.human_decision_required
        or parent_decision.accepted_decision != parent_semantic
        or parent_decision.accepted_decision_digest
        != accepted_semantic_decision_digest(parent_decision.identity, parent_semantic)
        or revised_decision.accepted_decision != revised_semantic
        or revised_decision.accepted_decision_digest
        != accepted_semantic_decision_digest(
            revised_decision.identity, revised_semantic
        )
    )
    if coordinator_decisions_invalid:
        raise AssetCompositionStateError(
            f"{prefix} coordinator decision provenance is stale"
        )
    outer_reviews_invalid = (
        parent_outer.canonical_graph_sha256 != record.parent_canonical_graph.sha256
        or revised_outer.canonical_graph_sha256 != record.revised_canonical_graph.sha256
        or parent_outer.identity_digest != record.identity_digest
        or revised_outer.identity_digest != record.identity_digest
        or parent_outer.reviewer != parent_decision.producer
        or revised_outer.reviewer != revised_decision.producer
        or parent_outer.evidence_digest != record.evidence_digest
        or revised_outer.evidence_digest != record.evidence_digest
        or parent_outer.proposal_digest != record.proposal_digest
        or revised_outer.proposal_digest != record.proposal_digest
        or parent_outer.disposition != parent_decision.disposition
        or revised_outer.disposition != revised_decision.disposition
        or parent_outer.rationale != parent_decision.rationale
        or revised_outer.rationale != revised_decision.rationale
        or parent_outer.revision_requests != parent_decision.revision_requests
        or revised_outer.revision_requests != revised_decision.revision_requests
        or revised_outer.human_review.status != "human_required"
    )
    if outer_reviews_invalid:
        raise AssetCompositionStateError(f"{prefix} outer review provenance is stale")
    return record


def _verify_articulation_review_history(
    reviews: Sequence[SupersededArticulationReview],
    *,
    current_candidates: ArtifactBinding | None,
    root: Path,
    prefix: str,
) -> None:
    prior_revised_graph: _BindingLike | None = None
    prior_revision_receipt: _BindingLike | None = None
    prior_revision_number: int | None = None
    for review_index, review in enumerate(reviews, start=1):
        record = _verify_superseded_articulation_review(
            review,
            index=review_index,
            root=root,
        )
        if prior_revised_graph is not None and not _same_binding(
            record.parent_canonical_graph,
            prior_revised_graph,
        ):
            raise AssetCompositionStateError(
                f"{prefix} Articulation review revision chain is disconnected"
            )
        if (prior_revision_receipt is None and record.parent_revision is not None) or (
            prior_revision_receipt is not None
            and (
                prior_revision_number is None
                or record.parent_revision is None
                or not _same_binding(
                    record.parent_revision,
                    prior_revision_receipt,
                )
                or record.revision != prior_revision_number + 1
            )
        ):
            raise AssetCompositionStateError(
                f"{prefix} Articulation revision receipt chain is disconnected"
            )
        prior_revised_graph = record.revised_canonical_graph
        prior_revision_receipt = review.revision_receipt
        prior_revision_number = record.revision
    if prior_revised_graph is not None and (
        current_candidates is None
        or not _same_binding(current_candidates, prior_revised_graph)
    ):
        raise AssetCompositionStateError(
            f"{prefix} candidates differ from the latest Articulation revision"
        )


def _load_execution_graph(binding: ArtifactBinding) -> AssetExecutionGraph:
    try:
        return AssetExecutionGraph.model_validate(
            _json_object_from_binding(binding, label="frozen execution graph")
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid frozen execution graph at {binding.path}: {exc}"
        ) from exc


def _load_leaf_receipt(binding: ArtifactBinding, *, label: str) -> AssetLeafReceipt:
    try:
        return AssetLeafReceipt.model_validate(
            _json_object_from_binding(binding, label=label)
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid {label} at {binding.path}: {exc}"
        ) from exc


def _verify_receipt_timing(
    *,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    label: str,
) -> None:
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        expected_duration = int((finished - started).total_seconds() * 1000)
    except ValueError as exc:
        raise AssetCompositionStateError(f"{label} timing is invalid") from exc
    if expected_duration < 0 or duration_ms != expected_duration:
        raise AssetCompositionStateError(f"{label} duration changed")


def _verify_leaf_receipt_artifacts(
    receipt: AssetLeafReceipt,
    *,
    attempt_root: Path,
    label: str,
) -> None:
    bindings = [
        ("invocation", receipt.invocation),
        ("result", receipt.result),
        ("projection", receipt.projection),
        ("native terminal receipt", receipt.native_terminal_receipt),
    ]
    for binding_label, binding in bindings:
        if binding is None:
            continue
        _verify_binding(
            binding,
            label=f"{label} {binding_label}",
            required_root=attempt_root,
        )
    for group_label, group in (
        ("operation index", receipt.operation_indexes),
        ("evidence index", receipt.evidence_indexes),
        ("evidence", receipt.evidence),
        ("saved-stage readback", receipt.saved_stage_readbacks),
        ("resource release", receipt.resource_release_receipts),
    ):
        for binding_index, binding in enumerate(group, start=1):
            _verify_binding(
                binding,
                label=f"{label} {group_label} {binding_index}",
                required_root=attempt_root,
            )


def _verify_leaf_receipt_projection(
    receipt: AssetLeafReceipt,
    *,
    runtime_binding: AssetLeafRuntimeBinding,
    attempt_root: Path,
    label: str,
) -> None:
    if receipt.schema_version != ASSET_LEAF_RECEIPT_SCHEMA_VERSION:
        raise AssetCompositionStateError(
            f"{label} legacy receipt cannot carry a deterministic projection"
        )
    if receipt.projection is None or receipt.result is None:
        raise AssetCompositionStateError(
            f"{label} v2 projection identity is incomplete"
        )
    if Path(receipt.projection.path) != attempt_root / "leaf_projection.json":
        raise AssetCompositionStateError(f"{label} projection path changed")
    try:
        invocation = runtime_binding.invocation_model.model_validate(
            _json_object_from_binding(
                receipt.invocation,
                label=f"{label} invocation",
            )
        )
        result = runtime_binding.result_model.model_validate(
            _json_object_from_binding(
                receipt.result,
                label=f"{label} result",
            )
        )
        projection = AssetLeafProjection.model_validate(
            _json_object_from_binding(
                receipt.projection,
                label=f"{label} projection",
            )
        )
        expected = runtime_binding.project(
            invocation,
            result,
            invocation_artifact=receipt.invocation,
            result_artifact=receipt.result,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"{label} descriptor-resolved projection is invalid"
        ) from exc
    payload = projection.payload
    if (
        projection != expected
        or receipt.descriptor_digest != runtime_binding.descriptor.descriptor_digest
        or receipt.invocation_schema_digest
        != runtime_binding.descriptor.invocation_schema_digest
        or receipt.result_schema_digest
        != runtime_binding.descriptor.result_schema_digest
        or receipt.projection_schema_digest
        != runtime_binding.descriptor.projection_schema_digest
        or receipt.projector_id != runtime_binding.descriptor.projector_id
        or receipt.projector_digest != runtime_binding.descriptor.projector_digest
        or receipt.required_artifact_categories
        != runtime_binding.descriptor.required_artifact_categories
        or receipt.native_terminal_receipt != payload.native_terminal_receipt
        or receipt.operation_indexes != list(payload.operation_indexes)
        or receipt.evidence_indexes != list(payload.evidence_indexes)
        or receipt.evidence != list(payload.evidence)
        or receipt.saved_stage_readbacks != list(payload.saved_stage_readbacks)
        or receipt.resource_claims != list(payload.resource_claims)
        or receipt.resource_release_receipts != list(payload.resource_release_receipts)
        or receipt.native_disposition != payload.native_disposition
        or receipt.native_status != payload.native_status
        or receipt.summary != payload.summary
        or receipt.error != payload.error
    ):
        raise AssetCompositionStateError(
            f"{label} receipt differs from its declared deterministic projection"
        )


def _validate_graph_catalog(
    graph: AssetExecutionGraph,
    catalog: AssetLeafCatalog,
    *,
    required_leaf_ids: Sequence[str] = (),
    required_terminal_leaf_ids: Sequence[str] = (),
    required_leaf_dependencies: Mapping[str, Sequence[str]] | None = None,
    exact_leaf_scope: bool = False,
) -> None:
    expected_catalog_version = (
        LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
        if graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
        else ASSET_LEAF_CATALOG_SCHEMA_VERSION
    )
    if catalog.schema_version != expected_catalog_version:
        raise AssetCompositionStateError(
            "Execution graph and leaf catalog versions are incompatible: "
            f"graph={graph.schema_version}, catalog={catalog.schema_version}"
        )
    descriptors = {descriptor.leaf_id: descriptor for descriptor in catalog.descriptors}
    catalog_ids = set(descriptors)
    dispositioned = set(graph.selected_leaf_ids).union(graph.omitted_leaf_ids)
    if dispositioned != catalog_ids:
        missing = sorted(catalog_ids.difference(dispositioned))
        unknown = sorted(dispositioned.difference(catalog_ids))
        raise AssetCompositionStateError(
            "Frozen graph must disposition every catalog leaf exactly once; "
            f"missing={missing}, unknown={unknown}"
        )
    selected_ids = set(graph.selected_leaf_ids)
    nodes = {node.leaf_id: node for node in graph.nodes}
    missing_required = sorted(set(required_leaf_ids).difference(selected_ids))
    if missing_required:
        raise AssetCompositionStateError(
            f"Frozen graph omits request-required leaves: {missing_required}"
        )
    optional_required = sorted(
        leaf_id
        for leaf_id in required_leaf_ids
        if nodes[leaf_id].requirement != "required"
    )
    if optional_required:
        raise AssetCompositionStateError(
            f"Frozen graph marks request-required leaves optional: {optional_required}"
        )
    if exact_leaf_scope:
        unexpected_selected = sorted(selected_ids.difference(required_leaf_ids))
        if unexpected_selected:
            raise AssetCompositionStateError(
                "Frozen graph over-selects leaves outside the exact prompt scope: "
                f"{unexpected_selected}"
            )
        missing_rationales = sorted(
            leaf_id
            for leaf_id in graph.selected_leaf_ids
            if not (nodes[leaf_id].selection_rationale or "").strip()
        )
        if missing_rationales:
            raise AssetCompositionStateError(
                "Frozen graph omits prompt-relevance rationale for exact-scope "
                f"leaves: {missing_rationales}"
            )
    terminal_ids = {node.leaf_id for node in graph.nodes if node.terminal_output}
    missing_required_terminals = sorted(
        set(required_terminal_leaf_ids).difference(terminal_ids)
    )
    if missing_required_terminals:
        raise AssetCompositionStateError(
            "Frozen graph does not mark request-required terminal leaves as "
            f"terminal outputs: {missing_required_terminals}"
        )
    unexpected_terminals = sorted(
        terminal_ids.difference(required_terminal_leaf_ids)
        if required_terminal_leaf_ids
        else ()
    )
    if unexpected_terminals:
        raise AssetCompositionStateError(
            "Frozen graph adds terminal outputs outside the frozen request: "
            f"{unexpected_terminals}"
        )
    for leaf_id, dependencies in (required_leaf_dependencies or {}).items():
        missing_edges = sorted(set(dependencies).difference(nodes[leaf_id].depends_on))
        if missing_edges:
            raise AssetCompositionStateError(
                "Frozen graph omits request-required dependency edges for "
                f"{leaf_id}: {missing_edges}"
            )
    for leaf_id in graph.selected_leaf_ids:
        node = nodes[leaf_id]
        descriptor = descriptors[leaf_id]
        if node.descriptor_digest != descriptor.descriptor_digest:
            raise AssetCompositionStateError(
                f"Frozen graph has a stale descriptor digest for {leaf_id}"
            )
        missing_dependencies = sorted(
            set(descriptor.required_dependencies).difference(node.depends_on)
        )
        if missing_dependencies:
            raise AssetCompositionStateError(
                f"Frozen graph omits declared dependencies for {leaf_id}: "
                f"{missing_dependencies}"
            )
        for dependent_id in descriptor.required_dependents:
            if dependent_id not in selected_ids:
                continue
            dependent = nodes[dependent_id]
            if leaf_id not in dependent.depends_on:
                raise AssetCompositionStateError(
                    "Frozen graph omits declared producer dependency for "
                    f"{dependent_id}: {leaf_id}"
                )
        incompatible = sorted(
            selected_ids.intersection(descriptor.incompatible_leaf_ids)
        )
        if incompatible:
            raise AssetCompositionStateError(
                f"Frozen graph selects incompatible leaves with {leaf_id}: "
                f"{incompatible}"
            )


def _resolve_graph_leaf_runtime_binding(
    request: AssetRunRequest,
    graph: AssetExecutionGraph,
    leaf_id: str,
) -> AssetLeafRuntimeBinding:
    if graph.schema_version != ASSET_EXECUTION_GRAPH_SCHEMA_VERSION:
        raise AssetCompositionStateError(
            "Legacy graph leaves have no repository runtime binding"
        )
    catalog = request.leaf_catalog
    if catalog is None:
        raise AssetCompositionStateError("Agentic request lacks a leaf catalog")
    try:
        runtime = resolve_repository_asset_leaf_catalog(catalog)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AssetCompositionStateError(
            "Frozen leaf catalog does not resolve to the complete repository catalog"
        ) from exc
    descriptor = next(
        (item for item in catalog.descriptors if item.leaf_id == leaf_id),
        None,
    )
    node = next((item for item in graph.nodes if item.leaf_id == leaf_id), None)
    if descriptor is None or node is None:
        raise AssetCompositionStateError(
            f"Active leaf is absent from the frozen graph catalog: {leaf_id}"
        )
    if node.descriptor_digest != descriptor.descriptor_digest:
        raise AssetCompositionStateError(
            f"Frozen descriptor identity drifted for {leaf_id}"
        )
    try:
        return runtime.resolve(descriptor)
    except (TypeError, ValueError) as exc:
        raise AssetCompositionStateError(
            f"Repository runtime binding drifted for {leaf_id}"
        ) from exc


def _verify_leaf_transition_log(
    run: AssetCompositionRun,
    graph: AssetExecutionGraph,
) -> None:
    statuses = {leaf_id: "pending" for leaf_id in graph.selected_leaf_ids}
    attempts = {leaf_id: 0 for leaf_id in graph.selected_leaf_ids}
    allowed = {
        ("pending", "ready"),
        ("ready", "running"),
        ("ready", "failed"),
        ("ready", "cancelled"),
        ("running", "passed"),
        ("running", "not_evaluated"),
        ("running", "failed"),
        ("running", "cancelled"),
        ("failed", "ready"),
        ("cancelled", "ready"),
    }
    for index, transition in enumerate(run.leaf_transitions, start=1):
        if transition.leaf_id not in statuses:
            raise AssetCompositionStateError(
                f"Leaf transition {index} names an unselected leaf"
            )
        observed = statuses[transition.leaf_id]
        edge = (transition.from_status, transition.to_status)
        if observed != transition.from_status or edge not in allowed:
            raise AssetCompositionStateError(
                f"Leaf transition {index} is not append-only and contiguous"
            )
        if edge in {
            ("ready", "running"),
            ("ready", "failed"),
            ("ready", "cancelled"),
        }:
            attempts[transition.leaf_id] += 1
        if transition.attempt_count != attempts[transition.leaf_id]:
            raise AssetCompositionStateError(
                f"Leaf transition {index} attempt identity changed"
            )
        statuses[transition.leaf_id] = transition.to_status
    for leaf_id, state in run.leaf_states.items():
        if (
            statuses[leaf_id] != state.status
            or attempts[leaf_id] != state.attempt_count
        ):
            raise AssetCompositionStateError(
                f"Leaf transition history differs from durable state for {leaf_id}"
            )


def _verify_graph_run(run: AssetCompositionRun, *, root: Path) -> None:
    request_payload = _json_object_from_binding(
        run.request,
        label="frozen asset request",
    )
    try:
        request = AssetRunRequest.model_validate(
            _normalize_historical_asset_request_payload(request_payload)
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid frozen composed asset request: {exc}"
        ) from exc
    if request.selected_mode != "agentic":
        raise AssetCompositionStateError(
            "Agentic graph run is bound to a non-agentic request"
        )
    if run.execution_graph is None:
        if run.leaf_states or run.current_leaf_id is not None:
            raise AssetCompositionStateError(
                "Unfrozen graph run contains leaf progress"
            )
        return
    _verify_binding(
        run.execution_graph,
        label="frozen execution graph",
        required_root=root,
    )
    graph = _load_execution_graph(run.execution_graph)
    coordinator = request.sole_coordinator_identity
    catalog = request.leaf_catalog
    if coordinator is None or catalog is None:
        raise AssetCompositionStateError(
            "Agentic request lacks coordinator or catalog identity"
        )
    expected_graph_identity = (
        graph.sole_coordinator_identity_digest == coordinator.identity_digest
        and graph.prompt_digest == request.prompt_digest
        and graph.source_digest == request.source_digest
        and graph.configuration_digest == request.configuration_digest
        and graph.reference_digest == request.reference_digest
        and graph.leaf_catalog_digest == catalog.catalog_digest
    )
    if not expected_graph_identity:
        raise AssetCompositionStateError(
            "Frozen graph identity differs from the frozen request"
        )
    _validate_graph_catalog(
        graph,
        catalog,
        required_leaf_ids=request.required_leaf_ids,
        required_terminal_leaf_ids=request.required_terminal_leaf_ids,
        required_leaf_dependencies=request.required_leaf_dependencies,
        exact_leaf_scope=request.exact_leaf_scope,
    )
    legacy_graph = graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
    expected_receipt_version = (
        LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION
        if legacy_graph
        else ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    )
    expected_terminal_version = (
        LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
        if legacy_graph
        else ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
    )
    if not legacy_graph:
        for leaf_id in graph.selected_leaf_ids:
            _resolve_graph_leaf_runtime_binding(request, graph, leaf_id)
    if set(run.leaf_states) != set(graph.selected_leaf_ids):
        raise AssetCompositionStateError(
            "Durable leaf states differ from the frozen selected graph"
        )
    _verify_leaf_transition_log(run, graph)
    selected_receipts: dict[str, ArtifactBinding] = {}
    terminal_artifacts: list[ArtifactBinding] = []
    leaf_release_receipts: list[ArtifactBinding] = []
    terminal_statuses = {"passed", "failed", "cancelled", "not_evaluated"}
    if (
        run.current_leaf_id is not None
        and run.current_leaf_id not in graph.selected_leaf_ids
    ):
        raise AssetCompositionStateError(
            "Current leaf is absent from the frozen selected graph"
        )
    running_leaf_ids = [
        leaf_id
        for leaf_id, state in run.leaf_states.items()
        if state.status == "running"
    ]
    expected_running = (
        [run.current_leaf_id]
        if run.coordinator.next_action == "execute_leaf"
        and run.current_leaf_id is not None
        else []
    )
    if running_leaf_ids != expected_running:
        raise AssetCompositionStateError(
            "Active graph leaf differs from the sole running leaf"
        )
    current_index = (
        graph.selected_leaf_ids.index(run.current_leaf_id)
        if legacy_graph and run.current_leaf_id is not None
        else None
    )
    for index, node in enumerate(graph.nodes):
        state = run.leaf_states[node.leaf_id]
        runtime_binding = (
            None
            if legacy_graph
            else _resolve_graph_leaf_runtime_binding(
                request,
                graph,
                node.leaf_id,
            )
        )
        if (
            state.leaf_id != node.leaf_id
            or state.requirement != node.requirement
            or state.depends_on != node.depends_on
            or state.terminal_output != node.terminal_output
        ):
            raise AssetCompositionStateError(
                f"Leaf state identity changed for {node.leaf_id}"
            )
        if legacy_graph:
            if current_index is not None:
                if index < current_index and state.status not in {
                    "passed",
                    "not_evaluated",
                }:
                    raise AssetCompositionStateError(
                        f"Predecessor leaf {node.leaf_id} is not terminal-successful"
                    )
                if index > current_index and state.status != "pending":
                    raise AssetCompositionStateError(
                        f"Future leaf {node.leaf_id} is not pending"
                    )
        else:
            dependencies_passed = all(
                run.leaf_states[dependency].status == "passed"
                for dependency in node.depends_on
            )
            if state.status == "ready" and not dependencies_passed:
                raise AssetCompositionStateError(
                    f"Ready leaf {node.leaf_id} has unsatisfied dependencies"
                )
            if state.status == "pending" and dependencies_passed:
                raise AssetCompositionStateError(
                    f"Dependency-ready leaf {node.leaf_id} remained pending"
                )
        expected_dependencies: dict[str, ArtifactBinding] | None = None
        if state.superseded_receipts or state.receipt is not None:
            missing_dependencies = [
                dependency
                for dependency in node.depends_on
                if run.leaf_states[dependency].receipt is None
            ]
            if missing_dependencies:
                raise AssetCompositionStateError(
                    f"Dependency receipts are missing for {node.leaf_id}: "
                    f"{missing_dependencies}"
                )
            expected_dependencies = {
                dependency: cast(
                    ArtifactBinding,
                    run.leaf_states[dependency].receipt,
                )
                for dependency in node.depends_on
            }
        prior_superseded: ArtifactBinding | None = None
        for attempt, superseded_binding in enumerate(
            state.superseded_receipts,
            start=1,
        ):
            attempt_root = _graph_attempt_root(
                root,
                graph,
                node.leaf_id,
                attempt=attempt,
            )
            if Path(superseded_binding.path) != attempt_root / "leaf_receipt.json":
                raise AssetCompositionStateError(
                    f"Superseded receipt path changed for {node.leaf_id}"
                )
            _verify_binding(
                superseded_binding,
                label=f"{node.leaf_id} superseded receipt {attempt}",
                required_root=attempt_root,
            )
            superseded = _load_leaf_receipt(
                superseded_binding,
                label=f"{node.leaf_id} superseded receipt {attempt}",
            )
            if (
                superseded.schema_version != expected_receipt_version
                or superseded.run_id != run.run_id
                or superseded.graph_digest != graph.graph_digest
                or superseded.leaf_catalog_digest != graph.leaf_catalog_digest
                or superseded.sole_coordinator_identity_digest
                != graph.sole_coordinator_identity_digest
                or superseded.leaf_id != node.leaf_id
                or superseded.requirement != node.requirement
                or superseded.depends_on != node.depends_on
                or superseded.dependency_receipts != expected_dependencies
                or superseded.attempt != attempt
                or superseded.native_disposition not in {"failed", "cancelled"}
                or superseded.supersedes != prior_superseded
            ):
                raise AssetCompositionStateError(
                    f"Superseded receipt chain changed for {node.leaf_id}"
                )
            _verify_receipt_timing(
                started_at=superseded.started_at,
                finished_at=superseded.finished_at,
                duration_ms=superseded.duration_ms,
                label=f"{node.leaf_id} superseded receipt {attempt}",
            )
            _verify_leaf_receipt_artifacts(
                superseded,
                attempt_root=attempt_root,
                label=f"{node.leaf_id} superseded receipt {attempt}",
            )
            if runtime_binding is not None:
                _verify_leaf_receipt_projection(
                    superseded,
                    runtime_binding=runtime_binding,
                    attempt_root=attempt_root,
                    label=f"{node.leaf_id} superseded receipt {attempt}",
                )
            prior_superseded = superseded_binding
        expected_attempt_count = len(state.superseded_receipts)
        if state.status in {"running", *terminal_statuses}:
            expected_attempt_count += 1
        if state.attempt_count != expected_attempt_count:
            raise AssetCompositionStateError(
                f"Leaf attempt count changed for {node.leaf_id}"
            )
        if state.receipt is None:
            if state.status in terminal_statuses:
                raise AssetCompositionStateError(
                    f"Terminal leaf {node.leaf_id} lacks its receipt"
                )
            continue
        _verify_binding(
            state.receipt,
            label=f"{node.leaf_id} receipt",
            required_root=_graph_attempt_root(
                root,
                graph,
                node.leaf_id,
                attempt=state.attempt_count,
            ),
        )
        current_attempt_root = _graph_attempt_root(
            root,
            graph,
            node.leaf_id,
            attempt=state.attempt_count,
        )
        if Path(state.receipt.path) != current_attempt_root / "leaf_receipt.json":
            raise AssetCompositionStateError(
                f"Leaf receipt path changed for {node.leaf_id}"
            )
        receipt = _load_leaf_receipt(
            state.receipt,
            label=f"{node.leaf_id} receipt",
        )
        if expected_dependencies is None:  # pragma: no cover - receipt set above
            raise AssetCompositionStateError(
                f"Dependency receipt verification was skipped for {node.leaf_id}"
            )
        if (
            receipt.schema_version != expected_receipt_version
            or receipt.run_id != run.run_id
            or receipt.graph_digest != graph.graph_digest
            or receipt.leaf_catalog_digest != graph.leaf_catalog_digest
            or receipt.sole_coordinator_identity_digest
            != graph.sole_coordinator_identity_digest
            or receipt.leaf_id != node.leaf_id
            or receipt.requirement != node.requirement
            or receipt.depends_on != node.depends_on
            or receipt.dependency_receipts != expected_dependencies
            or receipt.attempt != state.attempt_count
            or receipt.run_revision >= run.revision
            or receipt.native_disposition != state.status
            or receipt.error != state.error
            or receipt.supersedes != prior_superseded
        ):
            raise AssetCompositionStateError(
                f"Leaf receipt identity changed for {node.leaf_id}"
            )
        _verify_receipt_timing(
            started_at=receipt.started_at,
            finished_at=receipt.finished_at,
            duration_ms=receipt.duration_ms,
            label=f"{node.leaf_id} receipt",
        )
        _verify_leaf_receipt_artifacts(
            receipt,
            attempt_root=current_attempt_root,
            label=node.leaf_id,
        )
        if runtime_binding is not None:
            _verify_leaf_receipt_projection(
                receipt,
                runtime_binding=runtime_binding,
                attempt_root=current_attempt_root,
                label=node.leaf_id,
            )
        selected_receipts[node.leaf_id] = state.receipt
        leaf_release_receipts.extend(receipt.resource_release_receipts)
        if node.terminal_output and receipt.result is not None:
            terminal_artifacts.append(receipt.result)
    if run.graph_terminal_receipt is not None:
        if (
            Path(run.graph_terminal_receipt.path)
            != root / "graph_terminal_receipt.json"
        ):
            raise AssetCompositionStateError("Graph terminal receipt path changed")
        _verify_binding(
            run.graph_terminal_receipt,
            label="graph terminal receipt",
            required_root=root,
        )
        try:
            terminal = AssetGraphTerminalReceipt.model_validate(
                _json_object_from_binding(
                    run.graph_terminal_receipt,
                    label="graph terminal receipt",
                )
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid graph terminal receipt: {exc}"
            ) from exc
        if (
            terminal.schema_version != expected_terminal_version
            or terminal.run_id != run.run_id
            or terminal.graph != run.execution_graph
            or terminal.graph_digest != graph.graph_digest
            or terminal.leaf_catalog_digest != graph.leaf_catalog_digest
            or terminal.sole_coordinator_identity_digest
            != graph.sole_coordinator_identity_digest
            or terminal.prompt_digest != graph.prompt_digest
            or terminal.source_digest != graph.source_digest
            or terminal.configuration_digest != graph.configuration_digest
            or terminal.reference_digest != graph.reference_digest
            or terminal.selected_leaf_ids != graph.selected_leaf_ids
            or terminal.selected_leaf_receipts != selected_receipts
            or terminal.omitted_leaf_ids != graph.omitted_leaf_ids
            or terminal.omitted_leaf_dispositions
            != {leaf_id: "not_requested" for leaf_id in graph.omitted_leaf_ids}
            or terminal.terminal_artifacts != terminal_artifacts
        ):
            raise AssetCompositionStateError(
                "Graph terminal receipt differs from durable graph state"
            )
        if not legacy_graph and any(
            item is None
            for item in (
                terminal.parent_release_receipt,
                terminal.parent_command_receipt_journal,
                terminal.parent_command_receipt_checkpoint,
            )
        ):
            raise AssetCompositionStateError(
                "Graph terminal receipt omits launcher post-teardown evidence"
            )
        if (
            legacy_graph
            and request.requires_parent_resource_release
            and not terminal.resource_release_receipts
        ):
            raise AssetCompositionStateError(
                "Legacy graph terminal receipt omits parent resource release"
            )
        _verify_receipt_timing(
            started_at=terminal.started_at,
            finished_at=terminal.finished_at,
            duration_ms=terminal.duration_ms,
            label="graph terminal receipt",
        )
        terminal_release_identities = {
            (binding.path, binding.sha256, binding.size_bytes)
            for binding in terminal.resource_release_receipts
        }
        if any(
            (binding.path, binding.sha256, binding.size_bytes)
            not in terminal_release_identities
            for binding in leaf_release_receipts
        ):
            raise AssetCompositionStateError(
                "Graph terminal receipt omits leaf resource releases"
            )
        parent_command_evidence = [
            binding
            for binding in (
                terminal.parent_command_receipt_journal,
                terminal.parent_command_receipt_checkpoint,
            )
            if binding is not None
        ]
        for index, binding in enumerate(
            [
                *terminal.terminal_artifacts,
                *terminal.resource_release_receipts,
                *parent_command_evidence,
            ],
            start=1,
        ):
            _verify_binding(
                binding,
                label=f"graph terminal artifact {index}",
                required_root=root,
            )


def _verify_run(run: AssetCompositionRun, *, root: Path) -> None:
    _verify_binding(run.request, label="frozen request")
    source_required_root: Path | None = None
    if run.coordinator.mode == "single_reasoning_loop":
        request_payload = _json_object_from_binding(
            run.request,
            label="frozen asset request",
        )
        if request_payload.get("source_staging") is not None:
            source_required_root = root
    _verify_binding(
        run.source_asset,
        label="source asset",
        required_root=source_required_root,
    )
    _verify_dependency_bindings(
        Path(run.source_asset.path),
        run.source_dependencies,
        label="source asset",
        required_root=source_required_root,
    )
    _verify_coordinator_state(run, root=root)
    if run.selected_mode == "agentic":
        _verify_graph_run(run, root=root)
        return
    physics_validation_mode = _physics_validation_mode_from_run(run)

    expected_input = run.source_asset
    expected_dependencies = run.source_dependencies
    expected_readiness: Literal["yes", "conditional"] = "yes"
    for stage in run.stage_order:
        state = run.stages[stage]
        if stage != "articulation" and state.superseded_reviews:
            raise AssetCompositionStateError(
                f"{stage} cannot retain Articulation review revisions"
            )
        _verify_articulation_review_history(
            state.superseded_reviews,
            current_candidates=state.review_candidates,
            root=root,
            prefix=f"current {stage} stage",
        )
        for attempt_index, attempt in enumerate(
            state.superseded_attempts,
            start=1,
        ):
            _verify_superseded_attempt(
                attempt,
                stage=stage,
                index=attempt_index,
                root=root,
            )
        if state.input_asset is not None and state.input_asset != expected_input:
            raise AssetCompositionStateError(
                f"{stage} input does not match its predecessor output"
            )
        if state.input_asset is not None and (
            state.input_dependencies != expected_dependencies
        ):
            raise AssetCompositionStateError(
                f"{stage} input dependencies do not match its predecessor output"
            )
        if (
            state.input_asset is not None
            and state.input_readiness != expected_readiness
        ):
            raise AssetCompositionStateError(
                f"{stage} input readiness does not match its predecessor handoff"
            )
        if state.review_candidates is not None:
            _verify_binding(
                state.review_candidates,
                label=f"{stage} review candidates",
                required_root=root,
            )
        if state.review_decisions is not None:
            _verify_binding(
                state.review_decisions,
                label=f"{stage} review decisions",
                required_root=root,
            )
        if state.status == "completed":
            if state.output_asset is None or state.handoff is None:
                raise AssetCompositionStateError(
                    f"Completed stage {stage} lacks its output or handoff binding"
                )
            _verify_binding(
                state.output_asset,
                label=f"{stage} output asset",
                required_root=root,
            )
            _verify_dependency_bindings(
                Path(state.output_asset.path),
                state.output_dependencies,
                label=f"{stage} output asset",
                required_root=root,
            )
            _verify_binding(
                state.handoff,
                label=f"{stage} handoff",
                required_root=root,
            )
            for index, evidence in enumerate(state.evidence):
                _verify_binding(
                    evidence,
                    label=f"{stage} evidence {index + 1}",
                    required_root=root,
                )
            if stage == "physics":
                _validate_physics_runtime_evidence(
                    state.evidence,
                    output=state.output_asset,
                    validation_mode=physics_validation_mode,
                )
                _validate_articulated_physics_output(Path(state.output_asset.path))
            try:
                handoff = AssetStageHandoff.model_validate(
                    _json_object_from_binding(
                        state.handoff,
                        label=f"{stage} handoff",
                    )
                )
            except ValidationError as exc:
                raise AssetCompositionStateError(
                    f"{stage} handoff is invalid: {exc}"
                ) from exc
            if (
                handoff.stage != stage
                or handoff.input_asset != state.input_asset
                or handoff.input_dependencies != state.input_dependencies
                or handoff.output_asset != state.output_asset
                or handoff.output_dependencies != state.output_dependencies
                or handoff.evidence != state.evidence
                or handoff.readiness not in {"yes", "conditional"}
            ):
                raise AssetCompositionStateError(
                    f"{stage} handoff does not match accepted stage state"
                )
            expected_input = state.output_asset
            expected_dependencies = state.output_dependencies
            expected_readiness = handoff.readiness
        elif state.status == "pending":
            if state.input_asset is not None or state.input_dependencies:
                raise AssetCompositionStateError(
                    f"pending stage {stage} unexpectedly has an input binding"
                )
        elif stage != run.current_stage:
            raise AssetCompositionStateError(
                f"nonterminal stage {stage} is not the current stage"
            )
    if run.stages["finalization"].status == "completed":
        _verify_combined_report(run, root=root)


def _combined_report_from_evidence(
    evidence: Sequence[ArtifactBinding],
) -> tuple[ArtifactBinding, AssetCombinedReport]:
    matches: list[tuple[ArtifactBinding, AssetCombinedReport]] = []
    for binding in evidence:
        if Path(binding.path).suffix.lower() != ".json":
            continue
        payload = _json_object_from_binding(
            binding,
            label=f"finalization evidence {Path(binding.path).name}",
        )
        if not isinstance(payload, dict) or payload.get("schema_version") != (
            "content-agent-workflows.asset-combined-report.v1"
        ):
            continue
        try:
            matches.append((binding, AssetCombinedReport.model_validate(payload)))
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Combined final report is invalid: {exc}"
            ) from exc
    if len(matches) > 1:
        raise AssetCompositionStateError(
            "Finalization evidence must include exactly one combined asset report"
        )
    if matches:
        return matches[0]
    raise AssetCompositionStateError(
        "Finalization evidence must include the combined asset report"
    )


def _verify_combined_report(run: AssetCompositionRun, *, root: Path) -> None:
    finalization = run.stages["finalization"]
    if finalization.output_asset is None:
        raise AssetCompositionStateError(
            "Completed finalization stage lacks its output binding"
        )
    _, report = _combined_report_from_evidence(finalization.evidence)
    _validate_combined_report(
        run,
        report=report,
        final_asset=finalization.output_asset,
    )
    _validate_final_usdz(Path(finalization.output_asset.path))


def _validate_combined_report(
    run: AssetCompositionRun,
    *,
    report: AssetCombinedReport,
    final_asset: ArtifactBinding,
) -> None:
    """Validate a combined report before or after terminal persistence."""

    expected_handoffs = {
        stage: run.stages[stage].handoff for stage in run.stage_order[:-1]
    }
    if any(binding is None for binding in expected_handoffs.values()):
        raise AssetCompositionStateError(
            "Combined report cannot be verified before every prior handoff"
        )
    validation_evidence = run.stages["validation"].evidence
    if (
        report.run_id != run.run_id
        or report.request != run.request
        or report.source_asset != run.source_asset
        or report.source_dependencies != run.source_dependencies
        or report.final_asset != final_asset
        or report.stage_handoffs != expected_handoffs
        or report.validation_summary not in validation_evidence
    ):
        raise AssetCompositionStateError(
            "Combined final report does not match accepted workflow state"
        )


def _load_verified_run_snapshot(path: str | Path) -> _VerifiedRunSnapshot:
    run = load_run_state(path)
    binding_token = _VERIFIED_BINDINGS.set({})
    dependency_token = _VERIFIED_DEPENDENCIES.set({})
    plan_token = _VERIFIED_COORDINATOR_PLANS.set({})
    try:
        _verify_run(run, root=_run_root(_resolved(path)))
        coordinator_plans = dict(_VERIFIED_COORDINATOR_PLANS.get() or {})
    finally:
        _VERIFIED_COORDINATOR_PLANS.reset(plan_token)
        _VERIFIED_DEPENDENCIES.reset(dependency_token)
        _VERIFIED_BINDINGS.reset(binding_token)
    return _VerifiedRunSnapshot(
        run=run,
        coordinator_plans=coordinator_plans,
    )


def load_verified_run(path: str | Path) -> AssetCompositionRun:
    """Load state and rehash every frozen or accepted artifact."""

    return _load_verified_run_snapshot(path).run


def build_embedded_domain_execution_context(
    path: str | Path,
    *,
    domain: DomainName,
    input_asset: str | Path,
    output_dir: str | Path,
) -> DomainExecutionContext:
    """Bind an embedded domain request to the exact active coordinator attempt."""

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        verified = _load_verified_run_snapshot(state_path)
        load_verified_asset_request(state_path, run=verified.run)
        plan_binding = (
            verified.run.coordinator.plan_revisions[-1]
            if verified.run.coordinator.plan_revisions
            else None
        )
        return _build_embedded_domain_execution_context_locked(
            state_path,
            domain=domain,
            input_asset=input_asset,
            output_dir=output_dir,
            verified_run=verified.run,
            verified_plan=(
                verified.coordinator_plans.get(plan_binding.sha256)
                if plan_binding is not None
                else None
            ),
        ).context


def build_embedded_domain_decision_identity(
    path: str | Path,
    *,
    domain: DomainName,
    input_asset: str | Path,
    output_dir: str | Path,
    capability_digests: Mapping[str, str],
    implementation_digests: Mapping[str, str],
    configuration_digests: Mapping[str, str] | None = None,
) -> EmbeddedDecisionIdentity:
    """Bind a future domain decision chain to the exact active asset attempt.

    Prompt, references, frozen request configuration, source bytes, and outer
    plan identity are read from verified coordinator state.  Domain adapters
    add their exact capability and implementation manifest digests.
    """

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        verified = _load_verified_run_snapshot(state_path)
        run = verified.run
        request = load_verified_asset_request(state_path, run=run)
        plan_binding = (
            run.coordinator.plan_revisions[-1]
            if run.coordinator.plan_revisions
            else None
        )
        context_snapshot = _build_embedded_domain_execution_context_locked(
            state_path,
            domain=domain,
            input_asset=input_asset,
            output_dir=output_dir,
            verified_run=run,
            verified_plan=(
                verified.coordinator_plans.get(plan_binding.sha256)
                if plan_binding is not None
                else None
            ),
        )
        context = context_snapshot.context
        configuration = {
            "asset_request": run.request.sha256,
            "joint_config": request.joint_config_binding.sha256,
            "materials_manifest": request.materials_yaml_binding.sha256,
        }
        materials_library = request.materials_usd_binding
        if materials_library is None:  # pragma: no cover - verified request invariant
            raise AssetCompositionStateError(
                "Verified request lacks the required materials library binding"
            )
        configuration["materials_library"] = materials_library.sha256
        for name, digest in dict(configuration_digests or {}).items():
            if name in configuration and configuration[name] != digest:
                raise AssetCompositionStateError(
                    f"Additional configuration digest conflicts with {name}"
                )
            configuration[name] = digest
        reference_digests = {
            f"reference_{index:03d}": binding.sha256
            for index, binding in enumerate(request.reference_bindings, start=1)
        }
        reference_digests.update(
            {
                f"source_dependency_{index:03d}": binding.sha256
                for index, binding in enumerate(run.source_dependencies, start=1)
            }
        )
        reference_digests.update(
            {
                f"active_input_dependency_{index:03d}": binding.sha256
                for index, binding in enumerate(
                    run.stages[domain].input_dependencies,
                    start=1,
                )
            }
        )
        reference_digests.update(
            {
                f"materials_library_dependency_{index:03d}": binding.sha256
                for index, binding in enumerate(
                    request.materials_usd_dependencies,
                    start=1,
                )
            }
        )
        if context.embedded_stage is None:  # pragma: no cover - model invariant
            raise AssetCompositionStateError("Embedded execution context lacks stage")
        plan_binding = context_snapshot.plan_binding
        plan = context_snapshot.plan
        try:
            return EmbeddedDecisionIdentity(
                execution_context=context,
                source=context.embedded_stage.input_asset,
                coordinator_plan=ContractArtifactReference(
                    artifact_kind="coordinator_plan",
                    artifact_id=Path(plan_binding.path).stem,
                    schema_version=plan.schema_version,
                    sha256=plan_binding.sha256,
                ),
                digests=NamedDecisionDigests(
                    configuration=configuration,
                    prompt={
                        "asset_prompt": hashlib.sha256(
                            request.prompt.encode("utf-8")
                        ).hexdigest()
                    },
                    references=reference_digests,
                    capabilities=dict(capability_digests),
                    implementations=dict(implementation_digests),
                ),
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid embedded {domain} decision identity: {exc}"
            ) from exc


def _build_embedded_domain_execution_context_locked(
    state_path: Path,
    *,
    domain: DomainName,
    input_asset: str | Path,
    output_dir: str | Path,
    verified_run: AssetCompositionRun | None = None,
    verified_plan: AssetCoordinatorPlan | None = None,
) -> _EmbeddedDomainContextSnapshot:
    """Build one context while holding the composed-state transition lock."""

    run = verified_run or _load_verified_transition_run(state_path)
    if run.terminal_status != "active" or run.current_stage != domain:
        raise AssetCompositionStateError(
            f"Embedded {domain} execution requires {domain} to be the active stage"
        )
    stage = run.stages[domain]
    if stage.status != "running" or stage.attempt_count < 1:
        raise AssetCompositionStateError(
            f"Embedded {domain} execution requires a running stage attempt"
        )
    if (
        run.coordinator.mode != "single_reasoning_loop"
        or run.coordinator.next_action != "execute_stage"
        or not run.coordinator.plan_revisions
    ):
        raise AssetCompositionStateError(
            f"Embedded {domain} execution requires an active coordinator plan"
        )
    plan_binding = run.coordinator.plan_revisions[-1]
    if verified_run is not None and verified_plan is None:
        raise AssetCompositionStateError(
            "Verified run is missing its coordinator plan snapshot"
        )
    plan = verified_plan or _load_coordinator_plan(plan_binding)
    if plan.stage != domain or plan.stage_attempt != stage.attempt_count:
        raise AssetCompositionStateError(
            f"Latest coordinator plan does not own the active {domain} attempt"
        )
    if stage.input_asset is None:  # pragma: no cover - guarded by StageState
        raise AssetCompositionStateError(f"Running {domain} stage lacks an input")
    supplied_input = _binding(input_asset, label=f"embedded {domain} input")
    if supplied_input != stage.input_asset:
        raise AssetCompositionStateError(
            f"Embedded {domain} input differs from the active stage handoff"
        )
    expected_output_dir = (
        stage_directory(state_path, domain, attempt=stage.attempt_count) / "domain-run"
    ).resolve()
    supplied_output_dir = Path(output_dir).expanduser().resolve()
    if supplied_output_dir != expected_output_dir:
        raise AssetCompositionStateError(
            f"Embedded {domain} output directory must be {expected_output_dir}"
        )

    def portable(binding: ArtifactBinding) -> ExecutionArtifactBinding:
        return ExecutionArtifactBinding.model_validate(binding.model_dump())

    return _EmbeddedDomainContextSnapshot(
        context=DomainExecutionContext(
            schema_version=(
                DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2
                if domain == "validation"
                else DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION
            ),
            domain=domain,
            mode="embedded",
            reasoning_loop_owner="asset_coordinator",
            embedded_stage=EmbeddedStageBinding(
                outer_run_id=run.run_id,
                outer_request=portable(run.request),
                stage=domain,
                stage_attempt=stage.attempt_count,
                coordinator_plan=portable(plan_binding),
                input_asset=portable(stage.input_asset),
                domain_run_root=str(expected_output_dir),
            ),
        ),
        plan_binding=plan_binding,
        plan=plan,
    )


def _planned_attempt_number(state: StageState) -> int:
    """Return the attempt a ready stage will begin or continue."""

    if state.status == "ready" and not state.continue_current_attempt:
        return state.attempt_count + 1
    return max(state.attempt_count, 1)


def stage_directory(
    path: str | Path,
    stage: StageName,
    *,
    attempt: int | None = None,
) -> Path:
    """Return the canonical output directory for one domain stage."""

    state_path = _resolved(path)
    if not state_path.is_file():
        raise AssetCompositionStateError(
            f"Composed asset run does not exist: {state_path}"
        )
    run = load_run_state(state_path)
    if stage not in run.stage_order:
        raise AssetCompositionStateError(
            f"Stage {stage!r} is not enabled for this composed run"
        )
    index = run.stage_order.index(stage) + 1
    base = _run_root(state_path) / "stages" / f"{index:02d}-{stage}"
    if attempt is None:
        state = run.stages[stage]
        attempt = _planned_attempt_number(state)
    if attempt is None or attempt <= 1:
        return base
    return base / "attempts" / f"{attempt:02d}"


def _belongs_to_stage_attempt(path: Path, *, attempt_root: Path, attempt: int) -> bool:
    """Reject retry namespaces when validating the first stage attempt."""

    if not path.is_relative_to(attempt_root):
        return False
    relative = path.relative_to(attempt_root)
    return attempt > 1 or not relative.parts or relative.parts[0] != "attempts"


def _ensure_stage_directory(path: Path, stage: StageName, *, attempt: int) -> Path:
    """Create the canonical stage directory without following redirected roots."""

    root = _run_root(path)
    stages_root = root / "stages"
    if stages_root.exists() and (stages_root.is_symlink() or not stages_root.is_dir()):
        raise AssetCompositionStateError(
            f"Canonical stages root must be a regular directory: {stages_root}"
        )
    stages_root.mkdir(exist_ok=True)

    base = stage_directory(path, stage, attempt=1)
    if base.exists() and (base.is_symlink() or not base.is_dir()):
        raise AssetCompositionStateError(
            f"Canonical {stage} stage path must be a regular directory: {base}"
        )
    base.mkdir(exist_ok=True)
    attempts_root = base / "attempts"
    if attempt > 1:
        if attempts_root.exists() and (
            attempts_root.is_symlink() or not attempts_root.is_dir()
        ):
            raise AssetCompositionStateError(
                f"Canonical {stage} attempts path must be a directory: {attempts_root}"
            )
        attempts_root.mkdir(exist_ok=True)

    directory = stage_directory(path, stage, attempt=attempt)
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise AssetCompositionStateError(
            f"Canonical {stage} stage path must be a regular directory: {directory}"
        )
    directory.mkdir(exist_ok=True)
    if not directory.resolve().is_relative_to(root):
        raise AssetCompositionStateError(
            f"Canonical {stage} stage directory escaped the composed run: {directory}"
        )
    return directory


def _transition(
    run: AssetCompositionRun,
    stage: StageName,
    *,
    from_status: StageStatus,
    to_status: StageStatus,
    reason: str,
    actor: str,
) -> None:
    state = run.stages[stage]
    run.transitions.append(
        StageTransition(
            timestamp=_timestamp(),
            stage=stage,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            actor=actor,
            attempt_count=state.attempt_count,
            input_sha256=(
                state.input_asset.sha256 if state.input_asset is not None else None
            ),
            output_sha256=(
                state.output_asset.sha256 if state.output_asset is not None else None
            ),
        )
    )


def create_run(
    path: str | Path,
    *,
    run_id: str,
    request_path: str | Path,
    source_asset: str | Path,
    actor: str = "content-workflow-cli",
    coordinator_mode: CoordinatorMode = "single_reasoning_loop",
    include_geometry_stage: bool = False,
    include_cad_modeling_stage: bool = False,
) -> AssetCompositionRun:
    """Create an unfrozen agentic graph or explicit fixed compatibility run."""

    _final_usdz_validator()
    state_path = _resolved(path)
    root = _run_root(state_path)
    request = _binding(
        request_path,
        label="frozen request",
        required_root=root,
    )
    if include_cad_modeling_stage and not include_geometry_stage:
        raise AssetCompositionStateError(
            "CAD modeling requires the downstream Geometry stage"
        )
    request_payload = _json_object_from_binding(request, label="frozen request")
    request_schema = request_payload.get("schema_version")
    is_typed_asset_request = isinstance(
        request_schema, str
    ) and request_schema.startswith("content-agents.asset-composition-request.")
    declares_agentic = (
        request_payload.get("selected_mode") == "agentic"
        or request_payload.get("schema_version") == ASSET_REQUEST_SCHEMA_VERSION
        or any(
            field in request_payload
            for field in (
                "prompt_digest",
                "source_digest",
                "configuration_digest",
                "reference_digest",
                "sole_coordinator_identity",
                "leaf_catalog",
            )
        )
    )
    frozen_request: AssetRunRequest | None = None
    requires_request_validation = (
        include_geometry_stage
        or include_cad_modeling_stage
        or is_typed_asset_request
        or declares_agentic
        or request_payload.get("geometry") is not None
        or request_payload.get("cad_modeling") is not None
    )
    if requires_request_validation:
        try:
            frozen_request = AssetRunRequest.model_validate(
                _normalize_historical_asset_request_payload(request_payload)
            )
        except ValidationError as exc:
            message = (
                "Configured run requires a valid frozen asset request"
                if (
                    include_geometry_stage
                    or include_cad_modeling_stage
                    or request_payload.get("geometry") is not None
                    or request_payload.get("cad_modeling") is not None
                )
                else "Invalid frozen composed asset request"
            )
            raise AssetCompositionStateError(f"{message}: {exc}") from exc
    if frozen_request is not None:
        if (frozen_request.geometry is not None) != include_geometry_stage:
            raise AssetCompositionStateError(
                "Frozen Geometry policy must match the configured Geometry stage"
            )
        if (frozen_request.cad_modeling is not None) != include_cad_modeling_stage:
            raise AssetCompositionStateError(
                "Frozen CAD modeling policy must match the configured CAD modeling stage"
            )
    if (
        frozen_request is not None
        and frozen_request.selected_mode == "agentic"
        and coordinator_mode == "legacy"
    ):
        raise AssetCompositionStateError(
            "Agentic request cannot enter legacy fixed compatibility mode"
        )
    source = _binding(source_asset, label="source asset")
    source_path = Path(source.path)
    if (
        frozen_request is not None
        and frozen_request.source_staging is not None
        and source_path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}
    ):
        staging = frozen_request.source_staging
        if staging.staged_source != source:
            raise AssetCompositionStateError(
                "Frozen non-USD staging root differs from the source asset"
            )
        source_dependencies = []
        observed_root = False
        run_root = state_path.parent.resolve()
        for index, binding in enumerate(staging.staged_dependencies, start=1):
            _verify_binding(
                binding,
                label=f"staged source artifact {index}",
                required_root=run_root,
            )
            if binding.path == source.path:
                if binding != source or observed_root:
                    raise AssetCompositionStateError(
                        "Frozen non-USD staging has an ambiguous root identity"
                    )
                observed_root = True
            else:
                source_dependencies.append(binding)
        if not observed_root:
            raise AssetCompositionStateError(
                "Frozen non-USD staging omitted its root source"
            )
    else:
        source_dependencies = _dependency_bindings(
            source_path,
            label="source asset",
        )
    with _exclusive_lock(state_path):
        if state_path.exists():
            raise AssetCompositionStateError(
                f"Composed asset run already exists: {state_path}"
            )
        if frozen_request is not None and frozen_request.selected_mode == "agentic":
            coordinator = AssetCoordinatorState(
                mode="single_reasoning_loop",
                next_action="freeze_graph",
            )
            run = AssetCompositionRun(
                schema_version=ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
                run_id=run_id,
                request=request,
                source_asset=source,
                source_dependencies=source_dependencies,
                selected_mode="agentic",
                current_stage=None,
                stages={},
                coordinator=coordinator,
            )
            return _write_run(state_path, run)
        if include_cad_modeling_stage:
            stage_order = CAD_MODELING_STAGE_ORDER
        elif include_geometry_stage:
            stage_order = GEOMETRY_STAGE_ORDER
        else:
            stage_order = LEGACY_STAGE_ORDER
        first_stage = stage_order[0]
        stages: dict[StageName, StageState] = {
            stage: StageState() for stage in stage_order
        }
        stages[first_stage] = StageState(
            status="ready",
            input_asset=source,
            input_dependencies=source_dependencies,
        )
        run = AssetCompositionRun(
            schema_version=COMPATIBILITY_FIXED_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            run_id=run_id,
            request=request,
            source_asset=source,
            source_dependencies=source_dependencies,
            stage_order=stage_order,
            current_stage=first_stage,
            stages=stages,
            selected_mode="compatibility_fixed",
            coordinator=AssetCoordinatorState(mode=coordinator_mode),
        )
        _transition(
            run,
            first_stage,
            from_status="pending",
            to_status="ready",
            reason=(
                "Frozen request and source asset were digest-bound; the first "
                f"configured stage is {first_stage}."
            ),
            actor=actor,
        )
        return _write_run(state_path, run)


def _graph_attempt_root(
    root: Path,
    graph: AssetExecutionGraph,
    leaf_id: str,
    *,
    attempt: int,
) -> Path:
    try:
        index = graph.selected_leaf_ids.index(leaf_id) + 1
    except ValueError as exc:
        raise AssetCompositionStateError(
            f"Leaf {leaf_id!r} is not selected by the frozen graph"
        ) from exc
    return root / "leaves" / f"{index:03d}-{leaf_id}" / "attempts" / f"{attempt:02d}"


def _graph_leaf_directory(
    state_path: Path,
    graph: AssetExecutionGraph,
    leaf_id: str,
    *,
    attempt: int,
) -> Path:
    return _graph_attempt_root(
        _run_root(state_path),
        graph,
        leaf_id,
        attempt=attempt,
    )


def leaf_directory(
    path: str | Path,
    leaf_id: str,
    *,
    attempt: int | None = None,
) -> Path:
    """Return the canonical directory for one selected opaque leaf attempt."""

    state_path = _resolved(path)
    run = load_verified_run(state_path)
    if run.selected_mode != "agentic" or run.execution_graph is None:
        raise AssetCompositionStateError("Agentic execution graph is not frozen")
    graph = _load_execution_graph(run.execution_graph)
    state = run.leaf_states.get(leaf_id)
    if state is None:
        raise AssetCompositionStateError(
            f"Leaf {leaf_id!r} is not selected by the frozen graph"
        )
    expected_attempt = max(1, state.attempt_count)
    if attempt is not None and attempt != expected_attempt:
        raise AssetCompositionStateError(
            f"Leaf {leaf_id} attempt is {expected_attempt}, not {attempt}"
        )
    selected_attempt = expected_attempt
    return _graph_leaf_directory(
        state_path,
        graph,
        leaf_id,
        attempt=selected_attempt,
    )


def _append_leaf_transition(
    run: AssetCompositionRun,
    *,
    leaf_id: str,
    from_status: str,
    to_status: str,
    reason: str,
    actor: str,
) -> None:
    state = run.leaf_states[leaf_id]
    run.leaf_transitions.append(
        AssetLeafTransition(
            timestamp=_timestamp(),
            leaf_id=leaf_id,
            from_status=cast(Any, from_status),
            to_status=cast(Any, to_status),
            reason=reason,
            actor=actor,
            attempt_count=state.attempt_count,
        )
    )


def freeze_execution_graph(
    path: str | Path,
    *,
    graph_path: str | Path,
    actor: str = "asset-coordinator",
) -> AssetCompositionRun:
    """Validate and bind the exact graph supplied by the sole outer reasoner."""

    state_path = _resolved(path)
    root = _run_root(state_path)
    graph_binding = _binding(
        graph_path,
        label="outer-supplied execution graph",
        required_root=root,
    )
    graph = _load_execution_graph(graph_binding)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        request = load_verified_asset_request(state_path, run=run)
        if run.selected_mode != "agentic" or request.selected_mode != "agentic":
            raise AssetCompositionStateError(
                "Execution graph is accepted only in agentic mode"
            )
        if run.execution_graph is not None:
            raise AssetCompositionStateError("Execution graph is already frozen")
        if run.coordinator.next_action != "freeze_graph":
            raise AssetCompositionStateError(
                "Coordinator is not awaiting an outer-supplied graph"
            )
        coordinator = request.sole_coordinator_identity
        catalog = request.leaf_catalog
        if coordinator is None or catalog is None:
            raise AssetCompositionStateError(
                "Agentic request lacks coordinator or leaf catalog identity"
            )
        if (
            graph.sole_coordinator_identity_digest != coordinator.identity_digest
            or graph.prompt_digest != request.prompt_digest
            or graph.source_digest != request.source_digest
            or graph.configuration_digest != request.configuration_digest
            or graph.reference_digest != request.reference_digest
            or graph.leaf_catalog_digest != catalog.catalog_digest
        ):
            raise AssetCompositionStateError(
                "Outer-supplied graph identity differs from the frozen request"
            )
        _validate_graph_catalog(
            graph,
            catalog,
            required_leaf_ids=request.required_leaf_ids,
            required_terminal_leaf_ids=request.required_terminal_leaf_ids,
            required_leaf_dependencies=request.required_leaf_dependencies,
            exact_leaf_scope=request.exact_leaf_scope,
        )
        legacy_graph = (
            graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
        )
        if not legacy_graph:
            for leaf_id in graph.selected_leaf_ids:
                _resolve_graph_leaf_runtime_binding(request, graph, leaf_id)
        run.execution_graph = graph_binding
        run.graph_started_at = _timestamp()
        run.current_leaf_id = graph.selected_leaf_ids[0] if legacy_graph else None
        run.leaf_states = {
            node.leaf_id: AssetLeafState(
                leaf_id=node.leaf_id,
                requirement=node.requirement,
                depends_on=node.depends_on,
                terminal_output=node.terminal_output,
                status=(
                    "ready"
                    if (
                        node.leaf_id == graph.selected_leaf_ids[0]
                        if legacy_graph
                        else not node.depends_on
                    )
                    else "pending"
                ),
            )
            for node in graph.nodes
        }
        run.coordinator.next_action = "begin_leaf"
        for node in graph.nodes:
            if legacy_graph and node.leaf_id != graph.selected_leaf_ids[0]:
                continue
            if not legacy_graph and node.depends_on:
                continue
            _append_leaf_transition(
                run,
                leaf_id=node.leaf_id,
                from_status="pending",
                to_status="ready",
                reason=(
                    "The sole outer coordinator froze the exact execution graph."
                    if legacy_graph
                    else "The sole outer coordinator froze a dependency-ready leaf."
                ),
                actor=actor,
            )
        return _write_run(state_path, run)


def begin_leaf(
    path: str | Path,
    leaf_id: str,
    *,
    actor: str = "asset-coordinator",
) -> AssetCompositionRun:
    """Begin one dependency-ready leaf from the outer-frozen graph DAG."""

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.execution_graph is None or run.selected_mode != "agentic":
            raise AssetCompositionStateError("Agentic execution graph is not frozen")
        graph = _load_execution_graph(run.execution_graph)
        if graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION:
            if run.current_leaf_id != leaf_id:
                raise AssetCompositionStateError(
                    f"Current leaf is {run.current_leaf_id!r}, not {leaf_id!r}"
                )
        elif run.current_leaf_id is not None:
            raise AssetCompositionStateError(
                f"Another graph leaf is already active: {run.current_leaf_id!r}"
            )
        if run.coordinator.next_action != "begin_leaf":
            raise AssetCompositionStateError(
                "Current graph state does not allow begin-leaf"
            )
        state = run.leaf_states[leaf_id]
        if state.status != "ready":
            raise AssetCompositionStateError(
                f"Cannot begin {leaf_id} from {state.status}; expected ready"
            )
        missing = [
            dependency
            for dependency in state.depends_on
            if run.leaf_states[dependency].status != "passed"
        ]
        if missing:
            raise AssetCompositionStateError(
                f"Leaf {leaf_id} has unsatisfied dependencies: {missing}"
            )
        previous = state.status
        state.status = "running"
        state.attempt_count += 1
        state.started_at = _timestamp()
        state.error = None
        run.current_leaf_id = leaf_id
        attempt_root = _graph_leaf_directory(
            state_path,
            graph,
            leaf_id,
            attempt=state.attempt_count,
        )
        attempt_root.mkdir(parents=True, exist_ok=False)
        run.coordinator.next_action = "execute_leaf"
        _append_leaf_transition(
            run,
            leaf_id=leaf_id,
            from_status=previous,
            to_status="running",
            reason="Frozen dependencies and input identities were revalidated.",
            actor=actor,
        )
        return _write_run(state_path, run)


def _project_active_leaf(
    *,
    request: AssetRunRequest,
    graph: AssetExecutionGraph,
    leaf_id: str,
    attempt_root: Path,
    invocation_path: str | Path,
    result_path: str | Path,
    expected_dispositions: set[str],
    expected_error: str | None,
) -> tuple[ArtifactBinding, ArtifactBinding, AssetLeafProjection, ArtifactBinding]:
    runtime_binding = _resolve_graph_leaf_runtime_binding(request, graph, leaf_id)
    invocation_artifact = _binding(
        invocation_path,
        label=f"{leaf_id} invocation",
        required_root=attempt_root,
    )
    result_artifact = _binding(
        result_path,
        label=f"{leaf_id} result",
        required_root=attempt_root,
    )
    try:
        invocation = runtime_binding.invocation_model.model_validate(
            _json_object_from_binding(
                invocation_artifact,
                label=f"{leaf_id} invocation",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"{leaf_id} invocation violates its frozen schema digest"
        ) from exc
    try:
        result = runtime_binding.result_model.model_validate(
            _json_object_from_binding(
                result_artifact,
                label=f"{leaf_id} result",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"{leaf_id} result violates its frozen schema digest"
        ) from exc
    try:
        projection = runtime_binding.project(
            invocation,
            result,
            invocation_artifact=invocation_artifact,
            result_artifact=result_artifact,
        )
    except (TypeError, ValueError) as exc:
        raise AssetCompositionStateError(
            f"{leaf_id} deterministic projector rejected the native result"
        ) from exc
    projected = projection.payload
    if projected.native_disposition not in expected_dispositions:
        raise AssetCompositionStateError(
            f"{leaf_id} projected native disposition "
            f"{projected.native_disposition!r} is invalid for this transition"
        )
    if projected.error != expected_error:
        raise AssetCompositionStateError(
            f"{leaf_id} projected native terminal error differs from this transition"
        )
    if projected.native_disposition == "not_evaluated":
        node = next(node for node in graph.nodes if node.leaf_id == leaf_id)
        if node.requirement != "optional":
            raise AssetCompositionStateError(
                f"Required leaf {leaf_id} cannot be not_evaluated"
            )
        if any(leaf_id in node.depends_on for node in graph.nodes):
            raise AssetCompositionStateError(
                f"not_evaluated leaf {leaf_id} is required by another selected leaf"
            )
    for label, artifacts in (
        ("native terminal receipt", (projected.native_terminal_receipt,)),
        ("operation index", projected.operation_indexes),
        ("evidence index", projected.evidence_indexes),
        ("evidence", projected.evidence),
        ("saved-stage readback", projected.saved_stage_readbacks),
        ("resource release", projected.resource_release_receipts),
    ):
        for index, artifact in enumerate(artifacts, start=1):
            _verify_binding(
                artifact,
                label=f"{leaf_id} projected {label} {index}",
                required_root=attempt_root,
            )
    projection_path = attempt_root / "leaf_projection.json"
    if projection_path.exists() or projection_path.is_symlink():
        raise AssetCompositionStateError(
            f"Refusing to replace immutable leaf projection: {projection_path}"
        )
    atomic_write_json(projection_path, projection)
    projection_artifact = _binding(
        projection_path,
        label=f"{leaf_id} projection",
        required_root=attempt_root,
    )
    return invocation_artifact, result_artifact, projection, projection_artifact


def _activate_ready_graph_leaves(
    run: AssetCompositionRun,
    graph: AssetExecutionGraph,
    *,
    completed_leaf_id: str,
    actor: str,
) -> None:
    for node in graph.nodes:
        state = run.leaf_states[node.leaf_id]
        if state.status != "pending" or not all(
            run.leaf_states[dependency].status == "passed"
            for dependency in node.depends_on
        ):
            continue
        state.status = "ready"
        _append_leaf_transition(
            run,
            leaf_id=node.leaf_id,
            from_status="pending",
            to_status="ready",
            reason=(f"Frozen dependencies became satisfied after {completed_leaf_id}."),
            actor=actor,
        )
    statuses = {state.status for state in run.leaf_states.values()}
    if statuses.issubset({"passed", "not_evaluated"}):
        run.coordinator.next_action = "finalize_receipts"
    elif "ready" in statuses:
        run.coordinator.next_action = "begin_leaf"
    else:
        raise AssetCompositionStateError(
            "Frozen graph has no dependency-ready leaf and cannot advance"
        )


def _bind_legacy_leaf_artifacts(
    paths: Sequence[str | Path],
    *,
    label: str,
    attempt_root: Path,
) -> list[ArtifactBinding]:
    return [
        _binding(
            item,
            label=f"{label} {index}",
            required_root=attempt_root,
        )
        for index, item in enumerate(paths, start=1)
    ]


def _complete_legacy_leaf(
    *,
    state_path: Path,
    run: AssetCompositionRun,
    graph: AssetExecutionGraph,
    leaf_id: str,
    invocation_path: str | Path,
    result_path: str | Path,
    native_terminal_receipt_path: str | Path | None,
    operation_index_paths: Sequence[str | Path],
    evidence_index_paths: Sequence[str | Path],
    evidence_paths: Sequence[str | Path],
    saved_stage_readback_paths: Sequence[str | Path],
    native_disposition: Literal["passed", "not_evaluated"] | None,
    resource_claims: Sequence[str],
    resource_release_paths: Sequence[str | Path],
    summary: str | None,
    actor: str,
) -> AssetCompositionRun:
    if native_disposition is None or summary is None:
        raise AssetCompositionStateError(
            "Graph v1 complete-leaf requires compatibility-only native "
            "disposition and summary"
        )
    normalized_summary = summary.strip()
    if not normalized_summary:
        raise AssetCompositionStateError("Leaf summary must not be empty")
    state = run.leaf_states[leaf_id]
    if native_disposition == "not_evaluated" and state.requirement != "optional":
        raise AssetCompositionStateError(
            f"Required leaf {leaf_id} cannot be not_evaluated"
        )
    index = graph.selected_leaf_ids.index(leaf_id)
    if native_disposition != "passed" and any(
        leaf_id in node.depends_on for node in graph.nodes[index + 1 :]
    ):
        raise AssetCompositionStateError(
            f"not_evaluated leaf {leaf_id} is required by a later leaf"
        )
    attempt_root = _graph_leaf_directory(
        state_path,
        graph,
        leaf_id,
        attempt=state.attempt_count,
    )
    invocation = _binding(
        invocation_path,
        label=f"{leaf_id} invocation",
        required_root=attempt_root,
    )
    result = _binding(
        result_path,
        label=f"{leaf_id} result",
        required_root=attempt_root,
    )
    native_terminal_receipt = (
        _binding(
            native_terminal_receipt_path,
            label=f"{leaf_id} native terminal receipt",
            required_root=attempt_root,
        )
        if native_terminal_receipt_path is not None
        else None
    )
    operation_indexes = _bind_legacy_leaf_artifacts(
        operation_index_paths,
        label=f"{leaf_id} operation index",
        attempt_root=attempt_root,
    )
    evidence_indexes = _bind_legacy_leaf_artifacts(
        evidence_index_paths,
        label=f"{leaf_id} evidence index",
        attempt_root=attempt_root,
    )
    evidence = _bind_legacy_leaf_artifacts(
        evidence_paths,
        label=f"{leaf_id} evidence",
        attempt_root=attempt_root,
    )
    readbacks = _bind_legacy_leaf_artifacts(
        saved_stage_readback_paths,
        label=f"{leaf_id} saved-stage readback",
        attempt_root=attempt_root,
    )
    releases = _bind_legacy_leaf_artifacts(
        resource_release_paths,
        label=f"{leaf_id} resource release",
        attempt_root=attempt_root,
    )
    finished_at = _timestamp()
    started_at = datetime.fromisoformat(
        cast(str, state.started_at).replace("Z", "+00:00")
    )
    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    receipt = AssetLeafReceipt(
        schema_version=LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
        run_id=run.run_id,
        run_revision=run.revision,
        graph_digest=graph.graph_digest,
        leaf_catalog_digest=graph.leaf_catalog_digest,
        sole_coordinator_identity_digest=graph.sole_coordinator_identity_digest,
        leaf_id=leaf_id,
        requirement=state.requirement,
        depends_on=state.depends_on,
        dependency_receipts={
            dependency: cast(ArtifactBinding, run.leaf_states[dependency].receipt)
            for dependency in state.depends_on
        },
        attempt=state.attempt_count,
        invocation=invocation,
        result=result,
        native_terminal_receipt=native_terminal_receipt,
        operation_indexes=operation_indexes,
        evidence_indexes=evidence_indexes,
        evidence=evidence,
        saved_stage_readbacks=readbacks,
        native_disposition=native_disposition,
        started_at=cast(str, state.started_at),
        finished_at=finished_at,
        duration_ms=max(0, int((finished - started_at).total_seconds() * 1000)),
        resource_claims=list(resource_claims),
        resource_release_receipts=releases,
        supersedes=(
            state.superseded_receipts[-1] if state.superseded_receipts else None
        ),
        summary=normalized_summary,
        actor=actor,
    )
    receipt_path = attempt_root / "leaf_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise AssetCompositionStateError(
            f"Refusing to replace immutable leaf receipt: {receipt_path}"
        )
    atomic_write_json(receipt_path, receipt)
    state.receipt = _binding(
        receipt_path,
        label=f"{leaf_id} receipt",
        required_root=attempt_root,
    )
    previous = state.status
    state.status = native_disposition
    state.started_at = None
    _append_leaf_transition(
        run,
        leaf_id=leaf_id,
        from_status=previous,
        to_status=native_disposition,
        reason=normalized_summary,
        actor=actor,
    )
    if index + 1 == len(graph.selected_leaf_ids):
        run.current_leaf_id = None
        run.coordinator.next_action = "finalize_receipts"
    else:
        successor = graph.selected_leaf_ids[index + 1]
        successor_state = run.leaf_states[successor]
        if successor_state.status != "pending":
            raise AssetCompositionStateError(
                f"Successor leaf {successor} is unexpectedly {successor_state.status}"
            )
        successor_state.status = "ready"
        run.current_leaf_id = successor
        run.coordinator.next_action = "begin_leaf"
        _append_leaf_transition(
            run,
            leaf_id=successor,
            from_status="pending",
            to_status="ready",
            reason=f"Frozen predecessor {leaf_id} reached its native disposition.",
            actor=actor,
        )
    return _write_run(state_path, run)


def complete_leaf(
    path: str | Path,
    leaf_id: str,
    *,
    invocation_path: str | Path,
    result_path: str | Path,
    native_terminal_receipt_path: str | Path | None = None,
    operation_index_paths: Sequence[str | Path] = (),
    evidence_index_paths: Sequence[str | Path] = (),
    evidence_paths: Sequence[str | Path] = (),
    saved_stage_readback_paths: Sequence[str | Path] = (),
    native_disposition: Literal["passed", "not_evaluated"] | None = None,
    resource_claims: Sequence[str] = (),
    resource_release_paths: Sequence[str | Path] = (),
    summary: str | None = None,
    actor: str = "asset-coordinator",
) -> AssetCompositionRun:
    """Seal one version-matched leaf without upgrading its frozen graph."""

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        request = load_verified_asset_request(state_path, run=run)
        if run.execution_graph is None or run.current_leaf_id != leaf_id:
            raise AssetCompositionStateError(f"{leaf_id} is not the active graph leaf")
        if run.coordinator.next_action != "execute_leaf":
            raise AssetCompositionStateError(
                "Current graph state does not allow complete-leaf"
            )
        graph = _load_execution_graph(run.execution_graph)
        state = run.leaf_states[leaf_id]
        if state.status != "running" or state.started_at is None:
            raise AssetCompositionStateError(
                f"Cannot complete {leaf_id} from {state.status}; expected running"
            )
        if graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION:
            return _complete_legacy_leaf(
                state_path=state_path,
                run=run,
                graph=graph,
                leaf_id=leaf_id,
                invocation_path=invocation_path,
                result_path=result_path,
                native_terminal_receipt_path=native_terminal_receipt_path,
                operation_index_paths=operation_index_paths,
                evidence_index_paths=evidence_index_paths,
                evidence_paths=evidence_paths,
                saved_stage_readback_paths=saved_stage_readback_paths,
                native_disposition=native_disposition,
                resource_claims=resource_claims,
                resource_release_paths=resource_release_paths,
                summary=summary,
                actor=actor,
            )
        if (
            native_terminal_receipt_path is not None
            or operation_index_paths
            or evidence_index_paths
            or evidence_paths
            or saved_stage_readback_paths
            or native_disposition is not None
            or resource_claims
            or resource_release_paths
            or summary is not None
        ):
            raise AssetCompositionStateError(
                "Graph v2 complete-leaf rejects compatibility-only artifact flags"
            )
        attempt_root = _graph_leaf_directory(
            state_path,
            graph,
            leaf_id,
            attempt=state.attempt_count,
        )
        invocation, result, projection, projection_artifact = _project_active_leaf(
            request=request,
            graph=graph,
            leaf_id=leaf_id,
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            result_path=result_path,
            expected_dispositions={"passed", "not_evaluated"},
            expected_error=None,
        )
        payload = projection.payload
        runtime_binding = _resolve_graph_leaf_runtime_binding(request, graph, leaf_id)
        finished_at = _timestamp()
        started_at = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        receipt = AssetLeafReceipt(
            run_id=run.run_id,
            run_revision=run.revision,
            graph_digest=graph.graph_digest,
            leaf_catalog_digest=graph.leaf_catalog_digest,
            sole_coordinator_identity_digest=graph.sole_coordinator_identity_digest,
            leaf_id=leaf_id,
            descriptor_digest=runtime_binding.descriptor.descriptor_digest,
            invocation_schema_digest=(
                runtime_binding.descriptor.invocation_schema_digest
            ),
            result_schema_digest=runtime_binding.descriptor.result_schema_digest,
            projection_schema_digest=(
                runtime_binding.descriptor.projection_schema_digest
            ),
            projector_id=runtime_binding.descriptor.projector_id,
            projector_digest=runtime_binding.descriptor.projector_digest,
            required_artifact_categories=(
                runtime_binding.descriptor.required_artifact_categories
            ),
            requirement=state.requirement,
            depends_on=state.depends_on,
            dependency_receipts={
                dependency: cast(
                    ArtifactBinding,
                    run.leaf_states[dependency].receipt,
                )
                for dependency in state.depends_on
            },
            attempt=state.attempt_count,
            invocation=invocation,
            result=result,
            projection=projection_artifact,
            native_terminal_receipt=payload.native_terminal_receipt,
            operation_indexes=list(payload.operation_indexes),
            evidence_indexes=list(payload.evidence_indexes),
            evidence=list(payload.evidence),
            saved_stage_readbacks=list(payload.saved_stage_readbacks),
            native_disposition=payload.native_disposition,
            native_status=payload.native_status,
            started_at=state.started_at,
            finished_at=finished_at,
            duration_ms=max(0, int((finished - started_at).total_seconds() * 1000)),
            resource_claims=list(payload.resource_claims),
            resource_release_receipts=list(payload.resource_release_receipts),
            supersedes=(
                state.superseded_receipts[-1] if state.superseded_receipts else None
            ),
            summary=payload.summary,
            actor=actor,
        )
        receipt_path = attempt_root / "leaf_receipt.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            raise AssetCompositionStateError(
                f"Refusing to replace immutable leaf receipt: {receipt_path}"
            )
        atomic_write_json(receipt_path, receipt)
        receipt_binding = _binding(
            receipt_path,
            label=f"{leaf_id} receipt",
            required_root=attempt_root,
        )
        previous = state.status
        state.status = payload.native_disposition
        state.receipt = receipt_binding
        state.started_at = None
        run.current_leaf_id = None
        _append_leaf_transition(
            run,
            leaf_id=leaf_id,
            from_status=previous,
            to_status=payload.native_disposition,
            reason=payload.summary,
            actor=actor,
        )
        _activate_ready_graph_leaves(
            run,
            graph,
            completed_leaf_id=leaf_id,
            actor=actor,
        )
        return _write_run(state_path, run)


def _stop_legacy_leaf(
    *,
    state_path: Path,
    run: AssetCompositionRun,
    graph: AssetExecutionGraph,
    leaf_id: str,
    reason: str,
    disposition: Literal["failed", "cancelled"],
    invocation_path: str | Path | None,
    result_path: str | Path | None,
    native_terminal_receipt_path: str | Path | None,
    operation_index_paths: Sequence[str | Path],
    evidence_index_paths: Sequence[str | Path],
    evidence_paths: Sequence[str | Path],
    saved_stage_readback_paths: Sequence[str | Path],
    resource_claims: Sequence[str],
    resource_release_paths: Sequence[str | Path],
    actor: str,
) -> AssetCompositionRun:
    exact_native_artifacts = any(
        (
            result_path is not None,
            native_terminal_receipt_path is not None,
            bool(operation_index_paths),
            bool(evidence_index_paths),
            bool(evidence_paths),
            bool(saved_stage_readback_paths),
            bool(resource_claims),
            bool(resource_release_paths),
        )
    )
    if exact_native_artifacts and invocation_path is None:
        raise AssetCompositionStateError(
            "Native terminal artifacts require their exact leaf invocation"
        )
    state = run.leaf_states[leaf_id]
    if state.status not in {"ready", "running"}:
        raise AssetCompositionStateError(
            f"Cannot mark {leaf_id} {disposition} from {state.status}"
        )
    if state.status == "ready" and invocation_path is not None:
        raise AssetCompositionStateError(
            "Begin the leaf before binding native terminal artifacts"
        )
    if state.status == "ready":
        state.attempt_count += 1
        state.started_at = _timestamp()
        attempt_root = _graph_leaf_directory(
            state_path,
            graph,
            leaf_id,
            attempt=state.attempt_count,
        )
        attempt_root.mkdir(parents=True, exist_ok=False)
    else:
        attempt_root = _graph_leaf_directory(
            state_path,
            graph,
            leaf_id,
            attempt=state.attempt_count,
        )
    if invocation_path is None:
        invocation_path = attempt_root / "failed_invocation.json"
        atomic_write_json(
            invocation_path,
            {
                "schema_version": "content-agent-workflows.asset-leaf-failure.v1",
                "leaf_id": leaf_id,
                "reason": reason,
                "actor": actor,
            },
        )
    invocation = _binding(
        invocation_path,
        label=f"{leaf_id} failed invocation",
        required_root=attempt_root,
    )
    result = (
        _binding(
            result_path,
            label=f"{leaf_id} native terminal result",
            required_root=attempt_root,
        )
        if result_path is not None
        else None
    )
    native_terminal_receipt = (
        _binding(
            native_terminal_receipt_path,
            label=f"{leaf_id} native terminal receipt",
            required_root=attempt_root,
        )
        if native_terminal_receipt_path is not None
        else None
    )
    operation_indexes = _bind_legacy_leaf_artifacts(
        operation_index_paths,
        label=f"{leaf_id} operation index",
        attempt_root=attempt_root,
    )
    evidence_indexes = _bind_legacy_leaf_artifacts(
        evidence_index_paths,
        label=f"{leaf_id} evidence index",
        attempt_root=attempt_root,
    )
    evidence = _bind_legacy_leaf_artifacts(
        evidence_paths,
        label=f"{leaf_id} evidence",
        attempt_root=attempt_root,
    )
    readbacks = _bind_legacy_leaf_artifacts(
        saved_stage_readback_paths,
        label=f"{leaf_id} saved-stage readback",
        attempt_root=attempt_root,
    )
    releases = _bind_legacy_leaf_artifacts(
        resource_release_paths,
        label=f"{leaf_id} resource release",
        attempt_root=attempt_root,
    )
    finished_at = _timestamp()
    started_text = state.started_at or finished_at
    started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    receipt = AssetLeafReceipt(
        schema_version=LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
        run_id=run.run_id,
        run_revision=run.revision,
        graph_digest=graph.graph_digest,
        leaf_catalog_digest=graph.leaf_catalog_digest,
        sole_coordinator_identity_digest=graph.sole_coordinator_identity_digest,
        leaf_id=leaf_id,
        requirement=state.requirement,
        depends_on=state.depends_on,
        dependency_receipts={
            dependency: cast(ArtifactBinding, run.leaf_states[dependency].receipt)
            for dependency in state.depends_on
        },
        attempt=state.attempt_count,
        invocation=invocation,
        result=result,
        native_terminal_receipt=native_terminal_receipt,
        operation_indexes=operation_indexes,
        evidence_indexes=evidence_indexes,
        evidence=evidence,
        saved_stage_readbacks=readbacks,
        native_disposition=disposition,
        started_at=started_text,
        finished_at=finished_at,
        duration_ms=max(0, int((finished - started).total_seconds() * 1000)),
        resource_claims=list(resource_claims),
        resource_release_receipts=releases,
        supersedes=(
            state.superseded_receipts[-1] if state.superseded_receipts else None
        ),
        summary=reason,
        error=reason,
        actor=actor,
    )
    receipt_path = attempt_root / "leaf_receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise AssetCompositionStateError(
            f"Refusing to replace immutable leaf receipt: {receipt_path}"
        )
    atomic_write_json(receipt_path, receipt)
    state.receipt = _binding(
        receipt_path,
        label=f"{leaf_id} receipt",
        required_root=attempt_root,
    )
    previous = state.status
    state.status = disposition
    state.started_at = None
    state.error = reason
    run.terminal_status = "cancelled" if disposition == "cancelled" else "failed"
    run.coordinator.next_action = "stopped"
    run.coordinator.stop_reason = reason
    _append_leaf_transition(
        run,
        leaf_id=leaf_id,
        from_status=previous,
        to_status=disposition,
        reason=reason,
        actor=actor,
    )
    return _write_run(state_path, run)


def _stop_leaf(
    path: str | Path,
    leaf_id: str,
    *,
    reason: str,
    disposition: Literal["failed", "cancelled"],
    invocation_path: str | Path | None,
    result_path: str | Path | None,
    native_terminal_receipt_path: str | Path | None,
    operation_index_paths: Sequence[str | Path],
    evidence_index_paths: Sequence[str | Path],
    evidence_paths: Sequence[str | Path],
    saved_stage_readback_paths: Sequence[str | Path],
    resource_claims: Sequence[str],
    resource_release_paths: Sequence[str | Path],
    actor: str,
) -> AssetCompositionRun:
    state_path = _resolved(path)
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise AssetCompositionStateError("Leaf failure reason is required")
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.execution_graph is None or run.current_leaf_id != leaf_id:
            raise AssetCompositionStateError(f"{leaf_id} is not the active graph leaf")
        graph = _load_execution_graph(run.execution_graph)
        if graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION:
            return _stop_legacy_leaf(
                state_path=state_path,
                run=run,
                graph=graph,
                leaf_id=leaf_id,
                reason=normalized_reason,
                disposition=disposition,
                invocation_path=invocation_path,
                result_path=result_path,
                native_terminal_receipt_path=native_terminal_receipt_path,
                operation_index_paths=operation_index_paths,
                evidence_index_paths=evidence_index_paths,
                evidence_paths=evidence_paths,
                saved_stage_readback_paths=saved_stage_readback_paths,
                resource_claims=resource_claims,
                resource_release_paths=resource_release_paths,
                actor=actor,
            )
        if (
            native_terminal_receipt_path is not None
            or operation_index_paths
            or evidence_index_paths
            or evidence_paths
            or saved_stage_readback_paths
            or resource_claims
            or resource_release_paths
        ):
            raise AssetCompositionStateError(
                "Graph v2 stop transition rejects compatibility-only artifact flags"
            )
        if invocation_path is None or result_path is None:
            raise AssetCompositionStateError(
                "Graph v2 stop transition requires exact invocation and result"
            )
        request = load_verified_asset_request(state_path, run=run)
        state = run.leaf_states[leaf_id]
        if state.status != "running" or state.started_at is None:
            raise AssetCompositionStateError(
                f"Cannot mark {leaf_id} {disposition} from {state.status}; "
                "begin the leaf and retain its exact invocation/result first"
            )
        attempt_root = _graph_leaf_directory(
            state_path,
            graph,
            leaf_id,
            attempt=state.attempt_count,
        )
        invocation, result, projection, projection_artifact = _project_active_leaf(
            request=request,
            graph=graph,
            leaf_id=leaf_id,
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            result_path=result_path,
            expected_dispositions={disposition},
            expected_error=normalized_reason,
        )
        payload = projection.payload
        runtime_binding = _resolve_graph_leaf_runtime_binding(request, graph, leaf_id)
        finished_at = _timestamp()
        started_text = state.started_at
        started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        receipt = AssetLeafReceipt(
            run_id=run.run_id,
            run_revision=run.revision,
            graph_digest=graph.graph_digest,
            leaf_catalog_digest=graph.leaf_catalog_digest,
            sole_coordinator_identity_digest=(graph.sole_coordinator_identity_digest),
            leaf_id=leaf_id,
            descriptor_digest=runtime_binding.descriptor.descriptor_digest,
            invocation_schema_digest=(
                runtime_binding.descriptor.invocation_schema_digest
            ),
            result_schema_digest=runtime_binding.descriptor.result_schema_digest,
            projection_schema_digest=(
                runtime_binding.descriptor.projection_schema_digest
            ),
            projector_id=runtime_binding.descriptor.projector_id,
            projector_digest=runtime_binding.descriptor.projector_digest,
            required_artifact_categories=(
                runtime_binding.descriptor.required_artifact_categories
            ),
            requirement=state.requirement,
            depends_on=state.depends_on,
            dependency_receipts={
                dependency: cast(
                    ArtifactBinding,
                    run.leaf_states[dependency].receipt,
                )
                for dependency in state.depends_on
            },
            attempt=state.attempt_count,
            invocation=invocation,
            result=result,
            projection=projection_artifact,
            native_terminal_receipt=payload.native_terminal_receipt,
            operation_indexes=list(payload.operation_indexes),
            evidence_indexes=list(payload.evidence_indexes),
            evidence=list(payload.evidence),
            saved_stage_readbacks=list(payload.saved_stage_readbacks),
            native_disposition=disposition,
            native_status=payload.native_status,
            started_at=started_text,
            finished_at=finished_at,
            duration_ms=max(0, int((finished - started).total_seconds() * 1000)),
            resource_claims=list(payload.resource_claims),
            resource_release_receipts=list(payload.resource_release_receipts),
            supersedes=(
                state.superseded_receipts[-1] if state.superseded_receipts else None
            ),
            summary=payload.summary,
            error=payload.error,
            actor=actor,
        )
        receipt_path = attempt_root / "leaf_receipt.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            raise AssetCompositionStateError(
                f"Refusing to replace immutable leaf receipt: {receipt_path}"
            )
        atomic_write_json(receipt_path, receipt)
        receipt_binding = _binding(
            receipt_path,
            label=f"{leaf_id} receipt",
            required_root=attempt_root,
        )
        previous = state.status
        state.status = disposition
        state.receipt = receipt_binding
        state.started_at = None
        state.error = payload.error
        run.terminal_status = "cancelled" if disposition == "cancelled" else "failed"
        run.coordinator.next_action = "stopped"
        run.coordinator.stop_reason = payload.error
        _append_leaf_transition(
            run,
            leaf_id=leaf_id,
            from_status=previous,
            to_status=disposition,
            reason=payload.summary,
            actor=actor,
        )
        return _write_run(state_path, run)


def fail_leaf(
    path: str | Path,
    leaf_id: str,
    *,
    reason: str,
    invocation_path: str | Path | None = None,
    result_path: str | Path | None = None,
    native_terminal_receipt_path: str | Path | None = None,
    operation_index_paths: Sequence[str | Path] = (),
    evidence_index_paths: Sequence[str | Path] = (),
    evidence_paths: Sequence[str | Path] = (),
    saved_stage_readback_paths: Sequence[str | Path] = (),
    resource_claims: Sequence[str] = (),
    resource_release_paths: Sequence[str | Path] = (),
    actor: str = "asset-coordinator",
) -> AssetCompositionRun:
    return _stop_leaf(
        path,
        leaf_id,
        reason=reason,
        disposition="failed",
        invocation_path=invocation_path,
        result_path=result_path,
        native_terminal_receipt_path=native_terminal_receipt_path,
        operation_index_paths=operation_index_paths,
        evidence_index_paths=evidence_index_paths,
        evidence_paths=evidence_paths,
        saved_stage_readback_paths=saved_stage_readback_paths,
        resource_claims=resource_claims,
        resource_release_paths=resource_release_paths,
        actor=actor,
    )


def cancel_leaf(
    path: str | Path,
    leaf_id: str,
    *,
    reason: str,
    invocation_path: str | Path | None = None,
    result_path: str | Path | None = None,
    native_terminal_receipt_path: str | Path | None = None,
    operation_index_paths: Sequence[str | Path] = (),
    evidence_index_paths: Sequence[str | Path] = (),
    evidence_paths: Sequence[str | Path] = (),
    saved_stage_readback_paths: Sequence[str | Path] = (),
    resource_claims: Sequence[str] = (),
    resource_release_paths: Sequence[str | Path] = (),
    actor: str = "asset-coordinator",
) -> AssetCompositionRun:
    return _stop_leaf(
        path,
        leaf_id,
        reason=reason,
        disposition="cancelled",
        invocation_path=invocation_path,
        result_path=result_path,
        native_terminal_receipt_path=native_terminal_receipt_path,
        operation_index_paths=operation_index_paths,
        evidence_index_paths=evidence_index_paths,
        evidence_paths=evidence_paths,
        saved_stage_readback_paths=saved_stage_readback_paths,
        resource_claims=resource_claims,
        resource_release_paths=resource_release_paths,
        actor=actor,
    )


def recover_leaf(
    path: str | Path,
    leaf_id: str,
    *,
    reason: str,
    actor: str = "operator",
) -> AssetCompositionRun:
    """Supersede one stopped attempt without changing the frozen graph."""

    state_path = _resolved(path)
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise AssetCompositionStateError("Leaf recovery reason is required")
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.execution_graph is None or run.current_leaf_id != leaf_id:
            raise AssetCompositionStateError(f"{leaf_id} is not the current graph leaf")
        state = run.leaf_states[leaf_id]
        if state.status not in {"failed", "cancelled"} or state.receipt is None:
            raise AssetCompositionStateError(
                f"Cannot recover {leaf_id} from {state.status}"
            )
        previous = state.status
        state.superseded_receipts.append(state.receipt)
        state.receipt = None
        state.status = "ready"
        state.error = None
        graph = _load_execution_graph(run.execution_graph)
        run.current_leaf_id = (
            leaf_id
            if graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
            else None
        )
        run.terminal_status = "active"
        run.coordinator.next_action = "begin_leaf"
        run.coordinator.stop_reason = None
        _append_leaf_transition(
            run,
            leaf_id=leaf_id,
            from_status=previous,
            to_status="ready",
            reason=normalized_reason,
            actor=actor,
        )
        return _write_run(state_path, run)


def finalize_graph_run(
    path: str | Path,
    *,
    resource_release_paths: Sequence[str | Path] = (),
    parent_release_receipt_path: str | Path | None = None,
    parent_command_receipt_journal_path: str | Path | None = None,
    parent_command_receipt_checkpoint_path: str | Path | None = None,
    expected_parent_release_receipt: ArtifactBinding | None = None,
    expected_parent_command_receipt_journal: ArtifactBinding | None = None,
    expected_parent_command_receipt_checkpoint: ArtifactBinding | None = None,
    actor: str = "asset-coordinator",
) -> AssetCompositionRun:
    """Seal complete graph, terminal-output, timing, and release identities."""

    state_path = _resolved(path)
    root = _run_root(state_path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        request = load_verified_asset_request(state_path, run=run)
        if run.execution_graph is None or run.graph_started_at is None:
            raise AssetCompositionStateError("Agentic execution graph is not frozen")
        if run.coordinator.next_action != "finalize_receipts":
            raise AssetCompositionStateError(
                "Graph is not awaiting comprehensive receipt finalization"
            )
        graph = _load_execution_graph(run.execution_graph)
        legacy_graph = (
            graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
        )
        expected_leaf_receipt_version = (
            LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION
            if legacy_graph
            else ASSET_LEAF_RECEIPT_SCHEMA_VERSION
        )
        invalid: list[str] = []
        selected_receipts: dict[str, ArtifactBinding] = {}
        terminal_artifacts: list[ArtifactBinding] = []
        leaf_release_receipts: list[ArtifactBinding] = []
        for node in graph.nodes:
            state = run.leaf_states[node.leaf_id]
            if state.requirement == "required" and state.status != "passed":
                invalid.append(f"required leaf {node.leaf_id} is {state.status}")
            elif state.requirement == "optional" and state.status not in {
                "passed",
                "not_evaluated",
            }:
                invalid.append(f"optional leaf {node.leaf_id} is {state.status}")
            if state.receipt is None:
                invalid.append(f"leaf {node.leaf_id} lacks a receipt")
                continue
            selected_receipts[node.leaf_id] = state.receipt
            receipt = _load_leaf_receipt(
                state.receipt,
                label=f"{node.leaf_id} receipt",
            )
            if receipt.schema_version != expected_leaf_receipt_version:
                invalid.append(
                    f"leaf {node.leaf_id} receipt version differs from graph"
                )
            leaf_release_receipts.extend(receipt.resource_release_receipts)
            if node.terminal_output:
                if state.status != "passed" or receipt.result is None:
                    invalid.append(f"terminal output leaf {node.leaf_id} did not pass")
                else:
                    terminal_artifacts.append(receipt.result)
        parent_paths = (
            parent_release_receipt_path,
            parent_command_receipt_journal_path,
            parent_command_receipt_checkpoint_path,
        )
        if legacy_graph:
            if any(item is not None for item in parent_paths) or any(
                item is not None
                for item in (
                    expected_parent_release_receipt,
                    expected_parent_command_receipt_journal,
                    expected_parent_command_receipt_checkpoint,
                )
            ):
                invalid.append("graph v1 rejects graph-v2 parent receipt flags")
        else:
            if resource_release_paths:
                invalid.append("graph v2 rejects compatibility-only resource releases")
            if any(item is None for item in parent_paths):
                invalid.append(
                    "graph v2 requires launcher release receipt, journal, and checkpoint"
                )
            if any(
                item is None
                for item in (
                    expected_parent_release_receipt,
                    expected_parent_command_receipt_journal,
                    expected_parent_command_receipt_checkpoint,
                )
            ):
                invalid.append(
                    "graph v2 finalization requires launcher-bound expected parent "
                    "receipt identities"
                )
        legacy_parent_releases = (
            [
                _binding(
                    item,
                    label=f"graph resource release {index}",
                    required_root=root,
                )
                for index, item in enumerate(resource_release_paths, start=1)
            ]
            if legacy_graph
            else []
        )
        if (
            legacy_graph
            and request.requires_parent_resource_release
            and not legacy_parent_releases
        ):
            invalid.append("parent-owned resource release receipt is missing")
        if invalid:
            raise AssetCompositionStateError(
                "Cannot finalize agentic graph: " + "; ".join(invalid)
            )
        parent_release = (
            _binding(
                parent_release_receipt_path,
                label="graph parent release receipt",
                required_root=root,
            )
            if parent_release_receipt_path is not None
            else None
        )
        parent_journal = (
            _binding(
                parent_command_receipt_journal_path,
                label="graph parent command receipt journal",
                required_root=root,
            )
            if parent_command_receipt_journal_path is not None
            else None
        )
        parent_checkpoint = (
            _binding(
                parent_command_receipt_checkpoint_path,
                label="graph parent command receipt checkpoint",
                required_root=root,
            )
            if parent_command_receipt_checkpoint_path is not None
            else None
        )
        for label, observed, expected in (
            (
                "parent release receipt",
                parent_release,
                expected_parent_release_receipt,
            ),
            (
                "parent command receipt journal",
                parent_journal,
                expected_parent_command_receipt_journal,
            ),
            (
                "parent command receipt checkpoint",
                parent_checkpoint,
                expected_parent_command_receipt_checkpoint,
            ),
        ):
            if expected is not None and observed != expected:
                invalid.append(f"{label} identity was substituted or drifted")
        if invalid:
            raise AssetCompositionStateError(
                "Cannot finalize agentic graph: " + "; ".join(invalid)
            )
        release_receipts: list[ArtifactBinding] = []
        seen_releases: set[tuple[str, str]] = set()
        for binding in [
            *leaf_release_receipts,
            *legacy_parent_releases,
            *([parent_release] if parent_release is not None else []),
        ]:
            identity = (binding.path, binding.sha256)
            if identity not in seen_releases:
                release_receipts.append(binding)
                seen_releases.add(identity)
        finished_at = _timestamp()
        started = datetime.fromisoformat(run.graph_started_at.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
        terminal_payload: dict[str, Any] = {
            "schema_version": (
                LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
                if legacy_graph
                else ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
            ),
            "run_id": run.run_id,
            "graph": run.execution_graph,
            "graph_digest": graph.graph_digest,
            "leaf_catalog_digest": graph.leaf_catalog_digest,
            "sole_coordinator_identity_digest": (
                graph.sole_coordinator_identity_digest
            ),
            "prompt_digest": graph.prompt_digest,
            "source_digest": graph.source_digest,
            "configuration_digest": graph.configuration_digest,
            "reference_digest": graph.reference_digest,
            "selected_leaf_ids": graph.selected_leaf_ids,
            "selected_leaf_receipts": selected_receipts,
            "omitted_leaf_ids": graph.omitted_leaf_ids,
            "omitted_leaf_dispositions": {
                leaf_id: "not_requested" for leaf_id in graph.omitted_leaf_ids
            },
            "terminal_artifacts": terminal_artifacts,
            "resource_release_receipts": release_receipts,
            "started_at": run.graph_started_at,
            "finished_at": finished_at,
            "duration_ms": max(0, int((finished - started).total_seconds() * 1000)),
            "actor": actor,
        }
        if not legacy_graph:
            terminal_payload.update(
                {
                    "parent_release_receipt": parent_release,
                    "parent_command_receipt_journal": parent_journal,
                    "parent_command_receipt_checkpoint": parent_checkpoint,
                }
            )
        terminal = AssetGraphTerminalReceipt.model_validate(terminal_payload)
        receipt_path = root / "graph_terminal_receipt.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            raise AssetCompositionStateError(
                f"Refusing to replace graph terminal receipt: {receipt_path}"
            )
        atomic_write_json(receipt_path, terminal)
        run.graph_terminal_receipt = _binding(
            receipt_path,
            label="graph terminal receipt",
            required_root=root,
        )
        run.terminal_status = "completed"
        run.coordinator.next_action = "terminal"
        return _write_run(state_path, run)


def activate_single_reasoning_loop(
    path: str | Path,
) -> AssetCompositionRun:
    """Upgrade a legacy durable run before entering the shared coordinator."""

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        coordinator = run.coordinator
        if coordinator.mode == "single_reasoning_loop":
            return run
        request_payload = _json_object_from_binding(
            run.request,
            label="legacy composed asset request",
        )
        if request_payload.get("coordinator_mode") is None:
            _normalize_historical_asset_request_payload(request_payload)
            try:
                AssetRunRequest.model_validate(request_payload)
            except ValidationError as exc:
                raise AssetCompositionStateError(
                    f"Legacy request cannot be upgraded to the single loop: {exc}"
                ) from exc
            request_path = Path(run.request.path)
            atomic_write_json(request_path, request_payload)
            run.request = _binding(
                request_path,
                label="upgraded composed asset request",
                required_root=_run_root(state_path),
            )
        coordinator.mode = "single_reasoning_loop"
        if run.terminal_status == "completed":
            coordinator.next_action = "terminal"
        elif run.terminal_status in {"failed", "cancelled"}:
            coordinator.next_action = "stopped"
            current = run.current_stage
            coordinator.stop_reason = (
                run.stages[current].error
                if current is not None and run.stages[current].error
                else f"Legacy run is {run.terminal_status}."
            )
        else:
            if run.current_stage is None:
                raise AssetCompositionStateError(
                    "Active legacy run lacks a current stage"
                )
            status = run.stages[run.current_stage].status
            coordinator.next_action = (
                "await_human_review" if status == "needs_review" else "plan"
            )
        return _write_run(state_path, run)


def begin_stage(
    path: str | Path,
    stage: StageName,
    *,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Begin the current ready stage after revalidating all accepted work."""

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.terminal_status != "active":
            raise AssetCompositionStateError(
                f"Cannot begin a stage while run is {run.terminal_status}"
            )
        if run.current_stage != stage:
            raise AssetCompositionStateError(
                f"Current stage is {run.current_stage}, not {stage}"
            )
        state = run.stages[stage]
        if state.status != "ready":
            raise AssetCompositionStateError(
                f"Cannot begin {stage} from {state.status}; expected ready"
            )
        if run.coordinator.mode == "single_reasoning_loop":
            if run.coordinator.next_action != "begin_stage":
                raise AssetCompositionStateError(
                    "Current stage requires a coordinator plan before begin-stage"
                )
            if not run.coordinator.plan_revisions:
                raise AssetCompositionStateError(
                    "Current stage requires a coordinator plan before begin-stage"
                )
            plan = _load_coordinator_plan(run.coordinator.plan_revisions[-1])
            if plan.stage != stage or plan.stage_attempt != _planned_attempt_number(
                state
            ):
                raise AssetCompositionStateError(
                    "Latest coordinator plan does not match the next stage attempt"
                )
        previous = state.status
        state.status = "running"
        if not state.continue_current_attempt:
            state.attempt_count += 1
        state.continue_current_attempt = False
        _ensure_stage_directory(
            state_path,
            stage,
            attempt=state.attempt_count,
        )
        state.error = None
        if run.coordinator.mode == "single_reasoning_loop":
            run.coordinator.next_action = "execute_stage"
        _transition(
            run,
            stage,
            from_status=previous,
            to_status="running",
            reason="Current input and every accepted predecessor were verified.",
            actor=actor,
        )
        return _write_run(state_path, run)


def require_review(
    path: str | Path,
    *,
    candidates_path: str | Path,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Pause articulation before authoring and bind the exact candidate evidence."""

    state_path = _resolved(path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        stage: StageName = "articulation"
        if run.current_stage != stage or run.stages[stage].status != "running":
            raise AssetCompositionStateError(
                "Articulation must be the running stage before review is required"
            )
        candidates = _binding(
            candidates_path,
            label="articulation review candidates",
            required_root=_run_root(state_path),
        )
        state = run.stages[stage]
        if run.coordinator.mode == "single_reasoning_loop":
            if (
                run.coordinator.next_action != "await_human_review"
                or not run.coordinator.evidence_reviews
            ):
                raise AssetCompositionStateError(
                    "Articulation candidates require a coordinator await_review decision"
                )
            review = _load_coordinator_review(run.coordinator.evidence_reviews[-1])
            if (
                review.stage != stage
                or review.stage_attempt != state.attempt_count
                or review.decision != "await_review"
                or candidates not in review.evidence
            ):
                raise AssetCompositionStateError(
                    "Latest coordinator review does not bind these Joint candidates"
                )
        previous = state.status
        state.status = "needs_review"
        state.review_candidates = candidates
        _transition(
            run,
            stage,
            from_status=previous,
            to_status="needs_review",
            reason="Exact articulation candidates require user review before authoring.",
            actor=actor,
        )
        return _write_run(state_path, run)


def record_review_decisions(
    path: str | Path,
    *,
    decisions_path: str | Path,
    reviewer: str,
) -> AssetCompositionRun:
    """Persist exact review bytes inside the run and make articulation resumable."""

    state_path = _resolved(path)
    source = Path(decisions_path).expanduser()
    try:
        payload_bytes = _read_stable_regular_bytes(source, label="review decisions")
        payload = payload_bytes.decode("utf-8")
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssetCompositionStateError(
            f"Review decisions must be valid UTF-8 JSON: {source}"
        ) from exc
    if not isinstance(parsed, dict) or not parsed:
        raise AssetCompositionStateError(
            "Review decisions must be a non-empty JSON object"
        )

    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        stage: StageName = "articulation"
        state = run.stages[stage]
        if run.current_stage != stage or state.status != "needs_review":
            raise AssetCompositionStateError(
                "Review decisions are accepted only while articulation needs review"
            )
        if run.coordinator.mode == "single_reasoning_loop":
            if state.review_candidates is None:
                raise AssetCompositionStateError(
                    "Articulation review decisions require frozen Joint candidates"
                )
            candidate_payload = _json_object_from_binding(
                state.review_candidates,
                label="frozen Joint review candidates",
            )
            raw_candidates = candidate_payload.get("candidates")
            embedded_canonical_graph = candidate_payload.get("schema_version") == (
                "content-agent-workflows.embedded-articulation-graph.v1"
            )
            if isinstance(raw_candidates, list):
                candidate_ids = [
                    item.get("candidate_id") if isinstance(item, dict) else None
                    for item in raw_candidates
                ]
            else:
                raw_candidate_ids = candidate_payload.get("candidate_ids")
                candidate_ids = (
                    list(raw_candidate_ids)
                    if isinstance(raw_candidate_ids, list)
                    else []
                )
            if (
                not candidate_ids
                or any(
                    not isinstance(candidate_id, str) or not candidate_id
                    for candidate_id in candidate_ids
                )
                or len(set(candidate_ids)) != len(candidate_ids)
            ):
                raise AssetCompositionStateError(
                    "Frozen Joint review candidates must contain unique candidate IDs"
                )
            wrapped_decisions = parsed.get("decisions")
            raw_decisions = (
                wrapped_decisions if isinstance(wrapped_decisions, dict) else parsed
            )
            if not isinstance(raw_decisions, dict) or not raw_decisions:
                raise AssetCompositionStateError(
                    "Review decisions must be a non-empty candidate-to-decision object"
                )
            invalid_decisions = {
                candidate_id: decision
                for candidate_id, decision in raw_decisions.items()
                if not isinstance(candidate_id, str)
                or not candidate_id
                or not isinstance(decision, str)
                or decision
                not in (
                    {"accept", "reject", "revise"}
                    if embedded_canonical_graph
                    else {"accept", "reject"}
                )
            }
            if invalid_decisions:
                raise AssetCompositionStateError(
                    "Each Joint review decision must map a candidate ID to "
                    + (
                        "'accept', 'reject', or 'revise'"
                        if embedded_canonical_graph
                        else "'accept' or 'reject'"
                    )
                )
            observed_ids = {str(candidate_id) for candidate_id in raw_decisions}
            expected_ids = {str(candidate_id) for candidate_id in candidate_ids}
            if observed_ids != expected_ids:
                missing = sorted(expected_ids - observed_ids)
                unexpected = sorted(observed_ids - expected_ids)
                raise AssetCompositionStateError(
                    "Joint review decisions must match the frozen candidate IDs "
                    f"exactly; missing={missing}, unexpected={unexpected}"
                )
        review_dir = _run_root(state_path) / "reviews"
        source_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        destination = review_dir / f"articulation-{source_sha256}.json"
        if destination.exists():
            existing = _binding(
                destination,
                label="persisted review decisions",
                required_root=_run_root(state_path),
            )
            if existing.sha256 != source_sha256:
                raise AssetCompositionStateError(
                    "Persisted review destination contains different bytes"
                )
        else:
            atomic_write_text(destination, payload)
        decisions = _binding(
            destination,
            label="persisted review decisions",
            required_root=_run_root(state_path),
        )
        if decisions.sha256 != source_sha256:
            raise AssetCompositionStateError(
                "Persisted review decisions do not match the captured bytes"
            )
        previous = state.status
        state.status = "ready"
        state.review_decisions = decisions
        state.continue_current_attempt = True
        state.error = None
        _transition(
            run,
            stage,
            from_status=previous,
            to_status="ready",
            reason=f"Review decisions were bound by {reviewer}.",
            actor=reviewer,
        )
        if run.coordinator.mode == "single_reasoning_loop":
            run.coordinator.next_action = "plan"
        return _write_run(state_path, run)


def record_articulation_graph_revision(
    path: str | Path,
    *,
    revision_receipt_path: str | Path,
    candidates_path: str | Path,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Archive one human-revised graph review and require the revised digest."""

    from content_agent_workflows.articulation import (
        EmbeddedArticulationGraphRevision,
    )

    state_path = _resolved(path)
    root = _run_root(state_path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        stage: StageName = "articulation"
        state = run.stages[stage]
        revision_binding = _binding(
            revision_receipt_path,
            label="Articulation graph revision receipt",
            required_root=root,
        )
        candidates = _binding(
            candidates_path,
            label="revised Articulation review candidates",
            required_root=root,
        )
        if (
            run.current_stage == stage
            and state.status == "needs_review"
            and state.review_candidates == candidates
            and state.review_decisions is None
            and state.superseded_reviews
            and state.superseded_reviews[-1].revision_receipt == revision_binding
        ):
            return run
        if (
            run.current_stage != stage
            or state.status != "ready"
            or state.review_candidates is None
            or state.review_decisions is None
        ):
            raise AssetCompositionStateError(
                "Graph revision requires the active Articulation stage with exact "
                "bound human decisions"
            )
        try:
            record = EmbeddedArticulationGraphRevision.model_validate_json(
                Path(revision_binding.path).read_bytes()
            )
        except (OSError, ValidationError) as exc:
            raise AssetCompositionStateError(
                f"Invalid Articulation graph revision receipt: {exc}"
            ) from exc
        if not _same_binding(record.parent_canonical_graph, state.review_candidates):
            raise AssetCompositionStateError(
                "Graph revision does not supersede the frozen review candidates"
            )
        if (
            record.human_decisions.path != state.review_decisions.path
            or record.human_decisions.sha256 != state.review_decisions.sha256
            or record.human_decisions.size_bytes != state.review_decisions.size_bytes
        ):
            raise AssetCompositionStateError(
                "Graph revision does not bind the frozen human decision bytes"
            )
        if not _same_binding(record.revised_canonical_graph, candidates):
            raise AssetCompositionStateError(
                "Graph revision receipt names another revised canonical graph"
            )
        if any(
            item.revision_receipt == revision_binding
            for item in state.superseded_reviews
        ):
            raise AssetCompositionStateError(
                "Graph revision receipt is already bound to another review cycle"
            )
        state.superseded_reviews.append(
            SupersededArticulationReview(
                superseded_at=_timestamp(),
                revision_receipt=revision_binding,
                review_candidates=state.review_candidates,
                review_decisions=state.review_decisions,
                actor=actor,
            )
        )
        previous = state.status
        state.status = "needs_review"
        state.review_candidates = candidates
        state.review_decisions = None
        state.continue_current_attempt = False
        state.error = None
        _transition(
            run,
            stage,
            from_status=previous,
            to_status="needs_review",
            reason=(
                "Human-revised canonical graph superseded the prior review; the "
                "exact revised digest requires complete review."
            ),
            actor=actor,
        )
        if run.coordinator.mode == "single_reasoning_loop":
            run.coordinator.next_action = "await_human_review"
        return _write_run(state_path, run)


def validate_articulation_graph_revision_request(
    path: str | Path,
    *,
    candidates_path: str | Path,
    human_decisions: ExecutionArtifactBinding,
) -> AssetCompositionRun:
    """Validate exact outer review bindings before domain graph mutation."""

    state_path = _resolved(path)
    root = _run_root(state_path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        state = run.stages["articulation"]
        candidates = _binding(
            candidates_path,
            label="parent Articulation review candidates",
            required_root=root,
        )
        decisions = _binding(
            human_decisions.path,
            label="Articulation graph revision human decisions",
            required_root=root,
        )
        if (
            decisions.path != human_decisions.path
            or decisions.sha256 != human_decisions.sha256
            or decisions.size_bytes != human_decisions.size_bytes
        ):
            raise AssetCompositionStateError(
                "Graph revision human decision binding differs from captured bytes"
            )
        if (
            run.current_stage != "articulation"
            or state.status != "ready"
            or state.review_candidates != candidates
            or state.review_decisions != decisions
        ):
            raise AssetCompositionStateError(
                "Graph revision request differs from the exact active outer review"
            )
        return run


def load_embedded_articulation_human_acceptance(
    path: str | Path,
    *,
    canonical_graph: str | Path,
) -> EmbeddedArticulationHumanAcceptance | None:
    """Read the existing asset human gate as one exact shared human decision.

    This helper is deliberately articulation-only.  It does not transition the
    asset run; it verifies that ``record_review_decisions`` already persisted a
    complete decision for the exact frozen canonical graph.
    """

    from content_agent_workflows.articulation.embedded_decision import (
        EmbeddedArticulationCanonicalGraph,
        EmbeddedArticulationHumanAcceptance,
    )
    from content_agent_workflows.common.embedded_domain_decision import (
        canonical_json_digest,
    )

    state_path = _resolved(path)
    graph_path = Path(canonical_graph).expanduser().resolve()
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        state = run.stages["articulation"]
        if run.current_stage != "articulation":
            raise AssetCompositionStateError(
                "Embedded articulation human acceptance requires the active stage"
            )
        if state.status == "needs_review":
            return None
        if state.status != "ready" or state.review_decisions is None:
            raise AssetCompositionStateError(
                "Embedded articulation authoring requires persisted human decisions"
            )
        if state.review_candidates is None or (
            Path(state.review_candidates.path).expanduser().resolve() != graph_path
        ):
            raise AssetCompositionStateError(
                "Asset human gate is not bound to the canonical articulation graph"
            )
        graph_bytes = _read_stable_regular_bytes(
            graph_path,
            label="canonical articulation graph",
        )
        if hashlib.sha256(graph_bytes).hexdigest() != state.review_candidates.sha256:
            raise AssetCompositionStateError(
                "Canonical articulation graph changed after the human gate"
            )
        try:
            graph = EmbeddedArticulationCanonicalGraph.model_validate_json(graph_bytes)
            decisions_payload = _json_object_from_binding(
                state.review_decisions,
                label="embedded articulation human decisions",
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid embedded articulation human gate: {exc}"
            ) from exc
        wrapped = decisions_payload.get("decisions")
        raw_decisions = wrapped if isinstance(wrapped, dict) else decisions_payload
        if set(raw_decisions) != set(graph.candidate_ids):
            raise AssetCompositionStateError(
                "Embedded articulation human decisions do not cover the exact graph"
            )
        allowed = {"accept", "reject", "revise"}
        if any(value not in allowed for value in raw_decisions.values()):
            raise AssetCompositionStateError(
                "Embedded articulation human decision contains an invalid disposition"
            )
        reviewer = next(
            (
                transition.actor
                for transition in reversed(run.transitions)
                if transition.stage == "articulation"
                and transition.to_status == "ready"
                and transition.actor
            ),
            "asset-human-reviewer",
        )
        return EmbeddedArticulationHumanAcceptance(
            canonical_graph_sha256=canonical_json_digest(graph),
            decisions_path=state.review_decisions.path,
            decisions_sha256=state.review_decisions.sha256,
            reviewer=reviewer,
            decisions={
                candidate_id: cast(
                    Literal["accept", "reject", "revise"],
                    raw_decisions[candidate_id],
                )
                for candidate_id in graph.candidate_ids
            },
        )


def _load_plan_draft(path: str | Path) -> AssetCoordinatorPlanDraft:
    source = Path(path).expanduser()
    try:
        payload = json.loads(
            _read_stable_regular_bytes(source, label="coordinator plan draft")
        )
        return AssetCoordinatorPlanDraft.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Coordinator plan draft must be valid typed JSON: {source}: {exc}"
        ) from exc


def _load_review_draft(path: str | Path) -> AssetCoordinatorReviewDraft:
    source = Path(path).expanduser()
    try:
        payload = json.loads(
            _read_stable_regular_bytes(source, label="coordinator review draft")
        )
        return AssetCoordinatorReviewDraft.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Coordinator review draft must be valid typed JSON: {source}: {exc}"
        ) from exc


def _coordinator_directory(state_path: Path, child: str) -> Path:
    root = _run_root(state_path)
    coordinator_root = root / "coordinator"
    for directory in (coordinator_root, coordinator_root / child):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise AssetCompositionStateError(
                f"Coordinator path must be a regular directory: {directory}"
            )
        directory.mkdir(exist_ok=True)
        if not directory.resolve().is_relative_to(root):
            raise AssetCompositionStateError(
                f"Coordinator path escaped the composed run: {directory}"
            )
    return coordinator_root / child


def _write_immutable_model(
    path: Path,
    model: AssetCoordinatorPlan | AssetCoordinatorEvidenceReview,
    *,
    label: str,
    root: Path,
) -> ArtifactBinding:
    expected_bytes = (
        json.dumps(
            model.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
    if path.exists():
        existing = _binding(path, label=f"existing {label}", required_root=root)
        if existing.sha256 == expected_sha256 and existing.size_bytes == len(
            expected_bytes
        ):
            return existing
        # The caller always allocates len(sealed_chain) + 1, so a differing
        # existing file at this path is an unreferenced orphan left between the
        # artifact write and state commit. It is safe to replace atomically.
    atomic_write_json(path, model)
    return _binding(path, label=label, required_root=root)


def _coordinator_evidence_bindings(
    paths: Sequence[str],
    *,
    label: str,
    root: Path,
    state_path: Path,
    allowed_terminal_names: frozenset[str] = frozenset(),
    forbidden_root: Path | None = None,
    allowed_forbidden_paths: frozenset[Path] = frozenset(),
) -> list[ArtifactBinding]:
    bindings: list[ArtifactBinding] = []
    seen_paths: set[Path] = set()
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    terminal_path = root / "terminal_validation.json"
    mutable_names = {
        ".pipeline_state.json",
        "agent_prompt.md",
        "agent_run_prompt.md",
        "agent_review_prompt.md",
        "agent_resume_prompt.md",
        "asset_composition_items.json",
        "asset_composition_request.json",
        "asset_composition_result.json",
        "checkpoint.json",
        "validation_checkpoint.json",
        "workflow_checkpoint.json",
        "workflow_progress.json",
        "material_session_release.json",
        "material_decision_patch.json",
    }
    for index, item in enumerate(paths):
        candidate = Path(item).expanduser().resolve()
        if (
            forbidden_root is not None
            and candidate.is_relative_to(forbidden_root.resolve())
            and candidate not in allowed_forbidden_paths
        ):
            raise AssetCompositionStateError(
                f"{label} cannot bind mutable current stage-attempt evidence: "
                f"{candidate}"
            )
        if candidate in seen_paths:
            raise AssetCompositionStateError(
                f"{label} contains a duplicate evidence path: {candidate}"
            )
        seen_paths.add(candidate)
        candidate_name = candidate.name.lower()
        is_child_launcher_artifact = candidate_name.startswith("child-") and (
            "-output" in candidate_name or "-final" in candidate_name
        )
        is_coordinator_launcher_artifact = candidate_name.startswith(
            "coordinator-"
        ) and ("-output" in candidate_name or "-final" in candidate_name)
        is_mutable_runtime_artifact = (
            (
                candidate_name in mutable_names
                and candidate_name not in allowed_terminal_names
            )
            or (candidate.parent == root and candidate_name.endswith(".log"))
            or candidate_name.endswith(".lock")
            or "child-output" in candidate_name
            or "child-final" in candidate_name
            or is_child_launcher_artifact
            or is_coordinator_launcher_artifact
        )
        if (
            candidate in {state_path, lock_path, terminal_path}
            or is_mutable_runtime_artifact
        ):
            raise AssetCompositionStateError(
                f"{label} cannot bind mutable coordinator control artifact: {candidate}"
            )
        bindings.append(
            _binding(
                candidate,
                label=f"{label} {index + 1}",
                required_root=root,
            )
        )
    return bindings


def record_coordinator_plan(
    path: str | Path,
    *,
    plan_path: str | Path,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Seal one evidence-backed plan revision for the current stage attempt."""

    draft = _load_plan_draft(plan_path)
    state_path = _resolved(path)
    root = _run_root(state_path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        coordinator = run.coordinator
        if coordinator.mode != "single_reasoning_loop":
            raise AssetCompositionStateError(
                "Coordinator plan revisions require single_reasoning_loop mode"
            )
        if run.terminal_status != "active" or run.current_stage is None:
            raise AssetCompositionStateError("Cannot plan a terminal asset run")
        if draft.stage != run.current_stage:
            raise AssetCompositionStateError(
                f"Plan targets {draft.stage}, not current stage {run.current_stage}"
            )
        stage_state = run.stages[draft.stage]
        if stage_state.status not in {"ready", "running"}:
            raise AssetCompositionStateError(
                f"Cannot plan {draft.stage} from {stage_state.status}"
            )
        if coordinator.next_action != "plan":
            raise AssetCompositionStateError(
                f"Coordinator next action is {coordinator.next_action}, not plan"
            )
        if len(coordinator.plan_revisions) >= coordinator.max_plan_revisions:
            raise AssetCompositionStateError(
                "Coordinator plan revision budget exceeded"
            )
        stage_attempt = _planned_attempt_number(stage_state)
        sealed_attempt_evidence = {
            Path(binding.path).resolve()
            for review_binding in coordinator.evidence_reviews
            for binding in _load_coordinator_review(review_binding).evidence
        }
        sealed_attempt_evidence.update(
            Path(binding.path).resolve()
            for binding in (
                stage_state.review_candidates,
                stage_state.review_decisions,
            )
            if binding is not None
        )
        evidence = _coordinator_evidence_bindings(
            draft.evidence_paths,
            label="coordinator plan evidence",
            root=root,
            state_path=state_path,
            forbidden_root=stage_directory(
                state_path,
                draft.stage,
                attempt=stage_attempt,
            ),
            allowed_forbidden_paths=frozenset(sealed_attempt_evidence),
        )
        prior = coordinator.plan_revisions[-1] if coordinator.plan_revisions else None
        plan_revision = len(coordinator.plan_revisions) + 1
        plan = AssetCoordinatorPlan(
            plan_revision=plan_revision,
            run_revision=run.revision,
            run_id=run.run_id,
            request=run.request,
            source_asset=run.source_asset,
            stage=draft.stage,
            stage_attempt=stage_attempt,
            objective=draft.objective.strip(),
            steps=draft.steps,
            evidence=evidence,
            revision_reason=draft.revision_reason.strip(),
            prior_plan=prior,
            timestamp=_timestamp(),
            actor=actor,
        )
        destination = _coordinator_directory(state_path, "plans") / (
            f"plan-{plan_revision:03d}.json"
        )
        binding = _write_immutable_model(
            destination,
            plan,
            label=f"coordinator plan {plan_revision}",
            root=root,
        )
        coordinator.plan_revisions.append(binding)
        coordinator.next_action = (
            "begin_stage" if stage_state.status == "ready" else "execute_stage"
        )
        return _write_run(state_path, run)


def _archive_stage_for_revisit(
    run: AssetCompositionRun,
    stage: StageName,
    *,
    reason_review: ArtifactBinding,
    actor: str,
    target: bool,
    reviewed: bool = False,
    reviewed_output: ArtifactBinding | None = None,
    reviewed_output_dependencies: Sequence[ArtifactBinding] = (),
    reviewed_evidence: Sequence[ArtifactBinding] = (),
) -> None:
    state = run.stages[stage]
    meaningful = (
        state.status != "pending"
        or state.attempt_count > 0
        or state.input_asset is not None
        or state.output_asset is not None
        or state.error is not None
    )
    history = list(state.superseded_attempts)
    if meaningful:
        history.append(
            SupersededStageAttempt(
                archived_at=_timestamp(),
                reason_review=reason_review,
                status=state.status,
                attempt_count=state.attempt_count,
                input_asset=state.input_asset,
                input_dependencies=state.input_dependencies,
                input_readiness=state.input_readiness,
                output_asset=reviewed_output if reviewed else state.output_asset,
                output_dependencies=(
                    list(reviewed_output_dependencies)
                    if reviewed
                    else state.output_dependencies
                ),
                handoff=state.handoff,
                evidence=list(reviewed_evidence) if reviewed else state.evidence,
                review_candidates=state.review_candidates,
                review_decisions=state.review_decisions,
                superseded_reviews=state.superseded_reviews,
                error=state.error,
            )
        )
    previous = state.status
    if target:
        stage_index = run.stage_order.index(stage)
        predecessor = (
            run.source_asset
            if stage_index == 0
            else run.stages[run.stage_order[stage_index - 1]].output_asset
        )
        predecessor_dependencies = (
            run.source_dependencies
            if stage_index == 0
            else run.stages[run.stage_order[stage_index - 1]].output_dependencies
        )
        predecessor_readiness: Literal["yes", "conditional"] = "yes"
        if stage_index > 0:
            predecessor_state = run.stages[run.stage_order[stage_index - 1]]
            if predecessor_state.handoff is not None:
                predecessor_handoff = AssetStageHandoff.model_validate(
                    _json_object_from_binding(
                        predecessor_state.handoff,
                        label="revisit predecessor handoff",
                    )
                )
                predecessor_readiness = predecessor_handoff.readiness
        if predecessor is None:
            raise AssetCompositionStateError(
                f"Cannot revisit {stage} without an accepted predecessor"
            )
        replacement = StageState(
            status="ready",
            attempt_count=state.attempt_count,
            input_asset=predecessor,
            input_dependencies=predecessor_dependencies,
            input_readiness=predecessor_readiness,
            superseded_attempts=history,
        )
        next_status: StageStatus = "ready"
    else:
        replacement = StageState(
            attempt_count=state.attempt_count,
            superseded_attempts=history,
        )
        next_status = "pending"
    run.stages[stage] = replacement
    if previous != next_status:
        _transition(
            run,
            stage,
            from_status=previous,
            to_status=next_status,
            reason=(
                "Coordinator invalidated this stage after downstream evidence "
                f"review {reason_review.sha256}."
            ),
            actor=actor,
        )


def _active_stage_index(run: AssetCompositionRun, stage: StageName) -> int:
    try:
        return run.stage_order.index(stage)
    except ValueError as exc:
        raise AssetCompositionStateError(
            f"Stage {stage!r} is not enabled for this composed-asset run"
        ) from exc


def _apply_revisit(
    run: AssetCompositionRun,
    *,
    target_stage: StageName,
    reason_review: ArtifactBinding,
    reviewed_output: ArtifactBinding | None,
    reviewed_output_dependencies: Sequence[ArtifactBinding],
    reviewed_evidence: Sequence[ArtifactBinding],
    actor: str,
) -> None:
    current = run.current_stage
    if current is None:
        raise AssetCompositionStateError("Cannot revisit a terminal run")
    target_index = _active_stage_index(run, target_stage)
    current_index = _active_stage_index(run, current)
    if target_index >= current_index:
        raise AssetCompositionStateError(
            "Revisit target must be an earlier stage than the reviewed stage"
        )
    if target_stage == "articulation" and run.stages["articulation"].review_decisions:
        raise AssetCompositionStateError(
            "Cannot revisit articulation after binding the frozen Joint review receipt"
        )
    for index in range(target_index, len(run.stage_order)):
        stage = run.stage_order[index]
        _archive_stage_for_revisit(
            run,
            stage,
            reason_review=reason_review,
            actor=actor,
            target=index == target_index,
            reviewed=stage == current,
            reviewed_output=reviewed_output,
            reviewed_output_dependencies=reviewed_output_dependencies,
            reviewed_evidence=reviewed_evidence,
        )
    run.current_stage = target_stage
    run.terminal_status = "active"


def _prepare_refinement_attempt(
    run: AssetCompositionRun,
    *,
    stage: StageName,
    reason_review: ArtifactBinding,
    output_asset: ArtifactBinding | None,
    output_dependencies: list[ArtifactBinding],
    evidence: list[ArtifactBinding],
    actor: str,
) -> None:
    """Archive one reviewed attempt and make the same input ready again."""

    state = run.stages[stage]
    if state.status != "running" or state.input_asset is None:
        raise AssetCompositionStateError(
            f"Cannot refine {stage} without a running input-bound attempt"
        )
    history = [
        *state.superseded_attempts,
        SupersededStageAttempt(
            archived_at=_timestamp(),
            reason_review=reason_review,
            status=state.status,
            attempt_count=state.attempt_count,
            input_asset=state.input_asset,
            input_dependencies=state.input_dependencies,
            input_readiness=state.input_readiness,
            output_asset=output_asset,
            output_dependencies=output_dependencies,
            evidence=evidence,
            review_candidates=state.review_candidates,
            review_decisions=state.review_decisions,
            superseded_reviews=state.superseded_reviews,
        ),
    ]
    run.stages[stage] = StageState(
        status="ready",
        attempt_count=state.attempt_count,
        input_asset=state.input_asset,
        input_dependencies=state.input_dependencies,
        input_readiness=state.input_readiness,
        superseded_attempts=history,
    )
    _transition(
        run,
        stage,
        from_status="running",
        to_status="ready",
        reason=(
            "Coordinator opened a distinct refinement attempt after evidence "
            f"review {reason_review.sha256}."
        ),
        actor=actor,
    )


def _sealed_stage_output_paths(run: AssetCompositionRun) -> set[Path]:
    """Return output paths that an earlier accepted or rejected attempt sealed."""

    paths: set[Path] = set()
    for state in run.stages.values():
        if state.output_asset is not None:
            paths.add(Path(state.output_asset.path))
        paths.update(
            Path(attempt.output_asset.path)
            for attempt in state.superseded_attempts
            if attempt.output_asset is not None
        )
    return paths


def record_coordinator_evidence_review(
    path: str | Path,
    *,
    review_path: str | Path,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Seal one evidence review and atomically apply its bounded decision."""

    draft = _load_review_draft(review_path)
    state_path = _resolved(path)
    root = _run_root(state_path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        coordinator = run.coordinator
        if coordinator.mode != "single_reasoning_loop":
            raise AssetCompositionStateError(
                "Coordinator evidence reviews require single_reasoning_loop mode"
            )
        if len(coordinator.evidence_reviews) >= coordinator.max_evidence_reviews:
            raise AssetCompositionStateError(
                "Coordinator evidence review budget exceeded"
            )
        if run.terminal_status != "active" or run.current_stage is None:
            raise AssetCompositionStateError("Cannot review a terminal asset run")
        if draft.stage != run.current_stage:
            raise AssetCompositionStateError(
                f"Review targets {draft.stage}, not current stage {run.current_stage}"
            )
        state = run.stages[draft.stage]
        if state.status != "running":
            raise AssetCompositionStateError(
                f"Cannot review {draft.stage} from {state.status}; expected running"
            )
        if coordinator.next_action != "execute_stage":
            raise AssetCompositionStateError(
                f"Coordinator next action is {coordinator.next_action}, not execute_stage"
            )
        if not coordinator.plan_revisions:
            raise AssetCompositionStateError("Evidence review requires a plan revision")
        plan_binding = coordinator.plan_revisions[-1]
        plan = _load_coordinator_plan(plan_binding)
        if plan.stage != draft.stage or plan.stage_attempt != state.attempt_count:
            raise AssetCompositionStateError(
                "Latest coordinator plan does not match the running stage attempt"
            )
        if state.input_asset is None:
            raise AssetCompositionStateError(
                f"Running stage {draft.stage} lacks its input binding"
            )
        attempt_root = stage_directory(
            state_path,
            draft.stage,
            attempt=state.attempt_count,
        ).resolve()
        output: ArtifactBinding | None = None
        output_dependencies: list[ArtifactBinding] = []
        if draft.output_asset_path is not None:
            resolved_output_path = Path(draft.output_asset_path).expanduser().resolve()
            if not _belongs_to_stage_attempt(
                resolved_output_path,
                attempt_root=attempt_root,
                attempt=state.attempt_count,
            ):
                raise AssetCompositionStateError(
                    "Coordinator reviewed output must belong to the current "
                    "stage attempt directory"
                )
            if resolved_output_path in _sealed_stage_output_paths(run):
                raise AssetCompositionStateError(
                    "Each coordinator attempt requires a distinct output asset path"
                )
            output = _binding(
                resolved_output_path,
                label=f"{draft.stage} reviewed output",
                required_root=root,
            )
            output_dependencies = _dependency_bindings(
                Path(output.path),
                label=f"{draft.stage} reviewed output",
                required_root=root,
            )
        resolved_evidence_paths = [
            str(Path(item).expanduser().resolve()) for item in draft.evidence_paths
        ]
        if any(
            not _belongs_to_stage_attempt(
                Path(item),
                attempt_root=attempt_root,
                attempt=state.attempt_count,
            )
            for item in resolved_evidence_paths
        ):
            raise AssetCompositionStateError(
                "Coordinator review evidence must belong to the current stage "
                "attempt directory"
            )
        evidence = _coordinator_evidence_bindings(
            resolved_evidence_paths,
            label=f"{draft.stage} coordinator review evidence",
            root=root,
            state_path=state_path,
            allowed_terminal_names=(
                frozenset({"validation_checkpoint.json"})
                if draft.stage == "validation" and draft.decision == "accept"
                else (
                    frozenset({"workflow_checkpoint.json", "workflow_progress.json"})
                    if draft.stage == "texture" and draft.decision == "accept"
                    else (
                        frozenset({"checkpoint.json", "workflow_progress.json"})
                        if draft.stage == "articulation" and draft.decision == "accept"
                        else frozenset()
                    )
                )
            ),
        )
        if draft.decision == "accept":
            if output is None:
                raise AssetCompositionStateError(
                    "Accept decision requires a reviewed output asset"
                )
            _validate_stage_acceptance(
                draft.stage,
                state_path=state_path,
                run=run,
                input_asset=state.input_asset,
                input_dependencies=state.input_dependencies,
                output=output,
                output_dependencies=output_dependencies,
                evidence=evidence,
            )
        if draft.decision == "await_review":
            if draft.stage != "articulation" or output is not None:
                raise AssetCompositionStateError(
                    "await_review is only valid for articulation candidates"
                )
            if run.stages["articulation"].review_decisions is not None:
                raise AssetCompositionStateError(
                    "Cannot request another articulation review after binding the "
                    "frozen Joint review receipt"
                )
        if draft.decision == "refine":
            if (
                draft.stage == "articulation"
                and run.stages["articulation"].review_decisions is not None
            ):
                raise AssetCompositionStateError(
                    "Cannot refine articulation after binding the frozen Joint "
                    "review receipt"
                )
            count = coordinator.refinement_counts.get(draft.stage, 0)
            if count >= coordinator.max_refinements_per_stage:
                raise AssetCompositionStateError(
                    f"Coordinator refinement budget exceeded for {draft.stage}"
                )
            if not draft.repair_scope:
                raise AssetCompositionStateError(
                    "refine decision requires an explicit repair_scope"
                )
        if draft.decision == "revisit":
            if coordinator.revisit_count >= coordinator.max_revisits:
                raise AssetCompositionStateError("Coordinator revisit budget exceeded")
            if draft.target_stage is None:
                raise AssetCompositionStateError(
                    "Revisit decision requires a target stage"
                )
            if not draft.repair_scope:
                raise AssetCompositionStateError(
                    "revisit decision requires an explicit repair_scope"
                )
            if _active_stage_index(run, draft.target_stage) >= _active_stage_index(
                run, draft.stage
            ):
                raise AssetCompositionStateError(
                    "Revisit target must be earlier than the reviewed stage"
                )
            if (
                draft.target_stage == "articulation"
                and run.stages["articulation"].review_decisions is not None
            ):
                raise AssetCompositionStateError(
                    "Cannot revisit articulation after binding the frozen Joint "
                    "review receipt"
                )
        prior_review = (
            coordinator.evidence_reviews[-1] if coordinator.evidence_reviews else None
        )
        review_index = len(coordinator.evidence_reviews) + 1
        review = AssetCoordinatorEvidenceReview(
            review_index=review_index,
            run_revision=run.revision,
            run_id=run.run_id,
            request=run.request,
            stage=draft.stage,
            stage_attempt=state.attempt_count,
            input_asset=state.input_asset,
            output_asset=output,
            output_dependencies=output_dependencies,
            evidence=evidence,
            plan=plan_binding,
            articulation_review_decisions=(run.stages["articulation"].review_decisions),
            findings=draft.findings,
            decision=draft.decision,
            target_stage=draft.target_stage,
            decision_summary=draft.decision_summary.strip(),
            repair_scope=draft.repair_scope,
            prior_review=prior_review,
            timestamp=_timestamp(),
            actor=actor,
        )
        destination = _coordinator_directory(state_path, "reviews") / (
            f"review-{review_index:03d}.json"
        )
        binding = _write_immutable_model(
            destination,
            review,
            label=f"coordinator evidence review {review_index}",
            root=root,
        )
        coordinator.evidence_reviews.append(binding)
        if draft.decision == "accept":
            coordinator.next_action = "complete_stage"
        elif draft.decision == "await_review":
            coordinator.next_action = "await_human_review"
        elif draft.decision == "refine":
            coordinator.refinement_counts[draft.stage] = (
                coordinator.refinement_counts.get(draft.stage, 0) + 1
            )
            _prepare_refinement_attempt(
                run,
                stage=draft.stage,
                reason_review=binding,
                output_asset=output,
                output_dependencies=output_dependencies,
                evidence=evidence,
                actor=actor,
            )
            coordinator.next_action = "plan"
        elif draft.decision == "revisit":
            if draft.target_stage is None:
                raise AssetCompositionStateError(
                    "Revisit decision requires a target stage"
                )
            coordinator.revisit_count += 1
            _apply_revisit(
                run,
                target_stage=draft.target_stage,
                reason_review=binding,
                reviewed_output=output,
                reviewed_output_dependencies=output_dependencies,
                reviewed_evidence=evidence,
                actor=actor,
            )
            coordinator.next_action = "plan"
        else:
            previous = state.status
            stopped_status: StageStatus = (
                "cancelled" if draft.decision == "stop_cancelled" else "failed"
            )
            state.status = stopped_status
            # The review chain permanently seals this attempt's output and
            # evidence. Recovery must allocate a fresh numbered attempt rather
            # than rewrite those reviewed paths.
            state.continue_current_attempt = False
            state.error = draft.decision_summary.strip()
            run.terminal_status = (
                "cancelled" if stopped_status == "cancelled" else "failed"
            )
            coordinator.next_action = "stopped"
            coordinator.stop_reason = draft.decision_summary.strip()
            _transition(
                run,
                draft.stage,
                from_status=previous,
                to_status=stopped_status,
                reason=draft.decision_summary.strip(),
                actor=actor,
            )
        return _write_run(state_path, run)


def _json_object_from_binding(
    binding: ArtifactBinding,
    *,
    label: str,
) -> dict[str, object]:
    try:
        raw = _read_stable_regular_bytes(Path(binding.path), label=label)
        if (
            len(raw) != binding.size_bytes
            or hashlib.sha256(raw).hexdigest() != binding.sha256
        ):
            raise AssetCompositionStateError(f"{label} identity changed")
        payload = json.loads(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise AssetCompositionStateError(f"Invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssetCompositionStateError(f"{label} must contain a JSON object")
    return payload


def _required_bound_path(
    raw_path: object,
    *,
    evidence: Sequence[ArtifactBinding],
    label: str,
    additional: Sequence[ArtifactBinding] = (),
) -> ArtifactBinding:
    """Resolve one JSON path claim to an exact sealed artifact binding."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise AssetCompositionStateError(f"{label} must name a bound artifact path")
    resolved = Path(raw_path).expanduser().resolve()
    matches = {
        (binding.path, binding.sha256, binding.size_bytes): binding
        for binding in [*evidence, *additional]
        if Path(binding.path) == resolved
    }
    if len(matches) != 1:
        raise AssetCompositionStateError(
            f"{label} must match exactly one sealed evidence artifact: {resolved}"
        )
    return next(iter(matches.values()))


def _require_exact_claim(
    observed: object,
    expected: object,
    *,
    label: str,
) -> None:
    """Report one concrete acceptance mismatch for bounded coordinator repair."""

    if observed != expected:
        raise AssetCompositionStateError(
            f"{label} differs; expected={expected!r}, observed={observed!r}"
        )


def _material_identity(
    value: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise AssetCompositionStateError(f"{label} must be a bound file object")
    path_value = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise AssetCompositionStateError(f"{label} path is missing")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise AssetCompositionStateError(f"{label} digest is invalid")
    return Path(path_value).expanduser().resolve(), sha256


def _require_material_identity(
    value: object,
    expected: ArtifactBinding,
    *,
    label: str,
) -> None:
    path, sha256 = _material_identity(value, label=label)
    if path != Path(expected.path) or sha256 != expected.sha256:
        raise AssetCompositionStateError(
            f"{label} differs from the frozen coordinator input"
        )


def _require_material_evidence_binding(
    evidence: Sequence[ArtifactBinding],
    value: object,
    *,
    label: str,
) -> ArtifactBinding:
    path, sha256 = _material_identity(value, label=label)
    matches = [
        binding
        for binding in evidence
        if Path(binding.path) == path and binding.sha256 == sha256
    ]
    if len(matches) != 1:
        raise AssetCompositionStateError(
            f"{label} must be sealed exactly once as coordinator review evidence"
        )
    return matches[0]


def _frozen_material_reference_identities(
    request: dict[str, object],
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    raw_bindings = request.get("reference_bindings")
    raw_images = request.get("reference_images")
    raw_files = request.get("reference_files")
    if not isinstance(raw_bindings, list):
        raise AssetCompositionStateError(
            "Frozen asset request reference_bindings must be a list"
        )
    if not isinstance(raw_images, list) or not all(
        isinstance(item, str) for item in raw_images
    ):
        raise AssetCompositionStateError(
            "Frozen asset request reference_images must be a path list"
        )
    if not isinstance(raw_files, list) or not all(
        isinstance(item, str) for item in raw_files
    ):
        raise AssetCompositionStateError(
            "Frozen asset request reference_files must be a path list"
        )

    bindings_by_path: dict[Path, str] = {}
    for index, raw_binding in enumerate(raw_bindings):
        path, sha256 = _material_identity(
            raw_binding,
            label=f"frozen Material reference binding {index + 1}",
        )
        if path in bindings_by_path:
            raise AssetCompositionStateError(
                f"Frozen asset request repeats Material reference path {path}"
            )
        bindings_by_path[path] = sha256

    def resolve_paths(raw_paths: list[object], kind: str) -> list[tuple[Path, str]]:
        resolved: list[tuple[Path, str]] = []
        for raw_path in raw_paths:
            if not isinstance(raw_path, str):
                raise AssetCompositionStateError(
                    f"Frozen Material {kind} path must be a string"
                )
            path = Path(raw_path).expanduser().resolve()
            sha256 = bindings_by_path.get(path)
            if sha256 is None:
                raise AssetCompositionStateError(
                    f"Frozen Material {kind} lacks a digest binding: {path}"
                )
            resolved.append((path, sha256))
        return resolved

    images = resolve_paths(raw_images, "reference image")
    files = resolve_paths(raw_files, "reference file")
    if set(bindings_by_path) != {path for path, _sha256 in [*images, *files]}:
        raise AssetCompositionStateError(
            "Frozen Material reference bindings differ from the categorized paths"
        )
    return images, files


def _validate_material_reference_kind(
    value: object,
    expected: Sequence[tuple[Path, str]],
    *,
    label: str,
) -> None:
    if not isinstance(value, list):
        raise AssetCompositionStateError(f"{label} must be a list")
    observed = [
        _material_identity(item, label=f"{label} item {index + 1}")
        for index, item in enumerate(value)
    ]
    if len(observed) != len(expected) or set(observed) != set(expected):
        raise AssetCompositionStateError(
            f"{label} differs from the frozen asset request"
        )


def _validate_material_coordinator_evidence(
    run: AssetCompositionRun,
    *,
    input_asset: ArtifactBinding,
    output: ArtifactBinding,
    evidence: Sequence[ArtifactBinding],
) -> None:
    """Bind the two-phase Material result back to trusted asset-run inputs."""

    result_bindings = [
        binding
        for binding in evidence
        if Path(binding.path).name == "coordinator_result.json"
    ]
    if len(result_bindings) != 1:
        raise AssetCompositionStateError(
            "Material acceptance requires exactly one sealed coordinator result"
        )
    result = _json_object_from_binding(
        result_bindings[0],
        label="Material coordinator result",
    )
    if result.get("schema_version") != "content-agents.material-coordinator-result.v1":
        raise AssetCompositionStateError(
            "Material coordinator result has the wrong schema version"
        )
    if result.get("status") != "pass":
        raise AssetCompositionStateError(
            "Material coordinator result must pass before acceptance"
        )
    if result.get("unresolved_issues") != []:
        raise AssetCompositionStateError(
            "Material coordinator result must not contain unresolved issues"
        )

    request_binding = _require_material_evidence_binding(
        evidence,
        result.get("request"),
        label="Material coordinator request",
    )
    decision_binding = _require_material_evidence_binding(
        evidence,
        result.get("decision_patch"),
        label="Material applied decision patch",
    )
    raw_result_evidence = result.get("evidence")
    if not isinstance(raw_result_evidence, list) or not raw_result_evidence:
        raise AssetCompositionStateError(
            "Material coordinator result evidence must be a non-empty list"
        )
    for index, raw_binding in enumerate(raw_result_evidence):
        _require_material_evidence_binding(
            evidence,
            raw_binding,
            label=f"Material coordinator result evidence item {index + 1}",
        )
    request = _json_object_from_binding(
        request_binding,
        label="Material coordinator request",
    )
    if (
        request.get("schema_version")
        != "content-agents.material-coordinator-request.v1"
    ):
        raise AssetCompositionStateError(
            "Material coordinator request has the wrong schema version"
        )
    respect_existing_material_bindings = request.get(
        "respect_existing_material_bindings"
    )
    if not isinstance(respect_existing_material_bindings, bool):
        raise AssetCompositionStateError(
            "Material coordinator request lacks a valid existing-binding policy"
        )

    request_path = Path(request_binding.path)
    raw_run_dir = request.get("run_dir")
    if not isinstance(raw_run_dir, str) or (
        Path(raw_run_dir).expanduser().resolve() != request_path.parent
        or request_path.name != "coordinator_request.json"
    ):
        raise AssetCompositionStateError(
            "Material coordinator request path is not bound to its run directory"
        )
    material_run_dir = request_path.parent
    result_evidence_paths = {
        _material_identity(
            item,
            label=f"Material coordinator result evidence item {index + 1}",
        )[0]
        for index, item in enumerate(raw_result_evidence)
    }
    canonical_evidence = {
        material_run_dir / "coordinator_request.json",
        material_run_dir / "coordinator_preparation.json",
        material_run_dir / "assignments.json",
        material_run_dir / "api_operation_counts.json",
        material_run_dir / "visual_quality_assessment.json",
        material_run_dir / "validation_evidence.json",
        material_run_dir / "final_summary.md",
        material_run_dir / "raw" / "material_restore_response.json",
        material_run_dir / "raw" / "material_binding_audit.json",
        material_run_dir / "raw" / "material_application_receipt.json",
        material_run_dir / "raw" / "material_post_apply_review.json",
        material_run_dir / "raw" / "ovrtx_render_probe.json",
        material_run_dir / "raw" / "material_operation_receipts.json",
        material_run_dir / "raw" / "material_applied_decision_patch.json",
        material_run_dir / "raw" / "material_finalization_policy.json",
        material_run_dir / "raw" / "material_run_packet.json",
        material_run_dir / "raw" / "visible_candidate_prims.json",
        material_run_dir / "raw" / "material_palette.json",
        material_run_dir / "raw" / "material_authoring_context.md",
        material_run_dir / "raw" / "material_assignment_seed.json",
        material_run_dir / "raw" / "visible_candidate_table.tsv",
        material_run_dir / "raw" / "final_render_records.json",
        material_run_dir / "trace" / "operation_trace.json",
        material_run_dir / "trace" / "operation_trace.md",
        material_run_dir / "trace" / "run_retrospective.json",
        material_run_dir / "trace" / "replay_manifest.json",
    }
    if not respect_existing_material_bindings:
        canonical_evidence.add(
            material_run_dir / "raw" / "appearance_clear_report.json"
        )
    missing_canonical_evidence = canonical_evidence - result_evidence_paths
    if missing_canonical_evidence:
        raise AssetCompositionStateError(
            "Material coordinator result omits canonical evidence: "
            + ", ".join(
                str(path.relative_to(material_run_dir))
                for path in sorted(missing_canonical_evidence, key=str)
            )
        )
    if not any(
        path.is_relative_to(material_run_dir / "evidence_renders")
        for path in result_evidence_paths
    ):
        raise AssetCompositionStateError(
            "Material coordinator result requires an initial evidence render"
        )
    if not any(
        path.is_relative_to(material_run_dir / "final_renders")
        for path in result_evidence_paths
    ):
        raise AssetCompositionStateError(
            "Material coordinator result requires a final evidence render"
        )

    def canonical_binding(relative_path: str) -> ArtifactBinding:
        expected = material_run_dir / relative_path
        matches = [binding for binding in evidence if Path(binding.path) == expected]
        if len(matches) != 1:
            raise AssetCompositionStateError(
                f"Material native evidence must seal {relative_path} exactly once"
            )
        return matches[0]

    assignments = _json_object_from_binding(
        canonical_binding("assignments.json"),
        label="Material assignments",
    )
    visual_quality = _json_object_from_binding(
        canonical_binding("visual_quality_assessment.json"),
        label="Material visual quality assessment",
    )
    decision_patch = _json_object_from_binding(
        decision_binding,
        label="Material applied decision patch",
    )
    restore = _json_object_from_binding(
        canonical_binding("raw/material_restore_response.json"),
        label="Material restore response",
    )
    operation_counts = _json_object_from_binding(
        canonical_binding("api_operation_counts.json"),
        label="Material operation counts",
    )
    binding_audit = _json_object_from_binding(
        canonical_binding("raw/material_binding_audit.json"),
        label="Material binding audit",
    )
    if operation_counts.get(
        "schema_version"
    ) != "content-agents.api-operation-counts.v1" or any(
        not isinstance(operation_counts.get(key), int)
        or isinstance(operation_counts.get(key), bool)
        or operation_counts[key] < 0
        for key in (
            "api_operation_count_total",
            "render_count_total",
            "material_override_commands",
            "final_renders",
        )
    ):
        raise AssetCompositionStateError("Material operation counts are invalid")
    if (
        binding_audit.get("schema_version")
        != "content-agents.material-binding-audit.v1"
        or binding_audit.get("status") != "pass"
        or binding_audit.get("errors") != []
        or binding_audit.get("verified_target_count")
        != binding_audit.get("expected_target_count")
    ):
        raise AssetCompositionStateError(
            "Material binding audit does not verify every accepted target"
        )
    application_receipt = _json_object_from_binding(
        canonical_binding("raw/material_application_receipt.json"),
        label="Material application receipt",
    )
    final_render_records = _json_object_from_binding(
        canonical_binding("raw/final_render_records.json"),
        label="Material final render records",
    )
    post_apply_review = _json_object_from_binding(
        canonical_binding("raw/material_post_apply_review.json"),
        label="Material post-apply review",
    )
    if (
        application_receipt.get("schema_version")
        != "content-agents.material-application-receipt.v1"
        or application_receipt.get("status") != "review_required"
        or not isinstance(application_receipt.get("final_render_bindings"), list)
        or not application_receipt["final_render_bindings"]
    ):
        raise AssetCompositionStateError(
            "Material application receipt does not bind a post-apply review phase"
        )
    if (
        post_apply_review.get("schema_version")
        != "content-agents.material-post-apply-review.v1"
        or post_apply_review.get("status") not in {"pass", "fixed"}
        or post_apply_review.get("unresolved_issues") != []
    ):
        raise AssetCompositionStateError(
            "Material post-apply review is not a resolved final-render review"
        )
    raw_renders = final_render_records.get("renders")
    raw_turntable = final_render_records.get("turntable")
    if (
        final_render_records.get("schema_version")
        != "content-agents.material-final-renders.v1"
        or final_render_records.get("render_engine") != "ovrtx"
        or not isinstance(raw_renders, list)
        or not raw_renders
        or not isinstance(raw_turntable, dict)
        or raw_turntable.get("frame_count") != 24
    ):
        raise AssetCompositionStateError(
            "Material final render records lack the required OVRTX turntable"
        )
    render_image_paths: list[Path] = []
    turntable_frame_paths: list[Path] = []
    for index, record in enumerate(raw_renders):
        if not isinstance(record, dict):
            raise AssetCompositionStateError(
                f"Material final render record {index + 1} must be an object"
            )
        raw_image_path = record.get("image_path")
        if not isinstance(raw_image_path, str) or record.get("renderer") not in {
            "ovrtx",
            "remote",
        }:
            raise AssetCompositionStateError(
                f"Material final render record {index + 1} is not OVRTX evidence"
            )
        image_path = Path(raw_image_path).expanduser().resolve()
        if not image_path.is_relative_to(material_run_dir / "final_renders"):
            raise AssetCompositionStateError(
                f"Material final render record {index + 1} escapes final_renders"
            )
        render_image_paths.append(image_path)
        if record.get("kind") == "turntable_frame":
            turntable_frame_paths.append(image_path)
    if len(turntable_frame_paths) != 24 or len(set(turntable_frame_paths)) != 24:
        raise AssetCompositionStateError(
            "Material final render records must contain 24 distinct turntable frames"
        )
    if len(set(render_image_paths)) != len(render_image_paths):
        raise AssetCompositionStateError(
            "Material final render records repeat an image path"
        )
    raw_turntable_path = raw_turntable.get("gif_path")
    if not isinstance(raw_turntable_path, str):
        raise AssetCompositionStateError("Material turntable GIF path is missing")
    turntable_path = Path(raw_turntable_path).expanduser().resolve()
    if not turntable_path.is_relative_to(material_run_dir / "final_renders"):
        raise AssetCompositionStateError("Material turntable GIF escapes final_renders")
    expected_final_render_paths = {*render_image_paths, turntable_path}
    if not expected_final_render_paths.issubset(result_evidence_paths):
        raise AssetCompositionStateError(
            "Material coordinator result omits turntable render evidence"
        )
    expected_final_render_bindings: set[tuple[Path, str]] = set()
    for path in expected_final_render_paths:
        matches = [binding for binding in evidence if Path(binding.path) == path]
        if len(matches) != 1:
            raise AssetCompositionStateError(
                f"Material final render must be sealed exactly once: {path}"
            )
        expected_final_render_bindings.add((path, matches[0].sha256))
    receipt_review_bindings = {
        _material_identity(item, label="Material application final render")
        for item in application_receipt["final_render_bindings"]
    }
    if receipt_review_bindings != expected_final_render_bindings:
        raise AssetCompositionStateError(
            "Material application receipt differs from the OVRTX turntable evidence"
        )
    raw_review_bindings = post_apply_review.get("checked_view_bindings")
    if not isinstance(raw_review_bindings, list):
        raise AssetCompositionStateError(
            "Material post-apply review lacks exact checked-view bindings"
        )
    reviewed_bindings = {
        _material_identity(item, label="Material post-apply checked view")
        for item in raw_review_bindings
    }
    checked_views = post_apply_review.get("checked_views")
    result_evidence_identities = {
        _material_identity(item, label="Material coordinator result evidence")
        for item in raw_result_evidence
    }
    if (
        reviewed_bindings != receipt_review_bindings
        or not receipt_review_bindings.issubset(result_evidence_identities)
        or not isinstance(checked_views, list)
        or any(not isinstance(path, str) for path in checked_views)
        or {Path(path).expanduser().resolve() for path in checked_views}
        != {path for path, _sha256 in receipt_review_bindings}
    ):
        raise AssetCompositionStateError(
            "Material post-apply review differs from the applied final renders"
        )
    try:
        validation = ValidationEvidence.model_validate(
            _json_object_from_binding(
                canonical_binding("validation_evidence.json"),
                label="Material validation evidence",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid Material validation evidence: {exc}"
        ) from exc

    _require_material_identity(
        request.get("source"),
        input_asset,
        label="Material source",
    )
    output_path = Path(output.path)
    if (
        request.get("output_usd_path") != str(output_path)
        or result.get("output_usd_path") != str(output_path)
        or result.get("output_usd_sha256") != output.sha256
    ):
        raise AssetCompositionStateError(
            "Material result output differs from the accepted asset bytes"
        )
    if (
        assignments.get("schema_version") != "content-agents.assignments.v1"
        or assignments.get("source_usd") != input_asset.path
    ):
        raise AssetCompositionStateError(
            "Material assignments schema or source differs from the accepted input"
        )
    materialized = assignments.get("materialized_usd")
    if not isinstance(materialized, dict):
        raise AssetCompositionStateError(
            "Material assignments lack the materialized USD result"
        )
    if materialized.get("status") != "succeeded":
        raise AssetCompositionStateError(
            "Material assignments must report a succeeded materialized USD"
        )
    for path_key in ("requested_output_path", "output_path"):
        raw_materialized_path = materialized.get(path_key)
        if not isinstance(raw_materialized_path, str) or (
            Path(raw_materialized_path).expanduser().resolve() != output_path
        ):
            raise AssetCompositionStateError(
                f"Material assignments {path_key} differs from the accepted output"
            )
    forbidden_materialized_fields = (
        "error",
        "error_type",
        "unbound_source_prim_paths",
        "uncovered_assignment_groups",
        "unresolved_mappings",
    )
    unexpected_materialized_fields = [
        field
        for field in forbidden_materialized_fields
        if materialized.get(field) not in (None, [], {})
    ]
    if unexpected_materialized_fields:
        raise AssetCompositionStateError(
            "Material assignments contain unresolved materialization fields: "
            + ", ".join(unexpected_materialized_fields)
        )

    coverage = assignments.get("coverage")
    if not isinstance(coverage, dict):
        raise AssetCompositionStateError("Material assignments lack coverage evidence")
    unresolved_coverage = {
        key: coverage.get(key)
        for key in (
            "unassigned_visible_prim_count",
            "missing_assignment_prim_count",
            "rejected_assignment_prim_count",
        )
        if coverage.get(key) not in (0, None)
    }
    if unresolved_coverage:
        raise AssetCompositionStateError(
            "Material assignments retain unresolved coverage: "
            + json.dumps(unresolved_coverage, sort_keys=True)
        )
    if assignments.get("visual_quality_assessment") != visual_quality:
        raise AssetCompositionStateError(
            "Material assignments embed a different visual quality assessment"
        )
    authored_visual_quality = decision_patch.get("visual_quality_assessment")
    if (
        decision_patch.get("schema_version")
        != "content-agents.material-decision-patch.v1"
        or not isinstance(decision_patch.get("material_assignments"), list)
        or not isinstance(decision_patch.get("reviewed_no_override"), list)
        or not isinstance(authored_visual_quality, dict)
        or authored_visual_quality.get("status") not in {"pass", "fixed"}
        or authored_visual_quality.get("unresolved_issues") != []
        or not authored_visual_quality.get("checked_views")
    ):
        raise AssetCompositionStateError(
            "Material applied decision patch lacks a resolved evidence-backed VQA"
        )
    if (
        visual_quality.get("schema_version")
        != "content-agents.visual-quality-assessment.v1"
        or visual_quality.get("status") not in {"pass", "fixed"}
        or visual_quality.get("unresolved_issues") != []
    ):
        raise AssetCompositionStateError(
            "Material visual quality assessment is not a resolved pass"
        )
    checked_views = visual_quality.get("checked_views")
    if not isinstance(checked_views, list) or not checked_views:
        raise AssetCompositionStateError(
            "Material visual quality assessment must name checked final views"
        )
    sealed_paths = {Path(binding.path) for binding in evidence}
    authored_checked_views = authored_visual_quality.get("checked_views")
    if not isinstance(authored_checked_views, list):
        raise AssetCompositionStateError(
            "Material decision patch checked views must be a list"
        )
    invalid_authored_views = [
        raw_path
        for raw_path in authored_checked_views
        if not isinstance(raw_path, str)
        or Path(raw_path).expanduser().resolve() not in sealed_paths
        or not Path(raw_path).expanduser().resolve().is_relative_to(material_run_dir)
    ]
    if invalid_authored_views:
        raise AssetCompositionStateError(
            "Material decision patch checked views are not sealed run evidence: "
            + repr(invalid_authored_views)
        )
    invalid_checked_views = [
        raw_path
        for raw_path in checked_views
        if not isinstance(raw_path, str)
        or Path(raw_path).expanduser().resolve() not in sealed_paths
        or not Path(raw_path)
        .expanduser()
        .resolve()
        .is_relative_to(material_run_dir / "final_renders")
    ]
    if invalid_checked_views:
        raise AssetCompositionStateError(
            "Material visual quality checked views are not sealed final renders: "
            + repr(invalid_checked_views)
        )

    if (
        validation.schema_version != VALIDATION_EVIDENCE_SCHEMA_VERSION
        or validation.workflow != "material_assignment"
        or Path(validation.asset).expanduser().resolve() != Path(input_asset.path)
        or validation.sim_ready_status != "pass"
        or validation.failures
        or validation.warnings
        or validation.unresolved_issues
    ):
        raise AssetCompositionStateError(
            "Material validation evidence does not prove a clean pass for the input"
        )
    visual_checks = [
        check for check in validation.checks if check.name == "visual_materials"
    ]
    if len(visual_checks) != 1 or visual_checks[0].status != "pass":
        raise AssetCompositionStateError(
            "Material validation evidence lacks one passing visual_materials check"
        )
    validation_artifact_paths = {
        Path(artifact.path).expanduser().resolve()
        for artifact in [
            *validation.evidence_artifacts,
            *visual_checks[0].evidence_artifacts,
        ]
    }
    if not validation_artifact_paths or not validation_artifact_paths <= sealed_paths:
        raise AssetCompositionStateError(
            "Material validation artifacts must be sealed coordinator evidence"
        )

    restored_paths = restore.get("restored_source_prim_paths")
    restored_count = restore.get("restored_edit_count")
    if (
        restore.get("unresolved_mappings") != []
        or restore.get("unbound_source_prim_paths") != []
        or not isinstance(restored_paths, list)
        or any(
            not isinstance(path, str) or not path.startswith("/")
            for path in restored_paths
        )
        or len(set(restored_paths)) != len(restored_paths)
        or isinstance(restored_count, bool)
        or not isinstance(restored_count, int)
        or restored_count < 0
        or restored_count != len(restored_paths)
    ):
        raise AssetCompositionStateError(
            "Material restore response does not prove complete source coverage"
        )
    raw_restored_output = restore.get("output_usd_path")
    if (
        not isinstance(raw_restored_output, str)
        or not Path(raw_restored_output).expanduser().is_absolute()
    ):
        raise AssetCompositionStateError(
            "Material restore response lacks its absolute staged output path"
        )

    frozen_request = _normalize_historical_asset_request_payload(
        _json_object_from_binding(
            run.request,
            label="frozen asset request",
        )
    )
    for request_key, binding_key, label in (
        ("materials_yaml", "materials_yaml_binding", "Material manifest"),
        ("materials_usd", "materials_usd_binding", "Material library"),
    ):
        frozen_path = frozen_request.get(request_key)
        frozen_binding = frozen_request.get(binding_key)
        path, sha256 = _material_identity(frozen_binding, label=f"frozen {label}")
        if frozen_path != str(path):
            raise AssetCompositionStateError(
                f"Frozen {label} path differs from its digest binding"
            )
        observed_path, observed_sha256 = _material_identity(
            request.get(request_key),
            label=label,
        )
        if (observed_path, observed_sha256) != (path, sha256):
            raise AssetCompositionStateError(
                f"{label} differs from the frozen asset request"
            )

    frozen_dependencies = frozen_request.get("materials_usd_dependencies")
    observed_dependencies = request.get("materials_usd_dependencies")
    if not isinstance(frozen_dependencies, list) or not isinstance(
        observed_dependencies, list
    ):
        raise AssetCompositionStateError(
            "Material library dependency bindings must be lists"
        )
    frozen_dependency_identities = [
        _material_identity(item, label=f"frozen Material dependency {index + 1}")
        for index, item in enumerate(frozen_dependencies)
    ]
    observed_dependency_identities = [
        _material_identity(item, label=f"Material dependency {index + 1}")
        for index, item in enumerate(observed_dependencies)
    ]
    if observed_dependency_identities != frozen_dependency_identities:
        raise AssetCompositionStateError(
            "Material library dependency closure differs from the frozen asset request"
        )

    expected_images, expected_files = _frozen_material_reference_identities(
        frozen_request
    )
    _validate_material_reference_kind(
        request.get("reference_images"),
        expected_images,
        label="Material reference images",
    )
    _validate_material_reference_kind(
        request.get("reference_files"),
        expected_files,
        label="Material reference files",
    )

    raw_repository_root = frozen_request.get("repository_root")
    frozen_runtime = frozen_request.get("runtime")
    if not isinstance(raw_repository_root, str) or not isinstance(frozen_runtime, dict):
        raise AssetCompositionStateError(
            "Frozen asset request lacks Material runtime ownership fields"
        )
    frozen_scene_tool_timeout = frozen_runtime.get("scene_tool_timeout_seconds")
    if (
        isinstance(frozen_scene_tool_timeout, bool)
        or not isinstance(frozen_scene_tool_timeout, int | float)
        or frozen_scene_tool_timeout <= 0
    ):
        raise AssetCompositionStateError(
            "Frozen asset request lacks the Material scene-tool timeout"
        )
    observed_repository_root = request.get("repository_root")
    observed_scene_tool_timeout = request.get("scene_tool_timeout_seconds")
    if not isinstance(observed_repository_root, str) or (
        Path(observed_repository_root).expanduser().resolve()
        != Path(raw_repository_root).expanduser().resolve()
    ):
        raise AssetCompositionStateError(
            "Material repository root differs from the frozen asset request"
        )
    if (
        isinstance(observed_scene_tool_timeout, bool)
        or not isinstance(observed_scene_tool_timeout, int | float)
        or observed_scene_tool_timeout != frozen_scene_tool_timeout
    ):
        raise AssetCompositionStateError(
            "Material scene-tool timeout differs from the frozen asset request"
        )


def _validate_embedded_domain_request(
    metadata: dict[str, object],
    *,
    expected_domain: DomainName,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    output_dir: Path,
) -> None:
    """Require native evidence to identify the exact owning outer attempt."""

    try:
        context = domain_execution_context_from_metadata(
            metadata,
            expected_domain=expected_domain,
        )
    except ValueError as exc:
        raise AssetCompositionStateError(
            f"Invalid embedded {expected_domain} execution context: {exc}"
        ) from exc
    if (
        context is None
        or context.mode != "embedded"
        or context.reasoning_loop_owner != "asset_coordinator"
        or context.embedded_stage is None
    ):
        raise AssetCompositionStateError(
            f"{expected_domain.title()} acceptance requires an embedded execution "
            "context owned by the asset coordinator"
        )
    if not run.coordinator.plan_revisions:
        raise AssetCompositionStateError(
            f"{expected_domain.title()} acceptance lacks a coordinator plan"
        )
    stage = run.stages[expected_domain]
    embedded = context.embedded_stage

    def portable(binding: ArtifactBinding) -> ExecutionArtifactBinding:
        return ExecutionArtifactBinding.model_validate(binding.model_dump())

    matching_plan_bindings = [
        binding
        for binding in run.coordinator.plan_revisions
        if portable(binding) == embedded.coordinator_plan
    ]
    if len(matching_plan_bindings) != 1:
        raise AssetCompositionStateError(
            f"Embedded {expected_domain} execution context references an unknown "
            "coordinator plan"
        )
    embedded_plan = _load_coordinator_plan(matching_plan_bindings[0])
    canonical_domain_root = (
        stage_directory(
            state_path,
            expected_domain,
            attempt=stage.attempt_count,
        )
        / "domain-run"
    ).resolve()
    if (
        embedded.outer_run_id != run.run_id
        or embedded.outer_request != portable(run.request)
        or embedded.stage != expected_domain
        or embedded.stage_attempt != stage.attempt_count
        or embedded.input_asset != portable(input_asset)
        or Path(embedded.domain_run_root).expanduser().resolve()
        != output_dir.expanduser().resolve()
        or Path(embedded.domain_run_root).expanduser().resolve()
        != canonical_domain_root
        or embedded_plan.stage != expected_domain
        or embedded_plan.stage_attempt != stage.attempt_count
        or embedded_plan.request != run.request
    ):
        raise AssetCompositionStateError(
            f"Embedded {expected_domain} execution context differs from the "
            "active coordinator attempt"
        )


def _validate_texture_embedded_decision_receipt(
    summary_artifacts: Mapping[str, object],
    *,
    evidence: list[ArtifactBinding],
    request: object,
    checkpoint: object,
    output: ArtifactBinding,
) -> None:
    """Require the exact completed shared receipt for Texture publication."""

    from content_agent_workflows.common.embedded_domain_artifact_store import (
        EmbeddedDecisionArtifactStore,
    )
    from content_agent_workflows.common.embedded_domain_decision import (
        EmbeddedBoundedExecutionResult,
        EmbeddedDecisionContractError,
        EmbeddedDecisionIdentity,
        EmbeddedDecisionReceipt,
        artifact_reference,
    )
    from content_agent_workflows.texture.embedded_decision import (
        TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY,
        validate_completed_texture_decision_chain,
    )
    from content_agent_workflows.texture.runtime import (
        TextureWorkflowRuntimeError,
        verify_checkpoint_identity,
    )

    raw_path = summary_artifacts.get("embedded_decision_receipt")
    if not isinstance(raw_path, str) or not raw_path:
        raise AssetCompositionStateError(
            "Texture final summary does not reference a completed shared receipt"
        )
    binding = _required_bound_path(
        raw_path,
        evidence=evidence,
        label="Texture embedded decision receipt",
    )
    try:
        receipt = EmbeddedDecisionReceipt.model_validate(
            _json_object_from_binding(
                binding,
                label="Texture embedded decision receipt",
            )
        )
        request_output_dir = Path(getattr(request, "output_dir")).expanduser().resolve()
        decision_store = EmbeddedDecisionArtifactStore(request_output_dir)
        reference = artifact_reference(receipt)
        persisted = decision_store.load_typed(reference, EmbeddedDecisionReceipt)
        journal = decision_store.journal()
        embedded_state = getattr(checkpoint, "embedded_decision_state")
        if embedded_state is None:
            raise ValueError(
                "Texture terminal checkpoint lacks its shared decision state"
            )
        raw_identity = getattr(request, "metadata").get(
            TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY
        )
        if raw_identity is None:
            raise ValueError(
                "Texture request lacks its frozen shared decision identity"
            )
        frozen_identity = EmbeddedDecisionIdentity.model_validate(raw_identity)
        if (
            frozen_identity != embedded_state.identity
            or frozen_identity != receipt.identity
        ):
            raise ValueError(
                "Texture request, checkpoint, and receipt decision identities differ"
            )
        verify_checkpoint_identity(checkpoint, request=request)
        replayed = validate_completed_texture_decision_chain(
            decision_store,
            embedded_state,
            getattr(checkpoint, "plan"),
        )
    except (
        EmbeddedDecisionContractError,
        OSError,
        TextureWorkflowRuntimeError,
        TypeError,
        ValueError,
        ValidationError,
    ) as exc:
        raise AssetCompositionStateError(
            f"Invalid embedded Texture decision receipt: {exc}"
        ) from exc
    canonical_path = (
        decision_store.store_root
        / "artifacts"
        / reference.artifact_kind
        / f"{reference.sha256}.json"
    ).resolve()
    if Path(binding.path).expanduser().resolve() != canonical_path:
        raise AssetCompositionStateError(
            "Texture receipt does not use its canonical durable store path"
        )
    if persisted != receipt:
        raise AssetCompositionStateError(
            "Texture receipt differs from its durable shared-contract bytes"
        )
    _required_bound_path(
        str((decision_store.store_root / "journal.json").resolve()),
        evidence=evidence,
        label="Texture embedded decision journal",
    )
    for entry in journal.entries:
        artifact_path = (decision_store.store_root / entry.relative_path).resolve()
        _required_bound_path(
            str(artifact_path),
            evidence=evidence,
            label="Texture embedded decision artifact",
        )
    candidate_result_reference = embedded_state.accepted_candidate_result
    if candidate_result_reference is None:  # pragma: no cover - replay invariant
        raise AssetCompositionStateError(
            "Texture shared decision state lacks an accepted candidate result"
        )
    candidate_result = decision_store.load_typed(
        candidate_result_reference,
        EmbeddedBoundedExecutionResult,
    )
    for artifact in (
        *candidate_result.outputs,
        *(
            artifact
            for record in candidate_result.evidence
            for artifact in record.artifacts
        ),
    ):
        _required_bound_path(
            artifact.path,
            evidence=evidence,
            label="Texture accepted candidate artifact",
        )
    if replayed != receipt:
        raise AssetCompositionStateError(
            "Texture receipt differs from the replayed canonical decision chain"
        )
    execution_context = getattr(request, "execution_context")
    completed_reference = getattr(
        getattr(checkpoint, "embedded_decision_state"),
        "completed_receipt",
        None,
    )
    if (
        receipt.receipt_status != "completed"
        or receipt.execution_effect != "mutation"
        or receipt.execution_status != "succeeded"
        or receipt.review_disposition != "accept"
        or receipt.mutation_id != receipt.operation_id
        or receipt.identity.execution_context != execution_context
        or completed_reference != reference
        or len(receipt.outputs) != 1
    ):
        raise AssetCompositionStateError(
            "Texture shared receipt is incomplete, unreviewed, or bound to another run"
        )
    published = receipt.outputs[0]
    if (
        Path(published.path).expanduser().resolve() != Path(output.path)
        or published.sha256 != output.sha256
        or published.size_bytes != output.size_bytes
    ):
        raise AssetCompositionStateError(
            "Texture shared receipt does not bind the exact accepted publication"
        )
    canonical_plan = embedded_state.canonical_plan
    if canonical_plan is None or (
        getattr(checkpoint, "plan_digest") != canonical_plan.proposal_plan_digest
        or getattr(checkpoint, "selected_unit_ids") != canonical_plan.target_unit_ids
        or getattr(checkpoint, "accepted_unit_ids") != canonical_plan.target_unit_ids
        or getattr(checkpoint, "remaining_unit_ids")
        or set(getattr(checkpoint, "accepted_unit_material_state_digests"))
        != set(canonical_plan.target_unit_ids)
    ):
        raise AssetCompositionStateError(
            "Texture checkpoint does not bind every canonical target to the "
            "accepted publication"
        )


def _validate_texture_coordinator_evidence(
    evidence: list[ArtifactBinding],
    *,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    output: ArtifactBinding,
) -> None:
    """Require the canonical native Texture pass evidence for acceptance."""

    from content_agent_workflows.texture import (
        TextureWorkflowCheckpoint,
        TextureWorkflowRequest,
        TextureWorkflowValidationEvidence,
        texture_request_digest,
        verify_texture_decision_ledger,
    )

    request_matches = [
        binding for binding in evidence if Path(binding.path).name == "request.json"
    ]
    requires_embedded_context = (
        run.schema_version != LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION
    )
    validation_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "validation_evidence.json"
    ]
    summary_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "final_summary.json"
    ]
    progress_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "workflow_progress.json"
    ]
    checkpoint_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "workflow_checkpoint.json"
    ]
    ledger_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "texture_decision_ledger.json"
    ]
    if (
        (requires_embedded_context and len(request_matches) != 1)
        or (requires_embedded_context and len(ledger_matches) > 1)
        or len(validation_matches) != 1
        or len(summary_matches) != 1
        or len(progress_matches) != 1
        or len(checkpoint_matches) != 1
    ):
        required_names = (
            "request.json, validation_evidence.json, "
            "workflow_progress.json, final_summary.json, and terminal "
            "workflow_checkpoint.json"
            if requires_embedded_context
            else "validation_evidence.json, workflow_progress.json, "
            "final_summary.json, and terminal workflow_checkpoint.json"
        )
        raise AssetCompositionStateError(
            f"Texture acceptance requires exactly one {required_names}"
        )
    parents = {
        Path(binding.path).parent
        for binding in (
            validation_matches[0],
            progress_matches[0],
            summary_matches[0],
            checkpoint_matches[0],
            *(request_matches[:1] if requires_embedded_context else ()),
            *(ledger_matches[:1] if ledger_matches else ()),
        )
    }
    if len(parents) != 1:
        raise AssetCompositionStateError(
            "Texture terminal evidence must share one native run"
        )
    try:
        validation = TextureWorkflowValidationEvidence.model_validate(
            _json_object_from_binding(
                validation_matches[0],
                label="Texture validation evidence",
            )
        )
        checkpoint = TextureWorkflowCheckpoint.model_validate(
            _json_object_from_binding(
                checkpoint_matches[0],
                label="Texture terminal checkpoint",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid Texture validation evidence: {exc}"
        ) from exc
    if requires_embedded_context:
        try:
            request = TextureWorkflowRequest.model_validate(
                _json_object_from_binding(
                    request_matches[0],
                    label="Texture workflow request",
                )
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid Texture workflow request: {exc}"
            ) from exc
        _validate_embedded_domain_request(
            request.metadata,
            expected_domain="texture",
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            output_dir=request.output_dir,
        )
        if request.output_dir.expanduser().resolve() != Path(
            request_matches[0].path
        ).parent.resolve() or Path(request.source_asset).expanduser().resolve() != Path(
            input_asset.path
        ):
            raise AssetCompositionStateError(
                "Texture embedded request path or source differs from its native run"
            )
        if checkpoint.request_digest != texture_request_digest(request):
            raise AssetCompositionStateError(
                "Texture terminal checkpoint does not bind the embedded request"
            )
        if ledger_matches:
            try:
                decision_ledger = verify_texture_decision_ledger(
                    ledger_matches[0].path,
                    output_dir=request.output_dir,
                    request_digest=checkpoint.request_digest,
                    source_identity_digest=checkpoint.source_identity_digest,
                    plan_digest=checkpoint.plan_digest,
                    require_final_decision=True,
                    checkpoint=checkpoint,
                )
            except (OSError, ValueError) as exc:
                raise AssetCompositionStateError(
                    f"Invalid embedded Texture decision ledger: {exc}"
                ) from exc
            for record in decision_ledger.records:
                patch_binding = _required_bound_path(
                    record.decision_patch_path,
                    evidence=evidence,
                    label="Texture coordinator decision patch",
                )
                if patch_binding.sha256 != record.decision_patch_sha256:
                    raise AssetCompositionStateError(
                        "Texture coordinator decision patch differs from its ledger"
                    )
    summary = _json_object_from_binding(
        summary_matches[0],
        label="Texture final summary",
    )
    progress = _json_object_from_binding(
        progress_matches[0],
        label="Texture workflow progress",
    )
    if validation.status != "pass":
        raise AssetCompositionStateError(
            "Texture native terminal status must pass before acceptance"
        )
    validation_output = validation.output_asset_path
    if validation_output is None or (
        Path(validation_output).expanduser().resolve() != Path(output.path)
    ):
        raise AssetCompositionStateError(
            "Texture validation evidence does not bind the accepted output"
        )
    if validation.output_asset_sha256 != output.sha256:
        raise AssetCompositionStateError(
            "Texture validation evidence output digest differs from accepted bytes"
        )
    checkpoint_unit_paths = {
        unit_id: frozenset(
            str(Path(path).expanduser().resolve()) for path in digest.sha256_by_path
        )
        for unit_id, digest in checkpoint.artifact_digests.items()
    }
    validation_unit_paths = {
        unit_id: frozenset(str(Path(path).expanduser().resolve()) for path in paths)
        for unit_id, paths in validation.unit_artifact_paths.items()
    }
    checkpoint_visual_paths = frozenset(
        str(Path(path).expanduser().resolve())
        for path in checkpoint.validation_evidence_sha256_by_path
    )
    validation_visual_paths = frozenset(
        str(Path(path).expanduser().resolve())
        for path in validation.visual_evidence_paths
    )
    for digest_record in checkpoint.artifact_digests.values():
        for raw_path, expected_sha256 in digest_record.sha256_by_path.items():
            binding = _required_bound_path(
                raw_path,
                evidence=evidence,
                additional=(output,),
                label="Texture unit artifact",
            )
            if binding.sha256 != expected_sha256:
                raise AssetCompositionStateError(
                    "Texture unit artifact digest differs from its terminal manifest"
                )
    for (
        raw_path,
        expected_sha256,
    ) in checkpoint.validation_evidence_sha256_by_path.items():
        binding = _required_bound_path(
            raw_path,
            evidence=evidence,
            additional=(output,),
            label="Texture visual validation evidence",
        )
        if binding.sha256 != expected_sha256:
            raise AssetCompositionStateError(
                "Texture visual evidence digest differs from its terminal manifest"
            )
    summary_artifacts = summary.get("artifacts")
    texture_error = "Texture final summary does not bind a passing accepted output"
    if summary.get("schema_version") != "content-agent-workflows.texture-summary.v3":
        raise AssetCompositionStateError(f"{texture_error}: unsupported summary schema")
    if summary.get("status") != "pass":
        raise AssetCompositionStateError(f"{texture_error}: summary status is not pass")
    if not isinstance(summary.get("source_asset"), str) or Path(
        str(summary["source_asset"])
    ).expanduser().resolve() != Path(input_asset.path):
        raise AssetCompositionStateError(
            f"{texture_error}: summary source differs from the stage input"
        )
    if (
        not isinstance(summary.get("output_asset_path"), str)
        or Path(str(summary["output_asset_path"])).expanduser().resolve()
        != Path(output.path)
        or summary.get("output_asset_sha256") != output.sha256
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: summary output path or digest differs from reviewed bytes"
        )
    if (
        summary.get("selected_unit_ids") != list(validation.selected_unit_ids)
        or summary.get("accepted_unit_ids") != list(validation.accepted_unit_ids)
        or summary.get("remaining_unit_ids") != list(validation.remaining_unit_ids)
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: summary unit partition differs from validation evidence"
        )
    if (
        not isinstance(summary_artifacts, dict)
        or summary_artifacts.get("workflow_checkpoint") != checkpoint_matches[0].path
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: summary does not reference the sealed checkpoint"
        )
    if summary_artifacts.get("workflow_progress") != progress_matches[0].path:
        raise AssetCompositionStateError(
            f"{texture_error}: summary does not reference the sealed progress index"
        )
    if requires_embedded_context:
        _validate_texture_embedded_decision_receipt(
            summary_artifacts,
            evidence=evidence,
            request=request,
            checkpoint=checkpoint,
            output=output,
        )
        if ledger_matches and summary_artifacts.get("decision_ledger") != (
            ledger_matches[0].path
        ):
            raise AssetCompositionStateError(
                f"{texture_error}: summary does not reference the sealed decision "
                "ledger"
            )
    if (
        checkpoint.next_action != "done"
        or checkpoint.terminal_status != "pass"
        or checkpoint.source_identity_digest != input_asset.sha256
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: checkpoint is nonterminal or bound to another source"
        )
    expected_progress = {
        "schema_version": "content-agent-workflows.texture-progress-log.v2",
        "events": [item.model_dump(mode="json") for item in checkpoint.progress],
    }
    if progress != expected_progress:
        raise AssetCompositionStateError(
            f"{texture_error}: progress index differs from the terminal checkpoint"
        )
    if (
        checkpoint.output_asset_path is None
        or Path(checkpoint.output_asset_path).expanduser().resolve()
        != Path(output.path)
        or checkpoint.output_asset_sha256 != output.sha256
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: checkpoint output differs from reviewed bytes"
        )
    if (
        checkpoint.selected_unit_ids != validation.selected_unit_ids
        or checkpoint.accepted_unit_ids != validation.accepted_unit_ids
        or checkpoint.remaining_unit_ids != validation.remaining_unit_ids
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: checkpoint and validation unit partitions differ"
        )
    if (
        checkpoint_unit_paths != validation_unit_paths
        or checkpoint_visual_paths != validation_visual_paths
    ):
        raise AssetCompositionStateError(
            f"{texture_error}: checkpoint and validation artifact manifests differ"
        )


def _validate_embedded_articulation_acceptance(
    evidence: list[ArtifactBinding],
    *,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    output: ArtifactBinding,
    review_candidates: ArtifactBinding | None,
    review_decisions: ArtifactBinding | None,
    summary: dict[str, object],
) -> None:
    """Validate only the shared embedded Joint chain and exact outer gates."""

    from content_agent_workflows.articulation import (
        ArticulationAuthoringRequest,
        ArticulationAuthoringResult,
        ArticulationRunState,
        ArticulationWorkflowRequest,
        EmbeddedArticulationCanonicalGraph,
        EmbeddedArticulationGraphRevision,
        EmbeddedArticulationOuterReview,
        EmbeddedArticulationReadback,
        validate_completed_embedded_articulation_checkpoint,
        verify_articulation_workflow_summary,
    )
    from content_agent_workflows.common.embedded_domain_decision import (
        EmbeddedCoordinatorDecision,
        canonical_json_digest,
    )

    required_names = {
        "request.json": "embedded Articulation request",
        "checkpoint.json": "embedded Articulation checkpoint",
        "approved_articulation_candidates.json": "graph-only Stage2 adapter input",
        "authoring_request.json": "embedded Articulation authoring request",
        "authoring_result.json": "embedded Articulation authoring result",
        "validation_evidence.json": "embedded Articulation validation",
        "embedded_articulation_readback.json": "embedded Articulation readback",
        "workflow_progress.json": "embedded Articulation progress",
        "final_summary.json": "embedded Articulation final summary",
    }
    matches = {
        name: [binding for binding in evidence if Path(binding.path).name == name]
        for name in required_names
    }
    if any(len(items) != 1 for items in matches.values()):
        raise AssetCompositionStateError(
            "Embedded Articulation acceptance requires one complete graph-only "
            "request/checkpoint/authoring/readback/summary chain"
        )
    root_bound_names = set(required_names)
    parents = {Path(matches[name][0].path).parent for name in root_bound_names}
    if len(parents) != 1:
        raise AssetCompositionStateError(
            "Embedded Articulation native evidence must belong to one domain run"
        )
    domain_root = next(iter(parents)).resolve()
    try:
        request = ArticulationWorkflowRequest.model_validate(
            _json_object_from_binding(
                matches["request.json"][0], label=required_names["request.json"]
            )
        )
        checkpoint = ArticulationRunState.model_validate(
            _json_object_from_binding(
                matches["checkpoint.json"][0], label=required_names["checkpoint.json"]
            )
        )
        authoring_request = ArticulationAuthoringRequest.model_validate(
            _json_object_from_binding(
                matches["authoring_request.json"][0],
                label=required_names["authoring_request.json"],
            )
        )
        authoring = ArticulationAuthoringResult.model_validate(
            _json_object_from_binding(
                matches["authoring_result.json"][0],
                label=required_names["authoring_result.json"],
            )
        )
        readback = EmbeddedArticulationReadback.model_validate(
            _json_object_from_binding(
                matches["embedded_articulation_readback.json"][0],
                label=required_names["embedded_articulation_readback.json"],
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid embedded Articulation acceptance evidence: {exc}"
        ) from exc
    if checkpoint.embedded_canonical_graph is None:
        raise AssetCompositionStateError(
            "Embedded Articulation checkpoint lacks its canonical graph"
        )
    canonical_graph_matches = [
        binding
        for binding in evidence
        if Path(binding.path).expanduser().resolve()
        == Path(checkpoint.embedded_canonical_graph.path).expanduser().resolve()
        and binding.sha256 == checkpoint.embedded_canonical_graph.sha256
    ]
    if len(canonical_graph_matches) != 1:
        raise AssetCompositionStateError(
            "Embedded Articulation acceptance requires the exact active canonical "
            "graph once, independent of preserved parent graph names"
        )
    canonical_graph_binding = canonical_graph_matches[0]
    canonical_graph_path = Path(canonical_graph_binding.path).resolve()
    if not canonical_graph_path.is_relative_to(domain_root):
        raise AssetCompositionStateError(
            "Revised canonical Articulation graph must remain inside its domain run"
        )
    try:
        graph = EmbeddedArticulationCanonicalGraph.model_validate(
            _json_object_from_binding(
                canonical_graph_binding,
                label="active canonical Articulation graph",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid embedded Articulation acceptance evidence: {exc}"
        ) from exc
    matches["canonical_articulation_graph.json"] = [canonical_graph_binding]
    _validate_embedded_domain_request(
        request.metadata,
        expected_domain="articulation",
        state_path=state_path,
        run=run,
        input_asset=input_asset,
        output_dir=request.output_dir,
    )
    if authoring_request.predictions_path is not None or (
        authoring_request.predictions_sha256 is not None
    ):
        raise AssetCompositionStateError(
            "Embedded Articulation authoring must not bind raw predictions"
        )
    if (
        graph.candidate_ids != authoring_request.accepted_candidate_ids
        or graph.candidate_ids != authoring.authored_candidate_ids
        or graph.candidate_ids != readback.joint_ids
    ):
        raise AssetCompositionStateError(
            "Embedded Articulation graph, authoring, and readback scopes differ"
        )
    if (
        Path(authoring.output_asset_path).expanduser().resolve() != Path(output.path)
        or authoring.output_asset_sha256 != output.sha256
        or readback.output_asset_sha256 != output.sha256
    ):
        raise AssetCompositionStateError(
            "Embedded Articulation accepted output differs from saved readback"
        )
    if checkpoint.schema_version in {
        "content-agent-workflows.articulation-run-state.v3",
        "content-agent-workflows.articulation-run-state.v4",
    }:
        if checkpoint.embedded_outer_review is None:
            raise AssetCompositionStateError(
                "Embedded Articulation checkpoint lacks its exact outer graph review"
            )
        outer_review_matches = [
            binding
            for binding in evidence
            if Path(binding.path).expanduser().resolve()
            == Path(checkpoint.embedded_outer_review.path).expanduser().resolve()
            and binding.sha256 == checkpoint.embedded_outer_review.sha256
        ]
        if len(outer_review_matches) != 1:
            raise AssetCompositionStateError(
                "Embedded Articulation requires one exact outer graph review"
            )
        try:
            outer_review = EmbeddedArticulationOuterReview.model_validate(
                _json_object_from_binding(
                    outer_review_matches[0],
                    label="embedded Articulation outer graph review",
                )
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid embedded Articulation outer graph review: {exc}"
            ) from exc
        if (
            checkpoint.embedded_outer_review is None
            or checkpoint.embedded_outer_review.path != outer_review_matches[0].path
            or checkpoint.embedded_outer_review.sha256 != outer_review_matches[0].sha256
            or outer_review.canonical_graph_sha256
            != matches["canonical_articulation_graph.json"][0].sha256
            or outer_review.disposition != "accept"
        ):
            raise AssetCompositionStateError(
                "Embedded Articulation outer review does not accept the exact graph"
            )
    shared_path_fields = (
        (
            "embedded_evidence_path",
            "embedded_coordinator_decision_path",
            "embedded_execution_authorization_path",
            "embedded_execution_result_path",
            "embedded_coordinator_review_path",
            "embedded_decision_receipt_path",
        )
        + (
            ("embedded_outer_review_path",)
            if checkpoint.schema_version
            in {
                "content-agent-workflows.articulation-run-state.v3",
                "content-agent-workflows.articulation-run-state.v4",
            }
            else ()
        )
        + (
            ("embedded_graph_revision_path",)
            if checkpoint.embedded_graph_revision is not None
            else ()
        )
    )
    shared_bindings = {
        field_name: _required_bound_path(
            summary.get(field_name),
            evidence=evidence,
            label=f"Articulation {field_name}",
        )
        for field_name in shared_path_fields
    }
    if checkpoint.embedded_graph_revision is not None:
        current_revision = checkpoint.embedded_graph_revision
        seen_revision_digests: set[str] = set()
        while current_revision is not None:
            if current_revision.sha256 in seen_revision_digests:
                raise AssetCompositionStateError(
                    "Embedded Articulation graph revision evidence is cyclic"
                )
            seen_revision_digests.add(current_revision.sha256)
            sealed_revision = _required_bound_path(
                current_revision.path,
                evidence=evidence,
                label="Articulation graph revision receipt",
            )
            if sealed_revision.sha256 != current_revision.sha256:
                raise AssetCompositionStateError(
                    "Articulation graph revision receipt digest differs from its "
                    "checkpoint binding"
                )
            try:
                revision = EmbeddedArticulationGraphRevision.model_validate(
                    _json_object_from_binding(
                        sealed_revision,
                        label="Articulation graph revision receipt",
                    )
                )
            except ValidationError as exc:
                raise AssetCompositionStateError(
                    f"Invalid Articulation graph revision receipt: {exc}"
                ) from exc
            for label, revision_artifact in (
                ("parent canonical graph", revision.parent_canonical_graph),
                ("parent outer review", revision.parent_outer_review),
                (
                    "parent coordinator decision",
                    revision.parent_coordinator_decision,
                ),
                ("human graph decision", revision.human_decision),
                ("revised canonical graph", revision.revised_canonical_graph),
                ("revised outer review", revision.revised_outer_review),
                (
                    "revised coordinator decision",
                    revision.revised_coordinator_decision,
                ),
            ):
                sealed = _required_bound_path(
                    revision_artifact.path,
                    evidence=evidence,
                    label=f"Articulation revision {label}",
                )
                if sealed.sha256 != revision_artifact.sha256:
                    raise AssetCompositionStateError(
                        f"Articulation revision {label} digest differs from its "
                        "revision receipt"
                    )
            if revision.parent_revision is None:
                break
            current_revision = revision.parent_revision
    proposal_path = summary.get("embedded_proposal_path")
    if proposal_path is not None:
        _required_bound_path(
            proposal_path,
            evidence=evidence,
            label="Articulation embedded_proposal_path",
        )
    try:
        decision = EmbeddedCoordinatorDecision.model_validate(
            _json_object_from_binding(
                shared_bindings["embedded_coordinator_decision_path"],
                label="embedded Articulation coordinator decision",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid embedded Articulation coordinator decision: {exc}"
        ) from exc
    if decision.human_decision_required:
        if review_candidates is None or review_decisions is None:
            raise AssetCompositionStateError(
                "human_required Articulation completion lacks exact human review"
            )
        if (
            Path(review_candidates.path).expanduser().resolve()
            != Path(matches["canonical_articulation_graph.json"][0].path)
            or review_candidates.sha256
            != matches["canonical_articulation_graph.json"][0].sha256
        ):
            raise AssetCompositionStateError(
                "Asset human gate does not bind the exact canonical graph"
            )
        decisions = _json_object_from_binding(
            review_decisions,
            label="embedded Articulation human decisions",
        )
        wrapped = decisions.get("decisions")
        raw_decisions = wrapped if isinstance(wrapped, dict) else decisions
        if set(raw_decisions) != set(graph.candidate_ids) or any(
            value != "accept" for value in raw_decisions.values()
        ):
            raise AssetCompositionStateError(
                "Completed embedded Articulation requires exact human acceptance"
            )
        human_path = summary.get("embedded_human_decision_path")
        if human_path is None:
            raise AssetCompositionStateError(
                "human_required Articulation summary lacks human decision provenance"
            )
        _required_bound_path(
            human_path,
            evidence=evidence,
            label="Articulation embedded_human_decision_path",
        )
    elif (
        review_candidates is not None
        or review_decisions is not None
        or summary.get("embedded_human_decision_path") is not None
    ):
        raise AssetCompositionStateError(
            "not_requested Articulation human review cannot invent human authority"
        )
    if (
        readback.canonical_graph_digest != canonical_json_digest(graph)
        or readback.accepted_decision_digest != decision.accepted_decision_digest
        or readback.validation_sha256 != matches["validation_evidence.json"][0].sha256
    ):
        raise AssetCompositionStateError(
            "Embedded Articulation readback differs from its graph, decision, "
            "or validation evidence"
        )
    try:
        validate_completed_embedded_articulation_checkpoint(
            checkpoint,
            authoring=authoring,
        )
        verify_articulation_workflow_summary(
            checkpoint,
            output_dir=request.output_dir,
            authoring=authoring,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AssetCompositionStateError(
            f"Invalid completed embedded Articulation checkpoint: {exc}"
        ) from exc


def _validate_articulation_coordinator_evidence(
    evidence: list[ArtifactBinding],
    *,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    output: ArtifactBinding,
    review_candidates: ArtifactBinding | None,
    review_decisions: ArtifactBinding | None,
) -> None:
    """Require the native reviewed Joint graph and exact saved-output proof."""

    final_summaries = [
        binding
        for binding in evidence
        if Path(binding.path).name == "final_summary.json"
    ]
    if len(final_summaries) == 1:
        summary_payload = _json_object_from_binding(
            final_summaries[0],
            label="Articulation final summary",
        )
        if summary_payload.get("embedded_decision_receipt_path"):
            _validate_embedded_articulation_acceptance(
                evidence,
                state_path=state_path,
                run=run,
                input_asset=input_asset,
                output=output,
                review_candidates=review_candidates,
                review_decisions=review_decisions,
                summary=summary_payload,
            )
            return

    if review_candidates is None or review_decisions is None:
        raise AssetCompositionStateError(
            "Classic Articulation acceptance requires frozen Joint review decisions "
            "and candidates"
        )

    from content_agent_workflows.articulation import (
        ArticulationAuthoringRequest,
        ArticulationAuthoringResult,
        ArticulationDecisionLedger,
        ArticulationInferenceResult,
        ArticulationReviewReceipt,
        ArticulationRunState,
        ArticulationSceneEvidenceResult,
        ArticulationValidationResult,
        ArticulationWorkflowRequest,
        Stage2CandidateDocument,
        articulation_scene_artifact_bindings,
        load_articulation_decision_ledger,
        validate_completed_articulation_checkpoint,
        verify_articulation_workflow_summary,
    )

    requires_embedded_context = (
        run.schema_version != LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION
    )
    names = {
        "authoring_result.json": "Articulation authoring result",
        "review_receipt.json": "Articulation review receipt",
        "validation_evidence.json": "Articulation validation evidence",
    }
    if requires_embedded_context:
        names["request.json"] = "Articulation workflow request"
        names.update(
            {
                "checkpoint.json": "Articulation terminal checkpoint",
                "inference_result.json": "Articulation inference result",
                "articulation_candidates.json": (
                    "Articulation original candidate document"
                ),
                "articulation_decision_patch.json": (
                    "Articulation coordinator decision patch"
                ),
                "agent_reviewed_articulation_candidates.json": (
                    "Articulation agent-reviewed candidate document"
                ),
                "articulation_decision_ledger.json": (
                    "Articulation coordinator decision ledger"
                ),
                "approved_articulation_candidates.json": (
                    "Articulation approved candidate document"
                ),
                "authoring_request.json": "Articulation authoring request",
                "workflow_progress.json": "Articulation workflow progress",
                "final_summary.json": "Articulation final summary",
            }
        )
    matches = {
        name: [binding for binding in evidence if Path(binding.path).name == name]
        for name in names
    }
    if any(len(bindings) != 1 for bindings in matches.values()):
        request_name = (
            "request.json, terminal checkpoint, inference result, original/"
            "agent-reviewed/approved candidate documents, coordinator decision "
            "patch and ledger, authoring request, workflow progress, final summary, "
            if requires_embedded_context
            else ""
        )
        raise AssetCompositionStateError(
            "Articulation acceptance requires exactly one "
            f"{request_name}authoring_result.json, review_receipt.json, and "
            "validation_evidence.json"
        )
    parents = {Path(bindings[0].path).parent for bindings in matches.values()}
    if len(parents) != 1:
        raise AssetCompositionStateError(
            "Articulation native evidence must belong to one domain run"
        )
    try:
        authoring = ArticulationAuthoringResult.model_validate(
            _json_object_from_binding(
                matches["authoring_result.json"][0],
                label=names["authoring_result.json"],
            )
        )
        receipt = ArticulationReviewReceipt.model_validate(
            _json_object_from_binding(
                matches["review_receipt.json"][0],
                label=names["review_receipt.json"],
            )
        )
        validation = ArticulationValidationResult.model_validate(
            _json_object_from_binding(
                matches["validation_evidence.json"][0],
                label=names["validation_evidence.json"],
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid native Articulation acceptance evidence: {exc}"
        ) from exc
    request = None
    inference = None
    authoring_request = None
    reviewed_candidates = None
    approved_candidates = None
    if requires_embedded_context:
        try:
            request = ArticulationWorkflowRequest.model_validate(
                _json_object_from_binding(
                    matches["request.json"][0],
                    label=names["request.json"],
                )
            )
            checkpoint = ArticulationRunState.model_validate(
                _json_object_from_binding(
                    matches["checkpoint.json"][0],
                    label=names["checkpoint.json"],
                )
            )
            inference = ArticulationInferenceResult.model_validate(
                _json_object_from_binding(
                    matches["inference_result.json"][0],
                    label=names["inference_result.json"],
                )
            )
            ledger = ArticulationDecisionLedger.model_validate(
                _json_object_from_binding(
                    matches["articulation_decision_ledger.json"][0],
                    label=names["articulation_decision_ledger.json"],
                )
            )
            approved_candidates = Stage2CandidateDocument.model_validate(
                _json_object_from_binding(
                    matches["approved_articulation_candidates.json"][0],
                    label=names["approved_articulation_candidates.json"],
                )
            )
            authoring_request = ArticulationAuthoringRequest.model_validate(
                _json_object_from_binding(
                    matches["authoring_request.json"][0],
                    label=names["authoring_request.json"],
                )
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid embedded Articulation decision evidence: {exc}"
            ) from exc
        _validate_embedded_domain_request(
            request.metadata,
            expected_domain="articulation",
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            output_dir=request.output_dir,
        )
        try:
            validate_completed_articulation_checkpoint(
                checkpoint,
                request=request,
            )
            verify_articulation_workflow_summary(
                checkpoint,
                output_dir=request.output_dir,
                authoring=authoring,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise AssetCompositionStateError(
                f"Invalid embedded Articulation terminal checkpoint: {exc}"
            ) from exc
        checkpoint_bindings = {
            "request.json": checkpoint.request,
            "inference_result.json": checkpoint.inference_result,
            "articulation_candidates.json": checkpoint.candidate_document,
            "review_receipt.json": checkpoint.review_receipt,
            "approved_articulation_candidates.json": (
                checkpoint.approved_candidate_document
            ),
            "authoring_request.json": checkpoint.authoring_request,
            "authoring_result.json": checkpoint.authoring_result,
            "validation_evidence.json": checkpoint.validation_result,
        }
        for artifact_name, checkpoint_binding in checkpoint_bindings.items():
            if checkpoint_binding is None:
                raise AssetCompositionStateError(
                    "Articulation terminal checkpoint is missing its bound "
                    f"{artifact_name}"
                )
            evidence_binding = matches[artifact_name][0]
            if (
                Path(checkpoint_binding.path).expanduser().resolve()
                != Path(evidence_binding.path)
                or checkpoint_binding.sha256 != evidence_binding.sha256
            ):
                raise AssetCompositionStateError(
                    "Articulation terminal checkpoint differs from its sealed "
                    f"{artifact_name}"
                )
        assert inference is not None
        assert authoring_request is not None
        transitive_bindings = (
            (
                inference.predictions_path,
                inference.predictions_sha256,
                "Articulation Joint Agent predictions",
            ),
            (
                inference.report_path,
                inference.report_sha256,
                "Articulation Joint Agent candidate report",
            ),
            (
                authoring_request.predictions_path,
                authoring_request.predictions_sha256,
                "Articulation authoring predictions",
            ),
            (
                authoring.diagnostics_path,
                authoring.diagnostics_sha256,
                "Articulation Joint Rigger diagnostics",
            ),
            (
                authoring.joint_rigger_result_path,
                authoring.joint_rigger_result_sha256,
                "Articulation Joint Rigger result",
            ),
        )
        for artifact_path, expected_sha256, label in transitive_bindings:
            if artifact_path is None:
                continue
            sealed_binding = _required_bound_path(
                artifact_path,
                evidence=evidence,
                label=label,
            )
            if sealed_binding.sha256 != expected_sha256:
                raise AssetCompositionStateError(
                    f"{label} differs from its checkpointed digest"
                )
        try:
            decision_context = load_articulation_decision_ledger(
                request.output_dir,
                state=checkpoint,
            )
        except (OSError, ValueError) as exc:
            raise AssetCompositionStateError(
                f"Invalid embedded Articulation decision ledger: {exc}"
            ) from exc
        if decision_context is None:
            raise AssetCompositionStateError(
                "Articulation embedded acceptance requires a decision ledger"
            )
        _decision_patch, reviewed_candidates = decision_context
        ledger_bindings = {
            "articulation_candidates.json": ledger.original_candidate_document,
            "articulation_decision_patch.json": ledger.decision_patch,
            "agent_reviewed_articulation_candidates.json": (
                ledger.reviewed_candidate_document
            ),
        }
        for artifact_name, ledger_binding in ledger_bindings.items():
            evidence_binding = matches[artifact_name][0]
            if (
                Path(ledger_binding.path).expanduser().resolve()
                != Path(evidence_binding.path)
                or ledger_binding.sha256 != evidence_binding.sha256
            ):
                raise AssetCompositionStateError(
                    "Articulation coordinator decision ledger differs from its "
                    f"sealed {artifact_name}"
                )
        if checkpoint.scene_evidence is not None:
            scene_binding = _required_bound_path(
                checkpoint.scene_evidence.path,
                evidence=evidence,
                label="Articulation scene decision evidence",
            )
            if scene_binding.sha256 != checkpoint.scene_evidence.sha256:
                raise AssetCompositionStateError(
                    "Articulation scene evidence differs from its checkpoint"
                )
            try:
                scene_manifest = ArticulationSceneEvidenceResult.model_validate(
                    _json_object_from_binding(
                        scene_binding,
                        label="Articulation scene evidence manifest",
                    )
                )
            except ValidationError as exc:
                raise AssetCompositionStateError(
                    f"Invalid embedded Articulation scene evidence: {exc}"
                ) from exc
            for nested_binding in articulation_scene_artifact_bindings(scene_manifest):
                sealed_binding = _required_bound_path(
                    nested_binding.path,
                    evidence=evidence,
                    label="Articulation scene nested evidence",
                )
                if sealed_binding.sha256 != nested_binding.sha256:
                    raise AssetCompositionStateError(
                        "Articulation scene nested evidence differs from its manifest"
                    )
    accepted_ids = tuple(
        decision.candidate_id
        for decision in receipt.decisions
        if decision.decision == "accept"
    )
    raw_decisions = _json_object_from_binding(
        review_decisions,
        label="frozen Articulation review decisions",
    )
    nested_decisions = raw_decisions.get("decisions")
    decision_payload = (
        nested_decisions if isinstance(nested_decisions, dict) else raw_decisions
    )
    receipt_decisions = {
        decision.candidate_id: decision.decision for decision in receipt.decisions
    }
    articulation_error = (
        "Articulation native evidence does not prove the exact reviewed output"
    )
    if request is not None and (
        receipt.request_sha256 != matches["request.json"][0].sha256
        or Path(request.source_asset).expanduser().resolve() != Path(input_asset.path)
        or request.output_dir.expanduser().resolve()
        != Path(matches["request.json"][0].path).parent.resolve()
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: native request differs from the embedded input"
        )
    if reviewed_candidates is not None and (
        review_candidates.path
        != matches["agent_reviewed_articulation_candidates.json"][0].path
        or review_candidates.sha256
        != matches["agent_reviewed_articulation_candidates.json"][0].sha256
        or receipt.candidate_document_sha256 != review_candidates.sha256
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: human review did not bind the agent-reviewed "
            "candidate document"
        )
    reviewer_transitions = [
        transition
        for transition in run.transitions
        if transition.stage == "articulation"
        and transition.attempt_count == run.stages["articulation"].attempt_count
        and transition.from_status == "needs_review"
        and transition.to_status == "ready"
    ]
    if (
        len(reviewer_transitions) != 1
        or receipt.reviewer != reviewer_transitions[0].actor
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: receipt reviewer differs from the recorded "
            "human reviewer"
        )
    if validation.status != "pass":
        raise AssetCompositionStateError(
            f"{articulation_error}: saved-graph validation did not pass"
        )
    if (
        Path(validation.output_asset_path).expanduser().resolve() != Path(output.path)
        or validation.expected_output_asset_sha256 != output.sha256
        or validation.observed_output_asset_sha256 != output.sha256
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: validation output path or digest differs"
        )
    if authoring.output_asset_sha256 != output.sha256 or Path(
        authoring.output_asset_path
    ).expanduser().resolve() != Path(output.path):
        raise AssetCompositionStateError(
            f"{articulation_error}: authoring output path or digest differs"
        )
    if (
        authoring.authored_candidate_ids != validation.expected_candidate_ids
        or validation.validated_candidate_ids != validation.expected_candidate_ids
        or accepted_ids != validation.expected_candidate_ids
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: authored, reviewed, and validated candidate IDs differ"
        )
    if decision_payload != receipt_decisions:
        raise AssetCompositionStateError(
            f"{articulation_error}: receipt differs from frozen human decisions"
        )
    if (
        authoring.source_sha256 != input_asset.sha256
        or receipt.source_sha256 != input_asset.sha256
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: source digest differs from the stage input"
        )
    if approved_candidates is not None and reviewed_candidates is not None:
        reviewed_by_id = reviewed_candidates.candidate_by_id()
        if (
            approved_candidates.candidate_ids != accepted_ids
            or any(
                candidate != reviewed_by_id.get(candidate.candidate_id)
                for candidate in approved_candidates.candidates
            )
            or authoring.candidate_document_sha256
            != matches["approved_articulation_candidates.json"][0].sha256
            or Path(authoring.candidate_document_path).expanduser().resolve()
            != Path(matches["approved_articulation_candidates.json"][0].path)
        ):
            raise AssetCompositionStateError(
                f"{articulation_error}: approved authoring candidates differ from "
                "the reviewed accept set"
            )
    elif (
        authoring.candidate_document_sha256 != receipt.candidate_document_sha256
        or authoring.candidate_document_sha256 != review_candidates.sha256
        or Path(authoring.candidate_document_path).expanduser().resolve()
        != Path(review_candidates.path)
    ):
        raise AssetCompositionStateError(
            f"{articulation_error}: candidate receipt differs from frozen review bytes"
        )


def _validation_source_identities(
    payload: object,
    *,
    label: str,
) -> tuple[object, ...]:
    from content_agent_workflows.validation import ValidationArtifactIdentity

    if not isinstance(payload, list):
        raise AssetCompositionStateError(f"{label} must be a list")
    try:
        return tuple(
            ValidationArtifactIdentity.model_validate(item) for item in payload
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(f"Invalid {label}: {exc}") from exc


def _native_physics_digest(
    binding: ArtifactBinding,
    *,
    label: str,
) -> str:
    """Compute the digest format emitted by native Physics evidence."""

    from world_understanding.functions.physics.physics_topology import sha256_file

    try:
        digest = sha256_file(binding.path)
    except Exception as exc:  # noqa: BLE001 - normalize the USD boundary
        raise AssetCompositionStateError(
            f"Could not compute {label} composed digest: {exc}"
        ) from exc
    _verify_binding(binding, label=label)
    return digest


def _validate_physics_coordinator_evidence(
    evidence: list[ArtifactBinding],
    *,
    input_asset: ArtifactBinding,
    output: ArtifactBinding,
    validation_mode: PhysicsValidationMode,
) -> None:
    """Require coordinator-authored decisions and their deterministic result."""

    from content_agent_workflows.physics import (
        PhysicsComponentDecision,
        PhysicsComponentTargetDecision,
        inspect_physics_components,
        parse_physics_component_catalog_entry,
        parse_physics_decision_patch,
        physics_decision_assignment_payload,
        rebase_physics_v2_patch_to_components,
        resolve_physics_v2_patch_targets,
    )
    from content_agent_workflows.physics.workflow import (
        PHYSICS_ASSIGNMENTS_SCHEMA_VERSION,
        PHYSICS_DECISION_PATCH_SCHEMA_VERSION,
    )

    required_names = (
        "coordinator_physics_decision_patch.json",
        "physics_decision_patch.json",
        "physics_assignments.json",
        "physics_components.json",
        "physics_apply_report.json",
    )
    matches = {
        name: [binding for binding in evidence if Path(binding.path).name == name]
        for name in required_names
    }
    if any(len(bindings) != 1 for bindings in matches.values()):
        raise AssetCompositionStateError(
            "Physics acceptance requires one coordinator decision patch plus the "
            "native decision patch, assignments, and component inventory"
        )
    coordinator_binding = matches["coordinator_physics_decision_patch.json"][0]
    native_binding = matches["physics_decision_patch.json"][0]
    assignments_binding = matches["physics_assignments.json"][0]
    components_binding = matches["physics_components.json"][0]
    apply_report_binding = matches["physics_apply_report.json"][0]
    coordinator_patch = _json_object_from_binding(
        coordinator_binding,
        label="coordinator Physics decision patch",
    )
    native_patch = _json_object_from_binding(
        native_binding,
        label="native Physics decision patch",
    )
    assignments = _json_object_from_binding(
        assignments_binding,
        label="Physics assignments",
    )
    components = _json_object_from_binding(
        components_binding,
        label="Physics component inventory",
    )
    apply_report = _json_object_from_binding(
        apply_report_binding,
        label="Physics schema-application report",
    )
    input_physics_digest = _native_physics_digest(
        input_asset,
        label="Physics stage input",
    )
    try:
        coordinator_decisions = parse_physics_decision_patch(
            coordinator_patch,
            label="coordinator Physics decision patch",
        )
        native_decisions = parse_physics_decision_patch(
            native_patch,
            label="native Physics decision patch",
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Invalid coordinator-owned Physics decision evidence: {exc}"
        ) from exc
    physics_input_error = (
        "Physics native decisions do not match the coordinator-authored input"
    )
    if coordinator_patch.get("schema_version") != PHYSICS_DECISION_PATCH_SCHEMA_VERSION:
        raise AssetCompositionStateError(
            f"{physics_input_error}: coordinator patch schema is not V2"
        )
    if native_patch != coordinator_patch or native_decisions != coordinator_decisions:
        raise AssetCompositionStateError(
            f"{physics_input_error}: native patch differs from coordinator decisions"
        )
    if not coordinator_decisions:
        raise AssetCompositionStateError(
            f"{physics_input_error}: coordinator patch has no decisions"
        )
    if (
        coordinator_patch.get("source_digest") != input_physics_digest
        or not isinstance(coordinator_patch.get("asset"), str)
        or Path(str(coordinator_patch["asset"])).expanduser().resolve()
        != Path(input_asset.path)
    ):
        raise AssetCompositionStateError(
            f"{physics_input_error}: patch source differs from the stage input"
        )

    prepared_binding = _required_bound_path(
        assignments.get("prepared_asset"),
        evidence=evidence,
        additional=(input_asset, output),
        label="Physics prepared asset",
    )
    apply_patch_binding = _required_bound_path(
        assignments.get("apply_decision_patch"),
        evidence=evidence,
        label="Physics applied decision patch",
    )
    validation_binding = _required_bound_path(
        assignments.get("validation_evidence"),
        evidence=evidence,
        label="Physics validation evidence",
    )
    simulation_binding = _required_bound_path(
        assignments.get("simulation_report"),
        evidence=evidence,
        label="Physics runtime trajectory report",
    )
    simulation_report = _json_object_from_binding(
        simulation_binding,
        label="Physics runtime trajectory report",
    )
    trajectory_binding: ArtifactBinding | None = None
    if simulation_report.get("not_evaluated") is not True:
        trajectory_binding = _required_bound_path(
            simulation_report.get("trajectory_jsonl"),
            evidence=evidence,
            label="Physics runtime trajectory",
        )
    applied_patch = _json_object_from_binding(
        apply_patch_binding,
        label="Physics applied decision patch",
    )
    prepared_physics_digest = _native_physics_digest(
        prepared_binding,
        label="Physics prepared asset",
    )
    try:
        applied_decisions = parse_physics_decision_patch(
            applied_patch,
            label="applied Physics decision patch",
        )
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Invalid applied Physics decision patch: {exc}"
        ) from exc
    physics_assignment_error = (
        "Physics assignments do not prove the exact coordinator decision result"
    )
    if (
        applied_patch.get("schema_version") != PHYSICS_DECISION_PATCH_SCHEMA_VERSION
        or applied_patch.get("source_digest") != prepared_physics_digest
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: applied patch differs from prepared input"
        )
    resolved_applied_decisions: list[PhysicsComponentDecision] = []
    for decision in applied_decisions:
        if not isinstance(decision, PhysicsComponentDecision):
            raise AssetCompositionStateError(
                f"{physics_assignment_error}: applied patch must contain resolved V2 "
                "component decisions"
            )
        resolved_applied_decisions.append(decision)
    raw_components = components.get("components")
    raw_assignments = assignments.get("decisions")
    if (
        assignments.get("schema_version") != PHYSICS_ASSIGNMENTS_SCHEMA_VERSION
        or assignments.get("asset") != input_asset.path
        or assignments.get("source_asset_sha256") != input_asset.sha256
        or assignments.get("physics_usd") != output.path
        or assignments.get("decision_patch") != native_binding.path
        or assignments.get("prepared_asset_sha256") != prepared_binding.sha256
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: stage input, output, or native patch differs"
        )
    component_path_space = components.get("path_space")
    raw_component_expansions = components.get("source_path_expansions", {})
    if component_path_space not in {"source", "inspection"} or not isinstance(
        raw_component_expansions, dict
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: component path provenance is invalid"
        )
    component_expansions: dict[str, list[str]] = {}
    for runtime_path, source_paths in raw_component_expansions.items():
        if (
            not isinstance(runtime_path, str)
            or not isinstance(source_paths, list)
            or not all(isinstance(source_path, str) for source_path in source_paths)
        ):
            raise AssetCompositionStateError(
                f"{physics_assignment_error}: component path provenance is invalid"
            )
        component_expansions[runtime_path] = source_paths
    if (
        assignments.get("path_space") != component_path_space
        or assignments.get("source_path_expansions") != component_expansions
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: assignment path provenance differs from "
            "the inspected component inventory"
        )
    component_path_space = str(component_path_space)
    try:
        expected_assignments = [
            physics_decision_assignment_payload(
                decision,
                path_space=component_path_space,
                source_path_expansions=component_expansions,
            )
            for decision in resolved_applied_decisions
        ]
    except (TypeError, ValueError) as exc:
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: invalid path provenance: {exc}"
        ) from exc
    if (
        assignments.get("decision_count") != len(resolved_applied_decisions)
        or raw_assignments != expected_assignments
        or assignments.get("unresolved_components") not in ([], None)
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: applied decisions or unresolved set differs"
        )
    if assignments.get("apply_report") != apply_report:
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: assignments embed another apply report"
        )
    if (
        not isinstance(raw_components, list)
        or not raw_components
        or components.get("component_count") != len(raw_components)
        or components.get("asset") != prepared_binding.path
        or components.get("source_digest") != prepared_physics_digest
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: component inventory differs from prepared input"
        )
    if applied_patch != coordinator_patch:
        try:
            coordinator_rebase_patch = coordinator_patch
            if any(
                isinstance(decision, PhysicsComponentTargetDecision)
                for decision in coordinator_decisions
            ):
                coordinator_rebase_patch, _, _ = resolve_physics_v2_patch_targets(
                    coordinator_patch,
                    components=inspect_physics_components(input_asset.path),
                    source_digest=input_physics_digest,
                )
            prepared_components = [
                parse_physics_component_catalog_entry(component)
                for component in raw_components
            ]
            (
                expected_applied_patch,
                expected_applied_decisions,
                expected_unresolved,
            ) = rebase_physics_v2_patch_to_components(
                coordinator_rebase_patch,
                components=prepared_components,
                source_digest=prepared_physics_digest,
                asset=Path(prepared_binding.path),
            )
        except (RuntimeError, TypeError, ValueError, ValidationError) as exc:
            raise AssetCompositionStateError(
                f"{physics_assignment_error}: deterministic topology rebase failed: "
                f"{exc}"
            ) from exc
        if (
            expected_unresolved
            or applied_patch != expected_applied_patch
            or resolved_applied_decisions != expected_applied_decisions
        ):
            raise AssetCompositionStateError(
                f"{physics_assignment_error}: applied patch is not the deterministic "
                "rebase of coordinator decisions"
            )
    if (
        validation_binding.path == apply_report_binding.path
        or simulation_binding.path == apply_report_binding.path
        or simulation_binding.path == validation_binding.path
    ):
        raise AssetCompositionStateError(
            f"{physics_assignment_error}: apply, validation, and simulation evidence "
            "must be distinct"
        )
    runtime_not_evaluated = simulation_report.get("not_evaluated") is True
    if validation_mode == "runtime_required" and runtime_not_evaluated:
        raise AssetCompositionStateError(
            "Physics runtime report is marked not_evaluated; execute simulation"
        )
    _require_exact_claim(
        simulation_report.get("error"),
        None,
        label="Physics runtime error",
    )
    _require_exact_claim(
        simulation_report.get("failures"),
        [],
        label="Physics runtime failures",
    )
    engine = simulation_report.get("engine")
    if not isinstance(engine, str) or engine in {"", "none"}:
        raise AssetCompositionStateError(
            f"Physics runtime engine is not evaluated; observed={engine!r}"
        )
    raw_simulation_output = simulation_report.get("physics_usd")
    observed_simulation_output = (
        str(Path(raw_simulation_output).expanduser().resolve())
        if isinstance(raw_simulation_output, str)
        else raw_simulation_output
    )
    _require_exact_claim(
        observed_simulation_output,
        output.path,
        label="Physics runtime output path",
    )
    if runtime_not_evaluated:
        _require_exact_claim(
            simulation_report.get("trajectory_jsonl"),
            None,
            label="Physics unevaluated runtime trajectory",
        )
        warnings = simulation_report.get("warnings")
        if (
            not isinstance(warnings, list)
            or not warnings
            or any(
                not isinstance(warning, str) or not warning.strip()
                for warning in warnings
            )
        ):
            raise AssetCompositionStateError(
                "Physics schema-readback runtime gap requires an explicit warning"
            )
    elif trajectory_binding is None or trajectory_binding.size_bytes == 0:
        raise AssetCompositionStateError("Physics runtime trajectory is empty")

    topology_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "coordinator_physics_topology_plan.json"
    ]
    topology_report_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "physics_topology_report.json"
    ]
    # A prepared artifact is a topology handoff only when its complete binding
    # identity (resolved path, digest, and size) differs from the stage input.
    topology_changed = prepared_binding != input_asset
    if (
        len(topology_matches) > 1
        or len(topology_report_matches) > 1
        or (
            topology_changed
            and (len(topology_matches) != 1 or len(topology_report_matches) != 1)
        )
    ):
        raise AssetCompositionStateError(
            "Physics topology changes require one coordinator-owned topology plan "
            "and one native topology report"
        )
    if not topology_changed and (topology_matches or topology_report_matches):
        raise AssetCompositionStateError(
            "Physics topology evidence requires a distinct prepared asset"
        )
    if topology_changed:
        topology_plan = _json_object_from_binding(
            topology_matches[0],
            label="coordinator Physics topology plan",
        )
        topology_report = _json_object_from_binding(
            topology_report_matches[0],
            label="native Physics topology report",
        )
        expected_topology_schema = "content-workflows.physics-topology-plan.v1"
        _require_exact_claim(
            topology_plan.get("schema_version"),
            expected_topology_schema,
            label="Physics topology plan schema",
        )
        _require_exact_claim(
            topology_plan.get("expected_source_digest"),
            input_physics_digest,
            label="Physics topology plan source digest",
        )
        _require_exact_claim(
            topology_report.get("operation"),
            "physics.apply_topology_plan",
            label="Physics topology report operation",
        )
        _require_exact_claim(
            topology_report.get("schema_version"),
            expected_topology_schema,
            label="Physics topology report schema",
        )
        raw_topology_input = topology_report.get("input_usd_path")
        observed_topology_input = (
            str(Path(raw_topology_input).expanduser().resolve())
            if isinstance(raw_topology_input, str)
            else raw_topology_input
        )
        _require_exact_claim(
            observed_topology_input,
            input_asset.path,
            label="Physics topology report input path",
        )
        _require_exact_claim(
            topology_report.get("source_digest"),
            input_physics_digest,
            label="Physics topology report source digest",
        )
        raw_topology_output = topology_report.get("output_usd_path")
        observed_topology_output = (
            str(Path(raw_topology_output).expanduser().resolve())
            if isinstance(raw_topology_output, str)
            else raw_topology_output
        )
        _require_exact_claim(
            observed_topology_output,
            prepared_binding.path,
            label="Physics topology report output path",
        )
        _require_exact_claim(
            topology_report.get("output_digest"),
            prepared_physics_digest,
            label="Physics topology report output digest",
        )
        for key in ("mobility_intent", "invariants"):
            _require_exact_claim(
                topology_report.get(key),
                topology_plan.get(key),
                label=f"Physics topology report {key}",
            )
        raw_applied_operations = topology_report.get("applied_operations")
        normalized_applied_operations: list[dict[str, object]] = []
        if not isinstance(raw_applied_operations, list):
            raise AssetCompositionStateError(
                "Physics topology report applied_operations must be a list"
            )
        for index, operation in enumerate(raw_applied_operations):
            # This is a frozen v1 evidence receipt, not extensible debug
            # telemetry. Unknown fields must fail closed until the shared
            # topology-plan schema and both producer and consumer are revised.
            if not isinstance(operation, dict) or set(operation) - {
                "op",
                "prim_path",
                "reset_xform_stack",
            }:
                raise AssetCompositionStateError(
                    "Physics topology report has an invalid applied operation at "
                    f"index {index}"
                )
            if "reset_xform_stack" in operation and (
                operation.get("op") != "ensure_rigid_body_api"
                or operation.get("reset_xform_stack") != "preserve_world"
            ):
                raise AssetCompositionStateError(
                    "Physics topology report has an invalid reset-xform annotation"
                )
            normalized_applied_operations.append(
                {
                    "op": operation.get("op"),
                    "prim_path": operation.get("prim_path"),
                }
            )
        _require_exact_claim(
            normalized_applied_operations,
            topology_plan.get("operations"),
            label="Physics topology report applied_operations",
        )
        raw_planned_promotions = topology_plan.get(
            "joint_endpoint_owner_promotions", []
        )
        raw_applied_promotions = topology_report.get(
            "applied_joint_endpoint_owner_promotions", []
        )
        if not isinstance(raw_planned_promotions, list) or not isinstance(
            raw_applied_promotions, list
        ):
            raise AssetCompositionStateError(
                "Physics joint endpoint owner promotions must be lists"
            )
        promotion_fields = {
            "joint_prim_path",
            "relationship",
            "relationship_target_path",
            "requested_rigid_body_ancestor_path",
        }
        report_promotion_fields = promotion_fields | {
            "before_rigid_body_paths",
            "after_rigid_body_paths",
        }

        def normalize_promotion(
            promotion: object,
            *,
            index: int,
            report: bool,
        ) -> dict[str, object]:
            expected_fields = report_promotion_fields if report else promotion_fields
            if not isinstance(promotion, dict) or set(promotion) != expected_fields:
                raise AssetCompositionStateError(
                    "Physics topology evidence has an invalid joint endpoint owner "
                    f"promotion at index {index}"
                )
            normalized = {field: promotion.get(field) for field in promotion_fields}
            if normalized["relationship"] not in {"body0", "body1"} or any(
                not isinstance(normalized[field], str) or not normalized[field]
                for field in promotion_fields - {"relationship"}
            ):
                raise AssetCompositionStateError(
                    "Physics topology evidence has malformed joint endpoint owner "
                    f"paths at index {index}"
                )
            if report:
                before_paths = promotion.get("before_rigid_body_paths")
                after_paths = promotion.get("after_rigid_body_paths")
                if (
                    not isinstance(before_paths, list)
                    or any(not isinstance(path, str) for path in before_paths)
                    or after_paths != [normalized["requested_rigid_body_ancestor_path"]]
                ):
                    raise AssetCompositionStateError(
                        "Physics topology report has invalid observed owner paths at "
                        f"promotion index {index}"
                    )
            return normalized

        planned_promotions = [
            normalize_promotion(promotion, index=index, report=False)
            for index, promotion in enumerate(raw_planned_promotions)
        ]
        applied_promotions = [
            normalize_promotion(promotion, index=index, report=True)
            for index, promotion in enumerate(raw_applied_promotions)
        ]

        def promotion_sort_key(promotion: dict[str, object]) -> tuple[str, str, str]:
            return (
                str(promotion["joint_prim_path"]),
                str(promotion["relationship"]),
                str(promotion["relationship_target_path"]),
            )

        _require_exact_claim(
            sorted(applied_promotions, key=promotion_sort_key),
            sorted(planned_promotions, key=promotion_sort_key),
            label="Physics topology report joint endpoint owner promotions",
        )
        _require_exact_claim(
            topology_report.get("rejected_operations"),
            [],
            label="Physics topology report rejected operations",
        )
        _require_exact_claim(
            topology_report.get("invariant_results"),
            {
                "enabled_collider_count_preserved": True,
                "articulation_changes_rejected": True,
            },
            label="Physics topology report invariant results",
        )


def _validate_validation_coordinator_evidence(
    evidence: list[ArtifactBinding],
    *,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    input_dependencies: list[ArtifactBinding],
    output: ArtifactBinding,
    output_dependencies: list[ArtifactBinding],
) -> None:
    """Require native evidence plus the accepted outer Validation receipt."""

    from world_understanding.validation import (
        ValidationResult,
        aggregate_validation_verdict,
    )

    from content_agent_workflows.validation import (
        CANONICAL_VALIDATION_ASSESSMENT_NAME,
        EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
        EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME,
        VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION,
        VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION,
        EmbeddedValidationAssessmentError,
        ValidationWorkflowCheckpoint,
        validate_completed_embedded_validation_receipt,
    )

    names = (
        "validation_result.json",
        "validation_evidence.json",
        "final_summary.json",
        "validation_checkpoint.json",
    )
    matches = {
        name: [binding for binding in evidence if Path(binding.path).name == name]
        for name in names
    }
    if any(len(bindings) != 1 for bindings in matches.values()):
        raise AssetCompositionStateError(
            "Validation acceptance requires exactly one result, evidence, summary, "
            "and checkpoint artifact"
        )
    parents = {Path(bindings[0].path).parent for bindings in matches.values()}
    if len(parents) != 1:
        raise AssetCompositionStateError(
            "Validation native evidence must belong to one domain run"
        )
    cross_stage_matches = [
        binding
        for binding in evidence
        if Path(binding.path).name == "cross_stage_validation.json"
    ]
    if len(cross_stage_matches) != 1:
        raise AssetCompositionStateError(
            "Validation acceptance requires one cross_stage_validation.json"
        )
    embedded_names = (
        EMBEDDED_VALIDATION_EVIDENCE_INDEX_NAME,
        EMBEDDED_VALIDATION_EXECUTION_INDEX_NAME,
        EMBEDDED_VALIDATION_RECEIPT_INDEX_NAME,
        CANONICAL_VALIDATION_ASSESSMENT_NAME,
    )
    embedded_matches = {
        name: [binding for binding in evidence if Path(binding.path).name == name]
        for name in embedded_names
    }
    if any(len(bindings) != 1 for bindings in embedded_matches.values()):
        raise AssetCompositionStateError(
            "Validation acceptance requires exactly one embedded evidence index, "
            "execution index, completed receipt index, and canonical outer "
            "assessment"
        )
    embedded_parents = {
        Path(bindings[0].path).parent for bindings in embedded_matches.values()
    }
    if embedded_parents != parents:
        raise AssetCompositionStateError(
            "Validation embedded assessment artifacts must belong to the native "
            "domain run"
        )
    try:
        result = ValidationResult.model_validate(
            _json_object_from_binding(
                matches["validation_result.json"][0],
                label="Validation result",
            )
        )
        checkpoint = ValidationWorkflowCheckpoint.model_validate(
            _json_object_from_binding(
                matches["validation_checkpoint.json"][0],
                label="Validation checkpoint",
            )
        )
        cross_stage = AssetCrossStageValidation.model_validate(
            _json_object_from_binding(
                cross_stage_matches[0],
                label="cross-stage Validation receipt",
            )
        )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Invalid native Validation acceptance evidence: {exc}"
        ) from exc
    native_evidence = _json_object_from_binding(
        matches["validation_evidence.json"][0],
        label="Validation evidence",
    )
    summary = _json_object_from_binding(
        matches["final_summary.json"][0],
        label="Validation final summary",
    )
    domain_run_root = next(iter(parents))
    try:
        assessment, decision_receipt = validate_completed_embedded_validation_receipt(
            domain_run_root
        )
    except EmbeddedValidationAssessmentError as exc:
        raise AssetCompositionStateError(
            f"Invalid embedded Validation assessment receipt: {exc}"
        ) from exc
    decision_identity = decision_receipt.identity
    execution_context = decision_identity.execution_context
    embedded_stage = execution_context.embedded_stage
    if not run.coordinator.plan_revisions:
        raise AssetCompositionStateError(
            "Validation acceptance lacks a coordinator plan"
        )
    current_plan = run.coordinator.plan_revisions[-1]
    expected_context_bindings = {
        "outer_request": (embedded_stage.outer_request if embedded_stage else None),
        "coordinator_plan": (
            embedded_stage.coordinator_plan if embedded_stage else None
        ),
        "input_asset": embedded_stage.input_asset if embedded_stage else None,
    }
    expected_live_bindings = {
        "outer_request": run.request,
        "coordinator_plan": current_plan,
        "input_asset": input_asset,
    }
    context_bindings_match = embedded_stage is not None and all(
        observed is not None
        and observed.model_dump(mode="json") == expected.model_dump(mode="json")
        for name, observed in expected_context_bindings.items()
        for expected in (expected_live_bindings[name],)
    )
    if (
        execution_context.domain != "validation"
        or execution_context.mode != "embedded"
        or execution_context.reasoning_loop_owner != "asset_coordinator"
        or embedded_stage is None
        or embedded_stage.outer_run_id != run.run_id
        or embedded_stage.stage != "validation"
        or embedded_stage.stage_attempt != run.stages["validation"].attempt_count
        or Path(embedded_stage.domain_run_root).expanduser().resolve()
        != domain_run_root
        or not context_bindings_match
        or decision_identity.source.model_dump(mode="json")
        != input_asset.model_dump(mode="json")
        or decision_identity.coordinator_plan.sha256 != current_plan.sha256
        or decision_identity.coordinator_plan.artifact_id
        != Path(current_plan.path).stem
    ):
        raise AssetCompositionStateError(
            "Embedded Validation decision identity differs from the active outer "
            "coordinator attempt"
        )
    expected_validation_digests = {
        "validation_workflow_identity": checkpoint.workflow_identity.identity_digest,
        "validation_plan": checkpoint.plan_digest,
        "validation_result": matches["validation_result.json"][0].sha256,
        "validation_evidence": matches["validation_evidence.json"][0].sha256,
    }
    for name, expected_digest in expected_validation_digests.items():
        if decision_identity.digests.configuration.get(name) != expected_digest:
            raise AssetCompositionStateError(
                f"Embedded Validation decision identity has stale {name}"
            )
    if (
        assessment.terminal_disposition != "pass"
        or decision_receipt.receipt_status != "completed"
        or decision_receipt.review_disposition != "accept"
        or decision_receipt.execution_effect != "non_mutating"
        or decision_receipt.mutation_id is not None
    ):
        raise AssetCompositionStateError(
            "Outer-authored Validation assessment is not accepted non-mutating "
            "completion authority"
        )
    before = _validation_source_identities(
        native_evidence.get("source_before"),
        label="Validation source_before",
    )
    after = _validation_source_identities(
        native_evidence.get("source_after"),
        label="Validation source_after",
    )
    source_matches = any(
        getattr(identity, "kind", None) == "file"
        and Path(str(getattr(identity, "path", ""))).expanduser().resolve()
        == Path(input_asset.path)
        and getattr(identity, "sha256", None) == input_asset.sha256
        for identity in before
    )
    all_completed = bool(checkpoint.records) and all(
        record.state.value == "completed" and record.accepted_result is not None
        for record in checkpoint.records
    )
    accepted_template_results = tuple(
        record.accepted_result.result
        for record in checkpoint.records
        if record.accepted_result is not None
    )
    checkpoint_templates = tuple(record.template_name for record in checkpoint.records)
    result_templates = tuple(
        template_result.template_name for template_result in result.template_results
    )
    native_templates = native_evidence.get("templates")
    native_template_names = (
        tuple(native_templates) if isinstance(native_templates, dict) else ()
    )
    native_template_statuses_match = isinstance(native_templates, dict) and all(
        isinstance(native_templates.get(record.template_name), dict)
        and native_templates[record.template_name].get("status")
        == record.accepted_result.result.status
        for record in checkpoint.records
        if record.accepted_result is not None
    )
    upstream_stages: tuple[CrossStageHandoffName, ...] = (
        "articulation",
        "material",
        "texture",
        "physics",
    )
    expected_handoffs: dict[CrossStageHandoffName, ArtifactBinding] = {}
    for stage in upstream_stages:
        handoff = run.stages[stage].handoff
        if handoff is None:
            raise AssetCompositionStateError(
                f"Cross-stage Validation requires the accepted {stage} handoff"
            )
        expected_handoffs[stage] = handoff
    native_validation_bindings = {matches[name][0] for name in names}
    stage_claim_bindings = {
        stage: {
            *run.stages[stage].evidence,
            expected_handoffs[stage],
        }
        for stage in upstream_stages
    }
    articulation_output = run.stages["articulation"].output_asset
    if articulation_output is None:  # pragma: no cover - guarded by handoff checks
        raise AssetCompositionStateError(
            "Cross-stage Validation requires the accepted Articulation output"
        )
    reviewed_joint_graph = _joint_graph_signature(
        Path(articulation_output.path),
        label="accepted Articulation",
    )
    final_joint_graph = _joint_graph_signature(
        Path(input_asset.path),
        label="Validation input",
    )
    if final_joint_graph != reviewed_joint_graph:
        raise AssetCompositionStateError(
            "Validation input Joint graph differs from the accepted Articulation graph"
        )
    allowed_claim_bindings = {
        "joint_graph": {
            *stage_claim_bindings["articulation"],
            *native_validation_bindings,
        },
        "appearance": {
            *stage_claim_bindings["material"],
            *stage_claim_bindings["texture"],
        },
        "physics_behavior": stage_claim_bindings["physics"],
        "render_and_package": native_validation_bindings,
        "non_target_preservation": {
            *stage_claim_bindings["articulation"],
            *stage_claim_bindings["material"],
            *stage_claim_bindings["texture"],
            *stage_claim_bindings["physics"],
            *native_validation_bindings,
        },
    }
    cross_claims_valid = all(
        set(claim.evidence) <= allowed_claim_bindings[claim.name]
        and (
            claim.name not in {"joint_graph", "non_target_preservation"}
            or bool(set(claim.evidence) & native_validation_bindings)
        )
        and (
            claim.name != "joint_graph"
            or bool(set(claim.evidence) & stage_claim_bindings["articulation"])
        )
        for claim in cross_stage.claims
    )
    physics_claim = next(
        claim for claim in cross_stage.claims if claim.name == "physics_behavior"
    )
    expected_physics_claim_status = (
        "warn"
        if _physics_validation_mode_from_run(run) == "schema_readback"
        else "pass"
    )
    if physics_claim.status != expected_physics_claim_status:
        raise AssetCompositionStateError(
            "Validation physics_behavior claim must match the frozen Physics "
            f"acceptance mode; expected={expected_physics_claim_status!r}, "
            f"observed={physics_claim.status!r}"
        )
    validation_error = (
        "Validation native evidence does not prove a completed unchanged input"
    )
    if (
        output.sha256 != input_asset.sha256
        or output.size_bytes != input_asset.size_bytes
    ):
        raise AssetCompositionStateError(
            f"{validation_error}: stage output bytes differ from the input"
        )
    if output_dependencies != input_dependencies:
        raise AssetCompositionStateError(
            f"{validation_error}: stage output dependency closure differs from "
            "the input"
        )
    if not result.template_results:
        raise AssetCompositionStateError(
            f"{validation_error}: result has no template results"
        )
    if "render_valid" not in checkpoint_templates:
        raise AssetCompositionStateError(
            f"{validation_error}: checkpoint omits the required render_valid gate"
        )
    _require_exact_claim(
        tuple(result.request.inputs),
        (input_asset.path,),
        label="Validation result inputs",
    )
    if not source_matches:
        raise AssetCompositionStateError(
            f"{validation_error}: source_before omits the accepted input identity"
        )
    if before != after:
        raise AssetCompositionStateError(
            f"{validation_error}: source identities changed; "
            f"before={before!r}, after={after!r}"
        )
    for key, expected in (
        ("schema_version", VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION),
        ("workflow_identity_digest", checkpoint.workflow_identity.identity_digest),
        ("plan_digest", checkpoint.plan_digest),
        ("source_unchanged", True),
    ):
        _require_exact_claim(
            native_evidence.get(key),
            expected,
            label=f"Validation native evidence {key}",
        )
    if not all_completed:
        raise AssetCompositionStateError(
            f"{validation_error}: checkpoint contains an incomplete accepted record"
        )
    if checkpoint.cancellation_requested:
        raise AssetCompositionStateError(
            f"{validation_error}: checkpoint retains a cancellation request"
        )
    _require_exact_claim(
        result_templates,
        checkpoint_templates,
        label="Validation result template order",
    )
    _require_exact_claim(
        result.template_results,
        accepted_template_results,
        label="Validation result accepted template records",
    )
    _require_exact_claim(
        result.verdict,
        aggregate_validation_verdict(accepted_template_results),
        label="Validation aggregate verdict",
    )
    _require_exact_claim(
        frozenset(native_template_names),
        frozenset(checkpoint_templates),
        label="Validation native template names",
    )
    if not native_template_statuses_match:
        raise AssetCompositionStateError(
            f"{validation_error}: native template statuses differ from accepted records"
        )
    summary_claims: tuple[tuple[str, object], ...] = (
        ("schema_version", VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION),
        ("status", "completed"),
        ("verdict", result.verdict),
        ("source_asset_unchanged", True),
        ("completed_templates", list(checkpoint_templates)),
        ("remaining_templates", []),
    )
    for key, summary_expected in summary_claims:
        _require_exact_claim(
            summary.get(key),
            summary_expected,
            label=f"Validation final summary {key}",
        )
    _require_exact_claim(
        cross_stage.run_id,
        run.run_id,
        label="Validation cross-stage run ID",
    )
    _require_exact_claim(
        cross_stage.validation_input,
        input_asset,
        label="Validation cross-stage input",
    )
    _require_exact_claim(
        cross_stage.accepted_handoffs,
        expected_handoffs,
        label="Validation cross-stage accepted handoffs",
    )
    if not cross_claims_valid:
        raise AssetCompositionStateError(
            f"{validation_error}: cross-stage claim evidence violates its claim scope"
        )


def _cad_process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_cad_process_group_exit(
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        process.poll()
        if not _cad_process_group_exists(process.pid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_CAD_PROCESS_GROUP_POLL_SECONDS, remaining))


def _signal_cad_process_group(
    process_group_id: int,
    signal_number: signal.Signals,
) -> bool:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise AssetCompositionStateError(
            f"Could not signal CAD process group {process_group_id} with "
            f"{signal_number.name}"
        ) from exc
    return True


def _close_cad_process_pipes(process: subprocess.Popen[str]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def _terminate_cad_process_group(process: subprocess.Popen[str]) -> None:
    """Bound cleanup of a CAD CLI and descendants in its isolated group."""

    process_group_id = process.pid
    try:
        term_sent = _signal_cad_process_group(
            process_group_id,
            signal.SIGTERM,
        )
        if term_sent and not _wait_for_cad_process_group_exit(
            process,
            timeout=_CAD_PROCESS_TERMINATION_GRACE_SECONDS,
        ):
            kill_sent = _signal_cad_process_group(
                process_group_id,
                signal.SIGKILL,
            )
            if kill_sent:
                _wait_for_cad_process_group_exit(
                    process,
                    timeout=_CAD_PROCESS_KILL_GRACE_SECONDS,
                )
        process.poll()
        if process.returncode is None:
            raise AssetCompositionStateError(
                f"CAD process group {process_group_id} did not exit after SIGKILL"
            )
    finally:
        _close_cad_process_pipes(process)


def _run_geometry_authoring_provider(
    *,
    request_path: Path,
    workspace: Path,
    artifact_dir: Path,
    source_manifest_path: Path,
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one typed authoring request without a shell or provider-native source."""

    if not command or any(not str(item).strip() for item in command):
        raise AssetCompositionStateError("CAD agent command must not be empty")
    if os.name != "posix" or not hasattr(os, "killpg"):
        raise AssetCompositionStateError(
            "CAD product jobs require POSIX process-group supervision"
        )

    argv = [
        *[str(item) for item in command],
        "author-source",
        str(request_path),
        "--workspace",
        str(workspace),
        "--artifact-dir",
        str(artifact_dir),
        "--out",
        str(source_manifest_path),
    ]
    watched_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        watched_signals.append(signal.SIGHUP)
    previous_handlers: dict[signal.Signals, Any] = {}
    process: subprocess.Popen[str] | None = None
    pending_signal: int | None = None
    cleanup_in_progress = False

    def interrupt(signum: int, _frame: Any) -> None:
        nonlocal pending_signal
        if pending_signal is None:
            pending_signal = signum
        if process is not None and not cleanup_in_progress:
            raise _CadProductJobInterrupted(signum)

    try:
        for watched_signal in watched_signals:
            previous_handler = signal.getsignal(watched_signal)
            signal.signal(watched_signal, interrupt)
            previous_handlers[watched_signal] = previous_handler
    except (OSError, RuntimeError, ValueError) as exc:
        for watched_signal, previous_handler in previous_handlers.items():
            signal.signal(watched_signal, previous_handler)
        raise AssetCompositionStateError(
            "CAD product jobs require main-thread POSIX signal supervision"
        ) from exc

    try:
        if pending_signal is not None:
            raise _CadProductJobInterrupted(pending_signal)
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            start_new_session=True,
        )
        if pending_signal is not None:
            raise _CadProductJobInterrupted(pending_signal)
        stdout, stderr = process.communicate()
        cleanup_in_progress = True
        _terminate_cad_process_group(process)
        cleanup_in_progress = False
        if pending_signal is not None:
            raise _CadProductJobInterrupted(pending_signal)
        returncode = process.returncode
        if returncode is None:
            raise AssetCompositionStateError("CAD product job did not exit")
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)
    except BaseException as exc:
        cleanup_in_progress = True
        if process is not None:
            _terminate_cad_process_group(process)
        cleanup_in_progress = False
        if pending_signal is not None and not isinstance(
            exc,
            _CadProductJobInterrupted,
        ):
            raise _CadProductJobInterrupted(pending_signal) from exc
        raise
    finally:
        for watched_signal, previous_handler in previous_handlers.items():
            signal.signal(watched_signal, previous_handler)


def _resolved_geometry_authoring_command(
    command: Sequence[str] | None,
) -> tuple[str, ...]:
    """Resolve administrator-controlled provider argv without shell evaluation."""

    if command is not None:
        resolved = tuple(str(item) for item in command)
    else:
        raw = os.environ.get(_GEOMETRY_AUTHORING_COMMAND_ENV, "").strip()
        try:
            resolved = (
                tuple(shlex.split(raw)) if raw else ("geometry-authoring-provider",)
            )
        except ValueError as exc:
            raise AssetCompositionStateError(
                f"{_GEOMETRY_AUTHORING_COMMAND_ENV} contains invalid shell-style "
                "quoting"
            ) from exc
    if not resolved or any(not item.strip() for item in resolved):
        raise AssetCompositionStateError(
            "Geometry authoring provider command must not be empty"
        )
    return resolved


def _canonical_authoring_parameter_values(
    values: Mapping[str, object],
) -> dict[str, str | int | float | bool]:
    """Retain bounded scalar values admitted by the frozen public request."""

    canonical: dict[str, str | int | float | bool] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise AssetCompositionStateError(
                "Geometry authoring parameter names must be non-empty strings"
            )
        if not isinstance(value, str | int | float | bool):
            raise AssetCompositionStateError(
                f"Geometry authoring parameter {name!r} is not a scalar"
            )
        canonical[name] = value
    return canonical


def _source_bundle_parameter_values(
    bundle: GeometrySourceBundle,
) -> dict[str, str | int | float | bool]:
    return {parameter.name: parameter.value for parameter in bundle.parameters}


def _require_new_variant_source_revision(
    *,
    baseline_revision: str,
    variant_revision: str,
) -> None:
    if variant_revision == baseline_revision:
        raise ValueError("variant reused the immutable baseline source revision")


def _parameter_values_match(
    observed: Mapping[str, object],
    requested: Mapping[str, object],
) -> bool:
    for name, expected in requested.items():
        if name not in observed:
            return False
        actual = observed[name]
        if isinstance(expected, bool) or isinstance(actual, bool):
            if type(actual) is not type(expected) or actual != expected:
                return False
        elif actual != expected:
            return False
    return True


def _source_bundle_with_absolute_artifacts(
    manifest_path: str | Path,
    bundle: GeometrySourceBundle,
) -> GeometrySourceBundle:
    """Rebase provider outputs for a subsequent digest-bound revision call."""

    representations = tuple(
        representation.model_copy(
            update={
                "artifact": representation.artifact.model_copy(
                    update={
                        "path": str(
                            source_representation_path(manifest_path, representation)
                        )
                    }
                )
            }
        )
        for representation in bundle.representations
    )
    provenance = bundle.provenance.model_copy(
        update={
            "input_artifacts": tuple(
                artifact.model_copy(
                    update={"path": str(source_artifact_path(manifest_path, artifact))}
                )
                for artifact in bundle.provenance.input_artifacts
            )
        }
    )
    return bundle.model_copy(
        update={"representations": representations, "provenance": provenance}
    )


def _requested_source_formats_satisfied(
    bundle: GeometrySourceBundle,
    requested_formats: Sequence[str],
) -> bool:
    observed = {
        representation.format
        for representation in bundle.representations
        if representation.role not in {"native_source", "supporting_asset"}
    }
    for requested in requested_formats:
        if requested == "usd":
            if not observed.intersection({"usd", "usda", "usdc"}):
                return False
        elif requested not in observed:
            return False
    return True


def _source_representation_binding(
    representation: GeometryRepresentationBinding,
    *,
    path: Path,
    root: Path,
    label: str,
) -> ArtifactBinding:
    binding = _binding(path, label=label, required_root=root)
    if (
        binding.sha256 != representation.artifact.sha256
        or binding.size_bytes != representation.artifact.size_bytes
    ):
        raise AssetCompositionStateError(
            f"{label} differs from its geometry source identity"
        )
    return binding


def _bind_source_bundle(
    manifest_path: Path,
    bundle: GeometrySourceBundle,
    *,
    root: Path,
    key_prefix: str = "source_representation",
) -> tuple[dict[str, ArtifactBinding], GeometryRepresentationBinding, ArtifactBinding]:
    """Bind every source representation and select the single Geometry USD."""

    bindings: dict[str, ArtifactBinding] = {}
    for index, representation in enumerate(bundle.representations):
        source = source_representation_path(manifest_path, representation)
        key = f"{key_prefix}_{index:03d}_{representation.representation_id}"
        bindings[key] = _source_representation_binding(
            representation,
            path=source,
            root=root,
            label=f"geometry source representation {representation.representation_id}",
        )
    selected, selected_path = select_source_usd(manifest_path, bundle)
    selected_binding = _source_representation_binding(
        selected,
        path=selected_path,
        root=root,
        label="selected geometry source USD",
    )
    return bindings, selected, selected_binding


def _cad_refinement_seed(state: StageState) -> GeometrySourceBundle | None:
    """Recover only the canonical source bundle admitted by the latest review."""

    if not state.superseded_attempts:
        return None
    reviewed_evidence = state.superseded_attempts[-1].evidence
    stage_results: list[AssetCadModelingStageResult] = []
    for binding in reviewed_evidence:
        if Path(binding.path).suffix.lower() != ".json":
            continue
        payload = _json_object_from_binding(binding, label="CAD refinement evidence")
        if (
            payload.get("schema_version")
            != ASSET_CAD_MODELING_STAGE_RESULT_SCHEMA_VERSION
        ):
            continue
        try:
            stage_results.append(AssetCadModelingStageResult.model_validate(payload))
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Invalid reviewed CAD modeling result: {exc}"
            ) from exc
    if not stage_results:
        return None
    if len(stage_results) != 1:
        raise AssetCompositionStateError(
            "CAD refinement evidence must bind exactly one CAD modeling result"
        )
    source_manifest = stage_results[0].source_manifest
    if source_manifest is None:
        return None
    if source_manifest not in set(reviewed_evidence):
        raise AssetCompositionStateError(
            "CAD refinement source manifest must be explicitly admitted by review"
        )
    _json_object_from_binding(
        source_manifest,
        label="reviewed geometry source manifest",
    )
    try:
        bundle = load_external_source_bundle(source_manifest.path)
        return _source_bundle_with_absolute_artifacts(source_manifest.path, bundle)
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Reviewed geometry source bundle is invalid: {exc}"
        ) from exc


def execute_cad_modeling_stage(
    path: str | Path,
    *,
    actor: str = "asset-cad-modeling-executor",
    geometry_authoring_command: Sequence[str] | None = None,
) -> AssetCadModelingStageResult:
    """Exclusively run semantic CAD composition for the active stage attempt."""

    state_path = _resolved(path)
    run = _load_verified_transition_run(state_path)
    state = run.stages.get("cad_modeling")
    if run.current_stage != "cad_modeling" or state is None:
        raise AssetCompositionStateError("CAD modeling is not the active stage")
    if state.status != "running":
        raise AssetCompositionStateError(
            f"Cannot execute CAD modeling from {state.status}; expected running"
        )

    resolved_command = _resolved_geometry_authoring_command(geometry_authoring_command)
    with _cad_attempt_execution_lease(
        state_path,
        run_id=run.run_id,
        stage_attempt=state.attempt_count,
    ) as execution_lease:
        return _execute_cad_modeling_stage_owned(
            state_path,
            actor=actor,
            geometry_authoring_command=resolved_command,
            execution_lease=execution_lease,
        )


def _execute_cad_modeling_stage_owned(
    path: str | Path,
    *,
    actor: str,
    geometry_authoring_command: Sequence[str],
    execution_lease: _CadAttemptExecutionLease,
) -> AssetCadModelingStageResult:
    """Run one provider-neutral authoring attempt while its lease is held."""

    state_path = _resolved(path)
    run = _load_verified_transition_run(state_path)
    request = load_verified_asset_request(state_path, run=run)
    if run.current_stage != "cad_modeling" or "cad_modeling" not in run.stages:
        raise AssetCompositionStateError("CAD modeling is not the active stage")
    state = run.stages["cad_modeling"]
    if state.status != "running":
        raise AssetCompositionStateError(
            f"Cannot execute CAD modeling from {state.status}; expected running"
        )
    execution_lease.require_owner(
        run_id=run.run_id,
        stage_attempt=state.attempt_count,
    )
    if run.coordinator.mode != "single_reasoning_loop":
        raise AssetCompositionStateError(
            "Composed CAD modeling requires single_reasoning_loop mode"
        )
    if run.coordinator.next_action != "execute_stage":
        raise AssetCompositionStateError(
            "CAD modeling requires coordinator next_action=execute_stage"
        )
    if request.cad_modeling is None or request.source_mode != "cad_modeling":
        raise AssetCompositionStateError("Frozen request lacks CAD modeling policy")
    if state.input_asset is None:
        raise AssetCompositionStateError(
            "Running CAD modeling stage lacks input intent"
        )
    if not run.coordinator.plan_revisions:
        raise AssetCompositionStateError("CAD modeling requires a sealed plan")
    plan_binding = run.coordinator.plan_revisions[-1]
    plan = _load_coordinator_plan(plan_binding)
    if plan.stage != "cad_modeling" or plan.stage_attempt != state.attempt_count:
        raise AssetCompositionStateError(
            "Latest coordinator plan does not own this CAD modeling attempt"
        )

    attempt_root = stage_directory(
        state_path,
        "cad_modeling",
        attempt=state.attempt_count,
    ).resolve()
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise AssetCompositionStateError(
            f"CAD modeling attempt directory is unavailable: {attempt_root}"
        )
    domain_dir = attempt_root / "domain-run"
    workspace = domain_dir / "workspace"
    artifact_dir = domain_dir / "artifact-store"
    staged_images_dir = workspace / "inputs"
    staged_images_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    media_types = {
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    references: list[GeometryAuthoringReference] = []
    for index, binding in enumerate(request.source_image_bindings):
        suffix = Path(binding.path).suffix.lower()
        if suffix not in _CAD_SOURCE_IMAGE_SUFFIXES:
            raise AssetCompositionStateError(
                f"CAD source image {index + 1} has unsupported format {suffix!r}"
            )
        destination = staged_images_dir / f"source-image-{index:03d}{suffix}"
        payload = _read_stable_regular_bytes(
            Path(binding.path),
            label=f"CAD source image {index + 1}",
        )
        if (
            len(payload) != binding.size_bytes
            or hashlib.sha256(payload).hexdigest() != binding.sha256
        ):
            raise AssetCompositionStateError(
                f"CAD source image {index + 1} identity changed"
            )
        atomic_write_bytes(destination, payload, within=attempt_root)
        references.append(
            GeometryAuthoringReference(
                reference_id=f"source-image-{index:03d}",
                kind="image",
                media_type=media_types[suffix],
                artifact=GeometryArtifactBinding(
                    path=str(destination.resolve(strict=True)),
                    sha256=binding.sha256,
                    size_bytes=binding.size_bytes,
                ),
            )
        )

    policy = request.cad_modeling
    if isinstance(policy, LegacyAssetCadModelingRequest):
        raise AssetCompositionStateError(
            "asset request v4 uses the retired direct-authoring policy and must "
            "be resumed with the pre-provider runner"
        )
    base_parameters = _canonical_authoring_parameter_values(policy.parameter_values)
    parameter_values = tuple(
        GeometryAuthoringParameterValue(name=name, value=value)
        for name, value in sorted(base_parameters.items())
    )

    def request_id(label: str) -> str:
        identity = hashlib.sha256(
            f"{run.run_id}:{state.attempt_count}:{label}".encode()
        ).hexdigest()[:32]
        return f"asset-authoring-{identity}"

    refinement_seed = _cad_refinement_seed(state)
    authoring_request: GeometryAuthoringRequest | GeometryAuthoringRevisionRequest
    if refinement_seed is None:
        authoring_request = GeometryAuthoringRequest(
            request_id=request_id("baseline"),
            prompt=request.prompt,
            references=tuple(references),
            target_profile=policy.target_profile,
            requested_formats=tuple(policy.required_outputs),
            parameters=parameter_values,
        )
    else:
        if refinement_seed.producer.provider_id != policy.provider_id:
            raise AssetCompositionStateError(
                "Reviewed source bundle belongs to a different authoring provider"
            )
        authoring_request = GeometryAuthoringRevisionRequest(
            request_id=request_id("baseline-revision"),
            source_bundle=refinement_seed,
            instructions=request.prompt,
            references=tuple(references),
            parameter_overrides=parameter_values,
            requested_formats=tuple(policy.required_outputs),
        )

    authoring_request_path = domain_dir / "geometry_authoring_request.json"
    source_manifest_path = domain_dir / "geometry.source.json"
    atomic_write_json(
        authoring_request_path,
        authoring_request.model_dump(mode="json"),
    )
    stdout_path = domain_dir / "authoring_stdout.log"
    stderr_path = domain_dir / "authoring_stderr.log"
    command_error: str | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = _run_geometry_authoring_provider(
            request_path=authoring_request_path,
            workspace=workspace,
            artifact_dir=artifact_dir,
            source_manifest_path=source_manifest_path,
            command=geometry_authoring_command,
        )
        atomic_write_text(stdout_path, completed.stdout)
        atomic_write_text(stderr_path, completed.stderr)
        if completed.returncode != 0:
            command_error = (
                f"Geometry authoring provider exited with code {completed.returncode}"
            )
    except OSError as exc:
        atomic_write_text(stdout_path, "")
        atomic_write_text(stderr_path, f"{type(exc).__name__}: {exc}\n")
        command_error = f"Geometry authoring provider could not start: {exc}"

    root = _run_root(state_path)
    artifacts: dict[str, ArtifactBinding] = {
        "authoring_request": _binding(
            authoring_request_path,
            label="geometry authoring request",
            required_root=root,
        ),
        "authoring_stdout": _binding(
            stdout_path,
            label="geometry authoring stdout",
            required_root=root,
        ),
        "authoring_stderr": _binding(
            stderr_path,
            label="geometry authoring stderr",
            required_root=root,
        ),
    }
    errors: list[str] = [command_error] if command_error else []
    source_manifest_binding: ArtifactBinding | None = None
    family_binding: ArtifactBinding | None = None
    output: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = []
    bundle: GeometrySourceBundle | None = None
    selected_representation: GeometryRepresentationBinding | None = None
    observed_parameter_values: dict[str, str | int | float | bool] = {}

    if completed is not None and completed.returncode == 0:
        if not source_manifest_path.is_file():
            errors.append(
                "Geometry authoring provider did not publish a source manifest"
            )
        else:
            try:
                bundle = load_external_source_bundle(source_manifest_path)
                if bundle.producer.provider_id != policy.provider_id:
                    raise ValueError(
                        "source bundle producer differs from the selected provider"
                    )
                if (
                    bundle.provenance.request_digest
                    != geometry_authoring_request_digest(authoring_request)
                ):
                    raise ValueError(
                        "source bundle provenance differs from the authoring request"
                    )
                expected_parent = (
                    refinement_seed.bundle_id if refinement_seed is not None else None
                )
                if bundle.provenance.parent_bundle_id != expected_parent:
                    raise ValueError("source bundle revision lineage is invalid")
                if (
                    refinement_seed is not None
                    and bundle.source_revision == refinement_seed.source_revision
                ):
                    raise ValueError(
                        "source bundle refinement reused the immutable source revision"
                    )
                if not _requested_source_formats_satisfied(
                    bundle,
                    policy.required_outputs,
                ):
                    raise ValueError(
                        "source bundle omitted one or more requested output formats"
                    )
                observed_parameter_values = _source_bundle_parameter_values(bundle)
                if refinement_seed is None:
                    if not _parameter_values_match(
                        observed_parameter_values,
                        base_parameters,
                    ):
                        raise ValueError(
                            "source bundle parameters differ from the frozen request"
                        )
                else:
                    expected_parameters = resolve_semantic_parameter_overrides(
                        refinement_seed.parameters,
                        parameter_values,
                    )
                    validate_returned_parameter_state(
                        bundle.parameters,
                        expected_parameters,
                    )
                source_manifest_binding = _binding(
                    source_manifest_path,
                    label="geometry source manifest",
                    required_root=root,
                )
                artifacts["source_manifest"] = source_manifest_binding
                source_artifacts, selected_representation, output = _bind_source_bundle(
                    source_manifest_path,
                    bundle,
                    root=root,
                )
                artifacts.update(source_artifacts)
                artifacts["generated_usd"] = output
                validate_external_source_manifest(
                    source_manifest_path,
                    output_path=output.path,
                    output_sha256=output.sha256,
                    output_size_bytes=output.size_bytes,
                )
                output_dependencies = _dependency_bindings(
                    Path(output.path),
                    label="generated authoring USD",
                    required_root=root,
                )
            except (
                OSError,
                ValueError,
                ValidationError,
                AssetCompositionStateError,
            ) as exc:
                errors.append(f"Geometry source bundle is invalid: {exc}")

    family_rows: list[dict[str, object]] = []
    if (
        bundle is not None
        and selected_representation is not None
        and policy.parameter_variants
    ):
        baseline_for_revision = _source_bundle_with_absolute_artifacts(
            source_manifest_path,
            bundle,
        )
        variants_root = domain_dir / "parameter-variants"
        variants_root.mkdir(parents=True, exist_ok=True)
        baseline_parameter_values = _source_bundle_parameter_values(
            baseline_for_revision
        )
        for index, variant in enumerate(policy.parameter_variants):
            variant_parameters = _canonical_authoring_parameter_values(
                {
                    **baseline_parameter_values,
                    **variant.parameter_values,
                }
            )
            variant_dir = variants_root / f"{index:03d}-{variant.id}"
            variant_workspace = variant_dir / "workspace"
            variant_artifacts = variant_dir / "artifact-store"
            variant_workspace.mkdir(parents=True, exist_ok=True)
            variant_artifacts.mkdir(parents=True, exist_ok=True)
            variant_request = GeometryAuthoringRevisionRequest(
                request_id=request_id(f"variant-{variant.id}"),
                source_bundle=baseline_for_revision,
                parameter_overrides=tuple(
                    GeometryAuthoringParameterValue(name=name, value=value)
                    for name, value in sorted(variant_parameters.items())
                ),
                requested_formats=tuple(policy.required_outputs),
            )
            variant_request_path = variant_dir / "geometry_authoring_request.json"
            variant_manifest_path = variant_dir / "geometry.source.json"
            atomic_write_json(
                variant_request_path,
                variant_request.model_dump(mode="json"),
            )
            artifacts[f"variant_request_{variant.id}"] = _binding(
                variant_request_path,
                label=f"geometry variant {variant.id} request",
                required_root=root,
            )
            try:
                variant_completed = _run_geometry_authoring_provider(
                    request_path=variant_request_path,
                    workspace=variant_workspace,
                    artifact_dir=variant_artifacts,
                    source_manifest_path=variant_manifest_path,
                    command=geometry_authoring_command,
                )
                atomic_write_text(
                    variant_dir / "authoring_stdout.log",
                    variant_completed.stdout,
                )
                atomic_write_text(
                    variant_dir / "authoring_stderr.log",
                    variant_completed.stderr,
                )
                if variant_completed.returncode != 0:
                    raise AssetCompositionStateError(
                        f"provider exited with code {variant_completed.returncode}"
                    )
                variant_bundle = load_external_source_bundle(variant_manifest_path)
                if variant_bundle.producer.provider_id != policy.provider_id:
                    raise ValueError("variant producer differs from selected provider")
                if (
                    variant_bundle.provenance.parent_bundle_id
                    != baseline_for_revision.bundle_id
                ):
                    raise ValueError("variant lineage does not reference the baseline")
                _require_new_variant_source_revision(
                    baseline_revision=baseline_for_revision.source_revision,
                    variant_revision=variant_bundle.source_revision,
                )
                if (
                    variant_bundle.provenance.request_digest
                    != geometry_authoring_request_digest(variant_request)
                ):
                    raise ValueError("variant provenance differs from its request")
                if not _requested_source_formats_satisfied(
                    variant_bundle,
                    policy.required_outputs,
                ):
                    raise ValueError("variant omitted a requested output format")
                actual_variant_parameters = _source_bundle_parameter_values(
                    variant_bundle
                )
                expected_variant_parameters = resolve_semantic_parameter_overrides(
                    baseline_for_revision.parameters,
                    variant_request.parameter_overrides,
                )
                validate_returned_parameter_state(
                    variant_bundle.parameters,
                    expected_variant_parameters,
                )
                variant_manifest_binding = _binding(
                    variant_manifest_path,
                    label=f"geometry variant {variant.id} source manifest",
                    required_root=root,
                )
                variant_source_artifacts, variant_selected, variant_output = (
                    _bind_source_bundle(
                        variant_manifest_path,
                        variant_bundle,
                        root=root,
                        key_prefix=f"variant_{variant.id}_representation",
                    )
                )
                artifacts[f"variant_manifest_{variant.id}"] = variant_manifest_binding
                artifacts[f"variant_usd_{variant.id}"] = variant_output
                artifacts.update(variant_source_artifacts)
                family_rows.append(
                    {
                        "id": variant.id,
                        "parameter_values": actual_variant_parameters,
                        "source_bundle_id": variant_bundle.bundle_id,
                        "source_revision": variant_bundle.source_revision,
                        "source_provider_id": variant_bundle.producer.provider_id,
                        "source_manifest": variant_manifest_binding.model_dump(
                            mode="json"
                        ),
                        "selected_representation_id": (
                            variant_selected.representation_id
                        ),
                        "output_asset": variant_output.model_dump(mode="json"),
                    }
                )
            except (
                OSError,
                ValueError,
                ValidationError,
                AssetCompositionStateError,
            ) as exc:
                errors.append(
                    f"Geometry parameter variant {variant.id!r} failed: {exc}"
                )

        family_path = domain_dir / "geometry_parameter_family.json"
        atomic_write_json(
            family_path,
            {
                "schema_version": (
                    "content-agent-workflows.geometry-parameter-family.v1"
                ),
                "provider_id": policy.provider_id,
                "parent_bundle_id": bundle.bundle_id,
                "requested_rows": len(policy.parameter_variants),
                "succeeded": len(family_rows) == len(policy.parameter_variants),
                "rows": family_rows,
            },
        )
        family_binding = _binding(
            family_path,
            label="geometry parameter family",
            required_root=root,
        )
        artifacts["parameter_family"] = family_binding

    current = _load_verified_transition_run(state_path)
    current_state = current.stages.get("cad_modeling")
    if (
        current.revision != run.revision
        or current.current_stage != "cad_modeling"
        or current_state is None
        or current_state.status != "running"
        or current_state.attempt_count != state.attempt_count
        or not current.coordinator.plan_revisions
        or current.coordinator.plan_revisions[-1] != plan_binding
    ):
        raise AssetCompositionStateError(
            "Composed state changed while the CAD modeling executor was running"
        )

    success = (
        not errors
        and bundle is not None
        and selected_representation is not None
        and source_manifest_binding is not None
        and output is not None
    )
    result_path = attempt_root / "cad_modeling_stage_result.json"
    result = AssetCadModelingStageResult(
        run_id=run.run_id,
        run_revision=run.revision,
        stage_attempt=state.attempt_count,
        request=run.request,
        plan=plan_binding,
        input_asset=state.input_asset,
        input_dependencies=state.input_dependencies,
        result_path=str(result_path),
        authoring_request=artifacts["authoring_request"],
        source_manifest=source_manifest_binding,
        parameter_family=family_binding,
        output_asset=output,
        output_dependencies=output_dependencies,
        artifacts=artifacts,
        source_bundle_id=bundle.bundle_id if bundle is not None else None,
        source_revision=bundle.source_revision if bundle is not None else None,
        source_provider_id=(
            bundle.producer.provider_id if bundle is not None else None
        ),
        source_representation_id=(
            selected_representation.representation_id
            if selected_representation is not None
            else None
        ),
        parameter_values=observed_parameter_values,
        success=success,
        error="; ".join(dict.fromkeys(errors)) if errors else None,
        timestamp=_timestamp(),
        actor=actor,
    )
    execution_lease.require_owner(
        run_id=run.run_id,
        stage_attempt=state.attempt_count,
    )
    atomic_write_json(result_path, result)
    return result


def _cad_modeling_stage_result_from_evidence(
    evidence: Sequence[ArtifactBinding],
) -> tuple[ArtifactBinding, AssetCadModelingStageResult]:
    matches: list[tuple[ArtifactBinding, AssetCadModelingStageResult]] = []
    for binding in evidence:
        if Path(binding.path).suffix.lower() != ".json":
            continue
        payload = _json_object_from_binding(
            binding,
            label=f"CAD modeling evidence {Path(binding.path).name}",
        )
        if (
            payload.get("schema_version")
            != ASSET_CAD_MODELING_STAGE_RESULT_SCHEMA_VERSION
        ):
            continue
        try:
            matches.append(
                (binding, AssetCadModelingStageResult.model_validate(payload))
            )
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Composed CAD modeling stage result is invalid: {exc}"
            ) from exc
    if len(matches) != 1:
        raise AssetCompositionStateError(
            "CAD modeling acceptance requires exactly one typed stage result"
        )
    return matches[0]


def execute_geometry_stage(
    path: str | Path,
    *,
    actor: str = "asset-geometry-executor",
) -> AssetGeometryStageResult:
    """Exclusively run Geometry for the active composed stage attempt."""

    state_path = _resolved(path)
    run = _load_verified_transition_run(state_path)
    state = run.stages.get("geometry")
    if run.current_stage != "geometry" or state is None:
        raise AssetCompositionStateError("Geometry is not the active composed stage")
    if state.status != "running":
        raise AssetCompositionStateError(
            f"Cannot execute Geometry from {state.status}; expected running"
        )
    with _geometry_attempt_execution_lease(
        state_path,
        run_id=run.run_id,
        stage_attempt=state.attempt_count,
    ) as execution_lease:
        return _execute_geometry_stage_owned(
            state_path,
            actor=actor,
            execution_lease=execution_lease,
        )


def _execute_geometry_stage_owned(
    path: str | Path,
    *,
    actor: str,
    execution_lease: _CadAttemptExecutionLease,
) -> AssetGeometryStageResult:
    """Run Geometry while its durable stage-attempt lease is held."""

    state_path = _resolved(path)
    run = _load_verified_transition_run(state_path)
    request = load_verified_asset_request(state_path, run=run)
    if run.current_stage != "geometry" or "geometry" not in run.stages:
        raise AssetCompositionStateError("Geometry is not the active composed stage")
    state = run.stages["geometry"]
    if state.status != "running":
        raise AssetCompositionStateError(
            f"Cannot execute Geometry from {state.status}; expected running"
        )
    execution_lease.require_owner(
        run_id=run.run_id,
        stage_attempt=state.attempt_count,
        stage="geometry",
    )
    if run.coordinator.mode != "single_reasoning_loop":
        raise AssetCompositionStateError(
            "Composed Geometry execution requires single_reasoning_loop mode"
        )
    if run.coordinator.next_action != "execute_stage":
        raise AssetCompositionStateError(
            "Geometry execution requires coordinator next_action=execute_stage"
        )
    if request.geometry is None:
        raise AssetCompositionStateError("Frozen request lacks Geometry policy")
    if state.input_asset is None:
        raise AssetCompositionStateError("Running Geometry stage lacks input asset")
    if not run.coordinator.plan_revisions:
        raise AssetCompositionStateError("Geometry execution requires a sealed plan")
    plan_binding = run.coordinator.plan_revisions[-1]
    plan = _load_coordinator_plan(plan_binding)
    if plan.stage != "geometry" or plan.stage_attempt != state.attempt_count:
        raise AssetCompositionStateError(
            "Latest coordinator plan does not own this Geometry attempt"
        )

    attempt_root = stage_directory(
        state_path,
        "geometry",
        attempt=state.attempt_count,
    ).resolve()
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise AssetCompositionStateError(
            f"Geometry attempt directory is unavailable: {attempt_root}"
        )
    domain_dir = attempt_root / "domain-run"
    if domain_dir.exists() and (domain_dir.is_symlink() or not domain_dir.is_dir()):
        raise AssetCompositionStateError(
            f"Geometry domain directory must be a regular directory: {domain_dir}"
        )
    domain_dir.mkdir(exist_ok=True)
    output_path = domain_dir / "geometry.usdc"

    from content_agent_workflows.geometry.workflow import (
        GeometryWorkflowInput,
        run_geometry_workflow,
    )

    geometry = request.geometry
    segmentation_run_dir = (
        Path(geometry.segmentation_run.run_dir)
        if geometry.segmentation_run is not None
        else None
    )
    source_path: Path | None = Path(state.input_asset.path)
    source_manifest_path: Path | None = None
    source_representation_id: str | None = None
    generated_usd_path: Path | None = None
    expected_source_manifest_sha256: str | None = None
    if "cad_modeling" in run.stage_order:
        cad_state = run.stages["cad_modeling"]
        if (
            cad_state.status != "completed"
            or cad_state.output_asset != state.input_asset
        ):
            raise AssetCompositionStateError(
                "Geometry input is not the accepted CAD modeling output"
            )
        _binding_result, cad_result = _cad_modeling_stage_result_from_evidence(
            cad_state.evidence
        )
        if (
            not cad_result.success
            or cad_result.output_asset != state.input_asset
            or cad_result.source_manifest is None
            or cad_result.source_representation_id is None
        ):
            raise AssetCompositionStateError(
                "Accepted CAD modeling evidence is not admissible for Geometry"
            )
        # Geometry consumes the selected immutable representation plus the
        # provider-neutral source specification. Provider-native source remains
        # inert and is never imported or executed by this workflow.
        source_path = Path(state.input_asset.path)
        source_manifest_path = Path(cad_result.source_manifest.path)
        source_representation_id = cad_result.source_representation_id
        expected_source_manifest_sha256 = cad_result.source_manifest.sha256
        generated_usd_path = None
    workflow_result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source_path,
            source_manifest_path=source_manifest_path,
            expected_source_sha256=state.input_asset.sha256,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            source_representation_id=source_representation_id,
            prompt=request.prompt,
            image_path=(
                Path((request.source_images or request.reference_images)[0])
                if request.source_images or request.reference_images
                else None
            ),
            generated_usd_path=generated_usd_path,
            output_dir=domain_dir,
            output_usd_path=output_path,
            target_profile=geometry.target_profile,
            target_runtime=geometry.target_runtime,
            render_evidence=geometry.render_evidence,
            render_backend="ovrtx",
            optimization_policy=geometry.optimization_policy,
            optimizer_backend=geometry.optimizer_backend,
            canonicalize_stage_metrics=True,
            segmentation_run_dir=segmentation_run_dir,
            segmentation_required=geometry.segmentation_required,
            segmentation_required_parts=list(geometry.segmentation_required_parts),
            repair_mode=geometry.repair_mode,
            repair_profile=geometry.repair_profile,
            runtime_validation_mode=geometry.runtime_validation_mode,
            simready_mode=geometry.simready_mode,
            fail_on_validation_error=False,
        )
    )
    workflow_result_path = domain_dir / "geometry_workflow_result.json"
    atomic_write_json(workflow_result_path, workflow_result)

    current = _load_verified_transition_run(state_path)
    current_state = current.stages.get("geometry")
    if (
        current.revision != run.revision
        or current.current_stage != "geometry"
        or current_state is None
        or current_state.status != "running"
        or current_state.attempt_count != state.attempt_count
        or not current.coordinator.plan_revisions
        or current.coordinator.plan_revisions[-1] != plan_binding
    ):
        raise AssetCompositionStateError(
            "Composed state changed while the Geometry executor was running"
        )

    root = _run_root(state_path)
    workflow_binding = _binding(
        workflow_result_path,
        label="Geometry workflow result",
        required_root=root,
    )
    artifact_paths = {
        "handoff_manifest": workflow_result.handoff_manifest_path,
        "validation_evidence": workflow_result.validation_evidence_path,
        "evidence_bundle": workflow_result.evidence_bundle_path,
        "optimization_metadata": workflow_result.optimization_metadata_path,
        "usd_validation": workflow_result.usd_validation_report_path,
        "render_report": workflow_result.render_report_path,
        "inspection": workflow_result.inspection_path,
        "asset_audit": workflow_result.asset_audit_path,
        "asset_audit_summary": workflow_result.asset_audit_summary_path,
        "repair_certificate": workflow_result.repair_certificate_path,
        "segmentation_validation": workflow_result.segmentation_validation_path,
    }
    contract_errors: list[str] = []
    artifacts: dict[str, ArtifactBinding] = {}
    for label, raw_path in artifact_paths.items():
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if not candidate.is_file():
            continue
        artifacts[label] = _binding(
            candidate,
            label=f"Geometry {label.replace('_', ' ')}",
            required_root=root,
        )
    render_binding = artifacts.get("render_report")
    render_report: GeometryRenderEvidence | None = None
    if render_binding is not None:
        try:
            render_report = _load_geometry_render_evidence(render_binding)
            artifacts.update(
                _geometry_render_image_bindings(
                    render_report,
                    required_root=root,
                )
            )
        except AssetCompositionStateError as exc:
            contract_errors.append(str(exc))

    output: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = []
    if workflow_result.geometry_usd_path:
        candidate = Path(workflow_result.geometry_usd_path).expanduser().resolve()
        if candidate.is_file():
            output = _binding(
                candidate,
                label="Geometry output asset",
                required_root=root,
            )
            try:
                output_dependencies = _dependency_bindings(
                    candidate,
                    label="Geometry output asset",
                    required_root=root,
                )
            except AssetCompositionStateError as exc:
                contract_errors.append(str(exc))
            try:
                _validate_durable_geometry_usdc(candidate)
            except AssetCompositionStateError as exc:
                contract_errors.append(str(exc))
        else:
            contract_errors.append("Geometry workflow output path is not a file")
    else:
        contract_errors.append("Geometry workflow did not return an output asset")

    if render_report is not None and output is not None and render_binding is not None:
        if (
            not render_report.report_path
            or Path(render_report.report_path).expanduser().resolve()
            != Path(render_binding.path)
            or Path(render_report.source_usd_path).expanduser().resolve()
            != Path(output.path)
            or render_report.source_usd_sha256 != output.sha256
            or render_report.source_usd_sha256_after_render != output.sha256
        ):
            contract_errors.append(
                "Geometry OVRTX evidence does not bind the output bytes"
            )
        if geometry.render_evidence and (
            render_report.status != "pass"
            or "ovrtx" not in render_report.backend.lower()
            or not render_report.image_bindings
        ):
            contract_errors.append(
                "Geometry request requires passing OVRTX image evidence"
            )

    required_artifacts = {
        "handoff_manifest",
        "validation_evidence",
        "evidence_bundle",
        "optimization_metadata",
        "usd_validation",
    }
    missing_artifacts = sorted(required_artifacts.difference(artifacts))
    if missing_artifacts:
        contract_errors.append(
            f"Geometry workflow omitted required artifacts: {missing_artifacts}"
        )
    if geometry.render_evidence and "render_report" not in artifacts:
        contract_errors.append("Geometry workflow omitted required OVRTX evidence")

    if workflow_result.success and workflow_result.validation_status not in {
        "pass",
        "conditional",
    }:
        contract_errors.append(
            "Successful Geometry workflow lacks pass or conditional validation"
        )
    success = bool(workflow_result.success and not contract_errors)
    validation_status: Literal["pass", "conditional", "fail", "not_evaluated"] = (
        cast(
            Literal["pass", "conditional"],
            workflow_result.validation_status,
        )
        if success
        else "fail"
    )
    handoff_ready = workflow_result.handoff_ready if success else "no"
    error = None
    if not success:
        details = [item for item in [workflow_result.error, *contract_errors] if item]
        error = (
            "; ".join(details)
            or "Geometry workflow did not produce an admissible result"
        )

    result_path = attempt_root / "geometry_stage_result.json"
    result = AssetGeometryStageResult(
        run_id=run.run_id,
        run_revision=run.revision,
        stage_attempt=state.attempt_count,
        request=run.request,
        plan=plan_binding,
        input_asset=state.input_asset,
        input_dependencies=state.input_dependencies,
        result_path=str(result_path),
        workflow_result=workflow_binding,
        output_asset=output,
        output_dependencies=output_dependencies,
        artifacts=artifacts,
        success=success,
        validation_status=validation_status,
        handoff_ready=handoff_ready,
        error=error,
        timestamp=_timestamp(),
        actor=actor,
    )
    atomic_write_json(result_path, result)
    return result


def _geometry_stage_result_from_evidence(
    evidence: Sequence[ArtifactBinding],
) -> tuple[ArtifactBinding, AssetGeometryStageResult]:
    matches: list[tuple[ArtifactBinding, AssetGeometryStageResult]] = []
    for binding in evidence:
        if Path(binding.path).suffix.lower() != ".json":
            continue
        payload = _json_object_from_binding(
            binding,
            label=f"Geometry evidence {Path(binding.path).name}",
        )
        if payload.get("schema_version") != (
            "content-agent-workflows.asset-geometry-stage-result.v1"
        ):
            continue
        try:
            matches.append((binding, AssetGeometryStageResult.model_validate(payload)))
        except ValidationError as exc:
            raise AssetCompositionStateError(
                f"Composed Geometry stage result is invalid: {exc}"
            ) from exc
    if len(matches) != 1:
        raise AssetCompositionStateError(
            "Geometry acceptance requires exactly one typed Geometry stage result"
        )
    return matches[0]


def _validate_durable_geometry_usdc(path: Path) -> None:
    if path.suffix.lower() != ".usdc":
        raise AssetCompositionStateError(
            "Geometry output must use the durable .usdc suffix"
        )
    header = _read_stable_regular_prefix(
        path,
        label="Geometry durable USD",
        size=8,
    )
    if header != b"PXR-USDC":
        raise AssetCompositionStateError("Geometry output is not a binary USDC layer")
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
        if stage is None:
            raise RuntimeError("Usd.Stage.Open returned None")
        default_prim = stage.GetDefaultPrim()
        up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception as exc:
        raise AssetCompositionStateError(
            f"Geometry output cannot be reopened as USDC: {exc}"
        ) from exc
    if not default_prim:
        raise AssetCompositionStateError("Geometry USDC requires a default prim")
    if up_axis != "Z":
        raise AssetCompositionStateError(
            f"Geometry USDC must be Z-up, observed {up_axis!r}"
        )
    if abs(meters_per_unit - 1.0) > 1e-12:
        raise AssetCompositionStateError(
            f"Geometry USDC must author metersPerUnit=1.0, observed {meters_per_unit}"
        )


def _load_geometry_render_evidence(
    binding: ArtifactBinding,
) -> GeometryRenderEvidence:
    """Load one typed Geometry render report from its immutable binding."""

    from content_agent_workflows.geometry.rendering import GeometryRenderEvidence

    payload = _json_object_from_binding(binding, label="Geometry OVRTX evidence")
    try:
        return GeometryRenderEvidence.model_validate(payload)
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Geometry OVRTX evidence is invalid: {exc}"
        ) from exc


def _geometry_render_image_bindings(
    report: GeometryRenderEvidence,
    *,
    required_root: Path,
) -> dict[str, ArtifactBinding]:
    """Bind every image claimed by a typed Geometry render report."""

    result: dict[str, ArtifactBinding] = {}
    seen_paths: set[Path] = set()
    for index, image in enumerate(report.image_bindings):
        path = Path(image.path).expanduser().resolve()
        if path in seen_paths:
            raise AssetCompositionStateError(
                "Geometry OVRTX evidence repeats an image path"
            )
        seen_paths.add(path)
        binding = _binding(
            path,
            label=f"Geometry OVRTX image {index + 1}",
            required_root=required_root,
        )
        if binding.sha256 != image.sha256:
            raise AssetCompositionStateError(
                "Geometry OVRTX image digest differs from its render report"
            )
        result[f"render_image_{index:03d}"] = binding
    return result


def _validate_cad_modeling_coordinator_evidence(
    evidence: Sequence[ArtifactBinding],
    *,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    input_dependencies: list[ArtifactBinding],
    output: ArtifactBinding,
    output_dependencies: list[ArtifactBinding],
) -> None:
    """Validate the provider-neutral source specification admitted to Geometry."""

    result_binding, result = _cad_modeling_stage_result_from_evidence(evidence)
    state = run.stages["cad_modeling"]
    if not run.coordinator.plan_revisions:
        raise AssetCompositionStateError(
            "CAD modeling acceptance requires a sealed plan"
        )
    expected_plan = run.coordinator.plan_revisions[-1]
    expected_result_path = (
        stage_directory(
            state_path,
            "cad_modeling",
            attempt=state.attempt_count,
        )
        / "cad_modeling_stage_result.json"
    ).resolve()
    if (
        result.run_id != run.run_id
        or result.run_revision > run.revision
        or result.stage_attempt != state.attempt_count
        or result.request != run.request
        or result.plan != expected_plan
        or result.input_asset != input_asset
        or result.input_dependencies != input_dependencies
        or result.output_asset != output
        or result.output_dependencies != output_dependencies
        or Path(result.result_path).resolve() != expected_result_path
        or Path(result_binding.path) != expected_result_path
    ):
        raise AssetCompositionStateError(
            "CAD modeling result does not match the active run, plan, or artifacts"
        )
    if not result.success or result.error is not None:
        raise AssetCompositionStateError(
            "CAD modeling result is rejected and cannot enter Geometry"
        )
    required_bindings = {result_binding, *result.artifacts.values()}
    if not required_bindings.issubset(set(evidence)):
        raise AssetCompositionStateError(
            "CAD modeling review omitted one or more typed result artifacts"
        )
    if (
        result.source_manifest is None
        or result.output_asset is None
        or result.source_bundle_id is None
        or result.source_revision is None
        or result.source_provider_id is None
        or result.source_representation_id is None
    ):
        raise AssetCompositionStateError(
            "CAD modeling result lacks canonical source identity"
        )

    try:
        bundle = validate_external_source_manifest(
            result.source_manifest.path,
            output_path=output.path,
            output_sha256=output.sha256,
            output_size_bytes=output.size_bytes,
        )
        selected_representation, selected_path = select_source_usd(
            result.source_manifest.path,
            bundle,
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise AssetCompositionStateError(
            f"Geometry authoring source manifest is invalid: {exc}"
        ) from exc
    if (
        bundle.bundle_id != result.source_bundle_id
        or bundle.source_revision != result.source_revision
        or bundle.producer.provider_id != result.source_provider_id
        or selected_representation.representation_id != result.source_representation_id
        or selected_path != Path(output.path).resolve()
        or _source_bundle_parameter_values(bundle) != result.parameter_values
    ):
        raise AssetCompositionStateError(
            "CAD modeling result differs from its canonical source bundle"
        )

    frozen_request = load_verified_asset_request(state_path, run=run)
    request_policy = frozen_request.cad_modeling
    if request_policy is None:
        raise AssetCompositionStateError("Frozen CAD modeling policy is missing")
    if isinstance(request_policy, LegacyAssetCadModelingRequest):
        raise AssetCompositionStateError(
            "asset request v4 uses the retired direct-authoring policy and must "
            "be reviewed with the pre-provider runner"
        )
    if (
        bundle.producer.provider_id != request_policy.provider_id
        or not _requested_source_formats_satisfied(
            bundle,
            request_policy.required_outputs,
        )
        or not _parameter_values_match(
            result.parameter_values,
            request_policy.parameter_values,
        )
    ):
        raise AssetCompositionStateError(
            "Geometry source bundle does not satisfy the selected provider policy"
        )

    request_payload = _json_object_from_binding(
        result.authoring_request,
        label="geometry authoring request",
    )
    schema_version = request_payload.get("schema_version")
    try:
        if (
            schema_version
            == GeometryAuthoringRequest.model_fields["schema_version"].default
        ):
            authoring_request: (
                GeometryAuthoringRequest | GeometryAuthoringRevisionRequest
            ) = GeometryAuthoringRequest.model_validate(request_payload)
            if (
                authoring_request.prompt != frozen_request.prompt
                or authoring_request.target_profile != request_policy.target_profile
                or bundle.provenance.parent_bundle_id is not None
            ):
                raise AssetCompositionStateError(
                    "Generation request differs from the frozen authoring policy"
                )
            requested_parameters = authoring_request.parameters
        elif (
            schema_version
            == GeometryAuthoringRevisionRequest.model_fields["schema_version"].default
        ):
            authoring_request = GeometryAuthoringRevisionRequest.model_validate(
                request_payload
            )
            if (
                authoring_request.instructions != frozen_request.prompt
                or authoring_request.source_bundle.producer.provider_id
                != request_policy.provider_id
                or bundle.provenance.parent_bundle_id
                != authoring_request.source_bundle.bundle_id
            ):
                raise AssetCompositionStateError(
                    "Revision request differs from the reviewed source lineage"
                )
            requested_parameters = authoring_request.parameter_overrides
        else:
            raise AssetCompositionStateError(
                "CAD modeling used an unsupported authoring request schema"
            )
    except ValidationError as exc:
        raise AssetCompositionStateError(
            f"Geometry authoring request is invalid: {exc}"
        ) from exc

    if (
        authoring_request.requested_formats != tuple(request_policy.required_outputs)
        or {item.name: item.value for item in requested_parameters}
        != request_policy.parameter_values
        or bundle.provenance.request_digest
        != geometry_authoring_request_digest(authoring_request)
    ):
        raise AssetCompositionStateError(
            "Geometry authoring request or provenance differs from frozen policy"
        )

    references = authoring_request.references
    if len(references) != len(frozen_request.source_image_bindings):
        raise AssetCompositionStateError(
            "Geometry authoring request does not cover every frozen source image"
        )
    attempt_root = stage_directory(
        state_path,
        "cad_modeling",
        attempt=state.attempt_count,
    ).resolve()
    for index, (reference, binding) in enumerate(
        zip(references, frozen_request.source_image_bindings, strict=True),
        start=1,
    ):
        if reference.kind != "image":
            raise AssetCompositionStateError(
                "CAD source references must be bound images"
            )
        staged_path = Path(reference.artifact.path).expanduser().resolve()
        try:
            staged_path.relative_to(attempt_root)
        except ValueError as exc:
            raise AssetCompositionStateError(
                "CAD source image escaped the active attempt"
            ) from exc
        staged = _binding(
            staged_path,
            label=f"staged CAD source image {index}",
            required_root=_run_root(state_path),
        )
        if (
            staged.sha256 != binding.sha256
            or staged.size_bytes != binding.size_bytes
            or reference.artifact.sha256 != binding.sha256
            or reference.artifact.size_bytes != binding.size_bytes
        ):
            raise AssetCompositionStateError(
                "Staged CAD source image differs from its frozen input"
            )

    if Path(output.path).suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise AssetCompositionStateError("CAD modeling output must be USD")
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(output.path, load=Usd.Stage.LoadNone)
        if stage is None or not any(True for _prim in stage.TraverseAll()):
            raise RuntimeError("USD has no inspectable prims")
    except Exception as exc:  # noqa: BLE001 - normalize USD admission errors
        raise AssetCompositionStateError(
            f"CAD modeling output is not an inspectable USD stage: {exc}"
        ) from exc

    if request_policy.parameter_variants:
        if result.parameter_family is None:
            raise AssetCompositionStateError(
                "Requested CAD parameter variants lack family evidence"
            )
        family = _json_object_from_binding(
            result.parameter_family,
            label="geometry parameter family",
        )
        rows = family.get("rows")
        if (
            family.get("schema_version")
            != "content-agent-workflows.geometry-parameter-family.v1"
            or family.get("provider_id") != request_policy.provider_id
            or family.get("parent_bundle_id") != bundle.bundle_id
            or family.get("requested_rows") != len(request_policy.parameter_variants)
            or family.get("succeeded") is not True
            or not isinstance(rows, list)
            or len(rows) != len(request_policy.parameter_variants)
            or not all(isinstance(row, dict) for row in rows)
        ):
            raise AssetCompositionStateError(
                "Geometry parameter-family evidence is incomplete"
            )
        artifact_bindings = set(result.artifacts.values())
        baseline_parameter_values = _source_bundle_parameter_values(bundle)
        for variant, row in zip(
            request_policy.parameter_variants,
            rows,
            strict=True,
        ):
            expected_values = {
                **baseline_parameter_values,
                **variant.parameter_values,
            }
            try:
                row_manifest = ArtifactBinding.model_validate(
                    row.get("source_manifest")
                )
                row_output = ArtifactBinding.model_validate(row.get("output_asset"))
            except ValidationError as exc:
                raise AssetCompositionStateError(
                    f"Geometry parameter variant {variant.id!r} has invalid bindings"
                ) from exc
            if (
                row.get("id") != variant.id
                or row.get("parameter_values") != expected_values
                or row.get("source_provider_id") != request_policy.provider_id
                or row_manifest not in artifact_bindings
                or row_output not in artifact_bindings
            ):
                raise AssetCompositionStateError(
                    f"Geometry parameter variant {variant.id!r} differs from policy"
                )
            try:
                row_bundle = validate_external_source_manifest(
                    row_manifest.path,
                    output_path=row_output.path,
                    output_sha256=row_output.sha256,
                    output_size_bytes=row_output.size_bytes,
                )
                row_selected, _row_path = select_source_usd(
                    row_manifest.path,
                    row_bundle,
                )
            except (OSError, ValueError, ValidationError) as exc:
                raise AssetCompositionStateError(
                    f"Geometry parameter variant {variant.id!r} is invalid: {exc}"
                ) from exc
            if (
                row_bundle.bundle_id != row.get("source_bundle_id")
                or row_bundle.source_revision != row.get("source_revision")
                or row_bundle.producer.provider_id != row.get("source_provider_id")
                or row_bundle.provenance.parent_bundle_id != bundle.bundle_id
                or row_selected.representation_id
                != row.get("selected_representation_id")
                or not _parameter_values_match(
                    _source_bundle_parameter_values(row_bundle),
                    expected_values,
                )
            ):
                raise AssetCompositionStateError(
                    f"Geometry parameter variant {variant.id!r} identity is invalid"
                )
    elif result.parameter_family is not None:
        raise AssetCompositionStateError(
            "CAD modeling result contains an unrequested parameter family"
        )


def _validate_geometry_coordinator_evidence(
    evidence: Sequence[ArtifactBinding],
    *,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    input_dependencies: list[ArtifactBinding],
    output: ArtifactBinding,
    output_dependencies: list[ArtifactBinding],
) -> None:
    result_binding, result = _geometry_stage_result_from_evidence(evidence)
    state = run.stages["geometry"]
    if not run.coordinator.plan_revisions:
        raise AssetCompositionStateError("Geometry acceptance requires a sealed plan")
    expected_plan = run.coordinator.plan_revisions[-1]
    expected_result_path = (
        stage_directory(state_path, "geometry", attempt=state.attempt_count)
        / "geometry_stage_result.json"
    ).resolve()
    if (
        result.run_id != run.run_id
        or result.run_revision > run.revision
        or result.stage_attempt != state.attempt_count
        or result.request != run.request
        or result.plan != expected_plan
        or result.input_asset != input_asset
        or result.input_dependencies != input_dependencies
        or result.output_asset != output
        or result.output_dependencies != output_dependencies
        or Path(result.result_path).resolve() != expected_result_path
        or Path(result_binding.path) != expected_result_path
    ):
        raise AssetCompositionStateError(
            "Geometry stage result does not match the active run, plan, or artifacts"
        )
    if not result.success or result.handoff_ready == "no":
        raise AssetCompositionStateError(
            "Geometry result is rejected and cannot enter downstream stages"
        )
    if result.validation_status not in {"pass", "conditional"}:
        raise AssetCompositionStateError(
            "Geometry result lacks an admissible validation status"
        )
    required_bindings = {
        result_binding,
        result.workflow_result,
        *result.artifacts.values(),
    }
    if not required_bindings.issubset(set(evidence)):
        raise AssetCompositionStateError(
            "Geometry review omitted one or more typed result artifacts"
        )

    workflow_payload = _json_object_from_binding(
        result.workflow_result,
        label="Geometry workflow result",
    )
    for key, expected in (
        ("success", True),
        ("geometry_usd_path", output.path),
        ("validation_status", result.validation_status),
        ("handoff_ready", result.handoff_ready),
    ):
        _require_exact_claim(
            workflow_payload.get(key),
            expected,
            label=f"Geometry workflow result {key}",
        )

    manifest = _json_object_from_binding(
        result.artifacts["handoff_manifest"],
        label="Geometry handoff manifest",
    )
    geometry_record = manifest.get("workflow_geometry")
    if not isinstance(geometry_record, dict):
        raise AssetCompositionStateError(
            "Geometry handoff manifest lacks workflow_geometry identity"
        )
    for key, expected in (
        ("schema_version", "content-agent-workflows.geometry-handoff.v3"),
        ("geometry_usd", output.path),
        ("handoff_ready", result.handoff_ready),
    ):
        _require_exact_claim(
            manifest.get(key),
            expected,
            label=f"Geometry handoff manifest {key}",
        )
    _require_exact_claim(
        geometry_record.get("path"),
        output.path,
        label="Geometry manifest output path",
    )
    _require_exact_claim(
        geometry_record.get("sha256"),
        output.sha256,
        label="Geometry manifest output digest",
    )

    validation_payload = _json_object_from_binding(
        result.artifacts["validation_evidence"],
        label="Geometry validation evidence",
    )
    _require_exact_claim(
        validation_payload.get("workflow"),
        "geometry",
        label="Geometry validation workflow",
    )
    _require_exact_claim(
        validation_payload.get("asset"),
        output.path,
        label="Geometry validation asset",
    )
    metadata = validation_payload.get("metadata")
    if not isinstance(metadata, dict):
        raise AssetCompositionStateError(
            "Geometry validation evidence lacks typed metadata"
        )
    _require_exact_claim(
        metadata.get("handoff_ready"),
        result.handoff_ready,
        label="Geometry validation handoff readiness",
    )

    bundle = _json_object_from_binding(
        result.artifacts["evidence_bundle"],
        label="Geometry evidence bundle",
    )
    for key, expected in (
        ("geometry_usd", output.path),
        ("geometry_validation_status", result.validation_status),
    ):
        _require_exact_claim(
            bundle.get(key),
            expected,
            label=f"Geometry evidence bundle {key}",
        )

    optimization = _json_object_from_binding(
        result.artifacts["optimization_metadata"],
        label="Geometry optimization metadata",
    )
    optimization_status = str(optimization.get("status") or "")
    fallback_statuses = {
        "optimization_unavailable",
        "fidelity_fallback",
        "semantic_fidelity_fallback",
        "semantic_lock_skip",
    }
    if optimization_status in fallback_statuses and (
        result.validation_status == "pass" or result.handoff_ready == "yes"
    ):
        raise AssetCompositionStateError(
            "Geometry optimization fallback cannot be represented as an unconditional pass"
        )
    request = load_verified_asset_request(state_path, run=run)
    if request.geometry is None:
        raise AssetCompositionStateError(
            "Geometry stage lacks its frozen Geometry request policy"
        )
    render_binding = result.artifacts.get("render_report")
    if request.geometry.render_evidence and render_binding is None:
        raise AssetCompositionStateError(
            "Geometry request requires digest-bound OVRTX render evidence"
        )
    if render_binding is not None:
        render_report = _load_geometry_render_evidence(render_binding)
        expected_images = _geometry_render_image_bindings(
            render_report,
            required_root=_run_root(state_path),
        )
        actual_images = {
            label: binding
            for label, binding in result.artifacts.items()
            if label.startswith("render_image_")
        }
        if actual_images != expected_images:
            raise AssetCompositionStateError(
                "Geometry stage result omits or changes an OVRTX image binding"
            )
        if (
            not render_report.report_path
            or Path(render_report.report_path).expanduser().resolve()
            != Path(render_binding.path)
            or Path(render_report.source_usd_path).expanduser().resolve()
            != Path(output.path)
            or render_report.source_usd_sha256 != output.sha256
            or render_report.source_usd_sha256_after_render != output.sha256
        ):
            raise AssetCompositionStateError(
                "Geometry OVRTX evidence does not bind the accepted output bytes"
            )
        if request.geometry.render_evidence and (
            render_report.status != "pass"
            or "ovrtx" not in render_report.backend.lower()
            or not expected_images
        ):
            raise AssetCompositionStateError(
                "Geometry request requires passing OVRTX image evidence"
            )
    _validate_durable_geometry_usdc(Path(output.path))


def _validate_stage_acceptance(
    stage: StageName,
    *,
    state_path: Path,
    run: AssetCompositionRun,
    input_asset: ArtifactBinding,
    input_dependencies: list[ArtifactBinding],
    output: ArtifactBinding,
    output_dependencies: list[ArtifactBinding],
    evidence: list[ArtifactBinding],
) -> None:
    """Reject an invalid accept decision while the loop can still revise it."""

    if stage not in _ACCEPTANCE_VALIDATED_STAGES:
        raise AssetCompositionStateError(
            f"No coordinator acceptance validator is registered for stage {stage!r}"
        )
    output_path = Path(output.path)
    if stage == "cad_modeling":
        _validate_cad_modeling_coordinator_evidence(
            evidence,
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            input_dependencies=input_dependencies,
            output=output,
            output_dependencies=output_dependencies,
        )
    if stage == "geometry":
        _validate_geometry_coordinator_evidence(
            evidence,
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            input_dependencies=input_dependencies,
            output=output,
            output_dependencies=output_dependencies,
        )
    if stage == "articulation" and run.stages["articulation"].review_decisions is None:
        raise AssetCompositionStateError(
            "Articulation acceptance requires frozen Joint review decisions"
        )
    if stage == "articulation" and run.coordinator.mode == "single_reasoning_loop":
        review_candidates = run.stages["articulation"].review_candidates
        review_decisions = run.stages["articulation"].review_decisions
        _validate_articulation_coordinator_evidence(
            evidence,
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            output=output,
            review_candidates=review_candidates,
            review_decisions=review_decisions,
        )
    if (
        stage == "physics"
        and Path(input_asset.path).suffix.lower() == ".usdz"
        and output_path.suffix.lower() != ".usdz"
    ):
        raise AssetCompositionStateError(
            "Physics output must remain a self-contained USDZ when its exact "
            "input is USDZ"
        )
    if stage == "material" and run.coordinator.mode == "single_reasoning_loop":
        _validate_material_coordinator_evidence(
            run,
            input_asset=input_asset,
            output=output,
            evidence=evidence,
        )
    if stage == "texture" and run.coordinator.mode == "single_reasoning_loop":
        _validate_texture_coordinator_evidence(
            evidence,
            state_path=state_path,
            run=run,
            input_asset=input_asset,
            output=output,
        )
    if stage == "physics":
        validation_mode: PhysicsValidationMode = "runtime_required"
        if run.coordinator.mode == "single_reasoning_loop":
            request = load_verified_asset_request(state_path, run=run)
            validation_mode = request.physics_validation_mode
        _validate_physics_runtime_evidence(
            evidence,
            output=output,
            validation_mode=validation_mode,
        )
        if run.coordinator.mode == "single_reasoning_loop":
            _validate_physics_coordinator_evidence(
                evidence,
                input_asset=input_asset,
                output=output,
                validation_mode=validation_mode,
            )
        _validate_articulated_physics_output(output_path)
    if stage == "validation" and run.coordinator.mode == "single_reasoning_loop":
        _validate_validation_coordinator_evidence(
            evidence,
            run=run,
            input_asset=input_asset,
            input_dependencies=input_dependencies,
            output=output,
            output_dependencies=output_dependencies,
        )
    if stage == "finalization":
        if output_path.suffix.lower() != ".usdz":
            raise AssetCompositionStateError(
                "Finalization output must be a self-contained USDZ package"
            )
        _validate_final_usdz(output_path)
        _, report = _combined_report_from_evidence(evidence)
        _validate_combined_report(run, report=report, final_asset=output)


def complete_stage(
    path: str | Path,
    stage: StageName,
    *,
    output_asset: str | Path,
    evidence_paths: Sequence[str | Path],
    summary: str,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Seal one successful stage and advance only through a verified handoff."""

    state_path = _resolved(path)
    root = _run_root(state_path)
    normalized_summary = summary.strip()
    if not normalized_summary:
        raise AssetCompositionStateError("Stage summary must not be empty")
    if not evidence_paths:
        raise AssetCompositionStateError(
            "At least one evidence artifact is required to complete a stage"
        )
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.terminal_status != "active" or run.current_stage != stage:
            raise AssetCompositionStateError(f"{stage} is not the active stage")
        state = run.stages[stage]
        if state.status != "running":
            raise AssetCompositionStateError(
                f"Cannot complete {stage} from {state.status}; expected running"
            )
        if state.input_asset is None:
            raise AssetCompositionStateError(
                f"Running stage {stage} lacks its input binding"
            )
        output = _binding(
            output_asset,
            label=f"{stage} output asset",
            required_root=root,
        )
        output_path = Path(output.path)
        output_dependencies = _dependency_bindings(
            output_path,
            label=f"{stage} output asset",
            required_root=root,
        )
        evidence = [
            _binding(
                item,
                label=f"{stage} evidence {index + 1}",
                required_root=root,
            )
            for index, item in enumerate(evidence_paths)
        ]
        if run.coordinator.mode == "single_reasoning_loop":
            if (
                run.coordinator.next_action != "complete_stage"
                or not run.coordinator.evidence_reviews
            ):
                raise AssetCompositionStateError(
                    f"{stage} requires an accepting coordinator evidence review"
                )
            review = _load_coordinator_review(run.coordinator.evidence_reviews[-1])
            if (
                review.stage != stage
                or review.stage_attempt != state.attempt_count
                or review.decision != "accept"
                or review.input_asset != state.input_asset
                or review.output_asset != output
                or review.output_dependencies != output_dependencies
                or review.evidence != evidence
                or review.articulation_review_decisions
                != run.stages["articulation"].review_decisions
            ):
                raise AssetCompositionStateError(
                    f"Latest coordinator review does not accept exact {stage} artifacts"
                )
        _validate_stage_acceptance(
            stage,
            state_path=state_path,
            run=run,
            input_asset=state.input_asset,
            input_dependencies=state.input_dependencies,
            output=output,
            output_dependencies=output_dependencies,
            evidence=evidence,
        )
        handoff_readiness = state.input_readiness
        if stage == "geometry":
            _result_binding, geometry_result = _geometry_stage_result_from_evidence(
                evidence
            )
            if geometry_result.handoff_ready == "conditional":
                handoff_readiness = "conditional"
        handoff_path = (
            stage_directory(state_path, stage, attempt=state.attempt_count)
            / "handoff.json"
        )
        handoff = AssetStageHandoff(
            stage=stage,
            input_asset=state.input_asset,
            input_dependencies=state.input_dependencies,
            output_asset=output,
            output_dependencies=output_dependencies,
            evidence=evidence,
            readiness=handoff_readiness,
            summary=normalized_summary,
        )
        atomic_write_json(handoff_path, handoff)
        handoff_binding = _binding(
            handoff_path,
            label=f"{stage} handoff",
            required_root=root,
        )

        previous = state.status
        state.status = "completed"
        state.output_asset = output
        state.output_dependencies = output_dependencies
        state.handoff = handoff_binding
        state.evidence = evidence
        state.error = None
        _transition(
            run,
            stage,
            from_status=previous,
            to_status="completed",
            reason="Output asset and evidence were sealed into a verified handoff.",
            actor=actor,
        )

        index = run.stage_order.index(stage)
        if index + 1 == len(run.stage_order):
            run.current_stage = None
            run.terminal_status = "completed"
            if run.coordinator.mode == "single_reasoning_loop":
                run.coordinator.next_action = "terminal"
        else:
            successor = run.stage_order[index + 1]
            successor_state = run.stages[successor]
            if successor_state.status != "pending":
                raise AssetCompositionStateError(
                    f"Successor {successor} is unexpectedly {successor_state.status}"
                )
            successor_state.status = "ready"
            successor_state.input_asset = output
            successor_state.input_dependencies = output_dependencies
            successor_state.input_readiness = handoff_readiness
            run.current_stage = successor
            if run.coordinator.mode == "single_reasoning_loop":
                run.coordinator.next_action = "plan"
            _transition(
                run,
                successor,
                from_status="pending",
                to_status="ready",
                reason=f"Accepted {stage} output became the exact stage input.",
                actor=actor,
            )
        return _write_run(state_path, run)


def build_combined_report(
    path: str | Path,
    *,
    final_asset: str | Path,
    validation_summary: str | Path,
    output_path: str | Path,
) -> ArtifactBinding:
    """Publish the final digest index from already accepted stage evidence."""

    state_path = _resolved(path)
    root = _run_root(state_path)
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        state = run.stages["finalization"]
        if run.current_stage != "finalization" or state.status != "running":
            raise AssetCompositionStateError(
                "Combined report can be built only while finalization is running"
            )
        package = _binding(
            final_asset,
            label="final USDZ package",
            required_root=root,
        )
        if Path(package.path).suffix.lower() != ".usdz":
            raise AssetCompositionStateError("Final asset must use the .usdz suffix")
        _validate_final_usdz(Path(package.path))
        validation = _binding(
            validation_summary,
            label="validation summary",
            required_root=root,
        )
        if validation not in run.stages["validation"].evidence:
            raise AssetCompositionStateError(
                "Combined report validation summary must be accepted stage evidence"
            )
        canonical_summaries = [
            binding
            for binding in run.stages["validation"].evidence
            if Path(binding.path).name == "final_summary.json"
        ]
        if len(canonical_summaries) != 1 or validation != canonical_summaries[0]:
            raise AssetCompositionStateError(
                "Combined report requires the canonical accepted final_summary.json"
            )
        handoffs: dict[StageName, ArtifactBinding] = {}
        for stage in run.stage_order[:-1]:
            handoff = run.stages[stage].handoff
            if handoff is None:
                raise AssetCompositionStateError(
                    f"Cannot build combined report before {stage} completes"
                )
            handoffs[stage] = handoff
        destination = Path(output_path).expanduser()
        resolved_destination = destination.parent.resolve() / destination.name
        if not resolved_destination.is_relative_to(root):
            raise AssetCompositionStateError(
                "Combined report must be written inside the composed run directory"
            )
        report = AssetCombinedReport(
            run_id=run.run_id,
            request=run.request,
            source_asset=run.source_asset,
            source_dependencies=run.source_dependencies,
            final_asset=package,
            stage_handoffs=handoffs,
            validation_summary=validation,
        )
        expected_bytes = (
            json.dumps(
                report.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        if resolved_destination.is_symlink():
            raise AssetCompositionStateError(
                f"Refusing to replace combined report: {resolved_destination}"
            )
        if resolved_destination.exists():
            existing = _binding(
                resolved_destination,
                label="existing combined asset report",
                required_root=root,
            )
            if existing.sha256 != hashlib.sha256(
                expected_bytes
            ).hexdigest() or existing.size_bytes != len(expected_bytes):
                raise AssetCompositionStateError(
                    f"Refusing to replace combined report: {resolved_destination}"
                )
            return existing
        atomic_write_json(resolved_destination, report)
        return _binding(
            resolved_destination,
            label="combined asset report",
            required_root=root,
        )


def _stop_stage(
    path: str | Path,
    stage: StageName,
    *,
    reason: str,
    actor: str,
    status: StageStatus,
) -> AssetCompositionRun:
    state_path = _resolved(path)
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise AssetCompositionStateError("Failure or cancellation reason is required")
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.terminal_status != "active" or run.current_stage != stage:
            raise AssetCompositionStateError(f"{stage} is not the active stage")
        state = run.stages[stage]
        if state.status not in {"ready", "running", "needs_review"}:
            raise AssetCompositionStateError(
                f"Cannot mark {stage} {status} from {state.status}"
            )
        previous = state.status
        continue_current_attempt = previous == "running"
        if run.coordinator.mode == "single_reasoning_loop":
            # Only an executor that was still producing unreviewed evidence may
            # resume in place. Accept/await-review already sealed current paths.
            continue_current_attempt = (
                continue_current_attempt
                and run.coordinator.next_action == "execute_stage"
            )
            if previous == "ready" and state.continue_current_attempt:
                continue_current_attempt = True
        state.status = status
        state.continue_current_attempt = continue_current_attempt
        state.error = normalized_reason
        run.terminal_status = "cancelled" if status == "cancelled" else "failed"
        if run.coordinator.mode == "single_reasoning_loop":
            run.coordinator.next_action = "stopped"
            run.coordinator.stop_reason = normalized_reason
        _transition(
            run,
            stage,
            from_status=previous,
            to_status=status,
            reason=normalized_reason,
            actor=actor,
        )
        return _write_run(state_path, run)


def fail_stage(
    path: str | Path,
    stage: StageName,
    *,
    reason: str,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Durably stop the run with an explicit failed stage."""

    return _stop_stage(
        path,
        stage,
        reason=reason,
        actor=actor,
        status="failed",
    )


def cancel_stage(
    path: str | Path,
    stage: StageName,
    *,
    reason: str,
    actor: str = "agent",
) -> AssetCompositionRun:
    """Durably stop the run with an explicit cancelled stage."""

    return _stop_stage(
        path,
        stage,
        reason=reason,
        actor=actor,
        status="cancelled",
    )


def recover_stage(
    path: str | Path,
    stage: StageName,
    *,
    reason: str,
    actor: str = "operator",
) -> AssetCompositionRun:
    """Explicitly reopen only the current failed/cancelled stage for resume."""

    state_path = _resolved(path)
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise AssetCompositionStateError("Recovery reason is required")
    with _exclusive_lock(state_path):
        run = _load_verified_transition_run(state_path)
        if run.current_stage != stage:
            raise AssetCompositionStateError(f"{stage} is not the current stage")
        state = run.stages[stage]
        if state.status not in {"failed", "cancelled"}:
            raise AssetCompositionStateError(
                f"Cannot recover {stage} from {state.status}"
            )
        previous = state.status
        state.status = "ready"
        state.error = None
        run.terminal_status = "active"
        if run.coordinator.mode == "single_reasoning_loop":
            run.coordinator.next_action = "plan"
            run.coordinator.stop_reason = None
        _transition(
            run,
            stage,
            from_status=previous,
            to_status="ready",
            reason=normalized_reason,
            actor=actor,
        )
        return _write_run(state_path, run)


def validate_terminal(path: str | Path) -> AssetTerminalValidation:
    """Return exact terminal validation without converting failures to success."""

    try:
        run = load_verified_run(path)
        if run.coordinator.mode == "single_reasoning_loop":
            load_verified_asset_request(path, run=run)
    except AssetCompositionStateError as exc:
        return AssetTerminalValidation(
            schema_version=ASSET_TERMINAL_VALIDATION_SCHEMA_VERSION,
            valid=False,
            terminal_status="failed",
            current_stage=None,
            errors=[str(exc)],
        )
    errors: list[str] = []
    if run.terminal_status != "completed":
        errors.append(f"Run terminal status is {run.terminal_status}, not completed")
    if run.selected_mode == "agentic":
        terminal_artifact: ArtifactBinding | None = None
        if run.execution_graph is None:
            errors.append("Agentic run has no frozen execution graph")
        if run.graph_terminal_receipt is None:
            errors.append("Agentic run has no comprehensive terminal receipt")
        else:
            try:
                terminal = AssetGraphTerminalReceipt.model_validate(
                    _json_object_from_binding(
                        run.graph_terminal_receipt,
                        label="graph terminal receipt",
                    )
                )
            except (AssetCompositionStateError, ValidationError) as exc:
                errors.append(f"Invalid graph terminal receipt: {exc}")
            else:
                terminal_artifact = terminal.terminal_artifacts[0]
        invalid_leaves = [
            f"{leaf_id}={state.status}"
            for leaf_id, state in run.leaf_states.items()
            if (state.requirement == "required" and state.status != "passed")
            or (
                state.requirement == "optional"
                and state.status not in {"passed", "not_evaluated"}
            )
        ]
        if invalid_leaves:
            errors.append(f"Nonterminal graph dispositions: {invalid_leaves}")
        return AssetTerminalValidation(
            schema_version=ASSET_TERMINAL_VALIDATION_SCHEMA_VERSION,
            valid=not errors,
            terminal_status=run.terminal_status,
            current_stage=None,
            current_leaf_id=run.current_leaf_id,
            final_asset=terminal_artifact,
            graph=run.execution_graph,
            terminal_receipt=run.graph_terminal_receipt,
            errors=errors,
        )
    incomplete = [
        stage for stage in run.stage_order if run.stages[stage].status != "completed"
    ]
    if incomplete:
        errors.append(f"Incomplete stages: {incomplete}")
    final_asset = run.stages["finalization"].output_asset
    if final_asset is None:
        errors.append("Finalization has no output asset")
    final_handoff_binding = run.stages["finalization"].handoff
    if final_handoff_binding is not None:
        try:
            final_handoff = AssetStageHandoff.model_validate(
                _json_object_from_binding(
                    final_handoff_binding,
                    label="Finalization handoff",
                )
            )
        except ValidationError as exc:
            errors.append(f"Finalization handoff is invalid: {exc}")
        else:
            if final_handoff.readiness == "conditional":
                errors.append(
                    "Finalization remains conditional because an upstream Geometry "
                    "condition was not resolved"
                )
    return AssetTerminalValidation(
        schema_version="content-agent-workflows.asset-terminal-validation.v1",
        valid=not errors,
        terminal_status=run.terminal_status,
        current_stage=run.current_stage,
        final_asset=final_asset,
        errors=errors,
    )
