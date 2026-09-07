# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed USD extraction and authoring for controlled distance v1/v2."""

from __future__ import annotations

import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NoReturn, cast

from pydantic import BaseModel, ConfigDict, ValidationError
from world_understanding.functions.physics.joint_rigger import (
    ArtifactIdentityV1,
    FieldProvenanceV1,
)
from world_understanding.functions.physics.joint_rigger.reference import (
    RetainedUsdArtifactInspection,
)

from joint_agent.functions.articulation_contract import (
    ContractSummaryV1,
    LinkRecordV1,
    PrimRecordV1,
)
from joint_agent.functions.articulation_contract_v2 import (
    ArticulationContractV2,
    AttachmentFrameV2,
    DistanceConstraintV2,
    ExplicitAttachmentFramesV2,
    FieldEvidenceV2,
    JointRecordV2,
    canonical_articulation_v2_sha256,
)
from joint_agent.functions.articulation_v2_controlled_distance import (
    CONTROLLED_DISTANCE_BASE_FIELD_PROPERTIES,
    CONTROLLED_DISTANCE_FIELD_PROPERTIES,
    CONTROLLED_DISTANCE_SCHEMA_VERSION_V2,
    ControlledDistanceContractV1,
    ControlledDistanceContractV2,
    ControlledDistanceError,
    DistanceLinearDriveV1,
    DistanceLinearStateV1,
    bind_controlled_distance_contract_v1,
    bind_controlled_distance_contract_v2,
)

type Vector3 = tuple[float, float, float]
type QuaternionWxyz = tuple[float, float, float, float]
type ProvenanceSource = Literal[
    "accepted_manifest",
    "authored_metadata",
    "authored_reference",
    "source_metadata",
]
type ControlledDistanceV2SourceReadbackProtocol = Literal[
    "controlled-distance-retained-source-readback-v2",
    "controlled-distance-retained-source-readback-v2-fixture-audit",
]

_FRAME_TOLERANCE = 1e-6
_READBACK_RELATIVE_TOLERANCE = 1e-6
_EXPECTED_APPLIED_SCHEMAS = (
    "PhysicsDriveAPI:linear",
    "PhysicsJointStateAPI:linear",
)
_EXPECTED_AUTHORED_PROPERTIES = frozenset(
    {
        "drive:linear:physics:damping",
        "drive:linear:physics:maxForce",
        "drive:linear:physics:stiffness",
        "drive:linear:physics:targetPosition",
        "drive:linear:physics:targetVelocity",
        "drive:linear:physics:type",
        "physics:axis",
        "physics:body0",
        "physics:body1",
        "physics:localPos0",
        "physics:localPos1",
        "physics:localRot0",
        "physics:localRot1",
        "physics:lowerLimit",
        "physics:upperLimit",
        "state:linear:physics:position",
        "state:linear:physics:velocity",
    }
)
_EXPECTED_AUTHORED_METADATA = frozenset({"apiSchemas", "specifier", "typeName"})
_EXPECTED_ATTRIBUTE_TYPES = {
    "drive:linear:physics:damping": "float",
    "drive:linear:physics:maxForce": "float",
    "drive:linear:physics:stiffness": "float",
    "drive:linear:physics:targetPosition": "float",
    "drive:linear:physics:targetVelocity": "float",
    "drive:linear:physics:type": "token",
    "physics:axis": "token",
    "physics:localPos0": "point3f",
    "physics:localPos1": "point3f",
    "physics:localRot0": "quatf",
    "physics:localRot1": "quatf",
    "physics:lowerLimit": "float",
    "physics:upperLimit": "float",
    "state:linear:physics:position": "float",
    "state:linear:physics:velocity": "float",
}
_EXPECTED_UNIFORM_PROPERTIES = frozenset(
    {
        "drive:linear:physics:type",
        "physics:axis",
    }
)
_SOURCE_READBACK_WORKER_MODE = "controlled-distance-retained-source-readback-v1"
_SOURCE_READBACK_WORKER_MODE_V2: Literal[
    "controlled-distance-retained-source-readback-v2"
] = "controlled-distance-retained-source-readback-v2"
_SOURCE_READBACK_WORKER_MODE_V2_FIXTURE_AUDIT: Literal[
    "controlled-distance-retained-source-readback-v2-fixture-audit"
] = "controlled-distance-retained-source-readback-v2-fixture-audit"
_SOURCE_READBACK_TIMEOUT_SECONDS = 120.0
_SOURCE_READBACK_WORKER_RELATIVE_PATH = (
    "apps/joint_agent/joint_agent/functions/"
    "articulation_v2_controlled_distance_worker.py"
)
_SOURCE_READBACK_MODULE_RELATIVE_PATH = (
    "apps/joint_agent/joint_agent/functions/articulation_v2_controlled_distance_usd.py"
)


@dataclass(frozen=True)
class _ControlledDistanceV2IsolatedReadbackAuthority:
    source_root: Path
    dependency_snapshot: Path


@dataclass(frozen=True)
class _ControlledDistanceV2FixtureAuditReadbackAuthority:
    mode: Literal["explicit_fixture_audit"] = "explicit_fixture_audit"


_CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY: ContextVar[
    _ControlledDistanceV2IsolatedReadbackAuthority
    | _ControlledDistanceV2FixtureAuditReadbackAuthority
    | None
] = ContextVar(
    "controlled_distance_v2_isolated_readback_authority",
    default=None,
)


class _UsdResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ControlledDistanceStageReadbackV1(_UsdResultModel):
    """Exact values reconstructed from one saved controlled-distance joint."""

    joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    attachments: ExplicitAttachmentFramesV2
    axis_stage: Vector3
    axial_interval: DistanceConstraintV2
    drive: DistanceLinearDriveV1
    state: DistanceLinearStateV1
    meters_per_unit: float
    kilograms_per_unit: float


class ControlledDistanceStageReadbackV2(ControlledDistanceStageReadbackV1):
    """V2 readback using positive body1-to-body0 prismatic state."""

    contract_version: Literal[2] = 2


@contextmanager
def _controlled_distance_v2_isolated_readback_authority(
    *,
    sources: Mapping[str, bytes],
    dependency_snapshot: str | os.PathLike[str],
    private_source_parent: str | os.PathLike[str] | None = None,
) -> Iterator[None]:
    """Use only sealed bootstrap snapshots for production V2 child readback."""

    exact_sources = _validate_controlled_distance_v2_source_snapshot(sources)
    dependency_root = _require_canonical_snapshot_directory(
        dependency_snapshot,
        label="controlled-distance V2 dependency snapshot",
    )
    source_parent = (
        dependency_root.parent
        if private_source_parent is None
        else _require_canonical_snapshot_directory(
            private_source_parent,
            label="controlled-distance V2 private source parent",
        )
    )
    temporary_root = tempfile.TemporaryDirectory(
        prefix=".joint-agent-controlled-distance-v2-source-",
        dir=source_parent,
    )
    try:
        _require_private_readback_directory(Path(temporary_root.name))
        source_root = Path(temporary_root.name) / "source"
        source_root.mkdir(mode=0o700)
        _require_private_readback_directory(source_root)
        for relative, payload in exact_sources.items():
            path = source_root.joinpath(*PurePosixPath(relative).parts)
            parent = source_root
            for part in PurePosixPath(relative).parts[:-1]:
                parent /= part
                parent.mkdir(mode=0o700, exist_ok=True)
                _require_private_readback_directory(parent)
            _write_private_readback_source(path, payload)
        authority = _ControlledDistanceV2IsolatedReadbackAuthority(
            source_root=source_root,
            dependency_snapshot=dependency_root,
        )
        token = _CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY.set(authority)
        try:
            yield
        finally:
            _CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY.reset(token)
    finally:
        try:
            temporary_root.cleanup()
        except Exception as exc:
            # The reviewed source bytes are private copies. Cleanup occurs after
            # the executor has already returned a terminal result and therefore
            # cannot be allowed to rewrite that result as an execution failure,
            # even when the diagnostic stream is unavailable.
            try:
                sys.stderr.write(
                    "static execution warning "
                    "[articulation_v2_static_source_cleanup_failed]: remove the "
                    "private reviewed source snapshot manually at "
                    f"{temporary_root.name}: {exc}\n"
                )
            except Exception:
                _redirect_stderr_to_devnull()


def _redirect_stderr_to_devnull() -> None:
    """Prevent CPython's shutdown flush from changing a terminal result to 120."""

    descriptor = -1
    target = -1
    try:
        target = sys.stderr.fileno()
        descriptor = os.open(os.devnull, os.O_WRONLY)
        if descriptor != target:
            os.dup2(descriptor, target)
    except (AttributeError, OSError, ValueError):
        pass
    finally:
        if descriptor >= 0 and descriptor != target:
            os.close(descriptor)


@contextmanager
def _controlled_distance_v2_unsealed_readback_for_fixture_audit() -> Iterator[None]:
    """Explicitly select the non-production V2 fixture/audit readback protocol."""

    token = _CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY.set(
        _ControlledDistanceV2FixtureAuditReadbackAuthority()
    )
    try:
        yield
    finally:
        _CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY.reset(token)


def _validate_controlled_distance_v2_source_snapshot(
    sources: Mapping[str, bytes],
) -> dict[str, bytes]:
    if not isinstance(sources, Mapping):
        raise TypeError("controlled-distance V2 source snapshot must be a mapping")
    exact: dict[str, bytes] = {}
    for relative, payload in sources.items():
        if type(relative) is not str or type(payload) is not bytes:
            raise TypeError(
                "controlled-distance V2 source snapshot entries must be exact "
                "str-to-bytes pairs"
            )
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(
                "controlled-distance V2 source snapshot path is not canonical"
            )
        exact[relative] = payload
    required = {
        _SOURCE_READBACK_WORKER_RELATIVE_PATH,
        _SOURCE_READBACK_MODULE_RELATIVE_PATH,
    }
    if not required.issubset(exact):
        raise ValueError(
            "controlled-distance V2 source snapshot omits isolated readback source"
        )
    if any(not exact[relative] for relative in required):
        raise ValueError(
            "controlled-distance V2 isolated readback source must be nonempty"
        )
    supplemental_paths = [
        path.parts
        for relative in exact
        if (path := PurePosixPath(relative)).parts[0] == "packages"
    ]
    supplemental_roots = {parts[1] for parts in supplemental_paths if len(parts) >= 3}
    if (
        not supplemental_paths
        or any(len(parts) < 3 for parts in supplemental_paths)
        or len(supplemental_roots) != 1
    ):
        raise ValueError(
            "controlled-distance V2 source snapshot requires exactly one "
            "supplemental package root"
        )
    return exact


def _require_canonical_snapshot_directory(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be string-like") from exc
    if not raw or (isinstance(raw, str) and "\x00" in raw):
        raise ValueError(f"{label} must be an exact canonical directory")
    path = Path(os.path.abspath(raw))
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be an exact canonical directory")
    return path


def _write_private_readback_source(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("reviewed readback source write made no progress")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o400)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise OSError("reviewed readback source identity changed")
    finally:
        os.close(descriptor)


def _require_private_readback_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise OSError("reviewed readback source directory is not private")


class _SourceReadbackWorkerSuccessV1(_UsdResultModel):
    status: Literal["ok"]
    readback: ControlledDistanceStageReadbackV1


class _SourceReadbackWorkerErrorV1(_UsdResultModel):
    status: Literal["error"]
    code: str
    detail: str


class _SourceReadbackWorkerSuccessV2(_UsdResultModel):
    status: Literal["ok"]
    mode: Literal[
        "controlled-distance-retained-source-readback-v2",
        "controlled-distance-retained-source-readback-v2-fixture-audit",
    ]
    readback: ControlledDistanceStageReadbackV2


class SourceBackedControlledDistanceV1(_UsdResultModel):
    """Complete source facts for one axial controlled-distance joint."""

    source_artifact: ArtifactIdentityV1
    source_joint_path: str
    body0_prim_path: str
    body1_prim_path: str
    attachments: ExplicitAttachmentFramesV2
    axis_stage: Vector3
    axial_interval: DistanceConstraintV2
    drive: DistanceLinearDriveV1
    state: DistanceLinearStateV1
    articulation_field_evidence: tuple[FieldEvidenceV2, ...]
    controlled_field_evidence: tuple[FieldEvidenceV2, ...]


class SourceBackedControlledDistanceV2(SourceBackedControlledDistanceV1):
    """Complete v2 source facts using positive body1-to-body0 state."""

    contract_version: Literal[2] = 2
    source_readback_protocol: ControlledDistanceV2SourceReadbackProtocol


def extract_controlled_distance_source_from_retained_usd_v1(
    inspection: RetainedUsdArtifactInspection,
    *,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    provenance_source: ProvenanceSource = "authored_reference",
) -> SourceBackedControlledDistanceV1:
    """Extract source facts only from one immutable digest-verified projection."""

    return _extract_controlled_distance_source_from_retained_usd(
        inspection,
        source_joint_path=source_joint_path,
        expected_body0_prim_path=expected_body0_prim_path,
        expected_body1_prim_path=expected_body1_prim_path,
        provenance_source=provenance_source,
        contract_version=1,
    )


def extract_controlled_distance_source_from_retained_usd_v2(
    inspection: RetainedUsdArtifactInspection,
    *,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    provenance_source: ProvenanceSource = "authored_reference",
) -> SourceBackedControlledDistanceV2:
    """Extract v2 source facts from one immutable digest-verified projection."""

    return cast(
        SourceBackedControlledDistanceV2,
        _extract_controlled_distance_source_from_retained_usd(
            inspection,
            source_joint_path=source_joint_path,
            expected_body0_prim_path=expected_body0_prim_path,
            expected_body1_prim_path=expected_body1_prim_path,
            provenance_source=provenance_source,
            contract_version=2,
        ),
    )


def _extract_controlled_distance_source_from_retained_usd(
    inspection: RetainedUsdArtifactInspection,
    *,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    provenance_source: ProvenanceSource,
    contract_version: Literal[1, 2],
) -> SourceBackedControlledDistanceV1 | SourceBackedControlledDistanceV2:
    if type(inspection) is not RetainedUsdArtifactInspection:
        raise TypeError("inspection must be an exact RetainedUsdArtifactInspection")
    if provenance_source not in {
        "accepted_manifest",
        "authored_metadata",
        "authored_reference",
        "source_metadata",
    }:
        _fail(
            "controlled_distance_invalid_provenance_source",
            "controlled-distance provenance source is unsupported",
        )
    inspection.require_stage_unchanged()
    source_artifact = inspection.identity
    source_type: (
        type[SourceBackedControlledDistanceV1] | type[SourceBackedControlledDistanceV2]
    )
    source_readback_protocol: ControlledDistanceV2SourceReadbackProtocol | None
    if contract_version == 1:
        readback = _readback_retained_source_in_isolated_process(
            inspection,
            source_joint_path=source_joint_path,
            expected_body0_prim_path=expected_body0_prim_path,
            expected_body1_prim_path=expected_body1_prim_path,
        )
        source_type = SourceBackedControlledDistanceV1
        source_readback_protocol = None
    else:
        readback, source_readback_protocol = (
            _readback_retained_source_in_isolated_process_v2(
                inspection,
                source_joint_path=source_joint_path,
                expected_body0_prim_path=expected_body0_prim_path,
                expected_body1_prim_path=expected_body1_prim_path,
            )
        )
        source_type = SourceBackedControlledDistanceV2
    values: dict[str, Any] = {
        "source_artifact": source_artifact,
        "source_joint_path": source_joint_path,
        "body0_prim_path": readback.body0_prim_path,
        "body1_prim_path": readback.body1_prim_path,
        "attachments": readback.attachments,
        "axis_stage": readback.axis_stage,
        "axial_interval": readback.axial_interval,
        "drive": readback.drive,
        "state": readback.state,
        "articulation_field_evidence": _field_evidence(
            source_artifact=source_artifact,
            source_joint_path=source_joint_path,
            provenance_source=provenance_source,
            properties_by_field=CONTROLLED_DISTANCE_BASE_FIELD_PROPERTIES,
            body0_prim_path=readback.body0_prim_path,
            body1_prim_path=readback.body1_prim_path,
        ),
        "controlled_field_evidence": _field_evidence(
            source_artifact=source_artifact,
            source_joint_path=source_joint_path,
            provenance_source=provenance_source,
            properties_by_field=CONTROLLED_DISTANCE_FIELD_PROPERTIES,
            body0_prim_path=readback.body0_prim_path,
            body1_prim_path=readback.body1_prim_path,
        ),
    }
    if source_readback_protocol is not None:
        values["source_readback_protocol"] = source_readback_protocol
    result = source_type(**values)
    inspection.require_stage_unchanged()
    return result


def _readback_retained_source_in_isolated_process(
    inspection: RetainedUsdArtifactInspection,
    *,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
) -> ControlledDistanceStageReadbackV1:
    """Read one complete retained closure outside the caller's Sdf registry."""

    return _readback_retained_source_in_isolated_process_version(
        inspection,
        source_joint_path=source_joint_path,
        expected_body0_prim_path=expected_body0_prim_path,
        expected_body1_prim_path=expected_body1_prim_path,
        contract_version=1,
    )


def _readback_retained_source_in_isolated_process_v2(
    inspection: RetainedUsdArtifactInspection,
    *,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
) -> tuple[
    ControlledDistanceStageReadbackV2,
    ControlledDistanceV2SourceReadbackProtocol,
]:
    """Read one complete v2 retained closure outside the caller's Sdf registry."""

    authority = _CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY.get()
    protocol: ControlledDistanceV2SourceReadbackProtocol
    if type(authority) is _ControlledDistanceV2IsolatedReadbackAuthority:
        protocol = _SOURCE_READBACK_WORKER_MODE_V2
    elif type(authority) is _ControlledDistanceV2FixtureAuditReadbackAuthority:
        protocol = _SOURCE_READBACK_WORKER_MODE_V2_FIXTURE_AUDIT
    else:
        _fail(
            "controlled_distance_source_authority_missing",
            "controlled-distance V2 readback requires an explicit "
            "sealed-production or fixture-audit authority",
        )
    readback = cast(
        ControlledDistanceStageReadbackV2,
        _readback_retained_source_in_isolated_process_version(
            inspection,
            source_joint_path=source_joint_path,
            expected_body0_prim_path=expected_body0_prim_path,
            expected_body1_prim_path=expected_body1_prim_path,
            contract_version=2,
        ),
    )
    return readback, protocol


def _readback_retained_source_in_isolated_process_version(
    inspection: RetainedUsdArtifactInspection,
    *,
    source_joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    contract_version: Literal[1, 2],
) -> ControlledDistanceStageReadbackV1 | ControlledDistanceStageReadbackV2:
    _require_prim_path(source_joint_path, label="controlled-distance joint path")
    _require_prim_path(expected_body0_prim_path, label="expected body0 path")
    _require_prim_path(expected_body1_prim_path, label="expected body1 path")
    stage_path = inspection.stage_path
    authority = (
        _CONTROLLED_DISTANCE_V2_ISOLATED_READBACK_AUTHORITY.get()
        if contract_version == 2
        else None
    )
    sealed_authority: _ControlledDistanceV2IsolatedReadbackAuthority | None = None
    if contract_version == 2:
        if type(authority) is _ControlledDistanceV2IsolatedReadbackAuthority:
            sealed_authority = authority
        elif type(authority) is not _ControlledDistanceV2FixtureAuditReadbackAuthority:
            _fail(
                "controlled_distance_source_authority_missing",
                "controlled-distance V2 readback requires an explicit "
                "sealed-production or fixture-audit authority",
            )
    try:
        worker_path = (
            sealed_authority.source_root / _SOURCE_READBACK_WORKER_RELATIVE_PATH
            if sealed_authority is not None
            else Path(__file__).with_name(
                "articulation_v2_controlled_distance_worker.py"
            )
        ).resolve(strict=True)
    except OSError as exc:
        _fail(
            "controlled_distance_source_worker_failed",
            f"isolated controlled-distance worker is unavailable: {type(exc).__name__}",
        )
    if contract_version == 1:
        mode = _SOURCE_READBACK_WORKER_MODE
    elif sealed_authority is not None:
        mode = _SOURCE_READBACK_WORKER_MODE_V2
    else:
        mode = _SOURCE_READBACK_WORKER_MODE_V2_FIXTURE_AUDIT
    command: tuple[str, ...]
    if sealed_authority is None:
        command = (
            sys.executable,
            "-I",
            "-B",
            str(worker_path),
            mode,
            str(stage_path),
            source_joint_path,
            expected_body0_prim_path,
            expected_body1_prim_path,
        )
    else:
        command = (
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            str(worker_path),
            mode,
            str(sealed_authority.source_root),
            str(sealed_authority.dependency_snapshot),
            str(stage_path),
            source_joint_path,
            expected_body0_prim_path,
            expected_body1_prim_path,
        )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed trusted module/argv
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_SOURCE_READBACK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(
            "controlled_distance_source_worker_failed",
            f"isolated controlled-distance source readback failed: {type(exc).__name__}",
        )
    inspection.require_stage_unchanged()
    if completed.stderr:
        _fail(
            "controlled_distance_source_worker_failed",
            "isolated controlled-distance source readback emitted terminal diagnostics",
        )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(
            "controlled_distance_source_worker_failed",
            f"isolated controlled-distance source readback is invalid: {type(exc).__name__}",
        )
    if not isinstance(payload, dict):
        _fail(
            "controlled_distance_source_worker_failed",
            "isolated controlled-distance source readback is not an object",
        )
    status = payload.get("status")
    if completed.returncode == 0 and status == "ok":
        try:
            if contract_version == 1:
                return _SourceReadbackWorkerSuccessV1.model_validate_json(
                    completed.stdout,
                    strict=True,
                ).readback
            success = _SourceReadbackWorkerSuccessV2.model_validate_json(
                completed.stdout,
                strict=True,
            )
            if success.mode != mode:
                _fail(
                    "controlled_distance_source_worker_failed",
                    "isolated controlled-distance V2 source readback used the wrong "
                    "authority protocol",
                )
            return success.readback
        except ValidationError as exc:
            _fail(
                "controlled_distance_source_worker_failed",
                "isolated controlled-distance source readback violates its schema: "
                f"{exc.errors(include_url=False)}",
            )
    if completed.returncode == 2 and status == "error":
        try:
            error = _SourceReadbackWorkerErrorV1.model_validate_json(
                completed.stdout,
                strict=True,
            )
        except ValidationError as exc:
            _fail(
                "controlled_distance_source_worker_failed",
                "isolated controlled-distance source failure violates its schema: "
                f"{exc.errors(include_url=False)}",
            )
        _fail(error.code, error.detail)
    _fail(
        "controlled_distance_source_worker_failed",
        "isolated controlled-distance source readback returned an invalid "
        f"status/return code pair: status={status!r}, returncode={completed.returncode}",
    )


def controlled_distance_joint_record_v1(
    *,
    joint_id: str,
    body0_link: str,
    body1_link: str,
    source: SourceBackedControlledDistanceV1,
) -> JointRecordV2:
    """Build the ready base v2 record that the overlay binds exactly."""

    if type(source) is not SourceBackedControlledDistanceV1:
        raise TypeError("source must be an exact SourceBackedControlledDistanceV1")
    return JointRecordV2(
        kind="joint",
        joint_id=joint_id,
        body0_link=body0_link,
        body1_link=body1_link,
        attachments=source.attachments,
        constraint=source.axial_interval,
        field_evidence=source.articulation_field_evidence,
        review_status="ready_for_rigger_input",
    )


def controlled_distance_joint_record_v2(
    *,
    joint_id: str,
    body0_link: str,
    body1_link: str,
    source: SourceBackedControlledDistanceV2,
) -> JointRecordV2:
    """Build the ready base record for the corrected v2 source convention."""

    if type(source) is not SourceBackedControlledDistanceV2:
        raise TypeError("source must be an exact SourceBackedControlledDistanceV2")
    return JointRecordV2(
        kind="joint",
        joint_id=joint_id,
        body0_link=body0_link,
        body1_link=body1_link,
        attachments=source.attachments,
        constraint=source.axial_interval,
        field_evidence=source.articulation_field_evidence,
        review_status="ready_for_rigger_input",
    )


def articulation_contract_from_controlled_distance_source_v2(
    source: SourceBackedControlledDistanceV2,
    *,
    expected_articulation_contract_sha256: str,
) -> ArticulationContractV2:
    """Rebuild the exact admitted V2 base contract from retained source facts.

    The controlled-distance-v2 representation fixes the record identifiers and
    roles below as part of its wire convention.  The admitted contract digest
    is mandatory and rejects any source or convention drift before authoring.
    """

    if type(source) is not SourceBackedControlledDistanceV2:
        raise TypeError("source must be an exact SourceBackedControlledDistanceV2")
    if len(expected_articulation_contract_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_articulation_contract_sha256
    ):
        raise ValueError("expected articulation contract must be a lowercase SHA-256")

    def provenance(
        *, prim_path: str, properties: tuple[str, ...], evidence: str
    ) -> FieldProvenanceV1:
        return FieldProvenanceV1(
            source="authored_reference",
            artifact=source.source_artifact,
            prim_path=prim_path,
            properties=properties,
            evidence=evidence,
        )

    body0_membership = provenance(
        prim_path=source.body0_prim_path,
        properties=("link_id",),
        evidence="source-backed base membership",
    )
    body1_membership = provenance(
        prim_path=source.body1_prim_path,
        properties=("link_id",),
        evidence="source-backed carriage membership",
    )
    contract = ArticulationContractV2(
        schema_version="joint-agent-articulation-v2",
        status="ready_for_rigger_input",
        articulation_roots=("base",),
        source_identities=(source.source_artifact,),
        records=(
            PrimRecordV1(
                kind="prim",
                prim_path=source.body0_prim_path,
                link_id="base",
                membership_evidence=body0_membership,
            ),
            PrimRecordV1(
                kind="prim",
                prim_path=source.body1_prim_path,
                link_id="carriage",
                membership_evidence=body1_membership,
            ),
            LinkRecordV1(
                kind="link",
                link_id="base",
                body_prim_path=source.body0_prim_path,
                role="base",
                review_status="ready_for_rigger_input",
                field_evidence={
                    "body_prim_path": provenance(
                        prim_path=source.body0_prim_path,
                        properties=("body_prim_path",),
                        evidence="source-backed base body path",
                    ),
                    "role": provenance(
                        prim_path=source.body0_prim_path,
                        properties=("role",),
                        evidence="source-backed base role",
                    ),
                },
            ),
            LinkRecordV1(
                kind="link",
                link_id="carriage",
                body_prim_path=source.body1_prim_path,
                role="controlled_distance_carriage",
                review_status="ready_for_rigger_input",
                field_evidence={
                    "body_prim_path": provenance(
                        prim_path=source.body1_prim_path,
                        properties=("body_prim_path",),
                        evidence="source-backed carriage body path",
                    ),
                    "role": provenance(
                        prim_path=source.body1_prim_path,
                        properties=("role",),
                        evidence="source-backed carriage role",
                    ),
                },
            ),
            controlled_distance_joint_record_v2(
                joint_id="controlled_distance",
                body0_link="base",
                body1_link="carriage",
                source=source,
            ),
        ),
        diagnostics=(),
        summary=ContractSummaryV1(
            prim_count=2,
            link_count=2,
            joint_count=1,
            review_required_link_count=0,
            review_required_joint_count=0,
            diagnostic_count=0,
        ),
    )
    if canonical_articulation_v2_sha256(contract) != (
        expected_articulation_contract_sha256
    ):
        _fail(
            "controlled_distance_articulation_contract_mismatch",
            "retained source cannot reconstruct the admitted V2 base contract",
        )
    return contract


def controlled_distance_contract_from_source_v1(
    articulation_contract: ArticulationContractV2,
    *,
    joint_id: str,
    source: SourceBackedControlledDistanceV1,
) -> ControlledDistanceContractV1:
    """Bind extracted source facts to the exact completed base document."""

    if type(source) is not SourceBackedControlledDistanceV1:
        raise TypeError("source must be an exact SourceBackedControlledDistanceV1")
    controlled = ControlledDistanceContractV1(
        articulation_contract_sha256=canonical_articulation_v2_sha256(
            articulation_contract
        ),
        source_artifact=source.source_artifact,
        source_joint_path=source.source_joint_path,
        body0_prim_path=source.body0_prim_path,
        body1_prim_path=source.body1_prim_path,
        joint_id=joint_id,
        axis_stage=source.axis_stage,
        axial_interval=source.axial_interval,
        drive=source.drive,
        state=source.state,
        field_evidence=source.controlled_field_evidence,
    )
    matches = tuple(
        record
        for record in articulation_contract.records
        if isinstance(record, JointRecordV2) and record.joint_id == joint_id
    )
    links = {
        item.link_id: item
        for item in articulation_contract.records
        if isinstance(item, LinkRecordV1)
    }
    if (
        len(matches) != 1
        or matches[0].attachments != source.attachments
        or links.get(matches[0].body0_link) is None
        or links.get(matches[0].body1_link) is None
        or links[matches[0].body0_link].body_prim_path != source.body0_prim_path
        or links[matches[0].body1_link].body_prim_path != source.body1_prim_path
    ):
        _fail(
            "controlled_distance_source_attachment_conflict",
            "controlled source attachments or endpoints do not match the base record",
        )
    return bind_controlled_distance_contract_v1(
        articulation_contract,
        controlled,
    )


def controlled_distance_contract_from_source_v2(
    articulation_contract: ArticulationContractV2,
    *,
    joint_id: str,
    source: SourceBackedControlledDistanceV2,
) -> ControlledDistanceContractV2:
    """Bind corrected extracted source facts to the completed base document."""

    if type(source) is not SourceBackedControlledDistanceV2:
        raise TypeError("source must be an exact SourceBackedControlledDistanceV2")
    controlled = ControlledDistanceContractV2(
        schema_version=CONTROLLED_DISTANCE_SCHEMA_VERSION_V2,
        articulation_contract_sha256=canonical_articulation_v2_sha256(
            articulation_contract
        ),
        source_artifact=source.source_artifact,
        source_joint_path=source.source_joint_path,
        body0_prim_path=source.body0_prim_path,
        body1_prim_path=source.body1_prim_path,
        joint_id=joint_id,
        representation="prismatic_axial_interval_drive_body1_to_body0_v2",
        axis_stage=source.axis_stage,
        axial_interval=source.axial_interval,
        drive=source.drive,
        state=source.state,
        field_evidence=source.controlled_field_evidence,
    )
    matches = tuple(
        record
        for record in articulation_contract.records
        if isinstance(record, JointRecordV2) and record.joint_id == joint_id
    )
    links = {
        item.link_id: item
        for item in articulation_contract.records
        if isinstance(item, LinkRecordV1)
    }
    if (
        len(matches) != 1
        or matches[0].attachments != source.attachments
        or links.get(matches[0].body0_link) is None
        or links.get(matches[0].body1_link) is None
        or links[matches[0].body0_link].body_prim_path != source.body0_prim_path
        or links[matches[0].body1_link].body_prim_path != source.body1_prim_path
    ):
        _fail(
            "controlled_distance_source_attachment_conflict",
            "controlled source attachments or endpoints do not match the base record",
        )
    return bind_controlled_distance_contract_v2(
        articulation_contract,
        controlled,
    )


def _bind_controlled_distance_version(
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1 | ControlledDistanceContractV2,
    *,
    contract_version: Literal[1, 2],
) -> ControlledDistanceContractV1 | ControlledDistanceContractV2:
    if contract_version == 1:
        return bind_controlled_distance_contract_v1(
            articulation_contract,
            controlled_distance,
        )
    return bind_controlled_distance_contract_v2(
        articulation_contract,
        cast(ControlledDistanceContractV2, controlled_distance),
    )


def author_controlled_distance_stage_v1(
    stage: Any,
    *,
    source_inspection: RetainedUsdArtifactInspection,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1,
    joint_path: str,
) -> ControlledDistanceStageReadbackV1:
    """Revalidate source, author one create-only joint, and verify readback."""

    return _author_controlled_distance_stage(
        stage,
        source_inspection=source_inspection,
        articulation_contract=articulation_contract,
        controlled_distance=controlled_distance,
        joint_path=joint_path,
        contract_version=1,
    )


def author_controlled_distance_stage_v2(
    stage: Any,
    *,
    source_inspection: RetainedUsdArtifactInspection,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV2,
    joint_path: str,
) -> ControlledDistanceStageReadbackV2:
    """Author the corrected body1-to-body0 controlled-distance successor."""

    return cast(
        ControlledDistanceStageReadbackV2,
        _author_controlled_distance_stage(
            stage,
            source_inspection=source_inspection,
            articulation_contract=articulation_contract,
            controlled_distance=controlled_distance,
            joint_path=joint_path,
            contract_version=2,
        ),
    )


def _author_controlled_distance_stage(
    stage: Any,
    *,
    source_inspection: RetainedUsdArtifactInspection,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1 | ControlledDistanceContractV2,
    joint_path: str,
    contract_version: Literal[1, 2],
) -> ControlledDistanceStageReadbackV1 | ControlledDistanceStageReadbackV2:
    """Shared authoring after an explicit version selects state body order."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    controlled = _bind_controlled_distance_version(
        articulation_contract,
        controlled_distance,
        contract_version=contract_version,
    )
    record = next(
        cast(JointRecordV2, item)
        for item in articulation_contract.records
        if isinstance(item, JointRecordV2) and item.joint_id == controlled.joint_id
    )
    if not isinstance(  # pragma: no cover - strict ready-v2 binding invariant
        record.attachments,
        ExplicitAttachmentFramesV2,
    ):
        _fail(
            "controlled_distance_authoring_frames_missing",
            "controlled-distance authoring requires explicit attachment frames",
        )
    links = {
        item.link_id: item
        for item in articulation_contract.records
        if isinstance(item, LinkRecordV1)
    }
    body0 = links.get(record.body0_link)
    body1 = links.get(record.body1_link)
    if (  # pragma: no cover - strict articulation-v2 graph invariant
        body0 is None or body1 is None
    ):
        _fail(
            "controlled_distance_authoring_endpoint_missing",
            "controlled-distance body links must resolve exactly",
        )
    _require_controlled_distance_matches_retained_source(
        source_inspection,
        controlled_distance=controlled,
        attachments=record.attachments,
        contract_version=contract_version,
    )
    joint_sdf_path = _require_prim_path(joint_path, label="authoring joint path")
    if stage.GetPrimAtPath(joint_path).IsValid():
        _fail(
            "controlled_distance_authoring_path_collision",
            f"controlled-distance joint path already exists: {joint_path}",
        )
    parent_path = joint_sdf_path.GetParentPath()
    if parent_path != Sdf.Path.absoluteRootPath:
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            _fail(
                "controlled_distance_authoring_parent_missing",
                "controlled-distance authoring parent must already exist",
            )
        if not _is_concrete_active_prim(parent):
            _fail(
                "controlled_distance_authoring_parent_invalid",
                "controlled-distance authoring parent must be active and concrete",
            )
    body0_prim = _require_endpoint_rigid_body(
        stage,
        body0.body_prim_path,
        UsdPhysics=UsdPhysics,
    )
    body1_prim = _require_endpoint_rigid_body(
        stage,
        body1.body_prim_path,
        UsdPhysics=UsdPhysics,
    )
    _require_static_translation_endpoint(body0_prim, UsdGeom=UsdGeom)
    _require_static_translation_endpoint(body1_prim, UsdGeom=UsdGeom)
    _require_empty_target_joint_graph(stage)
    meters_per_unit, kilograms_per_unit = _stage_units(stage)
    _storage_projected_semantics(
        controlled,
        attachments=record.attachments,
        meters_per_unit=meters_per_unit,
        kilograms_per_unit=kilograms_per_unit,
    )
    edit_layer, edit_layer_backup = _backup_edit_layer(
        stage,
        joint_path=joint_sdf_path,
    )

    minimum = cast(float, controlled.axial_interval.minimum_meters)
    maximum = cast(float, controlled.axial_interval.maximum_meters)
    try:
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0.body_prim_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1.body_prim_path)])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(
                *(
                    value / meters_per_unit
                    for value in record.attachments.body0.position_meters
                )
            )
        )
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(
                *(
                    value / meters_per_unit
                    for value in record.attachments.body1.position_meters
                )
            )
        )
        joint.CreateLocalRot0Attr().Set(
            _quatf(record.attachments.body0.orientation_wxyz, Gf=Gf)
        )
        joint.CreateLocalRot1Attr().Set(
            _quatf(record.attachments.body1.orientation_wxyz, Gf=Gf)
        )
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.x)
        joint.CreateLowerLimitAttr().Set(minimum / meters_per_unit)
        joint.CreateUpperLimitAttr().Set(maximum / meters_per_unit)

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
        drive.CreateTargetPositionAttr().Set(
            controlled.drive.target_position_meters / meters_per_unit
        )
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateStiffnessAttr().Set(
            controlled.drive.stiffness_newtons_per_meter / kilograms_per_unit
        )
        drive.CreateDampingAttr().Set(
            controlled.drive.damping_newton_seconds_per_meter / kilograms_per_unit
        )
        drive.CreateMaxForceAttr().Set(
            controlled.drive.maximum_force_newtons
            / (kilograms_per_unit * meters_per_unit)
        )

        prim = joint.GetPrim()
        if not prim.AddAppliedSchema("PhysicsJointStateAPI:linear"):
            _fail(
                "controlled_distance_state_schema_authoring_failed",
                "could not author PhysicsJointStateAPI:linear",
            )
        prim.CreateAttribute(
            "state:linear:physics:position",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(controlled.state.position_meters / meters_per_unit)
        prim.CreateAttribute(
            "state:linear:physics:velocity",
            Sdf.ValueTypeNames.Float,
            custom=False,
        ).Set(0.0)

        if contract_version == 1:
            return _require_controlled_distance_stage_matches_bound_v1(
                stage,
                articulation_contract=articulation_contract,
                controlled_distance=controlled,
                joint_path=joint_path,
            )
        return _require_controlled_distance_stage_matches_bound_v2(
            stage,
            articulation_contract=articulation_contract,
            controlled_distance=cast(ControlledDistanceContractV2, controlled),
            joint_path=joint_path,
        )
    except BaseException as exc:
        rollback_exc: BaseException | None = None
        try:
            _restore_edit_layer(edit_layer, edit_layer_backup)
        except BaseException as caught_rollback_exc:
            rollback_exc = caught_rollback_exc
        if rollback_exc is not None:
            if not isinstance(exc, Exception):
                exc.add_note(
                    "Controlled-distance authoring rollback also failed: "
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
                raise exc from rollback_exc
            if not isinstance(rollback_exc, Exception):
                rollback_exc.add_note(
                    "Controlled-distance authoring also failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise rollback_exc from exc
            failures = ExceptionGroup(
                "controlled-distance authoring and rollback both failed",
                [exc, rollback_exc],
            )
            raise ControlledDistanceError(
                "controlled_distance_authoring_rollback_failed",
                "could not restore the authoring edit layer after "
                f"{type(exc).__name__}; rollback failed with "
                f"{type(rollback_exc).__name__}",
            ) from failures
        raise


def _require_controlled_distance_matches_retained_source(
    inspection: RetainedUsdArtifactInspection,
    *,
    controlled_distance: ControlledDistanceContractV1 | ControlledDistanceContractV2,
    attachments: ExplicitAttachmentFramesV2,
    contract_version: Literal[1, 2],
) -> None:
    """Bind authoring semantics to a fresh read of the retained source bytes."""

    controlled = controlled_distance
    if contract_version == 1:
        source = extract_controlled_distance_source_from_retained_usd_v1(
            inspection,
            source_joint_path=controlled.source_joint_path,
            expected_body0_prim_path=controlled.body0_prim_path,
            expected_body1_prim_path=controlled.body1_prim_path,
        )
    else:
        source = extract_controlled_distance_source_from_retained_usd_v2(
            inspection,
            source_joint_path=controlled.source_joint_path,
            expected_body0_prim_path=controlled.body0_prim_path,
            expected_body1_prim_path=controlled.body1_prim_path,
        )
    if (
        source.source_artifact != controlled.source_artifact
        or source.source_joint_path != controlled.source_joint_path
        or source.body0_prim_path != controlled.body0_prim_path
        or source.body1_prim_path != controlled.body1_prim_path
        or source.attachments != attachments
        or source.axis_stage != controlled.axis_stage
        or source.axial_interval != controlled.axial_interval
        or source.drive != controlled.drive
        or source.state != controlled.state
    ):
        _fail(
            "controlled_distance_source_semantics_conflict",
            "controlled-distance authoring semantics differ from retained source",
        )


def require_controlled_distance_stage_matches_v1(
    stage: Any,
    *,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1,
    joint_path: str,
) -> ControlledDistanceStageReadbackV1:
    """Require exact typed readback from a saved or in-memory stage."""

    controlled = bind_controlled_distance_contract_v1(
        articulation_contract,
        controlled_distance,
    )
    return _require_controlled_distance_stage_matches_bound_v1(
        stage,
        articulation_contract=articulation_contract,
        controlled_distance=controlled,
        joint_path=joint_path,
    )


def require_controlled_distance_stage_matches_v2(
    stage: Any,
    *,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV2,
    joint_path: str,
) -> ControlledDistanceStageReadbackV2:
    """Require corrected v2 typed readback from a saved or in-memory stage."""

    controlled = bind_controlled_distance_contract_v2(
        articulation_contract,
        controlled_distance,
    )
    return _require_controlled_distance_stage_matches_bound_v2(
        stage,
        articulation_contract=articulation_contract,
        controlled_distance=controlled,
        joint_path=joint_path,
    )


def _require_controlled_distance_stage_matches_bound_v1(
    stage: Any,
    *,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1,
    joint_path: str,
) -> ControlledDistanceStageReadbackV1:
    """Preserve the frozen v1 private verification seam."""

    return _require_controlled_distance_stage_matches_bound(
        stage,
        articulation_contract=articulation_contract,
        controlled_distance=controlled_distance,
        joint_path=joint_path,
        contract_version=1,
    )


def _require_controlled_distance_stage_matches_bound_v2(
    stage: Any,
    *,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV2,
    joint_path: str,
) -> ControlledDistanceStageReadbackV2:
    return cast(
        ControlledDistanceStageReadbackV2,
        _require_controlled_distance_stage_matches_bound(
            stage,
            articulation_contract=articulation_contract,
            controlled_distance=controlled_distance,
            joint_path=joint_path,
            contract_version=2,
        ),
    )


def _require_controlled_distance_stage_matches_bound(
    stage: Any,
    *,
    articulation_contract: ArticulationContractV2,
    controlled_distance: ControlledDistanceContractV1 | ControlledDistanceContractV2,
    joint_path: str,
    contract_version: Literal[1, 2],
) -> ControlledDistanceStageReadbackV1 | ControlledDistanceStageReadbackV2:
    controlled = controlled_distance
    _require_exact_saved_joint_graph(stage, joint_path=joint_path)
    links = {
        item.link_id: item
        for item in articulation_contract.records
        if isinstance(item, LinkRecordV1)
    }
    record = next(
        cast(JointRecordV2, item)
        for item in articulation_contract.records
        if isinstance(item, JointRecordV2) and item.joint_id == controlled.joint_id
    )
    if not isinstance(  # pragma: no cover - strict ready-v2 binding invariant
        record.attachments,
        ExplicitAttachmentFramesV2,
    ):
        _fail(
            "controlled_distance_saved_attachment_conflict",
            "controlled-distance saved readback requires explicit attachment frames",
        )
    body0 = links[record.body0_link]
    body1 = links[record.body1_link]
    meters_per_unit, kilograms_per_unit = _stage_units(stage)
    (
        expected_attachments,
        expected_interval,
        expected_drive,
        expected_state,
    ) = _storage_projected_semantics(
        controlled,
        attachments=record.attachments,
        meters_per_unit=meters_per_unit,
        kilograms_per_unit=kilograms_per_unit,
    )
    try:
        if contract_version == 1:
            readback = _readback_controlled_distance_stage_v1(
                stage,
                joint_path=joint_path,
                expected_body0_prim_path=body0.body_prim_path,
                expected_body1_prim_path=body1.body_prim_path,
                require_uncomposed_endpoints=False,
            )
        else:
            readback = _readback_controlled_distance_stage_v2(
                stage,
                joint_path=joint_path,
                expected_body0_prim_path=body0.body_prim_path,
                expected_body1_prim_path=body1.body_prim_path,
                require_uncomposed_endpoints=False,
            )
    except ControlledDistanceError as exc:
        if exc.code.startswith("controlled_distance_source_"):
            _fail(
                exc.code.replace(
                    "controlled_distance_source_",
                    "controlled_distance_saved_",
                    1,
                ),
                f"saved target readback failed: {exc}",
            )
        raise
    if readback.attachments != expected_attachments:
        _fail(
            "controlled_distance_saved_attachment_conflict",
            "saved controlled-distance attachment readback changed",
        )
    if not _vectors_close(readback.axis_stage, controlled.axis_stage):
        _fail(
            "controlled_distance_saved_axis_conflict",
            "saved controlled-distance stage axis changed",
        )
    if (
        readback.axial_interval != expected_interval
        or readback.drive != expected_drive
        or readback.state != expected_state
    ):
        _fail(
            "controlled_distance_saved_semantics_conflict",
            "saved controlled-distance interval, drive, or state changed",
        )
    return readback


def readback_controlled_distance_stage_v1(
    stage: Any,
    *,
    joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
) -> ControlledDistanceStageReadbackV1:
    """Reconstruct exact controlled-distance values from one source stage."""

    return _readback_controlled_distance_stage_v1(
        stage,
        joint_path=joint_path,
        expected_body0_prim_path=expected_body0_prim_path,
        expected_body1_prim_path=expected_body1_prim_path,
        require_uncomposed_endpoints=True,
    )


def readback_controlled_distance_stage_v2(
    stage: Any,
    *,
    joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
) -> ControlledDistanceStageReadbackV2:
    """Reconstruct v2 values using positive body1-to-body0 state."""

    return _readback_controlled_distance_stage_v2(
        stage,
        joint_path=joint_path,
        expected_body0_prim_path=expected_body0_prim_path,
        expected_body1_prim_path=expected_body1_prim_path,
        require_uncomposed_endpoints=True,
    )


def _readback_controlled_distance_stage_v1(
    stage: Any,
    *,
    joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    require_uncomposed_endpoints: bool,
) -> ControlledDistanceStageReadbackV1:
    return _readback_controlled_distance_stage(
        stage,
        joint_path=joint_path,
        expected_body0_prim_path=expected_body0_prim_path,
        expected_body1_prim_path=expected_body1_prim_path,
        require_uncomposed_endpoints=require_uncomposed_endpoints,
        contract_version=1,
    )


def _readback_controlled_distance_stage_v2(
    stage: Any,
    *,
    joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    require_uncomposed_endpoints: bool,
) -> ControlledDistanceStageReadbackV2:
    return cast(
        ControlledDistanceStageReadbackV2,
        _readback_controlled_distance_stage(
            stage,
            joint_path=joint_path,
            expected_body0_prim_path=expected_body0_prim_path,
            expected_body1_prim_path=expected_body1_prim_path,
            require_uncomposed_endpoints=require_uncomposed_endpoints,
            contract_version=2,
        ),
    )


def _readback_controlled_distance_stage(
    stage: Any,
    *,
    joint_path: str,
    expected_body0_prim_path: str,
    expected_body1_prim_path: str,
    require_uncomposed_endpoints: bool,
    contract_version: Literal[1, 2],
) -> ControlledDistanceStageReadbackV1 | ControlledDistanceStageReadbackV2:
    """Reconstruct exact source or authored controlled-distance values."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    _require_prim_path(joint_path, label="controlled-distance joint path")
    _require_prim_path(expected_body0_prim_path, label="expected body0 path")
    _require_prim_path(expected_body1_prim_path, label="expected body1 path")
    prim = stage.GetPrimAtPath(joint_path)
    if (
        not prim.IsValid()
        or not _is_concrete_active_prim(prim)
        or prim.IsInstanceProxy()
        or prim.GetTypeName() != "PhysicsPrismaticJoint"
        or not prim.IsA(UsdPhysics.PrismaticJoint)
    ):
        _fail(
            "controlled_distance_source_schema_conflict",
            "controlled-distance source must be an exact PhysicsPrismaticJoint",
        )
    _require_no_composition_arcs(prim)
    applied_schemas = _applied_schemas(prim)
    if applied_schemas != _EXPECTED_APPLIED_SCHEMAS:
        _fail(
            "controlled_distance_source_applied_schema_conflict",
            "controlled-distance source requires exactly linear drive and state APIs",
        )
    metadata = prim.GetAllAuthoredMetadata()
    if (
        frozenset(metadata) != _EXPECTED_AUTHORED_METADATA
        or metadata["specifier"] != Sdf.SpecifierDef
        or str(metadata["typeName"]) != "PhysicsPrismaticJoint"
    ):
        _fail(
            "controlled_distance_source_metadata_conflict",
            "controlled-distance source prim metadata closure differs",
        )
    authored_properties = frozenset(
        str(item.GetName()) for item in prim.GetAuthoredProperties()
    )
    if authored_properties != _EXPECTED_AUTHORED_PROPERTIES:
        missing = sorted(_EXPECTED_AUTHORED_PROPERTIES - authored_properties)
        extra = sorted(authored_properties - _EXPECTED_AUTHORED_PROPERTIES)
        _fail(
            "controlled_distance_source_property_conflict",
            f"controlled-distance property closure differs; missing={missing}; extra={extra}",
        )
    _require_property_type_closure(prim)

    body0_path = _single_target(prim, "physics:body0")
    body1_path = _single_target(prim, "physics:body1")
    if (
        body0_path != expected_body0_prim_path
        or body1_path != expected_body1_prim_path
        or body0_path == body1_path
    ):
        _fail(
            "controlled_distance_source_endpoint_conflict",
            "controlled-distance endpoints do not match the exact directed pair",
        )
    body0 = _require_endpoint_rigid_body(stage, body0_path, UsdPhysics=UsdPhysics)
    body1 = _require_endpoint_rigid_body(stage, body1_path, UsdPhysics=UsdPhysics)
    if require_uncomposed_endpoints:
        _require_no_composition_arcs(body0)
        _require_no_composition_arcs(body1)
    _require_static_translation_endpoint(body0, UsdGeom=UsdGeom)
    _require_static_translation_endpoint(body1, UsdGeom=UsdGeom)
    meters_per_unit, kilograms_per_unit = _stage_units(stage)

    axis = str(_explicit_attr(prim, "physics:axis"))
    if axis != "X":
        _fail(
            "controlled_distance_source_axis_conflict",
            "controlled-distance prismatic axis must be canonical X",
        )
    local_pos0 = _vec3(_explicit_attr(prim, "physics:localPos0"))
    local_pos1 = _vec3(_explicit_attr(prim, "physics:localPos1"))
    local_rot0 = _quat(_explicit_attr(prim, "physics:localRot0"))
    local_rot1 = _quat(_explicit_attr(prim, "physics:localRot1"))
    attachments = ExplicitAttachmentFramesV2(
        kind="explicit",
        body0=AttachmentFrameV2(
            position_meters=tuple(
                _finite_unit_product(
                    value,
                    meters_per_unit,
                    label="body0 attachment position",
                )
                for value in local_pos0
            ),
            orientation_wxyz=local_rot0,
        ),
        body1=AttachmentFrameV2(
            position_meters=tuple(
                _finite_unit_product(
                    value,
                    meters_per_unit,
                    label="body1 attachment position",
                )
                for value in local_pos1
            ),
            orientation_wxyz=local_rot1,
        ),
    )
    axis_stage, separation_meters = _aligned_positive_axis_and_separation(
        body0,
        body1,
        local_pos0=local_pos0,
        local_pos1=local_pos1,
        local_rot0=local_rot0,
        local_rot1=local_rot1,
        meters_per_unit=meters_per_unit,
        Gf=Gf,
        UsdGeom=UsdGeom,
        contract_version=contract_version,
    )

    minimum = _finite_unit_product(
        _float_attr(prim, "physics:lowerLimit"),
        meters_per_unit,
        label="lower limit",
    )
    maximum = _finite_unit_product(
        _float_attr(prim, "physics:upperLimit"),
        meters_per_unit,
        label="upper limit",
    )
    if minimum <= 0.0 or minimum >= maximum:
        _fail(
            "controlled_distance_invalid_axial_interval",
            "source requires 0 < lowerLimit < upperLimit",
        )
    interval = DistanceConstraintV2(
        kind="distance",
        minimum_meters=minimum,
        maximum_meters=maximum,
    )
    drive_type = str(_explicit_attr(prim, "drive:linear:physics:type"))
    target_position = _finite_unit_product(
        _float_attr(prim, "drive:linear:physics:targetPosition"),
        meters_per_unit,
        label="drive target position",
    )
    target_velocity = _finite_unit_product(
        _float_attr(prim, "drive:linear:physics:targetVelocity"),
        meters_per_unit,
        label="drive target velocity",
    )
    stiffness = _finite_unit_product(
        _float_attr(prim, "drive:linear:physics:stiffness"),
        kilograms_per_unit,
        label="drive stiffness",
    )
    damping = _finite_unit_product(
        _float_attr(prim, "drive:linear:physics:damping"),
        kilograms_per_unit,
        label="drive damping",
    )
    force_units = kilograms_per_unit * meters_per_unit
    maximum_force = _finite_unit_product(
        _float_attr(prim, "drive:linear:physics:maxForce"),
        force_units,
        label="drive maximum force",
    )
    state_position = _finite_unit_product(
        _float_attr(prim, "state:linear:physics:position"),
        meters_per_unit,
        label="state position",
    )
    state_velocity = _finite_unit_product(
        _float_attr(prim, "state:linear:physics:velocity"),
        meters_per_unit,
        label="state velocity",
    )
    if drive_type != "force":
        _fail(
            "controlled_distance_source_drive_type_conflict",
            "source linear drive must use force mode",
        )
    if not minimum < target_position < maximum:
        _fail(
            "controlled_distance_target_outside_axial_interval",
            "source drive target must be strictly inside the interval",
        )
    if not minimum < state_position < maximum:
        _fail(
            "controlled_distance_state_outside_axial_interval",
            "source state position must be strictly inside the interval",
        )
    if target_velocity != 0.0:
        _fail(
            "controlled_distance_nonzero_target_velocity",
            "source target velocity must be exactly zero",
        )
    if state_velocity != 0.0:
        _fail(
            "controlled_distance_nonzero_state_velocity",
            "source state velocity must be exactly zero",
        )
    if stiffness <= 0.0:
        _fail(
            "controlled_distance_non_positive_stiffness",
            "source drive stiffness must be positive",
        )
    if damping <= 0.0:
        _fail(
            "controlled_distance_non_positive_damping",
            "source drive damping must be positive",
        )
    if maximum_force <= 0.0:
        _fail(
            "controlled_distance_non_positive_maximum_force",
            "source drive maximum force must be positive",
        )
    drive = DistanceLinearDriveV1(
        target_position_meters=target_position,
        target_velocity_meters_per_second=target_velocity,
        stiffness_newtons_per_meter=stiffness,
        damping_newton_seconds_per_meter=damping,
        maximum_force_newtons=maximum_force,
        drive_type=drive_type,
    )
    state = DistanceLinearStateV1(
        position_meters=state_position,
        velocity_meters_per_second=state_velocity,
    )
    if not _scalars_close(
        state.position_meters,
        separation_meters,
    ):
        _fail(
            "controlled_distance_source_state_frame_conflict",
            "linear state position does not match positive frame separation",
        )
    readback_type = (
        ControlledDistanceStageReadbackV1
        if contract_version == 1
        else ControlledDistanceStageReadbackV2
    )
    return readback_type(
        joint_path=joint_path,
        body0_prim_path=body0_path,
        body1_prim_path=body1_path,
        attachments=attachments,
        axis_stage=axis_stage,
        axial_interval=interval,
        drive=drive,
        state=state,
        meters_per_unit=meters_per_unit,
        kilograms_per_unit=kilograms_per_unit,
    )


def _require_single_source_joint(stage: Any, *, source_joint_path: str) -> None:
    from pxr import UsdPhysics

    _require_prim_path(source_joint_path, label="source joint path")
    if stage.GetPrototypes():
        _fail(
            "controlled_distance_source_mixed_assembly",
            "controlled-distance source cannot contain instance prototypes",
        )
    joint_paths = tuple(
        str(prim.GetPath())
        for prim in stage.TraverseAll()
        if prim.IsA(UsdPhysics.Joint)
    )
    if joint_paths != (source_joint_path,):
        _fail(
            "controlled_distance_source_mixed_assembly",
            "controlled-distance source must contain exactly the selected joint",
        )


def _require_empty_target_joint_graph(stage: Any) -> None:
    from pxr import UsdPhysics

    if stage.GetPrototypes() or any(
        prim.IsA(UsdPhysics.Joint) for prim in stage.TraverseAll()
    ):
        _fail(
            "controlled_distance_target_joint_graph_conflict",
            "controlled-distance target must have an empty composed joint graph",
        )


def _backup_edit_layer(stage: Any, *, joint_path: Any) -> tuple[Any, Any]:
    """Return an editable identity-mapped layer and detached content backup."""

    try:
        from pxr import Sdf

        edit_target = stage.GetEditTarget()
        layer = edit_target.GetLayer()
        if (
            layer is None
            or not layer.permissionToEdit
            or edit_target.MapToSpecPath(joint_path) != joint_path
        ):
            _fail(
                "controlled_distance_authoring_edit_layer_unavailable",
                "the authoring edit layer must be editable and identity-mapped",
            )
        backup = Sdf.Layer.CreateAnonymous("controlled-distance-rollback.usda")
        backup.TransferContent(layer)
        return layer, backup
    except ControlledDistanceError:
        raise
    except Exception as exc:
        _fail(
            "controlled_distance_authoring_snapshot_failed",
            f"could not snapshot authoring edit layer: {type(exc).__name__}",
        )


def _restore_edit_layer(layer: Any, backup: Any) -> None:
    """Restore every edit-layer opinion and verify exact layer content."""

    expected = backup.ExportToString()
    layer.TransferContent(backup)
    if not expected or layer.ExportToString() != expected:
        raise RuntimeError("restored edit-layer content differs from its backup")


def _require_exact_saved_joint_graph(stage: Any, *, joint_path: str) -> None:
    from pxr import UsdPhysics

    _require_prim_path(joint_path, label="saved controlled-distance joint path")
    joint_paths = tuple(
        str(prim.GetPath())
        for prim in stage.TraverseAll()
        if prim.IsA(UsdPhysics.Joint)
    )
    if stage.GetPrototypes() or joint_paths != (joint_path,):
        _fail(
            "controlled_distance_saved_joint_graph_conflict",
            "saved controlled-distance stage must contain exactly the authored joint",
        )


def _require_static_translation_endpoint(prim: Any, *, UsdGeom: Any) -> None:
    from pxr import Sdf

    endpoint = UsdGeom.Xformable(prim)
    order_attribute = prim.GetAttribute("xformOpOrder")
    translation_attribute = prim.GetAttribute("xformOp:translate")
    raw_order = tuple(str(item) for item in _explicit_attr(prim, "xformOpOrder"))
    if (
        raw_order != ("xformOp:translate",)
        or order_attribute.GetAllAuthoredMetadata()
        != {
            "custom": False,
            "typeName": Sdf.ValueTypeNames.TokenArray,
            "variability": Sdf.VariabilityUniform,
        }
        or translation_attribute.GetAllAuthoredMetadata()
        != {
            "custom": False,
            "typeName": Sdf.ValueTypeNames.Double3,
            "variability": Sdf.VariabilityVarying,
        }
    ):
        _fail(
            "controlled_distance_endpoint_transform_conflict",
            "endpoint transforms require an exact single translation property closure",
        )
    _vec3(_explicit_attr(prim, "xformOp:translate"))
    operations = endpoint.GetOrderedXformOps()
    resets = endpoint.GetResetXformStack()
    if (  # pragma: no cover - exact raw transform closure dominates this check
        resets
        or len(operations) != 1
        or operations[0].GetOpType() != UsdGeom.XformOp.TypeTranslate
        or operations[0].GetPrecision() != UsdGeom.XformOp.PrecisionDouble
        or operations[0].IsInverseOp()
        or str(operations[0].GetOpName()) != "xformOp:translate"
    ):
        _fail(
            "controlled_distance_endpoint_transform_conflict",
            "endpoint transforms require one static double-precision translation",
        )
    parent = prim.GetParent()
    while parent and not parent.IsPseudoRoot():
        xformable = UsdGeom.Xformable(parent)
        if xformable:
            parent_operations = xformable.GetOrderedXformOps()
            parent_resets = xformable.GetResetXformStack()
            if parent_operations or parent_resets:
                _fail(
                    "controlled_distance_endpoint_transform_conflict",
                    "endpoint ancestors cannot contribute transforms",
                )
        parent = parent.GetParent()


def _storage_projected_semantics(
    controlled: ControlledDistanceContractV1,
    *,
    attachments: ExplicitAttachmentFramesV2,
    meters_per_unit: float,
    kilograms_per_unit: float,
) -> tuple[
    ExplicitAttachmentFramesV2,
    DistanceConstraintV2,
    DistanceLinearDriveV1,
    DistanceLinearStateV1,
]:
    minimum = cast(float, controlled.axial_interval.minimum_meters)
    maximum = cast(float, controlled.axial_interval.maximum_meters)
    expected_attachments = ExplicitAttachmentFramesV2(
        kind="explicit",
        body0=AttachmentFrameV2(
            position_meters=tuple(
                _project_float32(item, meters_per_unit)
                for item in attachments.body0.position_meters
            ),
            orientation_wxyz=tuple(
                _project_float32(item, 1.0)
                for item in attachments.body0.orientation_wxyz
            ),
        ),
        body1=AttachmentFrameV2(
            position_meters=tuple(
                _project_float32(item, meters_per_unit)
                for item in attachments.body1.position_meters
            ),
            orientation_wxyz=tuple(
                _project_float32(item, 1.0)
                for item in attachments.body1.orientation_wxyz
            ),
        ),
    )
    projected_minimum = _project_float32(minimum, meters_per_unit)
    projected_maximum = _project_float32(maximum, meters_per_unit)
    projected_target = _project_float32(
        controlled.drive.target_position_meters,
        meters_per_unit,
    )
    projected_state = _project_float32(
        controlled.state.position_meters,
        meters_per_unit,
    )
    projected_stiffness = _project_float32(
        controlled.drive.stiffness_newtons_per_meter,
        kilograms_per_unit,
    )
    projected_damping = _project_float32(
        controlled.drive.damping_newton_seconds_per_meter,
        kilograms_per_unit,
    )
    force_units = kilograms_per_unit * meters_per_unit
    projected_force = _project_float32(
        controlled.drive.maximum_force_newtons,
        force_units,
    )
    original_values = (
        minimum,
        maximum,
        controlled.drive.target_position_meters,
        controlled.state.position_meters,
        controlled.drive.stiffness_newtons_per_meter,
        controlled.drive.damping_newton_seconds_per_meter,
        controlled.drive.maximum_force_newtons,
        *attachments.body0.position_meters,
        *attachments.body0.orientation_wxyz,
        *attachments.body1.position_meters,
        *attachments.body1.orientation_wxyz,
    )
    projected_values = (
        projected_minimum,
        projected_maximum,
        projected_target,
        projected_state,
        projected_stiffness,
        projected_damping,
        projected_force,
        *expected_attachments.body0.position_meters,
        *expected_attachments.body0.orientation_wxyz,
        *expected_attachments.body1.position_meters,
        *expected_attachments.body1.orientation_wxyz,
    )
    if any(
        not _scalars_close(original, projected)
        for original, projected in zip(
            original_values,
            projected_values,
            strict=True,
        )
    ):
        _fail(
            "controlled_distance_authoring_precision_loss",
            "target units cannot preserve controlled-distance values in binary32",
        )
    if not (
        0.0 < projected_minimum < projected_maximum
        and projected_minimum < projected_target < projected_maximum
        and projected_minimum < projected_state < projected_maximum
        and projected_stiffness > 0.0
        and projected_damping > 0.0
        and projected_force > 0.0
    ):
        _fail(
            "controlled_distance_authoring_precision_loss",
            "binary32 projection changes controlled-distance ordering or activity",
        )
    try:
        return (
            expected_attachments,
            DistanceConstraintV2(
                kind="distance",
                minimum_meters=projected_minimum,
                maximum_meters=projected_maximum,
            ),
            DistanceLinearDriveV1(
                target_position_meters=projected_target,
                target_velocity_meters_per_second=0.0,
                stiffness_newtons_per_meter=projected_stiffness,
                damping_newton_seconds_per_meter=projected_damping,
                maximum_force_newtons=projected_force,
            ),
            DistanceLinearStateV1(
                position_meters=projected_state,
                velocity_meters_per_second=0.0,
            ),
        )
    except (
        ValidationError
    ) as exc:  # pragma: no cover - preconditions above mirror models
        _fail(
            "controlled_distance_authoring_precision_loss",
            f"binary32 projection is not representable: {exc.errors()[0]['type']}",
        )


def _project_float32(value: float, units: float) -> float:
    try:
        projected = struct.unpack("!f", struct.pack("!f", value / units))[0] * units
    except (OverflowError, struct.error) as exc:
        _fail(
            "controlled_distance_authoring_precision_loss",
            f"value cannot be represented as binary32: {type(exc).__name__}",
        )
    if not math.isfinite(projected):
        _fail(
            "controlled_distance_authoring_precision_loss",
            "value cannot be represented as finite binary32",
        )
    return 0.0 if projected == 0.0 else projected


def _scalars_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_READBACK_RELATIVE_TOLERANCE,
        abs_tol=_FRAME_TOLERANCE,
    )


def _require_prim_path(value: str, *, label: str) -> Any:
    from pxr import Sdf

    try:
        path = Sdf.Path(value)
    except Exception as exc:
        _fail(
            "controlled_distance_invalid_prim_path",
            f"{label} is invalid: {type(exc).__name__}",
        )
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path.IsAbsoluteRootPath():
        _fail(
            "controlled_distance_invalid_prim_path",
            f"{label} must be one non-root absolute prim path",
        )
    return path


def _stage_units(stage: Any) -> tuple[float, float]:
    from pxr import UsdGeom, UsdPhysics

    if not stage.HasAuthoredMetadata("metersPerUnit") or not stage.HasAuthoredMetadata(
        "kilogramsPerUnit"
    ):
        _fail(
            "controlled_distance_stage_units_unresolved",
            "controlled-distance stage requires explicit meter and kilogram units",
        )
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if (
        not math.isfinite(meters)
        or meters <= 0.0
        or not math.isfinite(kilograms)
        or kilograms <= 0.0
    ):
        _fail(
            "controlled_distance_stage_units_invalid",
            "controlled-distance stage units must be finite and positive",
        )
    return meters, kilograms


def _require_endpoint_rigid_body(stage: Any, path: str, *, UsdPhysics: Any) -> Any:
    prim = stage.GetPrimAtPath(path)
    if (
        not prim.IsValid()
        or not _is_concrete_active_prim(prim)
        or prim.IsInstanceProxy()
        or not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        _fail(
            "controlled_distance_endpoint_invalid",
            f"controlled-distance endpoint is not an exact rigid body: {path}",
        )
    enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr()
    if (
        not enabled
        or enabled.GetTimeSamples()
        or enabled.GetConnections()
        or enabled.Get() is not True
    ):
        _fail(
            "controlled_distance_endpoint_invalid",
            f"controlled-distance endpoint is not statically enabled: {path}",
        )
    return prim


def _is_concrete_active_prim(prim: Any) -> bool:
    return bool(
        prim.IsActive()
        and prim.IsDefined()
        and not prim.IsAbstract()
        and not prim.IsPrototype()
        and not prim.IsInPrototype()
        and not prim.IsInstance()
        and not prim.IsInstanceProxy()
    )


def _require_no_composition_arcs(prim: Any) -> None:
    from pxr import Pcp

    root = prim.GetPrimIndex().rootNode
    pending = [root]
    while pending:
        node = pending.pop()
        if node.arcType != Pcp.ArcTypeRoot:
            _fail(
                "controlled_distance_source_composition_arc_conflict",
                "controlled-distance source joint and endpoints cannot use composition arcs",
            )
        pending.extend(node.children)
    for spec in prim.GetPrimStack():
        if (
            spec.hasReferences
            or spec.hasPayloads
            or spec.hasInheritPaths
            or spec.hasSpecializes
            or len(spec.variantSelections) != 0
            or len(spec.variantSets) != 0
        ):  # pragma: no cover - prim-index traversal dominates authored arcs
            _fail(
                "controlled_distance_source_composition_arc_conflict",
                "controlled-distance source joint and endpoints cannot author composition arcs",
            )


def _applied_schemas(prim: Any) -> tuple[str, ...]:
    list_op = prim.GetMetadata("apiSchemas")
    if list_op is None:
        return ()
    explicit = tuple(str(item) for item in list_op.explicitItems)
    if (
        explicit != _EXPECTED_APPLIED_SCHEMAS
        or list_op.prependedItems
        or list_op.appendedItems
        or list_op.deletedItems
        or list_op.orderedItems
    ):
        return ()
    return explicit


def _require_property_type_closure(prim: Any) -> None:
    from pxr import Sdf

    for name, expected_type in _EXPECTED_ATTRIBUTE_TYPES.items():
        attribute = prim.GetAttribute(name)
        if (  # pragma: no cover - exact typed Prismatic schema invariant
            not attribute
            or attribute.IsCustom()
            or str(attribute.GetTypeName()) != expected_type
        ):
            _fail(
                "controlled_distance_source_property_type_conflict",
                f"{name} must be one exact non-custom {expected_type} attribute",
            )
        expected_metadata = {
            "custom": False,
            "typeName": attribute.GetTypeName(),
            "variability": (
                Sdf.VariabilityUniform
                if name in _EXPECTED_UNIFORM_PROPERTIES
                else Sdf.VariabilityVarying
            ),
        }
        if attribute.GetAllAuthoredMetadata() != expected_metadata:
            _fail(
                "controlled_distance_source_property_metadata_conflict",
                f"{name} has unsupported authored metadata",
            )
    for name in ("physics:body0", "physics:body1"):
        relationship = prim.GetRelationship(name)
        if (  # pragma: no cover - exact typed Prismatic schema invariant
            not relationship or relationship.IsCustom()
        ):
            _fail(
                "controlled_distance_source_property_type_conflict",
                f"{name} must be one exact non-custom relationship",
            )
        if relationship.GetAllAuthoredMetadata() != {
            "custom": False,
            "variability": Sdf.VariabilityUniform,
        }:
            _fail(
                "controlled_distance_source_property_metadata_conflict",
                f"{name} has unsupported authored metadata",
            )


def _single_target(prim: Any, name: str) -> str:
    relationship = prim.GetRelationship(name)
    if not relationship:  # pragma: no cover - property closure checked first
        _fail(
            "controlled_distance_source_relationship_conflict",
            f"{name} must have one exact target",
        )
    targets = tuple(str(item) for item in relationship.GetTargets())
    if len(targets) != 1:
        _fail(
            "controlled_distance_source_relationship_conflict",
            f"{name} must have one exact target",
        )
    return targets[0]


def _explicit_attr(prim: Any, name: str) -> Any:
    from pxr import Usd

    attribute = prim.GetAttribute(name)
    if (
        not attribute
        or not attribute.HasAuthoredValueOpinion()
        or attribute.GetTimeSamples()
        or attribute.GetConnections()
        or attribute.GetResolveInfo(Usd.TimeCode.Default()).ValueIsBlocked()
    ):
        _fail(
            "controlled_distance_source_attribute_conflict",
            f"{name} requires one unconnected static authored default",
        )
    value = attribute.Get(Usd.TimeCode.Default())
    if value is None:  # pragma: no cover - authored unblocked default invariant
        _fail(
            "controlled_distance_source_attribute_conflict",
            f"{name} has no authored default value",
        )
    return value


def _float_attr(prim: Any, name: str) -> float:
    value = float(_explicit_attr(prim, name))
    if not math.isfinite(value):
        _fail(
            "controlled_distance_source_non_finite_value",
            f"{name} must be finite",
        )
    return 0.0 if value == 0.0 else value


def _finite_unit_product(value: float, *units: float, label: str) -> float:
    result = value
    for unit in units:
        result *= unit
    if not math.isfinite(result):
        _fail(
            "controlled_distance_source_unit_conversion_invalid",
            f"{label} is non-finite after stage-unit conversion",
        )
    return 0.0 if result == 0.0 else result


def _vec3(value: Any) -> Vector3:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or any(not math.isfinite(item) for item in result):
        _fail(
            "controlled_distance_source_frame_invalid",
            "controlled-distance frame position must contain three finite values",
        )
    return result


def _quat(value: Any) -> QuaternionWxyz:
    imaginary = value.GetImaginary()
    result = (
        float(value.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )
    norm = math.sqrt(sum(item * item for item in result))
    if any(not math.isfinite(item) for item in result) or not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_FRAME_TOLERANCE,
    ):
        _fail(
            "controlled_distance_source_frame_invalid",
            "controlled-distance frame orientation must be finite and normalized",
        )
    return result


def _aligned_positive_axis_and_separation(
    body0: Any,
    body1: Any,
    *,
    local_pos0: Vector3,
    local_pos1: Vector3,
    local_rot0: QuaternionWxyz,
    local_rot1: QuaternionWxyz,
    meters_per_unit: float,
    Gf: Any,
    UsdGeom: Any,
    contract_version: Literal[1, 2],
) -> tuple[Vector3, float]:
    cache = UsdGeom.XformCache()
    world0 = cache.GetLocalToWorldTransform(body0)
    world1 = cache.GetLocalToWorldTransform(body1)
    point0 = world0.Transform(Gf.Vec3d(*local_pos0))
    point1 = world1.Transform(Gf.Vec3d(*local_pos1))
    basis0 = _transformed_basis(world0, local_rot0, Gf=Gf)
    basis1 = _transformed_basis(world1, local_rot1, Gf=Gf)
    axis0 = basis0[0]
    axis1 = basis1[0]
    if not _vectors_close(axis0, axis1):
        _fail(
            "controlled_distance_source_axis_frame_conflict",
            "controlled-distance endpoint +X frame axes must agree",
        )
    if not _vectors_close(basis0[1], basis1[1]) or not _vectors_close(
        basis0[2], basis1[2]
    ):
        _fail(
            "controlled_distance_source_basis_frame_conflict",
            "controlled-distance endpoint frame bases must agree",
        )
    if contract_version == 1:
        delta = tuple(float(point1[index] - point0[index]) for index in range(3))
    else:
        delta = tuple(float(point0[index] - point1[index]) for index in range(3))
    projection_stage = sum(delta[index] * axis0[index] for index in range(3))
    transverse = tuple(
        delta[index] - projection_stage * axis0[index] for index in range(3)
    )
    projection_meters = projection_stage * meters_per_unit
    transverse_meters = math.hypot(*(item * meters_per_unit for item in transverse))
    if (
        not math.isfinite(projection_meters)
        or not math.isfinite(transverse_meters)
        or projection_meters <= 0.0
        or transverse_meters > _FRAME_TOLERANCE
    ):
        detail = (
            "frame separation must lie strictly on the shared positive X axis"
            if contract_version == 1
            else "frame separation must lie strictly on the shared positive X axis "
            "from the body1 attachment toward the body0 attachment"
        )
        _fail(
            "controlled_distance_source_positive_axis_conflict",
            detail,
        )
    return axis0, projection_meters


def _transformed_basis(
    world: Any,
    orientation: QuaternionWxyz,
    *,
    Gf: Any,
) -> tuple[Vector3, Vector3, Vector3]:
    rotation = Gf.Rotation(_quatd(orientation, Gf=Gf))
    return (
        _normalized(world.TransformDir(rotation.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))),
        _normalized(world.TransformDir(rotation.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)))),
        _normalized(world.TransformDir(rotation.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)))),
    )


def _normalized(value: Any) -> Vector3:
    result = tuple(float(item) for item in value)
    norm = math.sqrt(sum(item * item for item in result))
    if (  # pragma: no cover - translation-only endpoints and unit quaternions
        len(result) != 3 or not math.isfinite(norm) or norm <= 0.0
    ):
        _fail(
            "controlled_distance_source_axis_invalid",
            "controlled-distance stage axis must be finite and nonzero",
        )
    return cast(Vector3, tuple(item / norm for item in result))


def _vectors_close(left: Vector3, right: Vector3) -> bool:
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=_FRAME_TOLERANCE)
        for a, b in zip(left, right, strict=True)
    )


def _quatd(value: QuaternionWxyz, *, Gf: Any) -> Any:
    return Gf.Quatd(value[0], Gf.Vec3d(value[1], value[2], value[3]))


def _quatf(value: QuaternionWxyz, *, Gf: Any) -> Any:
    return Gf.Quatf(value[0], Gf.Vec3f(value[1], value[2], value[3]))


def _field_evidence(
    *,
    source_artifact: ArtifactIdentityV1,
    source_joint_path: str,
    provenance_source: ProvenanceSource,
    properties_by_field: Mapping[str, tuple[str, ...]],
    body0_prim_path: str,
    body1_prim_path: str,
) -> tuple[FieldEvidenceV2, ...]:
    return tuple(
        FieldEvidenceV2(
            field=field,
            provenance=FieldProvenanceV1(
                source=provenance_source,
                artifact=source_artifact,
                prim_path=(
                    body0_prim_path
                    if field == "endpoint_frame.body0_world_transform"
                    else body1_prim_path
                    if field == "endpoint_frame.body1_world_transform"
                    else source_joint_path
                ),
                properties=properties,
                evidence=f"exact source-backed controlled-distance {field}",
            ),
        )
        for field, properties in sorted(properties_by_field.items())
    )


def _fail(code: str, detail: str) -> NoReturn:
    raise ControlledDistanceError(code, detail)


__all__ = [
    "ControlledDistanceStageReadbackV1",
    "ControlledDistanceStageReadbackV2",
    "ControlledDistanceV2SourceReadbackProtocol",
    "SourceBackedControlledDistanceV1",
    "SourceBackedControlledDistanceV2",
    "author_controlled_distance_stage_v1",
    "author_controlled_distance_stage_v2",
    "controlled_distance_contract_from_source_v1",
    "controlled_distance_contract_from_source_v2",
    "controlled_distance_joint_record_v1",
    "controlled_distance_joint_record_v2",
    "extract_controlled_distance_source_from_retained_usd_v1",
    "extract_controlled_distance_source_from_retained_usd_v2",
    "readback_controlled_distance_stage_v1",
    "readback_controlled_distance_stage_v2",
    "require_controlled_distance_stage_matches_v1",
    "require_controlled_distance_stage_matches_v2",
]
