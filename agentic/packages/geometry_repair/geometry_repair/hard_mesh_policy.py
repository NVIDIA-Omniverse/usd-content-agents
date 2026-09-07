# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed protocol and route policy for hard local mesh workers."""

from __future__ import annotations

import hmac
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import atomic_write_json, file_sha256
from .workers.base import WorkerResult

HARD_MESH_CAPABILITIES_SCHEMA = "geometry-repair.hard-mesh-capabilities.v1"
HARD_MESH_PROTOCOL = "geometry-repair.hard-mesh-json.v1"
HARD_MESH_REQUEST_SCHEMA = "geometry-repair.hard-mesh-request.v1"
HARD_MESH_REPORT_SCHEMA = "geometry-repair.hard-mesh-report.v1"

PMP_BUILD_ID = "pmp-library:2a2ad502743724ba90e09af816364c84032f9015"
PMP_SOURCE_COMMIT = "2a2ad502743724ba90e09af816364c84032f9015"
PMP_SOURCE_TREE_SHA256 = "ec134774c578b2ddd97b620daaa73ee1839ffb78b9d3355050443d8bdb6e5508"
PMP_EXECUTABLE_SHA256_ENVIRONMENT_VARIABLE = "GEOMETRY_REPAIR_PMP_EXECUTABLE_SHA256"
WILDMESHING_BUILD_ID = "wildmeshing-toolkit:34fdc3441fc6439a5f5c38b505cf17ad37b1925d"
MCUT_BUILD_ID = "mcut:d424aec52454c22cd9d436907f28db3595e706e1"

_MAX_CAPABILITY_BYTES = 64 * 1024
_MAX_REPORT_BYTES = 1024 * 1024


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HardMeshExecutableSpec(_StrictModel):
    """Pinned executable identity and required protocol surface."""

    worker: str = Field(min_length=1)
    environment_variable: str = Field(min_length=1)
    executable_names: tuple[str, ...] = Field(min_length=1)
    build_id: str = Field(min_length=1)
    operations: frozenset[str] = Field(min_length=1)
    required_capabilities: frozenset[str] = Field(min_length=1)
    shadow_only: bool = False


class HardMeshCapabilities(_StrictModel):
    """Machine-readable identity emitted by an audited native adapter."""

    schema_version: Literal["geometry-repair.hard-mesh-capabilities.v1"] = (
        HARD_MESH_CAPABILITIES_SCHEMA
    )
    protocol_version: Literal["geometry-repair.hard-mesh-json.v1"] = HARD_MESH_PROTOCOL
    worker: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    build_id: str = Field(min_length=1)
    operations: list[str] = Field(min_length=1)
    capabilities: list[str] = Field(min_length=1)
    deterministic: bool
    backend_source_commit: str | None = None
    backend_source_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    executable_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _reject_duplicates(self) -> HardMeshCapabilities:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("capability operations must not contain duplicates")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("capabilities must not contain duplicates")
        return self


class HardMeshRequest(_StrictModel):
    """Fixed JSON request passed to a native helper without a shell."""

    schema_version: Literal["geometry-repair.hard-mesh-request.v1"] = HARD_MESH_REQUEST_SCHEMA
    protocol_version: Literal["geometry-repair.hard-mesh-json.v1"] = HARD_MESH_PROTOCOL
    request_id: str = Field(min_length=1, max_length=256)
    worker: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    implementation_build_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str = Field(min_length=1)
    deterministic_seed: int = Field(ge=0, le=2**31 - 1)
    parameters: dict[str, Any]


class HardMeshFragment(_StrictModel):
    """One unselected MCUT fragment returned for later classification."""

    fragment_id: str = Field(min_length=1, max_length=256)
    source_part_path: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    face_count: int = Field(ge=1)


class HardMeshIntersectionCurve(_StrictModel):
    """One MCUT intersection-curve inventory record."""

    curve_id: str = Field(min_length=1, max_length=256)
    point_count: int = Field(ge=2)
    closed: bool


class HardMeshNativeReport(_StrictModel):
    """Strict result document produced by the native executable."""

    schema_version: Literal["geometry-repair.hard-mesh-report.v1"] = HARD_MESH_REPORT_SCHEMA
    protocol_version: Literal["geometry-repair.hard-mesh-json.v1"] = HARD_MESH_PROTOCOL
    status: Literal["success", "refused", "failed"]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    implementation_version: str = Field(min_length=1)
    build_id: str = Field(min_length=1)
    backend_source_commit: str | None = None
    backend_source_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    executable_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_path: str | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    changed: bool = False
    changed_face_ids: list[int] = Field(default_factory=list, max_length=2_000_000)
    generated_face_ids: list[int] = Field(default_factory=list, max_length=2_000_000)
    changed_region_path: str | None = None
    changed_region_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    correspondence_path: str | None = None
    correspondence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    correspondence_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    attribute_transfer_path: str | None = None
    attribute_transfer_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preserved_frozen_vertices: bool | None = None
    preserved_protected_edges: bool | None = None
    maximum_envelope_ratio: float | None = Field(default=None, ge=0.0)
    invariants_satisfied: list[str] = Field(default_factory=list)
    fragments: list[HardMeshFragment] = Field(default_factory=list)
    intersection_curves: list[HardMeshIntersectionCurve] = Field(default_factory=list)
    fragment_selection_performed: bool = False
    deleted_fragment_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status_shape(self) -> HardMeshNativeReport:
        if self.status == "success":
            if not self.changed:
                raise ValueError("successful hard-mesh reports must declare changed=true")
            if not self.output_path or not self.output_sha256:
                raise ValueError("successful hard-mesh reports require output path and digest")
            if self.failures:
                raise ValueError("successful hard-mesh reports cannot contain failures")
        elif self.output_path is not None or self.output_sha256 is not None or self.changed:
            raise ValueError("refused or failed reports cannot publish an output candidate")
        if (self.changed_region_path is None) != (self.changed_region_sha256 is None):
            raise ValueError("changed-region path and digest must be provided together")
        if (self.correspondence_path is None) != (self.correspondence_sha256 is None):
            raise ValueError("correspondence path and digest must be provided together")
        if (self.attribute_transfer_path is None) != (self.attribute_transfer_sha256 is None):
            raise ValueError("attribute-transfer path and digest must be provided together")
        if any(face_id < 0 for face_id in (*self.changed_face_ids, *self.generated_face_ids)):
            raise ValueError("face identifiers must be non-negative")
        return self


class HardMeshResult(_StrictModel):
    """Adapter-level result retaining unavailable and refused as distinct states."""

    status: Literal["success", "unavailable", "refused", "failed"]
    output_path: str | None = None
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    changed: bool = False
    operation: str | None = None
    implementation_version: str | None = None
    build_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    report_path: str | None = None
    report: HardMeshNativeReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_worker_result(self) -> WorkerResult:
        """Project the richer status onto the package's established worker contract."""

        status = {
            "success": "completed",
            "unavailable": "unavailable",
            "refused": "unavailable",
            "failed": "failed",
        }[self.status]
        return WorkerResult(
            status=status,
            output_path=self.output_path,
            output_sha256=self.output_sha256,
            changed=self.changed,
            operations=[self.operation] if self.status == "success" and self.operation else [],
            warnings=self.warnings,
            failures=self.failures,
            metadata={
                **self.metadata,
                "hard_mesh_status": self.status,
                "native_report_path": self.report_path,
                "implementation_version": self.implementation_version,
                "implementation_build_id": self.build_id,
            },
        )


class HardMeshRoute(_StrictModel):
    """One ordered route without implied acceptance authority."""

    worker: str
    operation: str
    authority: Literal["production", "candidate", "shadow", "reconstructive"]
    priority: int = Field(ge=0)


class HardMeshSelection(_StrictModel):
    """Ordered hard-local routes plus deterministic refusal reasons."""

    status: Literal["selected", "refused"]
    routes: list[HardMeshRoute] = Field(default_factory=list)
    refusal_reasons: list[str] = Field(default_factory=list)


def select_hard_mesh_routes(
    *,
    defect: Literal["classified_hole", "generated_patch_quality", "two_part_intersection"],
    explicit_intent: bool,
    protected_boundaries: bool,
    attributes_supported: bool,
    reconstructive_authorized: bool = False,
    boundary_vertex_count: int | None = None,
) -> HardMeshSelection:
    """Return the fixed least-destructive ordering for Phase 2 hard-local work."""

    routes: list[HardMeshRoute] = []
    refusals: list[str] = []
    local_authorized = explicit_intent and protected_boundaries and attributes_supported
    if not explicit_intent:
        refusals.append("hard-local mutation requires explicit classified defect intent")
    if not protected_boundaries:
        refusals.append("hard-local mutation requires frozen and protected boundaries")
    if not attributes_supported:
        refusals.append("hard-local mutation refuses unsupported attributed topology")

    if defect == "classified_hole":
        if boundary_vertex_count is not None and boundary_vertex_count < 3:
            return HardMeshSelection(
                status="refused",
                refusal_reasons=["a classified boundary loop requires at least three vertices"],
            )
        if boundary_vertex_count in {3, 4} and local_authorized:
            routes.append(
                HardMeshRoute(
                    worker="trimesh_bounded_hole_fill",
                    operation="bounded_hole_fill",
                    authority="production",
                    priority=10,
                )
            )
        if local_authorized:
            routes.extend(
                [
                    HardMeshRoute(
                        worker="pmp_patch",
                        operation="pmp_fill_classified_hole",
                        authority="candidate",
                        priority=20,
                    ),
                    HardMeshRoute(
                        worker="wildmeshing_shadow",
                        operation="wildmeshing_remesh_named_patch",
                        authority="shadow",
                        priority=30,
                    ),
                ]
            )
    elif defect == "generated_patch_quality" and local_authorized:
        routes.append(
            HardMeshRoute(
                worker="wildmeshing_shadow",
                operation="wildmeshing_remesh_named_patch",
                authority="shadow",
                priority=30,
            )
        )
    elif defect == "two_part_intersection":
        routes.append(
            HardMeshRoute(
                worker="geogram_local_repair",
                operation="intersection_repair",
                authority="production",
                priority=10,
            )
        )
        if local_authorized:
            routes.append(
                HardMeshRoute(
                    worker="mcut_shadow",
                    operation="mcut_partition_source_parts",
                    authority="shadow",
                    priority=20,
                )
            )

    if reconstructive_authorized:
        routes.append(
            HardMeshRoute(
                worker="sdf_rebuild",
                operation="per_part_reconstruction",
                authority="reconstructive",
                priority=90,
            )
        )
    routes.sort(key=lambda route: route.priority)
    return HardMeshSelection(
        status="selected" if routes else "refused",
        routes=routes,
        refusal_reasons=refusals,
    )


def _configured_executable(spec: HardMeshExecutableSpec) -> Path | None:
    configured = os.environ.get(spec.environment_variable)
    candidates = [configured, *(shutil.which(name) for name in spec.executable_names)]
    for value in candidates:
        if not value:
            continue
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def discover_hard_mesh_executable(
    spec: HardMeshExecutableSpec,
) -> tuple[Path | None, HardMeshCapabilities | None, str | None]:
    """Discover one exact native build and reject capability or version drift."""

    executable = _configured_executable(spec)
    if executable is None:
        return (
            None,
            None,
            f"{spec.worker} executable is unavailable; set {spec.environment_variable} "
            f"to the audited adapter build",
        )
    try:
        completed = subprocess.run(
            [str(executable), "--capabilities-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env={**os.environ, "OMP_NUM_THREADS": "1", "TBB_NUM_THREADS": "1"},
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        return None, None, f"capability discovery failed: {type(exc).__name__}: {exc}"
    raw = completed.stdout.strip()
    if completed.returncode != 0:
        return None, None, f"capability discovery exited {completed.returncode}"
    if not raw or len(raw.encode("utf-8")) > _MAX_CAPABILITY_BYTES:
        return None, None, "capability discovery returned an empty or oversized document"
    try:
        capabilities = HardMeshCapabilities.model_validate_json(raw)
    except Exception as exc:
        return None, None, f"capability discovery returned malformed JSON: {exc}"
    if capabilities.worker != spec.worker:
        return None, None, f"capability worker mismatch: {capabilities.worker!r}"
    if capabilities.build_id != spec.build_id:
        return (
            None,
            None,
            (
                f"{spec.worker} build drifted: expected {spec.build_id!r}, "
                f"got {capabilities.build_id!r}"
            ),
        )
    if not capabilities.deterministic:
        return None, None, f"{spec.worker} did not declare deterministic execution"
    missing_operations = spec.operations - set(capabilities.operations)
    missing_capabilities = spec.required_capabilities - set(capabilities.capabilities)
    if missing_operations or missing_capabilities:
        return (
            None,
            None,
            (
                f"{spec.worker} is missing operations {sorted(missing_operations)} or capabilities "
                f"{sorted(missing_capabilities)}"
            ),
        )
    return executable, capabilities, None


def verify_approved_executable_digest(
    executable: Path,
    *,
    digest_environment_variable: str,
) -> tuple[str | None, str | None]:
    """Verify an executable against an independently configured approved digest."""

    expected = os.environ.get(digest_environment_variable, "").strip().lower()
    digest_file_variable = f"{digest_environment_variable}_FILE"
    digest_file_value = os.environ.get(digest_file_variable, "").strip()
    if digest_file_value:
        digest_path = Path(digest_file_value).expanduser()
        if (
            not digest_path.is_absolute()
            or digest_path.is_symlink()
            or not digest_path.is_file()
            or digest_path.stat().st_size > 256
        ):
            return None, f"{digest_file_variable} must name a small absolute regular file"
        configured = digest_path.read_text(encoding="ascii").strip().lower()
        if expected and not hmac.compare_digest(expected, configured):
            return None, (f"{digest_environment_variable} and {digest_file_variable} disagree")
        expected = configured
    if not expected:
        return None, (
            "approved executable digest is unavailable; set "
            f"{digest_environment_variable} or {digest_file_variable} to the audited build "
            "SHA-256"
        )
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        return None, f"{digest_environment_variable} is not a lowercase SHA-256 digest"
    try:
        actual = file_sha256(executable)
    except OSError as exc:
        return None, f"could not hash native executable: {type(exc).__name__}: {exc}"
    if not hmac.compare_digest(actual, expected):
        return None, (f"native executable digest mismatch: expected {expected}, got {actual}")
    return actual, None


def _artifact_inside(path: str, root: Path) -> Path | None:
    candidate = Path(path).expanduser().resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _validate_report_artifacts(report: HardMeshNativeReport, work_dir: Path) -> str | None:
    pairs = [
        (report.changed_region_path, report.changed_region_sha256, "changed-region"),
        (report.correspondence_path, report.correspondence_sha256, "correspondence"),
        (report.attribute_transfer_path, report.attribute_transfer_sha256, "attribute-transfer"),
    ]
    pairs.extend(
        (fragment.artifact_path, fragment.artifact_sha256, f"fragment {fragment.fragment_id}")
        for fragment in report.fragments
    )
    for raw_path, expected_digest, label in pairs:
        if raw_path is None:
            continue
        artifact = _artifact_inside(raw_path, work_dir)
        if artifact is None:
            return f"{label} artifact is absent or escaped its assigned directory"
        try:
            if file_sha256(artifact) != expected_digest:
                return f"{label} artifact digest does not match the report"
        except OSError as exc:
            return f"{label} artifact could not be verified: {type(exc).__name__}: {exc}"
    return None


def _discard_output_candidate(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def run_hard_mesh_executable(
    *,
    spec: HardMeshExecutableSpec,
    executable: Path,
    capabilities: HardMeshCapabilities,
    source: Path,
    output: Path,
    request_id: str,
    operation: str,
    parameters: dict[str, Any],
    deterministic_seed: int,
    timeout_s: float,
    approved_executable_sha256: str | None = None,
    expected_backend_source_commit: str | None = None,
    expected_backend_source_tree_sha256: str | None = None,
) -> HardMeshResult:
    """Invoke a pinned helper through the fixed JSON protocol and validate its artifacts."""

    timeout = float(timeout_s)
    if not 1.0 <= timeout <= 300.0:
        return HardMeshResult(
            status="refused",
            operation=operation,
            failures=["hard-local worker timeout_s must be in [1, 300]"],
        )
    source_path = source.expanduser().resolve()
    output_path = output.expanduser().resolve()
    if not source_path.is_file():
        return HardMeshResult(
            status="refused",
            operation=operation,
            failures=["hard-local worker source does not exist"],
        )
    if source_path == output_path:
        return HardMeshResult(
            status="refused",
            operation=operation,
            failures=["hard-local worker output must not overwrite its source checkpoint"],
        )
    if approved_executable_sha256 is not None:
        try:
            executable_digest_before = file_sha256(executable)
        except OSError as exc:
            return HardMeshResult(
                status="failed",
                operation=operation,
                failures=[f"approved executable could not be hashed: {type(exc).__name__}: {exc}"],
            )
        if not hmac.compare_digest(executable_digest_before, approved_executable_sha256):
            return HardMeshResult(
                status="failed",
                operation=operation,
                failures=["approved executable digest changed before native invocation"],
            )
    work_dir = output_path.parent / f"{spec.worker}_protocol"
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    report_path = work_dir / "report.json"
    report_path.unlink(missing_ok=True)
    _discard_output_candidate(output_path)
    request = HardMeshRequest(
        request_id=request_id,
        worker=spec.worker,
        operation=operation,
        implementation_build_id=spec.build_id,
        source_path=str(source_path),
        source_sha256=file_sha256(source_path),
        output_path=str(output_path),
        deterministic_seed=deterministic_seed,
        parameters=parameters,
    )
    atomic_write_json(request_path, request)
    request_digest = file_sha256(request_path)
    command = [
        str(executable),
        "--request",
        str(request_path),
        "--result",
        str(report_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "TBB_NUM_THREADS": "1",
            },
        )
    except subprocess.TimeoutExpired:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=capabilities.implementation_version,
            build_id=capabilities.build_id,
            failures=[f"{spec.worker} exceeded its {timeout:g} second wall-clock limit"],
            metadata={"command": command, "memory_limit": "inherited_from_worker_runner"},
        )
    except OSError as exc:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=capabilities.implementation_version,
            build_id=capabilities.build_id,
            failures=[f"{spec.worker} launch failed: {type(exc).__name__}: {exc}"],
        )
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or report_path.stat().st_size > _MAX_REPORT_BYTES
    ):
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=capabilities.implementation_version,
            build_id=capabilities.build_id,
            failures=["native helper returned no bounded result document"],
            metadata={"return_code": completed.returncode, "command": command},
        )
    try:
        report = HardMeshNativeReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=capabilities.implementation_version,
            build_id=capabilities.build_id,
            report_path=str(report_path),
            failures=[f"native helper returned malformed report JSON: {exc}"],
            metadata={"return_code": completed.returncode, "command": command},
        )
    try:
        immutable_inputs_match = (
            file_sha256(request_path) == request_digest
            and file_sha256(source_path) == request.source_sha256
        )
        executable_identity_unchanged = approved_executable_sha256 is None or hmac.compare_digest(
            file_sha256(executable), approved_executable_sha256
        )
    except OSError as exc:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=capabilities.implementation_version,
            build_id=capabilities.build_id,
            report_path=str(report_path),
            report=report,
            failures=[f"native helper altered an immutable input: {type(exc).__name__}: {exc}"],
        )
    identity_mismatch = (
        report.request_sha256 != request_digest
        or report.worker != spec.worker
        or report.operation != operation
        or report.build_id != spec.build_id
        or report.implementation_version != capabilities.implementation_version
        or report.source_sha256 != request.source_sha256
        or not executable_identity_unchanged
        or (
            approved_executable_sha256 is not None
            and (
                capabilities.executable_sha256 != approved_executable_sha256
                or report.executable_sha256 != approved_executable_sha256
            )
        )
        or (
            expected_backend_source_commit is not None
            and (
                capabilities.backend_source_commit != expected_backend_source_commit
                or report.backend_source_commit != expected_backend_source_commit
            )
        )
        or (
            expected_backend_source_tree_sha256 is not None
            and (
                capabilities.backend_source_tree_sha256 != expected_backend_source_tree_sha256
                or report.backend_source_tree_sha256 != expected_backend_source_tree_sha256
            )
        )
        or not immutable_inputs_match
    )
    if identity_mismatch:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=capabilities.implementation_version,
            build_id=capabilities.build_id,
            report_path=str(report_path),
            report=report,
            failures=["native report identity does not match its immutable request/capabilities"],
        )
    if report.status != "success":
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status=report.status,
            operation=operation,
            implementation_version=report.implementation_version,
            build_id=report.build_id,
            warnings=report.warnings,
            failures=report.failures or [f"{spec.worker} {report.status}"],
            report_path=str(report_path),
            report=report,
            metadata={"return_code": completed.returncode, "shadow_only": spec.shadow_only},
        )
    if completed.returncode != 0:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=report.implementation_version,
            build_id=report.build_id,
            report_path=str(report_path),
            report=report,
            failures=[f"native helper reported success but exited {completed.returncode}"],
        )
    reported_output = Path(report.output_path or "").expanduser().resolve()
    if reported_output != output_path or not output_path.is_file() or output_path.is_symlink():
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=report.implementation_version,
            build_id=report.build_id,
            report_path=str(report_path),
            report=report,
            failures=["native helper output is absent or escaped its assigned path"],
        )
    actual_digest = file_sha256(output_path)
    if actual_digest != report.output_sha256:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=report.implementation_version,
            build_id=report.build_id,
            report_path=str(report_path),
            report=report,
            failures=["native helper output digest does not match its report"],
        )
    artifact_failure = _validate_report_artifacts(report, work_dir)
    if artifact_failure:
        _discard_output_candidate(output_path)
        return HardMeshResult(
            status="failed",
            operation=operation,
            implementation_version=report.implementation_version,
            build_id=report.build_id,
            report_path=str(report_path),
            report=report,
            failures=[artifact_failure],
        )
    return HardMeshResult(
        status="success",
        output_path=str(output_path),
        output_sha256=actual_digest,
        changed=True,
        operation=operation,
        implementation_version=report.implementation_version,
        build_id=report.build_id,
        warnings=report.warnings,
        report_path=str(report_path),
        report=report,
        metadata={
            "return_code": completed.returncode,
            "command": command,
            "request_path": str(request_path),
            "request_sha256": request_digest,
            "native_report_sha256": file_sha256(report_path),
            "memory_limit": "inherited_from_worker_runner",
            "shadow_only": spec.shadow_only,
            "approved_executable_sha256": approved_executable_sha256,
            "backend_source_commit": expected_backend_source_commit,
            "backend_source_tree_sha256": expected_backend_source_tree_sha256,
        },
    )
