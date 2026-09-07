# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unconditional workflow run manifest and artifact recorder."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import (
    atomic_write_json,
    file_sha256,
    read_contained_artifact,
    snapshot_contained_artifact,
)

WORKFLOW_RUN_MANIFEST_SCHEMA_VERSION = "content-agent-workflows.run-manifest.v1"
MAX_RUN_RECORD_JSON_BYTES = 16 * 1024 * 1024
MAX_RUN_RECORD_IMAGE_BYTES = 128 * 1024 * 1024


def _now() -> str:
    return datetime.now(UTC).isoformat()


class WorkflowArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_name: str
    kind: str
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    required: bool = True


class WorkflowCheckpointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str
    created_at: str
    artifacts: list[str] = Field(default_factory=list)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    sealed: bool = True


class WorkflowRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-agent-workflows.run-manifest.v1"] = (
        WORKFLOW_RUN_MANIFEST_SCHEMA_VERSION
    )
    workflow: str
    status: Literal["running", "pass", "fail", "blocked"] = "running"
    created_at: str
    updated_at: str
    request_path: str
    request_sha256: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    backend: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    required_artifacts: list[str] = Field(default_factory=list)
    artifacts: list[WorkflowArtifactRecord] = Field(default_factory=list)
    checkpoints: list[WorkflowCheckpointRecord] = Field(default_factory=list)
    failure: dict[str, Any] | None = None


def _record_paths(
    run_dir: Path,
    record_subdir: str | Path | None,
) -> tuple[Path, Path]:
    """Return contained request/manifest paths for one recorder namespace."""

    if record_subdir is None:
        directory = run_dir
    else:
        relative = Path(record_subdir)
        if relative.is_absolute() or any(
            component in {"", ".", ".."} for component in relative.parts
        ):
            raise ValueError(
                "Workflow run-record subdirectory must be a safe relative path."
            )
        directory = run_dir / relative
    return directory / "request.json", directory / "workflow_run_manifest.json"


class WorkflowRunRecorder:
    """Atomically maintain one backend-neutral run manifest."""

    def __init__(
        self,
        run_dir: Path,
        manifest: WorkflowRunManifest,
        *,
        manifest_path: Path | None = None,
    ):
        self.run_dir = run_dir.expanduser().resolve()
        self.path = manifest_path or self.run_dir / "workflow_run_manifest.json"
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        run_dir: Path,
        *,
        workflow: str,
        request: dict[str, Any],
        source_path: Path | None = None,
        backend: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        required_artifacts: list[str] | None = None,
        record_subdir: str | Path | None = None,
    ) -> WorkflowRunRecorder:
        root = run_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        request_target, manifest_path = _record_paths(root, record_subdir)
        request_path = atomic_write_json(
            request_target,
            request,
            within=root,
        )
        request_read = read_contained_artifact(
            root,
            request_path,
            max_bytes=MAX_RUN_RECORD_JSON_BYTES,
        )
        created = _now()
        source = source_path.expanduser().resolve() if source_path else None
        manifest = WorkflowRunManifest(
            workflow=workflow,
            created_at=created,
            updated_at=created,
            request_path=request_path.relative_to(root).as_posix(),
            request_sha256=request_read.sha256,
            source_path=str(source) if source else None,
            source_sha256=file_sha256(source) if source else None,
            backend=dict(backend or {}),
            policy=dict(policy or {}),
            required_artifacts=list(required_artifacts or []),
        )
        recorder = cls(root, manifest, manifest_path=manifest_path)
        recorder._write()
        return recorder

    @classmethod
    def adopt(
        cls,
        run_dir: Path,
        *,
        workflow: str,
        request: dict[str, Any],
        source_path: Path | None = None,
        backend: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        required_artifacts: list[str] | None = None,
        record_subdir: str | Path | None = None,
    ) -> WorkflowRunRecorder:
        """Create a manifest around an existing immutable request.

        This is used only to bring a pre-manifest resumable run under the
        durable contract.  The request bytes are deliberately preserved so
        workflow-specific input digests that already include the file remain
        valid.
        """

        root = run_dir.expanduser().resolve()
        request_path, manifest_path = _record_paths(root, record_subdir)
        request_read = read_contained_artifact(
            root,
            request_path,
            max_bytes=MAX_RUN_RECORD_JSON_BYTES,
            parse_json=True,
        )
        if request_read.json_object != request:
            raise ValueError(
                "Cannot adopt workflow run because request.json does not "
                "match the validated request."
            )
        created = _now()
        source = source_path.expanduser().resolve() if source_path else None
        manifest = WorkflowRunManifest(
            workflow=workflow,
            created_at=created,
            updated_at=created,
            request_path=request_read.path.relative_to(root).as_posix(),
            request_sha256=request_read.sha256,
            source_path=str(source) if source else None,
            source_sha256=file_sha256(source) if source else None,
            backend=dict(backend or {}),
            policy=dict(policy or {}),
            required_artifacts=list(required_artifacts or []),
        )
        recorder = cls(root, manifest, manifest_path=manifest_path)
        recorder._write()
        return recorder

    @classmethod
    def start(
        cls,
        run_dir: Path,
        *,
        workflow: str,
        request: dict[str, Any],
        source_path: Path | None = None,
        backend: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        required_artifacts: list[str] | None = None,
        resume: bool = False,
        record_subdir: str | Path | None = None,
    ) -> WorkflowRunRecorder:
        """Create a run record or resume the exact same recorded request.

        Resume is deliberately fail-closed: it requires an existing manifest
        whose request, source identity, backend, policy, and required-artifact
        contract match the caller. Existing artifacts and sealed checkpoints
        remain available as durable recovery evidence.
        """

        root = run_dir.expanduser().resolve()
        if not resume:
            return cls.create(
                root,
                workflow=workflow,
                request=request,
                source_path=source_path,
                backend=backend,
                policy=policy,
                required_artifacts=required_artifacts,
                record_subdir=record_subdir,
            )

        recorder = cls.resume(root, record_subdir=record_subdir)
        expected_backend = dict(backend or {})
        expected_policy = dict(policy or {})
        expected_required = list(required_artifacts or [])
        request_read = read_contained_artifact(
            root,
            root / recorder.manifest.request_path,
            max_bytes=MAX_RUN_RECORD_JSON_BYTES,
            parse_json=True,
        )
        mismatches: list[str] = []
        if recorder.manifest.workflow != workflow:
            mismatches.append("workflow")
        if request_read.json_object != request:
            mismatches.append("request")
        if (
            recorder.manifest.request_sha256 is not None
            and request_read.sha256 != recorder.manifest.request_sha256
        ):
            mismatches.append("request_sha256")
        if recorder.manifest.backend != expected_backend:
            mismatches.append("backend")
        if recorder.manifest.policy != expected_policy:
            mismatches.append("policy")
        if recorder.manifest.required_artifacts != expected_required:
            mismatches.append("required_artifacts")

        source = source_path.expanduser().resolve() if source_path else None
        expected_source_path = str(source) if source else None
        expected_source_sha256 = file_sha256(source) if source else None
        if recorder.manifest.source_path != expected_source_path:
            mismatches.append("source_path")
        if recorder.manifest.source_sha256 != expected_source_sha256:
            mismatches.append("source_sha256")
        if recorder.manifest.checkpoints:
            records = {
                record.logical_name: record for record in recorder.manifest.artifacts
            }
            checkpoint_invalid = False
            sealed_checkpoints = [
                checkpoint
                for checkpoint in recorder.manifest.checkpoints
                if checkpoint.sealed
            ]
            for checkpoint_index, checkpoint in enumerate(sealed_checkpoints):
                for logical_name, expected_sha256 in checkpoint.artifact_sha256.items():
                    checkpoint_path = checkpoint.artifact_paths.get(logical_name)
                    if checkpoint_path:
                        artifact_path = root / checkpoint_path
                    else:
                        record = records.get(logical_name)
                        artifact_path = (
                            root / record.path if record is not None else None
                        )
                    later_checkpoint_claims_logical_name = any(
                        logical_name in later.artifact_sha256
                        for later in sealed_checkpoints[checkpoint_index + 1 :]
                    )
                    try:
                        if artifact_path is None:
                            raise ValueError("checkpoint artifact has no path")
                        artifact_read = read_contained_artifact(
                            root,
                            artifact_path,
                        )
                    except ValueError:
                        # Pre-snapshot manifests could only point at the live
                        # logical path. A later checkpoint supersedes that
                        # mutable legacy claim; its own digest is still checked.
                        if later_checkpoint_claims_logical_name and not (
                            checkpoint_path
                            and ".workflow_checkpoints" in Path(checkpoint_path).parts
                        ):
                            continue
                        checkpoint_invalid = True
                        break
                    if artifact_read.sha256 != expected_sha256:
                        if later_checkpoint_claims_logical_name and not (
                            checkpoint_path
                            and ".workflow_checkpoints" in Path(checkpoint_path).parts
                        ):
                            continue
                        checkpoint_invalid = True
                        break
                if checkpoint_invalid:
                    mismatches.append("checkpoint_artifacts")
                    break
        if mismatches:
            raise ValueError(
                "Cannot resume workflow run because recorded inputs changed: "
                + ", ".join(mismatches)
            )

        recorder.manifest = recorder.manifest.model_copy(
            update={"status": "running", "failure": None, "updated_at": _now()}
        )
        recorder._write()
        return recorder

    @classmethod
    def resume(
        cls,
        run_dir: Path,
        *,
        record_subdir: str | Path | None = None,
    ) -> WorkflowRunRecorder:
        root = run_dir.expanduser().resolve()
        _request_path, manifest_path = _record_paths(root, record_subdir)
        manifest_read = read_contained_artifact(
            root,
            manifest_path,
            max_bytes=MAX_RUN_RECORD_JSON_BYTES,
            parse_json=True,
        )
        if manifest_read.json_object is None:
            raise ValueError(f"Expected a JSON object in {manifest_read.path}")
        manifest = WorkflowRunManifest.model_validate(manifest_read.json_object)
        return cls(root, manifest, manifest_path=manifest_path)

    def record_artifact(
        self,
        logical_name: str,
        path: Path,
        *,
        kind: str,
        required: bool = True,
        image: bool = False,
    ) -> WorkflowArtifactRecord:
        artifact_read = read_contained_artifact(
            self.run_dir,
            path,
            max_bytes=MAX_RUN_RECORD_IMAGE_BYTES if image else None,
            image=image,
        )
        record = WorkflowArtifactRecord(
            logical_name=logical_name,
            kind=kind,
            path=artifact_read.path.relative_to(self.run_dir).as_posix(),
            sha256=artifact_read.sha256,
            size_bytes=artifact_read.size_bytes,
            required=required,
        )
        artifacts = [
            existing
            for existing in self.manifest.artifacts
            if existing.logical_name != logical_name
        ]
        artifacts.append(record)
        self.manifest = self.manifest.model_copy(
            update={"artifacts": artifacts, "updated_at": _now()}
        )
        self._write()
        return record

    def seal_request(self) -> None:
        """Seal the current request bytes after workflow-owned enrichment.

        Some launchers add deterministic preflight and routing evidence to
        their request during setup.  They must call this before the first
        checkpoint; resumable workflows should instead keep request.json
        immutable and never call this while resuming.
        """

        request_read = read_contained_artifact(
            self.run_dir,
            self.run_dir / self.manifest.request_path,
            max_bytes=MAX_RUN_RECORD_JSON_BYTES,
        )
        self.manifest = self.manifest.model_copy(
            update={
                "request_sha256": request_read.sha256,
                "updated_at": _now(),
            }
        )
        self._write()

    def checkpoint(self, phase: str, artifacts: list[str]) -> None:
        records = {record.logical_name: record for record in self.manifest.artifacts}
        known = set(records)
        unknown = sorted(set(artifacts).difference(known))
        if unknown:
            raise ValueError(
                "Checkpoint references unknown workflow artifacts: "
                + ", ".join(unknown)
            )
        checkpoint_number = len(self.manifest.checkpoints) + 1
        record_parent = self.path.parent.relative_to(self.run_dir)
        snapshot_root = (
            record_parent / ".workflow_checkpoints" / f"{checkpoint_number:06d}"
        )
        artifact_paths: dict[str, str] = {}
        for artifact_index, logical_name in enumerate(
            dict.fromkeys(artifacts),
            start=1,
        ):
            record = records[logical_name]
            logical_digest = hashlib.sha256(logical_name.encode("utf-8")).hexdigest()
            snapshot_relative = snapshot_root / (
                f"{artifact_index:04d}-{logical_digest[:16]}.artifact"
            )
            snapshot = snapshot_contained_artifact(
                self.run_dir,
                self.run_dir / record.path,
                self.run_dir / snapshot_relative,
                expected_sha256=record.sha256,
                expected_size_bytes=record.size_bytes,
            )
            artifact_paths[logical_name] = snapshot.path.relative_to(
                self.run_dir
            ).as_posix()
        checkpoint = WorkflowCheckpointRecord(
            phase=phase,
            created_at=_now(),
            artifacts=list(artifacts),
            artifact_sha256={
                record.logical_name: record.sha256
                for record in self.manifest.artifacts
                if record.logical_name in artifacts
            },
            artifact_paths=artifact_paths,
        )
        self.manifest = self.manifest.model_copy(
            update={
                "checkpoints": [*self.manifest.checkpoints, checkpoint],
                "updated_at": _now(),
            }
        )
        self._write()

    def finalize(
        self,
        status: Literal["pass", "fail", "blocked"],
        *,
        failure: dict[str, Any] | None = None,
        annotate_integrity: bool = True,
    ) -> WorkflowRunManifest:
        input_drift: list[str] = []
        try:
            request_read = read_contained_artifact(
                self.run_dir,
                self.run_dir / self.manifest.request_path,
                max_bytes=MAX_RUN_RECORD_JSON_BYTES,
            )
            if (
                self.manifest.request_sha256 is not None
                and request_read.sha256 != self.manifest.request_sha256
            ):
                input_drift.append("request_sha256")
        except ValueError:
            input_drift.append("request")
        if self.manifest.source_path is not None:
            try:
                current_source_sha256 = file_sha256(self.manifest.source_path)
            except (OSError, ValueError):
                input_drift.append("source")
            else:
                if current_source_sha256 != self.manifest.source_sha256:
                    input_drift.append("source_sha256")

        recorded = {record.logical_name for record in self.manifest.artifacts}
        missing = sorted(set(self.manifest.required_artifacts).difference(recorded))
        invalid: list[str] = []
        required = set(self.manifest.required_artifacts)
        for record in self.manifest.artifacts:
            if record.logical_name not in required:
                continue
            try:
                artifact_read = read_contained_artifact(
                    self.run_dir,
                    self.run_dir / record.path,
                )
            except ValueError:
                invalid.append(record.logical_name)
                continue
            if (
                artifact_read.sha256 != record.sha256
                or artifact_read.size_bytes != record.size_bytes
            ):
                invalid.append(record.logical_name)
        entry_status = status
        if status == "pass" and input_drift:
            status = "fail"
            failure = {
                "code": "workflow_input_drift",
                "invalid": sorted(input_drift),
            }
        if status == "pass" and (missing or invalid):
            status = "fail"
            failure = {
                "code": "invalid_required_artifacts"
                if invalid
                else "missing_required_artifacts",
                "missing": missing,
                **({"invalid": sorted(invalid)} if invalid else {}),
            }
        elif (
            annotate_integrity
            and entry_status == "fail"
            and (input_drift or missing or invalid)
        ):
            # Blocked finalizations (dry runs, intentional stops) legitimately
            # lack terminal artifacts; only failed runs get the annotation.
            # An already-failed run must still surface contract violations:
            # dropping them lets a generic failure code coexist with a
            # rewritten sealed request or a missing required artifact, and
            # consumers that trust the failure object (e.g. conditional-exit
            # grading) would treat the incomplete contract as intact.
            failure = dict(failure or {})
            failure.setdefault(
                "integrity",
                {
                    **({"input_drift": sorted(input_drift)} if input_drift else {}),
                    **({"missing_required_artifacts": missing} if missing else {}),
                    **(
                        {"invalid_required_artifacts": sorted(invalid)}
                        if invalid
                        else {}
                    ),
                },
            )
        self.manifest = self.manifest.model_copy(
            update={
                "status": status,
                "failure": dict(failure) if failure is not None else None,
                "updated_at": _now(),
            }
        )
        self._write()
        return self.manifest

    def finalize_unexpected_failure(self, error: Exception) -> None:
        """Best-effort seal this run as failed without masking ``error``.

        Workflow entrypoints call this only after ``start`` returned.  Any
        recorder failure is deliberately suppressed so the workflow's
        original exception remains the one observed by its caller.
        """

        try:
            # A crash seal is already an explicit, untrusted failure: no
            # downgrade gate consumes it, and the run died before its
            # artifacts could exist, so the integrity annotation would add
            # only noise to every workflow's failure contract.
            self.finalize(
                "fail",
                failure={
                    "code": "unexpected_workflow_error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                annotate_integrity=False,
            )
        except Exception:
            pass

    def _write(self) -> None:
        atomic_write_json(
            self.path,
            self.manifest,
            within=self.run_dir,
        )
