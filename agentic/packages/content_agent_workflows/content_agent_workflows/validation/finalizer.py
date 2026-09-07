# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical artifact writing for the resumable Validation Agent workflow."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from world_understanding.utils.credentials import ensure_no_inline_secrets
from world_understanding.validation import (
    ValidationPlan,
    ValidationRequest,
    ValidationResult,
)
from world_understanding.validation.cli import finalize_validation_result

from content_agent_workflows.common.artifacts import (
    _directory_chain_matches,
    _open_directory_no_symlinks,
    atomic_write_json,
    atomic_write_json_at,
)

from .models import (
    VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION,
    VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION,
    ValidationArtifactIdentity,
    ValidationWorkflowCheckpoint,
    ValidationWorkflowRun,
    ValidationWorkflowStatus,
)

_WORKFLOW_INTEGRITY_ISSUE_CODES = frozenset(
    {
        "validation.accepted_evidence_missing",
        "validation.accepted_evidence_stale",
        "validation.artifact_publication_unstable",
        "validation.inline_credential_rejected",
        "validation.reference_evidence_stale",
        "validation.render_evidence_missing",
        "validation.render_evidence_input_collision",
        "validation.render_evidence_stale",
        "validation.render_handoff_missing",
        "validation.source_asset_modified",
        "validation.template_execution_error",
        "validation.template_result_mismatch",
        "validation.workflow_artifact_evidence_collision",
        "visual.reference_evidence_missing",
    }
)


@dataclass(frozen=True)
class ValidationWorkflowPaths:
    """Canonical files and attempt root for one validation workflow run."""

    output_dir: Path
    request: Path
    plan: Path
    result: Path
    checkpoint: Path
    evidence: Path
    final_summary: Path
    attempts: Path

    @classmethod
    def from_output_dir(cls, output_dir: str | Path) -> ValidationWorkflowPaths:
        root = Path(output_dir).expanduser().resolve()
        return cls(
            output_dir=root,
            request=root / "validation_request.json",
            plan=root / "validation_plan.json",
            result=root / "validation_result.json",
            checkpoint=root / "validation_checkpoint.json",
            evidence=root / "validation_evidence.json",
            final_summary=root / "final_summary.json",
            attempts=root / "attempts",
        )

    def artifact_paths(self) -> dict[str, str]:
        return {
            "validation_request": str(self.request),
            "validation_plan": str(self.plan),
            "validation_result": str(self.result),
            "validation_checkpoint": str(self.checkpoint),
            "validation_evidence": str(self.evidence),
            "final_summary": str(self.final_summary),
        }


def write_validation_planning_artifacts(
    request: ValidationRequest,
    plan: ValidationPlan,
    paths: ValidationWorkflowPaths,
    *,
    output_dir_fd: int | None = None,
    output_dir_identity: tuple[int, int] | None = None,
) -> None:
    """Persist the effective request and bound plan before backend work."""

    ensure_no_inline_secrets(
        request.model_dump(mode="json"),
        context="validation workflow request artifact",
    )
    ensure_no_inline_secrets(
        plan.model_dump(mode="json"),
        context="validation workflow plan artifact",
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir_fd is None and output_dir_identity is not None and os.name == "posix":
        with _pinned_output_directory_fd(
            paths.output_dir,
            expected_identity=output_dir_identity,
        ) as pinned_fd:
            write_validation_planning_artifacts(
                request,
                plan,
                paths,
                output_dir_fd=pinned_fd,
            )
        return
    _write_output_json(
        paths.request,
        request,
        output_dir=paths.output_dir,
        output_dir_fd=output_dir_fd,
    )
    _write_output_json(
        paths.plan,
        plan,
        output_dir=paths.output_dir,
        output_dir_fd=output_dir_fd,
    )


def _write_output_json(
    path: Path,
    payload: BaseModel | Mapping[str, Any],
    *,
    output_dir: Path,
    output_dir_fd: int | None,
) -> None:
    if path.parent != output_dir:
        raise ValueError(
            f"Validation output artifact must be directly inside output_dir: {path}"
        )
    if output_dir_fd is None:
        atomic_write_json(path, payload)
        return
    atomic_write_json_at(output_dir_fd, path.name, payload)


def _remove_output_artifact(
    path: Path,
    *,
    output_dir: Path,
    output_dir_fd: int | None,
) -> None:
    if path.parent != output_dir:
        raise ValueError(
            f"Validation output artifact must be directly inside output_dir: {path}"
        )
    try:
        if output_dir_fd is None:
            path.unlink()
        else:
            os.unlink(path.name, dir_fd=output_dir_fd)
    except FileNotFoundError:
        pass


@contextmanager
def _pinned_output_directory_fd(
    output_dir: Path,
    *,
    expected_identity: tuple[int, int],
) -> Iterator[int]:
    try:
        output_dir_fd, output_dir_chain = _open_directory_no_symlinks(output_dir)
    except OSError as exc:
        raise RuntimeError(
            f"Validation output directory could not be opened safely: {output_dir}"
        ) from exc
    try:
        opened_stat = os.fstat(output_dir_fd)
        try:
            path_stat = output_dir.stat()
        except OSError as exc:
            raise RuntimeError(
                f"Validation output directory identity changed: {output_dir}"
            ) from exc
        if (
            (opened_stat.st_dev, opened_stat.st_ino) != expected_identity
            or (path_stat.st_dev, path_stat.st_ino) != expected_identity
            or not _directory_chain_matches(output_dir, output_dir_chain)
        ):
            raise RuntimeError(
                f"Validation output directory identity changed: {output_dir}"
            )
        yield output_dir_fd
        try:
            final_path_stat = output_dir.stat()
        except OSError as exc:
            raise RuntimeError(
                f"Validation output directory identity changed: {output_dir}"
            ) from exc
        if (
            final_path_stat.st_dev,
            final_path_stat.st_ino,
        ) != expected_identity:
            raise RuntimeError(
                f"Validation output directory identity changed: {output_dir}"
            )
        if not _directory_chain_matches(output_dir, output_dir_chain):
            raise RuntimeError(
                f"Atomic artifact parent directory changed during write: {output_dir}"
            )
    finally:
        os.close(output_dir_fd)


def _has_workflow_integrity_failure(result: ValidationResult) -> bool:
    return any(
        issue.severity == "fail" and issue.code in _WORKFLOW_INTEGRITY_ISSUE_CODES
        for issue in result.issues
    )


def _without_expected_result_policy(request: ValidationRequest) -> ValidationRequest:
    policy = dict(request.policy)
    policy.pop("expected_verdict", None)
    policy.pop("expected_issue_codes", None)
    return request.model_copy(update={"policy": policy})


def _validation_evidence_payload(
    *,
    checkpoint: ValidationWorkflowCheckpoint,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
) -> dict[str, object]:
    accepted_results = {
        record.template_name: record.accepted_result
        for record in checkpoint.records
        if record.accepted_result is not None
    }
    return {
        "schema_version": VALIDATION_WORKFLOW_EVIDENCE_SCHEMA_VERSION,
        "workflow_identity_digest": checkpoint.workflow_identity.identity_digest,
        "plan_digest": checkpoint.plan_digest,
        "source_before": [
            identity.model_dump(mode="json") for identity in source_before
        ],
        "source_after": [identity.model_dump(mode="json") for identity in source_after],
        "source_unchanged": source_before == source_after,
        "templates": {
            template_name: {
                "status": accepted.result.status,
                "result_path": accepted.result_path,
                "result_sha256": accepted.result_sha256,
                "evidence_artifacts": [
                    artifact.model_dump(mode="json")
                    for artifact in accepted.evidence_artifacts
                ],
            }
            for template_name, accepted in accepted_results.items()
        },
    }


def _validation_summary_payload(
    *,
    status: ValidationWorkflowStatus,
    result: ValidationResult,
    checkpoint: ValidationWorkflowCheckpoint,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
    paths: ValidationWorkflowPaths,
) -> dict[str, object]:
    return {
        "schema_version": VALIDATION_WORKFLOW_SUMMARY_SCHEMA_VERSION,
        "status": status.value,
        "verdict": result.verdict,
        "recommendation": result.recommended_action,
        "source_asset_unchanged": source_before == source_after,
        "completed_templates": [
            template_result.template_name for template_result in result.template_results
        ],
        "remaining_templates": [
            record.template_name
            for record in checkpoint.records
            if record.accepted_result is None
        ],
        "artifacts": paths.artifact_paths(),
    }


def finalized_validation_artifacts_are_valid(
    run: ValidationWorkflowRun,
    *,
    checkpoint: ValidationWorkflowCheckpoint,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
    paths: ValidationWorkflowPaths,
    output_dir_fd: int | None,
) -> bool:
    """Return whether the published report bundle matches the finalized run."""

    expected_payloads = {
        paths.request: run.request.model_dump(mode="json"),
        paths.plan: run.plan.model_dump(mode="json"),
        paths.result: run.result.model_dump(mode="json"),
        paths.evidence: _validation_evidence_payload(
            checkpoint=checkpoint,
            source_before=source_before,
            source_after=source_after,
        ),
        paths.final_summary: _validation_summary_payload(
            status=run.status,
            result=run.result,
            checkpoint=checkpoint,
            source_before=source_before,
            source_after=source_after,
            paths=paths,
        ),
    }
    opened: list[tuple[Path, int, dict[str, object]]] = []
    try:
        for path, expected in expected_payloads.items():
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            if output_dir_fd is None:
                fd = os.open(path, flags)
                path_stat = os.stat(path, follow_symlinks=False)
            else:
                if path.parent != paths.output_dir:
                    return False
                fd = os.open(path.name, flags, dir_fd=output_dir_fd)
                path_stat = os.stat(
                    path.name,
                    dir_fd=output_dir_fd,
                    follow_symlinks=False,
                )
            opened.append((path, fd, expected))
            fd_stat = os.fstat(fd)
            if (
                not stat.S_ISREG(fd_stat.st_mode)
                or fd_stat.st_nlink != 1
                or not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_nlink != 1
                or (fd_stat.st_dev, fd_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                return False
        for _ in range(2):
            for path, fd, expected in opened:
                fd_stat = os.fstat(fd)
                path_stat = (
                    os.stat(path, follow_symlinks=False)
                    if output_dir_fd is None
                    else os.stat(
                        path.name,
                        dir_fd=output_dir_fd,
                        follow_symlinks=False,
                    )
                )
                if (
                    fd_stat.st_nlink != 1
                    or path_stat.st_nlink != 1
                    or (fd_stat.st_dev, fd_stat.st_ino)
                    != (path_stat.st_dev, path_stat.st_ino)
                ):
                    return False
                os.lseek(fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(fd), encoding="utf-8") as stream:
                    payload = json.load(stream)
                if payload != expected:
                    return False
    except (OSError, RuntimeError, ValueError):
        return False
    finally:
        for _, fd, _ in opened:
            os.close(fd)
    return True


def _finalize_validation_workflow_at(
    *,
    status: ValidationWorkflowStatus,
    request: ValidationRequest,
    plan: ValidationPlan,
    raw_result: ValidationResult,
    checkpoint: ValidationWorkflowCheckpoint,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
    paths: ValidationWorkflowPaths,
    output_dir_fd: int | None,
) -> ValidationWorkflowRun:
    """Write stable V1 reports plus workflow checkpoint/evidence summaries."""

    artifact_paths = paths.artifact_paths()
    expected_result_policy_configured = (
        "expected_verdict" in request.policy or "expected_issue_codes" in request.policy
    )
    integrity_policy_bypassed = (
        _has_workflow_integrity_failure(raw_result)
        and expected_result_policy_configured
    )
    finalization_request = (
        _without_expected_result_policy(request)
        if integrity_policy_bypassed
        else request
    )
    # validation_result.json is the terminal commit marker. Remove any result
    # from an earlier publication attempt before updating the bundle, then
    # publish the new result only after every supporting artifact succeeds.
    _remove_output_artifact(
        paths.result,
        output_dir=paths.output_dir,
        output_dir_fd=output_dir_fd,
    )
    finalized = finalize_validation_result(
        raw_result,
        request=finalization_request,
        artifact_paths=artifact_paths,
        runner="content-agent-workflows.validation",
    )
    if integrity_policy_bypassed:
        metadata = dict(finalized.metadata)
        metadata["workflow_integrity_expected_result_bypassed"] = True
        finalized = finalized.model_copy(
            update={
                "request": request,
                "metadata": metadata,
            }
        )
    write_validation_planning_artifacts(
        request,
        finalized.plan,
        paths,
        output_dir_fd=output_dir_fd,
    )
    ensure_no_inline_secrets(
        finalized.model_dump(mode="json"),
        context="validation workflow result artifact",
    )

    evidence_payload = _validation_evidence_payload(
        checkpoint=checkpoint,
        source_before=source_before,
        source_after=source_after,
    )
    ensure_no_inline_secrets(
        evidence_payload,
        context="validation workflow evidence artifact",
    )
    _write_output_json(
        paths.evidence,
        evidence_payload,
        output_dir=paths.output_dir,
        output_dir_fd=output_dir_fd,
    )

    summary_payload = _validation_summary_payload(
        status=status,
        result=finalized,
        checkpoint=checkpoint,
        source_before=source_before,
        source_after=source_after,
        paths=paths,
    )
    ensure_no_inline_secrets(
        summary_payload,
        context="validation workflow summary artifact",
    )
    _write_output_json(
        paths.final_summary,
        summary_payload,
        output_dir=paths.output_dir,
        output_dir_fd=output_dir_fd,
    )
    _write_output_json(
        paths.result,
        finalized,
        output_dir=paths.output_dir,
        output_dir_fd=output_dir_fd,
    )

    return ValidationWorkflowRun(
        status=status,
        output_dir=str(paths.output_dir),
        request=request,
        plan=finalized.plan,
        result=finalized,
        checkpoint=checkpoint,
        request_path=str(paths.request),
        plan_path=str(paths.plan),
        result_path=str(paths.result),
        checkpoint_path=str(paths.checkpoint),
        evidence_path=str(paths.evidence),
        final_summary_path=str(paths.final_summary),
    )


def finalize_validation_workflow(
    *,
    status: ValidationWorkflowStatus,
    request: ValidationRequest,
    plan: ValidationPlan,
    raw_result: ValidationResult,
    checkpoint: ValidationWorkflowCheckpoint,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
    paths: ValidationWorkflowPaths,
    output_dir_identity: tuple[int, int] | None,
) -> ValidationWorkflowRun:
    """Write stable V1 reports while pinning their output directory."""

    if os.name != "posix":  # pragma: win32 cover
        return _finalize_validation_workflow_at(
            status=status,
            request=request,
            plan=plan,
            raw_result=raw_result,
            checkpoint=checkpoint,
            source_before=source_before,
            source_after=source_after,
            paths=paths,
            output_dir_fd=None,
        )
    if output_dir_identity is None:
        raise ValueError(
            "POSIX validation finalization requires a pre-captured output "
            "directory identity"
        )
    with _pinned_output_directory_fd(
        paths.output_dir,
        expected_identity=output_dir_identity,
    ) as output_dir_fd:
        return _finalize_validation_workflow_at(
            status=status,
            request=request,
            plan=plan,
            raw_result=raw_result,
            checkpoint=checkpoint,
            source_before=source_before,
            source_after=source_after,
            paths=paths,
            output_dir_fd=output_dir_fd,
        )
