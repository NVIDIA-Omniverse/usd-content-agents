# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable identity, checkpoint, resume, and cancellation primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any, Literal, Self

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.embedded_domain_decision import (
    EmbeddedDecisionIdentity,
)

from .embedded_decision import (
    TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY,
    TextureEmbeddedDecisionState,
    validate_embedded_texture_identity_manifests,
)
from .models import (
    TextureExecutionResult,
    TextureFinalizationStatus,
    TexturePlanDocument,
    TextureUnitArtifact,
    TextureValidationResult,
    TextureWorkflowMode,
    TextureWorkflowProgress,
    TextureWorkflowRequest,
)
from .scope_validation import texture_unit_material_state_digests

TEXTURE_WORKFLOW_CHECKPOINT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-checkpoint.v1"
] = "content-agent-workflows.texture-checkpoint.v1"

TextureWorkflowAction = Literal[
    "execute",
    "validate",
    "refine",
    "finalize",
    "review_publication",
    "done",
]
CancellationCheck = Callable[[], bool]


class TextureWorkflowRuntimeError(RuntimeError):
    """Raised when a durable Texture workflow state cannot be trusted."""


class TextureWorkflowCancellationToken:
    """Thread-safe cooperative cancellation token for workflow boundaries."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation at the next durable workflow boundary."""

        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""

        return self._event.is_set()


class TextureArtifactDigest(BaseModel):
    """Byte identity for every local file owned by one selected unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    sha256_by_path: dict[str, str] = Field(min_length=1)


class TextureWorkflowCheckpoint(BaseModel):
    """Atomic outer-workflow state spanning plan, execute, VQA, and refine."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-agent-workflows.texture-checkpoint.v1"] = (
        TEXTURE_WORKFLOW_CHECKPOINT_SCHEMA_VERSION
    )
    revision: int = Field(default=0, ge=0)
    updated_at: str
    mode: TextureWorkflowMode
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan: TexturePlanDocument
    next_action: TextureWorkflowAction
    iteration: int = Field(default=0, ge=0)
    selected_unit_ids: tuple[str, ...] = Field(min_length=1)
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    pending_validation_unit_ids: tuple[str, ...] = ()
    executions: tuple[TextureExecutionResult, ...] = ()
    validations: tuple[TextureValidationResult, ...] = ()
    progress: tuple[TextureWorkflowProgress, ...] = ()
    unit_artifacts: dict[str, TextureUnitArtifact] = Field(default_factory=dict)
    artifact_digests: dict[str, TextureArtifactDigest] = Field(default_factory=dict)
    output_asset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    validation_evidence_sha256_by_path: dict[str, str] = Field(default_factory=dict)
    accepted_unit_material_state_digests: dict[str, str] = Field(default_factory=dict)
    output_asset_path: str | None = None
    client_resume_state: dict[str, Any] = Field(default_factory=dict)
    terminal_status: TextureFinalizationStatus | None = None
    cancellation_reason: str | None = None
    embedded_decision_state: TextureEmbeddedDecisionState | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        selected = self.selected_unit_ids
        if selected != self.plan.selected_unit_ids:
            raise ValueError("checkpoint selected IDs must match immutable plan order")
        if len(selected) != len(set(selected)):
            raise ValueError("checkpoint selected IDs must be unique")
        if set(self.accepted_unit_ids) & set(self.remaining_unit_ids):
            raise ValueError("checkpoint accepted and remaining IDs must be disjoint")
        if set(self.accepted_unit_ids) | set(self.remaining_unit_ids) != set(selected):
            raise ValueError(
                "checkpoint accepted and remaining IDs must partition the plan"
            )
        embedded_all_target_review = (
            self.embedded_decision_state is not None and self.next_action == "validate"
        )
        if embedded_all_target_review:
            if self.pending_validation_unit_ids != selected:
                raise ValueError(
                    "embedded pending validation IDs must cover every canonical "
                    "target in plan order"
                )
        elif not set(self.pending_validation_unit_ids) <= set(self.remaining_unit_ids):
            raise ValueError("pending validation IDs must be unresolved selected units")
        artifact_ids = set(self.unit_artifacts)
        if not artifact_ids <= set(selected):
            raise ValueError("checkpoint artifacts must remain within plan scope")
        if set(self.artifact_digests) != artifact_ids:
            raise ValueError(
                "checkpoint artifact digests must exactly cover unit artifacts"
            )
        if (self.output_asset_path is None) != (self.output_asset_sha256 is None):
            raise ValueError(
                "checkpoint output asset path and digest must be present together"
            )
        evidence_paths = {
            path
            for validation in self.validations
            for finding in validation.findings
            for path in finding.evidence_artifact_paths
        }
        if set(self.validation_evidence_sha256_by_path) != evidence_paths:
            raise ValueError(
                "checkpoint evidence digests must exactly cover validation evidence"
            )
        for digest in self.validation_evidence_sha256_by_path.values():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "validation evidence digests must be lowercase SHA-256"
                )
        if not set(self.accepted_unit_ids) <= artifact_ids:
            raise ValueError("accepted checkpoint units must have artifacts")
        if set(self.accepted_unit_material_state_digests) != set(
            self.accepted_unit_ids
        ):
            raise ValueError(
                "accepted material-state digests must exactly cover accepted units"
            )
        for digest in self.accepted_unit_material_state_digests.values():
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "accepted material-state digests must be lowercase SHA-256"
                )
        if self.next_action == "validate" and not self.pending_validation_unit_ids:
            raise ValueError("validate action requires pending unit IDs")
        if self.next_action == "review_publication" and (
            self.embedded_decision_state is None
            or self.embedded_decision_state.publication_result is None
        ):
            raise ValueError(
                "review_publication requires a persisted embedded publication result"
            )
        if self.terminal_status == "cancelled" and not self.cancellation_reason:
            raise ValueError("cancelled checkpoint requires a reason")
        if self.terminal_status != "cancelled" and self.cancellation_reason:
            raise ValueError(
                "cancellation_reason is only valid for a cancelled checkpoint"
            )
        if self.next_action == "done" and self.terminal_status not in {
            "pass",
            "conditional",
        }:
            raise ValueError("done checkpoint requires pass or conditional status")
        return self


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _stable_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_request_payload(request: TextureWorkflowRequest) -> dict[str, Any]:
    """Return the request identity excluding its storage location."""

    payload = request.model_dump(mode="json")
    source = request.source_asset
    if not source.startswith("s3://"):
        source_path = Path(source).expanduser()
        if source_path.exists():
            source = str(source_path.resolve())
    payload.pop("output_dir")
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop(TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY, None)
    payload["source_asset"] = source
    payload["intent"] = " ".join(request.intent.split())
    return payload


def texture_request_digest(request: TextureWorkflowRequest) -> str:
    """Hash the normalized semantic request used for resume validation."""

    return _stable_digest(normalized_request_payload(request))


def _validated_sha256(value: Any, *, field_name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise TextureWorkflowRuntimeError(
            f"{field_name} must be a 64-character SHA-256 digest"
        )
    return digest


def texture_source_identity_digest(
    request: TextureWorkflowRequest,
    *,
    plan: TexturePlanDocument | None = None,
) -> str:
    """Hash local source bytes or an explicitly supplied remote content digest."""

    supplied = request.metadata.get("source_asset_sha256")
    if request.source_asset.startswith("s3://"):
        if supplied is None:
            raise TextureWorkflowRuntimeError(
                "S3 Texture workflows require metadata.source_asset_sha256"
            )
        return _validated_sha256(
            supplied,
            field_name="metadata.source_asset_sha256",
        )

    supplied_digest = (
        _validated_sha256(
            supplied,
            field_name="metadata.source_asset_sha256",
        )
        if supplied is not None
        else None
    )
    source_path = Path(request.source_asset).expanduser()
    if source_path.is_file():
        local_digest = file_sha256(source_path)
        if supplied_digest is not None and supplied_digest != local_digest:
            raise TextureWorkflowRuntimeError(
                "metadata.source_asset_sha256 does not match local source bytes"
            )
        return local_digest

    if supplied_digest is not None:
        return supplied_digest

    if plan is not None:
        plan_payload = plan.model_dump(mode="json")
        plan_source = plan_payload.get("request", {}).get("source", {})
        plan_digest = (
            plan_source.get("source_asset_sha256")
            if isinstance(plan_source, dict)
            else None
        )
        if plan_digest:
            return _validated_sha256(
                plan_digest,
                field_name="plan.request.source.source_asset_sha256",
            )

    return _stable_digest({"unresolved_source_locator": request.source_asset})


def texture_plan_digest(plan: TexturePlanDocument) -> str:
    """Hash the complete immutable Texture Agent plan."""

    return _stable_digest(plan.model_dump(mode="json"))


def collect_artifact_digests(
    unit_artifacts: Mapping[str, TextureUnitArtifact],
) -> dict[str, TextureArtifactDigest]:
    """Record exact bytes for every selected-unit artifact."""

    result: dict[str, TextureArtifactDigest] = {}
    for unit_id, artifact in unit_artifacts.items():
        sha256_by_path: dict[str, str] = {}
        for raw_path in artifact.artifact_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise TextureWorkflowRuntimeError(
                    f"Texture artifact is not a local file: {path}"
                )
            sha256_by_path[str(path)] = file_sha256(path)
        result[unit_id] = TextureArtifactDigest(
            unit_id=unit_id,
            sha256_by_path=sha256_by_path,
        )
    return result


def collect_output_asset_digest(
    output_asset_path: str | Path | None,
) -> str | None:
    """Record exact bytes for the current candidate output asset."""

    if output_asset_path is None:
        return None
    path = Path(output_asset_path).expanduser().resolve()
    if not path.is_file():
        raise TextureWorkflowRuntimeError(
            f"Texture output asset is not a local file: {path}"
        )
    return file_sha256(path)


def collect_validation_evidence_digests(
    validations: tuple[TextureValidationResult, ...],
) -> dict[str, str]:
    """Record exact bytes for every usd-cli/VQA evidence artifact."""

    result: dict[str, str] = {}
    for validation in validations:
        for finding in validation.findings:
            for raw_path in finding.evidence_artifact_paths:
                path = Path(raw_path).expanduser().resolve()
                if not path.is_file():
                    raise TextureWorkflowRuntimeError(
                        f"Texture validation evidence is not a local file: {path}"
                    )
                result[raw_path] = file_sha256(path)
    return result


def verify_artifact_digests(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    unit_ids: tuple[str, ...] | None = None,
) -> None:
    """Reject missing or byte-changed artifacts before checkpoint reuse."""

    expected_ids = tuple(checkpoint.artifact_digests) if unit_ids is None else unit_ids
    for unit_id in expected_ids:
        digest_record = checkpoint.artifact_digests.get(unit_id)
        if digest_record is None:
            raise TextureWorkflowRuntimeError(
                f"Checkpoint has no artifact digest for {unit_id}"
            )
        for raw_path, expected_digest in digest_record.sha256_by_path.items():
            path = Path(raw_path)
            if not path.is_file():
                raise TextureWorkflowRuntimeError(
                    f"Checkpoint artifact is missing for {unit_id}: {path}"
                )
            if file_sha256(path) != expected_digest:
                raise TextureWorkflowRuntimeError(
                    f"Checkpoint artifact bytes changed for {unit_id}: {path}"
                )


def verify_output_asset_digest(checkpoint: TextureWorkflowCheckpoint) -> None:
    """Reject a missing or byte-changed checkpointed output asset."""

    if checkpoint.output_asset_path is None:
        return
    path = Path(checkpoint.output_asset_path).expanduser().resolve()
    if not path.is_file():
        raise TextureWorkflowRuntimeError(f"Checkpoint output asset is missing: {path}")
    if file_sha256(path) != checkpoint.output_asset_sha256:
        raise TextureWorkflowRuntimeError(
            f"Checkpoint output asset bytes changed: {path}"
        )


def verify_validation_evidence_digests(
    checkpoint: TextureWorkflowCheckpoint,
) -> None:
    """Reject missing or byte-changed checkpointed validation evidence."""

    for (
        raw_path,
        expected_digest,
    ) in checkpoint.validation_evidence_sha256_by_path.items():
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise TextureWorkflowRuntimeError(
                f"Checkpoint validation evidence is missing: {path}"
            )
        if file_sha256(path) != expected_digest:
            raise TextureWorkflowRuntimeError(
                f"Checkpoint validation evidence bytes changed: {path}"
            )


def verify_accepted_unit_material_state(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    output_asset_path: str | Path | None = None,
) -> None:
    """Reject byte-equivalent textures whose accepted USD material state changed."""

    if not checkpoint.accepted_unit_ids:
        return
    raw_output_path = output_asset_path or checkpoint.output_asset_path
    if raw_output_path is None:
        raise TextureWorkflowRuntimeError(
            "Checkpoint accepted units have no textured output asset"
        )
    actual = texture_unit_material_state_digests(
        output_asset_path=raw_output_path,
        plan=checkpoint.plan,
        unit_ids=checkpoint.accepted_unit_ids,
    )
    for unit_id in checkpoint.accepted_unit_ids:
        if actual[unit_id] != checkpoint.accepted_unit_material_state_digests[unit_id]:
            raise TextureWorkflowRuntimeError(
                f"Accepted Texture material state changed for {unit_id}"
            )


def _resume_required_artifact_unit_ids(
    checkpoint: TextureWorkflowCheckpoint,
) -> tuple[str, ...]:
    if checkpoint.next_action == "refine":
        return checkpoint.accepted_unit_ids
    return tuple(checkpoint.artifact_digests)


class TextureWorkflowCheckpointStore:
    """Locked atomic file store for one Texture workflow run."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.path = self.output_dir / "workflow_checkpoint.json"
        self._lock = FileLock(str(self.output_dir / ".texture_workflow.lock"))

    def exists(self) -> bool:
        """Return whether this run has durable workflow state."""

        return self.path.is_file()

    @contextmanager
    def transaction(
        self,
        *,
        ledger_output_dir: str | Path | None = None,
    ) -> Iterator[FileLock]:
        """Hold every durable lock across one complete workflow transition."""

        ledger_root = (
            self.output_dir
            if ledger_output_dir is None
            else Path(ledger_output_dir).expanduser().resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ledger_root.mkdir(parents=True, exist_ok=True)
        checkpoint_lock_path = Path(self._lock.lock_file)
        ledger_lock_path = ledger_root / ".texture_workflow.lock"
        locks_by_path = {checkpoint_lock_path: self._lock}
        ledger_lock = locks_by_path.get(ledger_lock_path)
        if ledger_lock is None:
            ledger_lock = FileLock(str(ledger_lock_path))
            locks_by_path[ledger_lock_path] = ledger_lock
        with ExitStack() as stack:
            for lock_path in sorted(locks_by_path, key=str):
                stack.enter_context(locks_by_path[lock_path])
            yield ledger_lock

    def load(self) -> TextureWorkflowCheckpoint:
        """Load and validate the current checkpoint."""

        try:
            with self._lock:
                payload = load_json(self.path)
            return TextureWorkflowCheckpoint.model_validate(payload)
        except (OSError, ValueError, ValidationError) as exc:
            raise TextureWorkflowRuntimeError(
                f"Invalid Texture workflow checkpoint at {self.path}: {exc}"
            ) from exc

    def save(
        self,
        checkpoint: TextureWorkflowCheckpoint,
    ) -> TextureWorkflowCheckpoint:
        """Increment and atomically publish the checkpoint."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        updated = TextureWorkflowCheckpoint.model_validate(
            checkpoint.model_dump(
                mode="python",
                round_trip=True,
            )
            | {
                "revision": checkpoint.revision + 1,
                "updated_at": _timestamp(),
            }
        )
        with self._lock:
            if self.path.is_file():
                current = TextureWorkflowCheckpoint.model_validate(load_json(self.path))
                if current.revision != checkpoint.revision:
                    raise TextureWorkflowRuntimeError(
                        "Texture workflow checkpoint changed concurrently; "
                        "stop the stale runner"
                    )
            atomic_write_json(self.path, updated)
        return updated

    def save_client_resume_state(
        self,
        checkpoint: TextureWorkflowCheckpoint,
        state: Mapping[str, Any],
    ) -> TextureWorkflowCheckpoint:
        """Persist adapter progress without advancing outer decision state."""

        updated = TextureWorkflowCheckpoint.model_validate(
            checkpoint.model_dump(
                mode="python",
                round_trip=True,
            )
            | {
                "updated_at": _timestamp(),
                "client_resume_state": dict(state),
            }
        )
        with self._lock:
            if not self.path.is_file():
                raise TextureWorkflowRuntimeError(
                    "Texture client resume state requires an existing checkpoint"
                )
            current = TextureWorkflowCheckpoint.model_validate(load_json(self.path))
            current_semantics = current.model_dump(
                mode="python",
                round_trip=True,
                exclude={"updated_at"},
            )
            expected_semantics = checkpoint.model_dump(
                mode="python",
                round_trip=True,
                exclude={"updated_at"},
            )
            if current_semantics != expected_semantics:
                raise TextureWorkflowRuntimeError(
                    "Texture workflow checkpoint changed while adapter progress was "
                    "being persisted; stop the stale runner"
                )
            atomic_write_json(self.path, updated)
        return updated

    def create(
        self,
        *,
        mode: TextureWorkflowMode,
        request: TextureWorkflowRequest,
        plan: TexturePlanDocument,
        source_identity_digest: str,
        next_action: TextureWorkflowAction,
        progress: tuple[TextureWorkflowProgress, ...],
        client_resume_state: Mapping[str, Any],
        embedded_decision_state: TextureEmbeddedDecisionState | None = None,
    ) -> TextureWorkflowCheckpoint:
        """Create the first checkpoint after immutable planning."""

        checkpoint = TextureWorkflowCheckpoint(
            updated_at=_timestamp(),
            mode=mode,
            request_digest=texture_request_digest(request),
            source_identity_digest=source_identity_digest,
            plan_digest=texture_plan_digest(plan),
            plan=plan,
            next_action=next_action,
            selected_unit_ids=plan.selected_unit_ids,
            accepted_unit_ids=(),
            remaining_unit_ids=plan.selected_unit_ids,
            progress=progress,
            client_resume_state=dict(client_resume_state),
            embedded_decision_state=embedded_decision_state,
        )
        return self.save(checkpoint)


def validate_resume_identity(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    request: TextureWorkflowRequest,
    mode: TextureWorkflowMode,
) -> None:
    """Fail closed when a run is resumed with changed identity inputs."""

    if checkpoint.mode != mode:
        raise TextureWorkflowRuntimeError(
            f"Checkpoint mode is {checkpoint.mode!r}, not {mode!r}"
        )
    verify_checkpoint_identity(checkpoint, request=request)
    verify_artifact_digests(
        checkpoint,
        unit_ids=_resume_required_artifact_unit_ids(checkpoint),
    )
    verify_output_asset_digest(checkpoint)
    verify_validation_evidence_digests(checkpoint)
    verify_accepted_unit_material_state(checkpoint)


def verify_checkpoint_identity(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    request: TextureWorkflowRequest,
    active_plan: TexturePlanDocument | None = None,
) -> None:
    """Reject changed request, source, or plan identity at phase boundaries."""

    if checkpoint.request_digest != texture_request_digest(request):
        raise TextureWorkflowRuntimeError(
            "Texture workflow request changed; start a new output directory"
        )
    identity_plan = active_plan or checkpoint.plan
    if checkpoint.source_identity_digest != texture_source_identity_digest(
        request,
        plan=identity_plan,
    ):
        raise TextureWorkflowRuntimeError(
            "Texture workflow source bytes changed; start a new output directory"
        )
    if checkpoint.plan_digest != texture_plan_digest(checkpoint.plan):
        raise TextureWorkflowRuntimeError(
            "Texture workflow checkpoint plan digest does not match its plan"
        )
    if active_plan is not None and checkpoint.plan_digest != texture_plan_digest(
        active_plan
    ):
        raise TextureWorkflowRuntimeError(
            "Texture workflow active plan changed after checkpointing"
        )
    embedded_state = checkpoint.embedded_decision_state
    if embedded_state is not None:
        raw_identity = request.metadata.get(
            TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY
        )
        if raw_identity is None:
            raise TextureWorkflowRuntimeError(
                "Embedded Texture request lacks its frozen decision identity"
            )
        try:
            request_identity = EmbeddedDecisionIdentity.model_validate(raw_identity)
        except ValidationError as exc:
            raise TextureWorkflowRuntimeError(
                f"Embedded Texture request decision identity is invalid: {exc}"
            ) from exc
        if request_identity != embedded_state.identity:
            raise TextureWorkflowRuntimeError(
                "Embedded Texture request decision identity changed; resume is rejected"
            )
        try:
            validate_embedded_texture_identity_manifests(request_identity)
        except ValueError as exc:
            raise TextureWorkflowRuntimeError(str(exc)) from exc


__all__ = [
    "CancellationCheck",
    "TEXTURE_WORKFLOW_CHECKPOINT_SCHEMA_VERSION",
    "TextureArtifactDigest",
    "TextureWorkflowAction",
    "TextureWorkflowCancellationToken",
    "TextureWorkflowCheckpoint",
    "TextureWorkflowCheckpointStore",
    "TextureWorkflowRuntimeError",
    "collect_artifact_digests",
    "collect_output_asset_digest",
    "collect_validation_evidence_digests",
    "normalized_request_payload",
    "texture_plan_digest",
    "texture_request_digest",
    "texture_source_identity_digest",
    "validate_resume_identity",
    "verify_accepted_unit_material_state",
    "verify_artifact_digests",
    "verify_checkpoint_identity",
    "verify_output_asset_digest",
    "verify_validation_evidence_digests",
]
