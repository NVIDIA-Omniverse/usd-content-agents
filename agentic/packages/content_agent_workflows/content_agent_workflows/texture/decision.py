# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, evidence-bound decisions for skill-routed Texture workflow steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)

from .runtime import TextureWorkflowAction, TextureWorkflowCheckpoint

TEXTURE_DECISION_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-decision-patch.v1"
] = "content-agent-workflows.texture-decision-patch.v1"
TEXTURE_STEP_OBSERVATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-step-observation.v1"
] = "content-agent-workflows.texture-step-observation.v1"
TEXTURE_DECISION_LEDGER_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-decision-ledger.v1"
] = "content-agent-workflows.texture-decision-ledger.v1"

TextureDecisionOperation = Literal[
    "inspect_scope",
    "inspect_uv",
    "generate_candidate",
    "preview_apply",
    "render_verification",
    "assess_visual_quality",
    "inspect_failed_units",
    "regenerate_failed_units",
    "verify_dependency_closure",
    "finalize_decision_patch",
    "publish_portable_asset",
]

_OPERATIONS_BY_ACTION: dict[
    TextureWorkflowAction, tuple[TextureDecisionOperation, ...]
] = {
    "execute": (
        "inspect_scope",
        "inspect_uv",
        "generate_candidate",
        "preview_apply",
    ),
    "validate": ("render_verification", "assess_visual_quality"),
    "refine": (
        "inspect_failed_units",
        "regenerate_failed_units",
        "preview_apply",
    ),
    "finalize": (
        "verify_dependency_closure",
        "finalize_decision_patch",
        "publish_portable_asset",
    ),
    "done": (),
}

_DECISION_ACTION_BY_PROGRESS_PHASE: dict[
    str, Literal["execute", "validate", "refine", "finalize"]
] = {
    "executing": "execute",
    "validating": "validate",
    "refining": "refine",
    "finalizing": "finalize",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextureDecisionPatch(_StrictModel):
    """One child-authored decision bound to an exact durable Texture state."""

    schema_version: Literal["content-agent-workflows.texture-decision-patch.v1"] = (
        TEXTURE_DECISION_PATCH_SCHEMA_VERSION
    )
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_revision: int = Field(ge=1)
    action: Literal["execute", "validate", "refine", "finalize"]
    iteration: int = Field(ge=0)
    target_unit_ids: tuple[str, ...]
    operations: tuple[TextureDecisionOperation, ...] = Field(min_length=2)
    evidence_sha256_by_path: dict[str, str] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        if len(self.target_unit_ids) != len(set(self.target_unit_ids)):
            raise ValueError("target_unit_ids must be unique")
        expected_operations = _OPERATIONS_BY_ACTION[self.action]
        if self.operations != expected_operations:
            raise ValueError(
                f"{self.action} operations must be {expected_operations!r}"
            )
        for path, digest in self.evidence_sha256_by_path.items():
            if not path:
                raise ValueError("evidence paths must not be empty")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("evidence digests must be lowercase SHA-256")
        return self


class TextureDecisionRecord(_StrictModel):
    """Digest-bound index entry for one canonical agent decision patch."""

    sequence: int = Field(ge=1)
    checkpoint_revision: int = Field(ge=1)
    action: Literal["execute", "validate", "refine", "finalize"]
    decision_patch_path: str = Field(min_length=1)
    decision_patch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TextureDecisionLedger(_StrictModel):
    """Append-only identity and byte bindings for skill-routed decisions."""

    schema_version: Literal["content-agent-workflows.texture-decision-ledger.v1"] = (
        TEXTURE_DECISION_LEDGER_SCHEMA_VERSION
    )
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[TextureDecisionRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_sequence(self) -> Self:
        expected_sequences = tuple(range(1, len(self.records) + 1))
        if tuple(record.sequence for record in self.records) != expected_sequences:
            raise ValueError("Texture decision record sequences must be contiguous")
        revisions = tuple(record.checkpoint_revision for record in self.records)
        if any(left >= right for left, right in zip(revisions, revisions[1:])):
            raise ValueError(
                "Texture decision checkpoint revisions must be strictly increasing"
            )
        actions = tuple(record.action for record in self.records)
        if actions and actions[0] != "execute":
            raise ValueError("Texture decision sequence must start with execute")
        for previous, current in zip(actions, actions[1:]):
            if current == previous:
                # Cooperative cancellation may leave the same durable action
                # pending. Resumption requires a fresh, revision-bound decision.
                continue
            allowed = {
                "execute": {"validate"},
                "validate": {"refine", "finalize"},
                "refine": {"validate"},
                "finalize": set(),
            }[previous]
            if current not in allowed:
                raise ValueError(
                    f"Invalid Texture decision transition: {previous} -> {current}"
                )
        return self


class TextureStepObservation(_StrictModel):
    """Compact deterministic packet used by a child or embedded coordinator."""

    schema_version: Literal["content-agent-workflows.texture-step-observation.v1"] = (
        TEXTURE_STEP_OBSERVATION_SCHEMA_VERSION
    )
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_revision: int = Field(ge=1)
    action: TextureWorkflowAction
    iteration: int = Field(ge=0)
    target_unit_ids: tuple[str, ...]
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    required_operations: tuple[TextureDecisionOperation, ...]
    evidence_sha256_by_path: dict[str, str]
    decision_patch_path: str | None = None
    terminal: bool


def texture_checkpoint_decision_digest(
    checkpoint: TextureWorkflowCheckpoint,
) -> str:
    """Digest decision-relevant state without volatile progress timestamps."""

    payload = {
        "schema_version": "content-agent-workflows.texture-decision-state.v1",
        "request_digest": checkpoint.request_digest,
        "source_identity_digest": checkpoint.source_identity_digest,
        "plan_digest": checkpoint.plan_digest,
        "revision": checkpoint.revision,
        "next_action": checkpoint.next_action,
        "iteration": checkpoint.iteration,
        "selected_unit_ids": checkpoint.selected_unit_ids,
        "accepted_unit_ids": checkpoint.accepted_unit_ids,
        "remaining_unit_ids": checkpoint.remaining_unit_ids,
        "pending_validation_unit_ids": checkpoint.pending_validation_unit_ids,
        "artifact_digests": {
            unit_id: artifact.model_dump(mode="json")
            for unit_id, artifact in sorted(checkpoint.artifact_digests.items())
        },
        "output_asset_sha256": checkpoint.output_asset_sha256,
        "validation_evidence_sha256_by_path": dict(
            sorted(checkpoint.validation_evidence_sha256_by_path.items())
        ),
        "accepted_unit_material_state_digests": dict(
            sorted(checkpoint.accepted_unit_material_state_digests.items())
        ),
        "embedded_decision_state": (
            checkpoint.embedded_decision_state.model_dump(mode="json")
            if checkpoint.embedded_decision_state is not None
            else None
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_unit_ids(
    checkpoint: TextureWorkflowCheckpoint,
) -> tuple[str, ...]:
    if checkpoint.next_action in {"execute", "refine"}:
        return checkpoint.remaining_unit_ids
    if checkpoint.next_action == "validate":
        return checkpoint.pending_validation_unit_ids
    if checkpoint.next_action == "finalize":
        return checkpoint.selected_unit_ids
    return ()


def _decision_evidence(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    output_dir: Path,
) -> dict[str, str]:
    candidates = [output_dir / "request.json", output_dir / "texture_plan.json"]
    if checkpoint.output_asset_path is not None:
        candidates.append(Path(checkpoint.output_asset_path))
    candidates.extend(
        Path(path) for path in checkpoint.validation_evidence_sha256_by_path
    )
    evidence: dict[str, str] = {}
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if not resolved.is_relative_to(output_dir):
            raise ValueError(
                f"Texture decision evidence escapes the run directory: {resolved}"
            )
        if resolved.is_file():
            evidence[str(resolved)] = file_sha256(resolved)
    return dict(sorted(evidence.items()))


def build_texture_step_observation(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    output_dir: str | Path,
) -> TextureStepObservation:
    """Build the exact packet a reasoning loop must review before the next step."""

    root = Path(output_dir).expanduser().resolve()
    action = checkpoint.next_action
    decision_path = None
    if action != "done":
        decision_path = str(
            root / "decisions" / f"{checkpoint.revision:04d}-{action}-decision.json"
        )
    return TextureStepObservation(
        request_digest=checkpoint.request_digest,
        source_identity_digest=checkpoint.source_identity_digest,
        plan_digest=checkpoint.plan_digest,
        checkpoint_decision_digest=texture_checkpoint_decision_digest(checkpoint),
        checkpoint_revision=checkpoint.revision,
        action=action,
        iteration=checkpoint.iteration,
        target_unit_ids=_target_unit_ids(checkpoint),
        accepted_unit_ids=checkpoint.accepted_unit_ids,
        remaining_unit_ids=checkpoint.remaining_unit_ids,
        required_operations=_OPERATIONS_BY_ACTION[action],
        evidence_sha256_by_path=_decision_evidence(checkpoint, output_dir=root),
        decision_patch_path=decision_path,
        terminal=action == "done",
    )


def validate_texture_decision_patch(
    patch: TextureDecisionPatch,
    *,
    checkpoint: TextureWorkflowCheckpoint,
    output_dir: str | Path,
) -> None:
    """Fail closed on stale, out-of-scope, or evidence-drifted decisions."""

    observation = build_texture_step_observation(checkpoint, output_dir=output_dir)
    expected = {
        "request_digest": observation.request_digest,
        "source_identity_digest": observation.source_identity_digest,
        "plan_digest": observation.plan_digest,
        "checkpoint_decision_digest": observation.checkpoint_decision_digest,
        "checkpoint_revision": observation.checkpoint_revision,
        "action": observation.action,
        "iteration": observation.iteration,
        "target_unit_ids": observation.target_unit_ids,
        "operations": observation.required_operations,
        "evidence_sha256_by_path": observation.evidence_sha256_by_path,
    }
    actual = patch.model_dump(mode="python", include=set(expected))
    if actual != expected:
        mismatches = sorted(
            field for field, value in expected.items() if actual.get(field) != value
        )
        raise ValueError(
            "Texture decision patch does not match the current durable state: "
            + ", ".join(mismatches)
        )

    root = Path(output_dir).expanduser().resolve()
    for raw_path, expected_digest in patch.evidence_sha256_by_path.items():
        evidence_path = Path(raw_path).expanduser().resolve()
        if not evidence_path.is_relative_to(root):
            raise ValueError(
                f"Texture decision evidence escapes the run directory: {evidence_path}"
            )
        if file_sha256(evidence_path) != expected_digest:
            raise ValueError(
                f"Texture decision evidence changed after review: {evidence_path}"
            )


def texture_decision_ledger_path(output_dir: str | Path) -> Path:
    """Return the canonical ledger path for one Texture workflow run."""

    return Path(output_dir).expanduser().resolve() / "texture_decision_ledger.json"


def _texture_decision_patch_path(
    output_dir: Path,
    *,
    checkpoint_revision: int,
    action: str,
) -> Path:
    return (
        output_dir / "decisions" / f"{checkpoint_revision:04d}-{action}-decision.json"
    )


def _verify_texture_decision_records(
    ledger: TextureDecisionLedger,
    *,
    output_dir: Path,
) -> None:
    for record in ledger.records:
        patch_path = Path(record.decision_patch_path).expanduser().resolve()
        canonical_patch_path = _texture_decision_patch_path(
            output_dir,
            checkpoint_revision=record.checkpoint_revision,
            action=record.action,
        )
        if patch_path != canonical_patch_path:
            raise ValueError(
                "Texture decision record must use its canonical patch path: "
                f"{canonical_patch_path}"
            )
        if not patch_path.is_relative_to(output_dir):
            raise ValueError(
                f"Texture decision patch escapes the run directory: {patch_path}"
            )
        if file_sha256(patch_path) != record.decision_patch_sha256:
            raise ValueError(
                f"Texture decision patch changed after acceptance: {patch_path}"
            )
        patch = TextureDecisionPatch.model_validate(load_json(patch_path))
        if (
            patch.checkpoint_revision != record.checkpoint_revision
            or patch.action != record.action
        ):
            raise ValueError(
                f"Texture decision record does not match its patch: {patch_path}"
            )
        if (
            patch.request_digest != ledger.request_digest
            or patch.source_identity_digest != ledger.source_identity_digest
            or patch.plan_digest != ledger.plan_digest
        ):
            raise ValueError(
                f"Texture decision patch identity differs from its ledger: {patch_path}"
            )


def _verify_texture_decision_checkpoint_coverage(
    ledger: TextureDecisionLedger,
    *,
    output_dir: Path,
    checkpoint: TextureWorkflowCheckpoint,
) -> TextureDecisionPatch | None:
    """Bind decisions to progress and return one exact recoverable current patch."""

    expected_actions = tuple(
        action
        for item in checkpoint.progress
        if (action := _DECISION_ACTION_BY_PROGRESS_PHASE.get(item.phase)) is not None
    )
    current_action = checkpoint.next_action
    ledger_contains_current_orphan = (
        current_action != "done"
        and bool(ledger.records)
        and ledger.records[-1].checkpoint_revision == checkpoint.revision
        and ledger.records[-1].action == current_action
    )
    coverage_records = (
        ledger.records[:-1] if ledger_contains_current_orphan else ledger.records
    )
    coverage_actions = tuple(record.action for record in coverage_records)

    def grouped_counts(actions: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        groups: list[tuple[str, int]] = []
        for action in actions:
            if groups and groups[-1][0] == action:
                previous_action, count = groups[-1]
                groups[-1] = (previous_action, count + 1)
            else:
                groups.append((action, 1))
        return tuple(groups)

    expected_groups = grouped_counts(expected_actions)
    coverage_groups = grouped_counts(coverage_actions)
    allow_cancelled_current_group = (
        checkpoint.terminal_status == "cancelled" and current_action != "done"
    )
    if (
        allow_cancelled_current_group
        and len(coverage_groups) == len(expected_groups) + 1
        and coverage_groups[-1][0] == current_action
    ):
        comparable_groups = coverage_groups[:-1]
        cancelled_decision_count = coverage_groups[-1][1]
    else:
        comparable_groups = coverage_groups
        cancelled_decision_count = 0
    coverage_matches = len(comparable_groups) == len(expected_groups)
    for (recorded_action, recorded_count), (expected_action, expected_count) in zip(
        comparable_groups,
        expected_groups,
    ):
        if recorded_action != expected_action or recorded_count < expected_count:
            coverage_matches = False
            break
        cancelled_decision_count += recorded_count - expected_count
    cancellation_count = sum(item.phase == "cancelled" for item in checkpoint.progress)
    coverage_matches = (
        coverage_matches and cancelled_decision_count <= cancellation_count
    )
    # Checkpoint v1 made progress optional, so pre-existing terminal receipts can
    # legitimately have no transition history. New focused runs always persist at
    # least the planned event and therefore take the strict coverage path.
    if checkpoint.progress and not coverage_matches:
        raise ValueError(
            "Texture decision ledger does not cover the durable checkpoint "
            f"transitions: expected {expected_actions!r}, found {coverage_actions!r}"
        )

    referenced_paths = {
        Path(record.decision_patch_path).expanduser().resolve()
        for record in ledger.records
    }
    decision_dir = output_dir / "decisions"
    persisted_paths = (
        {
            path.expanduser().resolve()
            for path in decision_dir.glob("*-decision.json")
            if path.is_file()
        }
        if decision_dir.is_dir()
        else set()
    )
    unreferenced_paths = persisted_paths - referenced_paths
    current_patch_path: Path | None = None
    if current_action != "done":
        current_patch_path = _texture_decision_patch_path(
            output_dir,
            checkpoint_revision=checkpoint.revision,
            action=current_action,
        )
    if unreferenced_paths and unreferenced_paths != {current_patch_path}:
        raise ValueError(
            "Texture resume found decision patches outside the checkpoint-bound "
            f"ledger: {sorted(str(path) for path in unreferenced_paths)!r}"
        )

    candidate_current_path = (
        Path(ledger.records[-1].decision_patch_path).expanduser().resolve()
        if ledger_contains_current_orphan
        else next(iter(unreferenced_paths), None)
    )
    if candidate_current_path is not None:
        current_patch = TextureDecisionPatch.model_validate(
            load_json(candidate_current_path)
        )
        validate_texture_decision_patch(
            current_patch,
            checkpoint=checkpoint,
            output_dir=output_dir,
        )
        return current_patch
    return None


def verify_texture_decision_ledger(
    ledger_path: str | Path,
    *,
    output_dir: str | Path,
    request_digest: str,
    source_identity_digest: str,
    plan_digest: str,
    require_final_decision: bool,
    checkpoint: TextureWorkflowCheckpoint | None = None,
) -> TextureDecisionLedger:
    """Validate the complete canonical decision history before publication."""

    root = Path(output_dir).expanduser().resolve()
    canonical_path = texture_decision_ledger_path(root)
    resolved_path = Path(ledger_path).expanduser().resolve()
    if resolved_path != canonical_path:
        raise ValueError(
            f"Texture decision ledger must use the canonical path: {canonical_path}"
        )
    ledger = TextureDecisionLedger.model_validate(load_json(resolved_path))
    expected_identity = (
        request_digest,
        source_identity_digest,
        plan_digest,
    )
    if (
        ledger.request_digest,
        ledger.source_identity_digest,
        ledger.plan_digest,
    ) != expected_identity:
        raise ValueError("Texture decision ledger identity does not match finalization")
    if not ledger.records:
        raise ValueError("Texture decision ledger must contain at least one decision")
    if require_final_decision and ledger.records[-1].action != "finalize":
        raise ValueError("Texture publication requires a final decision patch")
    _verify_texture_decision_records(ledger, output_dir=root)
    if checkpoint is not None:
        _verify_texture_decision_checkpoint_coverage(
            ledger,
            output_dir=root,
            checkpoint=checkpoint,
        )
    return ledger


def verify_texture_resume_decision_state(
    checkpoint: TextureWorkflowCheckpoint,
    *,
    output_dir: str | Path,
) -> TextureDecisionPatch | None:
    """Validate checkpoint coverage and return one exact reusable current patch."""

    root = Path(output_dir).expanduser().resolve()
    ledger_path = texture_decision_ledger_path(root)
    if ledger_path.is_file():
        ledger = verify_texture_decision_ledger(
            ledger_path,
            output_dir=root,
            request_digest=checkpoint.request_digest,
            source_identity_digest=checkpoint.source_identity_digest,
            plan_digest=checkpoint.plan_digest,
            require_final_decision=checkpoint.next_action == "done",
        )
    else:
        ledger = TextureDecisionLedger(
            request_digest=checkpoint.request_digest,
            source_identity_digest=checkpoint.source_identity_digest,
            plan_digest=checkpoint.plan_digest,
        )
    return _verify_texture_decision_checkpoint_coverage(
        ledger,
        output_dir=root,
        checkpoint=checkpoint,
    )


def record_texture_decision_patch(
    patch: TextureDecisionPatch,
    *,
    output_dir: str | Path,
    workflow_lock: FileLock | None = None,
) -> Path:
    """Canonically persist and append one already-validated decision patch."""

    root = Path(output_dir).expanduser().resolve()
    canonical_lock_path = root / ".texture_workflow.lock"
    if (
        workflow_lock is not None
        and Path(workflow_lock.lock_file) != canonical_lock_path
    ):
        raise ValueError(
            "Texture decision workflow_lock must protect the canonical output lock"
        )
    lock = workflow_lock or FileLock(str(canonical_lock_path))
    with lock:
        return _record_texture_decision_patch_locked(patch, output_dir=root)


def _record_texture_decision_patch_locked(
    patch: TextureDecisionPatch,
    *,
    output_dir: Path,
) -> Path:
    """Persist one patch while the run's checkpoint/decision lock is held."""

    root = output_dir
    ledger_path = texture_decision_ledger_path(root)
    if ledger_path.is_file():
        ledger = verify_texture_decision_ledger(
            ledger_path,
            output_dir=root,
            request_digest=patch.request_digest,
            source_identity_digest=patch.source_identity_digest,
            plan_digest=patch.plan_digest,
            require_final_decision=False,
        )
    else:
        ledger = TextureDecisionLedger(
            request_digest=patch.request_digest,
            source_identity_digest=patch.source_identity_digest,
            plan_digest=patch.plan_digest,
        )

    existing = next(
        (
            record
            for record in ledger.records
            if record.checkpoint_revision == patch.checkpoint_revision
        ),
        None,
    )
    if existing is not None:
        existing_patch = TextureDecisionPatch.model_validate(
            load_json(existing.decision_patch_path)
        )
        # Strict full-model equality is intentional: every patch field is part
        # of the idempotency identity and the model forbids unknown fields.
        if existing_patch != patch:
            raise ValueError(
                "Texture checkpoint revision already has a different decision patch"
            )
        return ledger_path

    patch_path = _texture_decision_patch_path(
        root,
        checkpoint_revision=patch.checkpoint_revision,
        action=patch.action,
    )
    competing_paths = {
        candidate.expanduser().resolve()
        for candidate in patch_path.parent.glob(
            f"{patch.checkpoint_revision:04d}-*-decision.json"
        )
        if candidate.is_file()
    } - {patch_path}
    if competing_paths:
        raise ValueError(
            "Texture checkpoint revision has conflicting orphaned decision patches: "
            f"{sorted(str(path) for path in competing_paths)!r}"
        )
    if patch_path.is_file():
        orphaned_patch = TextureDecisionPatch.model_validate(load_json(patch_path))
        if orphaned_patch != patch:
            raise ValueError(
                "Texture checkpoint revision has a different orphaned decision patch"
            )
    else:
        atomic_write_json(patch_path, patch)
    record = TextureDecisionRecord(
        sequence=len(ledger.records) + 1,
        checkpoint_revision=patch.checkpoint_revision,
        action=patch.action,
        decision_patch_path=str(patch_path),
        decision_patch_sha256=file_sha256(patch_path),
    )
    updated = ledger.model_copy(update={"records": (*ledger.records, record)})
    updated = TextureDecisionLedger.model_validate(
        updated.model_dump(mode="python", round_trip=True)
    )
    atomic_write_json(ledger_path, updated)
    return ledger_path


__all__ = [
    "TEXTURE_DECISION_LEDGER_SCHEMA_VERSION",
    "TEXTURE_DECISION_PATCH_SCHEMA_VERSION",
    "TEXTURE_STEP_OBSERVATION_SCHEMA_VERSION",
    "TextureDecisionLedger",
    "TextureDecisionOperation",
    "TextureDecisionPatch",
    "TextureDecisionRecord",
    "TextureStepObservation",
    "build_texture_step_observation",
    "record_texture_decision_patch",
    "texture_decision_ledger_path",
    "texture_checkpoint_decision_digest",
    "validate_texture_decision_patch",
    "verify_texture_decision_ledger",
    "verify_texture_resume_decision_state",
]
