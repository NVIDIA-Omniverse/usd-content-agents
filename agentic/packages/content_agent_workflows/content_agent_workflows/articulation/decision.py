# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-authored, evidence-bound decisions for Articulation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, Self

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256

from .models import (
    ArticulationRunState,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    Stage2ArticulationCandidate,
    Stage2CandidateDocument,
)

ARTICULATION_DECISION_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-decision-patch.v1"
] = "content-agent-workflows.articulation-decision-patch.v1"
ARTICULATION_DECISION_LEDGER_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-decision-ledger.v1"
] = "content-agent-workflows.articulation-decision-ledger.v1"
ARTICULATION_STEP_OBSERVATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-step-observation.v1"
] = "content-agent-workflows.articulation-step-observation.v1"
SKILL_ROUTED_DECISION_METADATA_KEY: Final = (
    "content_agent_workflows.skill_routed_decision_required"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArticulationCandidateDecision(_StrictModel):
    """One explicit child review of an inferred articulation candidate."""

    candidate_id: str = Field(min_length=1)
    decision: Literal["accept", "reject"]
    replacement_candidate: Stage2ArticulationCandidate | None = None
    rationale: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_paths: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_replacement(self) -> Self:
        if (
            self.replacement_candidate is not None
            and self.replacement_candidate.candidate_id != self.candidate_id
        ):
            raise ValueError("replacement candidate ID must remain immutable")
        if self.decision == "reject" and self.replacement_candidate is not None:
            raise ValueError("rejected candidates cannot carry an authoring edit")
        return self


class ArticulationDecisionPatch(_StrictModel):
    """Complete child decision over exact candidate and Scene evidence."""

    schema_version: Literal[
        "content-agent-workflows.articulation-decision-patch.v1"
    ] = ARTICULATION_DECISION_PATCH_SCHEMA_VERSION
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_evidence_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    checkpoint_revision: int = Field(ge=0)
    decisions: tuple[ArticulationCandidateDecision, ...] = Field(min_length=1)
    evidence_summary: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def _validate_unique_candidates(self) -> Self:
        candidate_ids = tuple(item.candidate_id for item in self.decisions)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("articulation decision candidate IDs must be unique")
        return self


class ArticulationStepObservation(_StrictModel):
    """Compact review packet shared by standalone and embedded reasoning loops."""

    schema_version: Literal[
        "content-agent-workflows.articulation-step-observation.v1"
    ] = ARTICULATION_STEP_OBSERVATION_SCHEMA_VERSION
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scene_evidence_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    checkpoint_revision: int = Field(ge=0)
    candidate_ids: tuple[str, ...] = Field(min_length=1)
    candidate_document_path: str
    scene_evidence_path: str | None = None
    evidence_sha256_by_path: dict[str, str]
    decision_patch_path: str
    required_operations: tuple[str, ...] = (
        "inspect_geometry_hierarchy",
        "inspect_existing_articulation",
        "review_renders_picks_properties",
        "propose_candidates",
        "select_reject_or_edit",
    )


class ArticulationDecisionLedger(_StrictModel):
    """Immutable binding from an original proposal to the child-reviewed set."""

    schema_version: Literal[
        "content-agent-workflows.articulation-decision-ledger.v1"
    ] = ARTICULATION_DECISION_LEDGER_SCHEMA_VERSION
    prepared_checkpoint_revision: int = Field(ge=0)
    original_candidate_document: ArtifactBinding
    decision_patch: ArtifactBinding
    reviewed_candidate_document: ArtifactBinding


def _verified_path(binding: ArtifactBinding, *, label: str) -> Path:
    path = Path(binding.path).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != binding.sha256:
        raise ValueError(f"{label} is missing or changed: {path}")
    return path


def build_articulation_step_observation(
    state: ArticulationRunState,
    *,
    output_dir: str | Path,
) -> ArticulationStepObservation:
    """Build the exact candidate/evidence packet presented to a reasoning loop."""

    root = Path(output_dir).expanduser().resolve()
    if state.candidate_document is None:
        raise ValueError("Articulation candidate evidence is not checkpointed")
    candidate_path = _verified_path(
        state.candidate_document,
        label="Articulation candidate document",
    )
    candidate_document = Stage2CandidateDocument.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )
    scene_path = None
    evidence = {
        str(_verified_path(state.request, label="Articulation request")): (
            state.request.sha256
        ),
        str(candidate_path): state.candidate_document.sha256,
    }
    if state.scene_evidence is not None:
        resolved = _verified_path(
            state.scene_evidence,
            label="Articulation Scene evidence",
        )
        scene_path = str(resolved)
        evidence[scene_path] = state.scene_evidence.sha256
    return ArticulationStepObservation(
        request_sha256=state.request.sha256,
        source_sha256=state.source_sha256,
        source_dependency_bundle_sha256=state.source_dependency_bundle_sha256,
        candidate_document_sha256=state.candidate_document.sha256,
        scene_evidence_sha256=(
            state.scene_evidence.sha256 if state.scene_evidence is not None else None
        ),
        checkpoint_revision=state.revision,
        candidate_ids=candidate_document.candidate_ids,
        candidate_document_path=str(candidate_path),
        scene_evidence_path=scene_path,
        evidence_sha256_by_path=dict(sorted(evidence.items())),
        decision_patch_path=str(root / "articulation_decision_patch.json"),
    )


def _validate_articulation_decision_patch(
    patch: ArticulationDecisionPatch,
    *,
    observation: ArticulationStepObservation,
    candidate_document: Stage2CandidateDocument,
    request: ArticulationWorkflowRequest,
    output_dir: Path,
) -> Stage2CandidateDocument:
    expected_bindings = {
        "request_sha256": observation.request_sha256,
        "source_sha256": observation.source_sha256,
        "source_dependency_bundle_sha256": (
            observation.source_dependency_bundle_sha256
        ),
        "candidate_document_sha256": observation.candidate_document_sha256,
        "scene_evidence_sha256": observation.scene_evidence_sha256,
        "checkpoint_revision": observation.checkpoint_revision,
    }
    actual_bindings = patch.model_dump(mode="python", include=set(expected_bindings))
    if actual_bindings != expected_bindings:
        mismatches = sorted(
            key
            for key, value in expected_bindings.items()
            if actual_bindings.get(key) != value
        )
        raise ValueError(
            "Articulation decision patch is stale or cross-run: "
            + ", ".join(mismatches)
        )
    decision_ids = tuple(item.candidate_id for item in patch.decisions)
    if decision_ids != candidate_document.candidate_ids:
        raise ValueError(
            "Articulation decisions must cover every candidate exactly once in order"
        )
    candidate_by_id = candidate_document.candidate_by_id()
    reviewed_candidates: list[Stage2ArticulationCandidate] = []
    for item in patch.decisions:
        original = candidate_by_id[item.candidate_id]
        replacement = item.replacement_candidate or original
        if item.replacement_candidate is not None:
            # This explicit allowlist is a fail-closed review boundary. Update it
            # only when the versioned candidate contract intentionally makes a
            # field reviewer-editable, with matching immutability coverage.
            editable_fields = {
                "confidence",
                "parent_hint",
                "child_hint",
                "component_name",
                "component_type",
                "evidence",
            }
            original_payload = original.model_dump(mode="json", round_trip=True)
            replacement_payload = replacement.model_dump(mode="json", round_trip=True)
            all_fields = set(original_payload) | set(replacement_payload)
            changed_fields = tuple(
                field_name
                for field_name in sorted(all_fields - editable_fields)
                if field_name not in original_payload
                or field_name not in replacement_payload
                or json.dumps(
                    replacement_payload[field_name],
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                != json.dumps(
                    original_payload[field_name],
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            if changed_fields:
                raise ValueError(
                    f"Candidate {item.candidate_id} edit changes immutable authoring "
                    "or evidence fields: " + ", ".join(changed_fields)
                )
        if item.decision == "accept" and (
            not replacement.is_articulation_v1_authorable
            or replacement.articulation_v1_type not in request.allowed_motion_types
        ):
            raise ValueError(
                f"Candidate {item.candidate_id} is unsupported or not native-ready"
            )
        for raw_path in item.evidence_paths:
            evidence_path = Path(raw_path).expanduser().resolve()
            if not evidence_path.is_relative_to(output_dir):
                raise ValueError(
                    f"Articulation decision evidence escapes the run: {evidence_path}"
                )
            expected_digest = observation.evidence_sha256_by_path.get(
                str(evidence_path)
            )
            if expected_digest is None or file_sha256(evidence_path) != expected_digest:
                raise ValueError(
                    f"Articulation decision evidence is unbound or changed: {evidence_path}"
                )
        reviewed_candidates.append(replacement)
    payload = candidate_document.model_dump(mode="python", round_trip=True)
    payload["candidates"] = tuple(reviewed_candidates)
    return Stage2CandidateDocument.model_validate(payload)


def apply_articulation_decision_patch(
    output_dir: str | Path,
    patch: ArticulationDecisionPatch,
    *,
    workflow_lock: FileLock | None = None,
) -> ArticulationDecisionLedger:
    """Validate and durably bind one patch before authoring can continue."""

    root = Path(output_dir).expanduser().resolve()
    canonical_lock_path = root / ".articulation-workflow.lock"
    if (
        workflow_lock is not None
        and Path(workflow_lock.lock_file) != canonical_lock_path
    ):
        raise ValueError(
            "Articulation decision workflow_lock must protect the canonical output lock"
        )
    lock = workflow_lock or FileLock(str(canonical_lock_path))
    with lock:
        state = ArticulationRunState.model_validate_json(
            (root / "checkpoint.json").read_text(encoding="utf-8")
        )
        if state.phase != "needs_review":
            raise ValueError(
                "Articulation decisions may be applied only at needs_review"
            )
        observation = build_articulation_step_observation(state, output_dir=root)
        request = ArticulationWorkflowRequest.model_validate_json(
            _verified_path(state.request, label="Articulation request").read_text(
                encoding="utf-8"
            )
        )
        candidate_path = _verified_path(
            state.candidate_document,
            label="Articulation candidate document",
        )
        candidate_document = Stage2CandidateDocument.model_validate_json(
            candidate_path.read_text(encoding="utf-8")
        )
        reviewed = _validate_articulation_decision_patch(
            patch,
            observation=observation,
            candidate_document=candidate_document,
            request=request,
            output_dir=root,
        )
        patch_path = root / "articulation_decision_patch.json"
        reviewed_path = root / "agent_reviewed_articulation_candidates.json"
        ledger_path = root / "articulation_decision_ledger.json"
        if ledger_path.exists():
            existing = load_articulation_decision_ledger(root, state=state)
            if existing is None:
                raise ValueError("Articulation decision ledger could not be loaded")
            existing_patch, existing_reviewed = existing
            if existing_patch != patch:
                raise ValueError("A conflicting Articulation decision patch exists")
            if existing_reviewed != reviewed:
                raise ValueError(
                    "Conflicting agent-reviewed Articulation candidates exist"
                )
            return ArticulationDecisionLedger.model_validate_json(
                ledger_path.read_text(encoding="utf-8")
            )
        if patch_path.exists():
            existing_patch = ArticulationDecisionPatch.model_validate_json(
                patch_path.read_text(encoding="utf-8")
            )
            if existing_patch != patch:
                raise ValueError("A conflicting Articulation decision patch exists")
        else:
            atomic_write_json(patch_path, patch.model_dump(mode="json"))
        if reviewed_path.exists():
            existing_reviewed = Stage2CandidateDocument.model_validate_json(
                reviewed_path.read_text(encoding="utf-8")
            )
            if existing_reviewed != reviewed:
                raise ValueError(
                    "Conflicting agent-reviewed Articulation candidates exist"
                )
        else:
            atomic_write_json(reviewed_path, reviewed.model_dump(mode="json"))
        ledger = ArticulationDecisionLedger(
            prepared_checkpoint_revision=state.revision,
            original_candidate_document=state.candidate_document,
            decision_patch=ArtifactBinding(
                path=str(patch_path),
                sha256=file_sha256(patch_path),
            ),
            reviewed_candidate_document=ArtifactBinding(
                path=str(reviewed_path),
                sha256=file_sha256(reviewed_path),
            ),
        )
        atomic_write_json(ledger_path, ledger.model_dump(mode="json"))
        return ledger


def load_articulation_decision_ledger(
    output_dir: str | Path,
    *,
    state: ArticulationRunState,
) -> tuple[ArticulationDecisionPatch, Stage2CandidateDocument] | None:
    """Load and reverify the immutable child decision and reviewed candidates."""

    root = Path(output_dir).expanduser().resolve()
    ledger_path = root / "articulation_decision_ledger.json"
    if not ledger_path.is_file():
        return None
    ledger = ArticulationDecisionLedger.model_validate_json(
        ledger_path.read_text(encoding="utf-8")
    )
    if state.candidate_document != ledger.original_candidate_document:
        raise ValueError("Articulation decision ledger candidate binding changed")
    patch_path = _verified_path(ledger.decision_patch, label="Articulation decision")
    reviewed_path = _verified_path(
        ledger.reviewed_candidate_document,
        label="Agent-reviewed articulation candidates",
    )
    canonical_reviewed_path = root / "agent_reviewed_articulation_candidates.json"
    if reviewed_path != canonical_reviewed_path:
        raise ValueError(
            "Articulation decision ledger reviewed candidates path is not canonical"
        )
    patch = ArticulationDecisionPatch.model_validate_json(
        patch_path.read_text(encoding="utf-8")
    )
    if patch.checkpoint_revision != ledger.prepared_checkpoint_revision:
        raise ValueError("Articulation decision ledger revision binding changed")
    if (
        patch.request_sha256 != state.request.sha256
        or patch.source_sha256 != state.source_sha256
        or patch.source_dependency_bundle_sha256
        != state.source_dependency_bundle_sha256
        or patch.candidate_document_sha256 != ledger.original_candidate_document.sha256
        or patch.scene_evidence_sha256
        != (state.scene_evidence.sha256 if state.scene_evidence else None)
    ):
        raise ValueError("Articulation decision ledger no longer matches the run")
    reviewed = Stage2CandidateDocument.model_validate_json(
        reviewed_path.read_text(encoding="utf-8")
    )
    request = ArticulationWorkflowRequest.model_validate_json(
        _verified_path(state.request, label="Articulation request").read_text(
            encoding="utf-8"
        )
    )
    candidate_path = _verified_path(
        ledger.original_candidate_document,
        label="Articulation candidate document",
    )
    candidate_document = Stage2CandidateDocument.model_validate_json(
        candidate_path.read_text(encoding="utf-8")
    )
    prepared_state = state.model_copy(
        update={"revision": ledger.prepared_checkpoint_revision}
    )
    observation = build_articulation_step_observation(
        prepared_state,
        output_dir=root,
    )
    expected_reviewed = _validate_articulation_decision_patch(
        patch,
        observation=observation,
        candidate_document=candidate_document,
        request=request,
        output_dir=root,
    )
    if reviewed != expected_reviewed:
        raise ValueError(
            "Agent-reviewed Articulation candidates do not match the validated patch"
        )
    return patch, expected_reviewed


__all__ = [
    "ARTICULATION_DECISION_LEDGER_SCHEMA_VERSION",
    "ARTICULATION_DECISION_PATCH_SCHEMA_VERSION",
    "ARTICULATION_STEP_OBSERVATION_SCHEMA_VERSION",
    "SKILL_ROUTED_DECISION_METADATA_KEY",
    "ArticulationCandidateDecision",
    "ArticulationDecisionLedger",
    "ArticulationDecisionPatch",
    "ArticulationStepObservation",
    "apply_articulation_decision_patch",
    "build_articulation_step_observation",
    "load_articulation_decision_ledger",
]
