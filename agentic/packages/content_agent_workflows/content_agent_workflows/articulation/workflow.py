# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable review, resume, authoring, and readback for articulation-v1."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import stat
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import BaseModel, ValidationError

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
)

from .client import (
    ArticulationWorkflowClient,
    CancelChecker,
    JointAgentInferenceTerminalError,
    _ArticulationEvidenceBindingError,
)
from .decision import (
    SKILL_ROUTED_DECISION_METADATA_KEY,
    load_articulation_decision_ledger,
)
from .finalizer import (
    articulation_workflow_progress_payload,
    verify_articulation_workflow_summary,
    write_articulation_workflow_summary,
)
from .models import (
    TERMINAL_ARTICULATION_PHASES,
    ArticulationAuthoringRequest,
    ArticulationAuthoringResult,
    ArticulationFinalizationResult,
    ArticulationInferenceResult,
    ArticulationReviewEntry,
    ArticulationReviewReceipt,
    ArticulationRunState,
    ArticulationStateTransition,
    ArticulationValidationResult,
    ArticulationWorkflowMode,
    ArticulationWorkflowPhase,
    ArticulationWorkflowRequest,
    ArtifactBinding,
    MembershipDispositionDocument,
    ReviewDecision,
    Stage2CandidateDocument,
)
from .scene_evidence import (
    ArticulationSceneEvidenceCollector,
    ArticulationSceneEvidenceResult,
    verify_articulation_scene_evidence,
)

PhaseBoundaryHook = Callable[[str], None]
_SCENE_EVIDENCE_REQUIRED_METADATA_KEY = (
    "content_agent_workflows.scene_evidence_required"
)
_LOGGER = logging.getLogger(__name__)
_DIGEST_CHUNK_SIZE = 1024 * 1024


class ArticulationWorkflowError(RuntimeError):
    """Raised when durable articulation evidence fails closed."""


class ArticulationWorkflowInterrupted(RuntimeError):
    """Testable process-interruption signal that intentionally remains resumable."""


class _ArticulationConfigurationDriftError(ArticulationWorkflowError):
    """Raised when a resumable checkpoint uses a different backend configuration."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _payload(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _write_once_json(
    path: Path,
    value: BaseModel | Mapping[str, Any],
    *,
    label: str,
) -> ArtifactBinding:
    expected = _payload(value)
    if not path.exists():
        atomic_write_json(path, expected)
    try:
        existing, binding = _read_json_artifact_binding(path, label=label)
    except (OSError, ValueError, ArticulationWorkflowError) as exc:
        raise ArticulationWorkflowError(
            f"Cannot recover {label} at {path}: {exc}"
        ) from exc
    if existing != expected:
        raise ArticulationWorkflowError(
            f"Existing {label} conflicts with the current workflow: {path}"
        )
    return binding


def _scene_manifest_matches(
    path: Path,
    result: ArticulationSceneEvidenceResult,
) -> bool:
    return _matching_scene_manifest_binding(path, result) is not None


def _matching_scene_manifest_binding(
    path: Path,
    result: ArticulationSceneEvidenceResult,
) -> ArtifactBinding | None:
    try:
        payload, binding = _read_json_artifact_binding(
            path,
            label="Scene evidence manifest",
        )
    except (OSError, ValueError, ArticulationWorkflowError):
        return None
    if payload != result.model_dump(mode="json"):
        return None
    return binding


def _discard_uncommitted_scene_collection(
    collector: ArticulationSceneEvidenceCollector,
    result: ArticulationSceneEvidenceResult | None,
    *,
    evidence_root: Path,
    original_error: BaseException,
) -> None:
    if result is None:
        return
    discard = getattr(collector, "discard_uncommitted_collection", None)
    if not callable(discard):
        return
    try:
        discard(
            result,
            evidence_root=evidence_root,
        )
    except Exception as cleanup_error:
        original_error.add_note(
            f"Could not discard the uncommitted Scene collection: {cleanup_error}"
        )
        _LOGGER.warning(
            "Failed to discard an uncommitted Scene collection.",
            exc_info=True,
        )


def _reclaim_orphaned_scene_collections(
    collector: ArticulationSceneEvidenceCollector,
    *,
    evidence_root: Path,
) -> None:
    reclaim = getattr(collector, "reclaim_orphaned_collections", None)
    if callable(reclaim):
        reclaim(evidence_root=evidence_root)


def _release_committed_scene_collection(
    collector: ArticulationSceneEvidenceCollector,
    result: ArticulationSceneEvidenceResult,
    *,
    manifest_path: Path,
) -> None:
    release = getattr(collector, "release_committed_collection", None)
    if not callable(release):
        return
    try:
        release(result, manifest_path=manifest_path)
    except Exception:
        _LOGGER.warning(
            "Failed to release a committed Scene collection capability.",
            exc_info=True,
        )


def _artifact_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns if os.name != "nt" else 0,
    )


def _read_json_artifact_binding(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], ArtifactBinding]:
    """Parse and bind one immutable JSON byte read."""

    resolved_path = path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = resolved_path.absolute()
    file_descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        file_descriptor = os.open(resolved_path, flags)
        opened_metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ArticulationWorkflowError(
                f"{label} is not a regular file: {resolved_path}"
            )
        with os.fdopen(os.dup(file_descriptor), "rb") as stream:
            payload = stream.read()
        final_descriptor_metadata = os.fstat(file_descriptor)
        current_path_metadata = os.stat(resolved_path, follow_symlinks=False)
        if _artifact_stat_signature(
            final_descriptor_metadata
        ) != _artifact_stat_signature(opened_metadata) or _artifact_stat_signature(
            current_path_metadata
        ) != _artifact_stat_signature(opened_metadata):
            raise ArticulationWorkflowError(
                f"{label} changed while being read: {resolved_path}"
            )
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} must contain a JSON object")
        return (
            parsed,
            ArtifactBinding(
                path=str(resolved_path),
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
        )
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _read_and_verify_binding(
    binding: ArtifactBinding,
    *,
    label: str,
    capture_bytes: bool,
) -> tuple[Path, bytes | None]:
    """Hash one pinned file and optionally return those exact bytes for parsing."""

    path = Path(binding.path).expanduser()
    if not path.is_absolute():
        path = path.absolute()
    file_descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        file_descriptor = os.open(path, flags)
        opened_metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ArticulationWorkflowError(f"{label} is not a regular file: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if capture_bytes else None
        with os.fdopen(os.dup(file_descriptor), "rb") as stream:
            while chunk := stream.read(_DIGEST_CHUNK_SIZE):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        final_descriptor_metadata = os.fstat(file_descriptor)
        current_path_metadata = os.stat(path, follow_symlinks=False)
        if _artifact_stat_signature(
            final_descriptor_metadata
        ) != _artifact_stat_signature(opened_metadata) or _artifact_stat_signature(
            current_path_metadata
        ) != _artifact_stat_signature(opened_metadata):
            raise ArticulationWorkflowError(f"{label} changed while being read: {path}")
        actual = digest.hexdigest()
        if actual != binding.sha256:
            raise ArticulationWorkflowError(
                f"{label} digest mismatch: expected {binding.sha256}, got {actual}"
            )
        payload = b"".join(chunks) if chunks is not None else None
        return path, payload
    except ArticulationWorkflowError:
        raise
    except FileNotFoundError as exc:
        raise ArticulationWorkflowError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise ArticulationWorkflowError(
            f"Cannot read {label} at {path}: {exc}"
        ) from exc
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _verify_binding(binding: ArtifactBinding, *, label: str) -> Path:
    path, _ = _read_and_verify_binding(
        binding,
        label=label,
        capture_bytes=False,
    )
    return path


def _read_verified_binding_bytes(
    binding: ArtifactBinding,
    *,
    label: str,
) -> tuple[Path, bytes]:
    path, payload = _read_and_verify_binding(
        binding,
        label=label,
        capture_bytes=True,
    )
    assert payload is not None
    return path, payload


def _require_scene_evidence_root(path: Path) -> Path:
    raw_path = path.expanduser()
    if not raw_path.is_absolute():
        raise ArticulationWorkflowError(
            "Scene evidence directory must be an absolute path."
        )
    if raw_path.is_symlink():
        raise ArticulationWorkflowError(
            "Scene evidence directory must not be a symbolic link."
        )
    if raw_path.exists() and not raw_path.is_dir():
        raise ArticulationWorkflowError(
            "Scene evidence path exists but is not a directory."
        )
    resolved_path = raw_path.resolve()
    if resolved_path != raw_path:
        raise ArticulationWorkflowError(
            "Scene evidence directory must resolve without traversing symlinks."
        )
    return resolved_path


def _verify_artifact_digest(
    path_value: str | Path,
    expected_sha256: str | None,
    *,
    label: str,
) -> Path:
    if expected_sha256 is None:
        raise ArticulationWorkflowError(f"{label} digest binding is missing.")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ArticulationWorkflowError(f"{label} is missing: {path}")
    try:
        actual = file_sha256(path)
    except OSError as exc:
        raise ArticulationWorkflowError(
            f"Cannot read {label} at {path}: {exc}"
        ) from exc
    if actual != expected_sha256:
        raise ArticulationWorkflowError(
            f"{label} digest mismatch: expected {expected_sha256}, got {actual}"
        )
    return path


def _binding(path: Path) -> ArtifactBinding:
    return ArtifactBinding(path=str(path.resolve()), sha256=file_sha256(path))


def _transition(
    state: ArticulationRunState,
    to_phase: ArticulationWorkflowPhase,
    reason: str,
    **updates: Any,
) -> ArticulationRunState:
    transition = ArticulationStateTransition(
        timestamp=_timestamp(),
        from_phase=state.phase,
        to_phase=to_phase,
        reason=reason,
    )
    return state.model_copy(
        update={
            **updates,
            "phase": to_phase,
            "transitions": (*state.transitions, transition),
        }
    )


def _persist_state(
    output_dir: Path, state: ArticulationRunState
) -> ArticulationRunState:
    updated = state.model_copy(update={"revision": state.revision + 1})
    atomic_write_json(output_dir / "checkpoint.json", updated)
    atomic_write_json(
        output_dir / "workflow_progress.json",
        articulation_workflow_progress_payload(updated),
    )
    return updated


def _load_state(path: Path) -> ArticulationRunState:
    try:
        payload, _ = _read_json_artifact_binding(
            path,
            label="articulation checkpoint",
        )
        return ArticulationRunState.model_validate(payload)
    except (OSError, ValueError, ValidationError) as exc:
        raise ArticulationWorkflowError(
            f"Invalid articulation checkpoint at {path}: {exc}"
        ) from exc


def _normalize_request(
    request: ArticulationWorkflowRequest,
) -> ArticulationWorkflowRequest:
    source = Path(request.source_asset).expanduser().resolve()
    if not source.is_file():
        raise ArticulationWorkflowError(
            f"Articulation source asset is not a file: {source}"
        )
    output_dir = request.output_dir.expanduser().resolve()
    return request.model_copy(
        update={"source_asset": str(source), "output_dir": output_dir}
    )


def _bind_scene_evidence_requirement(
    request: ArticulationWorkflowRequest,
    *,
    collector_enabled: bool,
) -> ArticulationWorkflowRequest:
    """Persist collector use in the write-once, digest-bound request artifact."""

    metadata = dict(request.metadata)
    marker = metadata.get(_SCENE_EVIDENCE_REQUIRED_METADATA_KEY)
    if marker is not None and not isinstance(marker, bool):
        raise ArticulationWorkflowError(
            "Scene evidence requirement metadata must be a boolean."
        )
    if collector_enabled:
        metadata[_SCENE_EVIDENCE_REQUIRED_METADATA_KEY] = True
    if metadata == request.metadata:
        return request
    return request.model_copy(update={"metadata": metadata})


def _request_requires_scene_evidence(
    request: ArticulationWorkflowRequest,
) -> bool:
    marker = request.metadata.get(_SCENE_EVIDENCE_REQUIRED_METADATA_KEY, False)
    if not isinstance(marker, bool):
        raise ArticulationWorkflowError(
            "Scene evidence requirement metadata must be a boolean."
        )
    return marker


def _source_identity(source_asset: str) -> tuple[str, str]:
    from world_understanding.functions.physics.joint_rigger import (
        identify_usd_artifact,
    )

    source = Path(source_asset).expanduser().resolve()
    try:
        identity = identify_usd_artifact(source, uri=source.as_uri())
    except Exception as exc:
        raise ArticulationWorkflowError(
            f"Cannot establish composed source identity for {source}: {exc}"
        ) from exc
    dependency_sha256 = identity.dependency_bundle_sha256
    if dependency_sha256 is None:
        raise ArticulationWorkflowError(
            f"Composed source identity lacks a dependency bundle digest: {source}"
        )
    return identity.root_sha256, dependency_sha256


def _load_inference(
    binding: ArtifactBinding,
    *,
    expected_configuration_sha256: str,
) -> ArticulationInferenceResult:
    path, payload = _read_verified_binding_bytes(
        binding,
        label="inference result",
    )
    try:
        result = ArticulationInferenceResult.model_validate(json.loads(payload))
    except (OSError, ValueError, ValidationError) as exc:
        raise ArticulationWorkflowError(
            f"Invalid articulation inference result at {path}: {exc}"
        ) from exc
    if result.backend_configuration_sha256 != expected_configuration_sha256:
        raise ArticulationWorkflowError(
            "Inference result backend configuration digest does not match "
            "the current adapter configuration."
        )
    _verify_inference_side_artifacts(result)
    return result


def _verify_inference_side_artifacts(
    result: ArticulationInferenceResult,
) -> None:
    if result.predictions_path is not None:
        _verify_artifact_digest(
            result.predictions_path,
            result.predictions_sha256,
            label="Joint Agent predictions",
        )
    if result.report_path is not None:
        _verify_artifact_digest(
            result.report_path,
            result.report_sha256,
            label="Joint Agent candidate report",
        )


def _load_scene_evidence(
    binding: ArtifactBinding,
    *,
    expected_manifest_path: Path,
    request: ArticulationWorkflowRequest,
    request_sha256: str,
    source_sha256: str,
    source_dependency_bundle_sha256: str,
    candidate_document: Stage2CandidateDocument,
    candidate_document_sha256: str,
    collector_configuration_sha256: str,
) -> ArticulationSceneEvidenceResult:
    evidence_root = _require_scene_evidence_root(expected_manifest_path.parent)
    expected_path = evidence_root / expected_manifest_path.name
    raw_binding_path = Path(binding.path).expanduser()
    if raw_binding_path.is_symlink() or expected_path.is_symlink():
        raise ArticulationWorkflowError(
            "Scene evidence manifest must not be a symbolic link."
        )
    path, payload = _read_verified_binding_bytes(
        binding,
        label="Scene evidence manifest",
    )
    if path != expected_path:
        raise ArticulationWorkflowError(
            f"Scene evidence manifest path does not match the workflow output: {path}"
        )
    try:
        result = ArticulationSceneEvidenceResult.model_validate(json.loads(payload))
        verify_articulation_scene_evidence(
            result,
            request=request,
            request_sha256=request_sha256,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            candidate_document=candidate_document,
            candidate_document_sha256=candidate_document_sha256,
            collector_configuration_sha256=collector_configuration_sha256,
            evidence_root=evidence_root,
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ArticulationWorkflowError(
            f"Invalid Scene articulation evidence at {path}: {exc}"
        ) from exc
    return result


def _load_authoring(binding: ArtifactBinding) -> ArticulationAuthoringResult:
    path, payload = _read_verified_binding_bytes(
        binding,
        label="authoring result",
    )
    try:
        result = ArticulationAuthoringResult.model_validate(json.loads(payload))
    except (OSError, ValueError, ValidationError) as exc:
        raise ArticulationWorkflowError(
            f"Invalid articulation authoring result at {path}: {exc}"
        ) from exc
    for artifact_path, expected_sha256, label in (
        (
            result.diagnostics_path,
            result.diagnostics_sha256,
            "Joint Rigger diagnostics",
        ),
        (
            result.joint_rigger_result_path,
            result.joint_rigger_result_sha256,
            "Joint Rigger result",
        ),
    ):
        if artifact_path is not None:
            _verify_artifact_digest(
                artifact_path,
                expected_sha256,
                label=label,
            )
    return result


def _load_validation(binding: ArtifactBinding) -> ArticulationValidationResult:
    path, payload = _read_verified_binding_bytes(
        binding,
        label="validation result",
    )
    try:
        result = ArticulationValidationResult.model_validate(json.loads(payload))
    except (OSError, ValueError, ValidationError) as exc:
        raise ArticulationWorkflowError(
            f"Invalid articulation validation result at {path}: {exc}"
        ) from exc
    _verify_artifact_digest(
        result.output_asset_path,
        result.observed_output_asset_sha256,
        label="Published articulation output",
    )
    return result


def _review_partition(
    candidate_document: Stage2CandidateDocument,
    *,
    review_policy: str,
    allowed_motion_types: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    review_required: list[str] = []
    auto_accepted: list[str] = []
    unresolved_without_review: list[str] = []
    for candidate in candidate_document.candidates:
        if (
            not candidate.is_articulation_v1_authorable
            or candidate.articulation_v1_type not in allowed_motion_types
        ):
            if review_policy == "none":
                unresolved_without_review.append(candidate.candidate_id)
            else:
                review_required.append(candidate.candidate_id)
            continue
        if review_policy == "all" or (
            review_policy == "uncertain" and candidate.confidence != "high"
        ):
            review_required.append(candidate.candidate_id)
        else:
            auto_accepted.append(candidate.candidate_id)
    return (
        tuple(review_required),
        tuple(auto_accepted),
        tuple(unresolved_without_review),
    )


def _membership_partition(
    membership_document: MembershipDispositionDocument | None,
    candidate_document: Stage2CandidateDocument,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Project typed membership decisions onto the current Stage 2 IDs."""

    authorable_candidate_prims = {
        candidate.moving_part_prims[0]: candidate.candidate_id
        for candidate in candidate_document.candidates
        if candidate.moving_part_prims and candidate.is_articulation_v1_authorable
    }
    if membership_document is None:
        if authorable_candidate_prims:
            raise ArticulationWorkflowError(
                "Typed membership dispositions are required before authorable "
                "Stage 2 candidates can reach review or authoring."
            )
        return (), (), (), ()
    covered_candidate_prims = {
        record.motion_candidate_prim
        for record in membership_document.dispositions
        if record.motion_candidate_prim is not None
    }
    missing_candidate_ids = tuple(
        candidate_id
        for prim, candidate_id in authorable_candidate_prims.items()
        if prim not in covered_candidate_prims
    )
    if missing_candidate_ids:
        raise ArticulationWorkflowError(
            "Typed membership dispositions do not cover authorable Stage 2 "
            f"candidates: {', '.join(missing_candidate_ids)}."
        )
    candidate_id_by_primary = {
        candidate.moving_part_prims[0]: candidate.candidate_id
        for candidate in candidate_document.candidates
        if candidate.moving_part_prims
    }
    suppressed: set[str] = set()
    pending_candidates: set[str] = set()
    review_memberships: list[str] = []
    pending_memberships: list[str] = []
    for record in membership_document.dispositions:
        candidate_id = (
            candidate_id_by_primary.get(record.motion_candidate_prim)
            if record.motion_candidate_prim is not None
            else None
        )
        owner_candidate_id = (
            candidate_id_by_primary.get(record.physical_owner_candidate_prim)
            if record.physical_owner_candidate_prim is not None
            else None
        )
        suppress_candidate = record.disposition != "independent_motion" and not (
            record.disposition == "co_rigid"
            and candidate_id is not None
            and candidate_id == owner_candidate_id
        )
        if suppress_candidate and candidate_id is not None:
            suppressed.add(candidate_id)
        if record.disposition in {"explicit_fixed", "unresolved"}:
            pending_memberships.append(record.disposition_id)
            if candidate_id is not None:
                pending_candidates.add(candidate_id)
        if record.review_status == "review_required":
            review_memberships.append(record.disposition_id)
    candidate_order = candidate_document.candidate_ids
    return (
        tuple(
            candidate_id
            for candidate_id in candidate_order
            if candidate_id in suppressed
        ),
        tuple(
            candidate_id
            for candidate_id in candidate_order
            if candidate_id in pending_candidates
        ),
        tuple(review_memberships),
        tuple(pending_memberships),
    )


def _review_and_membership_partition(
    candidate_document: Stage2CandidateDocument,
    membership_document: MembershipDispositionDocument | None,
    *,
    review_policy: str,
    allowed_motion_types: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Combine legacy candidate review with the separate membership artifact."""

    review_required, auto_accepted, unresolved = _review_partition(
        candidate_document,
        review_policy=review_policy,
        allowed_motion_types=allowed_motion_types,
    )
    (
        suppressed,
        pending_candidates,
        review_memberships,
        pending_memberships,
    ) = _membership_partition(membership_document, candidate_document)
    suppressed_set = set(suppressed)
    pending_candidate_set = set(pending_candidates)
    candidate_id_by_primary = {
        candidate.moving_part_prims[0]: candidate.candidate_id
        for candidate in candidate_document.candidates
        if candidate.moving_part_prims
    }
    membership_review_candidate_ids = {
        candidate_id
        for record in (
            membership_document.dispositions if membership_document is not None else ()
        )
        if record.review_status == "review_required"
        and record.motion_candidate_prim is not None
        and (candidate_id := candidate_id_by_primary.get(record.motion_candidate_prim))
        is not None
    }
    candidate_order = candidate_document.candidate_ids
    review_set = {
        candidate_id
        for candidate_id in review_required
        if candidate_id not in suppressed_set
        or candidate_id in membership_review_candidate_ids
    }
    review_set.update(membership_review_candidate_ids)
    unresolved_set = (set(unresolved) - suppressed_set) | pending_candidate_set
    return (
        tuple(
            candidate_id
            for candidate_id in candidate_order
            if candidate_id in review_set
        ),
        tuple(
            candidate_id
            for candidate_id in candidate_order
            if candidate_id in set(auto_accepted) - suppressed_set
        ),
        tuple(
            candidate_id
            for candidate_id in candidate_order
            if candidate_id in unresolved_set
        ),
        suppressed,
        review_memberships,
        pending_memberships,
    )


def _validate_review_receipt(
    receipt: ArticulationReviewReceipt,
    *,
    state: ArticulationRunState,
    candidate_document: Stage2CandidateDocument,
    candidate_document_sha256: str,
    allowed_motion_types: tuple[str, ...],
) -> None:
    candidate_binding = state.candidate_document
    if candidate_binding is None:
        raise ArticulationWorkflowError("Candidate evidence is not checkpointed.")
    if receipt.request_sha256 != state.request.sha256:
        raise ArticulationWorkflowError(
            "Review receipt request digest does not match this workflow."
        )
    if receipt.source_sha256 != state.source_sha256:
        raise ArticulationWorkflowError(
            "Review receipt source digest does not match this workflow."
        )
    if receipt.source_dependency_bundle_sha256 != state.source_dependency_bundle_sha256:
        raise ArticulationWorkflowError(
            "Review receipt source dependency bundle digest does not match "
            "this workflow."
        )
    if receipt.candidate_document_sha256 != candidate_document_sha256:
        raise ArticulationWorkflowError(
            "Review receipt candidate digest does not match current evidence."
        )
    expected_scene_sha256 = (
        state.scene_evidence.sha256 if state.scene_evidence is not None else None
    )
    if receipt.scene_evidence_sha256 != expected_scene_sha256:
        raise ArticulationWorkflowError(
            "Review receipt Scene evidence digest does not match current evidence."
        )
    decision_ids = tuple(decision.candidate_id for decision in receipt.decisions)
    if decision_ids != state.review_required_candidate_ids:
        raise ArticulationWorkflowError(
            "Review receipt must decide every required candidate exactly once "
            "in candidate order."
        )
    candidate_by_id = candidate_document.candidate_by_id()
    for decision in receipt.decisions:
        if decision.decision == "accept" and (
            not candidate_by_id[decision.candidate_id].is_articulation_v1_authorable
            or candidate_by_id[decision.candidate_id].articulation_v1_type
            not in allowed_motion_types
        ):
            raise ArticulationWorkflowError(
                f"Candidate {decision.candidate_id} is not native-ready evidence "
                "within the request's articulation-v1 motion scope and cannot be "
                "accepted. Re-run Joint Agent adjudication or reject it."
            )


def _review_outcome(
    candidate_document: Stage2CandidateDocument,
    *,
    auto_accepted_ids: tuple[str, ...],
    receipt: ArticulationReviewReceipt | None,
    unresolved_without_review: tuple[str, ...],
    membership_suppressed_candidate_ids: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    accepted = set(auto_accepted_ids)
    rejected: set[str] = set()
    if receipt is not None:
        for decision in receipt.decisions:
            if decision.decision == "accept":
                accepted.add(decision.candidate_id)
            else:
                rejected.add(decision.candidate_id)
    suppressed = set(membership_suppressed_candidate_ids)
    accepted.difference_update(suppressed)
    rejected.update(suppressed)
    candidate_order = candidate_document.candidate_ids
    return (
        tuple(
            candidate_id for candidate_id in candidate_order if candidate_id in accepted
        ),
        tuple(
            candidate_id for candidate_id in candidate_order if candidate_id in rejected
        ),
        tuple(
            candidate_id
            for candidate_id in candidate_order
            if candidate_id in set(unresolved_without_review)
        ),
    )


def _accepted_candidate_document(
    candidate_document: Stage2CandidateDocument,
    accepted_candidate_ids: tuple[str, ...],
) -> Stage2CandidateDocument:
    accepted = set(accepted_candidate_ids)
    payload = candidate_document.model_dump(mode="json")
    candidates = [
        candidate
        for candidate in payload["candidates"]
        if candidate["candidate_id"] in accepted
    ]
    if tuple(candidate["candidate_id"] for candidate in candidates) != (
        accepted_candidate_ids
    ):
        raise ArticulationWorkflowError(
            "Accepted candidate IDs do not preserve immutable candidate order."
        )
    joint_types: list[str] = []
    for candidate in candidates:
        joint_type = candidate["joint_type_hint"] or candidate["motion_type"]
        if not isinstance(joint_type, str) or not joint_type:
            raise ArticulationWorkflowError(
                "Accepted candidate lacks a normalized joint type."
            )
        joint_types.append(joint_type)
    joint_type_counts = Counter(joint_types)
    review_status_counts = Counter(
        str(candidate.get("review_status", "review_required"))
        for candidate in candidates
    )
    limit_readiness_counts = Counter(
        str(candidate.get("limit_readiness", "not_provided"))
        for candidate in candidates
    )
    reason_code_counts = Counter(
        str(reason_code)
        for candidate in candidates
        for reason_code in candidate.get("unresolved_reason_codes", [])
    )
    summary = dict(payload["summary"])
    summary.update(
        {
            "candidate_count": len(candidates),
            "ready_candidate_count": review_status_counts.get(
                "ready_for_rigger_input",
                0,
            ),
            "review_required_candidate_count": review_status_counts.get(
                "review_required",
                0,
            ),
            "joint_type_counts": dict(sorted(joint_type_counts.items())),
            "unresolved_axis_count": sum(
                candidate.get("motion_axis_world") is None for candidate in candidates
            ),
            "unresolved_parent_count": sum(
                candidate.get("fixed_parent_prim") is None for candidate in candidates
            ),
            "review_status_counts": dict(sorted(review_status_counts.items())),
            "limit_readiness_counts": dict(sorted(limit_readiness_counts.items())),
            "reason_code_counts": dict(sorted(reason_code_counts.items())),
        }
    )
    payload["summary"] = summary
    payload["candidates"] = candidates
    return Stage2CandidateDocument.model_validate(payload)


def _graph_closure_blockers(
    candidate_document: Stage2CandidateDocument,
    accepted_candidate_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Find accepted edges whose parent/child candidate subgraph is not closed."""

    accepted = set(accepted_candidate_ids)
    candidates_by_body: dict[str, list[str]] = {}
    candidate_by_id = candidate_document.candidate_by_id()
    for candidate in candidate_document.candidates:
        if candidate.moving_part_prims:
            candidates_by_body.setdefault(candidate.moving_part_prims[0], []).append(
                candidate.candidate_id
            )

    blockers: set[str] = set()
    accepted_children: dict[str, str] = {}
    accepted_topologies: dict[tuple[str, str, str, tuple[float, ...]], str] = {}
    for candidate_id in accepted_candidate_ids:
        candidate = candidate_by_id[candidate_id]
        if len(candidate.moving_part_prims) != 1:
            raise ArticulationWorkflowError(
                f"Accepted candidate {candidate_id} must bind exactly one "
                "moving-part prim."
            )
        child = candidate.moving_part_prims[0]
        previous_child = accepted_children.get(child)
        if previous_child is not None:
            blockers.update({previous_child, candidate_id})
        accepted_children[child] = candidate_id

        if candidate.fixed_parent_prim is None or candidate.motion_axis_world is None:
            raise ArticulationWorkflowError(
                f"Accepted candidate {candidate_id} lacks an authorable "
                "parent or signed motion axis."
            )
        topology = (
            candidate.articulation_v1_type,
            candidate.fixed_parent_prim,
            child,
            candidate.motion_axis_world,
        )
        previous_topology = accepted_topologies.get(topology)
        if previous_topology is not None:
            blockers.update({previous_topology, candidate_id})
        accepted_topologies[topology] = candidate_id

        parent_candidates = candidates_by_body.get(candidate.fixed_parent_prim, [])
        if len(parent_candidates) > 1:
            blockers.add(candidate_id)
            blockers.update(parent_candidates)
        elif parent_candidates and parent_candidates[0] not in accepted:
            blockers.update({candidate_id, parent_candidates[0]})

    return tuple(
        candidate_id
        for candidate_id in candidate_document.candidate_ids
        if candidate_id in blockers
    )


def _validate_inference_scope(
    result: ArticulationInferenceResult,
    *,
    request: ArticulationWorkflowRequest,
    configuration_sha256: str,
) -> None:
    if result.backend_configuration_sha256 != configuration_sha256:
        raise ArticulationWorkflowError(
            "Joint Agent inference used a different backend configuration."
        )
    if not result.candidate_document.candidates:
        evidence = result.metadata.get("structure_analysis_evidence")
        reasoning = result.metadata.get("structure_analysis_reasoning")
        diagnostics_path = (
            evidence.get("provider_response_diagnostics_path")
            if isinstance(evidence, dict)
            else None
        )
        diagnostics_sha256 = (
            evidence.get("provider_response_diagnostics_sha256")
            if isinstance(evidence, dict)
            else None
        )
        if not (
            result.metadata.get("structure_analysis_outcome") == "not_articulated"
            and isinstance(reasoning, str)
            and reasoning.strip()
            and isinstance(evidence, dict)
            and evidence.get("accepted") is True
            and isinstance(evidence.get("robot_type"), str)
            and evidence["robot_type"].strip()
            and evidence.get("dof") == 0
            and evidence.get("segment_names") == []
            and isinstance(evidence.get("source_prim_inventory"), list)
            and bool(evidence["source_prim_inventory"])
            and all(
                isinstance(path, str) and path.strip()
                for path in evidence["source_prim_inventory"]
            )
            and isinstance(diagnostics_path, str)
            and isinstance(diagnostics_sha256, str)
            and len(diagnostics_sha256) == 64
            and all(character in "0123456789abcdef" for character in diagnostics_sha256)
        ):
            raise ArticulationWorkflowError(
                "Joint Agent inference returned an unproven empty candidate set."
            )
        _verify_artifact_digest(
            diagnostics_path,
            diagnostics_sha256,
            label="Joint Agent structure provider-response diagnostics",
        )
        from joint_agent.functions.provider_response_conformance import (
            ProviderAttemptJournal,
        )

        try:
            structure = (
                ProviderAttemptJournal.load(
                    Path(diagnostics_path), expected_sha256=diagnostics_sha256
                )
                .snapshot()
                .whole_asset_structure
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ArticulationWorkflowError(
                "Joint Agent structure provider-response diagnostics are invalid."
            ) from exc
        if (
            structure is None
            or structure.accepted is not True
            or structure.reason_codes
            or structure.robot_type != evidence["robot_type"]
            or structure.dof != evidence["dof"]
            or structure.segment_names != tuple(evidence["segment_names"])
            or structure.source_prim_inventory
            != tuple(evidence["source_prim_inventory"])
        ):
            raise ArticulationWorkflowError(
                "Joint Agent structure provider-response diagnostics do not match "
                "the accepted non-articulated result."
            )
    if len(result.candidate_document.candidates) > request.max_candidate_count:
        raise ArticulationWorkflowError(
            "Joint Agent candidate count exceeds the request safety bound."
        )
    if (
        request.expected_candidate_count is not None
        and len(result.candidate_document.candidates)
        != request.expected_candidate_count
    ):
        raise ArticulationWorkflowError(
            "Joint Agent candidate count does not match the request: "
            f"expected {request.expected_candidate_count}, got "
            f"{len(result.candidate_document.candidates)}."
        )


def _validate_authoring_scope(
    result: ArticulationAuthoringResult,
    request: ArticulationAuthoringRequest,
) -> None:
    if result.authored_candidate_ids != request.accepted_candidate_ids:
        raise ArticulationWorkflowError(
            "Joint Agent authoring result differs from accepted candidate IDs."
        )
    if result.idempotency_key != request.idempotency_key:
        raise ArticulationWorkflowError(
            "Joint Agent authoring result idempotency key differs from the request."
        )
    if result.source_sha256 != request.source_sha256:
        raise ArticulationWorkflowError(
            "Joint Agent authoring result source digest differs from the request."
        )
    if (
        result.source_dependency_bundle_sha256
        != request.source_dependency_bundle_sha256
    ):
        raise ArticulationWorkflowError(
            "Joint Agent authoring result source dependency bundle digest "
            "differs from the request."
        )
    if (
        result.candidate_document_path != request.candidate_document_path
        or result.candidate_document_sha256 != request.candidate_document_sha256
    ):
        raise ArticulationWorkflowError(
            "Joint Agent authoring result candidate binding differs from the request."
        )


def _validate_authoring_request_scope(
    authoring_request: ArticulationAuthoringRequest,
    *,
    workflow_request: ArticulationWorkflowRequest,
    state: ArticulationRunState,
    inference: ArticulationInferenceResult | None,
) -> None:
    """Bind recovered authoring inputs to the exact workflow inference chain."""

    if (
        authoring_request.source_asset != workflow_request.source_asset
        or authoring_request.source_sha256 != state.source_sha256
        or authoring_request.source_dependency_bundle_sha256
        != state.source_dependency_bundle_sha256
    ):
        raise ArticulationWorkflowError(
            "Checkpointed authoring request differs from the workflow source."
        )
    if authoring_request.output_dir.expanduser().resolve() != (
        workflow_request.output_dir.expanduser().resolve()
    ):
        raise ArticulationWorkflowError(
            "Checkpointed authoring request differs from the workflow output directory."
        )
    if inference is None or (
        authoring_request.predictions_path != inference.predictions_path
        or authoring_request.predictions_sha256 != inference.predictions_sha256
    ):
        raise ArticulationWorkflowError(
            "Checkpointed authoring request prediction binding differs from inference "
            "evidence."
        )


def _validate_readback_scope(
    result: ArticulationValidationResult,
    *,
    authoring: ArticulationAuthoringResult,
    accepted_candidate_ids: tuple[str, ...],
) -> None:
    if result.output_asset_path != authoring.output_asset_path:
        raise ArticulationWorkflowError(
            "Readback result references a different articulation output."
        )
    if result.expected_output_asset_sha256 != authoring.output_asset_sha256:
        raise ArticulationWorkflowError(
            "Readback expected output digest differs from the authored output."
        )
    if result.expected_candidate_ids != accepted_candidate_ids:
        raise ArticulationWorkflowError(
            "Readback result expected scope differs from accepted candidates."
        )


def _validate_checkpoint_review_partition(
    state: ArticulationRunState,
    *,
    request: ArticulationWorkflowRequest,
    candidate_document: Stage2CandidateDocument,
    membership_document: MembershipDispositionDocument | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Re-derive immutable review scope instead of trusting checkpoint fields."""

    expected_partition = _review_and_membership_partition(
        candidate_document,
        membership_document,
        review_policy=request.review_policy,
        allowed_motion_types=request.allowed_motion_types,
    )
    (
        expected_review_required,
        expected_auto_accepted,
        expected_unresolved,
        _expected_suppressed,
        expected_membership_review,
        expected_membership_unresolved,
    ) = expected_partition
    if state.candidate_ids != candidate_document.candidate_ids:
        raise ArticulationWorkflowError(
            "Checkpointed candidate IDs differ from the bound candidate document."
        )
    if state.review_required_candidate_ids != expected_review_required:
        raise ArticulationWorkflowError(
            "Checkpointed review-required scope differs from the bound request "
            "and candidates."
        )
    if state.auto_accepted_candidate_ids != expected_auto_accepted:
        raise ArticulationWorkflowError(
            "Checkpointed auto-accepted scope differs from the bound request "
            "and candidates."
        )
    if state.unresolved_candidate_ids != expected_unresolved:
        raise ArticulationWorkflowError(
            "Checkpointed unresolved scope differs from the bound request "
            "and candidates."
        )
    if state.review_required_membership_disposition_ids != expected_membership_review:
        raise ArticulationWorkflowError(
            "Checkpointed membership review scope differs from inference evidence."
        )
    if state.unresolved_membership_disposition_ids != expected_membership_unresolved:
        raise ArticulationWorkflowError(
            "Checkpointed unresolved membership differs from inference evidence."
        )
    return expected_partition


def _verify_checkpoint_summary_evidence(
    state: ArticulationRunState,
    *,
    request: ArticulationWorkflowRequest,
    scene_evidence_path: Path,
) -> tuple[
    ArticulationInferenceResult | None,
    Stage2CandidateDocument | None,
    ArticulationReviewReceipt | None,
    Stage2CandidateDocument | None,
    ArticulationAuthoringRequest | None,
    ArticulationAuthoringResult | None,
    ArticulationValidationResult | None,
]:
    """Read and cross-check every checkpoint artifact exposed by a summary."""

    request_path, request_payload = _read_verified_binding_bytes(
        state.request,
        label="articulation request",
    )
    try:
        bound_request = ArticulationWorkflowRequest.model_validate(
            json.loads(request_payload)
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ArticulationWorkflowError(
            f"Invalid checkpointed articulation request at {request_path}: {exc}"
        ) from exc
    if bound_request != request:
        raise ArticulationWorkflowError(
            "Checkpointed articulation request differs from its bound artifact."
        )
    inference = (
        _load_inference(
            state.inference_result,
            expected_configuration_sha256=state.backend_configuration_sha256,
        )
        if state.inference_result is not None
        else None
    )
    if inference is not None:
        _validate_inference_scope(
            inference,
            request=request,
            configuration_sha256=state.backend_configuration_sha256,
        )
    candidate_document: Stage2CandidateDocument | None = None
    if state.candidate_document is not None:
        candidate_path, candidate_payload = _read_verified_binding_bytes(
            state.candidate_document,
            label="candidate document",
        )
        candidate_document = Stage2CandidateDocument.model_validate(
            json.loads(candidate_payload)
        )
        if inference is not None and candidate_document != inference.candidate_document:
            raise ArticulationWorkflowError(
                "Checkpointed candidate document differs from inference evidence."
            )
        if state.phase in {"needs_review", "completed"}:
            _validate_checkpoint_review_partition(
                state,
                request=request,
                candidate_document=candidate_document,
                membership_document=(
                    inference.membership_disposition_document
                    if inference is not None
                    else None
                ),
            )

    scene_evidence_required = _request_requires_scene_evidence(request)
    if (
        scene_evidence_required
        and state.phase
        in {"needs_review", "authoring", "validating", "completed", "conditional"}
        and (
            state.scene_evidence_configuration_sha256 is None
            or state.scene_evidence is None
        )
        and (candidate_document is None or candidate_document.candidates)
    ):
        raise ArticulationWorkflowError(
            "Checkpointed workflow is missing request-required Scene evidence."
        )
    if state.scene_evidence is not None:
        if (
            state.scene_evidence_configuration_sha256 is None
            or state.candidate_document is None
            or candidate_document is None
        ):
            raise ArticulationWorkflowError(
                "Checkpointed Scene evidence lacks its bound configuration "
                "or candidate evidence."
            )
        _load_scene_evidence(
            state.scene_evidence,
            expected_manifest_path=scene_evidence_path,
            request=request,
            request_sha256=state.request.sha256,
            source_sha256=state.source_sha256,
            source_dependency_bundle_sha256=(state.source_dependency_bundle_sha256),
            candidate_document=candidate_document,
            candidate_document_sha256=state.candidate_document.sha256,
            collector_configuration_sha256=(state.scene_evidence_configuration_sha256),
        )

    effective_candidate_document = candidate_document
    effective_candidate_document_sha256 = (
        state.candidate_document.sha256 if state.candidate_document else ""
    )
    if candidate_document is not None and request.metadata.get(
        SKILL_ROUTED_DECISION_METADATA_KEY
    ):
        try:
            decision_context = load_articulation_decision_ledger(
                request.output_dir,
                state=state,
            )
        except ValueError as exc:
            raise ArticulationWorkflowError(str(exc)) from exc
        if decision_context is not None:
            _decision_patch, effective_candidate_document = decision_context
            effective_candidate_document_sha256 = file_sha256(
                request.output_dir / "agent_reviewed_articulation_candidates.json"
            )

    receipt: ArticulationReviewReceipt | None = None
    if state.review_receipt is not None:
        _, review_payload = _read_verified_binding_bytes(
            state.review_receipt,
            label="review receipt",
        )
        receipt = ArticulationReviewReceipt.model_validate(json.loads(review_payload))
        if effective_candidate_document is None:
            raise ArticulationWorkflowError(
                "Checkpointed review receipt lacks candidate evidence."
            )
        _validate_review_receipt(
            receipt,
            state=state,
            candidate_document=effective_candidate_document,
            candidate_document_sha256=effective_candidate_document_sha256,
            allowed_motion_types=request.allowed_motion_types,
        )

    approved_document: Stage2CandidateDocument | None = None
    if state.approved_candidate_document is not None:
        _, approved_payload = _read_verified_binding_bytes(
            state.approved_candidate_document,
            label="approved candidate document",
        )
        approved_document = Stage2CandidateDocument.model_validate(
            json.loads(approved_payload)
        )
        if effective_candidate_document is None:
            raise ArticulationWorkflowError(
                "Checkpointed approved candidates lack candidate evidence."
            )
        expected_approved = _accepted_candidate_document(
            effective_candidate_document,
            state.accepted_candidate_ids,
        )
        if approved_document != expected_approved:
            raise ArticulationWorkflowError(
                "Checkpointed approved candidate document differs from accepted "
                "candidate IDs."
            )

    authoring_request: ArticulationAuthoringRequest | None = None
    if state.authoring_request is not None:
        _, authoring_request_payload = _read_verified_binding_bytes(
            state.authoring_request,
            label="authoring request",
        )
        authoring_request = ArticulationAuthoringRequest.model_validate(
            json.loads(authoring_request_payload)
        )
        if authoring_request.accepted_candidate_ids != state.accepted_candidate_ids:
            raise ArticulationWorkflowError(
                "Checkpointed authoring request differs from accepted candidate IDs."
            )
        if state.approved_candidate_document is None or (
            authoring_request.candidate_document_path
            != state.approved_candidate_document.path
            or authoring_request.candidate_document_sha256
            != state.approved_candidate_document.sha256
        ):
            raise ArticulationWorkflowError(
                "Checkpointed authoring request differs from approved candidates."
            )
        _validate_authoring_request_scope(
            authoring_request,
            workflow_request=request,
            state=state,
            inference=inference,
        )

    authoring: ArticulationAuthoringResult | None = None
    if state.authoring_result is not None:
        authoring = _load_authoring(state.authoring_result)
        if authoring_request is None:
            raise ArticulationWorkflowError(
                "Checkpointed authoring result lacks its request."
            )
        _validate_authoring_scope(authoring, authoring_request)
        _verify_artifact_digest(
            authoring.output_asset_path,
            authoring.output_asset_sha256,
            label="Published articulation output",
        )

    validation: ArticulationValidationResult | None = None
    if state.validation_result is not None:
        validation = _load_validation(state.validation_result)
        if authoring is None:
            raise ArticulationWorkflowError(
                "Checkpointed validation result lacks authoring evidence."
            )
        _validate_readback_scope(
            validation,
            authoring=authoring,
            accepted_candidate_ids=state.accepted_candidate_ids,
        )

    return (
        inference,
        candidate_document,
        receipt,
        approved_document,
        authoring_request,
        authoring,
        validation,
    )


def _validate_completed_checkpoint(
    state: ArticulationRunState,
    *,
    request: ArticulationWorkflowRequest,
    inference: ArticulationInferenceResult | None,
    candidate_document: Stage2CandidateDocument | None,
    receipt: ArticulationReviewReceipt | None,
    approved_document: Stage2CandidateDocument | None,
    authoring_request: ArticulationAuthoringRequest | None,
    authoring: ArticulationAuthoringResult | None,
    validation: ArticulationValidationResult | None,
) -> None:
    """Require a complete, passing, cross-bound evidence chain for success."""

    required = {
        "inference result": inference,
        "candidate document": candidate_document,
        "approved candidate document": approved_document,
        "authoring request": authoring_request,
        "authoring result": authoring,
        "validation result": validation,
    }
    missing = tuple(label for label, value in required.items() if value is None)
    if missing:
        raise ArticulationWorkflowError(
            f"Completed checkpoint is missing required evidence: {', '.join(missing)}."
        )
    if (
        state.scene_evidence_configuration_sha256 is not None
        and state.scene_evidence is None
    ):
        raise ArticulationWorkflowError(
            "Completed checkpoint is missing required Scene evidence."
        )
    if state.review_required_candidate_ids and state.review_receipt is None:
        raise ArticulationWorkflowError(
            "Completed checkpoint is missing its required review receipt."
        )
    assert candidate_document is not None
    (
        _expected_review_required,
        expected_auto_accepted,
        expected_unresolved,
        expected_suppressed,
        _expected_membership_review,
        expected_membership_unresolved,
    ) = _validate_checkpoint_review_partition(
        state,
        request=request,
        candidate_document=candidate_document,
        membership_document=(
            inference.membership_disposition_document if inference is not None else None
        ),
    )
    expected_accepted, expected_rejected, expected_unresolved = _review_outcome(
        candidate_document,
        auto_accepted_ids=expected_auto_accepted,
        receipt=receipt,
        unresolved_without_review=expected_unresolved,
        membership_suppressed_candidate_ids=expected_suppressed,
    )
    if (
        state.accepted_candidate_ids != expected_accepted
        or state.rejected_candidate_ids != expected_rejected
        or state.unresolved_candidate_ids != expected_unresolved
    ):
        raise ArticulationWorkflowError(
            "Completed checkpoint review outcome differs from its bound request, "
            "candidates, and receipt."
        )
    if (
        not state.accepted_candidate_ids
        or state.unresolved_candidate_ids
        or state.unresolved_membership_disposition_ids
        or expected_membership_unresolved
    ):
        raise ArticulationWorkflowError(
            "Completed checkpoint must have accepted candidates and no unresolved "
            "candidate or membership decisions."
        )
    assert validation is not None
    if validation.status != "pass":
        raise ArticulationWorkflowError(
            "Completed checkpoint requires passing readback validation."
        )


def validate_completed_articulation_checkpoint(
    state: ArticulationRunState,
    *,
    request: ArticulationWorkflowRequest,
) -> None:
    """Reverify one completed checkpoint and its full native evidence chain."""

    if state.phase != "completed":
        raise ArticulationWorkflowError(
            "Articulation checkpoint must be terminal completed."
        )
    (
        inference,
        candidate_document,
        receipt,
        approved_document,
        authoring_request,
        authoring,
        validation,
    ) = _verify_checkpoint_summary_evidence(
        state,
        request=request,
        scene_evidence_path=(request.output_dir / "scene_evidence" / "manifest.json"),
    )
    _validate_completed_checkpoint(
        state,
        request=request,
        inference=inference,
        candidate_document=candidate_document,
        receipt=receipt,
        approved_document=approved_document,
        authoring_request=authoring_request,
        authoring=authoring,
        validation=validation,
    )


def _authoring_idempotency_key(
    *,
    source_asset: str,
    source_sha256: str,
    source_dependency_bundle_sha256: str,
    candidate_document: ArtifactBinding,
    accepted_candidate_ids: tuple[str, ...],
    predictions_path: str | None,
    predictions_sha256: str | None,
    output_dir: Path,
) -> str:
    payload = {
        "schema_version": "content-agent-workflows.articulation-authoring-key.v1",
        "source_asset": source_asset,
        "source_sha256": source_sha256,
        "source_dependency_bundle_sha256": source_dependency_bundle_sha256,
        "candidate_document_path": candidate_document.path,
        "candidate_document_sha256": candidate_document.sha256,
        "accepted_candidate_ids": accepted_candidate_ids,
        "predictions_path": predictions_path,
        "predictions_sha256": predictions_sha256,
        "output_dir": str(output_dir),
        "apply_masses": False,
        "apply_collision": False,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invoke_hook(hook: PhaseBoundaryHook | None, boundary: str) -> None:
    if hook is not None:
        hook(boundary)


def _cancelled(
    output_dir: Path,
    state: ArticulationRunState,
    *,
    reason: str,
    authoring: ArticulationAuthoringResult | None = None,
) -> ArticulationFinalizationResult:
    if state.phase != "cancelled":
        state = _transition(state, "cancelled", reason)
        state = _persist_state(output_dir, state)
    return _publish_terminal_summary(
        output_dir,
        state,
        authoring=authoring,
    )


def _is_cancelled(cancel_checker: CancelChecker | None) -> bool:
    return bool(cancel_checker is not None and cancel_checker())


def _failed_backend_call(
    output_dir: Path,
    state: ArticulationRunState,
    *,
    operation: str,
    error: Exception,
) -> ArticulationFinalizationResult | None:
    message = f"Joint Agent {operation} failed: {error}"
    try:
        terminal_status = (
            dict(error.status)
            if isinstance(error, JointAgentInferenceTerminalError)
            else None
        )
        failed = _transition(
            state,
            "failed",
            message,
            error=message,
            backend_terminal_status=terminal_status,
        )
        failed = _persist_state(output_dir, failed)
        return _publish_terminal_summary(output_dir, failed)
    except (ArticulationWorkflowInterrupted, asyncio.CancelledError):
        raise
    except Exception as recording_error:
        error.add_note(
            f"Recording the durable backend failure also failed: {recording_error}"
        )
        return None


def _write_terminal_integrity_failure_summary(
    output_dir: Path,
    state: ArticulationRunState,
    error: Exception,
) -> None:
    """Atomically replace stale terminal success paths with a fail-closed index."""

    if state.phase == "cancelled":
        # The durable checkpoint keeps its review identities for diagnosis, but
        # an integrity-failure summary may no longer be able to load the typed
        # membership records that bind them.
        summary_state = state.model_copy(
            update={
                "review_required_membership_disposition_ids": (),
                "unresolved_membership_disposition_ids": (),
            }
        )
    else:
        # Keep the terminal checkpoint repairable while withholding every
        # downstream path whose trust depends on the failed preflight.
        summary_state = state.model_copy(
            update={
                "phase": "failed",
                "error": f"Terminal artifact integrity check failed: {error}",
                "inference_result": None,
                "candidate_document": None,
                "scene_evidence": None,
                "review_receipt": None,
                "approved_candidate_document": None,
                "authoring_request": None,
                "authoring_result": None,
                "validation_result": None,
                "review_required_membership_disposition_ids": (),
                "unresolved_membership_disposition_ids": (),
            }
        )
    _invalidate_stale_terminal_summary(output_dir)
    write_articulation_workflow_summary(
        summary_state,
        output_dir=output_dir,
        authoring=None,
    )


def _invalidate_stale_terminal_summary(output_dir: Path) -> None:
    """Durably remove a prior success index before fail-closed publication."""

    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(resolved_output_dir, directory_flags)
    try:
        try:
            os.unlink("final_summary.json", dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_terminal_summary(
    output_dir: Path,
    state: ArticulationRunState,
    *,
    authoring: ArticulationAuthoringResult | None = None,
) -> ArticulationFinalizationResult:
    """Publish one terminal index and replace stale success on any failure."""

    try:
        return write_articulation_workflow_summary(
            state,
            output_dir=output_dir,
            authoring=authoring,
        )
    except (ArticulationWorkflowInterrupted, asyncio.CancelledError):
        raise
    except Exception as publication_error:
        try:
            _write_terminal_integrity_failure_summary(
                output_dir,
                state,
                publication_error,
            )
        except Exception as fallback_error:
            publication_error.add_note(
                f"Fail-closed summary publication also failed: {fallback_error}"
            )
        raise


def _write_untrusted_checkpoint_failure_summary(
    output_dir: Path,
    *,
    mode: ArticulationWorkflowMode,
    error: Exception,
) -> None:
    """Replace a stale summary without trusting an unreadable checkpoint."""

    resolved_output_dir = output_dir.expanduser().resolve()
    checkpoint_path = resolved_output_dir / "checkpoint.json"
    progress_path = resolved_output_dir / "workflow_progress.json"
    summary_path = resolved_output_dir / "final_summary.json"
    _invalidate_stale_terminal_summary(resolved_output_dir)
    atomic_write_json(
        progress_path,
        {
            "schema_version": ("content-agent-workflows.articulation-progress-log.v1"),
            "revision": 0,
            "phase": "failed",
            "candidate_ids": (),
            "review_required_candidate_ids": (),
            "accepted_candidate_ids": (),
            "rejected_candidate_ids": (),
            "unresolved_candidate_ids": (),
            "review_required_membership_disposition_ids": (),
            "unresolved_membership_disposition_ids": (),
            "transitions": (),
        },
    )
    atomic_write_json(
        summary_path,
        ArticulationFinalizationResult(
            success=False,
            status="failed",
            mode=mode,
            output_dir=str(resolved_output_dir),
            checkpoint_path=str(checkpoint_path),
            workflow_progress_path=str(progress_path),
            final_summary_path=str(summary_path),
            message=(
                "Terminal artifact integrity check failed before checkpoint "
                f"recovery: {error}"
            ),
        ),
    )


def build_articulation_review_receipt(
    output_dir: str | Path,
    decisions: Mapping[str, ReviewDecision],
    *,
    reviewer: str,
) -> ArticulationReviewReceipt:
    """Build a digest-bound receipt in the exact checkpointed review order."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    with FileLock(str(resolved_output_dir / ".articulation-workflow.lock")):
        state = _load_state(resolved_output_dir / "checkpoint.json")
        if state.phase != "needs_review":
            raise ArticulationWorkflowError(
                "A review receipt can be built only while the workflow needs review."
            )
        required = state.review_required_candidate_ids
        if set(decisions) != set(required) or len(decisions) != len(required):
            raise ArticulationWorkflowError(
                "Review decisions must cover exactly the required candidate IDs."
            )
        _, request_payload = _read_verified_binding_bytes(
            state.request,
            label="request",
        )
        workflow_request = ArticulationWorkflowRequest.model_validate(
            json.loads(request_payload)
        )
        candidate_binding = state.candidate_document
        if candidate_binding is None:
            raise ArticulationWorkflowError("Candidate evidence is not checkpointed.")
        _, candidate_payload = _read_verified_binding_bytes(
            candidate_binding,
            label="candidate document",
        )
        candidate_document = Stage2CandidateDocument.model_validate(
            json.loads(candidate_payload)
        )
        scene_configuration_sha256 = state.scene_evidence_configuration_sha256
        if _request_requires_scene_evidence(workflow_request) and (
            state.scene_evidence is None or scene_configuration_sha256 is None
        ):
            raise ArticulationWorkflowError(
                "Checkpointed review is missing request-required Scene evidence."
            )
        if state.scene_evidence is not None:
            if scene_configuration_sha256 is None:
                raise ArticulationWorkflowError(
                    "Checkpointed Scene evidence lacks its bound configuration."
                )
            _load_scene_evidence(
                state.scene_evidence,
                expected_manifest_path=(
                    resolved_output_dir / "scene_evidence" / "manifest.json"
                ),
                request=workflow_request,
                request_sha256=state.request.sha256,
                source_sha256=state.source_sha256,
                source_dependency_bundle_sha256=(state.source_dependency_bundle_sha256),
                candidate_document=candidate_document,
                candidate_document_sha256=candidate_binding.sha256,
                collector_configuration_sha256=scene_configuration_sha256,
            )
        elif scene_configuration_sha256 is not None:
            raise ArticulationWorkflowError(
                "Checkpointed review is missing required Scene evidence."
            )
        effective_candidate_document = candidate_document
        effective_candidate_document_sha256 = candidate_binding.sha256
        if workflow_request.metadata.get(SKILL_ROUTED_DECISION_METADATA_KEY):
            try:
                decision_context = load_articulation_decision_ledger(
                    resolved_output_dir,
                    state=state,
                )
            except ValueError as exc:
                raise ArticulationWorkflowError(str(exc)) from exc
            if decision_context is None:
                raise ArticulationWorkflowError(
                    "Skill-routed Articulation review requires a validated agent "
                    "decision ledger."
                )
            _decision_patch, effective_candidate_document = decision_context
            effective_candidate_document_sha256 = file_sha256(
                resolved_output_dir / "agent_reviewed_articulation_candidates.json"
            )
        receipt = ArticulationReviewReceipt(
            request_sha256=state.request.sha256,
            source_sha256=state.source_sha256,
            source_dependency_bundle_sha256=(state.source_dependency_bundle_sha256),
            candidate_document_sha256=effective_candidate_document_sha256,
            scene_evidence_sha256=(
                state.scene_evidence.sha256
                if state.scene_evidence is not None
                else None
            ),
            reviewer=reviewer,
            decisions=tuple(
                ArticulationReviewEntry(
                    candidate_id=candidate_id,
                    decision=decisions[candidate_id],
                )
                for candidate_id in required
            ),
        )
        _validate_review_receipt(
            receipt,
            state=state,
            candidate_document=effective_candidate_document,
            candidate_document_sha256=effective_candidate_document_sha256,
            allowed_motion_types=workflow_request.allowed_motion_types,
        )
        receipt_binding = _write_once_json(
            resolved_output_dir / "review_receipt.json",
            receipt,
            label="review receipt",
        )
        if state.review_receipt is None:
            state = state.model_copy(update={"review_receipt": receipt_binding})
            _persist_state(resolved_output_dir, state)
        elif state.review_receipt != receipt_binding:
            raise ArticulationWorkflowError(
                "A conflicting review receipt is already checkpointed."
            )
        return receipt


def run_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    mode: ArticulationWorkflowMode,
    client: ArticulationWorkflowClient,
    scene_evidence_collector: ArticulationSceneEvidenceCollector | None = None,
    review_receipt: ArticulationReviewReceipt | None = None,
    cancel_checker: CancelChecker | None = None,
    phase_boundary_hook: PhaseBoundaryHook | None = None,
) -> ArticulationFinalizationResult:
    """Infer, review, author, resume, and exact-validate articulation-v1."""

    output_dir = request.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    request_path = output_dir / "request.json"
    inference_path = output_dir / "inference_result.json"
    candidate_path = output_dir / "articulation_candidates.json"
    review_path = output_dir / "review_receipt.json"
    approved_candidate_path = output_dir / "approved_articulation_candidates.json"
    authoring_request_path = output_dir / "authoring_request.json"
    authoring_result_path = output_dir / "authoring_result.json"
    validation_result_path = output_dir / "validation_evidence.json"
    scene_evidence_path = output_dir / "scene_evidence" / "manifest.json"
    lock_path = output_dir / ".articulation-workflow.lock"

    with FileLock(str(lock_path)):
        checkpointed_state: ArticulationRunState | None = None
        try:
            checkpointed_state = (
                _load_state(checkpoint_path) if checkpoint_path.exists() else None
            )
            normalized_request = _bind_scene_evidence_requirement(
                _normalize_request(request),
                collector_enabled=scene_evidence_collector is not None,
            )
            if (
                _request_requires_scene_evidence(normalized_request)
                and scene_evidence_collector is None
            ):
                raise _ArticulationConfigurationDriftError(
                    "The persisted articulation request requires a Scene "
                    "evidence collector."
                )
            if scene_evidence_collector is not None:
                _require_scene_evidence_root(scene_evidence_path.parent)
            request_binding = _write_once_json(
                request_path,
                normalized_request,
                label="articulation request",
            )
            source_sha256, source_dependency_bundle_sha256 = _source_identity(
                normalized_request.source_asset
            )
            configuration_sha256 = client.configuration_sha256(normalized_request)
            scene_evidence_configuration_sha256 = (
                scene_evidence_collector.configuration_sha256(normalized_request)
                if scene_evidence_collector is not None
                else None
            )

            if checkpointed_state is not None:
                state = checkpointed_state
                if state.mode != mode:
                    raise ArticulationWorkflowError(
                        "Workflow mode differs from the checkpointed run."
                    )
                if state.request != request_binding:
                    raise ArticulationWorkflowError(
                        "Request digest differs from the checkpointed run."
                    )
                if state.source_asset != normalized_request.source_asset:
                    raise ArticulationWorkflowError(
                        "Source path differs from the checkpointed run."
                    )
                if state.source_sha256 != source_sha256:
                    raise ArticulationWorkflowError(
                        "Source asset digest differs from the checkpointed run."
                    )
                if (
                    state.source_dependency_bundle_sha256
                    != source_dependency_bundle_sha256
                ):
                    raise ArticulationWorkflowError(
                        "Source dependency bundle digest differs from the "
                        "checkpointed run."
                    )
                if state.backend_configuration_sha256 != configuration_sha256:
                    raise _ArticulationConfigurationDriftError(
                        "Joint Agent configuration differs from the checkpointed run."
                    )
                if (
                    state.scene_evidence_configuration_sha256
                    != scene_evidence_configuration_sha256
                ):
                    raise _ArticulationConfigurationDriftError(
                        "Scene evidence configuration differs from the "
                        "checkpointed run."
                    )
                if state.candidate_document is not None:
                    _verify_binding(
                        state.candidate_document,
                        label="candidate document",
                    )
            else:
                state = ArticulationRunState(
                    mode=mode,
                    request=request_binding,
                    source_asset=normalized_request.source_asset,
                    source_sha256=source_sha256,
                    source_dependency_bundle_sha256=(source_dependency_bundle_sha256),
                    backend_configuration_sha256=configuration_sha256,
                    scene_evidence_configuration_sha256=(
                        scene_evidence_configuration_sha256
                    ),
                )
                state = _persist_state(output_dir, state)
        except (ArticulationWorkflowInterrupted, asyncio.CancelledError):
            raise
        except Exception as exc:
            if checkpointed_state is not None:
                preserve_existing_summary = (
                    isinstance(exc, _ArticulationConfigurationDriftError)
                    and checkpointed_state.phase == "needs_review"
                )
                summary_error = exc
                if preserve_existing_summary:
                    try:
                        verified_evidence = _verify_checkpoint_summary_evidence(
                            checkpointed_state,
                            request=normalized_request,
                            scene_evidence_path=scene_evidence_path,
                        )
                        verify_articulation_workflow_summary(
                            checkpointed_state,
                            output_dir=output_dir,
                            authoring=verified_evidence[5],
                        )
                    except Exception as integrity_error:
                        preserve_existing_summary = False
                        summary_error = integrity_error
                        exc.add_note(
                            "Checkpoint evidence also failed integrity validation: "
                            f"{integrity_error}"
                        )
                if not preserve_existing_summary:
                    _write_terminal_integrity_failure_summary(
                        output_dir,
                        checkpointed_state,
                        summary_error,
                    )
            elif (
                checkpointed_state is None
                and (output_dir / "final_summary.json").is_file()
            ):
                _write_untrusted_checkpoint_failure_summary(
                    output_dir,
                    mode=mode,
                    error=exc,
                )
            raise

        resuming_uncommitted_scene_collection = bool(
            checkpointed_state is not None
            and state.phase == "collecting_evidence"
            and state.scene_evidence is None
        )
        authoring: ArticulationAuthoringResult | None = None
        if state.phase in TERMINAL_ARTICULATION_PHASES:
            try:
                (
                    terminal_inference,
                    terminal_candidate_document,
                    terminal_review_receipt,
                    terminal_approved_document,
                    terminal_authoring_request,
                    authoring,
                    terminal_validation,
                ) = _verify_checkpoint_summary_evidence(
                    state,
                    request=normalized_request,
                    scene_evidence_path=scene_evidence_path,
                )
                if (
                    state.scene_evidence_configuration_sha256 is not None
                    and state.scene_evidence is None
                    and state.phase in {"completed", "conditional"}
                    and (
                        terminal_candidate_document is None
                        or terminal_candidate_document.candidates
                    )
                ):
                    raise ArticulationWorkflowError(
                        "Checkpointed terminal run is missing required Scene evidence."
                    )
                if state.phase == "completed":
                    _validate_completed_checkpoint(
                        state,
                        request=normalized_request,
                        inference=terminal_inference,
                        candidate_document=terminal_candidate_document,
                        receipt=terminal_review_receipt,
                        approved_document=terminal_approved_document,
                        authoring_request=terminal_authoring_request,
                        authoring=authoring,
                        validation=terminal_validation,
                    )
                return write_articulation_workflow_summary(
                    state,
                    output_dir=output_dir,
                    authoring=authoring,
                )
            except (ArticulationWorkflowInterrupted, asyncio.CancelledError):
                raise
            except Exception as exc:
                _write_terminal_integrity_failure_summary(
                    output_dir,
                    state,
                    exc,
                )
                raise

        if not resuming_uncommitted_scene_collection and _is_cancelled(cancel_checker):
            return _cancelled(
                output_dir,
                state,
                reason="Cancellation requested before candidate inference.",
            )

        inference_binding = state.inference_result
        if inference_binding is None and inference_path.exists():
            inference_binding = _binding(inference_path)
        if inference_binding is not None:
            inference = _load_inference(
                inference_binding,
                expected_configuration_sha256=configuration_sha256,
            )
        else:
            resume_backend = state.phase == "inferring"
            if state.phase != "inferring":
                state = _transition(
                    state,
                    "inferring",
                    "Starting Joint Agent prediction and Stage 2 inference.",
                )
                state = _persist_state(output_dir, state)
            try:
                inference = client.infer(
                    normalized_request,
                    resume=resume_backend,
                    cancel_checker=cancel_checker,
                )
            except asyncio.CancelledError:
                return _cancelled(
                    output_dir,
                    state,
                    reason="Joint Agent inference observed cancellation.",
                )
            except ArticulationWorkflowInterrupted:
                raise
            except Exception as exc:
                terminal_result = _failed_backend_call(
                    output_dir,
                    state,
                    operation="inference",
                    error=exc,
                )
                if isinstance(exc, JointAgentInferenceTerminalError):
                    if terminal_result is None:
                        raise ArticulationWorkflowError(
                            "Joint Agent terminal failure could not be published"
                        ) from exc
                    return terminal_result
                raise ArticulationWorkflowError(
                    f"Joint Agent inference failed: {exc}"
                ) from exc
            _verify_inference_side_artifacts(inference)
            inference_binding = _write_once_json(
                inference_path,
                inference,
                label="inference result",
            )

        _validate_inference_scope(
            inference,
            request=normalized_request,
            configuration_sha256=configuration_sha256,
        )
        candidate_binding = _write_once_json(
            candidate_path,
            inference.candidate_document,
            label="candidate document",
        )
        _invoke_hook(phase_boundary_hook, "inference_artifacts_written")
        if not resuming_uncommitted_scene_collection and _is_cancelled(cancel_checker):
            state = state.model_copy(
                update={
                    "inference_result": inference_binding,
                    "candidate_document": candidate_binding,
                }
            )
            return _cancelled(
                output_dir,
                state,
                reason="Cancellation requested after candidate inference.",
            )

        candidate_document = inference.candidate_document
        (
            review_required_ids,
            auto_accepted_ids,
            unresolved_without_review,
            membership_suppressed_candidate_ids,
            membership_review_required_ids,
            unresolved_membership_ids,
        ) = _review_and_membership_partition(
            candidate_document,
            inference.membership_disposition_document,
            review_policy=normalized_request.review_policy,
            allowed_motion_types=normalized_request.allowed_motion_types,
        )

        scene_evidence_was_checkpointed = state.scene_evidence is not None
        scene_evidence_binding = state.scene_evidence
        if (
            scene_evidence_collector is not None
            and scene_evidence_binding is None
            and scene_evidence_path.exists()
        ):
            scene_evidence_binding = _binding(scene_evidence_path)

        state = state.model_copy(
            update={
                "inference_result": inference_binding,
                "candidate_document": candidate_binding,
                "candidate_ids": candidate_document.candidate_ids,
                "review_required_candidate_ids": review_required_ids,
                "auto_accepted_candidate_ids": auto_accepted_ids,
                "unresolved_candidate_ids": unresolved_without_review,
                "review_required_membership_disposition_ids": (
                    membership_review_required_ids
                ),
                "unresolved_membership_disposition_ids": unresolved_membership_ids,
                "backend_terminal_status": None,
            }
        )
        if not candidate_document.candidates and not (
            membership_review_required_ids or unresolved_membership_ids
        ):
            state = _transition(
                state,
                "not_articulated",
                (
                    "Provider-backed structure analysis established that the asset "
                    "is not articulated; articulation evidence and authoring leaves "
                    "were skipped."
                ),
            )
            state = _persist_state(output_dir, state)
            return _publish_terminal_summary(output_dir, state)
        if scene_evidence_collector is not None and not scene_evidence_was_checkpointed:
            if state.phase != "collecting_evidence":
                state = _transition(
                    state,
                    "collecting_evidence",
                    (
                        "Joint Agent inference is checkpointed; collecting "
                        "source-bound usd-cli evidence."
                    ),
                )
            state = _persist_state(output_dir, state)

        if scene_evidence_collector is not None:
            if scene_evidence_configuration_sha256 is None:
                raise ArticulationWorkflowError(
                    "Scene evidence collector configuration is not bound."
                )
            if scene_evidence_binding is not None:
                _load_scene_evidence(
                    scene_evidence_binding,
                    expected_manifest_path=scene_evidence_path,
                    request=normalized_request,
                    request_sha256=request_binding.sha256,
                    source_sha256=source_sha256,
                    source_dependency_bundle_sha256=(source_dependency_bundle_sha256),
                    candidate_document=candidate_document,
                    candidate_document_sha256=candidate_binding.sha256,
                    collector_configuration_sha256=(
                        scene_evidence_configuration_sha256
                    ),
                )
                if resuming_uncommitted_scene_collection and _is_cancelled(
                    cancel_checker
                ):
                    state = state.model_copy(
                        update={
                            "scene_evidence": scene_evidence_binding,
                        }
                    )
                    return _cancelled(
                        output_dir,
                        state,
                        reason=(
                            "Cancellation requested after recovering Content "
                            "Scene evidence."
                        ),
                    )
            else:
                scene_evidence: ArticulationSceneEvidenceResult | None = None
                manifest_committed = False
                try:
                    if resuming_uncommitted_scene_collection:
                        _reclaim_orphaned_scene_collections(
                            scene_evidence_collector,
                            evidence_root=scene_evidence_path.parent,
                        )
                        if _is_cancelled(cancel_checker):
                            raise asyncio.CancelledError
                    scene_evidence = scene_evidence_collector.collect(
                        normalized_request,
                        request_sha256=request_binding.sha256,
                        source_sha256=source_sha256,
                        source_dependency_bundle_sha256=(
                            source_dependency_bundle_sha256
                        ),
                        candidate_document=candidate_document,
                        candidate_document_sha256=candidate_binding.sha256,
                        output_dir=scene_evidence_path.parent,
                        cancel_checker=cancel_checker,
                    )
                    if _is_cancelled(cancel_checker):
                        raise asyncio.CancelledError
                    verify_articulation_scene_evidence(
                        scene_evidence,
                        request=normalized_request,
                        request_sha256=request_binding.sha256,
                        source_sha256=source_sha256,
                        source_dependency_bundle_sha256=(
                            source_dependency_bundle_sha256
                        ),
                        candidate_document=candidate_document,
                        candidate_document_sha256=candidate_binding.sha256,
                        collector_configuration_sha256=(
                            scene_evidence_configuration_sha256
                        ),
                        evidence_root=scene_evidence_path.parent,
                    )
                    if _is_cancelled(cancel_checker):
                        raise asyncio.CancelledError
                    try:
                        scene_evidence_binding = _write_once_json(
                            scene_evidence_path,
                            scene_evidence,
                            label="Scene evidence manifest",
                        )
                    except Exception:
                        matching_manifest_binding = _matching_scene_manifest_binding(
                            scene_evidence_path,
                            scene_evidence,
                        )
                        if matching_manifest_binding is None:
                            raise
                        scene_evidence_binding = matching_manifest_binding
                    manifest_committed = True
                    _release_committed_scene_collection(
                        scene_evidence_collector,
                        scene_evidence,
                        manifest_path=scene_evidence_path,
                    )
                except asyncio.CancelledError as exc:
                    matching_manifest_binding = (
                        _matching_scene_manifest_binding(
                            scene_evidence_path,
                            scene_evidence,
                        )
                        if scene_evidence is not None and not manifest_committed
                        else None
                    )
                    if matching_manifest_binding is not None:
                        scene_evidence_binding = matching_manifest_binding
                        manifest_committed = True
                    if not manifest_committed:
                        _discard_uncommitted_scene_collection(
                            scene_evidence_collector,
                            scene_evidence,
                            evidence_root=scene_evidence_path.parent,
                            original_error=exc,
                        )
                    state = state.model_copy(
                        update={
                            "inference_result": inference_binding,
                            "candidate_document": candidate_binding,
                            "scene_evidence": (
                                scene_evidence_binding if manifest_committed else None
                            ),
                        }
                    )
                    return _cancelled(
                        output_dir,
                        state,
                        reason=("usd-cli evidence collection observed cancellation."),
                    )
                except ArticulationWorkflowInterrupted as exc:
                    if (
                        scene_evidence is not None
                        and not manifest_committed
                        and _scene_manifest_matches(
                            scene_evidence_path,
                            scene_evidence,
                        )
                    ):
                        manifest_committed = True
                    if not manifest_committed:
                        _discard_uncommitted_scene_collection(
                            scene_evidence_collector,
                            scene_evidence,
                            evidence_root=scene_evidence_path.parent,
                            original_error=exc,
                        )
                    raise
                except Exception as exc:
                    if (
                        scene_evidence is not None
                        and not manifest_committed
                        and _scene_manifest_matches(
                            scene_evidence_path,
                            scene_evidence,
                        )
                    ):
                        manifest_committed = True
                    if not manifest_committed:
                        _discard_uncommitted_scene_collection(
                            scene_evidence_collector,
                            scene_evidence,
                            evidence_root=scene_evidence_path.parent,
                            original_error=exc,
                        )
                    message = f"usd-cli evidence collection failed: {exc}"
                    state = _persist_state(
                        output_dir,
                        state.model_copy(update={"error": message}),
                    )
                    raise ArticulationWorkflowError(message) from exc

        state = state.model_copy(
            update={
                "inference_result": inference_binding,
                "candidate_document": candidate_binding,
                "scene_evidence": scene_evidence_binding,
                "candidate_ids": candidate_document.candidate_ids,
                "review_required_candidate_ids": review_required_ids,
                "auto_accepted_candidate_ids": auto_accepted_ids,
                "unresolved_candidate_ids": unresolved_without_review,
                "review_required_membership_disposition_ids": (
                    membership_review_required_ids
                ),
                "unresolved_membership_disposition_ids": unresolved_membership_ids,
                "error": None,
            }
        )
        if scene_evidence_binding is not None and not scene_evidence_was_checkpointed:
            state = _persist_state(output_dir, state)
            _invoke_hook(phase_boundary_hook, "scene_evidence_checkpointed")

        effective_candidate_document = candidate_document
        effective_candidate_document_sha256 = candidate_binding.sha256
        if normalized_request.metadata.get(SKILL_ROUTED_DECISION_METADATA_KEY):
            try:
                decision_context = load_articulation_decision_ledger(
                    output_dir,
                    state=state,
                )
            except ValueError as exc:
                raise ArticulationWorkflowError(str(exc)) from exc
            if decision_context is None:
                if state.phase != "needs_review":
                    state = _transition(
                        state,
                        "needs_review",
                        (
                            "Candidate and Scene evidence require a "
                            "skill-routed decision patch before review or authoring."
                        ),
                    )
                    state = _persist_state(output_dir, state)
                return write_articulation_workflow_summary(
                    state,
                    output_dir=output_dir,
                )
            _agent_decision_patch, effective_candidate_document = decision_context
            effective_candidate_document_sha256 = file_sha256(
                output_dir / "agent_reviewed_articulation_candidates.json"
            )

        effective_receipt = review_receipt
        if state.review_receipt is not None:
            _, persisted_review_payload = _read_verified_binding_bytes(
                state.review_receipt, label="review receipt"
            )
            persisted_receipt = ArticulationReviewReceipt.model_validate(
                json.loads(persisted_review_payload)
            )
            if effective_receipt is not None and persisted_receipt != effective_receipt:
                raise ArticulationWorkflowError(
                    "A conflicting review receipt is already checkpointed."
                )
            effective_receipt = persisted_receipt

        if membership_review_required_ids:
            if state.phase != "needs_review":
                state = _transition(
                    state,
                    "needs_review",
                    (
                        "Physical-membership evidence requires reviewed input "
                        "through the typed Joint method before authoring."
                    ),
                )
                state = _persist_state(output_dir, state)
            return write_articulation_workflow_summary(
                state,
                output_dir=output_dir,
            )
        if review_required_ids and effective_receipt is None:
            if state.phase != "needs_review":
                state = _transition(
                    state,
                    "needs_review",
                    "Candidate evidence requires explicit review decisions.",
                )
                state = _persist_state(output_dir, state)
            return write_articulation_workflow_summary(
                state,
                output_dir=output_dir,
            )
        if (
            not (review_required_ids or membership_review_required_ids)
            and effective_receipt is not None
        ):
            raise ArticulationWorkflowError(
                "This workflow has no candidate or membership review requirement."
            )

        review_binding: ArtifactBinding | None = state.review_receipt
        if effective_receipt is not None:
            _validate_review_receipt(
                effective_receipt,
                state=state,
                candidate_document=effective_candidate_document,
                candidate_document_sha256=effective_candidate_document_sha256,
                allowed_motion_types=normalized_request.allowed_motion_types,
            )
            review_binding = _write_once_json(
                review_path,
                effective_receipt,
                label="review receipt",
            )
        accepted_ids, rejected_ids, unresolved_ids = _review_outcome(
            effective_candidate_document,
            auto_accepted_ids=auto_accepted_ids,
            receipt=effective_receipt,
            unresolved_without_review=unresolved_without_review,
            membership_suppressed_candidate_ids=(membership_suppressed_candidate_ids),
        )
        state = state.model_copy(
            update={
                "review_receipt": review_binding,
                "accepted_candidate_ids": accepted_ids,
                "rejected_candidate_ids": rejected_ids,
                "unresolved_candidate_ids": unresolved_ids,
            }
        )

        if not accepted_ids:
            state = _transition(
                state,
                "conditional",
                "No native-ready articulation-v1 candidates were approved.",
            )
            state = _persist_state(output_dir, state)
            return _publish_terminal_summary(
                output_dir,
                state,
            )

        graph_closure_blockers = _graph_closure_blockers(
            effective_candidate_document,
            accepted_ids,
        )
        if graph_closure_blockers:
            unresolved = tuple(
                candidate_id
                for candidate_id in candidate_document.candidate_ids
                if candidate_id
                in {
                    *state.unresolved_candidate_ids,
                    *graph_closure_blockers,
                }
            )
            message = (
                "Approved candidates do not form a closed articulation graph; "
                "review the unresolved parent/child candidate set."
            )
            state = _transition(
                state,
                "conditional",
                message,
                unresolved_candidate_ids=unresolved,
                error=message,
            )
            state = _persist_state(output_dir, state)
            return _publish_terminal_summary(
                output_dir,
                state,
            )

        approved_document = _accepted_candidate_document(
            effective_candidate_document,
            accepted_ids,
        )
        approved_binding = _write_once_json(
            approved_candidate_path,
            approved_document,
            label="approved candidate document",
        )
        authoring_request = ArticulationAuthoringRequest(
            source_asset=normalized_request.source_asset,
            source_sha256=source_sha256,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            candidate_document_path=approved_binding.path,
            candidate_document_sha256=approved_binding.sha256,
            accepted_candidate_ids=accepted_ids,
            idempotency_key=_authoring_idempotency_key(
                source_asset=normalized_request.source_asset,
                source_sha256=source_sha256,
                source_dependency_bundle_sha256=(source_dependency_bundle_sha256),
                candidate_document=approved_binding,
                accepted_candidate_ids=accepted_ids,
                predictions_path=inference.predictions_path,
                predictions_sha256=inference.predictions_sha256,
                output_dir=output_dir,
            ),
            predictions_path=inference.predictions_path,
            predictions_sha256=inference.predictions_sha256,
            output_dir=output_dir,
        )
        authoring_request_binding = _write_once_json(
            authoring_request_path,
            authoring_request,
            label="authoring request",
        )
        state = state.model_copy(
            update={
                "approved_candidate_document": approved_binding,
                "authoring_request": authoring_request_binding,
            }
        )
        if state.phase != "authoring":
            state = _transition(
                state,
                "authoring",
                "Review is resolved; authoring accepted candidates only.",
            )
            state = _persist_state(output_dir, state)

        if _is_cancelled(cancel_checker):
            return _cancelled(
                output_dir,
                state,
                reason="Cancellation requested before Joint Rigger authoring.",
            )

        authoring_binding = state.authoring_result
        if authoring_binding is None and authoring_result_path.exists():
            authoring_binding = _binding(authoring_result_path)
        if authoring_binding is not None:
            authoring = _load_authoring(authoring_binding)
            _validate_authoring_scope(authoring, authoring_request)
        else:
            try:
                authoring = client.author(
                    authoring_request,
                    cancel_checker=cancel_checker,
                )
            except asyncio.CancelledError:
                return _cancelled(
                    output_dir,
                    state,
                    reason="Joint Rigger authoring observed cancellation.",
                )
            except ArticulationWorkflowInterrupted:
                raise
            except _ArticulationEvidenceBindingError as exc:
                raise ArticulationWorkflowError(
                    f"Joint Agent authoring evidence binding failed: {exc}"
                ) from exc
            except Exception as exc:
                _failed_backend_call(
                    output_dir,
                    state,
                    operation="authoring",
                    error=exc,
                )
                raise ArticulationWorkflowError(
                    f"Joint Agent authoring failed: {exc}"
                ) from exc
            _invoke_hook(phase_boundary_hook, "authoring_backend_returned")
            _validate_authoring_scope(authoring, authoring_request)
            authoring_binding = _write_once_json(
                authoring_result_path,
                authoring,
                label="authoring result",
            )
            _invoke_hook(phase_boundary_hook, "authoring_artifact_written")

        state = state.model_copy(update={"authoring_result": authoring_binding})
        if _is_cancelled(cancel_checker):
            return _cancelled(
                output_dir,
                state,
                reason=(
                    "Cancellation requested after authoring; published output was "
                    "preserved but success was not committed."
                ),
                authoring=authoring,
            )
        if state.phase != "validating":
            state = _transition(
                state,
                "validating",
                "Reopening the saved USDZ for exact graph and identity readback.",
            )
            state = _persist_state(output_dir, state)

        validation_binding = state.validation_result
        if validation_binding is None and validation_result_path.exists():
            validation_binding = _binding(validation_result_path)
        if validation_binding is not None:
            validation = _load_validation(validation_binding)
            _validate_readback_scope(
                validation,
                authoring=authoring,
                accepted_candidate_ids=accepted_ids,
            )
        else:
            try:
                validation = client.validate(
                    authoring,
                    expected_candidate_ids=accepted_ids,
                    cancel_checker=cancel_checker,
                )
            except asyncio.CancelledError:
                return _cancelled(
                    output_dir,
                    state,
                    reason="Exact articulation readback observed cancellation.",
                    authoring=authoring,
                )
            except ArticulationWorkflowInterrupted:
                raise
            except _ArticulationEvidenceBindingError as exc:
                raise ArticulationWorkflowError(
                    f"Joint Agent validation evidence binding failed: {exc}"
                ) from exc
            except Exception as exc:
                _failed_backend_call(
                    output_dir,
                    state,
                    operation="validation",
                    error=exc,
                )
                raise ArticulationWorkflowError(
                    f"Joint Agent validation failed: {exc}"
                ) from exc
            _validate_readback_scope(
                validation,
                authoring=authoring,
                accepted_candidate_ids=accepted_ids,
            )
            validation_binding = _write_once_json(
                validation_result_path,
                validation,
                label="validation result",
            )
            _invoke_hook(phase_boundary_hook, "validation_artifact_written")

        state = state.model_copy(update={"validation_result": validation_binding})
        if _is_cancelled(cancel_checker):
            return _cancelled(
                output_dir,
                state,
                reason=(
                    "Cancellation requested after readback; evidence was preserved "
                    "but success was not committed."
                ),
                authoring=authoring,
            )

        if state.scene_evidence is not None:
            scene_configuration = state.scene_evidence_configuration_sha256
            completion_candidate_binding = state.candidate_document
            if scene_configuration is None or completion_candidate_binding is None:
                raise ArticulationWorkflowError(
                    "Scene evidence lacks its configuration or candidate binding."
                )
            _load_scene_evidence(
                state.scene_evidence,
                expected_manifest_path=scene_evidence_path,
                request=normalized_request,
                request_sha256=state.request.sha256,
                source_sha256=state.source_sha256,
                source_dependency_bundle_sha256=(state.source_dependency_bundle_sha256),
                candidate_document=candidate_document,
                candidate_document_sha256=completion_candidate_binding.sha256,
                collector_configuration_sha256=scene_configuration,
            )
            if _is_cancelled(cancel_checker):
                return _cancelled(
                    output_dir,
                    state,
                    reason=(
                        "Cancellation requested after final Scene evidence "
                        "verification; evidence was preserved but success was not "
                        "committed."
                    ),
                    authoring=authoring,
                )

        if (
            validation.status == "pass"
            and not state.unresolved_candidate_ids
            and not state.unresolved_membership_disposition_ids
        ):
            state = _transition(
                state,
                "completed",
                "Exact saved graph and self-contained USDZ validation passed.",
            )
        else:
            failed_validation_ids = tuple(
                candidate_id
                for candidate_id in accepted_ids
                if candidate_id not in validation.validated_candidate_ids
            )
            targeted_validation_ids = (
                failed_validation_ids
                if failed_validation_ids
                else accepted_ids
                if validation.status == "fail"
                else ()
            )
            unresolved = tuple(
                candidate_id
                for candidate_id in candidate_document.candidate_ids
                if candidate_id
                in {
                    *state.unresolved_candidate_ids,
                    *targeted_validation_ids,
                }
            )
            state = _transition(
                state,
                "conditional",
                (
                    "Articulation output requires targeted reinspection of exact "
                    "candidate or physical-membership decisions."
                ),
                unresolved_candidate_ids=unresolved,
            )
        state = _persist_state(output_dir, state)
        return _publish_terminal_summary(
            output_dir,
            state,
            authoring=authoring,
        )


def run_interactive_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    client: ArticulationWorkflowClient,
    scene_evidence_collector: ArticulationSceneEvidenceCollector | None = None,
    review_receipt: ArticulationReviewReceipt | None = None,
    cancel_checker: CancelChecker | None = None,
    phase_boundary_hook: PhaseBoundaryHook | None = None,
) -> ArticulationFinalizationResult:
    """Run the durable articulation workflow from an interactive agent."""

    return run_articulation_workflow(
        request,
        mode="interactive",
        client=client,
        scene_evidence_collector=scene_evidence_collector,
        review_receipt=review_receipt,
        cancel_checker=cancel_checker,
        phase_boundary_hook=phase_boundary_hook,
    )


def run_batch_articulation_workflow(
    request: ArticulationWorkflowRequest,
    *,
    client: ArticulationWorkflowClient,
    scene_evidence_collector: ArticulationSceneEvidenceCollector | None = None,
    review_receipt: ArticulationReviewReceipt | None = None,
    cancel_checker: CancelChecker | None = None,
    phase_boundary_hook: PhaseBoundaryHook | None = None,
) -> ArticulationFinalizationResult:
    """Run the same durable articulation workflow from a batch child agent."""

    return run_articulation_workflow(
        request,
        mode="batch",
        client=client,
        scene_evidence_collector=scene_evidence_collector,
        review_receipt=review_receipt,
        cancel_checker=cancel_checker,
        phase_boundary_hook=phase_boundary_hook,
    )


__all__ = [
    "ArticulationWorkflowError",
    "ArticulationWorkflowInterrupted",
    "PhaseBoundaryHook",
    "build_articulation_review_receipt",
    "run_articulation_workflow",
    "run_batch_articulation_workflow",
    "run_interactive_articulation_workflow",
]
