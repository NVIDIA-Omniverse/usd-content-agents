# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed, non-authoring articulation-v2 workflow scaffolding.

The workflow can bind review state and prepare one idempotent handoff, but it
never invokes an authorer. Its USD identity verifier may load the existing
Joint Rigger package, whose public identity helper shares a package with trusted
authoring entry points; none of those entry points are called here. Publication
can only be recorded after the #870 release facade performs that work outside
this module. These modules remain deliberately absent from the articulation
package root until #870 owns release selection and public integration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from filelock import FileLock
from pydantic import BaseModel, ValidationError

from content_agent_workflows.common.artifacts import atomic_write_json

from .models_v2 import (
    ArticulationV2ArtifactBinding,
    ArticulationV2ArtifactIdentity,
    ArticulationV2AttachmentFrame,
    ArticulationV2AuthoringIntent,
    ArticulationV2Checkpoint,
    ArticulationV2Phase,
    ArticulationV2PublicationResult,
    ArticulationV2ReadbackResult,
    ArticulationV2ReleaseSelection,
    ArticulationV2ReviewContext,
    ArticulationV2ReviewReceipt,
    ArticulationV2Status,
    ArticulationV2Transition,
    ArticulationV2UsdArtifactIdentity,
    ArticulationV2WorkflowRequest,
    ArticulationV2WorkflowResult,
    _strict_revalidate_articulation_v2_model,
    articulation_v2_canonical_sha256,
)

ArticulationV2CancelChecker = Callable[[], bool]
_CHUNK_SIZE = 1024 * 1024
_UsdIdentityKey = tuple[Path, str, int, str]


class ArticulationV2WorkflowError(RuntimeError):
    """Raised when the v2 workflow cannot preserve an exact durable identity."""


class ArticulationV2ReleaseSelectionError(ArticulationV2WorkflowError):
    """Raised before Scene startup when #870 has not admitted the request."""


class ArticulationV2ReleaseRouter(Protocol):
    """Adapter implemented by the strict release-selected facade from #870."""

    def resolve(
        self,
        request: ArticulationV2WorkflowRequest,
    ) -> ArticulationV2ReleaseSelection:
        """Return the exact selected row or fail before any workflow side effect."""


class PackagedManifestArticulationV2ReleaseRouter:
    """Refuse to substitute packaged policy for the strict #870 facade.

    ``content-agent-workflows`` does not depend on ``joint-agent``. A packaged
    manifest alone would not provide the admitted selection object or the
    authoring seam, so the default stays dependency-free and fails closed.
    """

    def resolve(
        self,
        request: ArticulationV2WorkflowRequest,
    ) -> ArticulationV2ReleaseSelection:
        del request
        raise ArticulationV2ReleaseSelectionError(
            "The strict Joint #870 release-selected facade adapter is unavailable; "
            "articulation-v2 remains non-authoring before Scene startup."
        )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _strict_revalidate_model[ModelT: BaseModel](
    value: ModelT,
    *,
    label: str,
) -> ModelT:
    try:
        return _strict_revalidate_articulation_v2_model(value)
    except (TypeError, ValidationError) as exc:
        raise ArticulationV2WorkflowError(f"Invalid {label} model") from exc


def _payload(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_exact_file(
    path: Path,
    *,
    label: str,
    retain_payload: bool,
) -> tuple[bytes | None, os.stat_result, str]:
    descriptor = -1
    stream_descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ArticulationV2WorkflowError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] | None = [] if retain_payload else None
        digest = hashlib.sha256()
        stream_descriptor = os.dup(descriptor)
        with os.fdopen(stream_descriptor, "rb") as stream:
            stream_descriptor = -1
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
        final = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _stat_signature(opened) != _stat_signature(final) or _stat_signature(
            opened
        ) != _stat_signature(current):
            raise ArticulationV2WorkflowError(f"{label} changed while being read")
        payload = b"".join(chunks) if chunks is not None else None
        return payload, opened, digest.hexdigest()
    except ArticulationV2WorkflowError:
        raise
    except FileNotFoundError as exc:
        raise ArticulationV2WorkflowError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise ArticulationV2WorkflowError(
            f"Cannot read {label} at {path}: {exc}"
        ) from exc
    finally:
        if stream_descriptor >= 0:
            os.close(stream_descriptor)
        if descriptor >= 0:
            os.close(descriptor)


def _verify_identity(identity: ArticulationV2ArtifactIdentity, *, label: str) -> None:
    _, metadata, actual_sha256 = _read_exact_file(
        identity.path,
        label=label,
        retain_payload=False,
    )
    if actual_sha256 != identity.sha256:
        raise ArticulationV2WorkflowError(
            f"{label} digest mismatch: expected {identity.sha256}, got {actual_sha256}"
        )
    if metadata.st_size != identity.size_bytes:
        raise ArticulationV2WorkflowError(
            f"{label} size mismatch: expected {identity.size_bytes}, "
            f"got {metadata.st_size}"
        )


def _verify_usd_identity(
    identity: ArticulationV2UsdArtifactIdentity,
    *,
    label: str,
    verified: set[_UsdIdentityKey] | None = None,
) -> None:
    key: _UsdIdentityKey = (
        identity.path,
        identity.sha256,
        identity.size_bytes,
        identity.dependency_bundle_sha256,
    )
    if verified is not None and key in verified:
        return
    _verify_identity(identity, label=label)
    try:
        from world_understanding.functions.physics.joint_rigger import (
            identify_usd_artifact,
        )

        observed = identify_usd_artifact(identity.path, uri=identity.uri)
    except Exception as exc:
        raise ArticulationV2WorkflowError(
            f"Cannot verify the composed dependency identity for {label}: {exc}"
        ) from exc
    if observed.root_sha256 != identity.sha256:
        raise ArticulationV2WorkflowError(
            f"{label} root changed while its dependency closure was verified"
        )
    if observed.dependency_bundle_sha256 != identity.dependency_bundle_sha256:
        raise ArticulationV2WorkflowError(
            f"{label} dependency bundle digest mismatch: expected "
            f"{identity.dependency_bundle_sha256}, got "
            f"{observed.dependency_bundle_sha256}"
        )
    if verified is not None:
        verified.add(key)


def _read_json_binding(
    binding: ArticulationV2ArtifactBinding,
    *,
    label: str,
) -> dict[str, Any]:
    payload, _, actual_sha256 = _read_exact_file(
        binding.path,
        label=label,
        retain_payload=True,
    )
    if payload is None:
        raise ArticulationV2WorkflowError(f"{label} payload was not retained")
    if actual_sha256 != binding.sha256:
        raise ArticulationV2WorkflowError(
            f"{label} digest mismatch: expected {binding.sha256}, got {actual_sha256}"
        )
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArticulationV2WorkflowError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ArticulationV2WorkflowError(f"{label} must contain a JSON object")
    return parsed


def _write_once_json(
    path: Path,
    value: BaseModel | Mapping[str, Any],
    *,
    label: str,
) -> ArticulationV2ArtifactBinding:
    validated_value = (
        _strict_revalidate_model(value, label=label)
        if isinstance(value, BaseModel)
        else value
    )
    expected = _payload(validated_value)
    if not path.exists():
        atomic_write_json(path, expected)
    payload, _, actual_sha256 = _read_exact_file(
        path,
        label=label,
        retain_payload=True,
    )
    if payload is None:
        raise ArticulationV2WorkflowError(f"{label} payload was not retained")
    try:
        existing = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArticulationV2WorkflowError(f"{label} is not valid JSON") from exc
    try:
        type_sensitive_match = json.dumps(
            existing,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == json.dumps(
            expected,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        type_sensitive_match = False
    # Equality to the model's own JSON dump validates the persisted tree
    # without allowing Pydantic coercion (for example, JSON ``1`` vs ``true``).
    if not type_sensitive_match:
        raise ArticulationV2WorkflowError(
            f"Existing {label} conflicts with the current workflow: {path}"
        )
    return ArticulationV2ArtifactBinding(
        path=path,
        sha256=actual_sha256,
    )


def _load_checkpoint(path: Path) -> ArticulationV2Checkpoint:
    payload, _, _ = _read_exact_file(
        path,
        label="articulation-v2 checkpoint",
        retain_payload=True,
    )
    if payload is None:
        raise ArticulationV2WorkflowError(
            "articulation-v2 checkpoint payload was not retained"
        )
    try:
        return ArticulationV2Checkpoint.model_validate_json(payload)
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            f"Invalid articulation-v2 checkpoint at {path}: {exc}"
        ) from exc


def _persist_checkpoint(path: Path, state: ArticulationV2Checkpoint) -> None:
    validated = ArticulationV2Checkpoint.model_validate(state.model_dump(mode="python"))
    atomic_write_json(path, validated)


def _updated_checkpoint(
    state: ArticulationV2Checkpoint,
    updates: Mapping[str, Any],
) -> ArticulationV2Checkpoint:
    document = state.model_dump(mode="python")
    document.update(updates)
    try:
        return ArticulationV2Checkpoint.model_validate(document)
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            f"Invalid articulation-v2 checkpoint mutation: {exc}"
        ) from exc


def _transition(
    state: ArticulationV2Checkpoint,
    to_phase: ArticulationV2Phase,
    reason: str,
    *,
    updates: Mapping[str, Any] | None = None,
) -> ArticulationV2Checkpoint:
    payload = dict(updates or {})
    payload.update(
        {
            "revision": state.revision + 1,
            "phase": to_phase,
            "transitions": (
                *state.transitions,
                ArticulationV2Transition(
                    timestamp=_timestamp(),
                    from_phase=state.phase,
                    to_phase=to_phase,
                    reason=reason,
                ),
            ),
        }
    )
    return _updated_checkpoint(state, payload)


def _admit_release_selection(
    request: ArticulationV2WorkflowRequest,
    router: ArticulationV2ReleaseRouter,
) -> ArticulationV2ReleaseSelection:
    try:
        selection = _strict_revalidate_model(
            router.resolve(request),
            label="#870 release selection",
        )
    except ArticulationV2ReleaseSelectionError:
        raise
    except ArticulationV2WorkflowError as exc:
        raise ArticulationV2ReleaseSelectionError(
            "The #870 release facade returned an invalid release selection model."
        ) from exc
    if selection != request.release_selection:
        raise ArticulationV2ReleaseSelectionError(
            "The #870 release facade returned a selection that differs from the "
            "digest-bound workflow request."
        )
    return selection


def _validate_review_context(
    request: ArticulationV2WorkflowRequest,
    context: ArticulationV2ReviewContext,
    *,
    request_sha256: str,
    selection_sha256: str,
) -> None:
    expected = (
        context.request_sha256 == request_sha256
        and context.release_selection_sha256 == selection_sha256
        and context.selector_kind == request.selector.kind
        and context.capability_id == request.selector.capability_id
        and context.source_sha256 == request.source.sha256
        and context.reference_sha256 == request.reference.sha256
    )
    if not expected:
        raise ArticulationV2WorkflowError(
            "Scene review context does not match the exact v2 request."
        )
    _verify_identity(context.scene_evidence, label="Scene review evidence")


def _load_review_context(
    binding: ArticulationV2ArtifactBinding,
) -> ArticulationV2ReviewContext:
    try:
        return ArticulationV2ReviewContext.model_validate(
            _read_json_binding(binding, label="articulation-v2 review context")
        )
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            "Invalid articulation-v2 review context"
        ) from exc


def _load_review_receipt(
    binding: ArticulationV2ArtifactBinding,
) -> ArticulationV2ReviewReceipt:
    try:
        return ArticulationV2ReviewReceipt.model_validate(
            _read_json_binding(binding, label="articulation-v2 review receipt")
        )
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            "Invalid articulation-v2 review receipt"
        ) from exc


def _load_authoring_intent(
    binding: ArticulationV2ArtifactBinding,
) -> ArticulationV2AuthoringIntent:
    try:
        return ArticulationV2AuthoringIntent.model_validate(
            _read_json_binding(binding, label="articulation-v2 authoring intent")
        )
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            "Invalid articulation-v2 authoring intent"
        ) from exc


def _load_publication_result(
    binding: ArticulationV2ArtifactBinding,
) -> ArticulationV2PublicationResult:
    try:
        return ArticulationV2PublicationResult.model_validate(
            _read_json_binding(binding, label="articulation-v2 publication result")
        )
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            "Invalid articulation-v2 publication result"
        ) from exc


def _load_readback_result(
    binding: ArticulationV2ArtifactBinding,
) -> ArticulationV2ReadbackResult:
    try:
        return ArticulationV2ReadbackResult.model_validate(
            _read_json_binding(binding, label="articulation-v2 readback result")
        )
    except ValidationError as exc:
        raise ArticulationV2WorkflowError(
            "Invalid articulation-v2 readback result"
        ) from exc


def _validate_review_receipt(
    request: ArticulationV2WorkflowRequest,
    context: ArticulationV2ReviewContext,
    receipt: ArticulationV2ReviewReceipt,
    *,
    request_sha256: str,
    selection_sha256: str,
    context_sha256: str,
) -> None:
    expected = (
        receipt.request_sha256 == request_sha256
        and receipt.release_selection_sha256 == selection_sha256
        and receipt.review_context_sha256 == context_sha256
        and receipt.scene_evidence_sha256 == context.scene_evidence.sha256
        and receipt.selector_kind == request.selector.kind
        and receipt.capability_id == request.selector.capability_id
    )
    if not expected:
        raise ArticulationV2WorkflowError(
            "Review receipt does not match the exact v2 request and evidence."
        )


def _idempotency_key(
    *,
    request_sha256: str,
    selection_sha256: str,
    review_receipt_sha256: str,
    output: Mapping[str, Any],
) -> str:
    return articulation_v2_canonical_sha256(
        {
            "schema_version": "content-agent-workflows.articulation-idempotency.v2",
            "request_sha256": request_sha256,
            "release_selection_sha256": selection_sha256,
            "review_receipt_sha256": review_receipt_sha256,
            "output": dict(output),
        }
    )


def _validate_checkpoint_request(
    state: ArticulationV2Checkpoint,
    request: ArticulationV2WorkflowRequest,
    request_binding: ArticulationV2ArtifactBinding,
    selection_sha256: str,
) -> None:
    if state.mode != request.mode:
        raise ArticulationV2WorkflowError("Workflow mode differs from checkpoint")
    if state.request != request_binding:
        raise ArticulationV2WorkflowError("Request differs from checkpoint")
    if state.release_selection_sha256 != selection_sha256:
        raise ArticulationV2WorkflowError("Release selection differs from checkpoint")
    if state.source != request.source:
        raise ArticulationV2WorkflowError("Source identity differs from checkpoint")
    if state.reference != request.reference:
        raise ArticulationV2WorkflowError("Reference identity differs from checkpoint")
    if state.output != request.output:
        raise ArticulationV2WorkflowError("Output target differs from checkpoint")


def _constraint_matches_request(
    request: ArticulationV2WorkflowRequest,
    readback: ArticulationV2ReadbackResult,
) -> bool:
    contract = request.selector.contract
    if readback.constraint is None:
        return False
    tolerance = readback.readback_tolerance

    def _components_match(
        observed: tuple[float, ...],
        expected: tuple[float, ...],
    ) -> bool:
        return len(observed) == len(expected) and all(
            math.isclose(
                observed_component,
                expected_component,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            for observed_component, expected_component in zip(
                observed,
                expected,
                strict=True,
            )
        )

    def _frame_matches(
        observed: ArticulationV2AttachmentFrame,
        expected: ArticulationV2AttachmentFrame,
    ) -> bool:
        orientation_matches = _components_match(
            observed.orientation_wxyz,
            expected.orientation_wxyz,
        ) or _components_match(
            observed.orientation_wxyz,
            tuple(-component for component in expected.orientation_wxyz),
        )
        return (
            observed.sampling_mode == expected.sampling_mode
            and _components_match(observed.position_meters, expected.position_meters)
            and orientation_matches
        )

    matches = (
        readback.constraint.kind == contract.kind
        and readback.constraint.body0 == contract.body0
        and readback.constraint.body1 == contract.body1
        and readback.constraint.policy == contract.policy
        and _frame_matches(
            readback.constraint.body0_attachment,
            contract.body0_attachment,
        )
        and _frame_matches(
            readback.constraint.body1_attachment,
            contract.body1_attachment,
        )
    )
    if contract.kind == "distance" and readback.constraint.kind == "distance":
        matches = matches and (
            math.isclose(
                readback.constraint.minimum_distance_meters,
                contract.minimum_distance_meters,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            and math.isclose(
                readback.constraint.maximum_distance_meters,
                contract.maximum_distance_meters,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
    return matches


def _validate_persisted_state_artifacts(
    request: ArticulationV2WorkflowRequest,
    state: ArticulationV2Checkpoint,
    *,
    verified_usd_identities: set[_UsdIdentityKey],
) -> None:
    context: ArticulationV2ReviewContext | None = None
    if state.review_context is not None:
        context = _load_review_context(state.review_context)
        _validate_review_context(
            request,
            context,
            request_sha256=state.request.sha256,
            selection_sha256=state.release_selection_sha256,
        )

    if state.review_receipt is not None:
        if context is None or state.review_context is None:
            raise ArticulationV2WorkflowError(
                "Checkpointed review receipt has no bound review context."
            )
        receipt = _load_review_receipt(state.review_receipt)
        _validate_review_receipt(
            request,
            context,
            receipt,
            request_sha256=state.request.sha256,
            selection_sha256=state.release_selection_sha256,
            context_sha256=state.review_context.sha256,
        )

    if state.authoring_intent is not None:
        if state.review_receipt is None or state.idempotency_key is None:
            raise ArticulationV2WorkflowError(
                "Checkpointed authoring intent lacks review or idempotency."
            )
        intent = _load_authoring_intent(state.authoring_intent)
        expected_intent = (
            intent.request_sha256 == state.request.sha256
            and intent.release_selection_sha256 == state.release_selection_sha256
            and intent.review_receipt_sha256 == state.review_receipt.sha256
            and intent.idempotency_key == state.idempotency_key
            and intent.selector == request.selector
            and intent.source == request.source
            and intent.reference == request.reference
            and intent.output == request.output
        )
        if not expected_intent:
            raise ArticulationV2WorkflowError(
                "Checkpointed authoring intent differs from the accepted request."
            )

    publication: ArticulationV2PublicationResult | None = None
    if state.publication_result is not None:
        if state.review_receipt is None or state.authoring_intent is None:
            raise ArticulationV2WorkflowError(
                "Checkpointed publication lacks review or authoring intent."
            )
        publication = _load_publication_result(state.publication_result)
        expected_publication = (
            publication.request_sha256 == state.request.sha256
            and publication.release_selection_sha256 == state.release_selection_sha256
            and publication.review_receipt_sha256 == state.review_receipt.sha256
            and publication.authoring_intent_sha256 == state.authoring_intent.sha256
            and publication.idempotency_key == state.idempotency_key
            and publication.facade_call_id == state.facade_call_id
            and state.facade_call_count == 1
            and publication.selector_kind == request.selector.kind
            and publication.capability_id == request.selector.capability_id
            and publication.source == request.source
            and publication.reference == request.reference
            and publication.output_target == request.output
            and publication.authoring_contract_sha256
            == articulation_v2_canonical_sha256(request.selector.contract)
        )
        if not expected_publication:
            raise ArticulationV2WorkflowError(
                "Checkpointed publication differs from the accepted handoff."
            )
        _verify_usd_identity(
            publication.output_artifact,
            label="published v2 output",
            verified=verified_usd_identities,
        )

    if state.readback_result is not None:
        if publication is None or state.publication_result is None:
            raise ArticulationV2WorkflowError(
                "Checkpointed readback has no bound publication."
            )
        readback = _load_readback_result(state.readback_result)
        expected_identity = (
            readback.publication_result_sha256 == state.publication_result.sha256
            and readback.selector_kind == request.selector.kind
            and readback.capability_id == request.selector.capability_id
            and readback.source == request.source
            and readback.reference == request.reference
            and readback.output_artifact == publication.output_artifact
        )
        expected_outcome = (
            readback.status == "pass"
            and state.phase == "completed"
            and _constraint_matches_request(request, readback)
        ) or (readback.status == "fail" and state.phase == "failed")
        if not expected_identity or not expected_outcome:
            raise ArticulationV2WorkflowError(
                "Checkpointed saved-stage readback differs from the selected output."
            )
        _verify_usd_identity(
            readback.output_artifact,
            label="readback v2 output",
            verified=verified_usd_identities,
        )
        _verify_identity(
            readback.evidence_artifact,
            label="saved-stage readback evidence",
        )


def _result(
    request: ArticulationV2WorkflowRequest,
    state: ArticulationV2Checkpoint,
) -> ArticulationV2WorkflowResult:
    output_dir = request.output_dir
    status_by_phase: dict[ArticulationV2Phase, ArticulationV2Status] = {
        "initialized": "awaiting_review_context",
        "needs_review": "needs_review",
        "ready_for_authoring": "ready_for_authoring",
        "published": "published",
        "completed": "completed",
        "cancelled": "cancelled",
        "failed": "failed",
    }
    messages = {
        "initialized": (
            "Release selection is admitted; exact Scene review evidence is required."
        ),
        "needs_review": "Explicit review is required before any authoring handoff.",
        "ready_for_authoring": (
            "Review is accepted and one idempotent #870 facade handoff is prepared; "
            "this workflow has not invoked an authorer."
        ),
        "published": (
            "The external #870 facade publication is recorded; exact saved-stage "
            "readback is still required."
        ),
        "completed": (
            "The single publication and exact saved-stage constraint readback are bound."
        ),
        "cancelled": state.error or "The articulation-v2 workflow was cancelled.",
        "failed": state.error or "The articulation-v2 workflow failed.",
    }
    return ArticulationV2WorkflowResult(
        success=state.phase == "completed",
        status=status_by_phase[state.phase],
        mode=request.mode,
        capability_id=request.selector.capability_id,
        selector_kind=request.selector.kind,
        output_dir=output_dir,
        checkpoint_path=output_dir / "checkpoint.v2.json",
        request_path=state.request.path,
        review_context_path=(
            state.review_context.path if state.review_context is not None else None
        ),
        review_receipt_path=(
            state.review_receipt.path if state.review_receipt is not None else None
        ),
        authoring_intent_path=(
            state.authoring_intent.path if state.authoring_intent is not None else None
        ),
        publication_result_path=(
            state.publication_result.path
            if state.publication_result is not None
            else None
        ),
        readback_result_path=(
            state.readback_result.path if state.readback_result is not None else None
        ),
        output_asset_path=(
            request.output.path if state.publication_result is not None else None
        ),
        idempotency_key=state.idempotency_key,
        facade_call_count=state.facade_call_count,
        message=messages[state.phase],
    )


def _require_initialized_output_dir(output_dir: Path) -> None:
    if not output_dir.is_dir():
        raise ArticulationV2WorkflowError(
            "The articulation-v2 workflow output directory is not initialized."
        )


def _validate_post_handoff_review_inputs(
    state: ArticulationV2Checkpoint,
    *,
    review_context: ArticulationV2ReviewContext | None,
    review_receipt: ArticulationV2ReviewReceipt | None,
) -> None:
    if review_context is not None:
        if state.review_context is None:
            raise ArticulationV2WorkflowError(
                "A review context cannot be supplied after the durable handoff."
            )
        review_context = _strict_revalidate_model(
            review_context,
            label="articulation-v2 review context",
        )
        if _load_review_context(state.review_context) != review_context:
            raise ArticulationV2WorkflowError(
                "Review context differs from the checkpointed handoff evidence."
            )
    if review_receipt is not None:
        if state.review_receipt is None:
            raise ArticulationV2WorkflowError(
                "A review receipt cannot be supplied after the durable handoff."
            )
        review_receipt = _strict_revalidate_model(
            review_receipt,
            label="articulation-v2 review receipt",
        )
        if _load_review_receipt(state.review_receipt) != review_receipt:
            raise ArticulationV2WorkflowError(
                "Review receipt differs from the checkpointed handoff decision."
            )


def build_articulation_v2_review_receipt(
    output_dir: str | Path,
    *,
    release_router: ArticulationV2ReleaseRouter,
    decision: Literal["accept", "reject"],
    reviewer: str,
    note: str | None = None,
) -> ArticulationV2ReviewReceipt:
    """Build one exact review receipt from the durable v2 review context."""

    resolved = Path(output_dir).expanduser().resolve()
    _require_initialized_output_dir(resolved)
    checkpoint_path = resolved / "checkpoint.v2.json"
    with FileLock(str(resolved / ".articulation-v2-workflow.lock")):
        state = _load_checkpoint(checkpoint_path)
        if state.phase != "needs_review" or state.review_context is None:
            raise ArticulationV2WorkflowError(
                "A v2 review receipt can be built only while review is required."
            )
        request_payload = _read_json_binding(
            state.request,
            label="articulation-v2 request",
        )
        request = ArticulationV2WorkflowRequest.model_validate(request_payload)
        selection = _admit_release_selection(request, release_router)
        if (
            articulation_v2_canonical_sha256(selection)
            != state.release_selection_sha256
        ):
            raise ArticulationV2ReleaseSelectionError(
                "The re-admitted #870 selection differs from the review checkpoint."
            )
        context = _load_review_context(state.review_context)
        _validate_review_context(
            request,
            context,
            request_sha256=state.request.sha256,
            selection_sha256=state.release_selection_sha256,
        )
        receipt = ArticulationV2ReviewReceipt(
            request_sha256=state.request.sha256,
            release_selection_sha256=state.release_selection_sha256,
            review_context_sha256=state.review_context.sha256,
            scene_evidence_sha256=context.scene_evidence.sha256,
            selector_kind=request.selector.kind,
            capability_id=request.selector.capability_id,
            decision=decision,
            reviewer=reviewer,
            note=note,
        )
        receipt_binding = _write_once_json(
            resolved / "review_receipt.v2.json",
            receipt,
            label="articulation-v2 review receipt",
        )
        if state.review_receipt is None:
            state = _updated_checkpoint(
                state,
                {
                    "revision": state.revision + 1,
                    "review_receipt": receipt_binding,
                },
            )
            _persist_checkpoint(checkpoint_path, state)
        elif state.review_receipt != receipt_binding:
            raise ArticulationV2WorkflowError(
                "A conflicting v2 review receipt is already checkpointed."
            )
        return receipt


def run_articulation_v2_workflow(
    request: ArticulationV2WorkflowRequest,
    *,
    release_router: ArticulationV2ReleaseRouter | None = None,
    review_context: ArticulationV2ReviewContext | None = None,
    review_receipt: ArticulationV2ReviewReceipt | None = None,
    cancel_checker: ArticulationV2CancelChecker | None = None,
) -> ArticulationV2WorkflowResult:
    """Bind release selection, review, cancellation, resume, and idempotency.

    No authorer or resolver-capture callback exists in this entry point. The
    release router runs first, before output-directory creation or artifact IO.
    """

    request = _strict_revalidate_model(
        request,
        label="articulation-v2 request",
    )
    router = release_router or PackagedManifestArticulationV2ReleaseRouter()
    selection = _admit_release_selection(request, router)
    selection_sha256 = articulation_v2_canonical_sha256(selection)
    verified_usd_identities: set[_UsdIdentityKey] = set()

    _verify_usd_identity(
        request.source,
        label="articulation-v2 source",
        verified=verified_usd_identities,
    )
    _verify_usd_identity(
        request.reference,
        label="articulation-v2 reference",
        verified=verified_usd_identities,
    )

    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.v2.json"
    request_path = output_dir / "request.v2.json"
    lock_path = output_dir / ".articulation-v2-workflow.lock"

    with FileLock(str(lock_path)):
        request_binding = _write_once_json(
            request_path,
            request,
            label="articulation-v2 request",
        )
        if checkpoint_path.exists():
            state = _load_checkpoint(checkpoint_path)
            _validate_checkpoint_request(
                state,
                request,
                request_binding,
                selection_sha256,
            )
        else:
            state = ArticulationV2Checkpoint(
                mode=request.mode,
                request=request_binding,
                release_selection_sha256=selection_sha256,
                source=request.source,
                reference=request.reference,
                output=request.output,
            )
            _persist_checkpoint(checkpoint_path, state)

        _validate_persisted_state_artifacts(
            request,
            state,
            verified_usd_identities=verified_usd_identities,
        )

        if state.phase in {"completed", "cancelled", "failed"}:
            _validate_post_handoff_review_inputs(
                state,
                review_context=(
                    review_context if state.review_context is not None else None
                ),
                review_receipt=(
                    review_receipt if state.review_receipt is not None else None
                ),
            )
            return _result(request, state)

        # A durable authoring intent may already be consumed by #870.  It is
        # therefore the point of no return for local cancellation.
        if state.authoring_intent is not None:
            _validate_post_handoff_review_inputs(
                state,
                review_context=review_context,
                review_receipt=review_receipt,
            )
            return _result(request, state)

        if state.publication_result is None and os.path.lexists(request.output.path):
            raise ArticulationV2WorkflowError(
                "The v2 output target exists without a checkpointed #870 publication."
            )

        if cancel_checker is not None and cancel_checker():
            state = _transition(
                state,
                "cancelled",
                "Cancellation requested before the next durable boundary.",
                updates={"error": "The articulation-v2 workflow was cancelled."},
            )
            _persist_checkpoint(checkpoint_path, state)
            return _result(request, state)

        if review_context is not None:
            review_context = _strict_revalidate_model(
                review_context,
                label="articulation-v2 review context",
            )
            _validate_review_context(
                request,
                review_context,
                request_sha256=request_binding.sha256,
                selection_sha256=selection_sha256,
            )
            context_binding = _write_once_json(
                output_dir / "review_context.v2.json",
                review_context,
                label="articulation-v2 review context",
            )
            if state.review_context is None:
                if state.phase != "initialized":
                    raise ArticulationV2WorkflowError(
                        "Review context cannot be attached in the current phase."
                    )
                state = _transition(
                    state,
                    "needs_review",
                    "Exact Scene evidence was bound for explicit review.",
                    updates={"review_context": context_binding},
                )
                _persist_checkpoint(checkpoint_path, state)
            elif state.review_context != context_binding:
                raise ArticulationV2WorkflowError(
                    "Review context differs from the checkpointed evidence."
                )

        if state.review_context is None:
            return _result(request, state)
        context = _load_review_context(state.review_context)
        _validate_review_context(
            request,
            context,
            request_sha256=request_binding.sha256,
            selection_sha256=selection_sha256,
        )

        receipt_binding = state.review_receipt
        if review_receipt is not None:
            review_receipt = _strict_revalidate_model(
                review_receipt,
                label="articulation-v2 review receipt",
            )
            _validate_review_receipt(
                request,
                context,
                review_receipt,
                request_sha256=request_binding.sha256,
                selection_sha256=selection_sha256,
                context_sha256=state.review_context.sha256,
            )
            supplied_binding = _write_once_json(
                output_dir / "review_receipt.v2.json",
                review_receipt,
                label="articulation-v2 review receipt",
            )
            if receipt_binding is not None and receipt_binding != supplied_binding:
                raise ArticulationV2WorkflowError(
                    "Review receipt differs from the checkpointed decision."
                )
            if receipt_binding is None:
                receipt_binding = supplied_binding
                state = _updated_checkpoint(
                    state,
                    {
                        "revision": state.revision + 1,
                        "review_receipt": receipt_binding,
                    },
                )
                _persist_checkpoint(checkpoint_path, state)

        if receipt_binding is None:
            return _result(request, state)
        if state.review_context is None:
            raise ArticulationV2WorkflowError(
                "Checkpointed review receipt has no review context."
            )
        receipt = _load_review_receipt(receipt_binding)
        _validate_review_receipt(
            request,
            context,
            receipt,
            request_sha256=request_binding.sha256,
            selection_sha256=selection_sha256,
            context_sha256=state.review_context.sha256,
        )

        if receipt.decision == "reject":
            if state.phase != "cancelled":
                state = _transition(
                    state,
                    "cancelled",
                    "The reviewed fixed/distance request was rejected.",
                    updates={
                        "error": "The reviewed articulation-v2 request was rejected."
                    },
                )
                _persist_checkpoint(checkpoint_path, state)
            return _result(request, state)

        if state.authoring_intent is None:
            key = _idempotency_key(
                request_sha256=request_binding.sha256,
                selection_sha256=selection_sha256,
                review_receipt_sha256=receipt_binding.sha256,
                output=request.output.model_dump(mode="json"),
            )
            intent = ArticulationV2AuthoringIntent(
                request_sha256=request_binding.sha256,
                release_selection_sha256=selection_sha256,
                review_receipt_sha256=receipt_binding.sha256,
                idempotency_key=key,
                selector=request.selector,
                source=request.source,
                reference=request.reference,
                output=request.output,
            )
            intent_binding = _write_once_json(
                output_dir / "authoring_intent.v2.json",
                intent,
                label="articulation-v2 authoring intent",
            )
            state = _transition(
                state,
                "ready_for_authoring",
                "Review accepted; prepared one idempotent #870 facade handoff.",
                updates={
                    "authoring_intent": intent_binding,
                    "idempotency_key": key,
                },
            )
            _persist_checkpoint(checkpoint_path, state)

        return _result(request, state)


def _validated_request_state(
    request: ArticulationV2WorkflowRequest,
    *,
    release_router: ArticulationV2ReleaseRouter,
) -> tuple[
    ArticulationV2Checkpoint,
    ArticulationV2ArtifactBinding,
    str,
    set[_UsdIdentityKey],
]:
    selection = _admit_release_selection(request, release_router)
    selection_sha256 = articulation_v2_canonical_sha256(selection)
    verified_usd_identities: set[_UsdIdentityKey] = set()
    _verify_usd_identity(
        request.source,
        label="articulation-v2 source",
        verified=verified_usd_identities,
    )
    _verify_usd_identity(
        request.reference,
        label="articulation-v2 reference",
        verified=verified_usd_identities,
    )
    state = _load_checkpoint(request.output_dir / "checkpoint.v2.json")
    request_binding = _write_once_json(
        request.output_dir / "request.v2.json",
        request,
        label="articulation-v2 request",
    )
    _validate_checkpoint_request(state, request, request_binding, selection_sha256)
    _validate_persisted_state_artifacts(
        request,
        state,
        verified_usd_identities=verified_usd_identities,
    )
    return state, request_binding, selection_sha256, verified_usd_identities


def record_articulation_v2_publication(
    request: ArticulationV2WorkflowRequest,
    publication: ArticulationV2PublicationResult,
    *,
    release_router: ArticulationV2ReleaseRouter,
) -> ArticulationV2WorkflowResult:
    """Record, but never perform, the one publication returned by #870."""

    request = _strict_revalidate_model(
        request,
        label="articulation-v2 request",
    )
    publication = _strict_revalidate_model(
        publication,
        label="articulation-v2 publication result",
    )
    _require_initialized_output_dir(request.output_dir)
    lock_path = request.output_dir / ".articulation-v2-workflow.lock"
    with FileLock(str(lock_path)):
        (
            state,
            request_binding,
            selection_sha256,
            verified_usd_identities,
        ) = _validated_request_state(
            request,
            release_router=release_router,
        )
        replayable_failure = (
            state.phase == "failed" and state.publication_result is not None
        )
        if (
            state.phase
            not in {
                "ready_for_authoring",
                "published",
                "completed",
            }
            and not replayable_failure
        ):
            raise ArticulationV2WorkflowError(
                "Publication can be recorded only after accepted review."
            )
        if state.review_receipt is None or state.authoring_intent is None:
            raise ArticulationV2WorkflowError("Authoring handoff is incomplete")
        expected = (
            publication.request_sha256 == request_binding.sha256
            and publication.release_selection_sha256 == selection_sha256
            and publication.review_receipt_sha256 == state.review_receipt.sha256
            and publication.authoring_intent_sha256 == state.authoring_intent.sha256
            and publication.idempotency_key == state.idempotency_key
            and publication.selector_kind == request.selector.kind
            and publication.capability_id == request.selector.capability_id
            and publication.source == request.source
            and publication.reference == request.reference
            and publication.output_target == request.output
            and publication.authoring_contract_sha256
            == articulation_v2_canonical_sha256(request.selector.contract)
        )
        if not expected:
            raise ArticulationV2WorkflowError(
                "Publication does not match the accepted idempotent v2 handoff."
            )
        _verify_usd_identity(
            publication.output_artifact,
            label="published v2 output",
            verified=verified_usd_identities,
        )
        binding = _write_once_json(
            request.output_dir / "publication_result.v2.json",
            publication,
            label="articulation-v2 publication result",
        )
        if state.publication_result is None:
            state = _transition(
                state,
                "published",
                "Recorded the single external #870 facade publication.",
                updates={
                    "publication_result": binding,
                    "facade_call_id": publication.facade_call_id,
                    "facade_call_count": 1,
                },
            )
            _persist_checkpoint(request.output_dir / "checkpoint.v2.json", state)
        elif state.publication_result != binding:
            raise ArticulationV2WorkflowError(
                "A different publication is already checkpointed."
            )
        return _result(request, state)


def record_articulation_v2_readback(
    request: ArticulationV2WorkflowRequest,
    readback: ArticulationV2ReadbackResult,
    *,
    release_router: ArticulationV2ReleaseRouter,
) -> ArticulationV2WorkflowResult:
    """Bind exact saved-stage readback after the external publication."""

    request = _strict_revalidate_model(
        request,
        label="articulation-v2 request",
    )
    readback = _strict_revalidate_model(
        readback,
        label="articulation-v2 readback result",
    )
    _require_initialized_output_dir(request.output_dir)
    lock_path = request.output_dir / ".articulation-v2-workflow.lock"
    with FileLock(str(lock_path)):
        state, _, _, verified_usd_identities = _validated_request_state(
            request,
            release_router=release_router,
        )
        if state.phase not in {"published", "completed", "failed"}:
            raise ArticulationV2WorkflowError(
                "Saved-stage readback requires a checkpointed publication."
            )
        if state.publication_result is None:
            raise ArticulationV2WorkflowError("Publication result is missing")
        publication = _load_publication_result(state.publication_result)
        expected_identity = (
            readback.publication_result_sha256 == state.publication_result.sha256
            and readback.selector_kind == request.selector.kind
            and readback.capability_id == request.selector.capability_id
            and readback.source == request.source
            and readback.reference == request.reference
            and readback.output_artifact == publication.output_artifact
        )
        if not expected_identity:
            raise ArticulationV2WorkflowError(
                "Saved-stage readback identity does not match the selected output."
            )
        if readback.status == "pass" and not _constraint_matches_request(
            request,
            readback,
        ):
            raise ArticulationV2WorkflowError(
                "Passing saved-stage readback does not match the complete selected "
                "constraint."
            )
        _verify_usd_identity(
            readback.output_artifact,
            label="readback v2 output",
            verified=verified_usd_identities,
        )
        _verify_identity(
            readback.evidence_artifact, label="saved-stage readback evidence"
        )
        binding = _write_once_json(
            request.output_dir / "readback_result.v2.json",
            readback,
            label="articulation-v2 readback result",
        )
        if state.readback_result is None:
            passed = readback.status == "pass"
            state = _transition(
                state,
                "completed" if passed else "failed",
                (
                    "Exact saved-stage constraint readback was bound."
                    if passed
                    else "Failed saved-stage readback evidence was retained."
                ),
                updates={
                    "readback_result": binding,
                    "error": (
                        None
                        if passed
                        else "Saved-stage readback failed; see readback_result.v2.json."
                    ),
                },
            )
            _persist_checkpoint(request.output_dir / "checkpoint.v2.json", state)
        elif state.readback_result != binding:
            raise ArticulationV2WorkflowError(
                "A different saved-stage readback is already checkpointed."
            )
        return _result(request, state)


__all__ = [
    "ArticulationV2CancelChecker",
    "ArticulationV2ReleaseRouter",
    "ArticulationV2ReleaseSelectionError",
    "ArticulationV2WorkflowError",
    "PackagedManifestArticulationV2ReleaseRouter",
    "build_articulation_v2_review_receipt",
    "record_articulation_v2_publication",
    "record_articulation_v2_readback",
    "run_articulation_v2_workflow",
]
