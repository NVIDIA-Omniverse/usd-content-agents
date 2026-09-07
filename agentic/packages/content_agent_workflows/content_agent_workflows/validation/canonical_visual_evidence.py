# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical post-mutation OVRTX evidence as an independently callable leaf."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from content_agent_workflows.common.artifacts import (
    _open_directory_no_symlinks,
    atomic_write_json,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)

from .verified_operations import (
    VerifiedOperationComponentIdentity,
    VerifiedOperationError,
    VerifiedValidationOperationEnvelope,
    VerifiedValidationOperationProjection,
    execution_artifact_binding,
    verify_execution_artifact_binding,
    verify_operation_envelope,
)

CANONICAL_VISUAL_EVIDENCE_REQUEST_SCHEMA_VERSION: Final = (
    "content-agent-workflows.canonical-visual-evidence-request.v2"
)
CANONICAL_VISUAL_EVIDENCE_PAYLOAD_SCHEMA_VERSION: Final = (
    "content-agent-workflows.canonical-visual-evidence-payload.v2"
)
CANONICAL_VISUAL_EVIDENCE_PUBLICATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.canonical-visual-evidence-publication.v2"
)
CANONICAL_VISUAL_RENDER_REPORT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.canonical-visual-usd-cli-render-report.v1"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CanonicalVisualEvidenceRequest(_FrozenModel):
    """Exact source/output closure and explicitly selected render settings."""

    schema_version: Literal[
        "content-agent-workflows.canonical-visual-evidence-request.v2"
    ] = CANONICAL_VISUAL_EVIDENCE_REQUEST_SCHEMA_VERSION
    source: ExecutionArtifactBinding
    post_mutation_output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    dependency_closure_complete: Literal[True] = True
    backend: Literal["remote", "ovrtx"]
    views: tuple[str, ...] = Field(min_length=1)
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)


class CanonicalVisualEvidencePayload(_FrozenModel):
    """Raw OVRTX result identity without a semantic visual verdict."""

    schema_version: Literal[
        "content-agent-workflows.canonical-visual-evidence-payload.v2"
    ] = CANONICAL_VISUAL_EVIDENCE_PAYLOAD_SCHEMA_VERSION
    source: ExecutionArtifactBinding
    post_mutation_output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    dependency_closure_complete: Literal[True] = True
    render_report: ExecutionArtifactBinding
    images: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    render_responses: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    camera_records: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    usd_cli_command_receipt: ExecutionArtifactBinding
    usd_cli_receipt_checkpoint: ExecutionArtifactBinding
    usd_cli_source_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    backend_alias: Literal["remote", "ovrtx"]
    render_process_status: Literal["completed"] = "completed"
    render_metadata: dict[str, JsonValue]
    semantic_authority: Literal["outer_coordinator"] = "outer_coordinator"


class CanonicalVisualEvidencePublication(_FrozenModel):
    """Paths and exact bindings emitted by the public visual-evidence leaf."""

    schema_version: Literal[
        "content-agent-workflows.canonical-visual-evidence-publication.v2"
    ] = CANONICAL_VISUAL_EVIDENCE_PUBLICATION_SCHEMA_VERSION
    request: ExecutionArtifactBinding
    render_report: ExecutionArtifactBinding
    payload: ExecutionArtifactBinding
    projection: ExecutionArtifactBinding
    envelope: ExecutionArtifactBinding
    result: VerifiedValidationOperationEnvelope
    nested_agent_launched: Literal[False] = False


class _UsdCliRenderOutcome(NamedTuple):
    report: dict[str, JsonValue]
    images: tuple[ExecutionArtifactBinding, ...]
    responses: tuple[ExecutionArtifactBinding, ...]
    cameras: tuple[ExecutionArtifactBinding, ...]
    receipt: ExecutionArtifactBinding
    checkpoint: ExecutionArtifactBinding
    contract_path: Path


def _render_response_blank_suspects(
    binding: ExecutionArtifactBinding,
) -> tuple[str, ...]:
    """Return one usd-cli response's exact advisory blank-frame paths."""

    try:
        response = json.loads(
            verify_execution_artifact_binding(
                binding,
                label="canonical visual render response",
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedOperationError(
            "canonical visual render response is invalid"
        ) from exc
    if not isinstance(response, dict):
        raise VerifiedOperationError("canonical visual render response is invalid")
    data = response.get("data")
    if not isinstance(data, dict) or "blank_suspect" not in data:
        return ()
    suspects = data["blank_suspect"]
    if not isinstance(suspects, list) or any(
        not isinstance(item, str) or not item for item in suspects
    ):
        raise VerifiedOperationError(
            "canonical visual render response has invalid blank_suspect paths"
        )
    return tuple(suspects)


def _all_canonical_views_are_blank_suspects(
    *,
    images: tuple[ExecutionArtifactBinding, ...],
    responses: tuple[ExecutionArtifactBinding, ...],
) -> bool:
    """Fail only when no requested canonical view remains usable for review."""

    if len(images) != len(responses):
        raise VerifiedOperationError(
            "canonical visual images and render responses differ in count"
        )
    per_view: list[bool] = []
    for image, response in zip(images, responses, strict=True):
        image_path = Path(image.path).expanduser().absolute()
        suspects = _render_response_blank_suspects(response)
        per_view.append(
            any(Path(item).expanduser().absolute() == image_path for item in suspects)
        )
    return bool(per_view) and all(per_view)


def _usd_dependency_bindings(output_usd: Path) -> tuple[ExecutionArtifactBinding, ...]:
    from .workflow import _usd_dependency_paths

    closure = _usd_dependency_paths(output_usd)
    return tuple(
        execution_artifact_binding(path)
        for path in closure
        if path.resolve() != output_usd.resolve()
    )


def _component(
    *,
    component_id: str,
    version: str,
    contract: Path,
    configuration: ExecutionArtifactBinding,
) -> VerifiedOperationComponentIdentity:
    return VerifiedOperationComponentIdentity(
        component_id=component_id,
        version=version,
        contract=execution_artifact_binding(contract),
        configuration=configuration,
    )


def _render_with_package_owned_usd_cli(
    *,
    root: Path,
    output_path: Path,
    source_path: Path,
    dependencies: tuple[ExecutionArtifactBinding, ...],
    request: CanonicalVisualEvidenceRequest,
) -> _UsdCliRenderOutcome:
    """Run only the authenticated usd-cli/OVRTX render boundary."""

    # Keep this import local so provided-result ingestion stays tool-free.
    from content_agent_workflows.common import usd_cli_session as usd_cli_module

    project_dir = root / "usd_cli"
    parent_handoff = usd_cli_module._parent_usd_cli_session_identity_from_environment()
    render_config = (
        None
        if parent_handoff is not None
        else (
            usd_cli_module._environment_remote_render_config()
            if request.backend == "remote"
            else {
                # A nearer project config is deep-merged over the user's global
                # usd-cli config. Pin every transport selector so a standalone
                # local evidence request cannot inherit and contact a stale
                # remote renderer before its transport is verified.
                "renderer": "ovrtx",
                "remote_url": "",
                "remote_api_key": "",
                "backends": [],
            }
        )
    )
    session = usd_cli_module.WorkflowUsdCliSession.create(
        owner_root=root,
        project_dir=project_dir,
        identity=canonical_json_digest(request),
        workflow="validation-canonical-visual-evidence",
        render_config=render_config,
        input_roots=(
            source_path,
            output_path,
            *(Path(binding.path) for binding in dependencies),
        ),
    )
    records: list[dict[str, JsonValue]] = []
    primary_error = False
    try:
        probe = session.require_ovrtx(project_dir / "probe")
        # Directional evidence requires usd-cli to author and orbit an ephemeral
        # camera in the in-memory stage.  A read-only session correctly rejects
        # those commands.  Keep the session private/writable, never issue a save
        # or export, and re-bind the on-disk USD after rendering so any file
        # mutation still fails closed below.
        session.open(output_path, read_only=False)
        up_axis_y = session.stage_up_axis_is_y(output_path)
        for index, direction in enumerate(request.views):
            record = session.render_view(
                output_dir=project_dir / "renders",
                name=f"view-{index:03d}",
                direction=direction,
                backend=request.backend,
                width=request.image_width,
                height=request.image_height,
                up_axis_y=up_axis_y,
            )
            if record.get("renderer") != request.backend:
                raise VerifiedOperationError(
                    "canonical visual backend changed: requested "
                    f"{request.backend}, got {record.get('renderer')}"
                )
            records.append(record)
    except Exception:
        primary_error = True
        raise
    finally:
        try:
            session.close()
        except Exception as exc:
            if not primary_error:
                raise VerifiedOperationError(
                    f"could not close canonical visual usd-cli session: {exc}"
                ) from exc

    image_bindings = tuple(
        execution_artifact_binding(str(record["image_path"])) for record in records
    )
    response_bindings = tuple(
        execution_artifact_binding(str(record["response_path"])) for record in records
    )
    camera_bindings = tuple(
        execution_artifact_binding(str(record["camera_json_path"]))
        for record in records
    )
    receipt_binding = execution_artifact_binding(session.receipt_file)
    checkpoint_binding = execution_artifact_binding(session.receipt_checkpoint_file)
    source_revision = session.route.source_revision
    metadata: dict[str, JsonValue] = {
        "backend": request.backend,
        "renderer": "ovrtx",
        "scene_tool": "usd-cli",
        "scene_tool_source_revision": source_revision,
        "session_id": session.session_id,
        "workflow": session.workflow,
        "views": list(request.views),
        "image_width": request.image_width,
        "image_height": request.image_height,
        "renderer_identities": [record.get("renderer_identity") for record in records],
    }
    report: dict[str, JsonValue] = {
        "schema_version": CANONICAL_VISUAL_RENDER_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "backend": request.backend,
        "probe_schema_version": probe.get("schema_version"),
        "image_paths": [binding.path for binding in image_bindings],
        "images": [binding.model_dump(mode="json") for binding in image_bindings],
        "render_responses": [
            binding.model_dump(mode="json") for binding in response_bindings
        ],
        "camera_records": [
            binding.model_dump(mode="json") for binding in camera_bindings
        ],
        "usd_cli_command_receipt": receipt_binding.model_dump(mode="json"),
        "usd_cli_receipt_checkpoint": checkpoint_binding.model_dump(mode="json"),
        "metadata": metadata,
    }
    return _UsdCliRenderOutcome(
        report=report,
        images=image_bindings,
        responses=response_bindings,
        cameras=camera_bindings,
        receipt=receipt_binding,
        checkpoint=checkpoint_binding,
        contract_path=Path(usd_cli_module.__file__).resolve(),
    )


def produce_canonical_visual_evidence(
    *,
    post_mutation_usd: str | Path,
    output_dir: str | Path,
    source_usd: str | Path,
    backend: Literal["remote", "ovrtx"],
    views: tuple[str, ...] = ("+x+y+z",),
    image_width: int = 1024,
    image_height: int = 1024,
    operation_id: str = "visual.canonical-ovrtx",
    gate_id: str = "visual.canonical-evidence",
) -> CanonicalVisualEvidencePublication:
    """Render and bind exact post-mutation OVRTX evidence without judging it."""

    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).absolute()
    if root.exists() or root.is_symlink():
        raise VerifiedOperationError(
            f"canonical visual output already exists or is unsafe: {root}"
        )
    try:
        directory_fd, _chain = _open_directory_no_symlinks(root)
    except OSError as exc:
        raise VerifiedOperationError(
            f"canonical visual output is missing or unsafe: {root}"
        ) from exc
    os.close(directory_fd)
    output_path = Path(post_mutation_usd).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).absolute()
    source_path = Path(source_usd).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).absolute()
    source_before = execution_artifact_binding(source_path)
    output_before = execution_artifact_binding(output_path)
    dependencies_before = _usd_dependency_bindings(output_path)
    request = CanonicalVisualEvidenceRequest(
        source=source_before,
        post_mutation_output=output_before,
        dependencies=dependencies_before,
        backend=backend,
        views=views,
        image_width=image_width,
        image_height=image_height,
    )
    request_path = root / "canonical_visual_request.json"
    atomic_write_json(request_path, request)
    request_binding = execution_artifact_binding(request_path)

    try:
        render_outcome = _render_with_package_owned_usd_cli(
            root=root,
            output_path=output_path,
            source_path=source_path,
            dependencies=dependencies_before,
            request=request,
        )
    except VerifiedOperationError:
        raise
    except Exception as exc:
        raise VerifiedOperationError(
            f"canonical usd-cli OVRTX render failed: {exc}"
        ) from exc
    render_result = render_outcome.report
    image_bindings = render_outcome.images
    response_bindings = render_outcome.responses
    camera_bindings = render_outcome.cameras
    receipt_binding = render_outcome.receipt
    checkpoint_binding = render_outcome.checkpoint
    rendering_contract_path = render_outcome.contract_path
    render_report_path = root / "canonical_visual_render_report.json"
    atomic_write_json(render_report_path, render_result)
    render_report_binding = execution_artifact_binding(render_report_path)
    observed_backend = str(render_result.get("backend") or "")
    if observed_backend != backend:
        raise VerifiedOperationError(
            f"canonical visual backend changed: requested {backend}, got {observed_backend}"
        )
    if render_result.get("status") != "completed":
        raise VerifiedOperationError(
            "canonical OVRTX visual evidence did not complete; inspect "
            f"{render_report_path}"
        )
    if not image_bindings:
        raise VerifiedOperationError("canonical OVRTX result contains no images")
    if _all_canonical_views_are_blank_suspects(
        images=image_bindings,
        responses=response_bindings,
    ):
        raise VerifiedOperationError(
            "every canonical OVRTX view was flagged blank_suspect"
        )

    source_after = execution_artifact_binding(source_path)
    output_after = execution_artifact_binding(output_path)
    dependencies_after = _usd_dependency_bindings(output_path)
    if (
        source_after != source_before
        or output_after != output_before
        or dependencies_after != dependencies_before
    ):
        raise VerifiedOperationError(
            "source, post-mutation output, or dependency closure changed while rendering"
        )
    metadata = render_result.get("metadata")
    if not isinstance(metadata, dict):
        raise VerifiedOperationError("canonical OVRTX result lacks render metadata")
    source_revision = metadata.get("scene_tool_source_revision")
    if not isinstance(source_revision, str):
        raise VerifiedOperationError("canonical OVRTX result lacks usd-cli revision")
    payload = CanonicalVisualEvidencePayload(
        source=source_before,
        post_mutation_output=output_before,
        dependencies=dependencies_before,
        render_report=render_report_binding,
        images=image_bindings,
        render_responses=response_bindings,
        camera_records=camera_bindings,
        usd_cli_command_receipt=receipt_binding,
        usd_cli_receipt_checkpoint=checkpoint_binding,
        usd_cli_source_revision=source_revision,
        backend_alias=backend,
        render_process_status="completed",
        render_metadata=metadata,
    )
    payload_path = root / "canonical_visual_payload.json"
    atomic_write_json(payload_path, payload)
    payload_binding = execution_artifact_binding(payload_path)

    implementation_path = Path(__file__).resolve()
    producer = _component(
        component_id="canonical-ovrtx-visual-evidence-producer",
        version=CANONICAL_VISUAL_EVIDENCE_PAYLOAD_SCHEMA_VERSION,
        contract=implementation_path,
        configuration=request_binding,
    )
    tool = _component(
        component_id="package-owned-usd-cli-render",
        version="content-agent-workflows.workflow-usd-cli-render.v1",
        contract=rendering_contract_path,
        configuration=request_binding,
    )
    profile = _component(
        component_id="canonical-post-mutation-ovrtx",
        version=CANONICAL_VISUAL_EVIDENCE_REQUEST_SCHEMA_VERSION,
        contract=implementation_path,
        configuration=request_binding,
    )
    backend_identity = _component(
        component_id=f"ovrtx-backend-{backend}",
        version="content-agent-workflows.usd-cli-ovrtx-render-contract.v1",
        contract=rendering_contract_path,
        configuration=request_binding,
    )
    projector = _component(
        component_id="canonical-visual-evidence-projector",
        version="content-agent-workflows.canonical-visual-projector.v2",
        contract=implementation_path,
        configuration=request_binding,
    )
    verifier = _component(
        component_id="canonical-visual-evidence-byte-verifier",
        version="content-agent-workflows.verified-artifact-readback.v1",
        contract=implementation_path,
        configuration=request_binding,
    )
    all_artifacts = (
        *image_bindings,
        *response_bindings,
        *camera_bindings,
        receipt_binding,
        checkpoint_binding,
    )
    projection = VerifiedValidationOperationProjection(
        operation_id=operation_id,
        gate_id=gate_id,
        evidence_type="visual.canonical-ovrtx",
        native_report_type="usd-cli.ovrtx-render-report",
        native_payload_type="visual.canonical-usd-cli-ovrtx-payload",
        claim_scope="exact post-mutation USD OVRTX render evidence",
        native_status="pass",
        required=True,
        authority="outer_review_input",
        source=source_before,
        output=output_before,
        dependencies=dependencies_before,
        artifacts=all_artifacts,
        native_report=render_report_binding,
        native_payload=payload_binding,
        producer_identity_sha256=canonical_json_digest(producer),
        tool_identity_sha256=canonical_json_digest(tool),
        profile_identity_sha256=canonical_json_digest(profile),
        backend_identity_sha256=canonical_json_digest(backend_identity),
        verifier_identity_sha256=canonical_json_digest(verifier),
        projector_identity_sha256=canonical_json_digest(projector),
    )
    projection_path = root / "verified_operation_projection.json"
    atomic_write_json(projection_path, projection)
    projection_binding = execution_artifact_binding(projection_path)
    envelope = VerifiedValidationOperationEnvelope(
        operation_id=operation_id,
        gate_id=gate_id,
        evidence_type="visual.canonical-ovrtx",
        native_report_type="usd-cli.ovrtx-render-report",
        native_payload_type="visual.canonical-usd-cli-ovrtx-payload",
        claim_scope="exact post-mutation USD OVRTX render evidence",
        native_status="pass",
        required=True,
        authority="outer_review_input",
        source=source_before,
        output=output_before,
        dependencies=dependencies_before,
        artifacts=all_artifacts,
        native_report=render_report_binding,
        native_payload=payload_binding,
        producer=producer,
        tool=tool,
        profile=profile,
        backend=backend_identity,
        verifier=verifier,
        projector=projector,
        projection=projection_binding,
    )
    envelope_path = root / "verified_operation_envelope.json"
    atomic_write_json(envelope_path, envelope)
    envelope_binding = execution_artifact_binding(envelope_path)
    verify_operation_envelope(envelope)
    # Re-read every published binding before returning the public leaf receipt.
    for label, binding in (
        ("request", request_binding),
        ("render report", render_report_binding),
        ("payload", payload_binding),
        ("projection", projection_binding),
        ("envelope", envelope_binding),
    ):
        verify_execution_artifact_binding(binding, label=label)
    return CanonicalVisualEvidencePublication(
        request=request_binding,
        render_report=render_report_binding,
        payload=payload_binding,
        projection=projection_binding,
        envelope=envelope_binding,
        result=envelope,
    )


__all__ = [
    "CANONICAL_VISUAL_EVIDENCE_PAYLOAD_SCHEMA_VERSION",
    "CANONICAL_VISUAL_EVIDENCE_PUBLICATION_SCHEMA_VERSION",
    "CANONICAL_VISUAL_EVIDENCE_REQUEST_SCHEMA_VERSION",
    "CANONICAL_VISUAL_RENDER_REPORT_SCHEMA_VERSION",
    "CanonicalVisualEvidencePayload",
    "CanonicalVisualEvidencePublication",
    "CanonicalVisualEvidenceRequest",
    "produce_canonical_visual_evidence",
]
