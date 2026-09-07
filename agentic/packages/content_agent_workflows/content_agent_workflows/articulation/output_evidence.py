# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-local binding of shared canonical post-mutation OVRTX evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)

from .models import ArtifactBinding

EMBEDDED_ARTICULATION_OUTPUT_EVIDENCE_SCHEMA_VERSION = (
    "content-agent-workflows.embedded-articulation-output-evidence.v1"
)
EMBEDDED_ARTICULATION_TERMINAL_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.embedded-articulation-terminal-receipt.v1"
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbeddedArticulationOutputEvidence(_FrozenModel):
    """Exact Joint binding of one shared canonical visual payload."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-output-evidence.v1"
    ] = EMBEDDED_ARTICULATION_OUTPUT_EVIDENCE_SCHEMA_VERSION
    canonical_visual_envelope: ExecutionArtifactBinding
    canonical_visual_projection: ExecutionArtifactBinding
    canonical_visual_request: ExecutionArtifactBinding
    canonical_visual_payload: ExecutionArtifactBinding
    canonical_visual_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: ExecutionArtifactBinding
    post_mutation_output: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    dependency_closure_complete: Literal[True] = True
    dependency_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_report: ExecutionArtifactBinding
    images: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    backend_alias: Literal["remote", "ovrtx"]
    render_metadata: dict[str, JsonValue]
    semantic_authority: Literal["outer_coordinator"] = "outer_coordinator"

    @model_validator(mode="after")
    def validate_dependency_digest(self) -> Self:
        expected = canonical_json_digest(
            {
                "dependencies": [
                    item.model_dump(mode="json") for item in self.dependencies
                ]
            }
        )
        if self.dependency_closure_sha256 != expected:
            raise ValueError("Articulation output dependency closure digest is stale")
        return self


class EmbeddedArticulationTerminalReceipt(_FrozenModel):
    """Joint terminal index binding review, receipt, output, and OVRTX bytes."""

    schema_version: Literal[
        "content-agent-workflows.embedded-articulation-terminal-receipt.v1"
    ] = EMBEDDED_ARTICULATION_TERMINAL_RECEIPT_SCHEMA_VERSION
    accepted_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_review_patch: ArtifactBinding
    output_evidence: ArtifactBinding
    shared_coordinator_review: ArtifactBinding
    shared_decision_receipt: ArtifactBinding
    output_asset: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...] = ()
    dependency_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    render_report: ExecutionArtifactBinding
    images: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    renderer_backend_alias: Literal["remote", "ovrtx"]
    outer_review_disposition: Literal["accept"] = "accept"
    receipt_status: Literal["completed"] = "completed"

    @model_validator(mode="after")
    def validate_dependency_digest(self) -> Self:
        expected = canonical_json_digest(
            {
                "dependencies": [
                    item.model_dump(mode="json") for item in self.dependencies
                ]
            }
        )
        if self.dependency_closure_sha256 != expected:
            raise ValueError("Articulation terminal dependency closure digest is stale")
        return self


def build_embedded_articulation_output_evidence(
    canonical_visual_envelope_path: str | Path,
    *,
    expected_source: ExecutionArtifactBinding,
    expected_output: ExecutionArtifactBinding,
) -> EmbeddedArticulationOutputEvidence:
    """Validate #1153's complete publication and retain its exact review inputs."""

    from content_agent_workflows.validation.canonical_visual_evidence import (
        CANONICAL_VISUAL_EVIDENCE_PAYLOAD_SCHEMA_VERSION,
        CANONICAL_VISUAL_EVIDENCE_REQUEST_SCHEMA_VERSION,
        CANONICAL_VISUAL_RENDER_REPORT_SCHEMA_VERSION,
        CanonicalVisualEvidencePayload,
        CanonicalVisualEvidenceRequest,
    )
    from content_agent_workflows.validation.verified_operations import (
        VerifiedOperationComponentIdentity,
        load_verified_operation_envelope,
        verify_execution_artifact_binding,
    )

    envelope_binding, envelope = load_verified_operation_envelope(
        canonical_visual_envelope_path
    )
    if (
        envelope.operation_id != "visual.canonical-ovrtx"
        or envelope.gate_id != "visual.canonical-evidence"
        or envelope.evidence_type != "visual.canonical-ovrtx"
        or envelope.native_report_type != "usd-cli.ovrtx-render-report"
        or envelope.native_payload_type != "visual.canonical-usd-cli-ovrtx-payload"
        or envelope.native_status != "pass"
        or not envelope.required
        or envelope.authority != "outer_review_input"
        or envelope.native_payload is None
        or envelope.native_report is None
    ):
        raise ValueError("canonical visual envelope has unsupported semantics")
    payload_binding = envelope.native_payload
    payload = CanonicalVisualEvidencePayload.model_validate_json(
        verify_execution_artifact_binding(
            payload_binding,
            label="canonical visual payload",
        )
    )
    if payload.source != expected_source:
        raise ValueError("canonical visual evidence binds another source")
    if payload.post_mutation_output != expected_output:
        raise ValueError("canonical visual evidence binds another authored output")
    if payload.semantic_authority != "outer_coordinator":
        raise ValueError("canonical visual evidence changed semantic authority")
    if (
        envelope.source != payload.source
        or envelope.output != payload.post_mutation_output
        or envelope.dependencies != payload.dependencies
        or envelope.artifacts
        != (
            *payload.images,
            *payload.render_responses,
            *payload.camera_records,
            payload.usd_cli_command_receipt,
            payload.usd_cli_receipt_checkpoint,
        )
        or envelope.native_report != payload.render_report
    ):
        raise ValueError("canonical visual envelope differs from its payload")
    components: tuple[tuple[str, VerifiedOperationComponentIdentity | None], ...] = (
        ("producer", envelope.producer),
        ("tool", envelope.tool),
        ("profile", envelope.profile),
        ("backend", envelope.backend),
        ("verifier", envelope.verifier),
        ("projector", envelope.projector),
    )
    expected_components = {
        "producer": (
            "canonical-ovrtx-visual-evidence-producer",
            CANONICAL_VISUAL_EVIDENCE_PAYLOAD_SCHEMA_VERSION,
        ),
        "tool": (
            "package-owned-usd-cli-render",
            "content-agent-workflows.workflow-usd-cli-render.v1",
        ),
        "profile": (
            "canonical-post-mutation-ovrtx",
            CANONICAL_VISUAL_EVIDENCE_REQUEST_SCHEMA_VERSION,
        ),
        "backend": (
            f"ovrtx-backend-{payload.backend_alias}",
            "content-agent-workflows.usd-cli-ovrtx-render-contract.v1",
        ),
        "verifier": (
            "canonical-visual-evidence-byte-verifier",
            "content-agent-workflows.verified-artifact-readback.v1",
        ),
        "projector": (
            "canonical-visual-evidence-projector",
            "content-agent-workflows.canonical-visual-projector.v2",
        ),
    }
    configurations = []
    for label, component in components:
        if component is None:
            raise ValueError(f"canonical visual envelope lacks {label} identity")
        if (component.component_id, component.version) != expected_components[label]:
            raise ValueError(f"canonical visual envelope has unsupported {label}")
        if component.configuration is None:
            raise ValueError(f"canonical visual {label} lacks exact configuration")
        configurations.append(component.configuration)
    if len(set(configurations)) != 1:
        raise ValueError("canonical visual component configurations differ")
    request_binding = configurations[0]
    request = CanonicalVisualEvidenceRequest.model_validate_json(
        verify_execution_artifact_binding(
            request_binding,
            label="canonical visual request",
        )
    )
    if (
        request.source != payload.source
        or request.post_mutation_output != payload.post_mutation_output
        or request.dependencies != payload.dependencies
        or request.backend != payload.backend_alias
        or list(request.views) != payload.render_metadata.get("views")
        or request.image_width != payload.render_metadata.get("image_width")
        or request.image_height != payload.render_metadata.get("image_height")
    ):
        raise ValueError("canonical visual request differs from rendered payload")
    try:
        render_report = json.loads(
            verify_execution_artifact_binding(
                payload.render_report,
                label="canonical visual render report",
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical visual render report is invalid") from exc
    if (
        not isinstance(render_report, dict)
        or render_report.get("schema_version")
        != CANONICAL_VISUAL_RENDER_REPORT_SCHEMA_VERSION
        or render_report.get("status") != "completed"
        or render_report.get("backend") != payload.backend_alias
        or render_report.get("metadata") != payload.render_metadata
        or render_report.get("images")
        != [item.model_dump(mode="json") for item in payload.images]
        or render_report.get("render_responses")
        != [item.model_dump(mode="json") for item in payload.render_responses]
        or render_report.get("camera_records")
        != [item.model_dump(mode="json") for item in payload.camera_records]
        or render_report.get("usd_cli_command_receipt")
        != payload.usd_cli_command_receipt.model_dump(mode="json")
        or render_report.get("usd_cli_receipt_checkpoint")
        != payload.usd_cli_receipt_checkpoint.model_dump(mode="json")
        or payload.render_metadata.get("scene_tool") != "usd-cli"
        or payload.render_metadata.get("scene_tool_source_revision")
        != payload.usd_cli_source_revision
    ):
        raise ValueError("canonical visual render report differs from its payload")
    for label, binding in (
        ("canonical visual payload", payload_binding),
        ("canonical visual source", payload.source),
        ("canonical visual output", payload.post_mutation_output),
        ("canonical visual render report", payload.render_report),
        *(("canonical visual dependency", item) for item in payload.dependencies),
        *(("canonical visual image", item) for item in payload.images),
        *(
            ("canonical visual render response", item)
            for item in payload.render_responses
        ),
        *(("canonical visual camera record", item) for item in payload.camera_records),
        ("canonical visual usd-cli receipt", payload.usd_cli_command_receipt),
        (
            "canonical visual usd-cli receipt checkpoint",
            payload.usd_cli_receipt_checkpoint,
        ),
    ):
        verify_execution_artifact_binding(binding, label=label)
    dependencies = tuple(payload.dependencies)
    return EmbeddedArticulationOutputEvidence(
        canonical_visual_envelope=envelope_binding,
        canonical_visual_projection=envelope.projection,
        canonical_visual_request=request_binding,
        canonical_visual_payload=payload_binding,
        canonical_visual_contract_sha256=canonical_json_digest(payload),
        source=payload.source,
        post_mutation_output=payload.post_mutation_output,
        dependencies=dependencies,
        dependency_closure_sha256=canonical_json_digest(
            {"dependencies": [item.model_dump(mode="json") for item in dependencies]}
        ),
        render_report=payload.render_report,
        images=tuple(payload.images),
        backend_alias=payload.backend_alias,
        render_metadata=dict(payload.render_metadata),
    )


def validate_embedded_articulation_output_evidence(
    evidence: EmbeddedArticulationOutputEvidence,
) -> None:
    """Revalidate every shared visual artifact from disk."""

    from content_agent_workflows.validation.verified_operations import (
        load_verified_operation_envelope,
    )

    rebuilt = build_embedded_articulation_output_evidence(
        evidence.canonical_visual_envelope.path,
        expected_source=evidence.source,
        expected_output=evidence.post_mutation_output,
    )
    if rebuilt != evidence:
        raise ValueError("embedded Articulation output evidence is stale")
    # Ensure the explicitly retained projection remains the envelope's exact one.
    _binding, envelope = load_verified_operation_envelope(
        evidence.canonical_visual_envelope.path
    )
    if envelope.projection != evidence.canonical_visual_projection:
        raise ValueError("embedded Articulation visual projection is stale")


__all__ = [
    "EMBEDDED_ARTICULATION_OUTPUT_EVIDENCE_SCHEMA_VERSION",
    "EMBEDDED_ARTICULATION_TERMINAL_RECEIPT_SCHEMA_VERSION",
    "EmbeddedArticulationOutputEvidence",
    "EmbeddedArticulationTerminalReceipt",
    "build_embedded_articulation_output_evidence",
    "validate_embedded_articulation_output_evidence",
]
