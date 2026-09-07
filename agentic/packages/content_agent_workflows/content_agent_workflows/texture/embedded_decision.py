# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outer-owned embedded Texture decisions over replaceable provider leaves."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pxr import Ar, UsdUtils
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_artifact_store import (
    EmbeddedDecisionArtifactStore,
)
from content_agent_workflows.common.embedded_domain_decision import (
    AcceptedSemanticDecision,
    BoundedExecutionAuthorization,
    ContractArtifactReference,
    DomainProposalPayload,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorDecision,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionIdentity,
    EmbeddedDecisionReceipt,
    EmbeddedDomainEvidence,
    EmbeddedDomainProposal,
    NamedDecisionDigests,
    PersistedExecutionLineage,
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    accepted_semantic_decision_digest,
    artifact_reference,
    authorize_bounded_execution,
    build_decision_receipt,
    canonical_json_digest,
    validate_coordinator_decision_dependencies,
    validate_decision_receipt,
)

from .models import (
    TextureCandidateReviewDecision,
    TextureCanonicalPlan,
    TextureExecutionResult,
    TextureInspectionResult,
    TexturePlanDocument,
    TexturePublicationDecision,
    TexturePublicationReviewDecision,
    TextureUnitArtifact,
    TextureValidationResult,
    TextureWorkflowRequest,
)

TEXTURE_EMBEDDED_PATCH_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-embedded-decision-patch.v1"
] = "content-agent-workflows.texture-embedded-decision-patch.v1"
TEXTURE_EMBEDDED_CRITICAL_IMPLEMENTATION_FILES: tuple[str, ...] = (
    "__init__.py",
    "client.py",
    "decision.py",
    "embedded_decision.py",
    "embedded_workflow.py",
    "finalizer.py",
    "models.py",
    "runtime.py",
    "scope_validation.py",
    "scene_validation.py",
    "workflow.py",
)
TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY = (
    "content_agent_workflows.texture_embedded_decision_identity"
)

TextureEmbeddedAction = Literal[
    "execute",
    "validate",
    "refine",
    "finalize",
    "review_publication",
]


class TextureEmbeddedDecisionPatch(BaseModel):
    """One outer-authored semantic transition for an embedded Texture run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.texture-embedded-decision-patch.v1"
    ] = TEXTURE_EMBEDDED_PATCH_SCHEMA_VERSION
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_revision: int = Field(ge=1)
    action: TextureEmbeddedAction
    iteration: int = Field(ge=0)
    created_at: datetime
    canonical_plan: TextureCanonicalPlan | None = None
    regeneration_unit_ids: tuple[str, ...] = ()
    candidate_review: TextureCandidateReviewDecision | None = None
    publication: TexturePublicationDecision | None = None
    publication_review: TexturePublicationReviewDecision | None = None
    rationale: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_action_payload(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError(
                "embedded Texture decision timestamp must be timezone-aware"
            )
        payloads = {
            "canonical_plan": self.canonical_plan is not None,
            "candidate_review": self.candidate_review is not None,
            "publication": self.publication is not None,
            "publication_review": self.publication_review is not None,
        }
        expected = {
            "execute": "canonical_plan",
            "validate": "candidate_review",
            "refine": "canonical_plan",
            "finalize": "publication",
            "review_publication": "publication_review",
        }[self.action]
        if not payloads[expected] or any(
            present for name, present in payloads.items() if name != expected
        ):
            raise ValueError(f"embedded Texture {self.action} requires only {expected}")
        if self.action == "refine" and not self.regeneration_unit_ids:
            raise ValueError("embedded Texture refinement requires exact unit IDs")
        if self.action != "refine" and self.regeneration_unit_ids:
            raise ValueError("regeneration_unit_ids are valid only for refine")
        if len(self.regeneration_unit_ids) != len(set(self.regeneration_unit_ids)):
            raise ValueError("regeneration_unit_ids must be unique")
        return self


class TextureEmbeddedDecisionState(BaseModel):
    """Shared-contract references checkpointed with the native Texture state."""

    model_config = ConfigDict(extra="forbid")

    identity: EmbeddedDecisionIdentity
    inspection: TextureInspectionResult
    inspection_evidence: ContractArtifactReference
    plan_proposal: ContractArtifactReference
    canonical_plan: TextureCanonicalPlan | None = None
    current_decision: ContractArtifactReference | None = None
    current_authorization: ContractArtifactReference | None = None
    current_result: ContractArtifactReference | None = None
    current_candidate_evidence: ContractArtifactReference | None = None
    current_candidate_domain_review: ContractArtifactReference | None = None
    current_candidate_review: ContractArtifactReference | None = None
    current_candidate_receipt: ContractArtifactReference | None = None
    accepted_candidate_result: ContractArtifactReference | None = None
    accepted_candidate_domain_review: ContractArtifactReference | None = None
    accepted_candidate_review: ContractArtifactReference | None = None
    accepted_candidate_receipt: ContractArtifactReference | None = None
    publication_decision: ContractArtifactReference | None = None
    publication_authorization: ContractArtifactReference | None = None
    publication_result: ContractArtifactReference | None = None
    publication_review: ContractArtifactReference | None = None
    completed_receipt: ContractArtifactReference | None = None


class TextureEmbeddedStepObservation(BaseModel):
    """Typed outer-coordinator packet for one embedded Texture boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.texture-embedded-step-observation.v1"
    ] = "content-agent-workflows.texture-embedded-step-observation.v1"
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_decision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_revision: int = Field(ge=1)
    action: TextureEmbeddedAction | Literal["done"]
    iteration: int = Field(ge=0)
    target_unit_ids: tuple[str, ...]
    accepted_unit_ids: tuple[str, ...]
    remaining_unit_ids: tuple[str, ...]
    inspection: TextureInspectionResult
    service_plan_proposal: TexturePlanDocument
    canonical_plan: TextureCanonicalPlan | None = None
    current_candidate_result: ContractArtifactReference | None = None
    current_candidate_evidence: ContractArtifactReference | None = None
    candidate_output: ExecutionArtifactBinding | None = None
    visual_evidence_artifacts: tuple[ExecutionArtifactBinding, ...] = ()
    accepted_candidate_receipt: ContractArtifactReference | None = None
    publication_result: ContractArtifactReference | None = None
    publication_output: ExecutionArtifactBinding | None = None
    completed_receipt: ContractArtifactReference | None = None
    decision_patch_path: str | None = None
    terminal: bool


def _digest_descriptor(value: JsonValue) -> str:
    return canonical_json_digest(
        {
            "schema_version": "content-agent-workflows.texture-implementation.v1",
            "value": value,
        }
    )


def texture_embedded_implementation_digests() -> dict[str, str]:
    """Bind the exact coordinator, candidate, validator, and publisher code."""

    package_root = Path(__file__).resolve().parent
    critical_file_digests = {
        filename: file_sha256(package_root / filename)
        for filename in TEXTURE_EMBEDDED_CRITICAL_IMPLEMENTATION_FILES
    }
    return {
        "texture-embedded-implementation-set.v1": _digest_descriptor(
            critical_file_digests
        ),
        "asset-coordinator.texture.v1": file_sha256(
            package_root / "embedded_workflow.py"
        ),
        "texture-inspection-adapter.v1": file_sha256(
            package_root / "scene_validation.py"
        ),
        "texture-agent-plan-provider.v1": file_sha256(package_root / "client.py"),
        "texture-agent-candidate-executor.v1": file_sha256(package_root / "client.py"),
        "texture-publication-executor.v1": file_sha256(
            package_root / "embedded_workflow.py"
        ),
    }


def texture_embedded_capability_digests() -> dict[str, str]:
    """Return the provider-neutral capability manifest identity."""

    return {
        "texture-inspection": _digest_descriptor(
            {
                "material_scope": True,
                "uv_scope": True,
                "source_render_evidence": True,
            }
        ),
        "texture-candidate-generation": _digest_descriptor(
            {"execution_effect": "non_mutating", "surface_texturing_only": True}
        ),
        "texture-publication": _digest_descriptor(
            {
                "execution_effect": "mutation",
                "exact_candidate": True,
                "dependency_closure": True,
            }
        ),
    }


def validate_embedded_texture_identity_manifests(
    identity: EmbeddedDecisionIdentity,
) -> None:
    """Require a frozen identity to match the active embedded implementation."""

    if dict(identity.digests.capabilities) != texture_embedded_capability_digests():
        raise ValueError("Texture decision capability manifest changed")
    if (
        dict(identity.digests.implementations)
        != texture_embedded_implementation_digests()
    ):
        raise ValueError("Texture decision implementation manifest changed")


def outer_coordinator(identity: EmbeddedDecisionIdentity) -> ProducerIdentity:
    stage = identity.execution_context.embedded_stage
    if stage is None:  # pragma: no cover - identity invariant
        raise ValueError("embedded Texture identity lacks an outer stage")
    return ProducerIdentity(
        producer_id=f"asset-coordinator:{stage.outer_run_id}",
        role="outer_coordinator",
        implementation="asset-coordinator.texture.v1",
        implementation_digest=identity.digests.implementations[
            "asset-coordinator.texture.v1"
        ],
    )


def _producer(
    identity: EmbeddedDecisionIdentity,
    *,
    producer_id: str,
    role: Literal["evidence_provider", "proposal_provider", "executor"],
    implementation: str,
) -> ProducerIdentity:
    return ProducerIdentity(
        producer_id=producer_id,
        role=role,
        implementation=implementation,
        implementation_digest=identity.digests.implementations[implementation],
    )


def candidate_executor(identity: EmbeddedDecisionIdentity) -> ProducerIdentity:
    return _producer(
        identity,
        producer_id="texture-agent-candidate-leaf",
        role="executor",
        implementation="texture-agent-candidate-executor.v1",
    )


def publication_executor(identity: EmbeddedDecisionIdentity) -> ProducerIdentity:
    return _producer(
        identity,
        producer_id="texture-deterministic-publication",
        role="executor",
        implementation="texture-publication-executor.v1",
    )


def bind_file(path: str | Path) -> ExecutionArtifactBinding:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Texture contract artifact is not a file: {resolved}")
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _dependency_path(item: object, *, root: Path) -> Path | None:
    raw_path = getattr(item, "realPath", None) or str(item)
    raw_text = str(raw_path).strip()
    if not raw_text:
        return None
    if Ar.IsPackageRelativePath(raw_text):
        raw_text, _member_path = Ar.SplitPackageRelativePathOuter(raw_text)
    dependency = Path(raw_text).expanduser()
    if not dependency.is_absolute():
        dependency = root.parent / dependency
    return dependency.resolve()


def texture_candidate_dependency_closure_facts(
    candidate: ExecutionArtifactBinding,
) -> dict[str, JsonValue]:
    """Seal a root-only USD or byte-self-contained USDZ candidate."""

    candidate_path = Path(candidate.path).expanduser().resolve()
    if bind_file(candidate_path) != candidate:
        raise ValueError(
            "Texture candidate changed before dependency-closure inspection"
        )
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            str(candidate_path)
        )
    except Exception as exc:
        raise ValueError(
            f"Could not inspect Texture candidate dependency closure: {exc}"
        ) from exc
    unresolved_paths = tuple(sorted(str(item) for item in unresolved))
    if unresolved_paths:
        raise ValueError(
            "Texture candidate dependency closure is unresolved: "
            + ", ".join(unresolved_paths)
        )
    package_kind = "usdz" if candidate_path.suffix.lower() == ".usdz" else "root_only"
    external_dependencies = {
        dependency
        for item in (*layers, *assets)
        if (dependency := _dependency_path(item, root=candidate_path)) is not None
        and dependency != candidate_path
    }
    if external_dependencies:
        requirement = (
            "must be a self-contained USDZ"
            if package_kind != "usdz"
            else "USDZ must not resolve dependencies outside its package bytes"
        )
        raise ValueError(
            f"Embedded Texture candidate {requirement}; external dependency closure: "
            + ", ".join(str(path) for path in sorted(external_dependencies))
        )
    return {
        "schema_version": (
            "content-agent-workflows.texture-candidate-dependency-closure.v1"
        ),
        "candidate_sha256": candidate.sha256,
        "candidate_size_bytes": candidate.size_bytes,
        "package_kind": package_kind,
        "layer_count": len(layers),
        "asset_count": len(assets),
        "unresolved_dependency_count": 0,
        "external_dependency_count": 0,
        "closure_sealed_by_candidate_bytes": True,
    }


def build_embedded_texture_identity(
    request: TextureWorkflowRequest,
    *,
    inspection: TextureInspectionResult,
) -> EmbeddedDecisionIdentity:
    """Build the exact provider-neutral identity for one embedded stage attempt."""

    from .runtime import texture_request_digest

    context = request.execution_context
    if context is None or context.mode != "embedded" or context.embedded_stage is None:
        raise ValueError("shared Texture decisions require an embedded request")
    coordinator_plan = context.embedded_stage.coordinator_plan
    persisted_identity = request.metadata.get(
        TEXTURE_EMBEDDED_DECISION_IDENTITY_METADATA_KEY
    )
    if persisted_identity is not None:
        identity = EmbeddedDecisionIdentity.model_validate(persisted_identity)
        if identity.execution_context != context:
            raise ValueError(
                "Texture decision identity differs from the embedded stage context"
            )
        if identity.digests.configuration.get(
            "texture_request"
        ) != texture_request_digest(request):
            raise ValueError("Texture decision identity binds another request")
        validate_embedded_texture_identity_manifests(identity)
        return identity
    implementations = texture_embedded_implementation_digests()
    return EmbeddedDecisionIdentity(
        execution_context=context,
        source=context.embedded_stage.input_asset,
        coordinator_plan=ContractArtifactReference(
            artifact_kind="coordinator_plan",
            artifact_id=f"outer-plan-{coordinator_plan.sha256}",
            schema_version="content-agent-workflows.asset-coordinator-plan.v1",
            sha256=coordinator_plan.sha256,
        ),
        digests=NamedDecisionDigests(
            configuration={
                "texture_request": texture_request_digest(request),
            },
            prompt={"texture-intent": _digest_descriptor(request.intent)},
            references={
                reference.role: reference.artifact.sha256
                for reference in request.reference_artifacts
            },
            capabilities={
                **texture_embedded_capability_digests(),
            },
            implementations=implementations,
        ),
    )


def initialize_embedded_texture_decisions(
    request: TextureWorkflowRequest,
    plan: TexturePlanDocument,
    inspection: TextureInspectionResult,
    *,
    created_at: datetime | None = None,
) -> TextureEmbeddedDecisionState:
    """Persist inspection evidence and the service plan as a proposal only."""

    from .runtime import texture_plan_digest

    if inspection.proposal_plan_digest != texture_plan_digest(plan):
        raise ValueError("Texture inspection and service plan proposal differ")
    if tuple(unit.unit_id for unit in inspection.units) != plan.selected_unit_ids:
        raise ValueError(
            "Texture inspection must exactly cover the service proposal unit order"
        )
    if inspection.reference_artifacts != request.reference_artifacts:
        raise ValueError("Texture inspection reference identities differ from request")
    identity = build_embedded_texture_identity(request, inspection=inspection)
    timestamp = created_at or datetime.now(UTC)
    store = EmbeddedDecisionArtifactStore(request.output_dir)
    evidence = _inspection_evidence_artifact(
        identity,
        inspection,
        created_at=timestamp,
    )
    store.append(evidence)
    proposal = _plan_proposal_artifact(
        identity,
        inspection,
        plan,
        evidence=artifact_reference(evidence),
        created_at=timestamp,
    )
    store.append(proposal)
    return TextureEmbeddedDecisionState(
        identity=identity,
        inspection=inspection,
        inspection_evidence=artifact_reference(evidence),
        plan_proposal=artifact_reference(proposal),
    )


def _inspection_evidence_artifact(
    identity: EmbeddedDecisionIdentity,
    inspection: TextureInspectionResult,
    *,
    created_at: datetime,
) -> EmbeddedDomainEvidence:
    """Build the exact provider-neutral inspection evidence envelope."""

    evidence_provider = _producer(
        identity,
        producer_id="texture-inspection-provider",
        role="evidence_provider",
        implementation="texture-inspection-adapter.v1",
    )
    evidence = EmbeddedDomainEvidence(
        artifact_id=f"texture-inspection-{inspection.proposal_plan_digest}",
        identity=identity,
        producer=evidence_provider,
        parent_artifact=identity.coordinator_plan,
        created_at=created_at,
        records=(
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-source-material-uv-inspection",
                evidence_type="inspection",
                status="available",
                artifacts=(
                    inspection.source,
                    *inspection.inspection_artifacts,
                ),
                facts={
                    "units": [
                        unit.model_dump(mode="json") for unit in inspection.units
                    ],
                    "tool_metadata": inspection.tool_metadata,
                },
                summary="Exact material, UV, and source inspection before planning.",
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-before-renders",
                evidence_type="render",
                status="available",
                artifacts=inspection.before_render_artifacts,
                facts={"renderer_metadata": inspection.renderer_metadata},
                summary="Fresh provider-neutral source renders before planning.",
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-capability-constraints",
                evidence_type="capability",
                status="available",
                facts={"constraints": list(inspection.capability_constraints)},
                summary="Explicit capabilities and unsupported operations.",
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-reference-artifacts",
                evidence_type="artifact",
                status=(
                    "available" if inspection.reference_artifacts else "unavailable"
                ),
                required=bool(inspection.reference_artifacts),
                artifacts=tuple(
                    item.artifact for item in inspection.reference_artifacts
                ),
                facts=(
                    {"roles": [item.role for item in inspection.reference_artifacts]}
                    if inspection.reference_artifacts
                    else {}
                ),
                summary=(
                    "Exact reference artifacts available to outer visual reasoning."
                    if inspection.reference_artifacts
                    else "No reference artifact was requested for this Texture task."
                ),
            ),
        ),
    )
    return evidence


def _plan_proposal_artifact(
    identity: EmbeddedDecisionIdentity,
    inspection: TextureInspectionResult,
    plan: TexturePlanDocument,
    *,
    evidence: ContractArtifactReference,
    created_at: datetime,
) -> EmbeddedDomainProposal:
    """Build the service-plan proposal without granting it semantic authority."""

    proposal_provider = _producer(
        identity,
        producer_id="texture-agent-service-plan",
        role="proposal_provider",
        implementation="texture-agent-plan-provider.v1",
    )
    proposal_payload = DomainProposalPayload(
        schema_version="content-agent-workflows.texture-plan-proposal.v1",
        values={"service_plan": plan.model_dump(mode="json")},
    )
    proposal = EmbeddedDomainProposal(
        artifact_id=f"texture-plan-proposal-{inspection.proposal_plan_digest}",
        identity=identity,
        producer=proposal_provider,
        parent_artifact=evidence,
        created_at=created_at,
        evidence_artifacts=(evidence,),
        proposal=proposal_payload,
        proposal_digest=canonical_json_digest(proposal_payload),
    )
    return proposal


def validate_texture_initial_decision_inputs(
    store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
    plan: TexturePlanDocument,
) -> tuple[EmbeddedDomainEvidence, EmbeddedDomainProposal]:
    """Replay exact inspection and provider proposal bytes from durable state."""

    from .runtime import texture_plan_digest

    if (
        state.inspection.source != state.identity.source
        or state.inspection.proposal_plan_digest != texture_plan_digest(plan)
        or tuple(unit.unit_id for unit in state.inspection.units)
        != plan.selected_unit_ids
    ):
        raise ValueError("Texture inspection, identity, and checkpoint plan differ")
    inspection = store.load_typed(
        state.inspection_evidence,
        EmbeddedDomainEvidence,
    )
    expected_inspection = _inspection_evidence_artifact(
        state.identity,
        state.inspection,
        created_at=inspection.created_at,
    )
    if inspection != expected_inspection:
        raise ValueError("Persisted Texture inspection differs from checkpoint state")
    proposal = store.load_typed(state.plan_proposal, EmbeddedDomainProposal)
    expected_proposal = _plan_proposal_artifact(
        state.identity,
        state.inspection,
        plan,
        evidence=state.inspection_evidence,
        created_at=proposal.created_at,
    )
    if proposal != expected_proposal:
        raise ValueError(
            "Persisted Texture service proposal differs from checkpoint plan"
        )
    for binding in (
        state.inspection.source,
        *state.inspection.inspection_artifacts,
        *state.inspection.before_render_artifacts,
        *(item.artifact for item in state.inspection.reference_artifacts),
    ):
        if bind_file(binding.path) != binding:
            raise ValueError(
                "Texture initial inspection artifact bytes changed after planning"
            )
    return inspection, proposal


def validate_outer_texture_plan(
    canonical: TextureCanonicalPlan,
    *,
    state: TextureEmbeddedDecisionState,
    proposal: TexturePlanDocument,
) -> None:
    """Reject target, appearance-input, capability, or source drift."""

    from .runtime import texture_plan_digest

    if canonical.source != state.identity.source:
        raise ValueError("outer Texture plan source differs from embedded input")
    if canonical.proposal_plan_digest != texture_plan_digest(proposal):
        raise ValueError("outer Texture plan binds a stale service proposal")
    if canonical.capability_constraints != state.inspection.capability_constraints:
        raise ValueError("outer Texture plan capability constraints drifted")
    if canonical.reference_artifacts != state.inspection.reference_artifacts:
        raise ValueError("outer Texture plan reference identities drifted")
    if (
        not state.inspection.renderer_metadata
        or not state.inspection.tool_metadata
        or not state.inspection.before_render_artifacts
        or not state.inspection.inspection_artifacts
    ):
        raise ValueError(
            "outer Texture plan requires renderer, tool, source, and inspection "
            "evidence"
        )
    inspected = {unit.unit_id: unit for unit in state.inspection.units}
    inspected_unit_ids = tuple(unit.unit_id for unit in state.inspection.units)
    if canonical.target_unit_ids != inspected_unit_ids:
        raise ValueError("outer Texture plan must cover the exact inspected unit order")
    if canonical.target_unit_ids != proposal.selected_unit_ids:
        raise ValueError("Texture service proposal differs from inspected target scope")
    for target in canonical.targets:
        unit = inspected.get(target.unit_id)
        if unit is None:
            raise ValueError(f"outer Texture plan invented target {target.unit_id}")
        if unit.uv_status != "ready":
            raise ValueError(
                f"outer Texture plan cannot accept unavailable UV evidence: "
                f"{target.unit_id}={unit.uv_status}"
            )
        if (
            target.material_prim_paths != unit.material_prim_paths
            or target.member_prim_paths != unit.member_prim_paths
            or target.member_subset_paths != unit.member_subset_paths
        ):
            raise ValueError(
                f"outer Texture target scope differs from inspection: {target.unit_id}"
            )
        if target.generator_inputs.reference_artifacts != tuple(
            item.artifact for item in canonical.reference_artifacts
        ):
            raise ValueError(
                f"outer Texture generator references are incomplete: {target.unit_id}"
            )
        # The combined _agent-step compatibility adapter executes the service's
        # immutable plan and cannot forward an independently authored generator
        # request. Keep this legacy boundary fail-closed. The canonical focused
        # ``invoke_texture_generator`` capability passes outer inputs directly to
        # the explicitly selected generator leaf and does not apply this equality.
        if target.generator_inputs.prompt != target.requested_appearance:
            raise ValueError(
                "legacy outer Texture generator prompt must bind requested appearance"
            )
        if target.generator_inputs != unit.proposed_generator_inputs:
            raise ValueError(
                f"legacy outer Texture generator inputs differ from the executable "
                f"service proposal: {target.unit_id}"
            )


def build_plan_decision(
    patch: TextureEmbeddedDecisionPatch,
    *,
    store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
    proposal: TexturePlanDocument,
) -> tuple[
    EmbeddedCoordinatorDecision,
    tuple[EmbeddedDomainEvidence, ...],
    tuple[EmbeddedDomainProposal, ...],
]:
    canonical = patch.canonical_plan
    if canonical is None:  # pragma: no cover - patch invariant
        raise ValueError("Texture plan decision is missing canonical semantics")
    validate_outer_texture_plan(canonical, state=state, proposal=proposal)
    if patch.action == "refine":
        if state.canonical_plan is None:
            raise ValueError("Texture refinement requires an accepted canonical plan")
        if canonical != state.canonical_plan:
            raise ValueError("Texture refinement cannot replace the canonical plan")
        if set(patch.regeneration_unit_ids) - set(canonical.target_unit_ids):
            raise ValueError("Texture refinement contains units outside the plan")
    inspection, plan_proposal = validate_texture_initial_decision_inputs(
        store,
        state,
        proposal,
    )
    evidence: tuple[EmbeddedDomainEvidence, ...] = (inspection,)
    evidence_refs: tuple[ContractArtifactReference, ...] = (state.inspection_evidence,)
    parent = state.plan_proposal
    if patch.action == "refine":
        if state.current_candidate_evidence is None:
            raise ValueError("Texture refinement requires candidate critique evidence")
        current_evidence = store.load_typed(
            state.current_candidate_evidence,
            EmbeddedDomainEvidence,
        )
        evidence = (*evidence, current_evidence)
        evidence_refs = (*evidence_refs, state.current_candidate_evidence)
        parent = state.current_candidate_evidence
    accepted = AcceptedSemanticDecision(
        schema_version=canonical.schema_version,
        values={
            "canonical_plan": canonical.model_dump(mode="json"),
            "regeneration_unit_ids": list(patch.regeneration_unit_ids),
        },
    )
    decision = EmbeddedCoordinatorDecision(
        artifact_id=f"texture-{patch.action}-decision-{patch.checkpoint_revision:04d}",
        identity=state.identity,
        producer=outer_coordinator(state.identity),
        parent_artifact=parent,
        created_at=patch.created_at,
        disposition="accept",
        evidence_artifacts=evidence_refs,
        proposal_artifacts=(state.plan_proposal,),
        accepted_decision=accepted,
        accepted_decision_digest=accepted_semantic_decision_digest(
            state.identity, accepted
        ),
        rationale=patch.rationale,
    )
    validate_coordinator_decision_dependencies(
        decision,
        evidence=evidence,
        proposals=(plan_proposal,),
    )
    return decision, evidence, (plan_proposal,)


def authorize_candidate_generation(
    decision: EmbeddedCoordinatorDecision,
    *,
    evidence: tuple[EmbeddedDomainEvidence, ...],
    proposals: tuple[EmbeddedDomainProposal, ...],
) -> BoundedExecutionAuthorization:
    return authorize_bounded_execution(
        decision,
        expected_identity=decision.identity,
        evidence=evidence,
        proposals=proposals,
        existing_authorizations=(),
        executor=candidate_executor(decision.identity),
        execution_effect="non_mutating",
    )


def candidate_result_artifact(
    authorization: BoundedExecutionAuthorization,
    execution: TextureExecutionResult,
    validation: TextureValidationResult,
    *,
    candidate_closure_artifact: ExecutionArtifactBinding,
    candidate_closure_facts: Mapping[str, JsonValue],
    candidate_artifacts: Sequence[TextureUnitArtifact] | None = None,
    created_at: datetime | None = None,
) -> EmbeddedBoundedExecutionResult:
    output = bind_file(execution.output_asset_path)
    retained_artifacts = tuple(candidate_artifacts or execution.unit_artifacts)
    artifact_bindings = tuple(
        bind_file(path)
        for artifact in retained_artifacts
        for path in artifact.artifact_paths
    )
    visual_bindings = tuple(
        dict.fromkeys(
            bind_file(path)
            for finding in validation.findings
            for path in finding.evidence_artifact_paths
        )
    )
    return EmbeddedBoundedExecutionResult(
        artifact_id=f"texture-candidate-result-{authorization.operation_id}",
        identity=authorization.identity,
        producer=authorization.executor,
        parent_artifact=artifact_reference(authorization),
        created_at=created_at or datetime.now(UTC),
        accepted_decision=authorization.accepted_decision,
        accepted_decision_digest=authorization.accepted_decision_digest,
        execution_effect="non_mutating",
        operation_id=authorization.operation_id,
        mutation_id=None,
        attempt=authorization.attempt,
        status="succeeded",
        mutation_state="not_applicable",
        outputs=(output, *artifact_bindings),
        evidence=(
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-candidate-artifacts",
                evidence_type="artifact",
                status="available",
                artifacts=(output, *artifact_bindings),
                facts={
                    "requested_unit_ids": list(execution.requested_unit_ids),
                    "candidate_unit_ids": [
                        artifact.unit_id for artifact in retained_artifacts
                    ],
                    "candidate_output_sha256": output.sha256,
                    "candidate_identity": canonical_json_digest(
                        {
                            "schema_version": (
                                "content-agent-workflows.texture-candidate.v1"
                            ),
                            "authorization_operation_id": authorization.operation_id,
                            "output_sha256": output.sha256,
                            "requested_unit_ids": list(execution.requested_unit_ids),
                        }
                    ),
                },
                summary="Bounded candidate leaf output; not canonical publication.",
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-before-after-visual-critique",
                evidence_type="critique",
                status="available",
                artifacts=visual_bindings,
                facts={
                    "findings": [
                        finding.model_dump(mode="json")
                        for finding in validation.findings
                    ]
                },
                summary=(
                    "Fresh usd-cli/VQA output is evidence only; it does not accept "
                    "or reject a unit."
                ),
            ),
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-candidate-dependency-closure",
                evidence_type="inspection",
                status="available",
                artifacts=(candidate_closure_artifact,),
                facts=dict(candidate_closure_facts),
                summary=(
                    "Exact candidate dependency closure reviewed before publication."
                ),
            ),
        ),
    )


def candidate_evidence_artifact(
    result: EmbeddedBoundedExecutionResult,
    *,
    created_at: datetime | None = None,
) -> EmbeddedDomainEvidence:
    return EmbeddedDomainEvidence(
        artifact_id=f"texture-candidate-evidence-{artifact_reference(result).sha256}",
        identity=result.identity,
        producer=_producer(
            result.identity,
            producer_id="texture-candidate-evidence-adapter",
            role="evidence_provider",
            implementation="texture-inspection-adapter.v1",
        ),
        parent_artifact=result.identity.coordinator_plan,
        created_at=created_at or result.created_at,
        records=result.evidence,
    )


def map_candidate_review_disposition(
    review: TextureCandidateReviewDecision,
) -> Literal["accept", "reject", "revise"]:
    dispositions = {item.disposition for item in review.unit_dispositions}
    if dispositions == {"accept"}:
        return "accept"
    if "reject" in dispositions:
        return "reject"
    return "revise"


def build_candidate_review(
    patch: TextureEmbeddedDecisionPatch,
    *,
    state: TextureEmbeddedDecisionState,
    decision: EmbeddedCoordinatorDecision,
    result: EmbeddedBoundedExecutionResult,
) -> EmbeddedCoordinatorReview:
    domain_review = patch.candidate_review
    if domain_review is None:  # pragma: no cover - patch invariant
        raise ValueError("Texture candidate review payload is missing")
    result_ref = artifact_reference(result)
    if domain_review.candidate_result_sha256 != result_ref.sha256:
        raise ValueError("Texture candidate review binds another result")
    if state.canonical_plan is None:  # pragma: no cover - state invariant
        raise ValueError("Texture candidate review lacks a canonical plan")
    if domain_review.plan_digest != canonical_json_digest(state.canonical_plan):
        raise ValueError("Texture candidate review binds another canonical plan")
    if not result.outputs or domain_review.output_asset != result.outputs[0]:
        raise ValueError("Texture candidate review binds another output asset")
    visual_evidence = tuple(
        artifact
        for record in result.evidence
        if record.evidence_type == "critique"
        for artifact in record.artifacts
    )
    if domain_review.visual_evidence_artifacts != visual_evidence:
        raise ValueError("Texture candidate review visual evidence is stale or partial")
    if domain_review.reference_artifacts != state.canonical_plan.reference_artifacts:
        raise ValueError("Texture candidate review reference identities drifted")
    expected_unit_ids = state.canonical_plan.target_unit_ids
    if tuple(item.unit_id for item in domain_review.unit_dispositions) != (
        expected_unit_ids
    ):
        raise ValueError(
            "Texture candidate review must disposition every canonical target in "
            "the exact merged candidate"
        )
    return EmbeddedCoordinatorReview(
        artifact_id=f"texture-candidate-review-{patch.checkpoint_revision:04d}",
        identity=state.identity,
        producer=outer_coordinator(state.identity),
        parent_artifact=result_ref,
        created_at=patch.created_at,
        execution_result=result_ref,
        semantic_decision_owner=outer_coordinator(state.identity),
        accepted_decision_digest=result.accepted_decision_digest,
        disposition=map_candidate_review_disposition(domain_review),
        outputs=result.outputs,
        findings=domain_review.findings,
    )


def candidate_review_evidence_artifact(
    patch: TextureEmbeddedDecisionPatch,
    *,
    state: TextureEmbeddedDecisionState,
    result: EmbeddedBoundedExecutionResult,
) -> EmbeddedDomainEvidence:
    """Persist the complete typed outer unit dispositions and visual binding."""

    domain_review = patch.candidate_review
    if domain_review is None:  # pragma: no cover - patch invariant
        raise ValueError("Texture candidate review payload is missing")
    result_ref = artifact_reference(result)
    if domain_review.candidate_result_sha256 != result_ref.sha256:
        raise ValueError("Texture candidate domain review binds another result")
    return EmbeddedDomainEvidence(
        artifact_id=f"texture-outer-candidate-review-{patch.checkpoint_revision:04d}",
        identity=state.identity,
        producer=_producer(
            state.identity,
            producer_id="texture-outer-review-evidence-adapter",
            role="evidence_provider",
            implementation="texture-inspection-adapter.v1",
        ),
        parent_artifact=state.identity.coordinator_plan,
        created_at=patch.created_at,
        records=(
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-outer-candidate-unit-dispositions",
                evidence_type="artifact",
                status="available",
                artifacts=domain_review.visual_evidence_artifacts,
                facts={
                    "source": state.identity.source.model_dump(mode="json"),
                    "candidate_result": result_ref.model_dump(mode="json"),
                    "candidate_output": domain_review.output_asset.model_dump(
                        mode="json"
                    ),
                    "canonical_plan_digest": domain_review.plan_digest,
                    "visual_evidence_accepted": (
                        domain_review.visual_evidence_accepted
                    ),
                    "unit_dispositions": [
                        item.model_dump(mode="json")
                        for item in domain_review.unit_dispositions
                    ],
                    "findings": list(domain_review.findings),
                    "renderer_metadata": state.inspection.renderer_metadata,
                    "tool_metadata": state.inspection.tool_metadata,
                    "reference_artifacts": [
                        item.model_dump(mode="json")
                        for item in domain_review.reference_artifacts
                    ],
                },
                summary=(
                    "Exact outer-authored per-unit review bound to fresh visual "
                    "evidence and candidate identity."
                ),
            ),
        ),
    )


def build_review_receipt(
    *,
    artifact_id: str,
    decision: EmbeddedCoordinatorDecision,
    authorization: BoundedExecutionAuthorization,
    result: EmbeddedBoundedExecutionResult,
    review: EmbeddedCoordinatorReview,
    evidence: tuple[EmbeddedDomainEvidence, ...],
    proposals: tuple[EmbeddedDomainProposal, ...] = (),
) -> EmbeddedDecisionReceipt:
    return build_decision_receipt(
        artifact_id=artifact_id,
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
    )


def build_publication_decision(
    patch: TextureEmbeddedDecisionPatch,
    *,
    store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
) -> tuple[EmbeddedCoordinatorDecision, tuple[EmbeddedDomainEvidence, ...]]:
    """Author exact outer publication semantics for the accepted candidate."""

    publication = patch.publication
    if publication is None:  # pragma: no cover - patch invariant
        raise ValueError("Texture publication decision is missing")
    required_refs = (
        state.accepted_candidate_result,
        state.accepted_candidate_domain_review,
        state.accepted_candidate_review,
        state.accepted_candidate_receipt,
        state.current_candidate_evidence,
    )
    if any(reference is None for reference in required_refs):
        raise ValueError("Texture publication requires a completed candidate review")
    result_ref = state.accepted_candidate_result
    domain_review_ref = state.accepted_candidate_domain_review
    review_ref = state.accepted_candidate_review
    receipt_ref = state.accepted_candidate_receipt
    evidence_ref = state.current_candidate_evidence
    assert result_ref is not None
    assert domain_review_ref is not None
    assert review_ref is not None
    assert receipt_ref is not None
    assert evidence_ref is not None
    result = store.load_typed(result_ref, EmbeddedBoundedExecutionResult)
    domain_review_evidence = store.load_typed(
        domain_review_ref,
        EmbeddedDomainEvidence,
    )
    review = store.load_typed(review_ref, EmbeddedCoordinatorReview)
    receipt = store.load_typed(receipt_ref, EmbeddedDecisionReceipt)
    evidence = store.load_typed(evidence_ref, EmbeddedDomainEvidence)
    if receipt.receipt_status != "completed" or review.disposition != "accept":
        raise ValueError("Texture publication requires an accepted candidate receipt")
    if publication.candidate_result_sha256 != result_ref.sha256:
        raise ValueError("Texture publication substituted another candidate result")
    if not result.outputs or publication.candidate_output != result.outputs[0]:
        raise ValueError("Texture publication substituted another candidate output")
    if publication.candidate_review_sha256 != review_ref.sha256:
        raise ValueError("Texture publication binds another candidate review")
    if state.canonical_plan is None:
        raise ValueError("Texture publication lacks the canonical plan")
    if publication.plan_digest != canonical_json_digest(state.canonical_plan):
        raise ValueError("Texture publication binds another canonical plan")
    if publication.accepted_unit_ids != state.canonical_plan.target_unit_ids:
        raise ValueError("Texture publication must cover every canonical target")
    if publication.reference_artifacts != state.canonical_plan.reference_artifacts:
        raise ValueError("Texture publication reference identities drifted")
    accepted = AcceptedSemanticDecision(
        schema_version=publication.schema_version,
        values={"publication": publication.model_dump(mode="json")},
    )
    decision = EmbeddedCoordinatorDecision(
        artifact_id=f"texture-publication-decision-{patch.checkpoint_revision:04d}",
        identity=state.identity,
        producer=outer_coordinator(state.identity),
        parent_artifact=domain_review_ref,
        created_at=patch.created_at,
        disposition="accept",
        evidence_artifacts=(evidence_ref, domain_review_ref),
        accepted_decision=accepted,
        accepted_decision_digest=accepted_semantic_decision_digest(
            state.identity,
            accepted,
        ),
        rationale=patch.rationale,
    )
    validate_coordinator_decision_dependencies(
        decision,
        evidence=(evidence, domain_review_evidence),
        proposals=(),
    )
    return decision, (evidence, domain_review_evidence)


def authorize_publication(
    decision: EmbeddedCoordinatorDecision,
    *,
    evidence: Sequence[EmbeddedDomainEvidence],
) -> BoundedExecutionAuthorization:
    """Create the stage attempt's only mutating authorization."""

    return authorize_bounded_execution(
        decision,
        expected_identity=decision.identity,
        evidence=evidence,
        proposals=(),
        existing_authorizations=(),
        executor=publication_executor(decision.identity),
        execution_effect="mutation",
    )


def publication_result_artifact(
    authorization: BoundedExecutionAuthorization,
    *,
    published_asset: ExecutionArtifactBinding,
    verification_artifacts: Sequence[ExecutionArtifactBinding],
    verification_facts: dict[str, JsonValue],
    created_at: datetime | None = None,
) -> EmbeddedBoundedExecutionResult:
    """Bind deterministic publication and its scope/closure proof."""

    return EmbeddedBoundedExecutionResult(
        artifact_id=f"texture-publication-result-{authorization.operation_id}",
        identity=authorization.identity,
        producer=authorization.executor,
        parent_artifact=artifact_reference(authorization),
        created_at=created_at or datetime.now(UTC),
        accepted_decision=authorization.accepted_decision,
        accepted_decision_digest=authorization.accepted_decision_digest,
        execution_effect="mutation",
        operation_id=authorization.operation_id,
        mutation_id=authorization.mutation_id,
        attempt=authorization.attempt,
        status="succeeded",
        mutation_state="verified",
        outputs=(published_asset,),
        evidence=(
            ProviderNeutralEvidenceRecord(
                evidence_id="texture-publication-verification",
                evidence_type="validation",
                status="available",
                artifacts=tuple(verification_artifacts),
                facts=verification_facts,
                summary=(
                    "Exact accepted candidate was published with verified target "
                    "scope, package closure, and non-target preservation."
                ),
            ),
        ),
    )


def build_publication_review(
    patch: TextureEmbeddedDecisionPatch,
    *,
    state: TextureEmbeddedDecisionState,
    result: EmbeddedBoundedExecutionResult,
) -> EmbeddedCoordinatorReview:
    """Bind the outer final disposition to exact published bytes."""

    domain_review = patch.publication_review
    if domain_review is None:  # pragma: no cover - patch invariant
        raise ValueError("Texture publication review payload is missing")
    result_ref = artifact_reference(result)
    if domain_review.publication_result_sha256 != result_ref.sha256:
        raise ValueError("Texture publication review binds another result")
    if not result.outputs or domain_review.published_asset != result.outputs[0]:
        raise ValueError("Texture publication review binds another published asset")
    if state.canonical_plan is None:
        raise ValueError("Texture publication review lacks a canonical plan")
    if domain_review.reference_artifacts != state.canonical_plan.reference_artifacts:
        raise ValueError("Texture publication review reference identities drifted")
    return EmbeddedCoordinatorReview(
        artifact_id=f"texture-publication-review-{patch.checkpoint_revision:04d}",
        identity=state.identity,
        producer=outer_coordinator(state.identity),
        parent_artifact=result_ref,
        created_at=patch.created_at,
        execution_result=result_ref,
        semantic_decision_owner=outer_coordinator(state.identity),
        accepted_decision_digest=result.accepted_decision_digest,
        disposition=domain_review.disposition,
        outputs=result.outputs,
        findings=domain_review.findings,
    )


def _validate_persisted_receipt_chain(
    store: EmbeddedDecisionArtifactStore,
    *,
    decision_ref: ContractArtifactReference,
    authorization_ref: ContractArtifactReference,
    result_ref: ContractArtifactReference,
    review_ref: ContractArtifactReference,
    receipt_ref: ContractArtifactReference,
) -> tuple[
    EmbeddedCoordinatorDecision,
    BoundedExecutionAuthorization,
    EmbeddedBoundedExecutionResult,
    EmbeddedCoordinatorReview,
    EmbeddedDecisionReceipt,
]:
    decision = store.load_typed(decision_ref, EmbeddedCoordinatorDecision)
    authorization = store.load_typed(
        authorization_ref,
        BoundedExecutionAuthorization,
    )
    result = store.load_typed(result_ref, EmbeddedBoundedExecutionResult)
    review = store.load_typed(review_ref, EmbeddedCoordinatorReview)
    receipt = store.load_typed(receipt_ref, EmbeddedDecisionReceipt)
    evidence = tuple(
        store.load_typed(reference, EmbeddedDomainEvidence)
        for reference in decision.evidence_artifacts
    )
    proposals = tuple(
        store.load_typed(reference, EmbeddedDomainProposal)
        for reference in decision.proposal_artifacts
    )
    validate_decision_receipt(
        receipt,
        decision=decision,
        evidence=evidence,
        proposals=proposals,
        authorization=authorization,
        result=result,
        review=review,
        prior_lineage=PersistedExecutionLineage(),
    )
    return decision, authorization, result, review, receipt


def validate_rejected_texture_candidate_chain(
    store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
) -> EmbeddedDecisionReceipt:
    """Replay the exact terminal outer rejection for a Texture candidate."""

    required = {
        "current_decision": state.current_decision,
        "current_authorization": state.current_authorization,
        "current_result": state.current_result,
        "current_candidate_review": state.current_candidate_review,
        "current_candidate_receipt": state.current_candidate_receipt,
    }
    missing = [name for name, reference in required.items() if reference is None]
    if missing:
        raise ValueError(
            "rejected Texture candidate chain is incomplete: " + ", ".join(missing)
        )
    decision_ref = state.current_decision
    authorization_ref = state.current_authorization
    result_ref = state.current_result
    review_ref = state.current_candidate_review
    receipt_ref = state.current_candidate_receipt
    assert decision_ref is not None
    assert authorization_ref is not None
    assert result_ref is not None
    assert review_ref is not None
    assert receipt_ref is not None
    if any(
        reference is not None
        for reference in (
            state.accepted_candidate_result,
            state.accepted_candidate_domain_review,
            state.accepted_candidate_review,
            state.accepted_candidate_receipt,
            state.publication_decision,
            state.publication_authorization,
            state.publication_result,
            state.publication_review,
            state.completed_receipt,
        )
    ):
        raise ValueError(
            "rejected Texture candidate cannot retain acceptance or publication state"
        )
    (
        _decision,
        _authorization,
        _result,
        review,
        receipt,
    ) = _validate_persisted_receipt_chain(
        store,
        decision_ref=decision_ref,
        authorization_ref=authorization_ref,
        result_ref=result_ref,
        review_ref=review_ref,
        receipt_ref=receipt_ref,
    )
    if review.disposition != "reject" or receipt.receipt_status != "rejected":
        raise ValueError(
            "terminal Texture rejection must bind a rejected outer review receipt"
        )
    return receipt


def _validate_candidate_evidence_target_coverage(
    *,
    candidate_unit_ids: Sequence[JsonValue],
    raw_findings: Sequence[JsonValue],
    target_unit_ids: tuple[str, ...],
) -> None:
    if any(not isinstance(item, str) for item in candidate_unit_ids):
        raise ValueError("Texture candidate unit IDs must be strings")
    finding_unit_ids: list[str] = []
    for raw_finding in raw_findings:
        if not isinstance(raw_finding, Mapping):
            raise ValueError("Texture candidate critique finding is invalid")
        finding_unit_id = raw_finding.get("unit_id")
        if not isinstance(finding_unit_id, str):
            raise ValueError("Texture candidate critique finding lacks a unit ID")
        finding_unit_ids.append(finding_unit_id)
    if (
        tuple(candidate_unit_ids) != target_unit_ids
        or tuple(finding_unit_ids) != target_unit_ids
    ):
        raise ValueError(
            "Texture candidate evidence does not cover every canonical target"
        )


def validate_completed_texture_decision_chain(
    store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
    proposal_plan: TexturePlanDocument,
) -> EmbeddedDecisionReceipt:
    """Replay the complete accepted candidate and publication chains."""

    required = {
        "current_decision": state.current_decision,
        "current_authorization": state.current_authorization,
        "current_result": state.current_result,
        "current_candidate_evidence": state.current_candidate_evidence,
        "current_candidate_domain_review": state.current_candidate_domain_review,
        "current_candidate_review": state.current_candidate_review,
        "current_candidate_receipt": state.current_candidate_receipt,
        "accepted_candidate_result": state.accepted_candidate_result,
        "accepted_candidate_domain_review": (state.accepted_candidate_domain_review),
        "accepted_candidate_review": state.accepted_candidate_review,
        "accepted_candidate_receipt": state.accepted_candidate_receipt,
        "publication_decision": state.publication_decision,
        "publication_authorization": state.publication_authorization,
        "publication_result": state.publication_result,
        "publication_review": state.publication_review,
        "completed_receipt": state.completed_receipt,
    }
    missing = [name for name, reference in required.items() if reference is None]
    if state.canonical_plan is None or missing:
        raise ValueError(
            "completed Texture state lacks its canonical plan or decision chain: "
            + ", ".join(missing or ("canonical_plan",))
        )
    validate_texture_initial_decision_inputs(store, state, proposal_plan)
    validate_outer_texture_plan(
        state.canonical_plan,
        state=state,
        proposal=proposal_plan,
    )
    refs = {name: reference for name, reference in required.items() if reference}
    if (
        refs["accepted_candidate_result"] != refs["current_result"]
        or refs["accepted_candidate_domain_review"]
        != refs["current_candidate_domain_review"]
        or refs["accepted_candidate_review"] != refs["current_candidate_review"]
        or refs["accepted_candidate_receipt"] != refs["current_candidate_receipt"]
    ):
        raise ValueError(
            "completed Texture state does not retain the exact accepted candidate"
        )
    inspection = store.load_typed(
        state.inspection_evidence,
        EmbeddedDomainEvidence,
    )
    proposal = store.load_typed(state.plan_proposal, EmbeddedDomainProposal)
    if (
        inspection.identity != state.identity
        or proposal.identity != state.identity
        or state.canonical_plan.source != state.identity.source
        or proposal.evidence_artifacts != (state.inspection_evidence,)
    ):
        raise ValueError("Texture plan inputs do not bind the persisted run identity")

    (
        candidate_decision,
        _candidate_authorization,
        candidate_result,
        candidate_review,
        candidate_receipt,
    ) = _validate_persisted_receipt_chain(
        store,
        decision_ref=refs["current_decision"],
        authorization_ref=refs["current_authorization"],
        result_ref=refs["accepted_candidate_result"],
        review_ref=refs["accepted_candidate_review"],
        receipt_ref=refs["accepted_candidate_receipt"],
    )
    if candidate_decision.accepted_decision is None:
        raise ValueError("Texture candidate decision lacks accepted semantics")
    accepted_plan = TextureCanonicalPlan.model_validate(
        candidate_decision.accepted_decision.values.get("canonical_plan")
    )
    if (
        candidate_decision.proposal_artifacts != (state.plan_proposal,)
        or not candidate_decision.evidence_artifacts
        or candidate_decision.evidence_artifacts[0] != state.inspection_evidence
        or accepted_plan != state.canonical_plan
        or candidate_receipt.receipt_status != "completed"
        or candidate_receipt.execution_effect != "non_mutating"
        or candidate_receipt.execution_status != "succeeded"
        or candidate_review.disposition != "accept"
        or not candidate_result.outputs
    ):
        raise ValueError("Texture candidate receipt does not bind the canonical plan")
    candidate_evidence = store.load_typed(
        refs["current_candidate_evidence"],
        EmbeddedDomainEvidence,
    )
    if (
        candidate_evidence.identity != state.identity
        or candidate_evidence.records != candidate_result.evidence
    ):
        raise ValueError("Texture candidate evidence differs from the reviewed result")
    artifact_records = tuple(
        record
        for record in candidate_result.evidence
        if record.evidence_id == "texture-candidate-artifacts"
    )
    critique_records = tuple(
        record
        for record in candidate_result.evidence
        if record.evidence_type == "critique"
    )
    closure_records = tuple(
        record
        for record in candidate_result.evidence
        if record.evidence_id == "texture-candidate-dependency-closure"
    )
    if (
        len(artifact_records) != 1
        or len(critique_records) != 1
        or len(closure_records) != 1
        or len(closure_records[0].artifacts) != 1
    ):
        raise ValueError(
            "Texture candidate lacks exact artifact, critique, or closure evidence"
        )
    candidate_unit_ids = artifact_records[0].facts.get("candidate_unit_ids")
    raw_findings = critique_records[0].facts.get("findings")
    if not isinstance(candidate_unit_ids, tuple | list) or not isinstance(
        raw_findings, tuple | list
    ):
        raise ValueError("Texture candidate evidence lacks typed target coverage")
    _validate_candidate_evidence_target_coverage(
        candidate_unit_ids=candidate_unit_ids,
        raw_findings=raw_findings,
        target_unit_ids=state.canonical_plan.target_unit_ids,
    )
    if not critique_records[0].artifacts:
        raise ValueError(
            "Texture candidate evidence does not cover every canonical target"
        )
    closure_binding = closure_records[0].artifacts[0]
    if bind_file(closure_binding.path) != closure_binding:
        raise ValueError("Texture candidate dependency-closure manifest changed")
    try:
        closure_payload = json.loads(Path(closure_binding.path).read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Texture candidate dependency-closure manifest is invalid: {exc}"
        ) from exc
    closure_facts = closure_records[0].facts
    closure_facts_payload = dict(closure_facts)
    recomputed_closure = texture_candidate_dependency_closure_facts(
        candidate_result.outputs[0]
    )
    if (
        canonical_json_digest(closure_payload)
        != canonical_json_digest(closure_facts_payload)
        or canonical_json_digest(recomputed_closure)
        != canonical_json_digest(closure_facts_payload)
        or closure_facts.get("candidate_sha256") != candidate_result.outputs[0].sha256
        or closure_facts.get("candidate_size_bytes")
        != candidate_result.outputs[0].size_bytes
        or closure_facts.get("closure_sealed_by_candidate_bytes") is not True
        or closure_facts.get("external_dependency_count") != 0
        or closure_facts.get("unresolved_dependency_count") != 0
    ):
        raise ValueError(
            "Texture candidate dependency closure is stale or not byte-sealed"
        )
    domain_review = store.load_typed(
        refs["accepted_candidate_domain_review"],
        EmbeddedDomainEvidence,
    )
    if len(domain_review.records) != 1:
        raise ValueError("Texture candidate domain review is missing")
    review_record = domain_review.records[0]
    review_facts = review_record.facts
    reviewed_result_ref = ContractArtifactReference.model_validate(
        review_facts.get("candidate_result")
    )
    reviewed_output = ExecutionArtifactBinding.model_validate(
        review_facts.get("candidate_output")
    )
    dispositions = review_facts.get("unit_dispositions")
    if not isinstance(dispositions, tuple | list):
        raise ValueError("Texture candidate domain review lacks unit dispositions")
    disposition_by_unit: dict[str, str] = {}
    for raw_disposition in dispositions:
        if not isinstance(raw_disposition, Mapping):
            raise ValueError("Texture candidate domain review disposition is invalid")
        unit_id = raw_disposition.get("unit_id")
        disposition = raw_disposition.get("disposition")
        if isinstance(unit_id, str) and isinstance(disposition, str):
            disposition_by_unit[unit_id] = disposition
    visual_bindings = tuple(
        artifact
        for record in candidate_result.evidence
        if record.evidence_type == "critique"
        for artifact in record.artifacts
    )
    if (
        domain_review.identity != state.identity
        or reviewed_result_ref != refs["accepted_candidate_result"]
        or reviewed_output != candidate_result.outputs[0]
        or review_facts.get("canonical_plan_digest")
        != canonical_json_digest(state.canonical_plan)
        or review_facts.get("visual_evidence_accepted") is not True
        or review_facts.get("reference_artifacts")
        != tuple(
            item.model_dump(mode="json")
            for item in state.canonical_plan.reference_artifacts
        )
        or tuple(disposition_by_unit) != state.canonical_plan.target_unit_ids
        or any(value != "accept" for value in disposition_by_unit.values())
        or review_record.artifacts != visual_bindings
    ):
        raise ValueError(
            "Texture outer candidate review is partial, stale, or not accepted"
        )

    (
        publication_decision,
        _publication_authorization,
        publication_result,
        publication_review,
        completed_receipt,
    ) = _validate_persisted_receipt_chain(
        store,
        decision_ref=refs["publication_decision"],
        authorization_ref=refs["publication_authorization"],
        result_ref=refs["publication_result"],
        review_ref=refs["publication_review"],
        receipt_ref=refs["completed_receipt"],
    )
    if publication_decision.accepted_decision is None:
        raise ValueError("Texture publication decision lacks accepted semantics")
    publication_payload = publication_decision.accepted_decision.values.get(
        "publication"
    )
    publication = TexturePublicationDecision.model_validate(publication_payload)
    publication_verification = tuple(
        record
        for record in publication_result.evidence
        if record.evidence_id == "texture-publication-verification"
    )
    required_publication_facts = (
        "scope_invariants_passed",
        "geometry_unchanged",
        "non_target_materials_unchanged",
        "bindings_unchanged",
        "structure_unchanged_outside_target",
        "dependency_closure_complete",
        "exact_candidate_identity_preserved",
    )
    if (
        publication_decision.evidence_artifacts
        != (
            refs["current_candidate_evidence"],
            refs["accepted_candidate_domain_review"],
        )
        or publication.candidate_result_sha256
        != refs["accepted_candidate_result"].sha256
        or publication.candidate_output != candidate_result.outputs[0]
        or publication.plan_digest != canonical_json_digest(state.canonical_plan)
        or publication.accepted_unit_ids != state.canonical_plan.target_unit_ids
        or publication.candidate_review_sha256
        != refs["accepted_candidate_review"].sha256
        or completed_receipt.receipt_status != "completed"
        or completed_receipt.execution_effect != "mutation"
        or completed_receipt.execution_status != "succeeded"
        or publication_review.disposition != "accept"
        or publication_result.outputs != completed_receipt.outputs
        or len(publication_result.outputs) != 1
        or len(publication_verification) != 1
        or publication_verification[0].facts.get("published_asset_sha256")
        != publication_result.outputs[0].sha256
        or any(
            publication_verification[0].facts.get(name) is not True
            for name in required_publication_facts
        )
    ):
        raise ValueError(
            "Texture publication receipt does not consume the exact accepted candidate"
        )
    return completed_receipt


def load_generation_chain(
    store: EmbeddedDecisionArtifactStore,
    state: TextureEmbeddedDecisionState,
) -> tuple[
    EmbeddedCoordinatorDecision,
    BoundedExecutionAuthorization,
    EmbeddedBoundedExecutionResult,
]:
    decision_ref = state.current_decision
    authorization_ref = state.current_authorization
    result_ref = state.current_result
    if decision_ref is None or authorization_ref is None or result_ref is None:
        raise ValueError("embedded Texture candidate chain is incomplete")
    return (
        store.load_typed(decision_ref, EmbeddedCoordinatorDecision),
        store.load_typed(authorization_ref, BoundedExecutionAuthorization),
        store.load_typed(result_ref, EmbeddedBoundedExecutionResult),
    )


def ensure_embedded_patch_identity(
    patch: TextureEmbeddedDecisionPatch,
    *,
    request_digest: str,
    source_identity_digest: str,
    proposal_plan_digest: str,
    checkpoint_decision_digest: str,
    checkpoint_revision: int,
    action: str,
    iteration: int,
) -> None:
    expected = (
        request_digest,
        source_identity_digest,
        proposal_plan_digest,
        checkpoint_decision_digest,
        checkpoint_revision,
        action,
        iteration,
    )
    actual = (
        patch.request_digest,
        patch.source_identity_digest,
        patch.proposal_plan_digest,
        patch.checkpoint_decision_digest,
        patch.checkpoint_revision,
        patch.action,
        patch.iteration,
    )
    if actual != expected:
        raise ValueError("embedded Texture patch is stale or targets another action")


__all__ = [
    "TEXTURE_EMBEDDED_CRITICAL_IMPLEMENTATION_FILES",
    "TEXTURE_EMBEDDED_PATCH_SCHEMA_VERSION",
    "TextureEmbeddedDecisionPatch",
    "TextureEmbeddedDecisionState",
    "authorize_candidate_generation",
    "authorize_publication",
    "bind_file",
    "build_candidate_review",
    "build_embedded_texture_identity",
    "build_plan_decision",
    "build_publication_decision",
    "build_publication_review",
    "build_review_receipt",
    "candidate_evidence_artifact",
    "candidate_review_evidence_artifact",
    "candidate_executor",
    "candidate_result_artifact",
    "ensure_embedded_patch_identity",
    "initialize_embedded_texture_decisions",
    "load_generation_chain",
    "map_candidate_review_disposition",
    "outer_coordinator",
    "publication_executor",
    "publication_result_artifact",
    "texture_candidate_dependency_closure_facts",
    "validate_embedded_texture_identity_manifests",
    "validate_completed_texture_decision_chain",
    "validate_rejected_texture_candidate_chain",
    "validate_outer_texture_plan",
    "validate_texture_initial_decision_inputs",
]
