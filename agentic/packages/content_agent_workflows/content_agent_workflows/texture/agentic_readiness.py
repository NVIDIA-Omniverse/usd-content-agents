# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation-bound Texture planning and selected-only execution custody.

The focused Texture capabilities remain the operation leaves.  This module owns
only the one-attempt agentic boundary around them: a reasoning child proposes a
complete plan after deterministic preparation, the outer wrapper freezes that
exact proposal, and lazy adapters execute only the selected mutating actions.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from texture_agent.functions.detail_policy import apply_detail_policy_to_prompt

from content_agent_workflows.asset_composition import (
    bind_usd_dependency_closure,
    verify_usd_dependency_closure,
)
from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.common.usd_package_localizer import (
    create_localized_usdz_package,
)

from .capabilities import (
    TextureGeneratorLeaf,
    TextureGeneratorLeafRequest,
    TexturePreparationPacket,
    TextureUnitRenderEvidence,
)
from .models import (
    TextureAcceptanceCriteria,
    TextureExecutionResult,
    TextureGeneratorInputs,
    TextureInspectionUnit,
    TexturePlanDocument,
    TexturePreservationConstraints,
    TextureProvidedImageArtifact,
    TextureProvidedImageProducer,
    TextureReferenceArtifact,
)

TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-plan.v2"
] = "content-agent-workflows.texture-plan.v2"
TEXTURE_ACCEPTED_PLAN_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-accepted-plan.v1"
] = "content-agent-workflows.texture-accepted-plan.v1"
TEXTURE_ADAPTER_LEDGER_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-adapter-ledger.v1"
] = "content-agent-workflows.texture-adapter-ledger.v1"
TEXTURE_AGENTIC_EVIDENCE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-evidence.v1"
] = "content-agent-workflows.texture-agentic-evidence.v1"
TEXTURE_AGENTIC_REVIEW_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-review.v1"
] = "content-agent-workflows.texture-agentic-review.v1"
TEXTURE_AGENTIC_TERMINAL_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-terminal.v1"
] = "content-agent-workflows.texture-agentic-terminal.v1"
TEXTURE_AGENTIC_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-readback.v1"
] = "content-agent-workflows.texture-agentic-readback.v1"
TEXTURE_AGENTIC_PUBLICATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-publication.v1"
] = "content-agent-workflows.texture-agentic-publication.v1"
TEXTURE_AGENTIC_CLEANUP_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-cleanup.v1"
] = "content-agent-workflows.texture-agentic-cleanup.v1"
TEXTURE_AGENTIC_SOURCE_PREPARATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-agentic-source-preparation.v1"
] = "content-agent-workflows.texture-agentic-source-preparation.v1"

TextureAgenticAction = Literal[
    "preserve",
    "generate",
    "apply_provided",
    "defer",
    "reject",
]
TextureMutatingAction = Literal["generate", "apply_provided"]
_PROVIDED_CANDIDATES_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_provided_candidates"
)
_ACTION_POLICY_METADATA_KEY = "content_workflow_cli_texture_agentic_action_policy"
_APPEARANCE_POLICY_METADATA_KEY = (
    "content_workflow_cli_texture_agentic_appearance_policy"
)
_EVIDENCE_VIEWS_METADATA_KEY = "content_workflow_cli_texture_agentic_evidence_views"


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TextureAgenticSourcePreparation(_StrictFrozenModel):
    """Provider-free UV preparation performed before child reasoning."""

    schema_version: Literal[
        "content-agent-workflows.texture-agentic-source-preparation.v1"
    ] = TEXTURE_AGENTIC_SOURCE_PREPARATION_SCHEMA_VERSION
    original_source: ExecutionArtifactBinding
    uv_prepared_source: ExecutionArtifactBinding
    effective_source: ExecutionArtifactBinding
    target_prim_paths: tuple[str, ...] = Field(min_length=1)
    policy: Literal["generate_missing"] = "generate_missing"
    invocation: ExecutionArtifactBinding
    result: ExecutionArtifactBinding
    evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readbacks: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    provider_invoked: Literal[False] = False
    texture_service_constructed: Literal[False] = False
    vlm_assessor_constructed: Literal[False] = False
    image_generator_constructed: Literal[False] = False

    @model_validator(mode="after")
    def _require_unique_scope_and_artifacts(self) -> Self:
        if len(self.target_prim_paths) != len(set(self.target_prim_paths)):
            raise ValueError("Texture source-preparation targets must be unique")
        artifacts = (*self.evidence, *self.saved_stage_readbacks)
        paths = tuple(item.path for item in artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("Texture source-preparation evidence paths must be unique")
        return self


class TextureAgenticEvidenceRequirements(_StrictFrozenModel):
    """Evidence the outer wrapper must collect before publication."""

    required_views: tuple[str, ...] = Field(min_length=1)
    require_current_run_ovrtx: bool = True
    require_saved_stage_readback: bool = True
    require_dependency_closure: bool = True
    require_non_target_preservation: bool = True

    @model_validator(mode="after")
    def _require_fail_closed_evidence(self) -> Self:
        if len(self.required_views) != len(set(self.required_views)):
            raise ValueError("Texture required evidence views must be unique")
        unsupported = set(self.required_views) - {"+x-y+z", "+z", "-z"}
        if unsupported:
            raise ValueError(
                "Texture required evidence views must use supported OVRTX "
                f"directions: {sorted(unsupported)}"
            )
        if not all(
            (
                self.require_current_run_ovrtx,
                self.require_saved_stage_readback,
                self.require_dependency_closure,
                self.require_non_target_preservation,
            )
        ):
            raise ValueError("agentic Texture evidence requirements may not be relaxed")
        return self


class TextureAgenticStopPolicy(_StrictFrozenModel):
    """The v0.6 readiness slice permits exactly one immutable attempt."""

    max_attempts_per_unit: Literal[1] = 1
    plan_revision_count: Literal[0] = 0
    retry_unavailable_provider: Literal[False] = False
    fallback_provider: Literal[False] = False


class TexturePlanUnitDisposition(_StrictFrozenModel):
    """One child-authored semantic disposition for one prepared Texture unit."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    material_prim_paths: tuple[str, ...] = Field(min_length=1)
    member_prim_paths: tuple[str, ...] = ()
    member_subset_paths: tuple[str, ...] = ()
    action: TextureAgenticAction
    rationale: str = Field(min_length=1, max_length=4000)
    requested_appearance: str | None = Field(default=None, min_length=1)
    generator_inputs: TextureGeneratorInputs | None = None

    @model_validator(mode="after")
    def _validate_action_parameters(self) -> Self:
        if not self.member_prim_paths and not self.member_subset_paths:
            raise ValueError(
                "Texture plan disposition requires a prim or subset target"
            )
        if self.action == "generate":
            if self.generator_inputs is None:
                raise ValueError("generate disposition requires generator_inputs")
            if self.generator_inputs.execution_mode != "provider_generate":
                raise ValueError(
                    "generate disposition requires provider_generate inputs"
                )
            if not self.requested_appearance:
                raise ValueError("generate disposition requires requested_appearance")
        elif self.action == "apply_provided":
            if self.generator_inputs is None:
                raise ValueError("apply_provided disposition requires generator_inputs")
            if self.generator_inputs.execution_mode != "apply_provided":
                raise ValueError(
                    "apply_provided disposition requires apply_provided inputs"
                )
            provided = self.generator_inputs.provided_images
            if len(provided) != 1 or provided[0].unit_id != self.unit_id:
                raise ValueError(
                    "apply_provided requires one exact image for its disposition unit"
                )
            if not self.requested_appearance:
                raise ValueError("apply_provided requires requested_appearance")
        elif self.generator_inputs is not None or self.requested_appearance is not None:
            raise ValueError(
                f"{self.action} disposition cannot carry mutation parameters"
            )
        return self


class TextureAgenticPlan(_StrictFrozenModel):
    """Complete preparation-bound proposal authored by the reasoning child."""

    schema_version: Literal["content-agent-workflows.texture-plan.v2"] = (
        TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION
    )
    preparation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    scope_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dispositions: tuple[TexturePlanUnitDisposition, ...] = Field(min_length=1)
    reference_artifacts: tuple[TextureReferenceArtifact, ...] = ()
    preservation: TexturePreservationConstraints
    acceptance: TextureAcceptanceCriteria
    evidence: TextureAgenticEvidenceRequirements
    capability_constraints: tuple[str, ...] = Field(min_length=1)
    stop_policy: TextureAgenticStopPolicy = Field(
        default_factory=TextureAgenticStopPolicy
    )

    @model_validator(mode="after")
    def _validate_disposition_identity(self) -> Self:
        unit_ids = self.unit_ids
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Texture plan dispositions must contain unique unit IDs")
        reference_paths = tuple(item.artifact.path for item in self.reference_artifacts)
        if len(reference_paths) != len(set(reference_paths)):
            raise ValueError("Texture plan references must be unique")
        return self

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(item.unit_id for item in self.dispositions)

    def units_for(self, action: TextureAgenticAction) -> tuple[str, ...]:
        return tuple(
            item.unit_id for item in self.dispositions if item.action == action
        )


class TextureAcceptedPlan(_StrictFrozenModel):
    """Outer freeze of exact child-authored plan bytes."""

    schema_version: Literal["content-agent-workflows.texture-accepted-plan.v1"] = (
        TEXTURE_ACCEPTED_PLAN_SCHEMA_VERSION
    )
    proposal: ExecutionArtifactBinding
    proposal_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    accepted_by: Literal["texture-domain-wrapper"] = "texture-domain-wrapper"
    plan: TextureAgenticPlan

    @model_validator(mode="after")
    def _validate_frozen_plan(self) -> Self:
        if self.proposal_digest != canonical_json_digest(self.plan):
            raise ValueError("accepted Texture plan digest differs from proposal")
        if self.plan.preparation != self.preparation or self.plan.source != self.source:
            raise ValueError("accepted Texture plan identities differ from proposal")
        return self


class TextureAgenticUnitArtifacts(_StrictFrozenModel):
    """Digest-bound artifacts returned for one exactly selected unit."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_unique_artifacts(self) -> Self:
        paths = tuple(item.path for item in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("Texture unit artifact bindings must be unique")
        return self


class TextureAgenticAdapterResult(_StrictFrozenModel):
    """One selected adapter result over an exact ordered unit subset."""

    action: TextureMutatingAction
    unit_ids: tuple[str, ...] = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    input_asset: ExecutionArtifactBinding
    output_asset: ExecutionArtifactBinding
    unit_artifacts: tuple[TextureAgenticUnitArtifacts, ...] = Field(min_length=1)
    evidence_artifacts: tuple[ExecutionArtifactBinding, ...] = ()

    @model_validator(mode="after")
    def _validate_exact_result_scope(self) -> Self:
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise ValueError("Texture adapter result unit IDs must be unique")
        if tuple(item.unit_id for item in self.unit_artifacts) != self.unit_ids:
            raise ValueError("Texture adapter result artifacts differ from unit order")
        return self


class TextureAdapterCallRecord(_StrictFrozenModel):
    """Ledger entry for one actually invoked selected adapter."""

    sequence: int = Field(ge=1)
    action: TextureMutatingAction
    unit_ids: tuple[str, ...] = Field(min_length=1)
    adapter_id: str = Field(min_length=1)
    input_asset: ExecutionArtifactBinding
    output_asset: ExecutionArtifactBinding
    unit_artifacts: tuple[TextureAgenticUnitArtifacts, ...] = Field(min_length=1)
    evidence_artifacts: tuple[ExecutionArtifactBinding, ...] = ()


class TextureAdapterCallLedger(_StrictFrozenModel):
    """Exact selected-only invocation record for one accepted plan."""

    schema_version: Literal["content-agent-workflows.texture-adapter-ledger.v1"] = (
        TEXTURE_ADAPTER_LEDGER_SCHEMA_VERSION
    )
    accepted_plan: ExecutionArtifactBinding
    preparation: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    prepared_candidate: ExecutionArtifactBinding | None = None
    records: tuple[TextureAdapterCallRecord, ...] = ()
    final_candidate: ExecutionArtifactBinding
    unresolved_unit_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_chain(self) -> Self:
        if tuple(item.sequence for item in self.records) != tuple(
            range(1, len(self.records) + 1)
        ):
            raise ValueError("Texture adapter-call sequence must be contiguous")
        if self.prepared_candidate is not None:
            if self.prepared_candidate == self.source:
                raise ValueError(
                    "Texture prepared candidate must differ from its source"
                )
            if self.records:
                raise ValueError(
                    "Texture prepared candidate is only valid without mutating calls"
                )
        current = self.prepared_candidate or self.source
        observed_units: list[str] = []
        for record in self.records:
            if record.input_asset != current:
                raise ValueError("Texture adapter-call asset chain is discontinuous")
            if tuple(item.unit_id for item in record.unit_artifacts) != record.unit_ids:
                raise ValueError(
                    "Texture adapter-call artifacts differ from unit order"
                )
            observed_units.extend(record.unit_ids)
            current = record.output_asset
        if current != self.final_candidate:
            raise ValueError("Texture adapter ledger final candidate is stale")
        if len(observed_units) != len(set(observed_units)):
            raise ValueError("Texture adapter ledger executed a unit more than once")
        if len(self.unresolved_unit_ids) != len(set(self.unresolved_unit_ids)):
            raise ValueError("Texture unresolved unit IDs must be unique")
        return self


class _TexturePreserveCandidateReceipt(_StrictFrozenModel):
    """Crash-recovery identity for one completed preserve candidate."""

    schema_version: Literal["content-agent-workflows.texture-preserve-candidate.v1"] = (
        "content-agent-workflows.texture-preserve-candidate.v1"
    )
    source: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding


class TextureAgenticSavedStageReadback(_StrictFrozenModel):
    """Typed saved-stage verification over the exact executed candidate."""

    schema_version: Literal["content-agent-workflows.texture-agentic-readback.v1"] = (
        TEXTURE_AGENTIC_READBACK_SCHEMA_VERSION
    )
    accepted_plan: ExecutionArtifactBinding
    adapter_ledger: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding
    saved_stage: ExecutionArtifactBinding
    scope_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    uv_scope_matches: Literal[True] = True
    material_scope_matches: Literal[True] = True
    dependency_closure_complete: Literal[True] = True
    non_target_content_preserved: Literal[True] = True

    @model_validator(mode="after")
    def _require_exact_saved_candidate(self) -> Self:
        if self.saved_stage.sha256 != self.candidate.sha256:
            raise ValueError("saved Texture stage differs from executed candidate")
        return self


class TextureAgenticEvidenceReceipt(_StrictFrozenModel):
    """Current-run evidence and saved-stage readback over the exact candidate."""

    schema_version: Literal["content-agent-workflows.texture-agentic-evidence.v1"] = (
        TEXTURE_AGENTIC_EVIDENCE_SCHEMA_VERSION
    )
    accepted_plan: ExecutionArtifactBinding
    adapter_ledger: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding
    view_names: tuple[str, ...] = Field(min_length=1)
    unit_evidence: tuple[TextureUnitRenderEvidence, ...] = Field(min_length=1)
    static_evidence: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    saved_stage_readback: ExecutionArtifactBinding
    renderer_metadata: dict[str, object] = Field(min_length=1)
    qualifying_current_run_ovrtx: Literal[True] = True

    @model_validator(mode="after")
    def _require_unique_unit_evidence(self) -> Self:
        unit_ids = tuple(item.unit_id for item in self.unit_evidence)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Texture agentic evidence unit IDs must be unique")
        if len(self.view_names) != len(set(self.view_names)):
            raise ValueError("Texture agentic evidence view names must be unique")
        return self


class TextureAgenticUnitReview(_StrictFrozenModel):
    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    disposition: Literal["accept", "reject", "unresolved"]
    rationale: str = Field(min_length=1)


class TextureAgenticReviewReceipt(_StrictFrozenModel):
    """Separate outer review that cannot replace child-authored plan facts."""

    schema_version: Literal["content-agent-workflows.texture-agentic-review.v1"] = (
        TEXTURE_AGENTIC_REVIEW_SCHEMA_VERSION
    )
    accepted_plan: ExecutionArtifactBinding
    adapter_ledger: ExecutionArtifactBinding
    evidence: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    unit_reviews: tuple[TextureAgenticUnitReview, ...] = Field(min_length=1)
    inspected_visual_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(
        min_length=1
    )
    findings: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_review_scope(self) -> Self:
        unit_ids = tuple(item.unit_id for item in self.unit_reviews)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Texture agentic review unit IDs must be unique")
        return self

    @property
    def accepted(self) -> bool:
        return all(item.disposition == "accept" for item in self.unit_reviews)


class TextureAgenticPublicationReceipt(_StrictFrozenModel):
    """Guarded publication of the exact separately reviewed candidate."""

    schema_version: Literal[
        "content-agent-workflows.texture-agentic-publication.v1"
    ] = TEXTURE_AGENTIC_PUBLICATION_SCHEMA_VERSION
    accepted_plan: ExecutionArtifactBinding
    adapter_ledger: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding
    evidence: ExecutionArtifactBinding
    saved_stage_readback: ExecutionArtifactBinding
    review: ExecutionArtifactBinding
    published_asset: ExecutionArtifactBinding
    verification_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_exact_published_candidate(self) -> Self:
        if self.published_asset.sha256 != self.candidate.sha256:
            raise ValueError("published Texture asset differs from accepted candidate")
        return self


class TextureAgenticCleanupReceipt(_StrictFrozenModel):
    """Bound cleanup result emitted on every terminal path."""

    schema_version: Literal["content-agent-workflows.texture-agentic-cleanup.v1"] = (
        TEXTURE_AGENTIC_CLEANUP_SCHEMA_VERSION
    )
    accepted_plan: ExecutionArtifactBinding
    adapter_ledger: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding
    status: Literal["completed", "failed"]
    retained_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    failures: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_truthful_cleanup(self) -> Self:
        if self.status == "completed" and self.failures:
            raise ValueError("completed Texture cleanup cannot report failures")
        if self.status == "failed" and not self.failures:
            raise ValueError("failed Texture cleanup requires failure details")
        return self


class TextureAgenticTerminalReceipt(_StrictFrozenModel):
    """Truthful terminal binding for success and fail-closed dispositions."""

    schema_version: Literal["content-agent-workflows.texture-agentic-terminal.v1"] = (
        TEXTURE_AGENTIC_TERMINAL_SCHEMA_VERSION
    )
    request: ExecutionArtifactBinding
    preparation: ExecutionArtifactBinding
    proposed_plan: ExecutionArtifactBinding
    accepted_plan: ExecutionArtifactBinding
    adapter_ledger: ExecutionArtifactBinding
    candidate: ExecutionArtifactBinding
    evidence: ExecutionArtifactBinding | None = None
    saved_stage_readback: ExecutionArtifactBinding | None = None
    review: ExecutionArtifactBinding | None = None
    publication: ExecutionArtifactBinding | None = None
    cleanup: ExecutionArtifactBinding
    cleanup_status: Literal["completed", "failed"]
    disposition: Literal["published", "rejected", "blocked", "failed"]
    issue_codes: tuple[str, ...] = ()
    failure: str | None = None

    @model_validator(mode="after")
    def _validate_terminal_truth(self) -> Self:
        if len(self.issue_codes) != len(set(self.issue_codes)):
            raise ValueError("Texture terminal issue codes must be unique")
        if self.disposition == "published":
            if (
                self.evidence is None
                or self.saved_stage_readback is None
                or self.review is None
                or self.publication is None
                or self.cleanup_status != "completed"
                or self.failure is not None
            ):
                raise ValueError(
                    "published Texture terminal receipt requires complete accepted custody"
                )
        elif self.publication is not None:
            raise ValueError("non-published Texture receipt cannot bind publication")
        if self.disposition == "failed" and not self.failure:
            raise ValueError("failed Texture terminal receipt requires failure detail")
        return self


class TextureAgenticExecutionAdapter(Protocol):
    """Lazy outer-owned adapter for one selected mutating action."""

    adapter_id: str

    def execute(
        self,
        *,
        accepted_plan: TextureAcceptedPlan,
        preparation: TexturePreparationPacket,
        units: tuple[TexturePlanUnitDisposition, ...],
        input_asset: ExecutionArtifactBinding,
        output_dir: Path,
    ) -> TextureAgenticAdapterResult: ...


TextureAdapterFactory = Callable[[], TextureAgenticExecutionAdapter]


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) for item in value
    ):
        return ()
    return tuple(value)


def _narrow_texture_scope_plan(
    plan: TexturePlanDocument,
    units: tuple[TexturePlanUnitDisposition, ...],
) -> TexturePlanDocument:
    """Return the exact deterministic scope subset authorized for one adapter."""

    selected_by_id = {item.unit_id: item for item in plan.selected_units}
    unit_ids = tuple(item.unit_id for item in units)
    try:
        selected_units = tuple(selected_by_id[unit_id] for unit_id in unit_ids)
    except KeyError as exc:
        raise ValueError(
            f"Texture adapter unit is outside deterministic scope: {exc.args[0]}"
        ) from exc
    payload = plan.model_dump(mode="python")
    selected_unit_payloads = [item.model_dump(mode="python") for item in selected_units]
    payload["selected_units"] = selected_unit_payloads
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("deterministic Texture scope omitted counts")
    counts["selected_unit_count"] = len(selected_units)
    counts["selected_material_count"] = len(
        {
            path
            for selected_unit in selected_unit_payloads
            for path in selected_unit.get("material_prim_paths", ())
            if isinstance(path, str)
        }
    )
    counts["planned_generation_job_count"] = len(selected_units)
    scope_request = payload.get("request")
    if isinstance(scope_request, dict):
        discovery_mode = str(scope_request.get("discovery_mode") or "")
        if discovery_mode == "explicit":
            selected_material_paths = {
                path
                for selected_unit in selected_unit_payloads
                for field in ("material_prim_paths", "material_alias_paths")
                for path in selected_unit.get(field, ())
                if isinstance(path, str)
            }
            selected_member_paths = {
                path
                for disposition in units
                for path in (
                    *disposition.member_prim_paths,
                    *disposition.member_subset_paths,
                )
            }
            explicit_material_paths = tuple(
                path
                for path in scope_request.get("explicit_material_paths", ())
                if path in selected_material_paths
            )
            explicit_prim_paths = tuple(
                path
                for path in scope_request.get("explicit_prim_paths", ())
                if path in selected_member_paths
            )
            if not explicit_material_paths and not explicit_prim_paths:
                raise ValueError(
                    "selected Texture adapter units are outside the original "
                    "explicit request scope"
                )
            scope_request["explicit_material_paths"] = list(explicit_material_paths)
            scope_request["explicit_prim_paths"] = list(explicit_prim_paths)
    narrowed = TexturePlanDocument.model_validate(payload)
    if narrowed.selected_unit_ids != unit_ids:
        raise ValueError("narrowed Texture scope differs from adapter authorization")
    return narrowed


class TextureAgenticGeneratorLeafAdapter:
    """Production adapter from one accepted action to one trusted Texture leaf.

    The leaf factory is intentionally lazy. Preserve, defer, reject, and any
    unselected mutation action therefore cannot construct a provider client.
    """

    def __init__(
        self,
        *,
        action: TextureMutatingAction,
        leaf_factory: Callable[[], TextureGeneratorLeaf],
        adapter_id: str,
    ) -> None:
        normalized = str(adapter_id).strip()
        if not normalized:
            raise ValueError("Texture production adapter ID must not be empty")
        self.action = action
        self._leaf_factory = leaf_factory
        self.adapter_id = normalized

    def execute(
        self,
        *,
        accepted_plan: TextureAcceptedPlan,
        preparation: TexturePreparationPacket,
        units: tuple[TexturePlanUnitDisposition, ...],
        input_asset: ExecutionArtifactBinding,
        output_dir: Path,
    ) -> TextureAgenticAdapterResult:
        if not units or any(item.action != self.action for item in units):
            raise ValueError("Texture production adapter received another action")
        _verify_binding(input_asset, label=f"Texture {self.action} input")
        if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
            raise ValueError("Texture adapter output must be a regular directory")
        # A trusted leaf owns its exact resume contract inside this directory.
        # The service leaf persists a digest-bound plan/session checkpoint here;
        # deterministic leaves that do not support resume still fail closed on
        # their own pre-existing candidate directory.
        output_dir.mkdir(parents=True, exist_ok=True)
        leaf = self._leaf_factory()
        provider_id = str(leaf.provider_id).strip()
        capability_id = str(leaf.capability_id).strip()
        if not provider_id or not capability_id:
            raise ValueError("Texture leaf must declare provider and capability IDs")
        inputs = tuple(item.generator_inputs for item in units)
        if any(item is None for item in inputs):
            raise ValueError("Texture mutating adapter requires generator inputs")
        generator_inputs = tuple(item for item in inputs if item is not None)
        expected_mode = (
            "provider_generate" if self.action == "generate" else "apply_provided"
        )
        if any(item.execution_mode != expected_mode for item in generator_inputs):
            raise ValueError("Texture adapter execution mode differs from its action")
        if any(item.backend != provider_id for item in generator_inputs):
            raise ValueError("Texture accepted backend differs from selected leaf")
        narrowed_scope = _narrow_texture_scope_plan(preparation.scope_plan, units)
        dependencies = tuple(bind_usd_dependency_closure(input_asset.path))
        leaf_request = TextureGeneratorLeafRequest(
            outer_plan=accepted_plan.proposal,
            preparation=accepted_plan.preparation,
            source=input_asset,
            source_dependencies=dependencies,
            intent=preparation.request.intent,
            scope_plan=narrowed_scope,
            target_unit_ids=tuple(item.unit_id for item in units),
            generator_inputs=generator_inputs,
            reference_artifacts=accepted_plan.plan.reference_artifacts,
            output_dir=str(output_dir / "candidate"),
        )
        execution = TextureExecutionResult.model_validate(leaf.generate(leaf_request))
        expected_ids = leaf_request.target_unit_ids
        if execution.requested_unit_ids != expected_ids:
            raise ValueError("Texture leaf executed another unit scope")
        if execution.retry_count != 0:
            raise ValueError("Texture one-attempt adapter rejects provider retries")
        output = bind_texture_agentic_artifact(execution.output_asset_path)
        unit_artifacts = tuple(
            TextureAgenticUnitArtifacts(
                unit_id=item.unit_id,
                artifacts=tuple(
                    bind_texture_agentic_artifact(path) for path in item.artifact_paths
                ),
                metadata={
                    **item.metadata,
                    "generation": item.generation,
                    "provider": provider_id,
                    "capability": capability_id,
                },
            )
            for item in execution.unit_artifacts
        )
        execution_path = output_dir / "texture_leaf_execution.json"
        execution_binding = _write_packet(execution_path, execution)
        return TextureAgenticAdapterResult(
            action=self.action,
            unit_ids=expected_ids,
            adapter_id=self.adapter_id,
            input_asset=input_asset,
            output_asset=output,
            unit_artifacts=unit_artifacts,
            evidence_artifacts=(execution_binding,),
        )


def prepare_texture_agentic_source(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    target_prim_paths: tuple[str, ...],
) -> tuple[TextureAgenticSourcePreparation, ExecutionArtifactBinding]:
    """Generate missing UVs before semantic planning and bind every artifact.

    This is deliberately a provider-free pre-child operation.  The bounded UV
    leaf owns authoring and reopened-stage validation; this wrapper only freezes
    the resulting original/effective source chain for the agentic workflow.
    """

    from .uv_authoring import (
        TextureUvLeafResult,
        build_texture_uv_leaf_invocation,
        run_texture_uv_leaf,
    )

    if not target_prim_paths:
        raise ValueError("Texture source preparation requires explicit prim targets")
    operation_root = Path(output_dir).expanduser().resolve()
    operation_root.mkdir(parents=True, exist_ok=False)
    invocation = build_texture_uv_leaf_invocation(
        source_path,
        output_dir=operation_root,
        target_prim_paths=target_prim_paths,
        policy="generate_missing",
    )
    invocation_path = atomic_write_json(
        operation_root / "texture_uv_leaf_invocation.json",
        invocation,
    )
    invocation_binding = bind_texture_agentic_artifact(invocation_path)
    result = TextureUvLeafResult.model_validate(run_texture_uv_leaf(invocation_path))
    result_binding = bind_texture_agentic_artifact(
        operation_root / "texture_uv_leaf_result.json"
    )
    if result.invocation != invocation_binding:
        raise ValueError("Texture UV result binds another invocation")
    if result.source != invocation.source:
        raise ValueError("Texture UV result binds another original source")
    if result.native_disposition != "passed":
        raise ValueError(f"Texture UV preparation failed closed: {result.detail}")
    if any(
        (
            result.provider_invoked,
            result.texture_service_constructed,
            result.vlm_assessor_constructed,
            result.image_generator_constructed,
            result.fixed_pipeline_invoked,
            result.nested_coordinator_invoked,
        )
    ):
        raise ValueError("Texture UV preparation crossed a forbidden runtime boundary")
    for binding, label in (
        (result.source, "Texture original source"),
        (result.output, "Texture UV-prepared source"),
        (result_binding, "Texture UV result"),
        *((item, "Texture UV evidence") for item in result.evidence),
        *(
            (item, "Texture UV saved-stage readback")
            for item in result.saved_stage_readbacks
        ),
    ):
        _verify_binding(binding, label=label)
    from world_understanding.utils.usd.package import (
        extract_usdz_package_for_edit,
        write_usdz_package_from_directory,
    )

    # CreateNewUsdzPackage owns dependency localization and asset-path rewriting.
    # Stage.Flatten() alone preserves external asset paths, so writing only the
    # flattened layer would silently omit the verified output dependency closure.
    verify_usd_dependency_closure(result.output.path, result.output_dependencies)
    localized_path = operation_root / ".localized_texture.usdz"
    package_root = operation_root / ".localized_package"
    effective_path = operation_root / "prepared_texture.usdz"
    try:
        create_localized_usdz_package(
            result.output.path,
            localized_path,
            "prepared_texture.usdc",
        )
        verify_usd_dependency_closure(result.output.path, result.output_dependencies)
        localized_root = extract_usdz_package_for_edit(localized_path, package_root)
        write_usdz_package_from_directory(
            package_root,
            localized_root.relative_to(package_root),
            effective_path,
        )
    except BaseException:
        effective_path.unlink(missing_ok=True)
        raise
    finally:
        localized_path.unlink(missing_ok=True)
        shutil.rmtree(package_root, ignore_errors=True)
    effective_source = bind_texture_agentic_artifact(effective_path)
    if bind_usd_dependency_closure(effective_source.path):
        raise ValueError("Texture prepared source package is not byte-self-contained")
    receipt = TextureAgenticSourcePreparation(
        original_source=result.source,
        uv_prepared_source=result.output,
        effective_source=effective_source,
        target_prim_paths=target_prim_paths,
        invocation=invocation_binding,
        result=result_binding,
        evidence=result.evidence,
        saved_stage_readbacks=result.saved_stage_readbacks,
    )
    receipt_binding = _write_packet(
        operation_root / "texture_agentic_source_preparation.json",
        receipt,
    )
    return receipt, receipt_binding


def validate_texture_agentic_source_preparation(
    receipt: TextureAgenticSourcePreparation,
    *,
    receipt_binding: ExecutionArtifactBinding,
) -> None:
    """Replay a frozen provider-free source-preparation chain on resume."""

    from .uv_authoring import TextureUvLeafResult

    _verify_packet_binding(
        receipt_binding,
        receipt,
        label="Texture source preparation",
    )
    for binding, label in (
        (receipt.original_source, "Texture original source"),
        (receipt.uv_prepared_source, "Texture UV-prepared source"),
        (receipt.effective_source, "Texture UV-prepared source"),
        (receipt.invocation, "Texture UV invocation"),
        (receipt.result, "Texture UV result"),
        *((item, "Texture UV evidence") for item in receipt.evidence),
        *(
            (item, "Texture UV saved-stage readback")
            for item in receipt.saved_stage_readbacks
        ),
    ):
        _verify_binding(binding, label=label)
    result = TextureUvLeafResult.model_validate(load_json(receipt.result.path))
    if (
        result.invocation != receipt.invocation
        or result.source != receipt.original_source
        or result.output != receipt.uv_prepared_source
        or result.evidence != receipt.evidence
        or result.saved_stage_readbacks != receipt.saved_stage_readbacks
        or result.native_disposition != "passed"
    ):
        raise ValueError("Texture source preparation differs from its UV result")
    if bind_usd_dependency_closure(receipt.effective_source.path):
        raise ValueError("Texture prepared source package is not byte-self-contained")


def bind_texture_agentic_artifact(path: str | Path) -> ExecutionArtifactBinding:
    """Bind one regular non-symlink artifact by exact bytes."""

    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Texture agentic artifact must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"Texture agentic artifact is not a file: {resolved}")
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _verify_binding(binding: ExecutionArtifactBinding, *, label: str) -> None:
    if bind_texture_agentic_artifact(binding.path) != binding:
        raise ValueError(f"{label} bytes changed")


def _write_packet(path: str | Path, packet: BaseModel) -> ExecutionArtifactBinding:
    destination = Path(path).expanduser().resolve()
    atomic_write_json(destination, packet)
    type(packet).model_validate(load_json(destination))
    return bind_texture_agentic_artifact(destination)


def _verify_packet_binding(
    binding: ExecutionArtifactBinding,
    packet: BaseModel,
    *,
    label: str,
) -> None:
    _verify_binding(binding, label=label)
    persisted = type(packet).model_validate(load_json(binding.path))
    if persisted != packet:
        raise ValueError(f"{label} binding contains another typed packet")


def _texture_inspection_unit_target_paths(
    unit: TextureInspectionUnit,
) -> tuple[str, ...]:
    """Return every exact request-policy alias owned by one prepared unit."""

    return tuple(
        dict.fromkeys(
            (
                *unit.material_prim_paths,
                *unit.material_alias_paths,
                *unit.member_prim_paths,
                *unit.member_subset_paths,
            )
        )
    )


def validate_texture_agentic_plan(
    plan: TextureAgenticPlan,
    *,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
) -> None:
    """Reject stale, partial, unsafe, or unsupported child proposals."""

    _verify_binding(preparation_binding, label="Texture preparation")
    if plan.preparation != preparation_binding:
        raise ValueError("Texture plan binds a stale preparation")
    if plan.source != preparation.request.source:
        raise ValueError("Texture plan binds another source")
    _verify_binding(plan.source, label="Texture source")
    verify_usd_dependency_closure(
        plan.source.path,
        preparation.request.source_dependencies,
    )
    if plan.scope_plan_digest != preparation.scope_plan_digest:
        raise ValueError("Texture plan binds a stale deterministic scope")
    if plan.reference_artifacts != preparation.request.reference_artifacts:
        raise ValueError("Texture plan changed reference identities")
    for reference in plan.reference_artifacts:
        _verify_binding(
            reference.artifact,
            label=f"Texture reference {reference.role}",
        )
    if plan.capability_constraints != preparation.inspection.capability_constraints:
        raise ValueError("Texture plan changed capability constraints")
    required_views = preparation.request.metadata.get(_EVIDENCE_VIEWS_METADATA_KEY)
    if required_views is not None:
        if (
            not isinstance(required_views, list | tuple)
            or tuple(str(item) for item in required_views)
            != plan.evidence.required_views
        ):
            raise ValueError("Texture plan changed required OVRTX evidence views")
    inspected = {item.unit_id: item for item in preparation.inspection.units}
    if plan.unit_ids != tuple(inspected):
        raise ValueError(
            "Texture plan must cover every prepared unit exactly once in preparation order"
        )
    reference_bindings = tuple(item.artifact for item in plan.reference_artifacts)
    provided_by_unit: dict[str, TextureProvidedImageArtifact] = {}
    action_by_unit: dict[str, str] = {}
    appearance_by_unit: dict[str, str] = {}
    raw_action_policy = preparation.request.metadata.get(
        _ACTION_POLICY_METADATA_KEY, {}
    )
    if not isinstance(raw_action_policy, dict):
        raise ValueError("Texture action policy must be an object")
    supported_actions = {
        "preserve",
        "generate",
        "apply_provided",
        "defer",
        "reject",
    }
    for raw_target, raw_action in raw_action_policy.items():
        target_path = str(raw_target)
        action = str(raw_action)
        if action not in supported_actions:
            raise ValueError("Texture action policy names an unsupported action")
        matching_units = tuple(
            unit
            for unit in preparation.inspection.units
            if target_path in _texture_inspection_unit_target_paths(unit)
        )
        if len(matching_units) != 1:
            raise ValueError(
                "Texture action-policy target must map to one prepared unit"
            )
        unit_id = matching_units[0].unit_id
        existing_action = action_by_unit.get(unit_id)
        if existing_action is not None and existing_action != action:
            raise ValueError(
                "Texture action-policy targets for one prepared unit conflict"
            )
        action_by_unit[unit_id] = action
    raw_appearance_policy = preparation.request.metadata.get(
        _APPEARANCE_POLICY_METADATA_KEY, {}
    )
    if not isinstance(raw_appearance_policy, dict):
        raise ValueError("Texture appearance policy must be an object")
    for raw_target, raw_appearance in raw_appearance_policy.items():
        target_path = str(raw_target)
        if not isinstance(raw_appearance, str) or not raw_appearance.strip():
            raise ValueError("Texture appearance policy must not be empty")
        appearance = raw_appearance.strip()
        matching_units = tuple(
            unit
            for unit in preparation.inspection.units
            if target_path in _texture_inspection_unit_target_paths(unit)
        )
        if len(matching_units) != 1:
            raise ValueError(
                "Texture appearance-policy target must map to one prepared unit"
            )
        unit_id = matching_units[0].unit_id
        existing_appearance = appearance_by_unit.get(unit_id)
        if existing_appearance is not None and existing_appearance != appearance:
            raise ValueError(
                "Texture appearance-policy targets for one prepared unit conflict"
            )
        required_action = action_by_unit.get(unit_id)
        if required_action is not None and required_action not in {
            "generate",
            "apply_provided",
        }:
            raise ValueError(
                "Texture appearance policy requires a mutating per-unit action"
            )
        appearance_by_unit[unit_id] = appearance
    raw_provided = preparation.request.metadata.get(
        _PROVIDED_CANDIDATES_METADATA_KEY, ()
    )
    if not isinstance(raw_provided, list | tuple):
        raise ValueError("Texture provided-candidate inventory must be an array")
    for item in raw_provided:
        if not isinstance(item, dict):
            raise ValueError("Texture provided-candidate entry must be an object")
        target_path = str(item.get("target_path") or "")
        matching_units = tuple(
            unit
            for unit in preparation.inspection.units
            if target_path in _texture_inspection_unit_target_paths(unit)
        )
        if len(matching_units) != 1:
            raise ValueError(
                "Texture provided-candidate target must map to one prepared unit"
            )
        unit_id = matching_units[0].unit_id
        if unit_id in provided_by_unit:
            raise ValueError("Texture provided candidates must be unique per unit")
        provided_by_unit[unit_id] = TextureProvidedImageArtifact(
            unit_id=unit_id,
            channel="albedo",
            artifact=ExecutionArtifactBinding.model_validate(item.get("artifact")),
            producer=TextureProvidedImageProducer.model_validate(item.get("producer")),
        )
    for disposition in plan.dispositions:
        required_action = action_by_unit.get(disposition.unit_id)
        if required_action is not None and disposition.action != required_action:
            raise ValueError("Texture plan violates the exact per-unit action policy")
        required_appearance = appearance_by_unit.get(disposition.unit_id)
        if (
            required_appearance is not None
            and disposition.requested_appearance != required_appearance
        ):
            raise ValueError(
                "Texture plan violates the exact per-unit appearance policy"
            )
        unit = inspected[disposition.unit_id]
        if (
            disposition.material_prim_paths != unit.material_prim_paths
            or disposition.member_prim_paths != unit.member_prim_paths
            or disposition.member_subset_paths != unit.member_subset_paths
        ):
            raise ValueError(f"Texture plan target is unsafe: {disposition.unit_id}")
        if disposition.action in {"generate", "apply_provided"}:
            if unit.uv_status != "ready":
                raise ValueError(
                    "Texture mutation selected for unavailable UV scope: "
                    f"{disposition.unit_id}={unit.uv_status}"
                )
            inputs = disposition.generator_inputs
            assert inputs is not None
            if inputs.reference_artifacts != reference_bindings:
                raise ValueError(
                    f"Texture generator references are incomplete: {disposition.unit_id}"
                )
            for provided in inputs.provided_images:
                _verify_binding(
                    provided.artifact,
                    label=f"Texture provided image {provided.unit_id}",
                )
            assert disposition.requested_appearance is not None
            prepared_inputs = unit.proposed_generator_inputs
            unit_prompt = apply_detail_policy_to_prompt(
                disposition.requested_appearance,
                prepared_inputs.detail_policy,
            )
            if disposition.action == "generate":
                expected = prepared_inputs.model_copy(update={"prompt": unit_prompt})
                if inputs != expected:
                    raise ValueError(
                        "Texture generate inputs differ from the exact prepared unit "
                        "or its unit-specific requested appearance: "
                        f"{disposition.unit_id}"
                    )
            else:
                allowed = provided_by_unit.get(disposition.unit_id)
                if allowed is None or inputs.provided_images != (allowed,):
                    raise ValueError(
                        "Texture plan selected an unapproved provided candidate"
                    )
                expected = prepared_inputs.model_copy(
                    update={
                        "execution_mode": "apply_provided",
                        "backend": "outer_provided_image_apply",
                        "prompt": unit_prompt,
                        "engine": None,
                        "seed": None,
                        "parameters": {},
                        "provided_images": (allowed,),
                    }
                )
                if inputs != expected:
                    raise ValueError(
                        "Texture apply-provided inputs differ from the exact prepared "
                        f"unit and approved candidate: {disposition.unit_id}"
                    )


def accept_texture_agentic_plan(
    plan_path: str | Path,
    *,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    output_path: str | Path,
) -> tuple[TextureAcceptedPlan, ExecutionArtifactBinding]:
    """Validate child bytes and freeze the exact accepted plan."""

    proposal_binding = bind_texture_agentic_artifact(plan_path)
    plan = TextureAgenticPlan.model_validate(load_json(proposal_binding.path))
    validate_texture_agentic_plan(
        plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    accepted = TextureAcceptedPlan(
        proposal=proposal_binding,
        proposal_digest=canonical_json_digest(plan),
        preparation=preparation_binding,
        source=preparation.request.source,
        plan=plan,
    )
    binding = _write_packet(output_path, accepted)
    if bind_texture_agentic_artifact(plan_path) != proposal_binding:
        raise ValueError("Texture plan changed while it was being accepted")
    return accepted, binding


def validate_texture_accepted_plan(
    accepted: TextureAcceptedPlan,
    *,
    accepted_binding: ExecutionArtifactBinding,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
) -> None:
    """Replay the complete outer freeze before resume or execution."""

    _verify_packet_binding(
        accepted_binding,
        accepted,
        label="accepted Texture plan",
    )
    _verify_packet_binding(
        accepted.proposal,
        accepted.plan,
        label="proposed Texture plan",
    )
    validate_texture_agentic_plan(
        accepted.plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    if accepted.preparation != preparation_binding:
        raise ValueError("accepted Texture plan binds another preparation")


def _prepare_nonmutating_texture_candidate(
    source: ExecutionArtifactBinding,
    *,
    operation_root: Path,
    resume: bool,
) -> ExecutionArtifactBinding:
    """Package a dependency-bearing no-op result without semantic mutation."""

    dependencies = tuple(bind_usd_dependency_closure(source.path))
    if not dependencies:
        return source

    from world_understanding.utils.usd.package import (
        extract_usdz_package_for_edit,
        write_usdz_package_from_directory,
    )

    candidate_root = operation_root / "00-preserve" / "candidate"
    candidate_path = candidate_root / "preserved-texture-candidate.usdz"
    candidate_receipt_path = candidate_root / "preserve_candidate_receipt.json"

    def load_completed_candidate() -> ExecutionArtifactBinding:
        if candidate_receipt_path.is_symlink() or not candidate_receipt_path.is_file():
            raise ValueError(
                "persisted Texture preserve candidate receipt is missing or not regular"
            )
        receipt_binding = bind_texture_agentic_artifact(candidate_receipt_path)
        receipt = _TexturePreserveCandidateReceipt.model_validate(
            load_json(candidate_receipt_path)
        )
        _verify_packet_binding(
            receipt_binding,
            receipt,
            label="persisted Texture preserve candidate receipt",
        )
        if receipt.source != source:
            raise ValueError(
                "persisted Texture preserve candidate binds another source"
            )
        _verify_binding(receipt.candidate, label="persisted Texture preserve candidate")
        if Path(receipt.candidate.path).resolve() != candidate_path.resolve():
            raise ValueError(
                "persisted Texture preserve candidate binds another output path"
            )
        if bind_usd_dependency_closure(receipt.candidate.path):
            raise ValueError(
                "persisted Texture preserve candidate is not byte-self-contained"
            )
        return receipt.candidate

    if resume:
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise ValueError(
                "persisted Texture preserve candidate is missing or not regular"
            )
        candidate = bind_texture_agentic_artifact(candidate_path)
        if bind_usd_dependency_closure(candidate.path):
            raise ValueError(
                "persisted Texture preserve candidate is not byte-self-contained"
            )
        return candidate

    if candidate_root.is_symlink() or (
        candidate_root.exists() and not candidate_root.is_dir()
    ):
        raise ValueError(
            "Texture preserve candidate workspace must be a regular directory"
        )
    if candidate_root.exists():
        if candidate_receipt_path.exists() or candidate_receipt_path.is_symlink():
            return load_completed_candidate()
        # No ledger or completed-candidate receipt exists, so this exact
        # operation-owned directory is debris from an interrupted preparation.
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, mode=0o700)
    localized_path = candidate_root / ".localized-texture.usdz"
    package_root = candidate_root / ".localized-package"
    try:
        verify_usd_dependency_closure(source.path, dependencies)
        create_localized_usdz_package(
            source.path,
            localized_path,
            "preserved_texture.usdc",
        )
        verify_usd_dependency_closure(source.path, dependencies)
        localized_root = extract_usdz_package_for_edit(localized_path, package_root)
        write_usdz_package_from_directory(
            package_root,
            localized_root.relative_to(package_root),
            candidate_path,
        )
    finally:
        localized_path.unlink(missing_ok=True)
        if package_root.exists():
            if package_root.is_symlink() or not package_root.is_dir():
                raise ValueError(
                    "Texture preserve package workspace is not a regular directory"
                )
            shutil.rmtree(package_root)

    candidate = bind_texture_agentic_artifact(candidate_path)
    if bind_usd_dependency_closure(candidate.path):
        raise ValueError("Texture preserve candidate is not byte-self-contained")
    _write_packet(
        candidate_receipt_path,
        _TexturePreserveCandidateReceipt(source=source, candidate=candidate),
    )
    return candidate


def execute_texture_agentic_plan(
    accepted: TextureAcceptedPlan,
    *,
    accepted_binding: ExecutionArtifactBinding,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    adapter_factories: Mapping[TextureMutatingAction, TextureAdapterFactory],
    output_dir: str | Path,
) -> tuple[TextureAdapterCallLedger, ExecutionArtifactBinding]:
    """Construct and invoke exactly the selected mutating adapters, once each."""

    validate_texture_accepted_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    operation_root = Path(output_dir).expanduser().resolve()
    operation_root.mkdir(parents=True, exist_ok=True)
    ledger_path = operation_root / "texture_adapter_ledger.json"
    mutating_actions: tuple[TextureMutatingAction, ...] = (
        "generate",
        "apply_provided",
    )
    expected_calls = tuple(
        (
            action,
            tuple(
                item.unit_id
                for item in accepted.plan.dispositions
                if item.action == action
            ),
        )
        for action in mutating_actions
        if any(item.action == action for item in accepted.plan.dispositions)
    )
    current = accepted.source
    if not expected_calls:
        current = _prepare_nonmutating_texture_candidate(
            accepted.source,
            operation_root=operation_root,
            resume=ledger_path.exists() or ledger_path.is_symlink(),
        )
    records: list[TextureAdapterCallRecord] = []
    unresolved = tuple(
        item.unit_id
        for item in accepted.plan.dispositions
        if item.action in {"defer", "reject"}
    )

    def persist_observed_calls() -> tuple[
        TextureAdapterCallLedger,
        ExecutionArtifactBinding,
    ]:
        ledger = TextureAdapterCallLedger(
            accepted_plan=accepted_binding,
            preparation=preparation_binding,
            source=accepted.source,
            prepared_candidate=(
                current if not expected_calls and current != accepted.source else None
            ),
            records=tuple(records),
            final_candidate=current,
            unresolved_unit_ids=unresolved,
        )
        return ledger, _write_packet(ledger_path, ledger)

    if ledger_path.exists() or ledger_path.is_symlink():
        if ledger_path.is_symlink() or not ledger_path.is_file():
            raise ValueError("Texture adapter ledger must be a regular file")
        persisted_binding = bind_texture_agentic_artifact(ledger_path)
        persisted = TextureAdapterCallLedger.model_validate(load_json(ledger_path))
        if (
            persisted.accepted_plan != accepted_binding
            or persisted.preparation != preparation_binding
            or persisted.source != accepted.source
            or persisted.unresolved_unit_ids != unresolved
            or len(persisted.records) > len(expected_calls)
        ):
            raise ValueError("persisted Texture adapter ledger binds another execution")
        _verify_packet_binding(
            persisted_binding,
            persisted,
            label="persisted Texture adapter ledger",
        )
        for index, record in enumerate(persisted.records):
            expected_action, expected_ids = expected_calls[index]
            if (
                record.sequence != index + 1
                or record.action != expected_action
                or record.unit_ids != expected_ids
                or not record.adapter_id.strip()
                or record.input_asset != current
            ):
                raise ValueError(
                    "persisted Texture adapter ledger is not an exact selected prefix"
                )
            _verify_binding(record.input_asset, label="Texture adapter input")
            _verify_binding(record.output_asset, label="Texture adapter output")
            for unit_artifacts in record.unit_artifacts:
                for artifact in unit_artifacts.artifacts:
                    _verify_binding(
                        artifact,
                        label=(
                            f"Texture persisted unit artifact {unit_artifacts.unit_id}"
                        ),
                    )
            for evidence in record.evidence_artifacts:
                _verify_binding(evidence, label="Texture persisted adapter evidence")
            records.append(record)
            current = record.output_asset
        if persisted.final_candidate != current:
            raise ValueError("persisted Texture adapter candidate is stale")

    for action in mutating_actions:
        units = tuple(
            item for item in accepted.plan.dispositions if item.action == action
        )
        if not units:
            continue
        if any(record.action == action for record in records):
            continue
        factory = adapter_factories.get(action)
        if factory is None:
            raise RuntimeError(
                f"selected Texture capability is unavailable without fallback: {action}"
            )
        adapter = factory()
        adapter_id = str(adapter.adapter_id).strip()
        if not adapter_id:
            raise ValueError("Texture execution adapter must declare adapter_id")
        result = adapter.execute(
            accepted_plan=accepted,
            preparation=preparation,
            units=units,
            input_asset=current,
            output_dir=operation_root / f"{len(records) + 1:02d}-{action}",
        )
        expected_ids = tuple(item.unit_id for item in units)
        if (
            result.action != action
            or result.unit_ids != expected_ids
            or result.adapter_id != adapter_id
            or result.input_asset != current
        ):
            raise ValueError(f"Texture {action} adapter returned another authorization")
        _verify_binding(result.output_asset, label=f"Texture {action} output")
        for unit_artifacts in result.unit_artifacts:
            for artifact in unit_artifacts.artifacts:
                _verify_binding(
                    artifact,
                    label=f"Texture {action} unit artifact {unit_artifacts.unit_id}",
                )
        for evidence in result.evidence_artifacts:
            _verify_binding(evidence, label=f"Texture {action} evidence")
        records.append(
            TextureAdapterCallRecord(
                sequence=len(records) + 1,
                action=action,
                unit_ids=result.unit_ids,
                adapter_id=result.adapter_id,
                input_asset=result.input_asset,
                output_asset=result.output_asset,
                unit_artifacts=result.unit_artifacts,
                evidence_artifacts=result.evidence_artifacts,
            )
        )
        current = result.output_asset
        # Persist each completed adapter before attempting the next one. If a
        # later selected adapter fails, the terminal path can bind the exact
        # successful prefix instead of falsely reporting that nothing ran.
        persist_observed_calls()
    observed_calls = tuple((record.action, record.unit_ids) for record in records)
    if observed_calls != expected_calls:
        raise ValueError("Texture adapter-call ledger differs from selected operations")
    return persist_observed_calls()


def _validate_texture_agentic_saved_stage_readback(
    readback: TextureAgenticSavedStageReadback,
    *,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
) -> None:
    """Validate deterministic saved-stage readback facts."""

    for binding, label in (
        (accepted_binding, "accepted Texture plan"),
        (ledger_binding, "Texture adapter ledger"),
        (ledger.final_candidate, "Texture final candidate"),
        (readback.saved_stage, "saved Texture stage"),
    ):
        _verify_binding(binding, label=label)
    if (
        readback.accepted_plan != accepted_binding
        or readback.adapter_ledger != ledger_binding
        or readback.candidate != ledger.final_candidate
        or readback.scope_plan_digest != accepted.plan.scope_plan_digest
    ):
        raise ValueError("Texture saved-stage readback is stale or partial")
    for artifact in readback.verification_artifacts:
        _verify_binding(artifact, label="Texture saved-stage verification")


def validate_texture_agentic_saved_stage_readback(
    readback: TextureAgenticSavedStageReadback,
    *,
    readback_binding: ExecutionArtifactBinding,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
) -> None:
    """Revalidate an existing saved-stage readback and all bound bytes."""

    _verify_packet_binding(
        readback_binding,
        readback,
        label="Texture saved-stage readback",
    )
    _validate_texture_agentic_saved_stage_readback(
        readback,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
    )


def record_texture_agentic_saved_stage_readback(
    readback: TextureAgenticSavedStageReadback,
    *,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    output_path: str | Path,
) -> tuple[TextureAgenticSavedStageReadback, ExecutionArtifactBinding]:
    """Validate and bind deterministic saved-stage readback facts."""

    _validate_texture_agentic_saved_stage_readback(
        readback,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
    )
    binding = _write_packet(output_path, readback)
    return readback, binding


def _validate_texture_agentic_evidence(
    evidence: TextureAgenticEvidenceReceipt,
    *,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    readback: TextureAgenticSavedStageReadback,
    readback_binding: ExecutionArtifactBinding,
) -> None:
    """Validate exact current-run OVRTX evidence before semantic review."""

    for binding, label in (
        (accepted_binding, "accepted Texture plan"),
        (ledger_binding, "Texture adapter ledger"),
        (ledger.final_candidate, "Texture final candidate"),
    ):
        _verify_binding(binding, label=label)
    _verify_packet_binding(
        readback_binding,
        readback,
        label="Texture saved-stage readback",
    )
    if (
        evidence.accepted_plan != accepted_binding
        or evidence.adapter_ledger != ledger_binding
        or evidence.candidate != ledger.final_candidate
        or evidence.saved_stage_readback != readback_binding
        or readback.accepted_plan != accepted_binding
        or readback.adapter_ledger != ledger_binding
        or readback.candidate != ledger.final_candidate
    ):
        raise ValueError("Texture evidence is stale or binds another execution")
    if tuple(item.unit_id for item in evidence.unit_evidence) != accepted.plan.unit_ids:
        raise ValueError("Texture evidence must cover every plan unit in exact order")
    if evidence.view_names != accepted.plan.evidence.required_views:
        raise ValueError("Texture evidence view identity differs from the plan")
    view_count = len(accepted.plan.evidence.required_views)
    for unit in evidence.unit_evidence:
        if (
            len(unit.source_images) < view_count
            or len(unit.source_images) % view_count != 0
            or len(unit.candidate_images) < view_count
            or len(unit.candidate_images) % view_count != 0
        ):
            raise ValueError(
                f"Texture evidence views differ from the plan: {unit.unit_id}"
            )
        for artifact in (*unit.source_images, *unit.candidate_images):
            _verify_binding(
                artifact,
                label=f"Texture current-run OVRTX evidence {unit.unit_id}",
            )
    for artifact in evidence.static_evidence:
        _verify_binding(artifact, label="Texture static evidence")
    renderer = str(evidence.renderer_metadata.get("renderer") or "").lower()
    current_run = evidence.renderer_metadata.get("current_run")
    directions = _string_tuple(evidence.renderer_metadata.get("directions"))
    if (
        renderer != "ovrtx"
        or current_run is not True
        or directions != evidence.view_names
    ):
        raise ValueError("Texture evidence requires current-run OVRTX metadata")


def validate_texture_agentic_evidence(
    evidence: TextureAgenticEvidenceReceipt,
    *,
    evidence_binding: ExecutionArtifactBinding,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    readback: TextureAgenticSavedStageReadback,
    readback_binding: ExecutionArtifactBinding,
) -> None:
    """Revalidate an existing evidence receipt and every nested artifact."""

    _verify_packet_binding(
        evidence_binding,
        evidence,
        label="Texture evidence",
    )
    _validate_texture_agentic_evidence(
        evidence,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
    )


def record_texture_agentic_evidence(
    evidence: TextureAgenticEvidenceReceipt,
    *,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    readback: TextureAgenticSavedStageReadback,
    readback_binding: ExecutionArtifactBinding,
    output_path: str | Path,
) -> tuple[TextureAgenticEvidenceReceipt, ExecutionArtifactBinding]:
    """Validate and bind exact current-run OVRTX evidence."""

    _validate_texture_agentic_evidence(
        evidence,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
    )
    binding = _write_packet(output_path, evidence)
    return evidence, binding


def record_texture_agentic_review(
    review: TextureAgenticReviewReceipt,
    *,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    evidence: TextureAgenticEvidenceReceipt,
    evidence_binding: ExecutionArtifactBinding,
    output_path: str | Path,
) -> tuple[TextureAgenticReviewReceipt, ExecutionArtifactBinding]:
    """Recheck that a separate review covers exact plan and evidence bytes."""

    for binding, label in (
        (accepted_binding, "accepted Texture plan"),
        (ledger_binding, "Texture adapter ledger"),
        (ledger.final_candidate, "Texture final candidate"),
    ):
        _verify_binding(binding, label=label)
    _verify_packet_binding(
        evidence_binding,
        evidence,
        label="Texture evidence",
    )
    for record in ledger.records:
        for binding, label in (
            (record.input_asset, "Texture adapter input"),
            (record.output_asset, "Texture adapter output"),
        ):
            _verify_binding(binding, label=label)
        for unit_artifacts in record.unit_artifacts:
            for artifact in unit_artifacts.artifacts:
                _verify_binding(
                    artifact,
                    label=f"Texture unit artifact {unit_artifacts.unit_id}",
                )
        for artifact in record.evidence_artifacts:
            _verify_binding(artifact, label="Texture adapter evidence")
    for unit in evidence.unit_evidence:
        for artifact in (*unit.source_images, *unit.candidate_images):
            _verify_binding(
                artifact,
                label=f"Texture review visual evidence {unit.unit_id}",
            )
    for artifact in evidence.static_evidence:
        _verify_binding(artifact, label="Texture review static evidence")
    _verify_binding(
        evidence.saved_stage_readback,
        label="Texture saved-stage readback",
    )
    readback = TextureAgenticSavedStageReadback.model_validate(
        load_json(evidence.saved_stage_readback.path)
    )
    if (
        readback.accepted_plan != accepted_binding
        or readback.adapter_ledger != ledger_binding
        or readback.candidate != ledger.final_candidate
        or readback.scope_plan_digest != accepted.plan.scope_plan_digest
    ):
        raise ValueError("Texture review saved-stage readback is stale")
    for artifact in (readback.saved_stage, *readback.verification_artifacts):
        _verify_binding(artifact, label="Texture review saved-stage evidence")
    if (
        review.accepted_plan != accepted_binding
        or review.adapter_ledger != ledger_binding
        or review.evidence != evidence_binding
        or review.candidate != ledger.final_candidate
        or review.candidate != evidence.candidate
        or review.plan_digest != accepted.proposal_digest
        or tuple(item.unit_id for item in review.unit_reviews) != accepted.plan.unit_ids
    ):
        raise ValueError("Texture review is stale, partial, or rewrites plan facts")
    disposition_by_id = {
        item.unit_id: item.action for item in accepted.plan.dispositions
    }
    for item in review.unit_reviews:
        action = disposition_by_id[item.unit_id]
        if action == "defer" and item.disposition != "unresolved":
            raise ValueError("Texture review cannot upgrade a deferred plan unit")
        if action == "reject" and item.disposition != "reject":
            raise ValueError("Texture review cannot upgrade a rejected plan unit")
    required_visuals = tuple(
        binding
        for item in evidence.unit_evidence
        for binding in (*item.source_images, *item.candidate_images)
    )
    if review.inspected_visual_artifacts != required_visuals:
        raise ValueError("Texture review did not inspect exact current-run visuals")
    binding = _write_packet(output_path, review)
    return review, binding


def record_texture_agentic_publication(
    publication: TextureAgenticPublicationReceipt,
    *,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    evidence: TextureAgenticEvidenceReceipt,
    evidence_binding: ExecutionArtifactBinding,
    readback_binding: ExecutionArtifactBinding,
    review: TextureAgenticReviewReceipt,
    review_binding: ExecutionArtifactBinding,
    output_path: str | Path,
) -> tuple[TextureAgenticPublicationReceipt, ExecutionArtifactBinding]:
    """Validate guarded publication after exact evidence and separate review."""

    for packet_binding, packet, label in (
        (evidence_binding, evidence, "Texture evidence"),
        (review_binding, review, "Texture review"),
    ):
        _verify_packet_binding(packet_binding, packet, label=label)
    for binding, label in (
        (accepted_binding, "accepted Texture plan"),
        (ledger_binding, "Texture adapter ledger"),
        (readback_binding, "Texture saved-stage readback"),
        (publication.published_asset, "published Texture asset"),
    ):
        _verify_binding(binding, label=label)
    if ledger.unresolved_unit_ids:
        raise ValueError("unresolved Texture units block publication")
    if not review.accepted:
        raise ValueError("Texture publication requires an accepted separate review")
    if (
        publication.accepted_plan != accepted_binding
        or publication.adapter_ledger != ledger_binding
        or publication.candidate != ledger.final_candidate
        or publication.evidence != evidence_binding
        or publication.saved_stage_readback != readback_binding
        or publication.review != review_binding
        or evidence.saved_stage_readback != readback_binding
    ):
        raise ValueError("Texture publication is stale or changes custody")
    for artifact in publication.verification_artifacts:
        _verify_binding(artifact, label="Texture publication verification")
    binding = _write_packet(output_path, publication)
    return publication, binding


def record_texture_agentic_cleanup(
    cleanup: TextureAgenticCleanupReceipt,
    *,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    output_path: str | Path,
) -> tuple[TextureAgenticCleanupReceipt, ExecutionArtifactBinding]:
    """Bind cleanup status to the same accepted execution on every exit path."""

    if (
        cleanup.accepted_plan != accepted_binding
        or cleanup.adapter_ledger != ledger_binding
        or cleanup.candidate != ledger.final_candidate
    ):
        raise ValueError("Texture cleanup receipt binds another execution")
    for artifact in cleanup.retained_artifacts:
        _verify_binding(artifact, label="Texture cleanup retained artifact")
    binding = _write_packet(output_path, cleanup)
    return cleanup, binding


def seal_texture_agentic_terminal_receipt(
    *,
    request: ExecutionArtifactBinding,
    accepted: TextureAcceptedPlan,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
    evidence: TextureAgenticEvidenceReceipt | None,
    evidence_binding: ExecutionArtifactBinding | None,
    review: TextureAgenticReviewReceipt | None,
    review_binding: ExecutionArtifactBinding | None,
    publication: TextureAgenticPublicationReceipt | None,
    publication_binding: ExecutionArtifactBinding | None,
    cleanup: TextureAgenticCleanupReceipt,
    cleanup_binding: ExecutionArtifactBinding,
    disposition: Literal["published", "rejected", "blocked", "failed"],
    issue_codes: tuple[str, ...] = (),
    failure: str | None = None,
    output_path: str | Path,
) -> tuple[TextureAgenticTerminalReceipt, ExecutionArtifactBinding]:
    """Seal one success or fail-closed receipt after replaying exact custody."""

    for binding, label in (
        (request, "Texture request"),
        (accepted.proposal, "proposed Texture plan"),
        (accepted_binding, "accepted Texture plan"),
        (ledger_binding, "Texture adapter ledger"),
        (ledger.final_candidate, "Texture final candidate"),
    ):
        _verify_binding(binding, label=label)
    if (
        accepted_binding != ledger.accepted_plan
        or accepted.preparation != ledger.preparation
        or accepted.source != ledger.source
    ):
        raise ValueError("Texture terminal inputs do not share one accepted plan")
    if (evidence is None) != (evidence_binding is None):
        raise ValueError(
            "Texture evidence packet and binding must be supplied together"
        )
    if (review is None) != (review_binding is None):
        raise ValueError("Texture review packet and binding must be supplied together")
    if (publication is None) != (publication_binding is None):
        raise ValueError(
            "Texture publication packet and binding must be supplied together"
        )
    _verify_packet_binding(
        cleanup_binding,
        cleanup,
        label="Texture cleanup",
    )
    if (
        cleanup.accepted_plan != accepted_binding
        or cleanup.adapter_ledger != ledger_binding
        or cleanup.candidate != ledger.final_candidate
    ):
        raise ValueError("Texture terminal cleanup is stale")
    if evidence is not None and evidence_binding is not None:
        _verify_packet_binding(
            evidence_binding,
            evidence,
            label="Texture evidence",
        )
        _verify_binding(
            evidence.saved_stage_readback, label="Texture saved-stage readback"
        )
        readback = TextureAgenticSavedStageReadback.model_validate(
            load_json(evidence.saved_stage_readback.path)
        )
        if (
            evidence.accepted_plan != accepted_binding
            or evidence.adapter_ledger != ledger_binding
            or evidence.candidate != ledger.final_candidate
            or evidence.view_names != accepted.plan.evidence.required_views
            or tuple(item.unit_id for item in evidence.unit_evidence)
            != accepted.plan.unit_ids
            or readback.accepted_plan != accepted_binding
            or readback.adapter_ledger != ledger_binding
            or readback.candidate != ledger.final_candidate
            or readback.scope_plan_digest != accepted.plan.scope_plan_digest
        ):
            raise ValueError("Texture terminal evidence is stale")
        view_count = len(evidence.view_names)
        for unit in evidence.unit_evidence:
            if (
                len(unit.source_images) < view_count
                or len(unit.source_images) % view_count != 0
                or len(unit.candidate_images) < view_count
                or len(unit.candidate_images) % view_count != 0
            ):
                raise ValueError("Texture terminal evidence views are incomplete")
            for artifact in (*unit.source_images, *unit.candidate_images):
                _verify_binding(artifact, label="Texture terminal OVRTX evidence")
        for artifact in evidence.static_evidence:
            _verify_binding(artifact, label="Texture terminal static evidence")
        for artifact in (
            readback.saved_stage,
            *readback.verification_artifacts,
        ):
            _verify_binding(artifact, label="Texture terminal saved-stage readback")
        if (
            str(evidence.renderer_metadata.get("renderer") or "").lower() != "ovrtx"
            or evidence.renderer_metadata.get("current_run") is not True
            or _string_tuple(evidence.renderer_metadata.get("directions"))
            != evidence.view_names
        ):
            raise ValueError("Texture terminal evidence is not current-run OVRTX")
    if review is not None and review_binding is not None:
        _verify_packet_binding(
            review_binding,
            review,
            label="Texture review",
        )
        if (
            evidence_binding is None
            or review.accepted_plan != accepted_binding
            or review.adapter_ledger != ledger_binding
            or review.evidence != evidence_binding
            or review.candidate != ledger.final_candidate
            or review.plan_digest != accepted.proposal_digest
            or tuple(item.unit_id for item in review.unit_reviews)
            != accepted.plan.unit_ids
        ):
            raise ValueError("Texture terminal review is stale")
        if evidence is None:
            raise ValueError("Texture terminal review requires exact evidence")
        required_visuals = tuple(
            artifact
            for item in evidence.unit_evidence
            for artifact in (*item.source_images, *item.candidate_images)
        )
        if review.inspected_visual_artifacts != required_visuals:
            raise ValueError("Texture terminal review did not inspect exact visuals")
    if disposition == "published":
        if ledger.unresolved_unit_ids:
            raise ValueError("unresolved Texture units block publication")
        if evidence is None or evidence_binding is None:
            raise ValueError("Texture publication requires exact evidence")
        if review is None or review_binding is None or not review.accepted:
            raise ValueError("Texture publication requires an accepted separate review")
        if publication is None or publication_binding is None:
            raise ValueError("Texture publication requires a publication binding")
        _verify_packet_binding(
            publication_binding,
            publication,
            label="Texture publication",
        )
        if (
            publication.accepted_plan != accepted_binding
            or publication.adapter_ledger != ledger_binding
            or publication.candidate != ledger.final_candidate
            or publication.evidence != evidence_binding
            or publication.saved_stage_readback != evidence.saved_stage_readback
            or publication.review != review_binding
        ):
            raise ValueError("Texture terminal publication is stale")
    elif publication is not None or publication_binding is not None:
        raise ValueError("non-published Texture terminal cannot bind publication")
    if disposition == "rejected" and (review is None or review.accepted):
        raise ValueError("rejected Texture terminal requires a rejecting review")
    terminal = TextureAgenticTerminalReceipt(
        request=request,
        preparation=accepted.preparation,
        proposed_plan=accepted.proposal,
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        evidence=evidence_binding,
        saved_stage_readback=(
            evidence.saved_stage_readback if evidence is not None else None
        ),
        review=review_binding,
        publication=publication_binding,
        cleanup=cleanup_binding,
        cleanup_status=cleanup.status,
        disposition=disposition,
        issue_codes=issue_codes,
        failure=failure,
    )
    binding = _write_packet(output_path, terminal)
    return terminal, binding


__all__ = [
    "TEXTURE_ACCEPTED_PLAN_SCHEMA_VERSION",
    "TEXTURE_ADAPTER_LEDGER_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_CLEANUP_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_EVIDENCE_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_PLAN_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_PUBLICATION_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_READBACK_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_REVIEW_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_SOURCE_PREPARATION_SCHEMA_VERSION",
    "TEXTURE_AGENTIC_TERMINAL_SCHEMA_VERSION",
    "TextureAcceptedPlan",
    "TextureAdapterCallLedger",
    "TextureAdapterCallRecord",
    "TextureAdapterFactory",
    "TextureAgenticAction",
    "TextureAgenticAdapterResult",
    "TextureAgenticCleanupReceipt",
    "TextureAgenticEvidenceReceipt",
    "TextureAgenticEvidenceRequirements",
    "TextureAgenticExecutionAdapter",
    "TextureAgenticGeneratorLeafAdapter",
    "TextureAgenticPlan",
    "TextureAgenticPublicationReceipt",
    "TextureAgenticReviewReceipt",
    "TextureAgenticSavedStageReadback",
    "TextureAgenticSourcePreparation",
    "TextureAgenticStopPolicy",
    "TextureAgenticTerminalReceipt",
    "TextureAgenticUnitReview",
    "TextureAgenticUnitArtifacts",
    "TextureMutatingAction",
    "TexturePlanUnitDisposition",
    "accept_texture_agentic_plan",
    "bind_texture_agentic_artifact",
    "execute_texture_agentic_plan",
    "prepare_texture_agentic_source",
    "record_texture_agentic_cleanup",
    "record_texture_agentic_evidence",
    "record_texture_agentic_publication",
    "record_texture_agentic_review",
    "record_texture_agentic_saved_stage_readback",
    "seal_texture_agentic_terminal_receipt",
    "validate_texture_accepted_plan",
    "validate_texture_agentic_plan",
    "validate_texture_agentic_source_preparation",
]
