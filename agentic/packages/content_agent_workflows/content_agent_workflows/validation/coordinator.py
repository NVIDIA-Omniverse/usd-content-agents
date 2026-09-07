# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision-only coordinator boundary for agentic Validation readiness.

Preparation is deterministic and selects nothing. A confined child may author
only ``ValidationCoordinatorPlanPatch`` data. The trusted outer process then
validates that proposal, materializes the classic plan as a compatibility
projection, and invokes the existing exact Validation operation adapters.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)
from world_understanding.utils.credentials import ensure_no_inline_secrets
from world_understanding.validation import ValidationRequest
from world_understanding.validation.cli import scaffold_policy_from_request
from world_understanding.validation.scaffold_runner import (
    ScaffoldValidationStepExecutor,
)
from world_understanding.validation.templates import V1_TEMPLATE_NAMES

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)

from .models import (
    ValidationArtifactIdentity,
    ValidationWorkflowIdentity,
    ValidationWorkflowRun,
)
from .operations import (
    EXECUTED_VALIDATION_ARTIFACT_NAMES,
    VALIDATION_OPERATION_INDEX_NAME,
    VALIDATION_OPERATION_PREPARATION_NAME,
    VALIDATION_OPERATION_RULES,
    ValidationOperationResult,
    _prepare_validation_operations_from_coordinator,
    finalize_validation_operations,
    run_validation_operation,
)
from .workflow import (
    ValidationStepExecutor,
    ValidationWorkflowError,
    _artifact_identity,
    _validate_output_location,
    _workflow_identity,
)

VALIDATION_COORDINATOR_PREPARATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-preparation.v1"
)
VALIDATION_COORDINATOR_PLAN_PATCH_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-plan-patch.v1"
)
VALIDATION_COORDINATOR_ACCEPTED_PLAN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-accepted-plan.v1"
)
VALIDATION_COORDINATOR_EXECUTION_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-execution-receipt.v1"
)

VALIDATION_COORDINATOR_PREPARATION_NAME: Final = (
    "validation_coordinator_preparation.json"
)
VALIDATION_COORDINATOR_PLAN_PATCH_NAME: Final = "validation_coordinator_plan_patch.json"
VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME: Final = (
    "validation_coordinator_accepted_plan.json"
)
VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME: Final = (
    "validation_coordinator_execution_receipt.json"
)
_VALIDATION_CHILD_LAUNCH_DESCRIPTOR_NAME: Final = (
    "raw/validation_planner_launch_descriptor.json"
)

_CAPABILITY_IDS: Final[Mapping[str, str]] = {
    "render_valid": "validation.render_valid",
    "look_right": "validation.look_right",
    "physics_sane": "validation.physics_sane",
    "physical_behavior": "validation.physical_behavior",
}
_RULE_IDS: Final[Mapping[str, str]] = {
    template_name: rule_id
    for rule_id, template_name in VALIDATION_OPERATION_RULES.items()
}
_PARAMETER_FIELDS: Final[Mapping[str, tuple[str, ...]]] = {
    # Provider identity and render geometry are frozen by the outer request.
    # The planning child selects this capability but cannot author its policy.
    "render_valid": (),
    "look_right": (),
    "physics_sane": (),
    "physical_behavior": ("evidence_paths",),
}
_PROVIDER_FREE_CAPABILITY_IDS: Final = frozenset(
    {
        "validation.render_valid",
        "validation.physics_sane",
        "validation.physical_behavior",
    }
)
_EVIDENCE_REQUIREMENTS: Final[Mapping[str, tuple[str, ...]]] = {
    "render_valid": (
        "source_identity",
        "render_result",
        "qualified_render_evidence",
    ),
    "look_right": ("render_result", "reference_inventory", "critique_result"),
    "physics_sane": ("source_identity", "usd_schema_result"),
    "physical_behavior": ("source_identity", "runtime_evidence", "behavior_result"),
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationCoordinatorCapability(_FrozenModel):
    """One exact adapter available to the child for selection."""

    capability_id: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    provider_requirement: Literal["none", "explicit_provider"]
    allowed_parameters: tuple[str, ...]
    evidence_requirements: tuple[str, ...] = Field(min_length=1)
    preflight_requirements: tuple[str, ...] = Field(min_length=1)


class ValidationCoordinatorEvidenceInventoryItem(_FrozenModel):
    """One source-adjacent item known before any check is selected."""

    evidence_id: str = Field(min_length=1)
    evidence_kind: Literal["reference", "runtime", "render", "package", "upstream"]
    artifact: ValidationArtifactIdentity
    availability: Literal["available", "missing"]


class ValidationCoordinatorConstraints(_FrozenModel):
    """Mandatory outer policy frozen before child planning."""

    allowed_capability_ids: tuple[str, ...] = Field(min_length=1)
    required_capability_ids: tuple[str, ...] = ()
    provider_free_capability_ids: tuple[str, ...] = tuple(
        sorted(_PROVIDER_FREE_CAPABILITY_IDS)
    )
    minimum_provider_free_checks: Literal[2] = 2
    minimum_required_checks: Literal[1] = 1
    explicit_dependency_required: Literal[True] = True
    maximum_selected_checks: int = Field(default=4, ge=1, le=4)
    one_instance_per_capability: Literal[True] = True
    source_mutation_allowed: Literal[False] = False
    classic_plan_is_decision_authority: Literal[False] = False
    revision_allowed: Literal[False] = False


class ValidationCoordinatorPreparation(_FrozenModel):
    """Provider-neutral, selection-free input to the planning child."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-preparation.v1"
    ] = VALIDATION_COORDINATOR_PREPARATION_SCHEMA_VERSION
    output_dir: str = Field(min_length=1)
    config_base_dir: str = Field(min_length=1)
    request: ValidationRequest
    task_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_identity: ValidationWorkflowIdentity
    source_dependency_closure: tuple[ValidationArtifactIdentity, ...] = Field(
        min_length=1
    )
    approved_capabilities: tuple[ValidationCoordinatorCapability, ...] = Field(
        min_length=1
    )
    evidence_inventory: tuple[ValidationCoordinatorEvidenceInventoryItem, ...] = ()
    mandatory_constraints: ValidationCoordinatorConstraints
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_checks: tuple[()] = ()
    provider_invoked: Literal[False] = False
    operation_invoked: Literal[False] = False
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_preparation(self) -> Self:
        if self.request.requested_templates:
            raise ValueError("coordinator preparation must select no templates")
        if self.source_dependency_closure != self.adapter_identity.source_artifacts:
            raise ValueError("source dependency closure differs from adapter identity")
        if self.task_identity_sha256 != canonical_json_digest(
            {"task_description": self.request.task_description}
        ):
            raise ValueError("task identity is stale")
        if self.request_identity_sha256 != canonical_json_digest(self.request):
            raise ValueError("request identity is stale")
        if self.config_identity_sha256 != canonical_json_digest(
            cast(
                dict[str, JsonValue],
                {
                    "policy_digest": self.adapter_identity.policy_digest,
                    "backend_digest": self.adapter_identity.backend_digest,
                    "template_versions": self.adapter_identity.template_versions,
                },
            )
        ):
            raise ValueError("config identity is stale")
        expected = canonical_json_digest(
            self.model_dump(mode="json", exclude={"preparation_digest"})
        )
        if self.preparation_digest != expected:
            raise ValueError("preparation digest is stale")
        return self


class ValidationCoordinatorClaim(_FrozenModel):
    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)
    check_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if len(self.acceptance_criteria) != len(set(self.acceptance_criteria)):
            raise ValueError("claim acceptance criteria must be unique")
        if len(self.check_ids) != len(set(self.check_ids)):
            raise ValueError("claim check IDs must be unique")
        return self


class ValidationCoordinatorSelectedCheck(_FrozenModel):
    """One child-proposed, non-executing Validation adapter invocation."""

    check_id: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_.-]*$")
    capability_id: str = Field(min_length=1)
    template_name: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    targets: tuple[str, ...] = Field(min_length=1)
    focus_prim_paths: tuple[str, ...] = ()
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    required: bool = True
    depends_on: tuple[str, ...] = ()
    evidence_requirements: tuple[str, ...] = Field(min_length=1)
    completion_policy: Literal["all_required_evidence", "best_effort_advisory"] = (
        "all_required_evidence"
    )

    @model_validator(mode="after")
    def validate_local_shape(self) -> Self:
        for field_name, values in (
            ("targets", self.targets),
            ("focus_prim_paths", self.focus_prim_paths),
            ("depends_on", self.depends_on),
            ("evidence_requirements", self.evidence_requirements),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if any(not path.startswith("/") for path in self.focus_prim_paths):
            raise ValueError("focus prim paths must be absolute USD prim paths")
        if self.required and self.completion_policy != "all_required_evidence":
            raise ValueError("required checks need all_required_evidence completion")
        if not self.required and self.completion_policy != "best_effort_advisory":
            raise ValueError("advisory checks need best_effort_advisory completion")
        return self


class ValidationCoordinatorPlanPatch(_FrozenModel):
    """Only artifact the untrusted planning child may author."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-plan-patch.v1"
    ] = VALIDATION_COORDINATOR_PLAN_PATCH_SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer: Literal["codex", "claude"]
    child_session_id: str = Field(min_length=1)
    claims: tuple[ValidationCoordinatorClaim, ...] = Field(min_length=1)
    selected_checks: tuple[ValidationCoordinatorSelectedCheck, ...] = Field(
        min_length=1
    )
    decision_only: Literal[True] = True
    source_mutated: Literal[False] = False
    execution_performed: Literal[False] = False
    publication_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        check_ids = tuple(check.check_id for check in self.selected_checks)
        capability_ids = tuple(check.capability_id for check in self.selected_checks)
        template_names = tuple(check.template_name for check in self.selected_checks)
        claim_ids = tuple(claim.claim_id for claim in self.claims)
        for label, values in (
            ("check IDs", check_ids),
            ("capability IDs", capability_ids),
            ("templates", template_names),
            ("claim IDs", claim_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"coordinator plan contains duplicate {label}")
        known_checks = set(check_ids)
        claimed_checks: set[str] = set()
        for claim in self.claims:
            if not set(claim.check_ids).issubset(known_checks):
                raise ValueError("claim references an undeclared check")
            claimed_checks.update(claim.check_ids)
        if claimed_checks != known_checks:
            raise ValueError("every selected check must support an explicit claim")
        return self


class ValidationCoordinatorAcceptedPlan(_FrozenModel):
    """Outer receipt proving the child proposal was accepted before execution."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-accepted-plan.v1"
    ] = VALIDATION_COORDINATOR_ACCEPTED_PLAN_SCHEMA_VERSION
    plan_id: str = Field(min_length=1)
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ExecutionArtifactBinding
    plan_patch: ExecutionArtifactBinding
    child_output: ExecutionArtifactBinding
    child_launch_descriptor: ExecutionArtifactBinding
    producer: Literal["codex", "claude"]
    child_session_id: str = Field(min_length=1)
    selected_checks: tuple[ValidationCoordinatorSelectedCheck, ...] = Field(
        min_length=1
    )
    ordered_check_ids: tuple[str, ...] = Field(min_length=1)
    operation_preparation: ExecutionArtifactBinding
    compatibility_request: ExecutionArtifactBinding
    compatibility_plan: ExecutionArtifactBinding
    accepted_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_mutated: Literal[False] = False
    execution_started: Literal[False] = False

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.accepted_at.tzinfo is None or self.accepted_at.utcoffset() is None:
            raise ValueError("accepted_at must include a timezone")
        if self.accepted_plan_digest != canonical_json_digest(
            {
                "plan_id": self.plan_id,
                "preparation_digest": self.preparation_digest,
                "selected_checks": [
                    check.model_dump(mode="json") for check in self.selected_checks
                ],
                "ordered_check_ids": list(self.ordered_check_ids),
            }
        ):
            raise ValueError("accepted plan digest is stale")
        expected = canonical_json_digest(
            self.model_dump(mode="json", exclude={"receipt_digest"})
        )
        if self.receipt_digest != expected:
            raise ValueError("accepted plan receipt digest is stale")
        return self


class ValidationCoordinatorExecutionReceipt(_FrozenModel):
    """Digest-bound readback of exact adapter execution and publication inputs."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-execution-receipt.v1"
    ] = VALIDATION_COORDINATOR_EXECUTION_RECEIPT_SCHEMA_VERSION
    accepted_plan: ExecutionArtifactBinding
    accepted_plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation_results: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    operation_index: ExecutionArtifactBinding
    validation_result: ExecutionArtifactBinding
    validation_evidence: ExecutionArtifactBinding
    final_summary: ExecutionArtifactBinding
    source_before: tuple[ValidationArtifactIdentity, ...]
    source_after: tuple[ValidationArtifactIdentity, ...]
    execution_disposition: Literal["completed", "required_checks_failed"]
    required_checks_successful: bool
    receipt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    coordinator_planning_agent_launched: Literal[True] = True
    independent_assessment_required: Literal[True] = True
    publication_kind: Literal["validation_assessment"] = "validation_assessment"
    source_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.source_before != self.source_after:
            raise ValueError("Validation coordinator execution changed its source")
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("completed_at must include a timezone")
        if self.required_checks_successful != (
            self.execution_disposition == "completed"
        ):
            raise ValueError("execution disposition is inconsistent")
        expected = canonical_json_digest(
            self.model_dump(mode="json", exclude={"receipt_digest"})
        )
        if self.receipt_digest != expected:
            raise ValueError("execution receipt digest is stale")
        return self


def _binding(path: str | Path) -> ExecutionArtifactBinding:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValidationWorkflowError(
            f"Validation coordinator artifact must not be a symlink: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationWorkflowError(
            f"Validation coordinator artifact is missing: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise ValidationWorkflowError(
            f"Validation coordinator artifact must be a file: {resolved}"
        )
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationWorkflowError(
            f"Validation child launch descriptor has invalid {label}"
        )
    return value


def _bind_validation_child_launch_descriptor(
    preparation: ValidationCoordinatorPreparation,
    *,
    child_output: ExecutionArtifactBinding,
    producer: Literal["codex", "claude"],
    expected_descriptor_digest: str,
) -> ExecutionArtifactBinding:
    """Bind the shared launch contract before claiming a child was launched."""

    root = Path(preparation.output_dir)
    descriptor_path = root / _VALIDATION_CHILD_LAUNCH_DESCRIPTOR_NAME
    if descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise ValidationWorkflowError(
            "Validation child launch descriptor is missing or unsafe"
        )
    try:
        payload = load_json(descriptor_path)
    except (OSError, ValueError) as exc:
        raise ValidationWorkflowError(
            f"Invalid Validation child launch descriptor: {exc}"
        ) from exc
    descriptor = _required_mapping(payload, label="root")
    if (
        len(expected_descriptor_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_descriptor_digest
        )
        or canonical_json_digest(cast(dict[str, JsonValue], dict(descriptor)))
        != expected_descriptor_digest
    ):
        raise ValidationWorkflowError(
            "Validation child launch descriptor differs from the parent-owned "
            "pre-launch descriptor"
        )
    expected_identity = _binding(root / VALIDATION_COORDINATOR_PREPARATION_NAME)
    expected_artifact_identity = {
        "path": expected_identity.path,
        "sha256": expected_identity.sha256,
        "size_bytes": expected_identity.size_bytes,
    }
    expected_scalar_fields = {
        "schema_version": "content-agents.child-launch.v1",
        "profile_key": "validation.plan",
        "workflow": "validation.plan",
        "workflow_skill": "content-workflow-validation",
        "scene_backend": "none",
        "skill_staging_mode": "required",
    }
    if any(
        descriptor.get(key) != value for key, value in expected_scalar_fields.items()
    ):
        raise ValidationWorkflowError(
            "Validation child launch descriptor does not match validation.plan"
        )
    if descriptor.get("required_staged_skills") != ["content-workflow-validation"]:
        raise ValidationWorkflowError(
            "Validation child launch descriptor has the wrong staged skills"
        )
    for label in ("capability_inventory", "domain_policy_bounds"):
        if descriptor.get(label) != expected_artifact_identity:
            raise ValidationWorkflowError(
                f"Validation child launch descriptor has stale {label}"
            )
    network_policy = _required_mapping(
        descriptor.get("network_policy"), label="network policy"
    )
    if (
        network_policy.get("schema_version")
        != "content-agents.child-launch-network-policy.v1"
        or network_policy.get("mode") != "reasoning_transport_only"
        or network_policy.get("tool_network_access") is not False
        or network_policy.get("allowed_hosts") != []
    ):
        raise ValidationWorkflowError(
            "Validation child launch descriptor changes the network policy"
        )
    credential_policy = _required_mapping(
        descriptor.get("credential_policy"), label="credential policy"
    )
    forbidden_environment_names = credential_policy.get("forbidden_environment_names")
    if (
        credential_policy.get("mode") != "reasoning_transport_only"
        or not isinstance(forbidden_environment_names, list)
        or any(
            not isinstance(name, str) or not name
            for name in forbidden_environment_names
        )
    ):
        raise ValidationWorkflowError(
            "Validation child launch descriptor changes the credential policy"
        )
    runner_identity = _required_mapping(
        descriptor.get("runner_identity"), label="runner identity"
    )
    if runner_identity.get("runner") != producer:
        raise ValidationWorkflowError(
            "Validation child launch descriptor differs from the child producer"
        )
    artifacts = _required_mapping(descriptor.get("artifacts"), label="artifacts")
    if (
        artifacts.get("schema_version") != "content-agents.child-launch-artifacts.v1"
        or artifacts.get("bridge_artifact_prefix") != "validation_planner"
        or artifacts.get("run_root") != str(root)
        or artifacts.get("child_final_path") != child_output.path
    ):
        raise ValidationWorkflowError(
            "Validation child launch descriptor has stale child artifacts"
        )
    raw_root = root / "raw"
    raw_output_value = artifacts.get("child_output_path")
    if not isinstance(raw_output_value, str):
        raise ValidationWorkflowError(
            "Validation child launch descriptor has invalid child output"
        )
    raw_output_path = Path(raw_output_value).expanduser()
    if not raw_output_path.is_absolute() or raw_output_path.is_symlink():
        raise ValidationWorkflowError(
            "Validation child launch output is missing or unsafe"
        )
    raw_output = raw_output_path.resolve()
    try:
        raw_output.relative_to(raw_root)
    except ValueError as exc:
        raise ValidationWorkflowError(
            "Validation child launch output must stay under run/raw"
        ) from exc
    if not raw_output.is_file() or raw_output.stat().st_size == 0:
        raise ValidationWorkflowError(
            "Validation child launch output is missing or unsafe"
        )
    return _binding(descriptor_path)


def _verify_binding(binding: ExecutionArtifactBinding) -> None:
    if _binding(binding.path) != binding:
        raise ValidationWorkflowError(
            f"Validation coordinator artifact binding is stale: {binding.path}"
        )


def _capabilities() -> tuple[ValidationCoordinatorCapability, ...]:
    requirements = {
        "render_valid": (
            "Existing render evidence or an explicitly configured OVRTX renderer.",
        ),
        "look_right": (
            "A passed render_valid dependency and an explicitly configured provider.",
        ),
        "physics_sane": ("A local USD schema runtime; no model provider.",),
        "physical_behavior": (
            "Existing simulator, runtime, or recording evidence; no simulator launch.",
        ),
    }
    return tuple(
        ValidationCoordinatorCapability(
            capability_id=_CAPABILITY_IDS[template_name],
            template_name=template_name,
            rule_id=_RULE_IDS[template_name],
            provider_requirement=(
                "explicit_provider" if template_name == "look_right" else "none"
            ),
            allowed_parameters=_PARAMETER_FIELDS[template_name],
            evidence_requirements=_EVIDENCE_REQUIREMENTS[template_name],
            preflight_requirements=requirements[template_name],
        )
        for template_name in V1_TEMPLATE_NAMES
    )


def _path_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str | Path):
        return (str(value),)
    if isinstance(value, Mapping):
        path = value.get("path")
        return (str(path),) if isinstance(path, str | Path) else ()
    if isinstance(value, list | tuple):
        return tuple(path for item in value for path in _path_values(item))
    return ()


def _inventory(
    identity: ValidationWorkflowIdentity,
    *,
    request: ValidationRequest,
    config_base_dir: Path,
) -> tuple[ValidationCoordinatorEvidenceInventoryItem, ...]:
    records = list(
        ValidationCoordinatorEvidenceInventoryItem(
            evidence_id=f"reference:{index}",
            evidence_kind="reference",
            artifact=artifact,
            availability="missing" if artifact.kind == "missing" else "available",
        )
        for index, artifact in enumerate(identity.reference_artifacts)
    )
    policy = scaffold_policy_from_request(request, base_dir=config_base_dir)
    runtime_keys = (
        "physical_behavior_evidence",
        "behavior_evidence",
        "physical_behavior_refine_summary_path",
        "refine_summary_path",
        "physical_behavior_refine_output_dir",
        "physics_refine_output_dir",
        "refine_output_dir",
        "animation_usd_paths",
        "behavior_video_paths",
        "sampled_video_frame_paths",
        "simulation_json_paths",
        "time_sampled_usd_paths",
        "trajectory_metrics_paths",
        "video_paths",
    )
    render_keys = (
        "animation_frame_paths",
        "current_image_paths",
        "render_image_paths",
        "qualified_render_evidence",
    )
    focused_image_paths = policy.get("focused_image_paths")
    focused_render_paths = (
        tuple(
            path
            for value in focused_image_paths.values()
            for path in _path_values(value)
        )
        if isinstance(focused_image_paths, Mapping)
        else ()
    )
    runtime_render = policy.get("runtime_render")
    runtime_render_paths = (
        _path_values(runtime_render.get("image_paths"))
        if isinstance(runtime_render, Mapping)
        else ()
    )
    paths_by_kind: Mapping[
        Literal["runtime", "render", "package", "upstream"], tuple[str, ...]
    ] = {
        "runtime": tuple(
            path for key in runtime_keys for path in _path_values(policy.get(key))
        ),
        "render": (
            *(path for key in render_keys for path in _path_values(policy.get(key))),
            *focused_render_paths,
            *runtime_render_paths,
        ),
        "package": _path_values(policy.get("package_evidence_paths")),
        "upstream": _path_values(policy.get("upstream_evidence_paths")),
    }
    seen = {(record.evidence_kind, record.artifact.path) for record in records}
    counters = {kind: 0 for kind in paths_by_kind}
    for evidence_kind, paths in paths_by_kind.items():
        for path in paths:
            artifact = _artifact_identity(
                path,
                role="evidence",
                base_dir=config_base_dir,
            )
            identity = (evidence_kind, artifact.path)
            if identity in seen:
                continue
            seen.add(identity)
            index = counters[evidence_kind]
            counters[evidence_kind] += 1
            records.append(
                ValidationCoordinatorEvidenceInventoryItem(
                    evidence_id=f"{evidence_kind}:{index}",
                    evidence_kind=evidence_kind,
                    artifact=artifact,
                    availability=(
                        "missing" if artifact.kind == "missing" else "available"
                    ),
                )
            )
    return tuple(records)


def _coordinator_request(
    request: ValidationRequest,
    *,
    output_dir: Path,
) -> ValidationRequest:
    policy = dict(request.policy)
    policy.setdefault("visual_evidence_mode", "canonical_usd")
    return request.model_copy(
        deep=True,
        update={
            "project": request.project.model_copy(
                update={"working_dir": str(output_dir)}
            ),
            "requested_templates": (),
            "policy": policy,
        },
    )


def prepare_validation_coordinator(
    request: ValidationRequest,
    *,
    output_dir: str | Path,
    config_base_dir: str | Path,
    executor: ValidationStepExecutor | None = None,
) -> ValidationCoordinatorPreparation:
    """Freeze source, policy, inventory, and adapters without selecting or calling one."""

    root = Path(output_dir).expanduser().resolve()
    base_dir = Path(config_base_dir).expanduser().resolve()
    decision_request = _coordinator_request(request, output_dir=root)
    capability_ids = tuple(_CAPABILITY_IDS[name] for name in V1_TEMPLATE_NAMES)
    requested_required = request.metadata.get("validation_required_capability_ids", ())
    if not isinstance(requested_required, list | tuple):
        raise ValidationWorkflowError(
            "validation_required_capability_ids must be a list when provided"
        )
    required_capability_ids = tuple(str(item) for item in requested_required)
    if not set(required_capability_ids).issubset(capability_ids):
        raise ValidationWorkflowError(
            "validation_required_capability_ids contains an unknown capability"
        )
    if _CAPABILITY_IDS["look_right"] in required_capability_ids:
        raise ValidationWorkflowError(
            "validation.look_right is advisory and cannot be a required capability"
        )
    executor_impl = executor or ScaffoldValidationStepExecutor(base_dir)
    adapter_request = decision_request.model_copy(
        update={"requested_templates": tuple(V1_TEMPLATE_NAMES)}
    )
    adapter_identity = _workflow_identity(
        adapter_request,
        config_base_dir=base_dir,
        executor=executor_impl,
    )
    _validate_output_location(root, adapter_identity)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValidationWorkflowError(
            f"Validation coordinator output already exists; use a fresh output directory: {root}"
        ) from exc
    task_identity_sha256 = canonical_json_digest(
        {"task_description": decision_request.task_description}
    )
    request_identity_sha256 = canonical_json_digest(decision_request)
    config_identity_sha256 = canonical_json_digest(
        cast(
            dict[str, JsonValue],
            {
                "policy_digest": adapter_identity.policy_digest,
                "backend_digest": adapter_identity.backend_digest,
                "template_versions": adapter_identity.template_versions,
            },
        )
    )
    approved_capabilities = _capabilities()
    evidence_inventory = _inventory(
        adapter_identity,
        request=decision_request,
        config_base_dir=base_dir,
    )
    mandatory_constraints = ValidationCoordinatorConstraints(
        allowed_capability_ids=capability_ids,
        required_capability_ids=required_capability_ids,
    )
    unsealed = ValidationCoordinatorPreparation.model_construct(
        output_dir=str(root),
        config_base_dir=str(base_dir),
        request=decision_request,
        task_identity_sha256=task_identity_sha256,
        request_identity_sha256=request_identity_sha256,
        config_identity_sha256=config_identity_sha256,
        adapter_identity=adapter_identity,
        source_dependency_closure=adapter_identity.source_artifacts,
        approved_capabilities=approved_capabilities,
        evidence_inventory=evidence_inventory,
        mandatory_constraints=mandatory_constraints,
        preparation_digest="0" * 64,
    )
    preparation = ValidationCoordinatorPreparation(
        output_dir=str(root),
        config_base_dir=str(base_dir),
        request=decision_request,
        task_identity_sha256=task_identity_sha256,
        request_identity_sha256=request_identity_sha256,
        config_identity_sha256=config_identity_sha256,
        adapter_identity=adapter_identity,
        source_dependency_closure=adapter_identity.source_artifacts,
        approved_capabilities=approved_capabilities,
        evidence_inventory=evidence_inventory,
        mandatory_constraints=mandatory_constraints,
        preparation_digest=canonical_json_digest(
            unsealed.model_dump(mode="json", exclude={"preparation_digest"})
        ),
    )
    ensure_no_inline_secrets(
        preparation.model_dump(mode="json"),
        context="Validation coordinator preparation",
    )
    atomic_write_json(root / VALIDATION_COORDINATOR_PREPARATION_NAME, preparation)
    return preparation


def load_validation_coordinator_preparation(
    output_dir: str | Path,
    *,
    executor: ValidationStepExecutor | None = None,
) -> ValidationCoordinatorPreparation:
    """Load and revalidate a preparation against current source and adapter identity."""

    root = Path(output_dir).expanduser().resolve()
    path = root / VALIDATION_COORDINATOR_PREPARATION_NAME
    if path.is_symlink() or not path.is_file():
        raise ValidationWorkflowError(
            f"Validation coordinator preparation is missing or unsafe: {path}"
        )
    try:
        preparation = ValidationCoordinatorPreparation.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid Validation coordinator preparation {path}: {exc}"
        ) from exc
    if Path(preparation.output_dir) != root:
        raise ValidationWorkflowError(
            "Validation coordinator preparation belongs to another run"
        )
    base_dir = Path(preparation.config_base_dir)
    executor_impl = executor or ScaffoldValidationStepExecutor(base_dir)
    current_identity = _workflow_identity(
        preparation.request.model_copy(
            update={"requested_templates": tuple(V1_TEMPLATE_NAMES)}
        ),
        config_base_dir=base_dir,
        executor=executor_impl,
    )
    if current_identity != preparation.adapter_identity:
        raise ValidationWorkflowError(
            "Validation coordinator source, dependency closure, policy, config, or adapter changed"
        )
    for item in preparation.evidence_inventory:
        current_artifact = _artifact_identity(
            item.artifact.path,
            role=item.artifact.role,
            base_dir=base_dir,
        )
        if current_artifact != item.artifact:
            raise ValidationWorkflowError(
                f"Validation coordinator evidence inventory changed: {item.evidence_id}"
            )
    return preparation


def _validate_parameters(
    check: ValidationCoordinatorSelectedCheck,
    *,
    preparation: ValidationCoordinatorPreparation,
) -> None:
    allowed = set(_PARAMETER_FIELDS[check.template_name])
    unknown = set(check.parameters) - allowed
    if unknown:
        raise ValidationWorkflowError(
            f"Validation check {check.check_id!r} has unknown parameters: "
            + ", ".join(sorted(unknown))
        )
    parameters = check.parameters
    if check.template_name == "physical_behavior":
        raw_evidence_paths = parameters.get("evidence_paths", [])
        if not isinstance(raw_evidence_paths, list) or any(
            not isinstance(item, str) for item in raw_evidence_paths
        ):
            raise ValidationWorkflowError(
                "physical_behavior evidence_paths must be a string list"
            )
        evidence_paths = tuple(
            item for item in raw_evidence_paths if isinstance(item, str)
        )
        declared_runtime = {
            item.artifact.path: item.availability
            for item in preparation.evidence_inventory
            if item.evidence_kind == "runtime"
        }
        unknown_paths = tuple(
            path for path in evidence_paths if path not in declared_runtime
        )
        if unknown_paths:
            raise ValidationWorkflowError(
                "physical_behavior evidence_paths are not in the frozen inventory"
            )
        if check.required and (
            not evidence_paths
            or any(declared_runtime[path] != "available" for path in evidence_paths)
        ):
            raise ValidationWorkflowError(
                "required physical_behavior needs available frozen runtime evidence"
            )


def _topological_check_ids(
    checks: tuple[ValidationCoordinatorSelectedCheck, ...],
) -> tuple[str, ...]:
    by_id = {check.check_id: check for check in checks}
    for check in checks:
        undeclared = tuple(item for item in check.depends_on if item not in by_id)
        if undeclared:
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} has undeclared dependencies: "
                + ", ".join(undeclared)
            )
        if check.check_id in check.depends_on:
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} depends on itself"
            )
    ordered: list[str] = []
    pending = list(checks)
    while pending:
        ready = [check for check in pending if set(check.depends_on).issubset(ordered)]
        if not ready:
            raise ValidationWorkflowError(
                "Validation coordinator plan contains a cycle"
            )
        for check in ready:
            ordered.append(check.check_id)
            pending.remove(check)
    return tuple(ordered)


def _validate_plan_patch(
    preparation: ValidationCoordinatorPreparation,
    patch: ValidationCoordinatorPlanPatch,
) -> tuple[str, ...]:
    if patch.preparation_digest != preparation.preparation_digest:
        raise ValidationWorkflowError("Validation coordinator plan patch is stale")
    constraints = preparation.mandatory_constraints
    if len(patch.selected_checks) > constraints.maximum_selected_checks:
        raise ValidationWorkflowError(
            "Validation coordinator plan selects too many checks"
        )
    capabilities = {
        capability.capability_id: capability
        for capability in preparation.approved_capabilities
    }
    source_targets = tuple(preparation.request.inputs)
    declared_focus = set(preparation.request.focus.prim_paths)
    for check in patch.selected_checks:
        capability = capabilities.get(check.capability_id)
        if (
            capability is None
            or check.capability_id not in constraints.allowed_capability_ids
        ):
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} selects an unknown capability"
            )
        if (
            check.template_name != capability.template_name
            or check.rule_id != capability.rule_id
        ):
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} changes its adapter identity"
            )
        if check.targets != source_targets:
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} must preserve the exact "
                "top-level request inputs"
            )
        if not set(check.focus_prim_paths).issubset(declared_focus):
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} has undeclared focus prims"
            )
        if declared_focus and set(check.focus_prim_paths) != declared_focus:
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} must preserve all requested focus prims"
            )
        if check.evidence_requirements != capability.evidence_requirements:
            raise ValidationWorkflowError(
                f"Validation check {check.check_id!r} changes its evidence requirements"
            )
        if check.template_name == "look_right" and check.required:
            raise ValidationWorkflowError(
                "look_right is advisory and cannot be required"
            )
        _validate_parameters(check, preparation=preparation)
    selected_capability_ids = {check.capability_id for check in patch.selected_checks}
    provider_free_capability_ids = set(constraints.provider_free_capability_ids)
    if (
        len(selected_capability_ids & provider_free_capability_ids)
        < constraints.minimum_provider_free_checks
    ):
        raise ValidationWorkflowError(
            "Validation coordinator plan requires at least two distinct "
            "provider-free checks"
        )
    if (
        sum(check.required for check in patch.selected_checks)
        < constraints.minimum_required_checks
    ):
        raise ValidationWorkflowError(
            "Validation coordinator plan requires at least one required check"
        )
    if constraints.explicit_dependency_required and not any(
        check.depends_on for check in patch.selected_checks
    ):
        raise ValidationWorkflowError(
            "Validation coordinator plan requires at least one explicit dependency"
        )
    missing_required = (
        set(constraints.required_capability_ids) - selected_capability_ids
    )
    if missing_required:
        raise ValidationWorkflowError(
            "Validation coordinator plan omitted required capabilities: "
            + ", ".join(sorted(missing_required))
        )
    downgraded_required = {
        check.capability_id
        for check in patch.selected_checks
        if check.capability_id in constraints.required_capability_ids
        and not check.required
    }
    if downgraded_required:
        raise ValidationWorkflowError(
            "Validation coordinator plan made required capabilities advisory: "
            + ", ".join(sorted(downgraded_required))
        )
    ordered = _topological_check_ids(patch.selected_checks)
    by_template = {check.template_name: check for check in patch.selected_checks}
    critique = by_template.get("look_right")
    render = by_template.get("render_valid")
    if critique is not None and (
        render is None or render.check_id not in critique.depends_on
    ):
        raise ValidationWorkflowError(
            "look_right requires an explicit render_valid dependency"
        )
    return ordered


def _effective_accepted_request(
    preparation: ValidationCoordinatorPreparation,
    ordered_checks: tuple[ValidationCoordinatorSelectedCheck, ...],
) -> ValidationRequest:
    focus_prim_paths = tuple(
        dict.fromkeys(
            path for check in ordered_checks for path in check.focus_prim_paths
        )
    )
    return preparation.request.model_copy(
        deep=True,
        update={
            "requested_templates": tuple(
                check.template_name for check in ordered_checks
            ),
            "focus": preparation.request.focus.model_copy(
                update={"prim_paths": focus_prim_paths}
            ),
            "render": preparation.request.render,
        },
    )


def accept_validation_coordinator_plan(
    output_dir: str | Path,
    *,
    plan_patch_path: str | Path,
    child_output_path: str | Path,
    producer: Literal["codex", "claude"],
    expected_child_launch_descriptor_digest: str,
    child_session_id: str | None = None,
    child_plan_id: str | None = None,
    executor: ValidationStepExecutor | None = None,
) -> ValidationCoordinatorAcceptedPlan:
    """Validate one child decision and only then publish its adapter projection."""

    preparation = load_validation_coordinator_preparation(output_dir, executor=executor)
    root = Path(preparation.output_dir)
    patch_path = Path(plan_patch_path).expanduser().resolve()
    if patch_path != root / VALIDATION_COORDINATOR_PLAN_PATCH_NAME:
        raise ValidationWorkflowError(
            "Validation child plan patch must use the canonical run-local path"
        )
    if patch_path.is_symlink() or not patch_path.is_file():
        raise ValidationWorkflowError(
            "Validation child plan patch must be a regular run-local file"
        )
    try:
        patch = ValidationCoordinatorPlanPatch.model_validate(load_json(patch_path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid Validation coordinator plan patch: {exc}"
        ) from exc
    if patch.producer != producer:
        raise ValidationWorkflowError(
            "Validation child producer differs from the launcher"
        )
    if child_session_id is not None and patch.child_session_id != child_session_id:
        raise ValidationWorkflowError(
            "Validation child session differs from the launcher"
        )
    if child_plan_id is not None and patch.plan_id != child_plan_id:
        raise ValidationWorkflowError(
            "Validation child final response changed its plan ID"
        )
    child_output = _binding(child_output_path)
    try:
        Path(child_output.path).relative_to(root / "raw")
    except ValueError as exc:
        raise ValidationWorkflowError(
            "Validation child output must stay under run/raw"
        ) from exc
    child_launch_descriptor = _bind_validation_child_launch_descriptor(
        preparation,
        child_output=child_output,
        producer=producer,
        expected_descriptor_digest=expected_child_launch_descriptor_digest,
    )
    forbidden = tuple(
        name
        for name in EXECUTED_VALIDATION_ARTIFACT_NAMES
        if (root / name).exists() or (root / name).is_symlink()
    )
    if forbidden:
        raise ValidationWorkflowError(
            "Validation child crossed the decision-only boundary; found "
            + ", ".join(forbidden)
        )
    ordered_ids = _validate_plan_patch(preparation, patch)
    by_id = {check.check_id: check for check in patch.selected_checks}
    ordered_checks = tuple(by_id[check_id] for check_id in ordered_ids)
    effective_request = _effective_accepted_request(preparation, ordered_checks)
    _prepare_validation_operations_from_coordinator(
        effective_request,
        output_dir=root,
        config_base_dir=preparation.config_base_dir,
        check_ids=ordered_ids,
        dependencies={check.check_id: check.depends_on for check in ordered_checks},
        mandatory={check.template_name: check.required for check in ordered_checks},
        check_metadata={
            check.check_id: {
                "capability_id": check.capability_id,
                "rule_id": check.rule_id,
                "targets": list(check.targets),
                "focus_prim_paths": list(check.focus_prim_paths),
                "parameters": check.parameters,
                "required": check.required,
                "evidence_requirements": list(check.evidence_requirements),
                "completion_policy": check.completion_policy,
            }
            for check in ordered_checks
        },
        required_capability_ids=(
            preparation.mandatory_constraints.required_capability_ids
        ),
        executor=executor,
    )
    accepted_plan_digest = canonical_json_digest(
        {
            "plan_id": patch.plan_id,
            "preparation_digest": preparation.preparation_digest,
            "selected_checks": [
                check.model_dump(mode="json") for check in ordered_checks
            ],
            "ordered_check_ids": list(ordered_ids),
        }
    )
    preparation_binding = _binding(root / VALIDATION_COORDINATOR_PREPARATION_NAME)
    patch_binding = _binding(patch_path)
    operation_preparation_binding = _binding(
        root / VALIDATION_OPERATION_PREPARATION_NAME
    )
    compatibility_request = _binding(root / "validation_request.json")
    compatibility_plan = _binding(root / "validation_plan.json")
    accepted_at = datetime.now(UTC)
    unsealed = ValidationCoordinatorAcceptedPlan.model_construct(
        plan_id=patch.plan_id,
        preparation_digest=preparation.preparation_digest,
        preparation=preparation_binding,
        plan_patch=patch_binding,
        child_output=child_output,
        child_launch_descriptor=child_launch_descriptor,
        producer=producer,
        child_session_id=patch.child_session_id,
        selected_checks=ordered_checks,
        ordered_check_ids=ordered_ids,
        operation_preparation=operation_preparation_binding,
        compatibility_request=compatibility_request,
        compatibility_plan=compatibility_plan,
        accepted_plan_digest=accepted_plan_digest,
        accepted_at=accepted_at,
        receipt_digest="0" * 64,
    )
    receipt = ValidationCoordinatorAcceptedPlan(
        plan_id=patch.plan_id,
        preparation_digest=preparation.preparation_digest,
        preparation=preparation_binding,
        plan_patch=patch_binding,
        child_output=child_output,
        child_launch_descriptor=child_launch_descriptor,
        producer=producer,
        child_session_id=patch.child_session_id,
        selected_checks=ordered_checks,
        ordered_check_ids=ordered_ids,
        operation_preparation=operation_preparation_binding,
        compatibility_request=compatibility_request,
        compatibility_plan=compatibility_plan,
        accepted_plan_digest=accepted_plan_digest,
        accepted_at=accepted_at,
        receipt_digest=canonical_json_digest(
            unsealed.model_dump(mode="json", exclude={"receipt_digest"})
        ),
    )
    ensure_no_inline_secrets(
        receipt.model_dump(mode="json"),
        context="accepted Validation coordinator plan",
    )
    atomic_write_json(root / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME, receipt)
    return receipt


def load_validation_coordinator_accepted_plan(
    output_dir: str | Path,
    *,
    executor: ValidationStepExecutor | None = None,
) -> ValidationCoordinatorAcceptedPlan:
    root = Path(output_dir).expanduser().resolve()
    preparation = load_validation_coordinator_preparation(root, executor=executor)
    path = root / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME
    if path.is_symlink() or not path.is_file():
        raise ValidationWorkflowError(
            f"Accepted Validation coordinator plan is missing or unsafe: {path}"
        )
    try:
        receipt = ValidationCoordinatorAcceptedPlan.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid accepted Validation coordinator plan: {exc}"
        ) from exc
    if receipt.preparation_digest != preparation.preparation_digest:
        raise ValidationWorkflowError("Accepted Validation coordinator plan is stale")
    for binding in (
        receipt.preparation,
        receipt.plan_patch,
        receipt.child_output,
        receipt.child_launch_descriptor,
        receipt.operation_preparation,
        receipt.compatibility_request,
        receipt.compatibility_plan,
    ):
        _verify_binding(binding)
    return receipt


def execute_validation_coordinator_plan(
    output_dir: str | Path,
    *,
    executor: ValidationStepExecutor | None = None,
) -> tuple[ValidationWorkflowRun, ValidationCoordinatorExecutionReceipt]:
    """Execute an accepted plan through exact operation adapters in dependency order."""

    root = Path(output_dir).expanduser().resolve()
    accepted = load_validation_coordinator_accepted_plan(root, executor=executor)
    result_paths: dict[str, Path] = {}
    operations: list[ValidationOperationResult] = []
    for check in accepted.selected_checks:
        prior_paths = tuple(result_paths[check_id] for check_id in check.depends_on)
        operation = run_validation_operation(
            root,
            template_name=check.template_name,
            prior_result_paths=prior_paths,
            executor=executor,
        )
        operation_path = (
            root / "operations" / check.template_name / "operation_result.json"
        )
        result_paths[check.check_id] = operation_path
        operations.append(operation)
    run = finalize_validation_operations(root)
    preparation = load_validation_coordinator_preparation(root, executor=executor)
    source_after = _workflow_identity(
        preparation.request.model_copy(
            update={"requested_templates": tuple(V1_TEMPLATE_NAMES)}
        ),
        config_base_dir=Path(preparation.config_base_dir),
        executor=executor
        or ScaffoldValidationStepExecutor(Path(preparation.config_base_dir)),
    ).source_artifacts
    required_success = all(
        operation.template_result.passed
        for check, operation in zip(accepted.selected_checks, operations, strict=True)
        if check.required
    )
    execution_disposition: Literal["completed", "required_checks_failed"] = (
        "completed" if required_success else "required_checks_failed"
    )
    accepted_plan_binding = _binding(root / VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME)
    operation_results = tuple(_binding(path) for path in result_paths.values())
    operation_index = _binding(root / VALIDATION_OPERATION_INDEX_NAME)
    validation_result = _binding(run.result_path)
    validation_evidence = _binding(run.evidence_path)
    final_summary = _binding(run.final_summary_path)
    completed_at = datetime.now(UTC)
    unsealed = ValidationCoordinatorExecutionReceipt.model_construct(
        accepted_plan=accepted_plan_binding,
        accepted_plan_digest=accepted.accepted_plan_digest,
        operation_results=operation_results,
        operation_index=operation_index,
        validation_result=validation_result,
        validation_evidence=validation_evidence,
        final_summary=final_summary,
        source_before=preparation.source_dependency_closure,
        source_after=source_after,
        execution_disposition=execution_disposition,
        required_checks_successful=required_success,
        completed_at=completed_at,
        receipt_digest="0" * 64,
    )
    receipt = ValidationCoordinatorExecutionReceipt(
        accepted_plan=accepted_plan_binding,
        accepted_plan_digest=accepted.accepted_plan_digest,
        operation_results=operation_results,
        operation_index=operation_index,
        validation_result=validation_result,
        validation_evidence=validation_evidence,
        final_summary=final_summary,
        source_before=preparation.source_dependency_closure,
        source_after=source_after,
        execution_disposition=execution_disposition,
        required_checks_successful=required_success,
        completed_at=completed_at,
        receipt_digest=canonical_json_digest(
            unsealed.model_dump(mode="json", exclude={"receipt_digest"})
        ),
    )
    atomic_write_json(root / VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME, receipt)
    return run, receipt


__all__ = [
    "VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME",
    "VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME",
    "VALIDATION_COORDINATOR_PLAN_PATCH_NAME",
    "VALIDATION_COORDINATOR_PREPARATION_NAME",
    "ValidationCoordinatorAcceptedPlan",
    "ValidationCoordinatorCapability",
    "ValidationCoordinatorClaim",
    "ValidationCoordinatorConstraints",
    "ValidationCoordinatorEvidenceInventoryItem",
    "ValidationCoordinatorExecutionReceipt",
    "ValidationCoordinatorPlanPatch",
    "ValidationCoordinatorPreparation",
    "ValidationCoordinatorSelectedCheck",
    "accept_validation_coordinator_plan",
    "execute_validation_coordinator_plan",
    "load_validation_coordinator_accepted_plan",
    "load_validation_coordinator_preparation",
    "prepare_validation_coordinator",
]
