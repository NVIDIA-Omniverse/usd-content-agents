# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Validation operations selected directly by an outer reasoner.

This module deliberately does not select tasks or run an orchestration loop.
The caller freezes an explicit ``requested_templates`` list, invokes one named
operation at a time, and asks the deterministic finalizer to publish the
standard Validation Agent bundle after all mandatory selected operations exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from world_understanding.utils.credentials import ensure_no_inline_secrets
from world_understanding.validation import (
    ValidationIssue,
    ValidationPlan,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateContext,
    ValidationTemplateResult,
    aggregate_validation_verdict,
)
from world_understanding.validation.scaffold_runner import (
    ScaffoldValidationStepExecutor,
)
from world_understanding.validation.templates import V1_TEMPLATE_NAMES

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)

from .finalizer import (
    ValidationWorkflowPaths,
    finalize_validation_workflow,
    write_validation_planning_artifacts,
)
from .models import (
    ValidationAcceptedTemplateResult,
    ValidationArtifactIdentity,
    ValidationWorkflowCheckpoint,
    ValidationWorkflowIdentity,
    ValidationWorkflowRun,
    ValidationWorkflowStatus,
    ValidationWorkItemRecord,
    ValidationWorkItemState,
)
from .workflow import (
    VALIDATION_TEMPLATE_CAPABILITIES,
    ValidationStepExecutor,
    ValidationWorkflowError,
    _artifact_identity,
    _best_effort_artifact_identities,
    _bind_plan,
    _canonicalize_result_artifact_paths,
    _evidence_artifacts,
    _missing_declared_evidence_result,
    _plan_digest,
    _records_from_plan,
    _validate_render_evidence_result,
    _work_item_identity,
    _workflow_identity,
)

VALIDATION_OPERATION_PREPARATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-operation-preparation.v1"
)
VALIDATION_OPERATION_RESULT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-operation-result.v1"
)
VALIDATION_OPERATION_INDEX_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-operation-index.v1"
)

VALIDATION_OPERATION_PREPARATION_NAME: Final = "validation_operation_preparation.json"
VALIDATION_OPERATION_INDEX_NAME: Final = "validation_operation_index.json"
PROVIDED_OPERATION_ARTIFACT_NAMES: Final = (
    "verified_operation_ingest_index.json",
    "verified_operation_evidence_index.json",
    "verified_operation_execution_index.json",
    "canonical_verified_operation_assessment.json",
    "verified_operations",
)
EXECUTED_VALIDATION_ARTIFACT_NAMES: Final = (
    VALIDATION_OPERATION_PREPARATION_NAME,
    VALIDATION_OPERATION_INDEX_NAME,
    "operations",
    "validation_request.json",
    "validation_plan.json",
    "validation_result.json",
    "validation_checkpoint.json",
    "validation_evidence.json",
    "final_summary.json",
    "attempts",
    "standalone_validation_evidence.json",
    "standalone_validation_execution.json",
    "embedded_validation_evidence.json",
    "embedded_validation_execution.json",
    "embedded_validation_receipt.json",
    "canonical_validation_assessment.json",
)

_PHYSICAL_BEHAVIOR_REFINEMENT_OUTPUT_DIR_POLICY_KEYS: Final = (
    "physical_behavior_refine_output_dir",
    "physics_refine_output_dir",
    "refine_output_dir",
)

_PHYSICAL_BEHAVIOR_REFINEMENT_SUMMARY_POLICY_KEYS: Final = (
    "physical_behavior_refine_summary_path",
    "refine_summary_path",
)

_PHYSICAL_BEHAVIOR_REFINEMENT_POLICY_KEYS: Final = (
    *_PHYSICAL_BEHAVIOR_REFINEMENT_OUTPUT_DIR_POLICY_KEYS,
    *_PHYSICAL_BEHAVIOR_REFINEMENT_SUMMARY_POLICY_KEYS,
)

_PHYSICAL_BEHAVIOR_TYPED_PATH_POLICY: Final[Mapping[str, tuple[str, str]]] = {
    "animation_usd_paths": ("animation_usd", "animation_usd"),
    "behavior_video_paths": ("video", "video"),
    "sampled_video_frame_paths": ("sampled_frame", "sampled_video_frame"),
    "simulation_json_paths": ("simulation_json", "simulation_json"),
    "time_sampled_usd_paths": ("time_sampled_usd", "time_sampled_usd"),
    "trajectory_metrics_paths": ("trajectory_metrics", "trajectory_metrics"),
    "video_paths": ("video", "video"),
}

_PHYSICAL_BEHAVIOR_EVIDENCE_POLICY_KEYS: Final = (
    "physical_behavior_evidence",
    "behavior_evidence",
    *_PHYSICAL_BEHAVIOR_TYPED_PATH_POLICY,
)


def _physical_behavior_refinement_records_by_path(
    policy: Mapping[str, object],
    *,
    base_dir: Path,
) -> dict[str, tuple[str, object]]:
    records: dict[str, tuple[str, object]] = {}
    for key in _PHYSICAL_BEHAVIOR_REFINEMENT_POLICY_KEYS:
        value = policy.get(key)
        values = value if isinstance(value, list | tuple) else (value,)
        for item in values:
            if not isinstance(item, str | Path):
                continue
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            records.setdefault(str(path.resolve(strict=False)), (key, item))
    return records


def _physical_behavior_evidence_records_by_path(
    policy: Mapping[str, object],
    *,
    base_dir: Path,
) -> dict[str, object]:
    """Index original typed behavior-evidence records by resolved path."""

    records: dict[str, object] = {}
    for key in _PHYSICAL_BEHAVIOR_EVIDENCE_POLICY_KEYS:
        value = policy.get(key)
        values = value if isinstance(value, list | tuple) else (value,)
        for item in values:
            path: object
            if isinstance(item, Mapping):
                path = item.get("path")
            else:
                path = item
            if not isinstance(path, str | Path):
                continue
            candidate = Path(path).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            record: object = dict(item) if isinstance(item, Mapping) else item
            typed_path = _PHYSICAL_BEHAVIOR_TYPED_PATH_POLICY.get(key)
            if typed_path is not None:
                kind, role = typed_path
                record = {"path": path, "kind": kind, "role": role}
            records.setdefault(
                str(candidate.resolve(strict=False)),
                record,
            )
    return records


ValidationOperationFinalState = Literal[
    "not_requested",
    "evaluated",
    "not_evaluated",
]
ValidationOperationRole = Literal[
    "deterministic_check",
    "render_evidence_leaf",
    "runtime_evidence_consumer",
    "optional_advisory_critique",
]
ValidationOperationExpansionKind = Literal[
    "requested_templates",
    "requested_rules",
    "named_profile",
    "coordinator_plan_patch",
]

# These are stable public request shorthands, not an inference or routing layer.
# Each rule currently maps one-to-one to its owning V1 template so the result
# stays on the #38 ValidationResult/ValidationIssue contract.
VALIDATION_OPERATION_RULES: Final[Mapping[str, str]] = {
    "render.runtime_evidence": "render_valid",
    "physics.usd_schema_sanity": "physics_sane",
    "physics.behavior_evidence": "physical_behavior",
    "visual.optional_advisory_critique": "look_right",
}
VALIDATION_OPERATION_PROFILES: Final[Mapping[str, tuple[str, ...]]] = {
    "static": ("physics_sane",),
    "visual": ("render_valid",),
    "runtime": ("physical_behavior",),
    "comprehensive": (
        "physics_sane",
        "render_valid",
        "physical_behavior",
    ),
    "comprehensive-with-advisory-critique": (
        "physics_sane",
        "render_valid",
        "physical_behavior",
        "look_right",
    ),
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationOperationCapability(_FrozenModel):
    """One caller-selected or explicitly unselected Validation capability."""

    template_name: str = Field(min_length=1)
    role: ValidationOperationRole
    selection_state: Literal["requested", "not_requested"]
    mandatory_when_requested: bool
    required_capabilities: tuple[str, ...] = ()
    preflight_requirements: tuple[str, ...] = ()


class ValidationOperationExpansion(_FrozenModel):
    """Readback of one explicitly outer-selected request expansion."""

    kind: ValidationOperationExpansionKind
    requested_values: tuple[str, ...] = Field(min_length=1)
    expanded_templates: tuple[str, ...] = Field(min_length=1)


class ValidationOperationPreparation(_FrozenModel):
    """Frozen provider-free request, identity, and explicit operation plan."""

    schema_version: Literal[
        "content-agent-workflows.validation-operation-preparation.v1"
    ] = VALIDATION_OPERATION_PREPARATION_SCHEMA_VERSION
    execution_mode: Literal["execute"] = "execute"
    output_dir: str = Field(min_length=1)
    config_base_dir: str = Field(min_length=1)
    request: ValidationRequest
    plan: ValidationPlan
    workflow_identity: ValidationWorkflowIdentity
    expansion: ValidationOperationExpansion
    capabilities: tuple[ValidationOperationCapability, ...]
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_explicit_selection(self) -> Self:
        if self.expansion.expanded_templates != self.request.requested_templates:
            raise ValueError("request expansion must match requested_templates")
        requested = tuple(
            capability.template_name
            for capability in self.capabilities
            if capability.selection_state == "requested"
        )
        if requested != self.request.requested_templates:
            raise ValueError("capability selection must match requested_templates")
        if tuple(step.template_name for step in self.plan.steps) != requested:
            raise ValueError("prepared plan must preserve explicit operation order")
        capability_names = tuple(item.template_name for item in self.capabilities)
        expected_names = (
            *self.request.requested_templates,
            *(
                candidate
                for candidate in V1_TEMPLATE_NAMES
                if candidate not in self.request.requested_templates
            ),
        )
        if capability_names != expected_names:
            raise ValueError("preparation must classify every V1 template exactly once")
        for capability in self.capabilities:
            if capability.role != _operation_role(capability.template_name):
                raise ValueError("capability role differs from its template contract")
            if (
                capability.template_name == "look_right"
                and capability.mandatory_when_requested
            ):
                raise ValueError("look_right is advisory and cannot be mandatory")
        return self


class ValidationOperationResult(_FrozenModel):
    """One bounded template execution and its immutable evidence bindings."""

    schema_version: Literal[
        "content-agent-workflows.validation-operation-result.v1"
    ] = VALIDATION_OPERATION_RESULT_SCHEMA_VERSION
    execution_mode: Literal["execute"] = "execute"
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_name: str = Field(min_length=1)
    role: ValidationOperationRole
    mandatory: bool
    template_result: ValidationTemplateResult
    template_result_path: str = Field(min_length=1)
    template_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_artifacts: tuple[ValidationArtifactIdentity, ...] = ()
    source_before: tuple[ValidationArtifactIdentity, ...]
    source_after: tuple[ValidationArtifactIdentity, ...]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.template_result.template_name != self.template_name:
            raise ValueError("operation result template name differs")
        if self.role != _operation_role(self.template_name):
            raise ValueError("operation result role differs from its template contract")
        if self.template_name == "look_right" and self.mandatory:
            raise ValueError("look_right is advisory and cannot be mandatory")
        if self.source_before != self.source_after:
            raise ValueError("Validation operation changed its source identity")
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("completed_at must include a timezone")
        return self


class ValidationOperationStatus(_FrozenModel):
    """Explicit final state for one stable Validation Agent template."""

    template_name: str = Field(min_length=1)
    role: ValidationOperationRole
    state: ValidationOperationFinalState
    mandatory: bool
    result_path: str | None = None
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.state == "evaluated" and (
            self.result_path is None or self.result_sha256 is None
        ):
            raise ValueError("evaluated operation requires a result binding")
        if self.state != "evaluated" and (
            self.result_path is not None or self.result_sha256 is not None
        ):
            raise ValueError("unevaluated operation cannot claim a result binding")
        return self


class ValidationOperationIndex(_FrozenModel):
    """Selection/readback receipt; omitted leaves never appear as passes."""

    schema_version: Literal["content-agent-workflows.validation-operation-index.v1"] = (
        VALIDATION_OPERATION_INDEX_SCHEMA_VERSION
    )
    execution_mode: Literal["execute"] = "execute"
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    operations: tuple[ValidationOperationStatus, ...]
    required_operations_complete: bool
    optional_critique_evaluated: bool
    nested_agent_launched: Literal[False] = False

    @model_validator(mode="after")
    def validate_statuses(self) -> Self:
        if tuple(item.template_name for item in self.operations) != tuple(
            dict.fromkeys(item.template_name for item in self.operations)
        ):
            raise ValueError("operation index contains duplicate templates")
        if set(item.template_name for item in self.operations) != set(
            V1_TEMPLATE_NAMES
        ):
            raise ValueError("operation index must classify every V1 template")
        for item in self.operations:
            if item.role != _operation_role(item.template_name):
                raise ValueError("operation index role differs from template contract")
            if item.state == "not_requested" and item.mandatory:
                raise ValueError("unselected operations cannot be mandatory")
            if item.template_name == "look_right" and item.mandatory:
                raise ValueError("look_right is advisory and cannot be mandatory")
        required_complete = not any(
            item.mandatory and item.state != "evaluated" for item in self.operations
        )
        if self.required_operations_complete != required_complete:
            raise ValueError("required operation completion summary is inconsistent")
        critique_evaluated = any(
            item.template_name == "look_right" and item.state == "evaluated"
            for item in self.operations
        )
        if self.optional_critique_evaluated != critique_evaluated:
            raise ValueError("optional critique summary is inconsistent")
        return self


def _operation_role(template_name: str) -> ValidationOperationRole:
    roles: Mapping[str, ValidationOperationRole] = {
        "render_valid": "render_evidence_leaf",
        "look_right": "optional_advisory_critique",
        "physics_sane": "deterministic_check",
        "physical_behavior": "runtime_evidence_consumer",
    }
    try:
        return roles[template_name]
    except KeyError as exc:
        raise ValueError(f"unknown Validation template: {template_name!r}") from exc


def _operation_capability(
    template_name: str,
    *,
    requested: bool,
    mandatory_when_requested: bool | None = None,
) -> ValidationOperationCapability:
    requirements = {
        "render_valid": (
            "Existing image/render evidence, or an explicitly configured remote "
            "or local OVRTX renderer.",
        ),
        "look_right": (
            "Current render evidence and any requested references.",
            "An explicitly configured VLM/LLM provider only when advisory "
            "critique is requested.",
        ),
        "physics_sane": ("A local USD schema runtime; no model provider.",),
        "physical_behavior": (
            "Existing simulator/runtime/recording evidence; this operation does "
            "not launch a simulator.",
        ),
    }[template_name]
    capabilities = VALIDATION_TEMPLATE_CAPABILITIES[template_name]
    return ValidationOperationCapability(
        template_name=template_name,
        role=_operation_role(template_name),
        selection_state="requested" if requested else "not_requested",
        mandatory_when_requested=(
            template_name != "look_right"
            if mandatory_when_requested is None
            else mandatory_when_requested
        ),
        required_capabilities=capabilities,
        preflight_requirements=requirements,
    )


def _effective_operation_request(
    request: ValidationRequest,
    *,
    output_dir: Path,
    requested_rules: tuple[str, ...],
    named_profile: str | None,
) -> tuple[ValidationRequest, ValidationOperationExpansion]:
    sources = sum(
        (
            bool(request.requested_templates),
            bool(requested_rules),
            named_profile is not None,
        )
    )
    if sources != 1:
        raise ValidationWorkflowError(
            "Focused Validation preparation requires exactly one explicit outer "
            "choice: requested_templates, requested_rules, or named_profile."
        )
    if requested_rules:
        unknown_rules = tuple(
            name for name in requested_rules if name not in VALIDATION_OPERATION_RULES
        )
        if unknown_rules:
            raise ValidationWorkflowError(
                "Unknown focused Validation rules: " + ", ".join(unknown_rules)
            )
        requested = tuple(
            dict.fromkeys(VALIDATION_OPERATION_RULES[name] for name in requested_rules)
        )
        expansion = ValidationOperationExpansion(
            kind="requested_rules",
            requested_values=requested_rules,
            expanded_templates=requested,
        )
    elif named_profile is not None:
        if named_profile not in VALIDATION_OPERATION_PROFILES:
            raise ValidationWorkflowError(
                "Unknown focused Validation named profile: " + named_profile
            )
        requested = VALIDATION_OPERATION_PROFILES[named_profile]
        expansion = ValidationOperationExpansion(
            kind="named_profile",
            requested_values=(named_profile,),
            expanded_templates=requested,
        )
    else:
        requested = request.requested_templates
        expansion = ValidationOperationExpansion(
            kind="requested_templates",
            requested_values=requested,
            expanded_templates=requested,
        )
    if len(requested) != len(set(requested)):
        raise ValidationWorkflowError("explicit Validation choices must not duplicate")
    unknown = tuple(name for name in requested if name not in V1_TEMPLATE_NAMES)
    if unknown:
        raise ValidationWorkflowError(
            "Unknown focused Validation operations: " + ", ".join(unknown)
        )
    if "look_right" in requested and (
        "render_valid" not in requested
        or requested.index("render_valid") > requested.index("look_right")
    ):
        raise ValidationWorkflowError(
            "Explicit look_right critique requires render_valid earlier in the "
            "same frozen request."
        )
    policy = dict(request.policy)
    policy.setdefault("visual_evidence_mode", "canonical_usd")
    effective = request.model_copy(
        deep=True,
        update={
            "project": request.project.model_copy(
                update={"working_dir": str(output_dir)}
            ),
            "policy": policy,
            "requested_templates": requested,
        },
    )
    return effective, expansion


def prepare_validation_operations(
    request: ValidationRequest,
    *,
    output_dir: str | Path,
    config_base_dir: str | Path,
    requested_rules: tuple[str, ...] = (),
    named_profile: str | None = None,
    executor: ValidationStepExecutor | None = None,
) -> ValidationOperationPreparation:
    """Freeze an explicit plan without invoking a template, provider, or agent."""

    root = Path(output_dir).expanduser().resolve()
    if any(
        (root / name).exists() or (root / name).is_symlink()
        for name in PROVIDED_OPERATION_ARTIFACT_NAMES
    ):
        raise ValidationWorkflowError(
            "Validation execute and provided modes are mutually exclusive"
        )
    root.mkdir(parents=True, exist_ok=False)
    base_dir = Path(config_base_dir).expanduser().resolve()
    effective_request, expansion = _effective_operation_request(
        request,
        output_dir=root,
        requested_rules=requested_rules,
        named_profile=named_profile,
    )
    executor_impl = executor or ScaffoldValidationStepExecutor(base_dir)
    identity = _workflow_identity(
        effective_request,
        config_base_dir=base_dir,
        executor=executor_impl,
    )
    paths = ValidationWorkflowPaths.from_output_dir(root)
    plan = _bind_plan(
        executor_impl.plan(effective_request.model_copy(deep=True), working_dir=root),
        identity=identity,
        artifact_paths=paths.artifact_paths(),
    )
    preparation = ValidationOperationPreparation(
        output_dir=str(root),
        config_base_dir=str(base_dir),
        request=effective_request,
        plan=plan,
        workflow_identity=identity,
        expansion=expansion,
        capabilities=tuple(
            _operation_capability(
                name,
                requested=name in effective_request.requested_templates,
            )
            for name in (
                *effective_request.requested_templates,
                *(
                    candidate
                    for candidate in V1_TEMPLATE_NAMES
                    if candidate not in effective_request.requested_templates
                ),
            )
        ),
    )
    ensure_no_inline_secrets(
        preparation.model_dump(mode="json"),
        context="focused Validation operation preparation",
    )
    write_validation_planning_artifacts(effective_request, plan, paths)
    atomic_write_json(root / VALIDATION_OPERATION_PREPARATION_NAME, preparation)
    return preparation


def _prepare_validation_operations_from_coordinator(
    request: ValidationRequest,
    *,
    output_dir: str | Path,
    config_base_dir: str | Path,
    check_ids: tuple[str, ...],
    dependencies: Mapping[str, tuple[str, ...]],
    mandatory: Mapping[str, bool],
    check_metadata: Mapping[str, Mapping[str, object]],
    required_capability_ids: tuple[str, ...],
    executor: ValidationStepExecutor | None = None,
) -> ValidationOperationPreparation:
    """Publish the exact adapter projection of one accepted coordinator plan.

    Unlike :func:`prepare_validation_operations`, this internal bridge owns no
    selection authority. The caller has already validated a digest-bound child
    plan and supplies its exact topological order and dependency graph.
    """

    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValidationWorkflowError(
            f"Validation coordinator run directory is missing or unsafe: {root}"
        )
    forbidden = (
        VALIDATION_OPERATION_PREPARATION_NAME,
        VALIDATION_OPERATION_INDEX_NAME,
        "operations",
        "validation_request.json",
        "validation_plan.json",
        "validation_result.json",
    )
    present = tuple(
        name
        for name in forbidden
        if (root / name).exists() or (root / name).is_symlink()
    )
    if present:
        raise ValidationWorkflowError(
            "Coordinator plan must be accepted before adapter projection; found "
            + ", ".join(present)
        )
    base_dir = Path(config_base_dir).expanduser().resolve()
    requested = request.requested_templates
    if len(check_ids) != len(requested):
        raise ValidationWorkflowError(
            "Coordinator check IDs must match the accepted template order"
        )
    render_policy_parameter_checks = sorted(
        check_id
        for check_id, template_name in zip(check_ids, requested, strict=True)
        for parameters in (check_metadata.get(check_id, {}).get("parameters"),)
        if template_name == "render_valid"
        and isinstance(parameters, Mapping)
        and parameters
    )
    if render_policy_parameter_checks:
        raise ValidationWorkflowError(
            "Coordinator operation projection cannot author render policy: "
            + ", ".join(render_policy_parameter_checks)
        )
    provider_parameter_checks = sorted(
        check_id
        for check_id in check_ids
        for parameters in (check_metadata.get(check_id, {}).get("parameters"),)
        if isinstance(parameters, Mapping) and "backend" in parameters
    )
    if provider_parameter_checks:
        raise ValidationWorkflowError(
            "Coordinator operation projection cannot select a provider backend: "
            + ", ".join(provider_parameter_checks)
        )
    required = set(required_capability_ids)
    selected_required: set[str] = set()
    downgraded_required: set[str] = set()
    for check_id, template_name in zip(check_ids, requested, strict=True):
        capability_id = check_metadata.get(check_id, {}).get("capability_id")
        if not isinstance(capability_id, str) or capability_id not in required:
            continue
        selected_required.add(capability_id)
        if not mandatory.get(template_name, False):
            downgraded_required.add(capability_id)
    missing_required = required - selected_required
    if missing_required or downgraded_required:
        invalid = sorted(missing_required | downgraded_required)
        raise ValidationWorkflowError(
            "Coordinator operation projection did not preserve required "
            "capabilities: " + ", ".join(invalid)
        )
    executor_impl = executor or ScaffoldValidationStepExecutor(base_dir)
    identity = _workflow_identity(
        request,
        config_base_dir=base_dir,
        executor=executor_impl,
    )
    paths = ValidationWorkflowPaths.from_output_dir(root)
    plan = _bind_plan(
        executor_impl.plan(request.model_copy(deep=True), working_dir=root),
        identity=identity,
        artifact_paths=paths.artifact_paths(),
    )
    template_for_check = dict(zip(check_ids, requested, strict=True))
    accepted_steps = []
    for check_id, step in zip(check_ids, plan.steps, strict=True):
        expected_template = template_for_check[check_id]
        if step.template_name != expected_template:
            raise ValidationWorkflowError(
                "Coordinator operation plan reordered accepted template "
                f"{expected_template!r} for check {check_id!r} as "
                f"{step.template_name!r}"
            )
        dependency_ids = dependencies.get(check_id, ())
        dependency_templates = tuple(
            template_for_check[dependency_id] for dependency_id in dependency_ids
        )
        bound_dependencies = tuple(
            f"validation:{template_name}" for template_name in dependency_templates
        )
        metadata = dict(step.metadata)
        work_item = metadata.get("agentic_work_item")
        if not isinstance(work_item, dict):  # pragma: no cover - _bind_plan owns it
            raise ValidationWorkflowError(
                f"Validation template {step.template_name!r} has no work item"
            )
        work_item = dict(work_item)
        work_item["depends_on"] = list(bound_dependencies)
        work_item["identity_digest"] = _work_item_identity(
            identity,
            template_name=step.template_name,
            depends_on=bound_dependencies,
        )
        metadata["agentic_work_item"] = work_item
        metadata["validation_coordinator_check"] = {
            "check_id": check_id,
            **dict(check_metadata[check_id]),
        }
        accepted_steps.append(step.model_copy(update={"metadata": metadata}))
    plan = plan.model_copy(update={"steps": tuple(accepted_steps)})
    preparation = ValidationOperationPreparation(
        output_dir=str(root),
        config_base_dir=str(base_dir),
        request=request,
        plan=plan,
        workflow_identity=identity,
        expansion=ValidationOperationExpansion(
            kind="coordinator_plan_patch",
            requested_values=check_ids,
            expanded_templates=requested,
        ),
        capabilities=tuple(
            _operation_capability(
                name,
                requested=name in requested,
                mandatory_when_requested=(
                    mandatory[name] if name in requested else None
                ),
            )
            for name in (
                *requested,
                *(
                    candidate
                    for candidate in V1_TEMPLATE_NAMES
                    if candidate not in requested
                ),
            )
        ),
    )
    ensure_no_inline_secrets(
        preparation.model_dump(mode="json"),
        context="accepted Validation coordinator adapter projection",
    )
    write_validation_planning_artifacts(request, plan, paths)
    atomic_write_json(root / VALIDATION_OPERATION_PREPARATION_NAME, preparation)
    return preparation


def load_validation_operation_preparation(
    output_dir: str | Path,
) -> ValidationOperationPreparation:
    root = Path(output_dir).expanduser().resolve()
    mixed = [
        name
        for name in PROVIDED_OPERATION_ARTIFACT_NAMES
        if (root / name).exists() or (root / name).is_symlink()
    ]
    if mixed:
        raise ValidationWorkflowError(
            "Validation execute and provided modes are mutually exclusive; found "
            + ", ".join(mixed)
        )
    preparation_path = root / VALIDATION_OPERATION_PREPARATION_NAME
    if preparation_path.is_symlink() or not preparation_path.is_file():
        raise ValidationWorkflowError(
            f"Focused Validation preparation is missing or unsafe: {preparation_path}"
        )
    try:
        preparation = ValidationOperationPreparation.model_validate(
            load_json(preparation_path)
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid focused Validation preparation at {root}: {exc}"
        ) from exc
    if Path(preparation.output_dir) != root:
        raise ValidationWorkflowError("Validation preparation belongs to another run")
    return preparation


def _operation_result_path(root: Path, template_name: str) -> Path:
    return root / "operations" / template_name / "operation_result.json"


def _template_result_path(root: Path, template_name: str) -> Path:
    return root / "operations" / template_name / "template_result.json"


def _load_operation_result(path: Path) -> ValidationOperationResult:
    if path.is_symlink() or not path.is_file():
        raise ValidationWorkflowError(
            f"Validation operation result is missing or unsafe: {path}"
        )
    try:
        result = ValidationOperationResult.model_validate(load_json(path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid Validation operation result {path}: {exc}"
        ) from exc
    if path.parent.name != result.template_name:
        raise ValidationWorkflowError(
            f"Validation operation result is stored under another template: {path}"
        )
    template_path = Path(result.template_result_path)
    expected_template_path = path.parent / "template_result.json"
    if (
        template_path.is_symlink()
        or not template_path.is_file()
        or template_path.resolve() != expected_template_path.resolve()
        or file_sha256(template_path) != result.template_result_sha256
    ):
        raise ValidationWorkflowError(
            f"Validation operation template result binding is stale: {template_path}"
        )
    try:
        bound_template_result = ValidationTemplateResult.model_validate(
            load_json(template_path)
        )
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid bound Validation template result {template_path}: {exc}"
        ) from exc
    if bound_template_result != result.template_result:
        raise ValidationWorkflowError(
            f"Validation operation template result bytes differ: {template_path}"
        )
    for artifact in result.evidence_artifacts:
        candidate = Path(artifact.path)
        if (
            artifact.kind == "missing"
            or candidate.is_symlink()
            or not candidate.exists()
        ):
            raise ValidationWorkflowError(
                f"Validation operation evidence is missing or unsafe: {candidate}"
            )
        current = _artifact_identity(
            candidate,
            role="evidence",
            base_dir=candidate.parent,
        )
        if current != artifact:
            raise ValidationWorkflowError(
                f"Validation operation evidence identity is stale: {candidate}"
            )
    return result


def load_validated_validation_operation_index(
    output_dir: str | Path,
) -> ValidationOperationIndex:
    """Load an index whose states and result bindings match its preparation."""

    preparation = load_validation_operation_preparation(output_dir)
    root = Path(preparation.output_dir)
    index_path = root / VALIDATION_OPERATION_INDEX_NAME
    if index_path.is_symlink() or not index_path.is_file():
        raise ValidationWorkflowError(
            f"Validation operation index is missing or unsafe: {index_path}"
        )
    try:
        index = ValidationOperationIndex.model_validate(load_json(index_path))
    except (OSError, ValueError, ValidationError) as exc:
        raise ValidationWorkflowError(
            f"Invalid Validation operation index {index_path}: {exc}"
        ) from exc
    preparation_digest = canonical_json_digest(preparation)
    if index.preparation_digest != preparation_digest:
        raise ValidationWorkflowError(
            "Validation operation index belongs to another preparation"
        )
    if tuple(item.template_name for item in index.operations) != tuple(
        item.template_name for item in preparation.capabilities
    ):
        raise ValidationWorkflowError(
            "Validation operation index order differs from its preparation"
        )
    for capability, status in zip(
        preparation.capabilities,
        index.operations,
        strict=True,
    ):
        expected_mandatory = (
            capability.mandatory_when_requested
            if capability.selection_state == "requested"
            else False
        )
        if status.mandatory != expected_mandatory:
            raise ValidationWorkflowError(
                f"Validation operation {capability.template_name} mandatory state "
                "differs from its explicit preparation"
            )
        result_path = _operation_result_path(root, capability.template_name)
        if capability.selection_state == "not_requested":
            if status.state != "not_requested" or (
                result_path.exists() or result_path.is_symlink()
            ):
                raise ValidationWorkflowError(
                    f"Validation operation {capability.template_name} state differs "
                    "from its explicit preparation"
                )
            continue
        if status.state == "not_requested":
            raise ValidationWorkflowError(
                f"Validation operation {capability.template_name} state differs "
                "from its explicit preparation"
            )
        if status.state == "evaluated":
            if (
                status.result_path is None
                or Path(status.result_path) != result_path
                or result_path.is_symlink()
                or not result_path.is_file()
                or status.result_sha256 != file_sha256(result_path)
            ):
                raise ValidationWorkflowError(
                    f"Validation operation {capability.template_name} index binding "
                    "is stale"
                )
            operation = _load_operation_result(result_path)
            if operation.preparation_digest != preparation_digest:
                raise ValidationWorkflowError(
                    f"Validation operation {capability.template_name} belongs to "
                    "another preparation"
                )
            if (
                not capability.mandatory_when_requested
                and operation.template_result.status == "skipped"
            ):
                raise ValidationWorkflowError(
                    "skipped advisory operation cannot be recorded as evaluated"
                )
            continue
        if capability.mandatory_when_requested:
            if result_path.exists() or result_path.is_symlink():
                raise ValidationWorkflowError(
                    f"mandatory Validation operation {capability.template_name} "
                    "has an unbound result"
                )
            continue
        if result_path.exists() or result_path.is_symlink():
            operation = _load_operation_result(result_path)
            if (
                operation.preparation_digest != preparation_digest
                or operation.template_result.status != "skipped"
            ):
                raise ValidationWorkflowError(
                    "optional Validation operation has inconsistent not_evaluated "
                    "evidence"
                )
    return index


def _required_not_evaluated_result(
    result: ValidationTemplateResult,
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.required_capability_not_evaluated",
        severity="fail",
        message=(
            f"Required selected capability {result.template_name} did not produce "
            "an evaluated result."
        ),
        template_name=result.template_name,
    )
    metadata = dict(result.metadata)
    metadata["selection_state"] = "not_evaluated"
    metadata["original_status"] = result.status
    return result.model_copy(
        update={
            "status": "failed",
            "issues": (*result.issues, issue),
            "metadata": metadata,
        }
    )


def run_validation_operation(
    output_dir: str | Path,
    *,
    template_name: str,
    prior_result_paths: tuple[str | Path, ...] = (),
    executor: ValidationStepExecutor | None = None,
) -> ValidationOperationResult:
    """Execute exactly one outer-selected template and bind its evidence."""

    preparation = load_validation_operation_preparation(output_dir)
    if template_name not in preparation.request.requested_templates:
        raise ValidationWorkflowError(
            f"Validation operation {template_name!r} was not requested by the outer plan"
        )
    root = Path(preparation.output_dir)
    operation_dir = _operation_result_path(root, template_name).parent
    if operation_dir.exists():
        raise ValidationWorkflowError(
            f"Validation operation output already exists: {operation_dir}"
        )
    executor_impl = executor or ScaffoldValidationStepExecutor(
        Path(preparation.config_base_dir)
    )
    current_identity = _workflow_identity(
        preparation.request,
        config_base_dir=Path(preparation.config_base_dir),
        executor=executor_impl,
    )
    if current_identity != preparation.workflow_identity:
        raise ValidationWorkflowError(
            "Focused Validation request, inputs, references, policy, backend, or "
            "template implementation changed after preparation"
        )
    current_plan = _bind_plan(
        executor_impl.plan(
            preparation.request.model_copy(deep=True),
            working_dir=root,
        ),
        identity=current_identity,
        artifact_paths=ValidationWorkflowPaths.from_output_dir(root).artifact_paths(),
    )
    if preparation.expansion.kind == "coordinator_plan_patch":
        projected_steps = []
        for current_step, accepted_step in zip(
            current_plan.steps,
            preparation.plan.steps,
            strict=True,
        ):
            if current_step.template_name != accepted_step.template_name:
                raise ValidationWorkflowError(
                    "Focused Validation plan step order changed after preparation"
                )
            metadata = dict(current_step.metadata)
            for key in ("agentic_work_item", "validation_coordinator_check"):
                bound_value = accepted_step.metadata.get(key)
                if not isinstance(bound_value, dict):
                    raise ValidationWorkflowError(
                        f"Accepted Validation plan step {accepted_step.template_name!r} "
                        f"has no bound {key} metadata"
                    )
                metadata[key] = dict(bound_value)
            projected_steps.append(
                current_step.model_copy(update={"metadata": metadata})
            )
        current_plan = current_plan.model_copy(update={"steps": tuple(projected_steps)})
    if current_plan != preparation.plan:
        raise ValidationWorkflowError(
            "Focused Validation plan changed after preparation"
        )
    preparation_digest = canonical_json_digest(preparation)
    prior_results: list[ValidationOperationResult] = []
    for prior_path in prior_result_paths:
        prior = _load_operation_result(Path(prior_path).expanduser().resolve())
        if prior.preparation_digest != preparation_digest:
            raise ValidationWorkflowError(
                "Prior Validation result belongs to another plan"
            )
        prior_results.append(prior)
    planned_step = next(
        step for step in preparation.plan.steps if step.template_name == template_name
    )
    capability = next(
        item for item in preparation.capabilities if item.template_name == template_name
    )
    work_item = planned_step.metadata.get("agentic_work_item")
    if not isinstance(work_item, dict) or not isinstance(
        work_item.get("depends_on"), list
    ):
        raise ValidationWorkflowError(
            f"Validation operation {template_name!r} has no bound dependency plan"
        )
    expected_prior_templates = tuple(
        dependency.removeprefix("validation:")
        for dependency in work_item["depends_on"]
        if isinstance(dependency, str)
    )
    actual_prior_templates = tuple(item.template_name for item in prior_results)
    if actual_prior_templates != expected_prior_templates:
        raise ValidationWorkflowError(
            f"Validation operation {template_name!r} requires exact prior results "
            f"{expected_prior_templates!r}, got {actual_prior_templates!r}"
        )
    operation_dir.mkdir(parents=True, exist_ok=False)
    operation_request = preparation.request.model_copy(deep=True)
    coordinator_check = planned_step.metadata.get("validation_coordinator_check")
    if isinstance(coordinator_check, dict):
        targets = coordinator_check.get("targets")
        if isinstance(targets, list | tuple) and targets:
            operation_request = operation_request.model_copy(
                update={"inputs": tuple(str(target) for target in targets)}
            )
        focus_prim_paths = coordinator_check.get("focus_prim_paths")
        if isinstance(focus_prim_paths, list | tuple):
            operation_request = operation_request.model_copy(
                update={
                    "focus": operation_request.focus.model_copy(
                        update={
                            "prim_paths": tuple(str(item) for item in focus_prim_paths)
                        }
                    )
                }
            )
        parameters = coordinator_check.get("parameters")
        if template_name == "physical_behavior" and isinstance(parameters, dict):
            evidence_paths = parameters.get("evidence_paths")
            if isinstance(evidence_paths, list | tuple):
                policy = dict(operation_request.policy)
                base_dir = Path(preparation.config_base_dir)
                refinement_records = _physical_behavior_refinement_records_by_path(
                    policy,
                    base_dir=base_dir,
                )
                original_records = _physical_behavior_evidence_records_by_path(
                    policy,
                    base_dir=base_dir,
                )
                selected_records: list[object] = []
                selected_refinement: dict[str, list[object]] = {}
                for path in evidence_paths:
                    resolved_path = str(Path(path).expanduser().resolve(strict=False))
                    refinement_record = refinement_records.get(resolved_path)
                    if refinement_record is not None:
                        key, item = refinement_record
                        selected_refinement.setdefault(key, []).append(item)
                        continue
                    if resolved_path not in original_records:
                        raise ValidationWorkflowError(
                            "Selected physical_behavior evidence lost its frozen "
                            f"policy binding: {resolved_path}"
                        )
                    selected_records.append(original_records[resolved_path])
                for key in (
                    *_PHYSICAL_BEHAVIOR_EVIDENCE_POLICY_KEYS,
                    *_PHYSICAL_BEHAVIOR_REFINEMENT_POLICY_KEYS,
                ):
                    policy.pop(key, None)
                policy["physical_behavior_evidence"] = selected_records
                for key, items in selected_refinement.items():
                    if key in _PHYSICAL_BEHAVIOR_REFINEMENT_OUTPUT_DIR_POLICY_KEYS:
                        if len(items) != 1:
                            raise ValidationWorkflowError(
                                "Selected physical_behavior refinement output binding "
                                f"is ambiguous: {key}"
                            )
                        policy[key] = items[0]
                    else:
                        policy[key] = items
                policy["behavior_evidence_required"] = (
                    policy.get("behavior_evidence_required") is True
                    or capability.mandatory_when_requested
                )
                operation_request = operation_request.model_copy(
                    update={"policy": policy}
                )
    blocked_dependencies = tuple(
        item.template_name
        for item in prior_results
        if not item.template_result.passed
        and (
            item.mandatory
            or (template_name == "look_right" and item.template_name == "render_valid")
        )
    )
    if blocked_dependencies:
        result = ValidationTemplateResult(
            template_name=template_name,
            status="skipped",
            issues=(
                ValidationIssue(
                    code="validation.operation_dependency_not_satisfied",
                    severity="warn",
                    message=(
                        f"Validation operation {template_name} was not evaluated "
                        "because its accepted dependencies did not pass: "
                        + ", ".join(blocked_dependencies)
                    ),
                    template_name=template_name,
                    details={"blocked_dependencies": list(blocked_dependencies)},
                ),
            ),
            metadata={
                "selection_state": "not_evaluated",
                "blocked_dependencies": list(blocked_dependencies),
            },
        )
    else:
        result = executor_impl.run(
            template_name,
            ValidationTemplateContext(
                request=operation_request,
                plan=preparation.plan.model_copy(deep=True),
                working_dir=operation_dir,
                previous_template_results=tuple(
                    item.template_result.model_copy(deep=True) for item in prior_results
                ),
            ),
        )
    if result.template_name != template_name:
        raise ValidationWorkflowError(
            f"Validation operation returned {result.template_name!r}, expected {template_name!r}"
        )
    if isinstance(coordinator_check, dict):
        result_metadata = dict(result.metadata)
        result_metadata["validation_coordinator_check"] = dict(coordinator_check)
        result = result.model_copy(update={"metadata": result_metadata})
    result = _canonicalize_result_artifact_paths(result, base_dir=operation_dir)
    if template_name == "render_valid":
        result = _validate_render_evidence_result(
            result,
            base_dir=operation_dir,
            qualified_evidence_base_dir=Path(preparation.config_base_dir),
            request=operation_request,
            identity=preparation.workflow_identity,
            forbidden_artifacts=(
                *preparation.workflow_identity.source_artifacts,
                *preparation.workflow_identity.reference_artifacts,
            ),
        )
    if capability.mandatory_when_requested and result.status == "skipped":
        result = _required_not_evaluated_result(result)
    source_after = _best_effort_artifact_identities(
        preparation.request.inputs,
        role="source",
        base_dir=Path(preparation.config_base_dir),
    )
    if source_after != preparation.workflow_identity.source_artifacts:
        raise ValidationWorkflowError(
            "Focused Validation source identity changed while executing "
            f"{template_name!r}; the operation result was rejected"
        )
    evidence, missing = _evidence_artifacts(result, base_dir=operation_dir)
    if missing:
        result = _missing_declared_evidence_result(result, missing)
        evidence, _ = _evidence_artifacts(result, base_dir=operation_dir)
    ensure_no_inline_secrets(
        result.model_dump(mode="json"),
        context=f"focused Validation {template_name} result",
    )
    template_path = _template_result_path(root, template_name)
    atomic_write_json(template_path, result)
    operation = ValidationOperationResult(
        preparation_digest=preparation_digest,
        template_name=template_name,
        role=_operation_role(template_name),
        mandatory=capability.mandatory_when_requested,
        template_result=result,
        template_result_path=str(template_path),
        template_result_sha256=file_sha256(template_path),
        evidence_artifacts=evidence,
        source_before=preparation.workflow_identity.source_artifacts,
        source_after=source_after,
    )
    atomic_write_json(_operation_result_path(root, template_name), operation)
    return operation


def _optional_not_evaluated_result(template_name: str) -> ValidationTemplateResult:
    metrics: dict[str, object] = {}
    metadata: dict[str, object] = {"selection_state": "not_evaluated"}
    if template_name == "look_right":
        metrics["vlm_invoked"] = False
        metadata["authority"] = "optional_advisory_critique"
    return ValidationTemplateResult(
        template_name=template_name,
        status="skipped",
        issues=(
            ValidationIssue(
                code="validation.advisory_capability_not_evaluated",
                severity="info",
                message=(
                    f"The accepted plan requested advisory {template_name} but did "
                    "not evaluate that capability."
                ),
                template_name=template_name,
            ),
        ),
        metrics=metrics,
        metadata=metadata,
    )


def finalize_validation_operations(
    output_dir: str | Path,
) -> ValidationWorkflowRun:
    """Publish the standard #38 bundle from exact outer-selected results."""

    preparation = load_validation_operation_preparation(output_dir)
    root = Path(preparation.output_dir)
    preparation_digest = canonical_json_digest(preparation)
    by_name: dict[str, ValidationOperationResult] = {}
    operation_timestamps: dict[str, datetime] = {}
    statuses: list[ValidationOperationStatus] = []
    for capability in preparation.capabilities:
        path = _operation_result_path(root, capability.template_name)
        if capability.selection_state == "not_requested":
            statuses.append(
                ValidationOperationStatus(
                    template_name=capability.template_name,
                    role=capability.role,
                    state="not_requested",
                    mandatory=False,
                )
            )
            continue
        if path.is_file():
            operation = _load_operation_result(path)
            if operation.preparation_digest != preparation_digest:
                raise ValidationWorkflowError(
                    f"Validation operation {capability.template_name} is stale"
                )
            operation_timestamps[capability.template_name] = operation.completed_at
            optional_not_evaluated = (
                not capability.mandatory_when_requested
                and operation.template_result.status == "skipped"
            )
            by_name[capability.template_name] = operation
            statuses.append(
                ValidationOperationStatus(
                    template_name=capability.template_name,
                    role=capability.role,
                    state=("not_evaluated" if optional_not_evaluated else "evaluated"),
                    mandatory=capability.mandatory_when_requested,
                    result_path=None if optional_not_evaluated else str(path),
                    result_sha256=(
                        None if optional_not_evaluated else file_sha256(path)
                    ),
                )
            )
        elif capability.mandatory_when_requested:
            statuses.append(
                ValidationOperationStatus(
                    template_name=capability.template_name,
                    role=capability.role,
                    state="not_evaluated",
                    mandatory=True,
                )
            )
        else:
            statuses.append(
                ValidationOperationStatus(
                    template_name=capability.template_name,
                    role=capability.role,
                    state="not_evaluated",
                    mandatory=False,
                )
            )
    missing_required = [
        item.template_name
        for item in statuses
        if item.state == "not_evaluated" and item.mandatory
    ]
    index = ValidationOperationIndex(
        preparation_digest=preparation_digest,
        operations=tuple(statuses),
        required_operations_complete=not missing_required,
        optional_critique_evaluated=any(
            item.template_name == "look_right" and item.state == "evaluated"
            for item in statuses
        ),
    )
    atomic_write_json(root / VALIDATION_OPERATION_INDEX_NAME, index)
    if missing_required:
        raise ValidationWorkflowError(
            "Required Validation operations were not evaluated: "
            + ", ".join(missing_required)
        )
    if not operation_timestamps:  # pragma: no cover - explicit plans are non-empty
        raise ValidationWorkflowError(
            "Focused Validation finalization requires an executed operation"
        )
    fallback_completed_at = max(operation_timestamps.values())
    expected_records = _records_from_plan(preparation.plan)
    mandatory_by_name = {
        capability.template_name: capability.mandatory_when_requested
        for capability in preparation.capabilities
    }
    accepted_records: list[ValidationWorkItemRecord] = []
    template_results: list[ValidationTemplateResult] = []
    for record in expected_records:
        selected_operation = by_name.get(record.template_name)
        if selected_operation is None:
            if mandatory_by_name[
                record.template_name
            ]:  # pragma: no cover - guard above
                raise ValidationWorkflowError(
                    f"Missing required operation {record.template_name}"
                )
            template_result = _optional_not_evaluated_result(record.template_name)
            generated = (
                root
                / "operations"
                / record.template_name
                / "not_evaluated_template_result.json"
            )
            generated.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(generated, template_result)
            result_path = generated
            evidence_artifacts: tuple[ValidationArtifactIdentity, ...] = ()
        else:
            template_result = selected_operation.template_result
            result_path = Path(selected_operation.template_result_path)
            evidence_artifacts = selected_operation.evidence_artifacts
        completed_at = operation_timestamps.get(
            record.template_name,
            fallback_completed_at,
        )
        template_results.append(template_result)
        accepted_records.append(
            record.model_copy(
                update={
                    "state": ValidationWorkItemState.COMPLETED,
                    "attempts": 1,
                    "accepted_result": ValidationAcceptedTemplateResult(
                        work_item_identity_digest=record.identity_digest,
                        attempt=1,
                        result=template_result,
                        result_path=str(result_path),
                        result_sha256=file_sha256(result_path),
                        evidence_artifacts=evidence_artifacts,
                        accepted_at=completed_at,
                    ),
                    "started_at": completed_at,
                    "finished_at": completed_at,
                }
            )
        )
    checkpoint = ValidationWorkflowCheckpoint(
        workflow_identity=preparation.workflow_identity,
        plan_digest=_plan_digest(preparation.plan),
        ordered_work_item_ids=tuple(record.work_item_id for record in accepted_records),
        records=tuple(accepted_records),
        created_at=min(operation_timestamps.values()),
        updated_at=max(operation_timestamps.values()),
    )
    paths = ValidationWorkflowPaths.from_output_dir(root)
    atomic_write_json(paths.checkpoint, checkpoint)
    source_after = _best_effort_artifact_identities(
        preparation.request.inputs,
        role="source",
        base_dir=Path(preparation.config_base_dir),
    )
    raw_result = ValidationResult(
        verdict=aggregate_validation_verdict(tuple(template_results)),
        request=preparation.request,
        plan=preparation.plan,
        template_results=tuple(template_results),
        issues=tuple(issue for result in template_results for issue in result.issues),
        metrics={result.template_name: result.metrics for result in template_results},
        evidence={result.template_name: result.evidence for result in template_results},
        metadata={
            "execution_mode": "outer_selected_operations",
            "operation_index": str(root / VALIDATION_OPERATION_INDEX_NAME),
            "nested_agent_launched": False,
        },
    )
    stat_result = root.stat()
    return finalize_validation_workflow(
        status=ValidationWorkflowStatus.COMPLETED,
        request=preparation.request,
        plan=preparation.plan,
        raw_result=raw_result,
        checkpoint=checkpoint,
        source_before=preparation.workflow_identity.source_artifacts,
        source_after=source_after,
        paths=paths,
        output_dir_identity=(stat_result.st_dev, stat_result.st_ino),
    )


__all__ = [
    "PROVIDED_OPERATION_ARTIFACT_NAMES",
    "VALIDATION_OPERATION_INDEX_NAME",
    "VALIDATION_OPERATION_PREPARATION_NAME",
    "ValidationOperationCapability",
    "ValidationOperationExpansion",
    "ValidationOperationIndex",
    "ValidationOperationPreparation",
    "ValidationOperationResult",
    "ValidationOperationStatus",
    "VALIDATION_OPERATION_PROFILES",
    "VALIDATION_OPERATION_RULES",
    "finalize_validation_operations",
    "load_validated_validation_operation_index",
    "load_validation_operation_preparation",
    "prepare_validation_operations",
    "run_validation_operation",
]
