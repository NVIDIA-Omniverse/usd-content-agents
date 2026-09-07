# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable safe-restart disposition for non-resumable coordinator runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from world_understanding.validation import ValidationRequest

from content_agent_workflows.common.artifacts import atomic_write_json, load_json
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)

from .coordinator import (
    VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME,
    VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME,
    VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
    VALIDATION_COORDINATOR_PREPARATION_NAME,
)
from .embedded_assessment import VALIDATION_TERMINAL_RECEIPT_NAME
from .verified_operations import (
    VerifiedOperationError,
    execution_artifact_binding,
    verify_execution_artifact_binding,
)
from .workflow import ValidationWorkflowError

VALIDATION_COORDINATOR_SAFE_RESTART_SCHEMA_VERSION: Final = (
    "content-agent-workflows.validation-coordinator-safe-restart.v1"
)
VALIDATION_COORDINATOR_SAFE_RESTART_NAME: Final = "validation_safe_restart.json"

_CAPABILITY_IDS: Final = {
    "render_valid": "validation.render_valid",
    "look_right": "validation.look_right",
    "physics_sane": "validation.physics_sane",
    "physical_behavior": "validation.physical_behavior",
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationCoordinatorSafeRestart(_FrozenModel):
    """Receipt proving that coordinator continuity was refused safely."""

    schema_version: Literal[
        "content-agent-workflows.validation-coordinator-safe-restart.v1"
    ] = VALIDATION_COORDINATOR_SAFE_RESTART_SCHEMA_VERSION
    disposition: Literal["safe_restart_required"] = "safe_restart_required"
    reason_code: Literal["coordinator_resume_unsupported"] = (
        "coordinator_resume_unsupported"
    )
    original_run_dir: str = Field(min_length=1)
    original_request: ExecutionArtifactBinding
    coordinator_preparation: ExecutionArtifactBinding
    retained_coordinator_artifacts: tuple[ExecutionArtifactBinding, ...]
    unsupported_template_names: tuple[str, ...] = Field(min_length=1)
    unsupported_capability_ids: tuple[str, ...] = Field(min_length=1)
    recommended_command_prefix: tuple[
        Literal["content-workflow-cli"], Literal["validate"], Literal["run"]
    ] = ("content-workflow-cli", "validate", "run")
    required_new_output_argument: Literal["--output-dir <new-empty-directory>"] = (
        "--output-dir <new-empty-directory>"
    )
    prior_run_preserved: Literal[True] = True
    restart_requires_fresh_run_identity: Literal[True] = True
    restart_requires_fresh_output_dir: Literal[True] = True
    prior_results_reused: Literal[False] = False
    stale_result_promotion_refused: Literal[True] = True
    legacy_executor_invoked: Literal[False] = False
    nested_agent_launched: Literal[False] = False
    source_mutated: Literal[False] = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    receipt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("safe-restart timestamp must include a timezone")
        if len(self.unsupported_template_names) != len(
            set(self.unsupported_template_names)
        ):
            raise ValueError("safe-restart template names must be unique")
        if len(self.unsupported_capability_ids) != len(
            set(self.unsupported_capability_ids)
        ):
            raise ValueError("safe-restart capability IDs must be unique")
        unknown_templates = tuple(
            template
            for template in self.unsupported_template_names
            if template not in _CAPABILITY_IDS
        )
        if unknown_templates:
            raise ValueError(
                "safe-restart contains unknown templates: "
                + ", ".join(unknown_templates)
            )
        expected_capabilities = tuple(
            _CAPABILITY_IDS[template] for template in self.unsupported_template_names
        )
        if self.unsupported_capability_ids != expected_capabilities:
            raise ValueError("safe-restart capability IDs differ from templates")
        expected = canonical_json_digest(
            self.model_dump(mode="json", exclude={"receipt_digest"})
        )
        if self.receipt_digest != expected:
            raise ValueError("safe-restart receipt digest is stale")
        return self


def _verify_persisted_request(
    binding: ExecutionArtifactBinding,
    *,
    request: ValidationRequest,
) -> None:
    try:
        payload = verify_execution_artifact_binding(
            binding,
            label="Validation safe-restart original request",
        )
        persisted = ValidationRequest.model_validate_json(payload)
    except (ValidationError, VerifiedOperationError) as exc:
        raise ValidationWorkflowError(str(exc)) from exc
    if persisted != request:
        raise ValidationWorkflowError(
            "Validation safe-restart request differs from the persisted run"
        )


def _verify_receipt_bindings(
    receipt: ValidationCoordinatorSafeRestart,
    *,
    request: ValidationRequest,
) -> None:
    _verify_persisted_request(receipt.original_request, request=request)
    bindings = (
        receipt.coordinator_preparation,
        *receipt.retained_coordinator_artifacts,
    )
    try:
        for binding in bindings:
            verify_execution_artifact_binding(
                binding,
                label="Validation safe-restart retained artifact",
            )
    except VerifiedOperationError as exc:
        raise ValidationWorkflowError(str(exc)) from exc


def write_validation_coordinator_safe_restart(
    output_dir: str | Path,
    *,
    request: ValidationRequest,
) -> ValidationCoordinatorSafeRestart:
    """Write or revalidate one idempotent refusal to resume in place."""

    root = Path(output_dir).expanduser().resolve()
    receipt_path = root / VALIDATION_COORDINATOR_SAFE_RESTART_NAME
    if receipt_path.is_symlink():
        raise ValidationWorkflowError(
            f"Validation safe-restart artifact must not be a symlink: {receipt_path}"
        )
    if receipt_path.exists():
        try:
            receipt = ValidationCoordinatorSafeRestart.model_validate(
                load_json(receipt_path)
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise ValidationWorkflowError(
                f"Invalid existing Validation safe-restart artifact {receipt_path}: {exc}"
            ) from exc
        if Path(receipt.original_run_dir) != root:
            raise ValidationWorkflowError(
                "Validation safe-restart artifact belongs to another run"
            )
        _verify_receipt_bindings(receipt, request=request)
        return receipt

    try:
        request_binding = execution_artifact_binding(root / "validation_request.json")
        preparation_binding = execution_artifact_binding(
            root / VALIDATION_COORDINATOR_PREPARATION_NAME
        )
        retained_items: list[ExecutionArtifactBinding] = []
        for name in (
            VALIDATION_COORDINATOR_PLAN_PATCH_NAME,
            VALIDATION_COORDINATOR_ACCEPTED_PLAN_NAME,
            VALIDATION_COORDINATOR_EXECUTION_RECEIPT_NAME,
            VALIDATION_TERMINAL_RECEIPT_NAME,
        ):
            candidate = root / name
            if candidate.is_symlink():
                raise ValidationWorkflowError(
                    "Validation safe-restart retained artifact must not be a "
                    f"symlink: {candidate}"
                )
            if candidate.exists():
                retained_items.append(execution_artifact_binding(candidate))
        retained = tuple(retained_items)
    except VerifiedOperationError as exc:
        raise ValidationWorkflowError(str(exc)) from exc

    templates = tuple(request.requested_templates)
    _verify_persisted_request(request_binding, request=request)
    if not templates:
        raise ValidationWorkflowError(
            "Validation coordinator safe restart requires accepted template IDs"
        )
    unknown_templates = tuple(
        template for template in templates if template not in _CAPABILITY_IDS
    )
    if unknown_templates:
        raise ValidationWorkflowError(
            "Validation coordinator safe restart found unsupported template IDs: "
            + ", ".join(unknown_templates)
        )
    capability_ids = tuple(_CAPABILITY_IDS[template] for template in templates)
    created_at = datetime.now(UTC)
    unsealed = ValidationCoordinatorSafeRestart.model_construct(
        original_run_dir=str(root),
        original_request=request_binding,
        coordinator_preparation=preparation_binding,
        retained_coordinator_artifacts=retained,
        unsupported_template_names=templates,
        unsupported_capability_ids=capability_ids,
        created_at=created_at,
        receipt_digest="0" * 64,
    )
    receipt = ValidationCoordinatorSafeRestart(
        original_run_dir=str(root),
        original_request=request_binding,
        coordinator_preparation=preparation_binding,
        retained_coordinator_artifacts=retained,
        unsupported_template_names=templates,
        unsupported_capability_ids=capability_ids,
        created_at=created_at,
        receipt_digest=canonical_json_digest(
            unsealed.model_dump(mode="json", exclude={"receipt_digest"})
        ),
    )
    atomic_write_json(receipt_path, receipt)
    return receipt


__all__ = [
    "VALIDATION_COORDINATOR_SAFE_RESTART_NAME",
    "VALIDATION_COORDINATOR_SAFE_RESTART_SCHEMA_VERSION",
    "ValidationCoordinatorSafeRestart",
    "write_validation_coordinator_safe_restart",
]
