# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable Validation Agent V1 request, plan, and result contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic import field_validator as _pydantic_field_validator

from world_understanding.validation.json_normalization import (
    StructuredJsonNormalizer,
)
from world_understanding.validation.rendering_backend_contract import (
    VALIDATION_RENDERING_BACKEND_NAMES,
)

IssueSeverity = Literal["info", "warn", "fail"]
TemplateStatus = Literal[
    "passed",
    "warn",
    "failed",
    "needs_refinement",
    "skipped",
    "error",
]
ValidationVerdict = Literal["pass", "warn", "fail", "needs_refinement", "planned"]
PlannerBackend = Literal["auto", "llm", "vlm", "rules"]
InputKind = Literal[
    "usd",
    "image",
    "reference_image",
    "video",
    "render_bundle",
    "artifact_dir",
]

ISSUE_CODE_PATTERN = r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$"
SCHEMA_VERSION: Literal["1.0"] = "1.0"

_ValidatorFunc = TypeVar("_ValidatorFunc", bound=Callable[..., Any])
_JSON_NORMALIZER = StructuredJsonNormalizer(model_types=(BaseModel,))
_json_mapping = _JSON_NORMALIZER.mapping
_json_value = _JSON_NORMALIZER.value


def field_validator(
    *fields: str,
    mode: str = "after",
    check_fields: bool | None = None,
) -> Callable[[_ValidatorFunc], _ValidatorFunc]:
    """Typed wrapper around Pydantic's validator decorator for strict mypy."""

    kwargs: dict[str, object] = {"mode": mode}
    if check_fields is not None:
        kwargs["check_fields"] = check_fields
    validator = cast(Any, _pydantic_field_validator)
    return cast(
        Callable[[_ValidatorFunc], _ValidatorFunc], validator(*fields, **kwargs)
    )


class ValidationModel(BaseModel):
    """Base model for JSON-safe validation contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidationProject(ValidationModel):
    """Project/session metadata shared by all Validation Agent entry points."""

    name: str | None = Field(default=None, description="Human-readable run name.")
    working_dir: str | None = Field(
        default=None,
        description="Directory where plan, result, and evidence artifacts are written.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional caller-provided session identifier.",
    )

    @field_validator("working_dir", mode="before")
    @classmethod
    def _normalize_optional_path(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)


class ValidationPlannerConfig(ValidationModel):
    """Planner selection and model configuration."""

    backend: PlannerBackend = Field(
        default="rules",
        description="Planner implementation: auto, llm, vlm, or rules.",
    )
    model: str | None = Field(
        default=None,
        description="Optional planner model name for llm/vlm/auto backends.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationRenderConfig(ValidationModel):
    """Render policy used by validation templates that need visual evidence."""

    backend: str | None = Field(
        default=None,
        description=(
            "Validation Agent runtime rendering backend. Supported values are "
            f"{', '.join(VALIDATION_RENDERING_BACKEND_NAMES)}; omit to use remote. "
            "The field remains an open string so runtime validation can return "
            "structured unknown-versus-unsupported backend results."
        ),
    )
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    views: str | tuple[str, ...] | None = Field(default=None)
    animation_frames: str | tuple[int, ...] | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("views", mode="before")
    @classmethod
    def _normalize_views(cls, value: Any) -> str | tuple[str, ...] | None:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, Sequence):
            return tuple(str(item) for item in value)
        return str(value)

    @field_validator("animation_frames", mode="before")
    @classmethod
    def _normalize_animation_frames(cls, value: Any) -> str | tuple[int, ...] | None:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, Sequence):
            return tuple(int(item) for item in value)
        return str(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationFocusConfig(ValidationModel):
    """Manual focus-prim configuration.

    V1 does not allow planners to invent focus prims; these values come from
    request/config input or from deterministic fixtures.
    """

    prim_paths: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prim_paths", mode="before")
    @classmethod
    def _normalize_prim_paths(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence):
            return tuple(str(item) for item in value)
        raise ValueError("focus prim_paths must be a string or sequence of strings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationRequest(ValidationModel):
    """Stable Validation Agent V1 request/config model."""

    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION)
    task_description: str = Field(min_length=1)
    inputs: tuple[str, ...] = Field(
        description="File or directory inputs, resolved by the runner/input resolver."
    )
    project: ValidationProject = Field(default_factory=ValidationProject)
    planner: ValidationPlannerConfig = Field(default_factory=ValidationPlannerConfig)
    render: ValidationRenderConfig = Field(default_factory=ValidationRenderConfig)
    focus: ValidationFocusConfig = Field(default_factory=ValidationFocusConfig)
    requested_templates: tuple[str, ...] = Field(default_factory=tuple)
    policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs", mode="before")
    @classmethod
    def _normalize_inputs(cls, value: Any) -> tuple[str, ...]:
        inputs: tuple[str, ...]
        if isinstance(value, str | Path):
            inputs = (str(value),)
        elif isinstance(value, Sequence):
            inputs = tuple(str(item) for item in value)
        else:
            raise ValueError("inputs must be a path string or a sequence of paths")
        if not inputs:
            raise ValueError("At least one validation input is required")
        return inputs

    @field_validator("requested_templates", mode="before")
    @classmethod
    def _normalize_requested_templates(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Sequence):
            return tuple(str(item) for item in value)
        raise ValueError("requested_templates must be a string or sequence of strings")

    @field_validator("policy", "metadata", mode="before")
    @classmethod
    def _normalize_json_mapping(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationInput(ValidationModel):
    """One resolved validation input."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    original: str
    path: str
    kind: InputKind
    extension: str | None = None
    image_paths: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: Any) -> str:
        if isinstance(value, str):
            if value:
                return value
            raise ValueError("path must not be empty")
        if isinstance(value, Path):
            return str(value)
        raise ValueError("path must be a string or Path")

    @field_validator("image_paths", mode="before")
    @classmethod
    def _normalize_image_paths(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str | Path):
            return (str(value),)
        if isinstance(value, Sequence):
            return tuple(str(path) for path in value)
        raise ValueError("image_paths must be a string or sequence of strings")

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationInputGroups(ValidationModel):
    """Grouped resolved inputs for planner/template consumption."""

    items: tuple[ValidationInput, ...] = Field(default_factory=tuple)
    usd_paths: tuple[str, ...] = Field(default_factory=tuple)
    image_paths: tuple[str, ...] = Field(default_factory=tuple)
    reference_image_paths: tuple[str, ...] = Field(default_factory=tuple)
    video_paths: tuple[str, ...] = Field(default_factory=tuple)
    render_bundle_dirs: tuple[str, ...] = Field(default_factory=tuple)
    render_bundle_image_paths: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "usd_paths",
        "image_paths",
        "reference_image_paths",
        "video_paths",
        "render_bundle_dirs",
        "render_bundle_image_paths",
        mode="before",
    )
    @classmethod
    def _normalize_path_tuple(cls, value: Any) -> tuple[str, ...]:
        return _string_tuple(value)

    @classmethod
    def from_inventory_dict(
        cls,
        inventory: Mapping[str, Any],
    ) -> ValidationInputGroups:
        """Create grouped inputs from ``InputInventory.to_dict()`` output."""

        return cls(
            items=tuple(
                ValidationInput.model_validate(item)
                for item in _mapping_sequence(inventory, "items")
            ),
            usd_paths=_string_tuple(inventory.get("usd_paths")),
            image_paths=_string_tuple(inventory.get("image_paths")),
            reference_image_paths=_string_tuple(inventory.get("reference_image_paths")),
            video_paths=_string_tuple(inventory.get("video_paths")),
            render_bundle_dirs=_string_tuple(inventory.get("render_bundle_dirs")),
            render_bundle_image_paths=_string_tuple(
                inventory.get("render_bundle_image_paths")
            ),
        )


class ValidationPlanStep(ValidationModel):
    """One planned template invocation."""

    template_name: str = Field(min_length=1)
    reason: str = Field(default="")
    inputs_needed: tuple[str, ...] = Field(default_factory=tuple)
    required_capabilities: tuple[str, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inputs_needed", "required_capabilities", mode="before")
    @classmethod
    def _normalize_string_tuple(cls, value: Any) -> tuple[str, ...]:
        return _string_tuple(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationPlan(ValidationModel):
    """Stable, serializable plan consumed by Validation Agent templates."""

    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION)
    steps: tuple[ValidationPlanStep, ...]
    input_groups: ValidationInputGroups = Field(default_factory=ValidationInputGroups)
    focus_prim_paths: tuple[str, ...] = Field(default_factory=tuple)
    visual_prompt: str | None = None
    physical_prompt: str | None = None
    reasoning_summary: str = Field(default="")
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("steps")
    @classmethod
    def _validate_steps(
        cls, value: tuple[ValidationPlanStep, ...]
    ) -> tuple[ValidationPlanStep, ...]:
        if not value:
            raise ValueError("ValidationPlan requires at least one step")
        return value

    @field_validator("focus_prim_paths", mode="before")
    @classmethod
    def _normalize_focus_paths(cls, value: Any) -> tuple[str, ...]:
        return _string_tuple(value)

    @field_validator("artifact_paths", mode="before")
    @classmethod
    def _normalize_artifact_paths(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("artifact_paths must be a mapping")
        return {str(key): str(path) for key, path in value.items()}

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationIssue(ValidationModel):
    """Stable validation issue emitted by templates or runner infrastructure."""

    code: str = Field(pattern=ISSUE_CODE_PATTERN)
    severity: IssueSeverity
    message: str = Field(min_length=1)
    template_name: str | None = None
    subject: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details", mode="before")
    @classmethod
    def _normalize_details(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationEvidence(ValidationModel):
    """Typed evidence pointer for report summaries and service payloads."""

    kind: str = Field(min_length=1)
    path: str | None = None
    subject: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_optional_path(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationTemplateResult(ValidationModel):
    """Result from one validation template execution."""

    template_name: str = Field(min_length=1)
    status: TemplateStatus
    issues: tuple[ValidationIssue, ...] = Field(default_factory=tuple)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    evidence_items: tuple[ValidationEvidence, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed" and all(
            issue.severity != "fail" for issue in self.issues
        )

    @field_validator("metrics", "evidence", "metadata", mode="before")
    @classmethod
    def _normalize_json_mapping(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)


class ValidationResult(ValidationModel):
    """Stable Validation Agent V1 report model."""

    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION)
    verdict: ValidationVerdict
    request: ValidationRequest
    plan: ValidationPlan
    template_results: tuple[ValidationTemplateResult, ...] = Field(
        default_factory=tuple
    )
    issues: tuple[ValidationIssue, ...] = Field(default_factory=tuple)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    recommended_action: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metrics", "evidence", "metadata", mode="before")
    @classmethod
    def _normalize_json_mapping(cls, value: Any) -> dict[str, Any]:
        return _json_mapping(value)

    @field_validator("artifact_paths", mode="before")
    @classmethod
    def _normalize_artifact_paths(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("artifact_paths must be a mapping")
        return {str(key): str(path) for key, path in value.items()}


def aggregate_validation_verdict(
    template_results: Sequence[ValidationTemplateResult],
) -> ValidationVerdict:
    """Aggregate template results into a stable validation verdict.

    A run with no template results is a plan-only result, so it returns
    ``planned``. Executed validation results should include at least one
    template result and therefore aggregate to ``pass``, ``warn``,
    ``needs_refinement``, or ``fail``.
    """

    if not template_results:
        return "planned"

    if any(
        result.status in {"failed", "error"}
        or any(issue.severity == "fail" for issue in result.issues)
        for result in template_results
    ):
        return "fail"
    if any(result.status == "needs_refinement" for result in template_results):
        return "needs_refinement"
    if any(
        result.status == "warn"
        or result.status == "skipped"
        or any(issue.severity == "warn" for issue in result.issues)
        for result in template_results
    ):
        return "warn"
    return "pass"


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | Path):
        return (str(value),)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise ValueError("Expected a string or sequence of strings")


def _mapping_sequence(
    mapping: Mapping[str, Any],
    key: str,
) -> tuple[Mapping[str, Any], ...]:
    value = mapping.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))
