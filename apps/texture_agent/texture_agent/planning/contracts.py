# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned shared contracts for deterministic texture planning.

The models in this module intentionally do not inspect USD or call a model
backend. They are the serialization boundary shared by the CLI, service,
planner, executor, and long-running workflows described by issue #466.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TEXTURE_PLANNING_REQUEST_SCHEMA_VERSION: Literal["texture-agent-plan-request.v1"] = (
    "texture-agent-plan-request.v1"
)
TEXTURE_PLAN_SCHEMA_VERSION: Literal["texture-agent-plan.v1"] = "texture-agent-plan.v1"

TEXTURE_UNIT_DEFAULT_CAP: Literal[32] = 32
TEXTURE_UV_AWARE_DEFAULT_CAP: Literal[16] = 16
TEXTURE_PLAN_HARD_CAP: Literal[64] = 64
DEFAULT_TEXTURE_SIZE = 1024

_UNIT_ID_PREFIX = "tu_"
_UNIT_ID_HEX_LENGTH = 20


class TextureDiscoveryMode(StrEnum):
    """How candidate materials are discovered for a plan."""

    EFFECTIVE_BOUND = "effective_bound"
    EXPLICIT = "explicit"
    ALL_AUTHORED = "all_authored"


class TextureUnitMode(StrEnum):
    """How selected scene members are reduced to generation units."""

    PER_MATERIAL = "per_material"
    PER_GROUP = "per_group"
    PER_PRIM = "per_prim"


class TextureDetailPolicy(StrEnum):
    """Semantic-detail policy recorded in a plan."""

    DEFAULT = "default"
    SURFACE_ONLY = "surface_only"


class TextureSelectionKind(StrEnum):
    """Kinds of candidates that may be selected or skipped."""

    MATERIAL = "material"
    GROUP = "group"
    PRIM = "prim"
    SUBSET = "subset"


class TexturePlanDecisionState(StrEnum):
    """Planner decision before any LLM or image-generation work starts."""

    READY = "ready"
    REQUIRES_OPERATOR_OVERRIDE = "requires_operator_override"
    REQUIRES_NARROWING = "requires_narrowing"
    REQUIRES_CONSOLIDATION = "requires_consolidation"
    UNSUPPORTED = "unsupported"


class _FrozenContract(BaseModel):
    """Strict immutable base for persisted planning artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _canonical_prim_path(value: str, *, field_name: str) -> str:
    path = _non_empty(value, field_name=field_name)
    if not path.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute USD prim path: {path!r}")
    if path != "/" and path.endswith("/"):
        raise ValueError(f"{field_name} must not end with '/': {path!r}")
    if "//" in path:
        raise ValueError(f"{field_name} must not contain '//': {path!r}")
    return path


def _canonical_prim_paths(
    values: Sequence[str],
    *,
    field_name: str,
    require_non_empty: bool = False,
) -> tuple[str, ...]:
    paths = tuple(
        _canonical_prim_path(value, field_name=field_name) for value in values
    )
    if require_non_empty and not paths:
        raise ValueError(f"{field_name} must contain at least one USD prim path")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field_name} must not contain duplicate USD prim paths")
    return tuple(sorted(paths))


def stable_texture_unit_id(
    *,
    unit_mode: TextureUnitMode | str,
    material_prim_paths: Sequence[str],
    member_prim_paths: Sequence[str] = (),
    member_subset_paths: Sequence[str] = (),
    group_key: str | None = None,
) -> str:
    """Return a deterministic unit ID based on canonical scene identity.

    Display names are intentionally excluded. Per-material identity follows the
    canonical material path. Per-prim identity also includes the selected prim
    or subset. Per-group identity uses a stable caller-provided group key when
    available and otherwise falls back to sorted membership.
    """

    mode = TextureUnitMode(unit_mode)
    materials = _canonical_prim_paths(
        material_prim_paths,
        field_name="material_prim_paths",
        require_non_empty=True,
    )
    prims = _canonical_prim_paths(
        member_prim_paths,
        field_name="member_prim_paths",
    )
    subsets = _canonical_prim_paths(
        member_subset_paths,
        field_name="member_subset_paths",
    )
    normalized_group_key = group_key.strip() if group_key is not None else None
    if normalized_group_key == "":
        raise ValueError("group_key must be non-empty when provided")

    identity: dict[str, Any] = {
        "unit_mode": mode.value,
        "material_prim_paths": materials,
    }
    if mode is TextureUnitMode.PER_PRIM:
        identity["member_prim_paths"] = prims
        identity["member_subset_paths"] = subsets
    elif mode is TextureUnitMode.PER_GROUP:
        if normalized_group_key is not None:
            identity["group_key"] = normalized_group_key
        else:
            identity["member_prim_paths"] = prims
            identity["member_subset_paths"] = subsets

    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_UNIT_ID_PREFIX}{digest[:_UNIT_ID_HEX_LENGTH]}"


class TexturePlanSource(_FrozenContract):
    """Input asset and optional upstream Material Agent assignment artifact."""

    source_asset: str = Field(min_length=1)
    upstream_assignment_artifact: str | None = None
    source_asset_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Optional lowercase SHA-256 for durable source identity.",
    )

    @field_validator("source_asset")
    @classmethod
    def _validate_source_asset(cls, value: str) -> str:
        return _non_empty(value, field_name="source_asset")

    @field_validator("upstream_assignment_artifact")
    @classmethod
    def _validate_assignment_artifact(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, field_name="upstream_assignment_artifact")


class TexturePlanRequest(_FrozenContract):
    """Normalized deterministic request consumed by a texture planner."""

    schema_version: Literal["texture-agent-plan-request.v1"] = (
        TEXTURE_PLANNING_REQUEST_SCHEMA_VERSION
    )
    source: TexturePlanSource
    discovery_mode: TextureDiscoveryMode = TextureDiscoveryMode.EFFECTIVE_BOUND
    unit_mode: TextureUnitMode = TextureUnitMode.PER_MATERIAL
    explicit_material_paths: tuple[str, ...] = ()
    explicit_prim_paths: tuple[str, ...] = ()
    detail_policy: TextureDetailPolicy = TextureDetailPolicy.DEFAULT
    texture_size: int = Field(default=DEFAULT_TEXTURE_SIZE, ge=1, le=16384)
    backend: str = Field(default="simple_image_gen", min_length=1)
    backend_default_cap: int = Field(
        default=TEXTURE_UNIT_DEFAULT_CAP,
        ge=1,
        le=TEXTURE_UNIT_DEFAULT_CAP,
    )
    operator_override_cap: int | None = Field(
        default=None,
        ge=1,
        le=TEXTURE_PLAN_HARD_CAP,
    )
    max_concurrency: int = Field(default=4, ge=1, le=TEXTURE_PLAN_HARD_CAP)
    unit_timeout_seconds: int = Field(default=600, ge=1, le=86400)

    @field_validator("explicit_material_paths")
    @classmethod
    def _validate_explicit_material_paths(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        return _canonical_prim_paths(
            value,
            field_name="explicit_material_paths",
        )

    @field_validator("explicit_prim_paths")
    @classmethod
    def _validate_explicit_prim_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_prim_paths(value, field_name="explicit_prim_paths")

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        return _non_empty(value, field_name="backend")

    @model_validator(mode="after")
    def _validate_scope_and_override(self) -> Self:
        if (
            self.discovery_mode == TextureDiscoveryMode.EXPLICIT
            and not self.explicit_material_paths
            and not self.explicit_prim_paths
        ):
            raise ValueError(
                "explicit discovery requires explicit_material_paths or "
                "explicit_prim_paths"
            )

        effective_default = min(
            TEXTURE_UNIT_DEFAULT_CAP,
            self.backend_default_cap,
        )
        if (
            self.operator_override_cap is not None
            and self.operator_override_cap <= effective_default
        ):
            raise ValueError(
                "operator_override_cap must be greater than the effective "
                f"backend default cap ({effective_default})"
            )
        return self


class TexturePlanLimits(_FrozenContract):
    """Default, backend, override, and hard limits recorded in a plan."""

    global_default_cap: Literal[32] = TEXTURE_UNIT_DEFAULT_CAP
    backend_default_cap: int = Field(ge=1, le=TEXTURE_UNIT_DEFAULT_CAP)
    operator_override_cap: int | None = Field(
        default=None,
        ge=1,
        le=TEXTURE_PLAN_HARD_CAP,
    )
    effective_cap: int = Field(ge=1, le=TEXTURE_PLAN_HARD_CAP)
    hard_cap: Literal[64] = TEXTURE_PLAN_HARD_CAP

    @classmethod
    def from_request(cls, request: TexturePlanRequest) -> Self:
        """Build the exact effective limits represented by a request."""
        effective_default = min(
            TEXTURE_UNIT_DEFAULT_CAP,
            request.backend_default_cap,
        )
        return cls(
            backend_default_cap=request.backend_default_cap,
            operator_override_cap=request.operator_override_cap,
            effective_cap=request.operator_override_cap or effective_default,
        )

    @model_validator(mode="after")
    def _validate_effective_cap(self) -> Self:
        effective_default = min(self.global_default_cap, self.backend_default_cap)
        if (
            self.operator_override_cap is not None
            and self.operator_override_cap <= effective_default
        ):
            raise ValueError(
                "operator_override_cap must be greater than the effective "
                f"backend default cap ({effective_default})"
            )
        expected = self.operator_override_cap or effective_default
        if self.effective_cap != expected:
            raise ValueError(
                f"effective_cap must be {expected} for the recorded defaults "
                "and override"
            )
        return self


class TexturePlanExecution(_FrozenContract):
    """Bounded execution settings frozen into the plan."""

    backend: str = Field(min_length=1)
    texture_size: int = Field(ge=1, le=16384)
    max_concurrency: int = Field(ge=1, le=TEXTURE_PLAN_HARD_CAP)
    unit_timeout_seconds: int = Field(ge=1, le=86400)

    @classmethod
    def from_request(cls, request: TexturePlanRequest) -> Self:
        """Copy execution-affecting settings from a normalized request."""
        return cls(
            backend=request.backend,
            texture_size=request.texture_size,
            max_concurrency=request.max_concurrency,
            unit_timeout_seconds=request.unit_timeout_seconds,
        )

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        return _non_empty(value, field_name="backend")


class TexturePlanUnit(_FrozenContract):
    """One selected material, appearance group, or scoped-prim texture job."""

    unit_id: str = Field(pattern=r"^tu_[0-9a-f]{20}$")
    unit_mode: TextureUnitMode
    material_prim_paths: tuple[str, ...] = Field(min_length=1)
    member_prim_paths: tuple[str, ...] = ()
    member_subset_paths: tuple[str, ...] = ()
    group_key: str | None = None
    display_name: str = Field(min_length=1)
    selection_reason_code: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    detail_policy: TextureDetailPolicy

    @classmethod
    def build(
        cls,
        *,
        unit_mode: TextureUnitMode | str,
        material_prim_paths: Sequence[str],
        display_name: str,
        detail_policy: TextureDetailPolicy | str,
        selection_reason_code: str,
        selection_reason: str,
        member_prim_paths: Sequence[str] = (),
        member_subset_paths: Sequence[str] = (),
        group_key: str | None = None,
    ) -> Self:
        """Build a unit with its canonical stable identifier."""
        unit_id = stable_texture_unit_id(
            unit_mode=TextureUnitMode(unit_mode),
            material_prim_paths=material_prim_paths,
            member_prim_paths=member_prim_paths,
            member_subset_paths=member_subset_paths,
            group_key=group_key,
        )
        return cls(
            unit_id=unit_id,
            unit_mode=TextureUnitMode(unit_mode),
            material_prim_paths=tuple(material_prim_paths),
            member_prim_paths=tuple(member_prim_paths),
            member_subset_paths=tuple(member_subset_paths),
            group_key=group_key,
            display_name=display_name,
            selection_reason_code=selection_reason_code,
            selection_reason=selection_reason,
            detail_policy=TextureDetailPolicy(detail_policy),
        )

    @field_validator("material_prim_paths")
    @classmethod
    def _validate_material_prim_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_prim_paths(
            value,
            field_name="material_prim_paths",
            require_non_empty=True,
        )

    @field_validator("member_prim_paths")
    @classmethod
    def _validate_member_prim_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_prim_paths(value, field_name="member_prim_paths")

    @field_validator("member_subset_paths")
    @classmethod
    def _validate_member_subset_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_prim_paths(value, field_name="member_subset_paths")

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        return _non_empty(value, field_name="display_name")

    @field_validator("selection_reason_code", "selection_reason")
    @classmethod
    def _validate_selection_reason(cls, value: str, info: Any) -> str:
        return _non_empty(value, field_name=info.field_name)

    @field_validator("group_key")
    @classmethod
    def _validate_group_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, field_name="group_key")

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if (
            self.unit_mode in {TextureUnitMode.PER_MATERIAL, TextureUnitMode.PER_PRIM}
            and len(self.material_prim_paths) != 1
        ):
            raise ValueError(
                f"{self.unit_mode} units must reference exactly one material prim path"
            )
        if (
            self.unit_mode in {TextureUnitMode.PER_MATERIAL, TextureUnitMode.PER_PRIM}
            and self.group_key is not None
        ):
            raise ValueError(f"{self.unit_mode} units must not record group_key")
        if (
            self.unit_mode == TextureUnitMode.PER_PRIM
            and len(self.member_prim_paths) + len(self.member_subset_paths) != 1
        ):
            raise ValueError(
                "per_prim units must contain exactly one member prim or subset path"
            )
        if (
            self.unit_mode == TextureUnitMode.PER_GROUP
            and self.group_key is None
            and not self.member_prim_paths
            and not self.member_subset_paths
        ):
            raise ValueError(
                "per_group units must include a group_key or at least one member path"
            )

        expected = stable_texture_unit_id(
            unit_mode=self.unit_mode,
            material_prim_paths=self.material_prim_paths,
            member_prim_paths=self.member_prim_paths,
            member_subset_paths=self.member_subset_paths,
            group_key=self.group_key,
        )
        if self.unit_id != expected:
            raise ValueError(
                f"unit_id must be the stable path-based identifier {expected!r}"
            )
        return self


class TexturePlanSkippedItem(_FrozenContract):
    """One candidate omitted from selection, with a machine-readable reason."""

    item_kind: TextureSelectionKind
    canonical_id: str = Field(min_length=1)
    display_name: str | None = None
    reason_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("canonical_id")
    @classmethod
    def _validate_canonical_id(cls, value: str) -> str:
        return _non_empty(value, field_name="canonical_id")

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, field_name="display_name")

    @field_validator("reason_code", "reason")
    @classmethod
    def _validate_reason_fields(cls, value: str, info: Any) -> str:
        return _non_empty(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _validate_path_identity(self) -> Self:
        if self.item_kind in {
            TextureSelectionKind.MATERIAL,
            TextureSelectionKind.PRIM,
            TextureSelectionKind.SUBSET,
        }:
            _canonical_prim_path(self.canonical_id, field_name="canonical_id")
        return self


class TexturePlanCounts(_FrozenContract):
    """Auditable discovery, selection, and backend-job counts."""

    authored_material_count: int = Field(ge=0)
    renderable_prim_count: int = Field(ge=0)
    renderable_subset_count: int = Field(ge=0)
    effective_bound_material_count: int = Field(ge=0)
    selected_material_count: int = Field(ge=0)
    selected_unit_count: int = Field(ge=0)
    skipped_item_count: int = Field(ge=0)
    planned_generation_job_count: int = Field(ge=0)


class TexturePlanDecision(_FrozenContract):
    """Whether the immutable plan is approved for bounded execution."""

    state: TexturePlanDecisionState
    execution_allowed: bool
    consolidation_required: bool = False
    explicit_narrowing_required: bool = False
    reasons: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()

    @field_validator("reasons", "recommended_actions")
    @classmethod
    def _validate_messages(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        normalized = tuple(
            _non_empty(item, field_name=info.field_name) for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        if self.execution_allowed and self.state != TexturePlanDecisionState.READY:
            raise ValueError("execution_allowed plans must have state='ready'")
        if not self.execution_allowed and self.state == TexturePlanDecisionState.READY:
            raise ValueError("state='ready' requires execution_allowed=true")
        if self.state == TexturePlanDecisionState.READY and (
            self.consolidation_required or self.explicit_narrowing_required
        ):
            raise ValueError(
                "state='ready' must not require consolidation or explicit narrowing"
            )
        if (
            self.state == TexturePlanDecisionState.REQUIRES_CONSOLIDATION
            and not self.consolidation_required
        ):
            raise ValueError(
                "state='requires_consolidation' requires consolidation_required=true"
            )
        if (
            self.state == TexturePlanDecisionState.REQUIRES_NARROWING
            and not self.explicit_narrowing_required
        ):
            raise ValueError(
                "state='requires_narrowing' requires explicit_narrowing_required=true"
            )
        if self.state == TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE and (
            self.consolidation_required or self.explicit_narrowing_required
        ):
            raise ValueError(
                "state='requires_operator_override' must not require consolidation "
                "or explicit narrowing"
            )
        if not self.execution_allowed and not self.reasons:
            raise ValueError("non-executable plans must include at least one reason")
        if not self.execution_allowed and not self.recommended_actions:
            raise ValueError(
                "non-executable plans must include at least one recommended action"
            )
        return self


class TexturePlan(_FrozenContract):
    """Immutable plan artifact shared across all Texture Agent surfaces."""

    schema_version: Literal["texture-agent-plan.v1"] = TEXTURE_PLAN_SCHEMA_VERSION
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    request: TexturePlanRequest
    limits: TexturePlanLimits
    execution: TexturePlanExecution
    counts: TexturePlanCounts
    selected_units: tuple[TexturePlanUnit, ...] = ()
    skipped_items: tuple[TexturePlanSkippedItem, ...] = ()
    decision: TexturePlanDecision

    @field_validator("generated_at")
    @classmethod
    def _validate_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_plan(self) -> Self:
        expected_limits = TexturePlanLimits.from_request(self.request)
        if self.limits != expected_limits:
            raise ValueError("limits must exactly reflect the normalized request")
        expected_execution = TexturePlanExecution.from_request(self.request)
        if self.execution != expected_execution:
            raise ValueError("execution must exactly reflect the normalized request")

        if self.counts.selected_unit_count != len(self.selected_units):
            raise ValueError(
                "selected_unit_count must equal the number of selected_units"
            )
        if self.counts.skipped_item_count != len(self.skipped_items):
            raise ValueError(
                "skipped_item_count must equal the number of skipped_items"
            )
        if self.counts.planned_generation_job_count != len(self.selected_units):
            raise ValueError(
                "planned_generation_job_count must equal the number of "
                "selected texture units"
            )

        unit_ids = [unit.unit_id for unit in self.selected_units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("selected_units must have unique unit_id values")
        if any(
            unit.unit_mode != self.request.unit_mode for unit in self.selected_units
        ):
            raise ValueError("selected unit modes must match request.unit_mode")

        selected_material_paths = {
            material_path
            for unit in self.selected_units
            for material_path in unit.material_prim_paths
        }
        if self.counts.selected_material_count != len(selected_material_paths):
            raise ValueError(
                "selected_material_count must equal unique selected material paths"
            )

        jobs = self.counts.planned_generation_job_count
        if (
            self.decision.state == TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE
            and jobs <= self.limits.effective_cap
        ):
            raise ValueError(
                "state='requires_operator_override' requires planned jobs above "
                "the effective texture-unit cap"
            )
        if self.decision.execution_allowed:
            if jobs > self.limits.effective_cap:
                raise ValueError(
                    "execution cannot be allowed above the effective texture-unit cap"
                )
            if jobs > self.limits.hard_cap:
                raise ValueError(
                    "execution cannot be allowed above the hard texture-unit cap"
                )
        elif jobs > self.limits.hard_cap and not (
            self.decision.consolidation_required
            or self.decision.explicit_narrowing_required
        ):
            raise ValueError(
                "plans above the hard cap must require consolidation or explicit "
                "narrowing"
            )
        return self


def validate_texture_plan_payload(
    payload: TexturePlan | Mapping[str, Any] | str | bytes,
) -> TexturePlan:
    """Validate a model, mapping, or JSON payload as a Texture Plan v1."""
    if isinstance(payload, TexturePlan):
        return payload
    if isinstance(payload, str | bytes):
        return TexturePlan.model_validate_json(payload)
    return TexturePlan.model_validate(payload)


def texture_plan_json_schema() -> dict[str, Any]:
    """Return JSON Schema for the persisted Texture Plan v1 contract."""
    return TexturePlan.model_json_schema()
