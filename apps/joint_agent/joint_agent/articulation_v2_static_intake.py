# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure articulation-v2 static intake wire contract.

The intake carries already-reviewed source/reference identities and a fresh
output locator. Resolver capture and artifact production belong to later work.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from world_understanding.utils.artifacts import validated_artifact_relative_key

from joint_agent.articulation_v2_static_artifact_identity import (
    ArticulationV2StaticGeneratedOutputLocatorV1,
)
from joint_agent.articulation_v2_static_run_plan import (
    ArticulationV2StaticControlledDistanceRunIdentityV2,
    ArticulationV2StaticControlledDistanceRunPlanV2,
    ArticulationV2StaticRunIdentityV1,
    ArticulationV2StaticRunPlanV1,
    _revalidate_articulation_v2_static_controlled_distance_run_plan,
    _revalidate_articulation_v2_static_run_plan,
    articulation_v2_static_controlled_distance_run_plan_sha256,
    articulation_v2_static_run_plan_sha256,
    validate_articulation_v2_static_controlled_distance_run_plan,
    validate_articulation_v2_static_run_plan,
)
from joint_agent.articulation_v2_static_semantics import (
    ArticulationV2ControlledDistanceSelectorSemanticsV2,
    ArticulationV2StaticJointSemanticsV1,
    canonical_articulation_v2_controlled_distance_selector_semantics_bytes,
    canonical_articulation_v2_static_joint_semantics_bytes,
)
from joint_agent.capability_manifest import (
    CapabilityManifestError,
    LoadedCapabilityManifest,
)

ARTICULATION_V2_STATIC_INTAKE_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-intake-v1"
] = "joint-agent-articulation-v2-static-intake-v1"
ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_INTAKE_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-intake-v2"
] = "joint-agent-articulation-v2-static-intake-v2"
ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_REPRESENTATION: Literal[
    "prismatic_axial_interval_drive_body1_to_body0_v2"
] = "prismatic_axial_interval_drive_body1_to_body0_v2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArticulationV2StaticIntakeError(ValueError):
    """Stable parse failure for an articulation-v2 static intake."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ArticulationV2StaticIntakeV1(BaseModel):
    """One immutable fixed/distance contract intake for an exact run plan."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal["joint-agent-articulation-v2-static-intake-v1"]
    run_plan_sha256: str
    run: ArticulationV2StaticRunIdentityV1
    output: ArticulationV2StaticGeneratedOutputLocatorV1
    contract_uri: str
    contract_sha256: str
    contract_size_bytes: int = Field(gt=0)
    contract_semantics: ArticulationV2StaticJointSemanticsV1

    @field_validator("run_plan_sha256", "contract_sha256")
    @classmethod
    def _lowercase_sha256(cls, value: str, info: Any) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256")
        return value

    @field_validator("contract_uri")
    @classmethod
    def _canonical_contract_uri(cls, value: str) -> str:
        try:
            canonical: str = validated_artifact_relative_key(value)
        except ValueError as exc:
            raise ValueError(
                "contract_uri must be a canonical relative POSIX path"
            ) from exc
        if not canonical.startswith("input/contracts/") or not canonical.endswith(
            ".json"
        ):
            raise ValueError("contract_uri must name input/contracts/*.json")
        return canonical

    @model_validator(mode="after")
    def _roles_and_semantics_are_exact(self) -> ArticulationV2StaticIntakeV1:
        expected = self.run.selector.expected_joint_semantics
        if self.contract_semantics != expected:
            raise ValueError("intake contract differs from admitted selector semantics")
        semantic_bytes = canonical_articulation_v2_static_joint_semantics_bytes(
            self.contract_semantics
        )
        if self.contract_sha256 != hashlib.sha256(semantic_bytes).hexdigest():
            raise ValueError("contract digest differs from its canonical semantics")
        if self.contract_size_bytes != len(semantic_bytes):
            raise ValueError("contract size differs from its canonical semantics")
        if self.output.run_id != self.run.run_id:
            raise ValueError("intake output locator differs from run identity")
        expected_contract_uri = (
            f"input/contracts/{self.run.selector.constraint_kind}.json"
        )
        if self.contract_uri != expected_contract_uri:
            raise ValueError("contract URI differs from selected constraint")
        if self.contract_uri == self.output.output_key:
            raise ValueError("contract and output roles must be distinct")
        return self


class ArticulationV2StaticControlledDistanceIntakeV2(BaseModel):
    """Sibling intake for the exact admitted controlled-distance v2 semantics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal["joint-agent-articulation-v2-static-intake-v2"]
    representation: Literal["prismatic_axial_interval_drive_body1_to_body0_v2"]
    run_plan_sha256: str
    run: ArticulationV2StaticControlledDistanceRunIdentityV2
    output: ArticulationV2StaticGeneratedOutputLocatorV1
    contract_uri: str
    contract_sha256: str
    contract_size_bytes: int = Field(gt=0)
    contract_semantics: ArticulationV2ControlledDistanceSelectorSemanticsV2

    @field_validator("run_plan_sha256", "contract_sha256")
    @classmethod
    def _lowercase_sha256(cls, value: str, info: Any) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256")
        return value

    @field_validator("contract_uri")
    @classmethod
    def _canonical_contract_uri(cls, value: str) -> str:
        try:
            canonical: str = validated_artifact_relative_key(value)
        except ValueError as exc:
            raise ValueError(
                "contract_uri must be a canonical relative POSIX path"
            ) from exc
        if canonical != "input/contracts/distance.json":
            raise ValueError(
                "v2 intake must retain the admitted distance contract role"
            )
        return canonical

    @model_validator(mode="after")
    def _exact_v2_semantics(
        self,
    ) -> ArticulationV2StaticControlledDistanceIntakeV2:
        expected = self.run.selector.expected_joint_semantics
        semantic_bytes = (
            canonical_articulation_v2_controlled_distance_selector_semantics_bytes(
                self.contract_semantics
            )
        )
        if (
            self.representation != expected.controlled_distance_contract.representation
            or self.contract_semantics != expected
            or self.contract_sha256 != hashlib.sha256(semantic_bytes).hexdigest()
            or self.contract_size_bytes != len(semantic_bytes)
            or self.output.run_id != self.run.run_id
        ):
            raise ValueError("v2 intake differs from its admitted semantics")
        return self


_INTAKE_ADAPTER = TypeAdapter(ArticulationV2StaticIntakeV1)
_CONTROLLED_DISTANCE_INTAKE_ADAPTER = TypeAdapter(
    ArticulationV2StaticControlledDistanceIntakeV2
)


def canonical_articulation_v2_static_intake_bytes(
    intake: ArticulationV2StaticIntakeV1,
) -> bytes:
    if type(intake) is not ArticulationV2StaticIntakeV1:
        raise TypeError("intake must be an exact ArticulationV2StaticIntakeV1")
    encoded = json.dumps(
        _INTAKE_ADAPTER.dump_python(intake, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    validated = _INTAKE_ADAPTER.validate_json(encoded, strict=True)
    return json.dumps(
        _INTAKE_ADAPTER.dump_python(validated, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def articulation_v2_static_intake_sha256(
    intake: ArticulationV2StaticIntakeV1,
) -> str:
    return hashlib.sha256(
        canonical_articulation_v2_static_intake_bytes(intake)
    ).hexdigest()


def canonical_articulation_v2_static_controlled_distance_intake_bytes(
    intake: ArticulationV2StaticControlledDistanceIntakeV2,
) -> bytes:
    if type(intake) is not ArticulationV2StaticControlledDistanceIntakeV2:
        raise TypeError("intake must be an exact controlled-distance v2 intake")
    encoded = json.dumps(
        _CONTROLLED_DISTANCE_INTAKE_ADAPTER.dump_python(intake, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    validated = _CONTROLLED_DISTANCE_INTAKE_ADAPTER.validate_json(encoded, strict=True)
    return json.dumps(
        _CONTROLLED_DISTANCE_INTAKE_ADAPTER.dump_python(validated, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def articulation_v2_static_controlled_distance_intake_sha256(
    intake: ArticulationV2StaticControlledDistanceIntakeV2,
) -> str:
    return hashlib.sha256(
        canonical_articulation_v2_static_controlled_distance_intake_bytes(intake)
    ).hexdigest()


def validate_articulation_v2_static_intake(
    loaded: LoadedCapabilityManifest,
    plan: ArticulationV2StaticRunPlanV1,
    intake: ArticulationV2StaticIntakeV1,
) -> None:
    """Validate an intake and its plan using only already-loaded authority."""

    plan = _revalidate_articulation_v2_static_run_plan(plan)
    intake = _revalidate_articulation_v2_static_intake(intake)
    validate_articulation_v2_static_run_plan(loaded, plan)
    if intake.run_plan_sha256 != articulation_v2_static_run_plan_sha256(plan):
        raise CapabilityManifestError("static intake binds a different run plan")
    if intake.run != plan.run:
        raise CapabilityManifestError("static intake run identity differs from plan")
    if intake.output != plan.output:
        raise CapabilityManifestError("static intake output locator differs from plan")
    if intake.contract_uri not in plan.evidence_paths.paths:
        raise CapabilityManifestError("static intake contract path is not allowed")


def validate_articulation_v2_static_controlled_distance_intake(
    loaded: LoadedCapabilityManifest,
    plan: ArticulationV2StaticControlledDistanceRunPlanV2,
    intake: ArticulationV2StaticControlledDistanceIntakeV2,
) -> None:
    """Validate the v2 intake using only already-loaded authority."""

    plan = _revalidate_articulation_v2_static_controlled_distance_run_plan(plan)
    intake = _revalidate_articulation_v2_static_controlled_distance_intake(intake)
    validate_articulation_v2_static_controlled_distance_run_plan(loaded, plan)
    if intake.run_plan_sha256 != (
        articulation_v2_static_controlled_distance_run_plan_sha256(plan)
    ):
        raise CapabilityManifestError("v2 intake binds a different run plan")
    if intake.run != plan.run or intake.output != plan.output:
        raise CapabilityManifestError("v2 intake differs from its exact plan")
    if intake.contract_uri not in plan.evidence_paths.paths:
        raise CapabilityManifestError("v2 intake contract path is not allowed")


def _revalidate_articulation_v2_static_intake(
    intake: ArticulationV2StaticIntakeV1,
) -> ArticulationV2StaticIntakeV1:
    if type(intake) is not ArticulationV2StaticIntakeV1:
        raise CapabilityManifestError(
            "articulation-v2 static intake must use the exact 0.6 contract"
        )
    try:
        return _INTAKE_ADAPTER.validate_json(
            canonical_articulation_v2_static_intake_bytes(intake),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise CapabilityManifestError(
            "articulation-v2 static intake fails strict invariant revalidation"
        ) from exc


def _revalidate_articulation_v2_static_controlled_distance_intake(
    intake: ArticulationV2StaticControlledDistanceIntakeV2,
) -> ArticulationV2StaticControlledDistanceIntakeV2:
    if type(intake) is not ArticulationV2StaticControlledDistanceIntakeV2:
        raise CapabilityManifestError(
            "controlled-distance v2 intake must use the exact sibling contract"
        )
    try:
        return _CONTROLLED_DISTANCE_INTAKE_ADAPTER.validate_json(
            canonical_articulation_v2_static_controlled_distance_intake_bytes(intake),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise CapabilityManifestError(
            "controlled-distance v2 intake fails strict invariant revalidation"
        ) from exc


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ArticulationV2StaticIntakeError(
                "articulation_v2_static_intake_duplicate_key",
                f"duplicate JSON object key: {key!r}",
            )
        document[key] = value
    return document


def parse_articulation_v2_static_intake(
    payload: str | bytes | bytearray | dict[str, Any],
) -> ArticulationV2StaticIntakeV1:
    """Strictly parse an intake while rejecting duplicate JSON object keys."""

    try:
        document: Any
        if isinstance(payload, str | bytes | bytearray):
            document = json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_json_pairs,
            )
        elif type(payload) is dict:
            document = payload
        else:
            raise TypeError("mapping intake payloads must be exact dictionaries")
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return _INTAKE_ADAPTER.validate_json(encoded, strict=True)
    except ArticulationV2StaticIntakeError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ArticulationV2StaticIntakeError(
            "articulation_v2_static_intake_invalid",
            "payload is not a valid strict articulation-v2 static intake",
        ) from exc


__all__ = [
    "ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_INTAKE_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_REPRESENTATION",
    "ARTICULATION_V2_STATIC_INTAKE_SCHEMA_VERSION",
    "ArticulationV2StaticIntakeError",
    "ArticulationV2StaticIntakeV1",
    "ArticulationV2StaticControlledDistanceIntakeV2",
    "articulation_v2_static_controlled_distance_intake_sha256",
    "articulation_v2_static_intake_sha256",
    "canonical_articulation_v2_static_controlled_distance_intake_bytes",
    "canonical_articulation_v2_static_intake_bytes",
    "parse_articulation_v2_static_intake",
    "validate_articulation_v2_static_intake",
    "validate_articulation_v2_static_controlled_distance_intake",
]
