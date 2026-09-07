# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest-bound static qualification matrix for Joint Agent capabilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)
from world_understanding.functions.physics.joint_rigger import ArtifactIdentityV1

from joint_agent.capability_manifest import (
    _STATIC_QUALIFICATION_SCOPE_ISSUE,
    CapabilityManifestError,
    CapabilityRowV1,
    EvidenceBindingV1,
    LoadedCapabilityManifest,
    QualificationStatus,
    StaticQualificationStageBindingV1,
    load_capability_manifest,
    qualification_evidence_roster,
    reject_duplicate_json_keys,
    resolve_repository_path,
    validate_corpus_bindings,
)
from joint_agent.static_qualification_contracts import (
    ARTICULATION_V2_CONTRACT_LANE,
    RELEASE_GATE_V0_5_CONTRACT_LANE,
    STATIC_STAGE_VALIDATION_CONTRACTS,
    StaticQualificationContractLane,
    require_registered_static_validation_contract,
    static_validation_contract,
)

STATIC_QUALIFICATION_SCHEMA_VERSION: Literal[
    "joint-agent-static-qualification-matrix-v1"
] = "joint-agent-static-qualification-matrix-v1"
STATIC_QUALIFICATION_RUN_PLAN_SCHEMA_VERSION: Literal[
    "joint-agent-static-qualification-run-plan-v1"
] = "joint-agent-static-qualification-run-plan-v1"
STATIC_QUALIFICATION_RUN_REQUEST_SCHEMA_VERSION: Literal[
    "joint-agent-static-qualification-run-request-v1"
] = "joint-agent-static-qualification-run-request-v1"
STATIC_QUALIFICATION_RESULT_SCHEMA_VERSION: Literal[
    "joint-agent-static-qualification-result-v1"
] = "joint-agent-static-qualification-result-v1"
STATIC_QUALIFICATION_SCORECARD_SCHEMA_VERSION: Literal[
    "joint-agent-static-qualification-scorecard-v1"
] = "joint-agent-static-qualification-scorecard-v1"

type StaticStageName = Literal[
    "contract",
    "authoring",
    "readback",
    "gate3a",
    "gate3b",
]
type StaticPlanExecution = Literal["retain_frozen_0_5", "run", "blocked"]
type StaticExecutionStatus = Literal[
    "pass",
    "fail",
    "blocked",
    "error",
    "not_run",
    "na",
]
type StaticFailureCategory = Literal[
    "contract",
    "authoring",
    "package",
    "static_schema",
    "unsupported_capability",
]
type StaticRetainedArtifactRole = Literal[
    "contract",
    "output",
    "validator",
    "result",
    "raw_report",
]

STATIC_STAGE_NAMES: tuple[StaticStageName, ...] = (
    "contract",
    "authoring",
    "readback",
    "gate3a",
    "gate3b",
)

STATIC_RETAINED_ARTIFACT_ROLES: tuple[StaticRetainedArtifactRole, ...] = (
    "contract",
    "output",
    "validator",
    "result",
    "raw_report",
)

_SUPPORT_RANK = {
    "recognized": 0,
    "contract_ready": 1,
    "authorable": 2,
    "static_qualified": 3,
    "dynamic_qualified": 4,
}
_RAW_FAILURE_STATUS_TOKENS = frozenset(
    {
        "ABORT",
        "ABORTED",
        "BLOCKED",
        "CRASH",
        "CRASHED",
        "ERROR",
        "FAIL",
        "FAILED",
        "FATAL",
        "NOT_RUN",
        "TIMEOUT",
        "TIMED_OUT",
    }
)
_RAW_FAILURE_FINDING_SEVERITIES = frozenset(
    {"CRITICAL", "ERROR", "FAIL", "FAILED", "FATAL"}
)
_RAW_PASS_STATUSES = frozenset(
    {"COMPLETED", "OK", "PASS", "PASSED", "SUCCEEDED", "SUCCESS", "VALID"}
)
_RAW_NONFAIL_FINDING_SEVERITIES = frozenset({"INFO", "NOTICE", "WARN", "WARNING"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _QualificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StaticStageResultV1(_QualificationModel):
    status: QualificationStatus
    profile_id: str | None = None
    result_id: str | None = None
    result_sha256: str | None = None
    artifact_identity_set_sha256: str | None = None

    @field_validator("profile_id", "result_id")
    @classmethod
    def _nonblank_optional(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be blank")
        return normalized

    @field_validator("result_sha256", "artifact_identity_set_sha256")
    @classmethod
    def _valid_optional_sha256(
        cls,
        value: str | None,
        info: Any,
    ) -> str | None:
        if value is None:
            return value
        normalized = value.strip().lower()
        if _SHA256_RE.fullmatch(normalized) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest")
        return normalized

    @model_validator(mode="after")
    def _complete_terminal_result(self) -> StaticStageResultV1:
        values = (
            self.profile_id,
            self.result_id,
            self.result_sha256,
            self.artifact_identity_set_sha256,
        )
        if self.status in {"pass", "fail"} and any(value is None for value in values):
            raise ValueError("pass/fail stages require complete result identities")
        if self.status in {"not_run", "na"} and any(
            value is not None for value in values
        ):
            raise ValueError("not_run/na stages must not carry result identities")
        return self


class StaticStagesV1(_QualificationModel):
    contract: StaticStageResultV1
    authoring: StaticStageResultV1
    readback: StaticStageResultV1
    gate3a: StaticStageResultV1
    gate3b: StaticStageResultV1

    @model_validator(mode="after")
    def _same_artifact_identity(self) -> StaticStagesV1:
        identities = {
            result.artifact_identity_set_sha256
            for result in (
                self.contract,
                self.authoring,
                self.readback,
                self.gate3a,
                self.gate3b,
            )
            if result.status in {"pass", "fail"}
        }
        if len(identities) > 1:
            raise ValueError(
                "all completed static stages must bind the same artifact identity set"
            )
        return self


class StaticQualificationRowV1(_QualificationModel):
    capability_id: str
    evidence_bindings: tuple[EvidenceBindingV1, ...]
    stages: StaticStagesV1
    static_qualified: bool
    dynamic_qualified: Literal[False] = False
    blocking_reason_codes: tuple[str, ...]

    @field_validator("capability_id")
    @classmethod
    def _nonblank_capability_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("capability_id must not be blank")
        return normalized

    @field_validator("evidence_bindings")
    @classmethod
    def _canonical_evidence(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)

    @field_validator("blocking_reason_codes")
    @classmethod
    def _canonical_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("blocking reason codes must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("blocking reason codes must be unique")
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def _qualification_state(self) -> StaticQualificationRowV1:
        statuses = (
            self.stages.contract.status,
            self.stages.authoring.status,
            self.stages.readback.status,
            self.stages.gate3a.status,
            self.stages.gate3b.status,
        )
        all_pass = all(status == "pass" for status in statuses)
        if self.static_qualified != all_pass:
            raise ValueError(
                "static_qualified must be true exactly when every static stage passes"
            )
        if self.static_qualified and self.blocking_reason_codes:
            raise ValueError("qualified rows must not carry blocking reason codes")
        if not self.static_qualified and not self.blocking_reason_codes:
            raise ValueError("unqualified rows require blocking reason codes")
        return self


class StaticQualificationSummaryV1(_QualificationModel):
    row_count: int = Field(ge=0)
    completed_with_findings_count: int = Field(ge=0)
    not_run_row_count: int = Field(ge=0)
    static_qualified_count: int = Field(ge=0)


class StaticQualificationMatrixV1(_QualificationModel):
    schema_version: Literal["joint-agent-static-qualification-matrix-v1"]
    matrix_version: str
    capability_manifest_version: str
    capability_manifest_sha256: str
    corpus_id: str
    corpus_sha256: str
    coverage_report_sha256: str
    frozen_0_5_release_manifest_sha256: str
    frozen_0_5_gate3_baseline_sha256: str
    rows: tuple[StaticQualificationRowV1, ...]
    summary: StaticQualificationSummaryV1

    @field_validator(
        "capability_manifest_sha256",
        "corpus_sha256",
        "coverage_report_sha256",
        "frozen_0_5_release_manifest_sha256",
        "frozen_0_5_gate3_baseline_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        normalized = value.strip().lower()
        if _SHA256_RE.fullmatch(normalized) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest")
        return normalized

    @field_validator(
        "matrix_version",
        "capability_manifest_version",
        "corpus_id",
    )
    @classmethod
    def _nonblank_text(cls, value: str, info: Any) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be blank")
        return normalized

    @field_validator("rows")
    @classmethod
    def _canonical_rows(
        cls,
        value: tuple[StaticQualificationRowV1, ...],
    ) -> tuple[StaticQualificationRowV1, ...]:
        ids = [row.capability_id for row in value]
        if len(ids) != len(set(ids)):
            raise ValueError("qualification capability IDs must be unique")
        return tuple(sorted(value, key=lambda row: row.capability_id))

    @model_validator(mode="after")
    def _summary_matches_rows(self) -> StaticQualificationMatrixV1:
        completed_with_findings = sum(
            all(
                status in {"pass", "fail"}
                for status in (
                    row.stages.contract.status,
                    row.stages.authoring.status,
                    row.stages.readback.status,
                    row.stages.gate3a.status,
                    row.stages.gate3b.status,
                )
            )
            and not row.static_qualified
            for row in self.rows
        )
        not_run = sum(
            any(
                status == "not_run"
                for status in (
                    row.stages.contract.status,
                    row.stages.authoring.status,
                    row.stages.readback.status,
                    row.stages.gate3a.status,
                    row.stages.gate3b.status,
                )
            )
            for row in self.rows
        )
        expected = StaticQualificationSummaryV1(
            row_count=len(self.rows),
            completed_with_findings_count=completed_with_findings,
            not_run_row_count=not_run,
            static_qualified_count=sum(row.static_qualified for row in self.rows),
        )
        if self.summary != expected:
            raise ValueError("qualification summary does not match matrix rows")
        return self


class StaticQualificationPlanStageV1(_QualificationModel):
    """One explicit static stage in a manifest-driven run plan."""

    stage: StaticStageName
    execution: StaticPlanExecution
    validation_contract: str
    profile_id: str | None = None
    result_id: str | None = None
    command: tuple[str, ...] = ()
    tool_id: str | None = None
    tool_version: str | None = None
    blocking_reason_codes: tuple[str, ...] = ()
    retained_result: StaticStageResultV1 | None = None

    @field_validator(
        "validation_contract",
        "profile_id",
        "result_id",
        "tool_id",
        "tool_version",
    )
    @classmethod
    def _nonblank_optional_text(
        cls,
        value: str | None,
        info: Any,
    ) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be blank")
        return normalized

    @field_validator("command")
    @classmethod
    def _nonblank_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("command arguments must not be blank")
        return value

    @field_validator("blocking_reason_codes")
    @classmethod
    def _canonical_blockers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_reason_codes(value)

    @model_validator(mode="after")
    def _execution_shape(self) -> StaticQualificationPlanStageV1:
        require_registered_static_validation_contract(
            stage=self.stage,
            value=self.validation_contract,
        )
        tool_fields = (self.tool_id, self.tool_version)
        if (self.tool_id is None) != (self.tool_version is None):
            raise ValueError("tool_id and tool_version must be supplied together")
        if self.execution == "run":
            if (
                self.profile_id is None
                or self.result_id is None
                or not self.command
                or self.tool_id is None
                or self.blocking_reason_codes
                or self.retained_result is not None
            ):
                raise ValueError(
                    "run stages require a profile, command, and tool identity "
                    "without blockers or retained evidence"
                )
        elif self.execution == "blocked":
            if (
                not self.blocking_reason_codes
                or self.profile_id is not None
                or self.result_id is not None
                or self.command
                or any(value is not None for value in tool_fields)
                or self.retained_result is not None
            ):
                raise ValueError(
                    "blocked stages require blockers and no command, tool, or "
                    "retained result"
                )
        elif (
            self.profile_id is None
            or self.result_id is None
            or self.command
            or any(value is not None for value in tool_fields)
            or self.blocking_reason_codes
            or self.retained_result is None
        ):
            raise ValueError(
                "retained 0.5 stages require a profile and retained result only"
            )
        if self.retained_result is not None and (
            self.retained_result.profile_id != self.profile_id
            or self.retained_result.result_id != self.result_id
        ):
            raise ValueError("retained result identity differs from the plan")
        return self


class StaticQualificationPlanStagesV1(_QualificationModel):
    contract: StaticQualificationPlanStageV1
    authoring: StaticQualificationPlanStageV1
    readback: StaticQualificationPlanStageV1
    gate3a: StaticQualificationPlanStageV1
    gate3b: StaticQualificationPlanStageV1

    @model_validator(mode="after")
    def _stage_names_match_fields(self) -> StaticQualificationPlanStagesV1:
        for stage in STATIC_STAGE_NAMES:
            if getattr(self, stage).stage != stage:
                raise ValueError(f"{stage} plan must declare stage={stage!r}")
        return self


class StaticQualificationPlanRowV1(_QualificationModel):
    capability_id: str
    row_admission_sha256: str
    evidence_bindings: tuple[EvidenceBindingV1, ...]
    stages: StaticQualificationPlanStagesV1

    @field_validator("capability_id")
    @classmethod
    def _nonblank_capability(cls, value: str) -> str:
        return _nonblank_text(value, "capability_id")

    @field_validator("row_admission_sha256")
    @classmethod
    def _valid_row_admission_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _SHA256_RE.fullmatch(normalized) is None:
            raise ValueError("row_admission_sha256 must be a lowercase SHA-256 digest")
        return normalized

    @field_validator("evidence_bindings")
    @classmethod
    def _canonical_evidence(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)


class FrozenQualificationLaneV1(_QualificationModel):
    release_manifest_sha256: str
    gate3_baseline_sha256: str
    asset_count: int = Field(gt=0)
    joint_count: int = Field(gt=0)

    @field_validator("release_manifest_sha256", "gate3_baseline_sha256")
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)


class StaticQualificationRunPlanV1(_QualificationModel):
    schema_version: Literal["joint-agent-static-qualification-run-plan-v1"]
    plan_version: str
    capability_manifest_version: str
    capability_manifest_sha256: str
    corpus_id: str
    corpus_sha256: str
    coverage_report_sha256: str
    frozen_0_5: FrozenQualificationLaneV1
    qualification_assets: tuple[EvidenceBindingV1, ...]
    rows: tuple[StaticQualificationPlanRowV1, ...]

    @field_validator(
        "capability_manifest_sha256",
        "corpus_sha256",
        "coverage_report_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)

    @field_validator(
        "plan_version",
        "capability_manifest_version",
        "corpus_id",
    )
    @classmethod
    def _nonblank_text(cls, value: str, info: Any) -> str:
        return _nonblank_text(value, info.field_name)

    @field_validator("qualification_assets")
    @classmethod
    def _canonical_qualification_assets(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)

    @field_validator("rows")
    @classmethod
    def _canonical_rows(
        cls,
        value: tuple[StaticQualificationPlanRowV1, ...],
    ) -> tuple[StaticQualificationPlanRowV1, ...]:
        return _canonical_unique_rows(value, label="run-plan")


def _required_command_arguments(
    value: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    """Require a command that names something, on every artifact that carries one.

    Three artifacts assert a command -- the stage request, the trusted report,
    and the executed stage -- and they had grown three copies of this rule with
    only the label differing. Drift between them would let one artifact accept a
    command the others reject.
    """

    if not value or any(not item.strip() for item in value):
        raise ValueError(f"{label} command requires nonblank arguments")
    return value


class StaticQualificationStageRequestV1(_QualificationModel):
    """One stage an operator intends to execute, with its intended identity."""

    stage: StaticStageName
    profile_id: str
    result_id: str
    command: tuple[str, ...]
    tool_id: str
    tool_version: str

    @field_validator("profile_id", "result_id", "tool_id", "tool_version")
    @classmethod
    def _nonblank_text(cls, value: str, info: Any) -> str:
        return _nonblank_text(value, info.field_name)

    @field_validator("command")
    @classmethod
    def _required_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _required_command_arguments(value, "stage request")


class StaticQualificationRunRequestV1(_QualificationModel):
    """The stages of one capability row an operator intends to execute."""

    capability_id: str
    stages: tuple[StaticQualificationStageRequestV1, ...]

    @field_validator("capability_id")
    @classmethod
    def _nonblank_capability(cls, value: str) -> str:
        return _nonblank_text(value, "capability_id")

    @field_validator("stages")
    @classmethod
    def _unique_stages(
        cls,
        value: tuple[StaticQualificationStageRequestV1, ...],
    ) -> tuple[StaticQualificationStageRequestV1, ...]:
        if not value:
            raise ValueError("a run request must name at least one stage")
        stages = [item.stage for item in value]
        if len(stages) != len(set(stages)):
            raise ValueError("a run request must name each stage at most once")
        return tuple(
            sorted(value, key=lambda item: STATIC_STAGE_NAMES.index(item.stage))
        )


class StaticQualificationRunRequestDocumentV1(_QualificationModel):
    """The reviewed, loadable form of a static qualification run request."""

    schema_version: Literal["joint-agent-static-qualification-run-request-v1"]
    requests: tuple[StaticQualificationRunRequestV1, ...]

    @field_validator("requests")
    @classmethod
    def _unique_capabilities(
        cls,
        value: tuple[StaticQualificationRunRequestV1, ...],
    ) -> tuple[StaticQualificationRunRequestV1, ...]:
        ids = [item.capability_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("a run request document must name each capability once")
        return tuple(sorted(value, key=lambda item: item.capability_id))


class StaticEvidenceHashesV1(_QualificationModel):
    """Complete provenance chain for one executed static stage."""

    source_sha256: str
    manifest_sha256: str
    contract_sha256: str
    output_sha256: str
    output_dependency_bundle_sha256: str
    validator_sha256: str
    result_sha256: str

    @field_validator(
        "source_sha256",
        "manifest_sha256",
        "contract_sha256",
        "output_sha256",
        "output_dependency_bundle_sha256",
        "validator_sha256",
        "result_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)


class StaticRetainedArtifactV1(_QualificationModel):
    """One retained byte artifact that a trusted resolver must reopen."""

    role: StaticRetainedArtifactRole
    uri: str

    @field_validator("uri")
    @classmethod
    def _nonblank_uri(cls, value: str) -> str:
        return _nonblank_text(value, "retained artifact URI")


class StaticResolvedDependencyV1(_QualificationModel):
    """One immutable byte member in a freshly reopened USD dependency closure."""

    path: str
    payload: bytes

    @field_validator("path")
    @classmethod
    def _canonical_relative_path(cls, value: str) -> str:
        normalized = _nonblank_text(value, "dependency path")
        path = PurePosixPath(normalized)
        if (
            normalized == "<root>"
            or "\\" in normalized
            or path.is_absolute()
            or path.as_posix() != normalized
            or any(part in {".", ".."} for part in path.parts)
        ):
            raise ValueError("dependency path must be a canonical relative POSIX path")
        return normalized


class StaticResolvedUsdArtifactV1(_QualificationModel):
    """Root bytes plus the complete immutable closure observed in one reopen."""

    uri: str
    root_bytes: bytes
    dependencies: tuple[StaticResolvedDependencyV1, ...]

    @field_validator("uri")
    @classmethod
    def _nonblank_uri(cls, value: str) -> str:
        return _nonblank_text(value, "resolved output URI")

    @field_validator("dependencies")
    @classmethod
    def _canonical_dependencies(
        cls,
        value: tuple[StaticResolvedDependencyV1, ...],
    ) -> tuple[StaticResolvedDependencyV1, ...]:
        paths = [dependency.path for dependency in value]
        if len(paths) != len(set(paths)):
            raise ValueError("resolved dependency paths must be unique")
        return tuple(sorted(value, key=lambda dependency: dependency.path))


class StaticTrustedStageReportV1(_QualificationModel):
    """Status and findings recomputed by trusted code from retained bytes."""

    status: Literal["pass", "fail", "error"]
    raw_status: str
    capability_id: str
    source_evidence_bindings: tuple[EvidenceBindingV1, ...]
    profile_id: str
    result_id: str
    command: tuple[str, ...]
    tool_id: str
    tool_version: str
    observed_artifact_identity: ArtifactIdentityV1
    contract_validation_status: Literal["pass", "error"]
    raw_findings: tuple[dict[str, JsonValue], ...] = ()
    blocking_reason_codes: tuple[str, ...] = ()

    @field_validator(
        "raw_status",
        "capability_id",
        "profile_id",
        "result_id",
        "tool_id",
        "tool_version",
    )
    @classmethod
    def _nonblank_text(cls, value: str, info: Any) -> str:
        return _nonblank_text(value, info.field_name)

    @field_validator("source_evidence_bindings")
    @classmethod
    def _canonical_source_evidence(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)

    @field_validator("command")
    @classmethod
    def _required_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _required_command_arguments(value, "trusted report")

    @field_validator("blocking_reason_codes")
    @classmethod
    def _canonical_blockers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_reason_codes(value)

    @field_validator("raw_findings")
    @classmethod
    def _finite_raw_findings(
        cls,
        value: tuple[dict[str, JsonValue], ...],
    ) -> tuple[dict[str, JsonValue], ...]:
        _reject_nonfinite_json(value, label="trusted raw findings")
        return value

    @model_validator(mode="after")
    def _fail_closed_shape(self) -> StaticTrustedStageReportV1:
        _validate_completed_stage_claim(
            status=self.status,
            raw_status=self.raw_status,
            contract_validation_status=self.contract_validation_status,
            raw_findings=self.raw_findings,
            blocking_reason_codes=self.blocking_reason_codes,
        )
        return self


class StaticQualificationEvidenceAdapter(Protocol):
    """Trusted byte resolver and validator-specific report interpreter."""

    def read_bytes(self, uri: str) -> bytes: ...

    def reopen_usd_artifact(self, uri: str) -> StaticResolvedUsdArtifactV1: ...

    def interpret_stage(
        self,
        *,
        stage: StaticStageName,
        validation_contract: str,
        artifacts: Mapping[StaticRetainedArtifactRole, bytes],
        resolved_output: StaticResolvedUsdArtifactV1,
        observed_artifact_identity: ArtifactIdentityV1,
    ) -> StaticTrustedStageReportV1: ...


class StaticStageExecutionEvidenceV1(_QualificationModel):
    hashes: StaticEvidenceHashesV1
    command: tuple[str, ...]
    tool_id: str
    tool_version: str
    contract_validation_status: Literal["pass", "error"]
    raw_report_path: str
    raw_report_sha256: str
    retained_artifacts: tuple[StaticRetainedArtifactV1, ...]

    @field_validator("tool_id", "tool_version", "raw_report_path")
    @classmethod
    def _nonblank_text(cls, value: str, info: Any) -> str:
        return _nonblank_text(value, info.field_name)

    @field_validator("command")
    @classmethod
    def _required_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _required_command_arguments(value, "executed")

    @field_validator("raw_report_sha256")
    @classmethod
    def _valid_raw_report_sha256(cls, value: str) -> str:
        return _validated_sha256(value, "raw_report_sha256")

    @field_validator("retained_artifacts")
    @classmethod
    def _complete_retained_artifacts(
        cls,
        value: tuple[StaticRetainedArtifactV1, ...],
    ) -> tuple[StaticRetainedArtifactV1, ...]:
        roles = tuple(artifact.role for artifact in value)
        if len(roles) != len(set(roles)):
            raise ValueError("retained artifact roles must be unique")
        if set(roles) != set(STATIC_RETAINED_ARTIFACT_ROLES):
            raise ValueError(
                "executed stages must retain contract, output, validator, result, "
                "and raw_report bytes"
            )
        return tuple(sorted(value, key=lambda artifact: artifact.role))

    @model_validator(mode="after")
    def _raw_report_is_retained(self) -> StaticStageExecutionEvidenceV1:
        raw_report = next(
            artifact
            for artifact in self.retained_artifacts
            if artifact.role == "raw_report"
        )
        if raw_report.uri != self.raw_report_path:
            raise ValueError("raw_report_path must name the retained raw report")
        return self


class StaticQualificationExecutedStageV1(_QualificationModel):
    stage: StaticStageName
    status: StaticExecutionStatus
    raw_status: str
    profile_id: str | None = None
    result_id: str | None = None
    validation_contract: str
    evidence: StaticStageExecutionEvidenceV1 | None = None
    retained_result: StaticStageResultV1 | None = None
    raw_findings: tuple[dict[str, JsonValue], ...] = ()
    blocking_reason_codes: tuple[str, ...] = ()

    @field_validator(
        "raw_status",
        "validation_contract",
        "profile_id",
        "result_id",
    )
    @classmethod
    def _nonblank_optional_text(
        cls,
        value: str | None,
        info: Any,
    ) -> str | None:
        if value is None:
            return value
        return _nonblank_text(value, info.field_name)

    @field_validator("blocking_reason_codes")
    @classmethod
    def _canonical_blockers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_reason_codes(value)

    @field_validator("raw_findings")
    @classmethod
    def _finite_raw_findings(
        cls,
        value: tuple[dict[str, JsonValue], ...],
    ) -> tuple[dict[str, JsonValue], ...]:
        _reject_nonfinite_json(value, label="executed raw findings")
        return value

    @model_validator(mode="after")
    def _result_shape(self) -> StaticQualificationExecutedStageV1:
        require_registered_static_validation_contract(
            stage=self.stage,
            value=self.validation_contract,
        )
        if self.retained_result is not None:
            if (
                self.evidence is not None
                or self.status not in {"pass", "fail"}
                or self.status != self.retained_result.status
                or self.profile_id != self.retained_result.profile_id
                or self.result_id != self.retained_result.result_id
                or self.blocking_reason_codes
                or self.raw_findings
            ):
                raise ValueError("retained result shape is inconsistent")
            return self
        if self.status in {"pass", "fail", "error"}:
            if (
                self.evidence is None
                or self.profile_id is None
                or self.result_id is None
            ):
                raise ValueError(
                    "pass/fail/error stages require profile/result IDs and "
                    "complete evidence"
                )
            _validate_completed_stage_claim(
                status=_require_completed_status(self.status),
                raw_status=self.raw_status,
                contract_validation_status=(self.evidence.contract_validation_status),
                raw_findings=self.raw_findings,
                blocking_reason_codes=self.blocking_reason_codes,
            )
        elif self.evidence is not None:
            raise ValueError("blocked/not_run/na stages must not carry run evidence")
        elif (self.profile_id is None) != (self.result_id is None):
            raise ValueError(
                "nonexecuted stages must carry both planned identity fields or neither"
            )
        if self.status in {"blocked", "not_run"} and not self.blocking_reason_codes:
            raise ValueError(f"{self.status} stages require a blocking reason")
        if self.status in {"not_run", "na"} and self.raw_findings:
            raise ValueError("not_run/na stages must not carry raw findings")
        if self.status == "na" and self.blocking_reason_codes:
            raise ValueError("na stages must not carry blocking reasons")
        if self.status == "na" and (
            self.profile_id is not None or self.result_id is not None
        ):
            raise ValueError("na stages must not claim planned identities")
        return self


class StaticQualificationExecutedStagesV1(_QualificationModel):
    contract: StaticQualificationExecutedStageV1
    authoring: StaticQualificationExecutedStageV1
    readback: StaticQualificationExecutedStageV1
    gate3a: StaticQualificationExecutedStageV1
    gate3b: StaticQualificationExecutedStageV1

    @model_validator(mode="after")
    def _stage_names_and_identity(
        self,
    ) -> StaticQualificationExecutedStagesV1:
        evidence = []
        output_uris = set()
        for stage in STATIC_STAGE_NAMES:
            result = getattr(self, stage)
            if result.stage != stage:
                raise ValueError(f"{stage} result must declare stage={stage!r}")
            if result.evidence is not None:
                evidence.append(result.evidence.hashes)
                output_uris.add(
                    next(
                        retained.uri
                        for retained in result.evidence.retained_artifacts
                        if retained.role == "output"
                    )
                )
        for field in (
            "source_sha256",
            "manifest_sha256",
            "contract_sha256",
            "output_sha256",
            "output_dependency_bundle_sha256",
        ):
            identities = {getattr(item, field) for item in evidence}
            if len(identities) > 1:
                raise ValueError(
                    "completed static stages must bind the same "
                    f"{field.removesuffix('_sha256')} identity"
                )
        if len(output_uris) > 1:
            raise ValueError(
                "completed static stages must bind the same output artifact URI"
            )
        return self


class StaticQualificationResultRowV1(_QualificationModel):
    capability_id: str
    row_admission_sha256: str
    evidence_bindings: tuple[EvidenceBindingV1, ...]
    stages: StaticQualificationExecutedStagesV1

    @field_validator("capability_id")
    @classmethod
    def _nonblank_capability(cls, value: str) -> str:
        return _nonblank_text(value, "capability_id")

    @field_validator("row_admission_sha256")
    @classmethod
    def _valid_row_admission_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _SHA256_RE.fullmatch(normalized) is None:
            raise ValueError("row_admission_sha256 must be a lowercase SHA-256 digest")
        return normalized

    @field_validator("evidence_bindings")
    @classmethod
    def _canonical_evidence(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)


class StaticQualificationResultBundleV1(_QualificationModel):
    schema_version: Literal["joint-agent-static-qualification-result-v1"]
    run_id: str
    run_plan_sha256: str
    capability_manifest_sha256: str
    corpus_sha256: str
    rows: tuple[StaticQualificationResultRowV1, ...]

    @field_validator("run_id")
    @classmethod
    def _nonblank_run_id(cls, value: str) -> str:
        return _nonblank_text(value, "run_id")

    @field_validator(
        "run_plan_sha256",
        "capability_manifest_sha256",
        "corpus_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)

    @field_validator("rows")
    @classmethod
    def _canonical_rows(
        cls,
        value: tuple[StaticQualificationResultRowV1, ...],
    ) -> tuple[StaticQualificationResultRowV1, ...]:
        return _canonical_unique_rows(value, label="result")


class StaticQualificationScorecardRowV1(_QualificationModel):
    capability_id: str
    row_admission_sha256: str
    evidence_bindings: tuple[EvidenceBindingV1, ...]
    stages: StaticQualificationExecutedStagesV1
    promotion_eligible: bool
    failure_categories: tuple[StaticFailureCategory, ...]
    blocking_reason_codes: tuple[str, ...]

    @field_validator("capability_id")
    @classmethod
    def _nonblank_capability(cls, value: str) -> str:
        return _nonblank_text(value, "capability_id")

    @field_validator("row_admission_sha256")
    @classmethod
    def _valid_row_admission_sha256(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _SHA256_RE.fullmatch(normalized) is None:
            raise ValueError("row_admission_sha256 must be a lowercase SHA-256 digest")
        return normalized

    @field_validator("evidence_bindings")
    @classmethod
    def _canonical_evidence(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)

    @field_validator("failure_categories")
    @classmethod
    def _canonical_failure_categories(
        cls,
        value: tuple[StaticFailureCategory, ...],
    ) -> tuple[StaticFailureCategory, ...]:
        order: tuple[StaticFailureCategory, ...] = (
            "contract",
            "authoring",
            "package",
            "static_schema",
            "unsupported_capability",
        )
        if len(value) != len(set(value)):
            raise ValueError("failure categories must be unique")
        return tuple(item for item in order if item in value)

    @field_validator("blocking_reason_codes")
    @classmethod
    def _canonical_blockers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_reason_codes(value)

    @model_validator(mode="after")
    def _promotion_matches_stages(self) -> StaticQualificationScorecardRowV1:
        all_pass = all(
            getattr(self.stages, stage).status == "pass"
            and getattr(self.stages, stage).evidence is not None
            and getattr(
                self.stages,
                stage,
            ).evidence.contract_validation_status
            == "pass"
            for stage in STATIC_STAGE_NAMES
        )
        if self.promotion_eligible and not all_pass:
            raise ValueError(
                "promotion_eligible requires every static stage to pass with "
                "trusted contract evidence"
            )
        if self.promotion_eligible == bool(self.blocking_reason_codes):
            raise ValueError(
                "eligible rows must have no blockers; ineligible rows require them"
            )
        if self.promotion_eligible == bool(self.failure_categories):
            raise ValueError(
                "eligible rows must have no failure categories; ineligible rows "
                "require them"
            )
        return self


class StaticQualificationScorecardSummaryV1(_QualificationModel):
    row_count: int = Field(ge=0)
    promotion_eligible_count: int = Field(ge=0)
    blocked_row_count: int = Field(ge=0)
    error_row_count: int = Field(ge=0)
    not_run_row_count: int = Field(ge=0)


class StaticQualificationScorecardV1(_QualificationModel):
    schema_version: Literal["joint-agent-static-qualification-scorecard-v1"]
    run_id: str
    run_plan_sha256: str
    result_bundle_sha256: str
    capability_manifest_sha256: str
    corpus_sha256: str
    frozen_0_5: FrozenQualificationLaneV1
    qualification_assets: tuple[EvidenceBindingV1, ...]
    rows: tuple[StaticQualificationScorecardRowV1, ...]
    summary: StaticQualificationScorecardSummaryV1

    @field_validator("run_id")
    @classmethod
    def _nonblank_run_id(cls, value: str) -> str:
        return _nonblank_text(value, "run_id")

    @field_validator(
        "run_plan_sha256",
        "result_bundle_sha256",
        "capability_manifest_sha256",
        "corpus_sha256",
    )
    @classmethod
    def _valid_sha256(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)

    @field_validator("qualification_assets")
    @classmethod
    def _canonical_qualification_assets(
        cls,
        value: tuple[EvidenceBindingV1, ...],
    ) -> tuple[EvidenceBindingV1, ...]:
        return _canonical_evidence_bindings(value)

    @field_validator("rows")
    @classmethod
    def _canonical_rows(
        cls,
        value: tuple[StaticQualificationScorecardRowV1, ...],
    ) -> tuple[StaticQualificationScorecardRowV1, ...]:
        return _canonical_unique_rows(value, label="scorecard")

    @model_validator(mode="after")
    def _summary_matches_rows(self) -> StaticQualificationScorecardV1:
        expected = StaticQualificationScorecardSummaryV1(
            row_count=len(self.rows),
            promotion_eligible_count=sum(row.promotion_eligible for row in self.rows),
            blocked_row_count=sum(
                any(
                    getattr(row.stages, stage).status == "blocked"
                    for stage in STATIC_STAGE_NAMES
                )
                for row in self.rows
            ),
            error_row_count=sum(
                any(
                    getattr(row.stages, stage).status == "error"
                    for stage in STATIC_STAGE_NAMES
                )
                for row in self.rows
            ),
            not_run_row_count=sum(
                any(
                    getattr(row.stages, stage).status == "not_run"
                    for stage in STATIC_STAGE_NAMES
                )
                for row in self.rows
            ),
        )
        if self.summary != expected:
            raise ValueError("scorecard summary does not match rows")
        return self


def build_static_qualification_scaffold(
    *,
    repo_root: str | Path,
) -> StaticQualificationMatrixV1:
    """Build the initial exact-roster matrix from packaged policy authority."""

    return _build_static_qualification_scaffold(
        _load_packaged_capability_manifest(),
        repo_root=repo_root,
    )


def _static_contract_lane(
    row: CapabilityRowV1,
) -> StaticQualificationContractLane:
    """Select the contract lane from manifest-owned admission shape."""

    if row.qualification.selector_admissions is not None:
        return ARTICULATION_V2_CONTRACT_LANE
    return RELEASE_GATE_V0_5_CONTRACT_LANE


def _row_static_validation_contract(
    row: CapabilityRowV1,
    stage: StaticStageName,
) -> str:
    return static_validation_contract(_static_contract_lane(row), stage)


def _build_static_qualification_scaffold(
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
) -> StaticQualificationMatrixV1:
    """Internal manifest-parameterized scaffold builder for policy tooling."""

    root = Path(repo_root)
    validate_corpus_bindings(loaded, repo_root=root)
    manifest = loaded.manifest
    baseline_path = resolve_repository_path(
        root,
        manifest.frozen_0_5.gate3_baseline,
        label="frozen Gate 3 baseline",
    )
    baseline = _load_json_object(baseline_path, label="frozen Gate 3 baseline")
    artifact_set_sha256 = _frozen_artifact_set_sha256(baseline)
    source = baseline.get("source")
    if not isinstance(source, dict):
        raise CapabilityManifestError("frozen Gate 3 baseline source is missing")
    profiles = baseline.get("profiles")
    if not isinstance(profiles, dict):
        raise CapabilityManifestError("frozen Gate 3 profiles are missing")

    rows = tuple(
        _build_row(
            row,
            artifact_set_sha256=artifact_set_sha256,
            scoreboard_sha256=_required_sha256(source, "scoreboard_sha256"),
            gate3a_sha256=_required_sha256(source, "gate3a_results_sha256"),
            gate3b_sha256=_required_sha256(source, "gate3b_results_sha256"),
            gate3a_profile=_required_text(profiles, "gate3a"),
            gate3b_profile=_required_text(profiles, "gate3b"),
        )
        for row in manifest.capabilities
    )
    completed_with_findings = sum(
        all(
            stage.status in {"pass", "fail"}
            for stage in (
                row.stages.contract,
                row.stages.authoring,
                row.stages.readback,
                row.stages.gate3a,
                row.stages.gate3b,
            )
        )
        and not row.static_qualified
        for row in rows
    )
    return StaticQualificationMatrixV1(
        schema_version=STATIC_QUALIFICATION_SCHEMA_VERSION,
        matrix_version=manifest.manifest_version,
        capability_manifest_version=manifest.manifest_version,
        capability_manifest_sha256=loaded.sha256,
        corpus_id=manifest.corpus.corpus_id,
        corpus_sha256=manifest.corpus.corpus_sha256,
        coverage_report_sha256=manifest.corpus.coverage_report_sha256,
        frozen_0_5_release_manifest_sha256=(
            manifest.frozen_0_5.release_manifest_sha256
        ),
        frozen_0_5_gate3_baseline_sha256=(manifest.frozen_0_5.gate3_baseline_sha256),
        rows=rows,
        summary=StaticQualificationSummaryV1(
            row_count=len(rows),
            completed_with_findings_count=completed_with_findings,
            not_run_row_count=sum(
                any(
                    stage.status == "not_run"
                    for stage in (
                        row.stages.contract,
                        row.stages.authoring,
                        row.stages.readback,
                        row.stages.gate3a,
                        row.stages.gate3b,
                    )
                )
                for row in rows
            ),
            static_qualified_count=sum(row.static_qualified for row in rows),
        ),
    )


def load_static_qualification_matrix(
    path: str | Path,
) -> StaticQualificationMatrixV1:
    """Load a strict static qualification matrix."""

    document = _load_json_object(
        Path(path),
        label="static qualification matrix",
    )
    try:
        return StaticQualificationMatrixV1.model_validate(document)
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static qualification matrix is invalid: {exc}"
        ) from exc


def validate_static_qualification_matrix(
    matrix: StaticQualificationMatrixV1,
    *,
    repo_root: str | Path,
) -> None:
    """Verify a caller model against the packaged manifest authority."""

    canonical_matrix = _strict_revalidate_model(
        matrix,
        StaticQualificationMatrixV1,
        label="static qualification matrix",
    )
    _validate_static_qualification_matrix_against_manifest(
        canonical_matrix,
        _load_packaged_capability_manifest(),
        repo_root=repo_root,
    )


def _validate_static_qualification_matrix_against_manifest(
    matrix: StaticQualificationMatrixV1,
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
) -> None:
    """Internal manifest-parameterized matrix validator."""

    manifest = loaded.manifest
    expected_header = {
        "matrix_version": manifest.manifest_version,
        "capability_manifest_version": manifest.manifest_version,
        "capability_manifest_sha256": loaded.sha256,
        "corpus_id": manifest.corpus.corpus_id,
        "corpus_sha256": manifest.corpus.corpus_sha256,
        "coverage_report_sha256": manifest.corpus.coverage_report_sha256,
        "frozen_0_5_release_manifest_sha256": (
            manifest.frozen_0_5.release_manifest_sha256
        ),
        "frozen_0_5_gate3_baseline_sha256": (manifest.frozen_0_5.gate3_baseline_sha256),
    }
    for field, expected in expected_header.items():
        if getattr(matrix, field) != expected:
            raise CapabilityManifestError(
                f"qualification matrix {field} is stale or identity-mismatched"
            )

    manifest_rows = {row.capability_id: row for row in manifest.capabilities}
    matrix_rows = {row.capability_id: row for row in matrix.rows}
    scaffold_rows = {
        row.capability_id: row
        for row in _build_static_qualification_scaffold(
            loaded,
            repo_root=repo_root,
        ).rows
    }
    if set(matrix_rows) != set(manifest_rows):
        raise CapabilityManifestError(
            "qualification matrix must cover every manifest row exactly; "
            f"missing={sorted(set(manifest_rows) - set(matrix_rows))}, "
            f"extra={sorted(set(matrix_rows) - set(manifest_rows))}"
        )
    for capability_id, manifest_row in manifest_rows.items():
        matrix_row = matrix_rows[capability_id]
        expected_bindings = _row_evidence_bindings(manifest_row)
        if matrix_row.evidence_bindings != expected_bindings:
            raise CapabilityManifestError(
                f"qualification evidence drifted for {capability_id}"
            )
        frozen = any(
            binding.evidence_role == "frozen_0_5" for binding in expected_bindings
        )
        if frozen and matrix_row != scaffold_rows[capability_id]:
            raise CapabilityManifestError(
                f"frozen 0.5 qualification row drifted for {capability_id}"
            )
        # A scheduled row whose stages all passed is required by the row model
        # to set `static_qualified`, so treating that as exceeding the manifest
        # made no legal matrix expressible for a run in flight.
        if matrix_row.static_qualified and (
            manifest_row.qualification.static_status not in {"pass", "scheduled"}
        ):
            raise CapabilityManifestError(
                f"{capability_id} cannot exceed its manifest static status"
            )
        # Letting `scheduled` rows reach `static_qualified` above would
        # otherwise accept a caller-supplied matrix whose stages carry
        # fabricated passing identities. The 0.5 lane is exempt because it
        # carries retained identities the v1 stage admission never enumerated.
        if not frozen:
            for stage in STATIC_STAGE_NAMES:
                stage_result = getattr(matrix_row.stages, stage)
                _require_admitted_stage_identity(
                    label=f"{capability_id}/{stage}",
                    validation_contract=None,
                    profile_id=stage_result.profile_id,
                    result_id=stage_result.result_id,
                    admission=getattr(manifest_row.qualification.static_stages, stage),
                    require_present=False,
                )
        if matrix_row.dynamic_qualified:
            raise CapabilityManifestError(
                "static qualification must never imply dynamic qualification"
            )


def canonical_matrix_json(matrix: StaticQualificationMatrixV1) -> str:
    """Serialize a stable, reviewable matrix document."""

    return _canonical_model_json(
        _strict_revalidate_model(
            matrix,
            StaticQualificationMatrixV1,
            label="static qualification matrix",
        )
    )


def build_static_qualification_run_plan_scaffold(
    *,
    repo_root: str | Path,
) -> StaticQualificationRunPlanV1:
    """Build an exact-roster plan without scheduling unimplemented work.

    The immutable 0.5 lane points at its retained evidence. Every post-0.5 stage
    is explicit and blocked until an implementation work package replaces that
    stage with an independently reviewed ``run`` plan.
    """

    return _build_static_qualification_run_plan_scaffold(
        _load_packaged_capability_manifest(),
        repo_root=repo_root,
    )


def _build_static_qualification_run_plan_scaffold(
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
) -> StaticQualificationRunPlanV1:
    """Internal manifest-parameterized plan builder for policy tooling."""

    matrix = _build_static_qualification_scaffold(loaded, repo_root=repo_root)
    matrix_rows = {row.capability_id: row for row in matrix.rows}
    plan_rows = []
    for manifest_row in loaded.manifest.capabilities:
        evidence = _row_evidence_bindings(manifest_row)
        frozen = any(item.evidence_role == "frozen_0_5" for item in evidence)
        matrix_row = matrix_rows[manifest_row.capability_id]
        stages: dict[str, StaticQualificationPlanStageV1] = {}
        for stage in STATIC_STAGE_NAMES:
            retained = getattr(matrix_row.stages, stage)
            validation_contract = _row_static_validation_contract(
                manifest_row,
                stage,
            )
            if frozen:
                stages[stage] = StaticQualificationPlanStageV1(
                    stage=stage,
                    execution="retain_frozen_0_5",
                    validation_contract=validation_contract,
                    profile_id=retained.profile_id,
                    result_id=retained.result_id,
                    retained_result=retained,
                )
            else:
                # A row whose lane says `scheduled` has admitted a run, so the
                # scaffold must not report its stages as unschedulable; it has
                # simply not been handed an executable plan yet. Read that per
                # stage rather than from the aggregate lane: a partially
                # scheduled row leaves some stages with no admitted profile or
                # result, and those were never scheduled at all.
                stage_admission = getattr(
                    manifest_row.qualification.static_stages, stage
                )
                awaiting = (
                    manifest_row.qualification.static_status == "scheduled"
                    and bool(stage_admission.admitted_profile_ids())
                    and bool(stage_admission.admitted_result_ids())
                )
                blockers = tuple(
                    sorted(
                        {
                            *manifest_row.reason_codes,
                            f"{stage}_awaiting_plan"
                            if awaiting
                            else f"{stage}_not_scheduled",
                        }
                    )
                )
                stages[stage] = StaticQualificationPlanStageV1(
                    stage=stage,
                    execution="blocked",
                    validation_contract=validation_contract,
                    blocking_reason_codes=blockers,
                )
        plan_rows.append(
            StaticQualificationPlanRowV1(
                capability_id=manifest_row.capability_id,
                row_admission_sha256=manifest_row.admission_sha256,
                evidence_bindings=evidence,
                stages=StaticQualificationPlanStagesV1.model_validate(stages),
            )
        )
    manifest = loaded.manifest
    plan = StaticQualificationRunPlanV1(
        schema_version=STATIC_QUALIFICATION_RUN_PLAN_SCHEMA_VERSION,
        plan_version=manifest.manifest_version,
        capability_manifest_version=manifest.manifest_version,
        capability_manifest_sha256=loaded.sha256,
        corpus_id=manifest.corpus.corpus_id,
        corpus_sha256=manifest.corpus.corpus_sha256,
        coverage_report_sha256=manifest.corpus.coverage_report_sha256,
        frozen_0_5=FrozenQualificationLaneV1(
            release_manifest_sha256=manifest.frozen_0_5.release_manifest_sha256,
            gate3_baseline_sha256=manifest.frozen_0_5.gate3_baseline_sha256,
            asset_count=manifest.frozen_0_5.asset_count,
            joint_count=manifest.frozen_0_5.joint_count,
        ),
        qualification_assets=qualification_evidence_roster(
            loaded,
            repo_root=repo_root,
        ),
        rows=tuple(plan_rows),
    )
    _validate_static_qualification_run_plan_against_manifest(
        plan,
        loaded,
        repo_root=repo_root,
        executed=False,
    )
    return plan


def build_static_qualification_run_plan(
    *,
    repo_root: str | Path,
    requests: Collection[StaticQualificationRunRequestV1],
) -> StaticQualificationRunPlanV1:
    """Build an executable plan by scheduling manifest-admitted stages.

    The scaffold stays the base: every stage nobody asked for keeps whatever
    the manifest says about it, blockers included. A requested stage is
    replaced by a ``run`` stage carrying the profile, result, command, and tool
    identity the caller intends to execute, and the whole plan is then handed
    to the same manifest validator every other plan goes through. That is the
    only admission check -- this builder deliberately owns no membership logic
    of its own, so a stage can never be planned with an identity the manifest
    did not admit for that exact stage.
    """

    return _build_static_qualification_run_plan(
        _load_packaged_capability_manifest(),
        repo_root=repo_root,
        requests=requests,
    )


def _build_static_qualification_run_plan(
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
    requests: Collection[StaticQualificationRunRequestV1],
) -> StaticQualificationRunPlanV1:
    """Internal manifest-parameterized executable plan builder."""

    scaffold = _build_static_qualification_run_plan_scaffold(
        loaded,
        repo_root=repo_root,
    )
    requested: dict[str, StaticQualificationRunRequestV1] = {}
    for supplied in requests:
        # A caller can build a request with `model_construct` or
        # `model_copy(update=...)`, neither of which runs validators, so an
        # unsupported stage would reach a bare `KeyError` and a duplicated
        # stage would be silently collapsed by the mapping below. Every other
        # public policy entry point in this module dumps and reparses first.
        request = _strict_revalidate_model(
            supplied,
            StaticQualificationRunRequestV1,
            label="static qualification run request",
        )
        if request.capability_id in requested:
            raise CapabilityManifestError(
                "static qualification run request repeats capability "
                f"{request.capability_id}"
            )
        requested[request.capability_id] = request
    unknown = set(requested) - {row.capability_id for row in scaffold.rows}
    if unknown:
        raise CapabilityManifestError(
            "static qualification run request names capabilities the manifest "
            f"does not carry: {sorted(unknown)}"
        )
    manifest_rows = {row.capability_id: row for row in loaded.manifest.capabilities}
    rows = []
    for row in scaffold.rows:
        scheduled = requested.get(row.capability_id)
        if scheduled is None:
            rows.append(row)
            continue
        stages: dict[str, StaticQualificationPlanStageV1] = {
            stage: getattr(row.stages, stage) for stage in STATIC_STAGE_NAMES
        }
        for stage_request in scheduled.stages:
            stages[stage_request.stage] = StaticQualificationPlanStageV1(
                stage=stage_request.stage,
                execution="run",
                validation_contract=_row_static_validation_contract(
                    manifest_rows[row.capability_id],
                    stage_request.stage,
                ),
                profile_id=stage_request.profile_id,
                result_id=stage_request.result_id,
                command=stage_request.command,
                tool_id=stage_request.tool_id,
                tool_version=stage_request.tool_version,
            )
        rows.append(
            row.model_copy(
                update={
                    "stages": StaticQualificationPlanStagesV1.model_validate(stages)
                }
            )
        )
    plan = _strict_revalidate_model(
        scaffold.model_copy(update={"rows": tuple(rows)}),
        StaticQualificationRunPlanV1,
        label="static qualification run plan",
    )
    _validate_static_qualification_run_plan_against_manifest(
        plan,
        loaded,
        repo_root=repo_root,
    )
    return plan


def load_static_qualification_run_request(
    path: str | Path,
) -> tuple[StaticQualificationRunRequestV1, ...]:
    """Load the reviewed set of stages an operator intends to execute."""

    document = _load_json_object(
        Path(path),
        label="static qualification run request",
    )
    try:
        return StaticQualificationRunRequestDocumentV1.model_validate(document).requests
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static qualification run request is invalid: {exc}"
        ) from exc


def load_static_qualification_run_plan(
    path: str | Path,
) -> StaticQualificationRunPlanV1:
    """Load a strict static qualification run plan."""

    document = _load_json_object(Path(path), label="static qualification run plan")
    try:
        return StaticQualificationRunPlanV1.model_validate(document)
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static qualification run plan is invalid: {exc}"
        ) from exc


def validate_static_qualification_run_plan(
    plan: StaticQualificationRunPlanV1,
    *,
    repo_root: str | Path,
) -> None:
    """Fail closed against packaged authority and strict caller revalidation."""

    canonical_plan = _strict_revalidate_model(
        plan,
        StaticQualificationRunPlanV1,
        label="static qualification run plan",
    )
    _validate_static_qualification_run_plan_against_manifest(
        canonical_plan,
        _load_packaged_capability_manifest(),
        repo_root=repo_root,
    )


def _validate_attested_outcomes_against_plan(
    loaded: LoadedCapabilityManifest,
    plan: StaticQualificationRunPlanV1,
) -> None:
    """Require a recorded outcome to name the plan row it was run against.

    The manifest model can only bind an attestation structurally, to the row it
    sits on: at load time there is no plan to compare against, because a plan is
    something a run consumes rather than an input to manifest validation. So
    possession is enforced here, where the plan exists. Without it the
    attestation's digests are format-checked and never compared, and fabricated
    values satisfy it.
    """

    for row in loaded.manifest.capabilities:
        attestation = row.qualification.static_attestation
        if attestation is None:
            continue
        expected = attested_plan_row_sha256(plan, row.capability_id)
        if attestation.run_plan_sha256 != expected:
            raise CapabilityManifestError(
                f"{row.capability_id} records an outcome attested to a "
                "different run plan row than the one being validated"
            )


def _validate_static_qualification_run_plan_against_manifest(
    plan: StaticQualificationRunPlanV1,
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
    executed: bool = True,
) -> None:
    """Internal manifest-parameterized plan validator."""

    validate_corpus_bindings(loaded, repo_root=repo_root)
    if executed:
        # A recorded outcome attests to the plan that was *run*. The scaffold is
        # a template derived from the manifest -- its stages carry no profile,
        # result, or command identity -- so an attestation can never name it.
        _validate_attested_outcomes_against_plan(loaded, plan)
    manifest = loaded.manifest
    expected_header: Mapping[str, Any] = {
        "plan_version": manifest.manifest_version,
        "capability_manifest_version": manifest.manifest_version,
        "capability_manifest_sha256": loaded.sha256,
        "corpus_id": manifest.corpus.corpus_id,
        "corpus_sha256": manifest.corpus.corpus_sha256,
        "coverage_report_sha256": manifest.corpus.coverage_report_sha256,
        "frozen_0_5": FrozenQualificationLaneV1(
            release_manifest_sha256=manifest.frozen_0_5.release_manifest_sha256,
            gate3_baseline_sha256=manifest.frozen_0_5.gate3_baseline_sha256,
            asset_count=manifest.frozen_0_5.asset_count,
            joint_count=manifest.frozen_0_5.joint_count,
        ),
        "qualification_assets": qualification_evidence_roster(
            loaded,
            repo_root=repo_root,
        ),
    }
    for field, expected in expected_header.items():
        if getattr(plan, field) != expected:
            raise CapabilityManifestError(
                f"static qualification run plan {field} is stale or identity-mismatched"
            )

    manifest_rows = {row.capability_id: row for row in manifest.capabilities}
    plan_rows = {row.capability_id: row for row in plan.rows}
    _require_exact_capability_rows(
        expected=set(manifest_rows),
        actual=set(plan_rows),
        label="static qualification run plan",
    )
    retained_matrix = _build_static_qualification_scaffold(
        loaded,
        repo_root=repo_root,
    )
    retained_rows = {row.capability_id: row for row in retained_matrix.rows}
    for capability_id, manifest_row in manifest_rows.items():
        plan_row = plan_rows[capability_id]
        if plan_row.row_admission_sha256 != manifest_row.admission_sha256:
            # Without this, the field is only syntax-checked here: a caller
            # could bind a plan to any 64-character digest, recompute the plan
            # hash, and have public validation accept artifacts planned against
            # an admission identity the manifest never granted.
            raise CapabilityManifestError(
                f"static qualification plan row {capability_id} claims "
                "admission digest "
                f"{plan_row.row_admission_sha256} but the manifest admitted "
                f"{manifest_row.admission_sha256}"
            )
        bindings = _row_evidence_bindings(manifest_row)
        if plan_row.evidence_bindings != bindings:
            raise CapabilityManifestError(
                f"static qualification plan evidence drifted for {capability_id}"
            )
        frozen = any(item.evidence_role == "frozen_0_5" for item in bindings)
        for stage in STATIC_STAGE_NAMES:
            stage_plan = getattr(plan_row.stages, stage)
            expected_validation_contract = _row_static_validation_contract(
                manifest_row,
                stage,
            )
            if stage_plan.validation_contract != expected_validation_contract:
                raise CapabilityManifestError(
                    f"{capability_id}/{stage} uses validation contract "
                    f"{stage_plan.validation_contract!r}; the manifest lane admits "
                    f"{expected_validation_contract!r}"
                )
            if frozen:
                expected = getattr(retained_rows[capability_id].stages, stage)
                if (
                    stage_plan.execution != "retain_frozen_0_5"
                    or stage_plan.retained_result != expected
                ):
                    raise CapabilityManifestError(
                        f"immutable 0.5 stage drifted for {capability_id}/{stage}"
                    )
            elif stage_plan.execution == "retain_frozen_0_5":
                raise CapabilityManifestError(
                    f"post-0.5 row {capability_id}/{stage} cannot consume the "
                    "frozen 0.5 lane"
                )
            elif stage_plan.execution == "blocked":
                missing_blockers = set(manifest_row.reason_codes) - set(
                    stage_plan.blocking_reason_codes
                )
                if missing_blockers:
                    raise CapabilityManifestError(
                        f"{capability_id}/{stage} dropped undischarged manifest "
                        f"blockers: {sorted(missing_blockers)}"
                    )
            else:
                _validate_planned_stage_admission(
                    capability=manifest_row,
                    stage=stage,
                    stage_plan=stage_plan,
                )


def canonical_run_plan_json(plan: StaticQualificationRunPlanV1) -> str:
    """Serialize a deterministic machine-reviewable run plan."""

    return _canonical_model_json(
        _strict_revalidate_model(
            plan,
            StaticQualificationRunPlanV1,
            label="static qualification run plan",
        )
    )


def static_qualification_run_plan_sha256(
    plan: StaticQualificationRunPlanV1,
) -> str:
    """Hash the canonical run-plan bytes used by result ingestion."""

    return hashlib.sha256(canonical_run_plan_json(plan).encode("utf-8")).hexdigest()


def attested_plan_row_sha256(
    plan: StaticQualificationRunPlanV1,
    capability_id: str,
) -> str:
    """Digest one row's plan entry as an attestation can durably name it.

    A plan-wide digest cannot work here: the plan carries every capability, so
    any other row's edit moves it. Even this row's own entry moves when its
    advertisement changes, because `blocking_reason_codes` derives from
    `reason_codes`.

    This digests the *executable* row -- the plan a runner actually carried
    out -- not the manifest-derived scaffold. So it covers exactly what was
    admitted and planned for one capability --
    its `row_admission_sha256`, evidence bindings, and per-stage execution,
    profile, result, command, and tool identity -- while eliding the
    advertisement-derived blockers. Verified properties: stable when an
    unrelated row is edited, stable when this row records its own outcome, and
    moves when this row's admitted identity changes.
    """

    row = next(
        (
            candidate
            for candidate in plan.rows
            if candidate.capability_id == capability_id
        ),
        None,
    )
    if row is None:
        raise CapabilityManifestError(
            f"run plan has no row for capability {capability_id!r}"
        )
    document = json.loads(row.model_dump_json())
    stages = document.get("stages")
    if isinstance(stages, dict):
        for stage in stages.values():
            if isinstance(stage, dict):
                stage.pop("blocking_reason_codes", None)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attested_outcome_row_sha256(
    scorecard: StaticQualificationScorecardV1,
    capability_id: str,
) -> str:
    """Digest one row's outcome as an attestation can durably name it.

    Hashing the whole scorecard cannot work: it carries every row, and its
    header embeds the manifest digest, so recording this row's outcome would
    invalidate the very artifact the attestation names. Even this row alone
    embeds that digest once per stage, in `evidence.hashes.manifest_sha256`.

    So this covers the row's outcome -- admitted identity, evidence bindings,
    verdict, and per-stage status, profile, result, and artifact hashes --
    eliding only the whole-manifest digest, which is the sole circular field.
    Verified: stable when this row records its own outcome, and moves when the
    outcome itself changes.
    """

    row = next(
        (
            candidate
            for candidate in scorecard.rows
            if candidate.capability_id == capability_id
        ),
        None,
    )
    if row is None:
        raise CapabilityManifestError(
            f"scorecard has no row for capability {capability_id!r}"
        )
    return _outcome_row_sha256(row)


def _outcome_row_sha256(row: StaticQualificationScorecardRowV1) -> str:
    """Hash one already-resolved scorecard row. See the public wrapper above."""

    document = json.loads(row.model_dump_json())
    # Walk the declared stage names rather than guessing at the shape: the row
    # model fixes both, so defensive isinstance guards here would only add
    # branches nothing can reach. A stage that never ran carries no evidence.
    for stage in STATIC_STAGE_NAMES:
        evidence = document["stages"][stage]["evidence"]
        if evidence is not None:
            evidence["hashes"].pop("manifest_sha256", None)
    payload = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def static_qualification_source_sha256(
    bindings: tuple[EvidenceBindingV1, ...],
) -> str:
    """Hash one row's exact manifest-bound source evidence set."""

    try:
        return _evidence_identity_set_sha256(bindings)
    except Exception as exc:
        raise CapabilityManifestError(
            f"static qualification source evidence is invalid: {exc}"
        ) from exc


def load_static_qualification_result_bundle(
    path: str | Path,
) -> StaticQualificationResultBundleV1:
    """Load a strict result bundle without interpreting validator evidence."""

    document = _load_json_object(
        Path(path),
        label="static qualification result bundle",
    )
    try:
        return StaticQualificationResultBundleV1.model_validate(document)
    except ValueError as exc:
        raise CapabilityManifestError(
            f"static qualification result bundle is invalid: {exc}"
        ) from exc


def validate_static_qualification_result_bundle(
    bundle: StaticQualificationResultBundleV1,
    plan: StaticQualificationRunPlanV1,
    *,
    repo_root: str | Path,
    evidence_adapter: StaticQualificationEvidenceAdapter,
) -> None:
    """Validate untrusted caller models against packaged policy and retained bytes."""

    canonical_plan = _strict_revalidate_model(
        plan,
        StaticQualificationRunPlanV1,
        label="static qualification run plan",
    )
    canonical_bundle = _strict_revalidate_model(
        bundle,
        StaticQualificationResultBundleV1,
        label="static qualification result bundle",
    )
    _validate_static_qualification_result_bundle_against_manifest(
        canonical_bundle,
        canonical_plan,
        _load_packaged_capability_manifest(),
        repo_root=repo_root,
        evidence_adapter=evidence_adapter,
    )


def _validate_static_qualification_result_bundle_against_manifest(
    bundle: StaticQualificationResultBundleV1,
    plan: StaticQualificationRunPlanV1,
    loaded: LoadedCapabilityManifest,
    *,
    repo_root: str | Path,
    evidence_adapter: StaticQualificationEvidenceAdapter,
) -> None:
    """Internal manifest-parameterized result validator."""

    _validate_static_qualification_run_plan_against_manifest(
        plan,
        loaded,
        repo_root=repo_root,
    )
    expected_header = {
        "run_plan_sha256": static_qualification_run_plan_sha256(plan),
        "capability_manifest_sha256": loaded.sha256,
        "corpus_sha256": loaded.manifest.corpus.corpus_sha256,
    }
    for field, expected in expected_header.items():
        if getattr(bundle, field) != expected:
            raise CapabilityManifestError(
                f"qualification result {field} is stale or identity-mismatched"
            )
    plan_rows = {row.capability_id: row for row in plan.rows}
    result_rows = {row.capability_id: row for row in bundle.rows}
    _require_exact_capability_rows(
        expected=set(plan_rows),
        actual=set(result_rows),
        label="static qualification result",
    )
    for capability_id, plan_row in plan_rows.items():
        result_row = result_rows[capability_id]
        if result_row.row_admission_sha256 != plan_row.row_admission_sha256:
            # The plan row is already bound to the manifest above, so matching
            # it here transitively binds the result. Without this the check
            # lived only in `ingest_static_qualification_results`, leaving the
            # public bundle validator accepting a result whose admission
            # identity never matched the plan it claims to answer.
            raise CapabilityManifestError(
                f"static qualification result row {capability_id} claims "
                "admission digest "
                f"{result_row.row_admission_sha256} but its plan row was "
                f"admitted as {plan_row.row_admission_sha256}"
            )
        if result_row.evidence_bindings != plan_row.evidence_bindings:
            raise CapabilityManifestError(
                f"qualification result evidence drifted for {capability_id}"
            )
        expected_source_sha256 = _evidence_identity_set_sha256(
            plan_row.evidence_bindings
        )
        for stage in STATIC_STAGE_NAMES:
            stage_plan = getattr(plan_row.stages, stage)
            stage_result = getattr(result_row.stages, stage)
            _validate_stage_result_against_plan(
                capability_id=capability_id,
                stage=stage,
                stage_plan=stage_plan,
                stage_result=stage_result,
                expected_source_bindings=plan_row.evidence_bindings,
                expected_source_sha256=expected_source_sha256,
                manifest_sha256=loaded.sha256,
                evidence_adapter=evidence_adapter,
            )


def ingest_static_qualification_results(
    bundle: StaticQualificationResultBundleV1,
    plan: StaticQualificationRunPlanV1,
    *,
    repo_root: str | Path,
    evidence_adapter: StaticQualificationEvidenceAdapter,
    promote_capability_ids: Collection[str] = (),
) -> StaticQualificationScorecardV1:
    """Produce a scorecard using only packaged policy and trusted reopened bytes."""

    canonical_plan = _strict_revalidate_model(
        plan,
        StaticQualificationRunPlanV1,
        label="static qualification run plan",
    )
    canonical_bundle = _strict_revalidate_model(
        bundle,
        StaticQualificationResultBundleV1,
        label="static qualification result bundle",
    )
    loaded = _load_packaged_capability_manifest()
    _validate_static_qualification_result_bundle_against_manifest(
        canonical_bundle,
        canonical_plan,
        loaded,
        repo_root=repo_root,
        evidence_adapter=evidence_adapter,
    )
    bundle = canonical_bundle
    plan = canonical_plan
    score_rows = []
    eligible_ids = set()
    manifest_rows = {row.capability_id: row for row in loaded.manifest.capabilities}
    for result_row in bundle.rows:
        manifest_row = manifest_rows[result_row.capability_id]
        # Bind the run to the row's ADMITTED identity, not the whole-manifest
        # digest. Recording this row's outcome, or editing any other row, must
        # not invalidate a completed run -- which is exactly what comparing
        # against `loaded.sha256` did.
        if result_row.row_admission_sha256 != manifest_row.admission_sha256:
            raise CapabilityManifestError(
                f"{result_row.capability_id} result is bound to a different row "
                "admission digest than the manifest row it claims"
            )
        blockers = _result_blocking_reason_codes(result_row, manifest_row)
        eligible = not blockers
        if eligible:
            eligible_ids.add(result_row.capability_id)
        score_rows.append(
            StaticQualificationScorecardRowV1(
                capability_id=result_row.capability_id,
                row_admission_sha256=result_row.row_admission_sha256,
                evidence_bindings=result_row.evidence_bindings,
                stages=result_row.stages,
                promotion_eligible=eligible,
                failure_categories=_result_failure_categories(
                    result_row,
                    manifest_row,
                ),
                blocking_reason_codes=blockers,
            )
        )
    requested = set(promote_capability_ids)
    unknown = requested - {row.capability_id for row in bundle.rows}
    if unknown:
        raise CapabilityManifestError(
            f"promotion requested for unknown capabilities: {sorted(unknown)}"
        )
    rejected = requested - eligible_ids
    if rejected:
        raise CapabilityManifestError(
            "static promotion rejected; incomplete, blocked, error, NOT_RUN, "
            f"identity-mismatched, or finding-bearing rows: {sorted(rejected)}"
        )
    result_sha256 = hashlib.sha256(
        canonical_result_bundle_json(bundle).encode("utf-8")
    ).hexdigest()
    rows = tuple(score_rows)
    summary = StaticQualificationScorecardSummaryV1(
        row_count=len(rows),
        promotion_eligible_count=sum(row.promotion_eligible for row in rows),
        blocked_row_count=sum(
            any(
                getattr(row.stages, stage).status == "blocked"
                for stage in STATIC_STAGE_NAMES
            )
            for row in rows
        ),
        error_row_count=sum(
            any(
                getattr(row.stages, stage).status == "error"
                for stage in STATIC_STAGE_NAMES
            )
            for row in rows
        ),
        not_run_row_count=sum(
            any(
                getattr(row.stages, stage).status == "not_run"
                for stage in STATIC_STAGE_NAMES
            )
            for row in rows
        ),
    )
    scorecard = StaticQualificationScorecardV1(
        schema_version=STATIC_QUALIFICATION_SCORECARD_SCHEMA_VERSION,
        run_id=bundle.run_id,
        run_plan_sha256=bundle.run_plan_sha256,
        result_bundle_sha256=result_sha256,
        capability_manifest_sha256=bundle.capability_manifest_sha256,
        corpus_sha256=bundle.corpus_sha256,
        frozen_0_5=plan.frozen_0_5,
        qualification_assets=plan.qualification_assets,
        rows=rows,
        summary=summary,
    )
    return _strict_revalidate_model(
        scorecard,
        StaticQualificationScorecardV1,
        label="static qualification scorecard",
    )


def canonical_result_bundle_json(
    bundle: StaticQualificationResultBundleV1,
) -> str:
    """Serialize an immutable result bundle with raw findings intact."""

    return _canonical_model_json(
        _strict_revalidate_model(
            bundle,
            StaticQualificationResultBundleV1,
            label="static qualification result bundle",
        )
    )


def canonical_scorecard_json(
    scorecard: StaticQualificationScorecardV1,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Serialize the deterministic machine scorecard."""

    return _canonical_model_json(
        _revalidate_scorecard_policy(scorecard, repo_root=repo_root)
    )


def render_static_qualification_scorecard(
    scorecard: StaticQualificationScorecardV1,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Render a deterministic human scorecard with provenance and commands."""

    scorecard = _revalidate_scorecard_policy(scorecard, repo_root=repo_root)
    lines = [
        "# Joint Agent static qualification scorecard",
        "",
        f"- Run: `{scorecard.run_id}`",
        f"- Run plan SHA-256: `{scorecard.run_plan_sha256}`",
        f"- Result bundle SHA-256: `{scorecard.result_bundle_sha256}`",
        (f"- Capability manifest SHA-256: `{scorecard.capability_manifest_sha256}`"),
        f"- Corpus SHA-256: `{scorecard.corpus_sha256}`",
        (
            "- Frozen 0.5 lane: "
            f"{scorecard.frozen_0_5.asset_count} packages / "
            f"{scorecard.frozen_0_5.joint_count} joints"
        ),
        (
            f"- #{_STATIC_QUALIFICATION_SCOPE_ISSUE} qualification roster: "
            f"{len(scorecard.qualification_assets)} assets"
        ),
        "",
        "## Qualification assets",
        "",
        "| Asset ID | Evidence role | Reference SHA-256 |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                binding.asset_id,
                binding.evidence_role,
                f"`{binding.sha256}`",
            )
        )
        + " |"
        for binding in scorecard.qualification_assets
    )
    lines.extend(
        [
            "",
            "| Capability | Promotion eligible | Failure categories | Blockers |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in scorecard.rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.capability_id,
                    "yes" if row.promotion_eligible else "no",
                    ", ".join(row.failure_categories) or "-",
                    ", ".join(row.blocking_reason_codes) or "-",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "| Capability | Stage | Status | Source | Manifest | Contract | "
                "Output root/bundle | Validator | Result | Tool |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    commands = []
    for row in scorecard.rows:
        for stage in STATIC_STAGE_NAMES:
            result = getattr(row.stages, stage)
            hashes: tuple[str, ...]
            if result.evidence is None:
                hashes = ("-",) * 6
                tool = "retained 0.5" if result.retained_result is not None else "-"
            else:
                evidence_hashes = result.evidence.hashes
                hashes = (
                    _short_sha256(evidence_hashes.source_sha256),
                    _short_sha256(evidence_hashes.manifest_sha256),
                    _short_sha256(evidence_hashes.contract_sha256),
                    (
                        f"{_short_sha256(evidence_hashes.output_sha256)}/"
                        f"{_short_sha256(evidence_hashes.output_dependency_bundle_sha256)}"
                    ),
                    _short_sha256(evidence_hashes.validator_sha256),
                    _short_sha256(evidence_hashes.result_sha256),
                )
                tool = f"{result.evidence.tool_id}@{result.evidence.tool_version}"
                commands.append(
                    (
                        row.capability_id,
                        stage,
                        json.dumps(
                            list(result.evidence.command),
                            allow_nan=False,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ),
                    )
                )
            lines.append(
                "| "
                + " | ".join(
                    (
                        row.capability_id,
                        stage,
                        result.status,
                        *hashes,
                        tool,
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    if commands:
        lines.extend(
            f"- `{capability_id}/{stage}`: `{command}`"
            for capability_id, stage, command in commands
        )
    else:
        lines.append("- No validator command was executed.")
    return "\n".join(lines) + "\n"


def _build_row(
    row: CapabilityRowV1,
    *,
    artifact_set_sha256: str,
    scoreboard_sha256: str,
    gate3a_sha256: str,
    gate3b_sha256: str,
    gate3a_profile: str,
    gate3b_profile: str,
) -> StaticQualificationRowV1:
    frozen = any(
        binding.evidence_role == "frozen_0_5" for binding in _row_evidence_bindings(row)
    )
    if frozen:
        stages = StaticStagesV1(
            contract=StaticStageResultV1(
                status="pass",
                profile_id="joint-agent-articulation-v1",
                result_id="issue-579-release-scoreboard",
                result_sha256=scoreboard_sha256,
                artifact_identity_set_sha256=artifact_set_sha256,
            ),
            authoring=StaticStageResultV1(
                status="pass",
                profile_id="owned-core-v1",
                result_id="issue-579-release-scoreboard",
                result_sha256=scoreboard_sha256,
                artifact_identity_set_sha256=artifact_set_sha256,
            ),
            readback=StaticStageResultV1(
                status="pass",
                profile_id="exact-saved-stage-readback-v1",
                result_id="issue-579-release-scoreboard",
                result_sha256=scoreboard_sha256,
                artifact_identity_set_sha256=artifact_set_sha256,
            ),
            gate3a=StaticStageResultV1(
                status="fail",
                profile_id=gate3a_profile,
                result_id="issue-529-gate3a-baseline",
                result_sha256=gate3a_sha256,
                artifact_identity_set_sha256=artifact_set_sha256,
            ),
            gate3b=StaticStageResultV1(
                status="fail",
                profile_id=gate3b_profile,
                result_id="issue-529-gate3b-baseline",
                result_sha256=gate3b_sha256,
                artifact_identity_set_sha256=artifact_set_sha256,
            ),
        )
        blockers = ("gate3a_findings_retained", "gate3b_findings_retained")
    else:
        not_run = StaticStageResultV1(status="not_run")
        stages = StaticStagesV1(
            contract=not_run,
            authoring=not_run,
            readback=not_run,
            gate3a=not_run,
            gate3b=not_run,
        )
        blockers = tuple(sorted({*row.reason_codes, "qualification_not_run"}))
    return StaticQualificationRowV1(
        capability_id=row.capability_id,
        evidence_bindings=_row_evidence_bindings(row),
        stages=stages,
        static_qualified=False,
        dynamic_qualified=False,
        blocking_reason_codes=blockers,
    )


def _row_evidence_bindings(
    row: CapabilityRowV1,
) -> tuple[EvidenceBindingV1, ...]:
    return tuple(
        sorted(
            (
                *row.fixtures.positive,
                *row.fixtures.negative,
                *row.fixtures.representative,
            ),
            key=lambda binding: (
                binding.asset_id,
                binding.evidence_role,
                binding.artifact_key,
                binding.sha256,
            ),
        )
    )


def _frozen_artifact_set_sha256(baseline: dict[str, Any]) -> str:
    assets = baseline.get("assets")
    if not isinstance(assets, list) or not assets:
        raise CapabilityManifestError("frozen Gate 3 baseline assets are missing")
    identities = []
    for asset in assets:
        if not isinstance(asset, dict):
            raise CapabilityManifestError("frozen Gate 3 asset must be an object")
        asset_id = _required_text(asset, "asset_id")
        package_sha256 = _required_sha256(asset, "package_sha256")
        identities.append({"asset_id": asset_id, "package_sha256": package_sha256})
    identities.sort(key=lambda item: item["asset_id"])
    payload = json.dumps(
        identities,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _required_sha256(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CapabilityManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _required_text(document: dict[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CapabilityManifestError(f"{field} must be a nonblank string")
    return value.strip()


def _nonblank_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


def _validated_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _reject_nonfinite_json(value: object, *, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_json(nested, label=f"{label}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _reject_nonfinite_json(nested, label=f"{label}[{index}]")


def _canonical_reason_codes(value: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(item.strip() for item in value)
    if any(not item for item in normalized):
        raise ValueError("blocking reason codes must not be blank")
    if len(normalized) != len(set(normalized)):
        raise ValueError("blocking reason codes must be unique")
    return tuple(sorted(normalized))


def _canonical_evidence_bindings(
    value: tuple[EvidenceBindingV1, ...],
) -> tuple[EvidenceBindingV1, ...]:
    canonical = []
    for index, binding in enumerate(value):
        if not isinstance(binding, EvidenceBindingV1):
            raise ValueError(
                f"qualification evidence binding {index} must be EvidenceBindingV1"
            )
        try:
            canonical.append(
                EvidenceBindingV1.model_validate(
                    binding.model_dump(mode="json", warnings="error")
                )
            )
        except Exception as exc:
            raise ValueError(
                f"qualification evidence binding {index} is invalid: {exc}"
            ) from exc
    keys = [
        (
            binding.asset_id,
            binding.evidence_role,
            binding.artifact_key,
            binding.sha256,
        )
        for binding in canonical
    ]
    if not canonical or len(keys) != len(set(keys)):
        raise ValueError("qualification evidence bindings must be nonempty and unique")
    return tuple(
        sorted(
            canonical,
            key=lambda binding: (
                binding.asset_id,
                binding.evidence_role,
                binding.artifact_key,
                binding.sha256,
            ),
        )
    )


def _canonical_unique_rows(value: tuple[Any, ...], *, label: str) -> tuple[Any, ...]:
    ids = [row.capability_id for row in value]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} capability IDs must be unique")
    return tuple(sorted(value, key=lambda row: row.capability_id))


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream, object_pairs_hook=reject_duplicate_json_keys)
    except ValueError as exc:
        raise CapabilityManifestError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CapabilityManifestError(f"{label} root must be an object")
    return document


def _canonical_model_json(model: BaseModel) -> str:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_packaged_capability_manifest() -> LoadedCapabilityManifest:
    """Load the installed checked-in policy; callers cannot substitute authority."""

    return load_capability_manifest()


def _strict_revalidate_model(
    value: _ModelT,
    model_type: type[_ModelT],
    *,
    label: str,
) -> _ModelT:
    """Dump and parse an immutable contract so copy/construct cannot bypass checks."""

    if not isinstance(value, model_type):
        raise CapabilityManifestError(f"{label} must be a {model_type.__name__}")
    try:
        payload = value.model_dump(mode="json", warnings="error")
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        document = json.loads(encoded, object_pairs_hook=reject_duplicate_json_keys)
        return model_type.model_validate(document)
    except Exception as exc:
        raise CapabilityManifestError(
            f"{label} failed strict dump-and-parse revalidation: {exc}"
        ) from exc


def _revalidate_scorecard_policy(
    scorecard: StaticQualificationScorecardV1,
    *,
    repo_root: str | Path | None,
) -> StaticQualificationScorecardV1:
    """Bind a scorecard back to packaged capability and source authority."""

    canonical = _strict_revalidate_model(
        scorecard,
        StaticQualificationScorecardV1,
        label="static qualification scorecard",
    )
    loaded = _load_packaged_capability_manifest()
    manifest = loaded.manifest
    # This digest cannot simply equal `loaded.sha256`: it covers the
    # advertisement fields, which are exactly what changes when a row records
    # the outcome this scorecard reports, so equality made a completed run
    # invalidate its own evidence.
    #
    # But leaving it unchecked would let a caller publish forged provenance
    # through the renderer. So it is validated against the manifest digest the
    # run recorded executing under, falling back to the live digest when this
    # scorecard predates any recorded outcome of its own.
    recorded_manifest_digests = {
        row.qualification.static_attestation.capability_manifest_sha256
        for row in manifest.capabilities
        if row.qualification.static_attestation is not None
        and row.qualification.static_attestation.run_id == canonical.run_id
    }
    if len(recorded_manifest_digests) > 1:
        raise CapabilityManifestError(
            f"run {canonical.run_id!r} recorded outcomes under more than one "
            "capability manifest digest"
        )
    expected_manifest_sha256 = (
        recorded_manifest_digests.pop() if recorded_manifest_digests else loaded.sha256
    )
    if canonical.capability_manifest_sha256 != expected_manifest_sha256:
        raise CapabilityManifestError(
            "static qualification scorecard capability manifest is "
            "stale or identity-mismatched"
        )
    if canonical.corpus_sha256 != manifest.corpus.corpus_sha256:
        raise CapabilityManifestError(
            "static qualification scorecard corpus is stale or identity-mismatched"
        )
    expected_frozen = FrozenQualificationLaneV1(
        release_manifest_sha256=manifest.frozen_0_5.release_manifest_sha256,
        gate3_baseline_sha256=manifest.frozen_0_5.gate3_baseline_sha256,
        asset_count=manifest.frozen_0_5.asset_count,
        joint_count=manifest.frozen_0_5.joint_count,
    )
    if canonical.frozen_0_5 != expected_frozen:
        raise CapabilityManifestError(
            "static qualification scorecard frozen 0.5 lane is "
            "stale or identity-mismatched"
        )
    authority_root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    expected_qualification_assets = qualification_evidence_roster(
        loaded,
        repo_root=authority_root,
    )
    if canonical.qualification_assets != expected_qualification_assets:
        raise CapabilityManifestError(
            "static qualification scorecard qualification asset roster drifted"
        )
    manifest_rows = {row.capability_id: row for row in manifest.capabilities}
    scorecard_rows = {row.capability_id: row for row in canonical.rows}
    _require_exact_capability_rows(
        expected=set(manifest_rows),
        actual=set(scorecard_rows),
        label="static qualification scorecard",
    )
    for capability_id, row in scorecard_rows.items():
        manifest_row = manifest_rows[capability_id]
        if row.row_admission_sha256 != manifest_row.admission_sha256:
            # Same gap the plan validator had: without this the digest is only
            # syntax-checked, so a tampered scorecard carrying any valid
            # 64-character digest revalidates cleanly and the admission
            # identity it claims is never actually checked.
            raise CapabilityManifestError(
                f"static qualification scorecard row {capability_id} claims "
                "admission digest "
                f"{row.row_admission_sha256} but the manifest admitted "
                f"{manifest_row.admission_sha256}"
            )
        expected_bindings = _row_evidence_bindings(manifest_row)
        if row.evidence_bindings != expected_bindings:
            raise CapabilityManifestError(
                "static qualification scorecard source evidence drifted for "
                f"{capability_id}"
            )
        attestation = manifest_row.qualification.static_attestation
        # A scorecard must carry the manifest's exact row set, so rows attested
        # by an EARLIER run ride along in every later scorecard. This artifact
        # is not the one those attestations name, so only the rows this run
        # recorded are checked against it.
        if attestation is not None and attestation.run_id == canonical.run_id:
            # The recorded outcome names a run and a row outcome. Check both
            # here, where the scorecard is in hand: without this the
            # attestation proves only that someone could recompute the public
            # plan-row digest, and `scorecard_sha256`/`run_id` could be any
            # well-formed values.
            expected_outcome = _outcome_row_sha256(row)
            if attestation.scorecard_sha256 != expected_outcome:
                raise CapabilityManifestError(
                    f"{capability_id} records an outcome attested to a "
                    "different scorecard row than the one being validated"
                )
        expected_source_sha256 = _evidence_identity_set_sha256(expected_bindings)
        frozen = any(
            binding.evidence_role == "frozen_0_5" for binding in expected_bindings
        )
        for stage in STATIC_STAGE_NAMES:
            stage_result = getattr(row.stages, stage)
            # The 0.5 lane carries its own retained identities, which the v1
            # stage admission never enumerated, so it is exempt here.
            if not frozen:
                _require_admitted_stage_identity(
                    label=(f"static qualification scorecard {capability_id}/{stage}"),
                    validation_contract=stage_result.validation_contract,
                    profile_id=stage_result.profile_id,
                    result_id=stage_result.result_id,
                    admission=getattr(manifest_row.qualification.static_stages, stage),
                    require_present=False,
                )
            evidence = stage_result.evidence
            if (
                evidence is not None
                and evidence.hashes.source_sha256 != expected_source_sha256
            ):
                raise CapabilityManifestError(
                    "static qualification scorecard stage source identity "
                    f"drifted for {capability_id}/{stage}"
                )
    return canonical


def _require_exact_capability_rows(
    *,
    expected: set[str],
    actual: set[str],
    label: str,
) -> None:
    if actual != expected:
        raise CapabilityManifestError(
            f"{label} must cover every manifest row exactly; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _evidence_identity_set_sha256(
    bindings: tuple[EvidenceBindingV1, ...],
) -> str:
    payload = [
        binding.model_dump(mode="json")
        for binding in _canonical_evidence_bindings(bindings)
    ]
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _revalidate_resolved_usd_artifact(
    value: StaticResolvedUsdArtifactV1,
    *,
    expected_uri: str,
    label: str,
) -> StaticResolvedUsdArtifactV1:
    if not isinstance(value, StaticResolvedUsdArtifactV1):
        raise CapabilityManifestError(
            f"{label} output resolver must return StaticResolvedUsdArtifactV1"
        )
    if not isinstance(value.root_bytes, bytes) or any(
        not isinstance(dependency.payload, bytes) for dependency in value.dependencies
    ):
        raise CapabilityManifestError(
            f"{label} resolved root and dependency payloads must be immutable bytes"
        )
    try:
        canonical = StaticResolvedUsdArtifactV1.model_validate(
            value.model_dump(mode="python")
        )
    except ValueError as exc:
        raise CapabilityManifestError(
            f"{label} output resolver returned an invalid dependency closure"
        ) from exc
    if canonical.uri != expected_uri:
        raise CapabilityManifestError(
            f"{label} resolved output URI differs from the retained output URI"
        )
    return canonical


def _resolved_artifact_identity(
    resolved: StaticResolvedUsdArtifactV1,
) -> ArtifactIdentityV1:
    entries = [
        {"path": "<root>", "sha256": hashlib.sha256(resolved.root_bytes).hexdigest()},
        *(
            {
                "path": dependency.path,
                "sha256": hashlib.sha256(dependency.payload).hexdigest(),
            }
            for dependency in resolved.dependencies
        ),
    ]
    closure_payload = json.dumps(
        entries,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return ArtifactIdentityV1(
        uri=resolved.uri,
        root_sha256=hashlib.sha256(resolved.root_bytes).hexdigest(),
        dependency_bundle_sha256=hashlib.sha256(closure_payload).hexdigest(),
    )


def _require_admitted_stage_identity(
    *,
    label: str,
    validation_contract: str | None,
    profile_id: str | None,
    result_id: str | None,
    admission: StaticQualificationStageBindingV1,
    require_present: bool,
) -> None:
    """Reject a stage identity the manifest never admitted for that exact stage.

    Three artifacts assert stage identity -- the run plan, the scorecard, and
    the matrix -- and they must stay in lockstep: drift between them is a hole,
    not a style problem. Membership stays per stage and exact, because reading
    a union of lanes would let a profile from one lane pair with a result from
    another.

    A planned stage must name both identities (`require_present`); recorded and
    reported stages may leave them unset for a stage that never ran.
    """

    if require_present and (profile_id is None or result_id is None):
        raise CapabilityManifestError(
            f"{label} must name a complete admitted stage identity"
        )
    if admission.admissions is not None:
        if profile_id is None and result_id is None and not require_present:
            if validation_contract is None or any(
                item.validation_contract == validation_contract
                for item in admission.admissions
            ):
                return
        if (profile_id is None) != (result_id is None):
            raise CapabilityManifestError(
                f"{label} reports an incomplete atomic stage identity"
            )
        matching = tuple(
            item
            for item in admission.admissions
            if item.profile_id == profile_id
            and item.result_id == result_id
            and (
                validation_contract is None
                or item.validation_contract == validation_contract
            )
        )
        if len(matching) != 1:
            raise CapabilityManifestError(
                f"{label} reports validation contract/profile/result "
                f"{validation_contract!r}/{profile_id!r}/{result_id!r} which is "
                "not admitted as one exact atomic identity for this stage"
            )
        return
    if (require_present or profile_id is not None) and profile_id not in (
        admission.profile_ids
    ):
        raise CapabilityManifestError(
            f"{label} reports profile {profile_id!r} which is not admitted "
            "for this exact stage"
        )
    if (require_present or result_id is not None) and (
        result_id not in admission.result_ids
    ):
        raise CapabilityManifestError(
            f"{label} reports result {result_id!r} which is not admitted "
            "for this exact stage"
        )


def _validate_planned_stage_admission(
    *,
    capability: CapabilityRowV1,
    stage: StaticStageName,
    stage_plan: StaticQualificationPlanStageV1,
) -> None:
    label = f"{capability.capability_id}/{stage}"
    profile_id = stage_plan.profile_id
    result_id = stage_plan.result_id
    # A row must have admitted a run before one can be planned against it.
    # `not_run`/`na` mean nothing was admitted; `scheduled` means identities are
    # pinned with no outcome claimed. Note the membership tests below stay
    # per-stage and exact -- reading a union of lanes would let a profile from
    # one lane pair with a result from another.
    lane_status = capability.qualification.static_status
    if lane_status not in {"scheduled", "pass", "fail"}:
        raise CapabilityManifestError(
            f"{label} cannot be scheduled while the capability static lane is "
            f"{lane_status!r}"
        )
    _require_admitted_stage_identity(
        label=label,
        validation_contract=stage_plan.validation_contract,
        profile_id=profile_id,
        result_id=result_id,
        admission=getattr(capability.qualification.static_stages, stage),
        require_present=True,
    )
    minimum_support = "contract_ready" if stage == "contract" else "authorable"
    if _SUPPORT_RANK[capability.support_level] < _SUPPORT_RANK[minimum_support]:
        raise CapabilityManifestError(
            f"{label} cannot run before the row reaches {minimum_support}"
        )
    if stage != "contract" and not capability.authoring_backends:
        raise CapabilityManifestError(
            f"{label} cannot run without a manifest-admitted authoring backend"
        )


def _validate_completed_stage_claim(
    *,
    status: Literal["pass", "fail", "error"],
    raw_status: str,
    contract_validation_status: Literal["pass", "error"],
    raw_findings: tuple[dict[str, JsonValue], ...],
    blocking_reason_codes: tuple[str, ...],
) -> None:
    expected_contract_status = "error" if status == "error" else "pass"
    if contract_validation_status != expected_contract_status:
        raise ValueError(
            f"{status} stage requires contract_validation_status="
            f"{expected_contract_status!r}"
        )
    if status == "error" and not blocking_reason_codes:
        raise ValueError("validator errors require a blocking reason")
    if status in {"pass", "fail"} and blocking_reason_codes:
        raise ValueError("completed pass/fail stages must not carry blockers")
    if status == "fail" and not raw_findings:
        raise ValueError("failed stages must preserve their raw findings")
    if status == "pass" and _raw_status_is_failure(raw_status):
        raise ValueError("pass stage raw_status reports failure")
    if status == "pass" and not _raw_status_is_pass(raw_status):
        raise ValueError("pass stage raw_status is not an admitted pass status")
    if status == "pass" and _raw_findings_have_failure(raw_findings):
        raise ValueError("pass stage raw findings contain a failure severity")


def _require_completed_status(
    status: StaticExecutionStatus,
) -> Literal["pass", "fail", "error"]:
    if status == "pass":
        return "pass"
    if status == "fail":
        return "fail"
    if status == "error":
        return "error"
    raise ValueError(f"{status} is not a completed execution status")


def _raw_status_is_failure(raw_status: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]+", "_", raw_status.upper()).strip("_")
    tokens = normalized.split("_")
    return normalized in _RAW_FAILURE_STATUS_TOKENS or any(
        token in _RAW_FAILURE_STATUS_TOKENS
        and (index == 0 or tokens[index - 1] != "NO")
        for index, token in enumerate(tokens)
    )


def _raw_status_is_pass(raw_status: str) -> bool:
    normalized = re.sub(r"[^A-Z0-9]+", "_", raw_status.upper()).strip("_")
    return normalized in _RAW_PASS_STATUSES


def _raw_findings_have_failure(
    findings: tuple[dict[str, JsonValue], ...],
) -> bool:
    for finding in findings:
        classified = False
        for raw_field, value in finding.items():
            field = raw_field.strip().lower()
            if field == "severity":
                classified = True
                if not isinstance(value, str):
                    return True
                severity = value.strip().upper()
                if (
                    severity in _RAW_FAILURE_FINDING_SEVERITIES
                    or severity not in _RAW_NONFAIL_FINDING_SEVERITIES
                ):
                    return True
            elif field == "status":
                classified = True
                if not isinstance(value, str) or (
                    not _raw_status_is_pass(value)
                    and value.strip().upper() not in {"INFO", "WARN", "WARNING"}
                ):
                    return True
        if not classified:
            return True
    return False


def _validate_retained_stage_evidence(
    *,
    label: str,
    capability_id: str,
    stage: StaticStageName,
    stage_result: StaticQualificationExecutedStageV1,
    expected_source_bindings: tuple[EvidenceBindingV1, ...],
    evidence_adapter: StaticQualificationEvidenceAdapter,
) -> None:
    evidence = stage_result.evidence
    if evidence is None:
        raise CapabilityManifestError(f"{label} omitted required run evidence")
    expected_sha256: Mapping[StaticRetainedArtifactRole, str] = {
        "contract": evidence.hashes.contract_sha256,
        "output": evidence.hashes.output_sha256,
        "validator": evidence.hashes.validator_sha256,
        "result": evidence.hashes.result_sha256,
        "raw_report": evidence.raw_report_sha256,
    }
    artifacts: dict[StaticRetainedArtifactRole, bytes] = {}
    resolved_output: StaticResolvedUsdArtifactV1 | None = None
    observed_output_identity: ArtifactIdentityV1 | None = None
    for retained in evidence.retained_artifacts:
        try:
            if retained.role == "output":
                resolved_output = _revalidate_resolved_usd_artifact(
                    evidence_adapter.reopen_usd_artifact(retained.uri),
                    expected_uri=retained.uri,
                    label=label,
                )
                payload = resolved_output.root_bytes
                observed_output_identity = _resolved_artifact_identity(resolved_output)
            else:
                payload = evidence_adapter.read_bytes(retained.uri)
        except Exception as exc:
            unavailable = (
                "bytes or dependency closure are unavailable"
                if retained.role == "output"
                else "bytes are unavailable"
            )
            raise CapabilityManifestError(
                f"{label} retained {retained.role} {unavailable}"
            ) from exc
        if not isinstance(payload, bytes):
            raise CapabilityManifestError(
                f"{label} retained {retained.role} resolver output must be bytes"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha256[retained.role]:
            raise CapabilityManifestError(
                f"{label} retained {retained.role} bytes do not match the "
                "declared SHA-256"
            )
        if retained.role == "output":
            expected_output_identity = ArtifactIdentityV1(
                uri=retained.uri,
                root_sha256=evidence.hashes.output_sha256,
                dependency_bundle_sha256=(
                    evidence.hashes.output_dependency_bundle_sha256
                ),
            )
            if observed_output_identity != expected_output_identity:
                raise CapabilityManifestError(
                    f"{label} retained output root or dependency-bundle identity "
                    "does not match the declared ArtifactIdentityV1"
                )
        artifacts[retained.role] = payload

    if resolved_output is None or observed_output_identity is None:
        raise CapabilityManifestError(f"{label} retained output was not resolved")
    try:
        trusted_report = evidence_adapter.interpret_stage(
            stage=stage,
            validation_contract=stage_result.validation_contract,
            artifacts=artifacts,
            resolved_output=resolved_output,
            observed_artifact_identity=observed_output_identity,
        )
        trusted_report = _strict_revalidate_model(
            trusted_report,
            StaticTrustedStageReportV1,
            label=f"{label} trusted retained report",
        )
    except Exception as exc:
        raise CapabilityManifestError(
            f"{label} retained report could not be interpreted by the trusted adapter"
        ) from exc
    if stage_result.profile_id is None or stage_result.result_id is None:
        raise CapabilityManifestError(
            f"{label} completed evidence omitted its planned identities"
        )
    expected_report = StaticTrustedStageReportV1(
        status=_require_completed_status(stage_result.status),
        raw_status=stage_result.raw_status,
        capability_id=capability_id,
        source_evidence_bindings=expected_source_bindings,
        profile_id=stage_result.profile_id,
        result_id=stage_result.result_id,
        command=evidence.command,
        tool_id=evidence.tool_id,
        tool_version=evidence.tool_version,
        observed_artifact_identity=observed_output_identity,
        contract_validation_status=evidence.contract_validation_status,
        raw_findings=stage_result.raw_findings,
        blocking_reason_codes=stage_result.blocking_reason_codes,
    )
    if trusted_report != expected_report:
        raise CapabilityManifestError(
            f"{label} untrusted JSON disagrees with the trusted retained report"
        )


def _validate_stage_result_against_plan(
    *,
    capability_id: str,
    stage: StaticStageName,
    stage_plan: StaticQualificationPlanStageV1,
    stage_result: StaticQualificationExecutedStageV1,
    expected_source_bindings: tuple[EvidenceBindingV1, ...],
    expected_source_sha256: str,
    manifest_sha256: str,
    evidence_adapter: StaticQualificationEvidenceAdapter,
) -> None:
    label = f"{capability_id}/{stage}"
    if (
        stage_result.validation_contract != stage_plan.validation_contract
        or stage_result.profile_id != stage_plan.profile_id
        or stage_result.result_id != stage_plan.result_id
    ):
        raise CapabilityManifestError(
            f"{label} result differs from its planned profile/result/validation "
            "contract"
        )
    if stage_plan.execution == "retain_frozen_0_5":
        retained = stage_plan.retained_result
        if (
            retained is None
            or stage_result.retained_result != retained
            or stage_result.raw_status != f"retained_{retained.status}"
        ):
            raise CapabilityManifestError(
                f"{label} retained 0.5 result identity drifted"
            )
        return
    if stage_result.retained_result is not None:
        raise CapabilityManifestError(
            f"{label} cannot substitute frozen evidence for a post-0.5 stage"
        )
    if stage_plan.execution == "blocked":
        if (
            stage_result.status not in {"blocked", "not_run"}
            or stage_result.blocking_reason_codes != stage_plan.blocking_reason_codes
        ):
            raise CapabilityManifestError(
                f"{label} was blocked in the run plan; its blockers must be "
                "preserved exactly"
            )
        return
    evidence = stage_result.evidence
    if evidence is None:
        if stage_result.status not in {"blocked", "not_run"}:
            raise CapabilityManifestError(f"{label} omitted required run evidence")
        return
    if (
        evidence.command != stage_plan.command
        or evidence.tool_id != stage_plan.tool_id
        or evidence.tool_version != stage_plan.tool_version
    ):
        raise CapabilityManifestError(
            f"{label} command or tool identity differs from the run plan"
        )
    if evidence.hashes.source_sha256 != expected_source_sha256:
        raise CapabilityManifestError(
            f"{label} source evidence identity does not match the manifest row"
        )
    if evidence.hashes.manifest_sha256 != manifest_sha256:
        raise CapabilityManifestError(
            f"{label} result is bound to a stale capability manifest"
        )
    _validate_retained_stage_evidence(
        label=label,
        capability_id=capability_id,
        stage=stage,
        stage_result=stage_result,
        expected_source_bindings=expected_source_bindings,
        evidence_adapter=evidence_adapter,
    )


def _result_blocking_reason_codes(
    row: StaticQualificationResultRowV1,
    manifest_row: CapabilityRowV1,
) -> tuple[str, ...]:
    blockers = set(manifest_row.reason_codes)
    if manifest_row.disposition != "supported":
        blockers.add(f"capability_disposition_{manifest_row.disposition}")
    if _SUPPORT_RANK[manifest_row.support_level] < _SUPPORT_RANK["authorable"]:
        blockers.add("capability_not_authorable")
    if not manifest_row.authoring_backends:
        blockers.add("authoring_backend_missing")
    # `scheduled` means a run was admitted with its identities pinned and no
    # outcome claimed, so it must not block that run's own verdict. Recording
    # the outcome is a separate manifest edit that requires an attestation
    # bound to this row's admission digest, which is what keeps this safe.
    if manifest_row.qualification.static_status not in {"pass", "scheduled"}:
        blockers.add(
            f"manifest_static_status_{manifest_row.qualification.static_status}"
        )
    for stage in STATIC_STAGE_NAMES:
        result = getattr(row.stages, stage)
        blockers.update(result.blocking_reason_codes)
        if result.status == "pass":
            continue
        elif result.status == "fail":
            suffix = (
                "findings_retained"
                if result.retained_result is not None
                else "findings"
            )
            blockers.add(f"{stage}_{suffix}")
        elif result.status == "error":
            blockers.add(f"{stage}_validator_error")
        elif result.status == "blocked":
            blockers.add(f"{stage}_blocked")
        elif result.status == "not_run":
            blockers.add(f"{stage}_not_run")
        else:
            blockers.add(f"{stage}_not_applicable")
    return tuple(sorted(blockers))


def _result_failure_categories(
    row: StaticQualificationResultRowV1,
    manifest_row: CapabilityRowV1,
) -> tuple[StaticFailureCategory, ...]:
    categories: set[StaticFailureCategory] = set()
    stage_category: Mapping[StaticStageName, StaticFailureCategory] = {
        "contract": "contract",
        "authoring": "authoring",
        "readback": "package",
        "gate3a": "static_schema",
        "gate3b": "static_schema",
    }
    for stage in STATIC_STAGE_NAMES:
        result = getattr(row.stages, stage)
        if result.status != "pass" or (
            result.evidence is not None
            and result.evidence.contract_validation_status != "pass"
        ):
            categories.add(stage_category[stage])
    if (
        manifest_row.disposition != "supported"
        or manifest_row.reason_codes
        or _SUPPORT_RANK[manifest_row.support_level] < _SUPPORT_RANK["authorable"]
        or not manifest_row.authoring_backends
        or manifest_row.qualification.static_status not in {"pass", "scheduled"}
    ):
        categories.add("unsupported_capability")
    order: tuple[StaticFailureCategory, ...] = (
        "contract",
        "authoring",
        "package",
        "static_schema",
        "unsupported_capability",
    )
    return tuple(item for item in order if item in categories)


def _short_sha256(value: str) -> str:
    return f"`{value[:12]}`"


__all__ = [
    "STATIC_QUALIFICATION_RESULT_SCHEMA_VERSION",
    "STATIC_QUALIFICATION_RUN_PLAN_SCHEMA_VERSION",
    "STATIC_QUALIFICATION_RUN_REQUEST_SCHEMA_VERSION",
    "STATIC_QUALIFICATION_SCORECARD_SCHEMA_VERSION",
    "STATIC_QUALIFICATION_SCHEMA_VERSION",
    "STATIC_RETAINED_ARTIFACT_ROLES",
    "STATIC_STAGE_NAMES",
    "STATIC_STAGE_VALIDATION_CONTRACTS",
    "FrozenQualificationLaneV1",
    "StaticEvidenceHashesV1",
    "StaticQualificationEvidenceAdapter",
    "StaticQualificationExecutedStageV1",
    "StaticQualificationExecutedStagesV1",
    "StaticQualificationMatrixV1",
    "StaticQualificationPlanRowV1",
    "StaticQualificationPlanStageV1",
    "StaticQualificationPlanStagesV1",
    "StaticQualificationResultBundleV1",
    "StaticQualificationResultRowV1",
    "StaticQualificationRowV1",
    "StaticQualificationRunPlanV1",
    "StaticQualificationRunRequestDocumentV1",
    "StaticQualificationRunRequestV1",
    "StaticQualificationScorecardRowV1",
    "StaticQualificationScorecardSummaryV1",
    "StaticQualificationScorecardV1",
    "StaticQualificationStageRequestV1",
    "StaticRetainedArtifactRole",
    "StaticRetainedArtifactV1",
    "StaticResolvedDependencyV1",
    "StaticResolvedUsdArtifactV1",
    "StaticStageExecutionEvidenceV1",
    "StaticStageResultV1",
    "StaticStagesV1",
    "StaticTrustedStageReportV1",
    "build_static_qualification_run_plan",
    "build_static_qualification_run_plan_scaffold",
    "build_static_qualification_scaffold",
    "canonical_matrix_json",
    "canonical_result_bundle_json",
    "canonical_run_plan_json",
    "canonical_scorecard_json",
    "ingest_static_qualification_results",
    "load_static_qualification_result_bundle",
    "load_static_qualification_run_plan",
    "load_static_qualification_run_request",
    "load_static_qualification_matrix",
    "render_static_qualification_scorecard",
    "static_qualification_run_plan_sha256",
    "static_qualification_source_sha256",
    "validate_static_qualification_result_bundle",
    "validate_static_qualification_run_plan",
    "validate_static_qualification_matrix",
]
