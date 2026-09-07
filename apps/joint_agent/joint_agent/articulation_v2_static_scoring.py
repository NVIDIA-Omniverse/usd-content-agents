# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static-only scorecard and attestation contracts for retained 0.6 runs.

These models adapt one already-verified articulation-v2 fixed or distance
five-stage bundle into the issue #868 matrix outcome.  They have no execution,
manifest mutation, release-selection, dynamic-qualification, or public-surface
behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

from joint_agent.articulation_v2_static_run_plan import (
    ArticulationV2StaticRunIdentityV1,
)
from joint_agent.static_qualification_contracts import (
    ARTICULATION_V2_STATIC_STAGE_VALIDATION_CONTRACTS,
)

ARTICULATION_V2_STATIC_SCORECARD_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-scorecard-v1"
] = "joint-agent-articulation-v2-static-scorecard-v1"
ARTICULATION_V2_STATIC_ATTESTATION_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-attestation-v1"
] = "joint-agent-articulation-v2-static-attestation-v1"
ARTICULATION_V2_STATIC_SCORER_VERSION: Literal["0.6.0"] = "0.6.0"

type ArticulationV2StaticStageName = Literal[
    "contract", "authoring", "readback", "gate3a", "gate3b"
]
type ArticulationV2StaticScoredStatus = Literal["pass", "fail"]

_STAGE_NAMES: tuple[ArticulationV2StaticStageName, ...] = (
    "contract",
    "authoring",
    "readback",
    "gate3a",
    "gate3b",
)
_CAPABILITY_BY_KIND = {
    "fixed": "fixed.explicit_two_body_constraint",
    "distance": "distance.bounded_two_body_constraint",
}
_RECEIPT_PATH_BY_STAGE = {
    "contract": "receipts/contract.json",
    "authoring": "receipts/authoring.json",
    "readback": "receipts/readback.json",
    "gate3a": "claims/gate3a.json",
    "gate3b": "claims/gate3b.json",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _ScoringModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )


class ArticulationV2StaticScoringHashesV1(_ScoringModel):
    """Every authority and retained-artifact hash used by the outcome."""

    capability_manifest_sha256: str
    row_admission_sha256: str
    selected_source_root_sha256: str
    selected_source_dependency_bundle_sha256: str
    effective_reference_manifest_sha256: str
    selector_semantics_sha256: str
    run_plan_sha256: str
    intake_sha256: str
    authoring_contract_sha256: str
    semantic_contract_sha256: str
    target_root_sha256: str
    generated_root_sha256: str
    generated_authorer_dependency_bundle_sha256: str
    generated_gate3_dependency_bundle_sha256: str
    authoring_result_sha256: str
    result_bundle_sha256: str
    completion_sha256: str
    evidence_root_sha256: str

    @field_validator("*")
    @classmethod
    def _lowercase_sha256(cls, value: str, info: Any) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _root_and_generated_hashes_agree(
        self,
    ) -> ArticulationV2StaticScoringHashesV1:
        if self.evidence_root_sha256 != self.completion_sha256:
            raise ValueError("evidence-root identity must be the completion digest")
        return self


class ArticulationV2StaticScoredStageV1(_ScoringModel):
    """One #868 matrix stage with raw producer outcome retained unchanged."""

    stage: ArticulationV2StaticStageName
    validation_contract: str
    profile_id: str
    result_id: str
    tool_id: str
    tool_version: str
    command_sha256: str
    command_launcher_sha256: str
    command_source_sha256: str
    status: ArticulationV2StaticScoredStatus
    raw_status: str
    raw_findings: JsonValue
    receipt_path: str
    receipt_sha256: str
    raw_report_path: str | None = None
    raw_report_sha256: str | None = None

    @field_validator(
        "validation_contract",
        "profile_id",
        "result_id",
        "tool_id",
        "tool_version",
        "raw_status",
        "receipt_path",
    )
    @classmethod
    def _nonblank(cls, value: str, info: Any) -> str:
        if not value or value.strip() != value:
            raise ValueError(f"{info.field_name} must be exact nonblank text")
        return value

    @field_validator(
        "command_sha256",
        "command_launcher_sha256",
        "command_source_sha256",
        "receipt_sha256",
        "raw_report_sha256",
    )
    @classmethod
    def _optional_sha256(cls, value: str | None, info: Any) -> str | None:
        if value is not None and _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _raw_report_shape(self) -> ArticulationV2StaticScoredStageV1:
        gate = self.stage in {"gate3a", "gate3b"}
        report_fields = (self.raw_report_path, self.raw_report_sha256)
        if gate and not all(value is not None for value in report_fields):
            raise ValueError("Gate 3 stages require complete raw report identity")
        if not gate and any(value is not None for value in report_fields):
            raise ValueError("non-Gate stages cannot carry raw report identity")
        if not gate and self.raw_findings != []:
            raise ValueError("non-Gate stages cannot carry raw findings")
        if (
            self.validation_contract
            != ARTICULATION_V2_STATIC_STAGE_VALIDATION_CONTRACTS[self.stage]
        ):
            raise ValueError("stage validation contract differs from articulation-v2")
        if self.receipt_path != _RECEIPT_PATH_BY_STAGE[self.stage]:
            raise ValueError("stage receipt path differs from the retained contract")
        if gate:
            if self.raw_report_path != f"reports/{self.stage}.json":
                raise ValueError(
                    "Gate 3 raw report path differs from the retained claim"
                )
            if type(self.raw_findings) is not dict:
                raise ValueError("Gate 3 raw findings must be a JSON object")
        expected_status = _normalized_raw_status(self.stage, self.raw_status)
        if expected_status != self.status:
            raise ValueError("raw producer status differs from the scored status")
        return self


class ArticulationV2StaticScorecardV1(_ScoringModel):
    """One fixed-or-distance static matrix outcome from retained evidence."""

    schema_version: Literal["joint-agent-articulation-v2-static-scorecard-v1"]
    scorer_version: Literal["0.6.0"]
    run: ArticulationV2StaticRunIdentityV1
    constraint_kind: Literal["fixed", "distance"]
    representation: Literal["raw_usd", "usdz"]
    hashes: ArticulationV2StaticScoringHashesV1
    stages: tuple[ArticulationV2StaticScoredStageV1, ...]
    static_qualified: bool
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]
    capability_manifest_updated: Literal[False]

    @field_validator("stages")
    @classmethod
    def _exact_stage_order(
        cls,
        value: tuple[ArticulationV2StaticScoredStageV1, ...],
    ) -> tuple[ArticulationV2StaticScoredStageV1, ...]:
        if tuple(stage.stage for stage in value) != _STAGE_NAMES:
            raise ValueError("scorecard requires the exact five-stage order")
        return value

    @model_validator(mode="after")
    def _static_only_outcome(self) -> ArticulationV2StaticScorecardV1:
        if self.run.capability_id != _CAPABILITY_BY_KIND[self.constraint_kind]:
            raise ValueError("fixed and distance scorecard identities must not mix")
        if self.run.selected_source.representation != self.representation:
            raise ValueError("scorecard representation differs from selected source")
        _require_run_hashes(self.run, self.hashes)
        expected = all(stage.status == "pass" for stage in self.stages)
        if self.static_qualified != expected:
            raise ValueError("static qualification requires all five stages to pass")
        return self


class ArticulationV2StaticAttestationV1(_ScoringModel):
    """Hash-bound, static-only outcome attestation; never a release decision."""

    schema_version: Literal["joint-agent-articulation-v2-static-attestation-v1"]
    scorer_version: Literal["0.6.0"]
    scope: Literal["static_only"]
    run: ArticulationV2StaticRunIdentityV1
    constraint_kind: Literal["fixed", "distance"]
    scorecard_sha256: str
    report_sha256: str
    hashes: ArticulationV2StaticScoringHashesV1
    static_qualified: bool
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]
    capability_manifest_updated: Literal[False]

    @field_validator("scorecard_sha256", "report_sha256")
    @classmethod
    def _lowercase_sha256(cls, value: str, info: Any) -> str:
        if _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _same_capability_kind(self) -> ArticulationV2StaticAttestationV1:
        if self.run.capability_id != _CAPABILITY_BY_KIND[self.constraint_kind]:
            raise ValueError("attestation must not mix fixed and distance evidence")
        _require_run_hashes(self.run, self.hashes)
        return self


def _normalized_raw_status(
    stage: ArticulationV2StaticStageName,
    status: str,
) -> ArticulationV2StaticScoredStatus:
    if stage in {"contract", "authoring", "readback"}:
        if status == "pass":
            return "pass"
    elif stage == "gate3a":
        if status == "pass":
            return "pass"
        if status in {"warning", "fail"}:
            return "fail"
    elif stage == "gate3b":
        if status == "PASS":
            return "pass"
        if status == "FAIL":
            return "fail"
    raise ValueError(f"{stage} raw producer status is not a terminal outcome")


def _require_run_hashes(
    run: ArticulationV2StaticRunIdentityV1,
    hashes: ArticulationV2StaticScoringHashesV1,
) -> None:
    effective_manifest = run.selector.expected_joint_semantics.source_authority.effective_reference_manifest
    expected = (
        run.capability_manifest_sha256,
        run.row_admission_sha256,
        run.selected_source.root_sha256,
        run.selected_source.gate3_dependency_bundle.sha256,
        effective_manifest.sha256,
        run.selector.expected_joint_semantics_sha256,
    )
    actual = (
        hashes.capability_manifest_sha256,
        hashes.row_admission_sha256,
        hashes.selected_source_root_sha256,
        hashes.selected_source_dependency_bundle_sha256,
        hashes.effective_reference_manifest_sha256,
        hashes.selector_semantics_sha256,
    )
    if actual != expected:
        raise ValueError("scorecard hashes differ from the embedded run identity")


def build_articulation_v2_static_attestation(
    scorecard: ArticulationV2StaticScorecardV1,
    report: str,
) -> ArticulationV2StaticAttestationV1:
    """Bind canonical scorecard and report bytes to the verified evidence root."""

    canonical = canonical_articulation_v2_static_scorecard_bytes(scorecard)
    report_bytes = report.encode("utf-8")
    attestation = ArticulationV2StaticAttestationV1(
        schema_version=ARTICULATION_V2_STATIC_ATTESTATION_SCHEMA_VERSION,
        scorer_version=ARTICULATION_V2_STATIC_SCORER_VERSION,
        scope="static_only",
        run=scorecard.run,
        constraint_kind=scorecard.constraint_kind,
        scorecard_sha256=hashlib.sha256(canonical).hexdigest(),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        hashes=scorecard.hashes,
        static_qualified=scorecard.static_qualified,
        dynamic_qualified=False,
        public_enabled=False,
        release_selection_applied=False,
        capability_manifest_updated=False,
    )
    return _strict_revalidate(attestation, ArticulationV2StaticAttestationV1)


def canonical_articulation_v2_static_scorecard_bytes(
    scorecard: ArticulationV2StaticScorecardV1,
) -> bytes:
    return _canonical_bytes(scorecard, ArticulationV2StaticScorecardV1)


def canonical_articulation_v2_static_attestation_bytes(
    attestation: ArticulationV2StaticAttestationV1,
) -> bytes:
    return _canonical_bytes(attestation, ArticulationV2StaticAttestationV1)


def render_articulation_v2_static_scorecard(
    scorecard: ArticulationV2StaticScorecardV1,
) -> str:
    """Render deterministic Markdown while retaining raw Gate 3 results."""

    scorecard = _strict_revalidate(scorecard, ArticulationV2StaticScorecardV1)
    outcome = "PASS" if scorecard.static_qualified else "FAIL"
    lines = [
        "# Joint Agent articulation-v2 static scorecard",
        "",
        f"- Static outcome: **{outcome}**",
        f"- Capability: `{scorecard.run.capability_id}`",
        f"- Constraint kind: `{scorecard.constraint_kind}`",
        f"- Run: `{scorecard.run.run_id}`",
        f"- Representation: `{scorecard.representation}`",
        "- Dynamic qualified: `false`",
        "- Public enabled: `false`",
        "- Release selection applied: `false`",
        "- Capability manifest updated: `false`",
        "",
        "> This attests only to retained static evidence. It does not establish "
        "dynamic qualification or public release selection.",
        "",
        "## Bound hashes",
        "",
        "| Identity | SHA-256 |",
        "| --- | --- |",
    ]
    for name, value in scorecard.hashes.model_dump(mode="python").items():
        lines.append(f"| {name} | `{value}` |")
    lines.extend(
        [
            "",
            "## Five-stage outcome",
            "",
            "| Stage | Outcome | Raw status | Profile | Result | Receipt SHA-256 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for stage in scorecard.stages:
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_table_cell(stage.stage),
                    _markdown_table_cell(stage.status),
                    _markdown_table_cell(stage.raw_status),
                    _markdown_table_cell(stage.profile_id),
                    _markdown_table_cell(stage.result_id),
                    f"`{stage.receipt_sha256}`",
                )
            )
            + " |"
        )
    for stage in scorecard.stages:
        if stage.raw_report_path is None:
            continue
        lines.extend(
            [
                "",
                f"### {stage.stage} raw retained findings",
                "",
                f"- Raw report: `{stage.raw_report_path}`",
                f"- Raw report SHA-256: `{stage.raw_report_sha256}`",
                "",
                "```json",
                json.dumps(
                    stage.raw_findings,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def _markdown_table_cell(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("|", "`", "*", "_", "[", "]"):
        escaped = escaped.replace(character, f"\\{character}")
    return (
        escaped.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "\\n")
    )


def _strict_revalidate(value: _ModelT, model: type[_ModelT]) -> _ModelT:
    if type(value) is not model:
        raise TypeError(f"canonical encoding requires exact {model.__name__}")
    adapter = TypeAdapter(model)
    return adapter.validate_json(_dump_model_bytes(value, adapter), strict=True)


def _canonical_bytes(value: _ModelT, model: type[_ModelT]) -> bytes:
    adapter = TypeAdapter(model)
    validated = _strict_revalidate(value, model)
    return _dump_model_bytes(validated, adapter)


def _dump_model_bytes(value: _ModelT, adapter: TypeAdapter[_ModelT]) -> bytes:
    return json.dumps(
        adapter.dump_python(value, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "ARTICULATION_V2_STATIC_ATTESTATION_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_SCORECARD_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_SCORER_VERSION",
    "ArticulationV2StaticAttestationV1",
    "ArticulationV2StaticScorecardV1",
    "ArticulationV2StaticScoredStageV1",
    "ArticulationV2StaticScoringHashesV1",
    "build_articulation_v2_static_attestation",
    "canonical_articulation_v2_static_attestation_bytes",
    "canonical_articulation_v2_static_scorecard_bytes",
    "render_articulation_v2_static_scorecard",
]
