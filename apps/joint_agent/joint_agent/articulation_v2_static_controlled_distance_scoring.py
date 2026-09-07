# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static-only scorecard and attestation contracts for controlled-distance-v2."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_validator,
    model_validator,
)

from joint_agent.articulation_v2_static_run_plan import (
    ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_REPRESENTATION,
    ArticulationV2StaticControlledDistanceRunIdentityV2,
)
from joint_agent.articulation_v2_static_scoring import (
    ARTICULATION_V2_STATIC_SCORER_VERSION,
    ArticulationV2StaticScoredStageV1,
    ArticulationV2StaticScoringHashesV1,
)
from joint_agent.functions.articulation_v2_controlled_distance_usd import (
    ControlledDistanceV2SourceReadbackProtocol,
)

ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_SCORECARD_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-controlled-distance-scorecard-v2"
] = "joint-agent-articulation-v2-static-controlled-distance-scorecard-v2"
ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_ATTESTATION_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-controlled-distance-attestation-v2"
] = "joint-agent-articulation-v2-static-controlled-distance-attestation-v2"

_STAGES = ("contract", "authoring", "readback", "gate3a", "gate3b")


class _ScoringModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )


class ArticulationV2StaticControlledDistanceScorecardV2(_ScoringModel):
    schema_version: Literal[
        "joint-agent-articulation-v2-static-controlled-distance-scorecard-v2"
    ]
    scorer_version: Literal["0.6.0"]
    representation: Literal["prismatic_axial_interval_drive_body1_to_body0_v2"]
    run: ArticulationV2StaticControlledDistanceRunIdentityV2
    constraint_kind: Literal["distance"]
    source_readback_protocol: ControlledDistanceV2SourceReadbackProtocol
    hashes: ArticulationV2StaticScoringHashesV1
    stages: tuple[ArticulationV2StaticScoredStageV1, ...]
    static_qualified: bool
    dynamic_qualified: Literal[False]
    public_enabled: Literal[False]
    release_selection_applied: Literal[False]
    capability_manifest_updated: Literal[False]

    @field_validator("stages")
    @classmethod
    def _stage_order(
        cls, value: tuple[ArticulationV2StaticScoredStageV1, ...]
    ) -> tuple[ArticulationV2StaticScoredStageV1, ...]:
        if tuple(stage.stage for stage in value) != _STAGES:
            raise ValueError("V2 scorecard requires the exact five-stage order")
        return value

    @model_validator(mode="after")
    def _static_only(self) -> ArticulationV2StaticControlledDistanceScorecardV2:
        if (
            self.representation
            != ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_REPRESENTATION
            or self.run.selected_source.representation != self.representation
            or self.run.capability_id != "distance.bounded_two_body_constraint"
        ):
            raise ValueError("V2 scorecard mixes representation or capability")
        _require_run_hashes(self.run, self.hashes)
        expected_qualification = self.source_readback_protocol == (
            "controlled-distance-retained-source-readback-v2"
        ) and all(stage.status == "pass" for stage in self.stages)
        if self.static_qualified != expected_qualification:
            raise ValueError(
                "V2 static qualification requires sealed source readback and "
                "five passing stages"
            )
        return self


class ArticulationV2StaticControlledDistanceAttestationV2(_ScoringModel):
    schema_version: Literal[
        "joint-agent-articulation-v2-static-controlled-distance-attestation-v2"
    ]
    scorer_version: Literal["0.6.0"]
    scope: Literal["static_only"]
    representation: Literal["prismatic_axial_interval_drive_body1_to_body0_v2"]
    run: ArticulationV2StaticControlledDistanceRunIdentityV2
    constraint_kind: Literal["distance"]
    source_readback_protocol: ControlledDistanceV2SourceReadbackProtocol
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
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("V2 attestation hashes must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _same_run(self) -> ArticulationV2StaticControlledDistanceAttestationV2:
        if self.representation != self.run.selected_source.representation:
            raise ValueError("V2 attestation mixes representations")
        if (
            self.static_qualified
            and self.source_readback_protocol
            != "controlled-distance-retained-source-readback-v2"
        ):
            raise ValueError("V2 qualification requires sealed source readback")
        _require_run_hashes(self.run, self.hashes)
        return self


def build_articulation_v2_static_controlled_distance_attestation(
    scorecard: ArticulationV2StaticControlledDistanceScorecardV2,
    report: str,
) -> ArticulationV2StaticControlledDistanceAttestationV2:
    scorecard_bytes = (
        canonical_articulation_v2_static_controlled_distance_scorecard_bytes(scorecard)
    )
    return ArticulationV2StaticControlledDistanceAttestationV2(
        schema_version=ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_ATTESTATION_SCHEMA_VERSION,
        scorer_version=ARTICULATION_V2_STATIC_SCORER_VERSION,
        scope="static_only",
        representation=scorecard.representation,
        run=scorecard.run,
        constraint_kind="distance",
        source_readback_protocol=scorecard.source_readback_protocol,
        scorecard_sha256=hashlib.sha256(scorecard_bytes).hexdigest(),
        report_sha256=hashlib.sha256(report.encode("utf-8")).hexdigest(),
        hashes=scorecard.hashes,
        static_qualified=scorecard.static_qualified,
        dynamic_qualified=False,
        public_enabled=False,
        release_selection_applied=False,
        capability_manifest_updated=False,
    )


def render_articulation_v2_static_controlled_distance_scorecard(
    scorecard: ArticulationV2StaticControlledDistanceScorecardV2,
) -> str:
    scorecard = _strict(scorecard, ArticulationV2StaticControlledDistanceScorecardV2)
    lines = [
        "# Joint Agent controlled-distance-v2 static scorecard",
        "",
        f"- Static outcome: **{'PASS' if scorecard.static_qualified else 'FAIL'}**",
        f"- Capability: `{scorecard.run.capability_id}`",
        f"- Run: `{scorecard.run.run_id}`",
        f"- Representation: `{scorecard.representation}`",
        f"- Source readback protocol: `{scorecard.source_readback_protocol}`",
        "- Dynamic qualified: `false`",
        "- Public enabled: `false`",
        "- Release selection applied: `false`",
        "- Capability manifest updated: `false`",
        "",
        "> This attests only to retained static controlled-distance-v2 evidence. "
        "It does not establish dynamic qualification or release selection.",
        "",
        "## Bound hashes",
        "",
        "| Identity | SHA-256 |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| {name} | `{value}` |"
        for name, value in scorecard.hashes.model_dump(mode="python").items()
    )
    lines.extend(
        [
            "",
            "## Five-stage outcome",
            "",
            "| Stage | Outcome | Raw status | Receipt SHA-256 |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {stage.stage} | {stage.status} | {stage.raw_status} | `{stage.receipt_sha256}` |"
        for stage in scorecard.stages
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


def canonical_articulation_v2_static_controlled_distance_scorecard_bytes(
    scorecard: ArticulationV2StaticControlledDistanceScorecardV2,
) -> bytes:
    return _canonical(scorecard, ArticulationV2StaticControlledDistanceScorecardV2)


def canonical_articulation_v2_static_controlled_distance_attestation_bytes(
    attestation: ArticulationV2StaticControlledDistanceAttestationV2,
) -> bytes:
    return _canonical(attestation, ArticulationV2StaticControlledDistanceAttestationV2)


def _require_run_hashes(
    run: ArticulationV2StaticControlledDistanceRunIdentityV2,
    hashes: ArticulationV2StaticScoringHashesV1,
) -> None:
    expected = (
        run.capability_manifest_sha256,
        run.row_admission_sha256,
        run.selected_source.root_sha256,
        run.selected_source.gate3_dependency_bundle.sha256,
        run.selector.expected_joint_semantics.source_authority.effective_reference_manifest.sha256,
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
        raise ValueError("V2 scoring hashes differ from embedded run identity")


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _strict(value: _ModelT, model: type[_ModelT]) -> _ModelT:
    if type(value) is not model:
        raise TypeError(f"value must be an exact {model.__name__}")
    return TypeAdapter(model).validate_json(_canonical(value, model), strict=True)


def _canonical(value: _ModelT, model: type[_ModelT]) -> bytes:
    if type(value) is not model:
        raise TypeError(f"value must be an exact {model.__name__}")
    adapter = TypeAdapter(model)
    payload = json.dumps(
        adapter.dump_python(value, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    validated = adapter.validate_json(payload, strict=True)
    return json.dumps(
        adapter.dump_python(validated, mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_ATTESTATION_SCHEMA_VERSION",
    "ARTICULATION_V2_STATIC_CONTROLLED_DISTANCE_SCORECARD_SCHEMA_VERSION",
    "ArticulationV2StaticControlledDistanceAttestationV2",
    "ArticulationV2StaticControlledDistanceScorecardV2",
    "build_articulation_v2_static_controlled_distance_attestation",
    "canonical_articulation_v2_static_controlled_distance_attestation_bytes",
    "canonical_articulation_v2_static_controlled_distance_scorecard_bytes",
    "render_articulation_v2_static_controlled_distance_scorecard",
]
