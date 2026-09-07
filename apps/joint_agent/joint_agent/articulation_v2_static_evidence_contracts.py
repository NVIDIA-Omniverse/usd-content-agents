# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic retained-evidence path contracts for Joint 0.6 static runs.

This module names the only paths a later evidence publisher may retain. It
does not create a root, open a file, or publish bytes. ``completion.json`` is
the sole terminal receipt and is included in the allowlist but excluded from
the terminal receipt's non-circular retained-file inventory.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from world_understanding.utils.artifacts import validated_artifact_relative_key

from joint_agent.articulation_v2_static_semantics import (
    ArticulationV2StaticConstraintKind,
)

ARTICULATION_V2_STATIC_EVIDENCE_PATH_ALLOWLIST_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-v2-static-evidence-path-allowlist-v1"
] = "joint-agent-articulation-v2-static-evidence-path-allowlist-v1"

_USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})

STATIC_EVIDENCE_TERMINAL_RECEIPT_PATH: Literal["completion.json"] = "completion.json"
STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH: Literal["failure.json"] = "failure.json"
STATIC_EVIDENCE_RUN_PLAN_PATH: Literal["input/run-plan.json"] = "input/run-plan.json"
STATIC_EVIDENCE_INTAKE_PATH: Literal["input/intake.json"] = "input/intake.json"
STATIC_EVIDENCE_GATE3_CLOSEOUT_PATH: Literal["input/gate3-closeout.json"] = (
    "input/gate3-closeout.json"
)
STATIC_EVIDENCE_AUTHORING_RESULT_PATH: Literal["artifacts/authoring-result.json"] = (
    "artifacts/authoring-result.json"
)
STATIC_EVIDENCE_GATE3B_SUPPORT_PATH: Literal["reports/gate3b-support.json"] = (
    "reports/gate3b-support.json"
)
STATIC_EVIDENCE_RESULT_BUNDLE_PATH: Literal["result-bundle.json"] = "result-bundle.json"

_FIXED_PATHS = (
    STATIC_EVIDENCE_AUTHORING_RESULT_PATH,
    "claims/gate3a.json",
    "claims/gate3b.json",
    STATIC_EVIDENCE_GATE3_CLOSEOUT_PATH,
    STATIC_EVIDENCE_INTAKE_PATH,
    STATIC_EVIDENCE_RUN_PLAN_PATH,
    "receipts/authoring.json",
    "receipts/contract.json",
    "receipts/readback.json",
    "logs/gate3a.stderr.json",
    "logs/gate3a.stdout.json",
    "logs/gate3b.stderr.json",
    "logs/gate3b.stdout.json",
    "reports/gate3a.json",
    "reports/gate3b.json",
    STATIC_EVIDENCE_GATE3B_SUPPORT_PATH,
    STATIC_EVIDENCE_RESULT_BUNDLE_PATH,
    STATIC_EVIDENCE_TERMINAL_RECEIPT_PATH,
)


class ArticulationV2StaticEvidencePathAllowlistV1(BaseModel):
    """Exact sorted path set for one fixed or distance evidence root."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    schema_version: Literal[
        "joint-agent-articulation-v2-static-evidence-path-allowlist-v1"
    ]
    constraint_kind: ArticulationV2StaticConstraintKind
    generated_output: str
    paths: tuple[str, ...]
    terminal_receipt: Literal["completion.json"]

    @field_validator("generated_output")
    @classmethod
    def _canonical_generated_output(cls, value: str) -> str:
        return _canonical_generated_output_path(value)

    @field_validator("paths")
    @classmethod
    def _canonical_unique_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(_canonical_path(path) for path in value)
        if canonical != tuple(sorted(canonical)):
            raise ValueError("evidence path allowlist must be sorted")
        if len(canonical) != len(set(canonical)):
            raise ValueError("evidence path allowlist must be unique")
        return canonical

    @model_validator(mode="after")
    def _exact_allowlist(self) -> ArticulationV2StaticEvidencePathAllowlistV1:
        expected = articulation_v2_static_evidence_paths(
            constraint_kind=self.constraint_kind,
            generated_output=self.generated_output,
        )
        if self.paths != expected:
            raise ValueError("evidence path allowlist differs from the static contract")
        if self.terminal_receipt != STATIC_EVIDENCE_TERMINAL_RECEIPT_PATH:
            raise ValueError("static evidence requires completion.json as terminal")
        return self

    @property
    def retained_paths(self) -> tuple[str, ...]:
        """Return every required pre-terminal retained file path."""

        return tuple(
            path for path in self.paths if path != STATIC_EVIDENCE_TERMINAL_RECEIPT_PATH
        )


def articulation_v2_static_evidence_paths(
    *,
    constraint_kind: ArticulationV2StaticConstraintKind,
    generated_output: str,
) -> tuple[str, ...]:
    """Build the sole deterministic allowlist for one planned output."""

    output = _canonical_generated_output_path(generated_output)
    if constraint_kind not in {"fixed", "distance"}:
        raise ValueError("constraint_kind must be fixed or distance")
    contract = _canonical_path(f"input/contracts/{constraint_kind}.json")
    return tuple(sorted((*_FIXED_PATHS, contract, output)))


def articulation_v2_static_gate3a_failure_paths(
    *,
    constraint_kind: ArticulationV2StaticConstraintKind,
    generated_output: str,
) -> tuple[str, ...]:
    """Build the exact terminal inventory for a missing Gate 3A report.

    This is deliberately not a qualification allowlist.  It retains the
    completed first-three transaction, the Gate 3 closeout input, and both raw
    Gate 3A process streams, then terminates at ``failure.json`` without a
    validator report, claim, result bundle, or completion receipt.
    """

    output = _canonical_generated_output_path(generated_output)
    if constraint_kind not in {"fixed", "distance"}:
        raise ValueError("constraint_kind must be fixed or distance")
    contract = _canonical_path(f"input/contracts/{constraint_kind}.json")
    return tuple(
        sorted(
            (
                STATIC_EVIDENCE_AUTHORING_RESULT_PATH,
                STATIC_EVIDENCE_GATE3_CLOSEOUT_PATH,
                STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH,
                STATIC_EVIDENCE_INTAKE_PATH,
                STATIC_EVIDENCE_RUN_PLAN_PATH,
                contract,
                "logs/gate3a.stderr.json",
                "logs/gate3a.stdout.json",
                output,
                "receipts/authoring.json",
                "receipts/contract.json",
                "receipts/readback.json",
            )
        )
    )


def articulation_v2_static_gate3b_failure_paths(
    *,
    constraint_kind: ArticulationV2StaticConstraintKind,
    generated_output: str,
) -> tuple[str, ...]:
    """Build the exact terminal inventory for a missing Gate 3B report.

    The completed first three stages and validated Gate 3A evidence remain
    available, but the root terminates at ``failure.json`` after retaining both
    raw Gate 3B process streams.  It contains no Gate 3B report, claim, support
    bundle, result bundle, or completion receipt and cannot qualify the run.
    """

    output = _canonical_generated_output_path(generated_output)
    if constraint_kind not in {"fixed", "distance"}:
        raise ValueError("constraint_kind must be fixed or distance")
    contract = _canonical_path(f"input/contracts/{constraint_kind}.json")
    return tuple(
        sorted(
            (
                STATIC_EVIDENCE_AUTHORING_RESULT_PATH,
                "claims/gate3a.json",
                STATIC_EVIDENCE_GATE3_CLOSEOUT_PATH,
                STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH,
                STATIC_EVIDENCE_INTAKE_PATH,
                STATIC_EVIDENCE_RUN_PLAN_PATH,
                contract,
                "logs/gate3a.stderr.json",
                "logs/gate3a.stdout.json",
                "logs/gate3b.stderr.json",
                "logs/gate3b.stdout.json",
                output,
                "receipts/authoring.json",
                "receipts/contract.json",
                "receipts/readback.json",
                "reports/gate3a.json",
            )
        )
    )


def _canonical_generated_output_path(value: str) -> str:
    output = _canonical_path(value)
    if not output.startswith("output/"):
        raise ValueError("generated evidence must use the output/ namespace")
    if PurePosixPath(output).suffix.lower() not in _USD_EXTENSIONS:
        raise ValueError("generated evidence must name USD or USDZ output")
    return output


def _canonical_path(value: str) -> str:
    try:
        canonical: str = validated_artifact_relative_key(value)
    except ValueError as exc:
        raise ValueError("evidence paths must be canonical relative keys") from exc
    return canonical


__all__ = [
    "ARTICULATION_V2_STATIC_EVIDENCE_PATH_ALLOWLIST_SCHEMA_VERSION",
    "STATIC_EVIDENCE_AUTHORING_RESULT_PATH",
    "STATIC_EVIDENCE_GATE3_FAILURE_RECEIPT_PATH",
    "STATIC_EVIDENCE_GATE3B_SUPPORT_PATH",
    "STATIC_EVIDENCE_GATE3_CLOSEOUT_PATH",
    "STATIC_EVIDENCE_INTAKE_PATH",
    "STATIC_EVIDENCE_RESULT_BUNDLE_PATH",
    "STATIC_EVIDENCE_RUN_PLAN_PATH",
    "STATIC_EVIDENCE_TERMINAL_RECEIPT_PATH",
    "ArticulationV2StaticEvidencePathAllowlistV1",
    "articulation_v2_static_evidence_paths",
    "articulation_v2_static_gate3a_failure_paths",
    "articulation_v2_static_gate3b_failure_paths",
]
