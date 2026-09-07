# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-owned external-runtime (BYOR) refinement loop contracts.

These models define the artifact contract between the coding agent (which
owns the refinement outer loop: reviewing rendered sweep evidence against the
user's behavior goal, revising the active parameter search, and deciding when
to stop) and the deterministic wrapper (which owns qualification approval,
enforces the sweep budget, verifies the digest-bound decision chain against
the external sweep broker's ledger, and publishes the accepted result).

This is the agentic counterpart of ``physics-agent refine-external``'s
built-in loop: the agent's visual review replaces the VLM judge and the
agent's ``revise_search`` decisions replace the LLM refiner. Unlike the
in-house tuning loop there is no USD candidate to materialize — the
deliverable is the accepted sweep's best parameters plus its exact,
self-contained rollout recording, both pinned by content digest.

Decision chain integrity mirrors ``tuning_contract``: every decision after
the first carries the SHA-256 of the previous decision file's bytes in
``prior_decision_sha256``. ``accept``/``revise_search`` decisions must bind
the judged sweep's ``evidence_sha256``; ``accept`` additionally binds the
rendered frames the agent actually reviewed (``reviewed_frames``) and the
selected recording digest. The wrapper compares every digest against the
broker ledger and rehashes the referenced artifacts at conclusion.

Terminal semantics: only ``accepted`` may publish a final result.
``stopped``, ``budget_exhausted``, ``unresolved``, and ``tool_failure`` are
honest failures — the wrapper must not publish anything for them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from content_agent_workflows.physics.tuning_contract import (
    _coerce_text,
    _drop_unknown_keys,
    sha256_file,
)

EXTERNAL_TUNING_DECISION_SCHEMA_VERSION = (
    "content-agents.physics-external-tuning-decision.v1"
)
EXTERNAL_TUNING_RESULT_SCHEMA_VERSION = (
    "content-agents.physics-external-tuning-result.v1"
)

PhysicsExternalTuningDecisionKind = Literal["accept", "revise_search", "stop"]
PhysicsExternalTuningTerminalStatus = Literal[
    "accepted", "stopped", "budget_exhausted", "unresolved", "tool_failure"
]
EXTERNAL_PROMOTABLE_STATUSES: frozenset[str] = frozenset({"accepted"})


class PhysicsExternalReviewedFrame(BaseModel):
    """One rendered evidence frame the agent reviewed, pinned by digest."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _drop_unknown_keys(dict(data), cls)
        return data


class PhysicsExternalSelectedResult(BaseModel):
    """The sweep result the agent selected, pinned by content digests."""

    model_config = ConfigDict(extra="forbid")

    sweep_id: str = Field(min_length=1)
    best_params: dict[str, float] = Field(min_length=1)
    evidence_sha256: str = Field(min_length=64, max_length=64)
    recording_sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _drop_unknown_keys(dict(data), cls)
        return data


class PhysicsExternalTuningDecision(BaseModel):
    """One digest-bound outer-loop decision authored by the coding agent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXTERNAL_TUNING_DECISION_SCHEMA_VERSION
    iteration: int = Field(ge=1)
    decision: PhysicsExternalTuningDecisionKind
    sweep_id: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    reviewed_frames: list[PhysicsExternalReviewedFrame] = Field(default_factory=list)
    selected: PhysicsExternalSelectedResult | None = None
    next_active_search: dict[str, dict[str, float]] | None = None
    prior_decision_sha256: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["rationale"] = _coerce_text(data.get("rationale"))
        return _drop_unknown_keys(data, cls)

    @model_validator(mode="after")
    def _check_decision_requirements(self) -> PhysicsExternalTuningDecision:
        if self.decision in {"accept", "revise_search"}:
            if not self.sweep_id:
                raise ValueError(
                    f"{self.decision} decisions must reference the sweep_id they judged"
                )
        if self.decision == "accept" and not self.evidence_sha256:
            # revise_search may legitimately cite a FAILED sweep, which never
            # publishes evidence, so a bare digest omission stays
            # schema-valid for it (the wrapper tolerates the omission at
            # conclusion; a digest that IS present is still fully verified
            # against the broker ledger).
            raise ValueError(
                "accept decisions require evidence_sha256 "
                "binding the judged sweep evidence"
            )
        if (
            self.decision == "revise_search"
            and self.evidence_path
            and not self.evidence_sha256
        ):
            # Write-time guard for the common slip: a revise_search that
            # records which evidence file it judged must bind its digest
            # while the agent can still correct the file, not fail hours
            # later at conclusion.
            raise ValueError(
                "revise_search decisions that cite the evidence they judged "
                "must bind evidence_sha256 alongside evidence_path"
            )
        if self.decision == "accept":
            if self.selected is None:
                raise ValueError("accept decisions require a selected result")
            if not self.reviewed_frames:
                raise ValueError(
                    "accept decisions require reviewed_frames: the agent must "
                    "bind the rendered evidence frames it visually reviewed"
                )
            if self.selected.sweep_id != self.sweep_id:
                raise ValueError(
                    "accept decisions must select the sweep they cite: "
                    f"sweep_id {self.sweep_id!r} does not match "
                    f"selected.sweep_id {self.selected.sweep_id!r}"
                )
            if self.selected.evidence_sha256 != self.evidence_sha256:
                raise ValueError(
                    "accept decisions must bind the same evidence digest in "
                    "selected.evidence_sha256 and evidence_sha256"
                )
            if not self.selected.recording_sha256:
                # A succeeded sweep always publishes a digest-bound recording
                # (evidence publication fails closed without one), so an
                # accept that does not claim it cannot be verified against —
                # or published from — the broker record.
                raise ValueError(
                    "accept decisions must bind selected.recording_sha256, "
                    "copied exactly from the accepted sweep record"
                )
        if self.decision == "revise_search":
            if not self.next_active_search:
                raise ValueError("revise_search decisions require next_active_search")
            for name, bounds in self.next_active_search.items():
                if {"min", "max"} - set(bounds):
                    raise ValueError(
                        f"next_active_search parameter {name!r} must provide "
                        "'min' and 'max'"
                    )
        return self


class PhysicsExternalTuningResult(BaseModel):
    """Terminal result of the agent-owned external refinement loop."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXTERNAL_TUNING_RESULT_SCHEMA_VERSION
    status: PhysicsExternalTuningTerminalStatus
    selected: PhysicsExternalSelectedResult | None = None
    decision_paths: list[str] = Field(default_factory=list)
    final_decision_sha256: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["rationale"] = _coerce_text(data.get("rationale"))
        paths = data.get("decision_paths")
        if isinstance(paths, str):
            data["decision_paths"] = [paths]
        return _drop_unknown_keys(data, cls)

    @model_validator(mode="after")
    def _check_terminal_semantics(self) -> PhysicsExternalTuningResult:
        if self.status == "accepted" and self.selected is None:
            raise ValueError("accepted results require a selected result")
        if self.status == "accepted":
            assert self.selected is not None
            if not self.selected.recording_sha256:
                raise ValueError(
                    "accepted results must bind selected.recording_sha256, "
                    "copied exactly from the accepted sweep record"
                )
        if self.status != "accepted" and self.selected is not None:
            raise ValueError(
                f"status {self.status!r} must not carry a selected result; "
                "only accepted results may publish"
            )
        return self

    @property
    def promotable(self) -> bool:
        return self.status in EXTERNAL_PROMOTABLE_STATUSES


def load_physics_external_tuning_decision(
    path: Path | str,
) -> PhysicsExternalTuningDecision:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhysicsExternalTuningDecision.model_validate(payload)


def load_physics_external_tuning_result(
    path: Path | str,
) -> PhysicsExternalTuningResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhysicsExternalTuningResult.model_validate(payload)


def verify_external_decision_chain(
    decision_paths: list[Path | str],
) -> list[PhysicsExternalTuningDecision]:
    """Load decisions in order and verify the SHA-256 hash chain.

    Chain rule: decision ``i`` (1-indexed) must carry
    ``prior_decision_sha256 == sha256(bytes of decision file i-1)``; the
    first decision must carry ``None``. Iterations must be 1..N in order.
    ``accept`` and ``stop`` are terminal: only the final decision may carry
    them, so a chain cannot continue past a concluded loop (for example
    ``stop`` followed by ``accept``, or two accepts).

    Raises ``ValueError`` on any break; returns the parsed decisions.
    """

    decisions: list[PhysicsExternalTuningDecision] = []
    prior_sha: str | None = None
    for index, raw_path in enumerate(decision_paths, start=1):
        path = Path(raw_path)
        decision = load_physics_external_tuning_decision(path)
        if decision.iteration != index:
            raise ValueError(
                f"decision chain broken at {path}: iteration "
                f"{decision.iteration} != expected {index}"
            )
        if decision.prior_decision_sha256 != prior_sha:
            raise ValueError(
                f"decision chain broken at {path}: prior_decision_sha256 "
                f"{decision.prior_decision_sha256!r} != expected {prior_sha!r}"
            )
        if decisions and decisions[-1].decision != "revise_search":
            raise ValueError(
                f"decision chain broken at {path}: decision "
                f"{decisions[-1].iteration} was terminal "
                f"({decisions[-1].decision!r}); every non-final decision "
                "must be revise_search"
            )
        decisions.append(decision)
        prior_sha = sha256_file(path)
    return decisions


__all__ = [
    "EXTERNAL_PROMOTABLE_STATUSES",
    "EXTERNAL_TUNING_DECISION_SCHEMA_VERSION",
    "EXTERNAL_TUNING_RESULT_SCHEMA_VERSION",
    "PhysicsExternalReviewedFrame",
    "PhysicsExternalSelectedResult",
    "PhysicsExternalTuningDecision",
    "PhysicsExternalTuningResult",
    "load_physics_external_tuning_decision",
    "load_physics_external_tuning_result",
    "verify_external_decision_chain",
]
