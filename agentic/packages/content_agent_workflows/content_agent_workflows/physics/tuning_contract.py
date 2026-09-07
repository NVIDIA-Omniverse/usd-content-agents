# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-owned physics tuning loop contracts.

These models define the artifact contract between the coding agent (which
owns the tuning outer loop: judging sweep evidence, revising scenarios or
physics patches, and deciding when to stop) and the deterministic wrapper
(which enforces budgets, verifies the digest-bound decision chain against the
sweep broker's ledger, and promotes the accepted candidate).

Decision chain integrity: every decision after the first must carry the
SHA-256 of the *previous decision file's bytes* in ``prior_decision_sha256``,
and references sweeps/evidence/candidates by content digest. The digest
bindings are mandatory, not advisory: every decision that cites a sweep must
carry ``scenario_sha256`` and every digest available in that broker record.
Successful sweeps always provide ``evidence_sha256``. ``accept`` additionally
requires the selected candidate's ``usd_sha256``; ``revise_patch`` requires a
``rebuilt_physics_usd_sha256`` for any rebuilt USD it names. The wrapper
compares every digest unconditionally against the broker ledger and rehashes
the referenced artifacts at conclusion — an agent cannot swap or edit
artifacts after the fact, nor omit a digest to skip the comparison.

Terminal semantics: only ``accepted`` may promote a candidate. ``stopped``,
``budget_exhausted``, ``unresolved``, and ``tool_failure`` are honest
failures — the wrapper must not promote anything for them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TUNING_DECISION_SCHEMA_VERSION = "content-agents.physics-tuning-decision.v1"
TUNING_RESULT_SCHEMA_VERSION = "content-agents.physics-tuning-result.v1"

PhysicsTuningDecisionKind = Literal["accept", "revise_scenario", "revise_patch", "stop"]
PhysicsTuningTerminalStatus = Literal[
    "accepted", "stopped", "budget_exhausted", "unresolved", "tool_failure"
]
PROMOTABLE_STATUSES: frozenset[str] = frozenset({"accepted"})


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_text(value: Any) -> Any:
    """Coerce LLM-shaped rationale values (lists/objects) to a string."""

    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if value is not None and not isinstance(value, str):
        return str(value)
    return value


def _drop_unknown_keys(data: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in model.model_fields}


class PhysicsTuningSelectedCandidate(BaseModel):
    """A sweep trial the agent selected, pinned by content digest."""

    model_config = ConfigDict(extra="forbid")

    sweep_id: str = Field(min_length=1)
    trial_index: int = Field(ge=0)
    usd_path: str | None = None
    usd_sha256: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _drop_unknown_keys(dict(data), cls)
        return data


class PhysicsTuningDecision(BaseModel):
    """One digest-bound outer-loop decision authored by the coding agent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TUNING_DECISION_SCHEMA_VERSION
    iteration: int = Field(ge=1)
    decision: PhysicsTuningDecisionKind
    sweep_id: str | None = None
    scenario_path: str | None = None
    scenario_sha256: str | None = None
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    selected: PhysicsTuningSelectedCandidate | None = None
    next_scenario_path: str | None = None
    revised_patch_path: str | None = None
    rebuilt_physics_usd: str | None = None
    rebuilt_physics_usd_sha256: str | None = None
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

    @field_validator("rationale")
    @classmethod
    def _require_non_blank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must contain non-whitespace characters")
        return value

    @field_validator("sweep_id")
    @classmethod
    def _require_non_blank_sweep_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("sweep_id must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _check_decision_requirements(self) -> PhysicsTuningDecision:
        if self.sweep_id is not None and not self.scenario_sha256:
            raise ValueError(
                f"{self.decision} decisions citing a sweep require "
                "scenario_sha256 binding the sweep scenario"
            )
        if self.decision == "stop" and self.sweep_id is None:
            orphaned_fields = (
                "scenario_path",
                "scenario_sha256",
                "evidence_path",
                "evidence_sha256",
                "selected",
            )
            populated = [
                field_name
                for field_name in orphaned_fields
                if getattr(self, field_name) is not None
            ]
            if populated:
                raise ValueError(
                    "sweep-less stop decisions cannot carry sweep artifact fields: "
                    + ", ".join(populated)
                )
        if self.decision == "accept":
            if self.selected is None:
                raise ValueError("accept decisions require a selected candidate")
            if not self.selected.usd_path:
                raise ValueError(
                    "accept decisions require selected.usd_path (the "
                    "broker-materialized candidate path)"
                )
            if not self.selected.usd_sha256:
                raise ValueError(
                    "accept decisions require selected.usd_sha256 (the "
                    "materialized candidate digest)"
                )
            if not self.evidence_sha256:
                raise ValueError(
                    "accept decisions require evidence_sha256 binding the "
                    "judged sweep evidence"
                )
            if (
                self.sweep_id
                and self.selected.sweep_id
                and self.selected.sweep_id != self.sweep_id
            ):
                # The wrapper verifies scenario/evidence digests against the
                # decision's sweep_id but materializes from selected.sweep_id.
                # Allowing them to differ would let the evidence digests
                # describe a sweep other than the one the promoted candidate
                # came from.
                raise ValueError(
                    "accept decisions must select a candidate from the sweep "
                    f"they cite: sweep_id {self.sweep_id!r} does not match "
                    f"selected.sweep_id {self.selected.sweep_id!r}"
                )
        if self.decision == "revise_scenario" and not self.next_scenario_path:
            raise ValueError("revise_scenario decisions require next_scenario_path")
        if self.decision == "revise_patch":
            if not self.revised_patch_path:
                raise ValueError("revise_patch decisions require revised_patch_path")
            if self.rebuilt_physics_usd and not self.rebuilt_physics_usd_sha256:
                raise ValueError(
                    "revise_patch decisions naming a rebuilt_physics_usd must "
                    "bind it with rebuilt_physics_usd_sha256"
                )
        if self.decision in {"accept", "revise_scenario", "revise_patch"}:
            if self.sweep_id is None:
                raise ValueError(
                    f"{self.decision} decisions must reference the sweep_id they judged"
                )
        return self


class PhysicsTuningResult(BaseModel):
    """Terminal result of the agent-owned tuning loop."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TUNING_RESULT_SCHEMA_VERSION
    status: PhysicsTuningTerminalStatus
    selected: PhysicsTuningSelectedCandidate | None = None
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

    @field_validator("rationale")
    @classmethod
    def _require_non_blank_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rationale must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def _check_terminal_semantics(self) -> PhysicsTuningResult:
        if self.status == "accepted":
            if self.selected is None:
                raise ValueError("accepted results require a selected candidate")
            if not self.selected.usd_path:
                raise ValueError(
                    "accepted results require selected.usd_path (the "
                    "broker-materialized candidate path)"
                )
            if not self.selected.usd_sha256:
                raise ValueError(
                    "accepted results require selected.usd_sha256 (the "
                    "materialized candidate digest)"
                )
        if self.status != "accepted" and self.selected is not None:
            raise ValueError(
                f"status {self.status!r} must not carry a selected candidate; "
                "only accepted results may promote"
            )
        return self

    @property
    def promotable(self) -> bool:
        return self.status in PROMOTABLE_STATUSES


def load_physics_tuning_decision(path: Path | str) -> PhysicsTuningDecision:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhysicsTuningDecision.model_validate(payload)


def load_physics_tuning_result(path: Path | str) -> PhysicsTuningResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PhysicsTuningResult.model_validate(payload)


def verify_decision_chain(
    decision_paths: list[Path | str],
    *,
    max_iterations: int | None = None,
) -> list[PhysicsTuningDecision]:
    """Load decisions in order and verify the SHA-256 hash chain.

    Chain rule: decision ``i`` (1-indexed) must carry
    ``prior_decision_sha256 == sha256(bytes of decision file i-1)``; the first
    decision must carry ``None``. Iterations must be 1..N in order. The shared
    refinement transition primitive is replayed over the artifact chain, so
    ``accept`` and ``stop`` are terminal. ``max_iterations`` is the broker's
    sweep budget, so each nonterminal revision must cite a newly reserved
    sweep. A terminal decision may reuse only the immediately preceding sweep
    without consuming another reservation. It must still carry every
    scenario/evidence digest present in that broker record; a successful sweep
    always has an evidence digest. A sweep-less terminal ``stop`` claims no
    broker sweep.

    Raises ``ValueError`` on any break; returns the parsed decisions.
    """

    from world_understanding.optimization import RefinementLoop

    if max_iterations is not None and (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be positive")

    # Replay validates an already agent-authored trace; it does not move
    # judgment, revision, or completion ownership into the wrapper.
    replay_iterations = (
        max_iterations + 1
        if max_iterations is not None
        else max(1, len(decision_paths))
    )
    refinement = RefinementLoop[PhysicsTuningDecision | None](
        initial_state=None,
        max_iterations=replay_iterations,
    )
    decisions: list[PhysicsTuningDecision] = []
    prior_sha: str | None = None
    referenced_sweep_ids: set[str] = set()
    previous_sweep_id: str | None = None
    for index, raw_path in enumerate(decision_paths, start=1):
        path = Path(raw_path)
        decision = load_physics_tuning_decision(path)
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

        iteration = refinement.begin_iteration()
        if iteration is None:
            if decisions and decisions[-1].decision in {"accept", "stop"}:
                terminal = decisions[-1]
                detail = (
                    f"decision {terminal.iteration} was terminal "
                    f"({terminal.decision!r}); terminal actions must "
                    "terminate the verified decision chain"
                )
            else:
                detail = "the refinement replay terminated before this decision"
            raise ValueError(f"decision chain broken at {path}: {detail}")

        if decision.sweep_id is not None:
            already_referenced = decision.sweep_id in referenced_sweep_ids
            is_terminal = decision.decision in {"accept", "stop"}
            if already_referenced and (
                not is_terminal or decision.sweep_id != previous_sweep_id
            ):
                raise ValueError(
                    f"decision chain broken at {path}: sweep_id "
                    f"{decision.sweep_id!r} was already judged; each nonterminal "
                    "decision must cite a new sweep, and only a terminal "
                    "decision may reuse the immediately preceding sweep"
                )
            if not already_referenced:
                referenced_sweep_ids.add(decision.sweep_id)
                if (
                    max_iterations is not None
                    and len(referenced_sweep_ids) > max_iterations
                ):
                    raise ValueError(
                        f"decision chain broken at {path}: configured "
                        f"max_iterations={max_iterations} was exhausted"
                    )

        if decision.decision == "accept":
            refinement.approve()
        elif decision.decision == "stop":
            refinement.stop(decision.rationale)
        else:
            refinement.continue_with(decision)
        decisions.append(decision)
        prior_sha = sha256_file(path)
        previous_sweep_id = decision.sweep_id
    return decisions
