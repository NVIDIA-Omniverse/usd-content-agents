# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External-runtime tuning contract tests: decision requirements, terminal
semantics, and hash-chain verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from content_agent_workflows.physics.external_tuning_contract import (
    PhysicsExternalTuningDecision,
    PhysicsExternalTuningResult,
    sha256_file,
    verify_external_decision_chain,
)

_DIGEST = "a" * 64
_FRAME = {
    "path": "/run/tuning/iter_1/evidence-x/frames/frame_0000.png",
    "sha256": _DIGEST,
}


def _accept_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "iteration": 1,
        "decision": "accept",
        "sweep_id": "external-sweep-001-ab",
        "evidence_sha256": _DIGEST,
        "reviewed_frames": [_FRAME],
        "selected": {
            "sweep_id": "external-sweep-001-ab",
            "best_params": {"restitution": 0.8},
            "evidence_sha256": _DIGEST,
            "recording_sha256": _DIGEST,
        },
        "rationale": "matches the goal",
    }
    payload.update(overrides)
    return payload


def test_accept_decision_requirements() -> None:
    decision = PhysicsExternalTuningDecision.model_validate(_accept_payload())
    assert decision.selected is not None
    assert decision.reviewed_frames[0].sha256 == _DIGEST

    with pytest.raises(ValueError, match="reviewed_frames"):
        PhysicsExternalTuningDecision.model_validate(
            _accept_payload(reviewed_frames=[])
        )
    with pytest.raises(ValueError, match="selected result"):
        PhysicsExternalTuningDecision.model_validate(_accept_payload(selected=None))
    with pytest.raises(ValueError, match="evidence_sha256"):
        PhysicsExternalTuningDecision.model_validate(
            _accept_payload(evidence_sha256=None)
        )
    mismatched = _accept_payload()
    mismatched["selected"]["sweep_id"] = "other-sweep"
    with pytest.raises(ValueError, match="select the sweep they cite"):
        PhysicsExternalTuningDecision.model_validate(mismatched)
    divergent = _accept_payload()
    divergent["selected"]["evidence_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="same evidence digest"):
        PhysicsExternalTuningDecision.model_validate(divergent)
    # A succeeded sweep always publishes a digest-bound recording, so an
    # accept must claim it: verification and publication compare against
    # exactly this digest.
    unclaimed = _accept_payload()
    unclaimed["selected"]["recording_sha256"] = None
    with pytest.raises(ValueError, match="recording_sha256"):
        PhysicsExternalTuningDecision.model_validate(unclaimed)


def test_revise_search_requires_bounds() -> None:
    decision = PhysicsExternalTuningDecision.model_validate(
        {
            "iteration": 2,
            "decision": "revise_search",
            "sweep_id": "external-sweep-001-ab",
            "evidence_sha256": _DIGEST,
            "next_active_search": {"restitution": {"min": 0.4, "max": 0.9}},
            "prior_decision_sha256": _DIGEST,
            "rationale": "widen the rail",
        }
    )
    assert decision.next_active_search is not None

    with pytest.raises(ValueError, match="next_active_search"):
        PhysicsExternalTuningDecision.model_validate(
            {
                "iteration": 2,
                "decision": "revise_search",
                "sweep_id": "s",
                "evidence_sha256": _DIGEST,
                "rationale": "no search",
            }
        )
    with pytest.raises(ValueError, match="'min' and 'max'"):
        PhysicsExternalTuningDecision.model_validate(
            {
                "iteration": 2,
                "decision": "revise_search",
                "sweep_id": "s",
                "evidence_sha256": _DIGEST,
                "next_active_search": {"restitution": {"min": 0.4}},
                "rationale": "half bounds",
            }
        )


def test_revise_search_citing_evidence_must_bind_its_digest() -> None:
    """A revise_search that records which evidence file it judged must bind
    the digest at write time, while the agent can still correct the file —
    not hours later at conclusion."""

    with pytest.raises(ValueError, match="alongside evidence_path"):
        PhysicsExternalTuningDecision.model_validate(
            {
                "iteration": 2,
                "decision": "revise_search",
                "sweep_id": "external-sweep-001-ab",
                "evidence_path": "/run/tuning/iter_1/evidence-x/evidence.json",
                "next_active_search": {"restitution": {"min": 0.4, "max": 0.9}},
                "prior_decision_sha256": _DIGEST,
                "rationale": "judged evidence without binding it",
            }
        )


def test_revise_search_may_cite_a_failed_sweep_without_evidence() -> None:
    """A failed or deadline-exceeded sweep never publishes evidence, yet it is
    the sweep a revise_search legitimately learns from. The digest binding is
    enforced by broker verification instead, which rejects a missing or
    mismatched digest whenever the cited sweep did publish evidence."""

    decision = PhysicsExternalTuningDecision.model_validate(
        {
            "iteration": 2,
            "decision": "revise_search",
            "sweep_id": "external-sweep-001-ab",
            "next_active_search": {"restitution": {"min": 0.4, "max": 0.9}},
            "prior_decision_sha256": _DIGEST,
            "rationale": "sweep failed under the deadline; narrow and retry",
        }
    )
    assert decision.evidence_sha256 is None


def test_stop_decision_needs_no_sweep() -> None:
    decision = PhysicsExternalTuningDecision.model_validate(
        {"iteration": 1, "decision": "stop", "rationale": "unreachable"}
    )
    assert decision.sweep_id is None


def test_result_terminal_semantics() -> None:
    accepted = PhysicsExternalTuningResult.model_validate(
        {
            "status": "accepted",
            "selected": {
                "sweep_id": "s",
                "best_params": {"restitution": 0.8},
                "evidence_sha256": _DIGEST,
                "recording_sha256": _DIGEST,
            },
            "rationale": "done",
        }
    )
    assert accepted.promotable

    with pytest.raises(ValueError, match="recording_sha256"):
        PhysicsExternalTuningResult.model_validate(
            {
                "status": "accepted",
                "selected": {
                    "sweep_id": "s",
                    "best_params": {"restitution": 0.8},
                    "evidence_sha256": _DIGEST,
                },
                "rationale": "no recording claim",
            }
        )

    with pytest.raises(ValueError, match="require a selected result"):
        PhysicsExternalTuningResult.model_validate(
            {"status": "accepted", "rationale": "no selection"}
        )
    with pytest.raises(ValueError, match="must not carry"):
        PhysicsExternalTuningResult.model_validate(
            {
                "status": "stopped",
                "selected": {
                    "sweep_id": "s",
                    "best_params": {"restitution": 0.8},
                    "evidence_sha256": _DIGEST,
                },
                "rationale": "sneaky",
            }
        )
    stopped = PhysicsExternalTuningResult.model_validate(
        {"status": "stopped", "rationale": "honest"}
    )
    assert not stopped.promotable


def _write_decision(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_external_decision_chain(tmp_path: Path) -> None:
    first = _write_decision(
        tmp_path / "physics_external_tuning_decision_1.json",
        {
            "iteration": 1,
            "decision": "revise_search",
            "sweep_id": "s1",
            "evidence_sha256": _DIGEST,
            "next_active_search": {"restitution": {"min": 0.0, "max": 0.5}},
            "rationale": "first",
        },
    )
    second = _write_decision(
        tmp_path / "physics_external_tuning_decision_2.json",
        _accept_payload(iteration=2, prior_decision_sha256=sha256_file(first)),
    )
    decisions = verify_external_decision_chain([first, second])
    assert [decision.iteration for decision in decisions] == [1, 2]

    # A broken chain digest is rejected.
    tampered = _write_decision(
        tmp_path / "physics_external_tuning_decision_2b.json",
        _accept_payload(iteration=2, prior_decision_sha256="c" * 64),
    )
    with pytest.raises(ValueError, match="prior_decision_sha256"):
        verify_external_decision_chain([first, tampered])
    # Iterations must be contiguous from 1.
    with pytest.raises(ValueError, match="iteration"):
        verify_external_decision_chain([second])


def test_chain_rejects_decisions_after_a_terminal_action(tmp_path: Path) -> None:
    """accept and stop conclude the loop: a chain must not continue past
    them (stop -> accept, accept -> accept), so every non-final decision
    must be revise_search."""

    stop = _write_decision(
        tmp_path / "physics_external_tuning_decision_1.json",
        {"iteration": 1, "decision": "stop", "rationale": "concluded"},
    )
    late_accept = _write_decision(
        tmp_path / "physics_external_tuning_decision_2.json",
        _accept_payload(iteration=2, prior_decision_sha256=sha256_file(stop)),
    )
    with pytest.raises(ValueError, match="must be revise_search"):
        verify_external_decision_chain([stop, late_accept])

    first_accept = _write_decision(
        tmp_path / "physics_external_tuning_decision_1.json",
        _accept_payload(),
    )
    second_accept = _write_decision(
        tmp_path / "physics_external_tuning_decision_2.json",
        _accept_payload(iteration=2, prior_decision_sha256=sha256_file(first_accept)),
    )
    with pytest.raises(ValueError, match="must be revise_search"):
        verify_external_decision_chain([first_accept, second_accept])
