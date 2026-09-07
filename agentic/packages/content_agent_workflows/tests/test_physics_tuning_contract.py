# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Digest-bound tuning decision/result contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from content_agent_workflows.physics import (
    PhysicsTuningDecision,
    PhysicsTuningResult,
    verify_decision_chain,
)
from content_agent_workflows.physics.tuning_contract import sha256_file


def _decision_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "content-agents.physics-tuning-decision.v1",
        "iteration": 1,
        "decision": "revise_scenario",
        "sweep_id": "sweep-001-abcd",
        "scenario_sha256": "a" * 64,
        "next_scenario_path": "tuning/iter_2/scenario.yaml",
        "prior_decision_sha256": None,
        "rationale": "Restitution range too low for the bounce goal.",
    }
    payload.update(overrides)
    return payload


def test_accept_requires_selected_candidate() -> None:
    with pytest.raises(ValueError, match="selected candidate"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(decision="accept", next_scenario_path=None)
        )


def test_revise_scenario_requires_next_scenario_path() -> None:
    with pytest.raises(ValueError, match="next_scenario_path"):
        PhysicsTuningDecision.model_validate(_decision_payload(next_scenario_path=None))


def test_revise_patch_requires_revised_patch_path() -> None:
    with pytest.raises(ValueError, match="revised_patch_path"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(decision="revise_patch", next_scenario_path=None)
        )


def test_judging_decisions_require_a_sweep_reference() -> None:
    with pytest.raises(ValueError, match="sweep_id"):
        PhysicsTuningDecision.model_validate(_decision_payload(sweep_id=None))


def test_stop_decision_needs_no_sweep_artifacts() -> None:
    decision = PhysicsTuningDecision.model_validate(
        _decision_payload(
            decision="stop",
            sweep_id=None,
            scenario_sha256=None,
            next_scenario_path=None,
        )
    )
    assert decision.decision == "stop"


def test_stop_decision_citing_sweep_requires_scenario_digest() -> None:
    with pytest.raises(ValueError, match="scenario_sha256"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="stop",
                scenario_sha256=None,
                next_scenario_path=None,
            )
        )


@pytest.mark.parametrize("sweep_id", ["", " \t\n"])
def test_decision_rejects_blank_sweep_id(sweep_id: str) -> None:
    with pytest.raises(ValueError, match="sweep_id must contain non-whitespace"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="stop",
                sweep_id=sweep_id,
                scenario_sha256=None,
                next_scenario_path=None,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("scenario_path", "tuning/iter_1/scenario.yaml"),
        ("scenario_sha256", "a" * 64),
        ("evidence_path", "tuning/iter_1/evidence.json"),
        ("evidence_sha256", "b" * 64),
        (
            "selected",
            {
                "sweep_id": "sweep-001-abcd",
                "trial_index": 0,
                "usd_path": "tuning/iter_1/candidates/trial_0000.usda",
                "usd_sha256": "c" * 64,
            },
        ),
    ],
)
def test_sweep_less_stop_rejects_orphaned_sweep_artifact_fields(
    field_name: str,
    value: object,
) -> None:
    payload = _decision_payload(
        decision="stop",
        sweep_id=None,
        scenario_sha256=None,
        next_scenario_path=None,
    )
    payload[field_name] = value

    with pytest.raises(ValueError, match="sweep artifact fields"):
        PhysicsTuningDecision.model_validate(payload)


@pytest.mark.parametrize("rationale", [" ", "\t\n", [" "]])
def test_decision_rejects_whitespace_only_rationale(rationale: object) -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        PhysicsTuningDecision.model_validate(_decision_payload(rationale=rationale))


def test_result_rejects_whitespace_only_rationale() -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        PhysicsTuningResult.model_validate({"status": "stopped", "rationale": " \n\t"})


def test_digest_bindings_are_mandatory() -> None:
    """accept/revise decisions must digest-bind what they reference."""

    with pytest.raises(ValueError, match="scenario_sha256"):
        PhysicsTuningDecision.model_validate(_decision_payload(scenario_sha256=None))

    selected = {
        "sweep_id": "sweep-001-abcd",
        "trial_index": 0,
        "usd_path": "tuning/iter_1/candidates/trial_0000.usda",
        "usd_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="evidence_sha256"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="accept", next_scenario_path=None, selected=selected
            )
        )
    with pytest.raises(ValueError, match="usd_sha256"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="accept",
                next_scenario_path=None,
                evidence_sha256="b" * 64,
                selected={
                    "sweep_id": "sweep-001-abcd",
                    "trial_index": 0,
                    "usd_path": "tuning/iter_1/candidates/trial_0000.usda",
                },
            )
        )
    with pytest.raises(ValueError, match="usd_path"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="accept",
                next_scenario_path=None,
                evidence_sha256="b" * 64,
                selected={
                    "sweep_id": "sweep-001-abcd",
                    "trial_index": 0,
                    "usd_sha256": "c" * 64,
                },
            )
        )
    accept = PhysicsTuningDecision.model_validate(
        _decision_payload(
            decision="accept",
            next_scenario_path=None,
            evidence_sha256="b" * 64,
            selected=selected,
        )
    )
    assert accept.decision == "accept"


def test_accept_selected_candidate_must_come_from_the_cited_sweep() -> None:
    """The evidence digests are verified against the decision's sweep_id but
    the candidate is materialized from selected.sweep_id; they must agree or
    the evidence describes a different sweep than the promoted candidate."""

    with pytest.raises(ValueError, match="does not match selected.sweep_id"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="accept",
                next_scenario_path=None,
                evidence_sha256="b" * 64,
                sweep_id="sweep-001-abcd",
                selected={
                    "sweep_id": "sweep-002-efgh",
                    "trial_index": 0,
                    "usd_path": "tuning/iter_2/candidates/trial_0000.usdc",
                    "usd_sha256": "c" * 64,
                },
            )
        )


def test_revise_patch_rebuilt_usd_requires_digest() -> None:
    with pytest.raises(ValueError, match="rebuilt_physics_usd_sha256"):
        PhysicsTuningDecision.model_validate(
            _decision_payload(
                decision="revise_patch",
                next_scenario_path=None,
                revised_patch_path="raw/patch_v2.json",
                rebuilt_physics_usd="tuning/rebuilt.usda",
            )
        )
    decision = PhysicsTuningDecision.model_validate(
        _decision_payload(
            decision="revise_patch",
            next_scenario_path=None,
            revised_patch_path="raw/patch_v2.json",
            rebuilt_physics_usd="tuning/rebuilt.usda",
            rebuilt_physics_usd_sha256="d" * 64,
        )
    )
    assert decision.rebuilt_physics_usd_sha256 == "d" * 64


def test_accepted_result_requires_selected_digest() -> None:
    with pytest.raises(ValueError, match="usd_sha256"):
        PhysicsTuningResult.model_validate(
            {
                "status": "accepted",
                "selected": {
                    "sweep_id": "sweep-001-abcd",
                    "trial_index": 2,
                    "usd_path": "tuning/iter_1/candidates/trial_0002.usda",
                },
                "decision_paths": ["raw/physics_tuning_decision_1.json"],
                "final_decision_sha256": "1" * 64,
                "rationale": "matched the behavior goal",
            }
        )

    with pytest.raises(ValueError, match="usd_path"):
        PhysicsTuningResult.model_validate(
            {
                "status": "accepted",
                "selected": {
                    "sweep_id": "sweep-001-abcd",
                    "trial_index": 2,
                    "usd_sha256": "c" * 64,
                },
                "decision_paths": ["raw/physics_tuning_decision_1.json"],
                "final_decision_sha256": "1" * 64,
                "rationale": "matched the behavior goal",
            }
        )


def test_lenient_coercion_drops_unknown_keys_and_joins_rationale() -> None:
    decision = PhysicsTuningDecision.model_validate(
        _decision_payload(
            rationale=["bounds too tight", "widen restitution"],
            confidence=0.9,  # unknown key an LLM might add
        )
    )
    assert decision.rationale == "bounds too tight\nwiden restitution"


def test_result_terminal_semantics_only_accepted_promotes() -> None:
    selected = {
        "sweep_id": "sweep-001-abcd",
        "trial_index": 2,
        "usd_path": "tuning/iter_1/candidates/trial_0002.usda",
        "usd_sha256": "0" * 64,
    }
    accepted = PhysicsTuningResult.model_validate(
        {
            "status": "accepted",
            "selected": selected,
            "decision_paths": ["raw/physics_tuning_decision_1.json"],
            "final_decision_sha256": "1" * 64,
            "rationale": "matched the behavior goal",
        }
    )
    assert accepted.promotable

    with pytest.raises(ValueError, match="selected candidate"):
        PhysicsTuningResult.model_validate(
            {"status": "accepted", "rationale": "missing selected"}
        )

    for status in ("stopped", "budget_exhausted", "unresolved", "tool_failure"):
        result = PhysicsTuningResult.model_validate(
            {"status": status, "rationale": "honest failure"}
        )
        assert not result.promotable
        with pytest.raises(ValueError, match="must not carry"):
            PhysicsTuningResult.model_validate(
                {"status": status, "selected": selected, "rationale": "sneaky"}
            )


def _write_decision(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_verify_decision_chain_accepts_linked_stop_chain(tmp_path: Path) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json", _decision_payload()
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            decision="stop",
            sweep_id=None,
            scenario_sha256=None,
            next_scenario_path=None,
            prior_decision_sha256=sha256_file(first),
            rationale="goal unreachable within budget",
        ),
    )

    decisions = verify_decision_chain([first, second], max_iterations=1)

    assert [decision.decision for decision in decisions] == [
        "revise_scenario",
        "stop",
    ]


def test_verify_decision_chain_allows_terminal_stop_to_cite_last_sweep(
    tmp_path: Path,
) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json", _decision_payload()
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            decision="stop",
            evidence_sha256="b" * 64,
            next_scenario_path=None,
            prior_decision_sha256=sha256_file(first),
            rationale="The broker refused another sweep after budget exhaustion.",
        ),
    )

    decisions = verify_decision_chain([first, second], max_iterations=1)

    assert decisions[-1].decision == "stop"
    assert decisions[-1].sweep_id == decisions[0].sweep_id
    assert decisions[-1].evidence_sha256 == "b" * 64


def test_verify_decision_chain_accepts_mixed_revision_chain(tmp_path: Path) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json", _decision_payload()
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            decision="revise_patch",
            sweep_id="sweep-002-efgh",
            next_scenario_path=None,
            revised_patch_path="raw/patch_v2.json",
            prior_decision_sha256=sha256_file(first),
            rationale="The authored collider must be rebuilt before retuning.",
        ),
    )
    selected = {
        "sweep_id": "sweep-003-ijkl",
        "trial_index": 2,
        "usd_path": "tuning/iter_3/candidates/trial_0002.usda",
        "usd_sha256": "c" * 64,
    }
    third = _write_decision(
        tmp_path / "physics_tuning_decision_3.json",
        _decision_payload(
            iteration=3,
            decision="accept",
            sweep_id="sweep-003-ijkl",
            next_scenario_path=None,
            evidence_sha256="b" * 64,
            selected=selected,
            prior_decision_sha256=sha256_file(second),
            rationale="The selected candidate matches the behavior goal.",
        ),
    )

    decisions = verify_decision_chain(
        [first, second, third],
        max_iterations=3,
    )

    assert [decision.decision for decision in decisions] == [
        "revise_scenario",
        "revise_patch",
        "accept",
    ]


@pytest.mark.parametrize(
    ("terminal_decision", "terminal_rationale"),
    [
        ("accept", "terminal accept"),
        ("stop", "terminal stop"),
        ("stop", "max_iterations"),
    ],
)
def test_verify_decision_chain_rejects_decision_after_terminal_action(
    tmp_path: Path,
    terminal_decision: str,
    terminal_rationale: str,
) -> None:
    terminal_overrides: dict[str, object] = {
        "decision": terminal_decision,
        "next_scenario_path": None,
        "rationale": terminal_rationale,
    }
    if terminal_decision == "accept":
        terminal_overrides.update(
            evidence_sha256="b" * 64,
            selected={
                "sweep_id": "sweep-001-abcd",
                "trial_index": 0,
                "usd_path": "tuning/iter_1/candidates/trial_0000.usda",
                "usd_sha256": "c" * 64,
            },
        )
    else:
        terminal_overrides.update(sweep_id=None, scenario_sha256=None)
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json",
        _decision_payload(**terminal_overrides),
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            prior_decision_sha256=sha256_file(first),
        ),
    )

    with pytest.raises(ValueError, match=rf"terminal \('{terminal_decision}'\)"):
        verify_decision_chain([first, second])


def test_verify_decision_chain_rejects_configured_iteration_cap_overflow(
    tmp_path: Path,
) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json", _decision_payload()
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            sweep_id="sweep-002-efgh",
            prior_decision_sha256=sha256_file(first),
        ),
    )

    with pytest.raises(ValueError, match="configured max_iterations=1 was exhausted"):
        verify_decision_chain([first, second], max_iterations=1)


def test_verify_decision_chain_rejects_repeated_nonterminal_sweep(
    tmp_path: Path,
) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json", _decision_payload()
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            prior_decision_sha256=sha256_file(first),
        ),
    )

    with pytest.raises(ValueError, match="each nonterminal decision"):
        verify_decision_chain([first, second], max_iterations=1)


@pytest.mark.parametrize("max_iterations", [0, -1, True, 1.5])
def test_verify_decision_chain_rejects_invalid_iteration_cap(
    max_iterations: object,
) -> None:
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        verify_decision_chain([], max_iterations=max_iterations)  # type: ignore[arg-type]


def test_verify_decision_chain_rejects_tampered_prior_file(tmp_path: Path) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json", _decision_payload()
    )
    second = _write_decision(
        tmp_path / "physics_tuning_decision_2.json",
        _decision_payload(
            iteration=2,
            decision="stop",
            sweep_id=None,
            scenario_sha256=None,
            next_scenario_path=None,
            prior_decision_sha256=sha256_file(first),
            rationale="stop",
        ),
    )
    # Retroactive edit to decision 1 breaks the chain.
    tampered = _decision_payload(rationale="rewritten history")
    _write_decision(first, tampered)
    with pytest.raises(ValueError, match="prior_decision_sha256"):
        verify_decision_chain([first, second])


def test_verify_decision_chain_rejects_out_of_order_iterations(
    tmp_path: Path,
) -> None:
    first = _write_decision(
        tmp_path / "physics_tuning_decision_1.json",
        _decision_payload(iteration=2),
    )
    with pytest.raises(ValueError, match="iteration"):
        verify_decision_chain([first])
