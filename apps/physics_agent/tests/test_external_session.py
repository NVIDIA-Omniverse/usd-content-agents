# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for wrapper-facing external refinement session helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from physics_agent.tuning.external import (
    ExternalRuntime,
    ExternalTuneSpec,
    QualificationSettings,
    QualifiedParameter,
    derive_iteration_spec,
    validate_qualification_approval,
)
from physics_agent.tuning.external.types import EvidenceSettings
from physics_agent.tuning.types import (
    OptimizerSettings,
    TunableParam,
    TuningObjective,
)


@pytest.fixture()
def spec(tmp_path: Path) -> ExternalTuneSpec:
    script = tmp_path / "adapter.py"
    script.write_text("print('adapter')\n", encoding="utf-8")
    runtime = ExternalRuntime(
        python=Path(sys.executable),
        script=script,
        cwd=tmp_path,
        fingerprint_paths=(script,),
    )
    return ExternalTuneSpec(
        task="gear_assembly",
        runtime=runtime,
        parameter_catalog=(
            QualifiedParameter(name="restitution"),
            QualifiedParameter(name="solver_iterations", integer=True),
        ),
        params=(TunableParam(name="restitution", min_value=0.0, max_value=1.0),),
        objective=TuningObjective(name="grasp_slip", unit="m"),
        optimizer=OptimizerSettings(name="random", max_trials=4, seed=7),
        qualification=QualificationSettings(
            nominal_params={"restitution": 0.5, "solver_iterations": 8.0}
        ),
        evidence=EvidenceSettings(),
    )


def test_derive_iteration_spec_builds_active_search(spec: ExternalTuneSpec) -> None:
    derived = derive_iteration_spec(
        spec,
        active_search={"restitution": {"min": 0.2, "max": 0.8}},
        iteration=3,
    )
    assert [param.name for param in derived.params] == ["restitution"]
    assert derived.params[0].min_value == 0.2
    assert derived.params[0].max_value == 0.8
    # Seed schedule matches the built-in refine loop: seed + iteration - 1.
    assert derived.optimizer.seed == 9
    assert derived.optimizer.replica_seed == 7
    # The base spec is not mutated.
    assert spec.optimizer.seed == 7


def test_derive_iteration_spec_copies_integer_typing(
    spec: ExternalTuneSpec,
) -> None:
    derived = derive_iteration_spec(
        spec,
        active_search={"solver_iterations": {"min": 4, "max": 16}},
        iteration=1,
    )
    assert derived.params[0].integer is True


def test_derive_iteration_spec_rejects_unqualified_parameter(
    spec: ExternalTuneSpec,
) -> None:
    with pytest.raises(ValueError, match="not in the qualified"):
        derive_iteration_spec(
            spec,
            active_search={"unknown": {"min": 0.0, "max": 1.0}},
            iteration=1,
        )


def test_derive_iteration_spec_rejects_malformed_bounds(
    spec: ExternalTuneSpec,
) -> None:
    with pytest.raises(ValueError, match="'min' and 'max'"):
        derive_iteration_spec(
            spec, active_search={"restitution": {"min": 0.0}}, iteration=1
        )
    with pytest.raises(ValueError, match="min_value > max_value"):
        derive_iteration_spec(
            spec,
            active_search={"restitution": {"min": 0.9, "max": 0.1}},
            iteration=1,
        )
    with pytest.raises(ValueError, match="at least 1"):
        derive_iteration_spec(
            spec,
            active_search={"restitution": {"min": 0.0, "max": 1.0}},
            iteration=0,
        )
    with pytest.raises(ValueError, match="at least one parameter"):
        derive_iteration_spec(spec, active_search={}, iteration=1)


def test_validate_qualification_approval_fails_without_artifacts(
    spec: ExternalTuneSpec, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        validate_qualification_approval(
            spec,
            qualification_dir=tmp_path / "qualification",
            approval_digest="sha256:" + "0" * 64,
        )


def test_validate_qualification_approval_returns_path_on_success(
    spec: ExternalTuneSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a valid approval the helper returns the qualification.json path so
    a wrapper can record the reviewed evidence it is proceeding from."""

    from physics_agent.tuning.external import session

    qualification_path = tmp_path / "qualification" / "qualification.json"

    def fake_validate(**kwargs: object) -> tuple[object, Path, object, object]:
        return {}, qualification_path, None, None

    monkeypatch.setattr(session, "_validate_approved_qualification", fake_validate)
    result = validate_qualification_approval(
        spec,
        qualification_dir=tmp_path / "qualification",
        approval_digest="sha256:" + "0" * 64,
    )
    assert result == qualification_path
