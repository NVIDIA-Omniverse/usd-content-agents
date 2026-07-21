# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scenario YAML parsing + validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_agent.tuning.scenario import (
    ScenarioParseError,
    load_scenario,
    parse_scenario,
)
from physics_agent.tuning.types import (
    DEFAULT_PARAM_BOUNDS,
    SUPPORTED_PARAM_KEYS,
)


def _good_scenario() -> dict:
    return {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5},
        "parameters": [
            {"name": "mass_scale", "min": 0.5, "max": 2.0},
            {"name": "static_friction", "min": 0.1, "max": 1.0},
        ],
    }


def test_parse_scenario_happy_path() -> None:
    sc = parse_scenario(_good_scenario())
    assert sc.name == "drop_settle"
    assert sc.metric == "settle_distance"
    assert len(sc.params) == 2
    assert sc.params[0].name == "mass_scale"


def test_parse_scenario_defaults_metric_when_missing() -> None:
    raw = _good_scenario()
    raw.pop("metric")
    sc = parse_scenario(raw)
    assert sc.metric == "settle_distance"  # default


def test_parse_scenario_uses_default_param_bounds_when_omitted() -> None:
    raw = {
        "name": "drop_settle",
        "parameters": [{"name": "mass_scale"}],
    }
    sc = parse_scenario(raw)
    lo, hi = DEFAULT_PARAM_BOUNDS["mass_scale"]
    assert sc.params[0].min_value == lo
    assert sc.params[0].max_value == hi


def test_parse_scenario_rejects_unknown_scenario() -> None:
    raw = _good_scenario()
    raw["name"] = "rolling_ball"
    with pytest.raises(ScenarioParseError, match="Unsupported scenario"):
        parse_scenario(raw)


def test_parse_scenario_rejects_unsupported_drop_settle_metric() -> None:
    """LLM hallucinations / YAML typos like ``metric: max_velocity`` must
    surface as a clean parse error before the runner enters the per-trial
    backend loop. Otherwise the optimizer would burn the trial budget on
    every-trial-fails and the user would only see "all trials failed"."""
    raw = _good_scenario()
    raw["name"] = "drop_settle"
    raw["metric"] = "max_velocity"
    with pytest.raises(
        ScenarioParseError,
        match=r"Unsupported drop_settle metric 'max_velocity'",
    ):
        parse_scenario(raw)


def test_parse_scenario_accepts_all_registered_drop_settle_metrics() -> None:
    """Every key in ``_METRICS`` must round-trip through parse_scenario."""
    from physics_agent.tuning.scenarios.drop_settle import _METRICS

    for metric_name in _METRICS:
        raw = _good_scenario()
        raw["name"] = "drop_settle"
        raw["metric"] = metric_name
        sc = parse_scenario(raw)
        assert sc.metric == metric_name


def test_parse_scenario_freeform_defaults_metric_to_judge_score() -> None:
    """Round 13 (CodeRabbit thread #2): freeform with no explicit metric
    must default to ``judge_score`` — not ``settle_distance``, which is
    the drop_settle default and would silently mislabel artifacts."""
    raw = {
        "name": "freeform",
        "target": {"duration_s": 2.0},
        "parameters": [{"name": "restitution", "min": 0.4, "max": 0.95}],
    }
    sc = parse_scenario(raw)
    assert sc.metric == "judge_score"


def test_parse_scenario_freeform_rejects_non_judge_score_metric() -> None:
    """Freeform.evaluate ignores ``scenario.metric`` for scoring, but
    artifacts and audit records key off it. Reject any other value at
    parse time so a typo / LLM hallucination is caught before the run."""
    raw = {
        "name": "freeform",
        "metric": "settle_distance",  # belongs to drop_settle
        "target": {"duration_s": 2.0},
        "parameters": [{"name": "restitution", "min": 0.4, "max": 0.95}],
    }
    with pytest.raises(
        ScenarioParseError,
        match=r"Unsupported freeform metric 'settle_distance'",
    ):
        parse_scenario(raw)


def test_parse_scenario_freeform_accepts_explicit_judge_score() -> None:
    raw = {
        "name": "freeform",
        "metric": "judge_score",
        "target": {"duration_s": 2.0},
        "parameters": [{"name": "restitution", "min": 0.4, "max": 0.95}],
    }
    sc = parse_scenario(raw)
    assert sc.name == "freeform"
    assert sc.metric == "judge_score"


def test_parse_scenario_rejects_unknown_param() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"name": "viscosity", "min": 0.1, "max": 1.0}]
    with pytest.raises(ScenarioParseError, match="not a supported tunable"):
        parse_scenario(raw)


def test_parse_scenario_rejects_missing_param_name() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"min": 0.0, "max": 1.0}]
    with pytest.raises(ScenarioParseError, match="missing required key 'name'"):
        parse_scenario(raw)


def test_parse_scenario_rejects_min_greater_than_max() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"name": "mass_scale", "min": 5.0, "max": 1.0}]
    with pytest.raises(ScenarioParseError, match="min > max"):
        parse_scenario(raw)


def test_parse_scenario_rejects_infeasible_friction_ranges() -> None:
    raw = _good_scenario()
    raw["parameters"] = [
        {"name": "static_friction", "min": 0.1, "max": 0.5},
        {"name": "dynamic_friction", "min": 0.6, "max": 1.0},
    ]

    with pytest.raises(ScenarioParseError, match="dynamic_friction minimum"):
        parse_scenario(raw)


def test_parse_scenario_rejects_non_numeric_bound() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"name": "mass_scale", "min": "low"}]
    with pytest.raises(ScenarioParseError, match="must be a number"):
        parse_scenario(raw)


def test_parse_scenario_rejects_bool_bound() -> None:
    raw = _good_scenario()
    # `True` is technically int in Python — explicit reject avoids surprise.
    raw["parameters"] = [{"name": "mass_scale", "min": True, "max": 2.0}]
    with pytest.raises(ScenarioParseError, match="bool"):
        parse_scenario(raw)


def test_parse_scenario_rejects_empty_parameters() -> None:
    raw = _good_scenario()
    raw["parameters"] = []
    with pytest.raises(ScenarioParseError, match="at least one"):
        parse_scenario(raw)


def test_parse_scenario_rejects_missing_parameters() -> None:
    raw = _good_scenario()
    raw.pop("parameters")
    with pytest.raises(ScenarioParseError, match="must define 'parameters'"):
        parse_scenario(raw)


def test_parse_scenario_rejects_non_mapping_root() -> None:
    with pytest.raises(ScenarioParseError, match="must be a mapping"):
        parse_scenario([1, 2, 3])  # type: ignore[arg-type]


def test_parse_scenario_rejects_non_string_param_name() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"name": 42}]
    with pytest.raises(ScenarioParseError, match="must be a string"):
        parse_scenario(raw)


def test_parse_scenario_rejects_duplicate_param_names() -> None:
    raw = _good_scenario()
    raw["parameters"] = [
        {"name": "mass_scale", "min": 0.5, "max": 2.0},
        {"name": "mass_scale", "min": 0.6, "max": 1.5},
    ]
    with pytest.raises(ScenarioParseError, match="duplicates"):
        parse_scenario(raw)


def test_load_scenario_from_path(tmp_path: Path) -> None:
    p = tmp_path / "drop_settle.yaml"
    p.write_text(
        "name: drop_settle\n"
        "parameters:\n"
        "  - name: mass_scale\n"
        "    min: 0.5\n"
        "    max: 2.0\n"
    )
    sc = load_scenario(p)
    assert sc.name == "drop_settle"


def test_load_scenario_rejects_missing_file(tmp_path: Path) -> None:
    # CodeRabbit (R14): a hard-coded absolute path can flake if it
    # happens to exist on a runner image; use a guaranteed-missing
    # path under ``tmp_path`` instead.
    missing = tmp_path / "does_not_exist" / "scenario.yaml"
    with pytest.raises(FileNotFoundError):
        load_scenario(missing)


def test_load_scenario_rejects_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(ScenarioParseError, match="empty"):
        load_scenario(p)


def test_load_scenario_rejects_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: drop_settle\nparameters: : :\n  - bad\n")
    with pytest.raises(ScenarioParseError, match="Invalid YAML"):
        load_scenario(p)


def test_parse_scenario_rejects_nan_bound() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"name": "mass_scale", "min": float("nan"), "max": 2.0}]
    with pytest.raises(ScenarioParseError, match="finite"):
        parse_scenario(raw)


def test_parse_scenario_rejects_infinity_bound() -> None:
    raw = _good_scenario()
    raw["parameters"] = [{"name": "mass_scale", "min": 0.5, "max": float("inf")}]
    with pytest.raises(ScenarioParseError, match="finite"):
        parse_scenario(raw)


def test_parse_scenario_validates_drop_settle_target_numerics() -> None:
    raw = _good_scenario()
    raw["target"] = {"drop_height_m": [1, 2]}
    with pytest.raises(ScenarioParseError, match="target.drop_height_m"):
        parse_scenario(raw)


def test_supported_param_keys_match_default_bounds_keys() -> None:
    # Sanity guard: the SUPPORTED_PARAM_KEYS tuple and DEFAULT_PARAM_BOUNDS
    # must stay in lockstep — adding a new param without bounds would crash
    # the parser at default-fill time.
    assert set(SUPPORTED_PARAM_KEYS) == set(DEFAULT_PARAM_BOUNDS.keys())
