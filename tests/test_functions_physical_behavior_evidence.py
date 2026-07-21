# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for physical_behavior evidence shape helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from world_understanding.functions.physics.physical_behavior_evidence import (
    BEHAVIOR_EVIDENCE_MALFORMED,
    BEHAVIOR_EVIDENCE_MISSING,
    BEHAVIOR_EVIDENCE_UNSUPPORTED,
    BEHAVIOR_JUDGE_UNAVAILABLE,
    make_physical_behavior_placeholder_result,
    resolve_physical_behavior_evidence,
)


def _touch(path: Path, content: str = "fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _codes(result: object) -> set[str]:
    if isinstance(result, dict):
        issues = result["issues"]
    else:
        issues = result.issues
    return {issue.code if hasattr(issue, "code") else issue["code"] for issue in issues}


def test_resolves_supported_behavior_evidence_shapes(tmp_path: Path) -> None:
    rollout = _touch(tmp_path / "rollout.usda")
    animation = _touch(tmp_path / "animation.usd")
    video = _touch(tmp_path / "rollout.mp4")
    sampled_frame = _touch(tmp_path / "frame_0001.png")
    simulation = _touch(tmp_path / "sim.json", json.dumps({"success": True}))
    trajectory = _touch(
        tmp_path / "trajectory.json",
        json.dumps({"trajectory": [{"t": 0.0, "x": 0.0}]}),
    )
    history = _touch(
        tmp_path / "history.jsonl",
        json.dumps({"trial_index": 0, "score": 0.1}) + "\n",
    )

    result = resolve_physical_behavior_evidence(
        [
            rollout,
            {
                "path": animation,
                "kind": "animation_usd",
                "role": "animation",
                "description": "artist-authored animation clip",
            },
            video,
            sampled_frame,
            simulation,
            trajectory,
            history,
        ],
        behavior_evidence_required=True,
    )

    assert result.passed
    assert [item.kind for item in result.evidence] == [
        "time_sampled_usd",
        "animation_usd",
        "video",
        "sampled_frame",
        "simulation_json",
        "trajectory_metrics",
        "trajectory_metrics",
    ]
    assert [item.exists for item in result.evidence] == [True] * 7
    assert result.evidence[1].role == "animation"
    assert result.evidence[1].description == "artist-authored animation clip"
    assert result.evidence[6].details["jsonl_line_count"] == 1
    assert len(result.available_evidence) == 7


def test_missing_optional_evidence_warns_without_crashing() -> None:
    result = resolve_physical_behavior_evidence(
        behavior_evidence_required=False,
    )

    assert result.passed
    assert _codes(result) == {BEHAVIOR_EVIDENCE_MISSING}
    assert result.issues[0].severity == "warn"

    placeholder = make_physical_behavior_placeholder_result(result)

    assert placeholder["template"] == "physical_behavior"
    assert placeholder["status"] == "skipped"
    assert placeholder["verdict"] == "warn"
    assert placeholder["passed"] is True
    assert _codes(placeholder) == {BEHAVIOR_EVIDENCE_MISSING}


def test_missing_required_evidence_fails_without_crashing() -> None:
    result = resolve_physical_behavior_evidence(
        behavior_evidence_required=True,
    )

    assert not result.passed
    assert result.issues[0].severity == "fail"

    placeholder = make_physical_behavior_placeholder_result(
        result,
        behavior_evidence_required=True,
    )

    assert placeholder["status"] == "skipped"
    assert placeholder["verdict"] == "fail"
    assert placeholder["passed"] is False


def test_placeholder_required_flag_upgrades_missing_optional_evidence() -> None:
    result = resolve_physical_behavior_evidence(
        behavior_evidence_required=False,
    )

    placeholder = make_physical_behavior_placeholder_result(
        result,
        behavior_evidence_required=True,
    )

    assert placeholder["status"] == "skipped"
    assert placeholder["verdict"] == "fail"
    assert placeholder["passed"] is False
    assert placeholder["issues"][0]["code"] == BEHAVIOR_EVIDENCE_MISSING
    assert placeholder["issues"][0]["severity"] == "fail"
    assert placeholder["metrics"]["behavior_evidence_required"] is True


def test_missing_required_path_fails(tmp_path: Path) -> None:
    result = resolve_physical_behavior_evidence(
        [{"path": "missing.usda", "required": True}],
        base_dir=tmp_path,
    )

    assert not result.passed
    assert result.evidence[0].kind == "time_sampled_usd"
    assert result.evidence[0].exists is False
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MISSING
    assert result.issues[0].severity == "fail"


def test_unsupported_evidence_extension_reports_issue(tmp_path: Path) -> None:
    notes = _touch(tmp_path / "notes.txt")

    result = resolve_physical_behavior_evidence(notes)

    assert not result.passed
    assert result.evidence[0].kind is None
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_UNSUPPORTED


def test_malformed_json_evidence_reports_issue(tmp_path: Path) -> None:
    malformed = _touch(tmp_path / "sim.json", "{not-json")

    result = resolve_physical_behavior_evidence(malformed)

    assert not result.passed
    assert result.evidence[0].kind == "simulation_json"
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED


def test_malformed_jsonl_evidence_reports_issue(tmp_path: Path) -> None:
    malformed = _touch(tmp_path / "history.jsonl", '{"trial_index": 0}\n{bad')

    result = resolve_physical_behavior_evidence(malformed)

    assert not result.passed
    assert result.evidence[0].kind == "trajectory_metrics"
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].details["line_number"] == 2


def test_unreadable_json_evidence_reports_issue(tmp_path: Path) -> None:
    unreadable = tmp_path / "sim.json"
    unreadable.mkdir()

    result = resolve_physical_behavior_evidence(unreadable)

    assert not result.passed
    assert result.evidence[0].kind == "simulation_json"
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].details["error_type"] in {
        "IsADirectoryError",
        "PermissionError",
    }


def test_explicit_json_evidence_read_errors_report_issue(tmp_path: Path) -> None:
    unreadable = tmp_path / "metrics.json"
    unreadable.mkdir()

    result = resolve_physical_behavior_evidence(
        [{"path": unreadable, "kind": "trajectory_metrics"}],
    )

    assert not result.passed
    assert result.evidence[0].kind == "trajectory_metrics"
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].details["error_type"] in {
        "IsADirectoryError",
        "PermissionError",
    }


def test_explicit_json_evidence_validates_content_regardless_of_suffix(
    tmp_path: Path,
) -> None:
    malformed = _touch(tmp_path / "metrics.txt", "not-json")

    result = resolve_physical_behavior_evidence(
        [{"path": malformed, "kind": "trajectory_metrics"}],
    )

    assert not result.passed
    assert result.evidence[0].kind == "trajectory_metrics"
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED


def test_explicit_valid_json_evidence_has_no_validation_issue(tmp_path: Path) -> None:
    metrics = _touch(tmp_path / "metrics.json", json.dumps({"trajectory": []}))

    result = resolve_physical_behavior_evidence(
        [{"path": metrics, "kind": "trajectory_metrics"}],
    )

    assert result.passed
    assert result.evidence[0].kind == "trajectory_metrics"
    assert result.issues == ()


def test_explicit_jsonl_evidence_validates_line_records(tmp_path: Path) -> None:
    history = _touch(
        tmp_path / "history.jsonl",
        json.dumps({"trial_index": 0}) + "\n",
    )

    result = resolve_physical_behavior_evidence(
        [{"path": history, "kind": "trajectory_metrics"}],
    )

    assert result.passed
    assert result.evidence[0].kind == "trajectory_metrics"
    assert result.issues == ()


def test_malformed_mapping_spec_reports_issue() -> None:
    result = resolve_physical_behavior_evidence(
        [{"kind": "video"}],
        behavior_evidence_required=True,
    )

    assert not result.passed
    assert result.evidence == ()
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].severity == "fail"


def test_malformed_required_mapping_spec_fails_even_when_global_optional() -> None:
    result = resolve_physical_behavior_evidence(
        [{"path": "rollout.usda", "kind": "bogus", "required": True}],
    )

    assert not result.passed
    assert result.evidence == ()
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].severity == "fail"
    assert result.issues[0].subject == "rollout.usda"
    assert result.issues[0].details == {"input_index": 0, "required": True}


def test_malformed_non_mapping_spec_uses_default_requiredness() -> None:
    result = resolve_physical_behavior_evidence(
        [123],
        behavior_evidence_required=True,
    )

    assert not result.passed
    assert result.evidence == ()
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].severity == "fail"
    assert result.issues[0].subject is None
    assert result.issues[0].details == {"input_index": 0, "required": True}


def test_malformed_non_bool_required_spec_falls_back_to_global_default() -> None:
    result = resolve_physical_behavior_evidence(
        [{"path": "rollout.usda", "required": "yes"}],
        behavior_evidence_required=False,
    )

    assert result.passed
    assert result.evidence == ()
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MALFORMED
    assert result.issues[0].severity == "warn"
    assert result.issues[0].subject == "rollout.usda"
    assert result.issues[0].details == {"input_index": 0, "required": False}


def test_relative_base_dir_resolves_against_current_working_directory(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    rollout = _touch(tmp_path / "evidence" / "rollout.usda")

    result = resolve_physical_behavior_evidence(
        "rollout.usda",
        base_dir="evidence",
        behavior_evidence_required=True,
    )

    assert result.passed
    assert result.evidence[0].path == str(rollout.resolve())


def test_json_evidence_kind_inference_edge_cases(tmp_path: Path) -> None:
    missing_json = tmp_path / "missing.json"
    metrics_with_positions = _touch(
        tmp_path / "metrics.json",
        json.dumps({"metrics": {"ok": True}, "positions": []}),
    )
    top_level_list = _touch(
        tmp_path / "list.json",
        json.dumps([{"success": True}]),
    )

    result = resolve_physical_behavior_evidence(
        [missing_json, metrics_with_positions, top_level_list],
    )

    assert result.passed
    assert [item.kind for item in result.evidence] == [
        "simulation_json",
        "trajectory_metrics",
        "simulation_json",
    ]
    assert result.evidence[0].details == {"inferred": True}
    assert result.evidence[1].details["json_keys"] == ["metrics", "positions"]
    assert result.evidence[2].details["json_top_level"] == "list"
    assert result.issues[0].code == BEHAVIOR_EVIDENCE_MISSING
    assert result.issues[0].severity == "warn"


def test_missing_jsonl_and_explicit_missing_json_skip_content_validation(
    tmp_path: Path,
) -> None:
    result = resolve_physical_behavior_evidence(
        [
            tmp_path / "missing.jsonl",
            {"path": tmp_path / "explicit-missing.json", "kind": "trajectory_metrics"},
        ],
    )

    assert result.passed
    assert [item.kind for item in result.evidence] == [
        "trajectory_metrics",
        "trajectory_metrics",
    ]
    assert [issue.code for issue in result.issues] == [
        BEHAVIOR_EVIDENCE_MISSING,
        BEHAVIOR_EVIDENCE_MISSING,
    ]


def test_jsonl_evidence_ignores_blank_lines_when_counting_records(
    tmp_path: Path,
) -> None:
    history = _touch(
        tmp_path / "history.jsonl",
        "\n" + json.dumps({"trial_index": 0}) + "\n\n",
    )

    result = resolve_physical_behavior_evidence(history)

    assert result.passed
    assert result.evidence[0].kind == "trajectory_metrics"
    assert result.evidence[0].details["jsonl_line_count"] == 1


def test_placeholder_result_reports_unavailable_judge_for_valid_evidence(
    tmp_path: Path,
) -> None:
    video = _touch(tmp_path / "rollout.mp4")
    resolution = resolve_physical_behavior_evidence(video)

    placeholder = make_physical_behavior_placeholder_result(
        resolution,
        task_description="Validate that the cart rolls forward without tipping.",
    )

    assert placeholder["status"] == "unavailable"
    assert placeholder["verdict"] == "warn"
    assert placeholder["passed"] is True
    assert placeholder["metrics"] == {
        "behavior_evidence_required": False,
        "evidence_count": 1,
        "available_evidence_count": 1,
        "evidence_kinds": ["video"],
    }
    assert placeholder["task_description"] == (
        "Validate that the cart rolls forward without tipping."
    )
    assert _codes(placeholder) == {BEHAVIOR_JUDGE_UNAVAILABLE}


def test_required_placeholder_unavailable_judge_can_fail(tmp_path: Path) -> None:
    video = _touch(tmp_path / "rollout.mp4")
    resolution = resolve_physical_behavior_evidence(
        video,
        behavior_evidence_required=True,
    )

    placeholder = make_physical_behavior_placeholder_result(
        resolution,
    )

    assert placeholder["status"] == "unavailable"
    assert placeholder["verdict"] == "fail"
    assert placeholder["passed"] is False
    assert placeholder["metrics"]["behavior_evidence_required"] is True
    assert _codes(placeholder) == {BEHAVIOR_JUDGE_UNAVAILABLE}


def test_required_placeholder_respects_resolver_level_requiredness(
    tmp_path: Path,
) -> None:
    video = _touch(tmp_path / "rollout.mp4")
    resolution = resolve_physical_behavior_evidence(
        [{"path": video, "required": False}],
        behavior_evidence_required=True,
    )

    placeholder = make_physical_behavior_placeholder_result(resolution)

    assert resolution.evidence[0].required is False
    assert resolution.behavior_evidence_required is True
    assert placeholder["status"] == "unavailable"
    assert placeholder["verdict"] == "fail"
    assert placeholder["passed"] is False
    assert placeholder["metrics"]["behavior_evidence_required"] is True
    assert _codes(placeholder) == {BEHAVIOR_JUDGE_UNAVAILABLE}


def test_resolution_to_dict_is_json_friendly(tmp_path: Path) -> None:
    video = _touch(tmp_path / "rollout.mp4")
    result = resolve_physical_behavior_evidence(video)
    data = result.to_dict()

    json.dumps(data)

    assert data["passed"] is True
    assert data["behavior_evidence_required"] is False
    assert data["supported_kinds"] == [
        "time_sampled_usd",
        "animation_usd",
        "video",
        "sampled_frame",
        "simulation_json",
        "trajectory_metrics",
    ]
