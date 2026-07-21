# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Joint Agent service executor stats extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace

from apps.joint_agent_service.service.workers.executor import _extract_stats_from_result


def test_extract_stats_preserves_explicit_zero_articulation_candidates(tmp_path):
    candidates_dir = tmp_path / "cache" / "predictions"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "articulation_candidates.json").write_text(
        json.dumps({"summary": {"candidate_count": 3}}),
        encoding="utf-8",
    )
    result = SimpleNamespace(
        step_results={
            "predict": {"predictions_count": 2},
            "infer_articulation_candidates": {"articulation_candidate_count": 0},
        },
        raw_result={"build_dataset_usd_result": {"num_prims": 2, "num_images": 2}},
    )

    stats = _extract_stats_from_result(result, session_dir=tmp_path)

    assert stats["articulation_candidates"] == 0


def test_extract_stats_falls_back_to_candidate_file_when_step_count_missing(tmp_path):
    candidates_dir = tmp_path / "cache" / "predictions"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "articulation_candidates.json").write_text(
        json.dumps({"summary": {"candidate_count": 3}}),
        encoding="utf-8",
    )
    result = SimpleNamespace(
        step_results={"predict": {"predictions_count": 2}},
        raw_result={"build_dataset_usd_result": {"num_prims": 2, "num_images": 2}},
    )

    stats = _extract_stats_from_result(result, session_dir=tmp_path)

    assert stats["articulation_candidates"] == 3


def test_extract_stats_includes_joint_rigger_summary(tmp_path):
    result = SimpleNamespace(
        step_results={
            "apply_joint_rigger": {
                "joint_rigger_status": "authored",
                "authored_joint_count": 4,
            },
        },
        raw_result={},
    )

    stats = _extract_stats_from_result(result, session_dir=tmp_path)

    assert stats["joint_rigger_status"] == "authored"
    assert stats["joint_rigger_authored_joints"] == 4
