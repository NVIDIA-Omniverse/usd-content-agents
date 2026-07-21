# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for refine worker session metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from physics_agent.api.refine import IterationSummary

from ...service.workers.refine_executor import _refine_results_metadata


def test_refine_results_metadata_persists_final_best_params(tmp_path: Path) -> None:
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    best_params = {"mass_scale": 1.2, "restitution": 0.9}
    (final_dir / "best_params.json").write_text(
        json.dumps({"params": best_params, "best_score": 0.12}),
        encoding="utf-8",
    )

    result = SimpleNamespace(
        termination_reason="approved",
        iteration_count=1,
        final_iteration=1,
        final_judge_score=0.96,
        iterations=[
            IterationSummary(
                iteration=1,
                iteration_dir=tmp_path / "iter_1",
                judge_decision="approve",
                judge_score=0.96,
                judge_reasoning="stubbed",
                best_score=0.12,
                n_trials=3,
                metric_name="settle_distance",
                metric_value=0.12,
            )
        ],
        final_dir=final_dir,
        output_dir=tmp_path,
    )

    metadata = _refine_results_metadata(result)

    assert metadata["final_best_params"] == best_params
    assert metadata["iterations"][0]["best_params"] == best_params
