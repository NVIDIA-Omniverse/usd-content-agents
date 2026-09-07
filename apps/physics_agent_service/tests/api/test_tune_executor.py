# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tune worker session metadata persistence."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest


@pytest.mark.api
async def test_failed_tune_with_artifacts_persists_partial_results(
    session_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge-only fail-closed tunes stay failed but keep result discovery."""
    from physics_agent.tuning import TuneOutput

    from ...service.workers import tune_executor

    async def fake_arun_tune(params):
        out = params.output_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_params.json").write_text(
            json.dumps({"params": {"mass_scale": 1.2}, "best_score": 0.25}),
            encoding="utf-8",
        )
        (out / "history.jsonl").write_text("{}\n", encoding="utf-8")
        (out / "tune_results.json").write_text("{}", encoding="utf-8")
        (out / "report.md").write_text("# failed visual judge\n", encoding="utf-8")
        return TuneOutput(
            success=False,
            error=(
                "Visual judge evidence preparation failed with reference media; "
                "refusing to fall back to programmatic-only verdict."
            ),
            output_dir=out,
            best_params={"mass_scale": 1.2},
            best_score=0.25,
            best_objective=0.4,
            n_trials=3,
            optimizer_used="random",
            engine_used="fake",
            artifacts={"best_params": out / "best_params.json"},
        )

    monkeypatch.setattr(tune_executor, "arun_tune", fake_arun_tune)

    session_id = str(uuid4())
    session_dir = await session_manager.create_session(session_id)
    physics_usd = session_dir / "input" / "physics.usda"
    physics_usd.write_text("#usda 1.0\n", encoding="utf-8")

    await tune_executor.execute_tune_async(
        session_id=session_id,
        session_manager=session_manager,
        scenario_path=None,
        physics_usd=physics_usd,
        engine="fake",
        optimizer="random",
        max_trials=3,
        seed=7,
        reference_images=[session_dir / "input" / "reference.png"],
    )

    metadata = await session_manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "failed"
    assert metadata["can_cancel"] is False
    assert metadata["results"] == {
        "best_params": {"mass_scale": 1.2},
        "best_score": 0.25,
        "best_objective": 0.4,
        "n_trials": 3,
        "optimizer_used": "random",
        "engine_used": "fake",
    }
    assert metadata["partial_results"] == metadata["results"]
