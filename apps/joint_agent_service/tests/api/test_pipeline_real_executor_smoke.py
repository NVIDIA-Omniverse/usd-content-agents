# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the real Joint Agent Service executor path."""

import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from joint_agent.api.pipeline import PipelineInput, PipelineOutput

from ..conftest import make_pipeline_files


async def _emit_step(listener: Any, step_name: str, *, progress_message: str) -> None:
    listener.event(
        "step.started",
        {"step_name": step_name, "message": f"Starting {step_name}"},
    )
    await asyncio.sleep(0.01)
    progress_event_type = (
        "prediction.completed" if step_name == "predict" else "step.progress"
    )
    listener.event(
        progress_event_type,
        {
            "step_name": step_name,
            "current": 1,
            "total": 1,
            "percent": 100,
            "message": progress_message,
        },
    )
    await asyncio.sleep(0.01)
    listener.event(
        "step.completed",
        {"step_name": step_name, "message": f"Completed {step_name}"},
    )
    await asyncio.sleep(0.01)


@pytest.fixture
def _reset_event_bus() -> Generator[None, None, None]:
    from ...service.runtime import bus as bus_module

    bus_module._event_bus = None
    yield
    bus_module._event_bus = None


@pytest.mark.api
@pytest.mark.real_executor
async def test_pipeline_uses_real_executor_with_mocked_joint_agent(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    _reset_event_bus: None,
) -> None:
    """Exercise the real worker path while stubbing only JA's pipeline API."""
    from ...service.routers.pipeline_router import get_session_manager
    from ...service.workers import executor as executor_module

    async def fake_arun_pipeline(params: PipelineInput) -> PipelineOutput:
        session_dir = Path(params.event_listener.session_dir)
        dataset_dir = session_dir / "cache" / "dataset"
        predictions_dir = session_dir / "cache" / "predictions"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

        (dataset_dir / "dataset.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {"id": "/Root/Hinge", "images": {"prim_only": "hinge.png"}}
                    ),
                    json.dumps(
                        {"id": "/Root/Bracket", "images": {"prim_only": "bracket.png"}}
                    ),
                ]
            )
            + "\n"
        )
        (predictions_dir / "predictions.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "/Root/Hinge", "joint_type": "revolute"}),
                    json.dumps({"id": "/Root/Bracket", "joint_type": "fixed"}),
                ]
            )
            + "\n"
        )
        (predictions_dir / "articulation_candidates.json").write_text(
            json.dumps(
                {
                    "schema_version": "joint-agent-stage2-v0",
                    "summary": {"candidate_count": 1},
                    "candidates": [
                        {
                            "candidate_id": "candidate_0001",
                            "joint_type_hint": "revolute",
                        }
                    ],
                }
            )
        )
        (predictions_dir / "articulation_candidates.html").write_text(
            "<html><body>Articulation Candidates</body></html>"
        )
        joint_rigger_dir = session_dir / "cache" / "joint_rigger"
        joint_rigger_dir.mkdir(parents=True, exist_ok=True)
        (joint_rigger_dir / "rigged.usdz").write_bytes(b"PK\x03\x04owned-core")
        (joint_rigger_dir / "joint_rigger_diagnostics.json").write_text("{}")
        (joint_rigger_dir / "joint_rigger_validation.json").write_text("{}")

        await _emit_step(
            params.event_listener,
            "build_dataset_usd",
            progress_message="Rendered 2 parts",
        )
        await _emit_step(
            params.event_listener,
            "build_dataset_prepare_dataset",
            progress_message="Prepared 2 entries",
        )
        await _emit_step(
            params.event_listener,
            "predict",
            progress_message="Predicted 2 joints",
        )
        await _emit_step(
            params.event_listener,
            "consistency_pass",
            progress_message="Checked prediction consistency",
        )
        await _emit_step(
            params.event_listener,
            "infer_articulation_candidates",
            progress_message="Inferred 1 articulation candidate",
        )
        await _emit_step(
            params.event_listener,
            "restore_usd",
            progress_message="Restored prediction paths",
        )
        await _emit_step(
            params.event_listener,
            "apply_joint_rigger",
            progress_message="Authored 1 joint",
        )
        params.event_listener.event(
            "workflow.completed",
            {
                "workflow_type": "pipeline",
                "completed_steps": [
                    "build_dataset_usd",
                    "build_dataset_prepare_dataset",
                    "predict",
                    "consistency_pass",
                    "infer_articulation_candidates",
                    "restore_usd",
                    "apply_joint_rigger",
                ],
            },
        )
        await asyncio.sleep(0.01)

        return PipelineOutput(
            success=True,
            step_results={
                "build_dataset_prepare_dataset": {"num_entries": 2},
                "predict": {"predictions_count": 2},
                "consistency_pass": {
                    "predictions_count": 2,
                    "consistency_stats": {"groups_repeated": 0},
                },
                "infer_articulation_candidates": {
                    "articulation_candidate_count": 1,
                    "articulation_summary": {"candidate_count": 1},
                },
                "restore_usd": {"restored_predictions": 2},
                "apply_joint_rigger": {
                    "joint_rigger_status": "authored",
                    "authored_joint_count": 1,
                    "apply_joint_rigger_skipped": False,
                },
            },
            completed_steps=[
                "build_dataset_usd",
                "build_dataset_prepare_dataset",
                "predict",
                "consistency_pass",
                "infer_articulation_candidates",
                "restore_usd",
                "apply_joint_rigger",
            ],
            raw_result={
                "build_dataset_usd_result": {"num_prims": 2, "num_images": 2},
            },
        )

    monkeypatch.setattr(executor_module, "arun_pipeline", fake_arun_pipeline)

    response = await client.post(
        "/pipeline",
        files=make_pipeline_files(),
        data={"apply_joint_rigger": "true"},
    )
    assert response.status_code == 202
    session_id = response.json()["session_id"]

    seen_statuses: list[str] = []
    final_status = None
    for _ in range(200):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 200
        body = status_r.json()
        seen_statuses.append(body["status"])
        if body["status"] == "completed":
            final_status = body
            break
        await asyncio.sleep(0.01)

    assert final_status is not None
    assert "running" in seen_statuses
    assert final_status["overall_progress"]["percent"] == 100
    assert [step["name"] for step in final_status["completed_steps"]] == [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "consistency_pass",
        "infer_articulation_candidates",
        "restore_usd",
        "apply_joint_rigger",
    ]

    results = None
    for _ in range(100):
        results_r = await client.get(f"/pipeline/{session_id}/results")
        if results_r.status_code == 200:
            results = results_r.json()
            break
        assert results_r.status_code == 202
        await asyncio.sleep(0.01)

    assert results is not None
    assert results["status"] == "completed"
    assert results["stats"]["prims_processed"] == 2
    assert results["stats"]["images_generated"] == 2
    assert results["stats"]["predictions_made"] == 2
    assert results["stats"]["joint_rigger_status"] == "authored"
    assert results["stats"]["joint_rigger_artifacts"] == {
        "joint_rigger_output": True,
        "joint_rigger_diagnostics": True,
        "joint_rigger_validation": True,
    }
    assert "joint_rigger_output" in results["download_urls"]

    store_metadata = await get_session_manager().get_session_metadata(session_id)
    assert store_metadata is not None
    assert store_metadata["overall_progress"]["percent"] == 100
    assert store_metadata["overall_progress"]["current_step"] == 9
    assert store_metadata["overall_progress"]["total_steps"] == 9
    assert [step["name"] for step in store_metadata["completed_steps"]] == [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "consistency_pass",
        "infer_articulation_candidates",
        "restore_usd",
        "apply_joint_rigger",
    ]

    predictions_r = await client.get(f"/artifacts/{session_id}/predictions")
    assert predictions_r.status_code == 200
    dataset_r = await client.get(f"/artifacts/{session_id}/dataset")
    assert dataset_r.status_code == 200
    output_r = await client.get(results["download_urls"]["joint_rigger_output"])
    assert output_r.status_code == 200
    assert output_r.content == b"PK\x03\x04owned-core"
    assert 'filename="rigged.usdz"' in output_r.headers["content-disposition"]
