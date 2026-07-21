# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline regeneration.

Tests the regenerate endpoint for re-running specific steps.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
    b"\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
_INVALID_DURABLE_REQUEST_DETAIL = "Request content cannot contain inline credentials"


def _session_file_snapshot(session_dir: Path) -> dict[str, bytes]:
    """Return an exact snapshot of regular files beneath one session."""
    return {
        path.relative_to(session_dir).as_posix(): path.read_bytes()
        for path in session_dir.rglob("*")
        if path.is_file()
    }


async def _finish_stub_execution(
    session_manager: Any,
    session_id: str,
    regeneration_claim: Any | None,
    updates: dict[str, Any],
) -> None:
    """Finish a capture stub without bypassing regeneration fencing."""
    updates = dict(updates)
    if updates.get("status") == "completed":
        updates.setdefault("results", {})
        updates.setdefault("coverage", None)
        updates.setdefault("completed_at", "2026-01-01T00:00:00+00:00")
    if regeneration_claim is None:
        await session_manager.update_session(session_id, updates)
        return
    finalized = await session_manager.finalize_regeneration_claim(
        session_id,
        regeneration_claim,
        updates=updates,
    )
    assert finalized


async def _wait_for_completed(client: Any, session_id: str) -> dict[str, Any]:
    """Wait for asynchronous session finalization before regeneration."""
    status: dict[str, Any] = {}
    for _ in range(200):
        response = await client.get(f"/pipeline/{session_id}/status")
        assert response.status_code == 200
        status = response.json()
        if status["status"] == "completed":
            return status
        await asyncio.sleep(0.005)
    pytest.fail(f"session {session_id} did not complete: {status}")


@pytest.mark.api
class TestPipelineRegenerate:
    """Test pipeline regeneration."""

    async def test_regenerate_apply_only(self, client):
        """Test regenerating the apply step only."""
        # Create and complete a pipeline first
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        await asyncio.sleep(1)

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Now regenerate with apply only
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )

        assert regen_r.status_code == 202
        assert regen_r.json()["status"] == "pending"

    async def test_regenerate_multiple_steps(self, client):
        """Test regenerating multiple steps."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Regenerate predict and apply
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict", "apply"]},
        )

        assert regen_r.status_code == 202

    async def test_predict_regeneration_uses_only_current_run_evidence(
        self,
        client,
        monkeypatch,
    ):
        """Re-running predict must not inherit prior restore/apply evidence."""
        import material_agent.api as material_api

        from ...service.routers import pipeline_router

        manager = pipeline_router.get_session_manager()
        session_id = str(uuid4())
        session_dir = await manager.create_session(
            session_id,
            config={"coverage_policy": "strict"},
        )
        input_path = session_dir / "input" / "scene.usda"
        input_path.write_text("#usda 1.0\n", encoding="utf-8")
        raw_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps({"id": "/Current", "material": "CurrentRaw"}) + "\n",
            encoding="utf-8",
        )
        dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text('{"id": "/Current"}\n', encoding="utf-8")
        usd_dataset_dir = session_dir / "cache" / "dataset" / "usd"
        usd_dataset_dir.mkdir(parents=True, exist_ok=True)
        (usd_dataset_dir / "prims.jsonl").write_text(
            '{"id": "/Current"}\n',
            encoding="utf-8",
        )
        restored_path = (
            session_dir / "cache" / "restored" / "restored_predictions.jsonl"
        )
        restored_path.parent.mkdir(parents=True, exist_ok=True)
        restored_path.write_text(
            json.dumps({"id": "/Stale", "material": "StaleRestored"}) + "\n",
            encoding="utf-8",
        )
        state_path = session_dir / "cache" / ".pipeline_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "completed_steps": [
                        "build_dataset_usd",
                        "build_dataset_prepare_dataset",
                        "predict",
                        "restore_usd",
                        "apply",
                    ],
                    "failed_steps": [],
                    "step_errors": {},
                    "step_outputs": {
                        "build_dataset_usd": {
                            "output_dir": str(usd_dataset_dir),
                            "num_prims": 1,
                        },
                        "build_dataset_prepare_dataset": {
                            "dataset_jsonl_path": str(dataset_path),
                        },
                        "predict": {"predictions_path": "old-raw.jsonl"},
                        "restore_usd": {
                            "restored_predictions_path": str(restored_path),
                            "restore_stats": {
                                "restored_prim_sources": {"/Stale": "/Optimized/Stale"}
                            },
                        },
                        "apply": {
                            "assignment_stats": {
                                "bound_prim_ids": ["/Stale"],
                                "unbound_prim_ids": [],
                            }
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "coverage": {"readiness_grade": "complete"},
                "partial_results": {"old": True},
                "failed_step": "old_step",
                "error": "old failure",
                "results": {"materials_applied": 1},
                "completed_at": "2026-01-01T00:00:00+00:00",
                "failed_at": "2026-01-01T00:00:00+00:00",
                "cancelled_at": "2026-01-01T00:00:00+00:00",
                "duration_seconds": 9,
                "step_timings": {"apply": 1.0},
                "timings": {"apply": 1.0},
                "completed_steps": [
                    {"name": "restore_usd"},
                    {"name": "apply"},
                ],
            },
        )
        event_bus = pipeline_router.get_event_bus()
        await event_bus.get_queue(session_id).put(object())

        captured: dict[str, Any] = {}
        finished = asyncio.Event()

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            coverage_policy: str = "allow_partial",
            regeneration_claim: Any | None = None,
        ) -> None:
            captured["metadata"] = await session_manager.get_session_metadata(
                session_id
            )
            captured["state"] = json.loads(state_path.read_text(encoding="utf-8"))
            captured["snapshot"] = pipeline_router.get_event_bus().get_snapshot(
                session_id
            )
            captured["queue_size"] = (
                pipeline_router.get_event_bus().get_queue(session_id).qsize()
            )
            completed_step = "predict" if "predict" in config_dict["steps"] else "apply"
            current_state = json.loads(state_path.read_text(encoding="utf-8"))
            current_state.setdefault("completed_steps", []).append(completed_step)
            current_state.setdefault("step_outputs", {})[completed_step] = (
                {"predictions_path": str(raw_path)}
                if completed_step == "predict"
                else {
                    "output_usd_path": str(
                        session_dir / "output" / "scene_with_materials.usd"
                    )
                }
            )
            state_path.write_text(json.dumps(current_state), encoding="utf-8")
            current_metadata = await session_manager.get_session_metadata(session_id)
            artifact_validity = dict(current_metadata["artifact_validity"])
            artifact_validity[
                "raw_predictions"
                if completed_step == "predict"
                else "applied_output_usd"
            ] = True
            await _finish_stub_execution(
                session_manager,
                session_id,
                regeneration_claim,
                {
                    "status": "completed",
                    "completed_steps": [{"name": completed_step}],
                    "results": {"predictions_made": 1},
                    "artifact_validity": artifact_validity,
                },
            )
            finished.set()

        def minimal_config(**kwargs):
            enabled_steps = kwargs["enabled_steps"]
            return {
                "input": {"usd_path": kwargs["input_usd_path"]},
                "output": {"usd_path": kwargs["output_usd_path"]},
                "steps": {step: {} for step in enabled_steps},
            }

        monkeypatch.setattr(
            material_api,
            "build_unified_pipeline_config",
            minimal_config,
        )
        monkeypatch.setattr(
            pipeline_router,
            "execute_pipeline_async",
            capture_execute,
            raising=True,
        )

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )
        assert regen_r.status_code == 202
        await asyncio.wait_for(finished.wait(), timeout=2)

        metadata_at_start = captured["metadata"]
        for stale_field in (
            "coverage",
            "partial_results",
            "failed_step",
            "error",
            "completed_at",
            "failed_at",
            "cancelled_at",
            "results",
            "duration_seconds",
            "step_timings",
            "timings",
        ):
            assert stale_field not in metadata_at_start
        assert metadata_at_start["completed_steps"] == []
        assert metadata_at_start["overall_progress"]["percent"] == 0
        assert captured["snapshot"]["completed_steps"] == []
        assert captured["snapshot"]["status"] == "pending"
        assert captured["queue_size"] == 0
        assert set(captured["state"]["step_outputs"]) == {
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
        }
        first_metadata = await manager.get_session_metadata(session_id)
        assert first_metadata["artifact_validity"]["restored_predictions"] is False

        predictions_r = await client.get(f"/artifacts/{session_id}/predictions")
        assert predictions_r.status_code == 200
        assert "CurrentRaw" in predictions_r.text
        assert "StaleRestored" not in predictions_r.text

        finished.clear()
        apply_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )
        assert apply_r.status_code == 202
        await asyncio.wait_for(finished.wait(), timeout=2)
        apply_metadata = await manager.get_session_metadata(session_id)
        assert apply_metadata["artifact_validity"]["restored_predictions"] is False
        predictions_r = await client.get(f"/artifacts/{session_id}/predictions")
        assert "CurrentRaw" in predictions_r.text
        assert "StaleRestored" not in predictions_r.text

    async def test_apply_regeneration_preserves_valid_restored_lineage(
        self,
        client,
        monkeypatch,
    ):
        """Apply-only regeneration may reuse a still-valid restored artifact."""
        import material_agent.api as material_api

        from ...service.routers import pipeline_router

        manager = pipeline_router.get_session_manager()
        session_id = str(uuid4())
        session_dir = await manager.create_session(session_id)
        (session_dir / "input" / "scene.usda").write_text(
            "#usda 1.0\n",
            encoding="utf-8",
        )
        raw_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text('{"material": "raw"}\n', encoding="utf-8")
        restored_path = (
            session_dir / "cache" / "restored" / "restored_predictions.jsonl"
        )
        restored_path.parent.mkdir(parents=True, exist_ok=True)
        restored_path.write_text('{"material": "restored"}\n', encoding="utf-8")
        state_path = session_dir / "cache" / ".pipeline_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "completed_steps": ["predict", "restore_usd", "apply"],
                    "failed_steps": [],
                    "step_errors": {},
                    "step_outputs": {
                        "predict": {"predictions_path": str(raw_path)},
                        "restore_usd": {
                            "restored_predictions_path": str(restored_path)
                        },
                        "apply": {"output_usd_path": "old.usd"},
                    },
                }
            ),
            encoding="utf-8",
        )
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "restored_predictions_valid": True,
                "results": {},
                "coverage": None,
                "completed_at": "2026-01-01T00:00:00+00:00",
            },
        )

        finished = asyncio.Event()

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            coverage_policy: str = "allow_partial",
            regeneration_claim: Any | None = None,
        ) -> None:
            await _finish_stub_execution(
                session_manager,
                session_id,
                regeneration_claim,
                {
                    "status": "completed",
                    "completed_steps": [{"name": "apply"}],
                },
            )
            finished.set()

        monkeypatch.setattr(
            material_api,
            "build_unified_pipeline_config",
            lambda **kwargs: {
                "input": {"usd_path": kwargs["input_usd_path"]},
                "output": {"usd_path": kwargs["output_usd_path"]},
                "steps": {step: {} for step in kwargs["enabled_steps"]},
            },
        )
        monkeypatch.setattr(
            pipeline_router,
            "execute_pipeline_async",
            capture_execute,
            raising=True,
        )

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )
        assert regen_r.status_code == 202
        await asyncio.wait_for(finished.wait(), timeout=2)

        metadata = await manager.get_session_metadata(session_id)
        assert metadata["artifact_validity"]["restored_predictions"] is True
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "restore_usd" in state["step_outputs"]
        assert "apply" not in state["step_outputs"]
        predictions_r = await client.get(f"/artifacts/{session_id}/predictions")
        assert predictions_r.status_code == 200
        assert "restored" in predictions_r.text
        assert '"raw"' not in predictions_r.text

    async def test_regenerate_predict_uses_local_nim_routing(self, client, monkeypatch):
        """Regeneration should reuse create-time service VLM/LLM routing."""
        from ...service.routers import pipeline_router

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
        monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
        monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
        monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
        monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
        monkeypatch.setattr(pipeline_router.config, "vlm_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "vlm_model", "local-vlm")
        monkeypatch.setattr(pipeline_router.config, "llm_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "llm_model", "local-llm")

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            coverage_policy: str = "allow_partial",
            regeneration_claim: Any | None = None,
        ) -> None:
            captured_pipeline_configs.append(config_dict)
            await _finish_stub_execution(
                session_manager,
                session_id,
                regeneration_claim,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        create_r = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={"user_email": "test@example.com"},
        )

        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]

        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) == 1
        await _wait_for_completed(client, session_id)
        manager = pipeline_router.get_session_manager()
        session_dir = manager.get_session_dir(session_id)
        dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text('{"id": "/Root"}\n', encoding="utf-8")
        usd_dataset_dir = session_dir / "cache" / "dataset" / "usd"
        usd_dataset_dir.mkdir(parents=True, exist_ok=True)
        (usd_dataset_dir / "prims.jsonl").write_text(
            '{"id": "/Root"}\n',
            encoding="utf-8",
        )
        (session_dir / "cache" / ".pipeline_state.json").write_text(
            json.dumps(
                {
                    "completed_steps": [
                        "build_dataset_usd",
                        "build_dataset_prepare_dataset",
                    ],
                    "step_outputs": {
                        "build_dataset_usd": {
                            "output_dir": str(usd_dataset_dir),
                        },
                        "build_dataset_prepare_dataset": {
                            "dataset_jsonl_path": str(dataset_path),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )

        assert regen_r.status_code == 202
        for _ in range(20):
            if len(captured_pipeline_configs) == 2:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) == 2

        predict_config = captured_pipeline_configs[-1]["steps"]["predict"]
        assert predict_config["vlm"]["backend"] == "nim"
        assert predict_config["vlm"]["model"] == "local-vlm"
        assert predict_config["vlm"]["base_url"] == "http://vlm-nim:8000/v1"
        assert predict_config["llm"]["backend"] == "nim"
        assert predict_config["llm"]["model"] == "local-llm"
        assert predict_config["llm"]["base_url"] == "http://llm-nim:8000/v1"

    async def test_regenerate_layer_only_requires_apply(self, client, monkeypatch):
        """Regeneration should not inject apply just because layer_only=true."""
        from ...service.routers import pipeline_router

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            coverage_policy: str = "allow_partial",
            regeneration_claim: Any | None = None,
        ) -> None:
            await _finish_stub_execution(
                session_manager,
                session_id,
                regeneration_claim,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        create_r = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                )
            },
            data={"user_email": "test@example.com"},
        )
        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]
        await _wait_for_completed(client, session_id)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"], "layer_only": True},
        )

        assert regen_r.status_code == 400
        assert "layer_only=true requires the apply step" in regen_r.json()["detail"]

    async def test_regenerate_returns_400_while_running(self, client):
        """Test that regenerate returns 400 while pipeline is running."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Immediately try to regenerate (still running)
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )

        # Should return 400 - can't regenerate while running
        if regen_r.status_code != 202:
            assert regen_r.status_code == 400

    async def test_regenerate_nonexistent_session(self, client):
        """Test regenerate on nonexistent session returns 404."""
        regen_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/regenerate",
            json={"steps": ["apply"]},
        )

        assert regen_r.status_code == 404

    async def test_regenerate_rejects_empty_step_list(self, client):
        """The API contract rejects a no-op regeneration request."""
        regen_r = await client.post(
            f"/pipeline/{uuid4()}/regenerate",
            json={"steps": []},
        )

        assert regen_r.status_code == 422

    async def test_regenerate_with_prompt_override(self, client):
        """Test regenerate with user prompt override."""
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Regenerate with custom prompt
        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={
                "steps": ["predict", "apply"],
                "user_prompt": "Focus on shiny surfaces",
            },
        )

        assert regen_r.status_code == 202

    async def test_regenerate_rejects_secret_prompt_before_claim(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A prompt override must not claim or mutate a completed session."""
        from ...service.routers import pipeline_router

        manager = pipeline_router.get_session_manager()
        session_id = str(uuid4())
        session_dir = await manager.create_session(
            session_id,
            config={"coverage_policy": "allow_partial"},
        )
        (session_dir / "input" / "scene.usda").write_bytes(b"#usda 1.0\n")
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {},
                "coverage": None,
                "completed_at": "2026-07-15T00:00:00+00:00",
            },
        )
        metadata_before = await manager.get_session_metadata(session_id)
        files_before = _session_file_snapshot(session_dir)
        claim_calls = 0

        async def unexpected_claim(*args: Any, **kwargs: Any) -> Any:
            nonlocal claim_calls
            claim_calls += 1
            raise AssertionError("rejected regeneration must not claim the session")

        monkeypatch.setattr(manager, "claim_regeneration", unexpected_claim)
        sentinel = "material-regeneration-prompt-sentinel-713"

        response = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={
                "steps": ["apply"],
                "user_prompt": f"Authorization: Bearer {sentinel}",
            },
        )

        assert response.status_code == 400
        assert response.json() == {"detail": _INVALID_DURABLE_REQUEST_DETAIL}
        assert sentinel not in response.text
        assert sentinel not in caplog.text
        assert claim_calls == 0
        assert await manager.get_session_metadata(session_id) == metadata_before
        assert _session_file_snapshot(session_dir) == files_before

    async def test_regenerate_rejects_reused_secret_descriptions_before_claim(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Reused descriptions are validated before regeneration owns the session."""
        from ...service.routers import pipeline_router

        manager = pipeline_router.get_session_manager()
        session_id = str(uuid4())
        session_dir = await manager.create_session(
            session_id,
            config={"coverage_policy": "allow_partial"},
        )
        (session_dir / "input" / "scene.usda").write_bytes(b"#usda 1.0\n")
        reference_dir = session_dir / "input" / "reference_images"
        reference_dir.mkdir(parents=True, exist_ok=True)
        (reference_dir / "reference_0000.png").write_bytes(_PNG_BYTES)
        sentinel = "material-regeneration-description-sentinel-713"
        (reference_dir / "descriptions.json").write_text(
            json.dumps([f"Authorization: Bearer {sentinel}"]),
            encoding="utf-8",
        )
        await manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {},
                "coverage": None,
                "completed_at": "2026-07-15T00:00:00+00:00",
            },
        )
        metadata_before = await manager.get_session_metadata(session_id)
        files_before = _session_file_snapshot(session_dir)
        claim_calls = 0

        async def unexpected_claim(*args: Any, **kwargs: Any) -> Any:
            nonlocal claim_calls
            claim_calls += 1
            raise AssertionError("rejected regeneration must not claim the session")

        monkeypatch.setattr(manager, "claim_regeneration", unexpected_claim)

        response = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["apply"]},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": _INVALID_DURABLE_REQUEST_DETAIL}
        assert sentinel not in response.text
        assert sentinel not in caplog.text
        assert claim_calls == 0
        assert await manager.get_session_metadata(session_id) == metadata_before
        assert _session_file_snapshot(session_dir) == files_before

    async def test_regenerate_upload_first_preserves_render_num_workers(
        self, client, monkeypatch
    ):
        """Upload-first runs should keep render worker limits on regenerate."""
        from ...service.routers import pipeline_router

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            coverage_policy: str = "allow_partial",
            regeneration_claim: Any | None = None,
        ) -> None:
            captured_pipeline_configs.append(config_dict)
            await _finish_stub_execution(
                session_manager,
                session_id,
                regeneration_claim,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        usd_content = b"#usda 1.0\n"
        upload_r = await client.post(
            "/pipeline/upload-usd",
            files={"usd_file": ("scene.usda", usd_content, "application/octet-stream")},
        )
        assert upload_r.status_code == 201
        session_id = upload_r.json()["session_id"]

        start_r = await client.post(
            "/pipeline",
            data={
                "session_id": session_id,
                "render_num_workers": "1",
                "user_email": "test@example.com",
            },
        )
        assert start_r.status_code == 202

        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"]["num_workers"]
            == 1
        )
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"][
                "max_concurrent_requests"
            ]
            == 1
        )
        await _wait_for_completed(client, session_id)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["build_dataset_usd"]},
        )
        assert regen_r.status_code == 202

        for _ in range(20):
            if len(captured_pipeline_configs) >= 2:
                break
            await asyncio.sleep(0)
        assert len(captured_pipeline_configs) >= 2
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"]["num_workers"]
            == 1
        )
        assert (
            captured_pipeline_configs[-1]["steps"]["build_dataset_usd"][
                "max_concurrent_requests"
            ]
            == 1
        )

    @pytest.mark.parametrize(
        "manifest_text",
        [
            """
materials:
  library_path: material_library.usda
  entries:
    - name: Generated Orange Glossy Plastic
      binding: /World/Looks/Generated_Orange_Glossy_Plastic
      description: Glossy orange plastic generated for the enclosure.
""",
            """
library_path: material_library.usda
entries:
  - name: Generated Orange Glossy Plastic
    binding: /World/Looks/Generated_Orange_Glossy_Plastic
    description: Glossy orange plastic generated for the enclosure.
""",
        ],
    )
    async def test_regenerate_preserves_cached_generated_material_library(
        self, client, monkeypatch, manifest_text
    ):
        """Regeneration should keep generated-library mode from cached artifacts."""
        from ...service.routers import pipeline_router

        monkeypatch.setattr(pipeline_router.config, "image_gen_backend", "openai")
        monkeypatch.setattr(pipeline_router.config, "image_gen_model", "gpt-image-1")
        monkeypatch.setattr(
            pipeline_router.config,
            "image_gen_base_url",
            "http://image-gen.local/v1",
        )
        monkeypatch.setattr(
            pipeline_router.config,
            "image_gen_api_key",
            "super-secret-image-key",
        )

        captured_pipeline_configs: list[dict[str, Any]] = []

        async def capture_execute(
            session_id: str,
            config_dict: dict[str, Any],
            session_manager,
            user_email: str = "",
            coverage_policy: str = "allow_partial",
            regeneration_claim: Any | None = None,
        ) -> None:
            captured_pipeline_configs.append(config_dict)
            await _finish_stub_execution(
                session_manager,
                session_id,
                regeneration_claim,
                {"status": "completed", "results": {}, "can_cancel": False},
            )

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", capture_execute, raising=True
        )

        create_r = await client.post(
            "/pipeline",
            files={
                "usd_file": (
                    "scene.usda",
                    b"#usda 1.0\n",
                    "application/octet-stream",
                ),
                "reference_images": (
                    "reference.png",
                    _PNG_BYTES,
                    "image/png",
                ),
            },
            data={
                "user_email": "test@example.com",
                "enable_material_generation": "true",
            },
        )

        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]
        for _ in range(20):
            if captured_pipeline_configs:
                break
            await asyncio.sleep(0)
        assert captured_pipeline_configs
        await _wait_for_completed(client, session_id)

        session_dir = pipeline_router.get_session_manager().get_session_dir(session_id)
        generated_dir = session_dir / "cache" / "generated_material_library"
        generated_dir.mkdir(parents=True, exist_ok=True)
        material_library_path = generated_dir / "material_library.usda"
        material_library_path.write_text("#usda 1.0\n", encoding="utf-8")
        (generated_dir / "materials.yaml").write_text(manifest_text, encoding="utf-8")
        dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text('{"id": "/Root"}\n', encoding="utf-8")
        usd_dataset_dir = session_dir / "cache" / "dataset" / "usd"
        usd_dataset_dir.mkdir(parents=True, exist_ok=True)
        (usd_dataset_dir / "prims.jsonl").write_text(
            '{"id": "/Root"}\n',
            encoding="utf-8",
        )
        state_path = session_dir / "cache" / ".pipeline_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "completed_steps": [
                        "build_dataset_usd",
                        "build_dataset_prepare_dataset",
                    ],
                    "failed_steps": [],
                    "step_errors": {},
                    "step_outputs": {
                        "build_dataset_usd": {
                            "output_dir": str(usd_dataset_dir),
                        },
                        "build_dataset_prepare_dataset": {
                            "dataset_jsonl_path": str(dataset_path),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict", "apply"]},
        )

        assert regen_r.status_code == 202
        state = json.loads(state_path.read_text(encoding="utf-8"))
        generated_outputs = state["step_outputs"]["generate_material_library"]
        assert generated_outputs["generated_material_library_path"] == str(
            material_library_path.resolve()
        )
        assert generated_outputs["generated_materials_data"] == {
            "library_path": str(material_library_path.resolve()),
            "entries": [
                {
                    "name": "Generated Orange Glossy Plastic",
                    "binding": "/World/Looks/Generated_Orange_Glossy_Plastic",
                    "description": "Glossy orange plastic generated for the enclosure.",
                }
            ],
        }
