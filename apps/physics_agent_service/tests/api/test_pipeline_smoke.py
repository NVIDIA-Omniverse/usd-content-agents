# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for pipeline API endpoints (happy path).

Tests the core workflows:
- Create pipeline (upload USD file)
- Get status (poll for progress)
- Get results (download artifacts)
- Download endpoints
"""

import asyncio
from typing import Any

import pytest
import yaml

from ...service.routers import pipeline_router
from ..conftest import make_pipeline_files


@pytest.mark.api
class TestPipelineCreation:
    """Test pipeline creation endpoint."""

    async def test_create_pipeline_with_usd_file(self, client):
        """Test creating a pipeline with a USD file."""
        files = make_pipeline_files()

        response = await client.post("/pipeline", files=files)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"
        assert body["message"] == "Pipeline queued for execution"

    async def test_create_pipeline_generates_session_id(self, client):
        """Test that each pipeline creation generates a unique session ID."""
        r1 = await client.post("/pipeline", files=make_pipeline_files())
        r2 = await client.post("/pipeline", files=make_pipeline_files())

        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        assert sid1 != sid2
        assert len(sid1) > 0
        assert len(sid2) > 0

    async def test_create_pipeline_rejects_unsupported_usd_extension(self, client):
        """Test that unsupported USD file types are rejected."""
        files = [
            ("usd_file", ("model.obj", b"v 0 0 0\n", "application/octet-stream")),
        ]

        response = await client.post("/pipeline", files=files)

        assert response.status_code == 400
        assert "Invalid USD file type" in response.json()["detail"]

    async def test_create_pipeline_requires_usd_or_session_id(self, client):
        """Test that pipeline creation requires either usd_file or session_id."""
        response = await client.post("/pipeline")

        assert response.status_code == 400

    async def test_create_pipeline_rejects_unknown_backend_before_session_creation(
        self, client
    ):
        """A backend typo is a request error and must not leave session state."""
        manager = pipeline_router.get_session_manager()
        sessions_before = {path.name for path in manager.storage_path.iterdir()}

        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={"render_backend": "typo"},
        )

        assert response.status_code == 400
        assert "Unknown rendering backend: typo" in response.json()["detail"]
        assert {path.name for path in manager.storage_path.iterdir()} == sessions_before

    @pytest.mark.parametrize("route", ["/pipeline", "/pipeline/upload-usd"])
    @pytest.mark.parametrize(
        "allowed_buckets,s3_uri",
        [
            ("", "s3://trusted-input-bucket/path/scene.usda"),
            ("trusted-input-bucket", "s3://foreign-bucket/path/scene.usda"),
        ],
    )
    async def test_create_pipeline_rejects_unapproved_s3_before_download(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        route: str,
        allowed_buckets: str,
        s3_uri: str,
    ) -> None:
        """Rejected Physics S3 input must not acquire the session store."""
        monkeypatch.setattr(
            pipeline_router.config,
            "s3_allowed_buckets",
            allowed_buckets,
        )
        manager_calls = 0
        download_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("S3 policy must precede session-store access")

        def fail_if_downloaded(*_args: Any, **_kwargs: Any) -> None:
            nonlocal download_calls
            download_calls += 1
            raise AssertionError("foreign S3 bucket reached the downloader")

        monkeypatch.setattr(pipeline_router, "get_session_manager", fail_manager)
        monkeypatch.setattr(
            pipeline_router,
            "download_file_from_s3",
            fail_if_downloaded,
        )

        response = await client.post(
            route,
            data={"s3_uri": s3_uri},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "S3 URI is not permitted by the service's configured bucket allowlist"
        )
        assert manager_calls == 0
        assert download_calls == 0

    async def test_create_pipeline_session_id_precedes_lower_priority_sources(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lower-priority fields must not affect an existing-session run."""
        upload_response = await client.post(
            "/pipeline/upload-usd",
            files=make_pipeline_files(),
        )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]
        manager = pipeline_router.get_session_manager()
        before_config = (await manager.get_session_metadata(session_id))["config"]
        before_config = {**before_config, "has_usd_upload": False}
        await manager.update_session(session_id, {"config": before_config})

        def fail_validation(*_args: Any, **_kwargs: Any) -> str:
            raise AssertionError("unused S3 URI reached authorization")

        def fail_download(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("unused S3 URI reached the downloader")

        monkeypatch.setattr(
            pipeline_router,
            "_validate_and_authorize_s3_usd_uri",
            fail_validation,
        )
        monkeypatch.setattr(pipeline_router, "download_file_from_s3", fail_download)

        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(
                usd_content=b"ignored lower-priority upload",
                usd_filename="ignored.usdz",
            ),
            data={
                "session_id": session_id,
                "s3_uri": "s3://foreign/private/scene.usdz",
            },
        )

        assert response.status_code == 202, response.text
        assert response.json()["session_id"] == session_id
        after_config = (await manager.get_session_metadata(session_id))["config"]
        assert before_config["s3_uri"] is None
        assert after_config["s3_uri"] is None
        assert after_config["original_filename"] == before_config["original_filename"]
        assert after_config["usd_path"] == before_config["usd_path"]
        assert after_config["has_usd_upload"] is False

    async def test_create_pipeline_accepts_optimizer_boolean_form_values(self, client):
        """FastAPI should parse common boolean form values for optimizer flags."""
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "optimize_usd": "yes",
                "enable_deinstance": "on",
                "enable_split": "1",
                "enable_deduplicate": "0",
            },
        )

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        manager = pipeline_router.get_session_manager()
        config_path = manager.get_session_dir(session_id) / "input" / "config.yaml"
        pipeline_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        optimize = pipeline_config["steps"]["optimize_usd"]
        assert optimize["enabled"] is True
        assert optimize["scene_optimizer_settings"]["enable_deinstance"] is True
        assert optimize["scene_optimizer_settings"]["enable_split_meshes"] is True
        assert optimize["scene_optimizer_settings"]["enable_deduplicate"] is False
        assert pipeline_config["steps"]["restore_usd"]["enabled"] is True

        session_r = await client.get(f"/sessions/{session_id}")
        metadata_config = session_r.json()["config"]
        assert metadata_config["optimize_usd"] is True
        assert metadata_config["enable_deinstance"] is True
        assert metadata_config["enable_split"] is True
        assert metadata_config["enable_deduplicate"] is False

    async def test_create_pipeline_from_session_accepts_optimizer_flags(self, client):
        """A ready uploaded session can be started later with optimizer flags."""
        upload_response = await client.post(
            "/pipeline/upload-usd", files=make_pipeline_files()
        )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]

        response = await client.post(
            "/pipeline",
            data={
                "session_id": session_id,
                "optimize_usd": "true",
                "enable_deinstance": "true",
            },
        )

        assert response.status_code == 202
        assert response.json()["session_id"] == session_id

        manager = pipeline_router.get_session_manager()
        config_path = manager.get_session_dir(session_id) / "input" / "config.yaml"
        pipeline_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        optimize = pipeline_config["steps"]["optimize_usd"]
        assert optimize["enabled"] is True
        assert optimize["scene_optimizer_settings"]["enable_deinstance"] is True
        assert pipeline_config["steps"]["restore_usd"]["enabled"] is True

        session_r = await client.get(f"/sessions/{session_id}")
        metadata_config = session_r.json()["config"]
        assert metadata_config["has_usd_upload"] is True
        assert metadata_config["optimize_usd"] is True
        assert metadata_config["enable_deinstance"] is True

    async def test_create_pipeline_rejects_optimizer_with_no_operations(self, client):
        """optimize_usd=true requires at least one optimizer operation."""
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "optimize_usd": "true",
                "enable_deinstance": "false",
                "enable_split": "false",
                "enable_deduplicate": "false",
            },
        )

        assert response.status_code == 400
        assert "At least one optimization operation" in response.json()["detail"]


@pytest.mark.api
class TestPipelineStatus:
    """Test pipeline status endpoint."""

    async def test_status_for_accepted_pipeline_does_not_wait_on_metadata_store(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Accepted sessions must return lightweight status even if the store stalls."""
        worker_started = asyncio.Event()
        worker_blocker = asyncio.Event()

        async def blocked_execute(*args, **kwargs) -> None:
            worker_started.set()
            await worker_blocker.wait()

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", blocked_execute, raising=True
        )

        create_r = await client.post("/pipeline", files=make_pipeline_files())
        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]
        await asyncio.wait_for(worker_started.wait(), timeout=1.0)

        manager = pipeline_router.get_session_manager()

        async def stalled_get_session_metadata(
            session_id: str,
        ) -> dict[str, Any] | None:
            await asyncio.sleep(10)
            return None

        monkeypatch.setattr(
            manager, "get_session_metadata", stalled_get_session_metadata
        )

        try:
            status_r = await asyncio.wait_for(
                client.get(f"/pipeline/{session_id}/status"),
                timeout=0.25,
            )

            assert status_r.status_code == 200
            body = status_r.json()
            assert body["session_id"] == session_id
            assert body["status"] in {"pending", "running"}
        finally:
            worker_blocker.set()

    async def test_get_status_for_valid_session(self, client):
        """Test getting status for a valid session."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        status_r = await client.get(f"/pipeline/{session_id}/status")

        assert status_r.status_code == 200
        body = status_r.json()
        assert body["session_id"] == session_id
        assert "status" in body
        assert "overall_progress" in body
        assert "current_step" in body

    async def test_get_status_for_nonexistent_session(self, client):
        """Test getting status for nonexistent session returns 404."""
        response = await client.get(
            "/pipeline/00000000-0000-0000-0000-000000000000/status"
        )

        assert response.status_code == 404

    async def test_status_progress_updates(self, client):
        """Test that status progress updates as pipeline executes."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        # Poll status until completion
        previous_percent = 0

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            assert status_r.status_code == 200

            body = status_r.json()
            current_percent = body["overall_progress"]["percent"]

            if current_percent > previous_percent:
                pass
            previous_percent = current_percent

            if body["status"] == "completed":
                break

            await asyncio.sleep(0.01)

        final_status = (await client.get(f"/pipeline/{session_id}/status")).json()
        assert final_status["overall_progress"]["percent"] == 100
        assert final_status["status"] == "completed"

    async def test_status_shows_completed_steps(self, client):
        """Test that completed steps are shown in status."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        final_status = (await client.get(f"/pipeline/{session_id}/status")).json()
        assert len(final_status["completed_steps"]) == 4
        assert final_status["completed_steps"][0]["name"] == "build_dataset_usd"
        assert (
            final_status["completed_steps"][1]["name"]
            == "build_dataset_prepare_dataset"
        )
        assert final_status["completed_steps"][2]["name"] == "predict"
        assert final_status["completed_steps"][3]["name"] == "apply_physics"


@pytest.mark.api
class TestPipelineResults:
    """Test results endpoint."""

    async def test_get_results_returns_202_while_running(self, client):
        """Test that /results returns 202 while pipeline is running."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        results_r = await client.get(f"/pipeline/{session_id}/results")

        assert results_r.status_code == 202

    async def test_get_results_after_completion(self, client):
        """Test that /results returns completed results."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")

        assert results_r.status_code == 200
        body = results_r.json()
        assert body["session_id"] == session_id
        assert body["status"] == "completed"
        assert "stats" in body
        assert "download_urls" in body

    async def test_results_have_download_urls(self, client):
        """Test that results include download URLs."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        results_r = await client.get(f"/pipeline/{session_id}/results")
        body = results_r.json()
        urls = body["download_urls"]

        assert "predictions" in urls
        assert "report" in urls
        assert "dataset" in urls
        assert urls["predictions"].startswith("/")
        assert urls["report"].startswith("/")
        assert urls["dataset"].startswith("/")


@pytest.mark.api
class TestDownloadEndpoints:
    """Test artifact download endpoints."""

    async def test_download_predictions(self, client):
        """Test downloading the predictions JSONL file."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        download_r = await client.get(f"/artifacts/{session_id}/predictions")

        assert download_r.status_code == 200
        assert download_r.headers["content-type"] == "application/x-ndjson"
        assert len(download_r.content) > 0

    async def test_download_dataset(self, client):
        """Test downloading the dataset JSONL file."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        download_r = await client.get(f"/artifacts/{session_id}/dataset")

        assert download_r.status_code == 200
        assert len(download_r.content) > 0

    async def test_download_nonexistent_session_returns_404(self, client):
        """Test that downloading from nonexistent session returns 404."""
        response = await client.get(
            "/artifacts/00000000-0000-0000-0000-000000000000/predictions"
        )

        assert response.status_code == 404

    async def test_download_incomplete_returns_404(self, client, monkeypatch):
        """Test that downloading from incomplete pipeline returns 404."""
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def hold_pipeline_incomplete(**_kwargs: Any) -> None:
            started.set()
            try:
                await release.wait()
            finally:
                finished.set()

        monkeypatch.setattr(
            pipeline_router,
            "execute_pipeline_async",
            hold_pipeline_incomplete,
            raising=True,
        )

        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await asyncio.wait_for(started.wait(), timeout=5.0)
        try:
            download_r = await client.get(f"/artifacts/{session_id}/predictions")
            assert download_r.status_code == 404
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=5.0)
