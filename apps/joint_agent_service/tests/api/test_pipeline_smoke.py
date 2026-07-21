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
import uuid
from pathlib import Path
from typing import Any, NoReturn

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
        assert len(body["run_id"]) == 32
        assert int(body["run_id"], 16) >= 0
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
        self, client, session_manager
    ):
        """A backend typo is a request error and must not reserve a run or session."""
        sessions_before = {path.name for path in session_manager.storage_path.iterdir()}

        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={"render_backend": "typo"},
        )

        assert response.status_code == 400
        assert "Unknown rendering backend: typo" in response.json()["detail"]
        assert {
            path.name for path in session_manager.storage_path.iterdir()
        } == sessions_before

    @pytest.mark.parametrize("source_kind", ["upload", "s3"])
    async def test_create_pipeline_rejects_inline_credentials_and_cleans_new_session(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        source_kind: str,
    ) -> None:
        """Credential rejection removes only the upload/S3 session it created."""
        manager = pipeline_router.get_session_manager()
        sentinel = "never-return-this-joint-credential"
        created_session_ids: list[str] = []
        create_session = manager.create_session
        build_config = pipeline_router.build_default_pipeline_config

        async def track_created_session(
            session_id: str,
            config: dict[str, Any] | None = None,
        ) -> Path:
            created_session_ids.append(session_id)
            return await create_session(session_id, config)

        def build_unsafe_config(**kwargs: Any) -> dict[str, Any]:
            pipeline_config = build_config(**kwargs)
            pipeline_config["steps"]["predict"]["vlm"]["api_key"] = sentinel
            return pipeline_config

        monkeypatch.setattr(manager, "create_session", track_created_session)
        monkeypatch.setattr(
            pipeline_router,
            "build_default_pipeline_config",
            build_unsafe_config,
        )

        request_kwargs: dict[str, Any]
        if source_kind == "upload":
            request_kwargs = {"files": make_pipeline_files()}
        else:
            monkeypatch.setattr(
                pipeline_router.config,
                "s3_allowed_buckets",
                "approved",
            )

            def download_s3_input(_s3_uri: str, session_dir: Path) -> Path:
                path = session_dir / "input" / "scene.usda"
                path.write_bytes(b"#usda 1.0\n")
                return path

            monkeypatch.setattr(
                pipeline_router,
                "_download_s3_to_session",
                download_s3_input,
            )
            request_kwargs = {"data": {"s3_uri": "s3://approved/scene.usda"}}

        response = await client.post("/pipeline", **request_kwargs)

        assert response.status_code == 400
        assert response.json() == {"detail": "Pipeline configuration is invalid"}
        assert len(created_session_ids) == 1
        rejected_session_id = created_session_ids[0]
        assert not await manager.session_exists(rejected_session_id)
        assert not manager.get_session_dir(rejected_session_id).exists()
        assert await manager.store.list_keys(rejected_session_id) == []
        assert not pipeline_router.get_job_registry().is_running(rejected_session_id)
        assert sentinel not in response.text
        assert sentinel not in caplog.text

    async def test_create_pipeline_credential_rejection_preserves_existing_session(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Credential rejection releases admission without deleting prior state."""
        manager = pipeline_router.get_session_manager()
        upload_response = await client.post(
            "/pipeline/upload-usd",
            files=make_pipeline_files(),
        )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]
        session_dir = manager.get_session_dir(session_id)
        input_path = session_dir / "input" / "scene.usda"
        metadata_before = await manager.get_session_metadata(session_id)
        assert metadata_before is not None

        sentinel = "never-return-this-existing-session-credential"
        build_config = pipeline_router.build_default_pipeline_config

        def build_unsafe_config(**kwargs: Any) -> dict[str, Any]:
            pipeline_config = build_config(**kwargs)
            pipeline_config["steps"]["predict"]["vlm"]["api_key"] = sentinel
            return pipeline_config

        monkeypatch.setattr(
            pipeline_router,
            "build_default_pipeline_config",
            build_unsafe_config,
        )

        response = await client.post(
            "/pipeline",
            data={"session_id": session_id},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "Pipeline configuration is invalid"}
        assert await manager.session_exists(session_id)
        assert input_path.read_bytes() == b"#usda 1.0\n"
        assert not (session_dir / "input" / "config.yaml").exists()
        keys = await manager.store.list_keys(session_id)
        assert all("pipeline_configs" not in key for key in keys)
        metadata_after = await manager.get_session_metadata(session_id)
        assert metadata_after is not None
        assert metadata_after["status"] == metadata_before["status"]
        assert metadata_after["config"] == metadata_before["config"]
        assert "active_run_id" not in metadata_after
        assert "active_run_expires_at" not in metadata_after
        assert not pipeline_router.get_job_registry().is_running(session_id)
        assert sentinel not in response.text
        assert sentinel not in caplog.text

    @pytest.mark.parametrize("route", ["/pipeline", "/pipeline/upload-usd"])
    @pytest.mark.parametrize(
        "allowed_buckets,s3_uri",
        [
            ("", "s3://approved/private/scene.usdz"),
            ("approved", "s3://foreign/private/scene.usdz"),
        ],
    )
    async def test_client_s3_uri_rejects_foreign_bucket_before_download(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        route: str,
        allowed_buckets: str,
        s3_uri: str,
    ) -> None:
        """Joint rejects S3 inputs before acquiring a possibly remote store."""
        monkeypatch.setattr(
            pipeline_router.config, "s3_allowed_buckets", allowed_buckets
        )
        manager_calls = 0
        download_calls = 0

        def fail_manager() -> NoReturn:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("S3 policy must precede session-store access")

        def fail_download(*_args: Any, **_kwargs: Any) -> None:
            nonlocal download_calls
            download_calls += 1
            raise AssertionError("foreign S3 bucket must not be downloaded")

        monkeypatch.setattr(pipeline_router, "get_session_manager", fail_manager)
        monkeypatch.setattr(pipeline_router, "download_file_from_s3", fail_download)

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

    async def test_create_pipeline_session_id_precedes_unused_s3_uri(
        self,
        client,
        session_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lower-priority S3/file fields must not affect an existing-session run."""
        upload_response = await client.post(
            "/pipeline/upload-usd",
            files=make_pipeline_files(),
        )
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]
        original_s3_uri = "s3://approved/original/scene.usdz"
        await session_manager.update_session(
            session_id,
            {
                "config": {
                    "has_usd_upload": False,
                    "s3_uri": original_s3_uri,
                }
            },
        )

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
            files=make_pipeline_files(),
            data={
                "session_id": session_id,
                "s3_uri": "s3://foreign/private/scene.usdz",
            },
        )

        assert response.status_code == 202, response.text
        assert response.json()["session_id"] == session_id
        metadata = await session_manager.get_session_metadata(session_id)
        assert metadata is not None
        assert metadata["config"]["s3_uri"] == original_s3_uri
        assert metadata["config"]["has_usd_upload"] is False

    async def test_create_pipeline_can_enable_joint_rigger(
        self, client, session_manager
    ):
        """Joint Rigger options should be written into the generated config."""
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "apply_joint_rigger": "true",
                "joint_rigger_adapter": "usd_joint_rigger",
                "joint_rigger_on_missing_dependency": "block",
                "joint_rigger_on_unready_candidates": "warn",
                "joint_rigger_template": "generic_prop",
                "joint_rigger_apply_masses": "false",
                "joint_rigger_apply_collision": "true",
            },
        )

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        config_path = session_manager.get_session_dir(session_id) / "input/config.yaml"
        generated_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        step_config = generated_config["steps"]["apply_joint_rigger"]

        assert step_config["enabled"] is True
        assert step_config["adapter"] == "usd_joint_rigger"
        assert step_config["on_missing_dependency"] == "block"
        assert step_config["on_unready_candidates"] == "warn"
        assert step_config["joint_rigger_template"] == "generic_prop"
        assert step_config["apply_masses"] is False
        assert step_config["apply_collision"] is True

    async def test_create_pipeline_defaults_joint_rigger_policies(
        self, client, session_manager
    ):
        """The provisional adapter should inherit shared default policies."""
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "apply_joint_rigger": "true",
                "joint_rigger_adapter": "usd_joint_rigger",
            },
        )

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        config_path = session_manager.get_session_dir(session_id) / "input/config.yaml"
        generated_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        step_config = generated_config["steps"]["apply_joint_rigger"]

        assert step_config["enabled"] is True
        assert step_config["adapter"] == "usd_joint_rigger"
        assert step_config["on_missing_dependency"] == "skip"
        assert step_config["on_unready_candidates"] == "warn"

    @pytest.mark.parametrize("adapter", [None, "owned_core"])
    async def test_create_pipeline_defaults_or_selects_owned_core_usdz(
        self,
        client,
        session_manager,
        adapter: str | None,
    ):
        """Owned core should be explicit in config and publish a USDZ target."""
        data = {"apply_joint_rigger": "true"}
        if adapter is not None:
            data["joint_rigger_adapter"] = adapter
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(
                usd_content=b"PK\x03\x04test",
                usd_filename="scene.usdz",
            ),
            data=data,
        )

        assert response.status_code == 202
        session_id = response.json()["session_id"]
        config_path = session_manager.get_session_dir(session_id) / "input/config.yaml"
        generated_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        step_config = generated_config["steps"]["apply_joint_rigger"]

        assert step_config["enabled"] is True
        assert step_config["adapter"] == "owned_core"
        assert "predictions_path" not in step_config
        assert step_config["articulation_candidates_path"].endswith(
            "/predictions/articulation_candidates.json"
        )
        assert step_config["output_usd_path"].endswith("/joint_rigger/rigged.usdz")
        assert step_config["apply_masses"] is False
        assert step_config["apply_collision"] is False
        assert generated_config["steps"]["infer_articulation_candidates"][
            "candidate_joint_types"
        ] == ["revolute", "prismatic"]

    @pytest.mark.parametrize(
        "field",
        ["joint_rigger_apply_masses", "joint_rigger_apply_collision"],
    )
    async def test_create_pipeline_rejects_owned_core_physics_flags(
        self,
        client,
        field: str,
    ):
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={
                "apply_joint_rigger": "true",
                "joint_rigger_adapter": "owned_core",
                field: "true",
            },
        )

        assert response.status_code == 400
        assert "topology-only owned_core" in response.json()["detail"]

    async def test_owned_core_result_metadata_advertises_existing_usdz_artifact(
        self,
        client,
        session_manager,
    ):
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(
                usd_content=b"PK\x03\x04input-package",
                usd_filename="scene.usdz",
            ),
            data={"apply_joint_rigger": "true"},
        )
        assert response.status_code == 202
        session_id = response.json()["session_id"]

        for _ in range(200):
            status = await client.get(f"/pipeline/{session_id}/status")
            if status.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        assert status.json()["status"] == "completed"

        joint_rigger_dir = (
            session_manager.get_session_dir(session_id) / "cache" / "joint_rigger"
        )
        joint_rigger_dir.mkdir(parents=True, exist_ok=True)
        (joint_rigger_dir / "rigged.usdz").write_bytes(b"PK\x03\x04generated")
        await session_manager.update_session(
            session_id,
            {
                "results": {
                    "joint_rigger_status": "authored",
                    "joint_rigger_artifacts": {
                        "joint_rigger_output": True,
                        "joint_rigger_diagnostics": False,
                        "joint_rigger_validation": False,
                    },
                }
            },
        )

        results = await client.get(f"/pipeline/{session_id}/results")
        output_url = results.json()["download_urls"]["joint_rigger_output"]
        output = await client.get(output_url)

        assert output_url.endswith("/joint-rigger-output")
        assert output.content == b"PK\x03\x04generated"
        assert 'filename="rigged.usdz"' in output.headers["content-disposition"]

    async def test_create_pipeline_rejects_joint_rigger_options_without_enable(
        self, client
    ):
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={"joint_rigger_adapter": "usd_joint_rigger"},
        )

        assert response.status_code == 400
        assert "apply_joint_rigger must be true" in response.json()["detail"]

    async def test_create_pipeline_rejects_joint_rigger_boolean_option_without_enable(
        self, client
    ):
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={"joint_rigger_apply_masses": "false"},
        )

        assert response.status_code == 400
        assert "apply_joint_rigger must be true" in response.json()["detail"]

    async def test_create_pipeline_rejects_invalid_joint_rigger_adapter(self, client):
        response = await client.post(
            "/pipeline",
            files=make_pipeline_files(),
            data={"apply_joint_rigger": "true", "joint_rigger_adapter": "bad"},
        )

        assert response.status_code == 400
        assert "joint_rigger_adapter" in response.json()["detail"]


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
        assert [step["name"] for step in final_status["completed_steps"]] == [
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "consistency_pass",
            "infer_articulation_candidates",
        ]


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
        assert "articulation_candidates" in urls
        assert "articulation_report" in urls
        assert "report" in urls
        assert "dataset" in urls
        assert "joint_rigger_output" not in urls
        assert "joint_rigger_diagnostics" not in urls
        assert "joint_rigger_validation" not in urls
        assert urls["predictions"].startswith("/")
        assert urls["articulation_candidates"].startswith("/")
        assert urls["articulation_report"].startswith("/")
        assert urls["report"].startswith("/")
        assert urls["dataset"].startswith("/")

    async def test_results_include_joint_rigger_urls_when_artifacts_exist(
        self, client, session_manager
    ):
        """Test results advertise Joint Rigger URLs only for existing artifacts."""
        session_id = str(uuid.uuid4())
        session_dir = await session_manager.create_session(session_id)
        joint_rigger_dir = session_dir / "cache" / "joint_rigger"
        joint_rigger_dir.mkdir(parents=True, exist_ok=True)
        (joint_rigger_dir / "rigged.usdz").write_bytes(b"PK\x03\x04owned-core")
        (joint_rigger_dir / "joint_rigger_diagnostics.json").write_text(
            '{"status": "authored"}',
            encoding="utf-8",
        )
        (joint_rigger_dir / "joint_rigger_validation.json").write_text(
            '{"validation_skipped": true}',
            encoding="utf-8",
        )
        await session_manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {
                    "joint_rigger_status": "authored",
                    "joint_rigger_artifacts": {
                        "joint_rigger_output": True,
                        "joint_rigger_diagnostics": True,
                        "joint_rigger_validation": True,
                    },
                },
            },
        )

        results_r = await client.get(f"/pipeline/{session_id}/results")
        urls = results_r.json()["download_urls"]

        assert results_r.status_code == 200
        assert urls["joint_rigger_output"].startswith("/")
        assert urls["joint_rigger_diagnostics"].startswith("/")
        assert urls["joint_rigger_validation"].startswith("/")


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

    async def test_download_articulation_candidates(self, client):
        """Test downloading the Stage 2 articulation candidates JSON file."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        download_r = await client.get(
            f"/artifacts/{session_id}/articulation-candidates"
        )

        assert download_r.status_code == 200
        assert download_r.headers["content-type"] == "application/json"
        assert len(download_r.content) > 0

    async def test_view_articulation_report(self, client):
        """Test viewing the Stage 2 articulation candidates HTML report."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        report_r = await client.get(f"/artifacts/{session_id}/articulation-report")

        assert report_r.status_code == 200
        assert "text/html" in report_r.headers["content-type"]
        assert "Articulation Candidates" in report_r.text

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

    async def test_download_joint_rigger_artifacts(self, client, session_manager):
        """Test downloading generated Joint Rigger artifacts."""
        session_id = str(uuid.uuid4())
        session_dir = await session_manager.create_session(session_id)
        joint_rigger_dir = session_dir / "cache" / "joint_rigger"
        joint_rigger_dir.mkdir(parents=True, exist_ok=True)
        (joint_rigger_dir / "rigged.usdz").write_bytes(b"PK\x03\x04owned-core")
        (joint_rigger_dir / "rigged.usd").write_text("#usda 1.0\n", encoding="utf-8")
        (joint_rigger_dir / "joint_rigger_diagnostics.json").write_text(
            '{"status":"authored"}\n',
            encoding="utf-8",
        )
        (joint_rigger_dir / "joint_rigger_validation.json").write_text(
            '{"validation_skipped":true}\n',
            encoding="utf-8",
        )
        await session_manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {
                    "joint_rigger_artifacts": {
                        "joint_rigger_output": True,
                        "joint_rigger_diagnostics": True,
                        "joint_rigger_validation": True,
                    }
                },
            },
        )

        output_r = await client.get(f"/artifacts/{session_id}/joint-rigger-output")
        diagnostics_r = await client.get(
            f"/artifacts/{session_id}/joint-rigger-diagnostics"
        )
        validation_r = await client.get(
            f"/artifacts/{session_id}/joint-rigger-validation"
        )

        assert output_r.status_code == 200
        assert output_r.content == b"PK\x03\x04owned-core"
        assert 'filename="rigged.usdz"' in output_r.headers["content-disposition"]
        assert diagnostics_r.status_code == 200
        assert diagnostics_r.json()["status"] == "authored"
        assert validation_r.status_code == 200
        assert validation_r.json()["validation_skipped"] is True

        (joint_rigger_dir / "rigged.usdz").unlink()
        legacy_output = await client.get(f"/artifacts/{session_id}/joint-rigger-output")
        assert legacy_output.content.startswith(b"#usda")
        assert 'filename="rigged.usd"' in legacy_output.headers["content-disposition"]

    async def test_stale_joint_rigger_output_is_hidden_by_current_run_metadata(
        self, client, session_manager
    ):
        session_id = str(uuid.uuid4())
        session_dir = await session_manager.create_session(session_id)
        output = session_dir / "cache" / "joint_rigger" / "rigged.usdz"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"stale")
        await session_manager.update_session(
            session_id,
            {
                "status": "completed",
                "results": {
                    "joint_rigger_artifacts": {
                        "joint_rigger_output": False,
                        "joint_rigger_diagnostics": False,
                        "joint_rigger_validation": False,
                    }
                },
            },
        )

        response = await client.get(f"/artifacts/{session_id}/joint-rigger-output")

        assert response.status_code == 404

    @pytest.mark.parametrize(
        "artifact_path",
        [
            "joint-rigger-output",
            "joint-rigger-diagnostics",
            "joint-rigger-validation",
        ],
    )
    async def test_download_missing_joint_rigger_artifacts_returns_404(
        self, client, session_manager, artifact_path: str
    ):
        session_id = str(uuid.uuid4())
        await session_manager.create_session(session_id)

        response = await client.get(f"/artifacts/{session_id}/{artifact_path}")

        assert response.status_code == 404

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
