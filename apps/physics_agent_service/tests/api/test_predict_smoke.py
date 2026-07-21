# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the /predict route group.

Mirrors apps/physics_agent_service/tests/api/test_pipeline_smoke.py but
targets POST /predict. We deliberately do NOT modify any /pipeline
fixtures or tests — the conftest stub_executor patches
pipeline_router.execute_pipeline_async, and we add a similar predict_router
patch here so /predict runs deterministically without hitting a VLM.
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from ..conftest import make_pipeline_files


def _make_predict_stub(executor_label: str = "fake_predict_executor") -> Callable:
    """Build a deterministic stub for execute_predict_async.

    Mirrors the pipeline stub but only writes predict-relevant artifacts and
    records the predict_mode the route detected.
    """

    async def fake_execute_predict(
        session_id: str,
        config_dict: dict,
        session_manager,
        *,
        dataset_path=None,
    ):
        manager = session_manager
        try:
            session_dir = manager.get_session_dir(session_id)

            # Mirror predict_executor's mode detection so the test stub
            # records the same predict_mode the real executor would.
            existing_dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
            if dataset_path is not None and Path(dataset_path).is_file():
                mode = "dataset_only"
                steps_run = ["predict"]
            elif existing_dataset.exists():
                mode = "dataset_only"
                steps_run = ["predict"]
            else:
                mode = "full_predict"
                steps_run = [
                    "identify_asset",
                    "build_dataset_usd",
                    "build_dataset_prepare_dataset",
                    "predict",
                ]

            await manager.update_session(
                session_id,
                {
                    "status": "running",
                    "predict_mode": mode,
                    "predict_steps_run": steps_run,
                },
            )

            ds_dir = session_dir / "cache" / "dataset"
            preds_dir = session_dir / "cache" / "predictions"
            ds_dir.mkdir(parents=True, exist_ok=True)
            preds_dir.mkdir(parents=True, exist_ok=True)

            if mode == "full_predict":
                # Build dataset stub
                await manager.update_step_progress(
                    session_id,
                    "build_dataset_usd",
                    {
                        "current": 50,
                        "total": 100,
                        "percent": 50,
                        "message": "Rendering: 50%",
                    },
                )
                await asyncio.sleep(0.01)
                with (ds_dir / "dataset.jsonl").open("w") as f:
                    for i in range(5):
                        f.write(
                            json.dumps(
                                {
                                    "id": f"/p{i}",
                                    "type": "Mesh",
                                    "images": {"prim_only": f"img_{i}.png"},
                                }
                            )
                            + "\n"
                        )
                await manager.mark_step_completed(session_id, "build_dataset_usd")

                await manager.update_step_progress(
                    session_id,
                    "build_dataset_prepare_dataset",
                    {
                        "current": 5,
                        "total": 5,
                        "percent": 60,
                        "message": "Preparing dataset",
                    },
                )
                await asyncio.sleep(0.01)
                await manager.mark_step_completed(
                    session_id, "build_dataset_prepare_dataset"
                )

            # Predict step
            for i, pct in enumerate((70, 80, 90)):
                with (preds_dir / "predictions.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"/p{i}",
                                "classification": "metal",
                                "confidence": 0.9,
                            }
                        )
                        + "\n"
                    )
                await manager.update_step_progress(
                    session_id,
                    "predict",
                    {
                        "current": i + 1,
                        "total": 5,
                        "percent": pct,
                        "message": f"Predicting: {i + 1}/5",
                    },
                )
                await asyncio.sleep(0.01)
            await manager.mark_step_completed(session_id, "predict")

            await manager.update_session(
                session_id,
                {
                    "status": "completed",
                    "results": {
                        "predictions_made": 3,
                        "failed_count": 0,
                        "predictions_path": str(preds_dir / "predictions.jsonl"),
                        "token_stats": {
                            "prompt_tokens": 1000,
                            "completion_tokens": 200,
                        },
                    },
                    "duration_seconds": 1,
                    "completed_at": "1970-01-01T00:00:01Z",
                    "can_cancel": False,
                },
            )

        except asyncio.CancelledError:
            await manager.update_session(session_id, {"status": "cancelled"})
            raise
        except Exception as e:
            await manager.update_session(
                session_id,
                {"status": "failed", "error": str(e), "failed_step": "predict"},
            )
            raise

    fake_execute_predict.__name__ = executor_label
    return fake_execute_predict


@pytest.fixture(autouse=True)
def _stub_predict_executor(monkeypatch):
    """Replace predict_router.execute_predict_async with a deterministic stub."""
    from ...service.routers import predict_router

    monkeypatch.setattr(
        predict_router,
        "execute_predict_async",
        _make_predict_stub(),
        raising=True,
    )


# Helpers -----------------------------------------------------------------


async def _wait_completed(client, session_id: str, route: str = "predict") -> dict:
    """Poll status until completed or 200 timeouts; returns final status body."""
    for _ in range(300):
        status_r = await client.get(f"/{route}/{session_id}/status")
        if status_r.status_code == 200 and status_r.json()["status"] == "completed":
            return status_r.json()
        await asyncio.sleep(0.01)
    return (await client.get(f"/{route}/{session_id}/status")).json()


async def _bootstrap_terminal_predict_session(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> tuple[str, Path, Path, Path]:
    """Create one completed Mode-A session and a distinct retry dataset."""
    first_dir = tmp_path / "accepted"
    first_dir.mkdir()
    first_dataset = first_dir / "dataset.jsonl"
    first_dataset.write_text(
        json.dumps(
            {
                "id": "/accepted",
                "type": "Mesh",
                "images": {"prim_only": "accepted.png"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response = await client.post(
        "/predict",
        data={"dataset_path": str(first_dataset)},
    )
    assert response.status_code == 202, response.text
    session_id = response.json()["session_id"]
    await _wait_completed(client, session_id)

    from ...service.routers import predict_router

    session_dir = predict_router.get_session_manager().get_session_dir(session_id)
    config_path = session_dir / "input" / "predict_config.yaml"
    dataset_target = session_dir / "cache" / "dataset" / "dataset.jsonl"

    retry_dir = tmp_path / "retry"
    retry_dir.mkdir()
    retry_dataset = retry_dir / "dataset.jsonl"
    retry_dataset.write_text(
        json.dumps(
            {
                "id": "/replacement",
                "type": "Mesh",
                "images": {"prim_only": "replacement.png"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return session_id, config_path, dataset_target, retry_dataset


# ============================================================================
# POST /predict
# ============================================================================


@pytest.mark.api
class TestPredictCreation:
    async def test_create_predict_with_usd_returns_202(self, client):
        """POST /predict with uploaded USD must return 202."""
        files = make_pipeline_files()
        r = await client.post("/predict", files=files)

        assert r.status_code == 202
        body = r.json()
        assert "session_id" in body
        assert body["status"] == "pending"

    async def test_create_predict_unique_sessions(self, client):
        r1 = await client.post("/predict", files=make_pipeline_files())
        r2 = await client.post("/predict", files=make_pipeline_files())
        assert r1.json()["session_id"] != r2.json()["session_id"]

    async def test_create_predict_rejects_unsupported_extension(self, client):
        files = [("usd_file", ("model.obj", b"v 0 0 0\n", "application/octet-stream"))]
        r = await client.post("/predict", files=files)
        assert r.status_code == 400
        assert "Invalid USD file type" in r.json()["detail"]

    async def test_create_predict_requires_some_input(self, client):
        r = await client.post("/predict")
        assert r.status_code == 400

    @pytest.mark.parametrize(
        ("data", "expected_detail"),
        [
            ({"render_backend": "typo"}, "Unknown rendering backend: typo"),
            (
                {
                    "optimize_usd": "true",
                    "enable_deinstance": "false",
                    "enable_split": "false",
                    "enable_deduplicate": "false",
                },
                "At least one optimization operation",
            ),
        ],
    )
    async def test_create_predict_rejects_invalid_mode_b_options_before_session_creation(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        data: dict[str, str],
        expected_detail: str,
    ) -> None:
        """Invalid Mode B options return 400 before upload/session side effects."""
        from ...service.routers import predict_router

        manager = predict_router.get_session_manager()
        sessions_before = {path.name for path in manager.storage_path.iterdir()}
        create_session = manager.create_session
        create_calls = 0

        async def track_create_session(session_id: str) -> Path:
            nonlocal create_calls
            create_calls += 1
            return await create_session(session_id)

        monkeypatch.setattr(manager, "create_session", track_create_session)

        response = await client.post(
            "/predict",
            files=make_pipeline_files(),
            data=data,
        )

        assert response.status_code == 400
        assert expected_detail in response.json()["detail"]
        assert create_calls == 0
        assert {path.name for path in manager.storage_path.iterdir()} == sessions_before

    async def test_existing_mode_b_session_validates_before_syncing_remote_input(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bad backend cannot trigger a remote Mode B input download."""
        from ...service.routers import predict_router

        manager = predict_router.get_session_manager()
        session_id = "00000000-0000-4000-8000-000000000712"
        await manager.create_session(session_id)
        await manager.update_session(session_id, {"status": "completed"})
        sync_calls: list[str] = []

        async def track_sync_from_store(
            synced_session_id: str, prefix: str = ""
        ) -> int:
            assert synced_session_id == session_id
            sync_calls.append(prefix)
            return 0

        monkeypatch.setattr(manager, "sync_from_store", track_sync_from_store)

        response = await client.post(
            "/predict",
            data={"session_id": session_id, "render_backend": "typo"},
        )

        assert response.status_code == 400
        assert "Unknown rendering backend: typo" in response.json()["detail"]
        assert sync_calls == ["cache/dataset/"]

    @pytest.mark.parametrize(
        "allowed_buckets,s3_uri",
        [
            ("", "s3://trusted-input-bucket/path/scene.usdz"),
            ("trusted-input-bucket", "s3://foreign-bucket/path/scene.usdz"),
        ],
    )
    async def test_create_predict_rejects_unapproved_s3_before_preflight(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
        allowed_buckets: str,
        s3_uri: str,
    ) -> None:
        """Authorization must precede both the HEAD preflight and download."""
        from ...service.routers import predict_router

        monkeypatch.setattr(
            predict_router.config,
            "s3_allowed_buckets",
            allowed_buckets,
        )
        manager_calls = 0
        preflight_calls = 0
        download_calls = 0

        def fail_manager() -> None:
            nonlocal manager_calls
            manager_calls += 1
            raise AssertionError("S3 policy must precede session-store access")

        def fail_if_preflighted(*_args: Any, **_kwargs: Any) -> None:
            nonlocal preflight_calls
            preflight_calls += 1
            raise AssertionError("foreign S3 bucket reached the HEAD preflight")

        def fail_if_downloaded(*_args: Any, **_kwargs: Any) -> None:
            nonlocal download_calls
            download_calls += 1
            raise AssertionError("foreign S3 bucket reached the downloader")

        monkeypatch.setattr(
            predict_router,
            "get_session_manager",
            fail_manager,
        )
        monkeypatch.setattr(
            predict_router,
            "_preflight_s3_object_size",
            fail_if_preflighted,
        )
        monkeypatch.setattr(
            predict_router,
            "download_file_from_s3",
            fail_if_downloaded,
        )

        response = await client.post(
            "/predict",
            data={"s3_uri": s3_uri},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "S3 URI is not permitted by the service's configured bucket allowlist"
        )
        assert manager_calls == 0
        assert preflight_calls == 0
        assert download_calls == 0

    async def test_create_predict_with_dataset_path_runs_mode_a(self, client, tmp_path):
        """A pre-prepared dataset.jsonl forces Mode A (predict-only)."""
        ds = tmp_path / "dataset.jsonl"
        ds.write_text(
            json.dumps(
                {
                    "id": "/p0",
                    "type": "Mesh",
                    "images": {"prim_only": "img.png"},
                }
            )
            + "\n"
        )

        r = await client.post(
            "/predict",
            data={
                "dataset_path": str(ds),
                # Mode A never renders, so Mode-B-only fields are ignored.
                "render_backend": "future-server-backend",
                "optimize_usd": "true",
                "enable_deinstance": "false",
                "enable_split": "false",
                "enable_deduplicate": "false",
            },
        )
        assert r.status_code == 202
        session_id = r.json()["session_id"]

        body = await _wait_completed(client, session_id)
        assert body["status"] == "completed"

        results_r = await client.get(f"/predict/{session_id}/results")
        assert results_r.status_code == 200
        results = results_r.json()
        assert results["mode"] == "dataset_only"
        assert results["steps_run"] == ["predict"]

        # The staged dataset must be exposed via /artifacts so callers and
        # the report generator can resolve the dataset alongside predictions.
        assert "dataset" in results["download_urls"]
        dataset_r = await client.get(f"/artifacts/{session_id}/dataset")
        assert dataset_r.status_code == 200
        assert dataset_r.text == ds.read_text()

    async def test_create_predict_rejects_missing_dataset_path(self, client, tmp_path):
        r = await client.post(
            "/predict",
            data={"dataset_path": str(tmp_path / "nope" / "dataset.jsonl")},
        )
        assert r.status_code == 400

    async def test_create_predict_rejects_non_canonical_dataset_filename(
        self, client, tmp_path
    ):
        # dataset_path that points at any file other than `dataset.jsonl`
        # must be rejected — defense in depth against the route being used
        # to exfiltrate arbitrary readable files under the sessions root.
        bogus = tmp_path / "predictions.jsonl"
        bogus.write_text("{}\n")
        r = await client.post("/predict", data={"dataset_path": str(bogus)})
        assert r.status_code == 400
        assert "dataset.jsonl" in r.json()["detail"]

    async def test_create_predict_rejects_usd_file_plus_s3_uri(self, client):
        """Sending two primary input sources at once must 400."""
        files = make_pipeline_files()
        r = await client.post(
            "/predict",
            files=files,
            data={"s3_uri": "s3://bucket/scene.usdz"},
        )
        assert r.status_code == 400
        detail = r.json()["detail"]
        assert "exactly one" in detail
        assert "usd_file" in detail and "s3_uri" in detail

    async def test_create_predict_rejects_session_id_plus_s3_uri(self, client):
        """session_id and s3_uri must not be combined silently."""
        # Bootstrap a session via /pipeline/upload-usd so we have a real id.
        upload_r = await client.post(
            "/pipeline/upload-usd", files=make_pipeline_files()
        )
        assert upload_r.status_code == 201
        session_id = upload_r.json()["session_id"]

        r = await client.post(
            "/predict",
            data={"session_id": session_id, "s3_uri": "s3://bucket/other.usdz"},
        )
        assert r.status_code == 400
        assert "exactly one" in r.json()["detail"]

    async def test_create_predict_rejects_dataset_path_plus_s3_uri(
        self, client, tmp_path
    ):
        """dataset_path forces Mode A, so combining it with s3_uri (Mode B) must 400."""
        ds = tmp_path / "dataset.jsonl"
        ds.write_text(
            json.dumps({"id": "/p0", "type": "Mesh", "images": {"prim_only": "x.png"}})
            + "\n"
        )
        r = await client.post(
            "/predict",
            data={"dataset_path": str(ds), "s3_uri": "s3://bucket/scene.usdz"},
        )
        assert r.status_code == 400
        assert "dataset_path" in r.json()["detail"]

    async def test_create_predict_rejects_dataset_path_plus_usd_file(
        self, client, tmp_path
    ):
        """dataset_path + uploaded USD is contradictory and must 400."""
        ds = tmp_path / "dataset.jsonl"
        ds.write_text(
            json.dumps({"id": "/p0", "type": "Mesh", "images": {"prim_only": "x.png"}})
            + "\n"
        )
        r = await client.post(
            "/predict",
            files=make_pipeline_files(),
            data={"dataset_path": str(ds)},
        )
        assert r.status_code == 400
        assert "dataset_path" in r.json()["detail"]

    async def test_create_predict_rejects_concurrent_session_rerun(
        self, client, monkeypatch
    ):
        """A second POST /predict for a session whose previous predict is
        still pending/running/cancelling must 409. Otherwise two jobs race
        on the same cache/ paths and the second clobbers the first's
        metadata."""
        from ...service.routers import predict_router

        async def slow_predict_executor(
            session_id: str,
            config_dict: dict,
            session_manager,
            *,
            dataset_path=None,
        ):
            try:
                await session_manager.update_session(
                    session_id,
                    {"status": "running", "predict_mode": "full_predict"},
                )
                # Stay running long enough for the second POST to land.
                for _ in range(500):
                    await asyncio.sleep(0.01)
                await session_manager.update_session(
                    session_id, {"status": "completed"}
                )
            except asyncio.CancelledError:
                await session_manager.update_session(
                    session_id, {"status": "cancelled", "can_cancel": False}
                )
                raise

        monkeypatch.setattr(
            predict_router,
            "execute_predict_async",
            slow_predict_executor,
            raising=True,
        )

        first = await client.post("/predict", files=make_pipeline_files())
        assert first.status_code == 202
        session_id = first.json()["session_id"]

        # Wait for the executor to enter "running".
        for _ in range(200):
            status_r = await client.get(f"/predict/{session_id}/status")
            if status_r.status_code == 200 and status_r.json()["status"] == "running":
                break
            await asyncio.sleep(0.005)

        # Second POST for the same session_id while still running -> 409.
        second = await client.post(
            "/predict",
            data={"session_id": session_id},
        )
        assert second.status_code == 409
        detail = second.json()["detail"]
        assert "running" in detail or "pending" in detail

        # Cancel so the slow executor doesn't dominate test runtime.
        await client.post(f"/predict/{session_id}/cancel")

    async def test_concurrent_rerun_loser_does_not_mutate_session_state(
        self, client, monkeypatch
    ):
        """Same-pod rerun race: when two POST /predict reruns for the same
        terminal session arrive concurrently, the loser must return 409
        WITHOUT writing ``status=pending`` or its config block to session
        metadata.

        Under the previous "update_session before register()" ordering the
        loser passed the up-front status check, persisted its config, and
        only THEN got rejected by the JobRegistry — the accepted job ran
        with the rejected request's config. The fix reserves the slot
        before any session-state mutation, so the loser raises 409 at
        ``reserve()`` and never reaches ``update_session``.
        """
        from ...service.routers import predict_router

        # Build a terminal-state /predict session: run the default fast
        # stub once and wait for completion. After this point the
        # up-front is_running/status check will let a rerun through —
        # which is exactly the window the bug lived in.
        first = await client.post("/predict", files=make_pipeline_files())
        assert first.status_code == 202
        session_id = first.json()["session_id"]
        await _wait_completed(client, session_id)

        manager = predict_router.get_session_manager()
        baseline_metadata = await manager.get_session_metadata(session_id) or {}
        assert baseline_metadata.get("status") == "completed", (
            "test precondition: session must be terminal before the rerun race"
        )

        # Hold both reruns' executors so neither's update_session
        # writes ("running"/"completed") interferes with the assertions
        # below. The race we're testing happens inside the route handler,
        # before any executor work begins.
        executor_block = asyncio.Event()

        async def blocking_predict_executor(
            session_id: str,
            config_dict: dict,
            session_manager,
            *,
            dataset_path=None,
        ):
            try:
                await executor_block.wait()
            except asyncio.CancelledError:
                await session_manager.update_session(
                    session_id, {"status": "cancelled", "can_cancel": False}
                )
                raise

        monkeypatch.setattr(
            predict_router,
            "execute_predict_async",
            blocking_predict_executor,
            raising=True,
        )

        # Open the loser-vs-winner race by gating the route's
        # ``update_session("pending", ...)`` write. Holding it forces the
        # winner to remain in its critical section long enough for the
        # loser to also enter the route past the up-front status check —
        # exactly the interleaving that exposed the bug. Under the fix,
        # the loser fails at ``reserve()`` BEFORE reaching this gate, so
        # it never gets recorded.
        original_update_session = manager.update_session
        gate = asyncio.Event()
        pending_writes: list[dict] = []

        async def gated_update_session(target_session_id: str, updates: dict):
            if target_session_id == session_id and updates.get("status") == "pending":
                pending_writes.append(updates)
                await gate.wait()
            return await original_update_session(target_session_id, updates)

        monkeypatch.setattr(manager, "update_session", gated_update_session)

        # Two concurrent reruns with DISTINCT user_prompt values so the
        # winner's vs loser's persisted config is unambiguously identifiable.
        async def rerun(prompt: str):
            return await client.post(
                "/predict",
                data={"session_id": session_id, "user_prompt": prompt},
            )

        prompt_a = "RERUN_A_PROMPT"
        prompt_b = "RERUN_B_PROMPT"
        rerun_a = asyncio.create_task(rerun(prompt_a))
        rerun_b = asyncio.create_task(rerun(prompt_b))

        # Yield enough times that both POSTs reach their critical section
        # (or, on the fixed code, that the loser bails at reserve()).
        for _ in range(20):
            await asyncio.sleep(0.01)
            if pending_writes:
                # Winner is parked at the gate. Give the loser one more
                # tick to either also queue (old code) or 409-out (fix).
                await asyncio.sleep(0.02)
                break

        # Release the gated update_session and let both POSTs complete.
        gate.set()
        results = await asyncio.gather(rerun_a, rerun_b)

        statuses = sorted(r.status_code for r in results)
        assert statuses == [202, 409], (
            f"Expected exactly one 202 and one 409, got {statuses}: "
            f"{[r.text for r in results]}"
        )

        # Identify winner/loser by status code rather than positional
        # ordering, since which task wins the reserve() race is up to the
        # event loop's scheduling.
        winner = next(r for r in results if r.status_code == 202)
        idx = results.index(winner)
        winner_prompt = prompt_a if idx == 0 else prompt_b
        loser_prompt = prompt_b if winner_prompt == prompt_a else prompt_a

        # The accepted job's persisted config MUST reflect the winner's
        # user_prompt. Under the old ordering, the loser could overwrite
        # this with its own prompt before getting 409.
        post_metadata = await manager.get_session_metadata(session_id) or {}
        post_config = post_metadata.get("config") or {}
        assert post_config.get("user_prompt") == winner_prompt, (
            f"Session config user_prompt was clobbered by the loser: "
            f"got {post_config.get('user_prompt')!r}, expected winner's "
            f"{winner_prompt!r} (loser sent {loser_prompt!r})."
        )

        # Stronger assertion: only the winning request should have called
        # update_session("pending"). The loser must not appear at all —
        # this is the assertion that fails on the pre-fix ordering.
        assert len(pending_writes) == 1, (
            f"Loser wrote session state before getting 409: "
            f"{len(pending_writes)} pending-writes captured "
            f"(prompts: {[w.get('config', {}).get('user_prompt') for w in pending_writes]!r})"
        )
        assert pending_writes[0].get("config", {}).get("user_prompt") == winner_prompt

        persisted_config = (
            manager.get_session_dir(session_id) / "input" / "predict_config.yaml"
        ).read_text(encoding="utf-8")
        assert winner_prompt in persisted_config
        assert loser_prompt not in persisted_config

        # The registry must hold exactly the winner's task.
        registry = predict_router.get_job_registry()
        assert registry.is_running(session_id), (
            "winner's job should still be reserved/running at this point"
        )

        # Tear down: cancel the still-running winner via the route so the
        # autouse registry-cleanup fixture finds a clean state on exit.
        await client.post(f"/predict/{session_id}/cancel")
        executor_block.set()

    async def test_dataset_stage_failure_on_rerun_does_not_wedge_session(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Regression: a ``shutil.copyfile`` failure during dataset staging
        on a rerun (``session_created_here=False``) must NOT leave the
        session permanently stuck in ``status=pending``. A previous
        round-1 fix moved the copy *after* ``update_session("pending")``,
        which made any I/O error wedge the session — round-2 reverted
        that ordering. This test pins the contract: after the staging
        failure, ``GET /sessions/{id}`` shows the session at its previous
        terminal status, and a follow-up ``POST /predict`` with a usable
        dataset_path succeeds (instead of 409-ing on stale "pending").
        """
        from ...service.routers import predict_router

        # Bootstrap a terminal /predict session. dataset_path must point at
        # a file *named* dataset.jsonl per the route's canonical-filename
        # safety check, so each run lives in its own subdir.
        first_dir = tmp_path / "first"
        first_dir.mkdir()
        first_ds = first_dir / "dataset.jsonl"
        first_ds.write_text(
            json.dumps(
                {
                    "id": "/p0",
                    "type": "Mesh",
                    "images": {"prim_only": "img.png"},
                }
            )
            + "\n"
        )
        first_r = await client.post("/predict", data={"dataset_path": str(first_ds)})
        assert first_r.status_code == 202
        session_id = first_r.json()["session_id"]
        await _wait_completed(client, session_id)

        manager = predict_router.get_session_manager()
        baseline = await manager.get_session_metadata(session_id) or {}
        assert baseline.get("status") == "completed"
        config_path = (
            manager.get_session_dir(session_id) / "input" / "predict_config.yaml"
        )
        baseline_config = config_path.read_bytes()
        dataset_target = (
            manager.get_session_dir(session_id) / "cache" / "dataset" / "dataset.jsonl"
        )
        baseline_dataset = dataset_target.read_bytes()

        # Now force shutil.copyfile to fail on the next /predict. We need a
        # *different* dataset_path than the session's own staged file or
        # the route's already_staged samefile() short-circuit skips the
        # copy and we never hit the failure path.
        retry_dir = tmp_path / "retry"
        retry_dir.mkdir()
        retry_ds = retry_dir / "dataset.jsonl"
        retry_ds.write_text(first_ds.read_text())

        copyfile_calls = {"count": 0}
        original_copyfile = predict_router.shutil.copyfile

        def _exploding_copyfile(src: Any, dst: Any) -> None:
            copyfile_calls["count"] += 1
            assert Path(dst) != dataset_target
            Path(dst).write_bytes(b"partial-dataset-copy")
            raise OSError("simulated EIO during dataset stage")

        monkeypatch.setattr(predict_router.shutil, "copyfile", _exploding_copyfile)

        r = await client.post(
            "/predict",
            data={
                "session_id": session_id,
                "dataset_path": str(retry_ds),
                "user_prompt": "must-not-replace-the-accepted-config",
            },
        )
        assert r.status_code == 500
        assert "stage dataset" in r.json()["detail"].lower()
        assert copyfile_calls["count"] == 1

        # Crucial: the session must NOT be stuck in pending. Staging happens
        # before update_session, so the prior terminal status remains.
        post_failure = await manager.get_session_metadata(session_id) or {}
        assert post_failure.get("status") == "completed", (
            "rerun-staging-failure must not wedge the session in "
            f"pending; got status={post_failure.get('status')!r}"
        )
        assert config_path.read_bytes() == baseline_config
        assert dataset_target.read_bytes() == baseline_dataset
        assert not list(dataset_target.parent.glob(".dataset.jsonl.*.tmp"))

        # And a follow-up retry with a working copyfile must succeed
        # (otherwise the session would 409 forever).
        monkeypatch.setattr(predict_router.shutil, "copyfile", original_copyfile)
        retry = await client.post(
            "/predict",
            data={"session_id": session_id, "dataset_path": str(retry_ds)},
        )
        assert retry.status_code == 202, retry.text
        await _wait_completed(client, session_id)

    async def test_config_failure_after_dataset_stage_rolls_back_inputs(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A post-write config failure must not publish either retry input."""
        from ...service.routers import predict_router

        first_dir = tmp_path / "first"
        first_dir.mkdir()
        first_ds = first_dir / "dataset.jsonl"
        first_ds.write_text(
            json.dumps(
                {
                    "id": "/accepted",
                    "type": "Mesh",
                    "images": {"prim_only": "accepted.png"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        first_r = await client.post("/predict", data={"dataset_path": str(first_ds)})
        assert first_r.status_code == 202
        session_id = first_r.json()["session_id"]
        await _wait_completed(client, session_id)

        manager = predict_router.get_session_manager()
        session_dir = manager.get_session_dir(session_id)
        config_path = session_dir / "input" / "predict_config.yaml"
        dataset_target = session_dir / "cache" / "dataset" / "dataset.jsonl"
        baseline_config = config_path.read_bytes()
        baseline_dataset = dataset_target.read_bytes()

        retry_dir = tmp_path / "retry"
        retry_dir.mkdir()
        retry_ds = retry_dir / "dataset.jsonl"
        retry_ds.write_text(
            json.dumps(
                {
                    "id": "/replacement",
                    "type": "Mesh",
                    "images": {"prim_only": "replacement.png"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        original_persist = predict_router.build_and_write_pipeline_config
        persistence_calls = {"count": 0}

        async def _persist_then_fail(**kwargs: Any) -> None:
            persistence_calls["count"] += 1
            assert list(dataset_target.parent.glob(".dataset.jsonl.*.tmp"))
            await original_persist(**kwargs)
            assert b"must-be-rolled-back" in config_path.read_bytes()
            raise predict_router.HTTPException(
                status_code=500,
                detail="simulated config persistence failure",
            )

        monkeypatch.setattr(
            predict_router,
            "build_and_write_pipeline_config",
            _persist_then_fail,
        )

        response = await client.post(
            "/predict",
            data={
                "session_id": session_id,
                "dataset_path": str(retry_ds),
                "user_prompt": "must-be-rolled-back",
            },
        )

        assert response.status_code == 500
        assert persistence_calls["count"] == 1
        metadata = await manager.get_session_metadata(session_id) or {}
        assert metadata.get("status") == "completed"
        assert config_path.read_bytes() == baseline_config
        assert dataset_target.read_bytes() == baseline_dataset
        assert not list(dataset_target.parent.glob(".dataset.jsonl.*.tmp"))

    async def test_dataset_publish_failure_rolls_back_persisted_config(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A failed atomic dataset publish restores the prior config."""
        from ...service.routers import predict_router

        first_dir = tmp_path / "first"
        first_dir.mkdir()
        first_ds = first_dir / "dataset.jsonl"
        first_ds.write_text(
            json.dumps(
                {
                    "id": "/accepted",
                    "type": "Mesh",
                    "images": {"prim_only": "accepted.png"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        first_r = await client.post("/predict", data={"dataset_path": str(first_ds)})
        assert first_r.status_code == 202
        session_id = first_r.json()["session_id"]
        await _wait_completed(client, session_id)

        manager = predict_router.get_session_manager()
        session_dir = manager.get_session_dir(session_id)
        config_path = session_dir / "input" / "predict_config.yaml"
        dataset_target = session_dir / "cache" / "dataset" / "dataset.jsonl"
        baseline_config = config_path.read_bytes()
        baseline_dataset = dataset_target.read_bytes()

        retry_dir = tmp_path / "retry"
        retry_dir.mkdir()
        retry_ds = retry_dir / "dataset.jsonl"
        retry_ds.write_text(
            json.dumps(
                {
                    "id": "/replacement",
                    "type": "Mesh",
                    "images": {"prim_only": "replacement.png"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        original_replace = predict_router.os.replace
        publish_calls = {"count": 0}

        def _fail_dataset_publish(src: Any, dst: Any) -> None:
            if Path(dst) == dataset_target:
                publish_calls["count"] += 1
                if publish_calls["count"] == 1:
                    assert b"must-be-rolled-back" in config_path.read_bytes()
                    raise OSError("simulated atomic dataset publish failure")
            original_replace(src, dst)

        monkeypatch.setattr(predict_router.os, "replace", _fail_dataset_publish)

        response = await client.post(
            "/predict",
            data={
                "session_id": session_id,
                "dataset_path": str(retry_ds),
                "user_prompt": "must-be-rolled-back",
            },
        )

        assert response.status_code == 500
        # One failed publish followed by one atomic rollback publication.
        assert publish_calls["count"] == 2
        metadata = await manager.get_session_metadata(session_id) or {}
        assert metadata.get("status") == "completed"
        assert config_path.read_bytes() == baseline_config
        assert dataset_target.read_bytes() == baseline_dataset
        assert not list(dataset_target.parent.glob(".dataset.jsonl.*.tmp"))

    async def test_cancel_after_config_replace_quiesces_writer_and_rolls_back(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Cancellation cannot race a late config writer past input rollback."""
        from ...service import config_persistence
        from ...service.routers import predict_router

        (
            session_id,
            config_path,
            dataset_target,
            retry_dataset,
        ) = await _bootstrap_terminal_predict_session(client, tmp_path)
        manager = predict_router.get_session_manager()
        baseline_config = config_path.read_bytes()
        baseline_dataset = dataset_target.read_bytes()
        writer_replaced = threading.Event()
        writer_release = threading.Event()
        original_write = config_persistence.write_pipeline_config

        def _replace_then_block(path: Path, config: dict[str, Any]) -> None:
            original_write(path, config)
            writer_replaced.set()
            if not writer_release.wait(timeout=5):
                raise TimeoutError("test writer was not released")

        monkeypatch.setattr(
            config_persistence,
            "write_pipeline_config",
            _replace_then_block,
        )
        persist_task = asyncio.create_task(
            predict_router._persist_predict_inputs_transactionally(
                predict_config={"project": {"name": "cancelled-publication"}},
                config_path=config_path,
                dataset_source=retry_dataset,
                dataset_target=dataset_target,
                manager=manager,
                session_id=session_id,
                session_created_here=False,
            )
        )
        assert await asyncio.wait_for(
            asyncio.to_thread(writer_replaced.wait, 2),
            timeout=3,
        )
        assert b"cancelled-publication" in config_path.read_bytes()

        persist_task.cancel()
        await asyncio.sleep(0.05)
        assert not persist_task.done(), "cancellation must drain the writer first"
        writer_release.set()
        with pytest.raises(asyncio.CancelledError):
            await persist_task

        assert config_path.read_bytes() == baseline_config
        assert dataset_target.read_bytes() == baseline_dataset
        assert not list(config_path.parent.glob(".predict_config.yaml.*.tmp"))
        assert not list(dataset_target.parent.glob(".dataset.jsonl.*.tmp"))
        await asyncio.sleep(0.05)
        assert config_path.read_bytes() == baseline_config

    async def test_worker_start_failure_restores_existing_session_and_retries(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        """Registration failure restores exact inputs/metadata and frees its slot."""
        from ...service.routers import predict_router
        from ...service.runtime import registry as registry_module

        (
            session_id,
            config_path,
            dataset_target,
            retry_dataset,
        ) = await _bootstrap_terminal_predict_session(client, tmp_path)
        manager = predict_router.get_session_manager()
        registry = predict_router.get_job_registry()
        baseline_metadata = deepcopy(await manager.get_session_metadata(session_id))
        baseline_config = config_path.read_bytes()
        baseline_dataset = dataset_target.read_bytes()
        baseline_events = deepcopy(
            predict_router.get_event_bus().get_snapshot(session_id)
        )

        original_create_task = registry_module._create_job_task
        captured: dict[str, Any] = {}

        def _fail_start(wrapper: Any) -> None:
            captured["wrapper"] = wrapper
            raise RuntimeError("simulated worker registration failure")

        monkeypatch.setattr(registry_module, "_create_job_task", _fail_start)
        response = await client.post(
            "/predict",
            data={
                "session_id": session_id,
                "dataset_path": str(retry_dataset),
                "user_prompt": "must-not-survive-start-failure",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"] == predict_router._PREDICT_START_FAILED_DETAIL
        assert await manager.get_session_metadata(session_id) == baseline_metadata
        assert config_path.read_bytes() == baseline_config
        assert dataset_target.read_bytes() == baseline_dataset
        assert (
            predict_router.get_event_bus().get_snapshot(session_id) == baseline_events
        )
        assert registry.registered_count == 0
        assert registry.active_count == 0
        assert not registry.is_running(session_id)
        assert captured["wrapper"].cr_frame is None

        gc.collect()
        assert not any("was never awaited" in str(item.message) for item in recwarn)

        monkeypatch.setattr(
            registry_module,
            "_create_job_task",
            original_create_task,
        )
        retry = await client.post(
            "/predict",
            data={"session_id": session_id, "dataset_path": str(retry_dataset)},
        )
        assert retry.status_code == 202, retry.text
        await _wait_completed(client, session_id)

    async def test_worker_start_failure_deletes_only_fresh_session(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A failed start cannot orphan a request-owned fresh session."""
        from ...service.routers import predict_router
        from ...service.runtime import registry as registry_module

        dataset_dir = tmp_path / "fresh"
        dataset_dir.mkdir()
        dataset_path = dataset_dir / "dataset.jsonl"
        dataset_path.write_text(
            '{"id":"/fresh","type":"Mesh","images":{}}\n',
            encoding="utf-8",
        )
        manager = predict_router.get_session_manager()
        registry = predict_router.get_job_registry()
        sessions_before = set(await manager.list_sessions())
        original_create_task = registry_module._create_job_task

        def _fail_start(wrapper: Any) -> None:
            raise RuntimeError("simulated fresh worker registration failure")

        monkeypatch.setattr(registry_module, "_create_job_task", _fail_start)
        response = await client.post(
            "/predict",
            data={"dataset_path": str(dataset_path)},
        )

        assert response.status_code == 500
        assert set(await manager.list_sessions()) == sessions_before
        assert registry.registered_count == 0
        assert registry.active_count == 0

        monkeypatch.setattr(
            registry_module,
            "_create_job_task",
            original_create_task,
        )
        retry = await client.post(
            "/predict",
            data={"dataset_path": str(dataset_path)},
        )
        assert retry.status_code == 202, retry.text
        await _wait_completed(client, retry.json()["session_id"])

    async def test_metadata_transition_failure_restores_and_retries(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        """A post-registration metadata error cancels the gated worker safely."""
        from ...service.routers import predict_router

        (
            session_id,
            config_path,
            dataset_target,
            retry_dataset,
        ) = await _bootstrap_terminal_predict_session(client, tmp_path)
        manager = predict_router.get_session_manager()
        registry = predict_router.get_job_registry()
        baseline_metadata = deepcopy(await manager.get_session_metadata(session_id))
        baseline_config = config_path.read_bytes()
        baseline_dataset = dataset_target.read_bytes()
        original_update = manager.update_session

        async def _write_pending_then_fail(
            target_session_id: str,
            updates: dict[str, Any],
        ) -> None:
            await original_update(target_session_id, updates)
            if target_session_id == session_id and updates.get("status") == "pending":
                raise OSError("simulated metadata transition failure")

        monkeypatch.setattr(manager, "update_session", _write_pending_then_fail)
        response = await client.post(
            "/predict",
            data={
                "session_id": session_id,
                "dataset_path": str(retry_dataset),
                "user_prompt": "must-not-survive-metadata-failure",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"] == predict_router._PREDICT_START_FAILED_DETAIL
        assert await manager.get_session_metadata(session_id) == baseline_metadata
        assert config_path.read_bytes() == baseline_config
        assert dataset_target.read_bytes() == baseline_dataset
        assert registry.registered_count == 0
        assert registry.active_count == 0
        assert not registry.is_running(session_id)

        gc.collect()
        assert not any("was never awaited" in str(item.message) for item in recwarn)

        monkeypatch.setattr(manager, "update_session", original_update)
        retry = await client.post(
            "/predict",
            data={"session_id": session_id, "dataset_path": str(retry_dataset)},
        )
        assert retry.status_code == 202, retry.text
        await _wait_completed(client, session_id)

    async def test_rollback_failure_is_contained_and_session_is_retryable(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        """Rollback diagnostics stay fixed/value-free and do not leak a slot."""
        from ...service.routers import predict_router

        (
            session_id,
            config_path,
            dataset_target,
            retry_dataset,
        ) = await _bootstrap_terminal_predict_session(client, tmp_path)
        manager = predict_router.get_session_manager()
        registry = predict_router.get_job_registry()
        baseline_metadata = deepcopy(await manager.get_session_metadata(session_id))
        baseline_config = config_path.read_bytes()
        baseline_dataset = dataset_target.read_bytes()
        original_update = manager.update_session
        original_restore = predict_router._FileSnapshot.restore
        restore_calls = {"count": 0}

        async def _write_pending_then_fail(
            target_session_id: str,
            updates: dict[str, Any],
        ) -> None:
            await original_update(target_session_id, updates)
            if target_session_id == session_id and updates.get("status") == "pending":
                raise OSError("simulated startup transition failure")

        def _fail_first_restore(
            snapshot: predict_router._FileSnapshot,
        ) -> None:
            restore_calls["count"] += 1
            if restore_calls["count"] == 1:
                raise OSError("simulated rollback reporting failure")
            original_restore(snapshot)

        monkeypatch.setattr(manager, "update_session", _write_pending_then_fail)
        monkeypatch.setattr(
            predict_router._FileSnapshot,
            "restore",
            _fail_first_restore,
        )
        response = await client.post(
            "/predict",
            data={
                "session_id": session_id,
                "dataset_path": str(retry_dataset),
                "user_prompt": "must-not-appear-in-rollback-diagnostics",
            },
        )

        assert response.status_code == 500
        assert response.json()["detail"] == predict_router._PREDICT_START_FAILED_DETAIL
        assert restore_calls["count"] == 2
        assert "code=predict_start_input_restore_failed" in caplog.text
        assert "must-not-appear-in-rollback-diagnostics" not in caplog.text
        assert await manager.get_session_metadata(session_id) == baseline_metadata
        assert config_path.read_bytes() != baseline_config
        assert b"must-not-appear-in-rollback-diagnostics" in config_path.read_bytes()
        assert dataset_target.read_bytes() == baseline_dataset
        assert registry.registered_count == 0
        assert registry.active_count == 0
        assert not registry.is_running(session_id)

        gc.collect()
        assert not any("was never awaited" in str(item.message) for item in recwarn)

        monkeypatch.setattr(manager, "update_session", original_update)
        monkeypatch.setattr(
            predict_router._FileSnapshot,
            "restore",
            original_restore,
        )
        retry = await client.post(
            "/predict",
            data={"session_id": session_id, "dataset_path": str(retry_dataset)},
        )
        assert retry.status_code == 202, retry.text
        await _wait_completed(client, session_id)

    async def test_session_id_only_with_unrunnable_staged_dataset_rejected(
        self, client, monkeypatch, tmp_path
    ):
        """Regression: when a previous /predict run staged an external
        dataset.jsonl into ``cache/dataset/`` but its image PNGs live next
        to the *original* dataset_path (i.e. they were never copied),
        a follow-up ``POST /predict`` with ONLY ``session_id`` must NOT
        accept Mode A. The router's preflight detects unresolvable images
        up front and 400s with a recovery hint, instead of letting the
        executor fail asynchronously when it tries Mode A and finds the
        per-prim renders missing.
        """
        from ...service.routers import predict_router

        predict_router.get_session_manager()

        # Stage a session manually: a dataset.jsonl whose images aren't on
        # disk (mirrors the externally-staged-then-evaporated case).
        external_ds = tmp_path / "external" / "dataset.jsonl"
        external_ds.parent.mkdir()
        external_ds.write_text(
            json.dumps(
                {
                    "id": "/p0",
                    "media": {
                        "images": [{"path": "model_a/render_001.png", "type": "render"}]
                    },
                }
            )
            + "\n"
        )
        first_r = await client.post("/predict", data={"dataset_path": str(external_ds)})
        assert first_r.status_code == 202
        session_id = first_r.json()["session_id"]
        await _wait_completed(client, session_id)

        # The cache/dataset/dataset.jsonl now exists in this session_dir,
        # but model_a/render_001.png does not. session_id-only rerun must
        # 400 instead of letting the executor enter Mode A blindly.
        r = await client.post("/predict", data={"session_id": session_id})
        assert r.status_code == 400, r.text
        detail = r.json()["detail"].lower()
        assert (
            "staged dataset" in detail
            or "images are not present" in detail
            or "rebuild from source" in detail
        ), detail

    async def test_create_predict_s3_head_object_preflight_rejects_oversized(
        self, client, monkeypatch
    ):
        """An S3 object whose ContentLength exceeds the cap must 413 BEFORE
        the file is downloaded — head_object preflight, not a post-write
        size check that has already filled the volume."""
        from ...service.config import config as svc_config
        from ...service.routers import predict_router

        # Stub the s3_utils boto3 client so head_object returns a giant
        # ContentLength and download_file_from_s3 is never reached.
        oversized = (svc_config.max_upload_size_mb + 1) * 1024 * 1024

        class _StubS3Client:
            # boto3 head_object uses PascalCase kwargs per the wire API.
            def head_object(self, *, Bucket, Key):  # noqa: N803
                return {"ContentLength": oversized}

        from world_understanding.utils import s3_utils

        monkeypatch.setattr(svc_config, "s3_allowed_buckets", "bucket")
        monkeypatch.setattr(
            s3_utils, "_create_s3_client", lambda profile_name=None: _StubS3Client()
        )
        download_calls = {"count": 0}

        def _explode_download(*args, **kwargs):
            download_calls["count"] += 1
            raise AssertionError(
                "download_file_from_s3 must NOT run after preflight rejects"
            )

        monkeypatch.setattr(predict_router, "download_file_from_s3", _explode_download)

        r = await client.post(
            "/predict", data={"s3_uri": "s3://bucket/path/scene.usdz"}
        )
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()
        assert download_calls["count"] == 0


# ============================================================================
# GET /predict/{id}/status
# ============================================================================


@pytest.mark.api
class TestPredictStatus:
    async def test_status_for_valid_session(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        status_r = await client.get(f"/predict/{session_id}/status")

        assert status_r.status_code == 200
        body = status_r.json()
        assert body["session_id"] == session_id
        assert "status" in body
        assert "overall_progress" in body

    async def test_status_for_nonexistent_session_returns_404(self, client):
        r = await client.get("/predict/00000000-0000-0000-0000-000000000000/status")
        assert r.status_code == 404

    async def test_progresses_to_completed(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        body = await _wait_completed(client, session_id)
        assert body["status"] == "completed"


# ============================================================================
# GET /predict/{id}/results
# ============================================================================


@pytest.mark.api
class TestPredictResults:
    async def test_results_returns_202_while_running(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]
        # Wait until the worker observably entered pending/running before
        # asserting 202 — the stub can otherwise complete before the call.
        saw_inflight = False
        for _ in range(50):
            status_r = await client.get(f"/predict/{session_id}/status")
            if status_r.status_code == 200 and status_r.json()["status"] in (
                "pending",
                "running",
            ):
                saw_inflight = True
                break
            await asyncio.sleep(0.005)
        assert saw_inflight, "job reached terminal state before inflight assertion"
        r = await client.get(f"/predict/{session_id}/results")
        assert r.status_code == 202

    async def test_results_after_completion(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await _wait_completed(client, session_id)
        r = await client.get(f"/predict/{session_id}/results")

        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == session_id
        assert body["status"] == "completed"
        assert body["mode"] in ("dataset_only", "full_predict")
        # Issue: PredictOutput's sticky fields must be exposed.
        assert "predictions_count" in body
        assert "failed_count" in body
        assert "token_stats" in body
        assert "predictions_path" in body
        # Download URLs.
        urls = body["download_urls"]
        assert "predictions" in urls
        assert "report" in urls
        # /predict in Mode B builds a dataset, so dataset URL should be there
        assert "dataset" in urls

    async def test_results_predict_output_field_stability(self, client):
        """Regression: predictions_path/_count, failed_count, token_stats stay."""
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await _wait_completed(client, session_id)
        body = (await client.get(f"/predict/{session_id}/results")).json()
        assert body["predictions_count"] == 3
        assert body["failed_count"] == 0
        assert body["predictions_path"].endswith("predictions.jsonl")
        assert body["token_stats"]["prompt_tokens"] == 1000


# ============================================================================
# /predict/{id}/cancel
# ============================================================================


@pytest.mark.api
class TestPredictCancel:
    async def test_cancel_running_predict(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        cancel_r = await client.post(f"/predict/{session_id}/cancel")
        # Status either 200 with "cancelling" or 400 if already completed
        assert cancel_r.status_code in (200, 400)

    async def test_cancel_unknown_session(self, client):
        r = await client.post("/predict/00000000-0000-0000-0000-000000000000/cancel")
        assert r.status_code == 404

    async def test_cancel_rejects_non_predict_session(self, client):
        """/predict/{id}/cancel must refuse a /pipeline-only session.

        Without this guard, a client calling /predict/{id}/cancel against a
        session started via /pipeline would silently cancel the pipeline
        job while receiving 'Predict cancellation requested' — confusing
        and incorrect.
        """
        # Bootstrap a session through /pipeline/upload-usd. That route stamps
        # session metadata WITHOUT predict_route=True.
        upload_r = await client.post(
            "/pipeline/upload-usd", files=make_pipeline_files()
        )
        assert upload_r.status_code == 201
        session_id = upload_r.json()["session_id"]

        # Set the session metadata.config to look like a pipeline session
        # (no predict_route stamp). The fixture conftest may not have set
        # status to running; we just need the config block to exist.
        from ...service.routers import predict_router

        manager = predict_router.get_session_manager()
        existing = await manager.get_session_metadata(session_id) or {}
        existing_config = (existing.get("config") or {}).copy()
        existing_config["pipeline_route"] = True
        existing_config.pop("predict_route", None)
        await manager.update_session(
            session_id,
            {"status": "running", "config": existing_config},
        )

        cancel_r = await client.post(f"/predict/{session_id}/cancel")
        assert cancel_r.status_code == 409
        assert "not a predict session" in cancel_r.json()["detail"]

    async def test_cancel_after_completion_rejected(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]
        await _wait_completed(client, session_id)
        r = await client.post(f"/predict/{session_id}/cancel")
        assert r.status_code == 400

    async def test_cancel_reaches_terminal_cancelled(self, client, monkeypatch):
        """Cancel mid-flight must flip status to terminal 'cancelled', not
        leave it stuck at 'cancelling'. Uses a slow stub that yields
        before completing so the test has time to call /cancel."""
        from ...service.routers import predict_router

        async def slow_predict_executor(
            session_id: str,
            config_dict: dict,
            session_manager,
            *,
            dataset_path=None,
        ):
            try:
                await session_manager.update_session(
                    session_id,
                    {"status": "running", "predict_mode": "full_predict"},
                )
                # Long sleep so cancel can land before completion. Use lots
                # of small sleeps so CancelledError can arrive promptly.
                for _ in range(500):
                    await asyncio.sleep(0.01)
                await session_manager.update_session(
                    session_id, {"status": "completed"}
                )
            except asyncio.CancelledError:
                await session_manager.update_session(
                    session_id,
                    {
                        "status": "cancelled",
                        "can_cancel": False,
                    },
                )
                raise

        monkeypatch.setattr(
            predict_router,
            "execute_predict_async",
            slow_predict_executor,
            raising=True,
        )

        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        # Wait for the executor to enter "running"
        for _ in range(100):
            status_r = await client.get(f"/predict/{session_id}/status")
            if status_r.json()["status"] == "running":
                break
            await asyncio.sleep(0.01)

        cancel_r = await client.post(f"/predict/{session_id}/cancel")
        assert cancel_r.status_code == 200

        # Wait for terminal state.
        for _ in range(200):
            status_r = await client.get(f"/predict/{session_id}/status")
            if status_r.json()["status"] in ("cancelled", "completed", "failed"):
                break
            await asyncio.sleep(0.01)
        final_status = (await client.get(f"/predict/{session_id}/status")).json()
        assert final_status["status"] == "cancelled"


# ============================================================================
# Artifacts via existing /artifacts endpoints
# ============================================================================


@pytest.mark.api
class TestPredictArtifacts:
    async def test_artifacts_predictions_for_predict_session(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]
        await _wait_completed(client, session_id)

        r = await client.get(f"/artifacts/{session_id}/predictions")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/x-ndjson"
        assert len(r.content) > 0

    async def test_artifacts_dataset_for_predict_session(self, client):
        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]
        await _wait_completed(client, session_id)

        r = await client.get(f"/artifacts/{session_id}/dataset")
        assert r.status_code == 200

    async def test_artifacts_report_for_predict_session(self, client, tmp_path):
        """`/artifacts/{id}/report` must accept /predict-created sessions.

        We pre-author a report.html so the endpoint short-circuits to a plain
        FileResponse, validating the session_id resolution path. The on-demand
        generation branch is exercised by the existing pipeline tests.
        """
        from ...service.routers import predict_router

        create_r = await client.post("/predict", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]
        await _wait_completed(client, session_id)

        # Pre-create the report so the artifact endpoint serves it directly.
        manager = predict_router.get_session_manager()
        report_path = (
            manager.get_session_dir(session_id)
            / "cache"
            / "predictions"
            / "report.html"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html><body>predict report</body></html>")

        r = await client.get(f"/artifacts/{session_id}/report")
        assert r.status_code == 200
        assert "predict report" in r.text
