# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-instance simulation tests.

Part 1: Demonstrates that LOCAL storage breaks when multiple instances
        run behind a load balancer (each pod has its own disk).

Part 2: Demonstrates that SHARED storage (simulated by sharing a single
        LocalSessionStore) makes cross-instance operations work.

Part 3: Tests for specific cross-instance fixes:
        - Input USD sync after upload so other instances can start the pipeline
        - SSE returns 503 when session is not on this instance

Strategy: We use a SINGLE FastAPI app but swap the SessionManager between
requests to simulate a load balancer routing to different pods.
"""

import asyncio
import json
import shutil
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ..service.routers import (
    artifacts_router,
    pipeline_router,
    sessions_router,
)
from ..service.runtime.bus import EventBus
from ..service.runtime.events import ProgressEvent, StepState
from ..service.runtime.registry import JobRegistry
from ..service.session.cache_publications import (
    CACHE_PUBLICATIONS_FIELD,
    PIPELINE_CONFIG_PUBLICATION_ID_FIELD,
    PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD,
    cache_publication_path,
    pipeline_config_publication_key,
)
from ..service.session.manager import SessionManager
from ..service.storage import LocalSessionStore
from ..service.workers import executor
from ..service.workers.executor import finalize_pipeline_run

MINIMAL_USD = b'#usda 1.0\ndef Xform "Root" {}\n'


def _make_pipeline_files():
    return [("usd_file", ("cube.usda", MINIMAL_USD, "application/octet-stream"))]


def _switch_to(mgr: SessionManager):
    """Swap the global session manager — simulates request hitting a different pod."""
    pipeline_router.set_session_manager(mgr)
    artifacts_router.set_session_manager(mgr)
    sessions_router.set_session_manager(mgr)


def _rebase_expected_config(
    value,
    original_session_dir: Path,
    current_session_dir: Path,
):
    if isinstance(value, dict):
        return {
            key: _rebase_expected_config(
                item, original_session_dir, current_session_dir
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_expected_config(item, original_session_dir, current_session_dir)
            for item in value
        ]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value
    try:
        relative_path = Path(value).relative_to(original_session_dir)
    except ValueError:
        return value
    return str(current_session_dir / relative_path)


def _install_immediate_pipeline_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict], list[list[str] | None]]:
    captured_configs: list[dict] = []
    captured_steps: list[list[str] | None] = []

    async def capture_execute(
        session_id,
        run_id,
        config_dict,
        session_manager,
        only_steps=None,
    ):
        del session_id, run_id, session_manager
        captured_configs.append(deepcopy(config_dict))
        captured_steps.append(deepcopy(only_steps))

    class CapturingRegistry:
        def __init__(self) -> None:
            self.admissions: dict[str, str] = {}

        async def reserve_admission(self, session_id: str, run_id: str) -> bool:
            if session_id in self.admissions:
                return False
            self.admissions[session_id] = run_id
            return True

        async def release_admission(self, session_id: str, run_id: str) -> bool:
            if self.admissions.get(session_id) != run_id:
                return False
            del self.admissions[session_id]
            return True

        def is_running(self, session_id):
            return session_id in self.admissions

        async def register(
            self,
            _session_id,
            coro,
            *,
            run_id=None,
            liveness_guard=None,
            on_finish=None,
        ):
            del on_finish
            if run_id is not None:
                self.admissions.pop(_session_id, None)
            try:
                await coro
            finally:
                if hasattr(liveness_guard, "stop"):
                    await liveness_guard.stop()
                elif hasattr(liveness_guard, "close"):
                    liveness_guard.close()

    monkeypatch.setattr(pipeline_router, "execute_pipeline_async", capture_execute)
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: CapturingRegistry(),
    )
    return captured_configs, captured_steps


async def _complete_and_release_captured_run(
    manager: SessionManager,
    session_id: str,
    only_steps: list[str] | None = None,
) -> None:
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    run_id = metadata["active_run_id"]
    produced, published = await executor._publish_cache_publications(
        manager,
        session_id,
        run_id,
        only_steps,
    )
    cache_publications = await executor._merged_cache_publications(
        manager,
        session_id,
        produced,
        published,
    )
    for namespace in published:
        shutil.copytree(
            cache_publication_path(
                manager.get_session_dir(session_id),
                run_id,
                namespace,
            ),
            manager.get_session_dir(session_id) / "cache" / namespace,
        )
    assert await manager.terminalize_and_release_run(
        session_id,
        run_id,
        {
            "status": "completed",
            CACHE_PUBLICATIONS_FIELD: cache_publications,
        },
    )


def _write_prepared_dataset_closure(
    manager: SessionManager,
    session_id: str,
    *,
    write_image: bool = True,
) -> tuple[Path, Path]:
    dataset_dir = manager.get_session_dir(session_id) / "cache" / "dataset"
    image_path = dataset_dir / "usd" / "renders" / "component.png"
    if write_image:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"rendered-component")
    dataset_path = dataset_dir / "dataset.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        json.dumps(
            {
                "id": "/Root/Component",
                "media": {
                    "images": [
                        {
                            "path": "usd/renders/component.png",
                            "metadata": {"render_mode": "prim_only"},
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_path, image_path


def _write_render_dataset_closure(
    manager: SessionManager,
    session_id: str,
) -> tuple[Path, Path, Path]:
    usd_dir = manager.get_session_dir(session_id) / "cache" / "dataset" / "usd"
    image_path = usd_dir / "renders" / "component.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"source-render")
    dataset_metadata = usd_dir / "dataset.json"
    dataset_metadata.write_text(
        json.dumps({"statistics": {"total_prims": 1}}),
        encoding="utf-8",
    )
    prims_path = usd_dir / "prims.jsonl"
    prims_path.write_text(
        json.dumps(
            {
                "prim_path": "/Root/Component",
                "renders": [{"path": "renders/component.png"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_metadata, prims_path, image_path


@pytest.mark.parametrize(
    ("lease_seconds", "heartbeat_seconds"),
    [
        (0.0, 1.0),
        (float("nan"), 1.0),
        (10.0, 0.0),
        (10.0, float("inf")),
        (10.0, 10.0),
    ],
)
def test_session_manager_rejects_invalid_run_lease_settings(
    tmp_path,
    lease_seconds: float,
    heartbeat_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SessionManager(
            tmp_path / "invalid-lease",
            run_claim_lease_seconds=lease_seconds,
            run_claim_heartbeat_seconds=heartbeat_seconds,
        )


# ===========================================================================
# Fixtures: SEPARATE storage (simulates local-only, the broken case)
# ===========================================================================


@pytest.fixture()
def pod_a(tmp_path) -> SessionManager:
    """Pod A with its own local disk."""
    path = tmp_path / "pod_a_sessions"
    path.mkdir()
    store = LocalSessionStore(root_dir=str(path))
    return SessionManager(storage_path=path, ttl_hours=1, store=store)


@pytest.fixture()
def pod_b(tmp_path) -> SessionManager:
    """Pod B with its own local disk."""
    path = tmp_path / "pod_b_sessions"
    path.mkdir()
    store = LocalSessionStore(root_dir=str(path))
    return SessionManager(storage_path=path, ttl_hours=1, store=store)


# ===========================================================================
# Fixtures: SHARED storage (simulates S3, the fixed case)
# ===========================================================================


@pytest.fixture()
def shared_store(tmp_path) -> LocalSessionStore:
    """Shared store simulating S3 — both pods point to the same backend."""
    path = tmp_path / "shared_s3_sessions"
    path.mkdir()
    return LocalSessionStore(root_dir=str(path))


@pytest.fixture()
def shared_pod_a(tmp_path, shared_store) -> SessionManager:
    """Pod A with shared store (local working dir still separate)."""
    local_path = tmp_path / "shared_pod_a_local"
    local_path.mkdir()
    return SessionManager(storage_path=local_path, ttl_hours=1, store=shared_store)


@pytest.fixture()
def shared_pod_b(tmp_path, shared_store) -> SessionManager:
    """Pod B with shared store (local working dir still separate)."""
    local_path = tmp_path / "shared_pod_b_local"
    local_path.mkdir()
    return SessionManager(storage_path=local_path, ttl_hours=1, store=shared_store)


# ===========================================================================
# PART 1: Separate storage — proves the problem
# ===========================================================================


@pytest.mark.asyncio
async def test_session_not_visible_across_instances(client, pod_a, pod_b):
    """Session created on pod A is invisible to pod B (local storage)."""
    _switch_to(pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["session_id"]

    resp_a = await client.get(f"/sessions/{session_id}")
    assert resp_a.status_code == 200

    _switch_to(pod_b)
    resp_b = await client.get(f"/sessions/{session_id}")
    assert resp_b.status_code == 404, (
        f"Pod B should NOT see pod A's session, but got {resp_b.status_code}."
    )


@pytest.mark.asyncio
async def test_status_404_on_wrong_instance(client, pod_a, pod_b):
    """Pipeline status returns 404 when polled from wrong pod."""
    _switch_to(pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    resp_a = await client.get(f"/pipeline/{session_id}/status")
    assert resp_a.status_code == 200

    _switch_to(pod_b)
    resp_b = await client.get(f"/pipeline/{session_id}/status")
    assert resp_b.status_code == 404


@pytest.mark.asyncio
async def test_session_list_inconsistent_across_instances(client, pod_a, pod_b):
    """Each pod only lists sessions from its own local storage."""
    _switch_to(pod_a)
    for _ in range(2):
        await client.post("/pipeline/upload-usd", files=_make_pipeline_files())

    _switch_to(pod_b)
    await client.post("/pipeline/upload-usd", files=_make_pipeline_files())

    _switch_to(pod_a)
    sessions_a = (await client.get("/sessions")).json()

    _switch_to(pod_b)
    sessions_b = (await client.get("/sessions")).json()

    list_a = (
        sessions_a if isinstance(sessions_a, list) else sessions_a.get("sessions", [])
    )
    list_b = (
        sessions_b if isinstance(sessions_b, list) else sessions_b.get("sessions", [])
    )

    assert len(list_a) == 2, f"Pod A should see 2 sessions, got {len(list_a)}"
    assert len(list_b) == 1, f"Pod B should see 1 session, got {len(list_b)}"

    ids_a = {s["session_id"] for s in list_a}
    ids_b = {s["session_id"] for s in list_b}
    assert ids_a.isdisjoint(ids_b)


@pytest.mark.asyncio
async def test_cancel_fails_on_wrong_instance(client, pod_a, pod_b):
    """Cannot cancel a pipeline from a different pod (local storage)."""
    _switch_to(pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    _switch_to(pod_b)
    resp_b = await client.post(
        f"/pipeline/{session_id}/cancel",
        params={"run_id": "a" * 32},
    )
    assert resp_b.status_code in (404, 500)


@pytest.mark.asyncio
async def test_artifacts_not_downloadable_from_wrong_instance(client, pod_a, pod_b):
    """Artifact files live on pod A's disk, unreachable from pod B."""
    _switch_to(pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    preds_dir = pod_a.storage_path / session_id / "cache" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    (preds_dir / "predictions.jsonl").write_text(
        json.dumps({"id": "/Root", "classification": "furniture"}) + "\n"
    )
    metadata = await pod_a.get_session_metadata(session_id)
    assert metadata is not None
    metadata.pop(CACHE_PUBLICATIONS_FIELD)
    await pod_a.store.put_json(session_id, "session.json", metadata)
    await pod_a.update_session(session_id, {"status": "completed"})

    resp_a = await client.get(f"/artifacts/{session_id}/predictions")
    assert resp_a.status_code == 200

    _switch_to(pod_b)
    resp_b = await client.get(f"/artifacts/{session_id}/predictions")
    assert resp_b.status_code in (404, 500)


# ===========================================================================
# PART 2: Shared storage — proves the fix
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_session_visible_across_instances(
    client, shared_pod_a, shared_pod_b
):
    """Session created on pod A IS visible from pod B (shared store)."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Pod A sees it
    resp_a = await client.get(f"/sessions/{session_id}")
    assert resp_a.status_code == 200

    # Pod B also sees it via shared store
    _switch_to(shared_pod_b)
    resp_b = await client.get(f"/sessions/{session_id}")
    assert resp_b.status_code == 200, (
        f"Pod B should see shared session, got {resp_b.status_code}"
    )


@pytest.mark.asyncio
async def test_shared_status_available_from_any_instance(
    client, shared_pod_a, shared_pod_b
):
    """Pipeline status available from any pod (shared store)."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    # Status from pod A
    resp_a = await client.get(f"/pipeline/{session_id}/status")
    assert resp_a.status_code == 200

    # Status from pod B (reads from shared store)
    _switch_to(shared_pod_b)
    resp_b = await client.get(f"/pipeline/{session_id}/status")
    assert resp_b.status_code == 200


@pytest.mark.asyncio
async def test_shared_session_list_consistent(client, shared_pod_a, shared_pod_b):
    """Both pods return the same full session list (shared store)."""
    _switch_to(shared_pod_a)
    for _ in range(2):
        await client.post("/pipeline/upload-usd", files=_make_pipeline_files())

    _switch_to(shared_pod_b)
    await client.post("/pipeline/upload-usd", files=_make_pipeline_files())

    # Both pods should see all 3 sessions
    _switch_to(shared_pod_a)
    sessions_a = (await client.get("/sessions")).json()

    _switch_to(shared_pod_b)
    sessions_b = (await client.get("/sessions")).json()

    list_a = sessions_a.get("sessions", sessions_a)
    list_b = sessions_b.get("sessions", sessions_b)

    assert len(list_a) == 3, f"Pod A should see 3 sessions, got {len(list_a)}"
    assert len(list_b) == 3, f"Pod B should see 3 sessions, got {len(list_b)}"

    ids_a = {s["session_id"] for s in list_a}
    ids_b = {s["session_id"] for s in list_b}
    assert ids_a == ids_b, "Both pods should see identical session sets"


@pytest.mark.asyncio
async def test_shared_cancel_works_cross_instance(client, shared_pod_a, shared_pod_b):
    """Cancel signal written from pod B is visible to pod A (shared store)."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    # Simulate running status
    run_id = "a" * 32
    assert await shared_pod_a.reserve_run(session_id, run_id)
    assert await shared_pod_a.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )

    # Cancel from pod B
    _switch_to(shared_pod_b)
    resp_b = await client.post(
        f"/pipeline/{session_id}/cancel",
        params={"run_id": run_id},
    )
    assert resp_b.status_code == 200, f"Cancel should succeed: {resp_b.text}"

    # Verify cancel signal is visible from pod A
    _switch_to(shared_pod_a)
    is_cancelled = await shared_pod_a.is_cancelled(session_id)
    assert is_cancelled, "Cancel signal should be visible to pod A"


@pytest.mark.asyncio
async def test_shared_cancel_marker_stops_owner_and_terminalizes_cancelled(
    shared_pod_a: SessionManager,
    shared_pod_b: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = str(uuid.uuid4())
    run_id = "a" * 32
    await shared_pod_a.create_session(session_id)
    assert await shared_pod_a.reserve_run(session_id, run_id)
    assert await shared_pod_a.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )
    shared_pod_a.run_claim_heartbeat_seconds = 60.0
    monkeypatch.setattr(executor, "_RUN_CANCELLATION_POLL_SECONDS", 0.01)
    pipeline_started = asyncio.Event()

    async def slow_pipeline(params):
        pipeline_started.set()
        assert params.cancel_checker is not None
        while not params.cancel_checker():
            await asyncio.sleep(0.005)
        raise asyncio.CancelledError

    monkeypatch.setattr(executor, "arun_pipeline", slow_pipeline)
    registry = JobRegistry(max_concurrent=1)

    async def finish_run() -> None:
        await executor.finalize_pipeline_run(shared_pod_a, session_id, run_id)

    await registry.register(
        session_id,
        executor.execute_pipeline_async(
            session_id,
            run_id,
            {"project": {"name": "cross-instance-cancel"}},
            shared_pod_a,
        ),
        run_id=run_id,
        liveness_guard=executor.maintain_run_claim(
            shared_pod_a,
            session_id,
            run_id,
        ),
        on_finish=finish_run,
    )
    task = registry.get_task(session_id)
    assert task is not None
    await pipeline_started.wait()

    assert await shared_pod_b.request_cancellation(session_id, run_id)
    assert await shared_pod_a.is_cancellation_accepted(session_id, run_id)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.3)

    metadata = await shared_pod_b.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["failed_step"] == "cancelled"
    assert metadata["error"] == "Pipeline run was cancelled"
    assert "active_run_id" not in metadata
    assert "active_run_expires_at" not in metadata


@pytest.mark.asyncio
async def test_shared_cancel_polling_does_not_increase_lease_renewal_rate(
    shared_pod_a: SessionManager,
    shared_pod_b: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = str(uuid.uuid4())
    run_id = "a" * 32
    await shared_pod_a.create_session(session_id)
    assert await shared_pod_a.reserve_run(session_id, run_id)
    assert await shared_pod_a.update_session_for_run(
        session_id,
        run_id,
        {"status": "running"},
    )
    shared_pod_a.run_claim_heartbeat_seconds = 60.0
    monkeypatch.setattr(executor, "_RUN_CANCELLATION_POLL_SECONDS", 0.01)
    renewals = 0
    original_renew = shared_pod_a.renew_run

    async def count_renewal(selected_session_id: str, selected_run_id: str) -> bool:
        nonlocal renewals
        renewals += 1
        return await original_renew(selected_session_id, selected_run_id)

    monkeypatch.setattr(shared_pod_a, "renew_run", count_renewal)
    guard = asyncio.create_task(
        executor.maintain_run_claim(shared_pod_a, session_id, run_id)
    )
    await asyncio.sleep(0.02)

    assert await shared_pod_b.request_cancellation(session_id, run_id)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(guard, timeout=0.2)
    assert renewals == 0


@pytest.mark.asyncio
async def test_shared_artifacts_downloadable_from_any_instance(
    client, shared_pod_a, shared_pod_b, shared_store
):
    """Artifacts synced to shared store are downloadable from any pod."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    # Write predictions directly to the shared store (simulates sync_to_store)
    pred_data = json.dumps({"id": "/Root", "classification": "furniture"}) + "\n"
    run_id = "a" * 32
    prediction_key = f"artifacts/run_cache/{run_id}/cache/predictions/predictions.jsonl"
    await shared_store.put_bytes(
        session_id,
        prediction_key,
        pred_data.encode(),
        "application/x-ndjson",
    )
    await shared_pod_a.update_session(
        session_id,
        {
            "status": "completed",
            CACHE_PUBLICATIONS_FIELD: {"predictions": run_id},
        },
    )

    # Pod B can download (reads from shared store)
    _switch_to(shared_pod_b)
    resp_b = await client.get(f"/artifacts/{session_id}/predictions")
    assert resp_b.status_code == 200, (
        f"Pod B should serve shared artifacts, got {resp_b.status_code}: {resp_b.text}"
    )


# ===========================================================================
# PART 3: Cross-instance input sync and SSE 503
# ===========================================================================


@pytest.mark.asyncio
async def test_input_usd_synced_after_upload(client, shared_pod_a, shared_pod_b):
    """After upload-usd on pod A, pod B can start the pipeline (input in shared store)."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Verify input was synced to shared store by pod A
    input_keys = await shared_pod_a.store.list_keys(session_id, prefix="input/")
    assert any("scene" in k for k in input_keys), (
        f"Input file should be in shared store after upload, got keys: {input_keys}"
    )

    # Switch to pod B (separate local dir, same shared store) and start pipeline
    _switch_to(shared_pod_b)
    resp = await client.post(
        "/pipeline",
        data={"session_id": session_id, "render_backend": "warp"},
    )
    assert resp.status_code in (200, 201, 202), (
        f"Pod B should be able to start pipeline after pulling input from store: {resp.text}"
    )


@pytest.mark.asyncio
async def test_regenerate_restores_and_rebases_full_config_cross_instance(
    client,
    shared_store,
    shared_pod_a,
    shared_pod_b,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regeneration restores config plus step inputs under the current pod root."""
    captured_configs, captured_steps = _install_immediate_pipeline_capture(monkeypatch)

    _switch_to(shared_pod_a)
    created = await client.post(
        "/pipeline",
        files=_make_pipeline_files(),
        data={
            "render_backend": "warp",
            "user_prompt": "initial prompt",
            "apply_joint_rigger": "true",
            "joint_rigger_adapter": "mock",
            "joint_rigger_on_missing_dependency": "block",
            "joint_rigger_on_unready_candidates": "block",
            "joint_rigger_template": "custom-template",
            "joint_rigger_apply_masses": "true",
            "joint_rigger_apply_collision": "false",
        },
    )
    assert created.status_code == 202, created.text
    created_body = created.json()
    session_id = created_body["session_id"]
    created_run_id = created_body["run_id"]
    assert len(captured_configs) == 1

    published_config_key = pipeline_config_publication_key(created_run_id)
    assert await shared_store.exists(session_id, published_config_key)
    stored_config_stream = await shared_store.open_read(
        session_id,
        published_config_key,
    )
    try:
        stored_config = yaml.safe_load(stored_config_stream)
    finally:
        stored_config_stream.close()
    assert stored_config == captured_configs[0]

    pod_a_root = shared_pod_a.get_session_dir(session_id)
    pod_b_root = shared_pod_b.get_session_dir(session_id)
    pod_b_config = pod_b_root / "input/config.yaml"
    assert pod_a_root != pod_b_root
    pod_b_config.parent.mkdir(parents=True, exist_ok=True)
    pod_b_config.write_text("stale: true\n", encoding="utf-8")

    dataset_path, image_path = _write_prepared_dataset_closure(
        shared_pod_a,
        session_id,
    )
    assert (
        await shared_pod_a.sync_to_store(
            session_id,
            prefix="cache/dataset/",
            overwrite=True,
        )
        == 2
    )
    await _complete_and_release_captured_run(shared_pod_a, session_id)

    _switch_to(shared_pod_b)
    regenerated = await client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["predict"], "user_prompt": "regenerated prompt"},
    )
    assert regenerated.status_code == 202, regenerated.text
    assert pod_b_config.is_file()
    assert yaml.safe_load(pod_b_config.read_text(encoding="utf-8")) != {"stale": True}
    assert (pod_b_root / "input" / "scene.usda").is_file()
    restored_dataset = pod_b_root / dataset_path.relative_to(pod_a_root)
    restored_image = pod_b_root / image_path.relative_to(pod_a_root)
    assert restored_dataset.read_bytes() == dataset_path.read_bytes()
    assert restored_image.read_bytes() == image_path.read_bytes()
    assert len(captured_configs) == 2
    assert captured_steps == [None, ["predict"]]

    metadata = await shared_pod_b.get_session_metadata(session_id)
    assert metadata is not None
    regenerated_run_id = metadata["active_run_id"]
    assert metadata[PIPELINE_CONFIG_PUBLICATION_ID_FIELD] == regenerated_run_id
    assert await shared_store.exists(
        session_id,
        pipeline_config_publication_key(regenerated_run_id),
    )
    assert await shared_pod_b.release_run(session_id, regenerated_run_id)

    expected_config = _rebase_expected_config(
        captured_configs[0],
        pod_a_root,
        pod_b_root,
    )
    expected_config["steps"]["build_dataset_prepare_dataset"]["prompts"]["user"] = (
        "regenerated prompt"
    )
    assert captured_configs[1] == expected_config

    predictions_dir = pod_a_root / "cache" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = predictions_dir / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "id": "/Root/Component",
                "classification": {"component_type": "door"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidates_path = predictions_dir / "articulation_candidates.json"
    candidates_path.write_text(
        json.dumps({"candidates": [], "summary": {"candidate_count": 0}}),
        encoding="utf-8",
    )
    assert (
        await shared_pod_a.sync_to_store(
            session_id,
            prefix="cache/predictions/",
            overwrite=True,
        )
        == 2
    )
    prediction_run = "c" * 32
    assert await shared_pod_a.reserve_run(session_id, prediction_run)
    await _complete_and_release_captured_run(
        shared_pod_a,
        session_id,
        only_steps=["predict"],
    )

    inferred = await client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["infer_articulation_candidates"]},
    )
    assert inferred.status_code == 202, inferred.text
    restored_predictions = pod_b_root / predictions_path.relative_to(pod_a_root)
    restored_candidates = pod_b_root / candidates_path.relative_to(pod_a_root)
    assert restored_predictions.read_bytes() == predictions_path.read_bytes()
    assert restored_candidates.read_bytes() == candidates_path.read_bytes()
    assert captured_steps[-1] == ["infer_articulation_candidates"]
    await _complete_and_release_captured_run(shared_pod_b, session_id)


@pytest.mark.asyncio
async def test_reclaimed_config_upload_cannot_replace_successor_binding(
    tmp_path: Path,
) -> None:
    owner_run = "a" * 32
    successor_run = "b" * 32

    class PausingConfigStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.owner_upload_started = asyncio.Event()
            self.resume_owner_upload = asyncio.Event()

        async def sync_from_local(
            self,
            session_id: str,
            local_session_dir: str,
            prefix: str = "",
            *,
            overwrite: bool = False,
        ) -> int:
            if prefix == pipeline_config_publication_key(owner_run):
                self.owner_upload_started.set()
                await self.resume_owner_upload.wait()
            return await super().sync_from_local(
                session_id,
                local_session_dir,
                prefix=prefix,
                overwrite=overwrite,
            )

    store = PausingConfigStore(str(tmp_path / "config-store"))
    owner = SessionManager(
        tmp_path / "config-owner",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    successor = SessionManager(
        tmp_path / "config-successor",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    reader = SessionManager(tmp_path / "config-reader", store=store)
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    successor._claim_now = lambda: clock[0]  # type: ignore[method-assign]

    session_id = str(uuid.uuid4())
    owner_dir = await owner.create_session(session_id)
    owner_input = owner_dir / "input" / "scene.usda"
    owner_input.write_text("#usda 1.0\n", encoding="utf-8")
    assert await owner.reserve_run(session_id, owner_run)
    owner_config = {
        "input": {"usd_path": str(owner_input)},
        "source": "stale-owner",
    }
    stale_upload = asyncio.create_task(
        pipeline_router._persist_pipeline_config(
            owner,
            session_id,
            owner_run,
            owner_config,
            owner_input,
        )
    )
    await store.owner_upload_started.wait()

    clock[0] += timedelta(seconds=11)
    assert await successor.reserve_run(session_id, successor_run)
    successor_dir = successor.get_session_dir(session_id)
    successor_input = successor_dir / "input" / "scene.usda"
    successor_input.parent.mkdir(parents=True)
    successor_input.write_text("#usda 1.0\n", encoding="utf-8")
    successor_config = {
        "input": {"usd_path": str(successor_input)},
        "source": "accepted-successor",
    }
    successor_publication_sha256 = await pipeline_router._persist_pipeline_config(
        successor,
        session_id,
        successor_run,
        successor_config,
        successor_input,
    )
    assert await successor.update_session_for_run(
        session_id,
        successor_run,
        {
            PIPELINE_CONFIG_PUBLICATION_ID_FIELD: successor_run,
            PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD: successor_publication_sha256,
        },
    )

    store.resume_owner_upload.set()
    await stale_upload
    assert not await owner.update_session_for_run(
        session_id,
        owner_run,
        {PIPELINE_CONFIG_PUBLICATION_ID_FIELD: owner_run},
    )

    metadata = await reader.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata[PIPELINE_CONFIG_PUBLICATION_ID_FIELD] == successor_run
    restored_path = reader.get_session_dir(session_id) / "input" / "config.yaml"
    await pipeline_router._restore_pipeline_config(reader, session_id, restored_path)
    assert yaml.safe_load(restored_path.read_text(encoding="utf-8"))["source"] == (
        "accepted-successor"
    )
    assert await store.exists(
        session_id,
        pipeline_config_publication_key(owner_run),
    )
    assert await store.exists(
        session_id,
        pipeline_config_publication_key(successor_run),
    )
    assert await successor.release_run(session_id, successor_run)


@pytest.mark.asyncio
async def test_regenerate_restores_render_inputs_for_prepare_then_predict(
    client,
    shared_pod_a,
    shared_pod_b,
    monkeypatch: pytest.MonkeyPatch,
):
    """A selected prepare producer needs renders, not an old prepared JSONL."""
    _, captured_steps = _install_immediate_pipeline_capture(monkeypatch)

    _switch_to(shared_pod_a)
    created = await client.post("/pipeline", files=_make_pipeline_files())
    assert created.status_code == 202, created.text
    session_id = created.json()["session_id"]
    metadata_path, prims_path, image_path = _write_render_dataset_closure(
        shared_pod_a,
        session_id,
    )
    assert (
        await shared_pod_a.sync_to_store(
            session_id,
            prefix="cache/dataset/",
            overwrite=True,
        )
        == 3
    )
    await _complete_and_release_captured_run(shared_pod_a, session_id)

    _switch_to(shared_pod_b)
    regenerated = await client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["build_dataset_prepare_dataset", "predict"]},
    )
    assert regenerated.status_code == 202, regenerated.text
    pod_a_root = shared_pod_a.get_session_dir(session_id)
    pod_b_root = shared_pod_b.get_session_dir(session_id)
    for source_path in (metadata_path, prims_path, image_path):
        restored_path = pod_b_root / source_path.relative_to(pod_a_root)
        assert restored_path.read_bytes() == source_path.read_bytes()
    assert not (pod_b_root / "cache" / "dataset" / "dataset.jsonl").exists()
    assert captured_steps[-1] == ["build_dataset_prepare_dataset", "predict"]
    await _complete_and_release_captured_run(shared_pod_b, session_id)


@pytest.mark.asyncio
async def test_regenerate_fails_closed_when_dataset_sidecar_is_not_durable(
    client,
    shared_pod_a,
    shared_pod_b,
    monkeypatch: pytest.MonkeyPatch,
):
    """Admission rejects predict when its JSONL references an absent image."""
    _, captured_steps = _install_immediate_pipeline_capture(monkeypatch)

    _switch_to(shared_pod_a)
    created = await client.post("/pipeline", files=_make_pipeline_files())
    assert created.status_code == 202, created.text
    session_id = created.json()["session_id"]
    _write_prepared_dataset_closure(
        shared_pod_a,
        session_id,
        write_image=False,
    )
    assert (
        await shared_pod_a.sync_to_store(
            session_id,
            prefix="cache/dataset/",
            overwrite=True,
        )
        == 1
    )
    await _complete_and_release_captured_run(shared_pod_a, session_id)

    _switch_to(shared_pod_b)
    regenerated = await client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["predict"]},
    )
    assert regenerated.status_code == 409, regenerated.text
    assert "cache/dataset/usd/renders/component.png" in regenerated.text
    assert captured_steps == [None]

    retry_run = "f" * 32
    assert await shared_pod_b.reserve_run(session_id, retry_run)
    assert await shared_pod_b.release_run(session_id, retry_run)


@pytest.mark.asyncio
async def test_slow_cross_instance_restore_keeps_run_claim_live(
    client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A restore longer than the lease cannot be reclaimed on another instance."""
    shared_store = LocalSessionStore(root_dir=str(tmp_path / "slow-shared-store"))
    lease_seconds = 1.0
    heartbeat_seconds = 0.05
    pod_a = SessionManager(
        tmp_path / "slow-pod-a",
        store=shared_store,
        run_claim_lease_seconds=lease_seconds,
        run_claim_heartbeat_seconds=heartbeat_seconds,
    )
    pod_b = SessionManager(
        tmp_path / "slow-pod-b",
        store=shared_store,
        run_claim_lease_seconds=lease_seconds,
        run_claim_heartbeat_seconds=heartbeat_seconds,
    )
    _install_immediate_pipeline_capture(monkeypatch)

    _switch_to(pod_a)
    created = await client.post("/pipeline", files=_make_pipeline_files())
    assert created.status_code == 202, created.text
    session_id = created.json()["session_id"]
    _write_prepared_dataset_closure(pod_a, session_id)
    assert (
        await pod_a.sync_to_store(
            session_id,
            prefix="cache/dataset/",
            overwrite=True,
        )
        == 2
    )
    await _complete_and_release_captured_run(pod_a, session_id)

    execution_started = asyncio.Event()
    execution_release = asyncio.Event()
    executed_steps: list[list[str] | None] = []

    async def blocked_execute(
        session_id: str,
        run_id: str,
        config_dict,
        session_manager: SessionManager,
        only_steps=None,
    ) -> None:
        del config_dict
        executed_steps.append(only_steps)
        execution_started.set()
        await execution_release.wait()
        assert await session_manager.terminalize_and_release_run(
            session_id,
            run_id,
            {"status": "completed"},
        )

    registry = JobRegistry(max_concurrent=1)
    monkeypatch.setattr(pipeline_router, "execute_pipeline_async", blocked_execute)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: registry)

    restore_started = asyncio.Event()
    restore_release = asyncio.Event()
    original_sync_from_store = pod_b.sync_from_store

    async def slow_sync_from_store(
        requested_session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        if prefix == "cache/dataset/" or prefix.endswith("/cache/dataset/"):
            restore_started.set()
            await restore_release.wait()
        return await original_sync_from_store(
            requested_session_id,
            prefix=prefix,
            overwrite=overwrite,
        )

    monkeypatch.setattr(pod_b, "sync_from_store", slow_sync_from_store)
    _switch_to(pod_b)
    regeneration = asyncio.create_task(
        client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )
    )
    await asyncio.wait_for(restore_started.wait(), timeout=2)
    try:
        await asyncio.sleep(lease_seconds * 1.25)
        assert not await pod_a.reserve_run(session_id, "e" * 32)
    finally:
        restore_release.set()

    response = await asyncio.wait_for(regeneration, timeout=2)
    assert response.status_code == 202, response.text
    await asyncio.wait_for(execution_started.wait(), timeout=2)
    await asyncio.sleep(lease_seconds * 1.25)
    assert not await pod_a.reserve_run(session_id, "f" * 32)
    heartbeat_name = f"joint-run-claim-{session_id[:8]}"
    assert (
        sum(
            task.get_name() == heartbeat_name and not task.done()
            for task in asyncio.all_tasks()
        )
        == 1
    )
    assert executed_steps == [["predict"]]

    execution_release.set()
    for _ in range(100):
        if not registry.is_running(session_id):
            break
        await asyncio.sleep(0.01)
    assert not registry.is_running(session_id)
    assert not any(
        task.get_name() == heartbeat_name and not task.done()
        for task in asyncio.all_tasks()
    )
    final_probe = "a" * 32
    assert await pod_a.reserve_run(session_id, final_probe)
    assert await pod_a.release_run(session_id, final_probe)


@pytest.mark.asyncio
async def test_failed_pipeline_persists_dataset_for_cross_instance_regeneration(
    client,
    shared_store,
    shared_pod_a,
    shared_pod_b,
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed real executor run durably publishes usable predict inputs."""
    captured_configs, captured_steps = _install_immediate_pipeline_capture(monkeypatch)

    _switch_to(shared_pod_a)
    created = await client.post("/pipeline", files=_make_pipeline_files())
    assert created.status_code == 202, created.text
    session_id = created.json()["session_id"]
    await _complete_and_release_captured_run(shared_pod_a, session_id)
    assert await shared_store.list_keys(session_id, prefix="cache/dataset/") == []

    failed_run_id = "d" * 32
    assert await shared_pod_a.reserve_run(session_id, failed_run_id)
    assert await shared_pod_a.update_session_for_run(
        session_id,
        failed_run_id,
        {"status": "running"},
    )

    async def fail_after_writing_dataset(_params):
        _write_prepared_dataset_closure(shared_pod_a, session_id)
        return SimpleNamespace(
            success=False,
            error="dataset-stage failure",
            completed_steps=[
                "build_dataset_usd",
                "build_dataset_prepare_dataset",
            ],
        )

    monkeypatch.setattr(executor, "arun_pipeline", fail_after_writing_dataset)
    with pytest.raises(RuntimeError, match="^Pipeline failed$"):
        await executor.execute_pipeline_async(
            session_id,
            failed_run_id,
            captured_configs[0],
            shared_pod_a,
        )

    dataset_keys = sorted(
        await shared_store.list_keys(
            session_id,
            prefix=(f"artifacts/run_cache/{failed_run_id}/cache/dataset/"),
        )
    )
    assert dataset_keys == [
        f"artifacts/run_cache/{failed_run_id}/cache/dataset/dataset.jsonl",
        (
            f"artifacts/run_cache/{failed_run_id}/cache/dataset/"
            "usd/renders/component.png"
        ),
    ]
    failed_metadata = await shared_pod_a.get_session_metadata(session_id)
    assert failed_metadata is not None
    assert failed_metadata["status"] == "failed"
    assert failed_metadata["error"] == "joint_pipeline_execution_failed"
    assert failed_metadata["error_diagnostic"] == {
        "schema": "world-understanding-durable-diagnostic-v1",
        "code": "joint_pipeline_execution_failed",
        "phase": "pipeline_execution",
        "retryable": False,
    }
    assert "dataset-stage failure" not in repr(failed_metadata)
    assert failed_metadata["cache_publications"] == {"dataset": failed_run_id}
    await finalize_pipeline_run(shared_pod_a, session_id, failed_run_id)

    _switch_to(shared_pod_b)
    regenerated = await client.post(
        f"/pipeline/{session_id}/regenerate",
        json={"steps": ["predict"]},
    )
    assert regenerated.status_code == 202, regenerated.text
    restored_image = (
        shared_pod_b.get_session_dir(session_id)
        / "cache/dataset/usd/renders/component.png"
    )
    assert restored_image.read_bytes() == b"rendered-component"
    assert captured_steps[-1] == ["predict"]
    await _complete_and_release_captured_run(shared_pod_b, session_id)


@pytest.mark.asyncio
async def test_sse_returns_503_on_cross_instance(client, shared_pod_a, shared_pod_b):
    """SSE endpoint returns 503 when the session is not running on this instance."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Start pipeline on pod A (gives it a snapshot in the event bus)
    resp = await client.post(
        "/pipeline",
        data={"session_id": session_id, "render_backend": "warp"},
    )
    assert resp.status_code in (200, 201, 202)

    # Pod B has no event bus snapshot for this session — SSE should 503
    _switch_to(shared_pod_b)
    # Also reset the event bus to simulate a truly separate instance
    from ..service.runtime import bus as bus_module

    bus_module._event_bus = None

    resp = await client.get(f"/pipeline/{session_id}/events")
    assert resp.status_code == 503, (
        f"SSE on pod B should return 503 (cross-instance), got {resp.status_code}"
    )


@pytest.mark.asyncio
async def test_polling_works_cross_instance_after_sse_503(
    client, shared_pod_a, shared_pod_b
):
    """After SSE 503, polling /status works correctly from any pod."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]
    resp = await client.post(
        "/pipeline",
        data={"session_id": session_id, "render_backend": "warp"},
    )
    assert resp.status_code in (200, 201, 202)

    # Wait for stub executor to complete
    import asyncio

    await asyncio.sleep(0.2)

    # Poll status from pod B
    _switch_to(shared_pod_b)
    resp = await client.get(f"/pipeline/{session_id}/status")
    assert resp.status_code == 200
    status = resp.json()["status"]
    assert status in ("pending", "running", "completed"), (
        f"Unexpected status from pod B: {status}"
    )


@pytest.mark.asyncio
async def test_remote_regeneration_ignores_stale_terminal_snapshot_for_status_and_sse(
    client,
    shared_pod_a: SessionManager,
    shared_pod_b: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = str(uuid.uuid4())
    await shared_pod_a.create_session(session_id)
    await shared_pod_a.update_session(session_id, {"status": "completed"})

    pod_a_bus = EventBus()
    await pod_a_bus.emit(
        ProgressEvent(
            session_id=session_id,
            step="pipeline",
            state=StepState.COMPLETED,
            percent=100,
            extra={"pipeline_completed": True},
        )
    )
    assert pod_a_bus.get_snapshot(session_id)["status"] == "completed"

    run_id = "f" * 32
    assert await shared_pod_b.reserve_run(session_id, run_id)
    assert await shared_pod_b.update_session_for_run(
        session_id,
        run_id,
        {"status": "pending", "current_step": None},
    )
    pod_b_bus = EventBus()
    await pod_b_bus.seed_pending_session(session_id)

    _switch_to(shared_pod_a)
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: pod_a_bus)
    monkeypatch.setattr(
        pipeline_router,
        "get_job_registry",
        lambda: SimpleNamespace(is_running=lambda _session_id: False),
    )

    status = await client.get(f"/pipeline/{session_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] == "pending"

    events = await client.get(f"/pipeline/{session_id}/events")
    assert events.status_code == 503
    assert pod_a_bus.get_snapshot(session_id)["status"] == "completed"
    assert await shared_pod_b.release_run(session_id, run_id)


@pytest.mark.asyncio
async def test_same_instance_run_reservation_is_atomic(tmp_path) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid.uuid4())
    await manager.create_session(session_id)
    run_ids = ("a" * 32, "b" * 32)

    accepted = await asyncio.gather(
        *(manager.reserve_run(session_id, run_id) for run_id in run_ids)
    )

    assert accepted.count(True) == 1
    winner = run_ids[accepted.index(True)]
    loser = run_ids[accepted.index(False)]
    assert await manager.is_run_current(session_id, winner) is True
    assert (
        await manager.update_session_for_run(session_id, loser, {"status": "completed"})
        is False
    )
    assert await manager.release_run(session_id, winner) is True


@pytest.mark.asyncio
async def test_cross_instance_claim_fences_stale_completion(
    shared_pod_a: SessionManager,
    shared_pod_b: SessionManager,
) -> None:
    session_id = str(uuid.uuid4())
    await shared_pod_a.create_session(session_id)
    run_a = "a" * 32
    run_b = "b" * 32

    accepted_a, accepted_b = await asyncio.gather(
        shared_pod_a.reserve_run(session_id, run_a),
        shared_pod_b.reserve_run(session_id, run_b),
    )
    assert accepted_a != accepted_b
    winner_manager, winner_run, loser_manager, loser_run = (
        (shared_pod_a, run_a, shared_pod_b, run_b)
        if accepted_a
        else (shared_pod_b, run_b, shared_pod_a, run_a)
    )

    assert (
        await loser_manager.update_session_for_run(
            session_id,
            loser_run,
            {"status": "completed", "results": {"bytes": "loser"}},
        )
        is False
    )
    assert await winner_manager.release_run(session_id, winner_run) is True
    assert await loser_manager.reserve_run(session_id, loser_run) is True
    assert (
        await winner_manager.update_session_for_run(
            session_id,
            winner_run,
            {"status": "completed", "results": {"bytes": "stale"}},
        )
        is False
    )
    assert (
        await loser_manager.update_session_for_run(
            session_id,
            loser_run,
            {"status": "completed", "results": {"bytes": "current"}},
        )
        is True
    )
    metadata = await shared_pod_a.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["results"] == {"bytes": "current"}
    assert await loser_manager.release_run(session_id, loser_run) is True


@pytest.mark.asyncio
async def test_expired_cross_instance_claim_is_reclaimed_once(tmp_path) -> None:
    shared_path = tmp_path / "lease-store"
    shared_path.mkdir()
    store = LocalSessionStore(root_dir=str(shared_path))
    managers = [
        SessionManager(
            tmp_path / f"pod-{index}",
            store=store,
            run_claim_lease_seconds=10,
            run_claim_heartbeat_seconds=1,
        )
        for index in range(3)
    ]
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    for manager in managers:
        manager._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    session_id = str(uuid.uuid4())
    await managers[0].create_session(session_id)
    stale_run = "a" * 32
    assert await managers[0].reserve_run(session_id, stale_run)
    assert await managers[0].update_session_for_run(
        session_id,
        stale_run,
        {"status": "running"},
    )

    clock[0] += timedelta(seconds=11)
    assert await managers[0].renew_run(session_id, stale_run) is False
    contenders = ("b" * 32, "c" * 32)
    accepted = await asyncio.gather(
        *(
            manager.reserve_run(session_id, run_id)
            for manager, run_id in zip(managers[1:], contenders, strict=True)
        )
    )

    assert accepted.count(True) == 1
    winner_index = accepted.index(True)
    winner_manager = managers[winner_index + 1]
    winner_run = contenders[winner_index]
    assert await managers[0].is_run_current(session_id, stale_run) is False
    assert (
        await managers[0].update_session_for_run(
            session_id,
            stale_run,
            {"status": "completed", "results": {"owner": "stale"}},
        )
        is False
    )
    assert await winner_manager.is_run_current(session_id, winner_run) is True
    assert (
        await winner_manager.update_session_for_run(
            session_id,
            winner_run,
            {"status": "completed", "results": {"owner": "winner"}},
        )
        is True
    )
    metadata = await managers[0].get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["results"] == {"owner": "winner"}


@pytest.mark.asyncio
async def test_expired_owner_finalizer_terminalizes_before_successor(
    tmp_path,
) -> None:
    store = LocalSessionStore(root_dir=str(tmp_path / "terminal-store"))
    owner = SessionManager(
        tmp_path / "terminal-owner",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    successor = SessionManager(
        tmp_path / "terminal-successor",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    successor._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    session_id = str(uuid.uuid4())
    owner_run = "a" * 32
    successor_run = "b" * 32
    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, owner_run)
    assert await owner.update_session_for_run(
        session_id,
        owner_run,
        {"status": "running"},
    )
    clock[0] += timedelta(seconds=11)

    await finalize_pipeline_run(owner, session_id, owner_run)
    metadata = await owner.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "cancelled"
    assert metadata["error"] == "Pipeline run was cancelled before completion"
    assert "active_run_id" not in metadata

    assert await successor.reserve_run(session_id, successor_run)
    assert not await owner.terminalize_and_release_run(
        session_id,
        owner_run,
        {"status": "failed", "error": "stale owner"},
    )
    metadata = await owner.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["active_run_id"] == successor_run
    assert metadata["status"] == "cancelled"
    assert await successor.release_run(session_id, successor_run)


@pytest.mark.asyncio
async def test_reclaimed_run_fences_metadata_writer_already_in_flight(tmp_path) -> None:
    class PausingMetadataStore(LocalSessionStore):
        def __init__(self, root_dir: str) -> None:
            super().__init__(root_dir)
            self.pause_next_metadata_cas = False
            self.metadata_cas_started = asyncio.Event()
            self.resume_metadata_cas = asyncio.Event()

        async def compare_and_swap_bytes(
            self,
            session_id: str,
            key: str,
            expected: bytes,
            replacement: bytes | None,
            content_type: str | None = None,
        ) -> bool:
            if key == "session.json" and self.pause_next_metadata_cas:
                self.pause_next_metadata_cas = False
                self.metadata_cas_started.set()
                await self.resume_metadata_cas.wait()
            return await super().compare_and_swap_bytes(
                session_id,
                key,
                expected,
                replacement,
                content_type,
            )

    store = PausingMetadataStore(str(tmp_path / "metadata-cas-store"))
    owner = SessionManager(
        tmp_path / "metadata-owner",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    replacement = SessionManager(
        tmp_path / "metadata-replacement",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    replacement._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    session_id = str(uuid.uuid4())
    owner_run = "a" * 32
    replacement_run = "b" * 32
    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, owner_run)

    store.pause_next_metadata_cas = True
    stale_update = asyncio.create_task(
        owner.update_session_for_run(
            session_id,
            owner_run,
            {"status": "completed", "results": {"owner": "stale"}},
        )
    )
    await store.metadata_cas_started.wait()

    clock[0] += timedelta(seconds=11)
    assert await replacement.reserve_run(session_id, replacement_run)
    store.resume_metadata_cas.set()
    assert await stale_update is False
    assert await replacement.update_session_for_run(
        session_id,
        replacement_run,
        {"status": "completed", "results": {"owner": "replacement"}},
    )

    metadata = await replacement.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["results"] == {"owner": "replacement"}
    assert await replacement.release_run(session_id, replacement_run)


@pytest.mark.asyncio
async def test_renewed_cross_instance_claim_cannot_be_reclaimed(tmp_path) -> None:
    shared_path = tmp_path / "renew-store"
    shared_path.mkdir()
    store = LocalSessionStore(root_dir=str(shared_path))
    owner = SessionManager(
        tmp_path / "owner",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    contender = SessionManager(
        tmp_path / "contender",
        store=store,
        run_claim_lease_seconds=10,
        run_claim_heartbeat_seconds=1,
    )
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    owner._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    contender._claim_now = lambda: clock[0]  # type: ignore[method-assign]
    session_id = str(uuid.uuid4())
    run_id = "a" * 32
    await owner.create_session(session_id)
    assert await owner.reserve_run(session_id, run_id)

    clock[0] += timedelta(seconds=6)
    assert await owner.renew_run(session_id, run_id)
    clock[0] += timedelta(seconds=6)

    assert await contender.reserve_run(session_id, "b" * 32) is False
    assert await owner.is_run_current(session_id, run_id) is True
    assert await owner.release_run(session_id, run_id) is True


@pytest.mark.asyncio
async def test_cross_instance_pipeline_rejects_same_session_overlap(
    client,
    shared_pod_a: SessionManager,
    shared_pod_b: SessionManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_STEP_DELAY", "0.2")
    _switch_to(shared_pod_a)
    upload = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = upload.json()["session_id"]
    first = await client.post("/pipeline", data={"session_id": session_id})
    assert first.status_code == 202

    _switch_to(shared_pod_b)
    overlapping = await client.post("/pipeline", data={"session_id": session_id})
    assert overlapping.status_code == 409
    assert "already active" in overlapping.json()["detail"]
