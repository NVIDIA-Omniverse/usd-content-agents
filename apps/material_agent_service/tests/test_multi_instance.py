# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-instance simulation tests for Material Agent Service.

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

import json
from pathlib import Path

import pytest

from ..service.routers import (
    artifacts_router,
    assets_router,
    pipeline_router,
    sessions_router,
)
from ..service.session.manager import SessionManager
from ..service.storage.local_store import LocalSessionStore

MINIMAL_USD = b'#usda 1.0\ndef Xform "Root" {}\n'


def _make_pipeline_files():
    return [("usd_file", ("cube.usda", MINIMAL_USD, "application/octet-stream"))]


def _switch_to(mgr: SessionManager):
    """Swap the global session manager -- simulates request hitting a different pod."""
    pipeline_router.set_session_manager(mgr)
    artifacts_router.set_session_manager(mgr)
    assets_router.set_session_manager(mgr)
    sessions_router.set_session_manager(mgr)


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
    """Shared store simulating S3 -- both pods point to the same backend."""
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
# PART 1: Separate storage -- proves the problem
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

    list_a = sessions_a.get("sessions", sessions_a)
    list_b = sessions_b.get("sessions", sessions_b)

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
    resp_b = await client.post(f"/pipeline/{session_id}/cancel")
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
        json.dumps({"id": "/Root", "material": "Aluminum", "confidence": 0.95}) + "\n"
    )
    await pod_a.update_session(session_id, {"status": "completed"})

    resp_a = await client.get(f"/artifacts/{session_id}/predictions")
    assert resp_a.status_code == 200

    _switch_to(pod_b)
    resp_b = await client.get(f"/artifacts/{session_id}/predictions")
    assert resp_b.status_code in (404, 500)


# ===========================================================================
# PART 2: Shared storage -- proves the fix
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
    await shared_pod_a.update_session(session_id, {"status": "running"})

    # Cancel from pod B
    _switch_to(shared_pod_b)
    resp_b = await client.post(f"/pipeline/{session_id}/cancel")
    assert resp_b.status_code == 200, f"Cancel should succeed: {resp_b.text}"

    # Verify cancel signal is visible from pod A
    _switch_to(shared_pod_a)
    is_cancelled = await shared_pod_a.is_cancelled(session_id)
    assert is_cancelled, "Cancel signal should be visible to pod A"


@pytest.mark.asyncio
async def test_shared_artifacts_downloadable_from_any_instance(
    client, shared_pod_a, shared_pod_b, shared_store
):
    """Artifacts synced to shared store are downloadable from any pod."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    # Write predictions directly to the shared store (simulates sync_to_store)
    pred_data = (
        json.dumps({"id": "/Root", "material": "Aluminum", "confidence": 0.95}) + "\n"
    )
    await shared_store.put_bytes(
        session_id,
        "cache/predictions/predictions.jsonl",
        pred_data.encode(),
        "application/x-ndjson",
    )
    await shared_pod_a.update_session(session_id, {"status": "completed"})

    # Pod B can download (reads from shared store)
    _switch_to(shared_pod_b)
    resp_b = await client.get(f"/artifacts/{session_id}/predictions")
    assert resp_b.status_code == 200, (
        f"Pod B should serve shared artifacts, got {resp_b.status_code}: {resp_b.text}"
    )


@pytest.mark.asyncio
async def test_prediction_report_hydrates_inputs_from_shared_store(
    client, shared_pod_a, shared_pod_b, shared_store, monkeypatch
):
    """On-demand report generation must fetch prediction inputs from shared store."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    session_id = resp.json()["session_id"]

    pred_data = json.dumps({"id": "/Root", "material": "Aluminum"}) + "\n"
    dataset_data = json.dumps({"prim_path": "/Root", "renders": []}) + "\n"
    await shared_store.put_bytes(
        session_id,
        "cache/predictions/predictions.jsonl",
        pred_data.encode(),
        "application/x-ndjson",
    )
    await shared_store.put_bytes(
        session_id,
        "cache/dataset/dataset.jsonl",
        dataset_data.encode(),
        "application/x-ndjson",
    )
    generation_calls = 0

    async def fake_generate_report(
        session_dir: Path,
        predictions_path: Path,
        dataset_path: Path,
        prediction_lineage: str,
    ) -> Path:
        nonlocal generation_calls
        generation_calls += 1
        assert predictions_path.exists()
        assert dataset_path.exists()
        assert prediction_lineage
        report_path = (
            session_dir
            / "cache"
            / "predictions"
            / ".report-builds"
            / prediction_lineage
            / "prediction_report.html"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("<html>hydrated report</html>")
        return report_path

    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        fake_generate_report,
    )

    _switch_to(shared_pod_b)
    resp_b = await client.get(f"/artifacts/{session_id}/report")
    assert resp_b.status_code == 200, resp_b.text
    assert "hydrated report" in resp_b.text
    assert generation_calls == 1
    metadata = await shared_pod_a.get_session_metadata(session_id)
    assert metadata is not None
    publication = metadata["prediction_report_publication"]
    assert publication["key"].startswith(
        f"reports/{metadata['prediction_lineage_token']}/"
    )
    assert await shared_store.exists(session_id, publication["key"])
    assert not await shared_store.exists(
        session_id,
        "cache/predictions/prediction_report.html",
    )

    # A second request on a different pod must resolve the immutable pointer;
    # the lineage-specific staging directory on pod B has already been removed.
    _switch_to(shared_pod_a)
    resp_a = await client.get(f"/artifacts/{session_id}/report")
    assert resp_a.status_code == 200, resp_a.text
    assert "hydrated report" in resp_a.text
    assert generation_calls == 1


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
    assert any("scene" in k or "cube" in k for k in input_keys), (
        f"Input file should be in shared store after upload, got keys: {input_keys}"
    )

    # Switch to pod B (separate local dir, same shared store) and start pipeline
    _switch_to(shared_pod_b)
    resp = await client.post(
        "/pipeline",
        data={
            "session_id": session_id,
            "render_backend": "warp",
            "user_email": "test@test.com",
        },
    )
    assert resp.status_code in (200, 201, 202), (
        f"Pod B should be able to start pipeline after pulling input from store: {resp.text}"
    )


@pytest.mark.asyncio
async def test_sse_returns_503_on_cross_instance(client, shared_pod_a, shared_pod_b):
    """SSE endpoint returns 503 when the session is running on a different instance."""
    _switch_to(shared_pod_a)
    resp = await client.post("/pipeline/upload-usd", files=_make_pipeline_files())
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # Simulate a pipeline running on pod A by setting status directly
    # (avoids race with stub executor completing before we can test)
    await shared_pod_a.update_session(session_id, {"status": "running"})

    # Pod B has no event bus snapshot for this session -- SSE should 503
    _switch_to(shared_pod_b)
    # Reset the event bus to simulate a truly separate instance
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
        data={
            "session_id": session_id,
            "render_backend": "warp",
            "user_email": "test@test.com",
        },
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
