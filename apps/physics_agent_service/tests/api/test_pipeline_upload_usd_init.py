# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for persisted upload-session initialization.

``POST /pipeline/upload-usd`` previously created a session, wrote the
uploaded USD to disk, and returned ``status: "ready"`` in the HTTP
response, but never advanced the persisted session metadata past the
default ``status: "pending", config: {}`` written by
``manager.create_session``. Operators inspecting ``GET /sessions`` could
not tell an upload-only session from an empty placeholder. The fix
calls ``manager.update_session`` after the upload succeeds so persisted
state matches what the HTTP response advertises.
"""

import pytest

from ..conftest import make_pipeline_files


@pytest.mark.api
class TestUploadUsdInit:
    """Verify upload-usd persists status=ready and a populated config."""

    async def test_upload_usd_file_persists_ready_status(self, client):
        """Status reaches `ready` after a successful file upload."""
        upload_r = await client.post(
            "/pipeline/upload-usd", files=make_pipeline_files()
        )
        assert upload_r.status_code == 201, upload_r.text
        sid = upload_r.json()["session_id"]
        assert upload_r.json()["status"] == "ready"

        get_r = await client.get(f"/sessions/{sid}")
        assert get_r.status_code == 200, get_r.text
        body = get_r.json()
        # The persisted session must agree with the HTTP "ready" response.
        # Before the fix this was "pending".
        assert body["status"] == "ready"

    async def test_upload_usd_file_persists_populated_config(self, client):
        """``config`` records the uploaded artifact, not the empty default."""
        upload_r = await client.post(
            "/pipeline/upload-usd",
            files=make_pipeline_files(
                usd_content=b"#usda 1.0\n# pretend payload\n",
                usd_filename="ladder.usda",
            ),
        )
        assert upload_r.status_code == 201, upload_r.text
        sid = upload_r.json()["session_id"]

        get_r = await client.get(f"/sessions/{sid}")
        body = get_r.json()
        cfg = body.get("config")
        assert cfg, f"config was empty after upload: {body}"
        assert cfg["has_usd_upload"] is True
        assert cfg["usd_path"].endswith(".usda")
        assert cfg["s3_uri"] is None
        assert cfg["original_filename"] == "ladder.usda"
        assert isinstance(cfg["size_mb"], int | float)

    async def test_upload_usd_listed_sessions_show_ready(self, client):
        """An upload-only session shows ``ready`` in the listing too.

        This is the operator-facing surface the bug report called out: an
        upload-only session and a placeholder pending session must be
        distinguishable from ``GET /sessions`` alone.
        """
        upload_r = await client.post(
            "/pipeline/upload-usd", files=make_pipeline_files()
        )
        sid = upload_r.json()["session_id"]

        list_r = await client.get("/sessions")
        assert list_r.status_code == 200, list_r.text
        sessions = {s["session_id"]: s for s in list_r.json()["sessions"]}
        assert sid in sessions
        assert sessions[sid]["status"] == "ready"

    async def test_upload_usd_rejects_missing_payload(self, client):
        """Regression guard: empty multipart still rejected with 400.

        The bug noted the Swagger form makes it easy to leave both
        ``usd_file`` and ``s3_uri`` blank. The handler must continue to
        return 400 in that case (not 422; the existing check predates the
        4xx-hygiene work in MR !415).
        """
        r = await client.post("/pipeline/upload-usd")
        assert r.status_code == 400, r.text
        assert "usd_file or s3_uri" in r.json()["detail"]

    async def test_upload_usd_rejects_both_payloads(self, client):
        """Providing both ``usd_file`` and ``s3_uri`` is also a 400."""
        r = await client.post(
            "/pipeline/upload-usd",
            files=make_pipeline_files(),
            data={"s3_uri": "s3://bucket/key.usd"},
        )
        assert r.status_code == 400, r.text

    async def test_upload_usd_shared_store_sync_failure_does_not_leak_ready(
        self, client, monkeypatch: pytest.MonkeyPatch
    ):
        """If shared-store sync fails, the session must NOT advertise ready.

        In multi-instance deploys the durable session metadata is replicated
        but input artifacts only reach other replicas via sync_to_store.
        Marking the session ready before sync would let a follow-up
        POST /pipeline routed to another instance see ``status: "ready"``
        and then 400 with ``"Input USD not found for session"``. Sync runs
        first; on failure the session is cleaned up rather than left as a
        phantom ready entry.
        """
        from ...service.routers import pipeline_router

        # Snapshot the listing before the failure attempt so we can prove
        # the failed upload did not add a new session. (The class-scoped
        # client/manager fixtures preserve sessions from earlier tests in
        # this class, so a global "no ready sessions exist" assertion
        # would always fail.)
        before_r = await client.get("/sessions")
        before_ids = {s["session_id"] for s in before_r.json()["sessions"]}

        async def _failing_sync(session_id: str) -> int:
            raise RuntimeError("simulated shared-store outage")

        manager = pipeline_router.get_session_manager()
        monkeypatch.setattr(manager, "sync_to_store", _failing_sync)

        upload_r = await client.post(
            "/pipeline/upload-usd", files=make_pipeline_files()
        )
        # 5xx -- exact code is not contractually pinned here; the load-bearing
        # invariant is "no phantom ready session left behind".
        assert 500 <= upload_r.status_code < 600, upload_r.text

        after_r = await client.get("/sessions")
        after_ids = {s["session_id"] for s in after_r.json()["sessions"]}
        new_ids = after_ids - before_ids
        assert not new_ids, f"phantom session survived sync failure: {new_ids}"

    async def test_start_pipeline_resets_uploaded_session_to_pending(
        self, client, monkeypatch: pytest.MonkeyPatch
    ):
        """Starting a pipeline against an upload-only ``ready`` session must
        flip the persisted status to ``pending`` *before* the executor is
        registered.

        Otherwise a saturated semaphore can leave the session showing
        ``ready`` for the entire queue wait -- and the cancel handler
        would reject (it only accepts ``pending``/``running``). Codex
        adversarial-review finding on round 2 of !421.

        Stub the executor so it blocks indefinitely; this freezes the
        post-register state at "queued, not yet running" and lets us
        verify status is ``pending`` (not ``ready``) and that cancel is
        accepted.
        """
        import asyncio

        from ...service.routers import pipeline_router

        upload_r = await client.post(
            "/pipeline/upload-usd",
            files=make_pipeline_files(usd_filename="my-asset.usda"),
        )
        assert upload_r.status_code == 201
        sid = upload_r.json()["session_id"]
        # Sanity: the upload itself persisted status=ready.
        pre = (await client.get(f"/sessions/{sid}")).json()
        assert pre["status"] == "ready"
        pre_cfg = pre["config"]
        assert pre_cfg["original_filename"] == "my-asset.usda"
        assert pre_cfg["size_mb"] is not None

        block = asyncio.Event()

        async def _hanging_executor(
            session_id: str, config_dict: dict, session_manager
        ):
            # Don't advance to running -- park the task in the registry so
            # the post-register state is observable.
            await block.wait()

        monkeypatch.setattr(
            pipeline_router, "execute_pipeline_async", _hanging_executor
        )

        try:
            start_r = await client.post("/pipeline", data={"session_id": sid})
            assert start_r.status_code == 202, start_r.text

            get_r = await client.get(f"/sessions/{sid}")
            body = get_r.json()
            assert body["status"] == "pending", (
                f"upload-then-start left persisted status at {body['status']!r} "
                f"-- cancel would reject and /status would lie. Body: {body}"
            )
            assert body.get("can_cancel") is True

            # Upload-only metadata must survive the pipeline-start
            # update_session. Without the merge in create_pipeline these
            # fields would silently disappear from /sessions and
            # /sessions/{sid} the moment the user starts the run.
            cfg = body["config"]
            assert cfg.get("original_filename") == pre_cfg["original_filename"], (
                f"original_filename was clobbered by pipeline start: {cfg}"
            )
            assert cfg.get("size_mb") == pre_cfg["size_mb"], (
                f"size_mb was clobbered by pipeline start: {cfg}"
            )
            # has_usd_upload must remain True even though usd_file is None
            # on the start-from-existing-session path.
            assert cfg.get("has_usd_upload") is True, (
                f"has_usd_upload flipped to False on start: {cfg}"
            )

            cancel_r = await client.post(f"/pipeline/{sid}/cancel")
            assert cancel_r.status_code in (200, 202), cancel_r.text
        finally:
            block.set()

    async def test_upload_usd_s3_records_object_key_basename(
        self, client, monkeypatch: pytest.MonkeyPatch
    ):
        """The S3 branch must record the S3 object key basename, not the
        normalized local filename ``_download_s3_to_session`` writes.

        ``original_filename`` exists so operators can recognize uploads
        in ``GET /sessions``. Storing ``scene.usdz`` for every S3 upload
        makes that field useless. CodeRabbit MR-comment finding on !421.
        """
        from pathlib import Path

        from ...service.routers import pipeline_router

        monkeypatch.setattr(
            pipeline_router.config,
            "s3_allowed_buckets",
            "example-bucket",
        )

        def _fake_download(s3_uri: str, session_dir: Path) -> Path:
            ext = Path(s3_uri).suffix.lower() or ".usd"
            local_path = session_dir / "input" / f"scene{ext}"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"#usda 1.0\n")
            return local_path

        monkeypatch.setattr(pipeline_router, "_download_s3_to_session", _fake_download)

        upload_r = await client.post(
            "/pipeline/upload-usd",
            data={"s3_uri": "s3://example-bucket/path/Astronaut.usdz"},
        )
        assert upload_r.status_code == 201, upload_r.text
        sid = upload_r.json()["session_id"]

        body = (await client.get(f"/sessions/{sid}")).json()
        cfg = body["config"]
        assert cfg["original_filename"] == "Astronaut.usdz", (
            f"S3 branch recorded local filename instead of S3 object basename: "
            f"{cfg['original_filename']!r}"
        )
        assert cfg["s3_uri"] == "s3://example-bucket/path/Astronaut.usdz"
        assert cfg["has_usd_upload"] is True
