# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline cancellation.

Tests the cancel endpoint and cancellation semantics.
"""

import asyncio

import pytest


def _install_blocked_executor(monkeypatch):
    """Install an executor that cannot finish before a test cancels it."""
    from ...service.routers import pipeline_router

    started = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def blocked_execute(session_id, session_manager, **_kwargs) -> None:
        await session_manager.update_session(session_id, {"status": "running"})
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await session_manager.update_session(session_id, {"status": "cancelled"})
            raise
        finally:
            stopped.set()

    monkeypatch.setattr(
        pipeline_router,
        "execute_pipeline_async",
        blocked_execute,
        raising=True,
    )
    return started, release, stopped


@pytest.mark.api
class TestPipelineCancel:
    """Test pipeline cancellation."""

    async def test_cancel_running_pipeline(self, client, monkeypatch):
        """Test cancelling a running pipeline."""
        started, release, stopped = _install_blocked_executor(monkeypatch)
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        try:
            await asyncio.wait_for(started.wait(), timeout=5.0)
            cancel_r = await client.post(f"/pipeline/{session_id}/cancel")

            assert cancel_r.status_code == 200
            assert cancel_r.json()["status"] == "cancelling"
        finally:
            release.set()
            await asyncio.wait_for(stopped.wait(), timeout=5.0)

    async def test_cancel_returns_400_for_completed(self, client):
        """Test that cancelling a completed pipeline returns 400."""
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

        # Try to cancel completed pipeline
        cancel_r = await client.post(f"/pipeline/{session_id}/cancel")

        assert cancel_r.status_code == 400
        assert "Cannot cancel" in cancel_r.json()["detail"]

    async def test_cancel_returns_404_for_nonexistent(self, client):
        """Test that cancelling nonexistent session returns 404."""
        cancel_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/cancel"
        )

        assert cancel_r.status_code == 404

    async def test_cancelled_pipeline_stops_processing(self, client, monkeypatch):
        """Test that cancelled pipeline stops processing."""
        started, release, stopped = _install_blocked_executor(monkeypatch)
        usd_content = b"#usda 1.0\n"
        files = {"usd_file": ("scene.usda", usd_content, "application/octet-stream")}
        create_r = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_r.json()["session_id"]

        try:
            await asyncio.wait_for(started.wait(), timeout=5.0)
            cancel_r = await client.post(f"/pipeline/{session_id}/cancel")
            await asyncio.wait_for(stopped.wait(), timeout=5.0)

            assert cancel_r.status_code == 200
            assert stopped.is_set()
            assert not release.is_set()

            status_r = await client.get(f"/pipeline/{session_id}/status")
            assert status_r.json()["status"] == "cancelled"

        finally:
            release.set()
            await asyncio.wait_for(stopped.wait(), timeout=5.0)
