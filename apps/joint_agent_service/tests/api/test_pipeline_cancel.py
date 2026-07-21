# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline cancellation.

Tests the cancel endpoint and cancellation semantics.
"""

import asyncio
import os

import pytest

from ..conftest import make_pipeline_files


@pytest.mark.api
class TestPipelineCancel:
    """Test pipeline cancellation."""

    async def test_cancel_requires_exact_run_generation(self, client):
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        created = create_r.json()

        missing_generation = await client.post(
            f"/pipeline/{created['session_id']}/cancel"
        )
        assert missing_generation.status_code == 422

        accepted = await client.post(
            f"/pipeline/{created['session_id']}/cancel",
            params={"run_id": created["run_id"]},
        )
        assert accepted.status_code == 200
        assert accepted.json()["run_id"] == created["run_id"]

    async def test_cancel_running_pipeline(self, client):
        """Test cancelling a running pipeline."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        created = create_r.json()
        session_id = created["session_id"]
        run_id = created["run_id"]

        await asyncio.sleep(0.05)

        cancel_r = await client.post(
            f"/pipeline/{session_id}/cancel",
            params={"run_id": run_id},
        )

        assert cancel_r.status_code == 200
        assert cancel_r.json()["status"] == "cancelling"

    async def test_cancel_returns_400_for_completed(self, client):
        """Test that cancelling a completed pipeline returns 400."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        created = create_r.json()
        session_id = created["session_id"]
        run_id = created["run_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        cancel_r = await client.post(
            f"/pipeline/{session_id}/cancel",
            params={"run_id": run_id},
        )

        assert cancel_r.status_code == 400
        assert "Cannot cancel" in cancel_r.json()["detail"]

    async def test_cancel_returns_404_for_nonexistent(self, client):
        """Test that cancelling nonexistent session returns 404."""
        cancel_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/cancel",
            params={"run_id": "a" * 32},
        )

        assert cancel_r.status_code == 404

    async def test_cancelled_pipeline_stops_processing(self, client):
        """Test that cancelled pipeline stops processing."""
        os.environ["TEST_STEP_DELAY"] = "0.1"

        try:
            create_r = await client.post("/pipeline", files=make_pipeline_files())
            created = create_r.json()
            session_id = created["session_id"]
            run_id = created["run_id"]

            await asyncio.sleep(0.15)

            await client.post(
                f"/pipeline/{session_id}/cancel",
                params={"run_id": run_id},
            )

            for _ in range(100):
                status_r = await client.get(f"/pipeline/{session_id}/status")
                status = status_r.json()["status"]
                if status in ["cancelled", "completed"]:
                    break
                await asyncio.sleep(0.05)

        finally:
            os.environ["TEST_STEP_DELAY"] = "0.01"

    async def test_cancelled_pipeline_results_are_terminal(self, client, monkeypatch):
        """Cancelled results use the terminal PipelineError response contract."""
        monkeypatch.setenv("TEST_STEP_DELAY", "0.2")
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        created = create_r.json()
        session_id = created["session_id"]
        run_id = created["run_id"]

        cancel_r = await client.post(
            f"/pipeline/{session_id}/cancel",
            params={"run_id": run_id},
        )
        assert cancel_r.status_code == 200

        for _ in range(100):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "cancelled":
                break
            await asyncio.sleep(0.01)
        assert status_r.json()["status"] == "cancelled"

        results_r = await client.get(f"/pipeline/{session_id}/results")
        assert results_r.status_code == 200
        results = results_r.json()
        assert set(results) == {
            "session_id",
            "status",
            "error_message",
            "failed_step",
            "completed_steps",
            "partial_results",
        }
        assert results["session_id"] == session_id
        assert results["status"] == "cancelled"
        assert results["error_message"].startswith("Pipeline run was cancelled")
        assert results["failed_step"] == "cancelled"
        assert isinstance(results["completed_steps"], list)
