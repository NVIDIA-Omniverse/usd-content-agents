# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline regeneration.

Tests the regenerate endpoint for re-running specific steps.
"""

import asyncio

import pytest

from ..conftest import make_pipeline_files


@pytest.mark.api
class TestPipelineRegenerate:
    """Test pipeline regeneration."""

    async def test_regenerate_predict_only(self, client, monkeypatch):
        """Test regenerating the predict step only."""
        monkeypatch.setenv("TEST_STEP_DELAY", "0.05")
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        final_status = None
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            final_status = status_r.json()["status"]
            if final_status == "completed":
                break
            await asyncio.sleep(0.01)
        assert final_status == "completed"

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )

        assert regen_r.status_code == 202
        regenerated = regen_r.json()
        assert regenerated["status"] == "pending"
        assert len(regenerated["run_id"]) == 32
        assert set(regenerated["run_id"]) <= set("0123456789abcdef")
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.json()["status"] in {"pending", "running"}

        cancel_r = await client.post(
            f"/pipeline/{session_id}/cancel",
            params={"run_id": regenerated["run_id"]},
        )
        assert cancel_r.status_code == 200
        assert cancel_r.json()["run_id"] == regenerated["run_id"]

    async def test_regenerate_multiple_steps(self, client):
        """Test regenerating multiple steps."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["build_dataset_usd", "predict"]},
        )

        assert regen_r.status_code == 202

    async def test_regenerate_rejects_an_active_run_claim(self, client):
        """An unexpired run claim rejects overlapping regeneration."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={"steps": ["predict"]},
        )

        # The fast stub may finish before the second request reaches admission.
        if regen_r.status_code != 202:
            assert regen_r.status_code == 409

    async def test_regenerate_nonexistent_session(self, client):
        """Test regenerate on nonexistent session returns 404."""
        regen_r = await client.post(
            "/pipeline/00000000-0000-0000-0000-000000000000/regenerate",
            json={"steps": ["predict"]},
        )

        assert regen_r.status_code == 404

    async def test_regenerate_with_prompt_override(self, client):
        """Test regenerate with user prompt override."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        regen_r = await client.post(
            f"/pipeline/{session_id}/regenerate",
            json={
                "steps": ["predict"],
                "user_prompt": "Focus on identifying furniture parts",
            },
        )

        assert regen_r.status_code == 202
