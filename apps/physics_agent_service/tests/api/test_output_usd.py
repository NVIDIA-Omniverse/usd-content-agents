# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /artifacts/{session_id}/output-usd.

Covers the service endpoint that serves the apply_physics step's
simulation-ready USD.
"""

import asyncio
import zipfile
from io import BytesIO
from pathlib import PurePosixPath

import httpx
import pytest

from ..conftest import make_pipeline_files


async def _wait_for_completion(client: httpx.AsyncClient, session_id: str) -> None:
    """Poll status until the pipeline is completed; fail if it times out."""
    for _ in range(200):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        if status_r.json()["status"] == "completed":
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"Pipeline for {session_id} did not reach 'completed' in time")


@pytest.mark.api
class TestOutputUsdDownload:
    """Test the output-usd artifact endpoint."""

    async def test_download_returns_file_after_completion(
        self, client: httpx.AsyncClient
    ) -> None:
        """Once the pipeline completes, the USD file is served on GET."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await _wait_for_completion(client, session_id)

        r = await client.get(f"/artifacts/{session_id}/output-usd")

        assert r.status_code == 200
        # Stub writes a minimal `#usda 1.0` header.
        assert r.content.startswith(b"#usda")
        content_disposition = r.headers.get("content-disposition", "")
        assert "scene_physics.usda" in content_disposition

    async def test_download_defaults_usdz_input_to_usda_output(
        self, client: httpx.AsyncClient
    ) -> None:
        """USDZ input produces USDA output by default."""
        create_r = await client.post(
            "/pipeline",
            files=make_pipeline_files(usd_content=b"stub", usd_filename="scene.usdz"),
        )
        assert create_r.status_code == 202
        payload = create_r.json()
        assert "session_id" in payload
        session_id = payload["session_id"]

        await _wait_for_completion(client, session_id)

        r = await client.get(f"/artifacts/{session_id}/output-usd")

        assert r.status_code == 200
        assert r.content.startswith(b"#usda")
        content_disposition = r.headers.get("content-disposition", "")
        assert "scene_physics.usda" in content_disposition
        assert r.headers["content-type"].startswith("text/plain")

    async def test_download_bundles_usdz_sidecar_assets(
        self, client: httpx.AsyncClient
    ) -> None:
        """USDZ-derived USDA outputs with sidecar assets download as a bundle."""
        create_r = await client.post(
            "/pipeline",
            files=make_pipeline_files(usd_content=b"stub", usd_filename="scene.usdz"),
        )
        assert create_r.status_code == 202
        session_id = create_r.json()["session_id"]

        await _wait_for_completion(client, session_id)

        from ...service.routers import artifacts_router

        manager = artifacts_router.get_session_manager()
        output_path = (
            manager.get_session_dir(session_id)
            / "cache"
            / "physics"
            / "scene_physics.usda"
        )
        output_path.write_text(
            '#usda 1.0\n\ndef "Root"\n{\n'
            "    asset texture = "
            "@scene_physics.usda_assets/Textures/diffuse.png@\n"
            "}\n",
            encoding="utf-8",
        )
        texture_path = (
            output_path.parent
            / "scene_physics.usda_assets"
            / "Textures"
            / "diffuse.png"
        )
        texture_path.parent.mkdir(parents=True, exist_ok=True)
        texture_path.write_bytes(b"texture-bytes")

        r = await client.get(f"/artifacts/{session_id}/output-usd")

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/zip")
        content_disposition = r.headers.get("content-disposition", "")
        assert "scene_physics_bundle.zip" in content_disposition

        with zipfile.ZipFile(BytesIO(r.content)) as archive:
            names = set(archive.namelist())
            assert "scene_physics.usda" in names
            assert "scene_physics.usda_assets/Textures/diffuse.png" in names
            assert (
                archive.read("scene_physics.usda_assets/Textures/diffuse.png")
                == b"texture-bytes"
            )

    def test_store_sidecar_archive_name_rejects_unsafe_paths(self) -> None:
        """Store keys cannot create traversal or Windows-absolute ZIP entries."""
        from ...service.routers.artifacts_router import (
            _archive_name_for_store_sidecar,
        )

        output_key = "cache/physics/scene_physics.usda"

        assert (
            _archive_name_for_store_sidecar(
                output_key,
                "cache/physics/scene_physics.usda_assets/Textures/diffuse.png",
            )
            == "scene_physics.usda_assets/Textures/diffuse.png"
        )
        assert (
            _archive_name_for_store_sidecar(
                output_key,
                "cache/physics/scene_physics_assets/Textures/diffuse.png",
            )
            == "scene_physics_assets/Textures/diffuse.png"
        )

        for sidecar_key in (
            "cache/physics/scene_physics.usda_assets/../evil.txt",
            "cache/physics/scene_physics.usda_assets/C:/evil.txt",
            "cache/physics/scene_physics.usda_assets/Textures\\evil.txt",
        ):
            with pytest.raises(ValueError):
                _archive_name_for_store_sidecar(output_key, sidecar_key)

    def test_local_bundle_uses_safe_archive_names(self, tmp_path) -> None:
        """Local sidecar bundles contain only relative, traversal-free names."""
        from ...service.routers.artifacts_router import (
            _cleanup_temp_file,
            _write_local_output_usd_bundle,
        )

        output_path = tmp_path / "scene_physics.usda"
        output_path.write_text(
            '#usda 1.0\n\ndef "Root"\n{\n'
            "    asset texture = "
            "@scene_physics.usda_assets/Textures/diffuse.png@\n"
            "}\n",
            encoding="utf-8",
        )
        texture_path = (
            tmp_path / "scene_physics.usda_assets" / "Textures" / "diffuse.png"
        )
        texture_path.parent.mkdir(parents=True)
        texture_path.write_bytes(b"texture")

        zip_path = _write_local_output_usd_bundle(output_path)
        assert zip_path is not None
        try:
            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
                assert names == [
                    "scene_physics.usda",
                    "scene_physics.usda_assets/Textures/diffuse.png",
                ]
                for name in names:
                    path = PurePosixPath(name)
                    assert not path.is_absolute()
                    assert ".." not in path.parts
                    assert all(
                        "\\" not in part and ":" not in part for part in path.parts
                    )
        finally:
            _cleanup_temp_file(zip_path)

    async def test_returns_404_for_nonexistent_session(
        self, client: httpx.AsyncClient
    ) -> None:
        """Unknown session → 404."""
        r = await client.get(
            "/artifacts/00000000-0000-4000-8000-000000000000/output-usd"
        )
        assert r.status_code == 404

    async def test_results_response_includes_output_usd_link(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /pipeline/{id}/results advertises the output-usd download URL."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await _wait_for_completion(client, session_id)

        r = await client.get(f"/pipeline/{session_id}/results")

        assert r.status_code == 200
        download_urls = r.json()["download_urls"]
        assert "output_usd" in download_urls
        assert download_urls["output_usd"] == f"/artifacts/{session_id}/output-usd"
