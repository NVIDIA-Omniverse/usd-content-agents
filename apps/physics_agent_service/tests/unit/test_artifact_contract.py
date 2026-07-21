# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ...service.artifact_contract import (
    artifact_name_from_key,
    available_artifact_keys,
    collect_artifact_manifest,
    collect_public_artifact_manifest,
    is_safe_artifact_name,
)
from ...service.routers import refine_router, tune_router
from ...service.session.manager import SessionManager


def test_collect_manifest_and_path_validation(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "tune" / "nested").mkdir(parents=True)
    (session_dir / "tune" / "best_params.json").write_text("{}", encoding="utf-8")
    (session_dir / "tune" / "nested" / "evidence.json").write_text(
        "{}", encoding="utf-8"
    )

    assert collect_artifact_manifest(session_dir, "tune/") == [
        "tune/best_params.json",
        "tune/nested/evidence.json",
    ]
    assert collect_public_artifact_manifest(session_dir, "tune") == [
        "tune/best_params.json"
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        collect_public_artifact_manifest(session_dir, "pipeline")
    assert collect_artifact_manifest(session_dir, "missing") == []
    assert artifact_name_from_key("tune/report.md", "tune") == "report.md"
    with pytest.raises(ValueError, match="outside"):
        artifact_name_from_key("refine/report.md", "tune")

    assert is_safe_artifact_name("final/recording.usd")
    assert not is_safe_artifact_name("../session.json")
    assert not is_safe_artifact_name("final\\recording.usd")


@pytest.mark.asyncio
async def test_available_keys_intersect_manifest_and_support_legacy(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    tune_dir = session_dir / "tune"
    tune_dir.mkdir()
    (tune_dir / "best_params.json").write_text("{}", encoding="utf-8")
    (tune_dir / "unlisted.json").write_text("{}", encoding="utf-8")

    available = await available_artifact_keys(
        manager,
        session_id,
        {"artifact_manifest": ["tune/best_params.json", "tune/missing.json"]},
        "tune",
    )
    assert available == {"tune/best_params.json"}

    legacy_available = await available_artifact_keys(manager, session_id, {}, "tune")
    assert legacy_available == {
        "tune/best_params.json",
        "tune/unlisted.json",
    }


@pytest.mark.asyncio
async def test_result_urls_only_expose_canonical_available_files(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "sessions")

    tune_id = str(uuid4())
    tune_dir = (await manager.create_session(tune_id)) / "tune"
    tune_dir.mkdir()
    (tune_dir / "best_params.json").write_text("{}", encoding="utf-8")
    tune_metadata = {
        "artifact_manifest": [
            "tune/best_params.json",
            "tune/tuned_physics.usd",
            "tune/comparison.png",
        ]
    }
    assert await tune_router._tune_download_urls(manager, tune_id, tune_metadata) == {
        "best_params": f"/tune/{tune_id}/artifacts/best_params.json",
    }
    (tune_dir / "tuned_physics.usda").write_text("#usda 1.0\n", encoding="utf-8")
    assert await tune_router._tune_download_urls(
        manager,
        tune_id,
        {"artifact_manifest": ["tune/tuned_physics.usda"]},
    ) == {
        "tuned_usd": f"/tune/{tune_id}/artifacts/tuned_physics.usda",
    }

    refine_id = str(uuid4())
    refine_dir = (await manager.create_session(refine_id)) / "refine"
    render_dir = refine_dir / "final" / "render"
    render_dir.mkdir(parents=True)
    (refine_dir / "final" / "report.md").write_text("report", encoding="utf-8")
    (refine_dir / "final" / "tuned_physics.usda").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    (refine_dir / "final" / "recording.usd").write_bytes(b"recording")
    (render_dir / "render.mp4").write_bytes(b"canonical")
    (render_dir / "World_Camera_2__render.mp4").write_bytes(b"two")
    (render_dir / "World_Camera_10__render.mp4").write_bytes(b"ten")
    manifest = collect_public_artifact_manifest(refine_dir.parent, "refine")
    urls = await refine_router._refine_download_urls(
        manager,
        refine_id,
        {"artifact_manifest": manifest},
    )

    assert urls["final_report"] == (f"/refine/{refine_id}/artifacts/final/report.md")
    assert urls["final_tuned_usd"] == (
        f"/refine/{refine_id}/artifacts/final/tuned_physics.usda"
    )
    assert urls["final_recording_usd"] == (
        f"/refine/{refine_id}/artifacts/final/recording.usd"
    )
    assert not any(key.startswith("final_render_mp4") for key in urls)
    assert not any(key.endswith(".mp4") for key in manifest)
    assert "final_render_gif" not in urls
