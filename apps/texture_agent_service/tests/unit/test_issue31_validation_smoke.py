# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import texture_agent.tasks.generate_textures as generate_textures_task
from PIL import Image
from texture_agent.functions.texture_generation import (
    GeneratedTextures,
    GenerationResult,
    JobStatus,
)

from ...service.routers import pipeline_router, sessions_router
from ...service.runtime.bus import init_event_bus
from ...service.session.manager import SessionManager
from ...service.workers import executor

pytest.importorskip("pxr")


class _FakeImageGenEngine:
    name = "fake-lightweight"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _ensure_model(self) -> object:
        # GenerateTexturesTask calls this before constructing its client.
        return object()


class _FakeTextureVariationClient:
    def __init__(self, engine: object, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def generate(self, *, config: Any, **kwargs: Any) -> JobStatus:
        key = str(config.variant_name)
        textures = _write_fake_texture_set(self.output_dir, key)
        return JobStatus(
            job_id=f"fake-{key}",
            status="completed",
            result=GenerationResult(
                variant_asset_uri="",
                variant_name=key,
                generated_textures=textures,
            ),
        )


def _write_fake_texture_set(output_dir: Path, key: str) -> GeneratedTextures:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    colors = {
        "albedo": (180, 92, 40),
        "normal": (128, 128, 255),
        "orm": (255, 128, 20),
    }
    for suffix, color in colors.items():
        path = output_dir / f"{key}_{suffix}.png"
        Image.new("RGB", (16, 16), color).save(path)
        paths[suffix] = str(path)
    return GeneratedTextures(**paths)


def _ladder_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd"


@pytest.mark.asyncio
async def test_issue31_service_ladder_smoke_matches_cli_validation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate_textures_task, "ImageGenEngine", _FakeImageGenEngine)
    monkeypatch.setattr(
        generate_textures_task,
        "TextureVariationClient",
        _FakeTextureVariationClient,
    )
    # Packaging and manifest checks are owned by the sibling A31-1 slice.
    monkeypatch.setattr(executor, "_package_usdz", lambda context, session_dir: None)

    session_id = "issue31-service-smoke"
    manager = SessionManager(tmp_path)
    session_dir = manager.create_session(session_id)
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    init_event_bus(manager).clear_session_state(session_id)

    config = {
        "project": {"name": "issue31_service_smoke", "session_id": session_id},
        "input": {"usd_path": str(_ladder_path())},
        "texture": {
            "backend": "simple_image_gen",
            "image_gen": {"backend": "test"},
            "size": 16,
            "workers": 1,
            "skip_existing": False,
            "uv_policy": "generate_missing",
            "uv_projection": "box",
        },
        "material_textures": {
            "Aluminum_Matte": {
                "prompt": "deterministic rusty matte aluminum",
                "opacity": 0.85,
            }
        },
        "auto_prompt": {"enabled": False},
        "steps": {
            "render_previews": {"enabled": False},
            "blend_textures": {"output_size": 16},
            "render": {"enabled": False},
        },
    }

    await executor.execute_pipeline_async(
        session_id,
        config,
        manager,
        acquire_worker_lock=False,
    )

    results_response = await pipeline_router.get_pipeline_results(session_id)
    assert results_response.status == "completed"
    assert results_response.stats["materials_found"] == 4
    assert results_response.stats["textures_generated"] == 1
    assert results_response.stats["output_usd_count"] == 1
    assert results_response.stats["renders_count"] == 0
    assert results_response.download_urls["textures"] == (
        f"/artifacts/{session_id}/textures"
    )
    assert results_response.download_urls["output"] == (
        f"/artifacts/{session_id}/output"
    )

    uv_report_path = session_dir / "cache" / "prepared" / "uv_report.json"
    uv_report = json.loads(uv_report_path.read_text(encoding="utf-8"))
    assert uv_report["schema_version"] == "texture-agent-uv-report.v1"

    texture_plan = json.loads(
        (session_dir / "cache" / "texture_plan.json").read_text(encoding="utf-8")
    )
    texture_unit_id = texture_plan["selected_units"][0]["unit_id"]
    albedo_path = session_dir / "cache" / "textures" / f"{texture_unit_id}_albedo.png"
    with Image.open(albedo_path) as image:
        assert image.size == (16, 16)
        assert image.getbbox() is not None

    assert not (
        session_dir / "cache" / "generated" / "Aluminum_Brushed_albedo.png"
    ).exists()
