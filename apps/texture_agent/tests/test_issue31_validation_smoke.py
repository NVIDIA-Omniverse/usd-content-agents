# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

import texture_agent.tasks.generate_textures as generate_textures_task
from texture_agent.cli import app
from texture_agent.functions.texture_generation import (
    GeneratedTextures,
    GenerationResult,
    JobStatus,
)

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
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd"


def test_issue31_cli_ladder_smoke_uses_fake_backend_and_strict_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generate_textures_task, "ImageGenEngine", _FakeImageGenEngine)
    monkeypatch.setattr(
        generate_textures_task,
        "TextureVariationClient",
        _FakeTextureVariationClient,
    )

    working_dir = tmp_path / "work"
    config_path = tmp_path / "texture_ladder_issue31_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "issue31_cli_smoke",
                    "session_id": "issue31_cli_smoke",
                    "working_dir": str(working_dir),
                },
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
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["run", str(config_path)])

    assert result.exit_code == 0, result.output
    uv_report_path = working_dir / "prepared" / "uv_report.json"
    uv_report = json.loads(uv_report_path.read_text(encoding="utf-8"))
    assert uv_report["schema_version"] == "texture-agent-uv-report.v1"

    materials_path = working_dir / "discovery" / "materials.json"
    materials = json.loads(materials_path.read_text(encoding="utf-8"))
    # The primary ladder fixture has four materials; strict scope textures one.
    assert len(materials) == 4

    texture_plan = json.loads(
        (working_dir / "texture_plan.json").read_text(encoding="utf-8")
    )
    selected_unit_ids = {unit["unit_id"] for unit in texture_plan["selected_units"]}
    assert len(selected_unit_ids) == 1
    texture_unit_id = next(iter(selected_unit_ids))
    generated_dir = working_dir / "generated"
    generated_albedo_ids = {
        path.name.removesuffix("_albedo.png")
        for path in generated_dir.glob("*_albedo.png")
    }
    assert generated_albedo_ids == selected_unit_ids

    albedo_path = working_dir / "textures" / f"{texture_unit_id}_albedo.png"
    with Image.open(albedo_path) as image:
        assert image.size == (16, 16)
        assert image.getbbox() is not None

    output_usd = working_dir / "output" / "textured_output.usd"
    assert output_usd.exists()

    from world_understanding.functions.graphics.validate_usd import (
        TEXTURE_VALIDATION_CATEGORIES,
        is_available,
        validate_usd,
    )

    if not is_available():
        pytest.skip("usd-validation-nvidia not installed")

    validation = validate_usd(
        output_usd,
        categories=list(TEXTURE_VALIDATION_CATEGORIES),
    )
    assert validation["status"] == "success"
    assert set(validation["categories_checked"]) == set(TEXTURE_VALIDATION_CATEGORIES)
    assert validation["summary"]["failures"] == 0
    assert validation["summary"]["errors"] == 0
