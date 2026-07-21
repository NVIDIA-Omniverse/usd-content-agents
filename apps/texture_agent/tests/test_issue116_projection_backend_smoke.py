# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

pytest.importorskip("pxr")

from apps.texture_gen_service_common.artifacts import local_path_from_file_uri
from fake_projection_backend import FakeProjectionBackend

from texture_agent.cli import app
from texture_agent.functions.artifact_manifest import (
    _output_texture_references,
    validate_artifacts_manifest_schema,
)

_MATERIAL_NAME = "Aluminum_Matte"
_ALUMINUM_PRIM_PATH = "/RootNode/Geometry/M_AluminumStepLadder_B01_Aluminum"
_OTHER_MATERIALS = {
    "Aluminum_Brushed",
    "Rubber_Black_Matte",
    "Plastic_Black_Matte",
}


def _ladder_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd"


def _base_config(
    working_dir: Path,
    endpoint: str,
    *,
    variant: str,
    render_enabled: bool,
) -> dict:
    return {
        "project": {
            "name": "issue116_cli_smoke",
            "session_id": "issue116_cli_smoke",
            "working_dir": str(working_dir),
        },
        "input": {"usd_path": str(_ladder_path())},
        "texture": {
            "backend": "service",
            "endpoint": endpoint,
            "engine": "fake_projection",
            "custom_parameters": {"variant": variant},
            "size": 16,
            "workers": 1,
            "skip_existing": False,
            "failure_threshold": 0.0,
            "uv_policy": "generate_missing",
            "uv_projection": "box",
            "strict_scope": True,
            "capabilities": {
                "image_conditioning": True,
                "normal_map": True,
                "orm": True,
                "masks": True,
                "coverage": True,
                "geometry_output": "none",
            },
        },
        "material_textures": {
            _MATERIAL_NAME: {
                "prompt": "deterministic projection matte aluminum",
                "opacity": 0.85,
            }
        },
        "auto_prompt": {"enabled": False},
        "steps": {
            "render_previews": {"enabled": False},
            "blend_textures": {"output_size": 16},
            "render": {
                "enabled": render_enabled,
                "image_width": 16,
                "image_height": 16,
                "camera_paths": ["/Cameras/TextureAgentFinal"],
                "focus_cameras": True,
                "max_focus_cameras": 1,
            },
        },
    }


def _mock_render_all_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    def _render_all_cameras(
        *,
        stage: object,
        image_width: int,
        image_height: int,
        cameras: list[str],
        base_dir: object | None = None,
        max_workers: int | None = None,
        timeout: int | None = None,
        **_renderer_options: object,
    ) -> dict:
        del base_dir, stage
        assert max_workers == 1
        assert timeout == 3600
        assert "/Cameras/TextureAgentFinal" in cameras
        assert any(
            camera.startswith("/Cameras/TextureAgentFocus_") for camera in cameras
        )
        return {
            "results": [
                {
                    "camera": camera,
                    "status": "success",
                    "images": [
                        Image.new(
                            "RGB",
                            (image_width, image_height),
                            (32 + index, 96, 160),
                        )
                    ],
                }
                for index, camera in enumerate(cameras)
            ]
        }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote.render_all_cameras",
        _render_all_cameras,
    )


def _assert_png(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    assert path.is_file(), path
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == size
        assert image.getbbox() is not None


def _assert_manifest_paths_are_portable(
    manifest: dict,
    root: Path,
    texture_unit_id: str,
) -> None:
    generated = manifest["textures"]["generated"][texture_unit_id]
    blended = manifest["textures"]["blended"][texture_unit_id]
    paths = [
        generated["albedo"]["path"],
        generated["normal"]["path"],
        generated["orm"]["path"],
        blended["albedo"]["path"],
        blended["normal"]["path"],
        blended["orm"]["path"],
        manifest["prepared"]["prepared_usd"]["path"],
        manifest["prepared"]["uv_report"]["path"],
    ]
    paths.extend(item["path"] for item in manifest["renders"]["final"])

    for path in paths:
        assert path
        assert not Path(path).is_absolute()
        assert not urlparse(path).scheme
        assert (root / path).exists(), path


def _assert_output_usd_references_are_scoped_and_portable(
    output_usd: Path,
    texture_unit_id: str,
) -> None:
    refs = _output_texture_references(output_usd)
    assert refs
    for ref in refs:
        texture_path = str(ref["path"])
        assert texture_unit_id in texture_path
        assert not Path(texture_path).is_absolute()
        assert not urlparse(texture_path).scheme
        assert (output_usd.parent / texture_path).resolve().is_file()
        assert not any(other in texture_path for other in _OTHER_MATERIALS)


def test_issue116_cli_ladder_projection_backend_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working_dir = tmp_path / "work"
    _mock_render_all_cameras(monkeypatch)

    with FakeProjectionBackend(tmp_path / "backend") as backend:
        config_path = tmp_path / "texture_ladder_issue116_projection.yaml"
        config_path.write_text(
            yaml.safe_dump(
                _base_config(
                    working_dir,
                    endpoint=backend.endpoint_url,
                    variant="success_full_pbr",
                    render_enabled=True,
                )
            ),
            encoding="utf-8",
        )

        result = CliRunner().invoke(app, ["run", str(config_path)])

        assert result.exit_code == 0, result.output
        assert len(backend.requests) == 1
        request = backend.requests[0]
        uv_report = json.loads(
            (working_dir / "prepared" / "uv_report.json").read_text(encoding="utf-8")
        )
        assert (
            request["source_asset_uri"]
            == Path(uv_report["prepared_usd"]).resolve().as_uri()
        )
        assert request["target"] == {
            "material_name": _MATERIAL_NAME,
            "material_path": "/RootNode/Looks/Aluminum_Matte",
            "prim_paths": [_ALUMINUM_PRIM_PATH],
            "mode": "per_material",
            "strict_scope": True,
        }
        assert request["conditioning"]["text_prompt"] == (
            "deterministic projection matte aluminum"
        )
        assert request["configuration"]["texture_size"] == 16
        assert request["configuration"]["engine"] == "fake_projection"

    texture_plan = json.loads(
        (working_dir / "texture_plan.json").read_text(encoding="utf-8")
    )
    texture_unit_id = texture_plan["selected_units"][0]["unit_id"]
    generated_dir = working_dir / "generated"
    assert (generated_dir / f"{texture_unit_id}_albedo.png").is_file()
    for other in _OTHER_MATERIALS:
        assert not (generated_dir / f"{other}_albedo.png").exists()

    _assert_png(generated_dir / f"{texture_unit_id}_albedo.png")
    _assert_png(generated_dir / f"{texture_unit_id}_normal.png")
    _assert_png(generated_dir / f"{texture_unit_id}_orm.png")
    _assert_png(working_dir / "textures" / f"{texture_unit_id}_albedo.png")
    _assert_png(working_dir / "textures" / f"{texture_unit_id}_normal.png")
    _assert_png(working_dir / "textures" / f"{texture_unit_id}_orm.png")

    output_usd = working_dir / "output" / "textured_output.usd"
    assert output_usd.is_file()
    _assert_output_usd_references_are_scoped_and_portable(
        output_usd,
        texture_unit_id,
    )

    manifest = json.loads(
        (working_dir / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    assert validate_artifacts_manifest_schema(manifest) == []
    assert manifest["input"]["requested_material_scope"] == [_MATERIAL_NAME]
    assert manifest["materials"]["auto_prompt_additions"] == {}
    assert [item["material_name"] for item in manifest["materials"]["selected"]] == [
        _MATERIAL_NAME
    ]
    assert manifest["prompts"]["prompt_source"] == "material_textures"
    assert manifest["textures"]["generated_count"] == 1
    assert manifest["textures"]["blended_count"] == 1
    assert sorted(manifest["textures"]["projection_backend"]) == [texture_unit_id]
    assert manifest["outputs"]["portability"]["portable"] is True
    assert manifest["outputs"]["portability"]["texture_reference_count"] > 0
    assert manifest["outputs"]["portability"]["diagnostics"] == []
    assert manifest["backend"]["endpoint"] == "<configured>"
    assert manifest["backend"]["projection"]["unit_count"] == 1

    projection = manifest["textures"]["projection_backend"][texture_unit_id]
    assert projection["map_count"] == 3
    assert set(projection["maps"]) == {"albedo", "normal", "orm"}
    assert projection["channel_state"]["albedo"] == "present"
    assert projection["channel_state"]["normal"] == "present"
    assert projection["channel_state"]["orm"] == "present"
    assert projection["coverage"]["target_coverage"] == 0.97
    assert projection["diagnostics"] == []
    assert projection["warnings"] == []
    for channel, entry in projection["maps"].items():
        assert entry["width"] == 16
        assert entry["height"] == 16
        assert channel in entry["uri"]
        parsed_uri = urlparse(entry["uri"])
        assert parsed_uri.scheme == "file"
        entry_path = local_path_from_file_uri(entry["uri"])
        assert entry_path is not None
        assert entry_path.is_file()

    assert manifest["renders"]["render_available"] is True
    assert len(manifest["renders"]["final"]) >= 2
    assert manifest["renders"]["diagnostics"] == []
    assert manifest["status"]["state"] == "completed"
    assert manifest["status"]["errors"]["generate_textures"] == []
    assert manifest["status"]["errors"]["blend_textures"] == []
    assert manifest["status"]["diagnostics"] == []
    _assert_manifest_paths_are_portable(
        manifest,
        working_dir.parent,
        texture_unit_id,
    )


def test_issue116_cli_projection_backend_missing_albedo_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    working_dir = tmp_path / "work"
    with FakeProjectionBackend(tmp_path / "backend") as backend:
        config_path = tmp_path / "texture_ladder_issue116_missing_albedo.yaml"
        config_path.write_text(
            yaml.safe_dump(
                _base_config(
                    working_dir,
                    endpoint=backend.endpoint_url,
                    variant="missing_albedo",
                    render_enabled=False,
                )
            ),
            encoding="utf-8",
        )

        with caplog.at_level(logging.ERROR):
            result = CliRunner().invoke(app, ["run", str(config_path)])

    assert result.exit_code != 0
    assert len(backend.requests) == 1
    assert any(
        "texture generation requests failed" in rec.message for rec in caplog.records
    )
    assert not (working_dir / "output" / "textured_output.usd").exists()
    manifest_path = working_dir / "artifacts_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validate_artifacts_manifest_schema(manifest) == []
    assert manifest["status"]["state"] == "failed"
    assert manifest["status"]["completed_steps"] == [
        "prepare_uvs",
        "discover_materials",
        "plan_textures",
        "generate_prompts",
    ]
    assert manifest["status"]["failed_step"] == "generate_textures"
    assert manifest["status"]["error_code"] == "TEXTURE_PIPELINE_STEP_FAILED"
    assert manifest["status"]["error"] == "Texture Agent pipeline step failed."
    partial_paths = {
        artifact["path"] for artifact in manifest["status"]["partial_artifacts"]
    }
    assert "execution/texture_execution_checkpoint.json" in partial_paths
    assert "texture_plan.json" in partial_paths
    serialized = json.dumps(manifest, sort_keys=True)
    assert "Backend did not return required albedo map" not in serialized
    assert manifest["textures"]["generation_errors"] == []
    generated_dir = working_dir / "generated"
    assert not (generated_dir / "Aluminum_Matte_albedo.png").exists()
