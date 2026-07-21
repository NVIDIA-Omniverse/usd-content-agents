# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from PIL import Image

pytest.importorskip("pxr")

from fake_projection_backend import FakeProjectionBackend
from texture_agent.functions.artifact_manifest import (
    _output_texture_references,
    validate_artifacts_manifest_schema,
)

from ...client.client import TextureAgentClient
from ...service.routers import pipeline_router, sessions_router
from ...service.runtime.bus import init_event_bus
from ...service.session.manager import SessionManager
from ...service.workers import executor

_MATERIAL_NAME = "Aluminum_Matte"
_ALUMINUM_PRIM_PATH = "/RootNode/Geometry/M_AluminumStepLadder_B01_Aluminum"
_OTHER_MATERIALS = {
    "Aluminum_Brushed",
    "Rubber_Black_Matte",
    "Plastic_Black_Matte",
}


class _ClientResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"session_id": "issue116-service-smoke"}


class _ClientHttp:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _ClientResponse:
        self.posts.append({"url": url, **kwargs})
        return _ClientResponse()


def _ladder_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd"


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
                            (64, 48 + index, 128),
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


def _service_config_from_client_form(
    session_id: str,
    session_dir: Path,
    form: dict[str, str],
) -> dict[str, Any]:
    config = pipeline_router.build_default_pipeline_config(
        session_id=session_id,
        usd_path=str(_ladder_path()),
        working_dir=str(session_dir / "cache"),
        material_textures=json.loads(form["material_textures_json"]),
        auto_prompt_enabled=form["auto_prompt_enabled"] == "true",
        texture_backend=form["texture_backend"],
        texture_endpoint=form["texture_endpoint"],
        backend_engine=form["backend_engine"],
        backend_custom_parameters=json.loads(form["backend_custom_parameters_json"]),
        seed=int(form["seed"]),
        strength=float(form["strength"]),
        strict_scope=form["strict_scope"] == "true",
    )

    config["texture"].update(
        {
            "size": 16,
            "workers": 1,
            "skip_existing": False,
            "failure_threshold": 0.0,
            "uv_policy": "generate_missing",
            "uv_projection": "box",
            "capabilities": {
                "image_conditioning": True,
                "normal_map": False,
                "orm": False,
                "masks": True,
                "coverage": True,
                "geometry_output": "none",
            },
        }
    )
    config["steps"]["generate_textures"].update(
        {"skip_existing": False, "max_workers": 1}
    )
    config["steps"]["blend_textures"]["output_size"] = 16
    config["steps"]["render"] = {
        "enabled": True,
        "image_width": 16,
        "image_height": 16,
        "camera_paths": ["/Cameras/TextureAgentFinal"],
        "focus_cameras": True,
        "max_focus_cameras": 1,
    }
    return config


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


@pytest.mark.asyncio
async def test_issue116_service_client_ladder_projection_backend_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "_package_usdz", lambda context, session_dir: None)
    _mock_render_all_cameras(monkeypatch)

    client = TextureAgentClient("http://texture.test")
    fake_http = _ClientHttp()
    client._http = fake_http

    with FakeProjectionBackend(tmp_path / "backend") as backend:
        client_session_id = client.start_pipeline(
            session_id="uploaded-ladder",
            material_textures={
                _MATERIAL_NAME: {
                    "prompt": "service projection matte aluminum",
                    "opacity": 0.85,
                }
            },
            auto_prompt_enabled=False,
            texture_backend="service",
            texture_endpoint=backend.endpoint_url,
            backend_engine="fake_projection",
            backend_custom_parameters={"variant": "albedo_only_degraded"},
            seed=11631,
            strength=0.85,
            strict_scope=True,
        )

        assert client_session_id == "issue116-service-smoke"
        client_form = fake_http.posts[0]["data"]
        assert client_form["auto_prompt_enabled"] == "false"

        session_id = "issue116-service-smoke"
        manager = SessionManager(tmp_path / "sessions")
        session_dir = manager.create_session(session_id)
        pipeline_router.set_session_manager(manager)
        sessions_router.set_session_manager(manager)
        init_event_bus(manager).clear_session_state(session_id)

        await executor.execute_pipeline_async(
            session_id,
            _service_config_from_client_form(session_id, session_dir, client_form),
            manager,
            acquire_worker_lock=False,
        )

        assert len(backend.requests) == 1
        request = backend.requests[0]
        assert request["target"] == {
            "material_name": _MATERIAL_NAME,
            "material_path": "/RootNode/Looks/Aluminum_Matte",
            "prim_paths": [_ALUMINUM_PRIM_PATH],
            "mode": "per_material",
            "strict_scope": True,
        }
        assert request["configuration"]["custom_parameters"] == {
            "variant": "albedo_only_degraded"
        }

    results_response = await pipeline_router.get_pipeline_results(session_id)
    assert results_response.status == "completed"
    assert results_response.stats["materials_found"] == 4
    assert results_response.stats["textures_generated"] == 1
    assert results_response.stats.get("textures_failed", 0) == 0
    assert results_response.stats["projection_backend_units"] == 1
    texture_plan = json.loads(
        (session_dir / "cache" / "texture_plan.json").read_text(encoding="utf-8")
    )
    texture_unit_id = texture_plan["selected_units"][0]["unit_id"]
    assert results_response.stats["projection_backend_map_counts"] == {
        texture_unit_id: 1
    }
    assert results_response.stats["projection_backend_warnings"][0]["code"] == (
        "BACKEND_MAP_MISSING"
    )
    assert results_response.stats["renders_count"] >= 2
    assert results_response.stats["render_available"] is True
    assert "/Cameras/TextureAgentFinal" in results_response.stats["render_camera_paths"]
    assert len(results_response.stats["render_focus_cameras"]) == 1
    focus = results_response.stats["render_focus_cameras"][0]
    assert focus["prim_path"] == _ALUMINUM_PRIM_PATH
    assert focus["meets_target_frame_coverage"] is True
    assert results_response.download_urls["manifest"] == (
        f"/artifacts/{session_id}/manifest"
    )
    assert results_response.stats["manifest_url"] == f"/artifacts/{session_id}/manifest"

    cache_dir = session_dir / "cache"
    generated_dir = cache_dir / "generated"
    assert (generated_dir / f"{texture_unit_id}_albedo.png").is_file()
    for other in _OTHER_MATERIALS:
        assert not (generated_dir / f"{other}_albedo.png").exists()

    _assert_png(generated_dir / f"{texture_unit_id}_albedo.png")
    _assert_png(generated_dir / f"{texture_unit_id}_normal.png")
    _assert_png(generated_dir / f"{texture_unit_id}_orm.png")
    _assert_png(cache_dir / "textures" / f"{texture_unit_id}_albedo.png")
    _assert_png(cache_dir / "textures" / f"{texture_unit_id}_normal.png")
    _assert_png(cache_dir / "textures" / f"{texture_unit_id}_orm.png")

    output_usd = cache_dir / "output" / "textured_output.usd"
    assert output_usd.is_file()
    _assert_output_usd_references_are_scoped_and_portable(
        output_usd,
        texture_unit_id,
    )

    manifest = json.loads(
        (cache_dir / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    assert validate_artifacts_manifest_schema(manifest) == []
    assert manifest["input"]["config"]["auto_prompt"]["enabled"] is False
    assert manifest["input"]["requested_material_scope"] == [_MATERIAL_NAME]
    assert manifest["materials"]["auto_prompt_additions"] == {}
    assert [item["material_name"] for item in manifest["materials"]["selected"]] == [
        _MATERIAL_NAME
    ]
    assert manifest["textures"]["generated_count"] == 1
    assert manifest["textures"]["blended_count"] == 1
    assert manifest["outputs"]["portability"]["portable"] is True
    assert manifest["outputs"]["portability"]["texture_reference_count"] > 0
    assert manifest["outputs"]["portability"]["diagnostics"] == []
    assert manifest["renders"]["render_available"] is True
    assert len(manifest["renders"]["final"]) >= 2
    assert manifest["status"]["state"] == "completed"
    assert manifest["status"]["errors"]["generate_textures"] == []
    assert manifest["status"]["errors"]["blend_textures"] == []

    projection = manifest["textures"]["projection_backend"][texture_unit_id]
    assert projection["map_count"] == 1
    assert set(projection["maps"]) == {"albedo"}
    assert projection["channel_state"]["albedo"] == "present"
    assert projection["channel_state"]["normal"] == "synthesized_neutral"
    assert projection["channel_state"]["orm"] == "packed_from_channels_or_constants"
    assert projection["degraded_channels"] == ["normal", "orm"]
    assert projection["warnings"][0]["code"] == "BACKEND_MAP_MISSING"
    assert projection["diagnostics"][0]["severity"] == "warning"
    assert manifest["backend"]["projection"]["unit_count"] == 1
    assert manifest["backend"]["projection"]["diagnostics"][0]["code"] == (
        "BACKEND_MAP_MISSING"
    )
    assert {item["code"] for item in manifest["status"]["diagnostics"]} == {
        "BACKEND_MAP_MISSING"
    }
    _assert_manifest_paths_are_portable(manifest, session_dir, texture_unit_id)


@pytest.mark.asyncio
async def test_issue116_service_projection_backend_missing_albedo_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "_package_usdz", lambda context, session_dir: None)

    session_id = "issue116-service-missing-albedo"
    manager = SessionManager(tmp_path / "sessions")
    session_dir = manager.create_session(session_id)
    pipeline_router.set_session_manager(manager)
    sessions_router.set_session_manager(manager)
    init_event_bus(manager).clear_session_state(session_id)

    client_form = {
        "material_textures_json": json.dumps(
            {
                _MATERIAL_NAME: {
                    "prompt": "service projection matte aluminum",
                    "opacity": 0.85,
                }
            }
        ),
        "auto_prompt_enabled": "false",
        "texture_backend": "service",
        "backend_engine": "fake_projection",
        "backend_custom_parameters_json": json.dumps({"variant": "missing_albedo"}),
        "seed": "11631",
        "strength": "0.85",
        "strict_scope": "true",
    }

    with FakeProjectionBackend(tmp_path / "backend") as backend:
        client_form["texture_endpoint"] = backend.endpoint_url
        with pytest.raises(RuntimeError, match="texture generation requests failed"):
            await executor.execute_pipeline_async(
                session_id,
                _service_config_from_client_form(session_id, session_dir, client_form),
                manager,
                acquire_worker_lock=False,
            )

        assert len(backend.requests) == 1

    texture_plan = json.loads(
        (session_dir / "cache" / "texture_plan.json").read_text(encoding="utf-8")
    )
    texture_unit_id = texture_plan["selected_units"][0]["unit_id"]
    metadata = manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["status"] == "failed"
    assert metadata["failed_step"] == "generate_textures"
    failed_stats = metadata["failed_step_stats"]
    assert failed_stats["textures_generated"] == 0
    assert failed_stats["textures_failed"] == 1
    assert failed_stats["projection_backend_units"] == 1
    assert failed_stats["projection_backend_map_counts"] == {texture_unit_id: 1}
    assert failed_stats["projection_backend_diagnostics"][0]["code"] == (
        "BACKEND_MAP_MISSING"
    )
    diagnostic_codes = {
        item["code"] for item in failed_stats["projection_backend_diagnostics"]
    }
    assert "BACKEND_PARTIAL_FAILURE" not in diagnostic_codes
    assert failed_stats["manifest_available"] is True

    error_response = await pipeline_router.get_pipeline_results(session_id)
    assert error_response.status == "failed"
    assert error_response.failed_step == "generate_textures"
    assert error_response.failed_step_stats["textures_failed"] == 1
    assert not (session_dir / "cache" / "output" / "textured_output.usd").exists()

    manifest = json.loads(
        (session_dir / "cache" / "artifacts_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"]["state"] == "failed"
    assert manifest["textures"]["generated_count"] == 0
    assert manifest["outputs"]["portability"]["portable"] is False
    assert manifest["textures"]["generation_errors"][0]["material"] == texture_unit_id
    assert manifest["status"]["diagnostics"][0]["code"] == "BACKEND_MAP_MISSING"
