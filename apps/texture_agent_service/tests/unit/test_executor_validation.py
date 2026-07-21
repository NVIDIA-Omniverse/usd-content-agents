# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for executor terminal-state validation."""

import asyncio
from pathlib import Path

import pytest

from ...service.routers import pipeline_router
from ...service.session.manager import SessionManager
from ...service.workers.executor import (
    _extract_step_stats,
    _get_step_validation_error,
    execute_pipeline_async,
)

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402


def _write_multi_prim_material_stage(path: Path, *, prim_count: int) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.Scope.Define(stage, "/World/Looks")

    material = UsdShade.Material.Define(stage, "/World/Looks/Paint")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Paint/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 0.3, 0.1)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    for index in range(prim_count):
        cube = UsdGeom.Cube.Define(stage, f"/World/Panel_{index}")
        cube.CreateSizeAttr(0.1)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    return path


def test_service_generated_pipeline_fails_before_backend_on_texture_unit_fanout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline_router.config, "max_texture_units", 2)
    usd_path = _write_multi_prim_material_stage(
        tmp_path / "multi_prim.usda", prim_count=3
    )
    manager = SessionManager(tmp_path / "sessions", ttl_hours=2)
    session_id = "fanout-guard-local-smoke"
    session_dir = manager.create_session(session_id)
    config = pipeline_router.build_default_pipeline_config(
        session_id=session_id,
        usd_path=str(usd_path),
        working_dir=str(session_dir / "cache"),
        material_textures={
            "Paint": {
                "prompt": "slightly worn green paint",
                "per_prim": {"/World/Panel_0": {"prompt": "worn left panel"}},
            }
        },
        auto_prompt_enabled=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Texture generation would create 3 expanded texture units, exceeding "
            "texture.max_texture_units=2"
        ),
    ):
        asyncio.run(
            execute_pipeline_async(
                session_id,
                config,
                manager,
                only_steps=[
                    "discover_materials",
                    "plan_textures",
                    "generate_prompts",
                ],
            )
        )

    metadata = manager.get_session_metadata(session_id)
    assert metadata["status"] == "failed"
    assert metadata["failed_step"] == "generate_prompts"
    assert "texture.max_texture_units=2" in metadata["error"]


class TestGetStepValidationError:
    """Tests for executor validation of false-positive success states."""

    def test_discover_materials_fails_when_downstream_steps_need_materials(self):
        error = _get_step_validation_error(
            "discover_materials",
            {"materials_found": 0},
            [
                "prepare_uvs",
                "discover_materials",
                "generate_prompts",
                "generate_textures",
                "apply_textures",
            ],
        )

        assert error is not None
        assert "No discoverable materials" in error

    def test_discover_materials_allows_zero_materials_for_discovery_only_runs(self):
        error = _get_step_validation_error(
            "discover_materials",
            {"materials_found": 0},
            ["discover_materials"],
        )

        assert error is None

    def test_apply_textures_fails_when_no_output_usd_is_written(self):
        error = _get_step_validation_error(
            "apply_textures",
            {"output_usd_count": 0},
            ["discover_materials", "generate_textures", "apply_textures"],
        )

        assert error is not None
        assert "no output USD files" in error

    def test_apply_textures_succeeds_when_output_usd_exists(self):
        error = _get_step_validation_error(
            "apply_textures",
            {"output_usd_count": 1},
            ["discover_materials", "generate_textures", "apply_textures"],
        )

        assert error is None

    def test_apply_textures_message_names_upstream_generate_failure(self):
        error = _get_step_validation_error(
            "apply_textures",
            {"output_usd_count": 0},
            ["discover_materials", "generate_textures", "apply_textures"],
            context={
                "generated_textures": {},
                "blended_textures": {},
                "generate_textures_errors": [
                    {
                        "material": "A",
                        "type": "RuntimeError",
                        "status": 403,
                        "message": "x",
                    },
                    {
                        "material": "B",
                        "type": "RuntimeError",
                        "status": 403,
                        "message": "y",
                    },
                ],
            },
        )

        assert error is not None
        assert "no output USD files" in error
        assert "upstream generate_textures produced 0 textures" in error
        assert "2 per-material failure(s)" in error

    def test_apply_textures_message_names_upstream_blend_failure(self):
        error = _get_step_validation_error(
            "apply_textures",
            {"output_usd_count": 0},
            ["discover_materials", "generate_textures", "apply_textures"],
            context={
                "generated_textures": {"A": object()},
                "blended_textures": {},
                "blend_textures_errors": [
                    {
                        "material": "A",
                        "type": "MissingAlbedo",
                        "status": None,
                        "message": "x",
                    },
                ],
            },
        )

        assert error is not None
        assert "upstream blend_textures produced 0 textures" in error


class TestExtractStepStats:
    def test_generate_textures_includes_failed_count_and_errors(self):
        stats = _extract_step_stats(
            "generate_textures",
            {
                "generated_textures": {"A": object()},
                "generate_textures_errors": [
                    {
                        "material": "B",
                        "type": "RuntimeError",
                        "status": 403,
                        "message": "x",
                    },
                ],
                "generate_textures_failed_count": 1,
            },
        )
        assert stats == {
            "textures_generated": 1,
            "textures_failed": 1,
            "errors": [
                {
                    "material": "B",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "x",
                },
            ],
        }

    def test_blend_textures_includes_failed_count_and_errors(self):
        stats = _extract_step_stats(
            "blend_textures",
            {
                "blended_textures": {"A": object()},
                "blend_textures_errors": [
                    {
                        "material": "B",
                        "type": "MissingAlbedo",
                        "status": None,
                        "message": "x",
                    },
                ],
                "blend_textures_failed_count": 1,
            },
        )
        assert stats == {
            "textures_blended": 1,
            "textures_failed": 1,
            "errors": [
                {
                    "material": "B",
                    "type": "MissingAlbedo",
                    "status": None,
                    "message": "x",
                },
            ],
        }

    def test_generate_textures_omits_errors_key_when_no_failures(self):
        stats = _extract_step_stats(
            "generate_textures",
            {
                "generated_textures": {"A": object()},
                "generate_textures_errors": [],
                "generate_textures_failed_count": 0,
            },
        )
        assert stats == {"textures_generated": 1, "textures_failed": 0}
        assert "errors" not in stats
