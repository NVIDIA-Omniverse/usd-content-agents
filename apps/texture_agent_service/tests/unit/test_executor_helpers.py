# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
from texture_agent.functions.material_discovery import MaterialInfo
from texture_agent.planning import (
    TexturePlan,
    TexturePlanRequest,
    TexturePlanSource,
    TextureUnitMode,
    build_texture_plan,
)
from texture_agent.tasks.generate_prompts import (
    GeneratePromptsTask as TextureGeneratePromptsTask,
)

from ...service.workers.executor import (
    _MAX_ERROR_MESSAGE_CHARS,
    _MAX_ERRORS_IN_PAYLOAD,
    _MAX_RENDER_STATS_ITEMS,
    _artifact_manifest_status,
    _extract_final_stats,
    _extract_step_stats,
    _package_usdz,
    _prepare_config_and_context,
    _requires_executable_texture_plan,
    _task_to_step_name,
    _truncate_errors,
    _write_service_artifact_manifest,
)


class PrepareUVsTask:
    pass


class GeneratePromptsTask:
    pass


class ExecuteTexturePlanTask:
    pass


class _UnknownTask:
    pass


def _texture_plan(
    unit_count: int,
    *,
    unit_mode: TextureUnitMode = TextureUnitMode.PER_MATERIAL,
) -> TexturePlan:
    materials = [
        MaterialInfo(
            prim_path=f"/World/Looks/M{index}",
            name=f"M{index}",
            bound_prim_paths=[f"/World/Mesh{index}"],
        )
        for index in range(unit_count)
    ]
    return build_texture_plan(
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="session://sid/input/scene.usd"),
            unit_mode=unit_mode,
            backend_default_cap=32,
        ),
        discovered_materials=materials,
        material_textures={
            material.name: {"prompt": "cached surface"} for material in materials
        },
    )


def test_task_to_step_name_maps_known_and_unknown_classes() -> None:
    assert _task_to_step_name(PrepareUVsTask()) == "prepare_uvs"
    assert _task_to_step_name(_UnknownTask()) == "_UnknownTask"


def test_executable_plan_gate_only_exempts_marked_legacy_cached_apply() -> None:
    cached_apply = {
        "texture_plan": _texture_plan(33),
        "planning_config": {
            "resume_apply_textures": True,
            "apply_texture_plan_unit_ids": False,
            "allow_non_executable_cached_apply_plan": True,
        },
    }

    assert (
        _requires_executable_texture_plan(GeneratePromptsTask(), cached_apply) is False
    )
    assert (
        _requires_executable_texture_plan(
            GeneratePromptsTask(),
            {
                "planning_config": {
                    "resume_apply_textures": True,
                    "allow_non_executable_cached_apply_plan": False,
                }
            },
        )
        is True
    )
    assert (
        _requires_executable_texture_plan(
            GeneratePromptsTask(),
            {
                "planning_config": {
                    "allow_non_executable_cached_apply_plan": True,
                }
            },
        )
        is True
    )
    assert (
        _requires_executable_texture_plan(ExecuteTexturePlanTask(), cached_apply)
        is True
    )
    plan_id_apply = {
        **cached_apply,
        "planning_config": {
            **cached_apply["planning_config"],
            "apply_texture_plan_unit_ids": True,
        },
    }
    assert (
        _requires_executable_texture_plan(GeneratePromptsTask(), plan_id_apply) is True
    )
    assert _requires_executable_texture_plan(PrepareUVsTask(), cached_apply) is False


@pytest.mark.parametrize(
    ("plan", "expected_state", "expected_narrowing"),
    [
        (_texture_plan(65), "unsupported", True),
        (
            _texture_plan(1, unit_mode=TextureUnitMode.PER_GROUP),
            "unsupported",
            False,
        ),
    ],
    ids=["above-hard-cap-requires-narrowing", "unsupported-unit-mode"],
)
def test_cached_apply_plan_exemption_rejects_unsafe_decisions(
    plan: TexturePlan,
    expected_state: str,
    expected_narrowing: bool,
) -> None:
    assert plan.decision.state == expected_state
    assert plan.decision.explicit_narrowing_required is expected_narrowing
    context = {
        "texture_plan": plan,
        "planning_config": {
            "resume_apply_textures": True,
            "apply_texture_plan_unit_ids": False,
            "allow_non_executable_cached_apply_plan": True,
        },
    }

    assert _requires_executable_texture_plan(GeneratePromptsTask(), context) is True


def test_cached_apply_plan_exemption_requires_a_valid_plan(tmp_path: Path) -> None:
    context = {
        "texture_plan_path": str(tmp_path / "missing-plan.json"),
        "planning_config": {
            "resume_apply_textures": True,
            "apply_texture_plan_unit_ids": False,
            "allow_non_executable_cached_apply_plan": True,
        },
    }

    assert _requires_executable_texture_plan(GeneratePromptsTask(), context) is True


def test_cached_apply_plan_exemption_scopes_second_run_with_display_keys(
    tmp_path: Path,
) -> None:
    plan = _texture_plan(33)
    plan_path = tmp_path / "texture_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    materials = [
        MaterialInfo(
            prim_path=f"/World/Looks/M{index}",
            name=f"M{index}",
            bound_prim_paths=[f"/World/Mesh{index}"],
        )
        for index in range(34)
    ]
    context = {
        "texture_plan_path": str(plan_path),
        "working_dir": str(tmp_path),
        "discovered_materials": materials,
        "material_textures": {
            material.name: {"prompt": "cached surface", "opacity": 1.0}
            for material in materials
        },
        "auto_prompt_config": {"enabled": False},
        "texture_config": {"mode": "per_material", "max_texture_units": 64},
        "planning_config": {
            "resume_apply_textures": True,
            "apply_texture_plan_unit_ids": False,
            "allow_non_executable_cached_apply_plan": True,
        },
    }

    assert _requires_executable_texture_plan(GeneratePromptsTask(), context) is False
    result = TextureGeneratePromptsTask().run(context)

    assert result["texture_plan"] == plan
    assert [unit.key for unit in result["prim_texture_units"]] == [
        f"M{index}" for index in range(33)
    ]


def test_prepare_config_and_context_applies_defaults_and_creates_dirs(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    config, context = _prepare_config_and_context(
        {"input": {"usd_path": "/tmp/input.usd"}},
        session_dir,
    )

    working_dir = session_dir / "cache"
    assert config["project"]["working_dir"] == str(working_dir)
    assert context["working_dir"] == str(working_dir)
    assert context["usd_path"] == "/tmp/input.usd"
    assert (working_dir / "prepared").is_dir()
    assert (working_dir / "renders").is_dir()
    assert context["render_preview_config"]["image_width"] == 512
    assert context["render_config"]["image_width"] == 1024


@pytest.mark.parametrize("step_name", ("render_previews", "render"))
def test_prepare_config_rejects_invalid_render_backend_before_cache_creation(
    tmp_path: Path,
    step_name: str,
) -> None:
    session_dir = tmp_path / "session"

    with pytest.raises(ValueError, match="Unknown rendering backend: typo"):
        _prepare_config_and_context(
            {
                "input": {"usd_path": "/tmp/input.usd"},
                "steps": {step_name: {"backend": "typo"}},
            },
            session_dir,
        )

    assert not (session_dir / "cache").exists()


def test_prepare_config_and_context_reuses_existing_texture_plan_path(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}", encoding="utf-8")

    _config, context = _prepare_config_and_context(
        {"input": {"usd_path": "/tmp/input.usd"}},
        session_dir,
    )

    assert context["texture_plan_path"] == str(plan_path)


def test_prepare_config_and_context_applies_runtime_endpoint_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "session"
    monkeypatch.setenv("TA_IMAGE_GEN_BACKEND", "openai")
    monkeypatch.setenv("TA_IMAGE_GEN_BASE_URL", "http://image-gen-nim:8000/v1")
    monkeypatch.setenv("TA_IMAGE_GEN_MODEL", "black-forest-labs/flux.2-klein-4b")
    monkeypatch.setenv("TA_IMAGE_GEN_API_KEY", "not-used")

    config, context = _prepare_config_and_context(
        {"input": {"usd_path": "/tmp/input.usd"}},
        session_dir,
    )

    assert config["texture"]["image_gen"] == {
        "backend": "openai",
        "base_url": "http://image-gen-nim:8000/v1",
        "model": "black-forest-labs/flux.2-klein-4b",
        "api_key": "not-used",
    }
    assert context["texture_config"]["image_gen"]["api_key"] == "not-used"


def test_extract_step_stats_and_final_stats_fall_back_to_files(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache" / "textures").mkdir(parents=True)
    (session_dir / "cache" / "output").mkdir(parents=True)
    (session_dir / "cache" / "renders").mkdir(parents=True)
    (session_dir / "cache" / "textures" / "one.png").write_text("x", encoding="utf-8")
    (session_dir / "cache" / "textures" / "two.png").write_text("x", encoding="utf-8")
    (session_dir / "cache" / "output" / "a.usd").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    (session_dir / "cache" / "renders" / "final.png").write_text(
        "png", encoding="utf-8"
    )

    assert _extract_step_stats(
        "discover_materials", {"discovered_materials": [1, 2]}
    ) == {"materials_found": 2}
    # generate_textures stats include the failed-count counter (0 when
    # the step succeeded fully); the structured "errors" key is omitted
    # for the empty case so happy-path payloads stay compact.
    assert _extract_step_stats(
        "generate_textures", {"generated_textures": {"a": 1}}
    ) == {"textures_generated": 1, "textures_failed": 0}

    stats = _extract_final_stats({}, session_dir)

    assert stats == {
        "materials_found": 0,
        "textures_generated": 2,
        "output_usd_count": 1,
        "renders_count": 1,
        "render_available": True,
        "package_status": "not_available",
    }


def test_projection_backend_stats_surface_in_step_and_final_results(
    tmp_path: Path,
) -> None:
    context = {
        "generated_textures": {"Aluminum_Matte": object()},
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": {
                    "albedo": {"uri": "file:///tmp/a.png"},
                    "normal": {"uri": "file:///tmp/n.png"},
                    "orm": {"uri": "file:///tmp/o.png"},
                },
                "metadata": {
                    "backend_name": "fake_projection_backend",
                    "endpoint": "https://internal.backend.invalid/v1",
                    "seed": 11631,
                    "api_token": "SHOULD_NOT_SURFACE",
                    "debug_url": (
                        "https://backend.invalid/status?token=SHOULD_NOT_SURFACE"
                    ),
                    "headers": {"Authorization": "Bearer SHOULD_NOT_SURFACE"},
                    "credentials": {"password": "SHOULD_NOT_SURFACE"},
                },
            }
        },
        "generate_textures_diagnostics": [
            {
                "schema_version": "texture-agent-diagnostic.v1",
                "code": "BACKEND_LOW_COVERAGE",
                "severity": "warning",
                "stage": "generate_textures",
                "message": (
                    "Backend reported low target coverage with "
                    "Bearer SHOULD_NOT_SURFACE."
                ),
                "details": {
                    "debug_url": (
                        "https://backend.invalid/debug?api_key=SHOULD_NOT_SURFACE"
                    ),
                    "auth_url": (
                        "https://backend.invalid/auth?authorization=SHOULD_NOT_SURFACE"
                    ),
                    "headers": {"Authorization": "Bearer SHOULD_NOT_SURFACE"},
                },
            }
        ],
        "artifacts_manifest_path": str(
            tmp_path / "session" / "cache" / "artifacts_manifest.json"
        ),
        "output_usdz_path": str(tmp_path / "session" / "cache" / "output.usdz"),
    }

    step_stats = _extract_step_stats("generate_textures", context)
    final_stats = _extract_final_stats(context, tmp_path / "session")

    for stats in (step_stats, final_stats):
        assert stats["projection_backend_units"] == 1
        assert stats["projection_backend_map_counts"] == {"Aluminum_Matte": 3}
        assert stats["projection_backend_metadata"]["Aluminum_Matte"] == {
            "backend_name": "fake_projection_backend",
            "endpoint": "<configured>",
            "seed": 11631,
            "api_token": "<redacted>",
            "debug_url": "https://backend.invalid/status?token=<redacted>",
            "headers": {"Authorization": "<redacted>"},
            "credentials": "<redacted>",
        }
        serialized = json.dumps(stats, sort_keys=True)
        assert "SHOULD_NOT_SURFACE" not in serialized
        assert stats["projection_backend_diagnostics"][0]["code"] == (
            "BACKEND_LOW_COVERAGE"
        )
        assert stats["projection_backend_warnings"][0]["severity"] == "warning"

    assert final_stats["manifest_available"] is True
    assert final_stats["package_status"] == "succeeded"
    assert final_stats["output_usdz_available"] is True
    assert final_stats["render_available"] is False


def test_projection_backend_stats_ignore_malformed_context_shapes(
    tmp_path: Path,
) -> None:
    context = {
        "generated_textures": {},
        "projection_backend_results": ["malformed"],
        "generate_textures_diagnostics": {"malformed": True},
    }

    step_stats = _extract_step_stats("generate_textures", context)
    final_stats = _extract_final_stats(context, tmp_path / "session")

    assert step_stats == {"textures_generated": 0, "textures_failed": 0}
    assert "projection_backend_units" not in final_stats


def test_extract_render_stats_surfaces_bounded_diagnostics(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)
    camera_paths = [f"/Camera_{i}" for i in range(_MAX_RENDER_STATS_ITEMS + 2)]
    focus_cameras = [
        {"camera_path": path, "prim_path": f"/Root/Mesh_{i}"}
        for i, path in enumerate(camera_paths)
    ]
    render_error = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "RENDER_EMPTY_RESULT",
        "severity": "error",
        "stage": "render",
        "message": "Renderer returned no images",
    }
    render_warning = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "RENDER_NO_CAMERA",
        "severity": "warning",
        "stage": "render",
        "message": "Added fallback camera",
    }
    context = {
        "rendered_image_paths": [],
        "render_stats": {
            "render_available": False,
            "camera_paths": camera_paths,
            "focus_cameras": focus_cameras,
            "texture_detail_display_color_bakes": 3,
            "texture_detail_package_texture_localizations": 2,
            "texture_detail_uv_texture_fallbacks": 4,
            "textured_preview_fallbacks": 5,
        },
        "render_diagnostics": [render_warning, render_error],
        "render_errors": [render_error],
    }

    step_stats = _extract_step_stats("render", context)
    final_stats = _extract_final_stats(context, session_dir)

    assert step_stats["renders_count"] == 0
    assert step_stats["render_available"] is False
    assert step_stats["texture_detail_display_color_bakes"] == 3
    assert step_stats["texture_detail_package_texture_localizations"] == 2
    assert step_stats["texture_detail_uv_texture_fallbacks"] == 4
    assert step_stats["textured_preview_fallbacks"] == 5
    assert step_stats["camera_paths"] == camera_paths[:_MAX_RENDER_STATS_ITEMS]
    assert step_stats["focus_cameras"] == focus_cameras[:_MAX_RENDER_STATS_ITEMS]
    assert step_stats["diagnostics"] == [render_warning, render_error]
    assert step_stats["errors"] == [render_error]

    assert final_stats["render_available"] is False
    assert final_stats["texture_detail_display_color_bakes"] == 3
    assert final_stats["texture_detail_package_texture_localizations"] == 2
    assert final_stats["texture_detail_uv_texture_fallbacks"] == 4
    assert final_stats["textured_preview_fallbacks"] == 5
    assert final_stats["render_camera_paths"] == camera_paths[:_MAX_RENDER_STATS_ITEMS]
    assert (
        final_stats["render_focus_cameras"] == focus_cameras[:_MAX_RENDER_STATS_ITEMS]
    )
    assert final_stats["diagnostics"]["render"] == [render_warning, render_error]
    assert final_stats["errors"]["render"] == [render_error]


def test_extract_render_stats_surfaces_warning_only_diagnostics(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)
    render_warning = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "RENDER_FRAME_TOO_WIDE",
        "severity": "warning",
        "stage": "render",
        "message": "Focused render framing heuristic is below threshold",
    }
    context = {
        "rendered_image_paths": [],
        "render_stats": {"render_available": False},
        "render_diagnostics": [render_warning],
        "render_errors": [],
    }

    step_stats = _extract_step_stats("render", context)
    final_stats = _extract_final_stats(context, session_dir)

    assert "errors" not in step_stats
    assert step_stats["diagnostics"] == [render_warning]
    assert "errors" not in final_stats
    assert final_stats["diagnostics"]["render"] == [render_warning]


def test_extract_render_stats_classifies_mock_images_as_non_production(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)
    context = {
        "rendered_image_paths": [str(session_dir / "cache" / "renders" / "mock.png")],
        "render_stats": {
            "backend": "mock",
            "evidence_classification": "mock_placeholder",
            "production_visual_evidence": False,
            "render_available": True,
            "camera_paths": ["/Camera"],
            "focus_cameras": [],
        },
    }

    step_stats = _extract_step_stats("render", context)
    final_stats = _extract_final_stats(context, session_dir)

    for stats in (step_stats, final_stats):
        assert stats["render_available"] is True
        assert stats["backend"] == "mock"
        assert stats["evidence_classification"] == "mock_placeholder"
        assert stats["production_visual_evidence"] is False


def test_extract_step_stats_apply_textures_surfaces_mdl_overrides() -> None:
    """The apply_textures step must propagate MDL override counts into step
    stats and surface a `warnings` entry when SimReady-style
    pre-baked texture inputs had to be cleared, so /status and /results no
    longer silently succeed."""
    context = {
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "apply_textures_stats": {
            "applied_count": 8,
            "mdl_inputs_overridden": 2,
            "mdl_inputs_cleared": [
                "/Mat/Plastic_Blue_A:opacity_texture",
                "/Mat/Plastic_Blue_A:emissive_color_texture",
            ],
            "preview_texture_inputs_overridden": [
                "/Mat/Plastic_Blue_A/PreviewAlbedo:file"
            ],
        },
    }
    stats = _extract_step_stats("apply_textures", context)

    assert stats["output_usd_count"] == 1
    assert stats["mdl_inputs_overridden"] == 2
    assert stats["mdl_inputs_cleared"] == [
        "/Mat/Plastic_Blue_A:opacity_texture",
        "/Mat/Plastic_Blue_A:emissive_color_texture",
    ]
    assert stats["preview_texture_inputs_overridden"] == [
        "/Mat/Plastic_Blue_A/PreviewAlbedo:file"
    ]
    assert len(stats["warnings"]) == 1
    warning = stats["warnings"][0]
    assert "opacity_texture" in warning
    assert "emissive_color_texture" in warning


def test_extract_step_stats_apply_textures_no_mdl_inputs_no_warnings() -> None:
    """Materials without pre-baked MDL inputs (the common OpenPBR-only case)
    must not emit a warnings entry — that field is reserved for actual
    pipeline anomalies."""
    context = {
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "apply_textures_stats": {
            "applied_count": 3,
            "mdl_inputs_overridden": 0,
            "mdl_inputs_cleared": [],
            "mdl_inputs_localized": [],
        },
    }
    stats = _extract_step_stats("apply_textures", context)

    assert stats["output_usd_count"] == 1
    assert stats["mdl_inputs_overridden"] == 0
    assert "mdl_inputs_cleared" not in stats
    assert "mdl_inputs_localized" not in stats
    assert "warnings" not in stats


def test_extract_step_stats_apply_textures_localized_inputs_no_warning() -> None:
    """Localized MDL inputs (local files copied into the bundle textures dir)
    are reported as a count + list but must NOT trigger a warning — the bundle
    is self-consistent in that case."""
    context = {
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "apply_textures_stats": {
            "applied_count": 4,
            "mdl_inputs_overridden": 1,
            "mdl_inputs_cleared": [],
            "mdl_inputs_localized": ["/Mat/Plastic:opacity_texture"],
        },
    }
    stats = _extract_step_stats("apply_textures", context)

    assert stats["mdl_inputs_localized"] == ["/Mat/Plastic:opacity_texture"]
    assert "mdl_inputs_cleared" not in stats
    assert "warnings" not in stats


def test_extract_final_stats_persists_apply_textures_warnings() -> None:
    """Warnings emitted during apply_textures must survive into the final
    /results payload, not just the per-step
    stream. Otherwise clients polling /results after completion see a clean
    success and miss that MDL inputs were blanked."""
    session_dir = Path("/nonexistent")
    context = {
        "discovered_materials": [],
        "generated_textures": {},
        "output_usd_paths": ["/x/output/textured_output.usd"],
        "rendered_image_paths": [],
        "apply_textures_stats": {
            "applied_count": 1,
            "mdl_inputs_overridden": 2,
            "mdl_inputs_cleared": ["/Mat/X:opacity_texture"],
            "mdl_inputs_localized": ["/Mat/X:emissive_color_texture"],
            "preview_texture_inputs_overridden": ["/Mat/X/PreviewAlbedo:file"],
        },
    }

    stats = _extract_final_stats(context, session_dir)

    assert stats["mdl_inputs_overridden"] == 2
    assert stats["mdl_inputs_cleared"] == ["/Mat/X:opacity_texture"]
    assert stats["mdl_inputs_localized"] == ["/Mat/X:emissive_color_texture"]
    assert stats["preview_texture_inputs_overridden"] == ["/Mat/X/PreviewAlbedo:file"]
    assert len(stats["warnings"]) == 1
    assert "opacity_texture" in stats["warnings"][0]


def test_package_usdz_failure_surfaces_warning_in_final_stats(
    tmp_path: Path, monkeypatch
) -> None:
    """A USDZ packaging miss must be visible in /results, not logs only."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Usd, UsdUtils

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.DefinePrim("/Root", "Xform")
    stage.GetRootLayer().Save()

    monkeypatch.setattr(
        UsdUtils,
        "CreateNewUsdzPackage",
        lambda _src, _dst: False,
    )

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    stats = _extract_final_stats(context, tmp_path)

    assert usdz is None
    assert context["usdz_packaging_failed"] is True
    assert _artifact_manifest_status(context) == "partial"
    assert stats["package_status"] == "failed"
    assert stats["usdz_packaging_failed"] is True
    assert "Failed to create USDZ package" in stats["warnings"][0]
    assert "self-contained USDZ artifact was not produced" in stats["warnings"][0]
    assert stats["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"


def test_package_usdz_blocks_missing_relative_texture_refs(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    mat.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("../textures/missing.png"))
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    stats = _extract_final_stats(context, tmp_path)

    assert usdz is None
    assert context["output_portability"]["portable"] is False
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"
    assert _artifact_manifest_status(context) == "partial"
    assert stats["package_status"] == "failed"
    assert stats["usdz_packaging_failed"] is True
    assert stats["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"


def test_package_usdz_clears_missing_mdl_source_assets(tmp_path: Path) -> None:
    """Missing local MDL source assets must not prevent USDZ packaging."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/MDLShader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("OmniPBR.mdl"))
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)

    assert usdz is not None
    assert Path(usdz).exists()
    assert context["usdz_mdl_source_assets_cleared"] == [
        "/Root/Looks/Plastic/MDLShader.info:mdl:sourceAsset"
    ]

    rewritten_stage = Usd.Stage.Open(str(output_usd))
    rewritten_shader = rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/MDLShader")
    assert rewritten_shader.GetAttribute("info:mdl:sourceAsset").Get() is None


def test_package_usdz_clears_unbound_missing_material_layer_refs(
    tmp_path: Path,
) -> None:
    """Stale source material references should not ship as dangling USDZ deps."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stale_mat = UsdShade.Material.Define(stage, "/Root/Materials/physics_metal")
    stale_mat.GetPrim().GetReferences().AddReference("0/physics_metal.usda")
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, session_dir)
    context["output_usdz_path"] = usdz
    stats = _extract_final_stats(context, session_dir)

    assert usdz is not None
    assert Path(usdz).exists()
    assert context["usdz_layer_references_cleared"] == [
        "/Root/Materials/physics_metal: 0/physics_metal.usda"
    ]
    assert stats["package_status"] == "succeeded"
    assert stats["usdz_layer_references_cleared"] == [
        "/Root/Materials/physics_metal: 0/physics_metal.usda"
    ]

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    prim_spec = layer.GetPrimAtPath("/Root/Materials/physics_metal")
    assert list(prim_spec.referenceList.prependedItems) == []


def test_package_usdz_caps_cleared_layer_reference_payload(
    tmp_path: Path,
) -> None:
    """Layer-reference diagnostics keep total counts without oversized payloads."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    total_refs = _MAX_ERRORS_IN_PAYLOAD + 3
    for index in range(total_refs):
        mat = UsdShade.Material.Define(stage, f"/Root/Materials/stale_{index}")
        mat.GetPrim().GetReferences().AddReference(f"{index}/missing.usda")
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, session_dir)
    context["output_usdz_path"] = usdz
    stats = _extract_final_stats(context, session_dir)

    assert usdz is not None
    assert context["usdz_layer_references_cleared_count"] == total_refs
    assert len(context["usdz_layer_references_cleared"]) == _MAX_ERRORS_IN_PAYLOAD
    assert stats["usdz_layer_references_cleared_count"] == total_refs
    assert len(stats["usdz_layer_references_cleared"]) == _MAX_ERRORS_IN_PAYLOAD
    assert (
        f"... and {total_refs - _MAX_ERRORS_IN_PAYLOAD} more" in stats["warnings"][-1]
    )
    assert f"{total_refs - 1}/missing.usda" not in stats["warnings"][-1]


def test_package_usdz_blocks_bound_missing_material_layer_refs(
    tmp_path: Path,
) -> None:
    """A bound missing material layer is a real package error, not cleanup."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Materials/physics_metal")
    mat.GetPrim().GetReferences().AddReference("0/physics_metal.usda")
    geom = stage.DefinePrim("/Root/Geometry/Bucket", "Mesh")
    UsdShade.MaterialBindingAPI.Apply(geom).Bind(mat)
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, session_dir)
    stats = _extract_final_stats(context, session_dir)

    assert usdz is None
    assert context["usdz_packaging_failed"] is True
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"
    assert "0/physics_metal.usda" in context["package_diagnostics"][0]["message"]
    assert stats["package_status"] == "failed"


def test_package_usdz_clears_missing_physics_material_layer_refs(
    tmp_path: Path,
) -> None:
    """Nonvisual physics bindings must not block visual USDZ packaging."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    physics_mat = UsdShade.Material.Define(stage, "/Root/Materials/physics_metal")
    physics_mat.GetPrim().GetReferences().AddReference("0/physics_metal.usda")
    visual_mat = UsdShade.Material.Define(stage, "/Root/Materials/visual_metal")
    geom = stage.DefinePrim("/Root/Geometry/Bucket", "Mesh")
    UsdShade.MaterialBindingAPI.Apply(geom).Bind(visual_mat)
    geom.CreateRelationship("material:binding:physics").SetTargets(
        [physics_mat.GetPath()]
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, session_dir)

    assert usdz is not None
    assert context["usdz_layer_references_cleared"] == [
        "/Root/Materials/physics_metal: 0/physics_metal.usda"
    ]

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    prim_spec = layer.GetPrimAtPath("/Root/Materials/physics_metal")
    assert list(prim_spec.referenceList.prependedItems) == []


def test_package_usdz_localizes_upload_bundle_layer_refs(
    tmp_path: Path,
) -> None:
    """Resolvable local USD layer refs from the upload bundle are bundled."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    input_scene_dir = session_dir / "input" / ".step1x_package_assets" / "scene"
    source_layer_dir = input_scene_dir / "0"
    output_dir.mkdir(parents=True)
    source_layer_dir.mkdir(parents=True)
    (source_layer_dir / "physics_metal.usda").write_text(
        '#usda 1.0\n\ndef Material "PhysicsMetal"\n{\n}\n',
        encoding="utf-8",
    )

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Materials/physics_metal")
    mat.GetPrim().GetReferences().AddReference("0/physics_metal.usda")
    stage.GetRootLayer().Save()

    context = {
        "usd_path": str(input_scene_dir / "scene.usda"),
        "output_usd_paths": [str(output_usd)],
    }
    usdz = _package_usdz(context, session_dir)

    assert usdz is not None
    assert (output_dir / "0" / "physics_metal.usda").is_file()
    assert context["usdz_layer_references_localized"] == [
        "/Root/Materials/physics_metal: 0/physics_metal.usda"
    ]

    layer = Sdf.Layer.FindOrOpen(str(output_usd))
    prim_spec = layer.GetPrimAtPath("/Root/Materials/physics_metal")
    assert [ref.assetPath for ref in prim_spec.referenceList.prependedItems] == [
        "0/physics_metal.usda"
    ]


def test_write_service_artifact_manifest_sanitizes_and_updates_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from ...service.workers import executor

    monkeypatch.setattr(
        executor.service_config,
        "session_storage_path",
        str(tmp_path / "session"),
    )
    cache = tmp_path / "session" / "cache"
    cache.mkdir(parents=True)
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "session" / "input" / "scene.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "https://abc.invocation.api.nvcf.nvidia.com/v1",
            "custom_parameters": {"api_key": "SHOULD_NOT_SURFACE"},
        },
        "package_diagnostics": [
            {
                "schema_version": "texture-agent-diagnostic.v1",
                "code": "PACKAGE_MISSING_ARTIFACT",
                "severity": "error",
                "stage": "package",
                "message": f"missing {tmp_path / 'session' / 'cache' / 'textures' / 'x.png'}",
                "recommended_action": "inspect",
                "details": {},
            }
        ],
    }

    manifest = _write_service_artifact_manifest(
        context,
        status="failed",
        service_urls={"manifest": "/artifacts/sid/manifest"},
    )

    assert manifest == context["artifacts_manifest_path"]
    payload = Path(manifest).read_text(encoding="utf-8")
    assert "texture-agent-artifacts.v1" in payload
    assert "SHOULD_NOT_SURFACE" not in payload
    assert "<session>" in payload


def test_package_usdz_rewrites_string_and_token_png_paths(tmp_path: Path) -> None:
    """Codex round-8 finding: the packager only rewrote `Sdf.AssetPath`
    PNG attributes, leaving absolute cache paths in string/token-typed
    MDL texture inputs after download. Now string and token attributes
    are also rewritten to bundle-relative `../textures/<basename>` form.
    """
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    from PIL import Image

    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_albedo.PNG")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_normal.PNG")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_orm.png")

    # We use UsdPreviewSurface (not MDL) so USDZ packaging does not chase
    # an unresolvable `omniverse://...mdl` dep — the test focuses on the
    # path-rewriting behaviour, not the MDL resolution path.
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # Pre-rewrite shapes the packager must handle: absolute Asset, absolute
    # String, absolute Token — all PNG paths under the cache textures dir.
    shader.CreateInput("diffuseColor_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(textures_dir / "Plastic_albedo.PNG"))
    )
    shader.CreateInput("normal_texture", Sdf.ValueTypeNames.String).Set(
        str(textures_dir / "Plastic_normal.PNG")
    )
    shader.CreateInput("orm_texture", Sdf.ValueTypeNames.Token).Set(
        str(textures_dir / "Plastic_orm.png")
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is not None
    assert Path(usdz).exists()

    # Re-read the rewritten USD and confirm all three inputs were rewritten
    # to ../textures/<basename> (bundle-relative), regardless of authored
    # type.
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert (
        out_shader.GetInput("diffuseColor_texture").Get().path
        == "../textures/Plastic_albedo.PNG"
    )
    assert (
        out_shader.GetInput("normal_texture").Get() == "../textures/Plastic_normal.PNG"
    )
    assert out_shader.GetInput("orm_texture").Get() == "../textures/Plastic_orm.png"


def test_downloaded_usdz_preserves_active_preview_texture_graph(
    tmp_path: Path,
) -> None:
    """Backend-neutral blended maps remain active in the downloadable USDZ."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from texture_agent.functions.material_discovery import (
        MaterialInfo,
        PrimTextureUnit,
    )
    from texture_agent.tasks.apply_textures import ApplyTexturesTask
    from texture_agent.tasks.blend_textures import BlendedTextures
    from texture_agent.tasks.prepare_uvs import PrepareUVsTask

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    working_dir = session_dir / "cache"
    textures_dir = working_dir / "textures"
    input_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    input_path = input_dir / "constant_preview.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Root/LabelMesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 0.0),
            Gf.Vec3f(0.0, 1.0, 0.0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    material_path = "/Root/Looks/Label"
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.96, 0.96, 0.96)
    )
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).Set(
        Gf.Vec3f(0.0, 0.0, 1.0)
    )
    surface.CreateInput("occlusion", Sdf.ValueTypeNames.Float).Set(1.0)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    map_files = {
        "albedo": textures_dir / "Label_albedo.png",
        "normal": textures_dir / "Label_normal.png",
        "orm": textures_dir / "Label_orm.png",
    }
    Image.new("RGB", (8, 8), (210, 30, 25)).save(map_files["albedo"])
    Image.new("RGB", (8, 8), (128, 128, 255)).save(map_files["normal"])
    Image.new("RGB", (8, 8), (255, 90, 15)).save(map_files["orm"])

    unit = PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path=material_path,
            name="Label",
            bound_prim_paths=["/Root/LabelMesh"],
            base_color=(0.96, 0.96, 0.96),
            base_metalness=0.0,
            specular_roughness=0.6,
        ),
        key="Label",
        prompt="red printed equipment label",
        opacity=1.0,
    )
    context = {
        "usd_path": str(input_path),
        "blended_textures": {
            "Label": BlendedTextures(
                albedo=str(map_files["albedo"]),
                normal=str(map_files["normal"]),
                orm=str(map_files["orm"]),
            )
        },
        "prim_texture_units": [unit],
        "working_dir": str(working_dir),
        "texture_config": {
            "uv_policy": "generate_missing",
            "uv_projection": "planar",
        },
    }
    PrepareUVsTask().run(context)
    context = ApplyTexturesTask().run(context)

    usdz_path = _package_usdz(context, session_dir)
    assert usdz_path is not None
    assert Path(usdz_path).is_file()

    with zipfile.ZipFile(usdz_path) as package:
        package_members = package.namelist()

    downloaded_stage = Usd.Stage.Open(usdz_path)
    assert downloaded_stage is not None
    downloaded_mesh = downloaded_stage.GetPrimAtPath("/Root/LabelMesh")
    downloaded_st = UsdGeom.PrimvarsAPI(downloaded_mesh).GetPrimvar("st")
    assert downloaded_st.HasAuthoredValue()
    assert downloaded_st.GetInterpolation() == "faceVarying"
    assert len(downloaded_st.ComputeFlattened()) == 4
    bound_material, _binding_rel = UsdShade.MaterialBindingAPI(
        downloaded_mesh
    ).ComputeBoundMaterial()
    assert bound_material.GetPath() == Sdf.Path(material_path)

    downloaded_material = UsdShade.Material(
        downloaded_stage.GetPrimAtPath(material_path)
    )
    universal_surface = downloaded_material.GetSurfaceOutput()
    surface_source = universal_surface.GetConnectedSource()
    assert surface_source is not None
    downloaded_surface = UsdShade.Shader(surface_source[0].GetPrim())
    assert downloaded_surface.GetIdAttr().Get() == "UsdPreviewSurface"

    expected_connections = {
        "diffuseColor": ("Label_albedo.png", "rgb", "sRGB"),
        "normal": ("Label_normal.png", "rgb", "raw"),
        "occlusion": ("Label_orm.png", "r", "raw"),
        "roughness": ("Label_roughness.png", "r", "raw"),
        "metallic": ("Label_metalness.png", "r", "raw"),
    }
    st_reader_paths: set[Sdf.Path] = set()
    for input_name, (
        expected_file,
        expected_output,
        expected_color_space,
    ) in expected_connections.items():
        texture_source = downloaded_surface.GetInput(input_name).GetConnectedSource()
        assert texture_source is not None
        texture = UsdShade.Shader(texture_source[0].GetPrim())
        assert texture.GetIdAttr().Get() == "UsdUVTexture"
        assert str(texture_source[1]) == expected_output
        assert texture.GetInput("sourceColorSpace").Get() == expected_color_space

        packaged_asset = texture.GetInput("file").Get()
        assert Path(packaged_asset.path).name == expected_file
        matching_members = [
            member
            for member in package_members
            if member.rsplit("/", 1)[-1] == expected_file
        ]
        assert len(matching_members) == 1
        assert matching_members[0] in packaged_asset.resolvedPath

        st_source = texture.GetInput("st").GetConnectedSource()
        assert st_source is not None
        assert str(st_source[1]) == "result"
        st_reader = UsdShade.Shader(st_source[0].GetPrim())
        assert st_reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
        assert st_reader.GetInput("varname").Get() == "st"
        st_reader_paths.add(st_reader.GetPrim().GetPath())

    assert len(st_reader_paths) == 1
    normal_texture = UsdShade.Shader(
        downloaded_surface.GetInput("normal").GetConnectedSource()[0].GetPrim()
    )
    assert tuple(normal_texture.GetInput("scale").Get()) == (2.0, 2.0, 2.0, 2.0)
    assert tuple(normal_texture.GetInput("bias").Get()) == (-1.0, -1.0, -1.0, 0.0)


def test_package_usdz_does_not_rewrite_unrelated_string_attrs(tmp_path: Path) -> None:
    """Codex round-11 finding: the string/token rewrite must be scoped to
    Shader `inputs:*_texture` attributes. A non-shader string attribute, or
    a shader string attribute with a different name, that happens to end in
    ``.png`` must NOT be rewritten — those have no Asset-typed dep, so a
    rewrite would create a dangling USDZ ref.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))

    # Non-Shader prim, string attribute that happens to end in .png.
    meta_prim = stage.DefinePrim("/Root/Metadata", "Scope")
    meta_prim.CreateAttribute("note", Sdf.ValueTypeNames.String).Set(
        "see /assets/library/reference.png for the source"
    )

    # Shader prim with a string input named other than `inputs:*_texture`.
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("debug_label", Sdf.ValueTypeNames.String).Set("fallback.png")
    # Shader `inputs:*_texture` string — this one *is* in scope and must
    # be rewritten.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        str(textures_dir / "Plastic_albedo.png")
    )

    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is not None
    rewritten_stage = Usd.Stage.Open(str(output_usd))

    note = rewritten_stage.GetPrimAtPath("/Root/Metadata").GetAttribute("note").Get()
    assert note == "see /assets/library/reference.png for the source"

    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("debug_label").Get() == "fallback.png"
    assert (
        out_shader.GetInput("diffuse_texture").Get() == "../textures/Plastic_albedo.png"
    )


def test_package_usdz_skips_string_inputs_with_missing_files(tmp_path: Path) -> None:
    """Codex round-13 finding: even after the round-12 scope narrowing
    (Shader + `inputs:*_texture`), the packager could rewrite a string
    texture input on a non-MDL shader (or any shader skipped by
    apply_textures) to a `../textures/<basename>.png` path that the
    bundle does not actually ship. Now the packager additionally
    requires the basename to exist in `cache/textures/` before
    rewriting, so unrelated/skipped string texture refs are left as
    authored. A31-1 then blocks USDZ packaging with a structured
    PACKAGE_* diagnostic because the output is not self-contained.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    # Only the in-bundle texture exists.
    Image.new("RGB", (4, 4), (1, 2, 3)).save(textures_dir / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # In-bundle reference: must be rewritten.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        str(textures_dir / "Plastic_albedo.png")
    )
    # Out-of-scope shader string `inputs:*_texture` whose target does NOT
    # live in cache/textures: the packager must NOT rewrite this, since
    # USDZ packaging would not bundle the file and the relative rewrite
    # would dangle on the customer's machine. A31-1 should surface that
    # as a package diagnostic instead of shipping a bad archive.
    shader.CreateInput("mask_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/mask.png"
    )
    shader.CreateInput("emissive_texture", Sdf.ValueTypeNames.String).Set(
        "/private/path/that_does_not_exist.png"
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # In-bundle: rewritten.
    assert (
        out_shader.GetInput("diffuse_texture").Get() == "../textures/Plastic_albedo.png"
    )
    # Out-of-bundle: untouched.
    assert (
        out_shader.GetInput("mask_texture").Get()
        == "omniverse://nucleus.example/mask.png"
    )
    assert (
        out_shader.GetInput("emissive_texture").Get()
        == "/private/path/that_does_not_exist.png"
    )


def test_package_usdz_does_not_substitute_basename_collision(tmp_path: Path) -> None:
    """Codex round-15 finding: rewriting a string-typed shader input by
    basename match alone could silently substitute the wrong texture if
    the user has another local PNG whose basename happens to collide
    with a generated/localized file. The packager now resolves the
    *original* path and only rewrites when it lives under the session's
    own ``cache/textures`` directory.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    # The agent's generated file.
    Image.new("RGB", (4, 4), (200, 50, 50)).save(textures_dir / "Plastic_albedo.png")

    # An unrelated PNG that happens to share the basename, parked in a
    # totally separate directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    Image.new("RGB", (4, 4), (10, 200, 10)).save(elsewhere / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    # The user's intentional reference to elsewhere/Plastic_albedo.png.
    # Even though `cache/textures/Plastic_albedo.png` exists, the
    # packager must NOT substitute this string with `../textures/...`.
    # A31-1 should block USDZ packaging rather than ship an archive with
    # a host-local absolute path.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        str(elsewhere / "Plastic_albedo.png")
    )
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)
    assert usdz is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # Untouched — the original elsewhere/ path survived.
    assert out_shader.GetInput("diffuse_texture").Get() == str(
        elsewhere / "Plastic_albedo.png"
    )


def test_package_usdz_does_not_rewrite_out_of_bundle_asset_paths(
    tmp_path: Path,
) -> None:
    """Asset-typed refs must pass the same containment gate as string refs.

    Otherwise an absolute host-local AssetPath could be rewritten by basename
    to an in-bundle texture and pass portability validation with the wrong file.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (200, 50, 50)).save(textures_dir / "Plastic_albedo.png")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    Image.new("RGB", (4, 4), (10, 200, 10)).save(elsewhere / "Plastic_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    mat.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(str(elsewhere / "Plastic_albedo.png")))
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    usdz = _package_usdz(context, tmp_path)

    assert usdz is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_mat = UsdShade.Material(rewritten_stage.GetPrimAtPath("/Root/Looks/Plastic"))
    out_ref = out_mat.GetPrim().GetAttribute("inputs:base_color_texture_file").Get()
    assert out_ref.path == str(elsewhere / "Plastic_albedo.png")


def test_package_usdz_localizes_step1x_upload_bundle_texture_refs(
    tmp_path: Path,
) -> None:
    """Step1X package outputs can preserve original USD-relative texture refs.

    The referenced PNGs are not generated textures, but they are safe to bundle
    when they resolve under this session's uploaded/extracted input tree.
    """
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    input_scene_dir = session_dir / "input" / ".step1x_package_assets" / "scene"
    upload_textures_dir = input_scene_dir / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    upload_textures_dir.mkdir(parents=True)

    Image.new("RGB", (4, 4), (240, 210, 20)).save(
        upload_textures_dir / "trim_plastic_yellow_02_a.png"
    )

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Trim/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/trim_plastic_yellow_02_a.png")
    )
    stage.GetRootLayer().Save()

    context = {
        "usd_path": str(input_scene_dir / "scene.usda"),
        "output_usd_paths": [str(output_usd)],
    }
    usdz = _package_usdz(context, session_dir)

    assert usdz is not None
    assert Path(usdz).exists()
    localized = textures_dir / "trim_plastic_yellow_02_a.png"
    assert localized.is_file()
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Trim/Shader")
    )
    assert (
        out_shader.GetInput("diffuse_texture").Get().path
        == "../textures/trim_plastic_yellow_02_a.png"
    )
    assert context["output_portability"]["portable"] is True


def test_package_usdz_localizes_original_texture_refs_from_uploaded_usdz(
    tmp_path: Path,
) -> None:
    """Uploaded USDZ textures are valid sources for unedited material refs."""
    import zipfile

    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    input_dir = session_dir / "input"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    source_png = tmp_path / "trim_plastic_yellow_02_a.png"
    Image.new("RGB", (4, 4), (240, 210, 20)).save(source_png)
    input_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(input_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_png, "textures/trim_plastic_yellow_02_a.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Trim/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/trim_plastic_yellow_02_a.png")
    )
    stage.GetRootLayer().Save()

    context = {
        "usd_path": str(input_usdz),
        "output_usd_paths": [str(output_usd)],
    }
    usdz = _package_usdz(context, session_dir)

    assert usdz is not None
    assert (textures_dir / "trim_plastic_yellow_02_a.png").is_file()
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Trim/Shader")
    )
    assert (
        out_shader.GetInput("diffuse_texture").Get().path
        == "../textures/trim_plastic_yellow_02_a.png"
    )
    assert context["output_portability"]["portable"] is True


def test_package_usdz_reuses_localized_step1x_upload_texture_refs(
    tmp_path: Path,
) -> None:
    """Repeated refs to one uploaded atlas must package one copy, not many."""
    import pytest

    pytest.importorskip("pxr")
    import zipfile

    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    input_scene_dir = session_dir / "input" / ".step1x_package_assets" / "scene"
    upload_textures_dir = input_scene_dir / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    upload_textures_dir.mkdir(parents=True)

    atlas_name = "trim_plastic_yellow_02_a.png"
    Image.new("RGB", (4, 4), (240, 210, 20)).save(upload_textures_dir / atlas_name)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    for suffix in ("A", "B"):
        shader = UsdShade.Shader.Define(stage, f"/Root/Looks/Trim{suffix}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(f"./textures/{atlas_name}")
        )
    stage.GetRootLayer().Save()

    context = {
        "usd_path": str(input_scene_dir / "scene.usda"),
        "output_usd_paths": [str(output_usd)],
    }
    usdz = _package_usdz(context, session_dir)

    assert usdz is not None
    localized = sorted(textures_dir.glob("trim_plastic_yellow_02_a*.png"))
    assert [path.name for path in localized] == [atlas_name]

    rewritten_stage = Usd.Stage.Open(str(output_usd))
    rewritten_paths = []
    for suffix in ("A", "B"):
        shader = UsdShade.Shader(
            rewritten_stage.GetPrimAtPath(f"/Root/Looks/Trim{suffix}/Shader")
        )
        rewritten_paths.append(shader.GetInput("diffuse_texture").Get().path)
    assert rewritten_paths == [f"../textures/{atlas_name}", f"../textures/{atlas_name}"]

    with zipfile.ZipFile(usdz) as package:
        packaged_atlases = [
            name for name in package.namelist() if name.endswith(atlas_name)
        ]
    assert packaged_atlases == [f"0/{atlas_name}"]


def test_package_usdz_does_not_localize_out_of_session_relative_collision(
    tmp_path: Path,
) -> None:
    """A matching filename elsewhere on disk must not satisfy a package ref."""
    import pytest

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    cache = session_dir / "cache"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (200, 50, 50)).save(textures_dir / "collision.png")

    elsewhere_textures = tmp_path / "elsewhere" / "textures"
    elsewhere_textures.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 200, 10)).save(elsewhere_textures / "collision.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Trim/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/collision.png")
    )
    stage.GetRootLayer().Save()

    context = {
        "usd_path": str(session_dir / "input" / "scene.usda"),
        "output_usd_paths": [str(output_usd)],
    }
    usdz = _package_usdz(context, session_dir)

    assert usdz is None
    assert Image.open(textures_dir / "collision.png").getpixel((0, 0)) == (
        200,
        50,
        50,
    )
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Trim/Shader")
    )
    assert out_shader.GetInput("diffuse_texture").Get().path == (
        "./textures/collision.png"
    )


def test_extract_final_stats_no_apply_textures_stats_no_warning(
    tmp_path: Path,
) -> None:
    """Sessions that ran without apply_textures (or where the step recorded no
    MDL anomalies) must not emit a warnings entry into /results."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats({"output_usd_paths": ["/x.usd"]}, session_dir)

    assert "warnings" not in stats
    assert "mdl_inputs_overridden" not in stats
    assert "mdl_inputs_cleared" not in stats
    assert "mdl_inputs_localized" not in stats


def test_extract_final_stats_surfaces_partial_generate_failures(
    tmp_path: Path,
) -> None:
    """A run that completed below the threshold (e.g. 1 success + 3
    failures with default 1.0) must still expose the structured failures
    on the persisted final stats. Otherwise GET /result/{session_id} after
    the SSE snapshot has been GC'd looks identical to a clean run."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats(
        {
            "generated_textures": {"Good": object()},
            "generate_textures_failed_count": 3,
            "generate_textures_errors": [
                {
                    "material": "BadA",
                    "type": "RuntimeError",
                    "status": 500,
                    "message": "x",
                },
            ],
        },
        session_dir,
    )

    assert stats["textures_generated"] == 1
    assert stats["textures_generated_failed"] == 3
    assert stats["textures_failed"] == 3
    assert "generate_textures" in stats["errors"]
    assert stats["errors"]["generate_textures"][0]["status"] == 500


def test_extract_final_stats_sums_gen_and_blend_failure_counts(
    tmp_path: Path,
) -> None:
    """When both gen and blend partial-fail (different units), the
    top-level ``textures_failed`` must be the SUM, not just blend's.
    Otherwise an upstream auth issue (gen 403s) is hidden the moment
    blend introduces any of its own failures, defeating the purpose of
    the field."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats(
        {
            "generated_textures": {"Good1": object()},
            "generate_textures_failed_count": 2,
            "generate_textures_errors": [
                {
                    "material": "GenBadA",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "auth",
                },
                {
                    "material": "GenBadB",
                    "type": "RuntimeError",
                    "status": 403,
                    "message": "auth",
                },
            ],
            "blend_textures_failed_count": 1,
            "blend_textures_errors": [
                {
                    "material": "BlendBadA",
                    "type": "MissingAlbedo",
                    "status": None,
                    "message": "x",
                },
            ],
        },
        session_dir,
    )

    assert stats["textures_generated_failed"] == 2
    assert stats["textures_blended_failed"] == 1
    assert stats["textures_failed"] == 3
    assert set(stats["errors"]) == {"generate_textures", "blend_textures"}


def test_extract_final_stats_omits_failure_keys_when_no_errors(
    tmp_path: Path,
) -> None:
    """Happy-path runs must not gain new top-level keys -- existing
    consumers should see the same shape they always have."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    stats = _extract_final_stats(
        {
            "generated_textures": {"Good": object()},
            "generate_textures_failed_count": 0,
        },
        session_dir,
    )

    assert "textures_failed" not in stats
    assert "errors" not in stats


def test_truncate_errors_caps_list_length() -> None:
    """Per-prim mode with backend-wide outage can produce thousands of
    error records. Persisted payloads (session.json, event_log.jsonl,
    SSE) must cap them while leaving the count visible elsewhere."""
    errors = [
        {"material": f"m{i}", "type": "T", "status": 500, "message": "x"}
        for i in range(_MAX_ERRORS_IN_PAYLOAD * 4)
    ]
    out = _truncate_errors(errors)
    assert len(out) == _MAX_ERRORS_IN_PAYLOAD


def test_truncate_errors_truncates_long_messages() -> None:
    long_msg = "X" * (_MAX_ERROR_MESSAGE_CHARS * 5)
    errors = [{"material": "m", "type": "T", "status": 500, "message": long_msg}]
    out = _truncate_errors(errors)
    assert out[0]["message"].endswith("...(truncated)")
    assert len(out[0]["message"]) <= _MAX_ERROR_MESSAGE_CHARS + len("...(truncated)")


def test_truncate_errors_preserves_short_messages_unchanged() -> None:
    record = {"material": "m", "type": "HTTPError", "status": 403, "message": "x"}
    out = _truncate_errors([record])
    assert out == [record]


def test_extract_final_stats_truncates_oversized_error_lists(tmp_path: Path) -> None:
    """A 1000-prim per-prim run with an all-fail backend must NOT
    persist 1000 error records to /result. The count survives via
    ``textures_generated_failed`` / ``textures_failed``."""
    session_dir = tmp_path / "session"
    (session_dir / "cache").mkdir(parents=True)

    errors = [
        {"material": f"m{i}", "type": "T", "status": 500, "message": "x"}
        for i in range(1000)
    ]
    stats = _extract_final_stats(
        {
            "generated_textures": {},
            "generate_textures_failed_count": 1000,
            "generate_textures_errors": errors,
        },
        session_dir,
    )

    assert stats["textures_generated_failed"] == 1000
    assert stats["textures_failed"] == 1000
    assert len(stats["errors"]["generate_textures"]) == _MAX_ERRORS_IN_PAYLOAD
