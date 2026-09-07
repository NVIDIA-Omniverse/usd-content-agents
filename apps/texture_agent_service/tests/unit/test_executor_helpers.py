# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
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
    _map_sdf_paths_in_value,
    _package_usdz,
    _prepare_config_and_context,
    _prepare_source_usdz_stage,
    _redact_diagnostics_for_stats,
    _requires_executable_texture_plan,
    _resolver_stable_spec_value,
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


def test_resolver_stable_spec_value_normalizes_nested_asset_paths() -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf

    member_paths = {
        "/extract/Scene/legacy.png": "Scene/legacy.png",
        "/input/source.usdz[Scene/legacy.png]": "Scene/legacy.png",
    }

    def asset_path_signature(path_value: str) -> tuple[str, str]:
        member = member_paths.get(path_value)
        return (
            ("source-package-member", member)
            if member is not None
            else ("authored-path", path_value)
        )

    def normalized(value: Any) -> Any:
        return _resolver_stable_spec_value(
            value,
            sdf=Sdf,
            asset_path_signature=asset_path_signature,
        )

    extracted = Sdf.AssetPath("/extract/Scene/legacy.png")
    packaged = Sdf.AssetPath("/input/source.usdz[Scene/legacy.png]")
    assert normalized(
        {
            "array": Sdf.AssetPathArray([extracted]),
            "list": [extracted],
            "tuple": (extracted,),
            "set": {extracted},
            "plain": 1,
        }
    ) == normalized(
        {
            "array": Sdf.AssetPathArray([packaged]),
            "list": [packaged],
            "tuple": (packaged,),
            "set": {packaged},
            "plain": 1,
        }
    )

    explicit_paths = Sdf.PathListOp()
    explicit_paths.explicitItems = [Sdf.Path("/World/A.outputs:value")]
    prepended_paths = Sdf.PathListOp()
    prepended_paths.prependedItems = [Sdf.Path("/World/B.outputs:value")]
    assert normalized(Sdf.Path("relative/path")) == (
        "Sdf.Path",
        "relative/path",
    )
    assert normalized(explicit_paths) == (
        "Sdf.PathListOp",
        (("explicitItems", (("Sdf.Path", "/World/A.outputs:value"),)),),
    )
    normalized_prepended = normalized(prepended_paths)
    assert normalized_prepended[0] == "Sdf.PathListOp"
    assert ("prependedItems", (("Sdf.Path", "/World/B.outputs:value"),)) in (
        normalized_prepended[1]
    )
    assert normalized(Sdf.AssetPath("same.png", "/resolved/a")) == normalized(
        Sdf.AssetPath("same.png", "/resolved/b")
    )
    assert normalized(Sdf.AssetPath("same.png")) != normalized(
        Sdf.AssetPath("changed.png")
    )


def test_sdf_composition_arc_signatures_remap_synthetic_paths_semantically() -> None:
    """F1/F2 arc metadata compares by content, including target-path suffixes."""
    pytest.importorskip("pxr")
    from pxr import Sdf

    source_prefix = Sdf.Path("/Flattened_Prototype_2")
    destination_prefix = Sdf.Path("/Flattened_Prototype_1")

    def remap(path: Sdf.Path) -> Sdf.Path:
        return (
            path.ReplacePrefix(source_prefix, destination_prefix)
            if path.IsAbsolutePath() and path.HasPrefix(source_prefix)
            else path
        )

    source_target = Sdf.Path(
        "/Flattened_Prototype_1.rel[/Flattened_Prototype_1/Child].value"
    )
    edited_target = Sdf.Path(
        "/Flattened_Prototype_2.rel[/Flattened_Prototype_2/Child].value"
    )
    source_reference = Sdf.Reference(
        "",
        Sdf.Path("/Flattened_Prototype_1"),
        Sdf.LayerOffset(2.0, 3.0),
        {"target": source_target},
    )
    edited_reference = Sdf.Reference(
        "",
        Sdf.Path("/Flattened_Prototype_2"),
        Sdf.LayerOffset(2.0, 3.0),
        {"target": edited_target},
    )
    source_references = Sdf.ReferenceListOp.CreateExplicit([source_reference])
    edited_references = Sdf.ReferenceListOp.CreateExplicit([edited_reference])
    remapped_references = _map_sdf_paths_in_value(
        edited_references,
        sdf=Sdf,
        map_path=remap,
    )

    def signature(value: Any) -> Any:
        return _resolver_stable_spec_value(
            value,
            sdf=Sdf,
            asset_path_signature=lambda path: ("asset", path),
        )

    assert remapped_references.explicitItems == source_references.explicitItems
    assert remapped_references.explicitItems[0].customData["target"] == source_target
    assert signature(remapped_references) == signature(source_references)

    different_arc = Sdf.ReferenceListOp.CreateExplicit(
        [Sdf.Reference("", Sdf.Path("/Actually_Different"))]
    )
    assert signature(different_arc) != signature(source_references)

    payloads = Sdf.PayloadListOp()
    payloads.prependedItems = [Sdf.Payload("", Sdf.Path("/Flattened_Prototype_2"))]
    remapped_payloads = _map_sdf_paths_in_value(
        payloads,
        sdf=Sdf,
        map_path=remap,
    )
    assert remapped_payloads.prependedItems == [
        Sdf.Payload("", Sdf.Path("/Flattened_Prototype_1"))
    ]
    assert signature(remapped_payloads)[0] == "Sdf.PayloadListOp"

    external_reference = Sdf.Reference(
        "external.usda",
        Sdf.Path("/Flattened_Prototype_2"),
    )
    assert _map_sdf_paths_in_value(
        external_reference,
        sdf=Sdf,
        map_path=remap,
    ).primPath == Sdf.Path("/Flattened_Prototype_2")

    nested = _map_sdf_paths_in_value(
        (
            [Sdf.Path("/Flattened_Prototype_2/Child")],
            {
                "relative": Sdf.Path("Child"),
                Sdf.Path("/Flattened_Prototype_2/Key"): Sdf.Path(
                    "/Flattened_Prototype_2/Value"
                ),
            },
            7,
        ),
        sdf=Sdf,
        map_path=remap,
    )
    assert nested == (
        [Sdf.Path("/Flattened_Prototype_1/Child")],
        {
            "relative": Sdf.Path("Child"),
            Sdf.Path("/Flattened_Prototype_1/Key"): Sdf.Path(
                "/Flattened_Prototype_1/Value"
            ),
        },
        7,
    )


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

    invalid_plan = tmp_path / "invalid-plan.json"
    invalid_plan.write_text("{", encoding="utf-8")
    context["texture_plan_path"] = str(invalid_plan)

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
    assert context["source_usd_path"] == "/tmp/input.usd"
    assert context["usd_dependency_root"] == str(session_dir / "input")
    assert context["service_managed_usdz_reconstruction"] is True
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


def test_prepare_config_and_context_ignores_stale_plan_for_fresh_run(
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

    assert "texture_plan_path" not in context


@pytest.mark.parametrize(
    "resume_flag",
    ["resume_execution", "resume_apply_textures"],
)
def test_prepare_config_and_context_reuses_plan_only_for_resume(
    tmp_path: Path,
    resume_flag: str,
) -> None:
    session_dir = tmp_path / "session"
    plan_path = session_dir / "cache" / "texture_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}", encoding="utf-8")

    _config, context = _prepare_config_and_context(
        {
            "input": {"usd_path": "/tmp/input.usd"},
            "planning": {resume_flag: True},
        },
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


def test_prepare_config_filters_unsupported_llm_reasoning_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "session"
    hosted_backend = "nvidia" + "_inference"
    monkeypatch.setenv("TA_LLM_BACKEND", "nim")
    monkeypatch.setenv("TA_LLM_MODEL", "nvidia/cosmos-reason2-8b")
    monkeypatch.setenv("TA_LLM_REASONING_EFFORT", "xhigh")

    config, _context = _prepare_config_and_context(
        {
            "input": {"usd_path": "/tmp/input.usd"},
            "auto_prompt": {
                "llm": {
                    "backend": hosted_backend,
                    "model": "openai/openai/gpt-5.6-sol",
                    "reasoning_effort": "xhigh",
                }
            },
        },
        session_dir,
    )

    llm_config = config["auto_prompt"]["llm"]
    assert llm_config["backend"] == "nim"
    assert "reasoning_effort" not in llm_config


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


def test_authored_orm_success_diagnostic_reaches_event_and_final_stats(
    tmp_path: Path,
) -> None:
    diagnostic = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "AUTHORED_ORM_PRESERVED",
        "severity": "info",
        "stage": "blend_textures",
        "prim_path": "/World/Mesh",
        "material_name": "Metal",
        "message": "Authored channels preserved.",
        "recommended_action": "",
        "details": {},
    }
    context = {
        "blended_textures": {"Metal": object()},
        "blend_textures_diagnostics": [diagnostic],
    }

    event_stats = _extract_step_stats("blend_textures", context)
    final_stats = _extract_final_stats(context, tmp_path / "session")

    event_diagnostics = event_stats["diagnostics"]
    final_diagnostics = final_stats["diagnostics"]["blend_textures"]
    for diagnostics in (event_diagnostics, final_diagnostics):
        assert [item["code"] for item in diagnostics] == ["AUTHORED_ORM_PRESERVED"]
        assert diagnostics[0]["stage"] == "blend_textures"
        assert not any(
            item["code"] == "AUTHORED_ORM_NOT_PRESERVED" for item in diagnostics
        )


def test_authored_orm_failure_diagnostic_is_not_filtered_from_stats(
    tmp_path: Path,
) -> None:
    diagnostic = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "AUTHORED_ORM_NOT_PRESERVED",
        "severity": "warning",
        "stage": "blend_textures",
        "prim_path": "/World/Mesh",
        "material_name": "Metal",
        "message": "Authored channels were not preserved.",
        "recommended_action": "Inspect the UV layout.",
        "details": {"reason": "uv_layout_regenerated"},
    }
    context = {"blend_textures_diagnostics": [diagnostic]}

    event_stats = _extract_step_stats("blend_textures", context)
    final_stats = _extract_final_stats(context, tmp_path / "session")

    assert event_stats["diagnostics"] == [diagnostic]
    assert final_stats["diagnostics"]["blend_textures"] == [diagnostic]


def test_authored_orm_diagnostics_redact_secrets_and_session_paths(
    tmp_path: Path,
) -> None:
    from ...service.sanitization import sanitize_step_stats

    storage_root = tmp_path / "sessions"
    session_path = storage_root / "session-123" / "input" / "authored_orm.png"
    secret = "ORM_QUERY_SECRET_123"
    authored_ref = f"https://assets.invalid/orm.png?api_key={secret}"
    diagnostic = {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": "AUTHORED_ORM_NOT_PRESERVED",
        "severity": "warning",
        "stage": "blend_textures",
        "prim_path": "/World/Mesh",
        "material_name": "Metal",
        "message": f"Could not preserve {authored_ref} from {session_path}",
        "recommended_action": "Inspect the authored ORM.",
        "details": {
            "reason": "authored_orm_unresolvable",
            "authored_ref": authored_ref,
            "authored_orm": str(session_path),
        },
    }
    context = {"blend_textures_diagnostics": [diagnostic]}

    event_stats = sanitize_step_stats(
        _extract_step_stats("blend_textures", context),
        str(storage_root),
    )
    final_stats = sanitize_step_stats(
        _extract_final_stats(context, storage_root / "session-123"),
        str(storage_root),
    )
    assert event_stats is not None
    assert final_stats is not None

    event_diagnostic = event_stats["diagnostics"][0]
    final_diagnostic = final_stats["diagnostics"]["blend_textures"][0]
    for public_diagnostic in (event_diagnostic, final_diagnostic):
        assert public_diagnostic["code"] == "AUTHORED_ORM_NOT_PRESERVED"
        assert public_diagnostic["stage"] == "blend_textures"
        serialized = json.dumps(public_diagnostic, sort_keys=True)
        assert secret not in serialized
        assert f"api_key={secret}" not in serialized
        assert str(storage_root) not in serialized
        assert "<redacted>" in serialized
        assert "<session>" in serialized


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


@pytest.mark.parametrize("failure_mode", ["root", "member", "save"])
def test_package_usdz_failure_surfaces_warning_in_final_stats(
    tmp_path: Path,
    monkeypatch,
    failure_mode: str,
) -> None:
    """A USDZ packaging miss must be visible in /results, not logs only."""
    import pytest

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.DefinePrim("/Root", "Xform")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Shader")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/map.png")
    )
    stage.GetRootLayer().Save()
    textures_dir = cache / "textures"
    textures_dir.mkdir()
    (textures_dir / "map.png").write_bytes(b"png")

    class FailedWriter:
        def __init__(self) -> None:
            self.add_count = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback) -> bool:
            return False

        def AddFile(self, _src: str, dst: str) -> str:  # noqa: N802
            self.add_count += 1
            if failure_mode == "root" and self.add_count == 1:
                return ""
            if failure_mode == "member" and self.add_count == 2:
                return ""
            return dst

        def Save(self) -> bool:  # noqa: N802 - mirrors USD API
            return failure_mode != "save"

    zip_writer = getattr(Usd, "ZipFileWriter", None)
    if zip_writer is not None:
        monkeypatch.setattr(
            zip_writer,
            "CreateNew",
            staticmethod(lambda _dst: FailedWriter()),
        )
    else:
        from ...service.workers import executor as executor_module

        def fail_fallback(*_args, **_kwargs) -> None:
            raise RuntimeError(f"injected {failure_mode} packaging failure")

        monkeypatch.setattr(
            executor_module,
            "write_usdz_package_from_directory",
            fail_fallback,
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


def test_package_usdz_uses_aligned_fallback_without_low_level_zip_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Shader")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/map.png")
    )
    stage.GetRootLayer().Save()
    textures_dir = cache / "textures"
    textures_dir.mkdir()
    (textures_dir / "map.png").write_bytes(b"png")
    monkeypatch.delattr(Usd, "ZipFileWriter", raising=False)

    result = _package_usdz({"output_usd_paths": [str(output_usd)]}, tmp_path)

    assert result is not None
    package = Path(result)
    assert zipfile.is_zipfile(package)
    with zipfile.ZipFile(package) as archive:
        assert "map.png" in {Path(name).name for name in archive.namelist()}


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


def test_package_usdz_blocks_unresolved_nontexture_dependencies(
    tmp_path: Path,
) -> None:
    """Dependency-only archives fail rather than silently omit typed assets."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    cache = tmp_path / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    prim = stage.DefinePrim("/Root", "Xform")
    prim.CreateAttribute("inputs:data", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../data/missing.bin")
    )
    stage.GetRootLayer().Save()
    context = {"output_usd_paths": [str(output_usd)]}

    assert _package_usdz(context, tmp_path) is None
    assert context["usdz_packaging_failed"] is True
    assert "unresolved package dependencies" in context["usdz_packaging_error"]


def test_package_usdz_blocks_resolved_dependencies_outside_run_cache(
    tmp_path: Path,
) -> None:
    """Raw USD dependencies outside cache fail closed instead of leaking paths."""
    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    external_layer = input_dir / "payload.usda"
    external_layer.write_text(
        '#usda 1.0\ndef Xform "Root" {}\n',
        encoding="utf-8",
    )
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.GetRootLayer().subLayerPaths = [str(external_layer)]
    stage.GetRootLayer().Save()
    context = {"output_usd_paths": [str(output_usd)]}

    assert _package_usdz(context, session_dir) is None
    assert context["usdz_packaging_failed"] is True
    assert (
        context["usdz_packaging_error"]
        == "Failed to create USDZ package: Output USD dependency is outside "
        "the package workspace"
    )


def test_package_usdz_preserves_relative_symlink_member_layout(
    tmp_path: Path,
) -> None:
    """The archive member keeps the lexical path authored by the root layer."""
    import shutil
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir = tmp_path / "session"
    cache_dir = session_dir / "cache"
    output_dir = cache_dir / "output"
    shared_dir = cache_dir / "shared"
    assets_dir = cache_dir / "assets"
    output_dir.mkdir(parents=True)
    shared_dir.mkdir()
    assets_dir.mkdir()

    target_layer = assets_dir / "content.usda"
    target_stage = Usd.Stage.CreateNew(str(target_layer))
    target_stage.DefinePrim("/Referenced", "Xform")
    assert target_stage.GetRootLayer().Save()
    alias_layer = shared_dir / "content.usda"
    try:
        alias_layer.symlink_to("../assets/content.usda")
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    output_usd = output_dir / "textured_output.usda"
    output_stage = Usd.Stage.CreateNew(str(output_usd))
    output_stage.GetRootLayer().subLayerPaths = ["../shared/content.usda"]
    assert output_stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    packaged = _package_usdz(context, session_dir)
    assert packaged is not None
    packaged_path = Path(packaged)
    with zipfile.ZipFile(packaged_path) as archive:
        assert "shared/content.usda" in archive.namelist()

    relocated_package = tmp_path / "published.usdz"
    packaged_path.rename(relocated_package)
    shutil.rmtree(session_dir)
    relocated_stage = Usd.Stage.Open(str(relocated_package))
    assert relocated_stage is not None
    assert relocated_stage.GetPrimAtPath("/Referenced").IsValid()


def test_package_usdz_rejects_relative_symlink_member_escape(
    tmp_path: Path,
) -> None:
    """A lexical cache member cannot package a canonical target outside it."""
    pytest.importorskip("pxr")
    from pxr import Usd

    outside_layer = tmp_path / "outside.usda"
    outside_stage = Usd.Stage.CreateNew(str(outside_layer))
    outside_stage.DefinePrim("/Private", "Xform")
    assert outside_stage.GetRootLayer().Save()

    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    shared_dir = session_dir / "cache" / "shared"
    output_dir.mkdir(parents=True)
    shared_dir.mkdir()
    escape_layer = shared_dir / "escape.usda"
    try:
        escape_layer.symlink_to(outside_layer)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    output_usd = output_dir / "textured_output.usda"
    output_stage = Usd.Stage.CreateNew(str(output_usd))
    output_stage.GetRootLayer().subLayerPaths = ["../shared/escape.usda"]
    assert output_stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}
    assert _package_usdz(context, session_dir) is None
    assert context["usdz_packaging_error"] == (
        "Failed to create USDZ package: "
        "Output USD dependency is outside the package workspace"
    )


def test_package_usdz_ignores_inactive_string_texture_inputs(
    tmp_path: Path,
) -> None:
    """Inactive shader strings cannot probe or block host filesystem paths."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    inactive = UsdShade.Shader.Define(stage, "/Root/Inactive")
    inactive.CreateIdAttr("UsdPreviewSurface")
    inactive.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set("/etc/hosts")
    inactive.GetPrim().SetActive(False)
    metadata = UsdShade.Shader.Define(stage, "/Root/Metadata")
    metadata.CreateIdAttr("UsdPreviewSurface")
    metadata.CreateInput("metadata_texture", Sdf.ValueTypeNames.String).Set("notes.txt")
    stage.GetRootLayer().Save()
    context = {"output_usd_paths": [str(output_usd)]}

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    with zipfile.ZipFile(packaged_path) as package:
        assert package.namelist() == ["output/textured_output.usda"]


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


def test_package_usdz_does_not_clear_unrelated_missing_mdl_assets(
    tmp_path: Path,
) -> None:
    """Only the recognized MDL shader source attribute may be cleared."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    output_dir = tmp_path / "cache" / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    root.CreateAttribute(
        "asset:provenance",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("missing_model.mdl"))
    stage.GetRootLayer().Save()
    context = {"output_usd_paths": [str(output_usd)]}

    assert _package_usdz(context, tmp_path) is None
    reopened = Usd.Stage.Open(str(output_usd))
    value = reopened.GetPrimAtPath("/Root").GetAttribute("asset:provenance").Get()
    assert value.path == "missing_model.mdl"
    assert "usdz_mdl_source_assets_cleared" not in context


def test_package_usdz_clears_missing_mdl_from_layered_authoring_spec(
    tmp_path: Path,
) -> None:
    """Missing MDL defaults must be cleared in the package member that authored them."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    payload_text = """#usda 1.0
def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Plastic"
        {
            def Shader "MDLShader"
            {
                uniform token info:id = "mdlMaterial"
                asset info:mdl:sourceAsset = @OmniPBR.mdl@
            }
        }
    }
}
"""
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            '#usda 1.0\n(\n defaultPrim = "Root"\n'
            " subLayers = [@Payload/Contents.usda@]\n)\n",
        )
        package.writestr("Scene/Payload/Contents.usda", payload_text)

    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    prepared_usd.write_text(payload_text, encoding="utf-8")
    output_usd.write_text(payload_text, encoding="utf-8")
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    assert context["usdz_mdl_source_assets_cleared"] == [
        "/Root/Looks/Plastic/MDLShader.info:mdl:sourceAsset"
    ]
    packaged_stage = Usd.Stage.Open(packaged_path)
    shader = packaged_stage.GetPrimAtPath("/Root/Looks/Plastic/MDLShader")
    assert shader.GetAttribute("info:mdl:sourceAsset").Get() is None


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


def test_packaging_never_mutates_upload_bundle_authoring_layers(
    tmp_path: Path,
) -> None:
    """Generated-path overrides stay in cache even when source specs are input."""
    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd

    session_dir = tmp_path / "session"
    input_scene_dir = session_dir / "input" / ".step1x_package_assets" / "scene"
    source_layer_dir = input_scene_dir / "Materials"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    source_layer_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    source_layer = source_layer_dir / "Paint.usda"
    source_layer.write_text(
        """#usda 1.0
over "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:diffuse_texture = @Paint_albedo.png@
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    Image.new("RGB", (2, 2), (20, 40, 60)).save(source_layer_dir / "Paint_albedo.png")
    generated = textures_dir / "Paint_albedo.png"
    Image.new("RGB", (2, 2), (200, 40, 20)).save(generated)
    original_source = source_layer.read_bytes()

    output_usd = output_dir / "textured_output.usda"
    root_layer = Sdf.Layer.CreateNew(str(output_usd))
    root_layer.subLayerPaths = [str(source_layer)]
    root_layer.defaultPrim = "Root"
    Sdf.CreatePrimInLayer(root_layer, "/Root").specifier = Sdf.SpecifierDef
    assert root_layer.Save()
    context = {
        "usd_path": str(input_scene_dir / "scene.usda"),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key="Paint",
                prompt="paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            "Paint": {
                "albedo": str(generated),
                "normal": "",
                "orm": "",
            }
        },
    }

    _package_usdz(context, session_dir)

    assert source_layer.read_bytes() == original_source
    source_stage = Usd.Stage.Open(str(source_layer))
    source_value = (
        source_stage.GetPrimAtPath("/Root/Looks/Paint/Shader")
        .GetAttribute("inputs:diffuse_texture")
        .Get()
    )
    assert source_value.path == "Paint_albedo.png"


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


def test_real_apply_then_package_preserves_layered_usdz_composition(
    tmp_path: Path,
) -> None:
    """Service-owned direct USDZ apply is reconstructed into a portable package."""
    import shutil
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade
    from texture_agent.functions.material_discovery import PrimTextureUnit
    from texture_agent.tasks.apply_textures import ApplyTexturesTask
    from texture_agent.tasks.blend_textures import BlendedTextures

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    working_dir = session_dir / "cache"
    textures_dir = working_dir / "textures"
    input_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    root_layer = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend payload = @Payload/scene.usda@
)
{
}
"""
    payload_layer = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            token outputs:surface.connect = </Root/Looks/Paint/Preview.outputs:surface>

            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.3, 0.4)
                float inputs:metallic = 0.0
                float inputs:roughness = 0.6
                token outputs:surface
            }
        }
    }

    def Mesh "Body"
    {
        rel material:binding = </Root/Looks/Paint>
    }
}
"""
    source_usdz = input_dir / "layered.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_layer)
        package.writestr("Scene/Payload/scene.usda", payload_layer)

    map_files = {
        "albedo": textures_dir / "Paint_albedo.png",
        "normal": textures_dir / "Paint_normal.png",
        "orm": textures_dir / "Paint_orm.png",
    }
    Image.new("RGB", (8, 8), (210, 30, 25)).save(map_files["albedo"])
    Image.new("RGB", (8, 8), (128, 128, 255)).save(map_files["normal"])
    Image.new("RGB", (8, 8), (255, 90, 15)).save(map_files["orm"])
    unit = PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path="/Root/Looks/Paint",
            name="Paint",
            bound_prim_paths=["/Root/Body"],
            base_color=(0.2, 0.3, 0.4),
            base_metalness=0.0,
            specular_roughness=0.6,
        ),
        key="Paint",
        prompt="weathered paint",
        opacity=1.0,
    )
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(source_usdz),
        "usd_dependency_root": str(input_dir),
        "service_managed_usdz_reconstruction": True,
        "working_dir": str(working_dir),
        "blended_textures": {
            "Paint": BlendedTextures(
                albedo=str(map_files["albedo"]),
                normal=str(map_files["normal"]),
                orm=str(map_files["orm"]),
            )
        },
        "prim_texture_units": [unit],
    }

    context = ApplyTexturesTask().run(context)
    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    assert "source_usdz_prepared_edit_layer_paths" not in context
    assert context["render_output_usd_paths"] == [context["source_usdz_stage_path"]]
    packaged_stage = Usd.Stage.Open(packaged_path)
    assert packaged_stage.GetPrimAtPath("/Root/Body").IsValid()
    surface = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Preview"))
    for channel in ("diffuseColor", "normal", "occlusion", "roughness", "metallic"):
        connected, invalid = surface.GetInput(channel).GetConnectedSources()
        assert connected
        assert not invalid
        texture = UsdShade.Shader(connected[0].source.GetPrim())
        asset = texture.GetInput("file").Get()
        assert isinstance(asset, Sdf.AssetPath)
        assert asset.resolvedPath
    render_stage = Usd.Stage.Open(context["source_usdz_stage_path"])
    render_surface = UsdShade.Shader(
        render_stage.GetPrimAtPath("/Root/Looks/Paint/Preview")
    )
    for channel in ("diffuseColor", "normal", "occlusion", "roughness", "metallic"):
        texture = UsdShade.Shader(
            render_surface.GetInput(channel).GetConnectedSource()[0].GetPrim()
        )
        assert texture.GetInput("file").Get().resolvedPath
    packaged_stage.Unload("/Root")
    assert not packaged_stage.GetPrimAtPath("/Root/Body").IsValid()
    # Direct-USDZ apply opinions live in the reconstructed root overlay, while
    # the original geometry remains payload-backed.
    assert packaged_stage.GetPrimAtPath(
        "/Root/Looks/Paint/TextureAgentAlbedoTexture"
    ).IsValid()

    downloaded_package = tmp_path / "downloaded.usdz"
    shutil.copy2(packaged_path, downloaded_package)
    shutil.rmtree(session_dir)
    downloaded_stage = Usd.Stage.Open(str(downloaded_package))
    assert downloaded_stage.GetPrimAtPath("/Root/Body").IsValid()
    downloaded_surface = UsdShade.Shader(
        downloaded_stage.GetPrimAtPath("/Root/Looks/Paint/Preview")
    )
    downloaded_texture = UsdShade.Shader(
        downloaded_surface.GetInput("diffuseColor").GetConnectedSource()[0].GetPrim()
    )
    assert downloaded_texture.GetInput("file").Get().resolvedPath


def test_layered_delta_rebases_shader_connections_to_payload_namespace(
    tmp_path: Path,
) -> None:
    """Copied generated graphs use the payload namespace, not the composed one."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade
    from texture_agent.functions.material_discovery import PrimTextureUnit
    from texture_agent.tasks.apply_textures import ApplyTexturesTask
    from texture_agent.tasks.blend_textures import BlendedTextures

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    working_dir = session_dir / "cache"
    prepared_dir = working_dir / "prepared"
    textures_dir = working_dir / "textures"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_usdz = input_dir / "layered.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend payload = @Payload/model.usda@</Model>
)
{
}
""",
        )
        package.writestr(
            "Scene/Payload/model.usda",
            """#usda 1.0
(
    defaultPrim = "Model"
)
def Xform "Model"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            token outputs:surface.connect = </Model/Looks/Paint/Preview.outputs:surface>
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.3, 0.4)
                token outputs:surface
            }
        }
    }
    def Mesh "Body"
    {
        rel material:binding = </Model/Looks/Paint>
    }
}
""",
        )

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    maps = {
        "albedo": textures_dir / "Paint_albedo.png",
        "normal": textures_dir / "Paint_normal.png",
        "orm": textures_dir / "Paint_orm.png",
    }
    Image.new("RGB", (4, 4), (200, 40, 20)).save(maps["albedo"])
    Image.new("RGB", (4, 4), (128, 128, 255)).save(maps["normal"])
    Image.new("RGB", (4, 4), (255, 90, 20)).save(maps["orm"])
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "working_dir": str(working_dir),
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Asset/Looks/Paint",
                    name="Paint",
                    bound_prim_paths=["/Asset/Body"],
                ),
                key="Paint",
                prompt="paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            "Paint": BlendedTextures(
                albedo=str(maps["albedo"]),
                normal=str(maps["normal"]),
                orm=str(maps["orm"]),
            )
        },
    }

    ApplyTexturesTask().run(context)
    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    surface = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/Asset/Looks/Paint/Preview")
    )
    for channel in ("diffuseColor", "normal", "occlusion", "roughness", "metallic"):
        connected = surface.GetInput(channel).GetConnectedSource()
        assert connected is not None
        assert str(connected[0].GetPath()).startswith("/Asset/Looks/Paint/")
        file_value = UsdShade.Shader(connected[0].GetPrim()).GetInput("file").Get()
        assert isinstance(file_value, Sdf.AssetPath)
        assert file_value.resolvedPath
    with zipfile.ZipFile(packaged_path) as package:
        payload_text = package.read("Scene/Payload/model.usda").decode()
    assert "</Asset/Looks/Paint/" not in payload_text
    assert "</Model/Looks/Paint/" in payload_text
    packaged_stage.Unload("/Asset")
    assert not packaged_stage.GetPrimAtPath(
        "/Asset/Looks/Paint/TextureAgentAlbedoTexture"
    ).IsValid()


@pytest.mark.parametrize("edit_kind", ["property", "prim_deletion"])
def test_layered_delta_rejects_shared_reference_source_specs(
    tmp_path: Path,
    edit_kind: str,
) -> None:
    """Per-instance edits and deletions cannot mutate shared source specs."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "shared.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World"
{
    def Xform "A" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
    def Xform "B" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
}
""",
        )
        package.writestr(
            "Scene/Model/model.usda",
            """#usda 1.0
def Xform "Model"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.3, 0.4)
            }
        }
    }
}
""",
        )
    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))
    edited_stage = Usd.Stage.Open(str(output_usd))
    if edit_kind == "property":
        shader = UsdShade.Shader(
            edited_stage.GetPrimAtPath("/World/A/Looks/Paint/Preview")
        )
        shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.1, 0.1))
    else:
        assert edited_stage.RemovePrim("/World/A/Looks")
    edited_stage.GetRootLayer().Save()
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _package_usdz(context, session_dir) is None
    expected_error = (
        "shared source spec"
        if edit_kind == "property"
        else "whole-prim deletion is not supported"
    )
    assert expected_error in context["usdz_packaging_error"]


def test_layered_delta_preserves_uv_edit_with_unchanged_internal_instances(
    tmp_path: Path,
) -> None:
    """UV deltas coexist with unchanged internally referenced instances."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from texture_agent.tasks.prepare_uvs import (
        PrepareUVsTask as TexturePrepareUVsTask,
    )

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True)

    source_usdz = input_dir / "prepared_instances.usdz"
    legacy_texture = tmp_path / "legacy.png"
    Image.new("RGB", (2, 2), (20, 40, 60)).save(legacy_texture)
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "Prototype" (
        customData = {
            asset legacyAsset = @legacy.png@
            asset[] legacyAssets = [@legacy.png@]
        }
    )
    {
        def Mesh "InternalBody"
        {
        }
        def Shader "Legacy"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @legacy.png@
            asset[] inputs:alternates = [@legacy.png@]
        }
    }
    def Xform "Expanded" (
        prepend references = </Root/Prototype>
    )
    {
    }
    def Xform "Instanced" (
        instanceable = true
        prepend references = </Root/Prototype>
    )
    {
    }
    def Mesh "UvTarget"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }
}
""",
        )
        package.write(legacy_texture, "Scene/legacy.png")

    context = {
        "usd_path": str(source_usdz),
        "working_dir": str(session_dir / "cache"),
        "texture_config": {
            "uv_policy": "generate_missing",
            "uv_projection": "box",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": ["/Root/UvTarget"],
        },
    }
    TexturePrepareUVsTask().run(context)
    prepared_usd = Path(context["usd_path"])
    assert context["source_usd_path"] == str(source_usdz)
    assert context["uv_preparation"]["generated"] == 1
    context["output_usd_paths"] = [str(prepared_usd)]

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None, context.get("usdz_packaging_error")
    packaged_stage = Usd.Stage.Open(packaged_path)
    st = UsdGeom.PrimvarsAPI(packaged_stage.GetPrimAtPath("/Root/UvTarget")).GetPrimvar(
        "st"
    )
    assert st.HasValue()
    assert len(st.Get()) == 4
    instance = packaged_stage.GetPrimAtPath("/Root/Instanced")
    assert instance.IsInstanceable()
    assert instance.IsInstance()
    assert packaged_stage.GetPrimAtPath(
        "/Root/Instanced/InternalBody"
    ).IsInstanceProxy()
    assert packaged_stage.GetPrimAtPath("/Root/Expanded/InternalBody").IsValid()
    legacy = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Prototype/Legacy"))
    legacy_file = legacy.GetInput("file").Get()
    assert isinstance(legacy_file, Sdf.AssetPath)
    assert legacy_file.path == "legacy.png"
    assert legacy_file.resolvedPath
    with zipfile.ZipFile(packaged_path) as package:
        assert package.read("Scene/legacy.png") == legacy_texture.read_bytes()


def test_layered_packaging_preserves_instanceable_references(
    tmp_path: Path,
) -> None:
    """Packaging source textures must not de-instance reconstructed assets."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source_usdz = input_dir / "instances.usdz"
    texture = tmp_path / "albedo.png"
    Image.new("RGB", (2, 2), (20, 40, 60)).save(texture)
    root_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "A" (
        instanceable = true
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
    def Xform "B" (
        instanceable = true
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
}
"""
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr(
            "Scene/Model/model.usda",
            """#usda 1.0
def Xform "Model"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Texture"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @Textures/albedo.png@
            }
        }
    }
}
""",
        )
        package.write(texture, "Scene/Model/Textures/albedo.png")
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(root_text, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    render_stage = Usd.Stage.Open(context["source_usdz_stage_path"])
    for stage in (packaged_stage, render_stage):
        for path in ("/Root/A", "/Root/B"):
            prim = stage.GetPrimAtPath(path)
            assert prim.IsInstanceable()
            assert prim.IsInstance()


def test_package_usdz_restores_layered_source_package_dependencies(
    tmp_path: Path,
) -> None:
    """An edited USDZ root must retain its payload and sublayer dependencies."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd, UsdUtils

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    source_dir = tmp_path / "source"
    payload_dir = source_dir / "Payload"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    payload_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_root = source_dir / "scene.usda"
    source_root.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend payload = @./Payload/Contents.usda@
)
{
}
""",
        encoding="utf-8",
    )
    (payload_dir / "Contents.usda").write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
    subLayers = [
        @./Geometry.usda@,
        @./Materials.usda@
    ]
)

over "Root"
{
}
""",
        encoding="utf-8",
    )
    (payload_dir / "Geometry.usda").write_text(
        """#usda 1.0

def Xform "Root"
{
    def Mesh "Body"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        rel material:binding = </Root/Looks/Paint>
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )
    (payload_dir / "Materials.usda").write_text(
        """#usda 1.0

over "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            token outputs:surface.connect = </Root/Looks/Paint/Surface.outputs:surface>

            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.4, 0.1)
                token outputs:surface
            }
        }
    }
}
""",
        encoding="utf-8",
    )

    source_usdz = input_dir / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(source_root), str(source_usdz))
    with zipfile.ZipFile(source_usdz) as source_package:
        assert {
            "Payload/Contents.usda",
            "Payload/Geometry.usda",
            "Payload/Materials.usda",
        } <= set(source_package.namelist())

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_bytes(source_root.read_bytes())
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    with zipfile.ZipFile(packaged_path) as output_package:
        assert {
            "Payload/Contents.usda",
            "Payload/Geometry.usda",
            "Payload/Materials.usda",
        } <= set(output_package.namelist())
    packaged_stage = Usd.Stage.Open(packaged_path)
    assert packaged_stage is not None
    assert packaged_stage.GetPrimAtPath("/Root/Body").IsValid()
    assert packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Surface").IsValid()
    staged_root = Path(context["source_usdz_stage_path"])
    assert staged_root.name == "scene.usda"
    assert context["render_output_usd_paths"] == [str(staged_root)]
    assert (staged_root.parent / "Payload" / "Contents.usda").is_file()


def test_package_usdz_preserves_nested_source_root_references(
    tmp_path: Path,
) -> None:
    """Nested ``../`` references keep their original in-package meaning."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend references = @../Payload/Contents.usda@
)
{
}
"""
    payload = """#usda 1.0
(
    defaultPrim = "Root"
)

over "Root"
{
    def Mesh "Body"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr("Scene/scene.usda", source_root)
        package.writestr("Scene/Extras.usda", "#usda 1.0\n")
        package.writestr("Payload/Contents.usda", payload)

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir()
    generated_texture = textures_dir / "Body_albedo.png"
    Image.new("RGB", (4, 4), (130, 90, 40)).save(generated_texture)
    edited_stage = Usd.Stage.Open(str(output_usd))
    shader = UsdShade.Shader.Define(edited_stage, "/Root/Looks/Body/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(generated_texture))
    )
    edited_stage.GetRootLayer().Save()
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    assert packaged_stage is not None
    assert packaged_stage.GetPrimAtPath("/Root/Body").IsValid()
    packaged_shader = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/Root/Looks/Body/Shader")
    )
    assert packaged_shader.GetInput("diffuse_texture").Get().resolvedPath
    assert "usdz_layer_references_cleared" not in context
    staged_root = Path(context["source_usdz_stage_path"])
    assert (
        staged_root.relative_to(staged_root.parents[2])
        .as_posix()
        .endswith("Scene/scene.usda")
    )
    assert (staged_root.parent.parent / "Payload" / "Contents.usda").is_file()


def test_package_usdz_supports_deep_nested_source_root_dependencies(
    tmp_path: Path,
) -> None:
    """Deep package roots no longer depend on cache/output rebase headroom."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root" (
    prepend payload = @../../../../Payload/Contents.usda@
)
{
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr("A/B/C/D/scene.usda", source_root)
        package.writestr(
            "Payload/Contents.usda",
            (
                '#usda 1.0\n(\n    defaultPrim = "Root"\n)\n\n'
                'over "Root"\n{\n    def Xform "Body" {}\n}\n'
            ),
        )

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    textures_dir = session_dir / "cache" / "textures"
    textures_dir.mkdir()
    generated_texture = textures_dir / "Body_albedo.png"
    Image.new("RGB", (4, 4), (130, 90, 40)).save(generated_texture)
    edited_stage = Usd.Stage.Open(str(output_usd))
    shader = UsdShade.Shader.Define(edited_stage, "/Root/Looks/Body/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../textures/Body_albedo.png")
    )
    edited_stage.GetRootLayer().Save()
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    assert packaged_stage is not None
    assert packaged_stage.GetPrimAtPath("/Root/Body").IsValid()
    packaged_shader = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/Root/Looks/Body/Shader")
    )
    assert packaged_shader.GetInput("diffuse_texture").Get().resolvedPath
    assert (
        Path(context["source_usdz_stage_path"])
        .as_posix()
        .endswith("/A/B/C/D/scene.usda")
    )


@pytest.mark.parametrize("texture_suffix", [".jpeg", ".jpg", ".png"])
def test_package_usdz_uses_original_source_after_uv_path_replacement(
    tmp_path: Path,
    texture_suffix: str,
) -> None:
    """A prepared cache USD must not hide its original layered USDZ upload."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            (
                '#usda 1.0\n(\n    defaultPrim = "Root"\n'
                "    subLayers = [@../Payload/Contents.usda@]\n)\n"
            ),
        )
        package.writestr(
            "Payload/Contents.usda",
            """#usda 1.0
def Xform "Root"
{
    def Mesh "Body"
    {
    }
    def Scope "Looks"
    {
        def Material "Keep"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:diffuse_texture = @../Textures/keep{texture_suffix}@
                asset inputs:mdl_source = @../Materials/Test.mdl@
                string inputs:legacy_texture = "../Textures/keep{texture_suffix}"
            }
        }
    }
}
""".replace("{texture_suffix}", texture_suffix),
        )
        package.writestr("Materials/Test.mdl", "mdl 1.0;\n")
        texture_path = tmp_path / f"keep{texture_suffix}"
        Image.new("RGB", (2, 2), (10, 20, 30)).save(texture_path)
        package.write(texture_path, f"Textures/keep{texture_suffix}")

    flattened = f"""#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{{
    def Mesh "Body"
    {{
    }}
    def Scope "Looks"
    {{
        def Material "Keep"
        {{
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:diffuse_texture = @{source_usdz}[Textures/keep{texture_suffix}]@
                asset inputs:mdl_source = @{source_usdz}[Materials/Test.mdl]@
                string inputs:legacy_texture = "{source_usdz}[Textures/keep{texture_suffix}]"
            }}
        }}
    }}
}}
"""
    prepared_usd = prepared_dir / "prepared_input.usda"
    prepared_usd.write_text(flattened, encoding="utf-8")
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(flattened, encoding="utf-8")
    generated_dir = session_dir / "cache" / "textures"
    generated_dir.mkdir()
    generated_key = "tu_aaaaaaaaaaaaaaaaaaaa"
    generated_albedo = generated_dir / f"{generated_key}_albedo.png"
    Image.new("RGB", (2, 2), (140, 40, 220)).save(generated_albedo)
    edited_stage = Usd.Stage.Open(str(output_usd))
    edited_shader = UsdShade.Shader(
        edited_stage.GetPrimAtPath("/Root/Looks/Keep/Shader")
    )
    edited_shader.GetInput("diffuse_texture").Set(
        Sdf.AssetPath(f"../textures/{generated_albedo.name}")
    )
    edited_stage.GetRootLayer().Save()
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="/Root/Body",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Keep",
                    name="Keep",
                ),
                key=generated_key,
                prompt="generated paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            generated_key: {
                "albedo": str(generated_albedo),
                "normal": "",
                "orm": "",
            }
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    assert (
        Path(context["source_usdz_stage_path"]).as_posix().endswith("/Scene/root.usda")
    )
    packaged_stage = Usd.Stage.Open(packaged_path)
    assert packaged_stage.GetPrimAtPath("/Root/Body").IsValid()
    packaged_shader = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/Root/Looks/Keep/Shader")
    )
    assert packaged_shader.GetInput("mdl_source").Get().resolvedPath
    with zipfile.ZipFile(packaged_path) as package:
        assert "Materials/Test.mdl" in package.namelist()
    packaged_shader = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/Root/Looks/Keep/Shader")
    )
    packaged_texture = packaged_shader.GetInput("diffuse_texture").Get()
    assert Path(packaged_texture.path).name == generated_albedo.name
    assert packaged_texture.resolvedPath
    assert (
        packaged_shader.GetInput("legacy_texture").Get()
        == f"../Textures/keep{texture_suffix}"
    )
    assert context["usdz_source_portability"]["texture_reference_count"] == 2
    assert [
        entry["severity"] for entry in context["output_portability"]["diagnostics"]
    ] == ["warning"]


def test_package_usdz_ignores_prepare_only_asset_localization(tmp_path: Path) -> None:
    """An untouched prepared asset rewrite is not an Apply edit."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd
    from world_understanding.functions.graphics.so_export import export_stage_portably

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    texture = tmp_path / "Olive_Drab_Matte_albedo.png"
    Image.new("RGB", (3, 3), (55, 70, 25)).save(texture)
    original_texture_bytes = texture.read_bytes()
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "scene.usda",
            """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Material "Olive_Drab_Matte"
    {
        asset inputs:base_color_texture_file = @Textures/Olive_Drab_Matte_albedo.png@
    }
}
""",
        )
        package.write(texture, "Textures/Olive_Drab_Matte_albedo.png")

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared_input.usd"
    assert export_stage_portably(
        source_stage,
        prepared_usd,
        approved_dependency_roots=(input_dir, prepared_dir),
    )
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    assert prepared_stage is not None
    output_usd = output_dir / "textured_output.usda"
    assert prepared_stage.Flatten().Export(str(output_usd))
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    packaged_value = (
        packaged_stage.GetPrimAtPath("/Root/Olive_Drab_Matte")
        .GetAttribute("inputs:base_color_texture_file")
        .Get()
    )
    assert packaged_value.path == "Textures/Olive_Drab_Matte_albedo.png"
    assert packaged_value.resolvedPath
    with zipfile.ZipFile(packaged_path) as package:
        assert (
            package.read("Textures/Olive_Drab_Matte_albedo.png")
            == original_texture_bytes
        )


def test_prepare_source_usdz_reads_pristine_baseline_from_disk(
    tmp_path: Path,
) -> None:
    """A dirty cached prepared layer cannot absorb a genuine Apply edit."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "scene.usda",
            """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Material "Paint"
    {
        float inputs:roughness = 0.2
    }
}
""",
        )

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared_input.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))

    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    assert prepared_stage is not None
    roughness = prepared_stage.GetPrimAtPath("/Root/Paint").GetAttribute(
        "inputs:roughness"
    )
    assert roughness.Set(0.8)
    assert prepared_stage.GetRootLayer().dirty

    output_usd = output_dir / "textured_output.usda"
    assert prepared_stage.GetRootLayer().Export(str(output_usd))
    disk_baseline = Sdf.Layer.OpenAsAnonymous(str(prepared_usd))
    assert disk_baseline is not None
    disk_stage = Usd.Stage.Open(disk_baseline)
    assert disk_stage is not None
    assert disk_stage.GetPrimAtPath("/Root/Paint").GetAttribute(
        "inputs:roughness"
    ).Get() == pytest.approx(0.2)

    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }
    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed is not None
    assert reconstructed.GetPrimAtPath("/Root/Paint").GetAttribute(
        "inputs:roughness"
    ).Get() == pytest.approx(0.8)


def test_prepare_source_usdz_rejects_prepared_uv_deletion(tmp_path: Path) -> None:
    """Preparation cannot delete authored source UV or index opinions."""
    import zipfile

    pytest.importorskip("pxr")

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    source_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Mesh "Body"
    {
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
            interpolation = "vertex"
        )
        int[] primvars:st:indices = [0, 1, 2]
    }
}
"""
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("scene.usda", source_text)

    prepared_text = source_text.replace(
        "        int[] primvars:st:indices = [0, 1, 2]\n",
        "",
    )
    prepared_usd = prepared_dir / "prepared_input.usda"
    output_usd = output_dir / "textured_output.usda"
    prepared_usd.write_text(prepared_text, encoding="utf-8")
    output_usd.write_text(prepared_text, encoding="utf-8")
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _prepare_source_usdz_stage(context, session_dir) is None
    assert context["usdz_packaging_failed"] is True
    assert "removed a source UV opinion" in context["usdz_packaging_error"]


def test_prepare_source_usdz_strips_uvs_from_apply_added_prim_tree(
    tmp_path: Path,
) -> None:
    """E-added prim trees cannot bypass the B-vs-S UV allowlist."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("scene.usda", source_text)
    prepared_usd = prepared_dir / "prepared_input.usda"
    prepared_usd.write_text(source_text, encoding="utf-8")
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "ApplyAdded"
    {
        string applyMarker = "preserved"
        def Mesh "Body"
        {
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
                interpolation = "vertex"
            )
            int[] primvars:st:indices = [0, 1, 2]
        }
    }
}
""",
        encoding="utf-8",
    )
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None
    reconstructed = Usd.Stage.Open(str(staged_root))
    added = reconstructed.GetPrimAtPath("/Root/ApplyAdded")
    body = reconstructed.GetPrimAtPath("/Root/ApplyAdded/Body")
    assert added.GetAttribute("applyMarker").Get() == "preserved"
    assert not body.HasProperty("primvars:st")
    assert not body.HasProperty("primvars:st:indices")


def test_prepare_source_usdz_rejects_prepared_uv_subtree_deletion(
    tmp_path: Path,
) -> None:
    """B cannot hide authored UV removal by deleting the containing subtree."""
    import zipfile

    pytest.importorskip("pxr")

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "scene.usda",
            """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "UVBranch"
    {
        def Mesh "Body"
        {
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
                interpolation = "vertex"
            )
            int[] primvars:st:indices = [0, 1, 2]
        }
    }
}
""",
        )
    prepared_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
}
"""
    prepared_usd = prepared_dir / "prepared_input.usda"
    output_usd = output_dir / "textured_output.usda"
    prepared_usd.write_text(prepared_text, encoding="utf-8")
    output_usd.write_text(prepared_text, encoding="utf-8")
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _prepare_source_usdz_stage(context, session_dir) is None
    assert "subtree containing UV opinions" in context["usdz_packaging_error"]


def test_prepare_source_usdz_ignores_uvs_in_inactive_source_variant(
    tmp_path: Path,
) -> None:
    """S-to-B validation considers only the active source composition."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)

    source_layer = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_layer))
    root = UsdGeom.Xform.Define(source_stage, "/Root").GetPrim()
    source_stage.SetDefaultPrim(root)
    model = root.GetVariantSets().AddVariantSet("model")
    model.AddVariant("A")
    model.AddVariant("B")
    model.SetVariantSelection("A")
    with model.GetVariantEditContext():
        UsdGeom.Xform.Define(source_stage, "/Root/ActiveBranch")
    model.SetVariantSelection("B")
    with model.GetVariantEditContext():
        mesh = UsdGeom.Mesh.Define(source_stage, "/Root/InactiveUvBody")
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        assert st.Set([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(0.0, 1.0)])
        assert st.SetIndices([0, 1, 2])
    model.SetVariantSelection("A")
    assert source_stage.GetRootLayer().Save()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_layer, "scene.usda")
    prepared_usd = prepared_dir / "prepared_input.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(prepared_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    reconstructed = Usd.Stage.Open(str(staged_root))
    reconstructed_model = (
        reconstructed.GetPrimAtPath("/Root").GetVariantSets().GetVariantSet("model")
    )
    assert reconstructed_model.SetVariantSelection("B")
    assert reconstructed.GetPrimAtPath("/Root/InactiveUvBody").HasProperty(
        "primvars:st"
    )


def test_prepare_source_usdz_accepts_reflattened_instance_prototype_rename(
    tmp_path: Path,
) -> None:
    """Synthetic backing renames do not look like active prim deletion."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_layer = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_layer))
    root = UsdGeom.Xform.Define(source_stage, "/Root").GetPrim()
    source_stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(source_stage, "/Root/Prototype")
    UsdGeom.Mesh.Define(source_stage, "/Root/Prototype/Body")
    instance = UsdGeom.Xform.Define(source_stage, "/Root/Instance").GetPrim()
    assert instance.GetReferences().AddInternalReference("/Root/Prototype")
    assert instance.SetInstanceable(True)
    assert source_stage.GetRootLayer().Save()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_layer, "scene.usda")
    prepared_usd = prepared_dir / "prepared_input.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    assert prepared_stage.Flatten().Export(str(output_usd))
    prepared_backings = {
        prim.name
        for prim in prepared_stage.GetRootLayer().rootPrims
        if prim.name.startswith("Flattened_Prototype_")
    }
    output_stage = Usd.Stage.Open(str(output_usd))
    output_backings = {
        prim.name
        for prim in output_stage.GetRootLayer().rootPrims
        if prim.name.startswith("Flattened_Prototype_")
    }
    assert output_backings > prepared_backings
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed.GetPrimAtPath("/Root/Instance").IsInstance()
    assert reconstructed.GetPrimAtPath("/Root/Instance/Body").IsInstanceProxy()


@pytest.mark.parametrize(
    "instance_names",
    [("Instance",), ("A", "B")],
    ids=["stale-backing", "shared-instance"],
)
def test_prepare_source_usdz_rejects_instance_deletion_with_backing_retained(
    tmp_path: Path,
    instance_names: tuple[str, ...],
) -> None:
    """A stale synthetic backing cannot hide a missing active instance tree."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_layer = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_layer))
    root = UsdGeom.Xform.Define(source_stage, "/Root").GetPrim()
    source_stage.SetDefaultPrim(root)
    UsdGeom.Xform.Define(source_stage, "/Root/Prototype")
    UsdGeom.Mesh.Define(source_stage, "/Root/Prototype/Body")
    for instance_name in instance_names:
        instance = UsdGeom.Xform.Define(
            source_stage,
            f"/Root/{instance_name}",
        ).GetPrim()
        assert instance.GetReferences().AddInternalReference("/Root/Prototype")
        assert instance.SetInstanceable(True)
    assert source_stage.GetRootLayer().Save()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_layer, "scene.usda")
    prepared_usd = prepared_dir / "prepared_input.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    assert prepared_stage.Flatten().Export(str(output_usd))
    edited_stage = Usd.Stage.Open(str(output_usd))
    assert edited_stage.RemovePrim(f"/Root/{instance_names[0]}")
    assert edited_stage.GetRootLayer().Save()
    assert any(
        prim.name.startswith("Flattened_Prototype_")
        for prim in edited_stage.GetRootLayer().rootPrims
    )
    if len(instance_names) > 1:
        assert edited_stage.GetPrimAtPath(f"/Root/{instance_names[1]}").IsInstance()
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _prepare_source_usdz_stage(context, session_dir) is None
    assert "whole-prim deletion is not supported" in context["usdz_packaging_error"]


def test_prepare_source_usdz_rejects_prim_deletion_inside_active_variant(
    tmp_path: Path,
) -> None:
    """E cannot delete an active prepared prim, including inside a variant."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_layer = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_layer))
    root = UsdGeom.Xform.Define(source_stage, "/Root").GetPrim()
    source_stage.SetDefaultPrim(root)
    model = root.GetVariantSets().AddVariantSet("model")
    model.AddVariant("A")
    model.AddVariant("B")
    model.SetVariantSelection("A")
    with model.GetVariantEditContext():
        UsdGeom.Xform.Define(source_stage, "/Root/Branch")
        UsdGeom.Mesh.Define(source_stage, "/Root/Branch/DeleteMe")
    model.SetVariantSelection("B")
    with model.GetVariantEditContext():
        UsdGeom.Xform.Define(source_stage, "/Root/Branch")
        UsdGeom.Mesh.Define(source_stage, "/Root/Branch/KeepMe")
    model.SetVariantSelection("A")
    assert source_stage.GetRootLayer().Save()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_layer, "scene.usda")
    prepared_usd = prepared_dir / "prepared_input.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))
    edited_stage = Usd.Stage.Open(str(output_usd))
    assert edited_stage.RemovePrim("/Root/Branch")
    assert edited_stage.GetRootLayer().Save()
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _prepare_source_usdz_stage(context, session_dir) is None
    assert "whole-prim deletion is not supported" in context["usdz_packaging_error"]


def test_package_usdz_preserves_nested_custom_data_source_asset(
    tmp_path: Path,
) -> None:
    """Portable rewrites nested in customData remain source-package paths."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Sdf, Usd
    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    metadata_bytes = b"nested-custom-data-texture"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "scene.usda",
            """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root" (
    customData = {
        dictionary nested = {
            asset texture = @Textures/metadata.png@
        }
    }
)
{
}
""",
        )
        package.writestr("Textures/metadata.png", metadata_bytes)

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared_input.usd"
    assert export_stage_portably(
        source_stage,
        prepared_usd,
        approved_dependency_roots=(input_dir, prepared_dir),
    )
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    prepared_value = prepared_stage.GetPrimAtPath("/Root").GetCustomData()["nested"][
        "texture"
    ]
    assert isinstance(prepared_value, Sdf.AssetPath)
    assert "prepared_input.usd_assets" in prepared_value.path

    output_usd = output_dir / "textured_output.usda"
    assert prepared_stage.Flatten().Export(str(output_usd))
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    packaged_value = packaged_stage.GetPrimAtPath("/Root").GetCustomData()["nested"][
        "texture"
    ]
    assert isinstance(packaged_value, Sdf.AssetPath)
    assert packaged_value.path == "Textures/metadata.png"
    with zipfile.ZipFile(packaged_path) as package:
        assert package.read("Textures/metadata.png") == metadata_bytes


def test_prepare_source_usdz_ignores_prepare_only_udim_localization(
    tmp_path: Path,
) -> None:
    """A prepared UDIM rewrite does not replace the source package template."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd
    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    tiles = {
        "Textures/olive.1001.png": b"olive-tile-1001",
        "Textures/olive.1002.png": b"olive-tile-1002",
    }
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "scene.usda",
            """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Material "Olive_Drab_Matte"
    {
        asset inputs:base_color_texture_file = @Textures/olive.<UDIM>.png@
    }
}
""",
        )
        for member, data in tiles.items():
            package.writestr(member, data)

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared_input.usd"
    assert export_stage_portably(
        source_stage,
        prepared_usd,
        approved_dependency_roots=(input_dir, prepared_dir),
    )
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    assert prepared_stage is not None
    output_usd = output_dir / "textured_output.usda"
    assert prepared_stage.Flatten().Export(str(output_usd))
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None
    packaged_stage = Usd.Stage.Open(str(staged_root))
    packaged_value = (
        packaged_stage.GetPrimAtPath("/Root/Olive_Drab_Matte")
        .GetAttribute("inputs:base_color_texture_file")
        .Get()
    )
    assert packaged_value.path == "Textures/olive.<UDIM>.png"
    extract_root = Path(context["source_usdz_extract_root"])
    for member, data in tiles.items():
        assert (extract_root / member).read_bytes() == data


def test_package_usdz_localizes_nested_string_texture_for_flattened_render(
    tmp_path: Path,
) -> None:
    """Render-only String texture opinions survive anonymous stage flattening."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "scene.usdz"
    texture_path = tmp_path / "paint.png"
    Image.new("RGB", (2, 2), (35, 95, 155)).save(texture_path)
    material_layer = """#usda 1.0
over "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                string inputs:diffuse_texture = "../../Textures/paint.png"
            }
        }
    }
}
"""
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            '#usda 1.0\n(\n defaultPrim = "Root"\n'
            " subLayers = [@Libraries/material.usda@]\n)\n"
            'def Xform "Root" {}\n',
        )
        package.writestr("Scene/Libraries/material.usda", material_layer)
        package.write(texture_path, "Textures/paint.png")

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(output_usd))
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    packaged_value = (
        packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Shader")
        .GetAttribute("inputs:diffuse_texture")
        .Get()
    )
    assert not Path(packaged_value).is_absolute()

    render_stage = Usd.Stage.Open(context["source_usdz_stage_path"])
    render_flat = Usd.Stage.Open(render_stage.Flatten())
    render_value = (
        render_flat.GetPrimAtPath("/Root/Looks/Paint/Shader")
        .GetAttribute("inputs:diffuse_texture")
        .Get()
    )
    assert Path(render_value).is_file()
    assert Path(render_value).is_relative_to(Path(context["source_usdz_extract_root"]))
    assert context["render_string_texture_localizations"] == 1


def test_package_usdz_localizes_instance_proxy_string_textures_for_render(
    tmp_path: Path,
) -> None:
    """Render localization reaches referenced shaders without de-instancing."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_usdz = input_dir / "instances.usdz"
    albedo = tmp_path / "albedo.png"
    normal = tmp_path / "normal.png"
    Image.new("RGB", (2, 2), (35, 95, 155)).save(albedo)
    Image.new("RGB", (2, 2), (128, 128, 255)).save(normal)
    root_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "A" (
        instanceable = true
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
    def Xform "B" (
        instanceable = true
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
}
"""
    model_text = """#usda 1.0
def Xform "Model"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                string inputs:diffuse_texture = "Textures/albedo.png"
                token inputs:normal_texture = "Textures/normal.png"
            }
        }
    }
}
"""
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr("Scene/Model/model.usda", model_text)
        package.write(albedo, "Scene/Model/Textures/albedo.png")
        package.write(normal, "Scene/Model/Textures/normal.png")
    uploaded_bytes = source_usdz.read_bytes()

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(root_text, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    render_stage = Usd.Stage.Open(context["source_usdz_stage_path"])
    render_flat = Usd.Stage.Open(render_stage.Flatten())
    for stage in (packaged_stage, render_stage):
        for instance_path in ("/Root/A", "/Root/B"):
            instance = stage.GetPrimAtPath(instance_path)
            assert instance.IsInstanceable()
            assert instance.IsInstance()

    packaged_shader = packaged_stage.GetPrimAtPath("/Root/A/Looks/Paint/Shader")
    for input_name in ("diffuse_texture", "normal_texture"):
        packaged_value = packaged_shader.GetAttribute(f"inputs:{input_name}").Get()
        assert not Path(packaged_value).is_absolute()

    for instance_path in ("/Root/A", "/Root/B"):
        render_shader = render_stage.GetPrimAtPath(
            f"{instance_path}/Looks/Paint/Shader"
        )
        assert render_shader.IsInstanceProxy()
        for input_name in ("diffuse_texture", "normal_texture"):
            render_value = render_shader.GetAttribute(f"inputs:{input_name}").Get()
            assert Path(render_value).is_file()
            assert Path(render_value).is_relative_to(
                Path(context["source_usdz_extract_root"])
            )
            flattened_value = (
                render_flat.GetPrimAtPath(f"{instance_path}/Looks/Paint/Shader")
                .GetAttribute(f"inputs:{input_name}")
                .Get()
            )
            assert Path(flattened_value).is_file()

    assert context["render_string_texture_localizations"] == 2
    assert source_usdz.read_bytes() == uploaded_bytes


@pytest.mark.parametrize("matching_source", [False, True])
def test_package_usdz_rejects_unstaged_package_relative_texture(
    tmp_path: Path,
    matching_source: bool,
) -> None:
    """Only an existing member of the retained source USDZ can be rewritten."""
    import zipfile

    pytest.importorskip("pxr")

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source_usdz = input_dir / "scene.usdz"
    source_root = '#usda 1.0\n(\n    defaultPrim = "Root"\n)\ndef Xform "Root" {}\n'
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("root.usda", source_root)
    prepared_usd = prepared_dir / "prepared_input.usda"
    prepared_usd.write_text(source_root, encoding="utf-8")

    referenced_package = (
        source_usdz if matching_source else input_dir / "different.usdz"
    )
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(
        f"""#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{{
    def Shader "Shader"
    {{
        uniform token info:id = "UsdPreviewSurface"
        asset inputs:diffuse_texture = @{referenced_package}[missing.png]@
    }}
}}
""",
        encoding="utf-8",
    )
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _package_usdz(context, session_dir) is None
    assert context["usdz_source_portability"]["portable"] is False
    assert (
        f"{referenced_package}[missing.png]"
        in context["usdz_source_portability"]["non_relative_texture_paths"]
    )


def test_package_usdz_accepts_more_than_512_small_source_members(
    tmp_path: Path,
) -> None:
    """A valid production package is not rejected by the former 512-entry cap."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr("scene.usda", source_root)
        for index in range(600):
            package.writestr(f"Metadata/member_{index:04d}.txt", str(index))

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    assert Usd.Stage.Open(packaged_path) is not None
    staged_root = Path(context["source_usdz_stage_path"])
    assert len(list((staged_root.parent / "Metadata").glob("*.txt"))) == 600


def test_package_usdz_records_missing_source_archive(
    tmp_path: Path,
) -> None:
    """A missing uploaded source package produces a structured failure."""
    pytest.importorskip("pxr")

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text("#usda 1.0\n", encoding="utf-8")
    context = {
        "usd_path": str(input_dir / "missing.usdz"),
        "output_usd_paths": [str(output_usd)],
    }

    assert _package_usdz(context, session_dir) is None
    assert any(
        item["code"] == "PACKAGE_MISSING_ARTIFACT"
        for item in context["package_diagnostics"]
    )


def test_prepare_source_usdz_stage_rejects_missing_apply_output(
    tmp_path: Path,
) -> None:
    """Dependency preparation never turns a missing apply artifact into success."""
    pytest.importorskip("pxr")

    from ...service.workers import executor

    missing_output = tmp_path / "cache" / "output" / "missing.usda"
    context = {"output_usd_paths": [str(missing_output)]}

    assert executor._prepare_source_usdz_stage(context, tmp_path) is None
    assert context["usdz_packaging_failed"] is True
    assert str(missing_output) in context["usdz_packaging_error"]


def test_prepare_source_usdz_stage_ignores_absent_apply_outputs(
    tmp_path: Path,
) -> None:
    """Preparation is a no-op before ApplyTexturesTask produces an output."""
    from ...service.workers import executor

    assert executor._prepare_source_usdz_stage({}, tmp_path) is None


def test_prepare_source_usdz_stage_reuses_and_repairs_cached_stage(
    tmp_path: Path,
) -> None:
    """A prepared stage is memoized, while a stale marker is rebuilt."""
    import zipfile

    pytest.importorskip("pxr")

    from ...service.workers import executor

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source_usdz = input_dir / "scene.usdz"
    source_root = '#usda 1.0\n\ndef Xform "Root" {}\n'
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/scene.usda", source_root)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    stale_stage = session_dir / "cache" / "missing" / "scene.usda"
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "source_usdz_stage_path": str(stale_stage),
        "render_output_usd_paths": [str(stale_stage)],
    }

    prepared = executor._prepare_source_usdz_stage(context, session_dir)

    assert prepared is not None
    assert prepared.is_file()
    assert prepared != stale_stage
    assert executor._prepare_source_usdz_stage(context, session_dir) == prepared


def test_package_usdz_records_source_archive_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable uploaded packages fail packaging instead of the worker."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd

    from ...service.workers import executor

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w") as package:
        package.writestr("scene.usda", '#usda 1.0\n\ndef Xform "Root" {}\n')

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.DefinePrim("/Root", "Xform")
    stage.GetRootLayer().Save()

    def _raise_read_error(_path: Path) -> Path:
        raise PermissionError("source package is unreadable")

    monkeypatch.setattr(executor, "find_usdz_root_layer", _raise_read_error)
    context: dict[str, Any] = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    assert executor._package_usdz(context, session_dir) is None
    assert context["usdz_packaging_failed"] is True
    assert "Failed to read uploaded source USDZ" in context["usdz_packaging_error"]


@pytest.mark.parametrize(
    "asset_path",
    [
        "/definitely/missing/physics_metal.usda",
        "../../../../../../definitely-missing/physics_metal.usda",
    ],
)
def test_package_usdz_blocks_bound_layer_refs_outside_package_workspace(
    tmp_path: Path,
    asset_path: str,
) -> None:
    """Absolute and escaping local layer refs cannot enter a result package."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    material = UsdShade.Material.Define(stage, "/Root/Materials/physics_metal")
    material.GetPrim().GetReferences().AddReference(asset_path)
    mesh = stage.DefinePrim("/Root/Geometry/Body", "Mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}

    assert _package_usdz(context, session_dir) is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"


def test_package_usdz_blocks_existing_absolute_layer_ref_outside_workspace(
    tmp_path: Path,
) -> None:
    """Existing host files cannot be pulled into a customer USDZ by reference."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    outside_layer = tmp_path / "server-private.usda"
    outside_layer.write_text('#usda 1.0\n\ndef Scope "Private" {}\n', encoding="utf-8")
    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    output_dir.mkdir(parents=True)
    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    material = UsdShade.Material.Define(stage, "/Root/Materials/private")
    material.GetPrim().GetReferences().AddReference(str(outside_layer))
    mesh = stage.DefinePrim("/Root/Geometry/Body", "Mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    stage.GetRootLayer().Save()

    context = {"output_usd_paths": [str(output_usd)]}

    assert _package_usdz(context, session_dir) is None
    assert context["package_diagnostics"][0]["code"] == "PACKAGE_MISSING_ARTIFACT"


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


@pytest.mark.parametrize("texture_suffix", [".jpeg", ".jpg", ".png"])
def test_package_usdz_localizes_original_texture_refs_from_uploaded_usdz(
    tmp_path: Path,
    texture_suffix: str,
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

    texture_name = f"trim_plastic_yellow_02_a{texture_suffix}"
    source_texture = tmp_path / texture_name
    Image.new("RGB", (4, 4), (240, 210, 20)).save(source_texture)
    input_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(input_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_texture, f"textures/{texture_name}")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Trim/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(f"./textures/{texture_name}")
    )
    stage.GetRootLayer().Save()

    context = {
        "usd_path": str(input_usdz),
        "output_usd_paths": [str(output_usd)],
    }
    usdz = _package_usdz(context, session_dir)

    assert usdz is not None
    assert (textures_dir / texture_name).is_file()
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    out_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Trim/Shader")
    )
    assert (
        out_shader.GetInput("diffuse_texture").Get().path
        == f"../textures/{texture_name}"
    )
    assert context["output_portability"]["portable"] is True


def test_package_usdz_keeps_layered_source_textures_out_of_generated_artifacts(
    tmp_path: Path,
) -> None:
    """Asset and string source-package refs stay out of generated artifacts."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_png = tmp_path / "source_albedo.png"
    Image.new("RGB", (4, 4), (42, 88, 130)).save(source_png)
    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Trim"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:source_file = @../textures/source_albedo.png@
                string inputs:diffuse_texture = "../textures/source_albedo.png"
            }
        }
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/scene.usda", source_root)
        package.write(source_png, "textures/source_albedo.png")

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    assert not (session_dir / "cache" / "textures" / "source_albedo.png").exists()
    staged = Usd.Stage.Open(context["source_usdz_stage_path"])
    shader = UsdShade.Shader(staged.GetPrimAtPath("/Root/Looks/Trim/Shader"))
    source_string_ref = shader.GetInput("diffuse_texture").Get()
    assert Path(source_string_ref).is_file()
    assert Path(source_string_ref).is_relative_to(
        Path(context["source_usdz_extract_root"])
    )
    packaged_stage = Usd.Stage.Open(packaged_path)
    packaged_shader = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/Root/Looks/Trim/Shader")
    )
    packaged_ref = packaged_shader.GetInput("source_file").Get()
    assert packaged_ref.resolvedPath
    assert packaged_ref.path == "../textures/source_albedo.png"
    assert (
        packaged_shader.GetInput("diffuse_texture").Get()
        == "../textures/source_albedo.png"
    )


def test_package_usdz_localizes_nonlayered_source_textures_once(
    tmp_path: Path,
) -> None:
    """Asset and string refs share one collision-safe packaged source copy."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Sdf, Usd, UsdShade

    session_dir = tmp_path / "session"
    output_dir = session_dir / "cache" / "output"
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True)
    source_png = source_dir / "shared.png"
    Image.new("RGB", (4, 4), (19, 73, 131)).save(source_png)
    prompts_dir = session_dir / "cache" / "prompts"
    renders_dir = session_dir / "cache" / "renders"
    prompts_dir.mkdir()
    renders_dir.mkdir()
    (prompts_dir / "internal.json").write_text(
        '{"prompt": "must not ship"}',
        encoding="utf-8",
    )
    Image.new("RGB", (4, 4), (255, 0, 255)).save(renders_dir / "preview.png")

    output_usd = output_dir / "textured_output.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Trim/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("asset_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("source/shared.png")
    )
    shader.CreateInput("string_texture", Sdf.ValueTypeNames.String).Set(
        "source/shared.png"
    )
    stage.GetRootLayer().Save()
    context = {"output_usd_paths": [str(output_usd)]}

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    rewritten_stage = Usd.Stage.Open(str(output_usd))
    rewritten_shader = UsdShade.Shader(
        rewritten_stage.GetPrimAtPath("/Root/Looks/Trim/Shader")
    )
    asset_ref = rewritten_shader.GetInput("asset_texture").Get().path
    string_ref = rewritten_shader.GetInput("string_texture").Get()
    assert asset_ref == string_ref
    assert asset_ref.startswith(".texture_agent_source_assets/shared_")
    with zipfile.ZipFile(packaged_path) as package:
        members = package.namelist()
        matching_members = [
            name for name in members if ".texture_agent_source_assets/shared_" in name
        ]
    assert len(matching_members) == 1
    assert not any(name.startswith("prompts/") for name in members)
    assert not any(name.startswith("renders/") for name in members)
    assert not any(name.endswith("source/shared.png") for name in members)


def test_package_usdz_resolves_textures_from_each_authoring_layer(
    tmp_path: Path,
) -> None:
    """Relative texture refs are anchored to their material library layers."""
    import io
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
    subLayers = [
        @../Libraries/A/material.usda@,
        @../Libraries/B/material.usda@
    ]
)

def Xform "Root" {}
"""

    def material_layer(material_name: str) -> str:
        return f"""#usda 1.0

over "Root"
{{
    def Scope "Looks"
    {{
        def Material "{material_name}"
        {{
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:diffuse_texture = @./Textures/shared.png@
            }}
        }}
    }}
}}
"""

    def png_bytes(color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
        return buffer.getvalue()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", source_root)
        package.writestr("Libraries/A/material.usda", material_layer("A"))
        package.writestr("Libraries/B/material.usda", material_layer("B"))
        package.writestr("Libraries/A/Textures/shared.png", png_bytes((220, 20, 20)))
        package.writestr("Libraries/B/Textures/shared.png", png_bytes((20, 20, 220)))

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    assert not (session_dir / "cache" / "textures").exists()
    with zipfile.ZipFile(packaged_path) as package:
        members = package.namelist()
    assert members[0] == "Scene/root.usda"
    assert "Libraries/A/Textures/shared.png" in members
    assert "Libraries/B/Textures/shared.png" in members
    packaged_stage = Usd.Stage.Open(packaged_path)
    resolved_paths = []
    for material_name in ("A", "B"):
        shader = UsdShade.Shader(
            packaged_stage.GetPrimAtPath(f"/Root/Looks/{material_name}/Shader")
        )
        asset = shader.GetInput("diffuse_texture").Get()
        assert asset.resolvedPath
        resolved_paths.append(asset.resolvedPath)
    assert resolved_paths[0] != resolved_paths[1]
    assert context["usdz_source_portability"]["portable"] is True
    assert context["output_portability"]["portable"] is False


def test_package_usdz_scopes_missing_member_fallback_to_owning_unit(
    tmp_path: Path,
) -> None:
    """Another material's generated basename cannot replace a source texture."""
    import io
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Trim"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:orm_texture = @src/Paint_orm.png@
            }
        }
    }
}
"""
    source_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (12, 34, 56)).save(source_buffer, format="PNG")
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
        package.writestr("src/Paint_orm.png", source_buffer.getvalue())

    generated_path = textures_dir / "Paint_orm.png"
    Image.new("RGB", (4, 4), (255, 96, 32)).save(generated_path)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "blended_textures": {
            "Paint": {
                "albedo": "",
                "normal": "",
                "orm": str(generated_path),
            }
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    shader = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Looks/Trim/Shader"))
    packaged_ref = shader.GetInput("orm_texture").Get()
    assert packaged_ref.resolvedPath
    with zipfile.ZipFile(packaged_path) as package:
        matching_members = [
            name
            for name in package.namelist()
            if name.endswith("/Paint_orm.png") or name == "Paint_orm.png"
        ]
        assert len(matching_members) == 1
        packaged_pixel = Image.open(
            io.BytesIO(package.read(matching_members[0]))
        ).getpixel((0, 0))
    assert packaged_pixel == (12, 34, 56)


def test_package_usdz_prefers_owning_units_generated_map_over_source_member(
    tmp_path: Path,
) -> None:
    """An apply-selected generated map wins its same-unit source-name collision."""
    import io
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:albedo_texture = @src/Paint_albedo.png@
            }
        }
    }
}
"""
    source_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (220, 20, 20)).save(source_buffer, format="PNG")
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
        package.writestr("src/Paint_albedo.png", source_buffer.getvalue())

    generated_path = textures_dir / "Paint_albedo.png"
    Image.new("RGB", (4, 4), (20, 220, 20)).save(generated_path)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key="Paint",
                prompt="new paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            "Paint": {
                "albedo": str(generated_path),
                "normal": "",
                "orm": "",
            }
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    shader = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Shader"))
    packaged_ref = shader.GetInput("albedo_texture").Get()
    assert ".texture_agent_generated_textures" in packaged_ref.path
    with zipfile.ZipFile(packaged_path) as package:
        generated_member = next(
            name
            for name in package.namelist()
            if name.endswith("/Paint_albedo.png")
            and ".texture_agent_generated_textures" in name
        )
        packaged_pixel = Image.open(
            io.BytesIO(package.read(generated_member))
        ).getpixel((0, 0))
    assert packaged_pixel == (20, 220, 20)


def test_package_usdz_preserves_ambiguous_and_opacity_source_textures(
    tmp_path: Path,
) -> None:
    """Filename hints cannot override ambiguous or non-PBR source consumers."""
    import io
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            token outputs:surface.connect = </Root/Looks/Paint/Preview.outputs:surface>
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                float inputs:metallic.connect = </Root/Looks/Paint/Ambiguous.outputs:g>
                float inputs:opacity.connect = </Root/Looks/Paint/Mask.outputs:r>
                float inputs:roughness.connect = </Root/Looks/Paint/Ambiguous.outputs:r>
                token outputs:surface
            }
            def Shader "Ambiguous"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @src/Paint_Roughness_Metallic.png@
                float outputs:g
                float outputs:r
            }
            def Shader "Mask"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @src/Paint_albedo_mask.png@
                float outputs:r
            }
        }
    }
}
"""

    def _png(color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
        return buffer.getvalue()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
        package.writestr(
            "src/Paint_Roughness_Metallic.png",
            _png((20, 40, 60)),
        )
        package.writestr("src/Paint_albedo_mask.png", _png((80, 100, 120)))
    for channel, color in (
        ("albedo", (220, 20, 20)),
        ("roughness", (20, 220, 20)),
        ("metalness", (20, 20, 220)),
    ):
        Image.new("RGB", (4, 4), color).save(textures_dir / f"Paint_{channel}.png")
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key="Paint",
                prompt="paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            "Paint": {
                "albedo": str(textures_dir / "Paint_albedo.png"),
                "normal": "",
                "orm": "",
                "roughness": str(textures_dir / "Paint_roughness.png"),
                "metalness": str(textures_dir / "Paint_metalness.png"),
            }
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    for shader_name, expected in (
        ("Ambiguous", "Paint_Roughness_Metallic.png"),
        ("Mask", "Paint_albedo_mask.png"),
    ):
        shader = UsdShade.Shader(
            packaged_stage.GetPrimAtPath(f"/Root/Looks/Paint/{shader_name}")
        )
        packaged_ref = shader.GetInput("file").Get()
        assert Path(packaged_ref.path).name == expected
        assert ".texture_agent_generated_textures" not in packaged_ref.path
        assert packaged_ref.resolvedPath


def test_package_usdz_prefers_exact_material_path_over_colliding_unit_name(
    tmp_path: Path,
) -> None:
    """A different unit key matching the material name cannot steal its map."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Paint"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:orm_texture = @1/legacy_orm.png@
            }
        }
        def Material "Other"
        {
        }
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")

    correct_key = "tu_aaaaaaaaaaaaaaaaaaaa"
    wrong_key = "Paint"
    correct_orm = textures_dir / f"{correct_key}_orm.png"
    wrong_orm = textures_dir / f"{wrong_key}_orm.png"
    Image.new("RGB", (4, 4), (220, 20, 20)).save(correct_orm)
    Image.new("RGB", (4, 4), (20, 20, 220)).save(wrong_orm)
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key=correct_key,
                prompt="paint",
                opacity=1.0,
            ),
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Other",
                    name="Other",
                ),
                key=wrong_key,
                prompt="other",
                opacity=1.0,
            ),
        ],
        "blended_textures": {
            correct_key: {"albedo": "", "normal": "", "orm": str(correct_orm)},
            wrong_key: {"albedo": "", "normal": "", "orm": str(wrong_orm)},
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    shader = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Paint"))
    packaged_ref = shader.GetInput("orm_texture").Get()
    assert packaged_ref.resolvedPath
    assert Path(packaged_ref.path).name == f"{correct_key}_orm.png"


def test_package_usdz_avoids_generated_texture_namespace_collision(
    tmp_path: Path,
) -> None:
    """Generated maps cannot overwrite an identically named source member."""
    import io
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:diffuse_texture = @../textures/Paint_albedo.png@
            }
        }
    }
}
"""

    def _png_bytes(color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
        return buffer.getvalue()

    source_usdz = input_dir / "scene.usdz"
    source_member = ".texture_agent_generated_textures/Paint_albedo.png"
    source_bytes = _png_bytes((220, 20, 20))
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", source_root)
        package.writestr(source_member, source_bytes)

    generated_path = textures_dir / "Paint_albedo.png"
    Image.new("RGB", (4, 4), (20, 220, 20)).save(generated_path)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "blended_textures": {
            "Paint": {
                "albedo": str(generated_path),
                "normal": "",
                "orm": "",
            }
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    generated_member_dir = context["usdz_generated_textures_member"]
    assert generated_member_dir != ".texture_agent_generated_textures"
    generated_member = f"{generated_member_dir}/Paint_albedo.png"
    with zipfile.ZipFile(packaged_path) as package:
        assert package.read(source_member) == source_bytes
        generated_pixel = Image.open(
            io.BytesIO(package.read(generated_member))
        ).getpixel((0, 0))
    assert generated_pixel == (20, 220, 20)
    packaged_stage = Usd.Stage.Open(packaged_path)
    shader = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Shader"))
    packaged_ref = shader.GetInput("diffuse_texture").Get()
    assert packaged_ref.resolvedPath
    assert generated_member_dir in packaged_ref.path


def test_package_usdz_does_not_rewrite_uri_on_generated_basename_match(
    tmp_path: Path,
) -> None:
    """A generated map cannot satisfy a URI merely by sharing its basename."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    uri = "https://textures.example/Paint_orm.png"
    source_root = f"""#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{{
    def Scope "Looks"
    {{
        def Material "Paint"
        {{
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:orm_texture = @{uri}@
            }}
        }}
    }}
}}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
    generated_path = textures_dir / "Paint_orm.png"
    Image.new("RGB", (4, 4), (255, 96, 32)).save(generated_path)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "blended_textures": {
            "Paint": {
                "albedo": "",
                "normal": "",
                "orm": str(generated_path),
            }
        },
    }

    assert _package_usdz(context, session_dir) is None
    staged = Usd.Stage.Open(context["source_usdz_stage_path"])
    shader = UsdShade.Shader(staged.GetPrimAtPath("/Root/Looks/Paint/Shader"))
    assert shader.GetInput("orm_texture").Get().path == uri


def test_package_usdz_rewrites_missing_source_member_to_current_run_texture(
    tmp_path: Path,
) -> None:
    """Adam regression: a missing package-relative ORM uses this run's map."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:orm_texture = @1/legacy.png@
            }
        }
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    unit_key = "tu_0123456789abcdefabcd"
    orm_path = textures_dir / f"{unit_key}_orm.png"
    Image.new("RGB", (4, 4), (255, 96, 32)).save(orm_path)
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key=unit_key,
                prompt="paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            unit_key: {
                "albedo": "",
                "normal": "",
                "orm": str(orm_path),
            }
        },
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    shader = UsdShade.Shader(packaged_stage.GetPrimAtPath("/Root/Looks/Paint/Shader"))
    packaged_ref = shader.GetInput("orm_texture").Get()
    assert packaged_ref.resolvedPath
    assert Path(packaged_ref.path).name == f"{unit_key}_orm.png"


def test_package_usdz_does_not_infer_channel_from_embedded_substring(
    tmp_path: Path,
) -> None:
    """A basename like platform.png must not be mistaken for an ORM channel."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:surface_texture = @1/platform.png@
            }
        }
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    unit_key = "tu_0123456789abcdefabcd"
    orm_path = textures_dir / f"{unit_key}_orm.png"
    Image.new("RGB", (4, 4), (255, 96, 32)).save(orm_path)
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": [
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key=unit_key,
                prompt="paint",
                opacity=1.0,
            )
        ],
        "blended_textures": {
            unit_key: {
                "albedo": "",
                "normal": "",
                "orm": str(orm_path),
            }
        },
    }

    assert _package_usdz(context, session_dir) is None
    staged = Usd.Stage.Open(context["source_usdz_stage_path"])
    shader = UsdShade.Shader(staged.GetPrimAtPath("/Root/Looks/Paint/Shader"))
    assert shader.GetInput("surface_texture").Get().path == "1/platform.png"


def test_package_usdz_scopes_stable_units_by_full_material_path(
    tmp_path: Path,
) -> None:
    """Same-named materials resolve missing maps through their exact plan unit."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)

    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Xform "A"
    {
        def Scope "Looks"
        {
            def Material "Paint"
            {
                def Shader "Shader"
                {
                    uniform token info:id = "UsdPreviewSurface"
                    asset inputs:orm_texture = @1/Paint_orm.png@
                }
            }
        }
    }
    def Xform "B"
    {
        def Scope "Looks"
        {
            def Material "Paint"
            {
                def Shader "Shader"
                {
                    uniform token info:id = "UsdPreviewSurface"
                    asset inputs:orm_texture = @1/Paint_orm.png@
                }
            }
        }
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")

    unit_specs = (
        ("tu_aaaaaaaaaaaaaaaaaaaa", "/Root/A/Looks/Paint", (220, 20, 20)),
        ("tu_bbbbbbbbbbbbbbbbbbbb", "/Root/B/Looks/Paint", (20, 20, 220)),
    )
    units = []
    blended = {}
    for unit_key, material_path, color in unit_specs:
        orm_path = textures_dir / f"{unit_key}_orm.png"
        Image.new("RGB", (4, 4), color).save(orm_path)
        units.append(
            PrimTextureUnit(
                prim_path="",
                material_info=MaterialInfo(
                    prim_path=material_path,
                    name="Paint",
                ),
                key=unit_key,
                prompt="paint",
                opacity=1.0,
            )
        )
        blended[unit_key] = {
            "albedo": "",
            "normal": "",
            "orm": str(orm_path),
        }
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": units,
        "blended_textures": blended,
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    for unit_key, material_path, _color in unit_specs:
        shader = UsdShade.Shader(
            packaged_stage.GetPrimAtPath(f"{material_path}/Shader")
        )
        packaged_ref = shader.GetInput("orm_texture").Get()
        assert packaged_ref.resolvedPath
        assert Path(packaged_ref.path).name == f"{unit_key}_orm.png"


def test_package_usdz_scopes_per_prim_stable_units_by_clone_name(
    tmp_path: Path,
) -> None:
    """Per-prim stable IDs remain unambiguous on cloned material prims."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd, UsdShade

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    textures_dir = session_dir / "cache" / "textures"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    unit_specs = (
        ("tu_aaaaaaaaaaaaaaaaaaaa", "/Root/A", (220, 20, 20)),
        ("tu_bbbbbbbbbbbbbbbbbbbb", "/Root/B", (20, 20, 220)),
    )
    material_defs = "\n".join(
        f'''
        def Material "{unit_key}"
        {{
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                asset inputs:orm_texture = @1/legacy.png@
            }}
        }}'''
        for unit_key, _prim_path, _color in unit_specs
    )
    source_root = f"""#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{{
    def Scope "Looks"
    {{
{material_defs}
    }}
}}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("textured_output.usda", source_root)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")

    units = []
    blended = {}
    for unit_key, prim_path, color in unit_specs:
        orm_path = textures_dir / f"{unit_key}_orm.png"
        Image.new("RGB", (4, 4), color).save(orm_path)
        units.append(
            PrimTextureUnit(
                prim_path=prim_path,
                material_info=MaterialInfo(
                    prim_path="/Root/Looks/Paint",
                    name="Paint",
                ),
                key=unit_key,
                prompt="paint",
                opacity=1.0,
            )
        )
        blended[unit_key] = {
            "albedo": "",
            "normal": "",
            "orm": str(orm_path),
        }
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "prim_texture_units": units,
        "blended_textures": blended,
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None
    packaged_stage = Usd.Stage.Open(packaged_path)
    for unit_key, _prim_path, _color in unit_specs:
        shader = UsdShade.Shader(
            packaged_stage.GetPrimAtPath(f"/Root/Looks/{unit_key}/Shader")
        )
        packaged_ref = shader.GetInput("orm_texture").Get()
        assert packaged_ref.resolvedPath
        assert Path(packaged_ref.path).name == f"{unit_key}_orm.png"


def test_package_usdz_rejects_layered_texture_outside_archive_root(
    tmp_path: Path,
) -> None:
    """A cache-local file is not portable unless the layered archive ships it."""
    import zipfile

    pytest.importorskip("pxr")
    from PIL import Image

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source_root = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            asset inputs:base_color_texture_file = @../../outside.png@
        }
    }
}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", source_root)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(source_root, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }
    assert _prepare_source_usdz_stage(context, session_dir) is not None
    extract_root = Path(context["source_usdz_extract_root"])
    outside_texture = extract_root.parent / "outside.png"
    Image.new("RGB", (4, 4), (20, 80, 140)).save(outside_texture)

    assert _package_usdz(context, session_dir) is None
    assert context["usdz_source_portability"]["portable"] is False
    assert (
        "../../outside.png"
        in context["usdz_source_portability"]["non_relative_texture_paths"]
    )


def test_prepare_source_usdz_stage_rejects_lost_composed_prims(
    tmp_path: Path,
) -> None:
    """An edited root that drops a source sublayer must fail reconstruction."""
    import zipfile

    pytest.importorskip("pxr")

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            '#usda 1.0\n(\n defaultPrim = "Root"\n subLayers = [@Payload.usda@]\n)\n',
        )
        package.writestr(
            "Scene/Payload.usda",
            """#usda 1.0
def Xform "Root"
{
    def Mesh "Body"
    {
    }
}
""",
        )
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(
        '#usda 1.0\n(\n defaultPrim = "Root"\n)\ndef Xform "Root" {}\n',
        encoding="utf-8",
    )
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    assert _prepare_source_usdz_stage(context, session_dir) is None
    assert context["usdz_packaging_failed"] is True
    diagnostic = context["package_diagnostics"][-1]
    assert diagnostic["code"] == "PACKAGE_MISSING_ARTIFACT"
    assert diagnostic["details"]["missing_prim_paths"] == ["/Root/Body"]


def test_prepare_source_usdz_stage_overlays_prepared_edits_without_losing_variants(
    tmp_path: Path,
) -> None:
    """Prepared flattening is an overlay; source variants remain selectable."""
    import zipfile

    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    source_layer = tmp_path / "source.usda"
    payload_layer = tmp_path / "payload.usda"
    payload_stage = Usd.Stage.CreateNew(str(payload_layer))
    payload_root = payload_stage.DefinePrim("/PayloadRoot", "Xform")
    payload_stage.SetDefaultPrim(payload_root)
    payload_stage.DefinePrim("/PayloadRoot/LoadedPayload", "Mesh")
    payload_stage.GetRootLayer().Save()
    source_stage = Usd.Stage.CreateNew(str(source_layer))
    root = source_stage.DefinePrim("/Root", "Xform")
    source_stage.SetDefaultPrim(root)
    source_stage.DefinePrim("/Root/Dormant", "Mesh").SetActive(False)
    payload_host = source_stage.DefinePrim("/Root/PayloadHost", "Xform")
    payload_host.GetPayloads().AddPayload("payload.usda", "/PayloadRoot")
    source_stage.SetDefaultPrim(payload_host)
    model = root.GetVariantSets().AddVariantSet("model")
    model.AddVariant("A")
    model.AddVariant("B")
    model.SetVariantSelection("A")
    with model.GetVariantEditContext():
        source_stage.DefinePrim("/Root/BodyA", "Mesh")
    model.SetVariantSelection("B")
    with model.GetVariantEditContext():
        source_stage.DefinePrim("/Root/BodyB", "Mesh")
    model.SetVariantSelection("A")
    source_stage.GetRootLayer().Save()

    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_layer, "Scene/root.usda")
        package.write(payload_layer, "Scene/payload.usda")
        package.writestr("Scene/.texture_agent_source_root.usda", "source sentinel")
        package.writestr(
            "Scene/.texture_agent_prepared_overlay.usda",
            "overlay sentinel",
        )

    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    prepared_text = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
    def Mesh "BodyA"
    {
    }
    def Xform "PayloadHost"
    {
        def Mesh "LoadedPayload"
        {
        }
    }
}
"""
    output_text = (
        prepared_text.rsplit("}\n", maxsplit=1)[0]
        + """    def Scope "AppliedTextureMarker"
    {
    }
}
"""
    )
    prepared_usd.write_text(prepared_text, encoding="utf-8")
    output_usd.write_text(output_text, encoding="utf-8")
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None
    assert context["source_usdz_prepared_edit_layer_paths"]
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert str(reconstructed.GetDefaultPrim().GetPath()) == "/Root/PayloadHost"
    reconstructed_root = reconstructed.GetPrimAtPath("/Root")
    variants = reconstructed_root.GetVariantSets().GetVariantSet("model")
    assert variants.GetVariantNames() == ["A", "B"]
    assert reconstructed.GetPrimAtPath("/Root/AppliedTextureMarker").IsValid()
    assert reconstructed.GetPrimAtPath("/Root/Dormant").IsActive() is False
    assert variants.SetVariantSelection("B")
    assert reconstructed.GetPrimAtPath("/Root/BodyB").IsValid()
    assert not reconstructed.GetPrimAtPath("/Root/BodyA").IsValid()
    assert reconstructed.GetPrimAtPath("/Root/PayloadHost/LoadedPayload").IsValid()
    reconstructed.Unload("/Root/PayloadHost")
    assert not reconstructed.GetPrimAtPath("/Root/PayloadHost/LoadedPayload").IsValid()


def test_package_usdz_rejects_unresolved_layered_dependency_closure(
    tmp_path: Path,
) -> None:
    """Matching incomplete stages cannot hide a missing package sublayer."""
    import zipfile

    pytest.importorskip("pxr")

    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    root_layer = """#usda 1.0
(
    defaultPrim = "Root"
    subLayers = [@Missing.usda@]
)
def Xform "Root" {}
"""
    source_usdz = input_dir / "scene.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_layer)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(root_layer, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    assert _prepare_source_usdz_stage(context, session_dir) is not None
    assert _package_usdz(context, session_dir) is None
    diagnostic = context["package_diagnostics"][-1]
    assert diagnostic["code"] == "PACKAGE_MISSING_ARTIFACT"
    assert diagnostic["details"]["invalid_dependency_count"] == 1
    assert diagnostic["details"]["invalid_dependencies"] == ["Missing.usda"]


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
    assert packaged_atlases == [f"textures/{atlas_name}"]


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


def test_redact_diagnostics_for_stats_rejects_non_list_payload() -> None:
    """Persisted stats accept only the diagnostics list contract."""
    assert _redact_diagnostics_for_stats({"code": "MALFORMED"}) == []


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
