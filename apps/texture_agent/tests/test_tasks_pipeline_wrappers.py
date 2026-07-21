# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import numpy as np
import pytest
from PIL import Image

import texture_agent.tasks.apply_textures as apply_textures_task
import texture_agent.tasks.blend_textures as blend_textures_task
import texture_agent.tasks.discover_materials as discover_materials_task
import texture_agent.tasks.generate_prompts as generate_prompts_task
import texture_agent.tasks.generate_textures as generate_textures_task
import texture_agent.tasks.prepare_uvs as prepare_uvs_task
import texture_agent.tasks.render as render_task
import texture_agent.tasks.render_previews as render_previews_task
from texture_agent.functions.material_discovery import (
    EffectiveMaterialDiscovery,
    MaterialInfo,
    PrimTextureUnit,
)
from texture_agent.functions.texture_generation import GeneratedTextures

pytest.importorskip("pxr")
from pxr import Sdf  # noqa: E402


def _material(name: str, **overrides) -> MaterialInfo:
    data = {
        "prim_path": f"/Root/Looks/{name}",
        "name": name,
        "bound_prim_paths": [f"/Root/{name}_Mesh"],
        "base_color": (0.4, 0.5, 0.6),
        "base_metalness": 0.3,
        "specular_roughness": 0.2,
    }
    data.update(overrides)
    return MaterialInfo(**data)


def _unit(
    name: str = "Steel",
    key: str | None = None,
    prim_path: str = "",
    material_prim_path: str | None = None,
    prompt: str = "prompt",
    opacity: float = 0.8,
) -> PrimTextureUnit:
    material = _material(name, prim_path=material_prim_path or f"/Root/Looks/{name}")
    return PrimTextureUnit(
        prim_path=prim_path,
        material_info=material,
        key=key or name,
        prompt=prompt,
        opacity=opacity,
    )


def _save_png(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (8, 8), color).save(path)
    return str(path)


def _resolve_output_ref(output_path: Path, ref: str) -> Path:
    return (output_path.parent / ref).resolve()


def _write_quad_usd(
    path: Path,
    *,
    uvs: list[tuple[float, float]] | None = None,
    interpolation: str = "faceVarying",
) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    if uvs is not None:
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation
        )
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))
    stage.GetRootLayer().Save()
    return path


def _write_two_quad_usd(
    path: Path,
    *,
    target_uvs: list[tuple[float, float]],
    other_uvs: list[tuple[float, float]],
) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    for prim_path, uvs in {
        "/World/TargetMesh": target_uvs,
        "/World/OtherMesh": other_uvs,
    }.items():
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 1, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
        )
        st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uvs]))
    stage.GetRootLayer().Save()
    return path


@dataclass
class _FakeEditLayer:
    permissionToEdit: bool = True


@dataclass
class _FakeEditTarget:
    layer: _FakeEditLayer

    def GetLayer(self) -> _FakeEditLayer:
        return self.layer


@dataclass
class _FakeStage:
    permission_to_edit: bool = True

    def GetEditTarget(self) -> _FakeEditTarget:
        return _FakeEditTarget(_FakeEditLayer(self.permission_to_edit))


@dataclass
class _FakePrimPath:
    is_absolute_root_path: bool = False

    def IsAbsoluteRootPath(self) -> bool:
        return self.is_absolute_root_path


@dataclass
class _FakeParentPrim:
    is_valid: bool = True
    is_absolute_root_path: bool = False
    is_instance_proxy: bool = False
    is_prototype: bool = False
    is_in_prototype: bool = False
    is_defined: bool = False
    is_scope: bool = False
    type_name: str = ""
    specifier: Sdf.Specifier = Sdf.SpecifierDef

    def IsValid(self) -> bool:
        return self.is_valid

    def GetPath(self) -> _FakePrimPath:
        return _FakePrimPath(self.is_absolute_root_path)

    def IsInstanceProxy(self) -> bool:
        return self.is_instance_proxy

    def IsPrototype(self) -> bool:
        return self.is_prototype

    def IsInPrototype(self) -> bool:
        return self.is_in_prototype

    def IsDefined(self) -> bool:
        return self.is_defined

    def IsA(self, schema_type: object) -> bool:
        return self.is_scope

    def GetTypeName(self) -> str:
        return self.type_name

    def GetSpecifier(self) -> Sdf.Specifier:
        return self.specifier


@pytest.mark.parametrize(
    ("parent", "can_define"),
    [
        (_FakeParentPrim(is_valid=False), False),
        (_FakeParentPrim(is_absolute_root_path=True), False),
        (_FakeParentPrim(is_instance_proxy=True), False),
        (_FakeParentPrim(is_prototype=True), False),
        (_FakeParentPrim(is_in_prototype=True), False),
        (_FakeParentPrim(is_defined=True, is_scope=True), False),
        (_FakeParentPrim(is_defined=True, is_scope=False, type_name="Xform"), False),
        (_FakeParentPrim(is_defined=True, is_scope=False, type_name=""), True),
        (_FakeParentPrim(is_defined=False, specifier=Sdf.SpecifierOver), False),
        (_FakeParentPrim(is_defined=False), True),
    ],
)
def test_can_define_parent_scope_respects_usd_authoring_guards(
    parent: _FakeParentPrim, can_define: bool
) -> None:
    assert (
        apply_textures_task._can_define_parent_scope(_FakeStage(), parent) is can_define
    )


def test_can_define_parent_scope_requires_editable_layer() -> None:
    assert (
        apply_textures_task._can_define_parent_scope(
            _FakeStage(permission_to_edit=False), _FakeParentPrim()
        )
        is False
    )


def test_load_cached_blended_textures_uses_complete_texture_sets(
    tmp_path: Path,
) -> None:
    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    _save_png(textures_dir / "Steel_albedo.png", (10, 20, 30))
    _save_png(textures_dir / "Steel_normal.png", (128, 128, 255))
    _save_png(textures_dir / "Steel_orm.png", (255, 80, 10))
    _save_png(textures_dir / "Copper_albedo.png", (40, 50, 60))

    cached = apply_textures_task._load_cached_blended_textures(
        tmp_path,
        [_unit(key="Steel"), _unit(key="Copper")],
    )

    assert set(cached) == {"Steel"}
    assert cached["Steel"].albedo == str(textures_dir / "Steel_albedo.png")
    assert cached["Steel"].normal == str(textures_dir / "Steel_normal.png")
    assert cached["Steel"].orm == str(textures_dir / "Steel_orm.png")


def test_discover_materials_task_persists_summary(tmp_path: Path, monkeypatch) -> None:
    task = discover_materials_task.DiscoverMaterialsTask()
    materials = [_material("Steel", bound_prim_paths=["/Root/A", "/Root/B"])]
    discovery = EffectiveMaterialDiscovery(
        authored_materials=tuple(materials),
        effective_materials=tuple(materials),
        renderable_prim_paths=("/Root/A", "/Root/B"),
        renderable_subset_paths=(),
        skipped_materials=(),
    )
    monkeypatch.setattr(
        discover_materials_task,
        "discover_effective_materials_from_file",
        lambda usd_path, **kwargs: discovery,
    )

    context = {
        "usd_path": "/tmp/input.usd",
        "prim_paths": ["/Root/Looks/Steel"],
        "working_dir": str(tmp_path),
    }
    result = task.run(context)

    assert result["discovered_materials"] == materials
    assert result["effective_materials"] == materials
    assert result["material_discovery_counts"] == {
        "authored_material_count": 1,
        "renderable_prim_count": 2,
        "renderable_subset_count": 0,
        "effective_bound_material_count": 1,
    }
    summary_path = tmp_path / "discovery" / "materials.json"
    assert summary_path.exists()
    assert "Steel" in summary_path.read_text(encoding="utf-8")


def test_generate_prompts_task_handles_empty_materials() -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    context = task.run({"discovered_materials": [], "working_dir": "/tmp"})

    assert context["prim_texture_units"] == []


def test_generate_prompts_task_uses_fallback_when_llm_missing(
    tmp_path: Path, monkeypatch
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()
    materials = [_material("Steel")]
    captured = {}

    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models, "create_chat_model_from_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        generate_prompts_task,
        "_fallback_prompts",
        lambda needs_prompt, user_prompt, default_opacity: {
            "Steel": {"prompt": "fallback steel", "opacity": default_opacity}
        },
    )

    def fake_expand_to_prim_units(materials, material_textures, mode, **kwargs):
        captured["mode"] = mode
        captured["kwargs"] = kwargs
        captured["units"] = [
            PrimTextureUnit(
                prim_path="",
                material_info=materials[0],
                key="Steel",
                prompt=material_textures["Steel"]["prompt"],
                opacity=material_textures["Steel"]["opacity"],
            )
        ]
        return captured["units"]

    monkeypatch.setattr(
        generate_prompts_task,
        "expand_to_prim_units",
        fake_expand_to_prim_units,
    )

    result = task.run(
        {
            "discovered_materials": materials,
            "material_textures": {},
            "auto_prompt_config": {
                "enabled": True,
                "user_prompt": "aged",
                "default_opacity": 0.65,
            },
            "texture_config": {
                "mode": "per_material",
                "detail_policy": "surface_only",
            },
            "working_dir": str(tmp_path),
        }
    )

    assert captured["mode"] == "per_material"
    assert captured["kwargs"]["default_detail_policy"] == "surface_only"
    assert result["material_textures"]["Steel"]["prompt"] == "fallback steel"
    assert result["auto_prompt_additions"]["Steel"]["prompt"] == "fallback steel"
    assert result["prim_texture_units"][0].prompt == "fallback steel"
    assert (tmp_path / "prompts" / "material_prompts.json").exists()


def test_generate_prompts_task_rejects_unbounded_auto_prompt_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    import world_understanding.functions.models.chat_models as chat_models

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("LLM should not be created after limit rejection")

    monkeypatch.setattr(chat_models, "create_chat_model_from_config", fail_if_called)

    with pytest.raises(
        ValueError,
        match=(
            "Auto-prompt would select 11 discovered materials, exceeding "
            "auto_prompt.max_generated_materials=2"
        ),
    ):
        task.run(
            {
                "discovered_materials": [
                    _material(f"Material_{index}") for index in range(11)
                ],
                "material_textures": {},
                "auto_prompt_config": {
                    "enabled": True,
                    "user_prompt": "aged",
                    "max_generated_materials": 2,
                },
                "texture_config": {"mode": "per_material"},
                "working_dir": str(tmp_path),
            }
        )


def test_generate_prompts_task_rejects_boolean_auto_prompt_limit(
    tmp_path: Path,
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    with pytest.raises(
        ValueError,
        match="auto_prompt.max_generated_materials must be an integer",
    ):
        task.run(
            {
                "discovered_materials": [_material("Paint")],
                "material_textures": {},
                "auto_prompt_config": {
                    "enabled": True,
                    "user_prompt": "aged",
                    "max_generated_materials": True,
                },
                "texture_config": {"mode": "per_material"},
                "working_dir": str(tmp_path),
            }
        )


def test_generate_prompts_material_sample_label_falls_back_to_name() -> None:
    material = _material("Paint", prim_path="")

    assert generate_prompts_task._material_sample_label(material) == "Paint"


def test_generate_prompts_task_accepts_auto_prompt_limit_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models, "create_chat_model_from_config", lambda *args, **kwargs: None
    )

    result = task.run(
        {
            "discovered_materials": [
                _material(f"Material_{index}") for index in range(64)
            ],
            "material_textures": {},
            "auto_prompt_config": {
                "enabled": True,
                "user_prompt": "aged",
                "max_generated_materials": 64,
            },
            "texture_config": {
                "mode": "per_material",
                "max_texture_units": 64,
            },
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["material_textures"]) == 64
    assert len(result["auto_prompt_additions"]) == 64
    assert len(result["prim_texture_units"]) == 64


def test_generate_prompts_task_rejects_expanded_texture_unit_fanout(
    tmp_path: Path,
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()
    bound_prim_paths = [f"/Root/Panel_{index:02d}" for index in range(11)]

    with pytest.raises(
        ValueError,
        match=(
            "Texture generation would create 11 expanded texture units, exceeding "
            "texture.max_texture_units=2"
        ),
    ) as exc_info:
        task.run(
            {
                "discovered_materials": [
                    _material(
                        "Paint",
                        bound_prim_paths=bound_prim_paths,
                    )
                ],
                "material_textures": {
                    "Paint": {"prompt": "slightly worn green paint", "opacity": 0.8}
                },
                "auto_prompt_config": {"enabled": False},
                "texture_config": {"mode": "per_prim", "max_texture_units": 2},
                "working_dir": str(tmp_path),
            }
        )
    assert "Paint__Panel_09, ..." in str(exc_info.value)


def test_generate_prompts_task_rejects_boolean_texture_unit_limit(
    tmp_path: Path,
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    with pytest.raises(
        ValueError,
        match="texture.max_texture_units must be an integer",
    ):
        task.run(
            {
                "discovered_materials": [_material("Paint")],
                "material_textures": {
                    "Paint": {"prompt": "slightly worn green paint", "opacity": 0.8}
                },
                "auto_prompt_config": {"enabled": False},
                "texture_config": {"mode": "per_material", "max_texture_units": True},
                "working_dir": str(tmp_path),
            }
        )


def test_generate_prompts_task_skips_missing_materials_when_auto_prompt_disabled(
    tmp_path: Path,
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()
    materials = [_material("Steel"), _material("Copper")]

    result = task.run(
        {
            "discovered_materials": materials,
            "material_textures": {"Steel": {"prompt": "brushed steel", "opacity": 0.7}},
            "auto_prompt_config": {"enabled": False, "user_prompt": "aged"},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
        }
    )

    assert "Copper" not in result["material_textures"]
    assert result["auto_prompt_additions"] == {}
    assert [unit.key for unit in result["prim_texture_units"]] == ["Steel"]


def test_generate_prompts_task_skips_llm_when_every_material_has_prompt(
    tmp_path: Path,
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()
    result = task.run(
        {
            "discovered_materials": [_material("Steel")],
            "material_textures": {"Steel": {"prompt": "brushed", "opacity": 0.5}},
            "auto_prompt_config": {"enabled": True},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
        }
    )

    assert result["auto_prompt_additions"] == {}
    assert result["prim_texture_units"][0].prompt == "brushed"


def test_generate_prompts_resume_loads_cache_with_explicit_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "material_prompts.json").write_text(
        json.dumps(
            {
                "Steel": {"prompt": "cached steel", "opacity": 0.4},
                "Copper": {"prompt": "cached copper", "opacity": 0.6},
            }
        ),
        encoding="utf-8",
    )

    import world_understanding.functions.models.chat_models as chat_models

    def forbid_prompt_backend(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("resumed prompt cache must avoid the LLM backend")

    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        forbid_prompt_backend,
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Steel"), _material("Copper")],
            "material_textures": {
                "Steel": {"prompt": "explicit steel", "opacity": 0.9}
            },
            "auto_prompt_config": {"enabled": True},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
            "resume": True,
        }
    )

    units = {unit.key: unit for unit in result["prim_texture_units"]}
    assert units["Steel"].prompt == "explicit steel"
    assert units["Steel"].opacity == 0.9
    assert units["Copper"].prompt == "cached copper"
    assert units["Copper"].opacity == 0.6
    assert result["auto_prompt_additions"] == {}


def test_cached_apply_missing_prompt_fails_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.models.chat_models as chat_models

    provider_calls: list[bool] = []
    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        lambda *_args, **_kwargs: provider_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="Cached apply requires prompt specs"):
        generate_prompts_task.GeneratePromptsTask().run(
            {
                "discovered_materials": [_material("Steel")],
                "material_textures": {},
                "auto_prompt_config": {"enabled": True},
                "texture_config": {"mode": "per_material"},
                "working_dir": str(tmp_path),
                "cached_apply_only": True,
                "resume": True,
            }
        )

    assert provider_calls == []


def test_legacy_cached_apply_allows_explicit_subset_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.models.chat_models as chat_models

    provider_calls: list[bool] = []
    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        lambda *_args, **_kwargs: provider_calls.append(True),
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Steel"), _material("Copper")],
            "material_textures": {"Steel": {"prompt": "brushed steel", "opacity": 0.8}},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
            "cached_apply_only": True,
            "resume": True,
        }
    )

    assert [unit.key for unit in result["prim_texture_units"]] == ["Steel"]
    assert provider_calls == []


def test_cached_apply_plan_mismatch_fails_instead_of_returning_zero_units(
    tmp_path: Path,
) -> None:
    from texture_agent.tasks.plan_textures import PlanTexturesTask

    material = _material("Steel")
    planned = PlanTexturesTask().run(
        {
            "usd_path": str(tmp_path / "input.usd"),
            "working_dir": str(tmp_path),
            "texture_config": {"backend": "simple_image_gen", "size": 1024},
            "planning_config": {},
            "material_textures": {"Steel": {"prompt": "brushed steel", "opacity": 0.8}},
            "auto_prompt_config": {"enabled": False},
            "discovered_materials": [material],
            "steps": {"generate_textures": {"max_workers": 1}},
            "config": {"input": {}},
        }
    )

    planned.update(
        {
            "discovered_materials": [],
            "cached_apply_only": True,
            "resume": True,
        }
    )
    with pytest.raises(ValueError, match="has no matching runtime unit"):
        generate_prompts_task.GeneratePromptsTask().run(planned)


def test_generate_prompts_task_uses_llm_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = generate_prompts_task.GeneratePromptsTask()

    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        generate_prompts_task,
        "generate_texture_prompts",
        lambda **kwargs: {"Steel": {"prompt": "llm brushed steel", "opacity": 0.9}},
    )

    result = task.run(
        {
            "discovered_materials": [_material("Steel")],
            "material_textures": {},
            "auto_prompt_config": {"enabled": True},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
        }
    )

    assert result["auto_prompt_additions"]["Steel"]["prompt"] == "llm brushed steel"


def test_blend_textures_task_records_unknown_generated_key(tmp_path: Path) -> None:
    task = blend_textures_task.BlendTexturesTask()

    with pytest.raises(RuntimeError, match="1/1 blend operations failed"):
        task.run(
            {
                "prim_texture_units": [],
                "generated_textures": {
                    "Unknown": GeneratedTextures(albedo="", normal="", orm="")
                },
                "blend_config": {"output_size": 16},
                "working_dir": str(tmp_path),
            }
        )


def test_pipeline_executor_reexport_imports() -> None:
    import texture_agent.tasks.pipeline_executor as pipeline_executor

    assert pipeline_executor.__all__ == ["run_pipeline"]


def test_prepare_uvs_task_leaves_input_when_no_fixes(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()

    class FakeLayer:
        def Export(self, _path: str) -> None:
            raise AssertionError("Export should not be called when no fixes are needed")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    fake_stage = FakeStage()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: fake_stage)
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode: 0
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 0)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 0)
    monkeypatch.setattr(
        prepare_uvs_task,
        "inspect_uvs_for_stage",
        lambda stage: {
            "schema_version": "texture-agent-uv-report.v1",
            "summary": {},
            "meshes": [],
        },
    )

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {"uv_mode": "box"},
    }

    result = task.run(context)

    assert result["usd_path"] == "/tmp/original.usd"
    assert result["uv_preparation"]["backend"] == "python"
    assert result["uv_preparation"]["generated"] == 0
    assert result["uv_preparation"]["fixed_interpolation"] == 0
    assert result["uv_preparation"]["normalized"] == 0
    assert Path(result["uv_preparation"]["uv_report_path"]).exists()


def test_prepare_uvs_task_saves_prepared_copy(monkeypatch, tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    exported: list[str] = []

    class FakeLayer:
        def Export(self, path: str) -> None:
            exported.append(path)
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def __init__(self, layer=None):
            self._layer = layer or object()

        def Flatten(self):
            return self._layer

        def GetRootLayer(self):
            return FakeLayer()

    flat_layer = object()

    def fake_open(value):
        if value is flat_layer:
            return FakeStage(flat_layer)
        return FakeStage(flat_layer)

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", fake_open)
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode: 2
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 1)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 3)
    monkeypatch.setattr(
        prepare_uvs_task,
        "inspect_uvs_for_stage",
        lambda stage: {
            "schema_version": "texture-agent-uv-report.v1",
            "summary": {},
            "meshes": [],
        },
    )

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {"uv_mode": "box"},
    }

    result = task.run(context)

    prepared_path = tmp_path / "prepared" / "prepared_input.usd"
    assert result["usd_path"] == str(prepared_path)
    assert exported == [str(prepared_path)]
    assert result["uv_preparation"]["backend"] == "python"
    assert result["uv_preparation"]["generated"] == 2
    assert result["uv_preparation"]["fixed_interpolation"] == 1
    assert result["uv_preparation"]["normalized"] == 0
    assert Path(result["uv_preparation"]["uv_report_path"]).exists()


def test_prepare_uvs_task_falls_back_from_scene_optimizer(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    exported: list[str] = []
    so_call = {}
    fallback_modes = []

    class FakeLayer:
        def Export(self, path: str) -> None:
            exported.append(path)
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: FakeStage())

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        so_call.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                **kwargs,
            }
        )
        raise RuntimeError("SO package missing directory: python")

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )

    def fake_generate_uvs_for_stage(stage, mode):
        fallback_modes.append(mode)
        return 4

    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", fake_generate_uvs_for_stage
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 1)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 2)
    monkeypatch.setattr(
        prepare_uvs_task,
        "inspect_uvs_for_stage",
        lambda stage: {
            "schema_version": "texture-agent-uv-report.v1",
            "summary": {},
            "meshes": [],
        },
    )

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_mode": "planar",
            "uv_projection": "cube",
            "uv_so_backend": "local",
            "uv_allow_remote_fallback": False,
        },
    }

    result = task.run(context)

    flat_path = tmp_path / "prepared" / "prepared_input_flat.usd"
    prepared_path = tmp_path / "prepared" / "prepared_input.usd"
    assert so_call["input_path"] == flat_path
    assert so_call["output_path"] == prepared_path
    assert so_call["projection_type"] == prepare_uvs_task.ProjectionType.CUBE
    assert so_call["backend"] == "local"
    assert so_call["allow_remote_fallback"] is False
    assert so_call["overwrite_existing"] is False
    assert fallback_modes == [prepare_uvs_task.UVProjectionMode.BOX]
    assert exported == [str(flat_path), str(prepared_path)]
    assert result["usd_path"] == str(prepared_path)
    assert result["uv_preparation"]["backend"] == "python"
    assert result["uv_preparation"]["generated"] == 4
    assert result["uv_preparation"]["fixed_interpolation"] == 1
    assert result["uv_preparation"]["normalized"] == 0
    assert Path(result["uv_preparation"]["uv_report_path"]).exists()


def test_prepare_uvs_scene_optimizer_accepts_so_only_projection_without_uv_mode(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    so_call = {}

    class FakeLayer:
        def Export(self, path: str) -> None:
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: FakeStage())

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        so_call.update(kwargs)
        raise RuntimeError("SO unavailable")

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode: 0
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 0)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 0)
    monkeypatch.setattr(
        prepare_uvs_task,
        "inspect_uvs_for_stage",
        lambda stage: {
            "schema_version": "texture-agent-uv-report.v1",
            "summary": {},
            "meshes": [],
        },
    )

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "generate_missing",
            "uv_projection": "spherical",
        },
    }

    result = task.run(context)

    assert so_call["projection_type"] == prepare_uvs_task.ProjectionType.SPHERICAL
    assert result["uv_preparation"]["backend"] == "python"


def test_prepare_uvs_force_projection_overrides_so_overwrite_flag(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    so_call = {}

    class FakeLayer:
        def Export(self, path: str) -> None:
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: FakeStage())

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        so_call.update(kwargs)
        raise RuntimeError("SO unavailable")

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode, **kwargs: 0
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 0)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 0)
    monkeypatch.setattr(
        prepare_uvs_task,
        "inspect_uvs_for_stage",
        lambda stage: {
            "schema_version": "texture-agent-uv-report.v1",
            "summary": {},
            "meshes": [],
        },
    )

    context = {
        "usd_path": "/tmp/original.usd",
        "working_dir": str(tmp_path),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "force_projection",
            "uv_overwrite_existing": False,
        },
    }

    task.run(context)

    assert so_call["overwrite_existing"] is True


def test_prepare_uvs_validate_policy_fails_missing_uvs(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "missing_uvs.usda")

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "validate"},
    }

    with pytest.raises(prepare_uvs_task.UVPreparationError, match="UV_MISSING_ST"):
        task.run(context)

    report_path = tmp_path / "work" / "prepared" / "uv_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "texture-agent-uv-report.v1"
    assert report["policy"] == "validate"
    assert report["summary"]["missing"] == 1
    assert report["meshes"][0]["diagnostics"][0]["code"] == "UV_MISSING_ST"


def test_prepare_uvs_generate_missing_writes_report_and_prepared_usd(
    tmp_path: Path,
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "missing_uvs.usda")

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "generate_missing", "uv_projection": "box"},
    }

    result = task.run(context)

    prepared_path = tmp_path / "work" / "prepared" / "prepared_input.usd"
    assert result["usd_path"] == str(prepared_path)
    assert result["uv_preparation"]["generated"] == 1
    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert report["policy"] == "generate_missing"
    assert report["prepared_usd"] == str(prepared_path)
    assert report["summary"]["missing"] == 0
    assert report["summary"]["valid"] == 1


def test_prepare_uvs_preserves_out_of_range_uvs_by_default(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(
        tmp_path / "tiled_uvs.usda",
        uvs=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.5), (0.0, 1.5)],
    )

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "preserve_or_fix"},
    }

    result = task.run(context)

    assert result["usd_path"] == str(usd_path)
    assert result["uv_preparation"]["normalized"] == 0
    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert report["summary"]["out_of_range"] == 1
    assert report["meshes"][0]["diagnostics"][0]["code"] == "UV_OUT_OF_RANGE"


def test_prepare_uvs_validate_policy_succeeds_for_valid_uvs(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(
        tmp_path / "valid_uvs.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "validate"},
    }

    result = task.run(context)

    assert result["usd_path"] == str(usd_path)
    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert report["policy"] == "validate"
    assert report["summary"]["valid"] == 1


def test_prepare_uvs_validate_policy_ignores_so_only_projection(
    tmp_path: Path,
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(
        tmp_path / "valid_uvs.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "validate", "uv_projection": "spherical"},
    }

    result = task.run(context)

    assert result["usd_path"] == str(usd_path)
    assert result["uv_preparation"]["generated"] == 0


def test_prepare_uvs_invalid_policy_raises(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    with pytest.raises(ValueError, match="Invalid UV policy"):
        task.run(
            {
                "usd_path": str(usd_path),
                "working_dir": str(tmp_path / "work"),
                "texture_config": {"uv_policy": "unknown"},
            }
        )


def test_prepare_uvs_invalid_python_projection_raises(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    with pytest.raises(ValueError, match="Invalid UV projection mode"):
        task.run(
            {
                "usd_path": str(usd_path),
                "working_dir": str(tmp_path / "work"),
                "texture_config": {
                    "uv_policy": "generate_missing",
                    "uv_projection": "spherical",
                },
            }
        )


def test_prepare_uvs_force_projection_replaces_existing_uvs(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    task = prepare_uvs_task.PrepareUVsTask()
    original_uvs = [(0.2, 0.2), (0.2, 0.2), (0.2, 0.2), (0.2, 0.2)]
    usd_path = _write_quad_usd(tmp_path / "existing_uvs.usda", uvs=original_uvs)

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "force_projection", "uv_projection": "box"},
    }

    result = task.run(context)

    assert result["usd_path"].endswith("prepared_input.usd")
    assert result["uv_preparation"]["generated"] == 1
    stage = Usd.Stage.Open(result["usd_path"])
    st = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Mesh")).GetPrimvar("st")
    updated = np.array(st.Get())
    assert not np.allclose(updated, np.array(original_uvs))


def test_prepare_uvs_target_scope_replaces_only_material_texture_prims(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    task = prepare_uvs_task.PrepareUVsTask()
    target_uvs = [(0.2, 0.2)] * 4
    other_uvs = [(0.8, 0.8)] * 4
    usd_path = _write_two_quad_usd(
        tmp_path / "two_meshes.usda",
        target_uvs=target_uvs,
        other_uvs=other_uvs,
    )

    result = task.run(
        {
            "usd_path": str(usd_path),
            "working_dir": str(tmp_path / "work"),
            "texture_config": {
                "uv_policy": "force_projection",
                "uv_projection": "box",
                "uv_scope": "target_prims",
            },
            "material_textures": {
                "Body": {
                    "prompt": "blue body",
                    "prim_paths": ["/World/TargetMesh"],
                }
            },
        }
    )

    assert result["usd_path"].endswith("prepared_input.usd")
    assert result["uv_preparation"]["generated"] == 1
    assert result["uv_preparation"]["uv_scope"] == "target_prims"
    assert result["uv_preparation"]["target_prim_paths"] == ["/World/TargetMesh"]

    stage = Usd.Stage.Open(result["usd_path"])
    updated_target = np.array(
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/TargetMesh"))
        .GetPrimvar("st")
        .Get()
    )
    updated_other = np.array(
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/OtherMesh"))
        .GetPrimvar("st")
        .Get()
    )
    assert not np.allclose(updated_target, np.array(target_uvs))
    np.testing.assert_allclose(updated_other, np.array(other_uvs))

    uv_report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert uv_report["actions"]["target_prim_paths"] == ["/World/TargetMesh"]


def test_prepare_uvs_target_scope_ignores_unrelated_invalid_uvs(
    tmp_path: Path,
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    target_uvs = [(0.2, 0.2)] * 4
    other_uvs = [(np.nan, 0.8)] * 4
    usd_path = _write_two_quad_usd(
        tmp_path / "two_meshes.usda",
        target_uvs=target_uvs,
        other_uvs=other_uvs,
    )

    result = task.run(
        {
            "usd_path": str(usd_path),
            "working_dir": str(tmp_path / "work"),
            "texture_config": {
                "uv_policy": "force_projection",
                "uv_projection": "box",
                "uv_scope": "target_prims",
            },
            "material_textures": {
                "Body": {
                    "prompt": "blue body",
                    "prim_paths": ["/World/TargetMesh"],
                }
            },
        }
    )

    assert result["usd_path"].endswith("prepared_input.usd")
    assert result["uv_preparation"]["generated"] == 1
    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    mesh_statuses = {mesh["prim_path"]: mesh["status"] for mesh in report["meshes"]}
    assert mesh_statuses["/World/TargetMesh"] == "valid"
    assert mesh_statuses["/World/OtherMesh"] == "invalid"


def test_prepare_uvs_python_cube_projection_logs_box_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "missing_uvs.usda")

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "python",
            "uv_policy": "generate_missing",
            "uv_projection": "cube",
        },
    }

    with caplog.at_level(logging.WARNING):
        result = task.run(context)

    assert result["uv_preparation"]["generated"] == 1
    assert "using Python box projection instead" in caplog.text


def test_prepare_uvs_scene_optimizer_skipped_for_preserve_or_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(
        tmp_path / "valid_uvs.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    monkeypatch.setattr(
        prepare_uvs_task,
        "generate_projection_uvs",
        lambda *args, **kwargs: pytest.fail("Scene Optimizer should be skipped"),
    )

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "preserve_or_fix",
        },
    }

    with caplog.at_level(logging.INFO):
        result = task.run(context)

    assert result["usd_path"] == str(usd_path)
    assert "Scene Optimizer UV backend configured but skipped" in caplog.text


def test_prepare_uvs_scene_optimizer_policy_failure_does_not_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        _write_quad_usd(Path(output_path))
        return {"meshes_with_uvs": 0, "extra": Path(output_path)}

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )
    monkeypatch.setattr(
        prepare_uvs_task,
        "generate_uvs_for_stage",
        lambda *args, **kwargs: pytest.fail("Python fallback should not run"),
    )

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "generate_missing",
        },
    }

    with pytest.raises(
        prepare_uvs_task.UVPreparationError,
        match="Scene Optimizer UV preparation left meshes not UV-ready",
    ):
        task.run(context)

    report = json.loads(
        (tmp_path / "work" / "prepared" / "uv_report.json").read_text(encoding="utf-8")
    )
    assert report["actions"]["backend"] == "scene_optimizer"
    assert report["actions"]["so_result"]["extra"].endswith("prepared_input.usd")


def test_prepare_uvs_scene_optimizer_success_sets_prepared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        _write_quad_usd(
            Path(output_path),
            uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "generate_missing",
            "uv_projection": "spherical",
        },
    }

    result = task.run(context)

    assert result["usd_path"].endswith("prepared_input.usd")
    assert result["uv_preparation"]["backend"] == "scene_optimizer"
    assert result["uv_preparation"]["generated"] == 1
    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert report["projection"] == "spherical"
    assert report["actions"]["so_result"]["status"] == "completed"


def test_prepare_uvs_scene_optimizer_atlas_success_sets_prepared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")
    so_call = {}

    def fake_generate_atlas_uvs(input_path, output_path, **kwargs):
        so_call.update(
            {
                "input_path": input_path,
                "output_path": output_path,
                **kwargs,
            }
        )
        _write_quad_usd(
            Path(output_path),
            uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_task,
        "generate_projection_uvs",
        lambda *args, **kwargs: pytest.fail("Projection UVs should not run"),
    )
    monkeypatch.setattr(prepare_uvs_task, "generate_atlas_uvs", fake_generate_atlas_uvs)

    context = {
        "usd_path": str(usd_path),
        "working_dir": str(tmp_path / "work"),
        "texture_config": {
            "uv_backend": "scene_optimizer",
            "uv_policy": "generate_missing",
            "uv_generation_mode": "atlas",
            "uv_overwrite_existing": False,
            "uv_atlas_distortion_threshold": 2.25,
            "uv_atlas_enable_packing": False,
        },
    }

    result = task.run(context)

    assert result["usd_path"].endswith("prepared_input.usd")
    assert result["uv_preparation"]["backend"] == "scene_optimizer"
    assert result["uv_preparation"]["generation_mode"] == "atlas"
    assert result["uv_preparation"]["generated"] == 1
    assert so_call["overwrite_existing"] is False
    assert so_call["distortion_threshold"] == 2.25
    assert so_call["enable_atlas_packing"] is False
    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert report["actions"]["generation_mode"] == "atlas"
    assert report["actions"]["so_result"]["status"] == "completed"


def test_prepare_uvs_scene_optimizer_scoped_paths_are_passed_to_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_two_quad_usd(
        tmp_path / "two_meshes.usda",
        target_uvs=[(0.2, 0.2)] * 4,
        other_uvs=[(0.8, 0.8)] * 4,
    )
    so_call = {}

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        so_call.update(kwargs)
        _write_two_quad_usd(
            Path(output_path),
            target_uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            other_uvs=[(0.8, 0.8)] * 4,
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )
    fix_calls = []
    normalize_calls = []

    def fake_fix_uv_interpolation(stage, **kwargs):
        fix_calls.append(kwargs)
        return 0

    def fake_normalize_uvs(stage, **kwargs):
        normalize_calls.append(kwargs)
        return 0

    monkeypatch.setattr(
        prepare_uvs_task, "fix_uv_interpolation", fake_fix_uv_interpolation
    )
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", fake_normalize_uvs)

    result = task.run(
        {
            "usd_path": str(usd_path),
            "working_dir": str(tmp_path / "work"),
            "texture_config": {
                "uv_backend": "scene_optimizer",
                "uv_policy": "generate_missing",
                "uv_scope": "target_prims",
                "uv_normalize_out_of_range": True,
            },
            "material_textures": {
                "Body": {
                    "prompt": "blue body",
                    "prim_paths": ["/World/TargetMesh"],
                }
            },
        }
    )

    assert so_call["paths"] == ["/World/TargetMesh"]
    assert fix_calls == [{"target_prim_paths": ("/World/TargetMesh",)}]
    assert normalize_calls == [{"target_prim_paths": ("/World/TargetMesh",)}]
    assert result["uv_preparation"]["backend"] == "scene_optimizer"
    assert result["uv_preparation"]["uv_scope"] == "target_prims"
    assert result["uv_preparation"]["target_prim_paths"] == ["/World/TargetMesh"]


def test_prepare_uvs_atlas_requires_scene_optimizer_backend(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    with pytest.raises(ValueError, match="requires texture.uv_backend"):
        task.run(
            {
                "usd_path": str(usd_path),
                "working_dir": str(tmp_path / "work"),
                "texture_config": {"uv_generation_mode": "atlas"},
            }
        )


def test_prepare_uvs_invalid_uv_generation_mode_raises(tmp_path: Path) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    with pytest.raises(ValueError, match="Invalid UV generation mode"):
        task.run(
            {
                "usd_path": str(usd_path),
                "working_dir": str(tmp_path / "work"),
                "texture_config": {
                    "uv_backend": "scene_optimizer",
                    "uv_generation_mode": "bogus",
                },
            }
        )


def test_prepare_uvs_scene_optimizer_atlas_ignores_projection_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()
    usd_path = _write_quad_usd(tmp_path / "input.usda")

    def fake_generate_atlas_uvs(input_path, output_path, **kwargs):
        _write_quad_usd(
            Path(output_path),
            uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    monkeypatch.setattr(prepare_uvs_task, "generate_atlas_uvs", fake_generate_atlas_uvs)

    result = task.run(
        {
            "usd_path": str(usd_path),
            "working_dir": str(tmp_path / "work"),
            "texture_config": {
                "uv_backend": "scene_optimizer",
                "uv_policy": "generate_missing",
                "uv_generation_mode": "atlas",
                "uv_projection": "not-a-projection",
            },
        }
    )

    report = json.loads(
        Path(result["uv_preparation"]["uv_report_path"]).read_text(encoding="utf-8")
    )
    assert result["uv_preparation"]["generation_mode"] == "atlas"
    assert "projection" not in result["uv_preparation"]
    assert report["projection"] == "atlas"


def test_prepare_uvs_scene_optimizer_atlas_fallback_records_source(
    monkeypatch, tmp_path: Path
) -> None:
    task = prepare_uvs_task.PrepareUVsTask()

    class FakeLayer:
        def Export(self, path: str) -> None:
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")

    class FakeStage:
        def Flatten(self):
            return object()

        def GetRootLayer(self):
            return FakeLayer()

    monkeypatch.setattr(prepare_uvs_task.Usd.Stage, "Open", lambda value: FakeStage())
    monkeypatch.setattr(
        prepare_uvs_task,
        "generate_atlas_uvs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("atlas missing")),
    )
    monkeypatch.setattr(
        prepare_uvs_task, "generate_uvs_for_stage", lambda stage, mode, **kwargs: 1
    )
    monkeypatch.setattr(prepare_uvs_task, "fix_uv_interpolation", lambda stage: 0)
    monkeypatch.setattr(prepare_uvs_task, "normalize_uvs", lambda stage: 0)
    monkeypatch.setattr(
        prepare_uvs_task,
        "inspect_uvs_for_stage",
        lambda stage: {
            "schema_version": "texture-agent-uv-report.v1",
            "summary": {},
            "meshes": [],
        },
    )

    result = task.run(
        {
            "usd_path": "/tmp/original.usd",
            "working_dir": str(tmp_path),
            "texture_config": {
                "uv_backend": "scene_optimizer",
                "uv_generation_mode": "atlas",
            },
        }
    )

    assert result["uv_preparation"]["backend"] == "python"
    assert result["uv_preparation"]["fallback_from"]["backend"] == "scene_optimizer"
    assert result["uv_preparation"]["fallback_from"]["generation_mode"] == "atlas"


def test_generate_textures_task_reuses_existing_outputs(tmp_path: Path) -> None:
    task = generate_textures_task.GenerateTexturesTask()
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    albedo = out_dir / "Steel_albedo.png"
    normal = out_dir / "Steel_normal.png"
    orm = out_dir / "Steel_orm.png"
    _save_png(albedo, (10, 20, 30))
    _save_png(normal, (40, 50, 60))
    _save_png(orm, (70, 80, 90))

    result = task.run(
        {
            "prim_texture_units": [_unit()],
            "texture_config": {"skip_existing": True},
            "working_dir": str(tmp_path),
        }
    )

    assert result["generated_textures"]["Steel"] == GeneratedTextures(
        albedo=str(albedo),
        normal=str(normal),
        orm=str(orm),
    )


def test_generate_textures_task_unknown_backend_raises(tmp_path: Path) -> None:
    task = generate_textures_task.GenerateTexturesTask()

    with pytest.raises(ValueError, match="Unknown texture backend"):
        task.run(
            {
                "prim_texture_units": [_unit()],
                "texture_config": {"backend": "mystery"},
                "working_dir": str(tmp_path),
            }
        )


def test_localize_textures_copies_accessible_file_uri(tmp_path: Path) -> None:
    source = tmp_path / "remote_albedo.png"
    _save_png(source, (1, 2, 3))
    localized_dir = tmp_path / "localized"
    localized_dir.mkdir()

    result = generate_textures_task.GenerateTexturesTask._localize_textures(
        GeneratedTextures(
            albedo=f"file://{source}",
            normal="",
            orm="",
        ),
        key="Steel",
        out_dir=localized_dir,
        endpoint="http://service",
    )

    assert Path(result.albedo).exists()
    assert result.normal == ""
    assert result.orm == ""


def test_author_unit_source_usd_deinstances_material_graph(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdShade

    prepared_usd = tmp_path / "prepared.usda"
    output_usd = tmp_path / "unit_source.usda"
    source_albedo = Path(_save_png(tmp_path / "new_albedo.png", (10, 20, 30)))
    stage = Usd.Stage.CreateNew(str(prepared_usd))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    material.GetPrim().SetInstanceable(True)
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/base_color_texture")
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("old_albedo.png")
    )
    shader.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().Save()

    generate_textures_task._author_unit_source_usd(
        prepared_usd=prepared_usd,
        output_usd=output_usd,
        material_path="/Root/Looks/Steel",
        texture_paths={"albedo": source_albedo},
    )

    output_stage = Usd.Stage.Open(str(output_usd))
    output_material = output_stage.GetPrimAtPath("/Root/Looks/Steel")
    output_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Steel/base_color_texture")
    )
    assert output_material.IsInstanceable() is False
    assert output_shader.GetPrim().IsInstanceable() is False
    assert output_shader.GetInput("file").Get().path == str(source_albedo.resolve())


def test_blend_textures_task_creates_outputs(tmp_path: Path) -> None:
    task = blend_textures_task.BlendTexturesTask()
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    albedo = _save_png(generated_dir / "Steel_albedo.png", (200, 100, 50))
    orm = _save_png(generated_dir / "Steel_orm.png", (20, 40, 60))
    unit = _unit(opacity=0.5)

    result = task.run(
        {
            "prim_texture_units": [unit],
            "generated_textures": {
                "Steel": GeneratedTextures(albedo=albedo, normal="", orm=orm)
            },
            "blend_config": {"output_size": 16},
            "working_dir": str(tmp_path),
        }
    )

    blended = result["blended_textures"]["Steel"]
    assert Path(blended.albedo).exists()
    assert Path(blended.normal).exists()
    assert Path(blended.orm).exists()


def test_apply_textures_task_applies_per_material(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    prim = material.GetPrim()
    prim.CreateAttribute("inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset)
    prim.CreateAttribute(
        "inputs:geometry_normal_texture_file", Sdf.ValueTypeNames.Asset
    )
    prim.CreateAttribute(
        "inputs:specular_roughness_texture_file", Sdf.ValueTypeNames.Asset
    )
    prim.CreateAttribute("inputs:base_metalness_texture_file", Sdf.ValueTypeNames.Asset)
    for shader_name in [
        "tiledimage_base_color",
        "tiledimage_geometry_normal",
        "tiledimage_specular_roughness",
        "tiledimage_base_metalness",
    ]:
        shader = UsdShade.Shader.Define(stage, f"/Root/Looks/Steel/{shader_name}")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal=_save_png(textures_dir / "steel_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "steel_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [_unit()],
            "working_dir": str(tmp_path),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    assert output_path.exists()
    output_stage = Usd.Stage.Open(str(output_path))
    assert output_stage.GetPrimAtPath("/Root/Looks").IsA(UsdGeom.Scope)
    output_prim = output_stage.GetPrimAtPath("/Root/Looks/Steel")
    base_ref = output_prim.GetAttribute("inputs:base_color_texture_file").Get().path
    assert base_ref == "../textures/steel_albedo.png"
    assert _resolve_output_ref(output_path, base_ref).is_file()
    assert (
        output_prim.GetAttribute("inputs:base_color_texture_file")
        .Get()
        .path.endswith("steel_albedo.png")
    )
    assert (
        output_prim.GetAttribute("inputs:geometry_normal_texture_file")
        .Get()
        .path.endswith("steel_normal.png")
    )
    assert (
        output_prim.GetAttribute("inputs:specular_roughness_texture_file")
        .Get()
        .path.endswith("Steel_roughness.png")
    )
    assert (
        output_prim.GetAttribute("inputs:base_metalness_texture_file")
        .Get()
        .path.endswith("Steel_metalness.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Steel/tiledimage_base_color")
        )
        .GetInput("file")
        .Get()
        .path.endswith("steel_albedo.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Steel/tiledimage_geometry_normal")
        )
        .GetInput("file")
        .Get()
        .path.endswith("steel_normal.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath(
                "/Root/Looks/Steel/tiledimage_specular_roughness"
            )
        )
        .GetInput("file")
        .Get()
        .path.endswith("Steel_roughness.png")
    )
    assert (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Steel/tiledimage_base_metalness")
        )
        .GetInput("file")
        .Get()
        .path.endswith("Steel_metalness.png")
    )
    assert result["output_portability"]["portable"] is True

    moved_root = tmp_path / "moved_bundle"
    shutil.copytree(output_path.parent, moved_root / "output")
    shutil.copytree(textures_dir, moved_root / "textures")
    moved_output = moved_root / "output" / output_path.name
    moved_stage = Usd.Stage.Open(str(moved_output))
    moved_prim = moved_stage.GetPrimAtPath("/Root/Looks/Steel")
    moved_ref = moved_prim.GetAttribute("inputs:base_color_texture_file").Get().path
    assert moved_ref == "../textures/steel_albedo.png"
    assert _resolve_output_ref(moved_output, moved_ref).is_file()


def test_apply_textures_task_localizes_unedited_material_texture_refs(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    source_dir = tmp_path / "source_package"
    source_textures = source_dir / "textures"
    source_textures.mkdir(parents=True)
    shared_albedo = Path(
        _save_png(source_textures / "shared_trim_albedo.png", (220, 180, 20))
    )
    input_usd = source_dir / "input.usda"

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(input_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    untouched = UsdShade.Material.Define(stage, "/Root/Looks/Untouched")
    untouched.GetPrim().SetInstanceable(True)
    untouched.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(shared_albedo)))
    texture = UsdShade.Shader.Define(stage, "/Root/Looks/Untouched/Image_Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(shared_albedo))
    )
    stage.GetRootLayer().Save()

    working_dir = tmp_path / "run"
    generated_dir = working_dir / "textures"
    generated_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(generated_dir / "Steel_albedo.png", (20, 90, 180)),
        normal=_save_png(generated_dir / "Steel_normal.png", (128, 128, 255)),
        orm=_save_png(generated_dir / "Steel_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(input_usd),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [_unit("Steel")],
            "working_dir": str(working_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    untouched_prim = output_stage.GetPrimAtPath("/Root/Looks/Untouched")
    material_ref = (
        untouched_prim.GetAttribute("inputs:base_color_texture_file").Get().path
    )
    shader_ref = (
        UsdShade.Shader(
            output_stage.GetPrimAtPath("/Root/Looks/Untouched/Image_Texture")
        )
        .GetInput("file")
        .Get()
        .path
    )

    assert material_ref == "../textures/shared_trim_albedo.png"
    assert shader_ref == "../textures/shared_trim_albedo.png"
    assert untouched_prim.IsInstanceable() is False
    assert _resolve_output_ref(output_path, material_ref).is_file()
    assert result["output_portability"]["portable"] is True
    assert sorted(result["apply_textures_stats"]["stage_texture_refs_localized"]) == [
        "/Root/Looks/Untouched/Image_Texture:inputs:file",
        "/Root/Looks/Untouched:inputs:base_color_texture_file",
    ]


def test_apply_textures_task_preserves_typed_material_parent(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    parent = UsdGeom.Xform.Define(stage, "/Root")
    parent.AddTranslateOp().Set((1.0, 2.0, 3.0))
    material = UsdShade.Material.Define(stage, "/Root/Steel")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [_unit(material_prim_path="/Root/Steel")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    output_parent = output_stage.GetPrimAtPath("/Root")
    assert output_parent.IsA(UsdGeom.Xform)
    assert output_parent.GetTypeName() == "Xform"
    assert tuple(output_parent.GetAttribute("xformOp:translate").Get()) == (
        1.0,
        2.0,
        3.0,
    )


def test_apply_textures_task_overrides_prebaked_mdl_inputs(tmp_path: Path) -> None:
    """SimReady MDL materials can have pre-baked Nucleus texture inputs
    (e.g. inputs:normalmap_texture) that the agent must overwrite with freshly
    generated local textures, otherwise
    the output USD silently keeps the original references and renders broken
    once the bundle is downloaded outside Omniverse.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    # Mark the shader as MDL-sourced so the override path triggers.
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # Pre-bake Nucleus-hosted texture inputs (these would survive into the
    # output without the fix and produce broken refs after the service's
    # absolute → ../textures/<filename> rewrite step).
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Albedo.png")
    )
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Normal.png")
    )
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_ORM.png")
    )
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Opacity.png")
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert (
        out_shader.GetInput("diffuse_texture").Get().path.endswith("Plastic_albedo.png")
    )
    assert (
        out_shader.GetInput("normalmap_texture")
        .Get()
        .path.endswith("Plastic_normal.png")
    )
    assert out_shader.GetInput("ORM_texture").Get().path.endswith("Plastic_orm.png")
    # Inputs we cannot map to a generated channel must be cleared (not left
    # pointing at the original Nucleus PNG), otherwise the service packager's
    # absolute → ../textures/<basename> rewrite step would create a dangling
    # reference to a file the bundle does not ship.
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_overridden"] >= 3
    assert any("opacity_texture" in entry for entry in stats["mdl_inputs_cleared"])


def test_apply_textures_task_overrides_usd_preview_texture_nodes(
    tmp_path: Path,
) -> None:
    """Existing UsdPreviewSurface graphs must render generated textures.

    SimReady materials can carry both OmniPBR MDL and UsdPreviewSurface shader
    graphs. If the renderer cannot resolve OmniPBR, it falls back to preview
    shaders; those preview texture nodes must point at the generated maps too.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    albedo = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/AlbedoTexture")
    albedo.CreateIdAttr("UsdUVTexture")
    albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_albedo.png")
    )
    albedo_rgb = albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo_rgb
    )

    roughness = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/RoughnessTexture")
    roughness.CreateIdAttr("UsdUVTexture")
    roughness.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_roughness.png")
    )
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        roughness.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )
    metalness = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/MetalnessTexture")
    metalness.CreateIdAttr("UsdUVTexture")
    metalness.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_metalness.png")
    )
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
        metalness.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )

    normal = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/NormalTexture")
    normal.CreateIdAttr("UsdUVTexture")
    normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_normal.png")
    )
    preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )

    occlusion = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/OcclusionTexture")
    occlusion.CreateIdAttr("UsdUVTexture")
    occlusion.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_orm.png")
    )
    preview.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(
        occlusion.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_albedo = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/AlbedoTexture")
    )
    out_roughness = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/RoughnessTexture")
    )
    out_metalness = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/MetalnessTexture")
    )
    out_normal = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/NormalTexture")
    )
    out_occlusion = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/OcclusionTexture")
    )

    assert out_albedo.GetInput("file").Get().path.endswith("Plastic_albedo.png")
    assert out_roughness.GetInput("file").Get().path.endswith("Plastic_roughness.png")
    assert out_metalness.GetInput("file").Get().path.endswith("Plastic_metalness.png")
    assert out_normal.GetInput("file").Get().path.endswith("Plastic_normal.png")
    assert out_occlusion.GetInput("file").Get().path.endswith("Plastic_orm.png")
    out_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Preview")
    )
    for input_name, expected_node in {
        "diffuseColor": "AlbedoTexture",
        "roughness": "RoughnessTexture",
        "metallic": "MetalnessTexture",
        "normal": "NormalTexture",
        "occlusion": "OcclusionTexture",
    }.items():
        source = out_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName() == expected_node
    assert not output_stage.GetPrimAtPath(
        "/Root/Looks/Plastic/TextureAgentSTReader"
    ).IsValid()
    stats = result["apply_textures_stats"]
    assert len(stats["preview_texture_inputs_overridden"]) == 5


def test_apply_textures_task_authors_constant_preview_surface_texture_graph(
    tmp_path: Path,
) -> None:
    """Persist a renderable graph for backend-neutral blended texture maps."""
    from pxr import Gf, Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.1, 0.2, 0.3)
    )
    preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).Set(
        Gf.Vec3f(0.0, 0.0, 1.0)
    )
    preview.CreateInput("occlusion", Sdf.ValueTypeNames.Float).Set(1.0)
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(input_path),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    # Reopen the exported artifact so the assertions cover persisted package
    # content rather than only the in-memory edit target.
    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    output_material = UsdShade.Material(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic")
    )
    surface_sources, _ = output_material.GetSurfaceOutput().GetConnectedSources()
    assert len(surface_sources) == 1
    assert surface_sources[0].source.GetPrim().GetPath() == Sdf.Path(
        "/Root/Looks/Plastic/Surface"
    )

    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Surface")
    )
    expected = {
        "diffuseColor": (
            "TextureAgentAlbedoTexture",
            "Plastic_albedo.png",
            "rgb",
            "sRGB",
        ),
        "normal": (
            "TextureAgentNormalTexture",
            "Plastic_normal.png",
            "rgb",
            "raw",
        ),
        "occlusion": (
            "TextureAgentORMTexture",
            "Plastic_orm.png",
            "r",
            "raw",
        ),
        "roughness": (
            "TextureAgentRoughnessTexture",
            "Plastic_roughness.png",
            "r",
            "raw",
        ),
        "metallic": (
            "TextureAgentMetalnessTexture",
            "Plastic_metalness.png",
            "r",
            "raw",
        ),
    }
    for input_name, (
        expected_node,
        expected_file,
        expected_output,
        expected_color_space,
    ) in expected.items():
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        texture = UsdShade.Shader(source[0].GetPrim())
        assert texture.GetPrim().GetName() == expected_node
        assert texture.GetIdAttr().Get() == "UsdUVTexture"
        assert source[1] == expected_output
        assert texture.GetInput("file").Get().path.endswith(expected_file)
        assert texture.GetInput("sourceColorSpace").Get() == expected_color_space
        st_source = texture.GetInput("st").GetConnectedSource()
        assert st_source is not None
        st_reader = UsdShade.Shader(st_source[0].GetPrim())
        assert st_reader.GetPrim().GetName() == "TextureAgentSTReader"
        assert st_reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
        assert st_reader.GetInput("varname").Get() == "st"

    normal = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/TextureAgentNormalTexture")
    )
    assert tuple(normal.GetInput("scale").Get()) == (2.0, 2.0, 2.0, 2.0)
    assert tuple(normal.GetInput("bias").Get()) == (-1.0, -1.0, -1.0, 0.0)
    assert result["output_portability"]["portable"] is True
    assert len(
        result["apply_textures_stats"]["preview_texture_inputs_overridden"]
    ) == len(expected)


def test_apply_textures_task_constant_preview_authors_only_available_maps(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.2, 0.3))
    preview.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).Set((0.0, 0.0, 1.0))
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(input_path),
            "blended_textures": {
                "Plastic": apply_textures_task.BlendedTextures(
                    albedo=_save_png(
                        textures_dir / "Plastic_albedo.png", (200, 50, 50)
                    ),
                    normal="",
                    orm="",
                )
            },
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Surface")
    )
    albedo_source = output_preview.GetInput("diffuseColor").GetConnectedSource()
    assert albedo_source is not None
    assert albedo_source[0].GetPrim().GetName() == "TextureAgentAlbedoTexture"
    for input_name in ("normal", "roughness", "metallic"):
        assert output_preview.GetInput(input_name).GetConnectedSource() is None
    for node_name in (
        "TextureAgentNormalTexture",
        "TextureAgentORMTexture",
        "TextureAgentRoughnessTexture",
        "TextureAgentMetalnessTexture",
    ):
        assert not output_stage.GetPrimAtPath(
            f"/Root/Looks/Plastic/{node_name}"
        ).IsValid()
    assert result["apply_textures_stats"]["preview_texture_inputs_overridden"] == [
        "/Root/Looks/Plastic/TextureAgentAlbedoTexture:file"
    ]


def test_apply_textures_task_fills_partial_preview_graph_without_replacing_existing_node(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.2, 0.3))
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    existing_roughness = UsdShade.Shader.Define(
        stage, "/Root/Looks/Plastic/ExistingRoughnessTexture"
    )
    existing_roughness.CreateIdAttr("UsdUVTexture")
    existing_roughness.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_roughness.png")
    )
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        existing_roughness.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(input_path),
            "blended_textures": {
                "Plastic": apply_textures_task.BlendedTextures(
                    albedo=_save_png(
                        textures_dir / "Plastic_albedo.png", (200, 50, 50)
                    ),
                    normal=_save_png(
                        textures_dir / "Plastic_normal.png", (128, 128, 255)
                    ),
                    orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
                )
            },
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Surface")
    )

    diffuse_source = output_preview.GetInput("diffuseColor").GetConnectedSource()
    assert diffuse_source is not None
    assert diffuse_source[0].GetPrim().GetName() == "TextureAgentAlbedoTexture"

    roughness_source = output_preview.GetInput("roughness").GetConnectedSource()
    assert roughness_source is not None
    assert roughness_source[0].GetPrim().GetPath() == Sdf.Path(
        "/Root/Looks/Plastic/ExistingRoughnessTexture"
    )
    output_roughness = UsdShade.Shader(roughness_source[0].GetPrim())
    assert (
        output_roughness.GetInput("file").Get().path.endswith("Plastic_roughness.png")
    )
    assert not output_stage.GetPrimAtPath(
        "/Root/Looks/Plastic/TextureAgentRoughnessTexture"
    ).IsValid()


def test_apply_textures_task_enriches_albedo_only_preview_graph_on_rerun(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.2, 0.3))
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()

    first_working_dir = tmp_path / "first"
    first_textures_dir = first_working_dir / "textures"
    first_textures_dir.mkdir(parents=True)
    first_result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(input_path),
            "blended_textures": {
                "Plastic": apply_textures_task.BlendedTextures(
                    albedo=_save_png(
                        first_textures_dir / "Plastic_initial_albedo.png",
                        (200, 50, 50),
                    ),
                    normal="",
                    orm="",
                )
            },
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(first_working_dir),
        }
    )

    first_output_path = Path(first_result["output_usd_paths"][0])
    first_output_stage = Usd.Stage.Open(str(first_output_path))
    first_albedo = UsdShade.Shader(
        first_output_stage.GetPrimAtPath(
            "/Root/Looks/Plastic/TextureAgentAlbedoTexture"
        )
    )
    assert first_albedo.GetPrim().IsValid()
    first_albedo.GetPrim().CreateAttribute(
        "test:preservationMarker", Sdf.ValueTypeNames.String
    ).Set("first-pass")
    first_output_stage.GetRootLayer().Save()

    second_working_dir = tmp_path / "second"
    second_textures_dir = second_working_dir / "textures"
    second_textures_dir.mkdir(parents=True)
    second_result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(first_output_path),
            "blended_textures": {
                "Plastic": apply_textures_task.BlendedTextures(
                    albedo=_save_png(
                        second_textures_dir / "Plastic_enriched_albedo.png",
                        (50, 200, 50),
                    ),
                    normal=_save_png(
                        second_textures_dir / "Plastic_normal.png",
                        (128, 128, 255),
                    ),
                    orm=_save_png(
                        second_textures_dir / "Plastic_orm.png", (255, 64, 32)
                    ),
                )
            },
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(second_working_dir),
        }
    )

    output_stage = Usd.Stage.Open(second_result["output_usd_paths"][0])
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Surface")
    )
    expected_nodes = {
        "diffuseColor": "TextureAgentAlbedoTexture",
        "normal": "TextureAgentNormalTexture",
        "occlusion": "TextureAgentORMTexture",
        "roughness": "TextureAgentRoughnessTexture",
        "metallic": "TextureAgentMetalnessTexture",
    }
    for input_name, expected_node in expected_nodes.items():
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName() == expected_node

    output_albedo = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/TextureAgentAlbedoTexture")
    )
    assert (
        output_albedo.GetPrim().GetAttribute("test:preservationMarker").Get()
        == "first-pass"
    )
    assert (
        output_albedo.GetInput("file")
        .Get()
        .path.endswith("Plastic_enriched_albedo.png")
    )


def test_apply_textures_task_keeps_shared_usd_preview_orm_node(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    orm = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/OrmTexture")
    orm.CreateIdAttr("UsdUVTexture")
    orm.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_orm.png")
    )
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        orm.CreateOutput("g", Sdf.ValueTypeNames.Float)
    )
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
        orm.CreateOutput("b", Sdf.ValueTypeNames.Float)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_orm = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/OrmTexture")
    )

    assert out_orm.GetInput("file").Get().path.endswith("Plastic_orm.png")
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Preview")
    )
    for input_name, expected_node, expected_output in (
        ("roughness", "OrmTexture", "g"),
        ("metallic", "OrmTexture", "b"),
        ("diffuseColor", "TextureAgentAlbedoTexture", "rgb"),
        ("normal", "TextureAgentNormalTexture", "rgb"),
        ("occlusion", "TextureAgentORMTexture", "r"),
    ):
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName() == expected_node
        assert source[1] == expected_output
    for node_name in ("TextureAgentRoughnessTexture", "TextureAgentMetalnessTexture"):
        assert not output_stage.GetPrimAtPath(
            f"/Root/Looks/Plastic/{node_name}"
        ).IsValid()
    stats = result["apply_textures_stats"]
    assert set(stats["preview_texture_inputs_overridden"]) == {
        "/Root/Looks/Plastic/OrmTexture:file",
        "/Root/Looks/Plastic/TextureAgentAlbedoTexture:file",
        "/Root/Looks/Plastic/TextureAgentNormalTexture:file",
        "/Root/Looks/Plastic/TextureAgentORMTexture:file",
    }


def test_apply_textures_task_skips_shared_usd_preview_scalar_node_without_swizzles(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    shared = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/SharedScalar")
    shared.CreateIdAttr("UsdUVTexture")
    shared.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_shared_scalar.png")
    )
    scalar_r = shared.CreateOutput("r", Sdf.ValueTypeNames.Float)
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(scalar_r)
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(scalar_r)
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shared = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/SharedScalar")
    )

    assert out_shared.GetInput("file").Get().path.endswith("original_shared_scalar.png")
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Preview")
    )
    for input_name in ("roughness", "metallic"):
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetPath() == Sdf.Path(
            "/Root/Looks/Plastic/SharedScalar"
        )
        assert source[1] == "r"
    for input_name, expected_node in {
        "diffuseColor": "TextureAgentAlbedoTexture",
        "normal": "TextureAgentNormalTexture",
        "occlusion": "TextureAgentORMTexture",
    }.items():
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName() == expected_node
    assert set(result["apply_textures_stats"]["preview_texture_inputs_overridden"]) == {
        "/Root/Looks/Plastic/TextureAgentAlbedoTexture:file",
        "/Root/Looks/Plastic/TextureAgentNormalTexture:file",
        "/Root/Looks/Plastic/TextureAgentORMTexture:file",
    }


def test_apply_textures_task_skips_shared_usd_preview_mixed_channel_node(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    shared = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/SharedMixed")
    shared.CreateIdAttr("UsdUVTexture")
    shared.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_shared_mixed.png")
    )
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        shared.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        shared.CreateOutput("r", Sdf.ValueTypeNames.Float)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shared = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/SharedMixed")
    )

    assert out_shared.GetInput("file").Get().path.endswith("original_shared_mixed.png")
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Preview")
    )
    for input_name, expected_output in (("diffuseColor", "rgb"), ("roughness", "r")):
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetPath() == Sdf.Path(
            "/Root/Looks/Plastic/SharedMixed"
        )
        assert source[1] == expected_output
    for input_name, expected_node in {
        "normal": "TextureAgentNormalTexture",
        "occlusion": "TextureAgentORMTexture",
        "metallic": "TextureAgentMetalnessTexture",
    }.items():
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName() == expected_node
    assert set(result["apply_textures_stats"]["preview_texture_inputs_overridden"]) == {
        "/Root/Looks/Plastic/TextureAgentNormalTexture:file",
        "/Root/Looks/Plastic/TextureAgentORMTexture:file",
        "/Root/Looks/Plastic/TextureAgentMetalnessTexture:file",
    }


def test_apply_textures_task_skips_external_usd_preview_texture_nodes(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )

    shared = UsdShade.Shader.Define(stage, "/Root/Shared/SharedOrmTexture")
    shared.CreateIdAttr("UsdUVTexture")
    shared.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/original_shared_orm.png")
    )
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
        shared.CreateOutput("g", Sdf.ValueTypeNames.Float)
    )
    preview.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
        shared.CreateOutput("b", Sdf.ValueTypeNames.Float)
    )
    stage.GetRootLayer().Save()

    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(tmp_path),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shared = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Shared/SharedOrmTexture")
    )

    assert out_shared.GetInput("file").Get().path.endswith("original_shared_orm.png")
    output_preview = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Preview")
    )
    for input_name, expected_output in (("roughness", "g"), ("metallic", "b")):
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetPath() == Sdf.Path(
            "/Root/Shared/SharedOrmTexture"
        )
        assert source[1] == expected_output
    for input_name, expected_node in {
        "diffuseColor": "TextureAgentAlbedoTexture",
        "normal": "TextureAgentNormalTexture",
        "occlusion": "TextureAgentORMTexture",
    }.items():
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName() == expected_node
    assert set(result["apply_textures_stats"]["preview_texture_inputs_overridden"]) == {
        "/Root/Looks/Plastic/TextureAgentAlbedoTexture:file",
        "/Root/Looks/Plastic/TextureAgentNormalTexture:file",
        "/Root/Looks/Plastic/TextureAgentORMTexture:file",
    }


def test_apply_textures_task_localizes_local_unmapped_mdl_inputs(
    tmp_path: Path,
) -> None:
    """When an unmapped MDL `*_texture` input points at a *local* path that
    actually exists *inside the USD's upload directory*, the agent must copy
    that asset into the bundle textures dir and rewrite the input to that
    copy. Otherwise the service packager's `../textures/<basename>` rewrite
    would dangle on a file the bundle does not ship.
    """
    from pxr import Sdf, Usd, UsdShade

    # Local opacity map next to the input USD (relative path) and an
    # absolute-path emissive map under a sibling subdir — both within the
    # upload directory so the security gate accepts them.
    local_opacity = tmp_path / "local_opacity.png"
    _save_png(local_opacity, (10, 20, 30))
    abs_dir = tmp_path / "abs_assets"
    abs_dir.mkdir()
    abs_emissive = abs_dir / "local_emissive.png"
    _save_png(abs_emissive, (40, 50, 60))

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # One mapped input (Nucleus, must be overridden) + two unmapped local
    # paths (must be localized into the bundle textures dir).
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Normal.png")
    )
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./local_opacity.png")
    )
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(abs_emissive))
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert (
        out_shader.GetInput("normalmap_texture")
        .Get()
        .path.endswith("Plastic_normal.png")
    )
    # Each preserved local input was copied into work_dir/textures/<safe_name>.
    opacity_ref = out_shader.GetInput("opacity_texture").Get().path
    opacity_out = _resolve_output_ref(output_path, opacity_ref)
    assert opacity_ref == "../textures/Plastic__opacity_texture.png"
    assert opacity_out.parent == textures_dir
    assert opacity_out.exists()
    assert opacity_out.name == "Plastic__opacity_texture.png"
    emissive_ref = out_shader.GetInput("emissive_color_texture").Get().path
    emissive_out = _resolve_output_ref(output_path, emissive_ref)
    assert emissive_ref == "../textures/Plastic__emissive_color_texture.png"
    assert emissive_out.parent == textures_dir
    assert emissive_out.exists()
    assert emissive_out.name == "Plastic__emissive_color_texture.png"
    # And the bytes are preserved (it's a real copy, not a placeholder).
    assert opacity_out.read_bytes() == local_opacity.read_bytes()
    assert emissive_out.read_bytes() == abs_emissive.read_bytes()

    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_cleared"] == []
    assert sorted(stats["mdl_inputs_localized"]) == sorted(
        [
            "/Root/Looks/Plastic:opacity_texture",
            "/Root/Looks/Plastic:emissive_color_texture",
        ]
    )


def test_apply_textures_task_clears_unresolvable_local_mdl_inputs(
    tmp_path: Path,
) -> None:
    """If an unmapped MDL `*_texture` input points at a local path that does
    not exist on disk (asset author's reference is already broken), the agent
    must clear it rather than ship a dangling ref into the bundle.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./does_not_exist.png")
    )
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/nonexistent/abs/path.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert out_shader.GetInput("emissive_color_texture").Get().path == ""
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == []
    assert sorted(stats["mdl_inputs_cleared"]) == sorted(
        [
            "/Root/Looks/Plastic:opacity_texture",
            "/Root/Looks/Plastic:emissive_color_texture",
        ]
    )


@pytest.mark.parametrize(
    "path,unbundleable",
    [
        ("omniverse://nucleus.example/T.png", True),
        ("http://example.com/T.png", True),
        ("https://example.com/T.png", True),
        ("file:///abs/T.png", True),
        ("./local.png", False),
        ("relative/local.png", False),
        ("/abs/local.png", False),
        ("C:/Users/me/T.png", False),  # Windows drive letter is not a URI scheme
        ("", False),
    ],
)
def test_is_unbundleable_asset_path_classification(path: str, unbundleable: bool):
    """Direct unit coverage for the URI-scheme classifier (Claude review nit)."""
    assert apply_textures_task._is_unbundleable_asset_path(path) is unbundleable


def test_apply_textures_task_refuses_localize_outside_usd_directory(
    tmp_path: Path,
) -> None:
    """A malicious USD must not be able to use unmapped MDL `*_texture`
    inputs to copy host files outside the upload directory into the bundle
    textures dir, where they'd be exposed via the
    artifact download endpoint. Codex round-4 caught this CVE-class issue.
    """
    from pxr import Sdf, Usd, UsdShade

    # `outside_dir` is a sibling of the USD's directory, NOT under it. A
    # well-meaning author would never reach out here; an attacker would.
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.png"
    secret.write_bytes(b"secret-bytes")

    # Also a file with no/disallowed extension to verify the suffix gate.
    no_suffix = outside_dir / "passwd"
    no_suffix.write_bytes(b"root:x:0:0:")

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    usd_path = upload_dir / "input.usda"

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # Both inputs target files that exist on disk but live outside the USD's
    # upload directory. The agent must refuse to localize them.
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(secret))
    )
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(no_suffix))
    )
    # Symlink-escape attempt: a relative path that resolves outside the
    # upload root once symlinks are followed. Some Windows environments do not
    # grant symlink privileges, so keep the core outside-root checks active and
    # exercise the symlink branch only when the OS allows creating it.
    escape_link = upload_dir / "escape_link.png"
    has_symlink_escape = False
    try:
        escape_link.symlink_to(secret)
    except OSError:
        pass
    else:
        has_symlink_escape = True
        shader.CreateInput("displacement_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("./escape_link.png")
        )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(usd_path),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    # All malicious inputs were cleared, none localized.
    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert out_shader.GetInput("emissive_color_texture").Get().path == ""
    expected_cleared = [
        "/Root/Looks/Plastic:opacity_texture",
        "/Root/Looks/Plastic:emissive_color_texture",
    ]
    if has_symlink_escape:
        assert out_shader.GetInput("displacement_texture").Get().path == ""
        expected_cleared.append("/Root/Looks/Plastic:displacement_texture")

    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == []
    assert sorted(stats["mdl_inputs_cleared"]) == sorted(expected_cleared)

    # Critical: nothing from outside_dir was copied into the bundle dir.
    bundle_files = sorted(p.name for p in textures_dir.iterdir())
    assert "secret.png" not in bundle_files
    # No copy under the namespaced naming convention either.
    assert not any("opacity_texture" in name for name in bundle_files)
    assert not any("emissive_color_texture" in name for name in bundle_files)
    if has_symlink_escape:
        assert not any("displacement_texture" in name for name in bundle_files)


def test_apply_textures_task_resolves_relative_paths_against_authoring_layer(
    tmp_path: Path,
) -> None:
    """Codex round-5 finding: composed USDs (referenced material libraries)
    author texture paths relative to *their own* layer, not the root. The
    agent must resolve each MDL `*_texture` against the layer that authored
    the value, otherwise legitimate textures from referenced material USDs
    are silently dropped.
    """
    from pxr import Sdf, Usd, UsdShade

    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    materials_dir = upload_dir / "materials"
    materials_dir.mkdir()
    # `opacity.png` lives next to the *referenced* material USD, NOT next to
    # the root entry-point USD.
    materials_opacity = materials_dir / "opacity.png"
    _save_png(materials_opacity, (10, 20, 30))

    # Referenced material library file with the MDL shader and a layer-local
    # asset path.
    materials_usd_path = materials_dir / "library.usda"
    materials_stage = Usd.Stage.CreateNew(str(materials_usd_path))
    UsdShade.Material.Define(materials_stage, "/Materials/Plastic")
    sub_shader = UsdShade.Shader.Define(materials_stage, "/Materials/Plastic/Shader")
    sub_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    sub_shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./opacity.png")
    )
    materials_stage.GetRootLayer().Save()

    # Root USD references the material.
    root_usd_path = upload_dir / "scene.usda"
    root_stage = Usd.Stage.CreateNew(str(root_usd_path))
    plastic_mat = UsdShade.Material.Define(root_stage, "/Root/Looks/Plastic")
    plastic_mat.GetPrim().GetReferences().AddReference(
        "./materials/library.usda", "/Materials/Plastic"
    )
    root_stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (255, 64, 32)),
    )

    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(root_usd_path),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_stage = Usd.Stage.Open(str(output_path))
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    opacity_ref = out_shader.GetInput("opacity_texture").Get().path
    opacity_out = _resolve_output_ref(output_path, opacity_ref)
    assert opacity_ref == "../textures/Plastic__opacity_texture.png"
    # Resolution must have anchored on materials_dir (the referenced layer),
    # not on upload_dir (the root layer). Either way the localized copy lands
    # in work_dir/textures/Plastic__opacity_texture.png with the original
    # bytes — proving the texture was found and copied, not silently lost.
    assert opacity_out.parent == textures_dir
    assert opacity_out.exists()
    assert opacity_out.read_bytes() == materials_opacity.read_bytes()
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == ["/Root/Looks/Plastic:opacity_texture"]
    assert stats["mdl_inputs_cleared"] == []


def test_apply_textures_task_clears_non_png_local_mdl_inputs(tmp_path: Path) -> None:
    """The service packager and textures-artifact ZIP only handle
    case-sensitive ``.png``. Localizing a `.jpg` (or `.tif`/`.exr`/etc.)
    would create an inconsistent bundle — the file lands in cache/textures
    but the packager won't rewrite it and the ZIP glob won't include it.
    Codex round-5 medium finding: drop non-PNG suffixes at the localizer.
    """
    from pxr import Sdf, Usd, UsdShade

    local_jpg = tmp_path / "local_opacity.jpg"
    local_jpg.write_bytes(b"\xff\xd8\xff\xe0...not really jpg but suffix matters")

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./local_opacity.jpg")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert result["apply_textures_stats"]["mdl_inputs_localized"] == []
    assert result["apply_textures_stats"]["mdl_inputs_cleared"] == [
        "/Root/Looks/Plastic:opacity_texture"
    ]
    # Critical: the .jpg is NOT in the bundle textures dir.
    bundle_files = sorted(p.name for p in (work_dir / "textures").iterdir())
    assert not any(name.endswith(".jpg") for name in bundle_files)


def test_apply_textures_task_handles_string_typed_mdl_texture_inputs(
    tmp_path: Path,
) -> None:
    """Codex round-6/7 findings: an MDL shader can legally author
    ``inputs:*_texture`` as `string` or `token` (not `asset`). The previous
    fix skipped non-asset inputs entirely — but that left Nucleus URLs in
    string-typed inputs untouched, which the service packager later rewrites
    into broken `../textures/<basename>` refs (same bug, different surface).
    The agent must process string/token-typed texture inputs the same way as
    asset-typed: override mapped channels, clear unbundleable URI refs.
    Writes must use the authored type so we never crash the pipeline.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # Mapped channel authored as String + holding a Nucleus URL: the pre-fix
    # code skipped this and shipped a broken bundle ref. Must now be
    # overridden with the freshly generated local texture, written back as
    # a String (not silently coerced to Asset).
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/T_Plastic_Albedo.png"
    )
    # Unmapped channel authored as Token + holding a Nucleus URL: must be
    # cleared in its native token type.
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Token).Set(
        "omniverse://nucleus.example/T_Plastic_Opacity.png"
    )
    # Asset-typed mapped channel mixed in: business as usual.
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T_Plastic_Normal.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (200, 50, 50)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (128, 128, 255)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (255, 64, 32)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # String-typed mapped channel: overridden, type preserved, no Nucleus URL.
    diffuse = out_shader.GetInput("diffuse_texture")
    assert diffuse.GetTypeName() == Sdf.ValueTypeNames.String
    diffuse_val = diffuse.Get()
    assert diffuse_val.endswith("Plastic_albedo.png")
    assert "omniverse://" not in diffuse_val
    # Token-typed unmapped channel: cleared, type preserved.
    opacity = out_shader.GetInput("opacity_texture")
    assert opacity.GetTypeName() == Sdf.ValueTypeNames.Token
    assert opacity.Get() == ""
    # Asset-typed mapped channel: business as usual.
    normal = out_shader.GetInput("normalmap_texture")
    assert normal.GetTypeName() == Sdf.ValueTypeNames.Asset
    assert normal.Get().path.endswith("Plastic_normal.png")

    stats = result["apply_textures_stats"]
    # Both diffuse_texture (String) and normalmap_texture (Asset) overridden.
    assert stats["mdl_inputs_overridden"] == 2
    assert stats["mdl_inputs_cleared"] == ["/Root/Looks/Plastic:opacity_texture"]


def test_apply_textures_task_clears_string_typed_orm_texture(tmp_path: Path) -> None:
    """Codex round-9 finding: a string/token-typed mapped MDL input where
    the generated PNG has no parallel Asset-typed dep on the Material
    (today: only the packed ORM channel — `roughness`/`metalness` are
    written separately as OpenPBR Asset attrs, but `orm` itself is not)
    must be cleared, not overridden. Otherwise the service packager
    rewrites the path but USDZ packaging never bundles the file, leaving
    a dangling reference in the downloaded bundle.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # String-typed ORM_texture: must be cleared, not rewritten to the
    # generated packed-ORM path.
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/T_Plastic_ORM.png"
    )
    # String-typed diffuse_texture (channel='albedo' is USDZ-bundled via
    # the OpenPBR-side Asset attr): must still override.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(
        "omniverse://nucleus.example/T_Plastic_Albedo.png"
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # ORM_texture cleared (channel='orm' not in _USDZ_BUNDLED_CHANNELS for
    # string/token).
    orm = out_shader.GetInput("ORM_texture")
    assert orm.GetTypeName() == Sdf.ValueTypeNames.String
    assert orm.Get() == ""
    # diffuse_texture overridden (channel='albedo' is bundled via OpenPBR).
    diffuse = out_shader.GetInput("diffuse_texture")
    assert diffuse.GetTypeName() == Sdf.ValueTypeNames.String
    assert diffuse.Get().endswith("Plastic_albedo.png")
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_overridden"] == 1
    assert stats["mdl_inputs_cleared"] == ["/Root/Looks/Plastic:ORM_texture"]


def test_apply_textures_task_clears_string_typed_local_unmapped_inputs(
    tmp_path: Path,
) -> None:
    """Codex round-8 finding: string/token-typed unmapped MDL `*_texture`
    inputs cannot be safely localized. USDZ packaging only follows
    `Sdf.AssetPath` deps, so a localized PNG referenced only by a string
    input would not be bundled into the downloaded `.usdz`. Clear instead
    of localizing — the MDL falls back to its constant default, which is
    bundle-self-consistent.
    """
    from pxr import Sdf, Usd, UsdShade

    # Real local PNG inside the upload root that *would* pass the
    # security gate in the asset-typed code path.
    local_opacity = tmp_path / "local_opacity.png"
    _save_png(local_opacity, (10, 20, 30))

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    # String-typed unmapped local: must be cleared (not localized).
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.String).Set(
        "./local_opacity.png"
    )

    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    textures_dir = work_dir / "textures"
    textures_dir.mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(textures_dir / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(textures_dir / "Plastic_orm.png", (10, 10, 10)),
    )

    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    opacity = out_shader.GetInput("opacity_texture")
    assert opacity.GetTypeName() == Sdf.ValueTypeNames.String
    assert opacity.Get() == ""
    stats = result["apply_textures_stats"]
    assert stats["mdl_inputs_localized"] == []
    assert stats["mdl_inputs_cleared"] == ["/Root/Looks/Plastic:opacity_texture"]
    # Critical: the local opacity PNG was NOT copied into the bundle dir
    # under any name — string/token-typed inputs do not localize.
    bundle_files = sorted(p.name for p in textures_dir.iterdir())
    assert not any("opacity_texture" in name for name in bundle_files)


def test_apply_textures_task_skips_unsupported_mdl_input_types(tmp_path: Path) -> None:
    """Defense in depth: types outside the supported set
    (``Asset``/``String``/``Token``) — e.g. ``AssetArray``, numeric types
    named ``*_texture`` — must be left untouched and not crash. We'd rather
    skip a rare schema than emit a corrupted value or abort the step.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.AssetArray).Set(
        [Sdf.AssetPath("./pretend.png")]
    )
    # Real asset-typed input alongside, to prove the loop continues.
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("omniverse://nucleus.example/T.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    # Must not raise.
    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    # AssetArray input untouched.
    arr = out_shader.GetInput("opacity_texture").Get()
    assert len(arr) == 1
    assert arr[0].path == "./pretend.png"
    # Real asset-typed input was overridden.
    assert (
        out_shader.GetInput("diffuse_texture").Get().path.endswith("Plastic_albedo.png")
    )
    assert result["apply_textures_stats"]["mdl_inputs_overridden"] == 1


def test_apply_textures_task_handles_nul_byte_in_asset_path(tmp_path: Path) -> None:
    """Claude round-5 nit: ``Path('foo\\x00.png').resolve()`` raises
    ``ValueError``, not ``OSError``. A malicious USD with a NUL byte in an
    MDL `*_texture` asset path must not crash apply_textures — it should
    just clear the input.
    """
    from pxr import Sdf, Usd, UsdShade

    task = apply_textures_task.ApplyTexturesTask()
    stage = Usd.Stage.CreateNew(str(tmp_path / "input.usda"))
    UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Plastic.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./poison\x00.png")
    )
    stage.GetRootLayer().Save()

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "textures").mkdir()
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(work_dir / "textures" / "Plastic_albedo.png", (10, 10, 10)),
        normal=_save_png(work_dir / "textures" / "Plastic_normal.png", (10, 10, 10)),
        orm=_save_png(work_dir / "textures" / "Plastic_orm.png", (10, 10, 10)),
    )

    # Must not raise.
    result = task.run(
        {
            "usd_path": str(tmp_path / "input.usda"),
            "blended_textures": {"Plastic": blended},
            "prim_texture_units": [_unit("Plastic")],
            "working_dir": str(work_dir),
        }
    )

    output_stage = Usd.Stage.Open(result["output_usd_paths"][0])
    out_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Plastic/Shader")
    )
    assert out_shader.GetInput("opacity_texture").Get().path == ""
    assert result["apply_textures_stats"]["mdl_inputs_cleared"] == [
        "/Root/Looks/Plastic:opacity_texture"
    ]


def test_render_material_previews_task_skips_missing_template(tmp_path: Path) -> None:
    task = render_previews_task.RenderMaterialPreviewsTask()

    result = task.run(
        {
            "discovered_materials": [_material("Steel")],
            "usd_path": "/tmp/input.usd",
            "render_preview_config": {"template_scene": str(tmp_path / "missing.usd")},
            "working_dir": str(tmp_path),
        }
    )

    assert result["material_previews"] == {}


def test_render_material_previews_task_saves_preview(
    tmp_path: Path, monkeypatch
) -> None:
    task = render_previews_task.RenderMaterialPreviewsTask()
    template = tmp_path / "template.usd"
    template.write_text("#usda 1.0\n", encoding="utf-8")

    monkeypatch.setattr(
        task, "_compose_preview_stage", lambda *args, **kwargs: object()
    )
    import world_understanding.functions.graphics.render_remote as render_nvcf

    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: [{"images": [Image.new("RGB", (4, 4), (1, 2, 3))]}],
    )

    result = task.run(
        {
            "discovered_materials": [_material("Steel")],
            "usd_path": "/tmp/input.usd",
            "render_preview_config": {"template_scene": str(template)},
            "working_dir": str(tmp_path),
        }
    )

    preview_path = Path(result["material_previews"]["Steel"])
    assert preview_path.exists()


def test_render_output_task_handles_empty_outputs(tmp_path: Path) -> None:
    task = render_task.RenderOutputTask()

    result = task.run({"output_usd_paths": [], "working_dir": str(tmp_path)})

    assert result["rendered_image_paths"] == []
    assert result["render_stats"]["render_available"] is False
    assert result["render_stats"]["production_visual_evidence"] is False


def test_render_output_task_reports_global_slot_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.functions.graphics.render_remote_async as render_async
    import world_understanding.utils.usd.material as usd_material

    captured: dict[str, float | None] = {}

    @contextmanager
    def timeout_slot(*, timeout_seconds: float | None = None):
        captured["timeout_seconds"] = timeout_seconds
        raise render_async.RemoteRenderingSlotTimeoutError("test slot timeout")
        yield 0.0

    def unexpected_render_all_cameras(**kwargs):
        raise AssertionError("render should not start after slot timeout")

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_async, "global_remote_render_slot", timeout_slot)
    monkeypatch.setattr(
        render_nvcf, "render_all_cameras", unexpected_render_all_cameras
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {
                "image_width": 64,
                "render_slot_timeout_sec": 0.25,
            },
            "working_dir": str(tmp_path),
        }
    )

    assert result["rendered_image_paths"] == []
    assert captured["timeout_seconds"] == 0.25
    assert result["render_errors"][0]["code"] == "RENDER_GLOBAL_SLOT_TIMEOUT"


def test_render_output_task_adds_fallback_camera_and_saves_images(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom, UsdLux

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        render_stage = kwargs["stage"]
        captured["has_default_lights"] = any(
            prim.HasAPI(UsdLux.LightAPI) for prim in render_stage.Traverse()
        )
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (4, 5, 6))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64, "timeout_sec": 123},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert Path(result["rendered_image_paths"][0]).exists()
    assert captured["cameras"] == ["/Cameras/TextureAgentFinal"]
    assert captured["timeout"] == 123
    assert captured["has_default_lights"] is True
    assert result["render_stats"]["render_available"] is True
    assert any(
        item["code"] == "RENDER_NO_CAMERA" and item["severity"] == "warning"
        for item in result["render_diagnostics"]
    )


def test_render_output_task_saves_opaque_rgb_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    def fake_render_all_cameras(**kwargs):
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGBA", (4, 4), (20, 40, 60, 32))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    rendered = Image.open(result["rendered_image_paths"][0])
    assert rendered.mode == "RGB"
    assert rendered.getpixel((0, 0)) == (20, 40, 60)


def test_render_output_task_adds_textured_preview_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    (tmp_path / "textures").mkdir()
    texture_path = _save_png(tmp_path / "textures" / "Paint_albedo.png", (240, 190, 40))

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(1, 0, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(0, 1, 0),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "faceVarying",
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0, 0),
                Gf.Vec2f(1, 0),
                Gf.Vec2f(1, 1),
                Gf.Vec2f(0, 1),
            ],
        ),
    )
    material = UsdShade.Material.Define(stage, "/Root/Looks/Paint")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf

    captured = {}

    def fake_render_all_cameras(**kwargs):
        render_stage = kwargs["stage"]
        render_material = UsdShade.Material(
            render_stage.GetPrimAtPath("/Root/Looks/Paint"),
        )
        surface = render_material.GetSurfaceOutput()
        sources, _ = surface.GetConnectedSources()
        preview = UsdShade.Shader(sources[0].source.GetPrim())
        diffuse_sources, _ = preview.GetInput("diffuseColor").GetConnectedSources()
        display_color = UsdGeom.PrimvarsAPI(
            render_stage.GetPrimAtPath("/Root/Mesh"),
        ).GetPrimvar("displayColor")
        albedo = UsdShade.Shader(diffuse_sources[0].source.GetPrim())
        captured["preview_id"] = preview.GetIdAttr().Get()
        captured["diffuse_sources"] = diffuse_sources
        captured["albedo_id"] = albedo.GetIdAttr().Get()
        captured["albedo_file"] = albedo.GetInput("file").Get().path
        captured["has_display_color"] = bool(display_color and display_color.HasValue())
        captured["max_workers"] = kwargs["max_workers"]
        captured["base_dir"] = kwargs["base_dir"]
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (240, 190, 40))],
                }
            ]
        }

    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert captured["preview_id"] == "UsdPreviewSurface"
    assert captured["diffuse_sources"]
    assert captured["albedo_id"] == "UsdUVTexture"
    assert captured["albedo_file"] == str(texture_path)
    assert captured["has_display_color"] is False
    assert captured["max_workers"] == 1
    assert captured["base_dir"] == tmp_path
    assert result["render_stats"]["texture_detail_display_color_bakes"] == 0
    assert result["render_stats"]["texture_detail_uv_texture_fallbacks"] == 1

    original_stage = Usd.Stage.Open(str(usd_path))
    original_display_color = UsdGeom.PrimvarsAPI(
        original_stage.GetPrimAtPath("/Root/Mesh"),
    ).GetPrimvar("displayColor")
    assert not original_display_color.HasValue()


def test_render_output_task_overrides_existing_preview_texture_graph(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    (tmp_path / "textures").mkdir()
    texture_path = _save_png(
        tmp_path / "textures" / "Plastic_albedo.png", (210, 160, 20)
    )
    original_texture_path = _save_png(
        tmp_path / "textures" / "original_albedo.png", (20, 60, 210)
    )

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    cube = UsdGeom.Cube.Define(stage, "/Root/Cube")
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    albedo = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/AlbedoTexture")
    albedo.CreateIdAttr("UsdUVTexture")
    albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(original_texture_path))
    )
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf

    captured = {}

    def fake_render_all_cameras(**kwargs):
        render_stage = kwargs["stage"]
        render_material = UsdShade.Material(
            render_stage.GetPrimAtPath("/Root/Looks/Plastic"),
        )
        surface = render_material.GetSurfaceOutput()
        sources, _ = surface.GetConnectedSources()
        fallback = UsdShade.Shader(sources[0].source.GetPrim())
        diffuse_sources, _ = fallback.GetInput("diffuseColor").GetConnectedSources()
        albedo = UsdShade.Shader(diffuse_sources[0].source.GetPrim())
        captured["surface_source_name"] = sources[0].source.GetPrim().GetName()
        captured["preview_id"] = fallback.GetIdAttr().Get()
        captured["diffuse_sources"] = diffuse_sources
        captured["albedo_source_name"] = diffuse_sources[0].source.GetPrim().GetName()
        captured["albedo_id"] = albedo.GetIdAttr().Get()
        captured["albedo_file"] = albedo.GetInput("file").Get().path
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (210, 160, 20))],
                }
            ]
        }

    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert captured["surface_source_name"] == "OVRTXPreviewSurface"
    assert captured["preview_id"] == "UsdPreviewSurface"
    assert captured["diffuse_sources"]
    assert captured["albedo_source_name"] == "OVRTXPreviewAlbedoTexture"
    assert captured["albedo_id"] == "UsdUVTexture"
    assert captured["albedo_file"] == str(texture_path)
    assert result["render_stats"]["texture_detail_display_color_bakes"] == 0
    assert result["render_stats"]["texture_detail_uv_texture_fallbacks"] == 1


def test_render_output_task_preserves_connected_mdl_surface(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    (tmp_path / "textures").mkdir()
    texture_path = _save_png(tmp_path / "textures" / "Metal_albedo.png", (90, 80, 70))

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    cube = UsdGeom.Cube.Define(stage, "/Root/Cube")
    material = UsdShade.Material.Define(stage, "/Root/Looks/Metal")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    mdl_shader = UsdShade.Shader.Define(stage, "/Root/Looks/Metal/MdlShader")
    mdl_shader.CreateIdAttr("OmniPBR")
    mdl_shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path))
    )
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        render_stage = kwargs["stage"]
        render_material = UsdShade.Material(
            render_stage.GetPrimAtPath("/Root/Looks/Metal"),
        )
        universal_surface = render_material.GetSurfaceOutput()
        mdl_surface = render_material.GetSurfaceOutput("mdl")
        universal_sources, _ = universal_surface.GetConnectedSources()
        mdl_sources, _ = mdl_surface.GetConnectedSources()
        captured["universal_sources"] = universal_sources
        captured["mdl_source_name"] = mdl_sources[0].source.GetPrim().GetName()
        captured["base_dir"] = kwargs["base_dir"]
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (90, 80, 70))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert captured["universal_sources"] == []
    assert captured["mdl_source_name"] == "MdlShader"
    assert captured["base_dir"] == tmp_path
    assert result["render_stats"]["texture_detail_display_color_bakes"] == 0
    assert result["render_stats"]["texture_detail_uv_texture_fallbacks"] == 0


def test_render_output_task_overrides_stale_connected_mdl_texture(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    (tmp_path / "textures").mkdir()
    texture_path = _save_png(
        tmp_path / "textures" / "generated_bucket_albedo.png",
        (180, 90, 40),
    )
    stale_texture_path = _save_png(
        tmp_path / "textures" / "source_bucket_albedo.png",
        (20, 30, 40),
    )

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    cube = UsdGeom.Cube.Define(stage, "/Root/Cube")
    material = UsdShade.Material.Define(stage, "/Root/Looks/Bucket")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    mdl_shader = UsdShade.Shader.Define(stage, "/Root/Looks/Bucket/MdlShader")
    mdl_shader.CreateIdAttr("OmniPBR")
    mdl_shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(stale_texture_path))
    )
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        render_stage = kwargs["stage"]
        render_material = UsdShade.Material(
            render_stage.GetPrimAtPath("/Root/Looks/Bucket"),
        )
        universal_surface = render_material.GetSurfaceOutput()
        mdl_surface = render_material.GetSurfaceOutput("mdl")
        universal_sources, _ = universal_surface.GetConnectedSources()
        mdl_sources, _ = mdl_surface.GetConnectedSources()
        preview = UsdShade.Shader(universal_sources[0].source.GetPrim())
        diffuse_sources, _ = preview.GetInput("diffuseColor").GetConnectedSources()
        albedo = UsdShade.Shader(diffuse_sources[0].source.GetPrim())
        captured["universal_source_name"] = preview.GetPrim().GetName()
        captured["preview_id"] = preview.GetIdAttr().Get()
        captured["albedo_file"] = albedo.GetInput("file").Get().path
        captured["mdl_source_name"] = mdl_sources[0].source.GetPrim().GetName()
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (180, 90, 40))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert captured["universal_source_name"] == "OVRTXPreviewSurface"
    assert captured["preview_id"] == "UsdPreviewSurface"
    assert captured["albedo_file"] == str(texture_path)
    assert captured["mdl_source_name"] == "MdlShader"
    assert result["render_stats"]["texture_detail_display_color_bakes"] == 0
    assert result["render_stats"]["texture_detail_uv_texture_fallbacks"] == 1


def test_render_output_task_adds_preview_for_textureless_custom_mdl(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    (tmp_path / "textures").mkdir()
    texture_path = _save_png(
        tmp_path / "textures" / "Plastic_albedo.png", (20, 80, 210)
    )

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    cube = UsdGeom.Cube.Define(stage, "/Root/Cube")
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath(str(texture_path)))
    mdl_shader = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/MdlShader")
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("./materials/Plastic.mdl"))
    mdl_shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token
    ).Set("Plastic")
    mdl_shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.61, 0.0)
    )
    # An authored but unset texture input is not enough for OVRTX to render
    # generated maps through the custom MDL; the preview fallback is still
    # required.
    mdl_shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        mdl_shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        render_stage = kwargs["stage"]
        render_material = UsdShade.Material(
            render_stage.GetPrimAtPath("/Root/Looks/Plastic"),
        )
        universal_surface = render_material.GetSurfaceOutput()
        universal_sources, _ = universal_surface.GetConnectedSources()
        preview = UsdShade.Shader(universal_sources[0].source.GetPrim())
        diffuse_sources, _ = preview.GetInput("diffuseColor").GetConnectedSources()
        albedo = UsdShade.Shader(diffuse_sources[0].source.GetPrim())
        captured["universal_source_name"] = preview.GetPrim().GetName()
        captured["preview_id"] = preview.GetIdAttr().Get()
        captured["albedo_file"] = albedo.GetInput("file").Get().path
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (20, 80, 210))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert captured["universal_source_name"] == "OVRTXPreviewSurface"
    assert captured["preview_id"] == "UsdPreviewSurface"
    assert captured["albedo_file"] == str(texture_path)
    assert result["render_stats"]["texture_detail_display_color_bakes"] == 0
    assert result["render_stats"]["texture_detail_uv_texture_fallbacks"] == 1


def test_render_output_task_accepts_legacy_list_renderer_shape(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return [{"images": [Image.new("RGB", (4, 4), (4, 5, 6))]}]

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert captured["cameras"] == ["/Camera"]
    assert len(result["rendered_image_paths"]) == 1
    assert Path(result["rendered_image_paths"][0]).exists()
    assert result["render_errors"] == []


def test_render_output_task_uses_distinct_paths_for_multiple_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_paths = []
    for index in range(2):
        usd_path = tmp_path / f"output_{index}.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.Camera.Define(stage, "/Camera")
        stage.GetRootLayer().Save()
        usd_paths.append(str(usd_path))

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: {
            "results": [
                {
                    "camera": "/Camera",
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (4, 5, 6))],
                }
            ]
        },
    )

    result = task.run(
        {
            "output_usd_paths": usd_paths,
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 2
    assert len(set(result["rendered_image_paths"])) == 2
    assert all(Path(path).exists() for path in result["rendered_image_paths"])
    assert result["render_stats"]["camera_paths"] == ["/Camera"]


def test_render_output_task_adds_focus_camera_for_selected_prim(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "camera": camera,
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
                for camera in kwargs["cameras"]
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "prim_texture_units": [SimpleNamespace(prim_path="/Root/Cube")],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert captured["cameras"] == [
        "/Cameras/TextureAgentFinal",
        "/Cameras/TextureAgentFocus_0_0",
    ]
    assert len(result["rendered_image_paths"]) == 2
    assert result["render_stats"]["focus_cameras"] == [
        {
            "prim_path": "/Root/Cube",
            "camera_path": "/Cameras/TextureAgentFocus_0_0",
            "target_frame_coverage_threshold": 0.2,
            "target_frame_coverage_heuristic": pytest.approx(
                0.7561436672967864, rel=1e-3
            ),
            "coverage_metric_source": "focus_camera_bbox_margin_heuristic",
            "coverage_is_estimate": True,
            "meets_target_frame_coverage": True,
        }
    ]


def test_render_output_task_reports_focus_coverage_warning(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: {
            "results": [
                {
                    "camera": camera,
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
                for camera in kwargs["cameras"]
            ]
        },
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "prim_texture_units": [SimpleNamespace(prim_path="/Root/Cube")],
            "render_config": {
                "image_width": 64,
                "target_frame_coverage_threshold": 0.9,
            },
            "working_dir": str(tmp_path),
        }
    )

    assert any(
        item["code"] == "RENDER_FRAME_TOO_WIDE"
        and item["severity"] == "warning"
        and item["details"]["camera_path"] == "/Cameras/TextureAgentFocus_0_0"
        for item in result["render_diagnostics"]
    )


def test_render_output_task_uses_explicit_camera_paths(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "camera": "/ConfiguredCamera",
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {
                "camera_paths": ["/ConfiguredCamera"],
                "focus_cameras": False,
                "image_width": 64,
            },
            "working_dir": str(tmp_path),
        }
    )

    assert captured["cameras"] == ["/ConfiguredCamera"]
    assert result["render_stats"]["camera_paths"] == ["/ConfiguredCamera"]
    assert not any(
        item["code"] == "RENDER_NO_CAMERA" for item in result["render_diagnostics"]
    )
    assert len(result["rendered_image_paths"]) == 1


def test_render_output_task_honors_max_focus_cameras_zero(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "prim_texture_units": [SimpleNamespace(prim_path="/Root/Cube")],
            "render_config": {"image_width": 64, "max_focus_cameras": 0},
            "working_dir": str(tmp_path),
        }
    )

    assert captured["cameras"] == ["/Cameras/TextureAgentFinal"]
    assert result["render_stats"]["focus_cameras"] == []
    assert not any(
        item["code"] == "RENDER_FRAME_TOO_WIDE" for item in result["render_diagnostics"]
    )


def test_render_output_task_accepts_string_false_for_focus_cameras(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "prim_texture_units": [SimpleNamespace(prim_path="/Root/Cube")],
            "render_config": {"image_width": 64, "focus_cameras": "false"},
            "working_dir": str(tmp_path),
        }
    )

    assert captured["cameras"] == ["/Cameras/TextureAgentFinal"]
    assert result["render_stats"]["focus_cameras"] == []


def test_render_output_task_reports_missing_focus_prim(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
            ]
        },
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "prim_texture_units": [SimpleNamespace(prim_path="/Root/Missing")],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert any(
        item["code"] == "RENDER_FOCUS_PRIM_MISSING"
        and item["severity"] == "warning"
        and item["details"]["prim_path"] == "/Root/Missing"
        for item in result["render_diagnostics"]
    )


def test_render_output_task_skips_focus_camera_authoring_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.camera as usd_camera
    import world_understanding.utils.usd.material as usd_material

    captured = {}

    def fail_focus_camera(*args, **kwargs):
        raise RuntimeError("bad bounds")

    def fake_render_all_cameras(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "camera": kwargs["cameras"][0],
                    "status": "success",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
            ]
        }

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(usd_camera, "add_focused_corner_view_camera", fail_focus_camera)
    monkeypatch.setattr(render_nvcf, "render_all_cameras", fake_render_all_cameras)

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "prim_texture_units": [SimpleNamespace(prim_path="/Root/Cube")],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert captured["cameras"] == ["/Cameras/TextureAgentFinal"]
    assert len(result["rendered_image_paths"]) == 1
    assert result["render_stats"]["focus_cameras"] == []
    assert any(
        item["code"] == "RENDER_FOCUS_CAMERA_FAILED"
        and item["severity"] == "warning"
        and item["details"]["prim_path"] == "/Root/Cube"
        and item["details"]["exception_type"] == "RuntimeError"
        for item in result["render_diagnostics"]
    )


def test_render_output_task_reports_bad_renderer_result_shape(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: {"results": "not-a-list"},
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert result["rendered_image_paths"] == []
    assert result["render_stats"]["render_available"] is False
    assert result["render_errors"][0]["code"] == "RENDER_RESULT_PARSE_ERROR"
    assert result["render_errors"][0]["message"] == (
        "Renderer returned an unsupported result shape: "
        "render_all_cameras returned a dict without a list-valued 'results' key"
    )
    assert result["render_errors"][0]["details"]["exception_type"] == "ValueError"


def test_render_output_task_reports_empty_success_renderer_result(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: {
            "results": [
                {
                    "camera": "/Camera",
                    "status": "success",
                    "images": [],
                }
            ]
        },
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert result["rendered_image_paths"] == []
    assert result["render_stats"]["render_available"] is False
    assert result["render_errors"][0]["code"] == "RENDER_EMPTY_RESULT"
    assert result["render_errors"][0]["camera_path"] == "/Camera"
    assert "Renderer returned no images" in result["render_errors"][0]["message"]


def test_render_output_task_reports_per_camera_renderer_failure(
    tmp_path: Path, monkeypatch
) -> None:
    from pxr import Usd, UsdGeom

    task = render_task.RenderOutputTask()
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_nvcf
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        render_nvcf,
        "render_all_cameras",
        lambda **kwargs: {
            "results": [
                {
                    "status": "exception",
                    "error": "boom",
                    "images": [Image.new("RGB", (4, 4), (7, 8, 9))],
                }
            ]
        },
    )

    result = task.run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert result["rendered_image_paths"] == []
    assert result["render_stats"]["render_available"] is False
    assert result["render_errors"][0]["code"] == "RENDER_PER_CAMERA_FAILURE"
    assert result["render_errors"][0]["camera_path"] == "/Camera"
    assert result["render_errors"][0]["details"]["status"] == "exception"
    assert "boom" in result["render_errors"][0]["message"]


def test_render_output_task_reports_unopenable_output_usd(tmp_path: Path) -> None:
    task = render_task.RenderOutputTask()

    result = task.run(
        {
            "output_usd_paths": [str(tmp_path / "missing.usd")],
            "render_config": {"image_width": 64},
            "working_dir": str(tmp_path),
        }
    )

    assert result["rendered_image_paths"] == []
    assert result["render_stats"]["render_available"] is False
    assert result["render_errors"][0]["code"] == "RENDER_OUTPUT_USD_OPEN_FAILED"
    assert "Failed to open output USD" in result["render_errors"][0]["message"]
