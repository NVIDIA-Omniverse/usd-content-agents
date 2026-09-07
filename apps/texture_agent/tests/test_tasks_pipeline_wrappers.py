# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import shutil
import stat
import zipfile
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
from texture_agent.planning import (
    TexturePlan,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanExecution,
    TexturePlanLimits,
    TexturePlanRequest,
    TexturePlanSource,
    TexturePlanUnit,
)

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


def _write_layered_instance_material_stage(source_dir: Path) -> Path:
    """Create a real sublayer-backed, internally-instanced material fixture."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    layers_dir = source_dir / "layers"
    source_textures = layers_dir / "textures"
    source_textures.mkdir(parents=True)
    legacy_texture = source_textures / "legacy.png"
    _save_png(legacy_texture, (18, 36, 72))

    content_path = layers_dir / "content.usda"
    content = Usd.Stage.CreateNew(str(content_path))
    root = UsdGeom.Xform.Define(content, "/Root")
    prototype = UsdGeom.Xform.Define(content, "/Root/Prototype")
    UsdGeom.Mesh.Define(content, "/Root/Prototype/Body")
    material = UsdShade.Material.Define(content, "/Root/Prototype/Looks/Steel")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    )
    legacy = UsdShade.Shader.Define(
        content, "/Root/Prototype/Looks/Steel/LegacyTexture"
    )
    legacy.CreateIdAttr("UsdUVTexture")
    legacy.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/legacy.png")
    )
    instance = UsdGeom.Xform.Define(content, "/Root/Instance").GetPrim()
    instance.GetReferences().AddInternalReference(prototype.GetPath())
    instance.SetInstanceable(True)
    content.SetDefaultPrim(root.GetPrim())
    content.GetRootLayer().Save()

    source_path = source_dir / "scene.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_path))
    source_layer.subLayerPaths = ["layers/content.usda"]
    source_layer.defaultPrim = "Root"
    source_layer.Save()
    return source_path


def _write_layered_instance_material_usdz(source_dir: Path) -> Path:
    """Package the layered fixture with a nested root as a real USDZ."""
    package_source = source_dir / "package-source"
    root_path = _write_layered_instance_material_stage(package_source)
    package_path = source_dir / "scene.usdz"
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_STORED) as package:
        package.write(root_path, "Scene/scene.usda")
        for dependency in sorted((package_source / "layers").rglob("*")):
            if dependency.is_file():
                relative = dependency.relative_to(package_source).as_posix()
                package.write(dependency, f"Scene/{relative}")
    return package_path


def _write_sibling_layered_instance_material_stage(
    source_dir: Path,
) -> tuple[Path, Path]:
    """Create a root whose dependencies live in a sibling bundle directory."""
    from pxr import Sdf

    bundle_root = source_dir / "bundle"
    original_root = _write_layered_instance_material_stage(bundle_root)
    original_root.unlink()
    scene_dir = bundle_root / "Scene"
    scene_dir.mkdir()
    root_path = scene_dir / "scene.usda"
    root_layer = Sdf.Layer.CreateNew(str(root_path))
    root_layer.subLayerPaths = ["../layers/content.usda"]
    root_layer.defaultPrim = "Root"
    root_layer.Save()
    return root_path, bundle_root


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


def _write_three_repairable_quad_usd(path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    for prim_path in (
        "/World/Mesh_0",
        "/World/Mesh_1",
        "/World/Outside",
    ):
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
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
        )
        st.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0, 0),
                    Gf.Vec2f(1, 0),
                    Gf.Vec2f(1, 1),
                    Gf.Vec2f(0, 1),
                ]
            )
        )
    stage.GetRootLayer().Save()
    return path


def _ready_texture_plan(
    source_asset: Path,
    units: tuple[TexturePlanUnit, ...],
) -> TexturePlan:
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset=str(source_asset)),
        backend="simple_image_gen",
        max_concurrency=1,
        unit_timeout_seconds=30,
    )
    return TexturePlan(
        request=request,
        limits=TexturePlanLimits.from_request(request),
        execution=TexturePlanExecution.from_request(request),
        counts=TexturePlanCounts(
            authored_material_count=len(units),
            renderable_prim_count=len(units),
            renderable_subset_count=sum(
                len(unit.member_subset_paths) for unit in units
            ),
            effective_bound_material_count=len(units),
            selected_material_count=len(units),
            selected_unit_count=len(units),
            skipped_item_count=0,
            planned_generation_job_count=len(units),
        ),
        selected_units=units,
        decision=TexturePlanDecision(state="ready", execution_allowed=True),
    )


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


def test_deinstance_prim_skips_read_only_instance_proxy() -> None:
    class _InstanceProxy:
        def IsInstanceProxy(self) -> bool:
            return True

        def SetInstanceable(self, _value: bool) -> None:
            pytest.fail("SetInstanceable must not be called on an instance proxy")

    stage: Any = object()
    prim: Any = _InstanceProxy()
    deinstanced_paths = {"/AlreadyDeinstanced"}

    apply_textures_task._deinstance_prim(stage, prim, deinstanced_paths)

    assert deinstanced_paths == {"/AlreadyDeinstanced"}


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
    assert result["auto_prompt_source"] == "auto_prompt_fallback"
    assert result["auto_prompt_fallback_materials"] == ["Steel"]
    from texture_agent.functions.artifact_manifest import build_artifacts_manifest

    manifest = build_artifacts_manifest(result, status="completed")
    assert manifest["prompts"]["prompt_source"] == "auto_prompt_fallback"
    assert manifest["prompts"]["fallback_materials"] == ["Steel"]
    assert manifest["prompts"]["fallback_count"] == 1
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
    (prompts_dir / "prompt_provenance.json").write_text(
        json.dumps(
            {
                "prompt_source": "auto_prompt_fallback",
                "fallback_materials": ["Copper"],
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
    assert result["auto_prompt_source"] == "auto_prompt_fallback"
    assert result["auto_prompt_fallback_materials"] == ["Copper"]


def test_generate_prompts_resume_merges_cached_and_new_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "material_prompts.json").write_text(
        json.dumps({"Copper": {"prompt": "cached copper", "opacity": 0.6}}),
        encoding="utf-8",
    )
    (prompts_dir / "prompt_provenance.json").write_text(
        json.dumps(
            {
                "prompt_source": "auto_prompt_fallback",
                "auto_prompt_materials": ["Copper"],
                "fallback_materials": ["Copper"],
            }
        ),
        encoding="utf-8",
    )

    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models, "create_chat_model_from_config", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        generate_prompts_task,
        "generate_texture_prompts",
        lambda **_kwargs: {"Steel": {"prompt": "llm steel", "opacity": 0.8}},
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Copper"), _material("Steel")],
            "material_textures": {},
            "auto_prompt_config": {"enabled": True},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
            "resume": True,
        }
    )

    assert result["auto_prompt_source"] == "auto_prompt_llm_with_fallback"
    assert result["auto_prompt_fallback_materials"] == ["Copper"]


def test_generate_prompts_resume_drops_overridden_fallback_provenance(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "material_prompts.json").write_text(
        json.dumps({"Steel": {"prompt": "cached steel", "opacity": 0.6}}),
        encoding="utf-8",
    )
    (prompts_dir / "prompt_provenance.json").write_text(
        json.dumps(
            {
                "prompt_source": "auto_prompt_fallback",
                "auto_prompt_materials": ["Steel"],
                "fallback_materials": ["Steel"],
            }
        ),
        encoding="utf-8",
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Steel")],
            "material_textures": {
                "Steel": {"prompt": "explicit steel", "opacity": 0.9}
            },
            "auto_prompt_config": {"enabled": True},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
            "resume": True,
        }
    )

    assert result["auto_prompt_source"] == "material_textures"
    assert result["auto_prompt_fallback_materials"] == []


def test_generate_prompts_disabled_auto_prompt_preserves_cached_provenance(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    cached_copper = {"prompt": "cached copper", "opacity": 0.6}
    (prompts_dir / "material_prompts.json").write_text(
        json.dumps({"Copper": cached_copper}), encoding="utf-8"
    )
    (prompts_dir / "prompt_provenance.json").write_text(
        json.dumps(
            {
                "prompt_source": "auto_prompt_fallback",
                "auto_prompt_materials": ["Copper"],
                "fallback_materials": ["Copper"],
            }
        ),
        encoding="utf-8",
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Copper"), _material("Steel")],
            "material_textures": {"Copper": cached_copper},
            "cached_material_textures": {"Copper": cached_copper},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
            "resume": True,
        }
    )

    assert result["auto_prompt_source"] == "auto_prompt_fallback"
    assert result["auto_prompt_fallback_materials"] == ["Copper"]


def test_generate_prompts_resume_prunes_prompts_outside_active_scope(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "material_prompts.json").write_text(
        json.dumps(
            {
                "Steel": {"prompt": "cached steel", "opacity": 0.8},
                "Stale": {"prompt": "stale prompt", "opacity": 0.8},
            }
        ),
        encoding="utf-8",
    )
    (prompts_dir / "prompt_provenance.json").write_text(
        json.dumps(
            {
                "prompt_source": "auto_prompt_fallback",
                "auto_prompt_materials": ["Steel", "Stale"],
                "fallback_materials": ["Steel", "Stale"],
            }
        ),
        encoding="utf-8",
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Steel")],
            "material_textures": {},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
            "resume": True,
        }
    )

    assert result["material_textures"] == {
        "Steel": {"prompt": "cached steel", "opacity": 0.8}
    }
    assert json.loads((prompts_dir / "material_prompts.json").read_text()) == {
        "Steel": {"prompt": "cached steel", "opacity": 0.8}
    }
    assert json.loads((prompts_dir / "prompt_provenance.json").read_text()) == {
        "prompt_source": "auto_prompt_fallback",
        "auto_prompt_materials": ["Steel"],
        "fallback_materials": ["Steel"],
    }


def test_generate_prompts_restores_provenance_for_resume_execution(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    cached_copper = {"prompt": "cached copper", "opacity": 0.6}
    (prompts_dir / "material_prompts.json").write_text(
        json.dumps({"Copper": cached_copper}), encoding="utf-8"
    )
    (prompts_dir / "prompt_provenance.json").write_text(
        json.dumps(
            {
                "prompt_source": "auto_prompt_fallback",
                "auto_prompt_materials": ["Copper"],
                "fallback_materials": ["Copper"],
            }
        ),
        encoding="utf-8",
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Copper")],
            "material_textures": {"Copper": cached_copper},
            "cached_material_textures": {"Copper": cached_copper},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_material"},
            "planning_config": {"resume_execution": True},
            "working_dir": str(tmp_path),
        }
    )

    assert result["auto_prompt_source"] == "auto_prompt_fallback"
    assert result["auto_prompt_fallback_materials"] == ["Copper"]


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
    assert result["auto_prompt_source"] == "auto_prompt_llm"
    assert result["auto_prompt_fallback_materials"] == []


def test_generate_prompts_task_marks_all_llm_omissions_as_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models, "create_chat_model_from_config", lambda *args, **kwargs: object()
    )

    def all_fallbacks(*, materials, fallback_material_names, **_kwargs):
        fallback_material_names.update(material.name for material in materials)
        return {
            material.name: {"prompt": f"fallback {material.name}", "opacity": 0.8}
            for material in materials
        }

    monkeypatch.setattr(
        generate_prompts_task, "generate_texture_prompts", all_fallbacks
    )

    result = generate_prompts_task.GeneratePromptsTask().run(
        {
            "discovered_materials": [_material("Steel"), _material("Copper")],
            "material_textures": {},
            "auto_prompt_config": {"enabled": True},
            "texture_config": {"mode": "per_material"},
            "working_dir": str(tmp_path),
        }
    )

    assert result["auto_prompt_source"] == "auto_prompt_fallback"
    assert result["auto_prompt_fallback_materials"] == ["Copper", "Steel"]


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
        def Flatten(self, *, addSourceFileComment=True):
            assert addSourceFileComment is False
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
        prepare_uvs_task, "repair_degenerate_uvs", lambda stage, **_kwargs: 0
    )
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

        def Flatten(self, *, addSourceFileComment=True):
            assert addSourceFileComment is False
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
        prepare_uvs_task, "repair_degenerate_uvs", lambda stage, **_kwargs: 0
    )
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
        def Flatten(self, *, addSourceFileComment=True):
            assert addSourceFileComment is False
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
        prepare_uvs_task, "repair_degenerate_uvs", lambda stage, **_kwargs: 0
    )
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
        def Flatten(self, *, addSourceFileComment=True):
            assert addSourceFileComment is False
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
        prepare_uvs_task, "repair_degenerate_uvs", lambda stage, **_kwargs: 0
    )
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
        def Flatten(self, *, addSourceFileComment=True):
            assert addSourceFileComment is False
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
        prepare_uvs_task, "repair_degenerate_uvs", lambda stage, **_kwargs: 0
    )
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
        "source_usd_path": "/immutable/upload.usdz",
        "working_dir": str(tmp_path / "work"),
        "texture_config": {"uv_policy": "preserve_or_fix"},
    }

    result = task.run(context)

    assert result["usd_path"] == str(usd_path)
    assert result["source_usd_path"] == "/immutable/upload.usdz"
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


def test_prepare_uvs_plan_scope_repairs_all_selected_units_during_targeted_retry(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = _write_three_repairable_quad_usd(tmp_path / "scene.usda")
    units = tuple(
        TexturePlanUnit.build(
            unit_mode="per_material",
            material_prim_paths=(f"/World/Looks/Material_{index}",),
            member_prim_paths=(f"/World/Mesh_{index}",),
            display_name=f"Material {index}",
            selection_reason_code="explicit_material",
            selection_reason="Selected by the explicit material scope.",
            detail_policy="surface_only",
        )
        for index in range(2)
    )
    plan = _ready_texture_plan(source, units)
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    plan_path = working_dir / "texture_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    result = prepare_uvs_task.PrepareUVsTask().run(
        {
            "usd_path": str(source),
            "working_dir": str(working_dir),
            "texture_plan_path": str(plan_path),
            "texture_config": {
                "uv_policy": "generate_missing",
                "uv_scope": "target_prims",
                # Once an executable plan exists, it is the sole scope
                # authority; stale legacy targets must not widen the run.
                "uv_target_prim_paths": ["/World/Outside"],
            },
            "material_textures": {"StaleOutside": {"prim_paths": ["/World/Outside"]}},
            # Texture generation retries only this failed unit. UV preparation
            # must retain deterministic repairs for both approved units.
            "planning_config": {
                "regenerate_unit_ids": [units[0].unit_id],
            },
        }
    )

    assert result["uv_preparation"]["fixed_interpolation"] == 2
    assert result["uv_preparation"]["target_prim_paths"] == [
        "/World/Mesh_0",
        "/World/Mesh_1",
    ]
    stage = Usd.Stage.Open(result["usd_path"])
    for path in ("/World/Mesh_0", "/World/Mesh_1"):
        st = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).GetPrimvar("st")
        assert st.GetInterpolation() == "faceVarying"
        assert st.GetAttr().HasAuthoredMetadata("interpolation")
    outside_st = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Outside")).GetPrimvar(
        "st"
    )
    assert outside_st.GetInterpolation() == "constant"
    assert not outside_st.GetAttr().HasAuthoredMetadata("interpolation")


def test_prepare_uvs_collect_plan_targets_promotes_subset_to_deduplicated_parent(
    tmp_path: Path,
) -> None:
    unit = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=("/World/Mesh",),
        member_subset_paths=("/World/Mesh/PaintedFaces",),
        display_name="Paint",
        selection_reason_code="explicit_material",
        selection_reason="Selected by the explicit material scope.",
        detail_policy="surface_only",
    )
    plan = _ready_texture_plan(tmp_path / "scene.usda", (unit,))

    assert prepare_uvs_task._collect_uv_target_prim_paths(
        {
            "working_dir": str(tmp_path),
            "texture_plan": plan,
        }
    ) == ("/World/Mesh",)


def test_prepare_uvs_collect_targets_ignores_implicit_stale_workdir_plan(
    tmp_path: Path,
) -> None:
    stale_unit = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Stale",),
        member_prim_paths=("/World/StaleMesh",),
        display_name="Stale",
        selection_reason_code="explicit_material",
        selection_reason="Selected by a prior run.",
        detail_policy="surface_only",
    )
    stale_plan = _ready_texture_plan(tmp_path / "old-scene.usda", (stale_unit,))
    stale_plan = stale_plan.model_copy(
        update={
            "decision": TexturePlanDecision(
                state="unsupported",
                execution_allowed=False,
                reasons=("Prior run was rejected.",),
                recommended_actions=("Start a new run.",),
            )
        }
    )
    (tmp_path / "texture_plan.json").write_text(
        stale_plan.model_dump_json(indent=2),
        encoding="utf-8",
    )

    assert prepare_uvs_task._collect_uv_target_prim_paths(
        {
            "working_dir": str(tmp_path),
            "texture_config": {
                "uv_target_prim_paths": ["/World/CurrentMesh"],
            },
        }
    ) == ("/World/CurrentMesh",)


def test_prepare_uvs_collect_plan_targets_rejects_unbound_material(
    tmp_path: Path,
) -> None:
    unit = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Unbound",),
        display_name="Unbound",
        selection_reason_code="explicit_material",
        selection_reason="Selected by the explicit material scope.",
        detail_policy="surface_only",
    )
    plan = _ready_texture_plan(tmp_path / "scene.usda", (unit,))

    with pytest.raises(
        prepare_uvs_task.UVPreparationError,
        match="Selected material has no renderable bound geometry",
    ):
        prepare_uvs_task._collect_uv_target_prim_paths(
            {
                "working_dir": str(tmp_path),
                "texture_plan": plan,
            }
        )


@pytest.mark.parametrize(
    ("member_prim_paths", "member_subset_paths"),
    [
        (("/",), ()),
        ((), ("/Subset",)),
    ],
)
def test_prepare_uvs_collect_plan_targets_rejects_stage_root(
    tmp_path: Path,
    member_prim_paths: tuple[str, ...],
    member_subset_paths: tuple[str, ...],
) -> None:
    unit = TexturePlanUnit.build(
        unit_mode="per_material",
        material_prim_paths=("/World/Looks/Paint",),
        member_prim_paths=member_prim_paths,
        member_subset_paths=member_subset_paths,
        display_name="Paint",
        selection_reason_code="explicit_material",
        selection_reason="Selected by the explicit material scope.",
        detail_policy="surface_only",
    )
    plan = _ready_texture_plan(tmp_path / "scene.usda", (unit,))

    with pytest.raises(ValueError, match="absolute, non-root USD prim paths"):
        prepare_uvs_task._collect_uv_target_prim_paths(
            {
                "working_dir": str(tmp_path),
                "texture_plan": plan,
            }
        )


@pytest.mark.parametrize("target_path", ["/", "relative/path"])
def test_prepare_uvs_resolve_target_scope_rejects_invalid_direct_target(
    target_path: str,
) -> None:
    with pytest.raises(ValueError, match="absolute, non-root USD prim paths"):
        prepare_uvs_task._resolve_uv_target_scope(
            {
                "texture_config": {
                    "uv_scope": "target_prims",
                    "uv_target_prim_paths": [target_path],
                }
            }
        )


@pytest.mark.parametrize("documentation", [None, "original asset note\n"])
def test_prepare_uvs_flatten_preserves_authored_root_metadata(
    documentation: str | None,
) -> None:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    if documentation is not None:
        stage.GetPseudoRoot().SetMetadata("documentation", documentation)
    expected = stage.GetPseudoRoot().GetAllAuthoredMetadata()

    flattened = prepare_uvs_task._flatten_for_uv_preparation(
        stage,
        "/private/service/session/input/scene.usd",
    )

    assert flattened.GetPseudoRoot().GetAllAuthoredMetadata() == expected
    if documentation is None:
        assert not flattened.GetPseudoRoot().HasAuthoredMetadata("documentation")
    else:
        assert flattened.GetPseudoRoot().GetMetadata("documentation") == documentation


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


def test_prepare_uvs_scene_optimizer_publishes_only_uv_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_texture = source_dir / "olive.png"
    source_texture.write_bytes(b"original-olive-texture")
    replacement_texture = source_dir / "rewritten.png"
    replacement_texture.write_bytes(b"unexpected-so-texture")
    usd_path = _write_quad_usd(
        source_dir / "input.usda",
        uvs=[(0.2, 0.2)] * 4,
    )
    source_stage = Usd.Stage.Open(str(usd_path))
    shader = UsdShade.Shader.Define(source_stage, "/World/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("olive.png"))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    source_stage.GetRootLayer().Save()

    replacement_uvs = Vt.Vec2fArray(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(1.0, 1.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        optimized = Usd.Stage.Open(str(input_path))
        mesh = UsdGeom.Mesh(optimized.GetPrimAtPath("/World/Mesh"))
        UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Set(replacement_uvs)
        mesh.CreateDoubleSidedAttr(True)
        optimized_shader = UsdShade.Shader(optimized.GetPrimAtPath("/World/Shader"))
        optimized_shader.GetInput("file").Set(Sdf.AssetPath(str(replacement_texture)))
        optimized_shader.GetInput("roughness").Set(0.95)
        assert export_stage_portably(
            optimized,
            output_path,
            approved_dependency_roots=kwargs["approved_dependency_roots"],
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_task,
        "generate_projection_uvs",
        fake_generate_projection_uvs,
    )

    result = prepare_uvs_task.PrepareUVsTask().run(
        {
            "usd_path": str(usd_path),
            "working_dir": str(tmp_path / "work"),
            "texture_config": {
                "uv_backend": "scene_optimizer",
                "uv_policy": "force_projection",
            },
        }
    )

    prepared = Usd.Stage.Open(result["usd_path"])
    prepared_mesh = UsdGeom.Mesh(prepared.GetPrimAtPath("/World/Mesh"))
    assert list(UsdGeom.PrimvarsAPI(prepared_mesh).GetPrimvar("st").Get()) == list(
        replacement_uvs
    )
    assert prepared_mesh.GetDoubleSidedAttr().Get() is False
    prepared_shader = UsdShade.Shader(prepared.GetPrimAtPath("/World/Shader"))
    assert prepared_shader.GetInput("roughness").Get() == pytest.approx(0.25)
    prepared_texture = prepared_shader.GetInput("file").Get()
    assert prepared_texture.resolvedPath
    assert Path(prepared_texture.resolvedPath).read_bytes() == b"original-olive-texture"
    assert result["uv_preparation"]["uv_writeback_meshes"] == 1


def test_prepare_uvs_scene_optimizer_rejects_post_generation_output_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = _write_quad_usd(tmp_path / "input.usda")
    working_dir = tmp_path / "work"
    prepared_path = working_dir / "prepared" / "prepared_input.usd"
    displaced = working_dir / "prepared" / "displaced.usd"
    outside_output = tmp_path / "outside.usd"
    outside_output.write_bytes(b"outside-must-survive")

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        _write_quad_usd(
            Path(output_path),
            uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    def swap_output_after_stage_open(stage, **kwargs):
        prepared_path.rename(displaced)
        prepared_path.symlink_to(outside_output)
        return 0

    monkeypatch.setattr(
        prepare_uvs_task,
        "generate_projection_uvs",
        fake_generate_projection_uvs,
    )
    monkeypatch.setattr(
        prepare_uvs_task,
        "fix_uv_interpolation",
        swap_output_after_stage_open,
    )

    with pytest.raises(RuntimeError, match="symlink USD output"):
        prepare_uvs_task.PrepareUVsTask().run(
            {
                "usd_path": str(usd_path),
                "working_dir": str(working_dir),
                "texture_config": {
                    "uv_backend": "scene_optimizer",
                    "uv_policy": "generate_missing",
                },
            }
        )

    assert displaced.is_file()
    assert outside_output.read_bytes() == b"outside-must-survive"


def test_prepare_uvs_scene_optimizer_preserves_sibling_source_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Ar, Sdf, Usd, UsdShade
    from world_understanding.functions.graphics.so_export import (
        export_stage_portably,
    )

    bundle_root = tmp_path / "source-bundle"
    input_dir = bundle_root / "input"
    shared_dir = bundle_root / "shared"
    upload_dir = bundle_root / "upload"
    input_dir.mkdir(parents=True)
    shared_dir.mkdir()
    upload_dir.mkdir()
    texture_path = shared_dir / "albedo.png"
    texture_path.write_bytes(b"sibling-source-texture")
    source_usdz = upload_dir / "source.usdz"
    source_usdz.touch()

    usd_path = _write_quad_usd(
        input_dir / "input.usda",
        uvs=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    source_stage = Usd.Stage.Open(str(usd_path))
    shader = UsdShade.Shader.Define(source_stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../shared/albedo.png")
    )
    source_stage.GetRootLayer().Save()

    captured: dict[str, Any] = {}

    def fake_generate_projection_uvs(input_path, output_path, **kwargs):
        captured.update(kwargs)
        flat_stage = Usd.Stage.Open(str(input_path))
        assert flat_stage is not None
        assert export_stage_portably(
            flat_stage,
            output_path,
            approved_dependency_roots=kwargs["approved_dependency_roots"],
        )
        return {"meshes_with_uvs": 1, "status": "completed"}

    monkeypatch.setattr(
        prepare_uvs_task, "generate_projection_uvs", fake_generate_projection_uvs
    )
    working_dir = tmp_path / "run"
    result = prepare_uvs_task.PrepareUVsTask().run(
        {
            "usd_path": str(usd_path),
            "source_usd_path": str(source_usdz),
            "usd_dependency_root": str(bundle_root),
            "working_dir": str(working_dir),
            "texture_config": {
                "uv_backend": "scene_optimizer",
                "uv_policy": "force_projection",
            },
        }
    )

    assert captured["approved_dependency_roots"] == (
        working_dir.resolve(),
        input_dir.resolve(),
        upload_dir.resolve(),
        bundle_root.resolve(),
    )
    shutil.rmtree(bundle_root)

    prepared_stage = Usd.Stage.Open(result["usd_path"])
    assert prepared_stage is not None
    asset = (
        prepared_stage.GetPrimAtPath("/World/Shader").GetAttribute("inputs:file").Get()
    )
    assert asset.resolvedPath
    resolver_asset = Ar.GetResolver().OpenAsset(
        Ar.GetResolver().Resolve(asset.resolvedPath)
    )
    assert resolver_asset is not None
    assert bytes(resolver_asset.GetBuffer()) == b"sibling-source-texture"


def test_prepare_uvs_rejects_invalid_dependency_root_before_side_effects(
    tmp_path: Path,
) -> None:
    usd_path = _write_quad_usd(tmp_path / "input.usda")
    working_dir = tmp_path / "new-run"

    with pytest.raises(ValueError, match="must not contain filesystem roots"):
        prepare_uvs_task.PrepareUVsTask().run(
            {
                "usd_path": str(usd_path),
                "usd_dependency_root": str(Path(tmp_path.anchor)),
                "working_dir": str(working_dir),
                "texture_config": {
                    "uv_backend": "scene_optimizer",
                    "uv_policy": "generate_missing",
                },
            }
        )

    assert not working_dir.exists()


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
    assert so_call["approved_dependency_roots"] == (
        (tmp_path / "work").resolve(),
        tmp_path.resolve(),
    )
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
        def Flatten(self, *, addSourceFileComment=True):
            assert addSourceFileComment is False
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
        prepare_uvs_task, "repair_degenerate_uvs", lambda stage, **_kwargs: 0
    )
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


@pytest.mark.parametrize("packaged", [False, True], ids=["usd", "usdz"])
def test_apply_textures_preserves_relocated_layered_composition(
    tmp_path: Path,
    packaged: bool,
) -> None:
    """Direct CLI layered USD and USDZ outputs survive source removal."""
    from pxr import Usd

    source_dir = tmp_path / "source"
    source_path = (
        _write_layered_instance_material_usdz(source_dir)
        if packaged
        else _write_layered_instance_material_stage(source_dir)
    )
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    previous_output = working_dir / "output"
    previous_output.mkdir()
    (previous_output / "stale.txt").write_text("old", encoding="utf-8")
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal=_save_png(textures_dir / "steel_normal.png", (128, 128, 255)),
        orm=_save_png(textures_dir / "steel_orm.png", (255, 64, 32)),
    )

    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(source_path),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [
                _unit(material_prim_path="/Root/Instance/Looks/Steel")
            ],
            "working_dir": str(working_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    assert output_path == working_dir / "output" / "textured_output.usd"
    assert not (output_path.parent / "stale.txt").exists()
    assert not list(working_dir.glob(".output.backup-*"))
    assert result["output_portability"]["portable"] is True

    output_stage = Usd.Stage.Open(str(output_path))
    assert output_stage.GetDefaultPrim().GetPath() == Sdf.Path("/Root")
    assert output_stage.GetPrimAtPath("/Root/Instance/Body").IsValid()
    assert output_stage.GetPrimAtPath("/Root/Instance").IsInstanceable()

    applied = output_stage.GetAttributeAtPath(
        "/Root/Instance/Looks/Steel.inputs:base_color_texture_file"
    ).Get()
    assert applied.path and not Path(applied.path).is_absolute()
    assert Path(applied.resolvedPath).is_file()
    legacy = output_stage.GetAttributeAtPath(
        "/Root/Instance/Looks/Steel/LegacyTexture.inputs:file"
    ).Get()
    assert legacy.path and not Path(legacy.path).is_absolute()
    assert Path(legacy.resolvedPath).is_file()

    # Move only the produced run and delete the original source tree. Every
    # composition and texture dependency must continue to resolve.
    moved_work = tmp_path / "moved-work"
    shutil.copytree(working_dir, moved_work)
    shutil.rmtree(source_dir)
    moved_output = moved_work / "output" / "textured_output.usd"
    moved_stage = Usd.Stage.Open(str(moved_output))
    assert moved_stage.GetPrimAtPath("/Root/Instance/Body").IsValid()
    assert moved_stage.GetPrimAtPath("/Root/Instance").IsInstanceable()
    moved_applied = moved_stage.GetAttributeAtPath(
        "/Root/Instance/Looks/Steel.inputs:base_color_texture_file"
    ).Get()
    moved_legacy = moved_stage.GetAttributeAtPath(
        "/Root/Instance/Looks/Steel/LegacyTexture.inputs:file"
    ).Get()
    assert Path(moved_applied.resolvedPath).is_file()
    assert Path(moved_legacy.resolvedPath).is_file()


@pytest.mark.parametrize("package_source", [False, True], ids=["usd", "usdz"])
def test_prepare_then_apply_preserves_layered_composition_and_uv_edits(
    tmp_path: Path,
    package_source: bool,
) -> None:
    """The real CLI task sequence keeps layering after UV preparation flattens."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    source_dir = tmp_path / "source"
    layers_dir = source_dir / "layers"
    layers_dir.mkdir(parents=True)
    content_path = layers_dir / "content.usda"
    content_stage = Usd.Stage.CreateNew(str(content_path))
    root = UsdGeom.Xform.Define(content_stage, "/World")
    meshes = {}
    for prim_path in (
        "/World/Mesh",
        "/World/Unchanged",
        "/World/UntouchedMissing",
    ):
        mesh = UsdGeom.Mesh.Define(content_stage, prim_path)
        mesh.GetPointsAttr().Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(0, 0, 0),
                    Gf.Vec3f(1, 0, 0),
                    Gf.Vec3f(1, 1, 0),
                    Gf.Vec3f(0, 1, 0),
                ]
            )
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        meshes[prim_path] = mesh
    unchanged_uvs = Vt.Vec2fArray(
        [
            Gf.Vec2f(0, 0),
            Gf.Vec2f(1, 0),
            Gf.Vec2f(1, 1),
            Gf.Vec2f(0, 1),
        ]
    )
    unchanged_uv = UsdGeom.PrimvarsAPI(
        meshes["/World/Unchanged"].GetPrim()
    ).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "faceVarying",
    )
    assert unchanged_uv.Set(unchanged_uvs)
    UsdShade.Material.Define(content_stage, "/World/Looks/Steel")
    content_stage.SetDefaultPrim(root.GetPrim())
    assert content_stage.GetRootLayer().Save()

    source_path = source_dir / "scene.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_path))
    source_layer.subLayerPaths = ["layers/content.usda"]
    source_layer.defaultPrim = "World"
    assert source_layer.Save()
    if package_source:
        package_path = source_dir / "scene.usdz"
        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_STORED) as package:
            package.write(source_path, "scene.usda")
            package.write(content_path, "layers/content.usda")
        source_path = package_path

    working_dir = tmp_path / "work"
    context: dict[str, Any] = {
        "usd_path": str(source_path),
        "usd_dependency_root": str(source_dir),
        "working_dir": str(working_dir),
        "texture_config": {
            "uv_policy": "generate_missing",
            "uv_projection": "box",
            "uv_repair_degenerate": False,
            "uv_scope": "target_prims",
            "uv_target_prim_paths": ["/World/Mesh"],
        },
    }
    prepare_uvs_task.PrepareUVsTask().run(context)
    prepared_path = Path(context["usd_path"])
    assert prepared_path != source_path
    assert context["source_usd_path"] == str(source_path)
    prepared_stage = Usd.Stage.Open(str(prepared_path))
    prepared_uv = UsdGeom.PrimvarsAPI(
        prepared_stage.GetPrimAtPath("/World/Mesh")
    ).GetPrimvar("st")
    expected_uvs = prepared_uv.Get()
    assert expected_uvs

    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    context.update(
        {
            "blended_textures": {
                "Steel": apply_textures_task.BlendedTextures(
                    albedo=_save_png(
                        textures_dir / "steel_albedo.png",
                        (120, 130, 140),
                    ),
                    normal="",
                    orm="",
                )
            },
            "prim_texture_units": [_unit(material_prim_path="/World/Looks/Steel")],
        }
    )
    apply_textures_task.ApplyTexturesTask().run(context)

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    shutil.move(str(working_dir / "output"), str(delivery / "output"))
    shutil.move(str(textures_dir), str(delivery / "textures"))
    shutil.rmtree(source_dir)
    shutil.rmtree(working_dir)

    delivered_path = delivery / "output" / "textured_output.usd"
    delivered_stage = Usd.Stage.Open(str(delivered_path))
    assert delivered_stage
    assert delivered_stage.GetDefaultPrim().GetPath() == Sdf.Path("/World")
    assert len(delivered_stage.GetUsedLayers()) >= 3
    delivered_mesh = delivered_stage.GetPrimAtPath("/World/Mesh")
    assert delivered_mesh.IsValid()
    delivered_uv = UsdGeom.PrimvarsAPI(delivered_mesh).GetPrimvar("st")
    assert delivered_uv.Get() == expected_uvs
    delivered_unchanged = UsdGeom.PrimvarsAPI(
        delivered_stage.GetPrimAtPath("/World/Unchanged")
    ).GetPrimvar("st")
    assert delivered_unchanged.Get() == unchanged_uvs
    delivered_untouched = UsdGeom.PrimvarsAPI(
        delivered_stage.GetPrimAtPath("/World/UntouchedMissing")
    ).GetPrimvar("st")
    assert not delivered_untouched
    applied = delivered_stage.GetAttributeAtPath(
        "/World/Looks/Steel.inputs:base_color_texture_file"
    ).Get()
    assert applied.path and Path(applied.resolvedPath).is_file()


def test_prepare_then_apply_preserves_internal_composition_and_uv_edits(
    tmp_path: Path,
) -> None:
    """Prepared single-layer output keeps composition, assets, and metadata."""
    from pxr import Gf, Usd, UsdGeom, UsdShade, Vt

    source_dir = tmp_path / "source"
    scene_dir = source_dir / "scenes"
    shared_dir = source_dir / "shared"
    scene_dir.mkdir(parents=True)
    shared_dir.mkdir()
    legacy_texture = Path(_save_png(shared_dir / "legacy.png", (18, 36, 72)))
    legacy_texture_bytes = legacy_texture.read_bytes()
    color_config = shared_dir / "config.ocio"
    color_config.write_text("ocio_profile_version: 2\n", encoding="utf-8")
    color_config_bytes = color_config.read_bytes()
    lookup_table = shared_dir / "display.cube"
    lookup_table.write_text("TITLE display\n", encoding="utf-8")
    lookup_table_bytes = lookup_table.read_bytes()
    source_path = scene_dir / "scene.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(source_stage, "/World")
    mesh = UsdGeom.Mesh.Define(source_stage, "/World/Mesh")
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 1, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    UsdShade.Material.Define(source_stage, "/World/Looks/Steel")
    legacy_shader = UsdShade.Shader.Define(
        source_stage,
        "/World/Looks/Steel/LegacyTexture",
    )
    legacy_shader.CreateIdAttr("UsdUVTexture")
    legacy_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../shared/legacy.png")
    )

    prototype = UsdGeom.Xform.Define(source_stage, "/World/Prototype").GetPrim()
    UsdGeom.Scope.Define(source_stage, "/World/Prototype/Marker")
    instance = UsdGeom.Xform.Define(source_stage, "/World/Instance").GetPrim()
    instance.GetReferences().AddInternalReference(prototype.GetPath())
    instance.SetInstanceable(True)

    choice = UsdGeom.Xform.Define(source_stage, "/World/Choice").GetPrim()
    variants = choice.GetVariantSets().AddVariantSet("shape")
    for name in ("A", "B"):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            UsdGeom.Scope.Define(source_stage, f"/World/Choice/{name}Marker")
    variants.SetVariantSelection("A")
    source_stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(source_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(source_stage, 0.01)
    pseudo_root = source_stage.GetPseudoRoot()
    pseudo_root.SetMetadata("documentation", "preserve this asset note\n")
    pseudo_root.SetMetadata("comment", "source-layer comment")
    pseudo_root.SetMetadata(
        "customLayerData",
        {
            "pipeline": "texture-agent",
            "displayLut": Sdf.AssetPath("../shared/display.cube"),
        },
    )
    pseudo_root.SetMetadata(
        "colorConfiguration",
        Sdf.AssetPath("../shared/config.ocio"),
    )
    assert source_stage.GetRootLayer().Save()
    assert not apply_textures_task._stage_has_file_backed_composition(source_stage)

    working_dir = tmp_path / "work"
    context: dict[str, Any] = {
        "usd_path": str(source_path),
        "usd_dependency_root": str(source_dir),
        "working_dir": str(working_dir),
        "texture_config": {
            "uv_policy": "generate_missing",
            "uv_projection": "box",
            "uv_repair_degenerate": False,
            "uv_scope": "target_prims",
            "uv_target_prim_paths": ["/World/Mesh"],
        },
    }
    prepare_uvs_task.PrepareUVsTask().run(context)
    prepared_path = Path(context["usd_path"])
    assert prepared_path != source_path
    prepared_stage = Usd.Stage.Open(str(prepared_path))
    expected_uvs = (
        UsdGeom.PrimvarsAPI(prepared_stage.GetPrimAtPath("/World/Mesh"))
        .GetPrimvar("st")
        .Get()
    )
    assert expected_uvs

    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    context.update(
        {
            "blended_textures": {
                "Steel": apply_textures_task.BlendedTextures(
                    albedo=_save_png(
                        textures_dir / "steel_albedo.png",
                        (120, 130, 140),
                    ),
                    normal="",
                    orm="",
                )
            },
            "prim_texture_units": [_unit(material_prim_path="/World/Looks/Steel")],
        }
    )
    apply_textures_task.ApplyTexturesTask().run(context)

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    shutil.move(str(working_dir / "output"), str(delivery / "output"))
    shutil.move(str(textures_dir), str(delivery / "textures"))
    shutil.rmtree(source_dir)
    shutil.rmtree(working_dir)

    delivered_stage = Usd.Stage.Open(str(delivery / "output" / "textured_output.usd"))
    assert delivered_stage
    delivered_uvs = (
        UsdGeom.PrimvarsAPI(delivered_stage.GetPrimAtPath("/World/Mesh"))
        .GetPrimvar("st")
        .Get()
    )
    assert delivered_uvs == expected_uvs
    delivered_texture = delivered_stage.GetAttributeAtPath(
        "/World/Looks/Steel.inputs:base_color_texture_file"
    ).Get()
    assert delivered_texture.path == "../textures/steel_albedo.png"
    assert Path(delivered_texture.resolvedPath).is_file()
    assert delivered_stage.GetMetadata("upAxis") == "Z"
    assert delivered_stage.GetMetadata("metersPerUnit") == 0.01
    delivered_legacy = delivered_stage.GetAttributeAtPath(
        "/World/Looks/Steel/LegacyTexture.inputs:file"
    ).Get()
    assert not Path(delivered_legacy.path).is_absolute()
    assert Path(delivered_legacy.resolvedPath).read_bytes() == legacy_texture_bytes

    delivered_layer = Sdf.Layer.FindOrOpen(
        str(delivery / "output" / "textured_output.usd")
    )
    assert delivered_layer.documentation == "preserve this asset note\n"
    assert delivered_layer.comment == "source-layer comment"
    assert delivered_layer.customLayerData["pipeline"] == "texture-agent"
    delivered_lut = delivered_layer.customLayerData["displayLut"]
    assert isinstance(delivered_lut, Sdf.AssetPath)
    assert not Path(delivered_lut.path).is_absolute()
    delivered_lut_path = delivery / "output" / delivered_lut.path
    assert delivered_lut_path.read_bytes() == lookup_table_bytes
    delivered_color = delivered_stage.GetMetadata("colorConfiguration")
    assert isinstance(delivered_color, Sdf.AssetPath)
    assert not Path(delivered_color.path).is_absolute()
    assert Path(delivered_color.resolvedPath).read_bytes() == color_config_bytes

    delivered_instance = delivered_stage.GetPrimAtPath("/World/Instance")
    assert delivered_instance.IsInstanceable()
    assert delivered_instance.HasAuthoredReferences()
    assert delivered_stage.GetPrimAtPath("/World/Instance/Marker").IsValid()

    delivered_variants = delivered_stage.GetPrimAtPath("/World/Choice").GetVariantSets()
    assert delivered_variants.GetNames() == ["shape"]
    assert delivered_stage.GetPrimAtPath("/World/Choice/AMarker").IsValid()
    assert delivered_variants.SetSelection("shape", "B")
    assert delivered_stage.GetPrimAtPath("/World/Choice/BMarker").IsValid()
    assert not delivered_stage.GetPrimAtPath("/World/Choice/AMarker").IsValid()


def test_transfer_prepared_uv_deltas_preserves_indexed_uvs() -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    source_stage = Usd.Stage.CreateInMemory()
    localized_stage = Usd.Stage.CreateInMemory()
    prepared_stage = Usd.Stage.CreateInMemory()
    for stage in (source_stage, localized_stage, prepared_stage):
        UsdGeom.Mesh.Define(stage, "/World/Mesh")

    prepared_uv = UsdGeom.PrimvarsAPI(
        prepared_stage.GetPrimAtPath("/World/Mesh")
    ).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "faceVarying",
    )
    assert prepared_uv.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0, 0),
                Gf.Vec2f(1, 0),
                Gf.Vec2f(1, 1),
                Gf.Vec2f(0, 1),
            ]
        )
    )
    expected_indices = Vt.IntArray([0, 1, 2, 3])
    assert prepared_uv.SetIndices(expected_indices)

    assert (
        apply_textures_task._transfer_prepared_uv_deltas(
            prepared_stage,
            source_stage,
            localized_stage,
        )
        == 1
    )
    localized_uv = UsdGeom.PrimvarsAPI(
        localized_stage.GetPrimAtPath("/World/Mesh")
    ).GetPrimvar("st")
    assert localized_uv.IsIndexed()
    assert localized_uv.GetIndices() == expected_indices
    assert localized_uv.Get() == prepared_uv.Get()


def test_transfer_prepared_uv_deltas_rejects_variant_leakage() -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    def _variant_stage() -> Usd.Stage:
        stage = Usd.Stage.CreateInMemory()
        asset = UsdGeom.Xform.Define(stage, "/Root/Asset").GetPrim()
        variants = asset.GetVariantSets().AddVariantSet("shape")
        for name in ("A", "B"):
            variants.AddVariant(name)
            variants.SetVariantSelection(name)
            with variants.GetVariantEditContext():
                UsdGeom.Mesh.Define(stage, "/Root/Asset/Mesh")
        variants.SetVariantSelection("A")
        return stage

    source_stage = _variant_stage()
    localized_stage = _variant_stage()
    prepared_stage = Usd.Stage.Open(source_stage.Flatten())
    prepared_uv = UsdGeom.PrimvarsAPI(
        prepared_stage.GetPrimAtPath("/Root/Asset/Mesh")
    ).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        "constant",
    )
    assert prepared_uv.Set(Vt.Vec2fArray([Gf.Vec2f(0.25, 0.75)]))

    with pytest.raises(RuntimeError, match="crosses a variant arc"):
        apply_textures_task._transfer_prepared_uv_deltas(
            prepared_stage,
            source_stage,
            localized_stage,
        )

    localized_variants = localized_stage.GetPrimAtPath("/Root/Asset").GetVariantSets()
    localized_variants.SetSelection("shape", "B")
    leaked_uv = UsdGeom.PrimvarsAPI(
        localized_stage.GetPrimAtPath("/Root/Asset/Mesh")
    ).GetPrimvar("st")
    assert not leaked_uv


@pytest.mark.parametrize(
    ("variant_target", "error_prefix"),
    [
        ("material", "Texture application"),
        ("binding", "Material binding"),
    ],
)
def test_apply_textures_rejects_variant_scoped_root_overrides(
    tmp_path: Path,
    variant_target: str,
    error_prefix: str,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    source_dir = tmp_path / "source"
    layers_dir = source_dir / "layers"
    layers_dir.mkdir(parents=True)
    content_path = layers_dir / "content.usda"
    content_stage = Usd.Stage.CreateNew(str(content_path))
    root = UsdGeom.Xform.Define(content_stage, "/World")
    material_path = (
        "/Looks/Steel" if variant_target == "binding" else "/World/Looks/Steel"
    )
    variants = root.GetPrim().GetVariantSets().AddVariantSet("look")
    for variant_name in ("A", "B"):
        variants.AddVariant(variant_name)
        variants.SetVariantSelection(variant_name)
        with variants.GetVariantEditContext():
            mesh = UsdGeom.Mesh.Define(content_stage, "/World/Mesh")
            if variant_target == "material":
                material = UsdShade.Material.Define(
                    content_stage,
                    material_path,
                )
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            else:
                mesh.GetPrim().CreateRelationship("material:binding").SetTargets(
                    [Sdf.Path(material_path)]
                )
    variants.SetVariantSelection("A")
    content_stage.SetDefaultPrim(root.GetPrim())
    assert content_stage.GetRootLayer().Save()

    source_path = source_dir / "scene.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_path))
    source_layer.subLayerPaths = ["layers/content.usda"]
    source_layer.defaultPrim = "World"
    assert source_layer.Save()
    if variant_target == "binding":
        source_stage = Usd.Stage.Open(str(source_path))
        UsdShade.Material.Define(source_stage, material_path)
        assert source_stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    unit_key = "SteelClone" if variant_target == "binding" else "Steel"
    unit = _unit(
        key=unit_key,
        prim_path="/World/Mesh" if variant_target == "binding" else "",
        material_prim_path=material_path,
    )
    with pytest.raises(
        RuntimeError,
        match=rf"{error_prefix} crosses a variant arc",
    ):
        apply_textures_task.ApplyTexturesTask().run(
            {
                "usd_path": str(source_path),
                "blended_textures": {
                    unit_key: apply_textures_task.BlendedTextures(
                        albedo=_save_png(
                            textures_dir / "steel_albedo.png",
                            (120, 130, 140),
                        ),
                        normal="",
                        orm="",
                    )
                },
                "prim_texture_units": [unit],
                "working_dir": str(working_dir),
            }
        )

    assert not (working_dir / "output").exists()
    assert not list(working_dir.glob(".texture-output-stage-*"))


def test_stage_texture_localization_rejects_variant_scoped_override(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    source_path = tmp_path / "scene.usda"
    source_texture = tmp_path / "legacy.png"
    _save_png(source_texture, (25, 50, 75))
    stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    variants = root.GetPrim().GetVariantSets().AddVariantSet("look")
    for variant_name in ("A", "B"):
        variants.AddVariant(variant_name)
        variants.SetVariantSelection(variant_name)
        with variants.GetVariantEditContext():
            shader = UsdShade.Shader.Define(stage, "/World/Looks/Legacy")
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath("legacy.png")
            )
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    with pytest.raises(
        RuntimeError,
        match="Texture reference localization crosses a variant arc",
    ):
        apply_textures_task._localize_stage_texture_references(
            stage,
            usd_path=str(source_path),
            working_dir=working_dir,
            output_usd_path=working_dir / "output" / "textured_output.usd",
            context={},
        )

    assert not (working_dir / "textures" / source_texture.name).exists()


def test_nonvariant_edit_guard_rejects_dormant_ancestor_variant() -> None:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World")
    material = UsdShade.Material.Define(stage, "/World/Looks/Steel")
    variants = root.GetPrim().GetVariantSets().AddVariantSet("look")
    variants.AddVariant("A")
    variants.ClearVariantSelection()

    assert apply_textures_task._prim_crosses_variant_arc(material.GetPrim())


def test_apply_textures_preflights_variant_scoped_stage_localization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    source_path = tmp_path / "scene.usda"
    source_texture = tmp_path / "legacy.png"
    _save_png(source_texture, (25, 50, 75))
    stage = Usd.Stage.CreateNew(str(source_path))
    root = UsdGeom.Xform.Define(stage, "/World")
    material = UsdShade.Material.Define(stage, "/Looks/Steel")
    mesh = UsdGeom.Mesh.Define(stage, "/Stable/Mesh")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    variants = root.GetPrim().GetVariantSets().AddVariantSet("look")
    for variant_name in ("A", "B"):
        variants.AddVariant(variant_name)
        variants.SetVariantSelection(variant_name)
        with variants.GetVariantEditContext():
            shader = UsdShade.Shader.Define(stage, "/World/Looks/Legacy")
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath("legacy.png")
            )
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()

    calls: list[str] = []

    def unexpected_clone(*_args: object, **_kwargs: object) -> str:
        calls.append("clone")
        return "/Looks/SteelClone"

    def unexpected_apply(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[int, list[str], list[str], list[str]]:
        calls.append("apply")
        return 0, [], [], []

    monkeypatch.setattr(apply_textures_task, "_clone_material", unexpected_clone)
    monkeypatch.setattr(apply_textures_task, "_apply_pbr_textures", unexpected_apply)

    working_dir = tmp_path / "work"
    (working_dir / "textures").mkdir(parents=True)
    unit = _unit(
        key="SteelClone",
        prim_path="/Stable/Mesh",
        material_prim_path="/Looks/Steel",
    )
    with pytest.raises(
        RuntimeError,
        match="Texture reference localization crosses a variant arc",
    ):
        apply_textures_task.ApplyTexturesTask().run(
            {
                "usd_path": str(source_path),
                "blended_textures": {
                    unit.key: apply_textures_task.BlendedTextures(
                        albedo=_save_png(
                            working_dir / "textures" / "steel_albedo.png",
                            (120, 130, 140),
                        ),
                        normal="",
                        orm="",
                    )
                },
                "prim_texture_units": [unit],
                "working_dir": str(working_dir),
            }
        )

    assert calls == []
    assert not (working_dir / "output" / "textured_output.usd").exists()


def test_apply_textures_preserves_dependencies_below_explicit_bundle_root(
    tmp_path: Path,
) -> None:
    """A direct CLI root may compose sibling assets inside its trusted bundle."""
    from pxr import Usd

    source_dir = tmp_path / "source"
    source_path, bundle_root = _write_sibling_layered_instance_material_stage(
        source_dir
    )
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )

    apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(source_path),
            "usd_dependency_root": str(bundle_root),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [
                _unit(material_prim_path="/Root/Instance/Looks/Steel")
            ],
            "working_dir": str(working_dir),
        }
    )

    moved_work = tmp_path / "moved-work"
    shutil.copytree(working_dir, moved_work)
    shutil.rmtree(source_dir)
    moved_output = moved_work / "output" / "textured_output.usd"
    moved_stage = Usd.Stage.Open(str(moved_output))
    assert moved_stage.GetPrimAtPath("/Root/Instance/Body").IsValid()
    legacy = moved_stage.GetAttributeAtPath(
        "/Root/Instance/Looks/Steel/LegacyTexture.inputs:file"
    ).Get()
    assert Path(legacy.resolvedPath).is_file()


def test_apply_textures_preserves_nested_default_prim(tmp_path: Path) -> None:
    """A layered wrapper must retain a valid non-root default prim path."""
    from pxr import Usd, UsdGeom

    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    content_stage = Usd.Stage.Open(str(source_path.parent / "layers/content.usda"))
    nested_default = UsdGeom.Xform.Define(content_stage, "/Root/Nested").GetPrim()
    content_stage.SetDefaultPrim(nested_default)
    assert content_stage.GetRootLayer().Save()
    source_layer = Sdf.Layer.FindOrOpen(str(source_path))
    source_layer.defaultPrim = "/Root/Nested"
    assert source_layer.Save()
    source_stage = Usd.Stage.Open(str(source_path))
    assert source_stage.GetDefaultPrim().GetPath() == Sdf.Path("/Root/Nested")

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )
    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(source_path),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [
                _unit(material_prim_path="/Root/Instance/Looks/Steel")
            ],
            "working_dir": str(working_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    output_layer = Sdf.Layer.FindOrOpen(str(output_path))
    assert output_layer.defaultPrim == "/Root/Nested"
    output_stage = Usd.Stage.Open(str(output_path))
    assert output_stage.GetDefaultPrim().GetPath() == Sdf.Path("/Root/Nested")
    assert output_stage.GetPrimAtPath("/Root/Instance/Body").IsValid()


def test_composition_snapshot_ignores_pseudo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relocation identity set contains scene prims, not the pseudo-root."""

    class _Prim:
        def __init__(self, path: str, *, pseudo_root: bool = False) -> None:
            self._path = Sdf.Path(path)
            self._pseudo_root = pseudo_root

        def IsPseudoRoot(self) -> bool:  # noqa: N802 - mirrors the USD API
            return self._pseudo_root

        def GetPath(self) -> Sdf.Path:  # noqa: N802 - mirrors the USD API
            return self._path

        def IsA(self, _schema: Any) -> bool:  # noqa: N802 - mirrors the USD API
            return False

        def IsInstanceable(self) -> bool:  # noqa: N802 - mirrors the USD API
            return False

    monkeypatch.setattr(
        apply_textures_task.Usd.PrimRange,
        "Stage",
        lambda *_args: [_Prim("/", pseudo_root=True), _Prim("/Root")],
    )
    stage = SimpleNamespace(
        GetDefaultPrim=lambda: None,
        GetPseudoRoot=lambda: SimpleNamespace(GetAllAuthoredMetadata=lambda: {}),
    )

    snapshot = apply_textures_task._composition_snapshot(stage)

    assert snapshot.prim_paths == frozenset({"/Root"})


def test_texture_metadata_helpers_cover_typed_assets_and_rejections(
    tmp_path: Path,
) -> None:
    """Metadata relocation preserves containers and rejects weak matches."""
    output_dir = tmp_path / "output"
    source_anchor = output_dir / "source"
    source_anchor.mkdir(parents=True)
    localized_asset = source_anchor / "asset.bin"
    localized_asset.write_bytes(b"asset-bytes")
    output_path = output_dir / "textured_output.usda"

    assert (
        apply_textures_task._prim_crosses_variant_arc(
            SimpleNamespace(IsValid=lambda: False)
        )
        is False
    )
    assert (
        apply_textures_task._reanchor_stage_metadata_value(
            Sdf.AssetPath(),
            metadata_key="colorConfiguration",
            source_anchor=source_anchor,
            output_path=output_path,
        )
        == Sdf.AssetPath()
    )

    authored_asset = Sdf.AssetPath("asset.bin")
    reanchored_array = apply_textures_task._reanchor_stage_metadata_value(
        Sdf.AssetPathArray([authored_asset]),
        metadata_key="assetArray",
        source_anchor=source_anchor,
        output_path=output_path,
    )
    assert [item.path for item in reanchored_array] == ["source/asset.bin"]
    assert [
        item.path
        for item in apply_textures_task._reanchor_stage_metadata_value(
            [authored_asset],
            metadata_key="assetList",
            source_anchor=source_anchor,
            output_path=output_path,
        )
    ] == ["source/asset.bin"]
    assert tuple(
        item.path
        for item in apply_textures_task._reanchor_stage_metadata_value(
            (authored_asset,),
            metadata_key="assetTuple",
            source_anchor=source_anchor,
            output_path=output_path,
        )
    ) == ("source/asset.bin",)

    match = apply_textures_task._stage_metadata_values_match
    observed_asset = Sdf.AssetPath(
        "source/asset.bin",
        str(localized_asset),
    )
    assert not match(
        authored_asset,
        "not-an-asset-path",
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert match(
        Sdf.AssetPath(),
        Sdf.AssetPath(),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert not match(
        authored_asset,
        Sdf.AssetPath(str(localized_asset)),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert not match(
        authored_asset,
        Sdf.AssetPath("missing.bin"),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )

    outside_asset = tmp_path / "outside.bin"
    outside_asset.write_bytes(b"outside")
    assert not match(
        authored_asset,
        Sdf.AssetPath("outside.bin", str(outside_asset)),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    localized_directory = output_dir / "directory"
    localized_directory.mkdir()
    assert not match(
        authored_asset,
        Sdf.AssetPath("directory", str(localized_directory)),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert not match(
        authored_asset,
        observed_asset,
        allowed_root=output_dir,
        exact_asset_paths=False,
    )

    original_asset = tmp_path / "original.bin"
    original_asset.write_bytes(localized_asset.read_bytes())
    assert match(
        Sdf.AssetPath("../shared/asset.bin", str(original_asset)),
        observed_asset,
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert match(
        Sdf.AssetPath(
            "../source/asset.bin",
            "source.usdz[source/asset.bin]",
        ),
        observed_asset,
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    wrong_dir = output_dir / "wrong"
    wrong_dir.mkdir()
    wrong_config = wrong_dir / "config.ocio"
    wrong_config.write_text("wrong", encoding="utf-8")
    assert not match(
        Sdf.AssetPath(
            "config.ocio",
            "source.usdz[Scene/config.ocio]",
        ),
        Sdf.AssetPath("wrong/config.ocio", str(wrong_config)),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert not match(
        Sdf.AssetPath(
            "asset.bin",
            "source.usdz[nested.usdz[asset.bin]]",
        ),
        observed_asset,
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert match(
        Sdf.AssetPath("source/asset.bin"),
        observed_asset,
        allowed_root=output_dir,
        exact_asset_paths=True,
    )
    assert match(
        Sdf.AssetPathArray([Sdf.AssetPath()]),
        Sdf.AssetPathArray([Sdf.AssetPath()]),
        allowed_root=output_dir,
        exact_asset_paths=False,
    )
    assert match(
        ["one", ("two",)],
        ["one", ("two",)],
        allowed_root=output_dir,
        exact_asset_paths=False,
    )


def test_stage_composition_probe_tolerates_unavailable_dependency_api() -> None:
    """Older USD layers can lack a usable composition-dependency query."""

    class _RootLayer:
        subLayerPaths: list[str] = []

        def GetCompositionAssetDependencies(self) -> NoReturn:  # noqa: N802
            raise RuntimeError("dependency query unavailable")

    root_layer = _RootLayer()
    session_layer = object()
    stage = SimpleNamespace(
        GetRootLayer=lambda: root_layer,
        GetSessionLayer=lambda: session_layer,
        GetUsedLayers=lambda: [root_layer, session_layer],
    )

    assert apply_textures_task._stage_has_file_backed_composition(stage) is False


def test_localize_composed_stage_skips_source_layer_that_disappears_after_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copied dependency remains usable if its original vanishes afterward."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "scene.usda"
    source_path.write_text('#usda 1.0\ndef Xform "Root" {}\n', encoding="utf-8")
    transient_layer = source_root / "transient.usda"
    transient_layer.write_text(
        '#usda 1.0\ndef Scope "Transient" {}\n', encoding="utf-8"
    )
    dependency = SimpleNamespace(
        realPath=str(transient_layer),
        identifier=str(transient_layer),
    )
    monkeypatch.setattr(
        apply_textures_task.UsdUtils,
        "ComputeAllDependencies",
        lambda _path: ([dependency], [], []),
    )
    original_copy = shutil.copy2

    def _copy_then_remove(source: Path, destination: Path) -> str:
        result = original_copy(source, destination)
        if Path(source) == transient_layer:
            transient_layer.unlink()
        return result

    monkeypatch.setattr(apply_textures_task.shutil, "copy2", _copy_then_remove)

    localized_root = apply_textures_task._localize_composed_stage(
        source_path,
        tmp_path / "localized",
    )

    assert localized_root.is_file()
    assert (localized_root.parent / "transient.usda").is_file()


def test_root_relocation_rejects_dependency_omitted_from_computed_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dependency seen during rewriting must already be in the copy plan."""
    from pxr import Usd

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    texture_path = source_dir / "legacy.png"
    _save_png(texture_path, (12, 34, 56))
    source_path = source_dir / "scene.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    root = source_stage.DefinePrim("/Root", "Xform")
    root.CreateAttribute("legacy", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("legacy.png")
    )
    assert source_stage.GetRootLayer().Save()

    monkeypatch.setattr(
        apply_textures_task.UsdUtils,
        "ComputeAllDependencies",
        lambda _path: ([], [], []),
    )

    with pytest.raises(RuntimeError, match="dependency was not copied"):
        apply_textures_task._localize_composed_stage(
            source_path,
            tmp_path / "localized",
            root_layer_at_output_root=True,
        )


def test_localize_composed_stage_preserves_relative_symlink_layout(
    tmp_path: Path,
) -> None:
    """A bounded symlink keeps the lexical path authored by its USD layer."""
    from pxr import Sdf, Usd, UsdShade

    source_root = tmp_path / "bundle"
    scene_dir = source_root / "scenes"
    shared_dir = source_root / "shared"
    assets_dir = source_root / "assets"
    scene_dir.mkdir(parents=True)
    shared_dir.mkdir()
    assets_dir.mkdir()
    opacity_target = Path(_save_png(assets_dir / "opacity.png", (12, 34, 56)))
    opacity_bytes = opacity_target.read_bytes()
    opacity_alias = shared_dir / "opacity.png"
    try:
        opacity_alias.symlink_to("../assets/opacity.png")
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    source_path = scene_dir / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Shader")
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../shared/opacity.png")
    )
    assert stage.GetRootLayer().Save()

    localized_dir = tmp_path / "localized"
    localized_root = apply_textures_task._localize_composed_stage(
        source_path,
        localized_dir,
        dependency_root=source_root,
    )
    assert localized_root == localized_dir / "scenes" / "scene.usda"
    localized_alias = localized_dir / "shared" / "opacity.png"
    assert localized_alias.is_file()
    assert not localized_alias.is_symlink()
    assert (
        apply_textures_task.validate_output_texture_portability(
            localized_root,
            bundle_root=localized_dir,
        )["portable"]
        is True
    )

    relocated_dir = tmp_path / "published"
    localized_dir.rename(relocated_dir)
    shutil.rmtree(source_root)
    relocated_root = relocated_dir / "scenes" / "scene.usda"
    relocated_stage = Usd.Stage.Open(str(relocated_root))
    relocated_opacity = (
        UsdShade.Shader(relocated_stage.GetPrimAtPath("/Root/Looks/Steel/Shader"))
        .GetInput("opacity_texture")
        .Get()
    )
    assert relocated_opacity.path == "../shared/opacity.png"
    assert Path(relocated_opacity.resolvedPath).read_bytes() == opacity_bytes
    assert (
        apply_textures_task.validate_output_texture_portability(
            relocated_root,
            bundle_root=relocated_dir,
        )["portable"]
        is True
    )


def test_localize_composed_stage_rejects_relative_symlink_escape_without_output(
    tmp_path: Path,
) -> None:
    """A late escape rejects the closure before copying any valid layer."""
    from pxr import Sdf, Usd

    source_root = tmp_path / "bundle"
    scene_dir = source_root / "scenes"
    shared_dir = source_root / "shared"
    scene_dir.mkdir(parents=True)
    shared_dir.mkdir()
    outside_texture = Path(_save_png(tmp_path / "outside.png", (90, 80, 70)))
    escape_alias = shared_dir / "escape.png"
    try:
        escape_alias.symlink_to(outside_texture)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    source_path = scene_dir / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    root = stage.DefinePrim("/Root", "Xform")
    root.CreateAttribute("inputs:escape_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../shared/escape.png")
    )
    assert stage.GetRootLayer().Save()

    localized_dir = tmp_path / "localized"
    with pytest.raises(
        RuntimeError,
        match="dependency outside the source dependency root",
    ):
        apply_textures_task._localize_composed_stage(
            source_path,
            localized_dir,
            dependency_root=source_root,
        )
    assert not localized_dir.exists()


def test_localize_composed_stage_supports_symlinked_dependency_root(
    tmp_path: Path,
) -> None:
    """A trusted root alias preserves layout while containment stays canonical."""
    from pxr import Usd

    source_root = tmp_path / "bundle"
    scene_dir = source_root / "scenes"
    shared_dir = source_root / "shared"
    scene_dir.mkdir(parents=True)
    shared_dir.mkdir()
    dependency_layer = shared_dir / "content.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency_layer))
    dependency_stage.DefinePrim("/Dependency", "Xform")
    assert dependency_stage.GetRootLayer().Save()
    target_layer = scene_dir / "scene.usda"
    target_stage = Usd.Stage.CreateNew(str(target_layer))
    target_stage.GetRootLayer().subLayerPaths = ["../shared/content.usda"]
    target_stage.DefinePrim("/Root", "Xform")
    assert target_stage.GetRootLayer().Save()

    source_root_alias = tmp_path / "bundle_alias"
    try:
        source_root_alias.symlink_to(source_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    # Use the real source path with the trusted alias as the dependency root.
    # This exercises the canonical-root fallback without weakening lexical
    # containment for authored dependency paths.
    source_path = target_layer

    localized_dir = tmp_path / "localized"
    localized_root = apply_textures_task._localize_composed_stage(
        source_path,
        localized_dir,
        dependency_root=source_root_alias,
    )
    assert localized_root == localized_dir / "scenes" / "scene.usda"

    relocated_dir = tmp_path / "published"
    localized_dir.rename(relocated_dir)
    source_root_alias.unlink()
    shutil.rmtree(source_root)
    relocated_stage = Usd.Stage.Open(str(relocated_dir / "scenes" / "scene.usda"))
    assert relocated_stage is not None
    assert relocated_stage.GetPrimAtPath("/Root").IsValid()
    assert relocated_stage.GetPrimAtPath("/Dependency").IsValid()


def test_localize_composed_stage_preserves_source_symlink_layout(
    tmp_path: Path,
) -> None:
    """A bounded root-layer symlink keeps its lexical bundle location."""
    from pxr import Usd

    source_root = tmp_path / "bundle"
    entry_dir = source_root / "entry"
    assets_dir = source_root / "assets"
    entry_dir.mkdir(parents=True)
    assets_dir.mkdir()

    target_layer = assets_dir / "scene.usda"
    target_stage = Usd.Stage.CreateNew(str(target_layer))
    target_stage.DefinePrim("/Root", "Xform")
    assert target_stage.GetRootLayer().Save()
    source_path = entry_dir / "scene.usda"
    try:
        source_path.symlink_to("../assets/scene.usda")
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    localized_dir = tmp_path / "localized"
    localized_root = apply_textures_task._localize_composed_stage(
        source_path,
        localized_dir,
        dependency_root=source_root,
    )
    assert localized_root == localized_dir / "entry" / "scene.usda"
    assert localized_root.is_file()
    assert not localized_root.is_symlink()

    relocated_dir = tmp_path / "published"
    localized_dir.rename(relocated_dir)
    shutil.rmtree(source_root)
    relocated_root = relocated_dir / "entry" / "scene.usda"
    relocated_stage = Usd.Stage.Open(str(relocated_root))
    assert relocated_stage is not None
    assert relocated_stage.GetPrimAtPath("/Root").IsValid()


def test_localize_composed_stage_rejects_source_symlink_escape(
    tmp_path: Path,
) -> None:
    """A lexical in-root input cannot resolve to a layer outside the bundle."""
    from pxr import Usd

    outside_layer = tmp_path / "outside.usda"
    outside_stage = Usd.Stage.CreateNew(str(outside_layer))
    outside_stage.DefinePrim("/Private", "Xform")
    assert outside_stage.GetRootLayer().Save()

    source_root = tmp_path / "bundle"
    entry_dir = source_root / "entry"
    entry_dir.mkdir(parents=True)
    source_path = entry_dir / "scene.usda"
    try:
        source_path.symlink_to(outside_layer)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(
        RuntimeError,
        match="Layered texture input is outside its trusted dependency root",
    ):
        apply_textures_task._localize_composed_stage(
            source_path,
            tmp_path / "localized",
            dependency_root=source_root,
        )


@pytest.mark.parametrize(
    ("invalid_root_kind", "error"),
    [
        ("filesystem-root", "must not be the filesystem root"),
        ("file", "is not an existing directory"),
    ],
)
def test_localize_composed_stage_rejects_invalid_root_without_output(
    tmp_path: Path,
    invalid_root_kind: str,
    error: str,
) -> None:
    """The copy boundary independently rejects roots that are unsafe to trust."""
    from pxr import Usd

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "scene.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_stage.DefinePrim("/Root", "Xform")
    assert source_stage.GetRootLayer().Save()
    invalid_root = Path("/")
    if invalid_root_kind == "file":
        invalid_root = tmp_path / "not-a-directory"
        invalid_root.write_text("not a directory", encoding="utf-8")

    localized_dir = tmp_path / "localized"
    with pytest.raises(RuntimeError, match=error):
        apply_textures_task._localize_composed_stage(
            source_path,
            localized_dir,
            dependency_root=invalid_root,
        )
    assert not localized_dir.exists()


@pytest.mark.parametrize("layered", [False, True], ids=["single-layer", "layered"])
@pytest.mark.parametrize(
    ("invalid_root_kind", "error"),
    [
        ("filesystem-root", "must not be the filesystem root"),
        ("file", "is not an existing directory"),
    ],
)
def test_apply_textures_rejects_invalid_dependency_root_without_side_effects(
    tmp_path: Path,
    layered: bool,
    invalid_root_kind: str,
    error: str,
) -> None:
    """Every input shape rejects an unsafe root before clearing old results."""
    from pxr import Usd, UsdShade

    if layered:
        source_path = _write_layered_instance_material_stage(tmp_path / "source")
        material_path = "/Root/Instance/Looks/Steel"
    else:
        source_path = tmp_path / "source" / "scene.usda"
        source_path.parent.mkdir()
        source_stage = Usd.Stage.CreateNew(str(source_path))
        UsdShade.Material.Define(source_stage, "/Root/Looks/Steel")
        assert source_stage.GetRootLayer().Save()
        material_path = "/Root/Looks/Steel"

    invalid_root = Path("/")
    if invalid_root_kind == "file":
        invalid_root = tmp_path / "not-a-directory"
        invalid_root.write_text("not a directory", encoding="utf-8")

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    output_dir = working_dir / "output"
    output_dir.mkdir()
    sentinel = output_dir / "previous-result.txt"
    sentinel.write_text("keep", encoding="utf-8")
    context = {
        "usd_path": str(source_path),
        "usd_dependency_root": str(invalid_root),
        "blended_textures": {
            "Steel": apply_textures_task.BlendedTextures(
                albedo=_save_png(
                    textures_dir / "steel_albedo.png",
                    (120, 130, 140),
                ),
                normal="",
                orm="",
            )
        },
        "prim_texture_units": [
            _unit(material_prim_path=material_path),
        ],
        "working_dir": str(working_dir),
        "output_usd_paths": [str(output_dir / "textured_output.usd")],
        "apply_textures_stats": {"previous": True},
        "rendered_image_paths": [str(output_dir / "previous.png")],
    }

    with pytest.raises(RuntimeError, match=error):
        apply_textures_task.ApplyTexturesTask().run(context)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert context["output_usd_paths"] == [str(output_dir / "textured_output.usd")]
    assert context["apply_textures_stats"] == {"previous": True}
    assert context["rendered_image_paths"] == [str(output_dir / "previous.png")]
    assert not list(working_dir.glob(".texture-output-stage-*"))


def test_apply_textures_rejects_unrelated_dependency_root_without_side_effects(
    tmp_path: Path,
) -> None:
    """The immutable source must be bounded before old output is invalidated."""
    from pxr import Usd

    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    working_dir = tmp_path / "work"
    prepared_dir = working_dir / "prepared"
    prepared_dir.mkdir(parents=True)
    prepared_path = prepared_dir / "scene.usda"
    prepared_stage = Usd.Stage.CreateNew(str(prepared_path))
    prepared_stage.DefinePrim("/Prepared", "Xform")
    assert prepared_stage.GetRootLayer().Save()
    textures_dir = working_dir / "textures"
    textures_dir.mkdir()
    output_dir = working_dir / "output"
    output_dir.mkdir()
    sentinel = output_dir / "previous-result.txt"
    sentinel.write_text("keep", encoding="utf-8")
    context = {
        "usd_path": str(prepared_path),
        "source_usd_path": str(source_path),
        "usd_dependency_root": str(unrelated_root),
        "blended_textures": {
            "Steel": apply_textures_task.BlendedTextures(
                albedo=_save_png(
                    textures_dir / "steel_albedo.png",
                    (120, 130, 140),
                ),
                normal="",
                orm="",
            )
        },
        "prim_texture_units": [
            _unit(material_prim_path="/Root/Instance/Looks/Steel"),
        ],
        "working_dir": str(working_dir),
        "output_usd_paths": [str(output_dir / "textured_output.usd")],
        "apply_textures_stats": {"previous": True},
    }

    with pytest.raises(
        RuntimeError,
        match="Authoritative texture source must be inside Texture dependency root",
    ):
        apply_textures_task.ApplyTexturesTask().run(context)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert context["output_usd_paths"] == [str(output_dir / "textured_output.usd")]
    assert context["apply_textures_stats"] == {"previous": True}
    assert not list(working_dir.glob(".texture-output-stage-*"))


@pytest.mark.parametrize(
    ("root_kind", "error"),
    [
        ("usd-parent", "Texture USD parent must not be the filesystem root"),
        (
            "working-directory",
            "Texture working directory must not be the filesystem root",
        ),
        (
            "uv-input-parent",
            "Texture UV input parent must not be the filesystem root",
        ),
    ],
)
def test_allowed_texture_source_roots_never_include_filesystem_root(
    tmp_path: Path,
    root_kind: str,
    error: str,
) -> None:
    """No implicit source-root path may widen local texture access to ``/``."""
    usd_path = str(tmp_path / "scene.usda")
    working_dir = tmp_path / "work"
    context: dict[str, Any] = {}
    if root_kind == "usd-parent":
        usd_path = "/scene.usda"
    elif root_kind == "working-directory":
        working_dir = Path("/")
    else:
        report_path = tmp_path / "uv_report.json"
        report_path.write_text(json.dumps({"input_usd": "/scene.usda"}))
        context["uv_preparation"] = {"uv_report_path": str(report_path)}

    with pytest.raises(RuntimeError, match=error):
        apply_textures_task._allowed_texture_source_roots(
            usd_path,
            working_dir,
            context,
        )


def test_localize_composed_stage_makes_copied_layers_writable(tmp_path: Path) -> None:
    """Read-only source layers become writable private editing copies."""
    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    source_layers = [source_path, source_path.parent / "layers" / "content.usda"]
    for layer in source_layers:
        layer.chmod(layer.stat().st_mode & ~stat.S_IWUSR)

    localized_root = apply_textures_task._localize_composed_stage(
        source_path,
        tmp_path / "localized",
    )

    localized_layers = [
        localized_root,
        localized_root.parent / "layers" / "content.usda",
    ]
    assert all(layer.stat().st_mode & stat.S_IWUSR for layer in localized_layers)


def test_localize_composed_stage_preserves_external_absolute_asset_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute paths outside the copy root are never silently re-anchored."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "scene.usda"
    source_path.write_text('#usda 1.0\ndef Xform "Root" {}\n', encoding="utf-8")
    external_asset = tmp_path / "outside.png"
    observed: list[str] = []

    def _exercise_callback(_layer: Any, callback: Any) -> None:
        observed.append(callback(str(external_asset)))

    monkeypatch.setattr(
        apply_textures_task.UsdUtils,
        "ModifyAssetPaths",
        _exercise_callback,
    )

    localized_root = apply_textures_task._localize_composed_stage(
        source_path,
        tmp_path / "localized",
    )

    assert localized_root.is_file()
    assert observed and set(observed) == {str(external_asset)}


def test_publish_output_tree_restores_backup_after_validator_replaces_directory(
    tmp_path: Path,
) -> None:
    """A malformed file at the public path cannot block backup restoration."""
    staging_dir = tmp_path / ".output.stage"
    staging_dir.mkdir()
    (staging_dir / "new.txt").write_text("new", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "old.txt"
    sentinel.write_text("old", encoding="utf-8")

    def _replace_with_file(path: Path) -> NoReturn:
        shutil.rmtree(path)
        path.write_text("malformed", encoding="utf-8")
        raise RuntimeError("published output was replaced")

    with pytest.raises(RuntimeError, match="published output was replaced"):
        apply_textures_task._publish_output_tree(
            staging_dir,
            output_dir,
            validate=_replace_with_file,
        )

    assert output_dir.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "old"


def test_publish_output_tree_removes_backup_after_success(tmp_path: Path) -> None:
    staging_dir = tmp_path / ".output.stage"
    staging_dir.mkdir()
    (staging_dir / "new.txt").write_text("new", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    apply_textures_task._publish_output_tree(staging_dir, output_dir)

    assert (output_dir / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (output_dir / "old.txt").exists()
    assert not list(tmp_path.glob(".output.backup-*"))


def test_validate_composed_output_reports_only_local_dependency_escapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anonymous and resolver-backed assets are ignored; local escapes fail."""
    from pxr import Usd

    allowed_root = tmp_path / "run"
    allowed_root.mkdir()
    output_path = allowed_root / "scene.usda"
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.DefinePrim("/Root", "Xform")
    stage.GetRootLayer().Save()
    snapshot = apply_textures_task._composition_snapshot(stage)
    escaped_layer = tmp_path / "outside.usda"
    escaped_asset = tmp_path / "outside.png"
    anonymous = SimpleNamespace(realPath="", identifier="anon:session")
    external = SimpleNamespace(
        realPath=str(escaped_layer),
        identifier=str(escaped_layer),
    )
    monkeypatch.setattr(
        apply_textures_task.UsdUtils,
        "ComputeAllDependencies",
        lambda _path: (
            [anonymous, external],
            ["https://example.invalid/remote.png", str(escaped_asset)],
            [],
        ),
    )

    with pytest.raises(RuntimeError, match="outside the run directory") as exc_info:
        apply_textures_task._validate_composed_output(
            output_path,
            snapshot,
            allowed_root=allowed_root,
        )

    message = str(exc_info.value)
    assert str(escaped_layer) in message
    assert str(escaped_asset) in message
    assert "anon:session" not in message
    assert "example.invalid" not in message


def test_require_portable_output_reports_unique_diagnostic_codes() -> None:
    with pytest.raises(RuntimeError, match=r"portability validation: A, B"):
        apply_textures_task._require_portable_output(
            {
                "portable": False,
                "diagnostics": [
                    {"code": "B"},
                    {"code": "A"},
                    {"code": "B"},
                    {"message": "missing code"},
                    "invalid diagnostic",
                ],
            }
        )


def test_apply_textures_defers_usdz_localization_only_when_service_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service's explicit reconstruction contract is the only USDZ skip."""
    source_path = _write_layered_instance_material_usdz(tmp_path / "source")
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )

    def _unexpected_localization(*_args, **_kwargs):
        raise AssertionError("service-owned USDZ must defer to reconstruction")

    monkeypatch.setattr(
        apply_textures_task,
        "_stage_composed_source",
        _unexpected_localization,
    )
    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(source_path),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [
                _unit(material_prim_path="/Root/Instance/Looks/Steel")
            ],
            "working_dir": str(working_dir),
            "service_managed_usdz_reconstruction": True,
        }
    )

    assert Path(result["output_usd_paths"][0]).is_file()


@pytest.mark.parametrize("failure_mode", ["stage", "open"])
def test_apply_textures_cleans_partial_layered_staging_on_setup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """Neither staging exceptions nor unreadable localized roots leak files."""
    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )
    localized_root: Path | None = None

    def _stage_or_fail(
        _source_path: Path,
        staging_dir: Path,
        **_kwargs: Any,
    ) -> Path:
        nonlocal localized_root
        partial = staging_dir / "source"
        partial.mkdir(parents=True)
        (partial / "partial.txt").write_text("partial", encoding="utf-8")
        if failure_mode == "stage":
            raise RuntimeError("layer staging failed")
        localized_root = partial / "localized.usda"
        localized_root.write_text('#usda 1.0\ndef Xform "Root" {}\n', encoding="utf-8")
        return localized_root

    monkeypatch.setattr(
        apply_textures_task,
        "_stage_composed_source",
        _stage_or_fail,
    )
    if failure_mode == "open":
        original_open = apply_textures_task.Usd.Stage.Open

        def _open_stage(path: Any, *args: Any, **kwargs: Any) -> Any:
            if localized_root is not None and str(path) == str(localized_root):
                return None
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(apply_textures_task.Usd.Stage, "Open", _open_stage)

    expected = (
        "layer staging failed"
        if failure_mode == "stage"
        else "Localized layered USD could not be opened"
    )
    with pytest.raises(RuntimeError, match=expected):
        apply_textures_task.ApplyTexturesTask().run(
            {
                "usd_path": str(source_path),
                "blended_textures": {"Steel": blended},
                "prim_texture_units": [
                    _unit(material_prim_path="/Root/Instance/Looks/Steel")
                ],
                "working_dir": str(working_dir),
            }
        )

    assert not list(working_dir.glob(".texture-output-stage-*"))


def test_apply_textures_fails_closed_when_layered_localization_loses_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truthy but root-only localization cannot publish an empty artifact."""
    from pxr import Usd

    source_dir = tmp_path / "source"
    source_path = _write_layered_instance_material_stage(source_dir)
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )
    existing_output = working_dir / "output"
    existing_output.mkdir()
    sentinel = existing_output / "previous-result.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def _root_only_localize(
        source_path: Path,
        destination: Path,
        **_kwargs,
    ) -> Path:
        source_stage = Usd.Stage.Open(str(source_path))
        assert source_stage
        destination.mkdir(parents=True, exist_ok=True)
        root_only = destination / source_path.name
        assert source_stage.GetRootLayer().Export(str(root_only))
        return root_only

    monkeypatch.setattr(
        apply_textures_task,
        "_localize_composed_stage",
        _root_only_localize,
    )

    with pytest.raises(RuntimeError, match="lost composed prim paths"):
        apply_textures_task.ApplyTexturesTask().run(
            {
                "usd_path": str(source_path),
                "blended_textures": {"Steel": blended},
                "prim_texture_units": [
                    _unit(material_prim_path="/Root/Instance/Looks/Steel")
                ],
                "working_dir": str(working_dir),
            }
        )

    assert not (working_dir / "output" / "textured_output.usd").exists()
    assert not existing_output.exists()
    assert not sentinel.exists()
    assert not list(working_dir.glob(".texture-output-stage-*"))


def test_apply_textures_fails_closed_when_localization_loses_instanceability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only deinstancing explicitly performed during authoring is permitted."""
    from pxr import Usd

    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    previous_output = working_dir / "output"
    previous_output.mkdir()
    sentinel = previous_output / "previous-result.txt"
    sentinel.write_text("keep", encoding="utf-8")
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )
    original_localize = apply_textures_task._localize_composed_stage

    def _localize_without_instanceability(
        input_path: Path,
        destination: Path,
        **kwargs,
    ) -> Path:
        localized_root = original_localize(input_path, destination, **kwargs)
        localized_stage = Usd.Stage.Open(str(localized_root))
        instance = localized_stage.GetPrimAtPath("/Root/Instance")
        assert instance.IsInstanceable()
        instance.SetInstanceable(False)
        assert localized_stage.GetRootLayer().Save()
        return localized_root

    monkeypatch.setattr(
        apply_textures_task,
        "_localize_composed_stage",
        _localize_without_instanceability,
    )

    with pytest.raises(RuntimeError, match="lost composed instanceable paths"):
        apply_textures_task.ApplyTexturesTask().run(
            {
                "usd_path": str(source_path),
                "blended_textures": {"Steel": blended},
                "prim_texture_units": [
                    _unit(material_prim_path="/Root/Instance/Looks/Steel")
                ],
                "working_dir": str(working_dir),
            }
        )

    assert not previous_output.exists()
    assert not sentinel.exists()
    assert not list(working_dir.glob(".texture-output-stage-*"))


def test_apply_textures_invalidates_previous_output_when_publish_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical-path validation is part of the atomic publication boundary."""
    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    existing_output = working_dir / "output"
    existing_output.mkdir()
    sentinel = existing_output / "previous-result.txt"
    sentinel.write_text("keep", encoding="utf-8")
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )
    original_validate = apply_textures_task._validate_composed_output

    def _reject_canonical_path(output_path: Path, *args, **kwargs) -> None:
        if Path(output_path) == existing_output / "textured_output.usd":
            raise RuntimeError("canonical publish validation failure")
        original_validate(output_path, *args, **kwargs)

    monkeypatch.setattr(
        apply_textures_task,
        "_validate_composed_output",
        _reject_canonical_path,
    )

    context = {
        "usd_path": str(source_path),
        "blended_textures": {"Steel": blended},
        "prim_texture_units": [_unit(material_prim_path="/Root/Instance/Looks/Steel")],
        "working_dir": str(working_dir),
        "output_usd_paths": [str(existing_output / "textured_output.usd")],
        "output_usdz_path": str(working_dir / "previous.usdz"),
        "output_portability": {"portable": True},
        "apply_textures_stats": {"applied_count": 1},
        "render_output_usd_paths": [str(existing_output / "render.usd")],
    }
    for key in apply_textures_task._APPLY_OUTPUT_CONTEXT_KEYS:
        context.setdefault(key, "STALE_OUTPUT_METADATA")
    with pytest.raises(RuntimeError, match="canonical publish validation failure"):
        apply_textures_task.ApplyTexturesTask().run(context)

    assert not existing_output.exists()
    assert not sentinel.exists()
    assert not (existing_output / "textured_output.usd").exists()
    assert not list(working_dir.glob(".texture-output-stage-*"))
    assert not list(working_dir.glob(".output.backup-*"))
    assert not set(apply_textures_task._APPLY_OUTPUT_CONTEXT_KEYS) & context.keys()
    from texture_agent.functions.artifact_manifest import build_artifacts_manifest

    failed_manifest = build_artifacts_manifest(context, status="failed")
    assert failed_manifest["outputs"]["output_usd"] == []
    assert failed_manifest["outputs"]["output_usdz"] is None
    assert failed_manifest["renders"]["render_available"] is False
    assert failed_manifest["renders"]["final"] == []
    assert "STALE_OUTPUT_METADATA" not in json.dumps(failed_manifest)


def test_apply_textures_removes_staging_after_mid_authoring_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after localization cannot leak a private dependency tree."""
    source_path = _write_layered_instance_material_stage(tmp_path / "source")
    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    existing_output = working_dir / "output"
    existing_output.mkdir()
    sentinel = existing_output / "previous-result.txt"
    sentinel.write_text("keep", encoding="utf-8")
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "steel_albedo.png", (120, 130, 140)),
        normal="",
        orm="",
    )

    def _fail_during_authoring(*_args, **_kwargs) -> NoReturn:
        raise RuntimeError("mid-authoring failure")

    monkeypatch.setattr(
        apply_textures_task,
        "_apply_pbr_textures",
        _fail_during_authoring,
    )

    with pytest.raises(RuntimeError, match="mid-authoring failure"):
        apply_textures_task.ApplyTexturesTask().run(
            {
                "usd_path": str(source_path),
                "blended_textures": {"Steel": blended},
                "prim_texture_units": [
                    _unit(material_prim_path="/Root/Instance/Looks/Steel")
                ],
                "working_dir": str(working_dir),
            }
        )

    assert not existing_output.exists()
    assert not sentinel.exists()
    assert not list(working_dir.glob(".texture-output-stage-*"))
    assert not list(working_dir.glob(".output.backup-*"))


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


def test_apply_textures_honors_dependency_root_for_single_layer_texture(
    tmp_path: Path,
) -> None:
    """A single-layer root may localize a sibling texture in its trusted bundle."""
    from pxr import Sdf, Usd, UsdShade

    bundle_root = tmp_path / "bundle"
    scene_dir = bundle_root / "scenes"
    shared_dir = bundle_root / "shared"
    scene_dir.mkdir(parents=True)
    shared_dir.mkdir()
    shared_texture = Path(_save_png(shared_dir / "shared_albedo.png", (220, 180, 20)))
    shared_texture_bytes = shared_texture.read_bytes()
    input_usd = scene_dir / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    untouched = UsdShade.Material.Define(stage, "/Root/Looks/Untouched")
    untouched.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("../shared/shared_albedo.png"))
    assert stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Steel_albedo.png", (20, 90, 180)),
        normal="",
        orm="",
    )

    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(input_usd),
            "usd_dependency_root": str(bundle_root),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [_unit("Steel")],
            "working_dir": str(working_dir),
        }
    )

    output_path = Path(result["output_usd_paths"][0])
    shutil.rmtree(bundle_root)
    output_stage = Usd.Stage.Open(str(output_path))
    texture = output_stage.GetAttributeAtPath(
        "/Root/Looks/Untouched.inputs:base_color_texture_file"
    ).Get()
    assert texture.path == "../textures/shared_albedo.png"
    assert Path(texture.resolvedPath).read_bytes() == shared_texture_bytes
    assert result["apply_textures_stats"]["stage_texture_refs_localized"] == [
        "/Root/Looks/Untouched:inputs:base_color_texture_file"
    ]


def test_apply_textures_honors_dependency_root_for_single_layer_mdl_texture(
    tmp_path: Path,
) -> None:
    """An MDL input may use a sibling asset inside the trusted bundle root."""
    from pxr import Sdf, Usd, UsdShade

    bundle_root = tmp_path / "bundle"
    scene_dir = bundle_root / "scenes"
    shared_dir = bundle_root / "shared"
    scene_dir.mkdir(parents=True)
    shared_dir.mkdir()
    shared_opacity = Path(_save_png(shared_dir / "opacity.png", (30, 120, 210)))
    shared_opacity_bytes = shared_opacity.read_bytes()

    input_usd = scene_dir / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_usd))
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Shader")
    shader.GetPrim().CreateAttribute(
        "info:mdl:sourceAsset",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("omniverse://nucleus.example/Steel.mdl"))
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("../shared/opacity.png")
    )
    assert stage.GetRootLayer().Save()

    working_dir = tmp_path / "work"
    textures_dir = working_dir / "textures"
    textures_dir.mkdir(parents=True)
    blended = apply_textures_task.BlendedTextures(
        albedo=_save_png(textures_dir / "Steel_albedo.png", (20, 90, 180)),
        normal="",
        orm="",
    )

    result = apply_textures_task.ApplyTexturesTask().run(
        {
            "usd_path": str(input_usd),
            "usd_dependency_root": str(bundle_root),
            "blended_textures": {"Steel": blended},
            "prim_texture_units": [_unit("Steel")],
            "working_dir": str(working_dir),
        }
    )

    original_output = Path(result["output_usd_paths"][0])
    relocated_dir = tmp_path / "published"
    working_dir.rename(relocated_dir)
    shutil.rmtree(bundle_root)

    relocated_output = relocated_dir / "output" / original_output.name
    output_stage = Usd.Stage.Open(str(relocated_output))
    output_shader = UsdShade.Shader(
        output_stage.GetPrimAtPath("/Root/Looks/Steel/Shader")
    )
    opacity = output_shader.GetInput("opacity_texture").Get()
    assert opacity.path == "../textures/Steel__opacity_texture.png"
    assert Path(opacity.resolvedPath).read_bytes() == shared_opacity_bytes
    assert result["apply_textures_stats"]["mdl_inputs_cleared"] == []
    assert result["apply_textures_stats"]["mdl_inputs_localized"] == [
        "/Root/Looks/Steel:opacity_texture"
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


def test_apply_textures_task_replaces_material_constant_passthroughs(
    tmp_path: Path,
) -> None:
    """Material interface constants are not active texture connections.

    Some packaged assets connect every PreviewSurface input to a constant
    Material interface input. Generated maps must replace those passthroughs
    rather than treating the channels as already textured.
    """
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    for input_name, type_name, value in (
        ("diffuseColor", Sdf.ValueTypeNames.Color3f, (0.1, 0.2, 0.3)),
        ("normal", Sdf.ValueTypeNames.Normal3f, (0.0, 0.0, 1.0)),
        ("occlusion", Sdf.ValueTypeNames.Float, 1.0),
        ("roughness", Sdf.ValueTypeNames.Float, 0.7),
        ("metallic", Sdf.ValueTypeNames.Float, 0.1),
    ):
        material_input = material.CreateInput(input_name, type_name)
        material_input.Set(value)
        preview.CreateInput(input_name, type_name).ConnectToSource(material_input)
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
        assert source[0].GetPrim().IsA(UsdShade.Shader)


def test_apply_textures_task_replaces_nodegraph_output_constant_passthrough(
    tmp_path: Path,
) -> None:
    """A NodeGraph output forwarding a constant is not texture coverage."""
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    interface = UsdShade.NodeGraph.Define(
        stage, "/Root/Looks/Plastic/ConstantInterface"
    )
    interface_input = interface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    interface_input.Set((0.1, 0.2, 0.3))
    interface_output = interface.CreateOutput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    )
    interface_output.ConnectToSource(interface_input)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        interface_output
    )
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
    source = output_preview.GetInput("diffuseColor").GetConnectedSource()
    assert source is not None
    assert source[0].GetPrim().GetName() == "TextureAgentAlbedoTexture"


def test_apply_textures_task_handles_deep_interface_passthrough(
    tmp_path: Path,
) -> None:
    """Deep acyclic interface chains do not exceed Python's recursion limit."""
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")

    chain_inputs = []
    chain_outputs = []
    for index in range(1_100):
        nodegraph = UsdShade.NodeGraph.Define(
            stage, f"/Root/Looks/Plastic/Interface{index}"
        )
        chain_inputs.append(
            nodegraph.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        )
        chain_outputs.append(
            nodegraph.CreateOutput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        )
        chain_outputs[-1].ConnectToSource(chain_inputs[-1])
    for current_input, next_output in zip(
        chain_inputs[:-1], chain_outputs[1:], strict=True
    ):
        current_input.ConnectToSource(next_output)
    chain_inputs[-1].Set((0.1, 0.2, 0.3))
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        chain_outputs[0]
    )
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
    source = output_preview.GetInput("diffuseColor").GetConnectedSource()
    assert source is not None
    assert source[0].GetPrim().GetName() == "TextureAgentAlbedoTexture"


def test_preview_shader_connection_handles_malformed_and_cyclic_sources() -> None:
    """Malformed and cyclic interface sources terminate conservatively."""
    from pxr import UsdShade

    class _Prim:
        def IsA(self, _schema) -> bool:
            return False

    class _AttrValue:
        def __init__(self, path: str) -> None:
            self._path = path

        def GetPath(self) -> str:
            return self._path

    class _Attribute:
        def __init__(self, path: str) -> None:
            self._attr = _AttrValue(path)
            self.connected = None

        def GetConnectedSource(self):
            return self.connected

        def GetAttr(self) -> _AttrValue:
            return self._attr

    class _Source:
        def __init__(self, attribute: _Attribute | None) -> None:
            self._attribute = attribute

        def __bool__(self) -> bool:
            return True

        def GetPrim(self) -> _Prim:
            return _Prim()

        def GetInput(self, _name: str) -> _Attribute | None:
            return self._attribute

    class _FalsySource:
        def __bool__(self) -> bool:
            return False

    malformed_source = _Attribute("/Malformed.inputs:value")
    malformed_source.connected = (
        _FalsySource(),
        "value",
        UsdShade.AttributeType.Input,
    )
    assert not apply_textures_task._preview_input_has_shader_connection(
        malformed_source
    )

    missing_attribute = _Attribute("/Missing.inputs:value")
    missing_attribute.connected = (
        _Source(None),
        "value",
        UsdShade.AttributeType.Input,
    )
    assert apply_textures_task._preview_input_has_shader_connection(missing_attribute)

    first = _Attribute("/Cycle.inputs:first")
    second = _Attribute("/Cycle.inputs:second")
    first.connected = (_Source(second), "second", UsdShade.AttributeType.Input)
    second.connected = (_Source(first), "first", UsdShade.AttributeType.Input)
    assert apply_textures_task._preview_input_has_shader_connection(first)


def test_apply_textures_task_preserves_material_interface_shader_network(
    tmp_path: Path,
) -> None:
    """An interface input backed by a shader remains authored coverage."""
    from pxr import Sdf, Usd, UsdShade

    input_path = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_path))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Plastic/Surface")
    preview.CreateIdAttr("UsdPreviewSurface")
    interface_albedo = material.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        interface_albedo
    )
    external_albedo = UsdShade.Shader.Define(stage, "/Root/Looks/ExternalAlbedoTexture")
    external_albedo.CreateIdAttr("UsdUVTexture")
    external_albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./textures/authored_albedo.png")
    )
    interface_albedo.ConnectToSource(
        external_albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
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
    preview_source = output_preview.GetInput("diffuseColor").GetConnectedSource()
    assert preview_source is not None
    assert preview_source[0].GetPrim().GetPath() == Sdf.Path("/Root/Looks/Plastic")
    material_source = (
        UsdShade.Material(output_stage.GetPrimAtPath("/Root/Looks/Plastic"))
        .GetInput("diffuseColor")
        .GetConnectedSource()
    )
    assert material_source is not None
    assert material_source[0].GetPrim().GetPath() == Sdf.Path(
        "/Root/Looks/ExternalAlbedoTexture"
    )
    assert not output_stage.GetPrimAtPath(
        "/Root/Looks/Plastic/TextureAgentAlbedoTexture"
    ).IsValid()
    for input_name in ("normal", "occlusion", "roughness", "metallic"):
        source = output_preview.GetInput(input_name).GetConnectedSource()
        assert source is not None
        assert source[0].GetPrim().GetName().startswith("TextureAgent")


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
    opacity_asset = out_shader.GetInput("opacity_texture").Get()
    opacity_ref = opacity_asset.path
    opacity_out = Path(opacity_asset.resolvedPath)
    assert opacity_ref and not Path(opacity_ref).is_absolute()
    assert Path(opacity_ref).name == "Plastic__opacity_texture.png"
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
