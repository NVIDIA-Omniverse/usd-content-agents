# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

import texture_agent.cli as cli
import texture_agent.utils as utils


def test_version_callback_and_fallback_version(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._version_callback(False)

    monkeypatch.setattr(cli, "get_version", lambda: "9.9.9")
    with pytest.raises(typer.Exit):
        cli._version_callback(True)
    assert "texture-agent 9.9.9" in capsys.readouterr().out

    monkeypatch.setattr(
        utils,
        "version",
        lambda _name: (_ for _ in ()).throw(utils.PackageNotFoundError),
    )
    assert utils.get_version() == "0.0.0-dev"


def test_main_and_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    cli.main()

    called: list[bool] = []
    monkeypatch.setattr(cli, "app", lambda: called.append(True))

    runpy.run_module("texture_agent.__main__", run_name="__main__")

    assert called == [True]


def _patch_run_config(monkeypatch: pytest.MonkeyPatch, context: dict[str, Any]) -> None:
    import texture_agent.config.unified_config as unified_config

    def _load_config(
        _path: Path,
        session_id: str | None = None,
        *,
        config_data: dict[str, Any],
    ) -> dict[str, Any]:
        assert config_data == {"input": {}}
        return {"session_id": session_id}

    monkeypatch.setattr(
        unified_config,
        "load_config",
        _load_config,
    )
    monkeypatch.setattr(unified_config, "config_to_context", lambda _cfg: dict(context))


def _patch_direct_config(
    monkeypatch: pytest.MonkeyPatch,
    context: dict[str, Any],
) -> None:
    import texture_agent.config.unified_config as unified_config

    monkeypatch.setattr(
        unified_config,
        "load_config",
        lambda _path, session_id=None: {"session_id": session_id},
    )
    monkeypatch.setattr(unified_config, "config_to_context", lambda _cfg: dict(context))


def test_run_cli_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_run_config(monkeypatch, {"artifacts_manifest_path": "manifest.yaml"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")

    import texture_agent.workflows.factory as factory

    captured: dict[str, Any] = {}

    def fake_run_pipeline(
        context: dict[str, Any],
        *,
        skip: list[str] | None = None,
        only: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        captured.update(
            {"context": context, "skip": skip, "only": only, "dry": dry_run}
        )
        return {
            **context,
            "output_usd_paths": ["out.usda"],
            "rendered_image_paths": ["render.png"],
            "artifacts_manifest_path": "manifest.yaml",
        }

    monkeypatch.setattr(factory, "run_pipeline", fake_run_pipeline)

    cli.run(
        config=config_path,
        skip="discover_materials,render",
        only=None,
        dry_run=False,
        resume=True,
        session_id="sid",
        verbose=True,
    )

    out = capsys.readouterr().out
    assert "Pipeline complete!" in out
    assert "out.usda" in out
    assert "render.png" in out
    assert "manifest.yaml" in out
    assert captured["skip"] == ["discover_materials", "render"]
    assert captured["only"] == []
    assert captured["context"]["resume"] is True

    monkeypatch.setattr(
        factory,
        "run_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(typer.Exit) as exc:
        cli.run(config=config_path, verbose=True)
    assert exc.value.exit_code == 1


def test_run_cli_applies_detail_policy_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import texture_agent.config.unified_config as unified_config
    import texture_agent.workflows.factory as factory

    cfg: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    config_path = tmp_path / "config.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")

    def _load_config(
        _path: Path,
        session_id: str | None = None,
        *,
        config_data: dict[str, Any],
    ) -> dict[str, Any]:
        assert session_id is None
        assert config_data == {"input": {}}
        return cfg

    monkeypatch.setattr(unified_config, "load_config", _load_config)

    def fake_config_to_context(config: dict[str, Any]) -> dict[str, Any]:
        captured["config"] = config
        return {"texture_config": dict(config["texture"])}

    monkeypatch.setattr(unified_config, "config_to_context", fake_config_to_context)
    monkeypatch.setattr(factory, "run_pipeline", lambda context, **_kwargs: context)

    cli.run(
        config=config_path,
        skip=None,
        only=None,
        dry_run=False,
        resume=False,
        session_id=None,
        detail_policy="surface_only",
        verbose=False,
    )

    assert captured["config"]["texture"]["detail_policy"] == "surface_only"


def test_discover_generate_and_apply_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_direct_config(monkeypatch, {})

    import texture_agent.tasks as tasks
    import texture_agent.workflows.factory as factory

    material = SimpleNamespace(
        name="Brushed_Aluminum",
        base_color=(0.1, 0.2, 0.3),
        has_existing_texture=True,
        bound_prim_paths=["/World/A", "/World/B"],
    )

    class FakeDiscoverMaterialsTask:
        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            return {**context, "discovered_materials": [material]}

    monkeypatch.setattr(tasks, "DiscoverMaterialsTask", FakeDiscoverMaterialsTask)

    cli.discover(config=tmp_path / "config.yaml")
    discover_out = capsys.readouterr().out
    assert "Discovered 1 materials" in discover_out
    assert "Brushed_Aluminum" in discover_out

    def fake_run_pipeline(
        context: dict[str, Any],
        *,
        only: list[str],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        assert "discover_materials" in only
        if "blend_textures" in only:
            return {**context, "blended_textures": {"Mat": "Mat_albedo.png"}}
        return {**context, "output_usd_paths": ["textured.usda"]}

    monkeypatch.setattr(factory, "run_pipeline", fake_run_pipeline)

    cli.generate(config=tmp_path / "config.yaml")
    generate_out = capsys.readouterr().out
    assert "Generated and blended 1 textures" in generate_out
    assert "Mat_albedo.png" in generate_out

    cli.apply_cmd(config=tmp_path / "config.yaml")
    apply_out = capsys.readouterr().out
    assert "Applied textures to 1 USD file" in apply_out
    assert "textured.usda" in apply_out

    monkeypatch.setattr(
        factory,
        "run_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(typer.Exit):
        cli.generate(config=tmp_path / "config.yaml", verbose=True)
    with pytest.raises(typer.Exit):
        cli.apply_cmd(config=tmp_path / "config.yaml", verbose=True)

    monkeypatch.setattr(
        tasks,
        "DiscoverMaterialsTask",
        lambda: (_ for _ in ()).throw(RuntimeError("discover boom")),
    )
    with pytest.raises(typer.Exit):
        cli.discover(config=tmp_path / "config.yaml", verbose=True)


def test_apply_command_prepares_uvs_before_applying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_direct_config(monkeypatch, {})

    import texture_agent.workflows.factory as factory

    captured: dict[str, Any] = {}

    def fake_run_pipeline(
        context: dict[str, Any],
        *,
        only: list[str],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        captured["only"] = only
        captured["context"] = context
        return {**context, "output_usd_paths": []}

    monkeypatch.setattr(factory, "run_pipeline", fake_run_pipeline)

    cli.apply_cmd(config=tmp_path / "config.yaml")

    assert captured["only"] == [
        "prepare_uvs",
        "discover_materials",
        "generate_prompts",
        "apply_textures",
    ]
    assert captured["context"]["cached_apply_only"] is True


@pytest.mark.parametrize("with_plan", [False, True], ids=["legacy", "planned"])
def test_apply_command_resumes_cached_prompts_and_textures_without_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_plan: bool,
) -> None:
    import json

    import yaml
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    input_path = tmp_path / "input.usda"
    working_dir = tmp_path / "work"
    config_path = tmp_path / "config.yaml"

    stage = Usd.Stage.CreateNew(str(input_path))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-1, -1, 0),
            Gf.Vec3f(1, -1, 0),
            Gf.Vec3f(1, 1, 0),
            Gf.Vec3f(-1, 1, 0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    material = UsdShade.Material.Define(stage, "/World/Looks/Plastic")
    preview = UsdShade.Shader.Define(stage, "/World/Looks/Plastic/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.2, 0.2))
    material.CreateSurfaceOutput().ConnectToSource(
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {
                    "name": "cached_apply",
                    "working_dir": str(working_dir),
                },
                "input": {"usd_path": str(input_path)},
                "texture": {
                    "backend": "simple_image_gen",
                    "mode": "per_material",
                    "uv_policy": "generate_missing",
                    "uv_projection": "planar",
                },
                "auto_prompt": {"enabled": True, "user_prompt": "red"},
                "material_textures": {},
                "steps": {
                    "render_previews": {"enabled": False},
                    "render": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    unit_id = "Plastic"
    if with_plan:
        planned_context = run_pipeline(
            config_to_context(config),
            only=["discover_materials", "plan_textures"],
        )
        unit_id = planned_context["texture_plan"].selected_units[0].unit_id
        assert unit_id.startswith("tu_")
    (working_dir / "prompts" / "material_prompts.json").write_text(
        json.dumps({"Plastic": {"prompt": "cached red plastic", "opacity": 1.0}}),
        encoding="utf-8",
    )
    for suffix, color in (
        ("albedo", (200, 30, 30)),
        ("normal", (128, 128, 255)),
        ("orm", (255, 160, 0)),
    ):
        Image.new("RGB", (8, 8), color).save(
            working_dir / "textures" / f"{unit_id}_{suffix}.png"
        )

    import world_understanding.functions.models.chat_models as chat_models

    import texture_agent.tasks.execute_texture_plan as execute_texture_plan

    def forbid_backend(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("apply must not invoke prompt or image backends")

    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        forbid_backend,
    )
    monkeypatch.setattr(
        execute_texture_plan.ExecuteTexturePlanTask,
        "run",
        forbid_backend,
    )

    cli.apply_cmd(config=config_path)

    output_path = working_dir / "output" / "textured_output.usd"
    output_stage = Usd.Stage.Open(str(output_path))
    assert output_stage is not None
    output_mesh = output_stage.GetPrimAtPath("/World/Mesh")
    st = UsdGeom.PrimvarsAPI(output_mesh).GetPrimvar("st")
    assert st and st.HasAuthoredValue()
    assert st.GetInterpolation() == UsdGeom.Tokens.faceVarying
    assert len(st.Get()) == 4

    output_material = UsdShade.Material(
        output_stage.GetPrimAtPath("/World/Looks/Plastic")
    )
    surface_source = output_material.GetSurfaceOutput().GetConnectedSource()
    assert surface_source is not None
    assert surface_source[0].GetPrim().GetPath() == Sdf.Path(
        "/World/Looks/Plastic/Preview"
    )
    output_preview = UsdShade.Shader(surface_source[0].GetPrim())
    diffuse_source = output_preview.GetInput("diffuseColor").GetConnectedSource()
    assert diffuse_source is not None
    albedo_shader = UsdShade.Shader(diffuse_source[0].GetPrim())
    albedo_file = albedo_shader.GetInput("file").Get()
    assert albedo_file.path == f"../textures/{unit_id}_albedo.png"
