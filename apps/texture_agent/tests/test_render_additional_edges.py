# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import builtins
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

import texture_agent.tasks.render as render_task
import texture_agent.tasks.render_previews as render_previews_task
from texture_agent.functions.material_discovery import MaterialInfo

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdShade  # noqa: E402


def _material(name: str = "Steel") -> MaterialInfo:
    return MaterialInfo(
        prim_path=f"/Root/Looks/{name}",
        name=name,
        bound_prim_paths=["/Root/Sphere"],
    )


def _write_preview_template(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Root")
    UsdGeom.Sphere.Define(stage, "/Root/Sphere")
    UsdGeom.Camera.Define(stage, "/Root/thumbnail_CAM")
    stage.GetRootLayer().Save()


def _write_material_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Root")
    UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    stage.GetRootLayer().Save()


def test_render_preview_compose_stage_binds_template_sphere(tmp_path: Path) -> None:
    template = tmp_path / "template.usda"
    source = tmp_path / "source.usda"
    _write_preview_template(template)
    _write_material_stage(source)

    stage = render_previews_task.RenderMaterialPreviewsTask()._compose_preview_stage(
        template,
        str(source),
        _material(),
    )

    assert stage.GetPrimAtPath("/Root/Sphere").IsValid()
    assert stage.GetPrimAtPath("/Root/Looks/Steel").IsValid()


def test_render_preview_compose_stage_ignores_missing_material_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / "template.usda"
    source = tmp_path / "source.usda"
    _write_preview_template(template)
    _write_material_stage(source)
    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "world_understanding.utils.usd.material":
            raise ImportError("missing converter")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    stage = render_previews_task.RenderMaterialPreviewsTask()._compose_preview_stage(
        template,
        str(source),
        _material(),
    )

    assert stage.GetPrimAtPath("/Root/Sphere").IsValid()


def test_render_preview_run_uses_default_template_and_handles_empty_or_failed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = render_previews_task.RenderMaterialPreviewsTask()
    template = tmp_path / "template.usda"
    template.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(render_previews_task, "_DEFAULT_TEMPLATE", template)

    calls = {"count": 0}

    def fake_compose(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("compose failed")
        return object()

    monkeypatch.setattr(task, "_compose_preview_stage", fake_compose)

    import world_understanding.functions.graphics.render_remote as render_remote

    monkeypatch.setattr(render_remote, "render_all_cameras", lambda **kwargs: [])

    result = task.run(
        {
            "discovered_materials": [_material("A"), _material("B")],
            "usd_path": str(tmp_path / "source.usda"),
            "render_preview_config": {},
            "working_dir": str(tmp_path),
        }
    )

    assert result["material_previews"] == {}


def test_render_preview_run_uses_scoped_materials_without_fallback_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = render_previews_task.RenderMaterialPreviewsTask()
    template = tmp_path / "template.usda"
    template.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setattr(render_previews_task, "_DEFAULT_TEMPLATE", template)
    monkeypatch.setattr(task, "_compose_preview_stage", lambda *args: object())

    import world_understanding.functions.graphics.render_remote as render_remote

    monkeypatch.setattr(render_remote, "render_all_cameras", lambda **kwargs: [])

    result = task.run(
        {
            "texture_plan_scoped_materials": [_material("Scoped")],
            "usd_path": str(tmp_path / "source.usda"),
            "render_preview_config": {},
            "working_dir": str(tmp_path),
        }
    )

    assert result["material_previews"] == {}


def test_render_preview_empty_scope_does_not_construct_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.functions.graphics.rendering_backend_factory as factory

    def fail_create(*args: object, **kwargs: object) -> object:
        raise AssertionError("empty preview scope must not construct a backend")

    monkeypatch.setattr(factory, "create_rendering_backend", fail_create)
    working_dir = tmp_path / "work"

    result = render_previews_task.RenderMaterialPreviewsTask().run(
        {
            "texture_plan_scoped_materials": [],
            "usd_path": str(tmp_path / "source.usda"),
            "render_preview_config": {"backend": "ovrtx"},
            "working_dir": str(working_dir),
        }
    )

    assert result["material_previews"] == {}
    assert not working_dir.exists()


def test_render_preview_mock_backend_cpu_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / "template.usda"
    source = tmp_path / "source.usda"
    working_dir = tmp_path / "work"
    _write_preview_template(template)
    _write_material_stage(source)

    import world_understanding.functions.graphics.render_remote as render_remote

    monkeypatch.setattr(
        render_remote,
        "render_all_cameras",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("mock backend must not call the remote renderer")
        ),
    )

    result = render_previews_task.RenderMaterialPreviewsTask().run(
        {
            "discovered_materials": [_material()],
            "usd_path": str(source),
            "render_preview_config": {
                "backend": "mock",
                "template_scene": str(template),
                "image_width": 24,
                "image_height": 16,
            },
            "working_dir": str(working_dir),
        }
    )

    preview = Image.open(result["material_previews"]["Steel"])
    assert preview.size == (24, 16)
    assert preview.getbbox() is not None


def test_render_preview_rejects_invalid_backend_before_side_effects(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "work"

    with pytest.raises(ValueError, match="Unknown rendering backend: typo"):
        render_previews_task.RenderMaterialPreviewsTask().run(
            {
                "discovered_materials": [_material()],
                "usd_path": str(tmp_path / "source.usda"),
                "render_preview_config": {"backend": "typo"},
                "working_dir": str(working_dir),
            }
        )

    assert not working_dir.exists()


def test_render_preview_delegates_supported_backend_to_shared_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / "template.usda"
    source = tmp_path / "source.usda"
    _write_preview_template(template)
    _write_material_stage(source)
    captured: dict[str, object] = {}

    class FakeBackend:
        def render(self, **kwargs: object) -> dict[str, object]:
            captured["render"] = kwargs
            return {"results": [{"images": [Image.new("RGB", (2, 2))]}]}

    import world_understanding.functions.graphics.rendering_backend_factory as factory

    def fake_create(backend_type: object, config: object) -> FakeBackend:
        captured["create"] = (backend_type, config)
        return FakeBackend()

    monkeypatch.setattr(factory, "create_rendering_backend", fake_create)

    render_previews_task.RenderMaterialPreviewsTask().run(
        {
            "discovered_materials": [_material()],
            "usd_path": str(source),
            "render_preview_config": {
                "backend": "ovrtx",
                "template_scene": str(template),
            },
            "working_dir": str(tmp_path / "work"),
        }
    )

    assert captured["create"][0] == "ovrtx"  # type: ignore[index]
    render_call = captured["render"]
    assert render_call["cameras"] == ["/Root/thumbnail_CAM"]  # type: ignore[index]
    assert render_call["base_dir"] == source.parent  # type: ignore[index]


def test_render_helpers_cover_fallback_shapes_and_config_branches() -> None:
    assert render_task._as_path_list(("a", "", 3)) == ["a", "3"]
    assert render_task._as_path_list(object()) == []
    assert render_task._configured_camera_paths({"cameras": ("/A",)}) == ["/A"]
    assert render_task._configured_camera_paths({"camera_path": "/B"}) == ["/B"]
    assert render_task._config_float("bad", default=2.0) == 2.0
    assert render_task._config_float("-1", default=2.0) == 2.0
    assert (
        render_task._render_slot_timeout_seconds(
            {"global_render_slot_timeout_sec": "12.5"}
        )
        == 12.5
    )
    assert render_task._selected_prim_paths({}, {"focus_prim_path": "/World/A"}) == [
        "/World/A"
    ]

    rgba = Image.new("RGBA", (1, 1), (1, 2, 3, 4))
    assert (
        render_task._normalize_render_image_for_save(
            rgba,
            {"preserve_alpha": True},
        )
        is rgba
    )
    assert render_task._result_camera_path({}, 2, ["/A"]) == "camera_2"


@pytest.mark.parametrize(
    ("normalizer", "producer"),
    [
        pytest.param(
            render_task._render_result_items,
            "render_all_cameras",
            id="final-render",
        ),
        pytest.param(
            render_previews_task._render_result_items,
            "Rendering backend",
            id="preview-render",
        ),
    ],
)
def test_render_result_normalizers_preserve_caller_contract(
    normalizer: object,
    producer: str,
) -> None:
    assert callable(normalizer)
    assert normalizer([{"images": []}]) == [{"images": []}]

    with pytest.raises(ValueError) as invalid_envelope:
        normalizer({"results": "not-a-list"})
    assert str(invalid_envelope.value) == (
        f"{producer} returned a dict without a list-valued 'results' key"
    )

    with pytest.raises(TypeError) as invalid_items:
        normalizer([object()])
    assert str(invalid_items.value) == (
        f"{producer} returned unsupported result shape list; "
        "expected dict['results'] or list[dict]"
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {
                "timeout_sec": 10,
                "render_timeout_sec": 20,
                "request_timeout_sec": 30,
                "timeout": 40,
            },
            10,
            id="timeout-sec",
        ),
        pytest.param(
            {
                "render_timeout_sec": 20,
                "request_timeout_sec": 30,
                "timeout": 40,
            },
            20,
            id="render-timeout-sec",
        ),
        pytest.param(
            {"request_timeout_sec": 30, "timeout": 40},
            30,
            id="request-timeout-sec",
        ),
        pytest.param({"timeout": 40}, 40, id="timeout-compatibility"),
        pytest.param({}, 3600, id="default"),
    ],
)
def test_render_request_timeout_precedence(
    config: dict[str, object],
    expected: int,
) -> None:
    assert render_task._render_request_timeout_seconds(config) == expected


def test_render_stage_helpers_cover_lights_and_selected_materials() -> None:
    stage = Usd.Stage.CreateInMemory()
    assert (
        render_task._add_default_lights(stage, {"add_default_lights": False}) is False
    )
    assert render_task._add_default_lights(stage, {}) is True
    assert render_task._add_default_lights(stage, {}) is False

    selected = render_task._selected_prim_paths(
        {
            "material_textures": {"Steel": {}},
            "discovered_materials": [
                MaterialInfo(
                    prim_path="/Root/Looks/Steel",
                    name="Steel",
                    bound_prim_paths=["/Root/A", "/Root/A", "/Root/B"],
                )
            ],
        },
        {},
    )
    assert selected == ["/Root/A", "/Root/B"]


def test_render_output_reports_stage_open_returning_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pxr.Usd as usd_module

    monkeypatch.setattr(usd_module.Stage, "Open", lambda *args, **kwargs: None)

    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [str(tmp_path / "output.usda")],
            "render_config": {"image_width": 16},
            "working_dir": str(tmp_path),
        }
    )

    assert result["render_errors"][0]["details"] == {"reason": "stage_returned_none"}


def test_render_output_mock_backend_cpu_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Cube.Define(stage, "/Root/Cube")
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_remote

    monkeypatch.setattr(
        render_remote,
        "render_all_cameras",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("mock backend must not call the remote renderer")
        ),
    )

    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {
                "backend": "mock",
                "image_width": 24,
                "image_height": 16,
            },
            "working_dir": str(tmp_path / "work"),
        }
    )

    rendered = Image.open(result["rendered_image_paths"][0])
    assert rendered.size == (24, 16)
    assert result["render_stats"]["backend"] == "mock"
    assert result["render_stats"]["evidence_classification"] == "mock_placeholder"
    assert result["render_stats"]["production_visual_evidence"] is False
    assert any(
        item["code"] == "RENDER_MOCK_PLACEHOLDER"
        for item in result["render_diagnostics"]
    )


def test_render_output_rejects_unsupported_backend_before_side_effects(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "work"

    with pytest.raises(ValueError, match="recognized but unsupported"):
        render_task.RenderOutputTask().run(
            {
                "output_usd_paths": [str(tmp_path / "output.usda")],
                "render_config": {"backend": "warp"},
                "working_dir": str(working_dir),
            }
        )

    assert not working_dir.exists()


def test_render_output_without_images_cannot_claim_production_evidence(
    tmp_path: Path,
) -> None:
    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [],
            "render_config": {"backend": "ovrtx"},
            "working_dir": str(tmp_path / "work"),
        }
    )

    assert result["render_stats"]["backend"] == "ovrtx"
    assert result["render_stats"]["render_available"] is False
    assert result["render_stats"]["production_visual_evidence"] is False


def test_render_output_delegates_supported_backend_to_shared_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()
    captured: dict[str, object] = {}

    class FakeBackend:
        def render(self, **kwargs: object) -> dict[str, object]:
            captured["render"] = kwargs
            return {"results": [{"images": [Image.new("RGB", (2, 2))]}]}

    import world_understanding.functions.graphics.rendering_backend_factory as factory

    def fake_create(backend_type: object, config: object) -> FakeBackend:
        captured["create"] = (backend_type, config)
        return FakeBackend()

    monkeypatch.setattr(factory, "create_rendering_backend", fake_create)

    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {
                "backend": "ovrtx",
                "timeout_sec": 17,
                "render_slot_timeout_sec": 3,
            },
            "working_dir": str(tmp_path / "work"),
        }
    )

    create_call = captured["create"]
    assert create_call[0] == "ovrtx"  # type: ignore[index]
    assert create_call[1]["timeout"] == 17  # type: ignore[index]
    render_call = captured["render"]
    assert render_call["render_slot_timeout_sec"] == 3.0  # type: ignore[index]
    assert render_call["base_dir"] == usd_path.parent  # type: ignore[index]
    assert result["render_stats"]["backend"] == "ovrtx"
    assert result["render_stats"]["production_visual_evidence"] is True


def test_render_output_does_not_misclassify_ovrtx_daemon_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    class TimedOutOvRTXBackend:
        def render(self, **kwargs: object) -> dict[str, object]:
            raise TimeoutError("OvRTX daemon render timed out after 1.0s")

    import world_understanding.functions.graphics.rendering_backend_factory as factory

    monkeypatch.setattr(
        factory,
        "create_rendering_backend",
        lambda backend_type, config: TimedOutOvRTXBackend(),
    )

    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"backend": "ovrtx"},
            "working_dir": str(tmp_path / "work"),
        }
    )

    assert result["rendered_image_paths"] == []
    assert result["render_stats"]["production_visual_evidence"] is False
    assert result["render_errors"][0]["code"] == "RENDER_BACKEND_TIMEOUT"
    assert result["render_errors"][0]["details"] == {
        "backend": "ovrtx",
        "exception_type": "TimeoutError",
    }
    assert "OvRTX daemon render timed out" in result["render_errors"][0]["message"]
    assert all(
        item["code"] != "RENDER_GLOBAL_SLOT_TIMEOUT"
        for item in result["render_diagnostics"]
    )


def test_render_output_logs_asset_localization_bakes_and_queue_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_remote
    import world_understanding.functions.graphics.render_remote_async as render_async
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        usd_material,
        "localize_package_texture_assets_for_render",
        lambda stage, output_dir: 2,
    )
    monkeypatch.setattr(
        usd_material,
        "bake_texture_file_materials_to_display_color_for_render",
        lambda stage: 1,
    )
    monkeypatch.setattr(
        usd_material,
        "add_ovrtx_preview_fallbacks_for_texture_file_materials",
        lambda *args, **kwargs: 1,
    )

    @contextmanager
    def queued_slot(*, timeout_seconds: float | None = None):
        yield 0.1

    monkeypatch.setattr(render_async, "global_remote_render_slot", queued_slot)
    monkeypatch.setattr(
        render_remote,
        "render_all_cameras",
        lambda **kwargs: {"results": [{"images": [Image.new("RGB", (2, 2))]}]},
    )

    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 16, "preserve_mdl_surface": False},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert result["render_stats"]["texture_detail_package_texture_localizations"] == 2
    assert result["render_stats"]["texture_detail_display_color_bakes"] == 1
    assert result["render_stats"]["textured_preview_fallbacks"] == 1


def test_render_output_labels_solid_color_preview_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usd_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    import world_understanding.functions.graphics.render_remote as render_remote
    import world_understanding.utils.usd.material as usd_material

    monkeypatch.setattr(
        usd_material, "convert_custom_mdl_to_builtin", lambda stage: None
    )
    monkeypatch.setattr(
        usd_material,
        "localize_package_texture_assets_for_render",
        lambda stage, output_dir: 0,
    )
    monkeypatch.setattr(
        usd_material,
        "bake_texture_file_materials_to_display_color_for_render",
        lambda stage: 0,
    )
    monkeypatch.setattr(
        usd_material,
        "add_ovrtx_preview_fallbacks_for_texture_file_materials",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        render_remote,
        "render_all_cameras",
        lambda **kwargs: {"results": [{"images": [Image.new("RGB", (2, 2))]}]},
    )

    result = render_task.RenderOutputTask().run(
        {
            "output_usd_paths": [str(usd_path)],
            "render_config": {"image_width": 16, "preserve_mdl_surface": False},
            "working_dir": str(tmp_path),
        }
    )

    assert len(result["rendered_image_paths"]) == 1
    assert result["render_stats"]["textured_preview_fallbacks"] == 1
