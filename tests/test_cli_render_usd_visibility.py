# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
import typer
from PIL import Image
from pxr import Usd, UsdGeom, UsdLux
from typer.testing import CliRunner

from world_understanding.cli import (
    _apply_camera_clip_overrides,
    _apply_light_overrides,
    _collect_camera_paths,
    _expand_isolate_paths,
    _export_rendered_stage_for_camera_metadata,
    _get_single_camera_result_or_exit,
    _hide_render_paths,
    _prepare_render_stage,
    _render_remote,
    _save_all_camera_render_outputs,
    _save_render_images,
    app,
)
from world_understanding.render_usd_backend_policy import RENDER_USD_BACKEND_NAMES
from world_understanding.rendering_backend_contract import (
    RENDERING_BACKEND_NAMES,
    rendering_backend_subset,
)


def _write_render_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    UsdGeom.Camera.Define(stage, "/World/Camera")
    stage.DefinePrim("/Looks", "Material")
    UsdGeom.Mesh.Define(stage, "/Looks/PreviewMesh")
    stage.GetRootLayer().Save()
    return path


def _prepare_kwargs(usd_path: Path, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "usd_path": str(usd_path),
        "camera": "World/Camera",
        "width": 16,
        "height": 8,
        "all_cameras": False,
        "focus": None,
        "isolate_paths": None,
        "hide_paths": None,
        "direction": "+x+y+z",
        "margin": None,
        "focal_length": None,
        "aperture": None,
        "cam_x": None,
        "cam_y": None,
        "cam_z": None,
        "target_x": None,
        "target_y": None,
        "target_z": None,
        "near_clip": None,
        "far_clip": None,
        "dome_light": None,
        "distant_light": None,
        "stage_label": "test_render",
        "prepare_message": "Preparing test stage...",
    }
    kwargs.update(overrides)
    return kwargs


def test_render_usd_backend_policy_is_canonical_capability_subset() -> None:
    assert RENDER_USD_BACKEND_NAMES == rendering_backend_subset("remote", "ovrtx")
    assert "local-ovrtx" not in RENDERING_BACKEND_NAMES


def test_render_usd_help_lists_canonical_subset_and_deprecated_alias() -> None:
    result = CliRunner().invoke(app, ["render-usd", "--help"])
    help_text = " ".join(result.output.split())

    assert result.exit_code == 0
    assert "Rendering backend: remote," in help_text
    assert "ovrtx (default: remote)." in help_text
    assert "local-ovrtx" in help_text
    assert "deprecated" in help_text
    assert "compatibility alias" in help_text


def test_render_usd_parses_hide_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_render_remote(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("world_understanding.cli._render_remote", fake_render_remote)

    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--hide",
            "/World/Wall, /World/Glass",
            "--near-clip",
            "400",
            "--far-clip",
            "1300",
        ],
    )

    assert result.exit_code == 0
    assert captured["hide_paths"] == ["/World/Wall", "/World/Glass"]
    assert captured["near_clip"] == 400.0
    assert captured["far_clip"] == 1300.0
    assert captured["remote_max_workers"] == 4
    assert captured["save_camera_json_flag"] is False
    assert callable(captured["save_camera_json_fn"])


def test_render_usd_parses_ovrtx_backend_options(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_render_ovrtx(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("world_understanding.cli._render_ovrtx", fake_render_ovrtx)

    ovrtx_venv_dir = tmp_path / "ovrtx_venv"
    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--backend",
            "ovrtx",
            "--ovrtx-log-level",
            "info",
            "--ovrtx-venv-dir",
            str(ovrtx_venv_dir),
            "--ovrtx-num-sensor-updates",
            "64",
            "--ovrtx-render-mode",
            "pt",
            "--save-camera-json",
        ],
    )

    assert result.exit_code == 0
    assert captured["sensors"] is None
    assert captured["save_camera_json_flag"] is True
    assert callable(captured["save_camera_json_fn"])
    assert captured["ovrtx_log_level"] == "info"
    assert captured["ovrtx_venv_dir"] == str(ovrtx_venv_dir)
    assert captured["ovrtx_num_sensor_updates"] == 64
    assert captured["ovrtx_render_mode"] == "pt"


def test_render_usd_rejects_ovrtx_sensors(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    called = False

    def fake_render_ovrtx(**kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("world_understanding.cli._render_ovrtx", fake_render_ovrtx)

    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--backend",
            "ovrtx",
            "--sensors",
            "depth",
        ],
    )

    assert result.exit_code == 1
    assert "OvRTX currently supports color renders only" in result.output
    assert called is False


def test_render_usd_accepts_local_ovrtx_backend_alias(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_render_ovrtx(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("world_understanding.cli._render_ovrtx", fake_render_ovrtx)

    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--backend",
            "local-ovrtx",
        ],
    )

    assert result.exit_code == 0
    assert captured["usd_path"] == str(tmp_path / "scene.usd")
    output = " ".join(result.output.split())
    assert "--backend local-ovrtx is deprecated" in output
    assert "will be removed in 0.6.0" in output
    assert "use --backend ovrtx" in output


@pytest.mark.parametrize("backend", ["warp", "mock"])
def test_render_usd_rejects_canonical_but_unsupported_backend_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
) -> None:
    def fail_dispatch(**kwargs: Any) -> None:
        raise AssertionError("renderer dispatch must not run")

    monkeypatch.setattr("world_understanding.cli._render_remote", fail_dispatch)
    monkeypatch.setattr("world_understanding.cli._render_ovrtx", fail_dispatch)

    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--backend",
            backend,
        ],
    )

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert f"Rendering backend '{backend}' is recognized but unsupported" in output
    assert "by wu render-usd" in output


def test_render_usd_ovrtx_backend_renders_with_local_backend(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Camera.Define(stage, "/Camera")
    stage.GetRootLayer().Save()

    captured_init: dict[str, Any] = {}
    captured_render: dict[str, Any] = {}

    class FakeOvRTXRenderingBackend:
        def __init__(self, **kwargs: Any) -> None:
            captured_init.update(kwargs)

        def render(self, **kwargs: Any) -> dict[str, Any]:
            captured_render.update(kwargs)
            return {
                "total_cameras": 1,
                "successful_cameras": 1,
                "failed_cameras": 0,
                "total_render_time": 0.01,
                "results": [
                    {
                        "status": "success",
                        "camera": kwargs["cameras"][0],
                        "frame_count": 1,
                        "render_time": 0.01,
                        "images": [
                            Image.new(
                                "RGB",
                                (kwargs["image_width"], kwargs["image_height"]),
                                "white",
                            )
                        ],
                    }
                ],
            }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory.OvRTXRenderingBackend",
        FakeOvRTXRenderingBackend,
    )

    output_path = tmp_path / "render.png"
    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(usd_path),
            "--output",
            str(output_path),
            "--backend",
            "ovrtx",
            "--camera",
            "Camera",
            "--width",
            "16",
            "--height",
            "8",
            "--ovrtx-log-level",
            "info",
            "--ovrtx-num-sensor-updates",
            "9",
            "--ovrtx-render-mode",
            "pt",
            "--material-target",
            "openpbr_materialx",
            "--save-camera-json",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert captured_init["log_level"] == "info"
    assert captured_init["num_sensor_updates"] == 9
    assert captured_init["render_mode"] == "pt"
    assert captured_init["material_target"] == "openpbr_materialx"
    assert captured_render["image_width"] == 16
    assert captured_render["image_height"] == 8
    assert captured_render["cameras"] == ["/Camera"]
    assert captured_render["sensors"] is None
    assert captured_render["base_dir"] == str(tmp_path)


def test_render_usd_ovrtx_all_cameras_passes_scene_camera_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Camera.Define(stage, "/World/CameraA")
    UsdGeom.Camera.Define(stage, "/World/CameraB")
    stage.GetRootLayer().Save()

    captured_render: dict[str, Any] = {}

    class FakeOvRTXRenderingBackend:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def render(self, **kwargs: Any) -> dict[str, Any]:
            captured_render.update(kwargs)
            return {
                "total_cameras": len(kwargs["cameras"]),
                "successful_cameras": len(kwargs["cameras"]),
                "failed_cameras": 0,
                "total_render_time": 0.01,
                "results": [
                    {
                        "status": "success",
                        "camera": camera,
                        "frame_count": 1,
                        "render_time": 0.01,
                        "images": [Image.new("RGB", (8, 8), "white")],
                    }
                    for camera in kwargs["cameras"]
                ],
            }

    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory.OvRTXRenderingBackend",
        FakeOvRTXRenderingBackend,
    )

    output_dir = tmp_path / "renders"
    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(usd_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "ovrtx",
            "--all-cameras",
            "--width",
            "8",
            "--height",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert captured_render["cameras"] == ["/World/CameraA", "/World/CameraB"]
    assert (output_dir / "render_World_CameraA.png").exists()
    assert (output_dir / "render_World_CameraB.png").exists()


def test_hide_render_paths_hides_imageable_subtree() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/HiddenGroup")
    child_mesh = UsdGeom.Mesh.Define(stage, "/World/HiddenGroup/Mesh")
    visible_mesh = UsdGeom.Mesh.Define(stage, "/World/VisibleMesh")

    hidden_count, missing = _hide_render_paths(
        stage,
        ["/World/HiddenGroup", "/World/Missing"],
    )

    assert hidden_count == 1
    assert missing == ["/World/Missing"]
    assert (
        UsdGeom.Imageable(child_mesh.GetPrim()).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )
    assert (
        UsdGeom.Imageable(visible_mesh.GetPrim()).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )


def test_apply_camera_clip_overrides_updates_existing_camera() -> None:
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr().Set((1.0, 1000.0))

    updated = _apply_camera_clip_overrides(
        stage,
        ["/World/Camera"],
        near_clip=25.0,
        far_clip=250.0,
    )

    assert updated == 1
    assert tuple(camera.GetClippingRangeAttr().Get()) == (25.0, 250.0)

    second = UsdGeom.Camera.Define(stage, "/World/SecondCamera")
    updated = _apply_camera_clip_overrides(
        stage,
        None,
        near_clip=10.0,
        far_clip=None,
    )
    assert updated == 2
    assert tuple(second.GetClippingRangeAttr().Get()) == (10.0, 1000000.0)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--all-cameras"], "--output-dir is required when using --all-cameras"),
        (
            ["--frames", "0:2"],
            "--output-dir is required when rendering multiple frames",
        ),
        (
            [
                "--all-cameras",
                "--output-dir",
                "renders",
                "--output",
                "render.png",
            ],
            "Cannot use --output with multiple cameras or frames",
        ),
        ([], "Either --output or --output-dir is required"),
        (
            ["--output", "render.png", "--output-dir", "renders"],
            "Cannot specify both --output and --output-dir",
        ),
    ],
)
def test_render_usd_rejects_invalid_output_combinations(
    tmp_path: Path, args: list[str], expected: str
) -> None:
    result = CliRunner().invoke(app, ["render-usd", str(tmp_path / "scene.usd"), *args])

    assert result.exit_code == 1
    assert expected in result.output


def test_render_usd_rejects_unknown_backend_verbose(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(tmp_path / "render.png"),
            "--backend",
            "not-real",
            "--verbose",
        ],
    )

    assert result.exit_code == 1
    assert "Unknown rendering backend: not-real" in result.output
    assert "Supported by wu render-usd: remote, ovrtx" in result.output
    assert "recognized but unsupported" not in result.output


def test_render_usd_rejects_invalid_remote_worker_bound_before_stage_io(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--all-cameras",
            "--output-dir",
            str(tmp_path / "renders"),
            "--remote-max-workers",
            "33",
        ],
    )

    assert result.exit_code == 1
    assert "--remote-max-workers must be between 1 and 32" in result.output
    assert not (tmp_path / "renders").exists()


def test_render_usd_rejects_invalid_remote_worker_bound_for_single_camera(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "render.png"
    result = CliRunner().invoke(
        app,
        [
            "render-usd",
            str(tmp_path / "scene.usd"),
            "--output",
            str(output_path),
            "--remote-max-workers",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "--remote-max-workers must be between 1 and 32" in result.output
    assert not output_path.exists()


def test_render_stage_helpers_lights_isolate_and_hide_descendants() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    UsdGeom.Xform.Define(stage, "/World/Group")
    UsdGeom.Mesh.Define(stage, "/World/Group/NestedMesh")
    stage.DefinePrim("/Looks", "Material")
    material_mesh = UsdGeom.Mesh.Define(stage, "/Looks/PreviewMesh")
    UsdLux.DomeLight.Define(stage, "/OldLight")

    assert _collect_camera_paths(stage) == []
    _apply_light_overrides(stage, dome_light=10.0, distant_light=20.0)
    assert not stage.GetPrimAtPath("/OldLight").IsValid()
    assert stage.GetPrimAtPath("/RenderLights/DomeLight").IsValid()
    assert stage.GetPrimAtPath("/RenderLights/DistantLight").IsValid()
    _apply_light_overrides(stage, dome_light=None, distant_light=None)

    expanded = _expand_isolate_paths(
        stage,
        ["/World/Group", "/World/Mesh", "/Missing"],
    )
    assert set(expanded) == {"/World/Group/NestedMesh", "/World/Mesh"}

    hidden_count, missing = _hide_render_paths(stage, ["/Looks", "/Missing"])
    assert hidden_count == 1
    assert missing == ["/Missing"]
    assert (
        UsdGeom.Imageable(material_mesh.GetPrim()).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )


def test_export_rendered_stage_for_camera_metadata_failure(tmp_path: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    assert _export_rendered_stage_for_camera_metadata(
        stage, "fallback.usda", False
    ) == ("fallback.usda")
    metadata_path = _export_rendered_stage_for_camera_metadata(
        stage,
        "fallback.usda",
        True,
    )
    assert Path(metadata_path).exists()
    Path(metadata_path).unlink()

    class _BadRootLayer:
        def Export(self, path: str) -> bool:
            return False

    class _BadStage:
        def GetRootLayer(self) -> _BadRootLayer:
            return _BadRootLayer()

    with pytest.raises(RuntimeError):
        _export_rendered_stage_for_camera_metadata(_BadStage(), "fallback.usda", True)


def test_prepare_render_stage_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usd_path = _write_render_stage(tmp_path / "scene.usda")

    import world_understanding.utils.usd.camera as camera_utils
    import world_understanding.utils.usd.prim as prim_utils
    import world_understanding.utils.usd.stage as stage_utils

    monkeypatch.setattr(stage_utils, "duplicate_stage", lambda stage, label: stage)

    isolated: list[list[str]] = []
    monkeypatch.setattr(
        prim_utils,
        "disable_visibility_except_for_selected_mesh_prims",
        lambda stage, paths: isolated.append(paths),
    )

    focused_kwargs: dict[str, Any] = {}

    def fake_focused_camera(focus_prim: Any, camera_path: str, **kwargs: Any) -> None:
        focused_kwargs.update(kwargs)
        UsdGeom.Camera.Define(focus_prim.GetStage(), camera_path)

    monkeypatch.setattr(
        camera_utils,
        "add_focused_corner_view_camera",
        fake_focused_camera,
    )
    focused_stage, focused_camera = _prepare_render_stage(
        **_prepare_kwargs(
            usd_path,
            focus="/World/Mesh",
            isolate_paths=["/World"],
            hide_paths=["/Looks", "/Missing"],
            near_clip=0.25,
            far_clip=250.0,
            dome_light=5.0,
            distant_light=7.0,
        )
    )
    assert focused_camera == "/Cameras/FocusedCamera"
    assert focused_stage.GetPrimAtPath(focused_camera).IsValid()
    assert focused_kwargs["near_clip"] == 0.25
    assert isolated == [["/World/Mesh"]]

    corner_kwargs: dict[str, Any] = {}

    def fake_corner_camera(stage: Usd.Stage, camera_path: str, **kwargs: Any) -> None:
        corner_kwargs.update(kwargs)
        UsdGeom.Camera.Define(stage, camera_path)

    monkeypatch.setattr(camera_utils, "add_corner_view_camera", fake_corner_camera)
    _, corner_camera = _prepare_render_stage(
        **_prepare_kwargs(
            usd_path,
            camera="MissingCamera",
            width=8,
            height=16,
            near_clip=0.5,
            far_clip=500.0,
        )
    )
    assert corner_camera == "/MissingCamera"
    assert corner_kwargs["vertical_aperture"] == 36.0

    clipped_stage, existing_camera = _prepare_render_stage(
        **_prepare_kwargs(
            usd_path,
            camera="/World/Camera",
            near_clip=1.5,
            far_clip=150.0,
        )
    )
    assert existing_camera == "/World/Camera"
    camera = UsdGeom.Camera(clipped_stage.GetPrimAtPath("/World/Camera"))
    assert tuple(camera.GetClippingRangeAttr().Get()) == (1.5, 150.0)

    with pytest.raises(typer.Exit):
        _prepare_render_stage(**_prepare_kwargs(usd_path, focus="/Missing"))


def test_prepare_render_stage_open_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "bad.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    from pxr import Usd as PxrUsd

    monkeypatch.setattr(PxrUsd.Stage, "Open", lambda *args, **kwargs: None)
    with pytest.raises(typer.Exit):
        _prepare_render_stage(**_prepare_kwargs(source))


def test_save_all_camera_render_outputs_writes_images_and_metadata(
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    output_dir = tmp_path / "all"
    image = Image.new("RGB", (4, 3), "white")
    result = {
        "results": [
            {
                "status": "failed",
                "camera": "/Bad",
                "frame_count": 0,
                "render_time": 0.1,
                "error": "bad",
            },
            {
                "status": "success",
                "camera": "/Empty",
                "frame_count": 0,
                "render_time": 0.1,
                "images": [],
            },
            {
                "status": "success",
                "camera": "/Camera",
                "frame_count": 1,
                "render_time": 0.2,
                "images": [image],
            },
            {
                "status": "success",
                "camera": "/CameraMulti",
                "frame_count": 2,
                "render_time": 0.3,
                "images": [image, image],
            },
        ]
    }

    def save_camera_json(params: dict[str, Any], path: str) -> None:
        Path(path).write_text(str(params), encoding="utf-8")

    _save_all_camera_render_outputs(
        result=result,
        output_dir=str(output_dir),
        usd_stage=stage,
        usd_path="scene.usda",
        width=4,
        verbose=True,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=lambda **kwargs: {"camera": kwargs["camera_path"]},
        save_camera_json_fn=save_camera_json,
    )

    assert (output_dir / "render_Camera.png").exists()
    assert (output_dir / "render_Camera.json").exists()
    assert (output_dir / "render_CameraMulti_0000.png").exists()
    assert (output_dir / "render_CameraMulti_0001.png").exists()

    _save_all_camera_render_outputs(
        result={
            "results": [
                {
                    "status": "success",
                    "camera": "/Warn",
                    "frame_count": 1,
                    "render_time": 0.1,
                    "images": [image],
                }
            ]
        },
        output_dir=str(tmp_path / "warn"),
        usd_stage=stage,
        usd_path="scene.usda",
        width=4,
        verbose=True,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("json failed")
        ),
        save_camera_json_fn=save_camera_json,
    )


def test_save_render_images_output_dir_output_patterns_and_json_warnings(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (4, 3), "white")
    result = {"images": [image]}

    def extract_camera_parameters(**kwargs: Any) -> dict[str, Any]:
        return {"height": kwargs["image_height"]}

    def save_camera_json(params: dict[str, Any], path: str) -> None:
        Path(path).write_text(str(params), encoding="utf-8")

    output_dir = tmp_path / "single_dir"
    _save_render_images(
        result=result,
        output=None,
        output_dir=str(output_dir),
        camera="/World/Camera",
        usd_path="scene.usda",
        width=4,
        verbose=False,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=extract_camera_parameters,
        save_camera_json_fn=save_camera_json,
    )
    assert (output_dir / "render_World_Camera.png").exists()
    assert (output_dir / "render_World_Camera.json").exists()

    multi_dir = tmp_path / "multi_dir"
    _save_render_images(
        result={"images": [image, image]},
        output=None,
        output_dir=str(multi_dir),
        camera="/World/Camera",
        usd_path="scene.usda",
        width=4,
        verbose=False,
        save_camera_json_flag=False,
        extract_camera_parameters_fn=extract_camera_parameters,
        save_camera_json_fn=save_camera_json,
    )
    assert (multi_dir / "render_World_Camera_0000.png").exists()
    assert (multi_dir / "render_World_Camera_0001.png").exists()

    _save_render_images(
        result=result,
        output=None,
        output_dir=str(tmp_path / "warn_dir"),
        camera="/World/Warn",
        usd_path="scene.usda",
        width=4,
        verbose=True,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("json failed")
        ),
        save_camera_json_fn=save_camera_json,
    )

    multi = {"images": [image, image]}
    _save_render_images(
        result=multi,
        output=str(tmp_path / "frames" / "frame_###.png"),
        output_dir=None,
        camera="/World/Camera",
        usd_path="scene.usda",
        width=4,
        verbose=False,
        save_camera_json_flag=False,
        extract_camera_parameters_fn=extract_camera_parameters,
        save_camera_json_fn=save_camera_json,
    )
    assert (tmp_path / "frames" / "frame_000.png").exists()
    assert (tmp_path / "frames" / "frame_001.png").exists()

    _save_render_images(
        result=multi,
        output=str(tmp_path / "plain.png"),
        output_dir=None,
        camera="/World/Camera",
        usd_path="scene.usda",
        width=4,
        verbose=False,
        save_camera_json_flag=False,
        extract_camera_parameters_fn=extract_camera_parameters,
        save_camera_json_fn=save_camera_json,
    )
    assert (tmp_path / "plain_0000.png").exists()
    assert (tmp_path / "plain_0001.png").exists()

    output = tmp_path / "single.png"
    _save_render_images(
        result=result,
        output=str(output),
        output_dir=None,
        camera="/World/Camera",
        usd_path="scene.usda",
        width=4,
        verbose=False,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=extract_camera_parameters,
        save_camera_json_fn=save_camera_json,
    )
    assert output.exists()
    assert (tmp_path / "single.json").exists()

    _save_render_images(
        result=result,
        output=str(tmp_path / "single_warn.png"),
        output_dir=None,
        camera="/World/Camera",
        usd_path="scene.usda",
        width=4,
        verbose=True,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("json failed")
        ),
        save_camera_json_fn=save_camera_json,
    )


def test_get_single_camera_result_or_exit_reports_first_error() -> None:
    with pytest.raises(typer.Exit):
        _get_single_camera_result_or_exit(
            {
                "successful_cameras": 0,
                "results": [
                    {"camera": "/Camera", "error": "camera exploded"},
                ],
            }
        )

    result = _get_single_camera_result_or_exit(
        {"successful_cameras": 1, "results": [{"camera": "/Camera"}]}
    )
    assert result == {"camera": "/Camera"}


def test_render_remote_all_and_single_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, "/World/Camera")
    UsdGeom.Camera.Define(stage, "/World/CameraB")
    image = Image.new("RGB", (4, 3), "white")
    prepare_calls: list[dict[str, Any]] = []
    backend_configs: list[dict[str, Any]] = []
    render_calls: list[dict[str, Any]] = []

    def fake_prepare_render_stage(**kwargs: Any) -> tuple[Usd.Stage, str]:
        prepare_calls.append(kwargs)
        return stage, "/World/Camera"

    class FakeRemoteRenderingBackend:
        def __init__(self, **kwargs: Any) -> None:
            backend_configs.append(kwargs)

        def render(self, **kwargs: Any) -> dict[str, Any]:
            render_calls.append(kwargs)
            cameras = kwargs["cameras"] or ["/World/Camera"]
            return {
                "total_cameras": len(cameras),
                "successful_cameras": len(cameras),
                "failed_cameras": 0,
                "total_render_time": 0.1,
                "results": [
                    {
                        "status": "success",
                        "camera": camera,
                        "frame_count": 1,
                        "render_time": 0.1,
                        "images": [image],
                    }
                    for camera in cameras
                ],
            }

    monkeypatch.setattr(
        "world_understanding.cli._prepare_render_stage",
        fake_prepare_render_stage,
    )
    monkeypatch.setattr(
        "world_understanding.functions.graphics.rendering_backend_factory.RemoteRenderingBackend",
        FakeRemoteRenderingBackend,
    )

    def save_camera_json(params: dict[str, Any], path: str) -> None:
        Path(path).write_text(str(params), encoding="utf-8")

    output_dir = tmp_path / "remote_all"
    _render_remote(
        usd_path=str(tmp_path / "scene.usda"),
        camera="World/Camera",
        output=None,
        width=4,
        height=3,
        frames="0:1",
        sensors="depth, linear_depth",
        all_cameras=True,
        output_dir=str(output_dir),
        verbose=True,
        save_camera_json_flag=False,
        extract_camera_parameters_fn=lambda **kwargs: {"camera": kwargs["camera_path"]},
        save_camera_json_fn=save_camera_json,
        material_target="display_color",
    )
    assert render_calls[-1]["cameras"] == ["/World/Camera", "/World/CameraB"]
    assert render_calls[-1]["sensors"] == ["depth", "linear_depth"]
    assert render_calls[-1]["base_dir"] == str(tmp_path)
    assert render_calls[-1]["max_workers"] == 2
    assert backend_configs[-1]["bundle_mdl_assets"] is True
    assert backend_configs[-1]["material_target"] == "display_color"
    assert (output_dir / "render_World_Camera.png").exists()
    assert (output_dir / "render_World_CameraB.png").exists()

    output = tmp_path / "remote.png"
    _render_remote(
        usd_path=str(tmp_path / "scene.usda"),
        camera="World/Camera",
        output=str(output),
        width=4,
        height=3,
        frames="0",
        sensors=None,
        all_cameras=False,
        output_dir=None,
        verbose=True,
        save_camera_json_flag=True,
        extract_camera_parameters_fn=lambda **kwargs: {"camera": kwargs["camera_path"]},
        save_camera_json_fn=save_camera_json,
    )
    assert render_calls[-1]["cameras"] == ["/World/Camera"]
    assert render_calls[-1]["base_dir"] == str(tmp_path)
    assert backend_configs[-1]["bundle_mdl_assets"] is True
    assert "material_target" not in backend_configs[-1]
    assert output.exists()
    assert (tmp_path / "remote.json").exists()
    assert prepare_calls[-1]["stage_label"] == "remote_render"
