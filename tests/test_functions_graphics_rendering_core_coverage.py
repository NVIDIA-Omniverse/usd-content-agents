# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage tests for high-level rendering orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from world_understanding.functions.graphics import rendering
from world_understanding.utils.usd.stage import duplicate_stage


def _stage_with_meshes() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/A")
    UsdGeom.Mesh.Define(stage, "/World/B")
    UsdGeom.Xform.Define(stage, "/Other")
    UsdGeom.Mesh.Define(stage, "/Other/C")
    return stage


def _stage_with_nested_assembly_instance() -> tuple[Usd.Stage, str, str, str]:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/GuidePrototype")
    source_guide = UsdGeom.Cube.Define(stage, "/GuidePrototype/GuideCube")
    source_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    source_guide.CreateDisplayOpacityAttr([0.0])
    UsdGeom.Xform.Define(stage, "/World")
    owner_path = "/World/Owner"
    UsdGeom.Xform.Define(stage, owner_path)
    instance_path = f"{owner_path}/InstancedGuides"
    instance_root = UsdGeom.Xform.Define(stage, instance_path).GetPrim()
    instance_root.GetReferences().AddInternalReference("/GuidePrototype")
    instance_root.SetInstanceable(True)
    member_path = f"{instance_path}/GuideCube"
    return stage, owner_path, instance_path, member_path


def _result(
    camera: str = "/Camera", images: list[Image.Image] | None = None
) -> dict[str, Any]:
    return {
        "total_cameras": 1,
        "successful_cameras": 1,
        "failed_cameras": 0,
        "total_render_time": 0.01,
        "results": [
            {
                "status": "success",
                "camera": camera,
                "frame_count": len(images or [Image.new("RGB", (4, 4), "white")]),
                "render_time": 0.01,
                "images": images or [Image.new("RGB", (4, 4), "white")],
            }
        ],
    }


def test_filename_and_camera_config_helpers() -> None:
    assert asdict(rendering.RenderingConfig())["recovered_guide_gprim_targets"] == ()
    assert rendering.parse_camera_angle_from_view_name("negx_posy_negz") == "-X+Y-Z"
    assert rendering.parse_camera_angle_from_view_name("posy") == "+Y"
    assert rendering.parse_camera_angle_from_view_name("negy") == "-Y"
    assert rendering.parse_camera_angle_from_view_name("posz") == "+Z"
    assert rendering.parse_camera_angle_from_view_name("negz") == "-Z"
    assert rendering.parse_camera_angle_from_view_name("custom_posy") == "custom_+Y"

    spec = rendering.CameraSpec(
        direction="-z",
        margin=2.0,
        focal_length=50.0,
        horizontal_aperture=12.0,
        vertical_aperture=8.0,
        near_clip_margin=0.2,
        far_clip_margin=0.3,
        name="bottom",
    )
    merged = spec.merge_with_defaults(
        default_margin=1.0,
        default_focal_length=100.0,
        default_horizontal_aperture=1.0,
        default_vertical_aperture=1.0,
        default_near_clip_margin=0.1,
        default_far_clip_margin=0.1,
    )
    assert merged == spec
    assert merged.name == "bottom"

    config = rendering.RenderingConfig(
        camera_focus_mode=rendering.CameraFocusMode.STAGE,
        per_mode_focus_mode={"prim_only": rendering.CameraFocusMode.PRIM},
        per_mode_skip_occluded={"prim_only": True},
        per_mode_use_original_materials={"original": True},
        per_mode_base_mode={"original": "prim_only"},
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
            "__all__": [rendering.CameraSpec(direction="+x+y+z")],
        },
    )
    assert config.should_use_original_materials_for_mode("original") is True
    assert config.should_use_original_materials_for_mode("other") is False
    assert config.get_base_mode("original") == "prim_only"
    assert config.get_base_mode("plain") == "plain"
    assert config.get_focus_mode_for_mode("prim_only") == rendering.CameraFocusMode.PRIM
    assert (
        config.get_focus_mode_for_mode("composition") == rendering.CameraFocusMode.STAGE
    )
    assert config.should_skip_occluded_for_mode("prim_only") is True
    assert config.should_skip_occluded_for_mode("other") is False
    assert config.get_cameras_for_mode("prim_only")[0].direction == "+x"
    assert config.get_cameras_for_mode("composition")[0].direction == "+x+y+z"

    legacy = rendering.RenderingConfig(
        camera_ordering=["+x"], camera_view_type=rendering.CameraViewType.SIDE
    )
    assert legacy.get_cameras_for_mode("prim_with_stage")[0].margin == (
        legacy.camera_prim_with_stage_margin
    )
    assert (
        legacy.get_cameras_for_mode("composition")[0].margin
        == legacy.camera_composition_margin
    )
    assert (
        legacy.get_cameras_for_mode("prim_only")[0].margin
        == legacy.camera_prim_focus_margin
    )


def test_abstract_rendering_backend_default_methods() -> None:
    class ConcreteBackend(rendering.RenderingBackend):
        pass

    ConcreteBackend.__abstractmethods__ = frozenset()
    backend = ConcreteBackend()
    assert backend.supports_sensors() is False
    assert backend.get_supported_sensor_modes() == []
    assert (
        rendering.RenderingBackend.render(backend, Usd.Stage.CreateInMemory()) is None
    )


def test_make_render_ancestors_visible_skips_instance_proxies() -> None:
    pseudo_root = SimpleNamespace(
        IsValid=lambda: True,
        IsPseudoRoot=lambda: True,
    )
    instance_proxy = SimpleNamespace(
        IsValid=lambda: True,
        IsPseudoRoot=lambda: False,
        IsInstanceProxy=lambda: True,
        GetPath=lambda: SimpleNamespace(__str__=lambda self: "/World/Proxy"),
        GetParent=lambda: pseudo_root,
    )
    stage = SimpleNamespace(GetPrimAtPath=lambda path: instance_proxy)

    rendering._make_render_ancestors_visible(stage, ["/World/Proxy"])


def test_remote_render_helpers_and_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    slot_timeouts: list[float | None] = []

    @contextmanager
    def fake_slot(*, timeout_seconds: float | None = None) -> Iterator[float]:
        slot_timeouts.append(timeout_seconds)
        yield 0.1

    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_remote_async.global_remote_render_slot",
        fake_slot,
    )
    monkeypatch.setattr(
        rendering.render_remote,
        "render_all_cameras_from_url",
        lambda **kwargs: calls.append(kwargs) or {"from_url": True},
    )
    assert rendering._render_all_cameras_from_url_with_global_slot(
        usd_url="s3://stage"
    ) == {"from_url": True}
    assert calls[-1]["usd_url"] == "s3://stage"
    assert slot_timeouts == [None]

    monkeypatch.setattr(
        rendering.render_remote,
        "render_all_cameras",
        lambda **kwargs: calls.append(kwargs) or {"remote": True},
    )
    backend = rendering.RemoteRenderingBackend(
        api_key="key",
        base_url="https://render",
        s3_bucket="bucket",
        s3_region="region",
        s3_profile="profile",
        timeout=12,
        max_retries=4,
        retry_delay=0.5,
        retry_backoff_factor=3.0,
        retry_jitter=0.2,
        bundle_mdl_assets=False,
        use_data_uri=False,
        material_target="preview_surface",
    )
    assert backend.supports_sensors() is True
    modes = backend.get_supported_sensor_modes()
    modes.append("fake")
    assert "fake" not in backend.get_supported_sensor_modes()
    assert backend.render(
        stage=Usd.Stage.CreateInMemory(),
        cameras=["/Camera"],
        image_width=8,
        image_height=None,
        frames="1",
        sensors=["depth"],
        apply_background_mask=True,
        base_dir="/assets",
        render_slot_timeout_sec=2.5,
    ) == {"remote": True}
    assert slot_timeouts == [None, 2.5]
    assert calls[-1]["image_height"] == 8
    assert calls[-1]["base_dir"] == "/assets"
    assert calls[-1]["material_target"] == "preview_surface"


def test_ovrtx_and_warp_backend_lightweight_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ovrtx = object.__new__(rendering.OvRTXRenderingBackend)
    assert ovrtx.supports_sensors() is False
    assert ovrtx.get_supported_sensor_modes() == []

    warp_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "world_understanding.functions.graphics.render_warp.render_all_cameras",
        lambda **kwargs: warp_calls.append(kwargs) or {"warp": True},
    )
    backend = rendering.WarpRenderingBackend(
        device="cpu",
        color_boost=1.5,
        enable_shadows=False,
        enable_backface_culling=False,
    )
    assert backend.supports_sensors() is True
    assert backend.get_supported_sensor_modes() == ["depth", "normal"]
    assert backend.render(
        stage=Usd.Stage.CreateInMemory(),
        cameras=["/Camera"],
        image_width=6,
        image_height=None,
        cull_style="none",
    ) == {"warp": True}
    assert warp_calls[-1]["image_height"] == 6
    assert warp_calls[-1]["enable_backface_culling"] is False

    backend.render(
        stage=Usd.Stage.CreateInMemory(),
        cameras=["/Camera"],
        image_width=6,
        image_height=4,
        cull_style="front",
        sensors=["normal"],
    )
    assert warp_calls[-1]["image_height"] == 4
    assert warp_calls[-1]["sensors"] == ["normal"]
    assert warp_calls[-1]["enable_backface_culling"] is True


class _FakeRange:
    def GetMin(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def GetMax(self) -> tuple[float, float, float]:
        return (1.0, 2.0, 3.0)


class _FakeBBox:
    def ComputeAlignedRange(self) -> _FakeRange:
        return _FakeRange()


def test_prepare_render_prims_stage_and_prim_focus_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage_with_meshes()
    prim_paths = ["/World/A", "/World/B"]
    calls: dict[str, list[Any]] = {
        "remove_animation": [],
        "nullify": [],
        "remove_lights": [],
        "colors": [],
        "disable": [],
        "corner": [],
        "side": [],
        "focused_corner": [],
        "focused_side": [],
        "bbox": [],
    }

    monkeypatch.setattr(
        "world_understanding.utils.usd.stage.remove_animation",
        lambda stage, reference_time: calls["remove_animation"].append(reference_time)
        or 2,
    )
    monkeypatch.setattr(
        rendering, "nullify_materials", lambda stage: calls["nullify"].append(stage)
    )
    monkeypatch.setattr(
        rendering,
        "remove_all_lights",
        lambda stage: calls["remove_lights"].append(stage),
    )
    monkeypatch.setattr(
        rendering,
        "get_bbox_from_prim",
        lambda prim, **kwargs: calls["bbox"].append(kwargs) or _FakeBBox(),
    )
    monkeypatch.setattr(
        rendering,
        "set_gprim_display_color",
        lambda gprim, color, time: calls["colors"].append(
            (color, float(time.GetValue()))
        ),
    )
    monkeypatch.setattr(
        rendering,
        "disable_visibility_for_all_gprims",
        lambda stage, time: calls["disable"].append(float(time.GetValue())),
    )

    def add_stage_camera(stage: Usd.Stage, camera_path: str, **kwargs: Any) -> None:
        calls["corner"].append((camera_path, kwargs))
        UsdGeom.Camera.Define(stage, camera_path)

    def add_side_camera(stage: Usd.Stage, camera_path: str, **kwargs: Any) -> None:
        calls["side"].append((camera_path, kwargs))
        UsdGeom.Camera.Define(stage, camera_path)

    def add_focused_side(prim: Usd.Prim, camera_path: str, **kwargs: Any) -> None:
        calls["focused_side"].append((camera_path, kwargs))
        UsdGeom.Camera.Define(prim.GetStage(), camera_path)

    def add_focused_corner(prim: Usd.Prim, camera_path: str, **kwargs: Any) -> None:
        calls["focused_corner"].append((camera_path, kwargs))
        UsdGeom.Camera.Define(prim.GetStage(), camera_path)

    monkeypatch.setattr(rendering, "add_corner_view_camera", add_stage_camera)
    monkeypatch.setattr(rendering, "add_side_view_camera", add_side_camera)
    monkeypatch.setattr(rendering, "add_focused_corner_view_camera", add_focused_corner)
    monkeypatch.setattr(rendering, "add_focused_side_view_camera", add_focused_side)

    stage_config = rendering.RenderingConfig(
        camera_focus_mode=rendering.CameraFocusMode.STAGE,
        camera_specs={
            "composition": [
                rendering.CameraSpec(
                    direction="+x+y+z", view_type=rendering.CameraViewType.CORNER
                ),
                rendering.CameraSpec(
                    direction="+x", view_type=rendering.CameraViewType.SIDE
                ),
            ]
        },
        should_render_prim_only=True,
        should_highlight_prim=True,
        should_assign_random_colors=True,
        root_prim_path="/World",
        bbox_purposes=("default", "render"),
    )
    _, camera_paths, frames = rendering.prepare_render_prims(
        stage,
        prim_paths,
        stage_config,
        render_mode="composition",
    )
    assert frames == 2
    assert len(camera_paths) == 2
    assert calls["remove_animation"]
    assert calls["nullify"]
    assert calls["remove_lights"]
    assert calls["disable"] == [0.0]
    assert calls["corner"] and calls["side"]
    assert calls["bbox"][0]["included_purposes"] == ("default", "render")
    assert calls["corner"][0][1]["included_purposes"] == (
        "default",
        "render",
    )
    assert calls["side"][0][1]["included_purposes"] == ("default", "render")
    assert len(calls["colors"]) >= 4

    prim_config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=False,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        camera_specs={
            "prim_only": [
                rendering.CameraSpec(
                    direction="+x", view_type=rendering.CameraViewType.SIDE
                )
            ]
        },
    )
    _, prim_camera_paths, prim_frames = rendering.prepare_render_prims(
        _stage_with_meshes(),
        ["/World/A"],
        prim_config,
        render_mode="prim_only",
    )
    assert prim_frames == 1
    assert prim_camera_paths == ["/Cameras/SideViewCamera_posx"]
    assert calls["focused_side"]
    assert calls["focused_side"][0][1]["included_purposes"] == ("default",)

    _, default_camera_paths, default_frames = rendering.prepare_render_prims(
        _stage_with_meshes(),
        ["/World/A"],
    )
    assert default_frames == 1
    assert default_camera_paths
    assert calls["focused_corner"]
    assert calls["focused_corner"][0][1]["included_purposes"] == ("default",)

    instance_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Mesh.Define(instance_stage, "/Proto")
    instance_mesh = UsdGeom.Mesh.Define(instance_stage, "/World/InstMesh")
    instance_mesh.GetPrim().GetReferences().AddInternalReference("/Proto")
    instance_mesh.GetPrim().SetInstanceable(True)
    rendering.prepare_render_prims(
        instance_stage,
        ["/World/InstMesh"],
        prim_config,
        render_mode="prim_only",
    )
    assert instance_mesh.GetPrim().IsInstance() is False


def test_prepare_render_prims_isolates_mesh_cube_and_sphere() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=True,
        should_assign_random_colors=True,
        other_color_range=(0.25, 0.25),
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        stage,
        ["/World/Cube", "/World/Sphere"],
        config,
        render_mode="prim_only",
    )

    assert frames == 2
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    time_zero = Usd.TimeCode(0)
    time_one = Usd.TimeCode(1)
    mesh_imageable = UsdGeom.Imageable(mesh.GetPrim())
    cube_gprim = UsdGeom.Gprim(cube.GetPrim())
    sphere_gprim = UsdGeom.Gprim(sphere.GetPrim())
    assert mesh_imageable.GetVisibilityAttr().Get(time_zero) == (
        UsdGeom.Tokens.invisible
    )
    assert mesh_imageable.GetVisibilityAttr().Get(time_one) == (
        UsdGeom.Tokens.invisible
    )
    assert cube_gprim.GetVisibilityAttr().Get(time_zero) == UsdGeom.Tokens.inherited
    assert cube_gprim.GetVisibilityAttr().Get(time_one) == UsdGeom.Tokens.invisible
    assert sphere_gprim.GetVisibilityAttr().Get(time_zero) == UsdGeom.Tokens.invisible
    assert sphere_gprim.GetVisibilityAttr().Get(time_one) == (UsdGeom.Tokens.inherited)
    assert tuple(cube_gprim.GetDisplayColorAttr().Get(time_zero)[0]) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert tuple(sphere_gprim.GetDisplayColorAttr().Get(time_one)[0]) == (
        pytest.approx((1.0, 0.0, 0.0))
    )


def test_prepare_render_prims_represents_xform_subtree_as_one_target() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/World")
    UsdGeom.Xform.Define(source_stage, "/World/GuideOnlyAssembly")
    panel_a = UsdGeom.Cube.Define(
        source_stage,
        "/World/GuideOnlyAssembly/PanelA",
    )
    panel_a.CreatePurposeAttr(UsdGeom.Tokens.guide)
    panel_a.CreateDisplayOpacityAttr([0.0])
    panel_b = UsdGeom.Cube.Define(
        source_stage,
        "/World/GuideOnlyAssembly/PanelB",
    )
    panel_b.CreatePurposeAttr(UsdGeom.Tokens.guide)
    panel_b.CreateDisplayOpacityAttr([0.0])
    unrelated = UsdGeom.Cube.Define(source_stage, "/World/UnrelatedGuide")
    unrelated.CreatePurposeAttr(UsdGeom.Tokens.guide)
    unrelated.CreateDisplayOpacityAttr([0.0])
    rejected = UsdGeom.Cube.Define(
        source_stage,
        "/World/GuideOnlyAssembly/RejectedGuide",
    )
    rejected.CreatePurposeAttr(UsdGeom.Tokens.guide)
    rejected.CreateDisplayOpacityAttr([0.0])
    UsdGeom.Xform.Define(source_stage, "/World/GuideOnlyAssembly/NestedOwner")
    nested = UsdGeom.Cube.Define(
        source_stage,
        "/World/GuideOnlyAssembly/NestedOwner/NestedGuide",
    )
    nested.CreatePurposeAttr(UsdGeom.Tokens.guide)
    nested.CreateDisplayOpacityAttr([0.0])
    UsdGeom.Cube.Define(source_stage, "/World/NormalTarget")
    render_stage = duplicate_stage(source_stage)
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=True,
        should_assign_random_colors=True,
        bbox_purposes=("default",),
        assembly_target_members={
            "/World/GuideOnlyAssembly": (
                "/World/GuideOnlyAssembly/PanelA",
                "/World/GuideOnlyAssembly/PanelB",
            )
        },
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )

    with pytest.raises(ValueError, match="UsdGeom.Gprim or UsdGeom.Xform"):
        rendering.prepare_render_prims(
            duplicate_stage(source_stage),
            ["/World/GuideOnlyAssembly"],
            replace(config, assembly_target_members={}),
            render_mode="prim_only",
        )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        ["/World/GuideOnlyAssembly", "/World/NormalTarget"],
        config,
        render_mode="prim_only",
    )

    assert frames == 2
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    time_zero = Usd.TimeCode(0)
    time_one = Usd.TimeCode(1)
    assembly_baseline_colors: list[tuple[float, ...]] = []
    for panel_path in (
        "/World/GuideOnlyAssembly/PanelA",
        "/World/GuideOnlyAssembly/PanelB",
    ):
        source_panel = UsdGeom.Gprim(source_stage.GetPrimAtPath(panel_path))
        render_panel = UsdGeom.Gprim(render_stage.GetPrimAtPath(panel_path))
        assert UsdGeom.Imageable(source_panel).ComputePurpose() == UsdGeom.Tokens.guide
        assert source_panel.GetDisplayOpacityAttr().Get() == [0.0]
        assert UsdGeom.Imageable(render_panel).ComputePurpose() == (
            UsdGeom.Tokens.default_
        )
        assert render_panel.GetDisplayOpacityAttr().Get() == [1.0]
        assert render_panel.GetVisibilityAttr().Get(time_zero) == (
            UsdGeom.Tokens.inherited
        )
        assert tuple(render_panel.GetDisplayColorAttr().Get(time_zero)[0]) == (
            pytest.approx((1.0, 0.0, 0.0))
        )
        assembly_baseline_colors.append(
            tuple(render_panel.GetDisplayColorAttr().Get(time_one)[0])
        )
        assert render_panel.GetVisibilityAttr().Get(time_one) == (
            UsdGeom.Tokens.invisible
        )
    assert assembly_baseline_colors[0] == pytest.approx(assembly_baseline_colors[1])
    assert assembly_baseline_colors[0] != pytest.approx((1.0, 0.0, 0.0))
    normal_target = UsdGeom.Gprim(render_stage.GetPrimAtPath("/World/NormalTarget"))
    assert tuple(normal_target.GetDisplayColorAttr().Get(time_one)[0]) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    render_unrelated = UsdGeom.Gprim(
        render_stage.GetPrimAtPath("/World/UnrelatedGuide")
    )
    assert UsdGeom.Imageable(render_unrelated).ComputePurpose() == UsdGeom.Tokens.guide
    assert render_unrelated.GetDisplayOpacityAttr().Get() == [0.0]
    assert render_unrelated.GetVisibilityAttr().Get(time_zero) == (
        UsdGeom.Tokens.invisible
    )
    for excluded_path in (
        "/World/GuideOnlyAssembly/RejectedGuide",
        "/World/GuideOnlyAssembly/NestedOwner/NestedGuide",
    ):
        excluded = UsdGeom.Gprim(render_stage.GetPrimAtPath(excluded_path))
        assert UsdGeom.Imageable(excluded).ComputePurpose() == UsdGeom.Tokens.guide
        assert excluded.GetDisplayOpacityAttr().Get() == [0.0]
        assert excluded.GetVisibilityAttr().Get(time_zero) == UsdGeom.Tokens.invisible


def test_prepare_render_prims_deinstances_nested_assembly_member_proxy() -> None:
    source_stage, owner_path, instance_path, member_path = (
        _stage_with_nested_assembly_instance()
    )
    source_text = source_stage.GetRootLayer().ExportToString()
    assert source_stage.GetPrimAtPath(instance_path).IsInstance()
    assert source_stage.GetPrimAtPath(member_path).IsInstanceProxy()
    render_layer = Sdf.Layer.CreateAnonymous("nested-assembly-copy.usda")
    render_layer.TransferContent(source_stage.GetRootLayer())
    render_stage = Usd.Stage.Open(render_layer)
    assert render_stage is not None
    assert render_stage.GetPrimAtPath(member_path).IsInstanceProxy()
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        bbox_purposes=("default",),
        assembly_target_members={owner_path: (member_path,)},
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        [owner_path],
        config,
        render_mode="prim_only",
    )

    assert frames == 1
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    assert not render_stage.GetPrimAtPath(instance_path).IsInstance()
    assert not render_stage.GetPrimAtPath(member_path).IsInstanceProxy()
    render_member = UsdGeom.Gprim(render_stage.GetPrimAtPath(member_path))
    assert UsdGeom.Imageable(render_member).ComputePurpose() == UsdGeom.Tokens.default_
    assert render_member.GetDisplayOpacityAttr().Get() == [1.0]
    assert render_member.GetVisibilityAttr().Get(Usd.TimeCode(0)) == (
        UsdGeom.Tokens.inherited
    )
    source_member = UsdGeom.Gprim(source_stage.GetPrimAtPath(member_path))
    assert source_stage.GetPrimAtPath(instance_path).IsInstance()
    assert source_stage.GetPrimAtPath(member_path).IsInstanceProxy()
    assert UsdGeom.Imageable(source_member).ComputePurpose() == UsdGeom.Tokens.guide
    assert source_member.GetDisplayOpacityAttr().Get() == [0.0]
    assert source_stage.GetRootLayer().ExportToString() == source_text


def test_make_render_prims_editable_rejects_stale_assembly_before_deinstance() -> None:
    stage, owner_path, instance_path, member_path = (
        _stage_with_nested_assembly_instance()
    )
    source_text = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match="invalid member paths"):
        rendering._make_render_prims_editable(
            stage,
            [owner_path],
            {
                owner_path: (
                    member_path,
                    f"{owner_path}/MissingGuide",
                )
            },
        )

    assert stage.GetPrimAtPath(instance_path).IsInstance()
    assert stage.GetPrimAtPath(member_path).IsInstanceProxy()
    assert stage.GetRootLayer().ExportToString() == source_text


@pytest.mark.parametrize("render_mode", ["prim_with_stage", "composition"])
def test_non_prim_only_modes_show_represented_gprims_without_unhiding_context(
    render_mode: str,
) -> None:
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/World")
    owner_path = "/World/Owner"
    UsdGeom.Xform.Define(source_stage, owner_path)
    member_path = f"{owner_path}/GuideMember"
    member = UsdGeom.Cube.Define(source_stage, member_path)
    member.CreatePurposeAttr(UsdGeom.Tokens.guide)
    member.CreateDisplayOpacityAttr([0.0])
    member.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    normal_path = "/World/NormalLeaf"
    normal = UsdGeom.Cube.Define(source_stage, normal_path)
    normal.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    context_path = "/World/HiddenContext"
    hidden_context = UsdGeom.Cube.Define(source_stage, context_path)
    hidden_context.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    source_text = source_stage.GetRootLayer().ExportToString()
    render_stage = duplicate_stage(source_stage)
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=False,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        bbox_purposes=("default",),
        assembly_target_members={owner_path: (member_path,)},
        camera_specs={
            render_mode: [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        [owner_path, normal_path],
        config,
        render_mode=render_mode,
    )

    assert frames == 2
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    render_member = UsdGeom.Gprim(render_stage.GetPrimAtPath(member_path))
    assert UsdGeom.Imageable(render_member).ComputePurpose() == UsdGeom.Tokens.default_
    assert render_member.GetDisplayOpacityAttr().Get() == [1.0]
    assert render_member.GetVisibilityAttr().Get(Usd.TimeCode(0)) == (
        UsdGeom.Tokens.inherited
    )
    assert render_member.ComputeVisibility(Usd.TimeCode(0)) == (
        UsdGeom.Tokens.inherited
    )
    render_normal = UsdGeom.Gprim(render_stage.GetPrimAtPath(normal_path))
    assert render_normal.GetVisibilityAttr().Get(Usd.TimeCode(1)) == (
        UsdGeom.Tokens.inherited
    )
    assert render_normal.ComputeVisibility(Usd.TimeCode(1)) == (
        UsdGeom.Tokens.inherited
    )
    render_context = UsdGeom.Gprim(render_stage.GetPrimAtPath(context_path))
    for time_code in (Usd.TimeCode(0), Usd.TimeCode(1)):
        assert render_context.GetVisibilityAttr().Get(time_code) == (
            UsdGeom.Tokens.invisible
        )
        assert render_context.ComputeVisibility(time_code) == UsdGeom.Tokens.invisible
    assert member.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
    assert normal.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
    assert hidden_context.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible
    assert source_stage.GetRootLayer().ExportToString() == source_text


def test_prepare_render_prims_promotes_only_recovered_guide_gprim_leaf() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/World")
    owner = UsdGeom.Cube.Define(source_stage, "/World/GuideBody")
    owner.CreatePurposeAttr(UsdGeom.Tokens.guide)
    owner.CreateDisplayOpacityAttr([0.0])
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    child = UsdGeom.Cube.Define(source_stage, "/World/GuideBody/NestedGuide")
    child.CreatePurposeAttr(UsdGeom.Tokens.guide)
    child.CreateDisplayOpacityAttr([0.0])
    unrelated = UsdGeom.Cube.Define(source_stage, "/World/UnrelatedGuide")
    unrelated.CreatePurposeAttr(UsdGeom.Tokens.guide)
    unrelated.CreateDisplayOpacityAttr([0.0])
    source_text = source_stage.GetRootLayer().ExportToString()
    render_stage = duplicate_stage(source_stage)
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        bbox_purposes=("default",),
        recovered_guide_gprim_targets=(
            "/World/GuideBody",
            # A recovered target outside this batch must not be promoted.
            "/World/UnrelatedGuide",
        ),
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        ["/World/GuideBody"],
        config,
        render_mode="prim_only",
    )

    assert frames == 1
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    render_owner = UsdGeom.Gprim(render_stage.GetPrimAtPath("/World/GuideBody"))
    assert UsdGeom.Imageable(render_owner).ComputePurpose() == UsdGeom.Tokens.default_
    assert render_owner.GetDisplayOpacityAttr().Get() == [1.0]
    assert render_owner.GetVisibilityAttr().Get(Usd.TimeCode(0)) == (
        UsdGeom.Tokens.inherited
    )
    for path in (
        "/World/GuideBody/NestedGuide",
        "/World/UnrelatedGuide",
    ):
        excluded = UsdGeom.Gprim(render_stage.GetPrimAtPath(path))
        assert UsdGeom.Imageable(excluded).ComputePurpose() == UsdGeom.Tokens.guide
        assert excluded.GetDisplayOpacityAttr().Get() == [0.0]
        assert excluded.GetVisibilityAttr().Get(Usd.TimeCode(0)) == (
            UsdGeom.Tokens.invisible
        )
    assert UsdGeom.Imageable(owner.GetPrim()).ComputePurpose() == UsdGeom.Tokens.guide
    assert owner.GetDisplayOpacityAttr().Get() == [0.0]
    assert source_stage.GetRootLayer().ExportToString() == source_text


def test_prepare_render_prims_does_not_promote_stale_assembly_target() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/World")
    UsdGeom.Xform.Define(source_stage, "/World/SkippedAssembly")
    skipped_guide = UsdGeom.Cube.Define(
        source_stage,
        "/World/SkippedAssembly/GuidePanel",
    )
    skipped_guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    skipped_guide.CreateDisplayOpacityAttr([0.0])
    UsdGeom.Cube.Define(source_stage, "/World/NormalTarget")
    source_text = source_stage.GetRootLayer().ExportToString()
    render_stage = duplicate_stage(source_stage)
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=False,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        assembly_target_members={
            "/World/SkippedAssembly": ("/World/SkippedAssembly/GuidePanel",)
        },
        camera_specs={
            "prim_with_stage": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        ["/World/NormalTarget"],
        config,
        render_mode="prim_with_stage",
    )

    assert frames == 1
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    render_guide = UsdGeom.Gprim(
        render_stage.GetPrimAtPath("/World/SkippedAssembly/GuidePanel")
    )
    assert UsdGeom.Imageable(render_guide).ComputePurpose() == UsdGeom.Tokens.guide
    assert render_guide.GetDisplayOpacityAttr().Get() == [0.0]
    assert source_stage.GetRootLayer().ExportToString() == source_text


def test_guide_promotion_survives_frame_zero_animation_stripping() -> None:
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/World")
    UsdGeom.Xform.Define(source_stage, "/World/Assembly")
    member = UsdGeom.Cube.Define(source_stage, "/World/Assembly/GuideMember")
    member.CreatePurposeAttr(UsdGeom.Tokens.guide)
    member_opacity = member.CreateDisplayOpacityAttr()
    member_opacity.Set([0.0], Usd.TimeCode(0))
    member_opacity.Set([0.5], Usd.TimeCode(10))
    leaf = UsdGeom.Cube.Define(source_stage, "/World/GuideLeaf")
    leaf.CreatePurposeAttr(UsdGeom.Tokens.guide)
    leaf_opacity = leaf.CreateDisplayOpacityAttr()
    leaf_opacity.Set([0.0], Usd.TimeCode(0))
    leaf_opacity.Set([0.5], Usd.TimeCode(10))
    source_text = source_stage.GetRootLayer().ExportToString()
    render_stage = duplicate_stage(source_stage)
    config = rendering.RenderingConfig(
        strip_existing_animation=True,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=False,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        assembly_target_members={
            "/World/Assembly": ("/World/Assembly/GuideMember",),
        },
        recovered_guide_gprim_targets=("/World/GuideLeaf",),
        camera_specs={
            "prim_with_stage": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        ["/World/Assembly", "/World/GuideLeaf"],
        config,
        render_mode="prim_with_stage",
    )

    assert frames == 2
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    for path in ("/World/Assembly/GuideMember", "/World/GuideLeaf"):
        promoted = UsdGeom.Gprim(render_stage.GetPrimAtPath(path))
        assert promoted.GetDisplayOpacityAttr().GetNumTimeSamples() == 0
        assert promoted.GetDisplayOpacityAttr().Get() == [1.0]
        assert promoted.GetDisplayOpacityAttr().Get(Usd.TimeCode(0)) == [1.0]
        assert UsdGeom.Imageable(promoted).ComputePurpose() == (UsdGeom.Tokens.default_)
    assert member_opacity.GetTimeSamples() == [0.0, 10.0]
    assert leaf_opacity.GetTimeSamples() == [0.0, 10.0]
    assert source_stage.GetRootLayer().ExportToString() == source_text


def test_recovered_guide_gprim_promotion_rejects_non_gprim_and_noops() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    guide = UsdGeom.Cube.Define(stage, "/World/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    guide.CreateDisplayOpacityAttr([0.0])
    normal = UsdGeom.Cube.Define(stage, "/World/Normal")
    normal.CreateDisplayOpacityAttr([0.0])
    assembly = UsdGeom.Xform.Define(stage, "/World/Assembly")
    source_text = stage.GetRootLayer().ExportToString()

    rendering._promote_recovered_guide_gprims_for_render(
        stage,
        ["/World/Guide"],
        (),
    )
    rendering._promote_recovered_guide_gprims_for_render(
        stage,
        ["/World/Guide"],
        ("/World/StaleGuide",),
    )
    rendering._promote_recovered_guide_gprims_for_render(
        stage,
        ["/World/Normal"],
        ("/World/Normal",),
    )

    with pytest.raises(
        ValueError,
        match="Recovered guide render target is not a UsdGeom.Gprim",
    ):
        rendering._promote_recovered_guide_gprims_for_render(
            stage,
            [str(assembly.GetPath())],
            (str(assembly.GetPath()),),
        )

    assert UsdGeom.Imageable(guide.GetPrim()).ComputePurpose() == UsdGeom.Tokens.guide
    assert guide.GetDisplayOpacityAttr().Get() == [0.0]
    assert UsdGeom.Imageable(normal.GetPrim()).ComputePurpose() == (
        UsdGeom.Tokens.default_
    )
    assert normal.GetDisplayOpacityAttr().Get() == [0.0]
    assert stage.GetRootLayer().ExportToString() == source_text


def test_assembly_helpers_skip_unsupported_targets_and_non_guide_members() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    unsupported = UsdGeom.Scope.Define(stage, "/World/Unsupported").GetPrim()
    UsdGeom.Xform.Define(stage, "/World/Assembly")
    member = UsdGeom.Cube.Define(stage, "/World/Assembly/Member")
    member.CreateDisplayOpacityAttr([0.0])
    source_text = stage.GetRootLayer().ExportToString()

    assert (
        rendering._render_target_gprims(
            stage,
            unsupported,
            ("/World/Assembly/Member",),
        )
        == []
    )
    rendering._promote_assembly_target_geometry_for_render(
        stage,
        {"/World/Assembly": ("/World/Assembly/Member",)},
    )

    assert UsdGeom.Imageable(member.GetPrim()).ComputePurpose() == (
        UsdGeom.Tokens.default_
    )
    assert member.GetDisplayOpacityAttr().Get() == [0.0]
    assert stage.GetRootLayer().ExportToString() == source_text


def test_prepare_render_prims_deduplicates_overlapping_represented_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/World")
    UsdGeom.Xform.Define(source_stage, "/World/Assembly")
    source_member = UsdGeom.Cube.Define(source_stage, "/World/Assembly/Member")
    source_text = source_stage.GetRootLayer().ExportToString()
    render_stage = duplicate_stage(source_stage)
    random_components = iter((0.11, 0.12, 0.13, 0.17, 0.18, 0.19))
    monkeypatch.setattr(
        rendering.random,
        "uniform",
        lambda _minimum, _maximum: next(random_components),
    )
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=False,
        should_highlight_prim=False,
        should_assign_random_colors=True,
        bbox_purposes=("default",),
        assembly_target_members={
            "/World/Assembly": ("/World/Assembly/Member",),
        },
        camera_specs={
            "composition": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        render_stage,
        ["/World/Assembly", "/World/Assembly/Member"],
        config,
        render_mode="composition",
    )

    render_member = UsdGeom.Gprim(render_stage.GetPrimAtPath("/World/Assembly/Member"))
    assert frames == 2
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    assert tuple(render_member.GetDisplayColorAttr().Get(Usd.TimeCode(0))[0]) == (
        pytest.approx((0.11, 0.12, 0.13))
    )
    assert not source_member.GetDisplayColorAttr().HasAuthoredValueOpinion()
    assert source_stage.GetRootLayer().ExportToString() == source_text


@pytest.mark.parametrize(
    "prim_paths",
    [
        ["/World/Guide", "/World/Renderable"],
        ["/World/Renderable", "/World/Guide"],
    ],
)
def test_prepare_render_prims_isolates_late_focused_bbox_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    prim_paths: list[str],
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    guide = UsdGeom.Sphere.Define(stage, "/World/Guide")
    guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Cube.Define(stage, "/World/Renderable")
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        camera_view_type=rendering.CameraViewType.CORNER,
        camera_specs={
            "prim_only": [
                rendering.CameraSpec(
                    direction="+x+y+z",
                    view_type=rendering.CameraViewType.CORNER,
                )
            ],
        },
    )
    fallback_purposes: list[tuple[str, ...]] = []
    real_add_corner_view_camera = rendering.add_corner_view_camera

    def recording_add_corner_view_camera(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        fallback_purposes.append(tuple(kwargs["included_purposes"]))
        return real_add_corner_view_camera(*args, **kwargs)

    monkeypatch.setattr(
        rendering,
        "add_corner_view_camera",
        recording_add_corner_view_camera,
    )

    with caplog.at_level(logging.WARNING):
        _, camera_paths, frames = rendering.prepare_render_prims(
            stage,
            prim_paths,
            config,
            render_mode="prim_only",
        )

    assert frames == 2
    assert len(camera_paths) == 1
    assert "reason=focused_camera_fallback" in caplog.text
    assert "/World/Guide" in caplog.text
    assert fallback_purposes == [("default",)]


def test_prepare_render_prims_rehides_renderable_ancestor_after_child_frame() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    ancestor = UsdGeom.Cube.Define(stage, "/World/Ancestor")
    child = UsdGeom.Sphere.Define(stage, "/World/Ancestor/Child")
    return_child = UsdGeom.Cone.Define(stage, "/World/Ancestor/ReturnChild")
    later = UsdGeom.Cube.Define(stage, "/World/Later")
    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, _, frames = rendering.prepare_render_prims(
        stage,
        [
            str(child.GetPath()),
            str(later.GetPath()),
            str(return_child.GetPath()),
        ],
        config,
        render_mode="prim_only",
    )

    assert frames == 3
    time_zero = Usd.TimeCode(0)
    time_one = Usd.TimeCode(1)
    time_two = Usd.TimeCode(2)
    assert ancestor.GetVisibilityAttr().Get(time_zero) == UsdGeom.Tokens.inherited
    assert ancestor.GetVisibilityAttr().Get(time_one) == UsdGeom.Tokens.invisible
    assert ancestor.GetVisibilityAttr().Get(time_two) == UsdGeom.Tokens.inherited
    assert child.GetVisibilityAttr().Get(time_one) == UsdGeom.Tokens.invisible
    assert later.GetVisibilityAttr().Get(time_one) == UsdGeom.Tokens.inherited
    assert later.GetVisibilityAttr().Get(time_two) == UsdGeom.Tokens.invisible
    assert return_child.GetVisibilityAttr().Get(time_two) == UsdGeom.Tokens.inherited
    assert ancestor.ComputeVisibility(time_one) == UsdGeom.Tokens.invisible
    assert child.ComputeVisibility(time_zero) == UsdGeom.Tokens.inherited
    assert later.ComputeVisibility(time_one) == UsdGeom.Tokens.inherited
    assert return_child.ComputeVisibility(time_two) == UsdGeom.Tokens.inherited


def test_prepare_render_prims_deinstances_selected_gprim_proxies() -> None:
    source_layer = Sdf.Layer.CreateAnonymous("instance-source.usda")
    source_stage = Usd.Stage.Open(source_layer)
    prototype = UsdGeom.Xform.Define(source_stage, "/Prototype").GetPrim()
    UsdGeom.Cube.Define(source_stage, "/Prototype/Cube")
    source_stage.SetDefaultPrim(prototype)

    stage = Usd.Stage.CreateInMemory()
    instance_paths = ["/Selected", "/Unselected"]
    for instance_path in instance_paths:
        instance_root = UsdGeom.Xform.Define(stage, instance_path).GetPrim()
        instance_root.GetReferences().AddReference(
            source_layer.identifier,
            "/Prototype",
        )
        instance_root.SetInstanceable(True)
    selected_path = "/Selected/Cube"
    unselected_path = "/Unselected/Cube"

    assert stage.GetPrimAtPath(selected_path).IsInstanceProxy()
    assert stage.GetPrimAtPath(unselected_path).IsInstanceProxy()

    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=True,
        should_assign_random_colors=True,
        other_color_range=(0.25, 0.25),
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )

    _, camera_paths, frames = rendering.prepare_render_prims(
        stage,
        [selected_path],
        config,
        render_mode="prim_only",
    )

    assert frames == 1
    assert camera_paths == ["/Cameras/SideViewCamera_posx"]
    assert not stage.GetPrimAtPath("/Selected").IsInstance()
    assert not stage.GetPrimAtPath(selected_path).IsInstanceProxy()
    assert stage.GetPrimAtPath("/Unselected").IsInstance()
    assert stage.GetPrimAtPath(unselected_path).IsInstanceProxy()

    time_zero = Usd.TimeCode(0)
    selected = UsdGeom.Gprim(stage.GetPrimAtPath(selected_path))
    unselected = UsdGeom.Imageable(stage.GetPrimAtPath(unselected_path))
    assert selected.GetVisibilityAttr().Get(time_zero) == UsdGeom.Tokens.inherited
    assert unselected.ComputeVisibility(time_zero) == UsdGeom.Tokens.invisible
    assert tuple(selected.GetDisplayColorAttr().Get(time_zero)[0]) == pytest.approx(
        (1.0, 0.0, 0.0)
    )


def test_make_render_prims_editable_handles_nested_instances() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/CubePrototype")
    UsdGeom.Cube.Define(stage, "/CubePrototype/Cube")
    UsdGeom.Xform.Define(stage, "/NestedPrototype")
    nested_source = UsdGeom.Xform.Define(stage, "/NestedPrototype/Nested").GetPrim()
    nested_source.GetReferences().AddInternalReference("/CubePrototype")
    nested_source.SetInstanceable(True)
    outer = UsdGeom.Xform.Define(stage, "/Outer").GetPrim()
    outer.GetReferences().AddInternalReference("/NestedPrototype")
    outer.SetInstanceable(True)
    prim_path = "/Outer/Nested/Cube"

    assert stage.GetPrimAtPath(prim_path).IsInstanceProxy()
    assert stage.GetPrimAtPath("/Outer/Nested").IsInstanceProxy()

    config = rendering.RenderingConfig(
        strip_existing_animation=False,
        should_reset_materials=False,
        use_lights=True,
        should_render_prim_only=True,
        should_highlight_prim=False,
        should_assign_random_colors=False,
        camera_specs={
            "prim_only": [rendering.CameraSpec(direction="+x")],
        },
    )
    _, _, frames = rendering.prepare_render_prims(
        stage,
        [prim_path],
        config,
        render_mode="prim_only",
    )

    resolved = stage.GetPrimAtPath(prim_path)
    assert frames == 1
    assert not outer.IsInstance()
    assert not stage.GetPrimAtPath("/Outer/Nested").IsInstance()
    assert not resolved.IsInstanceProxy()
    assert resolved.IsA(UsdGeom.Gprim)
    assert (
        UsdGeom.Imageable(resolved).ComputeVisibility(Usd.TimeCode(0))
        == UsdGeom.Tokens.inherited
    )


def test_make_render_prims_editable_rejects_prototype_paths_before_mutation() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Prototype")
    UsdGeom.Cube.Define(stage, "/Prototype/Cube")
    instance = UsdGeom.Xform.Define(stage, "/Instance").GetPrim()
    instance.GetReferences().AddInternalReference("/Prototype")
    instance.SetInstanceable(True)
    prototype_cube = stage.GetPrototypes()[0].GetChildren()[0]

    with pytest.raises(ValueError, match="inside a USD prototype"):
        rendering._make_render_prims_editable(
            stage,
            [str(prototype_cube.GetPath())],
        )

    assert instance.IsInstance()


class _FakeBackend(rendering.RenderingBackend):
    def __init__(self, results: dict[str, Any] | None = None):
        self.results = results or _result(images=[Image.new("RGBA", (4, 4), "white")])
        self.calls: list[dict[str, Any]] = []

    def render(self, stage: Usd.Stage, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        copied = dict(self.results)
        copied["results"] = [
            {
                **result,
                "images": [image.copy() for image in result.get("images", [])],
            }
            for result in self.results["results"]
        ]
        return copied


class _SequenceBackend(rendering.RenderingBackend):
    def __init__(self, results: list[dict[str, Any]]):
        self.results = iter(results)

    def render(self, stage: Usd.Stage, **kwargs: Any) -> dict[str, Any]:
        return next(self.results)


def test_render_prims_and_all_prims_postprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage_with_meshes()
    images = [
        Image.new("RGBA", (4, 4), "white"),
        Image.new("RGBA", (4, 4), "black"),
    ]
    backend = _FakeBackend(_result(images=images))
    monkeypatch.setattr(
        rendering,
        "prepare_render_prims",
        lambda stage, prim_paths, config: (stage, ["/Camera"], len(prim_paths)),
    )

    config = rendering.RenderingConfig(
        use_background_color=True, background_color=(1, 2, 3)
    )
    result = rendering.render_prims(backend, stage, ["/World/A", "/World/B"], config)
    assert backend.calls[-1]["frames"] == "0:1"
    assert set(result["results"][0]["prim_to_images"]) == {"/World/A", "/World/B"}

    all_result = rendering.render_all_prims(backend, stage, None)
    assert all_result["results"][0]["prim_to_images"]

    one_image_backend = _FakeBackend(
        _result(images=[Image.new("RGBA", (4, 4), "white")])
    )
    default_result = rendering.render_prims(
        one_image_backend, stage, ["/World/A"], None
    )
    assert default_result["results"][0]["prim_to_images"]["/World/A"]


def test_hide_prims_outside_subtree() -> None:
    stage = _stage_with_meshes()
    rendering.hide_prims_outside_subtree(stage, "/Missing")
    rendering.hide_prims_outside_subtree(stage, "/World")

    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath("/Other")).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )
    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/A")).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )

    nested_stage = _stage_with_meshes()
    rendering.hide_prims_outside_subtree(nested_stage, "/World/A")
    assert (
        UsdGeom.Imageable(nested_stage.GetPrimAtPath("/World/B")).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )


def test_prepare_prims_with_composition_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage_with_meshes()
    prepare_calls: list[rendering.RenderingConfig] = []

    monkeypatch.setattr(
        "world_understanding.utils.usd.stage.duplicate_stage",
        lambda stage: stage,
    )
    monkeypatch.setattr(
        rendering,
        "prepare_render_prims",
        lambda stage, prim_paths, config, render_mode: prepare_calls.append(config)
        or (stage, ["/Camera"], len(prim_paths)),
    )

    rendering.prepare_prims_with_composition(stage, ["/World/A"], None)
    assert len(prepare_calls) == 2
    assert prepare_calls[0].highlight_color == (1.0, 0.0, 0.0)
    assert prepare_calls[0].should_render_prim_only is True

    config = rendering.RenderingConfig(
        root_prim_path="/World", composition_show_full_scene=False
    )
    rendering.prepare_prims_with_composition(stage, ["/World/A"], config)
    assert prepare_calls[-1].should_render_prim_only is True


def test_render_from_prepared_prims_url_standard_and_occlusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage_with_meshes()
    image = Image.new("RGBA", (4, 4), "white")
    monkeypatch.setattr(
        rendering,
        "_render_all_cameras_from_url_with_global_slot",
        lambda **kwargs: _result(images=[image]),
    )
    monkeypatch.setattr(
        "world_understanding.utils.image_utils.is_prim_visible_in_image",
        lambda *args, **kwargs: False,
    )
    config = rendering.RenderingConfig(
        image_width=4,
        use_background_color=True,
        should_highlight_prim=True,
        should_render_prim_only=False,
        per_mode_skip_occluded={"prim_with_stage": True},
    )
    remote_backend = rendering.RemoteRenderingBackend(api_key="key", base_url="url")
    result = rendering.render_from_prepared_prims(
        remote_backend,
        stage,
        ["/Camera"],
        1,
        ["/World/A"],
        config,
        frame_range=(2, 2),
        sensors=["depth"],
        image_height=3,
        stage_url="s3://stage",
        render_mode="prim_with_stage",
    )
    assert result["results"][0]["prim_to_images"]["/World/A"] is None
    assert result["results"][0]["prim_occlusion"]["/World/A"] is True

    backend = _FakeBackend(_result(images=[image, image]))
    standard = rendering.render_from_prepared_prims(
        backend,
        stage,
        ["/Camera"],
        2,
        ["/World/A", "/World/B"],
        rendering.RenderingConfig(image_width=4),
    )
    assert backend.calls[-1]["frames"] == "0:1"
    assert standard["results"][0]["prim_to_images"]["/World/B"].size == image.size

    monkeypatch.setattr(
        "world_understanding.utils.image_utils.is_prim_visible_in_image",
        lambda *args, **kwargs: True,
    )
    visible = rendering.render_from_prepared_prims(
        _FakeBackend(_result(images=[image])),
        stage,
        ["/Camera"],
        1,
        ["/World/A"],
        config,
        render_mode="prim_with_stage",
    )
    assert visible["results"][0]["prim_to_images"]["/World/A"] is not None
    assert visible["results"][0]["prim_occlusion"]["/World/A"] is False


def _patch_composition_image_helpers(
    monkeypatch: pytest.MonkeyPatch, visible: bool = True
) -> None:
    monkeypatch.setattr(
        rendering,
        "extract_non_black_outline",
        lambda img, **kwargs: Image.new("L", img.size, 255),
    )
    monkeypatch.setattr(
        rendering,
        "extract_red_outline",
        lambda img, **kwargs: Image.new("L", img.size, 255),
    )
    monkeypatch.setattr(
        rendering,
        "draw_bounding_box_on_red",
        lambda img, **kwargs: Image.new("L", img.size, 255),
    )
    monkeypatch.setattr(
        rendering, "paste_outline_to_image", lambda base, outline, color: base
    )
    monkeypatch.setattr(
        "world_understanding.utils.image_utils.is_prim_visible_in_image",
        lambda *args, **kwargs: visible,
    )


def test_render_from_prepared_composition_standard_url_and_occlusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage_with_meshes()
    image = Image.new("RGBA", (4, 4), "white")
    _patch_composition_image_helpers(monkeypatch, visible=True)
    config = rendering.RenderingConfig(
        image_width=4,
        use_background_color=True,
        enable_contour=True,
        contour_method="non_black",
        enable_bbox=True,
    )
    backend = _FakeBackend(_result(images=[image]))
    result = rendering.render_from_prepared_composition(
        backend,
        stage,
        ["/Camera"],
        stage,
        ["/Camera"],
        1,
        ["/World/A"],
        config,
    )
    assert backend.calls[0]["frames"] == "0"
    assert result["results"][0]["prim_to_images"]["/World/A"] is not None

    red_result = rendering.render_from_prepared_composition(
        backend,
        stage,
        ["/Camera"],
        stage,
        ["/Camera"],
        1,
        ["/World/A"],
        rendering.RenderingConfig(
            image_width=4, enable_contour=True, contour_method="red"
        ),
    )
    assert red_result["results"][0]["prim_to_images"]["/World/A"] is not None

    url_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        rendering,
        "_render_all_cameras_from_url_with_global_slot",
        lambda **kwargs: url_calls.append(kwargs) or _result(images=[image]),
    )
    remote_backend = rendering.RemoteRenderingBackend(api_key="key", base_url="url")
    rendering.render_from_prepared_composition(
        remote_backend,
        stage,
        ["/Camera"],
        stage,
        ["/Camera"],
        2,
        ["/World/A"],
        rendering.RenderingConfig(
            image_width=4, enable_contour=False, enable_bbox=False
        ),
        frame_range=(1, 2),
        highlight_url="s3://highlight",
        plain_url="s3://plain",
    )
    assert [call["usd_url"] for call in url_calls] == ["s3://highlight", "s3://plain"]

    _patch_composition_image_helpers(monkeypatch, visible=False)
    occluded = rendering.render_from_prepared_composition(
        backend,
        stage,
        ["/Camera"],
        stage,
        ["/Camera"],
        1,
        ["/World/A"],
        rendering.RenderingConfig(
            skip_occluded_images=True, enable_contour=False, enable_bbox=False
        ),
    )
    assert occluded["results"][0]["prim_to_images"]["/World/A"] is None
    assert occluded["results"][0]["prim_occlusion"]["/World/A"] is True

    _patch_composition_image_helpers(monkeypatch, visible=True)
    visible = rendering.render_from_prepared_composition(
        backend,
        stage,
        ["/Camera"],
        stage,
        ["/Camera"],
        1,
        ["/World/A"],
        rendering.RenderingConfig(
            skip_occluded_images=True, enable_contour=False, enable_bbox=False
        ),
    )
    assert visible["results"][0]["prim_to_images"]["/World/A"] is not None
    assert visible["results"][0]["prim_occlusion"]["/World/A"] is False


def test_render_prims_with_composition_and_all_prims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = _stage_with_meshes()
    image = Image.new("RGBA", (4, 4), "white")
    _patch_composition_image_helpers(monkeypatch)
    monkeypatch.setattr(
        rendering,
        "prepare_prims_with_composition",
        lambda stage, prim_paths, config: (
            (stage, ["/Camera"], len(prim_paths)),
            (stage, ["/Camera"], len(prim_paths)),
        ),
    )
    backend = _FakeBackend(_result(images=[image]))
    result = rendering.render_prims_with_composition(
        backend,
        stage,
        ["/World/A"],
        rendering.RenderingConfig(
            use_background_color=True,
            enable_contour=True,
            contour_method="non_black",
            enable_bbox=True,
        ),
    )
    assert result["results"][0]["prim_to_images"]["/World/A"] is not None

    default_result = rendering.render_prims_with_composition(
        backend,
        stage,
        ["/World/A"],
        None,
    )
    assert default_result["results"][0]["prim_to_images"]["/World/A"] is not None

    all_result = rendering.render_all_prims_with_composition(
        backend,
        stage,
        None,
    )
    assert all_result["results"][0]["prim_to_images"]

    mismatch_backend = _SequenceBackend(
        [
            {
                "results": [
                    {"camera": "/CameraA", "images": [image]},
                    {"camera": "/CameraB", "images": [image]},
                ]
            },
            {
                "results": [
                    {"camera": "/CameraA", "images": [image]},
                ]
            },
        ]
    )
    with pytest.raises(ValueError, match="number of results"):
        rendering.render_prims_with_composition(
            mismatch_backend,
            stage,
            ["/World/A"],
            rendering.RenderingConfig(),
        )
