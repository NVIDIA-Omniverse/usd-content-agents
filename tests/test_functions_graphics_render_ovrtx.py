# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OvRTX rendering backend.

Unit tests run without GPU/ovrtx. Integration tests require ovrtx + RTX GPU.
"""

import hashlib
import json
import logging
import os
import re
import stat
import subprocess
import sys
import unittest.mock
from pathlib import Path
from typing import Any

import pytest

from world_understanding.functions.graphics.render_ovrtx import (
    _DAEMON_SCRIPT,
    _NATIVE_DISPLAYCOLOR_PROBE_ENV,
    _WORKER_SCRIPT,
    DEFAULT_NUM_SENSOR_UPDATES,
    _build_render_products_usda,
    _build_visibility_frame_updates,
    _copy_exported_relative_assets,
    _ensure_lights,
    _frame_from_image_filename,
    _map_sensor_to_render_var,
    _native_displaycolor_probe_enabled,
    _OvRTXDaemon,
    _parse_frames,
    _probe_gpu_summary,
    _probe_image_metrics,
    _probe_mean_abs_rgb_diff,
    _run_sample_attribute_probe,
)


def _make_time_sampled_compliance_stage():
    """Create a small animated USD covering the OvRTX v1 time-sample contract."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(2.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    moving = UsdGeom.Cube.Define(stage, "/World/MovingCube")
    moving.GetSizeAttr().Set(0.9)
    moving_translate = UsdGeom.Xformable(moving.GetPrim()).AddTranslateOp()
    moving_color = moving.GetDisplayColorAttr()
    samples = [
        (0.0, Gf.Vec3d(-1.4, 0.0, 0.0), Gf.Vec3f(1.0, 0.05, 0.05)),
        (1.0, Gf.Vec3d(0.0, 0.0, 0.0), Gf.Vec3f(0.05, 1.0, 0.05)),
        (2.0, Gf.Vec3d(1.4, 0.0, 0.0), Gf.Vec3f(0.05, 0.05, 1.0)),
    ]
    for frame, position, color in samples:
        time_code = Usd.TimeCode(frame)
        moving_translate.Set(position, time_code)
        moving_color.Set(Vt.Vec3fArray([color]), time_code)

    toggle = UsdGeom.Cube.Define(stage, "/World/VisibilityCube")
    toggle.GetSizeAttr().Set(0.35)
    UsdGeom.Xformable(toggle.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 1.0, 0.0))
    toggle.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)]))
    visibility = UsdGeom.Imageable(toggle.GetPrim()).GetVisibilityAttr()
    visibility.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0.0))
    visibility.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(1.0))
    visibility.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(2.0))

    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(45.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    UsdGeom.Xformable(camera.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

    animated_camera = UsdGeom.Camera.Define(stage, "/AnimatedCamera")
    animated_camera.GetFocalLengthAttr().Set(45.0)
    animated_camera.GetHorizontalApertureAttr().Set(36.0)
    camera_translate = UsdGeom.Xformable(animated_camera.GetPrim()).AddTranslateOp()
    for frame, x_pos in [(0.0, -0.4), (1.0, 0.0), (2.0, 0.4)]:
        camera_translate.Set(Gf.Vec3d(x_pos, 0.0, 5.0), Usd.TimeCode(frame))

    key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(30000.0)
    key.CreateRadiusAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))

    return stage


def _make_visibility_only_stage():
    """Create a visibility-only animated USD for native visibility probes."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(2.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    positions = [-1.2, 0.0, 1.2]
    for visible_frame, x_pos in enumerate(positions):
        cube = UsdGeom.Cube.Define(stage, f"/World/Part{visible_frame}")
        cube.GetSizeAttr().Set(0.5)
        cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.8, 0.2, 0.6)]))
        UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(x_pos, 0.0, 0.0)
        )
        visibility = UsdGeom.Imageable(cube.GetPrim()).GetVisibilityAttr()
        for frame in range(3):
            token = (
                UsdGeom.Tokens.inherited
                if frame == visible_frame
                else UsdGeom.Tokens.invisible
            )
            visibility.Set(token, Usd.TimeCode(float(frame)))

    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(45.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    UsdGeom.Xformable(camera.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

    key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(30000.0)
    key.CreateRadiusAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))

    return stage


def _make_display_color_only_stage():
    """Create a stage where only primvars:displayColor is time-sampled."""
    from pxr import Gf, Usd, UsdGeom, UsdLux, Vt

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(2.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    cube = UsdGeom.Cube.Define(stage, "/World/ColorCube")
    cube.GetSizeAttr().Set(0.9)
    # Static transform: only the color is animated.
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    color_attr = cube.GetDisplayColorAttr()
    color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.05, 0.05)]), Usd.TimeCode(0.0))
    color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(0.05, 1.0, 0.05)]), Usd.TimeCode(1.0))
    color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.05, 1.0)]), Usd.TimeCode(2.0))

    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(45.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    UsdGeom.Xformable(camera.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

    key = UsdLux.SphereLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(30000.0)
    key.CreateRadiusAttr(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.0))

    return stage


def _mean_abs_rgb_diff(left, right) -> float:
    import numpy as np

    left_rgb = np.asarray(left.convert("RGB"), dtype=np.int16)
    right_rgb = np.asarray(right.convert("RGB"), dtype=np.int16)
    return float(np.abs(left_rgb - right_rgb).mean())


def _bright_centroid_x(image) -> float:
    import numpy as np

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    brightness = rgb.max(axis=2)
    threshold = max(20.0, float(brightness.max()) * 0.45)
    mask = brightness > threshold
    assert int(mask.sum()) > 16
    ys, xs = np.nonzero(mask)
    weights = brightness[ys, xs]
    return float(np.average(xs, weights=weights))


def _worker_params_from_command(cmd: list[str]) -> dict[str, Any]:
    """Extract the JSON worker payload without assuming an argv position."""
    for arg in reversed(cmd):
        if not isinstance(arg, str) or not arg.lstrip().startswith("{"):
            continue
        return json.loads(arg)
    raise AssertionError(f"OVRTX worker command has no JSON payload: {cmd!r}")


def _assert_red_green_blue_dominance(images) -> None:
    import numpy as np

    assert len(images) == 3
    frame_means = [
        np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        for image in images
    ]
    for frame_mean in frame_means:
        assert float(frame_mean.max()) > 5.0
    # Frame 0: red dominant; frame 1: green; frame 2: blue.
    assert frame_means[0][0] > max(frame_means[0][1], frame_means[0][2])
    assert frame_means[1][1] > max(frame_means[1][0], frame_means[1][2])
    assert frame_means[2][2] > max(frame_means[2][0], frame_means[2][1])


# ---------------------------------------------------------------------------
# Unit tests (no GPU required)
# ---------------------------------------------------------------------------


class TestParseFrames:
    """Test _parse_frames() for all three formats."""

    def test_single_frame(self):
        assert _parse_frames("0") == [0]
        assert _parse_frames("42") == [42]

    def test_frame_from_image_filename_prefers_encoded_frame_number(self):
        assert (
            _frame_from_image_filename(
                "cam0_f42.png",
                image_index=0,
                frame_list=[40, 41, 42],
                image_file_count=1,
            )
            == 42
        )

    def test_frame_from_image_filename_uses_index_only_for_complete_manifests(self):
        assert (
            _frame_from_image_filename(
                "legacy.png",
                image_index=1,
                frame_list=[10, 20],
                image_file_count=2,
            )
            == 20
        )
        assert (
            _frame_from_image_filename(
                "legacy.png",
                image_index=1,
                frame_list=[10, 20],
                image_file_count=1,
            )
            == 1
        )

    def test_frame_from_image_filename_rejects_negative_encoded_frame(self):
        assert (
            _frame_from_image_filename(
                "cam0_f-1.png",
                image_index=1,
                frame_list=[10, 20],
                image_file_count=2,
            )
            == 20
        )

    def test_frame_from_image_filename_accepts_extensionless_name(self):
        assert (
            _frame_from_image_filename(
                "cam0_f42",
                image_index=0,
                frame_list=[10],
                image_file_count=1,
            )
            == 42
        )

    def test_frame_range(self):
        assert _parse_frames("0:3") == [0, 1, 2, 3]
        assert _parse_frames("5:7") == [5, 6, 7]

    def test_comma_separated(self):
        assert _parse_frames("0,5,10") == [0, 5, 10]
        assert _parse_frames("10,0,5") == [0, 5, 10]  # sorted

    def test_single_frame_range(self):
        assert _parse_frames("3:3") == [3]

    def test_whitespace_handling(self):
        assert _parse_frames(" 0 ") == [0]
        assert _parse_frames("0, 5, 10") == [0, 5, 10]

    def test_invalid_frames_raises(self):
        with pytest.raises(ValueError):
            _parse_frames("not_a_number")


class TestBuildRenderProductsUsda:
    """Test _build_render_products_usda() generates valid USDA."""

    def test_single_camera(self):
        usda, paths = _build_render_products_usda(
            cameras=["/Cameras/Camera1"],
            image_width=512,
            image_height=512,
        )

        assert len(paths) == 1
        assert "/Render/" in paths[0]
        assert "#usda 1.0" in usda
        assert "RenderProduct" in usda
        assert "resolution = (512, 512)" in usda
        assert "LdrColor" in usda
        assert "rel camera = </Cameras/Camera1>" in usda
        assert "omni:rtx:pt:samplesPerPixel" not in usda
        assert "omni:rtx:rt:accumulationLimit" not in usda

    def test_multiple_cameras(self):
        cameras = ["/Cameras/Cam1", "/Cameras/Cam2", "/Cameras/Cam3"]
        usda, paths = _build_render_products_usda(
            cameras=cameras,
            image_width=1024,
            image_height=768,
        )

        assert len(paths) == 3
        assert "resolution = (1024, 768)" in usda
        for camera in cameras:
            assert f"rel camera = <{camera}>" in usda

    def test_with_depth_sensor(self):
        usda, paths = _build_render_products_usda(
            cameras=["/Camera"],
            image_width=512,
            image_height=512,
            sensors=["depth"],
        )

        assert "Depth" in usda
        assert "LdrColor" in usda
        assert len(paths) == 1

    def test_render_scope(self):
        usda, _ = _build_render_products_usda(
            cameras=["/Camera"],
            image_width=256,
            image_height=256,
        )

        assert 'def Scope "Render"' in usda

    def test_pt_samples_per_pixel_is_opt_in(self):
        usda, _ = _build_render_products_usda(
            cameras=["/Camera"],
            image_width=256,
            image_height=256,
            pt_samples_per_pixel=1,
        )

        assert "uint omni:rtx:pt:samplesPerPixel = 1" in usda
        assert "omni:rtx:rt:accumulationLimit" not in usda

    def test_rt_accumulation_limit_is_opt_in(self):
        usda, _ = _build_render_products_usda(
            cameras=["/Camera"],
            image_width=256,
            image_height=256,
            rt_accumulation_limit=512,
        )

        assert "omni:rtx:pt:samplesPerPixel" not in usda
        assert "int omni:rtx:rt:accumulationLimit = 512" in usda

    def test_probe_sample_attributes_reject_zero(self):
        with pytest.raises(ValueError, match="pt_samples_per_pixel"):
            _build_render_products_usda(
                cameras=["/Camera"],
                image_width=256,
                image_height=256,
                pt_samples_per_pixel=0,
            )

        with pytest.raises(ValueError, match="rt_accumulation_limit"):
            _build_render_products_usda(
                cameras=["/Camera"],
                image_width=256,
                image_height=256,
                rt_accumulation_limit=0,
            )

    def test_probe_sample_attributes_reject_negative(self):
        with pytest.raises(ValueError, match="pt_samples_per_pixel"):
            _build_render_products_usda(
                cameras=["/Camera"],
                image_width=256,
                image_height=256,
                pt_samples_per_pixel=-1,
            )

        with pytest.raises(ValueError, match="rt_accumulation_limit"):
            _build_render_products_usda(
                cameras=["/Camera"],
                image_width=256,
                image_height=256,
                rt_accumulation_limit=-1,
            )


class TestBuildVisibilityFrameUpdates:
    """Test visibility schedule compression before OVRTX write_attribute calls."""

    def test_skips_initial_inherited_values(self):
        updates = _build_visibility_frame_updates(
            {
                "0.0": {
                    "/World/A": "inherited",
                    "/World/B": "invisible",
                }
            },
            [0],
        )

        assert updates == {"0.0": {"/World/B": "invisible"}}

    def test_only_emits_deltas_across_frames(self):
        updates = _build_visibility_frame_updates(
            {
                "0.0": {
                    "/World/A": "inherited",
                    "/World/B": "invisible",
                    "/World/C": "invisible",
                },
                "1.0": {
                    "/World/A": "invisible",
                    "/World/B": "inherited",
                    "/World/C": "invisible",
                },
                "2.0": {
                    "/World/A": "invisible",
                    "/World/B": "invisible",
                    "/World/C": "inherited",
                },
            },
            [0, 1, 2],
        )

        assert updates == {
            "0.0": {
                "/World/B": "invisible",
                "/World/C": "invisible",
            },
            "1.0": {
                "/World/A": "invisible",
                "/World/B": "inherited",
            },
            "2.0": {
                "/World/B": "invisible",
                "/World/C": "inherited",
            },
        }

    def test_ignores_unrendered_frames(self):
        updates = _build_visibility_frame_updates(
            {
                "0.0": {"/World/A": "invisible"},
                "10.0": {"/World/A": "inherited"},
            },
            [10],
        )

        assert updates == {}

    def test_skips_rendered_frames_without_schedule_entries(self):
        updates = _build_visibility_frame_updates(
            {"0.0": {"/World/A": "invisible"}},
            [0, 1],
        )

        assert updates == {"0.0": {"/World/A": "invisible"}}


class TestOvRTXSampleAttributeProbeHelpers:
    """Test non-GPU pieces of the sample-attribute probe."""

    def test_probe_image_metrics_for_solid_rgb_image(self):
        from PIL import Image

        image = Image.new("RGB", (4, 4), (10, 20, 30))

        metrics = _probe_image_metrics(image)

        assert (
            metrics["sha256_rgb"]
            == hashlib.sha256(bytes([10, 20, 30]) * 16).hexdigest()
        )
        assert metrics["mean_rgb"] == [10.0, 20.0, 30.0]
        assert metrics["center_luma_std"] == 0.0
        assert metrics["unique_colors"] == 1

    def test_probe_mean_abs_rgb_diff(self):
        from PIL import Image

        left = Image.new("RGB", (2, 2), (0, 0, 0))
        right = Image.new("RGB", (2, 2), (10, 20, 30))

        assert _probe_mean_abs_rgb_diff(left, right) == 20.0

    def test_probe_gpu_summary_parses_nvidia_smi_output(self, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        def fake_run(cmd, **kwargs):
            assert cmd == [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True
            assert kwargs["timeout"] == 10
            assert kwargs["check"] is False
            return unittest.mock.Mock(
                returncode=0,
                stdout="NVIDIA RTX 6000 Ada Generation, 595.97, 49140 MiB\n",
                stderr="",
            )

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert (
            _probe_gpu_summary() == "NVIDIA RTX 6000 Ada Generation, 595.97, 49140 MiB"
        )

    def test_sample_attribute_probe_continues_on_ovrtx_version_drift(
        self, tmp_path, monkeypatch
    ):
        from PIL import Image

        from world_understanding.functions.graphics import render_ovrtx

        fake_python = tmp_path / "Scripts" / "python.exe"
        calls: list[dict[str, Any]] = []

        def fake_render_all_cameras(**kwargs):
            calls.append(kwargs)
            color = (10, 10, 10)
            if kwargs["num_sensor_updates"] == 4:
                color = (40, 40, 40)
            elif kwargs.get("rtx_pt_samples_per_pixel") == 1:
                color = (1, 1, 1)
            elif kwargs.get("rtx_pt_samples_per_pixel") == 8:
                color = (8, 8, 8)
            elif kwargs.get("rtx_rt_accumulation_limit") is not None:
                color = (3, 3, 3)

            return {
                "successful_cameras": 1,
                "results": [{"images": [Image.new("RGB", (8, 8), color)]}],
            }

        monkeypatch.setattr(
            render_ovrtx,
            "_get_ovrtx_python",
            lambda venv_dir=None: str(fake_python),
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python_path, venv_dir: "0.3.0.312916",
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_make_sample_attribute_probe_stage",
            lambda: object(),
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_gpu_summary",
            lambda: "fake gpu",
        )
        monkeypatch.setattr(render_ovrtx, "render_all_cameras", fake_render_all_cameras)

        result = _run_sample_attribute_probe(
            image_size=8,
            low_value=1,
            high_value=8,
            baseline_updates=4,
            ovrtx_venv_dir=tmp_path,
        )

        assert result["ovrtx_version"] == "0.3.0.312916"
        assert "expected 0.3.0.312915" in result["ovrtx_version_warning"]
        assert result["gpu"] == "fake gpu"
        assert len(result["variants"]) == 6
        assert len(calls) == 6
        assert (
            result["comparisons"]["num_sensor_updates_baseline"]["mean_abs_rgb_diff"]
            > 0.0
        )
        assert (
            result["comparisons"]["pt_samples_per_pixel_low_vs_high"][
                "mean_abs_rgb_diff"
            ]
            > 0.0
        )
        assert (
            result["comparisons"]["rt_accumulation_limit_low_vs_high"][
                "mean_abs_rgb_diff"
            ]
            == 0.0
        )

    def test_sample_attribute_probe_rejects_failed_variant(
        self, tmp_path, monkeypatch, caplog
    ):
        from world_understanding.functions.graphics import render_ovrtx

        fake_python = tmp_path / "Scripts" / "python.exe"

        def fake_render_all_cameras(**kwargs):
            return {
                "successful_cameras": 0,
                "failed_cameras": 1,
                "results": [],
            }

        monkeypatch.setattr(
            render_ovrtx,
            "_get_ovrtx_python",
            lambda venv_dir=None: str(fake_python),
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python_path, venv_dir: None,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_make_sample_attribute_probe_stage",
            lambda: object(),
        )
        monkeypatch.setattr(render_ovrtx, "render_all_cameras", fake_render_all_cameras)

        with (
            caplog.at_level("WARNING"),
            pytest.raises(
                RuntimeError,
                match="Probe variant baseline_steps_1 did not render successfully",
            ),
        ):
            _run_sample_attribute_probe(
                image_size=8,
                low_value=1,
                high_value=8,
                baseline_updates=4,
                ovrtx_venv_dir=tmp_path,
            )

        assert "could not confirm the installed version" in caplog.text
        assert "found None" not in caplog.text


class TestOvRTXCoverageEdges:
    """Focused non-GPU branches in the OVRTX renderer module."""

    def test_path_and_environment_helpers_handle_edge_cases(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        file_entry = tmp_path / "not-a-dir"
        file_entry.write_text("skip me", encoding="utf-8")
        directory_entry = tmp_path / "bin"
        directory_entry.mkdir()

        path_value = os.pathsep.join(["", str(file_entry), str(directory_entry)])

        assert str(render_ovrtx._local_asset_path("file://server/share/tex.png")) == (
            "//server/share/tex.png"
        )
        assert render_ovrtx._sanitize_ovrtx_path_env(path_value).split(os.pathsep) == [
            "",
            str(directory_entry),
        ]
        assert render_ovrtx._ovrtx_runtime_lock_path(tmp_path) == (
            render_ovrtx._ovrtx_provision_lock_path(tmp_path)
        )

        monkeypatch.setenv("WU_OVRTX_EXPERIMENTAL_NATIVE_VISIBILITY", " YES ")
        assert render_ovrtx._native_visibility_probe_enabled()

    def test_render_product_builder_skips_unknown_sensors(self, caplog):
        with caplog.at_level(logging.WARNING):
            usda, paths = _build_render_products_usda(
                cameras=["/Camera"],
                image_width=8,
                image_height=8,
                sensors=["mystery", "depth"],
            )

        assert paths == ["/Render/Product_Camera"]
        assert "Unknown sensor 'mystery'" in caplog.text
        assert 'RenderVar "Depth"' in usda
        assert "mystery" not in usda

    def test_marker_and_probe_stdout_helpers(self, tmp_path):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()

        render_ovrtx._try_refresh_ovrtx_managed_marker(
            venv_dir, render_ovrtx._ovrtx_runtime_lock_digest()
        )

        marker = venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
        assert marker.exists()
        marker_fields = render_ovrtx._read_ovrtx_managed_marker(marker)
        assert marker_fields == {
            "ovrtx_version": render_ovrtx._OVRTX_VERSION,
            "runtime_lock_sha256": render_ovrtx._ovrtx_runtime_lock_digest(),
        }
        assert len(marker_fields["runtime_lock_sha256"]) == 64
        assert not (venv_dir / render_ovrtx._OVRTX_PROVISIONING_MARKER).exists()
        assert (
            render_ovrtx._parse_ovrtx_probe_stdout("noise\nWU_OVRTX_VERSION=0.3.0\n")
            == "0.3.0"
        )

    def test_probe_ovrtx_version_reads_version_file(self, tmp_path, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        python_path = tmp_path / "python"

        def fake_run(cmd, **kwargs):
            version_path = Path(cmd[-1])
            version_path.write_text("noise\nWU_OVRTX_VERSION=9.8.7\n", encoding="utf-8")
            return unittest.mock.Mock(returncode=0, stdout="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._probe_ovrtx_version(python_path, tmp_path) == "9.8.7"

    def test_probe_existing_python_before_lock_fast_paths(self, tmp_path, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)

        remembered: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            render_ovrtx,
            "_remember_verified_ovrtx_python",
            lambda key, python: remembered.append((key, python)),
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda python, venv: True,
        )

        assert render_ovrtx._probe_existing_ovrtx_python_before_lock(
            python_path, venv_dir, cache_key
        ) == str(python_path)
        assert remembered == [(cache_key, str(python_path))]

        remembered.clear()
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda python, venv: False,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_is_managed_ovrtx_runtime_dir",
            lambda venv: False,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_ovrtx_import_probe_succeeds",
            lambda python, venv: True,
        )

        assert render_ovrtx._probe_existing_ovrtx_python_before_lock(
            python_path, venv_dir, cache_key
        ) == str(python_path)
        assert remembered == [(cache_key, str(python_path))]

        missing_python = venv_dir / "bin" / "missing-python"
        assert (
            render_ovrtx._probe_existing_ovrtx_python_before_lock(
                missing_python, venv_dir, cache_key
            )
            is None
        )

    @pytest.mark.parametrize(
        "exc",
        [
            subprocess.TimeoutExpired("probe", timeout=30),
            OSError("cannot launch"),
        ],
    )
    def test_probe_existing_python_before_lock_handles_probe_errors(
        self, tmp_path, monkeypatch, exc
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda python, venv: False,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_is_managed_ovrtx_runtime_dir",
            lambda venv: False,
        )

        def raise_probe(python_path_arg, venv_dir_arg):
            raise exc

        monkeypatch.setattr(render_ovrtx, "_ovrtx_import_probe_succeeds", raise_probe)

        assert (
            render_ovrtx._probe_existing_ovrtx_python_before_lock(
                python_path,
                venv_dir,
                render_ovrtx._ovrtx_runtime_cache_key(venv_dir),
            )
            is None
        )

    def test_cached_python_match_helper_clears_stale_state(self, tmp_path, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        missing_python = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        assert not render_ovrtx._cached_ovrtx_python_matches(missing_python, venv_dir)

        missing_python.parent.mkdir(parents=True)
        missing_python.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python, venv: render_ovrtx._OVRTX_VERSION,
        )
        assert render_ovrtx._cached_ovrtx_python_matches(missing_python, venv_dir)
        assert render_ovrtx._ovrtx_python == str(missing_python)

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", "stale")

        def raise_probe(python, venv):
            raise OSError("broken probe")

        monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", raise_probe)
        assert not render_ovrtx._cached_ovrtx_python_matches(missing_python, venv_dir)
        assert render_ovrtx._ovrtx_python is None

    def test_import_probe_uses_direct_import_and_site_package_fallback(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        site_dir = render_ovrtx._ovrtx_target_fallback_site_dir(venv_dir)
        (site_dir / "ovrtx").mkdir(parents=True)

        calls: list[list[str]] = []

        def fake_run_success(cmd, **kwargs):
            calls.append(cmd)
            return unittest.mock.Mock(returncode=0)

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run_success)
        assert render_ovrtx._ovrtx_import_probe_succeeds(python_path, venv_dir)
        assert calls == [[str(python_path), "-c", "import ovrtx"]]

        calls.clear()
        returns = iter([1, 0])

        def fake_run_fallback(cmd, **kwargs):
            calls.append(cmd)
            return unittest.mock.Mock(returncode=next(returns))

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run_fallback)
        assert render_ovrtx._ovrtx_import_probe_succeeds(python_path, venv_dir)
        assert len(calls) == 2
        assert str(site_dir) in calls[1][2]

    def test_unlocked_python_uses_cached_and_existing_runtime_paths(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {cache_key: "cached"})
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda python, venv: python == "cached",
        )
        remembered: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            render_ovrtx,
            "_remember_verified_ovrtx_python",
            lambda key, python: remembered.append((key, python)),
        )

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == "cached"
        assert remembered == [(cache_key, "cached")]

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(python_path))
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda python, venv: python == str(python_path),
        )
        remembered.clear()

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == str(python_path)
        assert remembered == [(cache_key, str(python_path))]

    def test_unlocked_python_uses_global_cached_runtime_after_probe_check(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
        readiness_calls = 0

        def fake_ready(python, venv):
            nonlocal readiness_calls
            readiness_calls += 1
            return readiness_calls == 2 and python == str(python_path)

        remembered: list[tuple[Path, str]] = []
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(python_path))
        monkeypatch.setattr(render_ovrtx, "_cached_ovrtx_python_ready", fake_ready)
        monkeypatch.setattr(
            render_ovrtx,
            "_remember_verified_ovrtx_python",
            lambda key, python: remembered.append((key, python)),
        )

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == str(python_path)
        assert remembered == [(cache_key, str(python_path))]

    def test_unlocked_python_rejects_missing_runtime_when_auto_provision_disabled(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")

        with pytest.raises(RuntimeError, match="AUTO_PROVISION is disabled"):
            render_ovrtx._get_ovrtx_python_unlocked(tmp_path / "missing_venv")

    def test_unlocked_python_discovers_uv_next_to_current_python(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        fake_uv = Path(render_ovrtx.sys.executable).with_name("uv")
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        (site_dir / "ovrtx" / "bin" / "library").mkdir(parents=True)

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: None)
        real_exists = os.path.exists
        monkeypatch.setattr(
            render_ovrtx.os.path,
            "exists",
            lambda path: str(path) == str(fake_uv) or real_exists(path),
        )

        def fake_run_checked(cmd, label):
            if label == "uv venv creation":
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python, venv: render_ovrtx._OVRTX_VERSION,
        )

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == str(python_path)
        assert (site_dir / "library").is_symlink()

    def test_unlocked_python_removes_partial_uv_runtime_on_lock_install_failure(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        removed: list[Path] = []

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")

        def fake_run_checked(cmd, label):
            if label == "uv venv creation":
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("", encoding="utf-8")
            if label == "locked OVRTX runtime install":
                raise RuntimeError("lock install failed")

        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_venv",
            lambda path: removed.append(Path(path)),
        )

        with pytest.raises(RuntimeError, match="lock install failed"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert removed == [venv_dir]

    def test_unlocked_python_reports_failed_installed_import_probe(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")

        def fake_run_checked(cmd, label):
            if label == "uv venv creation":
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(
            render_ovrtx, "_probe_ovrtx_version", lambda python, venv: None
        )

        with pytest.raises(RuntimeError, match="Installed ovrtx import probe failed"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

    def test_unlocked_python_removes_wrong_version_after_install(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        removed: list[Path] = []

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")

        def fake_run_checked(cmd, label):
            if label == "uv venv creation":
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(
            render_ovrtx, "_probe_ovrtx_version", lambda python, venv: "wrong"
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_venv",
            lambda path: removed.append(Path(path)),
        )

        with pytest.raises(RuntimeError, match="does not match expected"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert removed == [venv_dir]

    def test_remove_bundled_python_libraries_ignores_missing_files(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        missing = tmp_path / "libpython3.12.so"
        scans = iter([[missing], []])
        monkeypatch.setattr(
            render_ovrtx,
            "_ovrtx_bundled_python_libraries",
            lambda venv_dir: next(scans),
        )

        assert render_ovrtx._remove_ovrtx_bundled_python_libraries(tmp_path) == []

    def test_hdri_intensity_and_portable_asset_edges(
        self, tmp_path, monkeypatch, caplog
    ):
        from pxr import Usd

        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
        assert (
            render_ovrtx._resolve_default_hdri_intensity()
            == render_ovrtx._DEFAULT_HDRI_INTENSITY
        )

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", "not-a-float")
        with caplog.at_level(logging.WARNING):
            assert (
                render_ovrtx._resolve_default_hdri_intensity()
                == render_ovrtx._DEFAULT_HDRI_INTENSITY
            )
        assert "Invalid WU_OVRTX_DEFAULT_HDRI_INTENSITY" in caplog.text

        stage = Usd.Stage.CreateInMemory()
        assert (
            render_ovrtx._portable_stage_hdri_asset(
                stage, "https://example.invalid/studio.exr"
            )
            == "https://example.invalid/studio.exr"
        )
        assert render_ovrtx._portable_stage_hdri_asset(stage, "relative.exr") == (
            "relative.exr"
        )
        assert render_ovrtx._portable_stage_hdri_asset(
            stage, str(tmp_path / "missing.exr")
        ) == str(tmp_path / "missing.exr")

    def test_probe_stage_gpu_failure_and_cli_probe_path(
        self, monkeypatch, capsys: pytest.CaptureFixture[str]
    ):
        from world_understanding.functions.graphics import render_ovrtx

        stage = render_ovrtx._make_sample_attribute_probe_stage()
        assert str(stage.GetDefaultPrim().GetPath()) == "/World"
        assert stage.GetPrimAtPath("/Camera").IsValid()

        monkeypatch.setattr(
            render_ovrtx.subprocess,
            "run",
            lambda *args, **kwargs: unittest.mock.Mock(
                returncode=1,
                stdout="",
                stderr="driver unavailable",
            ),
        )
        assert render_ovrtx._probe_gpu_summary() == (
            "nvidia-smi failed: driver unavailable"
        )

        monkeypatch.setattr(
            render_ovrtx,
            "_run_sample_attribute_probe",
            lambda **kwargs: {"ok": True, "image_size": kwargs["image_size"]},
        )
        assert (
            render_ovrtx._main(
                [
                    "--probe-sample-attributes",
                    "--probe-image-size",
                    "12",
                    "--probe-low-value",
                    "2",
                    "--probe-high-value",
                    "4",
                    "--probe-baseline-updates",
                    "3",
                    "--log-level",
                    "debug",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out) == {"ok": True, "image_size": 12}

    def test_copy_exported_assets_skips_invalid_entries_and_updates_by_resolved_path(
        self, tmp_path, monkeypatch
    ):
        from pxr import Sdf, Usd, UsdShade

        from world_understanding.functions.graphics import render_ovrtx

        texture = tmp_path / "source" / "albedo.png"
        texture.parent.mkdir()
        texture.write_bytes(b"png")

        stage = Usd.Stage.CreateInMemory()
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Tex")
        shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(str(texture)))
        exported_shader = UsdShade.Shader.Define(stage, "/World/Looks/Exported")
        exported_shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(str(texture)))

        def fake_texture_assets(stage_arg, base_dir=None):
            return [
                {"is_local": False, "resolved_path": str(texture)},
                {"is_local": True, "resolved_path": ""},
                {
                    "is_local": True,
                    "resolved_path": str(texture),
                    "file_path": "https://example.invalid/albedo.png",
                },
                {
                    "is_local": True,
                    "resolved_path": str(tmp_path / "missing.png"),
                    "file_path": "missing.png",
                },
                {
                    "prim_path": "/World/Looks/Tex",
                    "attr_name": "inputs:file",
                    "file_path": str(texture),
                    "resolved_path": str(texture),
                    "is_local": True,
                },
            ]

        import world_understanding.utils.usd.material as usd_material

        monkeypatch.setattr(
            usd_material,
            "get_local_texture_file_assets",
            fake_texture_assets,
        )

        export_dir = tmp_path / "render"
        export_dir.mkdir()
        exported_stage_path = export_dir / "stage.usda"
        stage.GetRootLayer().Export(str(exported_stage_path))

        assert (
            render_ovrtx._copy_exported_relative_assets(
                stage,
                export_dir,
                exported_stage_path=exported_stage_path,
            )
            == 1
        )

        exported = Sdf.Layer.FindOrOpen(str(exported_stage_path))
        copied_path = (
            exported.GetPrimAtPath("/World/Looks/Tex")
            .attributes["inputs:file"]
            .default.path
        )
        fallback_path = (
            exported.GetPrimAtPath("/World/Looks/Exported")
            .attributes["inputs:file"]
            .default.path
        )

        assert copied_path == fallback_path
        assert (export_dir / copied_path).read_bytes() == b"png"

    def test_copy_exported_assets_resolves_relative_fallbacks_and_skips_attr_edges(
        self, tmp_path, monkeypatch
    ):
        from pxr import Sdf, Usd, UsdShade

        from world_understanding.functions.graphics import render_ovrtx

        source_dir = tmp_path / "source"
        texture = source_dir / "textures" / "albedo.png"
        texture.parent.mkdir(parents=True)
        texture.write_bytes(b"png")

        stage = Usd.Stage.CreateInMemory()
        tracked = UsdShade.Shader.Define(stage, "/World/Looks/Tracked")
        tracked.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/albedo.png"))
        fallback = UsdShade.Shader.Define(stage, "/World/Looks/Fallback")
        fallback.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/albedo.png"))
        fallback.GetPrim().CreateAttribute(
            "inputs:remote",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("https://example.invalid/albedo.png"))
        fallback.GetPrim().CreateAttribute(
            "inputs:scale",
            Sdf.ValueTypeNames.Float,
        ).Set(1.0)
        fallback.GetPrim().CreateAttribute(
            "inputs:bad",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("bad.png"))

        def fake_texture_assets(stage_arg, base_dir=None):
            return [
                {
                    "prim_path": "/World/Looks/Tracked",
                    "attr_name": "inputs:file",
                    "file_path": "textures/albedo.png",
                    "resolved_path": str(texture),
                    "is_local": True,
                }
            ]

        class BadLocalPath:
            def is_absolute(self):
                return True

            def resolve(self):
                raise OSError("cannot resolve")

        import world_understanding.utils.usd.material as usd_material

        original_local_asset_path = render_ovrtx._local_asset_path
        monkeypatch.setattr(
            usd_material,
            "get_local_texture_file_assets",
            fake_texture_assets,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_local_asset_path",
            lambda value: BadLocalPath()
            if value == "bad.png"
            else original_local_asset_path(value),
        )

        export_dir = tmp_path / "render"
        export_dir.mkdir()
        exported_stage_path = export_dir / "stage.usda"
        stage.GetRootLayer().Export(str(exported_stage_path))

        assert (
            render_ovrtx._copy_exported_relative_assets(
                stage,
                export_dir,
                base_dir=source_dir,
                exported_stage_path=exported_stage_path,
            )
            == 1
        )

        exported = Sdf.Layer.FindOrOpen(str(exported_stage_path))
        tracked_path = (
            exported.GetPrimAtPath("/World/Looks/Tracked")
            .attributes["inputs:file"]
            .default.path
        )
        fallback_path = (
            exported.GetPrimAtPath("/World/Looks/Fallback")
            .attributes["inputs:file"]
            .default.path
        )
        remote_path = (
            exported.GetPrimAtPath("/World/Looks/Fallback")
            .attributes["inputs:remote"]
            .default.path
        )
        bad_path = (
            exported.GetPrimAtPath("/World/Looks/Fallback")
            .attributes["inputs:bad"]
            .default.path
        )

        assert tracked_path == "textures/albedo.png"
        assert fallback_path == "textures/albedo.png"
        assert remote_path == "https://example.invalid/albedo.png"
        assert bad_path == "bad.png"

    def test_copy_exported_assets_returns_after_missing_export_layer(
        self, tmp_path, monkeypatch
    ):
        from pxr import Usd

        from world_understanding.functions.graphics import render_ovrtx

        texture = tmp_path / "texture.png"
        texture.write_bytes(b"png")
        stage = Usd.Stage.CreateInMemory()

        def fake_texture_assets(stage_arg, base_dir=None):
            return [
                {
                    "prim_path": "/World/Looks/Tex",
                    "attr_name": "inputs:file",
                    "file_path": "texture.png",
                    "resolved_path": str(texture),
                    "is_local": True,
                }
            ]

        import world_understanding.utils.usd.material as usd_material

        monkeypatch.setattr(
            usd_material,
            "get_local_texture_file_assets",
            fake_texture_assets,
        )

        export_dir = tmp_path / "render"
        export_dir.mkdir()

        assert (
            render_ovrtx._copy_exported_relative_assets(
                stage,
                export_dir,
                base_dir=tmp_path,
                exported_stage_path=tmp_path / "missing.usda",
            )
            == 1
        )

    def test_daemon_start_reports_empty_ready_line_exit(self, tmp_path, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeProcess:
            pid = 77
            stdin = None
            stdout = None
            stderr: list[str] = []

            def poll(self):
                return 17

            def wait(self, timeout=None):
                return 17

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        monkeypatch.setattr(
            render_ovrtx.subprocess,
            "Popen",
            lambda *args, **kwargs: FakeProcess(),
        )
        monkeypatch.setattr(
            render_ovrtx._OvRTXDaemon,
            "_read_stdout_line",
            lambda self, timeout_s, phase: "",
        )

        daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )

        with pytest.raises(RuntimeError, match="exit code 17"):
            daemon.ensure_running()
        assert daemon._process is None

    @pytest.mark.parametrize(
        ("ready_line", "error_match"),
        [
            ("not-json", "invalid startup JSON"),
            (json.dumps({"status": "starting"}), "unexpected init msg"),
        ],
    )
    def test_daemon_start_kills_process_after_invalid_ready_protocol(
        self, tmp_path, monkeypatch, ready_line, error_match
    ):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeProcess:
            pid = 77
            stdin = None
            stdout = None
            stderr: list[str] = []

            def __init__(self) -> None:
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                return 0

        process = FakeProcess()
        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        monkeypatch.setattr(
            render_ovrtx.subprocess,
            "Popen",
            lambda *args, **kwargs: process,
        )
        monkeypatch.setattr(
            render_ovrtx._OvRTXDaemon,
            "_read_stdout_line",
            lambda self, timeout_s, phase: ready_line,
        )

        daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )

        with pytest.raises(RuntimeError, match=error_match):
            daemon.ensure_running()
        assert process.killed
        assert daemon._process is None

    def test_daemon_running_and_stderr_and_render_failure_edges(
        self, tmp_path, monkeypatch, caplog
    ):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeStdin:
            def write(self, data: str) -> None:
                return None

            def flush(self) -> None:
                return None

        class FakeProcess:
            pid = 88
            stdin = FakeStdin()
            stdout = object()
            stderr = ["native warning\n"]

            def __init__(self) -> None:
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )
        daemon._process = FakeProcess()
        monkeypatch.setattr(
            daemon,
            "_start",
            unittest.mock.Mock(side_effect=AssertionError("already running")),
        )
        daemon.ensure_running()

        with caplog.at_level(logging.DEBUG, logger=render_ovrtx.__name__):
            daemon._drain_stderr()
        assert "native warning" in caplog.text

        monkeypatch.setattr(daemon, "_read_stdout_line", lambda timeout, phase: "")
        with pytest.raises(RuntimeError, match="died during render"):
            daemon.render(
                {
                    "cameras": ["/Camera"],
                    "usd_path": "/stage.usda",
                    "fps": 24.0,
                    "frames": [0],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )
        assert daemon._process is None

    def test_daemon_stdout_zero_timeout_and_shutdown_edges(self, tmp_path, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeStdout:
            def __init__(self) -> None:
                self.lines = iter(["tail\n", "next\n"])

            def readline(self) -> str:
                return next(self.lines)

        class FakeStdin:
            def __init__(self, *, broken: bool = False) -> None:
                self.broken = broken

            def write(self, data: str) -> None:
                if self.broken:
                    raise BrokenPipeError("closed")

            def flush(self) -> None:
                return None

        class FakeProcess:
            def __init__(self, *, stdin=None) -> None:
                self.stdin = stdin or FakeStdin()
                self.stdout = FakeStdout()
                self.stderr = []
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )

        daemon._process = FakeProcess()
        daemon._stdout_buffer = b"partial"
        assert daemon._read_stdout_line(0, "render") == "partialtail\n"
        assert daemon._read_stdout_line(0, "render") == "next\n"

        daemon._process = None
        daemon._kill_process()
        assert daemon._process is None

        broken_proc = FakeProcess(stdin=FakeStdin(broken=True))
        daemon._process = broken_proc
        daemon.shutdown()
        assert broken_proc.killed
        assert daemon._process is None

        class SlowProcess(FakeProcess):
            def wait(self, timeout=None):
                if not self.killed:
                    raise subprocess.TimeoutExpired("wait", timeout)
                return 0

        slow_proc = SlowProcess()
        daemon._process = slow_proc
        daemon.shutdown()
        assert slow_proc.killed
        assert daemon._process is None

    def test_daemon_read_timeout_eof_and_kill_exception_edges(
        self, tmp_path, monkeypatch, caplog
    ):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeStdout:
            def fileno(self):
                return 123

        class FakeProcess:
            stdout = FakeStdout()
            stderr = []

            def __init__(self) -> None:
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                return 0

        class NoEventSelector:
            def register(self, fd, event):
                return None

            def select(self, remaining):
                return []

            def close(self):
                return None

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )
        daemon._process = FakeProcess()
        monkeypatch.setattr(render_ovrtx.selectors, "DefaultSelector", NoEventSelector)

        with pytest.raises(TimeoutError):
            daemon._read_stdout_line(0.001, "render")
        assert daemon._process is None

        class EofSelector(NoEventSelector):
            def select(self, remaining):
                return [object()]

        daemon._process = FakeProcess()
        daemon._stdout_buffer = b"partial"
        monkeypatch.setattr(render_ovrtx.selectors, "DefaultSelector", EofSelector)
        monkeypatch.setattr(render_ovrtx.os, "read", lambda fd, size: b"")
        assert daemon._read_stdout_line(1.0, "render") == "partial"

        class BadKillProcess(FakeProcess):
            def kill(self):
                raise RuntimeError("no kill")

        daemon._process = BadKillProcess()
        with caplog.at_level(logging.ERROR):
            daemon._kill_process()
        assert "Failed to kill OvRTX daemon subprocess" in caplog.text
        assert daemon._process is None

    def test_daemon_read_timeout_breaks_when_deadline_has_passed(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeStdout:
            def fileno(self):
                return 456

        class FakeProcess:
            stdout = FakeStdout()
            stderr = []

            def poll(self):
                return None

            def kill(self):
                return None

            def wait(self, timeout=None):
                return 0

        class DeadlineSelector:
            def register(self, fd, event):
                return None

            def select(self, remaining):
                raise AssertionError("deadline branch should break before select")

            def close(self):
                return None

        times = iter([100.0, 100.2])

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        monkeypatch.setattr(render_ovrtx.selectors, "DefaultSelector", DeadlineSelector)
        monkeypatch.setattr(render_ovrtx.time, "monotonic", lambda: next(times))

        daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )
        daemon._process = FakeProcess()

        with pytest.raises(TimeoutError):
            daemon._read_stdout_line(0.1, "render")

        assert daemon._process is None

    def test_resource_path_falls_back_through_string_conversion(
        self, tmp_path, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        class FakeResource:
            def joinpath(self, *parts):
                return self

            def __str__(self) -> str:
                return str(tmp_path / "resource.dat")

        monkeypatch.setattr(
            render_ovrtx.importlib_resources,
            "files",
            lambda package: FakeResource(),
        )

        assert render_ovrtx._world_understanding_resource_path("data") == (
            tmp_path / "resource.dat"
        )

    def test_render_all_cameras_loads_sensors_failures_and_dump(
        self, tmp_path, monkeypatch
    ):
        import numpy as np
        from PIL import Image
        from pxr import Usd, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        stage = Usd.Stage.CreateInMemory()
        UsdLux.DomeLight.Define(stage, "/World/Light")
        dump_path = tmp_path / "dump" / "portable.usda"
        fake_daemon = unittest.mock.Mock()

        def fake_render(params):
            output_dir = Path(params["output_dir"])
            Image.new("RGB", (4, 4), (1, 2, 3)).save(output_dir / "cam0_f5.png")
            np.save(
                output_dir / "cam0_f5_depth.npy", np.array([[5.0]], dtype=np.float32)
            )
            return [
                {
                    "camera": "/Camera",
                    "image_files": ["cam0_f5.png"],
                    "sensor_files": {"depth": {"5": "cam0_f5_depth.npy"}},
                    "frame_count": 1,
                },
                {
                    "camera": "/Empty",
                    "image_files": [],
                    "sensor_files": {"depth": {"5": "missing.npy"}},
                    "frame_count": 0,
                },
            ]

        fake_daemon.render.side_effect = fake_render
        monkeypatch.setenv("WU_OVRTX_DUMP_COMBINED", str(dump_path))
        monkeypatch.setattr(
            render_ovrtx, "_get_ovrtx_python", lambda venv_dir=None: "/fake/python"
        )

        result = render_ovrtx.render_all_cameras(
            stage=stage,
            image_width=4,
            image_height=4,
            cameras=["/Camera", "/Empty"],
            frames="5",
            sensors=["depth"],
            num_sensor_updates=1,
            daemon=fake_daemon,
        )

        assert result["successful_cameras"] == 1
        assert result["failed_cameras"] == 1
        assert result["results"][0]["sensors"]["depth"][5].tolist() == [[5.0]]
        assert result["results"][1]["error"] == "No images produced"
        assert dump_path.exists()
        assert not (dump_path.parent / "combined.usda").exists()

    def test_render_all_cameras_restores_instance_visibility_and_overlay_sublayers(
        self, tmp_path, monkeypatch
    ):
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        stage = Usd.Stage.CreateInMemory()
        proto = UsdGeom.Xform.Define(stage, "/Proto")
        UsdGeom.Cube.Define(stage, "/Proto/Child")
        instance = UsdGeom.Xform.Define(stage, "/World/Inst")
        instance.GetPrim().GetReferences().AddInternalReference(str(proto.GetPath()))
        instance.GetPrim().SetInstanceable(True)
        visibility = UsdGeom.Imageable(instance.GetPrim()).GetVisibilityAttr()
        visibility.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0.0))
        visibility.Set(UsdGeom.Tokens.inherited, Usd.TimeCode(1.0))

        captured: dict[str, Any] = {}
        fake_daemon = unittest.mock.Mock()

        def fake_preview_overlay(stage_arg, output_path):
            Path(output_path).write_text("#usda 1.0\n", encoding="utf-8")
            return 1

        def fake_render(params):
            captured["params"] = params
            from pxr import Sdf

            frame_layer = Sdf.Layer.FindOrOpen(params["frame_usd_paths"]["0"])
            captured["frame_sublayers"] = list(frame_layer.subLayerPaths)

            # Simulate a caller-side mutation before cleanup to exercise the
            # restoration branch that sees the stripped prim as an instance again.
            stage.GetPrimAtPath("/World/Inst").SetInstanceable(True)
            UsdLux.DomeLight.Define(stage, "/OvRTXDefaultLights/Dome")

            output_dir = Path(params["output_dir"])
            Image.new("RGB", (4, 4), (10, 20, 30)).save(output_dir / "cam0_f0.png")
            return [
                {
                    "camera": "/Camera",
                    "image_files": ["cam0_f0.png"],
                    "sensor_files": {},
                    "frame_count": 1,
                }
            ]

        import world_understanding.utils.usd.material as usd_material

        fake_daemon.render.side_effect = fake_render
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "https://example.invalid/env.hdr")
        monkeypatch.setattr(
            render_ovrtx, "_get_ovrtx_python", lambda venv_dir=None: "/fake/python"
        )
        monkeypatch.setattr(
            usd_material,
            "write_ovrtx_preview_fallback_overlay_for_materialx_openpbr",
            fake_preview_overlay,
        )

        result = render_ovrtx.render_all_cameras(
            stage=stage,
            image_width=4,
            image_height=4,
            cameras=["/Camera"],
            frames="0:1",
            num_sensor_updates=1,
            daemon=fake_daemon,
            material_target="preview_surface",
        )

        assert result["successful_cameras"] == 1
        assert any(
            "ovrtx_material_fallbacks.usda" in p for p in captured["frame_sublayers"]
        )
        assert any("default_lights.usda" in p for p in captured["frame_sublayers"])
        assert not stage.GetPrimAtPath("/OvRTXDefaultLights").IsValid()
        assert stage.GetPrimAtPath("/World/Inst").IsInstance()
        restored_visibility = UsdGeom.Imageable(
            stage.GetPrimAtPath("/World/Inst")
        ).GetVisibilityAttr()
        assert restored_visibility.Get(Usd.TimeCode(0.0)) == UsdGeom.Tokens.invisible
        assert restored_visibility.Get(Usd.TimeCode(1.0)) == UsdGeom.Tokens.inherited

    def test_render_all_cameras_tolerates_stripped_prim_removed_before_cleanup(
        self, monkeypatch
    ):
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        stage = Usd.Stage.CreateInMemory()
        cube = UsdGeom.Cube.Define(stage, "/World/AnimatedVisibility")
        visibility = UsdGeom.Imageable(cube.GetPrim()).GetVisibilityAttr()
        visibility.Set(UsdGeom.Tokens.invisible, Usd.TimeCode(0.0))
        UsdLux.DomeLight.Define(stage, "/World/Light")
        fake_daemon = unittest.mock.Mock()

        def fake_render(params):
            stage.RemovePrim("/World/AnimatedVisibility")
            output_dir = Path(params["output_dir"])
            Image.new("RGB", (2, 2), (1, 2, 3)).save(output_dir / "cam0_f0.png")
            return [
                {
                    "camera": "/Camera",
                    "image_files": ["cam0_f0.png"],
                    "sensor_files": {},
                    "frame_count": 1,
                }
            ]

        fake_daemon.render.side_effect = fake_render
        monkeypatch.setattr(
            render_ovrtx, "_get_ovrtx_python", lambda venv_dir=None: "/fake/python"
        )

        result = render_ovrtx.render_all_cameras(
            stage=stage,
            image_width=2,
            image_height=2,
            cameras=["/Camera"],
            frames="0",
            num_sensor_updates=1,
            daemon=fake_daemon,
        )

        assert result["successful_cameras"] == 1

    def test_render_all_cameras_logs_copied_assets_and_subprocess_stdout_failure(
        self, monkeypatch
    ):
        from pxr import Usd, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        stage = Usd.Stage.CreateInMemory()
        UsdLux.DomeLight.Define(stage, "/World/Light")
        monkeypatch.setattr(
            render_ovrtx, "_get_ovrtx_python", lambda venv_dir=None: "/fake/python"
        )
        monkeypatch.setattr(
            render_ovrtx, "_copy_exported_relative_assets", lambda *args, **kwargs: 1
        )
        monkeypatch.setattr(
            render_ovrtx.subprocess,
            "run",
            lambda *args, **kwargs: unittest.mock.Mock(
                returncode=2,
                stdout="stdout details",
                stderr="",
            ),
        )

        with pytest.raises(RuntimeError, match="stdout details"):
            render_ovrtx.render_all_cameras(
                stage=stage,
                image_width=2,
                image_height=2,
                cameras=["/Camera"],
                frames="0",
                num_sensor_updates=1,
                daemon=None,
            )

    def test_render_all_cameras_cleanup_ignores_rmtree_error(
        self, tmp_path, monkeypatch
    ):
        from PIL import Image
        from pxr import Usd, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        stage = Usd.Stage.CreateInMemory()
        UsdLux.DomeLight.Define(stage, "/World/Light")
        fake_daemon = unittest.mock.Mock()

        def fake_render(params):
            output_dir = Path(params["output_dir"])
            Image.new("RGB", (2, 2), (1, 2, 3)).save(output_dir / "cam0_f0.png")
            return [
                {
                    "camera": "/Camera",
                    "image_files": ["cam0_f0.png"],
                    "sensor_files": {},
                    "frame_count": 1,
                }
            ]

        fake_daemon.render.side_effect = fake_render
        monkeypatch.setattr(
            render_ovrtx, "_get_ovrtx_python", lambda venv_dir=None: "/fake/python"
        )
        monkeypatch.setattr(
            render_ovrtx.shutil,
            "rmtree",
            unittest.mock.Mock(side_effect=OSError("busy")),
        )

        result = render_ovrtx.render_all_cameras(
            stage=stage,
            image_width=2,
            image_height=2,
            cameras=["/Camera"],
            frames="0",
            num_sensor_updates=1,
            daemon=fake_daemon,
        )

        assert result["successful_cameras"] == 1

    def test_main_without_action_exits_with_parser_error(self):
        from world_understanding.functions.graphics import render_ovrtx

        with pytest.raises(SystemExit) as exc_info:
            render_ovrtx._main([])

        assert exc_info.value.code == 2


class TestMapSensorToRenderVar:
    """Test _map_sensor_to_render_var() mapping correctness."""

    def test_depth_mapping(self):
        assert _map_sensor_to_render_var("depth") == "Depth"

    def test_normal_mapping(self):
        assert _map_sensor_to_render_var("normal") == "Normal"

    def test_albedo_mapping(self):
        assert _map_sensor_to_render_var("albedo") == "Albedo"

    def test_unknown_returns_none(self):
        assert _map_sensor_to_render_var("unknown_sensor") is None
        assert _map_sensor_to_render_var("") is None


class TestOvRTXBackendImportError:
    """Test graceful error when ovrtx venv cannot be provisioned."""

    def test_backend_error_when_venv_fails(self):
        """OvRTXRenderingBackend should raise RuntimeError when venv provisioning fails."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            side_effect=RuntimeError("ovrtx venv creation failed"),
        ):
            with pytest.raises(RuntimeError, match="ovrtx venv creation failed"):
                OvRTXRenderingBackend()


class TestOvRTXVenvPythonPath:
    """Test isolated ovrtx venv Python resolution without provisioning ovrtx."""

    @staticmethod
    def _write_fake_ovrtx_package(render_ovrtx: Any, venv_dir: Path) -> Path:
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        (site_dir / "ovrtx").mkdir(parents=True)
        return site_dir

    @staticmethod
    def _write_managed_marker(render_ovrtx: Any, venv_dir: Path) -> None:
        (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).write_text(
            "Created by world_understanding.functions.graphics.render_ovrtx\n"
            f"ovrtx_version={render_ovrtx._OVRTX_VERSION}\n"
            "runtime_lock_sha256="
            f"{render_ovrtx._ovrtx_runtime_lock_digest()}\n",
            encoding="utf-8",
        )

    def test_path_is_relative_to_returns_false_for_incomparable_paths(self) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        class IncomparablePath:
            def is_relative_to(self, parent: Path) -> bool:
                raise ValueError("different drive")

        assert not render_ovrtx._path_is_relative_to(IncomparablePath(), Path("C:/"))

    def test_infers_active_venv_from_cached_python_path(self, tmp_path: Path) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        requested_venv = tmp_path / "requested_venv"
        active_venv = tmp_path / "active_venv"

        assert (
            render_ovrtx._ovrtx_venv_dir_from_python_path(
                str(active_venv / "bin" / "python"), requested_venv
            )
            == active_venv
        )
        assert (
            render_ovrtx._ovrtx_venv_dir_from_python_path(
                str(active_venv / "Scripts" / "python.exe"), requested_venv
            )
            == active_venv
        )
        assert (
            render_ovrtx._ovrtx_venv_dir_from_python_path(
                "/fake/python", requested_venv
            )
            == requested_venv
        )

    def test_cached_python_does_not_mutate_global_site_dir_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        active_venv = tmp_path / "active_venv"
        python_path = active_venv / "Scripts" / "python.exe"
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        (active_venv / "pyvenv.cfg").write_text("")
        self._write_managed_marker(render_ovrtx, active_venv)
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(active_venv)[0]
        hdri_path = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        hdri_path.parent.mkdir(parents=True)
        hdri_path.write_bytes(b"fake-hdr")

        # This pins the cache fast path: the fake executable is not a real
        # ovrtx runtime, so this test should not reach the import probe.
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(python_path))
        stale_site_dir = tmp_path / "stale" / "site-packages"
        monkeypatch.setenv("_WU_OVRTX_SITE_DIR", str(stale_site_dir))

        assert render_ovrtx._get_ovrtx_python(active_venv) == str(python_path)
        assert os.environ["_WU_OVRTX_SITE_DIR"] == str(stale_site_dir)

    def test_custom_venv_request_ignores_different_cached_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        active_venv = tmp_path / "active_venv"
        requested_venv = tmp_path / "requested_venv"
        active_python = render_ovrtx._ovrtx_venv_python_path(active_venv)
        requested_python = render_ovrtx._ovrtx_venv_python_path(requested_venv)
        active_python.parent.mkdir(parents=True)
        active_python.write_text("")
        (active_venv / "pyvenv.cfg").write_text("")
        requested_python.parent.mkdir(parents=True)
        requested_python.write_text("")
        (requested_venv / "pyvenv.cfg").write_text("")
        self._write_fake_ovrtx_package(render_ovrtx, requested_venv)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(active_python))

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            assert cmd[0] == str(requested_python)
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._get_ovrtx_python(requested_venv) == str(requested_python)

    def test_existing_runtime_uses_standard_import_without_local_site_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "system_site_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        (venv_dir / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        calls: list[str] = []

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            calls.append(cmd[2])
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._get_ovrtx_python(venv_dir) == str(python_path)
        assert len(calls) == 1
        assert "metadata.version('ovrtx')" in calls[0]

    def test_existing_runtime_falls_back_to_target_site_dir_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "target_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        (venv_dir / "pyvenv.cfg").write_text("")
        self._write_fake_ovrtx_package(render_ovrtx, venv_dir)
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
        )

        assert render_ovrtx._get_ovrtx_python(venv_dir) == str(python_path)

    def test_default_venv_request_ignores_custom_cached_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        custom_venv = tmp_path / "custom_venv"
        default_venv = tmp_path / "default_venv"
        custom_python = render_ovrtx._ovrtx_venv_python_path(custom_venv)
        default_python = render_ovrtx._ovrtx_venv_python_path(default_venv)
        custom_python.parent.mkdir(parents=True)
        custom_python.write_text("")
        (custom_venv / "pyvenv.cfg").write_text("")
        default_python.parent.mkdir(parents=True)
        default_python.write_text("")
        (default_venv / "pyvenv.cfg").write_text("")
        self._write_fake_ovrtx_package(render_ovrtx, default_venv)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(custom_python))
        monkeypatch.setattr(render_ovrtx, "_OVRTX_VENV_DIR", default_venv)

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            assert cmd[0] == str(default_python)
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._get_ovrtx_python() == str(default_python)

    def test_default_venv_request_ignores_nonstandard_cached_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        cached_python = tmp_path / "external_wrapper" / "python"
        default_venv = tmp_path / "default_venv"
        default_python = render_ovrtx._ovrtx_venv_python_path(default_venv)
        cached_python.parent.mkdir(parents=True)
        cached_python.write_text("")
        default_python.parent.mkdir(parents=True)
        default_python.write_text("")
        self._write_fake_ovrtx_package(render_ovrtx, default_venv)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(cached_python))
        monkeypatch.setattr(render_ovrtx, "_OVRTX_VENV_DIR", default_venv)

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            assert cmd[0] == str(default_python)
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._get_ovrtx_python() == str(default_python)

    def test_custom_venv_request_ignores_unrelated_nonstandard_cached_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        cached_python = tmp_path / "external_wrapper" / "python"
        requested_venv = tmp_path / "requested_venv"
        requested_python = render_ovrtx._ovrtx_venv_python_path(requested_venv)
        cached_python.parent.mkdir(parents=True)
        cached_python.write_text("")
        requested_python.parent.mkdir(parents=True)
        requested_python.write_text("")
        self._write_fake_ovrtx_package(render_ovrtx, requested_venv)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", str(cached_python))

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            assert cmd[0] == str(requested_python)
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._get_ovrtx_python(requested_venv) == str(requested_python)

    def test_site_dir_env_is_omitted_for_unmanaged_python(self, tmp_path: Path) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        assert (
            render_ovrtx._ovrtx_site_dir_env_for_python(
                str(tmp_path / "external_wrapper" / "python")
            )
            is None
        )

    def test_site_dir_env_is_omitted_for_system_bin_python(
        self, tmp_path: Path
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        system_root = tmp_path / "usr"
        system_python = system_root / "bin" / "python3"
        system_python.parent.mkdir(parents=True)
        system_python.write_text("")
        (system_root / "lib" / "python3.11" / "site-packages").mkdir(parents=True)

        assert render_ovrtx._ovrtx_site_dir_env_for_python(str(system_python)) is None

    def test_site_dir_env_uses_explicit_runtime(self, tmp_path: Path) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        (site_dir / "ovrtx").mkdir(parents=True)

        assert render_ovrtx._ovrtx_site_dir_env_for_python(
            str(tmp_path / "external_wrapper" / "python"), venv_dir
        ) == str(site_dir)

    def test_site_dir_env_is_omitted_when_explicit_runtime_has_no_local_ovrtx(
        self, tmp_path: Path
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "system_site_venv"
        (venv_dir / "lib" / "python" / "site-packages").mkdir(parents=True)

        assert (
            render_ovrtx._ovrtx_site_dir_env_for_python(
                str(tmp_path / "external_wrapper" / "python"), venv_dir
            )
            is None
        )

    def test_site_dir_env_uses_managed_venv_python(self, tmp_path: Path) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = venv_dir / "bin" / "python"
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        (venv_dir / "pyvenv.cfg").write_text("")
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        (site_dir / "ovrtx").mkdir(parents=True)

        assert render_ovrtx._ovrtx_site_dir_env_for_python(str(python_path)) == str(
            site_dir
        )

    def test_site_packages_dir_uses_ovrtx_package_not_fixed_hdri_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        stale_site = venv_dir / "lib" / "python3.10" / "site-packages"
        active_site = venv_dir / "lib" / "python3.11" / "site-packages"
        stale_site.mkdir(parents=True)
        (active_site / "ovrtx" / "resources").mkdir(parents=True)
        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        assert render_ovrtx._ovrtx_site_packages_dir(venv_dir) == active_site

    def test_existing_windows_venv_uses_scripts_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx.os, "name", "nt")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
        )

        class FakeFileLock:
            def __init__(self, path: str, timeout: float) -> None:
                pass

            def __enter__(self) -> "FakeFileLock":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        monkeypatch.setattr(render_ovrtx, "FileLock", FakeFileLock)

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = venv_dir / "Scripts" / "python.exe"
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_fake_ovrtx_package(render_ovrtx, venv_dir)

        assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)

    def test_windows_uv_install_targets_scripts_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx.os, "name", "nt")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = venv_dir / "Scripts" / "python.exe"
        calls: list[tuple[list[str], str]] = []
        lock_calls: list[tuple[str, float]] = []
        lock_snapshots: list[tuple[Path, bytes]] = []

        class FakeFileLock:
            def __init__(self, path: str, timeout: float) -> None:
                lock_calls.append((path, timeout))

            def __enter__(self) -> "FakeFileLock":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def fake_run_checked(cmd: list[str], label: str) -> None:
            calls.append((cmd, label))
            if label == "uv venv creation":
                python_path.parent.mkdir(parents=True)
                python_path.write_text("")
            if label == "locked OVRTX runtime install":
                snapshot_path = type(render_ovrtx._OVRTX_RUNTIME_LOCK_FILE)(
                    cmd[cmd.index("-r") + 1]
                )
                assert snapshot_path.name.startswith("pylock.")
                lock_snapshots.append((snapshot_path, snapshot_path.read_bytes()))

        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(render_ovrtx, "FileLock", FakeFileLock)
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
        )

        assert render_ovrtx._get_ovrtx_python(venv_dir=venv_dir) == str(python_path)
        assert lock_calls
        venv_cmd, venv_label = calls[0]
        assert venv_label == "uv venv creation"
        assert "--allow-existing" in venv_cmd
        install_cmd, install_label = calls[1]
        assert install_label == "locked OVRTX runtime install"
        assert install_cmd[install_cmd.index("--python") + 1] == str(python_path)
        assert "--require-hashes" in install_cmd
        assert "--no-deps" in install_cmd
        assert lock_snapshots == [
            (
                type(render_ovrtx._OVRTX_RUNTIME_LOCK_FILE)(
                    install_cmd[install_cmd.index("-r") + 1]
                ),
                render_ovrtx._OVRTX_RUNTIME_LOCK_FILE.read_bytes(),
            )
        ]
        assert not lock_snapshots[0][0].exists()

    def test_uv_venv_uses_running_python(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx.os, "name", "posix")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda name: "uv")
        calls: list[tuple[list[str], str]] = []

        def fake_run_checked(cmd: list[str], label: str) -> None:
            calls.append((cmd, label))
            if label == "uv venv creation":
                python_path = render_ovrtx._ovrtx_venv_python_path(
                    tmp_path / "ovrtx_venv"
                )
                python_path.parent.mkdir(parents=True)
                python_path.write_text("")

        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda python_path_arg, venv_dir_arg: render_ovrtx._OVRTX_VERSION,
        )

        render_ovrtx._get_ovrtx_python_unlocked(venv_dir=tmp_path / "ovrtx_venv")

        venv_cmd, venv_label = calls[0]
        assert venv_label == "uv venv creation"
        assert "--allow-existing" in venv_cmd
        assert venv_cmd[venv_cmd.index("--python") + 1] == render_ovrtx.sys.executable

    def test_python_cache_is_keyed_by_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        first_venv = tmp_path / "first_venv"
        second_venv = tmp_path / "second_venv"
        first_python = render_ovrtx._ovrtx_venv_python_path(first_venv)
        second_python = render_ovrtx._ovrtx_venv_python_path(second_venv)
        for venv, python_path in (
            (first_venv, first_python),
            (second_venv, second_python),
        ):
            python_path.parent.mkdir(parents=True)
            python_path.write_text("")
            (venv / "pyvenv.cfg").write_text("")
            self._write_fake_ovrtx_package(render_ovrtx, venv)

        calls: list[str] = []

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            calls.append(cmd[0])
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)

        assert render_ovrtx._get_ovrtx_python(first_venv) == str(first_python)
        assert render_ovrtx._get_ovrtx_python(second_venv) == str(second_python)
        assert render_ovrtx._get_ovrtx_python(first_venv) == str(first_python)
        assert calls == [str(first_python), str(second_python)]

    def test_cached_managed_python_uses_marker_fast_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        monkeypatch.setattr(
            render_ovrtx,
            "_ovrtx_python_cache",
            {render_ovrtx._ovrtx_runtime_cache_key(venv_dir): str(python_path)},
        )

        class FailingLock:
            def __init__(self, path: str, timeout: int):
                raise AssertionError("cache hit should not acquire lock")

        monkeypatch.setattr(render_ovrtx, "FileLock", FailingLock)

        assert render_ovrtx._get_ovrtx_python(venv_dir) == str(python_path)

    def test_cached_validation_pins_managed_ownership_before_marker_loss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())

        def lose_marker(
            path: Path, unused_runtime_lock_digest: str | None = None
        ) -> bool:
            (path / render_ovrtx._OVRTX_MANAGED_MARKER).unlink()
            return False

        monkeypatch.setattr(
            render_ovrtx,
            "_ovrtx_managed_marker_matches_runtime_lock",
            lose_marker,
        )

        assert not render_ovrtx._cached_ovrtx_python_ready(str(python_path), venv_dir)
        assert cache_key in render_ovrtx._verified_managed_ovrtx_python_cache

    def test_ready_managed_runtime_skips_lock_without_process_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())

        class FailingLock:
            def __init__(self, path: str, timeout: int):
                raise AssertionError("ready runtime should not acquire lock")

        monkeypatch.setattr(render_ovrtx, "FileLock", FailingLock)

        assert render_ovrtx._get_ovrtx_python(venv_dir) == str(python_path)

    def test_verified_unmanaged_runtime_cache_skips_lock_and_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "prebuilt_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(
            render_ovrtx, "_ovrtx_python_cache", {cache_key: str(python_path)}
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_verified_ovrtx_python_cache",
            {(cache_key, render_ovrtx._ovrtx_runtime_lock_digest())},
        )
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> unittest.mock.Mock:
            calls.append(cmd)
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        class FailingLock:
            def __init__(self, path: str, timeout: int):
                raise AssertionError("verified unmanaged runtime should not lock")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fake_run)
        monkeypatch.setattr(render_ovrtx, "FileLock", FailingLock)

        assert render_ovrtx._get_ovrtx_python(venv_dir) == str(python_path)
        assert render_ovrtx._get_ovrtx_python(venv_dir) == str(python_path)
        assert calls == []
        assert not (venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER).exists()

    def test_cached_managed_python_without_marker_waits_for_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        monkeypatch.setattr(
            render_ovrtx,
            "_ovrtx_python_cache",
            {render_ovrtx._ovrtx_runtime_cache_key(venv_dir): str(python_path)},
        )
        lock_paths: list[str] = []

        class DummyLock:
            def __init__(self, path: str, timeout: int):
                lock_paths.append(path)

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def fake_unlocked(locked_venv_dir: Path) -> str:
            assert locked_venv_dir == venv_dir
            return "python-after-lock"

        monkeypatch.setattr(render_ovrtx, "FileLock", DummyLock)
        monkeypatch.setattr(render_ovrtx, "_get_ovrtx_python_unlocked", fake_unlocked)

        assert render_ovrtx._get_ovrtx_python(venv_dir) == "python-after-lock"
        assert lock_paths

    def test_interrupted_custom_provisioning_marker_allows_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "custom_ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        (venv_dir / render_ovrtx._OVRTX_PROVISIONING_MARKER).write_text("")
        monkeypatch.setattr(
            render_ovrtx.subprocess,
            "run",
            lambda *args, **kwargs: unittest.mock.Mock(
                returncode=1, stdout="", stderr="broken"
            ),
        )

        def fake_rmtree(path: Path, *args: Any, **kwargs: Any) -> None:
            assert path == venv_dir
            raise RuntimeError("cleanup attempted")

        monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)

        with pytest.raises(RuntimeError, match="cleanup attempted"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir=venv_dir)

    def test_broken_unmanaged_runtime_is_recreated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        venv_dir = tmp_path / "usr_local"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        rmtree_calls: list[Path] = []

        monkeypatch.setattr(
            render_ovrtx.subprocess,
            "run",
            lambda *args, **kwargs: unittest.mock.Mock(
                returncode=1, stdout="", stderr="broken"
            ),
        )
        real_rmtree = render_ovrtx.shutil.rmtree

        def fake_rmtree(path: Path, *args: Any, **kwargs: Any) -> None:
            rmtree_calls.append(path)
            real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)

        with pytest.raises(RuntimeError, match="uv venv creation failed"):
            render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)

        assert rmtree_calls == [venv_dir, venv_dir]

    @pytest.mark.parametrize(
        "probe_error",
        [
            OSError("cannot launch"),
            subprocess.TimeoutExpired("probe", timeout=30),
        ],
        ids=["launch-error", "timeout"],
    )
    def test_unmanaged_runtime_probe_error_is_preserved(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        probe_error: OSError | subprocess.TimeoutExpired,
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "1")
        venv_dir = tmp_path / "operator_ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        runtime_file = venv_dir / "operator-owned.txt"
        runtime_file.write_text("preserve")
        rmtree_calls: list[Path] = []

        def raise_probe(unused_python_path: Path, unused_venv_dir: Path) -> str:
            raise probe_error

        monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", raise_probe)
        monkeypatch.setattr(
            render_ovrtx.shutil,
            "rmtree",
            lambda path, ignore_errors=False: rmtree_calls.append(Path(path)),
        )

        with pytest.raises(RuntimeError, match="unmanaged.*refusing to replace"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert rmtree_calls == []
        assert python_path.exists()
        assert runtime_file.read_text() == "preserve"

    def test_disabled_auto_provision_preserves_managed_probe_launch_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")
        venv_dir = tmp_path / "managed_ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        rmtree_calls: list[Path] = []

        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda unused_python_path, unused_venv_dir: False,
        )

        def raise_probe(unused_python_path: Path, unused_venv_dir: Path) -> str:
            raise OSError("cannot launch")

        monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", raise_probe)
        monkeypatch.setattr(
            render_ovrtx.shutil,
            "rmtree",
            lambda path, ignore_errors=False: rmtree_calls.append(Path(path)),
        )

        with pytest.raises(RuntimeError, match="managed.*AUTO_PROVISION is disabled"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert rmtree_calls == []
        assert python_path.exists()
        assert render_ovrtx._ovrtx_managed_marker_matches_runtime_lock(venv_dir)

    def test_get_ovrtx_python_expands_default_runtime_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        captured: list[Path] = []
        lock_paths: list[str] = []

        class DummyLock:
            def __init__(self, path: str, timeout: int):
                lock_paths.append(path)

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def fake_unlocked(venv_dir: Path) -> str:
            captured.append(venv_dir)
            return "python"

        monkeypatch.setattr(render_ovrtx, "_OVRTX_VENV_DIR", Path("~/custom_ovrtx"))
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "FileLock", DummyLock)
        monkeypatch.setattr(render_ovrtx, "_get_ovrtx_python_unlocked", fake_unlocked)

        assert render_ovrtx._get_ovrtx_python() == "python"
        assert captured
        assert "~" not in str(captured[0])
        assert lock_paths
        lock_path = Path(lock_paths[0])
        assert lock_path.parent == captured[0].parent
        assert lock_path.name.startswith(f".{captured[0].name}-")
        assert lock_path.name.endswith(".lock")

    def test_unlocked_ready_managed_runtime_skips_import_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())

        def fail_run(*args: Any, **kwargs: Any) -> unittest.mock.Mock:
            raise AssertionError("ready runtime should not be import-probed")

        monkeypatch.setattr(render_ovrtx.subprocess, "run", fail_run)

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == str(python_path)

    def test_unlocked_matching_managed_runtime_refreshes_bundled_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        validated_digest = render_ovrtx._ovrtx_runtime_lock_digest()
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda unused_python_path, unused_venv_dir: False,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda unused_python_path, unused_venv_dir: render_ovrtx._OVRTX_VERSION,
        )
        removed: list[Path] = []
        refreshed: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_bundled_python_libraries",
            lambda path: removed.append(path) or [],
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_try_refresh_ovrtx_managed_marker",
            lambda path, digest: refreshed.append((path, digest)),
        )

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == str(python_path)
        assert removed == [venv_dir]
        assert refreshed == [(venv_dir, validated_digest)]
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
        assert (cache_key, validated_digest) in (
            render_ovrtx._verified_ovrtx_python_cache
        )

    @pytest.mark.parametrize("change_point", ["probe", "cleanup", "refresh", "marker"])
    def test_managed_runtime_identity_drift_requests_reprovision(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        change_point: str,
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        runtime_lock = tmp_path / "pylock.ovrtx-runtime.toml"
        runtime_lock.write_bytes(b"lock generation A")
        monkeypatch.setattr(render_ovrtx, "_OVRTX_RUNTIME_LOCK_FILE", runtime_lock)
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "1")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        validated_digest = render_ovrtx._ovrtx_runtime_lock_digest()
        removed_bundled: list[Path] = []
        refreshed: list[tuple[Path, str]] = []
        reprovisioned: list[Path] = []

        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda unused_python_path, unused_venv_dir: False,
        )

        def fake_probe(unused_python_path: Path, unused_venv_dir: Path) -> str:
            if change_point == "probe":
                runtime_lock.write_bytes(b"lock generation B")
            return render_ovrtx._OVRTX_VERSION

        def fake_remove_bundled(path: Path) -> list[Path]:
            removed_bundled.append(path)
            if change_point == "cleanup":
                runtime_lock.write_bytes(b"lock generation B")
            return []

        def fake_refresh(path: Path, digest: str) -> None:
            refreshed.append((path, digest))
            render_ovrtx._write_ovrtx_managed_marker(path, digest)
            if change_point == "refresh":
                runtime_lock.write_bytes(b"lock generation B")
            if change_point == "marker":
                (path / render_ovrtx._OVRTX_MANAGED_MARKER).unlink()

        def request_reprovision(path: Path) -> None:
            reprovisioned.append(path)
            raise RuntimeError("reprovision requested")

        monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_bundled_python_libraries",
            fake_remove_bundled,
        )
        monkeypatch.setattr(
            render_ovrtx, "_try_refresh_ovrtx_managed_marker", fake_refresh
        )
        monkeypatch.setattr(render_ovrtx, "_remove_ovrtx_venv", request_reprovision)

        with pytest.raises(RuntimeError, match="reprovision requested"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert reprovisioned == [venv_dir]
        assert removed_bundled == ([] if change_point == "probe" else [venv_dir])
        assert refreshed == (
            [(venv_dir, validated_digest)]
            if change_point in {"refresh", "marker"}
            else []
        )
        if change_point == "marker":
            assert not render_ovrtx._ovrtx_managed_marker_matches_runtime_lock(
                venv_dir, validated_digest
            )
            assert render_ovrtx._ovrtx_runtime_lock_digest() == validated_digest
        else:
            marker = render_ovrtx._read_ovrtx_managed_marker(
                venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
            )
            assert marker["runtime_lock_sha256"] == validated_digest
            assert render_ovrtx._ovrtx_runtime_lock_digest() != validated_digest
        assert render_ovrtx._verified_ovrtx_python_cache == set()
        assert render_ovrtx._ovrtx_python is None

    def test_managed_runtime_identity_drift_reprovisions_from_current_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        runtime_lock = tmp_path / "pylock.ovrtx-runtime.toml"
        runtime_lock.write_bytes(b"lock generation A")
        monkeypatch.setattr(render_ovrtx, "_OVRTX_RUNTIME_LOCK_FILE", runtime_lock)
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "1")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        probe_calls = 0
        installed_snapshots: list[bytes] = []

        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda unused_python_path, unused_venv_dir: False,
        )

        def fake_probe(unused_python_path: Path, unused_venv_dir: Path) -> str:
            nonlocal probe_calls
            probe_calls += 1
            if probe_calls == 1:
                runtime_lock.write_bytes(b"lock generation B")
            return render_ovrtx._OVRTX_VERSION

        def fake_run_checked(cmd: list[str], label: str) -> None:
            if label == "uv venv creation":
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.write_text("")
            elif label == "locked OVRTX runtime install":
                snapshot_path = Path(cmd[cmd.index("-r") + 1])
                installed_snapshots.append(snapshot_path.read_bytes())

        monkeypatch.setattr(render_ovrtx, "_probe_ovrtx_version", fake_probe)
        monkeypatch.setattr(render_ovrtx.shutil, "which", lambda unused_name: "uv")
        monkeypatch.setattr(render_ovrtx, "_run_checked", fake_run_checked)
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_bundled_python_libraries",
            lambda unused_venv_dir: [],
        )

        assert render_ovrtx._get_ovrtx_python_unlocked(venv_dir) == str(python_path)

        current_digest = render_ovrtx._ovrtx_runtime_lock_digest()
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
        marker = render_ovrtx._read_ovrtx_managed_marker(
            venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
        )
        assert installed_snapshots == [b"lock generation B"]
        assert marker["runtime_lock_sha256"] == current_digest
        assert (cache_key, current_digest) in (
            render_ovrtx._verified_ovrtx_python_cache
        )

    @pytest.mark.parametrize("change_point", ["probe", "cleanup", "refresh", "marker"])
    def test_disabled_auto_provision_preserves_managed_runtime_on_identity_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        change_point: str,
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        runtime_lock = tmp_path / "pylock.ovrtx-runtime.toml"
        runtime_lock.write_bytes(b"lock generation A")
        monkeypatch.setattr(render_ovrtx, "_OVRTX_RUNTIME_LOCK_FILE", runtime_lock)
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "0")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        validated_digest = render_ovrtx._ovrtx_runtime_lock_digest()
        marker_path = venv_dir / render_ovrtx._OVRTX_MANAGED_MARKER
        original_marker = marker_path.read_text(encoding="utf-8")
        removed_bundled: list[Path] = []
        refreshed: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda unused_python_path, unused_venv_dir: False,
        )

        def change_lock_during_probe(
            unused_python_path: Path, unused_venv_dir: Path
        ) -> str:
            if change_point == "probe":
                runtime_lock.write_bytes(b"lock generation B")
            return render_ovrtx._OVRTX_VERSION

        def fake_remove_bundled(path: Path) -> list[Path]:
            removed_bundled.append(path)
            if change_point == "cleanup":
                runtime_lock.write_bytes(b"lock generation B")
            return []

        def fake_refresh(path: Path, digest: str) -> None:
            refreshed.append((path, digest))
            render_ovrtx._write_ovrtx_managed_marker(path, digest)
            if change_point == "refresh":
                runtime_lock.write_bytes(b"lock generation B")
            if change_point == "marker":
                (path / render_ovrtx._OVRTX_MANAGED_MARKER).unlink()

        def fail_removal(*unused_args: Any, **unused_kwargs: Any) -> None:
            raise AssertionError("identity drift must preserve the managed runtime")

        monkeypatch.setattr(
            render_ovrtx, "_probe_ovrtx_version", change_lock_during_probe
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_bundled_python_libraries",
            fake_remove_bundled,
        )
        monkeypatch.setattr(
            render_ovrtx, "_try_refresh_ovrtx_managed_marker", fake_refresh
        )
        monkeypatch.setattr(render_ovrtx, "_remove_ovrtx_venv", fail_removal)

        with pytest.raises(RuntimeError, match="identity changed.*AUTO_PROVISION"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert python_path.exists()
        assert removed_bundled == ([] if change_point == "probe" else [venv_dir])
        assert refreshed == (
            [(venv_dir, validated_digest)]
            if change_point in {"refresh", "marker"}
            else []
        )
        if change_point == "marker":
            assert not marker_path.exists()
            assert render_ovrtx._ovrtx_runtime_lock_digest() == validated_digest
            with pytest.raises(RuntimeError, match="does not match.*runtime lock"):
                render_ovrtx._get_ovrtx_python_unlocked(venv_dir)
        else:
            assert marker_path.read_text(encoding="utf-8") == original_marker
            assert render_ovrtx._ovrtx_runtime_lock_digest() != validated_digest
        assert render_ovrtx._verified_ovrtx_python_cache == set()
        assert render_ovrtx._ovrtx_python is None

    def test_marker_loss_keeps_managed_provenance_after_partial_removal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        runtime_lock = tmp_path / "pylock.ovrtx-runtime.toml"
        runtime_lock.write_bytes(b"lock generation A")
        monkeypatch.setattr(render_ovrtx, "_OVRTX_RUNTIME_LOCK_FILE", runtime_lock)
        monkeypatch.setenv("WU_OVRTX_AUTO_PROVISION", "1")
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python", None)
        monkeypatch.setattr(render_ovrtx, "_ovrtx_python_cache", {})
        monkeypatch.setattr(render_ovrtx, "_verified_ovrtx_python_cache", set())
        monkeypatch.setattr(render_ovrtx, "_verified_managed_ovrtx_python_cache", set())

        venv_dir = tmp_path / "ovrtx_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("")
        self._write_managed_marker(render_ovrtx, venv_dir)
        cache_key = render_ovrtx._ovrtx_runtime_cache_key(venv_dir)
        monkeypatch.setattr(
            render_ovrtx,
            "_cached_ovrtx_python_ready",
            lambda unused_python_path, unused_venv_dir: False,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_probe_ovrtx_version",
            lambda unused_python_path, unused_venv_dir: render_ovrtx._OVRTX_VERSION,
        )
        monkeypatch.setattr(
            render_ovrtx,
            "_remove_ovrtx_bundled_python_libraries",
            lambda unused_venv_dir: [],
        )

        def remove_marker(path: Path, unused_digest: str) -> None:
            (path / render_ovrtx._OVRTX_MANAGED_MARKER).unlink()

        monkeypatch.setattr(
            render_ovrtx, "_try_refresh_ovrtx_managed_marker", remove_marker
        )
        monkeypatch.setattr(
            render_ovrtx.shutil,
            "rmtree",
            lambda unused_path, ignore_errors=False: None,
        )

        with pytest.raises(RuntimeError, match="could not be completely removed"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir)

        assert cache_key in render_ovrtx._verified_managed_ovrtx_python_cache
        assert (venv_dir / render_ovrtx._OVRTX_PROVISIONING_MARKER).exists()
        assert python_path.exists()

    def test_broken_managed_python_symlink_triggers_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "managed_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        self._write_managed_marker(render_ovrtx, venv_dir)
        original_exists = Path.exists
        original_is_symlink = Path.is_symlink

        def fake_exists(self: Path) -> bool:
            if self == python_path:
                return False
            return original_exists(self)

        def fake_is_symlink(self: Path) -> bool:
            if self == python_path:
                return True
            return original_is_symlink(self)

        def fake_rmtree(path: Path, *args: Any, **kwargs: Any) -> None:
            assert path == venv_dir
            raise RuntimeError("cleanup attempted")

        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
        monkeypatch.setattr(render_ovrtx.shutil, "rmtree", fake_rmtree)

        with pytest.raises(RuntimeError, match="cleanup attempted"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir=venv_dir)

    def test_broken_unmanaged_python_symlink_fails_clearly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "unmanaged_venv"
        python_path = render_ovrtx._ovrtx_venv_python_path(venv_dir)
        python_path.parent.mkdir(parents=True)
        original_exists = Path.exists
        original_is_symlink = Path.is_symlink

        def fake_exists(self: Path) -> bool:
            if self == python_path:
                return False
            return original_exists(self)

        def fake_is_symlink(self: Path) -> bool:
            if self == python_path:
                return True
            return original_is_symlink(self)

        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)

        with pytest.raises(RuntimeError, match="broken symlink"):
            render_ovrtx._get_ovrtx_python_unlocked(venv_dir=venv_dir)


class TestOvRTXBackendSensorSupport:
    """Test sensor capability methods without requiring ovrtx."""

    def test_supported_sensor_modes_class_var(self):
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        assert OvRTXRenderingBackend.SUPPORTED_SENSOR_MODES == []

    def test_render_warns_before_rejecting_sensors(self, tmp_path):
        from pxr import Usd

        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()
        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            backend = OvRTXRenderingBackend(ovrtx_venv_dir=str(venv_dir))

        stage = Usd.Stage.CreateInMemory()
        with pytest.warns(FutureWarning, match="no longer supports sensor/depth"):
            with pytest.raises(ValueError, match="color renders only"):
                backend.render(stage=stage, sensors=["depth"])


class TestEnsureLights:
    """Test _ensure_lights() adds default lights when needed."""

    def test_adds_hdri_dome_with_bundled_default(self, monkeypatch, tmp_path):
        """No env override: DomeLight uses the repo-bundled studio HDRI."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)
        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        expected_hdri = tmp_path / "studio.exr"
        expected_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setattr(render_ovrtx, "_BUNDLED_DEFAULT_HDRI_PATH", expected_hdri)

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        # No lights initially
        light_prims = [p for p in stage.Traverse() if "Light" in p.GetTypeName()]
        assert len(light_prims) == 0

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        assert len(dome_prims) == 1
        dome = UsdLux.DomeLight(dome_prims[0])
        assert dome.GetIntensityAttr().Get() == render_ovrtx._DEFAULT_HDRI_INTENSITY
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        assert tex is not None
        tex_value = tex.Get()
        resolved = str(tex_value.resolvedPath or tex_value.path)
        assert resolved == str(expected_hdri)
        assert Path(resolved).name == "studio.exr"
        # No distant light - env map provides direction.
        assert [p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)] == []

    def test_ensure_lights_uses_bundled_default_for_standalone_stage(
        self, monkeypatch, tmp_path
    ):
        """Direct stage prep should not provision OVRTX just to find the HDRI."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        expected_hdri = tmp_path / "studio.exr"
        expected_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setattr(render_ovrtx, "_BUNDLED_DEFAULT_HDRI_PATH", expected_hdri)

        def fake_get_ovrtx_python(venv_dir: Path | None = None) -> str:
            raise AssertionError("default HDRI resolution should not provision OVRTX")

        monkeypatch.setattr(render_ovrtx, "_get_ovrtx_python", fake_get_ovrtx_python)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        assert len(dome_prims) == 1
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        tex_value = tex.Get()
        assert Path(str(tex_value.resolvedPath or tex_value.path)).name == (
            "studio.exr"
        )

    def test_ensure_lights_missing_bundled_default_fails_without_provisioning(
        self, monkeypatch, tmp_path
    ):
        """Missing bundled default HDRI should fail clearly without provisioning."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from pxr import Usd, UsdGeom

        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(
            render_ovrtx, "_BUNDLED_DEFAULT_HDRI_PATH", tmp_path / "missing.exr"
        )

        def fake_get_ovrtx_python(venv_dir: Path | None = None) -> str:
            raise AssertionError("default HDRI failure should not provision OVRTX")

        monkeypatch.setattr(render_ovrtx, "_get_ovrtx_python", fake_get_ovrtx_python)

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        with pytest.raises(RuntimeError, match="Default OVRTX HDRI studio.exr"):
            _ensure_lights(stage)

    def test_ensure_lights_uses_runtime_only_for_packaged_hdri_intensity(
        self, monkeypatch, tmp_path
    ):
        """The runtime argument only classifies explicit packaged-HDRI overrides."""
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        requested_venv = tmp_path / "requested_venv"
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(requested_venv)[0]
        packaged_hdri = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        packaged_hdri.parent.mkdir(parents=True)
        packaged_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(packaged_hdri))

        def fake_get_ovrtx_python(venv_dir: Path | None = None) -> str:
            raise AssertionError("default HDRI resolution should not provision OVRTX")

        monkeypatch.setattr(render_ovrtx, "_get_ovrtx_python", fake_get_ovrtx_python)

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        _ensure_lights(stage, venv_dir=requested_venv)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        tex_value = tex.Get()
        resolved = str(tex_value.resolvedPath or tex_value.path)
        assert resolved == str(packaged_hdri)
        assert UsdLux.DomeLight(dome_prims[0]).GetIntensityAttr().Get() == 600.0

    def test_ensure_lights_copies_absolute_hdri_next_to_file_backed_stage(
        self, monkeypatch, tmp_path
    ):
        """Standalone light injection should avoid cache-absolute USD assets."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from pxr import Usd, UsdGeom, UsdLux

        from world_understanding.functions.graphics import render_ovrtx

        expected_hdri = tmp_path / "assets" / "studio.exr"
        expected_hdri.parent.mkdir()
        expected_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setattr(render_ovrtx, "_BUNDLED_DEFAULT_HDRI_PATH", expected_hdri)
        stage_path = tmp_path / "scene.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        UsdGeom.Cube.Define(stage, "/World/Cube")

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        tex_value = tex.Get()
        assert tex_value.path == expected_hdri.name
        assert (tmp_path / expected_hdri.name).read_bytes() == b"fake-hdr"

    def test_ensure_lights_copies_custom_hdri_override_next_to_file_backed_stage(
        self, monkeypatch, tmp_path
    ):
        """Custom local HDRI overrides should be portable for file-backed stages."""
        fake_hdri = tmp_path / "custom.hdr"
        fake_hdri.write_bytes(b"custom-hdr")
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(fake_hdri))

        from pxr import Usd, UsdGeom, UsdLux

        stage_dir = tmp_path / "stage"
        stage_dir.mkdir()
        stage = Usd.Stage.CreateNew(str(stage_dir / "scene.usda"))
        UsdGeom.Cube.Define(stage, "/World/Cube")

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        tex_value = tex.Get()
        assert tex_value.path == fake_hdri.name
        assert (stage_dir / fake_hdri.name).read_bytes() == b"custom-hdr"

    def test_default_hdri_returns_bundled_default(self, monkeypatch, tmp_path):
        """The active default HDRI is bundled with this package."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        expected_hdri = tmp_path / "studio.exr"
        expected_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setattr(render_ovrtx, "_BUNDLED_DEFAULT_HDRI_PATH", expected_hdri)

        resolved = Path(render_ovrtx._resolve_default_hdri())

        assert resolved == expected_hdri
        assert resolved.name == "studio.exr"

    def test_bundled_default_hdri_file_matches_documented_checksum(self):
        """The repo-owned default lighting asset should be present and intact."""
        from world_understanding.functions.graphics import render_ovrtx

        hdri_path = Path(render_ovrtx._default_bundled_hdri_path(require_exists=True))
        documented_checksums = [
            token.strip("`.,;")
            for token in (hdri_path.parent / "README.md").read_text().split()
            if re.fullmatch(r"[0-9a-fA-F]{64}", token.strip("`.,;"))
        ]

        assert hdri_path.name == "studio.exr"
        assert hdri_path.is_file()
        assert documented_checksums == [
            hashlib.sha256(hdri_path.read_bytes()).hexdigest()
        ]

    def test_site_packages_dir_fails_clearly_without_candidates(
        self, monkeypatch, tmp_path
    ):
        """Candidate discovery edge cases should not surface as IndexError."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(
            render_ovrtx, "_ovrtx_site_packages_candidates", lambda _: []
        )

        with pytest.raises(RuntimeError, match="No candidate site-packages"):
            render_ovrtx._ovrtx_site_packages_dir(tmp_path / "ovrtx_venv")

    def test_site_packages_dir_fails_clearly_without_ovrtx_package(
        self, monkeypatch, tmp_path
    ):
        """Known layouts without ovrtx should not return a silent fallback path."""
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        site_dir.mkdir(parents=True)
        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        with pytest.raises(RuntimeError, match="ovrtx package directory"):
            render_ovrtx._ovrtx_site_packages_dir(venv_dir)

    def test_ovrtx_packaged_hdri_fails_clearly_without_site_candidates(
        self, monkeypatch, tmp_path
    ):
        """Packaged StinsonBeach discovery should fail clearly without layouts."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(
            render_ovrtx, "_ovrtx_site_packages_candidates", lambda _: []
        )

        with pytest.raises(RuntimeError, match="No candidate site-packages"):
            render_ovrtx._default_ovrtx_hdri_path(tmp_path / "ovrtx_venv")

    def test_default_hdri_resolves_pip_target_fallback_layout(
        self, monkeypatch, tmp_path
    ):
        """Non-Windows pip --target fallback installs below lib/python."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        fallback_site = venv_dir / "lib" / "python" / "site-packages"
        fallback_hdri = fallback_site / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        fallback_hdri.parent.mkdir(parents=True)
        fallback_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setenv("_WU_OVRTX_SITE_DIR", str(fallback_site))

        assert Path(render_ovrtx._default_ovrtx_hdri_path(venv_dir)) == fallback_hdri

    def test_default_hdri_discovers_prebuilt_unix_venv_python_minor(
        self, monkeypatch, tmp_path
    ):
        """Prebuilt Unix venvs can use a different Python minor version."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        actual_site = venv_dir / "lib" / "python3.10" / "site-packages"
        actual_hdri = actual_site / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        actual_hdri.parent.mkdir(parents=True)
        actual_hdri.write_bytes(b"fake-hdr")

        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        assert render_ovrtx._default_ovrtx_hdri_path(venv_dir) == str(actual_hdri)

    def test_default_hdri_discovers_prebuilt_unix_lib64_layout(
        self, monkeypatch, tmp_path
    ):
        """Some Linux distributions place venv site-packages below lib64."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        actual_site = venv_dir / "lib64" / "python3.10" / "site-packages"
        actual_hdri = actual_site / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        actual_hdri.parent.mkdir(parents=True)
        actual_hdri.write_bytes(b"fake-hdr")

        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        assert render_ovrtx._default_ovrtx_hdri_path(venv_dir) == str(actual_hdri)

    def test_default_hdri_discovers_windows_pip_target_layout(
        self, monkeypatch, tmp_path
    ):
        """Windows discovery should not be locked to stdlib venv layout only."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        fallback_site = venv_dir / "lib" / "python" / "site-packages"
        fallback_hdri = fallback_site / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        fallback_hdri.parent.mkdir(parents=True)
        fallback_hdri.write_bytes(b"fake-hdr")

        monkeypatch.setattr(render_ovrtx.os, "name", "nt")

        assert render_ovrtx._default_ovrtx_hdri_path(
            venv_dir, require_exists=True
        ) == str(fallback_hdri)

    def test_default_hdri_discovers_moved_texture_within_ovrtx_package(
        self, monkeypatch, tmp_path
    ):
        """Keep working if ovrtx moves StinsonBeach.hdr within its package."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        moved_hdri = site_dir / "ovrtx" / "resources" / "textures" / "StinsonBeach.hdr"
        moved_hdri.parent.mkdir(parents=True)
        moved_hdri.write_bytes(b"fake-hdr")

        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        assert render_ovrtx._default_ovrtx_hdri_path(
            venv_dir, require_exists=True
        ) == str(moved_hdri)

    def test_default_hdri_moved_texture_search_is_cached(self, monkeypatch, tmp_path):
        """Moved-texture fallback should cache bounded misses."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        moved_hdri = site_dir / "ovrtx" / "resources" / "textures" / "StinsonBeach.hdr"
        moved_hdri.parent.mkdir(parents=True)
        moved_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        assert render_ovrtx._default_ovrtx_hdri_path(
            venv_dir, require_exists=True
        ) == str(moved_hdri)

        def fail_rglob(self: Path, pattern: str):
            raise AssertionError(f"unexpected recursive search for {pattern}")

        monkeypatch.setattr(Path, "rglob", fail_rglob)
        assert render_ovrtx._default_ovrtx_hdri_path(
            venv_dir, require_exists=True
        ) == str(moved_hdri)

    def test_default_hdri_strict_resolution_fails_when_bundled_default_missing(
        self, monkeypatch, tmp_path
    ):
        """Strict active default resolution should fail loudly if HDRI is absent."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setattr(
            render_ovrtx, "_BUNDLED_DEFAULT_HDRI_PATH", tmp_path / "missing.exr"
        )

        with pytest.raises(RuntimeError, match="Default OVRTX HDRI"):
            render_ovrtx._resolve_default_hdri(require_exists=True)

    def test_explicit_bundled_hdri_uses_legacy_intensity(self, monkeypatch, tmp_path):
        """An explicitly selected legacy EXR should not inherit StinsonBeach's multiplier."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        bundled_hdri = tmp_path / "bundled.exr"
        bundled_hdri.write_bytes(b"fake-exr")
        monkeypatch.setattr(render_ovrtx, "_BUNDLED_LEGACY_HDRI_PATH", bundled_hdri)

        assert render_ovrtx._resolve_default_hdri_intensity(str(bundled_hdri)) == 1.0

    def test_legacy_hdri_intensity_fallback_normalizes_when_resolve_fails(
        self, monkeypatch, tmp_path
    ):
        """The resolve() fallback should compare normalized absolute paths."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        bundled_hdri = tmp_path / "assets" / "bundled.exr"
        bundled_hdri.parent.mkdir()
        bundled_hdri.write_bytes(b"fake-exr")
        equivalent_path = bundled_hdri.parent / ".." / "assets" / bundled_hdri.name
        monkeypatch.setattr(render_ovrtx, "_BUNDLED_LEGACY_HDRI_PATH", bundled_hdri)

        original_resolve = Path.resolve

        def fake_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
            if self.name == bundled_hdri.name:
                raise OSError("synthetic resolve failure")
            return original_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

        assert render_ovrtx._resolve_default_hdri_intensity(str(equivalent_path)) == 1.0

    def test_default_hdri_moved_texture_negative_lookup_is_cached(
        self, monkeypatch, tmp_path
    ):
        """Repeated misses should not re-check moved HDRI paths before expiry."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        render_ovrtx._OVRTX_MOVED_HDRI_CACHE.clear()
        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        ovrtx_package_dir = site_dir / "ovrtx"
        ovrtx_package_dir.mkdir(parents=True)
        moved_hdri = ovrtx_package_dir / "resources" / "textures" / "StinsonBeach.hdr"
        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        def fail_rglob(self: Path, pattern: str):
            raise AssertionError(f"unexpected recursive search for {pattern}")

        monkeypatch.setattr(render_ovrtx.time, "monotonic", lambda: 0.0)
        monkeypatch.setattr(Path, "rglob", fail_rglob)
        with pytest.raises(RuntimeError, match="Default OVRTX HDRI"):
            render_ovrtx._default_ovrtx_hdri_path(venv_dir, require_exists=True)

        moved_hdri.parent.mkdir(parents=True)
        moved_hdri.write_bytes(b"fake-hdr")
        with pytest.raises(RuntimeError, match="Default OVRTX HDRI"):
            render_ovrtx._default_ovrtx_hdri_path(venv_dir, require_exists=True)

        assert ovrtx_package_dir in render_ovrtx._OVRTX_MOVED_HDRI_CACHE

    def test_default_hdri_moved_texture_negative_cache_expires(
        self, monkeypatch, tmp_path
    ):
        """Expired negative moved-HDRI lookups should re-check bounded paths."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        render_ovrtx._OVRTX_MOVED_HDRI_CACHE.clear()
        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        ovrtx_package_dir = site_dir / "ovrtx"
        ovrtx_package_dir.mkdir(parents=True)
        moved_hdri = ovrtx_package_dir / "resources" / "textures" / "StinsonBeach.hdr"
        monkeypatch.setattr(render_ovrtx.os, "name", "posix")
        monotonic_values = iter([0.0, 1.0, 61.0])

        def fail_rglob(self: Path, pattern: str):
            raise AssertionError(f"unexpected recursive search for {pattern}")

        monkeypatch.setattr(
            render_ovrtx.time, "monotonic", lambda: next(monotonic_values)
        )
        monkeypatch.setattr(Path, "rglob", fail_rglob)

        with pytest.raises(RuntimeError, match="Default OVRTX HDRI"):
            render_ovrtx._default_ovrtx_hdri_path(venv_dir, require_exists=True)
        with pytest.raises(RuntimeError, match="Default OVRTX HDRI"):
            render_ovrtx._default_ovrtx_hdri_path(venv_dir, require_exists=True)
        moved_hdri.parent.mkdir(parents=True)
        moved_hdri.write_bytes(b"fake-hdr")

        assert render_ovrtx._default_ovrtx_hdri_path(
            venv_dir, require_exists=True
        ) == str(moved_hdri)

    def test_default_hdri_candidate_fast_path_skips_fallback_lookup(
        self, monkeypatch, tmp_path
    ):
        """Primary packaged path should not inspect fallback locations."""
        from world_understanding.functions.graphics import render_ovrtx

        site_dir = tmp_path / "site-packages"
        primary_hdri = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        primary_hdri.parent.mkdir(parents=True)
        primary_hdri.write_bytes(b"fake-hdr")

        def fail_rglob(self: Path, pattern: str):
            raise AssertionError(f"unexpected package-tree recursion for {pattern}")

        monkeypatch.setattr(Path, "rglob", fail_rglob)

        assert render_ovrtx._default_ovrtx_hdri_candidates(site_dir) == [primary_hdri]

    def test_default_hdri_import_time_fallback_skips_fallback_lookup(
        self, monkeypatch, tmp_path
    ):
        """Import-time fallback path computation should not inspect ovrtx."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        (site_dir / "ovrtx").mkdir(parents=True)
        monkeypatch.setattr(render_ovrtx.os, "name", "posix")

        def fail_rglob(self: Path, pattern: str):
            raise AssertionError(f"unexpected package-tree recursion for {pattern}")

        monkeypatch.setattr(Path, "rglob", fail_rglob)

        expected = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        assert render_ovrtx._default_ovrtx_hdri_path(venv_dir) == str(expected)

    def test_default_hdri_require_exists_fails_when_packaged_asset_missing(
        self, monkeypatch, tmp_path
    ):
        """Render-time default resolution should fail clearly if HDRI is absent."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("_WU_OVRTX_SITE_DIR", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        with pytest.raises(RuntimeError, match="Default OVRTX HDRI"):
            render_ovrtx._default_ovrtx_hdri_path(
                tmp_path / "missing_ovrtx_venv", require_exists=True
            )

    def test_default_hdri_missing_local_override_fails_when_required(
        self, monkeypatch, tmp_path
    ):
        """Missing absolute WU_OVRTX_DEFAULT_HDRI should fail before render."""
        from world_understanding.functions.graphics import render_ovrtx

        missing_hdri = tmp_path / "missing.hdr"
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(missing_hdri))

        with pytest.raises(RuntimeError, match="WU_OVRTX_DEFAULT_HDRI"):
            render_ovrtx._resolve_default_hdri(require_exists=True)

    def test_default_hdri_missing_file_uri_override_fails_when_required(
        self, monkeypatch, tmp_path
    ):
        """Missing file:// HDRI overrides are local paths, not remote assets."""
        from world_understanding.functions.graphics import render_ovrtx

        missing_hdri_uri = (tmp_path / "missing.hdr").as_uri()
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", missing_hdri_uri)

        with pytest.raises(RuntimeError, match="WU_OVRTX_DEFAULT_HDRI"):
            render_ovrtx._resolve_default_hdri(require_exists=True)

    def test_default_hdri_missing_windows_file_uri_override_fails_when_required(
        self, monkeypatch
    ):
        """Missing file:///C:/... HDRI overrides should validate as local paths."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "file:///C:/missing.hdr")

        with pytest.raises(RuntimeError, match="WU_OVRTX_DEFAULT_HDRI"):
            render_ovrtx._resolve_default_hdri(require_exists=True)

    def test_two_slash_windows_file_uri_normalizes_to_drive_path(self):
        from world_understanding.functions.graphics import render_ovrtx

        assert render_ovrtx._local_asset_path("file://C:/missing.hdr") == Path(
            "C:/missing.hdr"
        )

    def test_ovrtx_subprocess_env_drops_inaccessible_path_entries(
        self, monkeypatch, tmp_path
    ):
        from world_understanding.functions.graphics import render_ovrtx

        ok_dir = tmp_path / "ok"
        blocked_dir = tmp_path / "blocked"
        ok_dir.mkdir()
        monkeypatch.setenv("PATH", f"{ok_dir}{os.pathsep}{blocked_dir}")
        original_exists = Path.exists

        def fake_exists(self: Path) -> bool:
            if self == blocked_dir:
                raise PermissionError("blocked")
            return original_exists(self)

        monkeypatch.setattr(Path, "exists", fake_exists)

        entries = render_ovrtx._ovrtx_subprocess_env()["PATH"].split(os.pathsep)
        assert str(ok_dir) in entries
        assert str(blocked_dir) not in entries

    def test_default_hdri_rejects_asset_delimiter_chars(self, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv(
            "WU_OVRTX_DEFAULT_HDRI",
            "https://example.invalid/env@maps/StinsonBeach.hdr",
        )

        with pytest.raises(RuntimeError, match="cannot be safely embedded"):
            render_ovrtx._resolve_default_hdri(require_exists=True)

    def test_default_hdri_windows_drive_relative_override_is_local(self, monkeypatch):
        """Windows drive-relative paths like C:foo should not parse as URLs."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "C:missing.hdr")

        with pytest.raises(RuntimeError, match="WU_OVRTX_DEFAULT_HDRI"):
            render_ovrtx._resolve_default_hdri(require_exists=True)

    def test_windows_drive_path_detection_requires_alpha_drive(self):
        from world_understanding.functions.graphics import render_ovrtx

        assert render_ovrtx._looks_like_windows_drive_path("C:missing.hdr")
        assert not render_ovrtx._looks_like_windows_drive_path("1:missing.hdr")
        assert not render_ovrtx._looks_like_windows_drive_path(":missing.hdr")

    def test_default_hdri_relative_override_defers_to_usd_resolver(self, monkeypatch):
        """Relative HDRI overrides may be valid in the USD asset context."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "assets/my_hdri.exr")

        assert (
            render_ovrtx._resolve_default_hdri(require_exists=True)
            == "assets/my_hdri.exr"
        )

    @pytest.mark.parametrize(
        "asset_url",
        [
            "https://example.invalid/StinsonBeach.hdr",
            "s3://bucket/StinsonBeach.hdr",
        ],
    )
    def test_default_hdri_remote_override_does_not_require_local_file(
        self, monkeypatch, asset_url
    ):
        """Remote WU_OVRTX_DEFAULT_HDRI values should pass through unchanged."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", asset_url)

        assert render_ovrtx._resolve_default_hdri(require_exists=True) == asset_url

    def test_default_lights_usda_uses_working_ovrtx_hdri(self, monkeypatch):
        """Sublayer default light generation should carry the new defaults."""
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        from world_understanding.functions.graphics import render_ovrtx

        usda = render_ovrtx.build_default_hdri_lights_usda()

        assert "studio.exr" in usda
        assert "float inputs:intensity = 600.0" in usda
        assert "asset inputs:texture:file" in usda

    def test_default_lights_usda_normalizes_windows_asset_paths(self):
        from world_understanding.functions.graphics import render_ovrtx

        usda = render_ovrtx._build_default_lights_usda(
            "C:\\Users\\Name\\StinsonBeach.hdr", 600.0
        )

        assert "@C:/Users/Name/StinsonBeach.hdr@" in usda
        assert "\\Users" not in usda

    def test_default_lights_usda_rejects_suspicious_asset_syntax(self):
        from world_understanding.functions.graphics import render_ovrtx

        with pytest.raises(ValueError, match="USDA asset syntax"):
            render_ovrtx._build_default_lights_usda(
                "https://example.invalid/env@maps/StinsonBeach.hdr", 600.0
            )

    def test_default_hdri_intensity_uses_legacy_default_for_custom_override(
        self, monkeypatch
    ):
        """Custom HDRI overrides keep the old 1.0 intensity unless configured."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "assets/my_hdri.exr")
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        assert render_ovrtx._resolve_default_hdri_intensity() == 1.0

    def test_default_hdri_intensity_env_override_still_wins(self, monkeypatch):
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "assets/my_hdri.exr")
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", "12.5")

        assert render_ovrtx._resolve_default_hdri_intensity() == 12.5

    def test_default_hdri_intensity_uses_packaged_asset_over_ambient_override(
        self, monkeypatch, tmp_path
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        packaged_hdri = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        monkeypatch.setattr(render_ovrtx, "_OVRTX_VENV_DIR", venv_dir)
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", "assets/my_hdri.exr")
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        assert render_ovrtx._resolve_default_hdri_intensity(str(packaged_hdri)) == 600.0

    def test_default_hdri_intensity_uses_packaged_env_override_when_asset_omitted(
        self, monkeypatch, tmp_path
    ):
        from world_understanding.functions.graphics import render_ovrtx

        venv_dir = tmp_path / "ovrtx_venv"
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        packaged_hdri = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        monkeypatch.setattr(render_ovrtx, "_OVRTX_VENV_DIR", venv_dir)
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(packaged_hdri))
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        assert render_ovrtx._resolve_default_hdri_intensity() == 600.0

    def test_default_hdri_intensity_uses_contextual_packaged_asset(
        self, monkeypatch, tmp_path
    ):
        from world_understanding.functions.graphics import render_ovrtx

        default_venv = tmp_path / "default_venv"
        custom_venv = tmp_path / "custom_venv"
        custom_site = render_ovrtx._ovrtx_site_packages_candidates(custom_venv)[0]
        custom_hdri = custom_site / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        monkeypatch.setattr(render_ovrtx, "_OVRTX_VENV_DIR", default_venv)
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(custom_hdri))
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        assert (
            render_ovrtx._resolve_default_hdri_intensity(str(custom_hdri), custom_venv)
            == 600.0
        )

    def test_default_hdri_lights_usda_expands_contextual_venv_dir(
        self, monkeypatch, tmp_path
    ):
        from world_understanding.functions.graphics import render_ovrtx

        home_dir = tmp_path / "home"
        venv_dir = home_dir / "ovrtx_venv"
        site_dir = render_ovrtx._ovrtx_site_packages_candidates(venv_dir)[0]
        custom_hdri = site_dir / render_ovrtx._OVRTX_DEFAULT_HDRI_RELATIVE_PATH
        custom_hdri.parent.mkdir(parents=True)
        custom_hdri.write_bytes(b"fake-hdr")
        monkeypatch.setenv("HOME", str(home_dir))
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(custom_hdri))
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        usda = render_ovrtx.build_default_hdri_lights_usda(
            venv_dir=Path("~/ovrtx_venv")
        )

        assert "float inputs:intensity = 600.0" in usda

    def test_default_hdri_intensity_uses_custom_fallback_for_stinson_beach_override(
        self, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv(
            "WU_OVRTX_DEFAULT_HDRI",
            "https://example.invalid/lighting/StinsonBeach.hdr",
        )
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        assert (
            render_ovrtx._resolve_default_hdri_intensity(
                "https://example.invalid/lighting/StinsonBeach.hdr"
            )
            == 1.0
        )

    def test_default_hdri_intensity_uses_custom_fallback_for_encoded_stinson_override(
        self, monkeypatch
    ):
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv(
            "WU_OVRTX_DEFAULT_HDRI",
            "https://example.invalid/lighting/StinsonBeach%2Ehdr",
        )
        monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

        assert (
            render_ovrtx._resolve_default_hdri_intensity(
                "https://example.invalid/lighting/StinsonBeach%2Ehdr"
            )
            == 1.0
        )

    def test_env_var_overrides_default_hdri(self, monkeypatch, tmp_path):
        """``WU_OVRTX_DEFAULT_HDRI`` overrides the bundled default HDRI.

        Operators can point at a different local ``.exr``/``.hdr`` or an
        S3 URL when they want to swap lighting (e.g. a public HDRI for
        restricted-network hosts). The override replaces the texture
        path but keeps every other DomeLight attribute identical.
        """
        fake_hdri = tmp_path / "env.exr"
        fake_hdri.write_bytes(b"fake-exr-contents")
        monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI", str(fake_hdri))

        from pxr import Usd, UsdGeom, UsdLux

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, "/World/Cube")

        _ensure_lights(stage)

        dome_prims = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
        assert len(dome_prims) == 1
        tex = dome_prims[0].GetAttribute("inputs:texture:file")
        tex_value = tex.Get()
        resolved = str(tex_value.resolvedPath or tex_value.path)
        assert resolved == str(fake_hdri)

    def test_does_not_add_lights_when_present(self):
        from pxr import Usd, UsdLux

        stage = Usd.Stage.CreateInMemory()
        UsdLux.DomeLight.Define(stage, "/Existing/DomeLight")

        _ensure_lights(stage)

        # Should still only have the original light
        light_prims = [p for p in stage.Traverse() if "Light" in p.GetTypeName()]
        assert len(light_prims) == 1


class TestCopyExportedRelativeAssets:
    """Test local asset mirroring for exported OVRTX stages."""

    def test_copies_relative_texture_assets_to_export_dir(self, tmp_path):
        from pxr import Sdf, Usd, UsdShade

        source_dir = tmp_path / "source"
        texture_dir = source_dir / "textures"
        texture_dir.mkdir(parents=True)
        texture = texture_dir / "checker.png"
        texture.write_bytes(b"fake-png")

        stage_path = source_dir / "stage.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Tex")
        shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/checker.png"))
        stage.GetRootLayer().Save()

        reopened = Usd.Stage.Open(str(stage_path))
        export_dir = tmp_path / "render"
        export_dir.mkdir()

        copied = _copy_exported_relative_assets(reopened, export_dir)

        assert copied == 1
        assert (export_dir / "textures" / "checker.png").read_bytes() == b"fake-png"

    def test_copies_relative_texture_assets_using_explicit_base_dir(self, tmp_path):
        from pxr import Sdf, Usd, UsdShade

        source_dir = tmp_path / "source"
        texture_dir = source_dir / "textures"
        texture_dir.mkdir(parents=True)
        texture = texture_dir / "checker.png"
        texture.write_bytes(b"fake-png")

        stage = Usd.Stage.CreateInMemory()
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Tex")
        shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/checker.png"))

        export_dir = tmp_path / "render"
        export_dir.mkdir()

        copied = _copy_exported_relative_assets(
            stage,
            export_dir,
            base_dir=source_dir,
        )

        assert copied == 1
        assert (export_dir / "textures" / "checker.png").read_bytes() == b"fake-png"

    def test_normalizes_dot_relative_texture_paths_in_exported_stage(self, tmp_path):
        from pxr import Sdf, Usd, UsdShade

        source_dir = tmp_path / "source"
        texture_dir = source_dir / "Textures"
        texture_dir.mkdir(parents=True)
        texture = texture_dir / "checker.png"
        texture.write_bytes(b"fake-png")

        stage_path = source_dir / "stage.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Tex")
        shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("./Textures/checker.png"))
        stage.GetRootLayer().Save()

        reopened = Usd.Stage.Open(str(stage_path))
        export_dir = tmp_path / "render"
        export_dir.mkdir()
        exported_stage_path = export_dir / "stage.usdc"
        reopened.GetRootLayer().Export(str(exported_stage_path))

        copied = _copy_exported_relative_assets(
            reopened,
            export_dir,
            exported_stage_path=exported_stage_path,
        )

        assert copied == 1
        assert (export_dir / "Textures" / "checker.png").read_bytes() == b"fake-png"

        exported = Sdf.Layer.FindOrOpen(str(exported_stage_path))
        attr = exported.GetPrimAtPath("/World/Looks/Tex").attributes["inputs:file"]
        assert attr.default == Sdf.AssetPath("Textures/checker.png")

    def test_disambiguates_colliding_relative_texture_assets(
        self,
        tmp_path,
        monkeypatch,
    ):
        from pxr import Sdf, Usd, UsdShade

        source_a = tmp_path / "a" / "textures"
        source_b = tmp_path / "b" / "textures"
        source_a.mkdir(parents=True)
        source_b.mkdir(parents=True)
        texture_a = source_a / "albedo.png"
        texture_b = source_b / "albedo.png"
        texture_a.write_bytes(b"texture-a")
        texture_b.write_bytes(b"texture-b")

        stage = Usd.Stage.CreateInMemory()
        shader_a = UsdShade.Shader.Define(stage, "/World/Looks/A/Tex")
        shader_a.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/albedo.png"))
        shader_b = UsdShade.Shader.Define(stage, "/World/Looks/B/Tex")
        shader_b.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("textures/albedo.png"))

        def fake_texture_assets(stage, base_dir=None):
            return [
                {
                    "prim_path": "/World/Looks/A/Tex",
                    "attr_name": "inputs:file",
                    "file_path": "textures/albedo.png",
                    "resolved_path": str(texture_a),
                    "is_local": True,
                },
                {
                    "prim_path": "/World/Looks/B/Tex",
                    "attr_name": "inputs:file",
                    "file_path": "textures/albedo.png",
                    "resolved_path": str(texture_b),
                    "is_local": True,
                },
            ]

        import world_understanding.utils.usd.material as usd_material

        monkeypatch.setattr(
            usd_material,
            "get_local_texture_file_assets",
            fake_texture_assets,
        )

        export_dir = tmp_path / "render"
        export_dir.mkdir()
        exported_stage_path = export_dir / "stage.usdc"
        stage.GetRootLayer().Export(str(exported_stage_path))

        copied = _copy_exported_relative_assets(
            stage,
            export_dir,
            exported_stage_path=exported_stage_path,
        )

        assert copied == 2
        assert (export_dir / "textures" / "albedo.png").read_bytes() == b"texture-a"
        digest_b = hashlib.sha256(str(texture_b.resolve()).encode("utf-8")).hexdigest()[
            :12
        ]
        localized_b = export_dir / "textures" / f"albedo_{digest_b}.png"
        assert localized_b.read_bytes() == b"texture-b"

        exported = Sdf.Layer.FindOrOpen(str(exported_stage_path))
        attr_a = exported.GetPrimAtPath("/World/Looks/A/Tex").attributes["inputs:file"]
        attr_b = exported.GetPrimAtPath("/World/Looks/B/Tex").attributes["inputs:file"]
        assert attr_a.default == Sdf.AssetPath("textures/albedo.png")
        assert attr_b.default == Sdf.AssetPath(
            localized_b.relative_to(export_dir).as_posix(),
        )

    def test_localizes_absolute_file_uri_texture_assets(self, tmp_path):
        from pxr import Sdf, Usd, UsdShade

        texture = tmp_path / "checker.png"
        texture.write_bytes(b"fake-png")

        stage = Usd.Stage.CreateInMemory()
        shader = UsdShade.Shader.Define(stage, "/World/Looks/Tex")
        shader.GetPrim().CreateAttribute(
            "inputs:file",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(texture.as_uri()))

        export_dir = tmp_path / "render"
        export_dir.mkdir()
        exported_stage_path = export_dir / "stage.usdc"
        stage.GetRootLayer().Export(str(exported_stage_path))

        copied = _copy_exported_relative_assets(
            stage,
            export_dir,
            exported_stage_path=exported_stage_path,
        )

        copied_textures = list((export_dir / "textures").glob("checker_*.png"))
        assert copied == 1
        assert len(copied_textures) == 1
        assert copied_textures[0].read_bytes() == b"fake-png"

        exported = Sdf.Layer.FindOrOpen(str(exported_stage_path))
        attr = exported.GetPrimAtPath("/World/Looks/Tex").attributes["inputs:file"]
        assert attr.default == Sdf.AssetPath(
            copied_textures[0].relative_to(export_dir).as_posix(),
        )

    def test_localizes_relative_mdl_directory(self, tmp_path):
        from pxr import Sdf, Usd, UsdShade

        source_dir = tmp_path / "source"
        mdl_dir = source_dir
        mdl_dir.mkdir()
        (mdl_dir / "OmniPBR.mdl").write_text("mdl 1.7;", encoding="utf-8")
        (mdl_dir / "OmniPBRBase.mdl").write_text("mdl 1.7;", encoding="utf-8")
        (source_dir / "stage.usdc").write_bytes(b"source sentinel")

        stage_path = source_dir / "asset.usda"
        stage = Usd.Stage.CreateNew(str(stage_path))
        shader = UsdShade.Shader.Define(stage, "/World/Looks/OmniPBR")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("./OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        reopened = Usd.Stage.Open(str(stage_path))
        export_dir = tmp_path / "render"
        export_dir.mkdir()
        exported_stage_path = export_dir / "stage.usdc"
        reopened.GetRootLayer().Export(str(exported_stage_path))

        copied = _copy_exported_relative_assets(
            reopened,
            export_dir,
            exported_stage_path=exported_stage_path,
        )

        assert copied == 1
        copied_mdl = list((export_dir / "mdl_materials").glob("*/OmniPBR.mdl"))
        assert len(copied_mdl) == 1
        localized_path = copied_mdl[0].relative_to(export_dir)
        assert (export_dir / localized_path.parent / "OmniPBRBase.mdl").is_file()
        assert not (export_dir / "OmniPBR.mdl").exists()
        assert exported_stage_path.read_bytes() != b"source sentinel"

        exported_layer = Sdf.Layer.FindOrOpen(str(exported_stage_path))
        shader_spec = exported_layer.GetPrimAtPath("/World/Looks/OmniPBR")
        mdl_attribute = shader_spec.attributes["info:mdl:sourceAsset"]
        assert mdl_attribute.default == Sdf.AssetPath(localized_path.as_posix())


class TestDefaultNumSensorUpdates:
    """Test DEFAULT_NUM_SENSOR_UPDATES constant."""

    def test_default_num_sensor_updates(self):
        # 32 is the fast-iteration default paired with rt2 render mode.
        # num_sensor_updates here is step-loop count, not samples-per-pixel
        # (ovrtx 0.2.0 ignores the SPP schema attr). For pt-mode quality
        # parity, callers should pass num_sensor_updates=500 explicitly.
        assert DEFAULT_NUM_SENSOR_UPDATES == 32


class TestOvRTXDaemonResourceLimits:
    def test_nonnegative_integer_environment_parser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        assert render_ovrtx.DEFAULT_OVRTX_DAEMON_MAX_RENDERS == 64
        assert render_ovrtx.DEFAULT_OVRTX_DAEMON_MAX_RSS_BYTES == 24 * 1024**3

        monkeypatch.delenv("TEST_OVRTX_LIMIT", raising=False)
        assert render_ovrtx._parse_nonnegative_int_env("TEST_OVRTX_LIMIT", 7) == 7
        monkeypatch.setenv("TEST_OVRTX_LIMIT", "")
        assert render_ovrtx._parse_nonnegative_int_env("TEST_OVRTX_LIMIT", 7) == 7
        monkeypatch.setenv("TEST_OVRTX_LIMIT", "0")
        assert render_ovrtx._parse_nonnegative_int_env("TEST_OVRTX_LIMIT", 7) == 0

        for invalid in ("-1", "many"):
            monkeypatch.setenv("TEST_OVRTX_LIMIT", invalid)
            with pytest.raises(ValueError, match="must be a non-negative integer"):
                render_ovrtx._parse_nonnegative_int_env("TEST_OVRTX_LIMIT", 7)

    def test_linux_process_rss_bytes_reads_statm_and_handles_bad_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        proc_root = tmp_path / "proc"
        process_dir = proc_root / "42"
        process_dir.mkdir(parents=True)
        statm = process_dir / "statm"
        statm.write_text("100 3 2 1 0 0 0\n", encoding="utf-8")
        monkeypatch.setattr(render_ovrtx.os, "sysconf", lambda _name: 4096)

        assert render_ovrtx._linux_process_rss_bytes(42, proc_root=proc_root) == 12_288

        for invalid in ("", "100", "100 nope", "100 -1"):
            statm.write_text(invalid, encoding="utf-8")
            assert (
                render_ovrtx._linux_process_rss_bytes(42, proc_root=proc_root) is None
            )
        statm.unlink()
        assert render_ovrtx._linux_process_rss_bytes(42, proc_root=proc_root) is None

    def test_linux_process_rss_bytes_rejects_invalid_page_size(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        process_dir = tmp_path / "42"
        process_dir.mkdir()
        (process_dir / "statm").write_text("100 3\n", encoding="utf-8")
        monkeypatch.setattr(render_ovrtx.os, "sysconf", lambda _name: 0)

        assert render_ovrtx._linux_process_rss_bytes(42, proc_root=tmp_path) is None


class TestOvRTXDaemonLifecycle:
    """Test _OvRTXDaemon start / render / shutdown / crash-recovery."""

    @staticmethod
    def _make_fake_daemon_script(tmp_path):
        """Write a tiny Python script that speaks the daemon JSON protocol."""
        script = tmp_path / "fake_daemon.py"
        script.write_text(
            "import json, sys\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    if req.get("command") == "render":\n'
            '        manifest = [{"camera": c, "image_files": [], '
            '"sensor_files": {}, "frame_count": 0} '
            'for c in req["cameras"]]\n'
            '        sys.stdout.write(json.dumps({"status": "ok", '
            '"manifest": manifest}) + "\\n")\n'
            "        sys.stdout.flush()\n"
        )
        return str(script)

    @staticmethod
    def _render_params(tmp_path: Path) -> dict[str, Any]:
        return {
            "cameras": ["/Camera"],
            "usd_path": "/fake/stage.usdc",
            "fps": 24.0,
            "frames": [],
            "sensors": [],
            "output_dir": str(tmp_path),
            "product_paths": [],
        }

    def test_start_and_shutdown(self, tmp_path):
        """Daemon starts, is running, then shuts down cleanly."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.ensure_running()
        assert daemon._is_running()
        daemon.shutdown()
        assert not daemon._is_running()

    def test_completed_render_limit_recycles_before_next_request(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RENDERS", "1")
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RSS_BYTES", "0")
        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)

        daemon.render(self._render_params(tmp_path))
        first_pid = daemon._process.pid
        first_snapshot = daemon.lifecycle_snapshot()
        assert first_snapshot["daemon_completed_renders"] == 1
        assert (
            first_snapshot["daemon_pending_recycle_reason"] == "completed_render_limit"
        )

        daemon.render(self._render_params(tmp_path))
        second_snapshot = daemon.lifecycle_snapshot()
        assert daemon._process.pid != first_pid
        assert second_snapshot["daemon_completed_renders"] == 1
        assert second_snapshot["daemon_recycle_count"] == 1
        assert second_snapshot["daemon_last_recycle_reason"] == "completed_render_limit"
        daemon.shutdown()

    def test_rss_limit_recycles_pathological_daemon_before_count_limit(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("OVRTX_DAEMON_MAX_RENDERS", "100")
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RSS_BYTES", "100")
        rss_samples = iter([10, 10, 200, 200, 200, 10, 20, 20])
        monkeypatch.setattr(
            render_ovrtx,
            "_linux_process_rss_bytes",
            lambda _pid: next(rss_samples),
        )
        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)

        daemon.render(self._render_params(tmp_path))
        first_pid = daemon._process.pid
        assert (
            daemon.lifecycle_snapshot()["daemon_pending_recycle_reason"] == "rss_limit"
        )

        daemon.render(self._render_params(tmp_path))
        snapshot = daemon.lifecycle_snapshot()
        assert daemon._process.pid != first_pid
        assert snapshot["daemon_recycle_count"] == 1
        assert snapshot["daemon_last_recycle_reason"] == "rss_limit"
        assert snapshot["daemon_completed_renders"] == 1
        daemon.shutdown()

    def test_zero_limits_preserve_persistent_daemon(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RENDERS", "0")
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RSS_BYTES", "0")
        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)

        daemon.render(self._render_params(tmp_path))
        first_pid = daemon._process.pid
        daemon.render(self._render_params(tmp_path))
        snapshot = daemon.lifecycle_snapshot()

        assert daemon._process.pid == first_pid
        assert snapshot["daemon_completed_renders"] == 2
        assert snapshot["daemon_recycle_count"] == 0
        assert snapshot["daemon_pending_recycle_reason"] is None
        daemon.shutdown()

    def test_recycle_counters_reset_only_after_successful_restart(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RENDERS", "1")
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RSS_BYTES", "0")
        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.render(self._render_params(tmp_path))

        monkeypatch.setattr(
            daemon,
            "_start",
            unittest.mock.Mock(side_effect=RuntimeError("restart failed")),
        )
        with pytest.raises(RuntimeError, match="restart failed"):
            daemon.render(self._render_params(tmp_path))

        snapshot = daemon.lifecycle_snapshot()
        assert snapshot["daemon_completed_renders"] == 1
        assert snapshot["daemon_recycle_count"] == 0
        assert snapshot["daemon_pending_recycle_reason"] == "completed_render_limit"
        daemon.shutdown()

    def test_ensure_running_completes_a_previously_failed_recycle(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RENDERS", "1")
        monkeypatch.setenv("OVRTX_DAEMON_MAX_RSS_BYTES", "0")
        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.render(self._render_params(tmp_path))

        original_start = daemon._start
        monkeypatch.setattr(
            daemon,
            "_start",
            unittest.mock.Mock(side_effect=RuntimeError("restart failed")),
        )
        with pytest.raises(RuntimeError, match="restart failed"):
            daemon.render(self._render_params(tmp_path))

        monkeypatch.setattr(daemon, "_start", original_start)
        daemon.ensure_running()
        replacement_pid = daemon._process.pid
        snapshot = daemon.lifecycle_snapshot()
        assert snapshot["daemon_completed_renders"] == 0
        assert snapshot["daemon_recycle_count"] == 1
        assert snapshot["daemon_last_recycle_reason"] == "completed_render_limit"
        assert snapshot["daemon_pending_recycle_reason"] is None

        daemon.render(self._render_params(tmp_path))
        assert daemon._process.pid == replacement_pid
        daemon.shutdown()

    def test_start_passes_site_dir_for_explicit_venv(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """Daemon startup should not depend on a stale parent-process env var."""
        from world_understanding.functions.graphics import render_ovrtx

        custom_venv = tmp_path / "custom_ovrtx_venv"
        expected_site_dir = render_ovrtx._ovrtx_site_packages_candidates(custom_venv)[0]
        (expected_site_dir / "ovrtx").mkdir(parents=True)
        monkeypatch.setenv("_WU_OVRTX_SITE_DIR", str(tmp_path / "stale"))

        captured_env: dict[str, str] = {}

        class FakeProcess:
            pid = 12345
            stdin = None
            stdout = None
            stderr: list[str] = []

            def poll(self):
                return None

        def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
            captured_env.update(kwargs["env"])
            return FakeProcess()

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        monkeypatch.setattr(render_ovrtx.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            render_ovrtx._OvRTXDaemon,
            "_read_stdout_line",
            lambda self, timeout_s, phase: json.dumps({"status": "ready"}),
        )

        daemon = _OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python"),
            daemon_script_path=str(tmp_path / "daemon.py"),
            ovrtx_venv_dir=custom_venv,
        )
        daemon.ensure_running()

        assert captured_env["_WU_OVRTX_SITE_DIR"] == str(expected_site_dir)

    def test_start_omits_site_dir_for_unmanaged_python(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """Custom wrappers should not inherit a stale/default ovrtx site dir."""
        from world_understanding.functions.graphics import render_ovrtx

        monkeypatch.setenv("_WU_OVRTX_SITE_DIR", str(tmp_path / "stale_default"))
        captured_env: dict[str, str] = {}

        class FakeProcess:
            pid = 12345
            stdin = None
            stdout = None
            stderr: list[str] = []

            def poll(self):
                return None

        def fake_popen(*args: Any, **kwargs: Any) -> FakeProcess:
            captured_env.update(kwargs["env"])
            return FakeProcess()

        monkeypatch.setattr(render_ovrtx.atexit, "register", lambda func: None)
        monkeypatch.setattr(render_ovrtx.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(
            render_ovrtx._OvRTXDaemon,
            "_read_stdout_line",
            lambda self, timeout_s, phase: json.dumps({"status": "ready"}),
        )

        daemon = _OvRTXDaemon(
            ovrtx_python=str(tmp_path / "python-wrapper"),
            daemon_script_path=str(tmp_path / "daemon.py"),
        )
        daemon.ensure_running()

        assert "_WU_OVRTX_SITE_DIR" not in captured_env

    def test_render_returns_manifest(self, tmp_path):
        """daemon.render() returns the manifest list."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.ensure_running()

        manifest = daemon.render(
            {
                "cameras": ["/Camera"],
                "usd_path": "/fake/stage.usdc",
                "fps": 24.0,
                "frames": [],
                "sensors": [],
                "output_dir": str(tmp_path),
                "product_paths": [],
            }
        )

        assert len(manifest) == 1
        assert manifest[0]["camera"] == "/Camera"
        daemon.shutdown()

    def test_render_preserves_buffered_stdout_after_first_line(self, tmp_path):
        """Extra stdout bytes after a newline should feed the next response."""
        import sys

        script = tmp_path / "buffered_daemon.py"
        script.write_text(
            "import json, sys\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "sent_pair = False\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    if req.get("command") == "render" and not sent_pair:\n'
            '        first = {"status": "ok", "manifest": [{"camera": "first"}]}\n'
            '        second = {"status": "ok", "manifest": [{"camera": "second"}]}\n'
            '        sys.stdout.write(json.dumps(first) + "\\n" + '
            'json.dumps(second) + "\\n")\n'
            "        sys.stdout.flush()\n"
            "        sent_pair = True\n"
            '    elif req.get("command") == "render":\n'
            '        late = {"status": "ok", "manifest": [{"camera": "late"}]}\n'
            '        sys.stdout.write(json.dumps(late) + "\\n")\n'
            "        sys.stdout.flush()\n"
        )

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        daemon.ensure_running()

        params = {
            "cameras": ["/Camera"],
            "usd_path": "/fake/stage.usdc",
            "fps": 24.0,
            "frames": [],
            "sensors": [],
            "output_dir": str(tmp_path),
            "product_paths": [],
        }

        assert daemon.render(params)[0]["camera"] == "first"
        assert daemon.render(params)[0]["camera"] == "second"
        daemon.shutdown()

    def test_auto_restart_on_crash(self, tmp_path):
        """Daemon auto-restarts if the subprocess dies between calls."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.ensure_running()

        # Kill the subprocess to simulate a crash
        daemon._process.kill()
        daemon._process.wait()
        assert not daemon._is_running()

        # Next render should auto-restart
        manifest = daemon.render(
            {
                "cameras": ["/Cam"],
                "usd_path": "/fake/stage.usdc",
                "fps": 24.0,
                "frames": [],
                "sensors": [],
                "output_dir": str(tmp_path),
                "product_paths": [],
            }
        )
        assert len(manifest) == 1
        assert daemon._is_running()
        daemon.shutdown()

    def test_shutdown_when_not_running_is_noop(self, tmp_path):
        """Calling shutdown() when the daemon is not started does not raise."""
        import sys

        script = self._make_fake_daemon_script(tmp_path)
        daemon = _OvRTXDaemon(ovrtx_python=sys.executable, daemon_script_path=script)
        daemon.shutdown()  # should not raise

    def test_render_error_propagates(self, tmp_path):
        """Daemon error responses become RuntimeError in the caller."""
        import sys

        # Script that always returns an error for render commands
        script = tmp_path / "err_daemon.py"
        script.write_text(
            "import json, sys\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    sys.stdout.write(json.dumps({"status": "error", '
            '"error": "boom"}) + "\\n")\n'
            "    sys.stdout.flush()\n"
        )

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        daemon.ensure_running()

        with pytest.raises(RuntimeError, match="boom"):
            daemon.render(
                {
                    "cameras": [],
                    "usd_path": "/fake/stage.usdc",
                    "fps": 24.0,
                    "frames": [],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )
        daemon.shutdown()

    def test_start_timeout_kills_wedged_daemon(self, tmp_path, monkeypatch):
        """A daemon that never sends ready should time out instead of hanging."""
        import sys

        script = tmp_path / "startup_hang_daemon.py"
        script.write_text("import time\ntime.sleep(60)\n")
        monkeypatch.setenv("OVRTX_DAEMON_START_TIMEOUT", "0.05")

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        with pytest.raises(TimeoutError, match="startup timed out"):
            daemon.ensure_running()

        assert not daemon._is_running()

    def test_start_timeout_kills_daemon_after_partial_line(self, tmp_path, monkeypatch):
        """Partial stdout without newline must not bypass the startup timeout."""
        import sys

        script = tmp_path / "startup_partial_hang_daemon.py"
        script.write_text(
            "import sys, time\n"
            "sys.stdout.write('{\"status\"')\n"
            "sys.stdout.flush()\n"
            "time.sleep(60)\n"
        )
        monkeypatch.setenv("OVRTX_DAEMON_START_TIMEOUT", "0.05")

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        with pytest.raises(TimeoutError, match="startup timed out"):
            daemon.ensure_running()

        assert not daemon._is_running()

    def test_render_timeout_kills_wedged_daemon(self, tmp_path, monkeypatch):
        """A daemon that stops responding should release the render lock path."""
        import sys

        script = tmp_path / "render_hang_daemon.py"
        script.write_text(
            "import json, sys, time\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    if req.get("command") == "render":\n'
            "        time.sleep(60)\n"
        )
        monkeypatch.setenv("OVRTX_DAEMON_RENDER_TIMEOUT", "0.05")

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        daemon.ensure_running()

        with pytest.raises(TimeoutError, match="render timed out"):
            daemon.render(
                {
                    "cameras": ["/Camera"],
                    "usd_path": "/fake/stage.usdc",
                    "fps": 24.0,
                    "frames": [],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )

        assert not daemon._is_running()

    def test_render_timeout_kills_daemon_after_partial_line(
        self, tmp_path, monkeypatch
    ):
        """Partial render response without newline must still hit the deadline."""
        import sys

        script = tmp_path / "render_partial_hang_daemon.py"
        script.write_text(
            "import json, sys, time\n"
            'sys.stdout.write(json.dumps({"status": "ready"}) + "\\n")\n'
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    req = json.loads(line.strip())\n"
            '    if req.get("command") == "shutdown":\n'
            "        break\n"
            '    if req.get("command") == "render":\n'
            '        sys.stdout.write(\'{"status": "ok"\')\n'
            "        sys.stdout.flush()\n"
            "        time.sleep(60)\n"
        )
        monkeypatch.setenv("OVRTX_DAEMON_RENDER_TIMEOUT", "0.05")

        daemon = _OvRTXDaemon(
            ovrtx_python=sys.executable, daemon_script_path=str(script)
        )
        daemon.ensure_running()

        with pytest.raises(TimeoutError, match="render timed out"):
            daemon.render(
                {
                    "cameras": ["/Camera"],
                    "usd_path": "/fake/stage.usdc",
                    "fps": 24.0,
                    "frames": [],
                    "sensors": [],
                    "output_dir": str(tmp_path),
                    "product_paths": [],
                }
            )

        assert not daemon._is_running()


class TestOvRTXTimeSampledSupport:
    """Unit coverage for the OvRTX time-sampled USD handoff."""

    def test_worker_paths_reset_between_usd_time_and_visibility_writes(self):
        worker_loop = _WORKER_SCRIPT[_WORKER_SCRIPT.index("for frame_num in frames:") :]
        daemon_loop = _DAEMON_SCRIPT[_DAEMON_SCRIPT.index("for frame_num in frames:") :]

        assert worker_loop.index("renderer.update_from_usd_time") < worker_loop.index(
            "renderer.reset()"
        )
        assert worker_loop.index("renderer.reset()") < worker_loop.index(
            "renderer.write_attribute"
        )
        assert daemon_loop.index("renderer.update_from_usd_time") < daemon_loop.index(
            "renderer.reset()"
        )
        assert daemon_loop.index("renderer.reset()") < daemon_loop.index(
            "renderer.write_attribute"
        )

    def test_time_sampled_attrs_survive_export_except_visibility(
        self, monkeypatch, tmp_path
    ):
        from PIL import Image
        from pxr import Usd, UsdGeom

        stage = _make_time_sampled_compliance_stage()
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["params"] = params

            exported = Usd.Stage.Open(params["usd_path"])
            moving = exported.GetPrimAtPath("/World/MovingCube")
            camera = exported.GetPrimAtPath("/AnimatedCamera")
            visibility_cube = exported.GetPrimAtPath("/World/VisibilityCube")

            captured["moving_translate_samples"] = moving.GetAttribute(
                "xformOp:translate"
            ).GetNumTimeSamples()
            captured["moving_color_samples"] = (
                UsdGeom.Gprim(moving).GetDisplayColorAttr().GetNumTimeSamples()
            )
            captured["camera_translate_samples"] = camera.GetAttribute(
                "xformOp:translate"
            ).GetNumTimeSamples()
            captured["visibility_samples"] = (
                UsdGeom.Imageable(visibility_cube)
                .GetVisibilityAttr()
                .GetNumTimeSamples()
            )

            frame_zero = Usd.Stage.Open(params["frame_usd_paths"]["0"])
            frame_zero_moving = frame_zero.GetPrimAtPath("/World/MovingCube")
            frame_zero_visibility = frame_zero.GetPrimAtPath("/World/VisibilityCube")
            captured["frame_zero_color_samples"] = (
                UsdGeom.Gprim(frame_zero_moving)
                .GetDisplayColorAttr()
                .GetNumTimeSamples()
            )
            captured["frame_zero_visibility_samples"] = (
                UsdGeom.Imageable(frame_zero_visibility)
                .GetVisibilityAttr()
                .GetNumTimeSamples()
            )
            captured["frame_zero_visibility"] = (
                UsdGeom.Imageable(frame_zero_visibility).GetVisibilityAttr().Get()
            )

            output_dir = Path(params["output_dir"])
            manifest = []
            for camera_idx, camera_path in enumerate(params["cameras"]):
                image_files = []
                for frame_num in params["frames"]:
                    filename = f"cam{camera_idx}_f{frame_num}.png"
                    Image.new(
                        "RGBA",
                        (params["image_width"], params["image_height"]),
                        (frame_num * 50, 32, 64, 255),
                    ).save(output_dir / filename)
                    image_files.append(filename)
                manifest.append(
                    {
                        "camera": camera_path,
                        "image_files": image_files,
                        "sensor_files": {},
                        "frame_count": len(image_files),
                    }
                )
            (output_dir / "manifest.json").write_text(json.dumps(manifest))
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera", "/AnimatedCamera"],
            frames="0:2",
            num_sensor_updates=1,
        )

        params = captured["params"]
        assert params["fps"] == 24.0
        assert params["frames"] == [0, 1, 2]
        assert set(params["frame_usd_paths"]) == {"0", "1", "2"}
        assert params["visibility_updates"] == {
            "0.0": {"/World/VisibilityCube": "invisible"},
            "1.0": {"/World/VisibilityCube": "inherited"},
            "2.0": {"/World/VisibilityCube": "invisible"},
        }
        assert captured["moving_translate_samples"] == 3
        assert captured["moving_color_samples"] == 3
        assert captured["camera_translate_samples"] == 3
        assert captured["visibility_samples"] == 0
        assert captured["frame_zero_color_samples"] == 0
        assert captured["frame_zero_visibility_samples"] == 0
        assert captured["frame_zero_visibility"] == UsdGeom.Tokens.invisible
        assert result["successful_cameras"] == 2

        restored_visibility = UsdGeom.Imageable(
            stage.GetPrimAtPath("/World/VisibilityCube")
        ).GetVisibilityAttr()
        assert restored_visibility.GetNumTimeSamples() == 3
        assert restored_visibility.Get(Usd.TimeCode(0.0)) == UsdGeom.Tokens.invisible
        assert restored_visibility.Get(Usd.TimeCode(1.0)) == UsdGeom.Tokens.inherited
        assert restored_visibility.Get(Usd.TimeCode(2.0)) == UsdGeom.Tokens.invisible

    def test_experimental_native_visibility_keeps_samples_and_skips_overlays(
        self, monkeypatch
    ):
        from PIL import Image
        from pxr import Usd, UsdGeom

        stage = _make_visibility_only_stage()
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["params"] = params

            exported = Usd.Stage.Open(params["usd_path"])
            for frame in range(3):
                prim_path = f"/World/Part{frame}"
                visibility = UsdGeom.Imageable(
                    exported.GetPrimAtPath(prim_path)
                ).GetVisibilityAttr()
                captured[f"{prim_path}_sample_count"] = visibility.GetNumTimeSamples()
                for sample_frame in range(3):
                    captured[f"{prim_path}_frame_{sample_frame}"] = visibility.Get(
                        Usd.TimeCode(float(sample_frame))
                    )

            output_dir = Path(params["output_dir"])
            image_files = []
            for frame_num in params["frames"]:
                filename = f"cam0_f{frame_num}.png"
                Image.new(
                    "RGBA",
                    (params["image_width"], params["image_height"]),
                    (frame_num * 50, 32, 64, 255),
                ).save(output_dir / filename)
                image_files.append(filename)
            manifest = [
                {
                    "camera": params["cameras"][0],
                    "image_files": image_files,
                    "sensor_files": {},
                    "frame_count": len(image_files),
                }
            ]
            (output_dir / "manifest.json").write_text(json.dumps(manifest))
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setenv("WU_OVRTX_EXPERIMENTAL_NATIVE_VISIBILITY", "1")
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0:2",
            num_sensor_updates=1,
        )

        params = captured["params"]
        assert params["frames"] == [0, 1, 2]
        assert params["frame_usd_paths"] == {}
        assert params["visibility_updates"] == {}
        for frame in range(3):
            prim_path = f"/World/Part{frame}"
            assert captured[f"{prim_path}_sample_count"] == 3
            for sample_frame in range(3):
                expected = (
                    UsdGeom.Tokens.inherited
                    if sample_frame == frame
                    else UsdGeom.Tokens.invisible
                )
                assert captured[f"{prim_path}_frame_{sample_frame}"] == expected
        assert result["successful_cameras"] == 1

        for frame in range(3):
            visibility = UsdGeom.Imageable(
                stage.GetPrimAtPath(f"/World/Part{frame}")
            ).GetVisibilityAttr()
            assert visibility.GetNumTimeSamples() == 3
            for sample_frame in range(3):
                expected = (
                    UsdGeom.Tokens.inherited
                    if sample_frame == frame
                    else UsdGeom.Tokens.invisible
                )
                assert visibility.Get(Usd.TimeCode(float(sample_frame))) == expected

    def test_display_color_only_uses_frame_overlay_by_default(self, monkeypatch):
        from PIL import Image
        from pxr import Usd, UsdGeom

        stage = _make_display_color_only_stage()
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["params"] = params

            exported = Usd.Stage.Open(params["usd_path"])
            color_cube = exported.GetPrimAtPath("/World/ColorCube")
            captured["exported_color_samples"] = (
                UsdGeom.Gprim(color_cube).GetDisplayColorAttr().GetNumTimeSamples()
            )

            frame_zero = Usd.Stage.Open(params["frame_usd_paths"]["0"])
            frame_zero_cube = frame_zero.GetPrimAtPath("/World/ColorCube")
            frame_zero_color = UsdGeom.Gprim(frame_zero_cube).GetDisplayColorAttr()
            captured["frame_zero_color_samples"] = frame_zero_color.GetNumTimeSamples()
            captured["frame_zero_color"] = tuple(frame_zero_color.Get()[0])

            output_dir = Path(params["output_dir"])
            image_files = []
            for frame_num in params["frames"]:
                filename = f"cam0_f{frame_num}.png"
                Image.new(
                    "RGBA",
                    (params["image_width"], params["image_height"]),
                    (frame_num * 50, 32, 64, 255),
                ).save(output_dir / filename)
                image_files.append(filename)
            manifest = [
                {
                    "camera": params["cameras"][0],
                    "image_files": image_files,
                    "sensor_files": {},
                    "frame_count": len(image_files),
                }
            ]
            (output_dir / "manifest.json").write_text(json.dumps(manifest))
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0:2",
            num_sensor_updates=1,
        )

        params = captured["params"]
        assert set(params["frame_usd_paths"]) == {"0", "1", "2"}
        assert params["visibility_updates"] == {}
        assert captured["exported_color_samples"] == 3
        assert captured["frame_zero_color_samples"] == 0
        assert captured["frame_zero_color"] == pytest.approx((1.0, 0.05, 0.05))
        assert result["successful_cameras"] == 1

    def test_native_displaycolor_probe_keeps_time_samples_without_color_overlays(
        self, monkeypatch
    ):
        from PIL import Image
        from pxr import Usd, UsdGeom

        stage = _make_display_color_only_stage()
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["params"] = params

            exported = Usd.Stage.Open(params["usd_path"])
            color_cube = exported.GetPrimAtPath("/World/ColorCube")
            color_attr = UsdGeom.Gprim(color_cube).GetDisplayColorAttr()
            captured["exported_color_samples"] = color_attr.GetNumTimeSamples()
            captured["exported_frame_0_color"] = tuple(
                color_attr.Get(Usd.TimeCode(0.0))[0]
            )
            captured["exported_frame_1_color"] = tuple(
                color_attr.Get(Usd.TimeCode(1.0))[0]
            )
            captured["exported_frame_2_color"] = tuple(
                color_attr.Get(Usd.TimeCode(2.0))[0]
            )

            output_dir = Path(params["output_dir"])
            image_files = []
            for frame_num in params["frames"]:
                filename = f"cam0_f{frame_num}.png"
                Image.new(
                    "RGBA",
                    (params["image_width"], params["image_height"]),
                    (frame_num * 50, 32, 64, 255),
                ).save(output_dir / filename)
                image_files.append(filename)
            manifest = [
                {
                    "camera": params["cameras"][0],
                    "image_files": image_files,
                    "sensor_files": {},
                    "frame_count": len(image_files),
                }
            ]
            (output_dir / "manifest.json").write_text(json.dumps(manifest))
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setenv("WU_OVRTX_EXPERIMENTAL_NATIVE_DISPLAYCOLOR", "1")
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0:2",
            num_sensor_updates=1,
        )

        params = captured["params"]
        assert params["frame_usd_paths"] == {}
        assert params["visibility_updates"] == {}
        assert captured["exported_color_samples"] == 3
        assert captured["exported_frame_0_color"] == pytest.approx((1.0, 0.05, 0.05))
        assert captured["exported_frame_1_color"] == pytest.approx((0.05, 1.0, 0.05))
        assert captured["exported_frame_2_color"] == pytest.approx((0.05, 0.05, 1.0))
        assert result["successful_cameras"] == 1

    def test_native_displaycolor_probe_composes_with_visibility_overlays(
        self, monkeypatch
    ):
        from PIL import Image
        from pxr import Usd, UsdGeom

        stage = _make_time_sampled_compliance_stage()
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["params"] = params

            exported = Usd.Stage.Open(params["usd_path"])
            moving = exported.GetPrimAtPath("/World/MovingCube")
            moving_color = UsdGeom.Gprim(moving).GetDisplayColorAttr()
            captured["exported_color_samples"] = moving_color.GetNumTimeSamples()

            frame_zero = Usd.Stage.Open(params["frame_usd_paths"]["0"])
            frame_zero_moving = frame_zero.GetPrimAtPath("/World/MovingCube")
            frame_zero_color = UsdGeom.Gprim(frame_zero_moving).GetDisplayColorAttr()
            captured["frame_zero_composed_color_samples"] = (
                frame_zero_color.GetNumTimeSamples()
            )
            captured["frame_zero_composed_color"] = tuple(
                frame_zero_color.Get(Usd.TimeCode(0.0))[0]
            )

            frame_zero_root_layer = frame_zero.GetRootLayer()
            frame_zero_overlay_path = frame_zero_root_layer.ComputeAbsolutePath(
                frame_zero_root_layer.subLayerPaths[0]
            )
            frame_zero_overlay = Usd.Stage.Open(frame_zero_overlay_path)
            captured["frame_zero_overlay_has_moving_cube"] = (
                frame_zero_overlay.GetPrimAtPath("/World/MovingCube").IsValid()
            )

            frame_zero_visibility_cube = frame_zero_overlay.GetPrimAtPath(
                "/World/VisibilityCube"
            )
            frame_zero_visibility = UsdGeom.Imageable(
                frame_zero_visibility_cube
            ).GetVisibilityAttr()
            captured["frame_zero_overlay_visibility_samples"] = (
                frame_zero_visibility.GetNumTimeSamples()
            )
            captured["frame_zero_overlay_visibility"] = frame_zero_visibility.Get()

            output_dir = Path(params["output_dir"])
            image_files = []
            for frame_num in params["frames"]:
                filename = f"cam0_f{frame_num}.png"
                Image.new(
                    "RGBA",
                    (params["image_width"], params["image_height"]),
                    (frame_num * 50, 32, 64, 255),
                ).save(output_dir / filename)
                image_files.append(filename)
            manifest = [
                {
                    "camera": params["cameras"][0],
                    "image_files": image_files,
                    "sensor_files": {},
                    "frame_count": len(image_files),
                }
            ]
            (output_dir / "manifest.json").write_text(json.dumps(manifest))
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setenv("WU_OVRTX_EXPERIMENTAL_NATIVE_DISPLAYCOLOR", "1")
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0:2",
            num_sensor_updates=1,
        )

        params = captured["params"]
        assert set(params["frame_usd_paths"]) == {"0", "1", "2"}
        assert params["visibility_updates"] == {
            "0.0": {"/World/VisibilityCube": "invisible"},
            "1.0": {"/World/VisibilityCube": "inherited"},
            "2.0": {"/World/VisibilityCube": "invisible"},
        }
        assert captured["exported_color_samples"] == 3
        assert captured["frame_zero_composed_color_samples"] == 3
        assert captured["frame_zero_composed_color"] == pytest.approx((1.0, 0.05, 0.05))
        assert captured["frame_zero_overlay_has_moving_cube"] is False
        assert captured["frame_zero_overlay_visibility_samples"] == 0
        assert captured["frame_zero_overlay_visibility"] == UsdGeom.Tokens.invisible
        assert result["successful_cameras"] == 1

    def test_render_all_cameras_preview_surface_target_adds_fallback_to_export(
        self, monkeypatch
    ):
        from PIL import Image
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Camera.Define(stage, "/Camera")
        UsdLux.DomeLight.Define(stage, "/World/Light")
        material = UsdShade.Material.Define(stage, "/World/Looks/Gold")
        material.GetPrim().CreateAttribute(
            "inputs:base_color",
            Sdf.ValueTypeNames.Color3f,
        ).Set((1.0, 0.766, 0.336))

        shader = UsdShade.Shader.Define(
            stage,
            "/World/Looks/Gold/open_pbr_surface_surfaceshader",
        )
        shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
        material.CreateSurfaceOutput("mtlx").ConnectToSource(
            shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
        )

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["material_target"] = params["material_target"]
            exported_stage = Usd.Stage.Open(params["usd_path"])
            exported_material = UsdShade.Material(
                exported_stage.GetPrimAtPath("/World/Looks/Gold"),
            )
            surface = exported_material.GetSurfaceOutput()
            sources, _ = surface.GetConnectedSources()
            preview_shader = UsdShade.Shader(sources[0].source.GetPrim())
            captured["shader_id"] = preview_shader.GetIdAttr().Get()
            captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
                "mtlx",
            ).HasConnectedSource()

            output_dir = Path(params["output_dir"])
            image_name = "camera_0.png"
            Image.new("RGBA", (16, 16), (255, 192, 84, 255)).save(
                output_dir / image_name,
            )
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "camera": "/Camera",
                            "image_files": [image_name],
                            "sensor_files": {},
                            "frame_count": 1,
                        }
                    ],
                ),
            )
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0",
            num_sensor_updates=1,
            material_target="preview_surface",
        )

        assert result["successful_cameras"] == 1
        assert captured == {
            "material_target": "preview_surface",
            "shader_id": "UsdPreviewSurface",
            "mtlx_connected": False,
        }

    def test_render_all_cameras_preview_surface_target_adds_fallback_for_sublayered_material(
        self, monkeypatch, tmp_path
    ):
        from PIL import Image
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

        material_layer_path = tmp_path / "materials.usda"
        material_stage = Usd.Stage.CreateNew(str(material_layer_path))
        material = UsdShade.Material.Define(material_stage, "/World/Looks/Gold")
        material.GetPrim().CreateAttribute(
            "inputs:base_color",
            Sdf.ValueTypeNames.Color3f,
        ).Set((1.0, 0.766, 0.336))
        shader = UsdShade.Shader.Define(
            material_stage,
            "/World/Looks/Gold/open_pbr_surface_surfaceshader",
        )
        shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
        material.CreateSurfaceOutput("mtlx").ConnectToSource(
            shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
        )
        material_stage.GetRootLayer().Save()

        scene_layer_path = tmp_path / "scene.usda"
        scene_stage = Usd.Stage.CreateNew(str(scene_layer_path))
        UsdGeom.Camera.Define(scene_stage, "/Camera")
        UsdLux.DomeLight.Define(scene_stage, "/World/Light")
        scene_stage.GetRootLayer().subLayerPaths.append(str(material_layer_path))
        scene_stage.GetRootLayer().Save()

        stage = Usd.Stage.Open(str(scene_layer_path))
        assert stage is not None
        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["material_target"] = params["material_target"]
            exported_stage = Usd.Stage.Open(params["usd_path"])
            exported_material = UsdShade.Material(
                exported_stage.GetPrimAtPath("/World/Looks/Gold"),
            )
            surface = exported_material.GetSurfaceOutput()
            sources, _ = surface.GetConnectedSources()
            preview_shader = UsdShade.Shader(sources[0].source.GetPrim())
            captured["shader_id"] = preview_shader.GetIdAttr().Get()
            captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
                "mtlx",
            ).HasConnectedSource()

            output_dir = Path(params["output_dir"])
            image_name = "camera_0.png"
            Image.new("RGBA", (16, 16), (255, 192, 84, 255)).save(
                output_dir / image_name,
            )
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "camera": "/Camera",
                            "image_files": [image_name],
                            "sensor_files": {},
                            "frame_count": 1,
                        }
                    ],
                ),
            )
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0",
            num_sensor_updates=1,
            material_target="preview_surface",
        )

        assert result["successful_cameras"] == 1
        assert captured == {
            "material_target": "preview_surface",
            "shader_id": "UsdPreviewSurface",
            "mtlx_connected": False,
        }

    @pytest.mark.parametrize("material_target", [None, "auto", "openpbr_materialx"])
    def test_render_all_cameras_preserving_targets_do_not_add_preview_fallback(
        self, monkeypatch, material_target
    ):
        from PIL import Image
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Camera.Define(stage, "/Camera")
        UsdLux.DomeLight.Define(stage, "/World/Light")
        material = UsdShade.Material.Define(stage, "/World/Looks/Gold")

        shader = UsdShade.Shader.Define(
            stage,
            "/World/Looks/Gold/open_pbr_surface_surfaceshader",
        )
        shader.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
        material.CreateSurfaceOutput("mtlx").ConnectToSource(
            shader.CreateOutput("out", Sdf.ValueTypeNames.Token),
        )

        captured: dict[str, object] = {}

        def fake_run(cmd, **kwargs):
            params = _worker_params_from_command(cmd)
            captured["material_target"] = params["material_target"]
            exported_stage = Usd.Stage.Open(params["usd_path"])
            exported_material = UsdShade.Material(
                exported_stage.GetPrimAtPath("/World/Looks/Gold"),
            )
            surface = exported_material.GetSurfaceOutput()
            captured["universal_connected"] = bool(
                surface and surface.HasConnectedSource(),
            )
            captured["mtlx_connected"] = exported_material.GetSurfaceOutput(
                "mtlx",
            ).HasConnectedSource()
            captured["has_preview_child"] = exported_stage.GetPrimAtPath(
                "/World/Looks/Gold/OVRTXPreviewSurface",
            ).IsValid()

            output_dir = Path(params["output_dir"])
            image_name = "camera_0.png"
            Image.new("RGBA", (16, 16), (255, 192, 84, 255)).save(
                output_dir / image_name,
            )
            (output_dir / "manifest.json").write_text(
                json.dumps(
                    [
                        {
                            "camera": "/Camera",
                            "image_files": [image_name],
                            "sensor_files": {},
                            "frame_count": 1,
                        }
                    ],
                ),
            )
            return unittest.mock.Mock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            fake_run,
        )

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        render_kwargs = {}
        if material_target is not None:
            render_kwargs["material_target"] = material_target
        result = render_all_cameras(
            stage=stage,
            image_width=16,
            image_height=16,
            cameras=["/Camera"],
            frames="0",
            num_sensor_updates=1,
            **render_kwargs,
        )

        assert result["successful_cameras"] == 1
        expected_material_target = material_target or "auto"
        assert captured == {
            "material_target": expected_material_target,
            "universal_connected": False,
            "mtlx_connected": True,
            "has_preview_child": False,
        }


class TestRenderAllCamerasBackwardCompat:
    """Verify render_all_cameras(daemon=None) still uses subprocess.run."""

    def test_daemon_none_uses_subprocess(self, monkeypatch):
        """When daemon=None, the old subprocess.run path should be taken."""
        monkeypatch.setenv(
            "WU_OVRTX_DEFAULT_HDRI", "https://example.invalid/StinsonBeach.hdr"
        )
        monkeypatch.delenv("DISPLAY", raising=False)
        with (
            unittest.mock.patch(
                "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
                return_value="/fake/python",
            ),
            unittest.mock.patch(
                "world_understanding.functions.graphics.render_ovrtx._ovrtx_site_dir_env_for_python",
                return_value="/fake/site-packages",
            ),
            unittest.mock.patch(
                "world_understanding.functions.graphics.render_ovrtx.subprocess.run",
            ) as mock_run,
        ):
            # Make subprocess.run return a failure so we can catch it quickly
            mock_run.return_value = unittest.mock.Mock(
                returncode=1, stdout="", stderr="test"
            )
            from pxr import Usd

            stage = Usd.Stage.CreateInMemory()
            from world_understanding.functions.graphics.render_ovrtx import (
                render_all_cameras,
            )

            with pytest.raises(RuntimeError, match="OvRTX subprocess failed"):
                render_all_cameras(stage=stage, daemon=None)

            mock_run.assert_called_once()
            assert (
                mock_run.call_args.kwargs["env"]["_WU_OVRTX_SITE_DIR"]
                == "/fake/site-packages"
            )
            assert mock_run.call_args.kwargs["env"]["DISPLAY"] == ":0"

    def test_old_positional_daemon_and_base_dir_tail_still_binds(
        self, tmp_path, monkeypatch
    ):
        """Old positional callers keep binding daemon/base_dir after render_mode."""
        from PIL import Image
        from pxr import Usd

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        monkeypatch.setenv(
            "WU_OVRTX_DEFAULT_HDRI", "https://example.invalid/StinsonBeach.hdr"
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: str(
                tmp_path / "ovrtx_venv" / "Scripts" / "python.exe"
            ),
        )

        stage = Usd.Stage.CreateInMemory()
        fake_daemon = unittest.mock.Mock()

        def fake_render(params):
            output_dir = Path(params["output_dir"])
            image_name = "camera0_frame0.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(output_dir / image_name)
            return [
                {
                    "camera": "/Camera",
                    "image_files": [image_name],
                    "sensor_files": {},
                    "frame_count": 1,
                }
            ]

        fake_daemon.render.side_effect = fake_render

        result = render_all_cameras(
            stage,
            4,
            4,
            ["/Camera"],
            "0",
            None,
            None,
            "warn",
            None,
            1,
            "rt2",
            fake_daemon,
            tmp_path,
        )

        fake_daemon.ensure_running.assert_called_once_with()
        fake_daemon.render.assert_called_once()
        params = fake_daemon.render.call_args.args[0]
        assert params["material_target"] == "auto"
        assert result["successful_cameras"] == 1
        assert result["results"][0]["images"][0].size == (4, 4)

    def test_daemon_payload_includes_explicit_material_target(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Persistent daemon requests should receive the normalized material target."""
        from PIL import Image
        from pxr import Usd

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        monkeypatch.setenv(
            "WU_OVRTX_DEFAULT_HDRI", "https://example.invalid/StinsonBeach.hdr"
        )
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: str(
                tmp_path / "ovrtx_venv" / "Scripts" / "python.exe"
            ),
        )

        stage = Usd.Stage.CreateInMemory()
        fake_daemon = unittest.mock.Mock()

        def fake_render(params):
            output_dir = Path(params["output_dir"])
            image_name = "camera0_frame0.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(output_dir / image_name)
            return [
                {
                    "camera": "/Camera",
                    "image_files": [image_name],
                    "sensor_files": {},
                    "frame_count": 1,
                }
            ]

        fake_daemon.render.side_effect = fake_render

        result = render_all_cameras(
            stage=stage,
            image_width=4,
            image_height=4,
            cameras=["/Camera"],
            frames="0",
            num_sensor_updates=1,
            daemon=fake_daemon,
            base_dir=tmp_path,
            material_target="openpbr",
        )

        fake_daemon.ensure_running.assert_called_once_with()
        fake_daemon.render.assert_called_once()
        params = fake_daemon.render.call_args.args[0]
        assert params["material_target"] == "openpbr_materialx"
        assert result["successful_cameras"] == 1

    @pytest.mark.parametrize(
        "sample_kwargs",
        [
            {"rtx_pt_samples_per_pixel": 512},
            {"rtx_rt_accumulation_limit": 512},
        ],
    )
    def test_probe_sample_attributes_rejected_with_daemon(
        self, monkeypatch, sample_kwargs
    ):
        """Probe-only sample attributes must not be silently ignored by daemon mode."""
        from pxr import Usd

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        stage = Usd.Stage.CreateInMemory()
        fake_daemon = unittest.mock.Mock()
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            unittest.mock.Mock(side_effect=AssertionError("should not provision")),
        )

        with pytest.raises(ValueError, match="unsupported with the persistent OvRTX"):
            render_all_cameras(stage=stage, daemon=fake_daemon, **sample_kwargs)

        fake_daemon.ensure_running.assert_not_called()
        fake_daemon.render.assert_not_called()


class TestOvRTXBackendCreatesDaemon:
    """Verify OvRTXRenderingBackend creates and shuts down a daemon."""

    def test_backend_creates_daemon(self, tmp_path, monkeypatch):
        """OvRTXRenderingBackend.__init__ should create a _OvRTXDaemon."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()
        runtime_dir = tmp_path / "runtime"
        monkeypatch.setenv("WU_OVRTX_RUNTIME_DIR", str(runtime_dir))
        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            backend = OvRTXRenderingBackend(ovrtx_venv_dir=str(venv_dir))
            assert hasattr(backend, "_daemon")
            assert isinstance(backend._daemon, _OvRTXDaemon)
            daemon_script_path = Path(backend._daemon._daemon_script_path)
            assert daemon_script_path.parent == runtime_dir
            assert daemon_script_path.is_file()
            assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
            assert stat.S_IMODE(daemon_script_path.stat().st_mode) == 0o600
            backend.__del__()
            assert not daemon_script_path.exists()

    def test_backend_uses_private_runtime_dir_by_default(self, tmp_path, monkeypatch):
        """Default daemon scripts should not use predictable shared /tmp paths."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()
        monkeypatch.delenv("WU_OVRTX_RUNTIME_DIR", raising=False)
        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            backend = OvRTXRenderingBackend(ovrtx_venv_dir=str(venv_dir))
            daemon_script_path = Path(backend._daemon._daemon_script_path)
            runtime_dir = daemon_script_path.parent
            assert runtime_dir.name.startswith("wu_ovrtx_")
            assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
            assert stat.S_IMODE(daemon_script_path.stat().st_mode) == 0o600
            backend.__del__()
            assert not runtime_dir.exists()

    def test_backend_rejects_symlink_runtime_dir(self, tmp_path, monkeypatch):
        """Explicit runtime dir must not be a symlink into a shared location."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        runtime_link = tmp_path / "runtime-link"
        runtime_link.symlink_to(target_dir, target_is_directory=True)
        monkeypatch.setenv("WU_OVRTX_RUNTIME_DIR", str(runtime_link))
        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            with pytest.raises(RuntimeError, match="must not be a symlink"):
                OvRTXRenderingBackend(ovrtx_venv_dir=str(venv_dir))

    def test_backend_rejects_shared_runtime_dir(self, tmp_path, monkeypatch):
        """Explicit runtime dir must already be private if it exists."""
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        runtime_dir.chmod(0o755)
        monkeypatch.setenv("WU_OVRTX_RUNTIME_DIR", str(runtime_dir))
        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            with pytest.raises(RuntimeError, match="must not be group or world"):
                OvRTXRenderingBackend(ovrtx_venv_dir=str(venv_dir))


class TestOvRTXBackendNumSensorUpdatesPrecedence:
    """Verify OvRTXRenderingBackend.render's per-call num_sensor_updates override."""

    def _make_backend(self, num_sensor_updates: int, tmp_path, **kwargs):
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        venv_dir = tmp_path / "ovrtx_venv"
        venv_dir.mkdir()
        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            return_value="/fake/python",
        ):
            return OvRTXRenderingBackend(
                num_sensor_updates=num_sensor_updates,
                ovrtx_venv_dir=str(venv_dir),
                **kwargs,
            )

    def test_none_falls_back_to_instance_default(self, tmp_path):
        """num_sensor_updates=None must pass the instance-level value through."""
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, num_sensor_updates=None)

        kwargs = mock_render.call_args.kwargs
        assert kwargs["num_sensor_updates"] == 7

    def test_explicit_value_overrides_instance_default(self, tmp_path):
        """An explicit num_sensor_updates must beat the instance-level value."""
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, num_sensor_updates=42)

        kwargs = mock_render.call_args.kwargs
        assert kwargs["num_sensor_updates"] == 42

    def test_base_dir_passes_through_to_ovrtx_render(self, tmp_path):
        """Prepared anonymous stages still need the original asset base dir."""
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)
        stage = Usd.Stage.CreateInMemory()
        asset_root = tmp_path / "asset"
        asset_root.mkdir()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, base_dir=asset_root)

        kwargs = mock_render.call_args.kwargs
        assert kwargs["base_dir"] == asset_root

    def test_material_target_passes_through_to_ovrtx_render(self, tmp_path):
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, material_target="openpbr_materialx")

        kwargs = mock_render.call_args.kwargs
        assert kwargs["material_target"] == "openpbr_materialx"

    def test_material_target_is_public_backend_attribute(self, tmp_path):
        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)

        assert backend.material_target == "auto"

    def test_add_preview_fallbacks_is_public_backend_attribute(self, tmp_path):
        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)

        assert backend.add_preview_fallbacks is False

    def test_preview_surface_target_sets_public_fallback_attribute(self, tmp_path):
        backend = self._make_backend(
            num_sensor_updates=7,
            tmp_path=tmp_path,
            material_target="preview_surface",
        )

        assert backend.material_target == "preview_surface"
        assert backend.add_preview_fallbacks is True

    def test_legacy_preview_fallbacks_request_maps_to_preview_target(self, tmp_path):
        from pxr import Usd

        backend = self._make_backend(
            num_sensor_updates=7,
            tmp_path=tmp_path,
            add_preview_fallbacks=True,
        )
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage)

        kwargs = mock_render.call_args.kwargs
        assert backend.material_target == "auto"
        assert backend.add_preview_fallbacks is True
        assert kwargs["material_target"] == "preview_surface"

    def test_explicit_material_target_beats_legacy_preview_fallbacks(self, tmp_path):
        from pxr import Usd

        backend = self._make_backend(
            num_sensor_updates=7,
            tmp_path=tmp_path,
            add_preview_fallbacks=True,
        )
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, material_target="openpbr")

        kwargs = mock_render.call_args.kwargs
        assert kwargs["material_target"] == "openpbr_materialx"

    def test_per_call_preview_fallbacks_request_maps_to_preview_target(self, tmp_path):
        from pxr import Usd

        backend = self._make_backend(num_sensor_updates=7, tmp_path=tmp_path)
        stage = Usd.Stage.CreateInMemory()

        with unittest.mock.patch(
            "world_understanding.functions.graphics.render_ovrtx.render_all_cameras",
            return_value={"results": []},
        ) as mock_render:
            backend.render(stage=stage, add_preview_fallbacks=True)

        kwargs = mock_render.call_args.kwargs
        assert kwargs["material_target"] == "preview_surface"


# ---------------------------------------------------------------------------
# Integration tests (require RTX GPU + ovrtx installed)
# ---------------------------------------------------------------------------

_has_ovrtx = False
_ovrtx_skip_reason = "ovrtx not available"
try:
    from world_understanding.functions.graphics.render_ovrtx import _get_ovrtx_python

    _python = _get_ovrtx_python()
    # Verify ovrtx can actually initialize (needs GPU + Vulkan + libGL).
    # Strip PYTHONPATH so the app's pxr provider doesn't leak into the isolated venv.
    import subprocess

    _env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    _probe = subprocess.run(
        [_python, "-c", "import ovrtx; ovrtx.Renderer()"],
        capture_output=True,
        timeout=60,
        env=_env,
    )
    _has_ovrtx = _probe.returncode == 0
    if not _has_ovrtx:
        _ovrtx_skip_reason = f"ovrtx Renderer() failed: {_probe.stderr.decode()[-200:]}"
except (ImportError, RuntimeError, subprocess.TimeoutExpired) as exc:
    _ovrtx_skip_reason = f"ovrtx probe error: {exc}"

requires_ovrtx = pytest.mark.skipif(not _has_ovrtx, reason=_ovrtx_skip_reason)
requires_native_displaycolor_probe = pytest.mark.skipif(
    not _native_displaycolor_probe_enabled(),
    reason=f"set {_NATIVE_DISPLAYCOLOR_PROBE_ENV}=1 to run native displayColor probe",
)


@pytest.fixture
def simple_usd_stage():
    """Create a simple USD stage with a cube and camera for testing."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

    # Add a simple cube
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.GetSizeAttr().Set(1.0)

    # Add a camera
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    camera.GetFocalLengthAttr().Set(50.0)
    camera.GetHorizontalApertureAttr().Set(36.0)

    # Position camera to see the cube
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(3.0, 3.0, 3.0))

    return stage


@requires_ovrtx
class TestOvRTXIntegrationSingleCamera:
    """Integration test: single camera rendering with OvRTX."""

    def test_render_single_camera(self, simple_usd_stage):
        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage,
            image_width=256,
            image_height=256,
            cameras=["/Camera"],
            frames="0",
        )

        assert result["total_cameras"] == 1
        assert result["successful_cameras"] == 1
        assert result["failed_cameras"] == 0
        assert len(result["results"]) == 1
        assert result["results"][0]["frame_count"] >= 1
        assert len(result["results"][0]["images"]) >= 1

        # Check that the image is a valid PIL Image
        img = result["results"][0]["images"][0]
        assert img.size == (256, 256)


@requires_ovrtx
class TestOvRTXIntegrationTimeSampledUsd:
    """Integration test: OvRTX renders authored USD time samples by frame."""

    def test_render_time_sampled_transform_color_visibility_and_camera(self):
        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=_make_time_sampled_compliance_stage(),
            image_width=96,
            image_height=96,
            cameras=["/Camera", "/AnimatedCamera"],
            frames="0:2",
            num_sensor_updates=8,
            render_mode="rt2",
        )

        assert result["total_cameras"] == 2
        assert result["successful_cameras"] == 2
        assert result["failed_cameras"] == 0

        static_camera = result["results"][0]
        animated_camera = result["results"][1]
        assert static_camera["frame_count"] == 3
        assert animated_camera["frame_count"] == 3

        static_images = static_camera["images"]
        animated_images = animated_camera["images"]

        assert _mean_abs_rgb_diff(static_images[0], static_images[1]) > 1.0
        assert _mean_abs_rgb_diff(static_images[1], static_images[2]) > 1.0
        assert _mean_abs_rgb_diff(animated_images[0], animated_images[2]) > 1.0

        assert (
            _bright_centroid_x(static_images[2])
            > _bright_centroid_x(static_images[0]) + 8.0
        )

        import numpy as np

        frame_means = [
            np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
            for image in static_images
        ]
        assert frame_means[0][0] > frame_means[0][1]
        assert frame_means[1][1] > frame_means[1][0]
        assert frame_means[2][2] > frame_means[2][1]

    def test_render_time_sampled_display_color_only(self):
        """displayColor-only animated stage must render frame-to-frame
        color deltas via the per-frame overlay path, even when no other
        channel (visibility, transform) is animated. Pins the contract
        that animated displayColor is a v1 supported channel
        independently — without this test, animated-color-only stages
        used to fall through to OvRTX's native handling whose 0.2.0
        behavior is unverified.
        """
        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=_make_display_color_only_stage(),
            image_width=96,
            image_height=96,
            cameras=["/Camera"],
            frames="0:2",
            num_sensor_updates=8,
            render_mode="rt2",
        )

        assert result["successful_cameras"] == 1
        camera_result = result["results"][0]
        images = camera_result["images"]
        _assert_red_green_blue_dominance(images)

    @requires_native_displaycolor_probe
    def test_render_time_sampled_display_color_only_native_probe(self):
        """Opt-in probe: animated displayColor must render without color overlays."""
        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=_make_display_color_only_stage(),
            image_width=96,
            image_height=96,
            cameras=["/Camera"],
            frames="0:2",
            num_sensor_updates=8,
            render_mode="rt2",
        )

        assert result["successful_cameras"] == 1
        camera_result = result["results"][0]
        images = camera_result["images"]
        _assert_red_green_blue_dominance(images)


@requires_ovrtx
class TestOvRTXIntegrationMultipleCameras:
    """Integration test: multi-camera rendering with OvRTX."""

    def test_render_multiple_cameras(self, simple_usd_stage):
        from pxr import Gf, UsdGeom

        # Add a second camera
        camera2 = UsdGeom.Camera.Define(simple_usd_stage, "/Camera2")
        camera2.GetFocalLengthAttr().Set(50.0)
        camera2.GetHorizontalApertureAttr().Set(36.0)
        xform2 = UsdGeom.Xformable(camera2.GetPrim())
        xform2.AddTranslateOp().Set(Gf.Vec3d(-3.0, 3.0, 3.0))

        from world_understanding.functions.graphics.render_ovrtx import (
            render_all_cameras,
        )

        result = render_all_cameras(
            stage=simple_usd_stage,
            image_width=256,
            image_height=256,
            cameras=["/Camera", "/Camera2"],
            frames="0",
        )

        assert result["total_cameras"] == 2
        assert result["successful_cameras"] == 2
        assert len(result["results"]) == 2
