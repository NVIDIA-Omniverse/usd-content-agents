# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resume coverage for the blank dataset render guard.

A resumed session (``skip_existing``/``resume``) must get the same
blank-render verdict a fresh run of the same asset would get: pre-existing
renders are part of the validated render set, the guard runs even when zero
prims were freshly rendered, and stale metadata inherited from a previous
session never replaces disk re-analysis.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image, ImageDraw
from pxr import Gf, Usd, UsdGeom, Vt

from world_understanding.agentic.usd_tasks import prim_traversal
from world_understanding.agentic.usd_tasks.prim_traversal import (
    USDPrimTraversalAndRenderingTask,
    prim_path_to_directory_structure,
)
from world_understanding.functions.graphics.rendering import RenderingConfig


def _save_blank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (0, 0, 0)).save(path)


def _save_nonblank_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([4, 4, 20, 20], fill=(0, 0, 0))
    draw.rectangle([22, 22, 30, 30], fill=(200, 30, 90))
    image.save(path)


def _skipped_prim_data(paths: list[Path]) -> list[dict[str, Any]]:
    """Build prim_data as `_record_existing_files` does for resumed prims."""
    return [
        {
            "prim_path": f"/World/Prim{index}",
            "images": [
                {
                    "view": f"view{index}",
                    "path": path.name,
                    "camera": f"Camera_view{index}",
                    "render_mode": "composition",
                    "skipped": True,
                }
            ],
            "metadata": {},
        }
        for index, path in enumerate(paths)
    ]


class TestResumeBlankGuard:
    """Pre-existing (skipped) renders must be validated like fresh ones."""

    def test_resume_with_preexisting_blank_renders_fails(self, tmp_path):
        """Zero freshly rendered prims + blank existing renders must fail."""
        task = USDPrimTraversalAndRenderingTask()
        blank_a = tmp_path / "a_composition.png"
        blank_b = tmp_path / "b_composition.png"
        nonblank = tmp_path / "c_composition.png"
        _save_blank_image(blank_a)
        _save_blank_image(blank_b)
        _save_nonblank_image(nonblank)
        context: dict[str, Any] = {}

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                _skipped_prim_data([blank_a, blank_b, nonblank]),
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context=context,
            )

        assert context["blank_render_checked_count"] == 3
        assert len(context["blank_renders"]) == 2

    def test_resume_with_healthy_preexisting_renders_passes(self, tmp_path):
        """Healthy pre-existing renders must not trip the guard."""
        task = USDPrimTraversalAndRenderingTask()
        paths = [tmp_path / f"prim{i}_composition.png" for i in range(3)]
        for path in paths:
            _save_nonblank_image(path)
        listener = Mock()
        context: dict[str, Any] = {}

        task._check_blank_dataset_renders(
            _skipped_prim_data(paths),
            tmp_path,
            rgb_modes=["composition"],
            sensor_modes=[],
            listener=listener,
            context=context,
        )

        assert "blank_renders" not in context
        listener.warning.assert_not_called()

    def test_skipped_render_with_stale_healthy_stats_is_reanalyzed(self, tmp_path):
        """Stale metadata claiming a blank file is healthy must not be trusted."""
        task = USDPrimTraversalAndRenderingTask()
        blank_path = tmp_path / "stale_composition.png"
        _save_blank_image(blank_path)
        prim_data = [
            {
                "prim_path": "/World/Stale",
                "images": [
                    {
                        "view": "front",
                        "path": blank_path.name,
                        "camera": "Camera_front",
                        "render_mode": "composition",
                        "skipped": True,
                        # Inherited metadata says the render is fine; the file
                        # on disk is blank. Disk re-analysis must win.
                        "blank_render": True,
                        "stats": {"blank": False},
                    }
                ],
            }
        ]

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                prim_data,
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context={},
            )

    def test_skipped_render_with_stale_blank_stats_is_reanalyzed(self, tmp_path):
        """Stale metadata claiming blankness must not fail a healthy render."""
        task = USDPrimTraversalAndRenderingTask()
        nonblank_path = tmp_path / "healthy_composition.png"
        _save_nonblank_image(nonblank_path)
        listener = Mock()
        context: dict[str, Any] = {}
        prim_data = [
            {
                "prim_path": "/World/Healthy",
                "images": [
                    {
                        "view": "front",
                        "path": nonblank_path.name,
                        "camera": "Camera_front",
                        "render_mode": "composition",
                        "skipped": True,
                        "blank_render": True,
                        "stats": {"blank": True, "reason": "remote_blank_render"},
                    }
                ],
            }
        ]

        task._check_blank_dataset_renders(
            prim_data,
            tmp_path,
            rgb_modes=["composition"],
            sensor_modes=[],
            listener=listener,
            context=context,
        )

        assert "blank_renders" not in context
        listener.warning.assert_not_called()

    def test_fresh_render_blank_stats_still_skip_disk_reanalysis(self, tmp_path):
        """Fresh (non-skipped) renders keep trusting renderer blank stats."""
        task = USDPrimTraversalAndRenderingTask()
        context: dict[str, Any] = {}

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            task._check_blank_dataset_renders(
                [
                    {
                        "prim_path": "/World/Blank",
                        "images": [
                            {
                                "path": "missing_composition.png",
                                "render_mode": "composition",
                                "view": "front",
                                "blank_render": True,
                                "stats": {"blank": True, "reason": "solid_color"},
                            }
                        ],
                    }
                ],
                tmp_path,
                rgb_modes=["composition"],
                sensor_modes=[],
                listener=Mock(),
                context=context,
            )

        assert context["blank_renders"][0]["stats"]["reason"] == "solid_color"
        assert "analysis_error" not in context["blank_renders"][0]


def _stage_with_mesh() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Cube")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(0, 1, 0),
                Gf.Vec3f(0, 0, 1),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 1, 1)]))
    return stage


def _preexisting_render_paths(
    renders_dir: Path,
    rendering_config: RenderingConfig,
    prim_path: str = "/World/Cube",
    render_mode: str = "prim_only",
) -> list[Path]:
    """Compute the file paths `skip_existing` expects for a prim."""
    prim_name = prim_path.strip("/").split("/")[-1]
    paths = []
    for camera_spec in rendering_config.get_cameras_for_mode(render_mode):
        dir_suffix = prim_traversal.format_direction_for_filename(camera_spec.direction)
        camera_name = f"{rendering_config.camera_name_prefix}_{dir_suffix}"
        view_name = camera_name.split("_", 1)[-1]
        filename = f"{prim_name}_{view_name}_{render_mode}.png"
        paths.append(prim_path_to_directory_structure(prim_path, renders_dir, filename))
    return paths


class TestResumeRunLevelBlankGuard:
    """Resume-like `run()` invocations (0 fresh prims) must stay guarded."""

    def _run_resume_session(self, tmp_path: Path, *, blank: bool) -> dict[str, Any]:
        stage = _stage_with_mesh()
        task = USDPrimTraversalAndRenderingTask()
        rendering_config = RenderingConfig(camera_ordering=["+x"])
        output_dir = tmp_path / "output"
        renders_dir = output_dir / "renders"
        for path in _preexisting_render_paths(renders_dir, rendering_config):
            if blank:
                _save_blank_image(path)
            else:
                _save_nonblank_image(path)

        object_store = Mock()
        object_store.get.side_effect = lambda key, default=None: {
            "usd_stage": stage,
            "rendering_backend": object(),
            "rendering_config": rendering_config,
            "usd_model": None,
        }.get(key, default)

        context = {
            "event_listener": Mock(),
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
            "render_output_dir": str(renders_dir),
            "output_dir": str(output_dir),
            "skip_existing": True,
            "rendering_modes": ["prim_only"],
            "batch_size": 1,
        }
        return task.run(context, object_store)

    def test_resumed_session_with_blank_existing_renders_fails(self, tmp_path):
        """skip_existing + all renders on disk must still trip the guard."""
        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            self._run_resume_session(tmp_path, blank=True)

    def test_resumed_session_with_healthy_existing_renders_passes(self, tmp_path):
        """skip_existing + healthy renders on disk completes normally."""
        result = self._run_resume_session(tmp_path, blank=False)

        assert result["rendered_prims"] == []
        assert result["total_images_rendered"] == 1
        assert result["prim_data"][0]["images"][0]["skipped"] is True

    @pytest.mark.asyncio
    async def test_resumed_async_session_with_blank_existing_renders_fails(
        self, tmp_path
    ):
        """The async traversal path applies the same resume guard."""
        stage = _stage_with_mesh()
        task = USDPrimTraversalAndRenderingTask()
        rendering_config = RenderingConfig(camera_ordering=["+x"])
        output_dir = tmp_path / "output"
        renders_dir = output_dir / "renders"
        for path in _preexisting_render_paths(renders_dir, rendering_config):
            _save_blank_image(path)

        object_store = Mock()
        object_store.get.side_effect = lambda key, default=None: {
            "usd_stage": stage,
            "rendering_backend": object(),
            "rendering_config": rendering_config,
            "usd_model": None,
        }.get(key, default)

        context = {
            "event_listener": Mock(),
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
            "render_output_dir": str(renders_dir),
            "output_dir": str(output_dir),
            "skip_existing": True,
            "rendering_modes": ["prim_only"],
            "batch_size": 1,
        }

        with pytest.raises(RuntimeError, match="dataset renders are blank"):
            await task.arun(context, object_store)
