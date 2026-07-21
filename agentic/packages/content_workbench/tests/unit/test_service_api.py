# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI tests for the content workbench service."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import content_workbench.sessions as sessions_module
from content_workbench.correspondence import SceneOptimizerPathMap
from content_workbench.main import (
    LOCALHOST_CORS_ORIGIN_REGEX,
    _cors_origin_regex_from_env,
    _first_nonempty_env,
    _http_error,
    create_app,
)
from content_workbench.material_apply_adapter import MaterialApplyUnavailableError
from content_workbench.models import (
    MAX_BATCH_REQUEST_ITEMS,
    MAX_OPTIMIZATION_CONFIG_DEPTH,
    MAX_PRIM_PATH_LENGTH,
    MAX_RENDER_DIMENSION,
    CreateSessionRequest,
    MaterialOverride,
    OptimizationState,
    PickRequest,
    RenderRequest,
    SceneSnapshotResponse,
)
from content_workbench.renderer_worker import (
    IsolatedOvRTXRendererWorker,
    PickRenderResult,
)
from content_workbench.sessions import (
    MAX_CAMERA_DISTANCE,
    PREVIEW_SCENE_RETENTION_COUNT,
    SceneSession,
    SessionManager,
    _export_preview_stage,
    _is_durable_material_override,
    _material_apply_bound_source_paths,
    _override_source_paths_or_fallback,
    _package_material_apply_usdz,
    _parse_direction,
    _parse_frame_spec,
    _prune_preview_scenes,
    _resolve_material_apply_output_path,
    _run_material_apply_task,
    _trim_material_override_coverage,
)
from content_workbench.version import SERVICE_VERSION


class FakeRenderer:
    selected_prim_paths: list[str] = []
    render_mode: str | None = None
    num_updates: int | None = None
    active_aov: str | None = None
    hdri_light: float | None = None
    dome_light: float | None = None
    distant_light: float | None = None
    frame_count: int = 0

    def render(
        self,
        *,
        scene_path: Path,
        output_path: Path,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float = 50.0,
        camera_horizontal_aperture: float = 36.0,
        num_updates: int = 64,
        render_mode: str = "rt2",
        hdri_light: float | None = 600.0,
        dome_light: float | None = None,
        distant_light: float | None = None,
        selected_prim_paths: list[str] | None = None,
        active_aov: str = "LdrColor",
        lock_timeout_seconds: float | None = None,
    ) -> float:
        self.selected_prim_paths = selected_prim_paths or []
        self.render_mode = render_mode
        self.num_updates = num_updates
        self.active_aov = active_aov
        self.hdri_light = hdri_light
        self.dome_light = dome_light
        self.distant_light = distant_light
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (width, height), (128, 128, 128, 255)).save(output_path)
        return 0.125

    def render_frames(
        self,
        *,
        scene_path: Path,
        output_paths: list[Path],
        width: int,
        height: int,
        frame_numbers: list[int],
        camera_transforms: list[list[list[float]]],
        camera_path: str | None = None,
        camera_focal_length: float = 50.0,
        camera_horizontal_aperture: float = 36.0,
        fps: float = 24.0,
        num_updates: int = 64,
        render_mode: str = "rt2",
        hdri_light: float | None = 600.0,
        dome_light: float | None = None,
        distant_light: float | None = None,
        selected_prim_paths: list[str] | None = None,
        active_aov: str = "LdrColor",
        lock_timeout_seconds: float | None = None,
    ) -> float:
        self.selected_prim_paths = selected_prim_paths or []
        self.render_mode = render_mode
        self.num_updates = num_updates
        self.active_aov = active_aov
        self.hdri_light = hdri_light
        self.dome_light = dome_light
        self.distant_light = distant_light
        self.frame_count = len(output_paths)
        self.frame_numbers = frame_numbers
        self.camera_path = camera_path
        self.fps = fps
        if camera_path is None:
            assert len(output_paths) == len(camera_transforms)
        for index, output_path in enumerate(output_paths):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGBA",
                (width, height),
                (128, 128, min(255, index), 255),
            ).save(output_path)
        return 0.25

    def pick(
        self,
        *,
        scene_path: Path,
        x: int,
        y: int,
        width: int,
        height: int,
        camera_transform: list[list[float]],
        camera_focal_length: float = 50.0,
        camera_horizontal_aperture: float = 36.0,
        num_updates: int = 1,
        render_mode: str = "rt2",
        hdri_light: float | None = 600.0,
        dome_light: float | None = None,
        distant_light: float | None = None,
        selected_prim_paths: list[str] | None = None,
        lock_timeout_seconds: float | None = None,
    ) -> PickRenderResult:
        self.selected_prim_paths = selected_prim_paths or []
        self.render_mode = render_mode
        self.num_updates = num_updates
        self.hdri_light = hdri_light
        self.dome_light = dome_light
        self.distant_light = distant_light
        return PickRenderResult(prim_paths=["/World/Step"], elapsed_seconds=0.05)

    def shutdown(self) -> None:
        return None


def test_session_manager_uses_process_isolated_renderer_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(tmp_path / "workspace"))
    manager = SessionManager()

    assert isinstance(manager._renderer, IsolatedOvRTXRendererWorker)

    manager.shutdown()


def test_default_workspace_root_is_per_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTENT_WORKBENCH_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("SCENE_INSPECTOR_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("RSI_WORKSPACE_DIR", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "runtime"))

    manager = SessionManager(renderer=FakeRenderer())

    assert manager._workspace_root == tmp_path / "runtime" / "content-workbench"


def test_default_workspace_root_falls_back_to_private_temp_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTENT_WORKBENCH_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("SCENE_INSPECTOR_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("RSI_WORKSPACE_DIR", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(
        "content_workbench.sessions.tempfile.gettempdir",
        lambda: str(tmp_path),
    )

    manager = SessionManager(renderer=FakeRenderer())
    second_manager = SessionManager(renderer=FakeRenderer())

    assert manager._workspace_root.parent == tmp_path
    assert manager._workspace_root.name.startswith("content-workbench-")
    assert second_manager._workspace_root == manager._workspace_root
    assert manager._workspace_root.exists()
    assert manager._workspace_root.stat().st_mode & 0o777 == 0o700


def test_workspace_root_logs_chmod_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))

    def fail_chmod(_path: Path, _mode: int) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr("content_workbench.sessions.os.chmod", fail_chmod)

    with caplog.at_level(logging.WARNING, logger="content_workbench.sessions"):
        SessionManager(renderer=FakeRenderer())

    assert "Unable to chmod Content Workbench workspace root" in caplog.text


def test_http_error_scrubs_unhandled_details() -> None:
    try:
        raise RuntimeError("/tmp/internal/scene.usd failed")
    except RuntimeError as error:
        response = _http_error(error)

    assert response.status_code == 500
    assert response.detail == "Internal server error"


def test_http_error_redacts_absolute_filesystem_paths(tmp_path: Path) -> None:
    response = _http_error(FileNotFoundError(f"missing {tmp_path / 'scene.usda'}"))

    assert response.status_code == 404
    assert str(tmp_path) not in response.detail
    assert "<path>" in response.detail


def test_http_error_exposes_optional_material_apply_guidance() -> None:
    response = _http_error(
        MaterialApplyUnavailableError(
            "Material assignment apply requires the optional material-agent package."
        )
    )

    assert response.status_code == 501
    assert "optional material-agent package" in response.detail


def _write_sample_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Blue"
        {
            token outputs:surface.connect = </World/Looks/Blue/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 1)
                token outputs:surface
            }
        }
    }

    def Mesh "Step"
    {
        rel material:binding = </World/Looks/Blue>
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )


def _write_time_sampled_recording_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    startTimeCode = 0
    endTimeCode = 2
    timeCodesPerSecond = 10
)

def Xform "World"
{
    double3 xformOp:translate.timeSamples = {
        0: (0, 0, 0),
        1: (0, 0.1, 0),
        2: (0, 0.2, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Step"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}

def Scope "Cameras"
{
    def Camera "plus_xplus_yplus_z"
    {
        float2 clippingRange = (0.01, 100000)
        float focalLength = 35
        float horizontalAperture = 20.955
        matrix4d xformOp:transform = (
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (3, 3, 3, 1)
        )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
""",
        encoding="utf-8",
    )


def _write_authored_appearance_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    rel material:binding = </World/Looks/Blue>

    def Scope "Looks"
    {
        def Material "Blue"
        {
            token outputs:surface.connect = </World/Looks/Blue/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 1)
                token outputs:surface
            }
        }
    }

    def Mesh "Step"
    {
        color3f[] primvars:displayColor = [(0, 1, 0)] (
            interpolation = "constant"
        )
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )


def _write_isolation_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "KeepGroup"
    {
        def Mesh "KeepMesh"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        }
    }

    def Xform "Backdrop"
    {
        def Mesh "BackdropMesh"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        }
    }
}
""",
        encoding="utf-8",
    )


def _write_hidden_ancestor_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "HiddenGroup"
    {
        token visibility = "invisible"

        def Xform "Deep"
        {
            def Mesh "Candidate"
            {
                int[] faceVertexCounts = [4]
                int[] faceVertexIndices = [0, 1, 2, 3]
                point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
            }
        }
    }
}
""",
        encoding="utf-8",
    )


def _write_instanced_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Blue"
        {
            token outputs:surface.connect = </World/Looks/Blue/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0, 0, 1)
                token outputs:surface
            }
        }
    }

    def Xform "Geom"
    {
        def Mesh "Mesh"
        {
            rel material:binding = </World/Looks/Blue>
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        }
    }

    def Xform "Part" (
        instanceable = true
        prepend references = </World/Geom>
    )
    {
    }
}
""",
        encoding="utf-8",
    )


def _write_material_library_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Steel_Painted_Orange"
        {
            token outputs:surface.connect = </World/Looks/Steel_Painted_Orange/Shader.outputs:surface>

            def Shader "Shader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.78, 0.27, 0.025)
                float inputs:roughness = 0.4
                token outputs:surface
            }
        }
    }
}
""",
        encoding="utf-8",
    )


def _write_split_source_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "Panel"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]

        def GeomSubset "FaceA"
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0]
        }
    }
}
""",
        encoding="utf-8",
    )


def _write_split_optimized_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "Panel_part_0"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )


def _write_dedup_source_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Mesh "BoltA"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }

    def Mesh "BoltB"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
    }
}
""",
        encoding="utf-8",
    )


def _write_dedup_optimized_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "BoltPrototype"
    {
        def Mesh "Geometry"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        }
    }
}
""",
        encoding="utf-8",
    )


def _write_mixed_alias_optimized_stage(path: Path) -> None:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "BoltPrototype"
    {
        def Mesh "Geometry"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        }
    }

    def Xform "BoltA"
    {
        def Mesh "Extra"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        }
    }
}
""",
        encoding="utf-8",
    )


class FakeMixedAliasOptimizingSessionManager(SessionManager):
    """Optimizer stand-in where one canonical prim has two runtime fragments.

    BoltA is represented at runtime by both the alias it shares with BoltB
    (deduplicated geometry) and its own unique fragment. This is the "extra
    runtime edge" shape that previously made content-workflow-cli's material
    finalizer reject the whole shared-alias group rather than risk data loss.
    """

    def __init__(self, *, optimized_path: Path) -> None:
        super().__init__(renderer=FakeRenderer())
        self.optimized_path = optimized_path

    def _optimize_scene(
        self,
        session_id: str,
        source_scene_path: Path,
        optimization_config: dict[str, object],
    ) -> tuple[Path, OptimizationState, SceneOptimizerPathMap]:
        metadata = {
            "correspondence_map": {
                "full_mapping": {
                    "original_to_prototype": {
                        "/World/BoltA": [
                            "/World/BoltPrototype/Geometry",
                            "/World/BoltA/Extra",
                        ],
                        "/World/BoltB": ["/World/BoltPrototype/Geometry"],
                    },
                },
            },
            "operations_executed": [
                {"operation": "deduplicateGeometry"},
                {"operation": "splitMeshes"},
            ],
        }
        path_map = SceneOptimizerPathMap.from_metadata(
            original_usd_path=source_scene_path,
            optimization_metadata=metadata,
        )
        optimization = OptimizationState(
            enabled=True,
            status="ready",
            source_scene_path=str(source_scene_path),
            inspection_scene_path=str(self.optimized_path),
            correspondence_summary=path_map.summary(),
            operations_executed=metadata["operations_executed"],
        )
        return self.optimized_path, optimization, path_map


def _client() -> TestClient:
    return TestClient(create_app(SessionManager(renderer=FakeRenderer())))


def test_empty_cors_origin_regex_env_keeps_localhost_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTENT_WORKBENCH_CORS_ORIGIN_REGEX", "")

    assert _cors_origin_regex_from_env() == LOCALHOST_CORS_ORIGIN_REGEX


def test_default_cors_origin_regex_allows_loopback_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CONTENT_WORKBENCH_CORS_ORIGIN_REGEX", raising=False)

    regex = _cors_origin_regex_from_env()

    assert regex is not None
    assert re.fullmatch(regex, "http://127.0.0.2:8088")
    assert re.fullmatch(regex, "http://[::ffff:127.0.0.1]:8088")
    assert not re.fullmatch(regex, "http://127.999.999.999:8088")


def test_cors_origin_regex_env_can_disable_default_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTENT_WORKBENCH_CORS_ORIGIN_REGEX", "none")

    assert _cors_origin_regex_from_env() is None


def test_first_nonempty_env_ignores_empty_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTENT_WORKBENCH_PORT", "")
    monkeypatch.setenv("SCENE_INSPECTOR_PORT", "")
    monkeypatch.setenv("RSI_PORT", "")

    assert (
        _first_nonempty_env(
            ("CONTENT_WORKBENCH_PORT", "SCENE_INSPECTOR_PORT", "RSI_PORT"),
            default="8088",
        )
        == "8088"
    )


def test_workspace_root_ignores_empty_primary_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_workspace = tmp_path / "legacy-workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", "")
    monkeypatch.setenv("SCENE_INSPECTOR_WORKSPACE_DIR", str(legacy_workspace))

    manager = SessionManager(renderer=FakeRenderer())

    assert manager._workspace_root == legacy_workspace


def test_direction_parser_requires_full_token_consumption() -> None:
    assert _parse_direction("+x.5x") == pytest.approx(
        [1.0 / 3.0**0.5, -1.0 / 3.0**0.5, 1.0 / 3.0**0.5]
    )
    assert _parse_direction("+1.5x-y") == pytest.approx(
        [0.8320502943, -0.5547001962, 0.0]
    )


class FakeOptimizingSessionManager(SessionManager):
    def __init__(self, *, optimized_path: Path) -> None:
        super().__init__(renderer=FakeRenderer())
        self.optimized_path = optimized_path
        self.last_optimization_config: dict[str, object] | None = None

    def _optimize_scene(
        self,
        session_id: str,
        source_scene_path: Path,
        optimization_config: dict[str, object],
    ) -> tuple[Path, OptimizationState, SceneOptimizerPathMap]:
        self.last_optimization_config = optimization_config
        metadata = {
            "correspondence_map": {
                "split_mapping": {
                    "/World/Panel": ["/World/Panel_part_0"],
                },
                "full_mapping": {
                    "original_to_prototype": {
                        "/World/Panel": ["/World/Panel_part_0"],
                    },
                },
            },
            "operations_executed": [{"operation": "splitMeshes"}],
        }
        path_map = SceneOptimizerPathMap.from_metadata(
            original_usd_path=source_scene_path,
            optimization_metadata=metadata,
        )
        optimization = OptimizationState(
            enabled=True,
            status="ready",
            source_scene_path=str(source_scene_path),
            inspection_scene_path=str(self.optimized_path),
            correspondence_summary=path_map.summary(),
            operations_executed=metadata["operations_executed"],
        )
        return self.optimized_path, optimization, path_map


class FakeDedupOptimizingSessionManager(SessionManager):
    def __init__(self, *, optimized_path: Path) -> None:
        super().__init__(renderer=FakeRenderer())
        self.optimized_path = optimized_path

    def _optimize_scene(
        self,
        session_id: str,
        source_scene_path: Path,
        optimization_config: dict[str, object],
    ) -> tuple[Path, OptimizationState, SceneOptimizerPathMap]:
        metadata = {
            "correspondence_map": {
                "full_mapping": {
                    "original_to_prototype": {
                        "/World/BoltA": ["/World/BoltPrototype/Geometry"],
                        "/World/BoltB": ["/World/BoltPrototype/Geometry"],
                    },
                },
            },
            "operations_executed": [{"operation": "deduplicateGeometry"}],
        }
        path_map = SceneOptimizerPathMap.from_metadata(
            original_usd_path=source_scene_path,
            optimization_metadata=metadata,
        )
        optimization = OptimizationState(
            enabled=True,
            status="ready",
            source_scene_path=str(source_scene_path),
            inspection_scene_path=str(self.optimized_path),
            correspondence_summary=path_map.summary(),
            operations_executed=metadata["operations_executed"],
        )
        return self.optimized_path, optimization, path_map


def test_agent_api_docs_are_served_from_canonical_location() -> None:
    client = _client()

    doc = client.get("/agent-api")
    assert doc.status_code == 200
    assert "Content Workbench Agent API" in doc.text
    assert "<workbench-endpoint>/openapi.json" in doc.text

    discovery = client.get("/agent-api.json")
    assert discovery.status_code == 200
    body = discovery.json()
    assert body["service"] == "content-workbench"
    assert body["version"] == SERVICE_VERSION
    assert body["agent_api_url"] == "/agent-api"
    assert body["openapi_url"] == "/openapi.json"
    assert body["agent_openapi_url"] == "/agent/openapi.json"
    assert body["capabilities_url"] == "/agent/capabilities"
    assert body["tool_manifest_url"] == "/agent/tool-manifest"
    assert body["agent_discovery_endpoints"] == [
        "/agent-api",
        "/agent-api.json",
        "/agent/capabilities",
        "/agent/openapi.json",
        "/agent/tool-manifest",
    ]
    assert "/sessions/{session_id}/scene/snapshot" in body["primary_endpoints"]
    assert "/sessions/{session_id}/scene/optimize" in body["primary_endpoints"]
    assert "/sessions/{session_id}/scene/restore" in body["primary_endpoints"]
    assert (
        "/sessions/{session_id}/physics/validate-runtime" in body["primary_endpoints"]
    )
    assert "/sessions/{session_id}/render-frames" in body["primary_endpoints"]
    assert "/sessions/{session_id}/renders/{filename:path}" in body["primary_endpoints"]
    assert (
        "/sessions/{session_id}/authoring/material-assignments:apply"
        in body["primary_endpoints"]
    )
    assert "target_agent_endpoints" not in body
    assert not any(
        "/agent/sessions/" in endpoint for endpoint in body["primary_endpoints"]
    )
    assert "/sessions/{session_id}/stream" not in body["primary_endpoints"]
    assert "material_override" in body["primary_commands"]

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["version"] == SERVICE_VERSION

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    openapi_body = openapi.json()
    assert openapi_body["info"]["version"] == SERVICE_VERSION
    assert "post" in openapi_body["paths"]["/sessions/{session_id}/paths/translate"]
    assert (
        "post" in openapi_body["paths"]["/sessions/{session_id}/paths/translate:batch"]
    )
    assert not any(
        path.startswith("/agent/sessions/") for path in openapi_body["paths"]
    )

    agent_openapi = client.get("/agent/openapi.json")
    assert agent_openapi.status_code == 200
    assert agent_openapi.json()["info"]["version"] == SERVICE_VERSION

    capabilities = client.get("/agent/capabilities")
    assert capabilities.status_code == 200
    capabilities_body = capabilities.json()
    assert capabilities_body["service"] == "content-workbench"
    assert "viewport_render" in capabilities_body["capabilities"]
    assert "frame_sequence_render" in capabilities_body["capabilities"]
    assert "physics_runtime_validation" in capabilities_body["capabilities"]
    assert "physics_component_inspection" in capabilities_body["capabilities"]
    assert "physics_topology_plan_apply" in capabilities_body["capabilities"]
    assert capabilities_body["agent_discovery_endpoints"] == [
        "/agent-api",
        "/agent-api.json",
        "/agent/capabilities",
        "/agent/openapi.json",
        "/agent/tool-manifest",
    ]

    tool_manifest = client.get("/agent/tool-manifest")
    assert tool_manifest.status_code == 200
    manifest_body = tool_manifest.json()
    assert manifest_body["transport"] == "rest"
    assert manifest_body["discovery"]["openapi"] == "/agent/openapi.json"
    operations = {
        operation["name"]: operation for operation in manifest_body["operations"]
    }
    assert set(operations) >= {
        "create_session",
        "optimize_scene",
        "snapshot_scene",
        "render",
        "pick",
        "translate_path",
        "translate_paths",
        "apply_command",
        "inspect_physics_candidates",
        "inspect_physics_components",
        "inspect_physics_topology",
        "apply_physics_topology_plan",
        "apply_physics_schema",
        "validate_physics_runtime",
        "render_frame_sequence",
    }
    assert operations["translate_path"]["path"] == (
        "/sessions/{session_id}/paths/translate"
    )
    assert operations["translate_paths"]["path"] == (
        "/sessions/{session_id}/paths/translate:batch"
    )


def test_physics_agent_routes_delegate_to_workbench_ops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_workbench import physics_ops

    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_inspect_mesh_candidates(
        usd_path: str,
        *,
        root_prim_path: str | None = None,
        include_existing_schema: bool = True,
        path_space: str = "source",
    ) -> dict[str, Any]:
        calls.append(
            (
                "inspect",
                {
                    "usd_path": usd_path,
                    "root_prim_path": root_prim_path,
                    "include_existing_schema": include_existing_schema,
                    "path_space": path_space,
                },
            )
        )
        return {
            "asset": usd_path,
            "path_space": path_space,
            "candidate_count": 1,
            "candidates": [{"prim_path": "/World/Bulb"}],
        }

    def fake_apply_schema(
        *,
        usd_path: str,
        decision_patch_path: str | None = None,
        predictions_jsonl_path: str,
        output_usd_path: Path | str,
        collision_approximation: str = "convexHull",
        output_key: str = "classification",
        author_rigid_body: bool = True,
    ) -> dict[str, Any]:
        output_usd_path = str(output_usd_path)
        calls.append(
            (
                "apply",
                {
                    "usd_path": usd_path,
                    "decision_patch_path": decision_patch_path,
                    "predictions_jsonl_path": predictions_jsonl_path,
                    "output_usd_path": output_usd_path,
                    "collision_approximation": collision_approximation,
                    "output_key": output_key,
                    "author_rigid_body": author_rigid_body,
                },
            )
        )
        return {
            "operation": "physics.apply_schema",
            "physics_usd": output_usd_path,
            "collision_count": 1,
            "rigid_body_count": 1,
        }

    def fake_validate_runtime(
        *,
        physics_usd: str,
        output_dir: Path | str,
        engine: str = "ovphysx",
        duration_s: float = 1.0,
        dt: float = 1.0 / 240.0,
        sample_fps: int = 30,
        drop_height_m: float | None = None,
        acceptance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_dir = str(output_dir)
        calls.append(
            (
                "validate",
                {
                    "physics_usd": physics_usd,
                    "output_dir": output_dir,
                    "engine": engine,
                    "duration_s": duration_s,
                    "dt": dt,
                    "sample_fps": sample_fps,
                    "drop_height_m": drop_height_m,
                    "acceptance": acceptance,
                },
            )
        )
        return {
            "engine": engine,
            "physics_usd": physics_usd,
            "runtime_report": str(tmp_path / "runtime_report.json"),
            "failures": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        physics_ops,
        "inspect_mesh_candidates",
        fake_inspect_mesh_candidates,
    )
    monkeypatch.setattr(physics_ops, "apply_schema", fake_apply_schema)
    monkeypatch.setattr(physics_ops, "validate_runtime", fake_validate_runtime)

    renderer = FakeRenderer()
    client = TestClient(create_app(SessionManager(renderer=renderer)))
    session_stage_path = tmp_path / "session.usda"
    _write_sample_stage(session_stage_path)
    session_id = client.post(
        "/sessions",
        json={"scene_path": str(session_stage_path)},
    ).json()["session_id"]
    usd_path = str(tmp_path / "lightbulb.usda")
    recording_path = tmp_path / "recording.usda"
    _write_time_sampled_recording_stage(recording_path)

    inspect_response = client.post(
        f"/sessions/{session_id}/physics/inspect-mesh-candidates",
        json={
            "usd_path": usd_path,
            "root_prim_path": "/World",
            "include_existing_schema": False,
            "path_space": "source",
        },
    )
    assert inspect_response.status_code == 200
    assert inspect_response.json()["candidate_count"] == 1

    apply_response = client.post(
        f"/sessions/{session_id}/physics/apply-schema",
        json={
            "usd_path": usd_path,
            "decision_patch_path": str(tmp_path / "decisions.json"),
            "predictions_jsonl_path": str(tmp_path / "predictions.jsonl"),
            "collision_approximation": "convexHull",
            "author_rigid_body": False,
        },
    )
    assert apply_response.status_code == 200
    assert apply_response.json()["operation"] == "physics.apply_schema"
    apply_call = next(call for call in calls if call[0] == "apply")
    assert apply_call[1]["author_rigid_body"] is False

    escaped_apply = client.post(
        f"/sessions/{session_id}/physics/apply-schema",
        json={
            "usd_path": usd_path,
            "predictions_jsonl_path": str(tmp_path / "predictions.jsonl"),
            "output_usd_path": str(tmp_path / "escaped.usda"),
        },
    )
    assert escaped_apply.status_code == 400
    assert "session workspace" in escaped_apply.text

    validation_response = client.post(
        f"/sessions/{session_id}/physics/validate-runtime",
        json={
            "physics_usd_path": str(tmp_path / "physics.usda"),
            "engine": "fake",
            "duration_s": 0.25,
            "dt": 0.01,
            "sample_fps": 10,
            "drop_height_m": 0.1,
            "acceptance": {
                "expected_body_count": 2,
                "max_ground_penetration_m": 0.01,
            },
        },
    )
    assert validation_response.status_code == 200
    assert validation_response.json()["engine"] == "fake"
    validate_call = [call for call in calls if call[0] == "validate"][-1][1]
    assert validate_call["acceptance"]["expected_body_count"] == 2
    assert validate_call["acceptance"]["max_ground_penetration_m"] == pytest.approx(
        0.01
    )
    assert "require_gravity_response" not in validate_call["acceptance"]
    assert "detect_initial_pose_discontinuity" not in validate_call["acceptance"]

    default_validation_response = client.post(
        f"/sessions/{session_id}/physics/validate-runtime",
        json={
            "physics_usd_path": str(tmp_path / "physics.usda"),
            "engine": "fake",
        },
    )
    assert default_validation_response.status_code == 200
    default_validate_call = [call for call in calls if call[0] == "validate"][-1][1]
    assert default_validate_call["acceptance"] is None

    too_long_validation = client.post(
        f"/sessions/{session_id}/physics/validate-runtime",
        json={
            "physics_usd_path": str(tmp_path / "physics.usda"),
            "duration_s": 60.0,
        },
    )
    assert too_long_validation.status_code == 422

    too_small_dt_validation = client.post(
        f"/sessions/{session_id}/physics/validate-runtime",
        json={
            "physics_usd_path": str(tmp_path / "physics.usda"),
            "dt": 0.0001,
        },
    )
    assert too_small_dt_validation.status_code == 422

    too_dense_sampling_validation = client.post(
        f"/sessions/{session_id}/physics/validate-runtime",
        json={
            "physics_usd_path": str(tmp_path / "physics.usda"),
            "sample_fps": 240,
        },
    )
    assert too_dense_sampling_validation.status_code == 422

    escaped_validation = client.post(
        f"/sessions/{session_id}/physics/validate-runtime",
        json={
            "physics_usd_path": str(tmp_path / "physics.usda"),
            "output_dir": str(tmp_path / "runtime"),
        },
    )
    assert escaped_validation.status_code == 400
    assert "session workspace" in escaped_validation.text

    render_response = client.post(
        f"/sessions/{session_id}/render-frames",
        json={
            "scene_path": str(recording_path),
            "camera_path": "+x+y+z",
            "width": 256,
            "height": 256,
            "make_mp4": False,
            "max_duration_seconds": 0.2,
            "ovrtx_num_sensor_updates": 2,
            "ovrtx_render_mode": "rt2",
        },
    )
    assert render_response.status_code == 200
    assert len(render_response.json()["frame_paths"]) == 3
    assert len(render_response.json()["frame_urls"]) == 3
    assert renderer.camera_path == "/Cameras/plus_xplus_yplus_z"
    assert renderer.frame_numbers == [0, 1, 2]
    assert renderer.fps == 10

    assert [name for name, _payload in calls] == [
        "inspect",
        "apply",
        "validate",
        "validate",
    ]
    apply_payload = calls[1][1]
    assert apply_payload["decision_patch_path"] == str(tmp_path / "decisions.json")
    assert Path(str(apply_payload["output_usd_path"])).name == "physics.usda"
    assert session_id in Path(str(apply_payload["output_usd_path"])).parts
    validate_payload = calls[2][1]
    assert session_id in Path(str(validate_payload["output_dir"])).parts

    escape = client.post(
        f"/sessions/{session_id}/render-frames",
        json={"scene_path": str(recording_path), "output_dir": str(tmp_path / "out")},
    )
    assert escape.status_code == 400
    assert "session workspace" in escape.text


def test_physics_component_and_topology_routes_use_workspace_derivative(
    tmp_path: Path,
) -> None:
    from world_understanding.functions.physics.physics_topology import sha256_file

    renderer = FakeRenderer()
    client = TestClient(create_app(SessionManager(renderer=renderer)))
    stage_path = tmp_path / "asset.usda"
    _write_sample_stage(stage_path)
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    components = client.post(
        f"/sessions/{session_id}/physics/inspect-components",
        json={"usd_path": str(stage_path), "path_space": "source"},
    )
    topology = client.post(
        f"/sessions/{session_id}/physics/inspect-topology",
        json={"usd_path": str(stage_path), "path_space": "source"},
    )
    applied = client.post(
        f"/sessions/{session_id}/physics/apply-topology-plan",
        json={
            "input_usd_path": str(stage_path),
            "expected_source_digest": sha256_file(stage_path),
            "mobility_intent": "preserve",
            "operations": [],
            "invariants": {
                "enabled_collider_count": 0,
                "reject_articulation_changes": True,
            },
        },
    )

    assert components.status_code == 200
    assert components.json()["component_count"] == 1
    assert topology.status_code == 200
    assert topology.json()["enabled_collider_count"] == 0
    assert applied.status_code == 200
    output = Path(applied.json()["output_usd_path"])
    assert output.name == "prepared.usda"
    assert session_id in output.parts
    assert output.is_file()
    assert Path(applied.json()["topology_report"]).is_file()


def test_parse_frame_spec_rejects_oversized_range() -> None:
    with pytest.raises(ValueError, match="at most 10000 frames"):
        _parse_frame_spec("0:10000")


def test_scene_snapshot_schema_excluded_non_candidates_are_paths() -> None:
    schema = SceneSnapshotResponse.model_json_schema()
    excluded_schema = schema["properties"]["excluded_non_candidates"]

    assert excluded_schema["type"] == "array"
    assert excluded_schema["items"]["type"] == "string"


def test_scene_snapshot_candidate_hints_share_max_prim_budget(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    stage_path = tmp_path / "bounded_snapshot.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    for index in range(12):
        UsdGeom.Cube.Define(stage, f"/World/Cube_{index:02d}")
    stage.GetRootLayer().Save()

    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    response = client.post(
        f"/sessions/{session_id}/scene/snapshot",
        json={"max_prims": 6},
    )

    assert response.status_code == 200
    snapshot = response.json()
    summary = snapshot["summary"]
    detailed_paths = set(snapshot["paths"]) | {
        candidate["inspection_path"] for candidate in snapshot["candidates"]
    }
    assert summary["snapshot_path_count"] <= 6
    assert len(detailed_paths) <= 6
    assert summary["candidate_hint_path_count"] <= 3
    assert summary["candidate_hints_truncated"] is True
    assert summary["truncated"] is True


def test_create_session_loads_scene_and_serves_queries(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()

    create = client.post("/sessions", json={"scene_path": str(stage_path)})
    assert create.status_code == 201
    session = create.json()
    session_id = session["session_id"]
    assert session["status"] == "ready"
    assert session["root_prim_path"] == "/World"
    assert session["viewport"]["mode"] == "still_render"
    assert session["viewport"]["width"] == 1280
    assert session["viewport"]["height"] == 720

    tree = client.get(f"/sessions/{session_id}/tree", params={"prim_path": "/World"})
    assert tree.status_code == 200
    assert [child["path"] for child in tree.json()["children"]] == [
        "/World/Looks",
        "/World/Step",
    ]

    props = client.get(
        f"/sessions/{session_id}/properties",
        params={"prim_path": "/World/Step"},
    )
    assert props.status_code == 200
    assert props.json()["properties"]["type_name"] == "Mesh"

    material = client.get(
        f"/sessions/{session_id}/material-binding",
        params={"prim_path": "/World/Step"},
    )
    assert material.status_code == 200
    assert material.json()["bound_material_path"] == "/World/Looks/Blue"

    properties_batch = client.post(
        f"/sessions/{session_id}/properties:batch",
        json={"prim_paths": ["/World", "/World/Step"]},
    )
    assert properties_batch.status_code == 200
    properties_results = properties_batch.json()["results"]
    assert [result["prim_path"] for result in properties_results] == [
        "/World",
        "/World/Step",
    ]
    assert properties_results[1]["properties"]["type_name"] == "Mesh"

    material_batch = client.post(
        f"/sessions/{session_id}/material-binding:batch",
        json={"prim_paths": ["/World", "/World/Step"]},
    )
    assert material_batch.status_code == 200
    material_results = material_batch.json()["results"]
    assert [result["prim_path"] for result in material_results] == [
        "/World",
        "/World/Step",
    ]
    assert material_results[1]["bound_material_path"] == "/World/Looks/Blue"

    snapshot = client.post(
        f"/sessions/{session_id}/scene/snapshot",
        json={},
    )
    assert snapshot.status_code == 200
    snapshot_body = snapshot.json()
    assert snapshot_body["session_id"] == session_id
    assert snapshot_body["root_prim_path"] == "/World"
    assert snapshot_body["summary"]["prim_count"] == 5
    assert snapshot_body["summary"]["candidate_count"] == 1
    assert snapshot_body["paths"] == [
        "/World",
        "/World/Looks",
        "/World/Step",
        "/World/Looks/Blue",
        "/World/Looks/Blue/Shader",
    ]
    assert [
        result["prim_path"] for result in snapshot_body["properties"]
    ] == snapshot_body["paths"]
    assert [
        result["prim_path"] for result in snapshot_body["material_bindings"]
    ] == snapshot_body["paths"]
    assert [
        result["input_path"] for result in snapshot_body["path_translations"]
    ] == snapshot_body["paths"]
    assert snapshot_body["candidates"][0]["inspection_path"] == "/World/Step"
    assert snapshot_body["candidates"][0]["source_paths"] == ["/World/Step"]
    assert snapshot_body["candidates"][0]["bound_material_path"] == "/World/Looks/Blue"
    assert snapshot_body["candidates"][0]["candidate_reason"] == "renderable_prim"

    oversized_snapshot = client.post(
        f"/sessions/{session_id}/scene/snapshot",
        json={"max_prims": MAX_BATCH_REQUEST_ITEMS + 1},
    )
    assert oversized_snapshot.status_code == 422

    oversized_batch_path = "/" + ("x" * MAX_PRIM_PATH_LENGTH) + "x"
    oversized_batch = client.post(
        f"/sessions/{session_id}/properties:batch",
        json={"prim_paths": [oversized_batch_path]},
    )
    assert oversized_batch.status_code == 422

    stream = client.get(f"/sessions/{session_id}/stream")
    assert stream.status_code == 404


def test_scene_load_route_loads_scene(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()

    create = client.post("/sessions", json={})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    load = client.post(
        f"/sessions/{session_id}/scene",
        json={"scene_path": str(stage_path)},
    )

    assert load.status_code == 200
    body = load.json()
    assert body["status"] == "ready"
    assert body["source_scene_path"] == str(stage_path)


def test_scene_optimize_route_reoptimizes_loaded_scene(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.usda"
    optimized_path = tmp_path / "optimized.usda"
    _write_split_source_stage(source_path)
    _write_split_optimized_stage(optimized_path)
    manager = FakeOptimizingSessionManager(optimized_path=optimized_path)
    client = TestClient(create_app(manager))

    create = client.post("/sessions", json={"scene_path": str(source_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    optimize = client.post(
        f"/sessions/{session_id}/scene/optimize",
        json={
            "enable_deinstance": True,
            "enable_split": True,
            "enable_deduplicate": False,
        },
    )

    assert optimize.status_code == 200
    body = optimize.json()
    assert body["optimization"]["enabled"] is True
    assert body["source_scene_path"] == str(source_path)
    assert body["inspection_scene_path"] == str(optimized_path)
    assert manager.last_optimization_config is not None
    settings = manager.last_optimization_config["scene_optimizer_settings"]
    assert isinstance(settings, dict)
    assert settings["enable_split_meshes"] is True
    assert settings["enable_deduplicate"] is False


def test_scene_restore_exports_preview_without_durable_edits(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()

    create = client.post("/sessions", json={"scene_path": str(stage_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    restore = client.post(f"/sessions/{session_id}/scene/restore")

    assert restore.status_code == 200
    body = restore.json()
    assert body["status"] == "preview_exported"
    assert body["output_usd_path"] is None
    assert Path(body["preview_scene_path"]).exists()
    assert body["restored_edit_count"] == 0
    assert "No durable source-space edits" in body["warnings"][0]


@pytest.mark.parametrize("output_suffix", [".usda", ".usdz"])
def test_scene_restore_zero_edit_export_honors_requested_format(
    tmp_path: Path,
    output_suffix: str,
) -> None:
    from pxr import Usd

    source_usda = tmp_path / "source.usda"
    _write_sample_stage(source_usda)
    source_stage = Usd.Stage.Open(str(source_usda))
    assert source_stage is not None
    source_path = tmp_path / "source.usdc"
    assert source_stage.Flatten().Export(str(source_path))
    output_path = tmp_path / "outputs" / f"restored{output_suffix}"
    client = _client()
    create = client.post("/sessions", json={"scene_path": str(source_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    restore = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={
            "output_usd_path": str(output_path),
            "output_mode": "flattened",
            "overwrite": True,
            "include_preview_artifact": False,
        },
    )

    assert restore.status_code == 200
    body = restore.json()
    assert body["output_usd_path"] == str(output_path)
    assert body["output_mode"] == "flattened"
    assert body["restored_edit_count"] == 0
    assert Usd.Stage.Open(str(output_path)) is not None
    if output_suffix == ".usda":
        assert output_path.read_bytes().startswith(b"#usda")
    else:
        assert zipfile.is_zipfile(output_path)


def test_scene_restore_zero_edit_layer_usdz_packages_relative_dependency(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdGeom

    dependency_dir = tmp_path / "dependencies"
    dependency_dir.mkdir()
    dependency_path = dependency_dir / "geometry.usda"
    dependency_stage = Usd.Stage.CreateNew(str(dependency_path))
    dependency_root = UsdGeom.Xform.Define(
        dependency_stage, "/DependencyRoot"
    ).GetPrim()
    dependency_stage.SetDefaultPrim(dependency_root)
    dependency_stage.GetRootLayer().Save()

    source_path = tmp_path / "source.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_path))
    source_layer.subLayerPaths.append("dependencies/geometry.usda")
    source_layer.Save()
    output_path = tmp_path / "outputs" / "restored.usdz"
    client = _client()
    create = client.post("/sessions", json={"scene_path": str(source_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    restore = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={
            "output_usd_path": str(output_path),
            "output_mode": "layer",
            "overwrite": True,
            "include_preview_artifact": False,
        },
    )

    assert restore.status_code == 200
    with zipfile.ZipFile(output_path) as package:
        assert any(name.endswith("geometry.usda") for name in package.namelist())
    dependency_path.unlink()
    restored_stage = Usd.Stage.Open(str(output_path))
    assert restored_stage is not None
    assert restored_stage.GetPrimAtPath("/DependencyRoot").IsValid()


def test_scene_restore_enforces_configured_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    run_dir = tmp_path / "run"
    allowed_output = run_dir / "materialized.usda"
    protected_output = tmp_path / "host" / "protected.usda"
    _write_sample_stage(source_path)
    run_dir.mkdir()
    protected_output.parent.mkdir()
    protected_output.write_text("protected\n", encoding="utf-8")
    monkeypatch.setenv("CONTENT_WORKBENCH_OUTPUT_ROOTS", str(run_dir))
    client = _client()
    create = client.post("/sessions", json={"scene_path": str(source_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    rejected = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={
            "output_usd_path": str(protected_output),
            "overwrite": True,
            "include_preview_artifact": False,
        },
    )

    assert rejected.status_code == 400
    assert "outside CONTENT_WORKBENCH_OUTPUT_ROOTS" in rejected.json()["detail"]
    assert protected_output.read_text(encoding="utf-8") == "protected\n"

    accepted = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={
            "output_usd_path": str(allowed_output),
            "overwrite": True,
            "include_preview_artifact": False,
        },
    )

    assert accepted.status_code == 200
    assert accepted.json()["output_usd_path"] == str(allowed_output)
    assert allowed_output.is_file()
    assert client.get("/healthz").json()["output_roots"] == [str(run_dir.resolve())]


def test_material_assignment_apply_enforces_configured_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    library_path = tmp_path / "materials.usda"
    run_dir = tmp_path / "run"
    protected_output = tmp_path / "host" / "protected.usda"
    _write_sample_stage(source_path)
    _write_material_library_stage(library_path)
    run_dir.mkdir()
    protected_output.parent.mkdir()
    protected_output.write_text("protected\n", encoding="utf-8")
    monkeypatch.setenv("CONTENT_WORKBENCH_OUTPUT_ROOTS", str(run_dir))
    apply_called = False

    def fail_apply(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal apply_called
        apply_called = True
        raise AssertionError("out-of-root output must be rejected before apply")

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fail_apply,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(source_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    rejected = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(protected_output), "overwrite": True},
    )

    assert rejected.status_code == 400
    assert "outside CONTENT_WORKBENCH_OUTPUT_ROOTS" in rejected.json()["detail"]
    assert protected_output.read_text(encoding="utf-8") == "protected\n"
    assert not apply_called


def test_material_apply_symlink_swap_cannot_escape_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.usda"
    library_path = tmp_path / "materials.usda"
    run_dir = tmp_path / "run"
    output_parent = run_dir / "nested"
    output_path = output_parent / "victim.usda"
    victim_dir = tmp_path / "victim"
    victim_path = victim_dir / output_path.name
    _write_sample_stage(source_path)
    _write_material_library_stage(library_path)
    output_parent.mkdir(parents=True)
    victim_dir.mkdir()
    victim_path.write_text("victim\n", encoding="utf-8")
    monkeypatch.setenv("CONTENT_WORKBENCH_OUTPUT_ROOTS", str(run_dir))

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        Path(context["output_usd_path"]).write_text("#usda 1.0\n", encoding="utf-8")
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {
                "bound_prim_ids": ["/World/Step"],
                "unbound_prim_ids": [],
            },
        }

    original_resolver = sessions_module._secure_output_root_and_relative_path

    def swap_parent_after_resolution(
        candidate: Path,
        *,
        allowed_output_roots: list[Path] | None,
    ) -> tuple[Path, Path]:
        resolved = original_resolver(
            candidate,
            allowed_output_roots=allowed_output_roots,
        )
        output_parent.rename(run_dir / "nested-original")
        output_parent.symlink_to(victim_dir, target_is_directory=True)
        return resolved

    monkeypatch.setattr(
        sessions_module,
        "_run_material_apply_task",
        fake_apply_task,
    )
    monkeypatch.setattr(
        sessions_module,
        "_secure_output_root_and_relative_path",
        swap_parent_after_resolution,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(source_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    response = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path), "overwrite": True},
    )

    assert response.status_code == 500
    assert victim_path.read_text(encoding="utf-8") == "victim\n"
    assert output_parent.is_symlink()


def test_secure_publish_copy_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    output_path.write_text("protected\n", encoding="utf-8")
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")

    def fail_copy(_descriptor: int, _contents: object) -> int:
        raise OSError("injected copy failure")

    monkeypatch.setattr(sessions_module.os, "write", fail_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=True,
        )

    assert output_path.read_text(encoding="utf-8") == "protected\n"
    assert staged_path.read_text(encoding="utf-8") == "replacement\n"
    assert not list(output_root.glob(".result.usda.*.tmp"))


@pytest.mark.parametrize("overwrite", [False, True])
def test_secure_publish_keeps_output_readable_across_service_uid(
    tmp_path: Path,
    overwrite: bool,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    if overwrite:
        output_path.write_text("protected\n", encoding="utf-8")
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    staged_path.chmod(0o600)

    sessions_module._secure_publish_staged_output(
        staged_path,
        output_path=output_path,
        allowed_output_roots=[output_root],
        overwrite=overwrite,
    )

    assert output_path.read_text(encoding="utf-8") == "replacement\n"
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o644


def test_secure_publish_keeps_new_output_directories_cross_uid_traversable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    existing_parent = output_root / "existing"
    existing_parent.mkdir()
    existing_parent.chmod(0o755)
    output_path = existing_parent / "published" / "nested" / "result.usda"
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    original_mkdir = sessions_module.os.mkdir

    def restrictive_mkdir(
        path: str | bytes,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        original_mkdir(path, mode=mode & 0o700, dir_fd=dir_fd)

    monkeypatch.setattr(sessions_module.os, "mkdir", restrictive_mkdir)

    sessions_module._secure_publish_staged_output(
        staged_path,
        output_path=output_path,
        allowed_output_roots=[output_root],
        overwrite=False,
    )

    assert output_path.read_text(encoding="utf-8") == "replacement\n"
    assert stat.S_IMODE(existing_parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(output_path.parent.parent.stat().st_mode) == 0o711
    assert stat.S_IMODE(output_path.parent.stat().st_mode) == 0o711
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o644


@pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="Linux O_PATH is required")
def test_secure_publish_traverses_execute_only_existing_root_with_o_path(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    output_parent = output_root / "shared"
    output_parent.mkdir(parents=True)
    output_root.chmod(0o111)
    output_parent.chmod(0o1777)
    output_path = output_parent / "result.usda"
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")

    try:
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=False,
        )

        assert output_path.read_text(encoding="utf-8") == "replacement\n"
        assert stat.S_IMODE(output_root.stat().st_mode) == 0o111
        assert stat.S_IMODE(output_parent.stat().st_mode) == 0o1777
    finally:
        output_root.chmod(0o700)
        output_parent.chmod(0o700)


def test_verify_anchored_output_parent_rejects_permission_change(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "run"
    output_parent = output_root / "shared"
    output_parent.mkdir(parents=True)
    expected = output_parent.stat(follow_symlinks=False)
    output_parent.chmod(0o777)

    with pytest.raises(RuntimeError, match="identity or permissions changed"):
        sessions_module._verify_anchored_output_parent(
            output_root,
            Path("shared"),
            expected=expected,
        )


def test_secure_publish_post_replace_failure_restores_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    output_path.write_text("protected\n", encoding="utf-8")
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    original_verify = sessions_module._verify_anchored_output_parent
    verify_calls = 0

    def fail_post_replace_verification(*args: object, **kwargs: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise RuntimeError("injected post-replace verification failure")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(
        sessions_module,
        "_verify_anchored_output_parent",
        fail_post_replace_verification,
    )

    with pytest.raises(RuntimeError, match="post-replace verification failure"):
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=True,
        )

    assert output_path.read_text(encoding="utf-8") == "protected\n"
    assert staged_path.read_text(encoding="utf-8") == "replacement\n"
    assert not list(output_root.glob(".result.usda.*.tmp"))
    assert not list(output_root.glob(".result.usda.*.bak"))


def test_secure_publish_failed_rollback_preserves_existing_output_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    output_path.write_text("protected\n", encoding="utf-8")
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    original_verify = sessions_module._verify_anchored_output_parent
    original_replace = sessions_module.os.replace
    verify_calls = 0
    replace_calls = 0

    def fail_post_replace_verification(*args: object, **kwargs: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise RuntimeError("injected post-replace verification failure")
        original_verify(*args, **kwargs)

    def fail_backup_restore(*args: object, **kwargs: object) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected rollback failure")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(
        sessions_module,
        "_verify_anchored_output_parent",
        fail_post_replace_verification,
    )
    monkeypatch.setattr(sessions_module.os, "replace", fail_backup_restore)

    with pytest.raises(
        RuntimeError,
        match="prior output could not be restored; it remains preserved as",
    ) as exc_info:
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=True,
        )

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "injected rollback failure"
    assert output_path.read_text(encoding="utf-8") == "replacement\n"
    backups = list(output_root.glob(".result.usda.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "protected\n"
    assert staged_path.read_text(encoding="utf-8") == "replacement\n"
    assert not list(output_root.glob(".result.usda.*.tmp"))


def test_secure_publish_post_replace_failure_removes_new_output_without_prior_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    original_verify = sessions_module._verify_anchored_output_parent
    verify_calls = 0

    def fail_post_replace_verification(*args: object, **kwargs: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise RuntimeError("injected post-replace verification failure")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(
        sessions_module,
        "_verify_anchored_output_parent",
        fail_post_replace_verification,
    )

    with pytest.raises(RuntimeError, match="post-replace verification failure"):
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=True,
        )

    assert not output_path.exists()
    assert staged_path.read_text(encoding="utf-8") == "replacement\n"
    assert not list(output_root.glob(".result.usda.*.tmp"))
    assert not list(output_root.glob(".result.usda.*.bak"))


def test_secure_publish_non_overwrite_post_link_failure_preserves_new_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    original_verify = sessions_module._verify_anchored_output_parent
    verify_calls = 0

    def fail_post_link_verification(*args: object, **kwargs: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise RuntimeError("injected post-link verification failure")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(
        sessions_module,
        "_verify_anchored_output_parent",
        fail_post_link_verification,
    )

    with pytest.raises(RuntimeError, match="post-link verification failure"):
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=False,
        )

    assert output_path.read_text(encoding="utf-8") == "replacement\n"
    assert staged_path.read_text(encoding="utf-8") == "replacement\n"
    assert not list(output_root.glob(".result.usda.*.tmp"))
    assert not list(output_root.glob(".result.usda.*.bak"))


def test_secure_publish_rejects_output_swap_before_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    output_path = output_root / "result.usda"
    output_path.write_text("protected\n", encoding="utf-8")
    staged_path = tmp_path / "staged.usda"
    staged_path.write_text("replacement\n", encoding="utf-8")
    original_copy = sessions_module._copy_staged_output_to_anchored_fd

    def swap_output_after_copy(*args: object, **kwargs: object) -> str:
        temporary_name = original_copy(*args, **kwargs)
        raced_path = output_root / ".raced-output"
        raced_path.write_text("concurrent replacement\n", encoding="utf-8")
        os.replace(raced_path, output_path)
        return temporary_name

    monkeypatch.setattr(
        sessions_module,
        "_copy_staged_output_to_anchored_fd",
        swap_output_after_copy,
    )

    with pytest.raises(
        RuntimeError,
        match="securely preserve the existing material output",
    ):
        sessions_module._secure_publish_staged_output(
            staged_path,
            output_path=output_path,
            allowed_output_roots=[output_root],
            overwrite=True,
        )

    assert output_path.read_text(encoding="utf-8") == "concurrent replacement\n"
    assert not list(output_root.glob(".result.usda.*.tmp"))
    assert not list(output_root.glob(".result.usda.*.bak"))


def test_material_apply_rebases_assets_without_mutating_cached_staging_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd

    source_path = tmp_path / "source.usda"
    library_path = tmp_path / "materials.usda"
    run_dir = tmp_path / "run"
    output_path = run_dir / "published" / "result.usda"
    _write_sample_stage(source_path)
    _write_material_library_stage(library_path)
    run_dir.mkdir()
    monkeypatch.setenv("CONTENT_WORKBENCH_OUTPUT_ROOTS", str(run_dir))
    captured: dict[str, object] = {}

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        task_output = Path(context["output_usd_path"])
        dependency = task_output.parent / "materials" / "surface.mdl"
        dependency.parent.mkdir()
        dependency.write_text("mdl 1.0;\n", encoding="utf-8")
        task_output.write_text(
            '#usda 1.0\n\ndef "Root" {\n'
            "    custom asset dependency = @materials/surface.mdl@\n"
            "}\n",
            encoding="utf-8",
        )
        captured["dependency"] = dependency
        captured["layer"] = Sdf.Layer.FindOrOpen(str(task_output))
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {
                "bound_prim_ids": ["/World/Step"],
                "unbound_prim_ids": [],
            },
        }

    monkeypatch.setattr(
        sessions_module,
        "_run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(source_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    response = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path)},
    )

    assert response.status_code == 200
    stage = Usd.Stage.Open(str(output_path))
    assert stage is not None
    asset_path = stage.GetPrimAtPath("/Root").GetAttribute("dependency").Get().path
    assert (
        Path(stage.GetRootLayer().ComputeAbsolutePath(asset_path)).resolve()
        == Path(captured["dependency"]).resolve()
    )
    cached_layer = captured["layer"]
    assert isinstance(cached_layer, Sdf.Layer)
    cached_value = cached_layer.GetPrimAtPath("/Root").attributes["dependency"].default
    assert cached_value.path == "materials/surface.mdl"


def test_scene_restore_rejects_no_output_when_preview_disabled(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()

    create = client.post("/sessions", json={"scene_path": str(stage_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    restore = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={"include_preview_artifact": False},
    )

    assert restore.status_code == 400
    assert "output_usd_path or include_preview_artifact" in restore.json()["detail"]


def test_scene_restore_projects_material_overrides_to_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_restored.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    captured_context: dict[str, Any] = {}

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        captured_context.update(context)
        Path(context["output_usd_path"]).write_text("#usda 1.0\n", encoding="utf-8")
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {
                "total_prims": 1,
                "materials_applied": 1,
                "materials_created": 1,
                "failed": 0,
                "bound_prim_ids": ["/World/Step"],
                "unbound_prim_ids": [],
            },
        }

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    create = client.post("/sessions", json={"scene_path": str(stage_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    restore = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={
            "output_usd_path": str(output_path),
            "output_mode": "layer",
            "material_profile": "preview_surface",
        },
    )

    assert restore.status_code == 200
    body = restore.json()
    assert body["status"] == "restored"
    assert body["output_usd_path"] == str(output_path)
    assert Path(body["output_usd_path"]).exists()
    assert Path(body["preview_scene_path"]).exists()
    assert body["restored_edit_count"] == 1
    assert body["restored_source_prim_paths"] == ["/World/Step"]
    assert body["unbound_source_prim_paths"] == []
    assert body["unresolved_mappings"] == []
    assert body["material_apply"]["applied_assignment_count"] == 1
    assert body["material_apply"]["applied_source_prim_paths"] == ["/World/Step"]
    assert body["material_apply"]["material_library_path"] == str(library_path)
    assert captured_context["input_usd_path"] == str(stage_path)
    predictions_path = Path(captured_context["predictions_path"])
    assert predictions_path.exists()
    assert predictions_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {"id": "/World/Step", "material": "Steel Painted Orange"},
            sort_keys=True,
        )
    ]


def test_scene_restore_handles_preview_only_material_override(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    create = client.post("/sessions", json={"scene_path": str(stage_path)})
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "display_name": "Debug Red",
                    "preview_color": [1.0, 0.0, 0.0],
                },
            },
        },
    )
    assert override.status_code == 200

    restore = client.post(f"/sessions/{session_id}/scene/restore")

    assert restore.status_code == 200
    body = restore.json()
    assert body["status"] == "preview_exported"
    assert body["output_usd_path"] is None
    assert Path(body["preview_scene_path"]).exists()
    assert body["restored_edit_count"] == 0
    assert body["material_apply"] is None
    assert any("preview-only" in warning for warning in body["warnings"])


def test_durable_material_override_requires_source_mapping(tmp_path: Path) -> None:
    library_path = tmp_path / "materials.usda"
    _write_material_library_stage(library_path)
    material = {
        "source": "material_library",
        "library_path": str(library_path),
        "material_path": "/World/Looks/Steel_Painted_Orange",
        "material_name": "Steel Painted Orange",
    }
    unresolved_override = MaterialOverride(
        prim_path="/World/Step",
        space="inspection",
        inspection_prim_paths=["/World/Step"],
        source_prim_paths=[],
        material=material,
    )
    resolved_override = unresolved_override.model_copy(
        update={"source_prim_paths": ["/World/Step"]}
    )

    assert not _is_durable_material_override(
        unresolved_override,
        source_scene_path=tmp_path / "source.usda",
    )
    assert _is_durable_material_override(
        resolved_override,
        source_scene_path=tmp_path / "source.usda",
    )


def _material_override_for_trim_tests(
    *, source_prim_paths: list[str], inspection_prim_paths: list[str]
) -> MaterialOverride:
    return MaterialOverride(
        prim_path=source_prim_paths[0] if source_prim_paths else "/World/Bolt",
        space="source" if source_prim_paths else "inspection",
        source_prim_paths=source_prim_paths,
        inspection_prim_paths=inspection_prim_paths,
        material={
            "source": "material_library",
            "library_path": "/materials.usda",
            "material_path": "/World/Looks/Steel",
            "material_name": "Steel",
        },
    )


def test_trim_material_override_coverage_narrows_without_desyncing() -> None:
    # Only the source side has explicit coverage; trimming one of its two
    # entries must narrow the override, not drop it, even though the
    # (never-tracked) inspection side stays empty throughout.
    override = _material_override_for_trim_tests(
        source_prim_paths=["/World/BoltA", "/World/BoltB"],
        inspection_prim_paths=[],
    )
    trimmed = _trim_material_override_coverage(
        override,
        source_paths={"/World/BoltA"},
        inspection_paths=set(),
    )
    assert trimmed is not None
    assert trimmed.source_prim_paths == ["/World/BoltB"]
    assert trimmed.inspection_prim_paths == []


def test_trim_material_override_coverage_marks_source_exhausted_on_side_desync() -> (
    None
):
    """A one-sided full trim must narrow, not drop, and flag the exhausted side.

    Regression test: `source_prim_paths` and `inspection_prim_paths` are two
    path-space representations of the *same* coverage, but a new command's
    overlap set can fully consume only one side (e.g. a shared runtime alias
    scenario where the source-space and inspection-space overlap checks
    disagree). Dropping the whole override in that case would discard the
    other side's still-valid, unrelated coverage. Instead the override must
    be narrowed and `source_coverage_exhausted` set, so downstream consumers
    know not to fall back to `override.prim_path` for the emptied side.
    """
    override = _material_override_for_trim_tests(
        source_prim_paths=["/World/BoltA"],
        inspection_prim_paths=["/World/Optimized/SharedBolt"],
    )
    trimmed = _trim_material_override_coverage(
        override,
        source_paths={"/World/BoltA"},
        inspection_paths=set(),
    )
    assert trimmed is not None
    assert trimmed.source_prim_paths == []
    assert trimmed.inspection_prim_paths == ["/World/Optimized/SharedBolt"]
    assert trimmed.source_coverage_exhausted is True


def test_trim_material_override_coverage_preserves_unique_source_on_inspection_trim() -> (
    None
):
    """Emptying only inspection_prim_paths must narrow, not drop.

    Regression test: candidate A maps to a shared runtime alias plus its own
    unique "extra" fragment (`source_prim_paths=["extra"]`,
    `inspection_prim_paths=["shared"]`). A sibling candidate B, which maps
    only to the shared alias, is excluded from A's coalesce merge and binds
    to "shared" on its own command. That command's overlap only trims A's
    `inspection_prim_paths` (the shared alias A no longer owns); A's
    `source_prim_paths` covering its own unique "extra" fragment is
    untouched by B's command and must survive. Dropping the whole override
    here would discard A's still-valid unique fragment for no reason --
    regardless of whether a downstream consumer would also fall back to
    `override.prim_path` for the emptied side (as
    `_material_overrides_for_inspection` does for missing inspection
    coverage), since a fallback substituting stale coverage is a second,
    independent problem, not a justification for keeping this one.
    """
    override = _material_override_for_trim_tests(
        source_prim_paths=["/World/BoltA_Extra"],
        inspection_prim_paths=["/World/Optimized/SharedBolt"],
    )
    trimmed = _trim_material_override_coverage(
        override,
        source_paths=set(),
        inspection_paths={"/World/Optimized/SharedBolt"},
    )
    assert trimmed is not None
    assert trimmed.source_prim_paths == ["/World/BoltA_Extra"]
    assert trimmed.inspection_prim_paths == []


def test_trim_material_override_coverage_drops_when_both_sides_consumed() -> None:
    override = _material_override_for_trim_tests(
        source_prim_paths=["/World/BoltA"],
        inspection_prim_paths=["/World/Optimized/SharedBolt"],
    )
    trimmed = _trim_material_override_coverage(
        override,
        source_paths={"/World/BoltA"},
        inspection_paths={"/World/Optimized/SharedBolt"},
    )
    assert trimmed is None


def test_trim_material_override_coverage_preserves_unique_inspection_when_source_exhausted() -> (
    None
):
    """kimbyn round-3 scenario: source fully consumed, inspection keeps a unique path.

    Regression test: an override covers a shared runtime alias
    ("/World/Optimized/Shared") plus its own unique fragment
    ("/World/Optimized/Unique") on the inspection side, backed by a single
    source path. A sibling command consumes the source path *and* the shared
    inspection alias, but not the unique inspection fragment. The override
    must be narrowed (not dropped) so the unique fragment survives, and
    `source_coverage_exhausted` must be set so fallback consumers do not
    reintroduce the now-superseded source path.
    """
    override = _material_override_for_trim_tests(
        source_prim_paths=["/World/BoltA"],
        inspection_prim_paths=[
            "/World/Optimized/Shared",
            "/World/Optimized/Unique",
        ],
    )
    trimmed = _trim_material_override_coverage(
        override,
        source_paths={"/World/BoltA"},
        inspection_paths={"/World/Optimized/Shared"},
    )
    assert trimmed is not None
    assert trimmed.source_prim_paths == []
    assert trimmed.inspection_prim_paths == ["/World/Optimized/Unique"]
    assert trimmed.source_coverage_exhausted is True
    assert _override_source_paths_or_fallback(trimmed) == []


def test_override_source_paths_or_fallback_uses_prim_path_when_never_tracked() -> None:
    override = _material_override_for_trim_tests(
        source_prim_paths=[],
        inspection_prim_paths=["/World/Optimized/Shared"],
    )
    assert _override_source_paths_or_fallback(override) == [override.prim_path]


def test_trim_material_override_coverage_keeps_override_when_no_overlap() -> None:
    override = _material_override_for_trim_tests(
        source_prim_paths=["/World/BoltA"],
        inspection_prim_paths=["/World/Optimized/SharedBolt"],
    )
    trimmed = _trim_material_override_coverage(
        override,
        source_paths={"/World/Unrelated"},
        inspection_paths=set(),
    )
    assert trimmed is override


def test_create_session_clear_materials_starts_from_clean_appearance(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "authored_appearance.usda"
    _write_authored_appearance_stage(stage_path)
    from pxr import Usd, UsdGeom

    source_stage = Usd.Stage.Open(str(stage_path))
    assert source_stage is not None
    UsdGeom.SetStageUpAxis(source_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(source_stage, 1.0)
    source_stage.GetRootLayer().customLayerData = {
        "SimReady_Metadata": '{"identifier":"appearance"}'
    }
    source_stage.GetRootLayer().Save()
    client = _client()

    create = client.post(
        "/sessions",
        json={"scene_path": str(stage_path), "clear_materials": True},
    )

    assert create.status_code == 201
    session = create.json()
    session_id = session["session_id"]
    clean_path = Path(session["artifacts"]["material_cleared_scene_path"])
    assert session["clear_materials"] is True
    assert session["source_scene_path"] == str(stage_path)
    assert session["inspection_scene_path"] == str(clean_path)
    assert session["scene_path"] == str(clean_path)
    assert clean_path.exists()

    material = client.get(
        f"/sessions/{session_id}/material-binding",
        params={"prim_path": "/World/Step"},
    )
    assert material.status_code == 200
    material_body = material.json()
    assert material_body["bound_material_path"] is None
    assert material_body["direct_targets"] == []
    assert material_body["material_override"] is None

    from pxr import UsdShade

    clean_stage = Usd.Stage.Open(str(clean_path))
    assert clean_stage is not None
    assert UsdGeom.GetStageUpAxis(clean_stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(clean_stage) == 1.0
    assert (
        clean_stage.GetRootLayer().customLayerData["SimReady_Metadata"]
        == '{"identifier":"appearance"}'
    )
    assert str(clean_stage.GetRootLayer().subLayerPaths[0]) == str(stage_path)
    mesh = clean_stage.GetPrimAtPath("/World/Step")
    bound_material, _rel = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert not bound_material.GetPrim().IsValid()
    assert mesh.GetAttribute("primvars:displayColor").Get() is None


def test_scene_snapshot_deep_root_preserves_hidden_ancestor_visibility(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "hidden_ancestor.usda"
    _write_hidden_ancestor_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    snapshot = client.post(
        f"/sessions/{session_id}/scene/snapshot",
        json={"root_prim_path": "/World/HiddenGroup/Deep"},
    )

    assert snapshot.status_code == 200
    snapshot_body = snapshot.json()
    assert snapshot_body["root_prim_path"] == "/World/HiddenGroup/Deep"
    assert snapshot_body["summary"]["candidate_count"] == 0
    assert snapshot_body["candidates"] == []


def test_visual_debug_commands_update_session_state(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    select = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "select", "payload": {"paths": ["/World/Step"]}},
    )
    assert select.status_code == 200
    assert select.json()["session"]["view"]["selected_prims"] == ["/World/Step"]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World",
                "material": {"preview_color": [1.0, 0.0, 0.0]},
            },
        },
    )
    assert override.status_code == 200
    override_body = override.json()
    assert override_body["session"]["view"]["material_overrides"][0] == {
        "prim_path": "/World",
        "material": {"preview_color": [1.0, 0.0, 0.0]},
        "mode": "material_assignment",
        "unbind_existing": True,
        "remove_material_libraries": False,
        "space": "source",
        "source_prim_paths": ["/World"],
        "inspection_prim_paths": ["/World"],
        "source_coverage_exhausted": False,
    }
    preview_path = Path(override_body["session"]["artifacts"]["preview_scene_path"])
    assert preview_path.exists()

    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    assert str(preview_stage.GetRootLayer().subLayerPaths[0]) == str(stage_path)
    material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/Step")
    ).ComputeBoundMaterial()
    assert str(material.GetPath()).startswith(
        "/World/PreviewMaterials/PreviewMaterial_World_"
    )

    material = client.get(
        f"/sessions/{session_id}/material-binding",
        params={"prim_path": "/World/Step"},
    )
    assert material.status_code == 200
    assert material.json()["material_override"]["prim_path"] == "/World"
    assert material.json()["material_override"]["material"] == {
        "preview_color": [1.0, 0.0, 0.0]
    }

    clear = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "clear_visual_overrides", "payload": {}},
    )
    assert clear.status_code == 200
    view = clear.json()["session"]["view"]
    assert view["selected_prims"] == []
    assert view["material_overrides"] == []


def test_preview_exports_prune_stale_preview_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    workspace_root = tmp_path / "workspace"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest(scene_path=str(stage_path)))

    for _ in range(PREVIEW_SCENE_RETENTION_COUNT + 3):
        manager.export_preview_scene(session.session_id)

    preview_dir = workspace_root / session.session_id / "previews"
    preview_paths = list(preview_dir.glob("preview-*.usda"))
    assert len(preview_paths) == PREVIEW_SCENE_RETENTION_COUNT
    latest_preview = manager.get_session(
        session.session_id
    ).artifacts.preview_scene_path
    assert latest_preview is not None
    assert Path(latest_preview).exists()


def test_create_session_cleans_workspace_when_initial_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())

    with pytest.raises(FileNotFoundError):
        manager.create_session(
            CreateSessionRequest(scene_path=str(tmp_path / "missing.usda"))
        )

    assert manager.active_session_count == 0
    assert list(workspace_root.iterdir()) == []


def test_create_session_accepts_usdz_scene(tmp_path: Path) -> None:
    source_path = tmp_path / "ladder_part.usda"
    usdz_path = tmp_path / "ladder_part.usdz"
    _write_sample_stage(source_path)
    _package_material_apply_usdz(source_path, usdz_path)

    client = _client()
    response = client.post("/sessions", json={"scene_path": str(usdz_path)})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready"
    assert body["source_scene_path"] == str(usdz_path)
    assert body["inspection_scene_path"] == str(usdz_path)


def test_invalid_session_id_does_not_create_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    client = _client()

    response = client.get("/sessions/not-a-uuid/renders/render.png")

    assert response.status_code == 400
    assert response.json()["detail"] == "session_id must be a canonical UUID"
    assert list(workspace_root.iterdir()) == []


def test_preview_export_removes_partial_scene_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    workspace_root = tmp_path / "workspace"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest(scene_path=str(stage_path)))

    def fake_export_preview_stage(*, output_path: Path, **_kwargs: object) -> None:
        output_path.write_text("#usda 1.0\n", encoding="utf-8")
        raise RuntimeError("export failed")

    monkeypatch.setattr(
        "content_workbench.sessions._export_preview_stage",
        fake_export_preview_stage,
    )

    with pytest.raises(RuntimeError, match="export failed"):
        manager.export_preview_scene(session.session_id)

    preview_dir = workspace_root / session.session_id / "previews"
    assert list(preview_dir.glob("preview-*.usda")) == []
    assert manager.get_session(session.session_id).artifacts.preview_scene_path is None


def test_close_session_does_not_recreate_missing_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest())
    workspace = workspace_root / session.session_id
    assert workspace.exists()

    shutil.rmtree(workspace)
    manager.close_session(session.session_id)

    assert not workspace.exists()


def test_close_session_removes_existing_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest())
    workspace = workspace_root / session.session_id
    (workspace / "renders").mkdir()
    (workspace / "renders" / "render.png").write_bytes(b"png")

    manager.close_session(session.session_id)

    assert not workspace.exists()


def test_close_final_session_shuts_down_idle_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShutdownRenderer(FakeRenderer):
        shutdown_count = 0

        def shutdown(self) -> None:
            self.shutdown_count += 1

    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    renderer = ShutdownRenderer()
    manager = SessionManager(renderer=renderer)
    first = manager.create_session(CreateSessionRequest())
    second = manager.create_session(CreateSessionRequest())

    manager.close_session(first.session_id)
    assert renderer.shutdown_count == 0

    manager.close_session(second.session_id)
    assert renderer.shutdown_count == 1


def test_close_session_removes_resolved_pinned_preview_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    workspace_root = parent / ".." / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest())
    workspace = manager._workspace_path(session.session_id)
    pinned_path = (workspace / "previews" / "preview-pinned.usda").resolve()
    manager._pinned_preview_paths.add(pinned_path)

    manager.close_session(session.session_id)

    assert pinned_path not in manager._pinned_preview_paths


def test_close_session_waits_for_active_workspace_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest())
    workspace = workspace_root / session.session_id
    closed: list[bool] = []

    with manager._lock:
        manager._begin_workspace_operation(session.session_id)

    thread = threading.Thread(
        target=lambda: closed.append(
            manager.close_session(session.session_id).status == "closed"
        )
    )
    thread.start()
    time.sleep(0.05)

    assert thread.is_alive()
    assert workspace.exists()

    with manager._lock:
        manager._end_workspace_operation(session.session_id)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert closed == [True]
    assert not workspace.exists()


def test_close_session_times_out_for_stuck_workspace_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    monkeypatch.setenv("CONTENT_WORKBENCH_RENDER_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("CONTENT_WORKBENCH_MATERIAL_APPLY_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(
        "content_workbench.sessions.CLOSE_SESSION_WORKSPACE_TIMEOUT_SECONDS",
        0.01,
    )
    manager = SessionManager(renderer=FakeRenderer())
    session = manager.create_session(CreateSessionRequest())

    with manager._lock:
        manager._begin_workspace_operation(session.session_id)
    try:
        with pytest.raises(
            TimeoutError,
            match="Timed out waiting for active workspace operation",
        ):
            manager.close_session(session.session_id)
    finally:
        with manager._lock:
            manager._end_workspace_operation(session.session_id)

    manager.close_session(session.session_id)


def test_app_shutdown_releases_sessions_and_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ShutdownRenderer(FakeRenderer):
        shutdown_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

    workspace_root = tmp_path / "workspace"
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    renderer = ShutdownRenderer()
    manager = SessionManager(renderer=renderer)

    with TestClient(create_app(manager)) as client:
        session = client.post("/sessions", json={"scene_path": str(stage_path)}).json()
        workspace = workspace_root / session["session_id"]
        assert workspace.exists()

    assert renderer.shutdown_called is True
    assert manager.active_session_count == 0
    assert not workspace.exists()


def test_render_timeout_releases_workspace_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowRenderer(FakeRenderer):
        def render(self, **kwargs: Any) -> float:
            time.sleep(0.05)
            return super().render(**kwargs)

    workspace_root = tmp_path / "workspace"
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    monkeypatch.setenv("CONTENT_WORKBENCH_RENDER_TIMEOUT_SECONDS", "0.01")
    manager = SessionManager(renderer=SlowRenderer())
    session = manager.create_session(CreateSessionRequest(scene_path=str(stage_path)))

    with pytest.raises(TimeoutError, match="Timed out waiting for OvRTX render"):
        manager.render_session(session.session_id, RenderRequest(width=64, height=48))

    assert manager._active_workspace_ops == {}
    assert manager._pinned_preview_paths == set()
    manager.shutdown()


def test_render_cleanup_runs_when_session_unavailable_after_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MarkingRenderer(FakeRenderer):
        rendered = False

        def render(self, **kwargs: Any) -> float:
            elapsed = super().render(**kwargs)
            self.rendered = True
            return elapsed

    class UnavailableAfterRenderManager(SessionManager):
        def _require_ready_session(self, session_id: str) -> SceneSession:
            if renderer.rendered:
                raise KeyError(session_id)
            return super()._require_ready_session(session_id)

    workspace_root = tmp_path / "workspace"
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    renderer = MarkingRenderer()
    manager = UnavailableAfterRenderManager(renderer=renderer)
    session = manager.create_session(CreateSessionRequest(scene_path=str(stage_path)))

    with pytest.raises(KeyError):
        manager.render_session(
            session.session_id,
            RenderRequest(width=64, height=48, save_camera_json=True),
        )

    assert manager._active_workspace_ops == {}
    assert manager._pinned_preview_paths == set()


def test_pick_cleanup_runs_when_session_unavailable_after_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MarkingRenderer(FakeRenderer):
        picked = False

        def pick(self, **kwargs: Any) -> PickRenderResult:
            result = super().pick(**kwargs)
            self.picked = True
            return result

    class UnavailableAfterPickManager(SessionManager):
        def _require_ready_session(self, session_id: str) -> SceneSession:
            if renderer.picked:
                raise KeyError(session_id)
            return super()._require_ready_session(session_id)

    workspace_root = tmp_path / "workspace"
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    renderer = MarkingRenderer()
    manager = UnavailableAfterPickManager(renderer=renderer)
    session = manager.create_session(CreateSessionRequest(scene_path=str(stage_path)))

    with pytest.raises(KeyError):
        manager.pick_session(
            session.session_id,
            PickRequest(x=0, y=0, width=64, height=48),
        )

    assert manager._active_workspace_ops == {}
    assert manager._pinned_preview_paths == set()


def test_preview_prune_preserves_protected_in_flight_scene(tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    preview_dir.mkdir()
    protected_path = preview_dir / "preview-protected.usda"
    keep_path = preview_dir / "preview-keep.usda"
    protected_path.write_text("#usda 1.0\n", encoding="utf-8")
    keep_path.write_text("#usda 1.0\n", encoding="utf-8")
    for index in range(PREVIEW_SCENE_RETENTION_COUNT + 3):
        (preview_dir / f"preview-stale-{index}.usda").write_text(
            "#usda 1.0\n",
            encoding="utf-8",
        )

    _prune_preview_scenes(
        preview_dir,
        keep_path=keep_path,
        protected_paths={protected_path.resolve()},
        retention_count=1,
    )

    assert keep_path.exists()
    assert protected_path.exists()


def test_preview_isolation_hides_non_isolated_imageables(tmp_path: Path) -> None:
    stage_path = tmp_path / "isolation.usda"
    preview_path = tmp_path / "preview.usda"
    _write_isolation_stage(stage_path)

    _export_preview_stage(
        source_path=stage_path,
        output_path=preview_path,
        root_prim_path="/World",
        overrides=[],
        hidden_prims=[],
        isolated_prims=["/World/KeepGroup/KeepMesh"],
    )

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(preview_path))
    assert stage is not None
    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath("/World")).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )
    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/KeepGroup")).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )
    assert (
        UsdGeom.Imageable(
            stage.GetPrimAtPath("/World/KeepGroup/KeepMesh")
        ).ComputeVisibility()
        == UsdGeom.Tokens.inherited
    )
    assert (
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Backdrop")).ComputeVisibility()
        == UsdGeom.Tokens.invisible
    )


def test_material_override_can_bind_material_from_library(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200
    preview_path = Path(override.json()["session"]["artifacts"]["preview_scene_path"])

    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    assert str(preview_stage.GetRootLayer().subLayerPaths[0]) == str(stage_path)
    assert str(preview_stage.GetRootLayer().subLayerPaths[1]) == str(library_path)
    material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/Step")
    ).ComputeBoundMaterial()
    assert str(material.GetPath()) == "/World/Looks/Steel_Painted_Orange"


def test_material_assignment_apply_uses_workbench_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    captured_context: dict[str, Any] = {}
    captured_task_options: dict[str, Any] = {}

    def fake_apply_task(context: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured_context.update(context)
        captured_task_options.update(kwargs)
        Path(context["output_usd_path"]).write_text("#usda 1.0\n", encoding="utf-8")
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {
                "total_prims": 1,
                "materials_applied": 1,
                "materials_created": 1,
                "failed": 0,
                "bound_prim_ids": ["/World/Step"],
                "unbound_prim_ids": [],
            },
        }

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    assignments = client.get(f"/sessions/{session_id}/authoring/material-assignments")
    assert assignments.status_code == 200
    assignment = assignments.json()["assignments"][0]
    assert assignment["source_prim_paths"] == ["/World/Step"]
    assert assignment["inspection_prim_paths"] == ["/World/Step"]
    assert assignment["material_library_path"] == str(library_path)
    assert assignment["material_path"] == "/World/Looks/Steel_Painted_Orange"

    apply = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={
            "output_usd_path": str(output_path),
            "output_mode": "layer",
            "material_profile": "preview_surface",
        },
    )

    assert apply.status_code == 200
    body = apply.json()
    assert body["status"] == "applied"
    assert body["output_usd_path"] == str(output_path)
    assert Path(body["output_usd_path"]).exists()
    assert body["material_library_path"] == str(library_path)
    assert body["applied_assignment_count"] == 1
    assert body["materials_applied"] == {
        "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
    }
    assert captured_context["input_usd_path"] == str(stage_path)
    task_output_path = Path(captured_context["output_usd_path"])
    assert task_output_path != output_path
    assert not task_output_path.is_relative_to(output_path.parent)
    assert "authoring" in task_output_path.parts
    assert task_output_path.name.startswith(f".{output_path.stem}.")
    assert task_output_path.suffix == output_path.suffix
    assert not task_output_path.exists()
    assert captured_context["resolved_materials"] == {
        "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
    }
    assert captured_context["is_library_based_mapping"] is True
    assert captured_context["material_library_path"] == str(library_path)
    assert captured_context["layer_only"] is True
    assert captured_context["flatten_output"] is False
    assert captured_context["material_profile"] == "preview_surface"
    assert captured_task_options["executor"] is not None
    assert captured_task_options["timeout_seconds"] == 300.0

    predictions_path = Path(captured_context["predictions_path"])
    assert predictions_path.exists()
    assert predictions_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {"id": "/World/Step", "material": "Steel Painted Orange"},
            sort_keys=True,
        )
    ]
    assignments_path = Path(body["assignments_path"])
    assert assignments_path.exists()
    assert (
        json.loads(assignments_path.read_text(encoding="utf-8"))["assignments"][0][
            "prim_path"
        ]
        == "/World/Step"
    )

    session = client.get(f"/sessions/{session_id}").json()
    assert session["artifacts"]["last_apply_output_path"] == str(output_path)
    assert session["artifacts"]["last_apply_assignments_path"] == str(assignments_path)
    assert session["artifacts"]["last_apply_predictions_path"] == str(predictions_path)


def test_material_apply_bound_source_paths_rejects_unbound_targets() -> None:
    prediction_records = [{"id": "/World/A", "material": "Steel"}]

    with pytest.raises(RuntimeError, match="exact bound/unbound prim coverage"):
        _material_apply_bound_source_paths(
            assignment_stats={},
            prediction_records=prediction_records,
        )

    with pytest.raises(RuntimeError, match="left requested prim targets unbound"):
        _material_apply_bound_source_paths(
            assignment_stats={
                "bound_prim_ids": [],
                "unbound_prim_ids": ["/World/A"],
            },
            prediction_records=prediction_records,
        )

    assert _material_apply_bound_source_paths(
        assignment_stats={
            "bound_prim_ids": ["/World/A"],
            "unbound_prim_ids": [],
        },
        prediction_records=prediction_records,
    ) == (["/World/A"], [])

    assert _material_apply_bound_source_paths(
        assignment_stats={
            "bound_prim_ids": ["/World/A"],
            "unbound_prim_ids": ["/World/B"],
        },
        prediction_records=[
            *prediction_records,
            {"id": "/World/B", "material": "Steel"},
        ],
        fail_on_invalid_assignment=False,
    ) == (["/World/A"], ["/World/B"])

    with pytest.raises(RuntimeError, match="inconsistent prim binding coverage"):
        _material_apply_bound_source_paths(
            assignment_stats={
                "bound_prim_ids": ["/World/A"],
                "unbound_prim_ids": ["/World/Unexpected"],
            },
            prediction_records=prediction_records,
            fail_on_invalid_assignment=False,
        )


def test_scene_restore_writes_best_effort_output_for_unbound_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        Path(context["output_usd_path"]).write_text("#usda 1.0\n", encoding="utf-8")
        return {
            **context,
            "assignment_stats": {
                "bound_prim_ids": [],
                "unbound_prim_ids": ["/World/Step"],
            },
        }

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    restore = client.post(
        f"/sessions/{session_id}/scene/restore",
        json={
            "output_usd_path": str(output_path),
            "fail_on_invalid_assignment": False,
            "include_preview_artifact": False,
        },
    )

    assert restore.status_code == 200
    body = restore.json()
    assert output_path.is_file()
    assert body["restored_edit_count"] == 0
    assert body["restored_source_prim_paths"] == []
    assert body["unbound_source_prim_paths"] == ["/World/Step"]
    assert body["warnings"] == [
        "Material apply left requested prim targets unbound: ['/World/Step']"
    ]
    assert body["material_apply"]["applied_assignment_count"] == 0
    assert body["material_apply"]["unbound_source_prim_paths"] == ["/World/Step"]


def test_material_assignment_apply_packages_usdz_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf

    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usdz"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    captured_context: dict[str, Any] = {}
    captured_package: dict[str, Path] = {}

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        captured_context.update(context)
        task_layer = Sdf.Layer.CreateNew(str(context["output_usd_path"]))
        task_layer.Save()
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {
                "total_prims": 1,
                "materials_applied": 1,
                "bound_prim_ids": ["/World/Step"],
                "unbound_prim_ids": [],
            },
        }

    def fake_package_usdz(source_usd_path: Path, usdz_path: Path) -> None:
        captured_package["source_usd_path"] = source_usd_path
        captured_package["usdz_path"] = usdz_path
        with zipfile.ZipFile(usdz_path, "w", zipfile.ZIP_STORED) as package:
            package.writestr("root.usdc", source_usd_path.read_bytes())

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    monkeypatch.setattr(
        "content_workbench.sessions._package_material_apply_usdz",
        fake_package_usdz,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    apply = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path), "overwrite": True},
    )

    assert apply.status_code == 200
    body = apply.json()
    assert body["output_usd_path"] == str(output_path)
    assert zipfile.is_zipfile(output_path)
    task_output_path = Path(captured_context["output_usd_path"])
    assert task_output_path.suffix == ".usdc"
    assert task_output_path.name.startswith(f".{output_path.stem}.")
    assert captured_package["source_usd_path"] != task_output_path
    assert captured_package["source_usd_path"].parent == task_output_path.parent
    assert captured_package["source_usd_path"].suffix == task_output_path.suffix
    assert captured_package["usdz_path"].suffix == ".usdz"
    assert captured_package["usdz_path"] != output_path
    assert not captured_package["source_usd_path"].exists()
    assert not captured_package["usdz_path"].exists()
    assert not task_output_path.exists()


def test_material_assignment_apply_usdz_packages_relative_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf, Usd, UsdGeom

    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usdz"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    captured_dependency: dict[str, Path] = {}

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        task_output_path = Path(context["output_usd_path"])
        dependency_path = task_output_path.parent / "relative-geometry.usda"
        dependency_stage = Usd.Stage.CreateNew(str(dependency_path))
        dependency_root = UsdGeom.Xform.Define(
            dependency_stage, "/DependencyRoot"
        ).GetPrim()
        dependency_stage.SetDefaultPrim(dependency_root)
        dependency_stage.GetRootLayer().Save()
        task_layer = Sdf.Layer.CreateNew(str(task_output_path))
        task_layer.subLayerPaths.append(dependency_path.name)
        task_layer.Save()
        captured_dependency["path"] = dependency_path
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {
                "total_prims": 1,
                "materials_applied": 1,
                "bound_prim_ids": ["/World/Step"],
                "unbound_prim_ids": [],
            },
        }

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    apply = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path), "overwrite": True},
    )

    assert apply.status_code == 200
    with zipfile.ZipFile(output_path) as package:
        assert any(
            name.endswith("relative-geometry.usda") for name in package.namelist()
        )
    captured_dependency["path"].unlink()
    packaged_stage = Usd.Stage.Open(str(output_path))
    assert packaged_stage is not None
    assert packaged_stage.GetPrimAtPath("/DependencyRoot").IsValid()


def test_package_material_apply_usdz_writes_openable_zip(tmp_path: Path) -> None:
    source_path = tmp_path / "source.usda"
    usdz_path = tmp_path / "source.usdz"
    _write_sample_stage(source_path)

    _package_material_apply_usdz(source_path, usdz_path)

    assert zipfile.is_zipfile(usdz_path)

    from pxr import Usd

    assert Usd.Stage.Open(str(usdz_path)) is not None


def test_material_assignment_apply_context_runs_real_apply_task(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "outputs" / "ladder_materials.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    predictions_path.write_text(
        json.dumps(
            {"id": "/World/Step", "material": "Steel Painted Orange"},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_material_apply_task(
        {
            "input_usd_path": str(stage_path),
            "output_usd_path": str(output_path),
            "predictions_path": str(predictions_path),
            "resolved_materials": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "is_library_based_mapping": True,
            "material_library_path": str(library_path),
            "layer_only": True,
            "flatten_output": False,
            "skip_instance_check": False,
            "material_profile": "preview_surface",
            "allow_empty_predictions": False,
            "fail_on_unknown_material": True,
        }
    )

    assert output_path.exists()
    assert result["materials_applied"] == {
        "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
    }
    assert result["assignment_stats"]["materials_applied"] == 1
    assert result["assignment_stats"]["materials_created"] == 1

    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(output_path))
    assert stage is not None
    material, _relationship = UsdShade.MaterialBindingAPI(
        stage.GetPrimAtPath("/World/Step")
    ).ComputeBoundMaterial()
    assert str(material.GetPath()) == "/World/Looks/Steel_Painted_Orange"


def test_material_assignment_apply_requires_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
            "assignment_stats": {"total_prims": 1, "failed": 0},
        }

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    response = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path)},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert not output_path.exists()


def test_material_assignment_apply_preserves_existing_output_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usda"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("original output\n", encoding="utf-8")
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    captured_context: dict[str, Any] = {}

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        captured_context.update(context)
        raise RuntimeError("apply failed")

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    response = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path), "overwrite": True},
    )

    assert response.status_code == 500
    assert output_path.read_text(encoding="utf-8") == "original output\n"
    assert captured_context["output_usd_path"] != str(output_path)
    assert not Path(captured_context["output_usd_path"]).is_relative_to(
        output_path.parent
    )
    assert not Path(captured_context["output_usd_path"]).exists()


def test_material_assignment_apply_preserves_existing_usdz_on_packaging_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "materials.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usdz"
    output_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as package:
        package.writestr("original.usda", "#usda 1.0\n")
    original_bytes = output_path.read_bytes()
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)

    def fake_apply_task(context: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        Path(context["output_usd_path"]).write_text("#usda 1.0\n", encoding="utf-8")
        return {
            **context,
            "materials_applied": {
                "Steel Painted Orange": "/World/Looks/Steel_Painted_Orange"
            },
        }

    def fake_package_usdz(_source_usd_path: Path, usdz_path: Path) -> None:
        usdz_path.write_bytes(b"partial package")
        raise RuntimeError("package failed")

    monkeypatch.setattr(
        "content_workbench.sessions._run_material_apply_task",
        fake_apply_task,
    )
    monkeypatch.setattr(
        "content_workbench.sessions._package_material_apply_usdz",
        fake_package_usdz,
    )
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )
    assert override.status_code == 200

    response = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path), "overwrite": True},
    )

    assert response.status_code == 500
    assert output_path.read_bytes() == original_bytes
    assert zipfile.is_zipfile(output_path)
    assert not any(output_path.parent.glob(".ladder_materials.*.usdz"))
    assert not any(output_path.parent.glob(".ladder_materials.*.usdc"))


def test_material_apply_output_path_rejects_unsafe_targets(tmp_path: Path) -> None:
    source_path = tmp_path / "source.usda"
    inspection_path = tmp_path / "inspection.usda"
    existing_path = tmp_path / "outputs" / "existing.usda"
    _write_sample_stage(source_path)
    _write_sample_stage(inspection_path)
    existing_path.parent.mkdir()
    existing_path.write_text("#usda 1.0\n", encoding="utf-8")

    cases = [
        ("relative.usda", False, "must be an absolute local path"),
        (
            str(tmp_path / "bad.txt"),
            False,
            "must end with .usd, .usda, .usdc, or .usdz",
        ),
        (str(source_path), True, "must not overwrite the loaded scene"),
        (str(inspection_path), True, "must not overwrite the loaded scene"),
        (
            str(existing_path),
            False,
            "already exists; set overwrite=true to replace it",
        ),
    ]

    for raw_path, overwrite, expected_message in cases:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            _resolve_material_apply_output_path(
                raw_path,
                source_scene_path=source_path,
                inspection_scene_path=inspection_path,
                overwrite=overwrite,
            )


def test_material_assignment_apply_rejects_preview_only_material(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    output_path = tmp_path / "outputs" / "ladder_materials.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "display_name": "Debug Red",
                    "preview_color": [1.0, 0.0, 0.0],
                },
            },
        },
    )
    assert override.status_code == 200

    response = client.post(
        f"/sessions/{session_id}/authoring/material-assignments:apply",
        json={"output_usd_path": str(output_path)},
    )

    assert response.status_code == 400
    assert (
        "durable material apply requires a material-library assignment"
        in response.json()["detail"]
    )


def test_material_override_rejects_usd_special_material_library_path(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    stage_path = source_dir / "ladder_part.usda"
    library_path = source_dir / "materials@bad.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )

    assert override.status_code == 400
    assert "USD-special characters" in override.json()["detail"]


def test_material_override_rejects_usd_special_source_scene_path(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder@bad.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "name": "neutral_gray",
                    "preview_color": [0.7, 0.7, 0.7],
                },
            },
        },
    )

    assert override.status_code == 400
    assert (
        "Source scene path contains USD-special characters" in override.json()["detail"]
    )


def test_relative_material_library_resolves_from_source_before_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    cwd_dir = tmp_path / "cwd"
    source_dir.mkdir()
    cwd_dir.mkdir()
    stage_path = source_dir / "ladder_part.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(source_dir / "materials.usda")
    (cwd_dir / "materials.usda").write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.chdir(cwd_dir)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": "materials.usda",
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )

    assert override.status_code == 200
    preview_path = Path(override.json()["session"]["artifacts"]["preview_scene_path"])

    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    assert str(preview_stage.GetRootLayer().subLayerPaths[1]) == str(
        source_dir / "materials.usda"
    )
    material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/Step")
    ).ComputeBoundMaterial()
    assert str(material.GetPath()) == "/World/Looks/Steel_Painted_Orange"


def test_material_library_defaults_to_source_directory_allowlist(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    outside_dir = tmp_path / "outside"
    source_dir.mkdir()
    outside_dir.mkdir()
    stage_path = source_dir / "ladder_part.usda"
    library_path = outside_dir / "materials.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )

    assert override.status_code == 400
    assert (
        "outside CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS" in override.json()["detail"]
    )
    assert str(source_dir) not in override.json()["detail"]
    assert "<path>" in override.json()["detail"]


def test_missing_relative_material_library_reports_source_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "source"
    cwd_dir = tmp_path / "cwd"
    source_dir.mkdir()
    cwd_dir.mkdir()
    stage_path = source_dir / "ladder_part.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(cwd_dir / "missing.usda")
    monkeypatch.chdir(cwd_dir)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": "missing.usda",
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )

    assert override.status_code == 404
    detail = override.json()["detail"]
    assert str(source_dir / "missing.usda") not in detail
    assert str(cwd_dir / "missing.usda") not in detail
    assert "<path>" in detail


def test_material_override_rejects_symlinked_material_library(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    outside_dir = tmp_path / "outside"
    source_dir.mkdir()
    outside_dir.mkdir()
    stage_path = source_dir / "ladder_part.usda"
    real_library_path = outside_dir / "materials.usda"
    symlink_path = source_dir / "materials-link.usda"
    _write_sample_stage(stage_path)
    _write_material_library_stage(real_library_path)
    try:
        symlink_path.symlink_to(real_library_path)
    except OSError:
        pytest.skip("symlinks are not supported on this filesystem")
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(symlink_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )

    assert override.status_code == 400
    assert "must not be a symlink" in override.json()["detail"]


def test_material_override_rejects_material_library_outside_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    library_path = tmp_path / "outside" / "materials.usda"
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    library_path.parent.mkdir()
    _write_sample_stage(stage_path)
    _write_material_library_stage(library_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS", str(allowed_root))
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "library_path": str(library_path),
                    "material_name": "Steel Painted Orange",
                },
            },
        },
    )

    assert override.status_code == 400
    assert (
        "outside CONTENT_WORKBENCH_MATERIAL_LIBRARY_ROOTS" in override.json()["detail"]
    )


def test_material_override_does_not_route_generic_library_metadata(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "name": "neutral_gray",
                    "library": "metadata-only-tag",
                    "preview_color": [0.7, 0.7, 0.7],
                },
            },
        },
    )

    assert override.status_code == 200
    material_override = client.get(f"/sessions/{session_id}").json()["view"][
        "material_overrides"
    ][0]
    assert material_override["material"]["library"] == "metadata-only-tag"


def test_material_override_rejects_non_numeric_material_color(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {"preview_color": ["red", 0, 0]},
            },
        },
    )

    assert override.status_code == 400
    assert "material color values must be numeric" in override.json()["detail"]
    session_view = client.get(f"/sessions/{session_id}").json()["view"]
    assert session_view["material_overrides"] == []


def test_material_override_rejects_out_of_range_material_color(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {"preview_color": [1.2, 0.0, 0.0]},
            },
        },
    )

    assert override.status_code == 400
    assert "material color values must be between 0 and 1" in override.json()["detail"]
    session_view = client.get(f"/sessions/{session_id}").json()["view"]
    assert session_view["material_overrides"] == []


def test_material_override_rejects_short_material_color(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {"preview_color": [0.5, 0.5]},
            },
        },
    )

    assert override.status_code == 400
    assert (
        "material color values must contain exactly 3 values"
        in override.json()["detail"]
    )
    session_view = client.get(f"/sessions/{session_id}").json()["view"]
    assert session_view["material_overrides"] == []


def test_material_override_rejects_string_material(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": "Steel_Painted_Orange",
            },
        },
    )

    assert override.status_code == 400
    assert (
        override.json()["detail"]
        == "material_override payload.material must be an object"
    )
    session_view = client.get(f"/sessions/{session_id}").json()["view"]
    assert session_view["material_overrides"] == []


def test_material_override_rejects_relative_material_path(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {
                    "source": "material_library",
                    "material_path": "World/Looks/Steel",
                },
            },
        },
    )

    assert override.status_code == 400
    assert (
        override.json()["detail"]
        == "material_override payload.material.material_path must be an "
        "absolute USD prim path"
    )
    session_view = client.get(f"/sessions/{session_id}").json()["view"]
    assert session_view["material_overrides"] == []


def test_clear_material_override_rejects_invalid_space(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    response = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "clear_material_override",
            "payload": {
                "prim_path": "/World/Step",
                "space": "inpsection",
            },
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "clear_material_override space must be source or inspection"
    )


def test_clear_material_override_rejects_unknown_prim_path(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    response = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "clear_material_override",
            "payload": {"prim_path": "/World/Missing"},
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Prim not found: /World/Missing"


def test_more_specific_material_override_wins_regardless_of_command_order(
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    mesh_override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Step",
                "material": {"name": "MeshRed", "preview_color": [1.0, 0.0, 0.0]},
            },
        },
    )
    assert mesh_override.status_code == 200
    root_override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World",
                "material": {"name": "RootGray", "preview_color": [0.5, 0.5, 0.5]},
            },
        },
    )
    assert root_override.status_code == 200
    preview_path = Path(
        root_override.json()["session"]["artifacts"]["preview_scene_path"]
    )

    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/Step")
    ).ComputeBoundMaterial()
    assert str(material.GetPath()).startswith(
        "/World/PreviewMaterials/MeshRed_World_Step_"
    )


def test_same_named_material_overrides_do_not_collide(tmp_path: Path) -> None:
    stage_path = tmp_path / "isolation.usda"
    _write_isolation_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    first = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/KeepGroup/KeepMesh",
                "material": {"preview_color": [1.0, 0.0, 0.0]},
            },
        },
    )
    assert first.status_code == 200
    second = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Backdrop/BackdropMesh",
                "material": {"preview_color": [0.0, 1.0, 0.0]},
            },
        },
    )
    assert second.status_code == 200
    preview_path = Path(second.json()["session"]["artifacts"]["preview_scene_path"])

    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    first_material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/KeepGroup/KeepMesh")
    ).ComputeBoundMaterial()
    second_material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/Backdrop/BackdropMesh")
    ).ComputeBoundMaterial()

    assert str(first_material.GetPath()) != str(second_material.GetPath())


def test_instance_proxy_selection_and_material_override(tmp_path: Path) -> None:
    stage_path = tmp_path / "instanced_part.usda"
    _write_instanced_stage(stage_path)
    renderer = FakeRenderer()
    client = TestClient(create_app(SessionManager(renderer=renderer)))
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    tree = client.get(
        f"/sessions/{session_id}/tree", params={"prim_path": "/World/Part"}
    )
    assert tree.status_code == 200
    assert [child["path"] for child in tree.json()["children"]] == ["/World/Part/Mesh"]

    select = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "select", "payload": {"paths": ["/World/Part"]}},
    )
    assert select.status_code == 200
    response = client.post(
        f"/sessions/{session_id}/render",
        json={"width": 64, "height": 48, "ovrtx_num_sensor_updates": 1},
    )
    assert response.status_code == 200
    assert renderer.selected_prim_paths == ["/World/Part/Mesh"]

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Part/Mesh",
                "material": {"preview_color": [1.0, 0.0, 0.0]},
            },
        },
    )
    assert override.status_code == 200
    preview_path = Path(override.json()["session"]["artifacts"]["preview_scene_path"])

    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    part = preview_stage.GetPrimAtPath("/World/Part")
    mesh = preview_stage.GetPrimAtPath("/World/Part/Mesh")
    assert part.IsValid()
    assert not part.IsInstance()
    assert mesh.IsValid()
    assert not mesh.IsInstanceProxy()
    material, _rel = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    assert str(material.GetPath()).startswith(
        "/World/PreviewMaterials/PreviewMaterial_World_Part_Mesh_"
    )


def test_optimized_session_recovers_material_override_to_source_path(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.usda"
    optimized_path = tmp_path / "optimized.usda"
    _write_split_source_stage(source_path)
    _write_split_optimized_stage(optimized_path)
    manager = FakeOptimizingSessionManager(optimized_path=optimized_path)
    client = TestClient(create_app(manager))

    create = client.post(
        "/sessions",
        json={"scene_path": str(source_path), "optimize": True},
    )
    assert create.status_code == 201
    session = create.json()
    session_id = session["session_id"]
    assert session["optimization"]["enabled"] is True
    assert session["source_scene_path"] == str(source_path)
    assert session["inspection_scene_path"] == str(optimized_path)

    translate = client.post(
        f"/sessions/{session_id}/paths/translate",
        json={
            "prim_path": "/World/Panel_part_0",
            "source_space": "inspection",
            "target_space": "source",
        },
    )
    assert translate.status_code == 200
    assert translate.json()["source_paths"] == ["/World/Panel/FaceA"]

    translate_batch = client.post(
        f"/sessions/{session_id}/paths/translate:batch",
        json={
            "requests": [
                {
                    "prim_path": "/World/Panel_part_0",
                    "source_space": "inspection",
                    "target_space": "source",
                }
            ]
        },
    )
    assert translate_batch.status_code == 200
    assert translate_batch.json()["results"][0]["source_paths"] == [
        "/World/Panel/FaceA"
    ]

    invalid_translate = client.post(
        f"/sessions/{session_id}/paths/translate",
        json={
            "prim_path": "/World/Panel_part_0",
            "source_space": "bad",
            "target_space": "source",
        },
    )
    assert invalid_translate.status_code == 422

    empty_translate_batch = client.post(
        f"/sessions/{session_id}/paths/translate:batch",
        json={"requests": []},
    )
    assert empty_translate_batch.status_code == 422

    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/Panel_part_0",
                "space": "inspection",
                "material": {"preview_color": [1.0, 0.0, 0.0]},
            },
        },
    )
    assert override.status_code == 200
    override_body = override.json()
    stored = override_body["session"]["view"]["material_overrides"][0]
    assert stored["prim_path"] == "/World/Panel_part_0"
    assert stored["space"] == "inspection"
    assert stored["source_prim_paths"] == ["/World/Panel/FaceA"]
    assert stored["inspection_prim_paths"] == ["/World/Panel_part_0"]

    preview_path = Path(override_body["session"]["artifacts"]["preview_scene_path"])
    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    assert str(preview_stage.GetRootLayer().subLayerPaths[0]) == str(optimized_path)
    material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/Panel_part_0")
    ).ComputeBoundMaterial()
    assert str(material.GetPath()).startswith(
        "/World/PreviewMaterials/PreviewMaterial_World_Panel_part_0_"
    )


def test_optimized_dedup_material_override_keeps_runtime_identity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.usda"
    optimized_path = tmp_path / "optimized.usda"
    _write_dedup_source_stage(source_path)
    _write_dedup_optimized_stage(optimized_path)
    manager = FakeDedupOptimizingSessionManager(optimized_path=optimized_path)
    client = TestClient(create_app(manager))

    create = client.post(
        "/sessions",
        json={"scene_path": str(source_path), "optimize": True},
    )
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    runtime_path = "/World/BoltPrototype/Geometry"
    override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": runtime_path,
                "space": "inspection",
                "material": {"preview_color": [0.3, 0.3, 0.3]},
            },
        },
    )
    assert override.status_code == 200
    override_body = override.json()
    stored_overrides = override_body["session"]["view"]["material_overrides"]
    assert len(stored_overrides) == 1
    stored = stored_overrides[0]
    assert stored["prim_path"] == runtime_path
    assert stored["space"] == "inspection"
    assert stored["source_prim_paths"] == ["/World/BoltA", "/World/BoltB"]
    assert stored["inspection_prim_paths"] == [runtime_path]

    assignments = client.get(f"/sessions/{session_id}/authoring/material-assignments")
    assert assignments.status_code == 200
    assignment_records = assignments.json()["assignments"]
    assert len(assignment_records) == 1
    assignment = assignment_records[0]
    assert assignment["prim_path"] == runtime_path
    assert assignment["space"] == "inspection"
    assert assignment["source_prim_paths"] == ["/World/BoltA", "/World/BoltB"]
    assert assignment["inspection_prim_paths"] == [runtime_path]

    preview_path = Path(override_body["session"]["artifacts"]["preview_scene_path"])
    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    assert str(preview_stage.GetRootLayer().subLayerPaths[0]) == str(optimized_path)
    material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath(runtime_path)
    ).ComputeBoundMaterial()
    assert str(material.GetPath()).startswith(
        "/World/PreviewMaterials/PreviewMaterial_World_BoltPrototype_Geometry_"
    )

    clear = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "clear_material_override",
            "payload": {"prim_path": runtime_path, "space": "inspection"},
        },
    )
    assert clear.status_code == 200
    assert clear.json()["session"]["view"]["material_overrides"] == []


def test_material_override_with_extra_runtime_edge_preserves_sibling_coverage(
    tmp_path: Path,
) -> None:
    """Binding a prim's unique fragment must not erase a sibling's coverage.

    BoltA and BoltB dedup onto a shared runtime mesh (BoltPrototype/Geometry).
    BoltA additionally has its own unique runtime fragment (BoltA/Extra). A
    command binding BoltA's unique fragment shares BoltA as a resolved source
    prim with the earlier shared-alias override, but must only narrow that
    override's coverage (drop BoltA, keep BoltB) rather than delete it
    outright.
    """
    source_path = tmp_path / "source.usda"
    optimized_path = tmp_path / "optimized.usda"
    _write_dedup_source_stage(source_path)
    _write_mixed_alias_optimized_stage(optimized_path)
    manager = FakeMixedAliasOptimizingSessionManager(optimized_path=optimized_path)
    client = TestClient(create_app(manager))

    session_id = client.post(
        "/sessions",
        json={"scene_path": str(source_path), "optimize": True},
    ).json()["session_id"]

    shared_override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/BoltPrototype/Geometry",
                "space": "inspection",
                "material": {"preview_color": [0.3, 0.3, 0.3]},
            },
        },
    )
    assert shared_override.status_code == 200
    overrides = shared_override.json()["session"]["view"]["material_overrides"]
    assert len(overrides) == 1
    assert overrides[0]["source_prim_paths"] == ["/World/BoltA", "/World/BoltB"]

    unique_override = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "material_override",
            "payload": {
                "prim_path": "/World/BoltA/Extra",
                "space": "inspection",
                "material": {"preview_color": [1.0, 0.0, 0.0]},
            },
        },
    )
    assert unique_override.status_code == 200
    overrides = unique_override.json()["session"]["view"]["material_overrides"]
    assert len(overrides) == 2
    by_prim_path = {override["prim_path"]: override for override in overrides}
    shared = by_prim_path["/World/BoltPrototype/Geometry"]
    unique = by_prim_path["/World/BoltA/Extra"]
    assert shared["source_prim_paths"] == ["/World/BoltB"]
    assert shared["inspection_prim_paths"] == ["/World/BoltPrototype/Geometry"]
    assert unique["source_prim_paths"] == ["/World/BoltA"]
    assert unique["inspection_prim_paths"] == ["/World/BoltA/Extra"]

    preview_path = Path(
        unique_override.json()["session"]["artifacts"]["preview_scene_path"]
    )
    from pxr import Usd, UsdShade

    preview_stage = Usd.Stage.Open(str(preview_path))
    assert preview_stage is not None
    shared_material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/BoltPrototype/Geometry")
    ).ComputeBoundMaterial()
    unique_material, _rel = UsdShade.MaterialBindingAPI(
        preview_stage.GetPrimAtPath("/World/BoltA/Extra")
    ).ComputeBoundMaterial()
    assert str(shared_material.GetPath()) != str(unique_material.GetPath())


def test_optimizer_request_options_merge_into_shared_task_config(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.usda"
    optimized_path = tmp_path / "optimized.usda"
    _write_split_source_stage(source_path)
    _write_split_optimized_stage(optimized_path)
    manager = FakeOptimizingSessionManager(optimized_path=optimized_path)
    client = TestClient(create_app(manager))

    create = client.post(
        "/sessions",
        json={
            "scene_path": str(source_path),
            "optimize": True,
            "optimizer_backend": "remote",
            "flatten_prototypes": False,
            "enable_deinstance": True,
            "enable_split": False,
            "enable_deduplicate": True,
            "optimization_config": {
                "scene_optimizer_settings": {
                    "verbose": True,
                    "deduplicate": {"tolerance": 0.01},
                }
            },
        },
    )

    assert create.status_code == 201
    assert manager.last_optimization_config is not None
    assert manager.last_optimization_config["backend"] == "remote"
    assert manager.last_optimization_config["flatten_prototypes"] is False
    settings = manager.last_optimization_config["scene_optimizer_settings"]
    assert isinstance(settings, dict)
    assert settings["enable_deinstance"] is True
    assert settings["enable_split_meshes"] is False
    assert settings["enable_deduplicate"] is True
    assert settings["verbose"] is True
    assert settings["deduplicate"] == {"tolerance": 0.01}


def test_optimizer_request_rejects_no_enabled_operations(tmp_path: Path) -> None:
    source_path = tmp_path / "source.usda"
    _write_split_source_stage(source_path)
    client = _client()

    create = client.post(
        "/sessions",
        json={
            "scene_path": str(source_path),
            "optimize": True,
            "enable_deinstance": False,
            "enable_split": False,
            "enable_deduplicate": False,
        },
    )

    assert create.status_code == 422
    assert "At least one Scene Optimizer operation" in create.text


def test_optimizer_request_rejects_raw_config_with_no_enabled_operations(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.usda"
    _write_split_source_stage(source_path)
    client = _client()

    create = client.post(
        "/sessions",
        json={
            "scene_path": str(source_path),
            "optimize": True,
            "optimization_config": {
                "scene_optimizer_settings": {
                    "enable_deinstance": False,
                    "enable_split_meshes": False,
                    "enable_deduplicate": False,
                }
            },
        },
    )

    assert create.status_code == 422
    assert "At least one Scene Optimizer operation" in create.text


def test_optimizer_request_rejects_deep_optimization_config(tmp_path: Path) -> None:
    source_path = tmp_path / "source.usda"
    _write_split_source_stage(source_path)
    nested: dict[str, object] = {}
    cursor = nested
    for _index in range(MAX_OPTIMIZATION_CONFIG_DEPTH + 2):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child
    client = _client()

    create = client.post(
        "/sessions",
        json={
            "scene_path": str(source_path),
            "optimize": True,
            "optimization_config": {
                "scene_optimizer_settings": {
                    "enable_deinstance": True,
                    "enable_split_meshes": True,
                    "enable_deduplicate": True,
                    "nested": nested,
                }
            },
        },
    )

    assert create.status_code == 422
    assert "optimization_config exceeds maximum depth" in create.text


def test_render_endpoint_uses_owned_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setattr("content_workbench.sessions.time.time", lambda: 1000.0)
    renderer = FakeRenderer()
    client = TestClient(create_app(SessionManager(renderer=renderer)))
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    response = client.post(
        f"/sessions/{session_id}/render",
        json={"width": 64, "height": 48, "render_quality": "final"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["renderer"] == "ovrtx"
    assert body["render_product_path"] == "/Session/Render/Viewport"
    assert body["render_quality"] == "final"
    assert body["ovrtx_render_mode"] == "rt2"
    assert body["ovrtx_num_sensor_updates"] == 256
    assert body["active_aov"] == "LdrColor"
    assert renderer.render_mode == "rt2"
    assert renderer.num_updates == 256
    assert renderer.active_aov == "LdrColor"
    assert renderer.hdri_light == 600.0
    assert renderer.dome_light is None
    assert renderer.distant_light is None
    assert Path(body["preview_scene_path"]).exists()
    assert Path(body["image_path"]).exists()
    assert body["image_url"].startswith(f"/sessions/{session_id}/renders/render-")
    assert body["camera_json_url"].startswith(f"/sessions/{session_id}/renders/render-")

    image = client.get(body["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    camera_json = client.get(body["camera_json_url"])
    assert camera_json.status_code == 200
    assert camera_json.headers["content-type"] == "application/json"
    camera_json_payload = camera_json.json()
    assert camera_json_payload["camera_path"] == "/Session/Cameras/Main"
    assert camera_json_payload["hdri_light"] == 600.0
    assert camera_json_payload["dome_light"] is None
    assert camera_json_payload["distant_light"] is None
    assert camera_json_payload["active_aov"] == "LdrColor"

    escape = client.get(f"/sessions/{session_id}/renders/../secret.png")
    assert escape.status_code in {400, 404}

    select = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "select", "payload": {"paths": ["/World"]}},
    )
    assert select.status_code == 200
    change_aov = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "change_aov", "payload": {"aov": "Albedo"}},
    )
    assert change_aov.status_code == 200
    assert change_aov.json()["session"]["view"]["active_aov"] == "Albedo"
    response = client.post(
        f"/sessions/{session_id}/render",
        json={
            "width": 64,
            "height": 48,
            "render_quality": "final",
            "ovrtx_render_mode": "rt2",
            "ovrtx_num_sensor_updates": 1,
            "hdri_light": None,
            "dome_light": 350.0,
            "distant_light": 1200.0,
        },
    )
    assert response.status_code == 200
    assert response.json()["ovrtx_render_mode"] == "rt2"
    assert response.json()["ovrtx_num_sensor_updates"] == 1
    assert response.json()["active_aov"] == "Albedo"
    assert renderer.render_mode == "rt2"
    assert renderer.num_updates == 1
    assert renderer.active_aov == "Albedo"
    assert renderer.hdri_light is None
    assert renderer.dome_light == 350.0
    assert renderer.distant_light == 1200.0
    assert response.json()["image_url"] != body["image_url"]
    assert response.json()["camera_json_url"] != body["camera_json_url"]
    camera_json = client.get(response.json()["camera_json_url"]).json()
    assert camera_json["hdri_light"] is None
    assert camera_json["dome_light"] == 350.0
    assert camera_json["distant_light"] == 1200.0
    assert camera_json["active_aov"] == "Albedo"

    invalid_aov = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "change_aov", "payload": {"aov": "Bad\nAOV"}},
    )
    assert invalid_aov.status_code == 400
    assert "ASCII letters, digits, or underscores" in invalid_aov.text

    screenshot = client.get(f"/sessions/{session_id}/screenshot?width=32&height=24")
    assert screenshot.status_code == 200
    assert screenshot.headers["content-type"] == "image/png"


def test_render_frames_endpoint_uses_owned_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setattr("content_workbench.sessions.time.time", lambda: 1000.0)
    renderer = FakeRenderer()
    client = TestClient(create_app(SessionManager(renderer=renderer)))
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    response = client.post(
        f"/sessions/{session_id}/render-frames",
        json={
            "width": 64,
            "height": 48,
            "frames": "0:2",
            "directions": ["+x+z", "+y+z", "-x+z"],
            "render_quality": "final",
            "ovrtx_num_sensor_updates": 2,
            "save_camera_json": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["render_quality"] == "final"
    assert body["ovrtx_num_sensor_updates"] == 2
    assert body["active_aov"] == "LdrColor"
    assert len(body["frame_urls"]) == 3
    assert len(body["camera_json_urls"]) == 3
    assert renderer.frame_count == 3
    assert renderer.num_updates == 2
    assert renderer.active_aov == "LdrColor"
    assert renderer.hdri_light == 600.0
    assert renderer.dome_light is None
    assert renderer.distant_light is None

    image = client.get(body["frame_urls"][0])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    camera_json = client.get(body["camera_json_urls"][0])
    assert camera_json.status_code == 200
    assert camera_json.headers["content-type"] == "application/json"
    camera_json_payload = camera_json.json()
    assert camera_json_payload["frames"] == "0:2"
    assert camera_json_payload["direction"] == "+x+z"
    assert camera_json_payload["hdri_light"] == 600.0
    assert camera_json_payload["dome_light"] is None
    assert camera_json_payload["distant_light"] is None

    bad = client.post(
        f"/sessions/{session_id}/render-frames",
        json={"frames": "0:2", "directions": ["+x"]},
    )
    assert bad.status_code == 400
    assert "directions length" in bad.text

    external_stage_path = tmp_path / "recording.usda"
    _write_time_sampled_recording_stage(external_stage_path)
    external_without_camera = client.post(
        f"/sessions/{session_id}/render-frames",
        json={"scene_path": str(external_stage_path), "frames": "0:1"},
    )
    assert external_without_camera.status_code == 400
    assert "external scene frame renders require" in external_without_camera.text

    external_with_directions = client.post(
        f"/sessions/{session_id}/render-frames",
        json={
            "scene_path": str(external_stage_path),
            "frames": "0:1",
            "directions": ["+x+z", "-x+z"],
        },
    )
    assert external_with_directions.status_code == 400
    assert "external scene frame renders require" in external_with_directions.text

    fallback_camera_response = client.post(
        f"/sessions/{session_id}/render-frames",
        json={
            "width": 64,
            "height": 48,
            "frames": "0:1",
            "save_camera_json": True,
        },
    )
    assert fallback_camera_response.status_code == 200
    fallback_camera_body = fallback_camera_response.json()
    assert len(fallback_camera_body["frame_urls"]) == 2
    assert len(fallback_camera_body["camera_json_urls"]) == 2
    fallback_camera_json = client.get(fallback_camera_body["camera_json_urls"][0])
    assert fallback_camera_json.status_code == 200
    assert fallback_camera_json.json()["camera_world_transform"] is not None

    def fake_write_mp4(
        _frame_paths: list[Path],
        output_path: Path,
        _fps: float,
    ) -> bool:
        output_path.write_bytes(b"fake mp4")
        return True

    monkeypatch.setattr("content_workbench.sessions._write_mp4", fake_write_mp4)
    mp4_response = client.post(
        f"/sessions/{session_id}/render-frames",
        json={
            "width": 64,
            "height": 48,
            "frames": "0:1",
            "make_mp4": True,
        },
    )
    assert mp4_response.status_code == 200
    mp4_body = mp4_response.json()
    assert len(mp4_body["frame_urls"]) == 2
    assert len(mp4_body["mp4_urls"]) == 1
    assert renderer.frame_count == 2
    mp4 = client.get(mp4_body["mp4_urls"][0])
    assert mp4.status_code == 200
    assert mp4.headers["content-type"] == "video/mp4"


def test_render_failure_removes_camera_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingRenderer(FakeRenderer):
        def render(self, **_kwargs: object) -> float:
            raise RuntimeError("render failed")

    workspace_root = tmp_path / "workspace"
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    monkeypatch.setenv("CONTENT_WORKBENCH_WORKSPACE_DIR", str(workspace_root))
    monkeypatch.setattr("content_workbench.sessions.time.time", lambda: 1000.0)
    client = TestClient(create_app(SessionManager(renderer=FailingRenderer())))
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    response = client.post(
        f"/sessions/{session_id}/render",
        json={"width": 64, "height": 48, "save_camera_json": True},
    )

    assert response.status_code == 500
    assert not list((workspace_root / session_id / "renders").glob("*.json"))


def test_render_dimension_requests_have_upper_bound() -> None:
    client = _client()
    too_large = MAX_RENDER_DIMENSION + 1

    render = client.post(
        "/sessions/missing/render",
        json={"width": too_large, "height": 32},
    )
    assert render.status_code == 422

    pick = client.post(
        "/sessions/missing/pick",
        json={"x": 0, "y": 0, "width": too_large, "height": 32},
    )
    assert pick.status_code == 422

    screenshot = client.get(f"/sessions/missing/screenshot?width={too_large}&height=32")
    assert screenshot.status_code == 422


def test_agent_viewport_camera_commands_are_stateful(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post(
        "/sessions",
        json={"scene_path": str(stage_path), "width": 640, "height": 480},
    ).json()["session_id"]

    camera = client.get(f"/sessions/{session_id}/camera")
    assert camera.status_code == 200
    initial_camera = camera.json()
    assert initial_camera["last_framed_prim_path"] == "/World"

    orbit = client.post(
        f"/sessions/{session_id}/commands",
        json={
            "command": "orbit",
            "payload": {"yaw_delta_degrees": 30, "pitch_delta_degrees": -10},
        },
    )
    assert orbit.status_code == 200
    orbit_camera = orbit.json()["session"]["view"]["camera"]
    assert orbit_camera["yaw_degrees"] == initial_camera["yaw_degrees"] + 30
    assert orbit_camera["pitch_degrees"] == initial_camera["pitch_degrees"] - 10

    dolly = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "dolly", "payload": {"amount": -1}},
    )
    assert dolly.status_code == 200
    dolly_camera = dolly.json()["session"]["view"]["camera"]
    assert dolly_camera["distance"] == orbit_camera["distance"] * 0.5

    huge_dolly = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "dolly", "payload": {"amount": 1e6}},
    )
    assert huge_dolly.status_code == 200
    huge_dolly_camera = huge_dolly.json()["session"]["view"]["camera"]
    assert huge_dolly_camera["distance"] == MAX_CAMERA_DISTANCE

    huge_factor_dolly = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "dolly", "payload": {"factor": 1e308}},
    )
    assert huge_factor_dolly.status_code == 200
    huge_factor_camera = huge_factor_dolly.json()["session"]["view"]["camera"]
    assert huge_factor_camera["distance"] == MAX_CAMERA_DISTANCE

    bool_factor_dolly = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "dolly", "payload": {"factor": True}},
    )
    assert bool_factor_dolly.status_code == 400
    assert (
        "dolly factor must be a finite positive number"
        in bool_factor_dolly.json()["detail"]
    )

    pan = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "pan", "payload": {"right": 0.5, "up": -0.25}},
    )
    assert pan.status_code == 200
    pan_camera = pan.json()["session"]["view"]["camera"]
    assert pan_camera["target"] != dolly_camera["target"]

    bad_pan = client.post(
        f"/sessions/{session_id}/commands",
        content='{"command":"pan","payload":{"right":1e309}}',
        headers={"content-type": "application/json"},
    )
    assert bad_pan.status_code == 400
    assert "finite" in bad_pan.json()["detail"]

    frame = client.post(
        f"/sessions/{session_id}/commands",
        json={"command": "frame", "payload": {"prim_path": "/World/Step"}},
    )
    assert frame.status_code == 200
    frame_camera = frame.json()["session"]["view"]["camera"]
    assert frame_camera["last_framed_prim_path"] == "/World/Step"
    assert frame_camera["yaw_degrees"] == pan_camera["yaw_degrees"]

    replacement = {
        "target": [1.0, 2.0, 3.0],
        "distance": 4.0,
        "yaw_degrees": 12.0,
        "pitch_degrees": 8.0,
        "focal_length": 35.0,
        "horizontal_aperture": 36.0,
        "last_framed_prim_path": "/World/Step",
    }
    set_camera = client.post(f"/sessions/{session_id}/camera", json=replacement)
    assert set_camera.status_code == 200
    assert set_camera.json()["target"] == [1.0, 2.0, 3.0]

    render = client.post(
        f"/sessions/{session_id}/render",
        json={"width": 64, "height": 48, "ovrtx_num_sensor_updates": 1},
    )
    assert render.status_code == 200
    camera_json_path = Path(render.json()["camera_json_path"])
    camera_json = json.loads(camera_json_path.read_text(encoding="utf-8"))
    assert camera_json["use_session_camera"] is True
    assert camera_json["camera_state"]["target"] == [1.0, 2.0, 3.0]
    assert camera_json["camera_state"]["yaw_degrees"] == 12.0


def test_pick_endpoint_updates_selection(tmp_path: Path) -> None:
    stage_path = tmp_path / "ladder_part.usda"
    _write_sample_stage(stage_path)
    client = _client()
    session_id = client.post("/sessions", json={"scene_path": str(stage_path)}).json()[
        "session_id"
    ]

    pick = client.post(
        f"/sessions/{session_id}/pick",
        json={"x": 12, "y": 10, "width": 64, "height": 48},
    )
    assert pick.status_code == 200
    body = pick.json()
    assert body["prim_paths"] == ["/World/Step"]
    assert body["selected_prims"] == ["/World/Step"]

    session = client.get(f"/sessions/{session_id}")
    assert session.status_code == 200
    assert session.json()["view"]["selected_prims"] == ["/World/Step"]
