# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OvRTX renderer worker orchestration."""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from content_workbench import renderer_worker
from content_workbench.renderer_worker import (
    DEFAULT_HDRI_LIGHT_INTENSITY,
    FALLBACK_DISTANT_LIGHT_INTENSITY,
    FALLBACK_DOME_LIGHT_INTENSITY,
    IsolatedOvRTXRendererWorker,
    OvRTXRendererWorker,
    _activate_managed_ovrtx_runtime,
    _configure_ovrtx_env,
    _managed_ovrtx_spawn_context,
    _prepend_unique_env_path,
    _runtime_process_identity,
    _source_light_paths,
    _usd_path_literal,
    _viewer_stage_usda,
    _viewer_stage_usda_for_frames,
    validate_aov_name,
)


def test_open_viewer_stage_preserves_requested_render_mode(tmp_path: Path) -> None:
    worker = OvRTXRendererWorker()
    observed_modes: list[str] = []
    scene_path = tmp_path / "scene.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")

    class FakeRenderer:
        def open_usd_from_string(self, _stage: str) -> None:
            return None

    def fake_renderer_and_device(*, render_mode: str = "rt2"):
        observed_modes.append(render_mode)
        return FakeRenderer(), object()

    worker._renderer_and_device = fake_renderer_and_device  # type: ignore[method-assign]

    worker._open_viewer_stage(
        scene_path=scene_path,
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        hdri_light=DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light=None,
        distant_light=None,
        render_mode="pt",
    )

    assert observed_modes == ["pt"]


def test_viewer_stage_deactivates_source_lights_but_keeps_workbench_light(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdLux

    scene_path = tmp_path / "lit_scene.usda"
    scene_path.write_text(
        """#usda 1.0

def Scope "PreviewLights"
{
    def DomeLight "Dome"
    {
        float inputs:intensity = 1200
    }

    def DistantLight "Key"
    {
        float inputs:intensity = 3500
    }
}

def Mesh "Board"
{
}
""",
        encoding="utf-8",
    )

    source_light_paths = _source_light_paths(scene_path)
    assert source_light_paths == [
        "/PreviewLights/Dome",
        "/PreviewLights/Key",
    ]

    viewer_usda = _viewer_stage_usda(
        scene_path=scene_path,
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        hdri_light=None,
        dome_light=42.0,
        distant_light=None,
        source_light_paths=source_light_paths,
    )
    layer = Sdf.Layer.CreateAnonymous("viewer.usda")
    assert layer.ImportFromString(viewer_usda)
    stage = Usd.Stage.Open(layer)

    assert not stage.GetPrimAtPath("/PreviewLights/Dome").IsActive()
    assert not stage.GetPrimAtPath("/PreviewLights/Key").IsActive()
    workbench_light = stage.GetPrimAtPath("/Session/Lights/Fill")
    assert workbench_light.IsActive()
    assert workbench_light.HasAPI(UsdLux.LightAPI)
    assert stage.GetPrimAtPath("/Board").IsActive()


def test_source_light_paths_loads_payload_lights(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.usda"
    payload_path.write_text(
        """#usda 1.0
(
    defaultPrim = "PayloadRoot"
)

def Xform "PayloadRoot"
{
    def DomeLight "PayloadDome"
    {
        float inputs:intensity = 900
    }
}
""",
        encoding="utf-8",
    )
    scene_path = tmp_path / "payload_scene.usda"
    scene_path.write_text(
        """#usda 1.0

def Xform "PayloadRoot" (
    payload = @payload.usda@
)
{
}
""",
        encoding="utf-8",
    )

    assert _source_light_paths(scene_path) == ["/PayloadRoot/PayloadDome"]


def test_frame_viewer_stage_deactivates_source_lights(tmp_path: Path) -> None:
    usda = _viewer_stage_usda_for_frames(
        scene_path=tmp_path / "scene.usda",
        width=64,
        height=48,
        frame_numbers=[0],
        camera_transforms=[
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        camera_path=None,
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        fps=24.0,
        hdri_light=None,
        dome_light=42.0,
        distant_light=None,
        source_light_paths=["/PreviewLights/Dome"],
    )

    assert 'over "Dome" (\n        active = false' in usda
    assert 'def DomeLight "Fill"' in usda


def test_renderer_worker_defaults_to_hdri_without_plain_lights(
    tmp_path: Path,
) -> None:
    worker = OvRTXRendererWorker()
    observed: dict[str, object] = {}

    def fake_render_locked(**kwargs: object) -> float:
        observed.update(kwargs)
        return 0.125

    worker._render_locked = fake_render_locked  # type: ignore[method-assign]

    worker.render(
        scene_path=tmp_path / "scene.usda",
        output_path=tmp_path / "render.png",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    assert observed["hdri_light"] == DEFAULT_HDRI_LIGHT_INTENSITY
    assert observed["dome_light"] is None
    assert observed["distant_light"] is None


def test_isolated_renderer_routes_render_through_process_boundary(
    tmp_path: Path,
) -> None:
    worker = IsolatedOvRTXRendererWorker()
    observed: dict[str, object] = {}
    scene_path = tmp_path / "scene.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")

    def fake_run_isolated(operation: str, kwargs: dict[str, object]) -> float:
        observed["operation"] = operation
        observed["scene_path"] = kwargs["scene_path"]
        observed["source_light_paths"] = kwargs["source_light_paths"]
        return 0.25

    worker._run_isolated = fake_run_isolated  # type: ignore[method-assign]

    elapsed = worker.render(
        scene_path=scene_path,
        output_path=tmp_path / "render.png",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    assert elapsed == 0.25
    assert observed == {
        "operation": "render",
        "scene_path": scene_path,
        "source_light_paths": [],
    }


def test_isolated_renderer_terminates_child_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def is_alive(self) -> bool:
            return not self.terminated and not self.killed

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def join(self, timeout: float | None = None) -> None:
            del timeout

    class FakeFuture:
        def result(self, timeout: float | None = None) -> object:
            del timeout
            raise renderer_worker.FutureTimeoutError()

    class FakeExecutor:
        process = FakeProcess()
        shutdown_calls: list[dict[str, object]] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self._processes = {123: self.process}

        def submit(self, *args: object, **kwargs: object) -> FakeFuture:
            del args, kwargs
            return FakeFuture()

        def shutdown(
            self,
            *,
            wait: bool = True,
            cancel_futures: bool = False,
        ) -> None:
            self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    monkeypatch.setattr(renderer_worker, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        renderer_worker,
        "_managed_ovrtx_runtime",
        lambda: (sys.executable, str(tmp_path)),
    )
    worker = IsolatedOvRTXRendererWorker(operation_timeout_seconds=0.01)

    with pytest.raises(TimeoutError, match="OvRTX isolated render exceeded"):
        worker._run_isolated("render", {})

    assert FakeExecutor.process.terminated
    assert FakeExecutor.shutdown_calls[-1] == {
        "wait": False,
        "cancel_futures": True,
    }


def test_isolated_renderer_retries_crashed_child_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFuture:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        def result(self, timeout: float | None = None) -> object:
            del timeout
            if self.attempt == 1:
                raise renderer_worker.BrokenProcessPool("native renderer crashed")
            return 0.25

    class FakeExecutor:
        attempts = 0
        shutdown_count = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def submit(self, *args: object, **kwargs: object) -> FakeFuture:
            del args, kwargs
            type(self).attempts += 1
            return FakeFuture(type(self).attempts)

        def shutdown(
            self,
            *,
            wait: bool = True,
            cancel_futures: bool = False,
        ) -> None:
            del wait, cancel_futures
            type(self).shutdown_count += 1

    monkeypatch.setattr(renderer_worker, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        renderer_worker,
        "_managed_ovrtx_runtime",
        lambda: (sys.executable, str(tmp_path)),
    )
    worker = IsolatedOvRTXRendererWorker(operation_timeout_seconds=1.0)

    assert worker._run_isolated("render", {}) == 0.25
    assert FakeExecutor.attempts == 2
    assert FakeExecutor.shutdown_count == 2


def test_render_releases_renderer_stage_after_success(tmp_path: Path) -> None:
    worker = OvRTXRendererWorker()
    reset_count = 0

    class FakeRenderer:
        def reset_stage(self) -> None:
            nonlocal reset_count
            reset_count += 1

    worker._renderer = FakeRenderer()
    worker._render_locked = lambda **_kwargs: 0.125  # type: ignore[method-assign]

    worker.render(
        scene_path=tmp_path / "scene.usda",
        output_path=tmp_path / "render.png",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )

    assert reset_count == 1


def test_render_releases_renderer_stage_after_failure(tmp_path: Path) -> None:
    worker = OvRTXRendererWorker()
    reset_count = 0

    class FakeRenderer:
        def reset_stage(self) -> None:
            nonlocal reset_count
            reset_count += 1

    def fail_render(**_kwargs: object) -> float:
        raise RuntimeError("render failed")

    worker._renderer = FakeRenderer()
    worker._render_locked = fail_render  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="render failed"):
        worker.render(
            scene_path=tmp_path / "scene.usda",
            output_path=tmp_path / "render.png",
            width=64,
            height=48,
            camera_transform=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

    assert reset_count == 1


def test_shutdown_resets_stage_when_renderer_has_no_shutdown() -> None:
    worker = OvRTXRendererWorker()
    reset_count = 0

    class FakeRenderer:
        def reset_stage(self) -> None:
            nonlocal reset_count
            reset_count += 1

    worker._renderer = FakeRenderer()
    worker._device = object()
    worker._render_mode = "rt2"

    worker.shutdown()

    assert reset_count == 1
    assert worker._renderer is None
    assert worker._device is None
    assert worker._render_mode is None


def test_viewer_stage_honors_explicit_hdri_light(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

    usda = _viewer_stage_usda(
        scene_path=tmp_path / "scene.usda",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        hdri_light=DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light=None,
        distant_light=None,
    )

    assert 'def "OvRTXDefaultLights"' in usda
    assert 'def DomeLight "DomeLight"' in usda
    assert "studio.exr" in usda
    assert "float inputs:intensity = 600.0" in usda
    assert 'token inputs:texture:format = "latlong"' in usda
    assert "asset inputs:texture:file" in usda
    assert "custom bool visibleInPrimaryRay = 0" in usda
    assert 'def DomeLight "Fill"' not in usda
    assert 'def DistantLight "Key"' not in usda


def test_viewer_stage_distinguishes_hdri_from_plain_dome_light(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", raising=False)

    usda = _viewer_stage_usda(
        scene_path=tmp_path / "scene.usda",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        hdri_light=None,
        dome_light=42.0,
        distant_light=None,
    )

    assert 'def "OvRTXDefaultLights"' not in usda
    assert 'def DomeLight "DomeLight"' not in usda
    assert 'def DomeLight "Fill"' in usda
    assert "float inputs:intensity = 42" in usda
    assert "asset inputs:texture:file" not in usda


def test_viewer_stage_honors_explicit_default_hdri_intensity_over_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WU_OVRTX_DEFAULT_HDRI", raising=False)
    monkeypatch.setenv("WU_OVRTX_DEFAULT_HDRI_INTENSITY", "12.5")

    usda = _viewer_stage_usda(
        scene_path=tmp_path / "scene.usda",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        hdri_light=DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light=None,
        distant_light=None,
    )

    assert "float inputs:intensity = 600.0" in usda
    assert "float inputs:intensity = 12.5" not in usda


def test_viewer_stage_falls_back_to_synthetic_lights_when_hdri_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_default_hdri(_intensity: float) -> str:
        raise RuntimeError("missing studio.exr")

    monkeypatch.setattr(renderer_worker, "_default_hdri_lights_usda", fail_default_hdri)
    caplog.set_level(logging.WARNING)

    usda = _viewer_stage_usda(
        scene_path=tmp_path / "scene.usda",
        width=64,
        height=48,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        camera_focal_length=50.0,
        camera_horizontal_aperture=36.0,
        hdri_light=DEFAULT_HDRI_LIGHT_INTENSITY,
        dome_light=None,
        distant_light=None,
    )

    assert 'def "OvRTXDefaultLights"' not in usda
    assert 'def DomeLight "Fill"' in usda
    assert f"float inputs:intensity = {FALLBACK_DOME_LIGHT_INTENSITY:.6g}" in usda
    assert 'def DistantLight "Key"' in usda
    assert f"float inputs:intensity = {FALLBACK_DISTANT_LIGHT_INTENSITY:.6g}" in usda
    assert "Falling back to synthetic Workbench lights" in caplog.text


def test_usd_path_literal_rejects_control_characters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="control characters"):
        _usd_path_literal(tmp_path / "bad\npath.usda")


def test_validate_aov_name_rejects_usd_control_syntax() -> None:
    with pytest.raises(ValueError, match="ASCII letters"):
        validate_aov_name('LdrColor"\n)')


def test_prepend_ovrtx_path_warns_about_existing_ovrtx_library_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    new_path = tmp_path / "ovrtx-new" / "plugins"
    old_path = tmp_path / "ovrtx-old" / "plugins"
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        os.pathsep.join([str(old_path), "/usr/lib"]),
    )

    with caplog.at_level(logging.WARNING, logger="content_workbench.renderer_worker"):
        _prepend_unique_env_path("LD_LIBRARY_PATH", new_path)

    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(new_path)
    assert "existing OvRTX-looking LD_LIBRARY_PATH entries" in caplog.text


def test_configure_ovrtx_env_memoizes_process_path_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    activation_calls: list[str] = []
    ovrtx_module = tmp_path / "ovrtx" / "__init__.py"
    ovrtx_module.parent.mkdir()
    ovrtx_module.write_text("", encoding="utf-8")

    def fake_find_spec(name: str) -> SimpleNamespace | None:
        calls.append(name)
        return SimpleNamespace(origin=str(ovrtx_module))

    monkeypatch.setattr(renderer_worker, "_OVRTX_ENV_CONFIGURED", False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(
        renderer_worker,
        "_activate_managed_ovrtx_runtime",
        lambda: activation_calls.append("activate"),
    )
    monkeypatch.setattr(renderer_worker.importlib.util, "find_spec", fake_find_spec)

    _configure_ovrtx_env(render_mode="rt2")
    _configure_ovrtx_env(render_mode="pt")

    plugin_path = str(ovrtx_module.parent.resolve() / "bin" / "plugins")
    assert activation_calls == ["activate"]
    assert calls == ["ovrtx"]
    assert os.environ["OVRTX_RENDER_MODE"] == "pt"
    assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep).count(plugin_path) == 1


def test_activate_managed_ovrtx_runtime_uses_locked_runtime_site_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from world_understanding.functions.graphics import render_ovrtx

    runtime_dir = tmp_path / "ovrtx-runtime"
    site_dir = runtime_dir / "lib" / "python3.12" / "site-packages"
    site_dir.mkdir(parents=True)
    python_path = runtime_dir / "bin" / "python"
    cache_invalidations: list[str] = []
    original_sys_path = list(renderer_worker.sys.path)

    monkeypatch.setattr(
        render_ovrtx,
        "_get_ovrtx_python",
        lambda: str(python_path),
    )
    monkeypatch.setattr(
        render_ovrtx,
        "_ovrtx_site_dir_env_for_python",
        lambda active_python: (
            str(site_dir) if active_python == str(python_path) else None
        ),
    )
    monkeypatch.setattr(
        renderer_worker.importlib,
        "invalidate_caches",
        lambda: cache_invalidations.append("invalidate"),
    )

    try:
        _activate_managed_ovrtx_runtime()
        _activate_managed_ovrtx_runtime()
        assert renderer_worker.sys.path[0] == str(site_dir.resolve())
        assert renderer_worker.sys.path.count(str(site_dir.resolve())) == 1
        assert cache_invalidations == ["invalidate", "invalidate"]
    finally:
        renderer_worker.sys.path[:] = original_sys_path


def test_managed_ovrtx_spawn_uses_one_locked_dependency_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_dir = tmp_path / "ovrtx-runtime"
    runtime_python = runtime_dir / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.symlink_to(Path(sys.executable).resolve())
    site_dir = runtime_dir / "lib" / "python3.12" / "site-packages"
    site_dir.mkdir(parents=True)

    packages = {
        "numpy": ("2.2.6", "numpy"),
        "pillow": ("12.3.0", "PIL"),
        "ovrtx": ("0.3.0.312915", "ovrtx"),
    }
    for distribution, (version, module_name) in packages.items():
        module_dir = site_dir / module_name
        module_dir.mkdir()
        (module_dir / "__init__.py").write_text("", encoding="utf-8")
        if module_name == "PIL":
            (module_dir / "Image.py").write_text("", encoding="utf-8")
            (module_dir / "__init__.py").write_text(
                "from . import Image\n", encoding="utf-8"
            )
        metadata_dir = site_dir / f"{distribution}-{version}.dist-info"
        metadata_dir.mkdir()
        (metadata_dir / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        renderer_worker,
        "_managed_ovrtx_runtime",
        lambda: (str(runtime_python), str(site_dir)),
    )

    with _managed_ovrtx_spawn_context() as context:
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            identity = executor.submit(_runtime_process_identity).result(timeout=30)

    assert Path(identity["python"]) == runtime_python
    assert identity["ovrtx_version"] == "0.3.0.312915"
    assert identity["numpy_version"] == "2.2.6"
    assert identity["pillow_version"] == "12.3.0"
    assert Path(identity["ovrtx_origin"]).is_relative_to(site_dir)
    assert Path(identity["numpy_origin"]).is_relative_to(site_dir)
    assert Path(identity["pillow_origin"]).is_relative_to(site_dir)


def test_pick_closes_intermediate_context_managed_products(
    tmp_path: Path,
) -> None:
    worker = OvRTXRendererWorker()
    entered: list[int] = []
    exited: list[int] = []

    class FakeProducts:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> dict[str, object]:
            entered.append(self.index)
            return {}

        def __exit__(self, *_args: object) -> None:
            exited.append(self.index)

    class FakeRenderer:
        def enqueue_pick_query(self, **_kwargs: object) -> None:
            return None

    step_products = [
        FakeProducts(0),
        FakeProducts(1),
        {"/Session/Render/Viewport": SimpleNamespace(frames=[])},
    ]

    def fake_renderer_and_device(
        *, render_mode: str = "rt2"
    ) -> tuple[FakeRenderer, object]:
        return FakeRenderer(), object()

    def fake_open_viewer_stage(**_kwargs: object) -> None:
        return None

    def fake_write_selection_groups(_paths: list[str]) -> None:
        return None

    def fake_step_renderer() -> object:
        return step_products.pop(0)

    worker._renderer_and_device = (  # type: ignore[method-assign]
        fake_renderer_and_device
    )
    worker._open_viewer_stage = fake_open_viewer_stage  # type: ignore[method-assign]
    worker._write_selection_groups = fake_write_selection_groups  # type: ignore[method-assign]
    worker._step_renderer = fake_step_renderer  # type: ignore[method-assign]

    result = worker.pick(
        scene_path=tmp_path / "scene.usda",
        x=1,
        y=1,
        width=4,
        height=4,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        num_updates=3,
        source_light_paths=[],
    )

    assert result.prim_paths == []
    assert entered == [0, 1]
    assert exited == [0, 1]


def test_pick_releases_renderer_stage(tmp_path: Path) -> None:
    worker = OvRTXRendererWorker()
    reset_count = 0

    class FakeRenderer:
        def enqueue_pick_query(self, **_kwargs: object) -> None:
            return None

        def reset_stage(self) -> None:
            nonlocal reset_count
            reset_count += 1

    fake_renderer = FakeRenderer()
    worker._renderer = fake_renderer
    worker._renderer_and_device = (  # type: ignore[method-assign]
        lambda **_kwargs: (fake_renderer, object())
    )
    worker._open_viewer_stage = lambda **_kwargs: None  # type: ignore[method-assign]
    worker._write_selection_groups = lambda _paths: None  # type: ignore[method-assign]
    worker._step_renderer = lambda: {  # type: ignore[method-assign]
        renderer_worker.RENDER_PRODUCT_PATH: SimpleNamespace(frames=[])
    }

    worker.pick(
        scene_path=tmp_path / "scene.usda",
        x=1,
        y=1,
        width=4,
        height=4,
        camera_transform=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        source_light_paths=[],
    )

    assert reset_count == 1


def test_render_frames_releases_renderer_stage(tmp_path: Path) -> None:
    worker = OvRTXRendererWorker()
    reset_count = 0

    class FakeRenderer:
        def reset_stage(self) -> None:
            nonlocal reset_count
            reset_count += 1

    worker._renderer = FakeRenderer()
    worker._render_frames_locked = lambda **_kwargs: None  # type: ignore[method-assign]

    worker.render_frames(
        scene_path=tmp_path / "scene.usda",
        output_paths=[tmp_path / "frame.png"],
        width=4,
        height=4,
        frame_numbers=[0],
        camera_transforms=[
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    )

    assert reset_count == 1


def test_render_restarts_renderer_once_when_active_aov_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = OvRTXRendererWorker()
    created_renderers: list[str] = []
    shutdown_renderers: list[str] = []
    output_path = tmp_path / "render.png"
    scene_path = tmp_path / "scene.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    rgba = renderer_worker.np.zeros((2, 2, 4), dtype=renderer_worker.np.uint8)

    class FakeDevice:
        CPU = object()

    class FakeMappedRenderVar:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeRenderVar:
        def map(self, *, device: object) -> FakeMappedRenderVar:
            assert device is FakeDevice.CPU
            return FakeMappedRenderVar()

    class FakeRenderer:
        def __init__(self, name: str, *, include_aov: bool) -> None:
            self.name = name
            self.include_aov = include_aov

        def open_usd_from_string(self, _stage: str) -> None:
            return None

        def step(self, **_kwargs: object) -> dict[str, object]:
            render_vars = {"LdrColor": FakeRenderVar()} if self.include_aov else {}
            return {
                renderer_worker.RENDER_PRODUCT_PATH: SimpleNamespace(
                    frames=[SimpleNamespace(render_vars=render_vars)]
                )
            }

        def shutdown(self) -> None:
            shutdown_renderers.append(self.name)

    renderers = [
        FakeRenderer("missing-aov", include_aov=False),
        FakeRenderer("replacement", include_aov=True),
    ]
    monotonic_values = iter([0.0, 1.0, 2.0, 8.0, 8.0])

    def fake_renderer_and_device(
        *, render_mode: str = "rt2"
    ) -> tuple[FakeRenderer, type[FakeDevice]]:
        if worker._renderer is None:
            renderer = renderers.pop(0)
            created_renderers.append(renderer.name)
            worker._renderer = renderer
            worker._device = FakeDevice
            worker._render_mode = render_mode
        return worker._renderer, FakeDevice

    worker._renderer_and_device = (  # type: ignore[method-assign]
        fake_renderer_and_device
    )
    monkeypatch.setattr(renderer_worker.np, "from_dlpack", lambda _value: rgba)
    monkeypatch.setattr(
        renderer_worker.time, "monotonic", lambda: next(monotonic_values)
    )

    with caplog.at_level(logging.WARNING, logger="content_workbench.renderer_worker"):
        elapsed = worker.render(
            scene_path=scene_path,
            output_path=output_path,
            width=2,
            height=2,
            camera_transform=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            num_updates=1,
        )

    assert elapsed == 8.0
    assert output_path.exists()
    assert created_renderers == ["missing-aov", "replacement"]
    assert shutdown_renderers == ["missing-aov"]
    assert "restarting renderer and retrying once" in caplog.text


def test_render_retry_propagates_when_active_aov_remains_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = OvRTXRendererWorker()
    scene_path = tmp_path / "scene.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    created_renderers: list[str] = []
    shutdown_renderers: list[str] = []

    class FakeDevice:
        CPU = object()

    class FakeRenderer:
        def __init__(self, name: str) -> None:
            self.name = name

        def open_usd_from_string(self, _stage: str) -> None:
            return None

        def step(self, **_kwargs: object) -> dict[str, object]:
            return {
                renderer_worker.RENDER_PRODUCT_PATH: SimpleNamespace(
                    frames=[SimpleNamespace(render_vars={})]
                )
            }

        def shutdown(self) -> None:
            shutdown_renderers.append(self.name)

    renderers = [
        FakeRenderer("missing-aov"),
        FakeRenderer("still-missing-aov"),
    ]

    def fake_renderer_and_device(
        *, render_mode: str = "rt2"
    ) -> tuple[FakeRenderer, type[FakeDevice]]:
        if worker._renderer is None:
            renderer = renderers.pop(0)
            created_renderers.append(renderer.name)
            worker._renderer = renderer
            worker._device = FakeDevice
            worker._render_mode = render_mode
        return worker._renderer, FakeDevice

    worker._renderer_and_device = (  # type: ignore[method-assign]
        fake_renderer_and_device
    )

    with (
        caplog.at_level(logging.WARNING, logger="content_workbench.renderer_worker"),
        pytest.raises(RuntimeError, match="OvRTX did not return LdrColor"),
    ):
        worker.render(
            scene_path=scene_path,
            output_path=tmp_path / "render.png",
            width=2,
            height=2,
            camera_transform=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            num_updates=1,
        )

    assert created_renderers == ["missing-aov", "still-missing-aov"]
    assert shutdown_renderers == ["missing-aov"]
    assert "restarting renderer and retrying once" in caplog.text


def test_render_frames_retry_removes_partial_outputs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    worker = OvRTXRendererWorker()
    calls = 0
    shutdowns = 0
    output_paths = [tmp_path / "frames" / "frame_000.png"]

    def fake_render_frames_locked(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        output_paths[0].parent.mkdir(parents=True, exist_ok=True)
        output_paths[0].write_bytes(b"partial")
        raise renderer_worker._MissingActiveAOVError("OvRTX did not return LdrColor")

    def fake_shutdown_renderer_locked() -> None:
        nonlocal shutdowns
        shutdowns += 1

    worker._render_frames_locked = fake_render_frames_locked  # type: ignore[method-assign]
    worker._shutdown_renderer_locked = (  # type: ignore[method-assign]
        fake_shutdown_renderer_locked
    )

    with (
        caplog.at_level(logging.WARNING, logger="content_workbench.renderer_worker"),
        pytest.raises(RuntimeError, match="OvRTX did not return LdrColor"),
    ):
        worker.render_frames(
            scene_path=tmp_path / "scene.usda",
            output_paths=output_paths,
            width=2,
            height=2,
            frame_numbers=[0],
            camera_transforms=[
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            num_updates=1,
        )

    assert calls == 2
    assert shutdowns == 1
    assert not output_paths[0].exists()
    assert "render_frames returned no LdrColor" in caplog.text


def test_render_frames_removes_partial_outputs_on_first_attempt_failure(
    tmp_path: Path,
) -> None:
    worker = OvRTXRendererWorker()
    output_paths = [tmp_path / "frames" / "frame_000.png"]

    def fake_render_frames_locked(**_kwargs: object) -> None:
        output_paths[0].parent.mkdir(parents=True, exist_ok=True)
        output_paths[0].write_bytes(b"partial")
        raise RuntimeError("first attempt failed after partial output")

    worker._render_frames_locked = fake_render_frames_locked  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="first attempt failed"):
        worker.render_frames(
            scene_path=tmp_path / "scene.usda",
            output_paths=output_paths,
            width=2,
            height=2,
            frame_numbers=[0],
            camera_transforms=[
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            num_updates=1,
        )

    assert not output_paths[0].exists()


def test_render_frames_retry_removes_partial_outputs_on_non_aov_failure(
    tmp_path: Path,
) -> None:
    worker = OvRTXRendererWorker()
    calls = 0
    output_paths = [tmp_path / "frames" / "frame_000.png"]

    def fake_render_frames_locked(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        output_paths[0].parent.mkdir(parents=True, exist_ok=True)
        output_paths[0].write_bytes(b"partial")
        if calls == 1:
            raise renderer_worker._MissingActiveAOVError(
                "OvRTX did not return LdrColor"
            )
        raise RuntimeError("retry failed after partial output")

    worker._render_frames_locked = fake_render_frames_locked  # type: ignore[method-assign]
    worker._shutdown_renderer_locked = lambda: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="retry failed"):
        worker.render_frames(
            scene_path=tmp_path / "scene.usda",
            output_paths=output_paths,
            width=2,
            height=2,
            frame_numbers=[0],
            camera_transforms=[
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            num_updates=1,
        )

    assert calls == 2
    assert not output_paths[0].exists()
