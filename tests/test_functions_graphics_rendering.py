# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for graphics rendering backend configuration."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from PIL import Image
from pxr import Gf, Usd, UsdGeom, Vt

from world_understanding.functions.graphics import render_remote_async, rendering
from world_understanding.functions.graphics.rendering import (
    NVCFRenderingBackend,
    OvRTXRenderingBackend,
    RemoteRenderingBackend,
)
from world_understanding.rendering_backend_contract import (
    RemoteRenderingSlotTimeoutError,
)


class TestRemoteRenderingBackendConfig:
    """Tests for remote REST renderer backend configuration precedence."""

    def test_reads_s3_env_at_instantiation(self, monkeypatch):
        monkeypatch.setenv("WU_S3_BUCKET", "runtime-bucket")
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        monkeypatch.setenv("WU_S3_PROFILE", "runtime-profile")
        monkeypatch.delenv("MA_RENDERING_USE_DATA_URI", raising=False)

        backend = RemoteRenderingBackend()

        assert backend.s3_bucket == "runtime-bucket"
        assert backend.s3_region == "eu-west-1"
        assert backend.s3_profile == "runtime-profile"
        assert backend.use_data_uri is True

    def test_explicit_false_uses_s3_transfer(self, monkeypatch):
        monkeypatch.setenv("MA_RENDERING_USE_DATA_URI", "true")

        backend = RemoteRenderingBackend(use_data_uri=False)

        assert backend.use_data_uri is False

    def test_legacy_nvcf_backend_aliases_remote_backend(self):
        assert NVCFRenderingBackend is RemoteRenderingBackend

    def test_explicit_s3_kwargs_override_runtime_env(self, monkeypatch):
        monkeypatch.setenv("WU_S3_BUCKET", "runtime-bucket")
        monkeypatch.setenv("WU_S3_REGION", "eu-west-1")
        monkeypatch.setenv("WU_S3_PROFILE", "runtime-profile")

        backend = RemoteRenderingBackend(
            s3_bucket="explicit-bucket",
            s3_region="us-west-2",
            s3_profile="explicit-profile",
        )

        assert backend.s3_bucket == "explicit-bucket"
        assert backend.s3_region == "us-west-2"
        assert backend.s3_profile == "explicit-profile"

    def test_falls_back_to_module_constants_when_no_env_or_kwargs(self, monkeypatch):
        monkeypatch.delenv("WU_S3_BUCKET", raising=False)
        monkeypatch.delenv("WU_S3_REGION", raising=False)
        monkeypatch.delenv("WU_S3_PROFILE", raising=False)
        monkeypatch.setattr(rendering, "WU_S3_BUCKET", "module-bucket")
        monkeypatch.setattr(rendering, "WU_S3_REGION", "ap-south-1")
        monkeypatch.setattr(rendering, "WU_S3_PROFILE", "module-profile")

        backend = RemoteRenderingBackend()

        assert backend.s3_bucket == "module-bucket"
        assert backend.s3_region == "ap-south-1"
        assert backend.s3_profile == "module-profile"

    def test_sync_render_passes_base_dir_to_remote_renderer(
        self,
        monkeypatch,
        tmp_path,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["base_dir"] = kwargs.get("base_dir")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(api_key="test")
        result = backend.render(object(), cameras=["/Camera"], base_dir=tmp_path)

        assert captured["base_dir"] == tmp_path
        assert result["successful_cameras"] == 1

    def test_sync_render_passes_preview_fallback_flag_to_remote_renderer(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["add_preview_fallbacks"] = kwargs.get("add_preview_fallbacks")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(
            api_key="test",
            add_preview_fallbacks=False,
        )
        result = backend.render(object(), cameras=["/Camera"])

        assert captured["add_preview_fallbacks"] is False
        assert result["successful_cameras"] == 1

    def test_sync_render_auto_preserves_materials_by_default(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["add_preview_fallbacks"] = kwargs.get("add_preview_fallbacks")
            captured["material_target"] = kwargs.get("material_target")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(api_key="test")
        result = backend.render(object(), cameras=["/Camera"])

        assert captured == {
            "add_preview_fallbacks": False,
            "material_target": "auto",
        }
        assert result["successful_cameras"] == 1

    def test_sync_render_derives_preview_fallback_flag_from_material_target(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["add_preview_fallbacks"] = kwargs.get("add_preview_fallbacks")
            captured["material_target"] = kwargs.get("material_target")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(
            api_key="test",
            material_target="openpbr_materialx",
        )
        result = backend.render(object(), cameras=["/Camera"])

        assert captured == {
            "add_preview_fallbacks": False,
            "material_target": "openpbr_materialx",
        }
        assert result["successful_cameras"] == 1

    def test_sync_render_preview_surface_target_requests_preview_fallback(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["add_preview_fallbacks"] = kwargs.get("add_preview_fallbacks")
            captured["material_target"] = kwargs.get("material_target")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(
            api_key="test",
            material_target="preview_surface",
        )
        result = backend.render(object(), cameras=["/Camera"])

        assert captured == {
            "add_preview_fallbacks": True,
            "material_target": "preview_surface",
        }
        assert result["successful_cameras"] == 1

    def test_sync_render_legacy_preview_fallback_true_requests_preview_fallback(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured["add_preview_fallbacks"] = kwargs.get("add_preview_fallbacks")
            captured["material_target"] = kwargs.get("material_target")
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(
            api_key="test",
            add_preview_fallbacks=True,
        )
        result = backend.render(object(), cameras=["/Camera"])

        assert captured == {
            "add_preview_fallbacks": True,
            "material_target": "auto",
        }
        assert result["successful_cameras"] == 1

    def test_sync_render_uses_global_request_limit(self, monkeypatch):
        active_requests = 0
        max_active_requests = 0
        calls = 0
        counters_lock = threading.Lock()

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            nonlocal active_requests, max_active_requests, calls
            with counters_lock:
                calls += 1
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            time.sleep(0.01)
            with counters_lock:
                active_requests -= 1
            return {
                "successful_cameras": 1,
                "results": [{"images": [], "status": "success"}],
            }

        monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
        render_remote_async._reset_global_remote_render_semaphore_for_tests()
        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(api_key="test")
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(backend.render, object(), cameras=["/Camera"])
                    for _ in range(2)
                ]
                results = [future.result() for future in futures]
        finally:
            render_remote_async._reset_global_remote_render_semaphore_for_tests()

        with counters_lock:
            assert calls == 2
            assert max_active_requests == 1
        assert [result["successful_cameras"] for result in results] == [1, 1]

    def test_sync_parallel_render_delegates_global_slots_per_camera(
        self,
        monkeypatch,
    ):
        captured: dict[str, object] = {}

        def fake_render_all_cameras(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "successful_cameras": 2,
                "results": [
                    {"camera": "/CameraA", "images": [], "status": "success"},
                    {"camera": "/CameraB", "images": [], "status": "success"},
                ],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            fake_render_all_cameras,
        )

        backend = RemoteRenderingBackend(api_key="test")
        result = backend.render(
            object(),
            cameras=["/CameraA", "/CameraB"],
            max_workers=2,
            render_slot_timeout_sec=7.5,
        )

        assert result["successful_cameras"] == 2
        assert captured["max_workers"] == 2
        assert captured["use_global_render_slots"] is True
        assert captured["render_slot_timeout_sec"] == 7.5

    @pytest.mark.parametrize("global_limit", (1, 2))
    def test_sync_parallel_render_caps_thread_pool_to_global_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        global_limit: int,
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setenv(
            "WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS",
            str(global_limit),
        )
        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            lambda **kwargs: captured.update(kwargs)
            or {"successful_cameras": 4, "results": []},
        )

        backend = RemoteRenderingBackend(api_key="test")
        backend.render(
            object(),
            cameras=[f"/Camera{i}" for i in range(4)],
            max_workers=4,
        )

        assert captured["max_workers"] == global_limit
        assert captured["use_global_render_slots"] is True

    def test_sync_serial_render_surfaces_slot_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class TimeoutSlot:
            def __enter__(self) -> float:
                raise RemoteRenderingSlotTimeoutError("slot capacity unavailable")

            def __exit__(self, *args: object) -> None:
                return None

        monkeypatch.setattr(
            render_remote_async,
            "global_remote_render_slot",
            lambda **kwargs: TimeoutSlot(),
        )
        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras",
            lambda **kwargs: pytest.fail("slot timeout must stop rendering"),
        )

        backend = RemoteRenderingBackend(api_key="test")
        with pytest.raises(
            RemoteRenderingSlotTimeoutError,
            match="slot capacity unavailable",
        ):
            backend.render(
                object(),
                cameras=["/Camera"],
                max_workers=1,
                render_slot_timeout_sec=0.01,
            )

    @pytest.mark.parametrize("max_workers", (0, -1, 33, True, "2"))
    def test_sync_render_rejects_invalid_worker_count(
        self,
        max_workers: object,
    ) -> None:
        backend = RemoteRenderingBackend(api_key="test")

        with pytest.raises(
            ValueError,
            match="Remote render max_workers must be an integer between 1 and 32",
        ):
            backend.render(object(), cameras=["/Camera"], max_workers=max_workers)

    def test_url_prim_render_passes_preview_fallback_flag(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_render_all_cameras_from_url(**kwargs: Any) -> dict[str, Any]:
            captured["add_preview_fallbacks"] = kwargs.get("add_preview_fallbacks")
            return {
                "successful_cameras": 1,
                "results": [{"images": ["image"], "status": "success"}],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras_from_url",
            fake_render_all_cameras_from_url,
        )

        backend = RemoteRenderingBackend(api_key="test", add_preview_fallbacks=False)
        config = rendering.RenderingConfig(image_width=64)

        rendering.render_from_prepared_prims(
            backend,
            object(),
            ["/Camera"],
            1,
            ["/World/Prim"],
            config,
            stage_url="https://example.com/scene.usd",
        )

        assert captured["add_preview_fallbacks"] is False

    def test_url_composition_render_passes_preview_fallback_flag(self, monkeypatch):
        captured: list[object] = []

        def fake_render_all_cameras_from_url(**kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs.get("add_preview_fallbacks"))
            return {
                "successful_cameras": 1,
                "results": [
                    {
                        "camera": "/Camera",
                        "images": [Image.new("RGB", (4, 4), (0, 0, 0))],
                        "status": "success",
                    }
                ],
            }

        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras_from_url",
            fake_render_all_cameras_from_url,
        )

        backend = RemoteRenderingBackend(api_key="test", add_preview_fallbacks=False)
        config = rendering.RenderingConfig(image_width=64)

        rendering.render_from_prepared_composition(
            backend,
            object(),
            ["/Camera"],
            object(),
            ["/Camera"],
            1,
            ["/World/Prim"],
            config,
            highlight_url="https://example.com/highlight.usd",
            plain_url="https://example.com/plain.usd",
        )

        assert captured == [False, False]

    def test_prepared_prims_with_ovrtx_backend_uses_standard_render(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            "world_understanding.functions.graphics.render_ovrtx._get_ovrtx_python",
            lambda venv_dir=None: "/fake/python",
        )

        backend = OvRTXRenderingBackend(ovrtx_venv_dir=str(tmp_path / "ovrtx_venv"))
        captured: dict[str, object] = {}

        def fake_render(stage, **kwargs: Any) -> dict[str, Any]:
            captured["stage"] = stage
            captured["kwargs"] = kwargs
            return {
                "successful_cameras": 1,
                "results": [
                    {
                        "camera": "/Camera",
                        "images": [Image.new("RGB", (4, 4), (0, 0, 0))],
                        "status": "success",
                    }
                ],
            }

        monkeypatch.setattr(backend, "render", fake_render)
        config = rendering.RenderingConfig(image_width=64)
        stage = object()

        result = rendering.render_from_prepared_prims(
            backend,
            stage,
            ["/Camera"],
            1,
            ["/World/Prim"],
            config,
            stage_url="https://example.com/scene.usd",
        )

        assert backend.material_target == "auto"
        assert backend.add_preview_fallbacks is False
        assert captured["stage"] is stage
        assert captured["kwargs"]["cameras"] == ["/Camera"]
        assert result["successful_cameras"] == 1
        assert result["results"][0]["prim_to_images"]["/World/Prim"] is not None


class TestPrepareRenderPrims:
    """Tests for USD render preparation."""

    def test_hidden_ancestor_is_made_visible_for_requested_prim(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Imageable(stage.GetPrimAtPath("/World")).MakeInvisible()

        target = UsdGeom.Mesh.Define(stage, "/World/Target")
        target.CreatePointsAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(0.0, 0.0, 0.0),
                    Gf.Vec3f(1.0, 0.0, 0.0),
                    Gf.Vec3f(0.0, 1.0, 0.0),
                ]
            )
        )
        target.CreateFaceVertexCountsAttr([3])
        target.CreateFaceVertexIndicesAttr([0, 1, 2])
        target.CreateExtentAttr(
            Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 1.0, 0.0)])
        )

        other = UsdGeom.Mesh.Define(stage, "/World/Other")
        other.CreatePointsAttr(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(2.0, 0.0, 0.0),
                    Gf.Vec3f(3.0, 0.0, 0.0),
                    Gf.Vec3f(2.0, 1.0, 0.0),
                ]
            )
        )
        other.CreateFaceVertexCountsAttr([3])
        other.CreateFaceVertexIndicesAttr([0, 1, 2])
        other.CreateExtentAttr(
            Vt.Vec3fArray([Gf.Vec3f(2.0, 0.0, 0.0), Gf.Vec3f(3.0, 1.0, 0.0)])
        )

        rendering.prepare_render_prims(
            stage,
            ["/World/Target"],
            rendering.RenderingConfig(),
            render_mode="prim_only",
        )

        assert (
            UsdGeom.Imageable(stage.GetPrimAtPath("/World")).GetVisibilityAttr().Get()
            == UsdGeom.Tokens.inherited
        )
        assert (
            target.GetVisibilityAttr().Get(Usd.TimeCode(0)) == UsdGeom.Tokens.inherited
        )
        assert (
            other.GetVisibilityAttr().Get(Usd.TimeCode(0)) == UsdGeom.Tokens.invisible
        )

    def test_url_render_uses_global_request_limit(self, monkeypatch):
        active_requests = 0
        max_active_requests = 0
        calls = 0
        counters_lock = threading.Lock()

        def fake_render_all_cameras_from_url(**kwargs: Any) -> dict[str, Any]:
            nonlocal active_requests, max_active_requests, calls
            with counters_lock:
                calls += 1
                active_requests += 1
                max_active_requests = max(max_active_requests, active_requests)
            time.sleep(0.01)
            with counters_lock:
                active_requests -= 1
            return {
                "successful_cameras": 1,
                "results": [{"images": ["image"], "status": "success"}],
            }

        monkeypatch.setenv("WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS", "1")
        render_remote_async._reset_global_remote_render_semaphore_for_tests()
        monkeypatch.setattr(
            rendering.render_remote,
            "render_all_cameras_from_url",
            fake_render_all_cameras_from_url,
        )

        backend = RemoteRenderingBackend(api_key="test")
        config = rendering.RenderingConfig(image_width=64)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        rendering.render_from_prepared_prims,
                        backend,
                        object(),
                        ["/Camera"],
                        1,
                        ["/World/Prim"],
                        config,
                        stage_url="https://example.com/scene.usd",
                    )
                    for _ in range(2)
                ]
                results = [future.result() for future in futures]
        finally:
            render_remote_async._reset_global_remote_render_semaphore_for_tests()

        with counters_lock:
            assert calls == 2
            assert max_active_requests == 1
        assert [result["successful_cameras"] for result in results] == [1, 1]
