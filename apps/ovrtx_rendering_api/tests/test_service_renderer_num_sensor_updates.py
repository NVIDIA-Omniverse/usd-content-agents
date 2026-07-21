# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-request ``num_sensor_updates`` plumbing.

Exercises that the render-settings field is forwarded into the backend's
``num_sensor_updates`` kwarg without touching pxr or ovrtx — the backend is
replaced with a recording stub.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock, patch

import pytest
from PIL import Image

# service/ is on sys.path via [tool.pytest.ini_options].pythonpath in the
# root pyproject.toml.
from service import renderer as renderer_module


class _RecordingBackend:
    """Stand-in for OvRTXRenderingBackend that records render() kwargs."""

    def __init__(self) -> None:
        self.last_num_sensor_updates: int | None | object = object()
        self.last_render_mode: str | None | object = object()
        self.last_material_target: str | None | object = object()
        self.last_kwargs: dict[str, Any] = {}
        self.render_calls = 0
        self.responses: list[dict[str, Any] | BaseException] = []

    def render(self, **kwargs: Any) -> dict[str, Any]:
        self.render_calls += 1
        self.last_num_sensor_updates = kwargs.get("num_sensor_updates")
        self.last_render_mode = kwargs.get("render_mode")
        self.last_material_target = kwargs.get("material_target")
        self.last_kwargs = kwargs
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return self.success_result()

    @staticmethod
    def success_result() -> dict[str, Any]:
        # Produce a minimally valid backend result so _to_v1_response
        # doesn't need to deal with empty payloads.
        image = Image.new("RGB", (2, 2), color=(8, 16, 32))
        image.putpixel((1, 0), (64, 16, 32))
        image.putpixel((0, 1), (8, 80, 32))
        image.putpixel((1, 1), (8, 16, 96))
        return {
            "results": [
                {
                    "camera": "/World/Camera",
                    "successful_frames": 1,
                    "failed_frames": 0,
                    "images": [image],
                    "sensor_files": {},
                }
            ]
        }


class _FakeStage:
    """Truthy placeholder so ``if not stage:`` does not short-circuit."""

    def __bool__(self) -> bool:
        return True


class _FakeDaemon:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.running = True

    def _is_running(self) -> bool:
        return self.running

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.running = False


class _BackendWithDaemon:
    def __init__(self, daemon: _FakeDaemon) -> None:
        self._daemon = daemon


@pytest.fixture
def renderer_with_stub_backend():
    """Produce a Renderer whose backend, stage open, and fetch are stubbed."""
    import types

    # Skip Renderer.__init__ — it constructs OvRTXRenderingBackend, which
    # requires the ovrtx venv and a GPU. The only attributes render() needs
    # are _backend, _render_lock, and _initialized.
    backend = _RecordingBackend()
    r = renderer_module.Renderer.__new__(renderer_module.Renderer)
    r._backend = backend  # type: ignore[attr-defined]
    r._initialized = True  # type: ignore[attr-defined]
    r._render_lock = threading.RLock()  # type: ignore[attr-defined]

    # Stub the fetch (no network / disk) and pxr (no USD C++ deps).
    pxr_mod = types.ModuleType("pxr")
    pxr_mod.Usd = types.SimpleNamespace(  # type: ignore[attr-defined]
        Stage=types.SimpleNamespace(Open=lambda _p: _FakeStage())
    )

    with (
        patch.object(renderer_module, "_fetch_usd", lambda _url, _path: None),
        patch.dict("sys.modules", {"pxr": pxr_mod}),
    ):
        yield r, backend


class TestNumSensorUpdatesPrecedence:
    def test_per_request_value_is_forwarded_to_backend(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            num_sensor_updates=32,
        )
        assert backend.last_num_sensor_updates == 32

    def test_unset_passes_none_so_backend_uses_instance_default(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )
        assert backend.last_num_sensor_updates is None


class TestRenderSettingsValidation:
    def test_num_sensor_updates_rejects_zero(self):
        from service.models import RenderSettings

        with pytest.raises(ValueError):
            RenderSettings(num_sensor_updates=0)

    def test_num_sensor_updates_accepts_positive(self):
        from service.models import RenderSettings

        assert RenderSettings(num_sensor_updates=1).num_sensor_updates == 1
        assert RenderSettings(num_sensor_updates=100).num_sensor_updates == 100

    def test_default_is_none_so_backend_instance_default_wins(self):
        """Unspecified field defaults to None so OVRTX_NUM_SENSOR_UPDATES still governs."""
        from service.models import RenderSettings

        assert RenderSettings().num_sensor_updates is None


class TestRenderModePrecedence:
    """Mirrors the num_sensor_updates precedence tests — per-request render_mode
    must reach the backend, and None must leave the backend's instance
    default (seeded from OVRTX_RENDER_MODE env at service startup) alone.
    """

    def test_per_request_mode_is_forwarded_to_backend(self, renderer_with_stub_backend):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            render_mode="rt2",
        )
        assert backend.last_render_mode == "rt2"

    def test_unset_mode_passes_none_to_backend(self, renderer_with_stub_backend):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )
        assert backend.last_render_mode is None

    @pytest.mark.parametrize("mode", ["rt1", "rt2", "pt"])
    def test_all_supported_modes_are_forwarded(self, renderer_with_stub_backend, mode):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            render_mode=mode,
        )
        assert backend.last_render_mode == mode


class TestMaterialTargetPrecedence:
    def test_per_request_target_is_forwarded_to_backend(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
            material_target="openpbr_materialx",
        )
        assert backend.last_material_target == "openpbr_materialx"

    def test_unset_target_passes_none_to_backend(self, renderer_with_stub_backend):
        r, backend = renderer_with_stub_backend
        r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )
        assert backend.last_material_target is None


class TestBlankRenderDetection:
    def test_all_blank_frames_return_success_with_warning_metadata(self):
        response = renderer_module._to_v1_response(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [Image.new("RGB", (4, 4), color=(0, 0, 0))],
                        "sensors": {},
                    }
                ]
            },
            requested_sensors=[],
            ovrtx_sensors=[],
            frame_start=0,
        )

        assert response["status"] == "success"
        assert "0" in response["images"]
        assert response["error"] is None
        assert "warnings" in response
        assert response["blank_render_frames"][0]["stats"]["blank"] is True

    def test_partial_blank_frames_return_success_with_warning(self):
        nonblank = Image.new("RGB", (2, 2), color=(8, 16, 32))
        nonblank.putpixel((1, 0), (64, 16, 32))
        nonblank.putpixel((0, 1), (8, 80, 32))
        nonblank.putpixel((1, 1), (8, 16, 96))

        response = renderer_module._to_v1_response(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [
                            Image.new("RGB", (4, 4), color=(255, 255, 255)),
                            nonblank,
                        ],
                        "sensors": {},
                    }
                ]
            },
            requested_sensors=[],
            ovrtx_sensors=[],
            frame_start=4,
        )

        assert response["status"] == "success"
        assert response["blank_render_frames"][0]["frame"] == 4
        assert "warnings" in response
        assert "4" in response["images"]
        assert "5" in response["images"]

    def test_v1_response_uses_upstream_blank_frame_metadata(self):
        nonblank = Image.new("RGB", (2, 2), color=(8, 16, 32))
        nonblank.putpixel((1, 0), (64, 16, 32))
        nonblank.putpixel((0, 1), (8, 80, 32))
        nonblank.putpixel((1, 1), (8, 16, 96))

        response = renderer_module._to_v1_response(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [nonblank],
                        "image_frames": [42],
                        "sensors": {},
                        "warnings": ["worker warning"],
                        "blank_render_frames": [
                            {
                                "frame": 42,
                                "camera": "/World/Camera",
                                "stats": {"blank": True, "reason": "solid_color"},
                            }
                        ],
                    }
                ]
            },
            requested_sensors=[],
            ovrtx_sensors=[],
            frame_start=0,
        )

        assert response["status"] == "success"
        assert "42" in response["images"]
        assert response["blank_render_frames"][0]["frame"] == 42
        assert "worker warning" in response["warnings"]

    def test_v1_response_ignores_out_of_range_upstream_blank_frame(self):
        nonblank = Image.new("RGB", (2, 2), color=(8, 16, 32))
        nonblank.putpixel((1, 0), (64, 16, 32))
        nonblank.putpixel((0, 1), (8, 80, 32))
        nonblank.putpixel((1, 1), (8, 16, 96))

        response = renderer_module._to_v1_response(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [nonblank],
                        "image_frames": [42],
                        "sensors": {},
                        "blank_render_frames": [
                            {
                                "frame": 99,
                                "camera": "/World/Camera",
                                "stats": {"blank": True, "reason": "solid_color"},
                            }
                        ],
                    }
                ]
            },
            requested_sensors=[],
            ovrtx_sensors=[],
            frame_start=0,
        )

        assert response["status"] == "success"
        assert "warnings" not in response

    def test_normalize_blank_frame_whitelists_expected_fields(self):
        normalized = renderer_module._normalize_blank_frame(
            {
                "frame": 7,
                "camera": "/World/Camera",
                "stats": {"blank": True, "reason": "solid_color"},
                "image_file": "frame_7.png",
                "unexpected": "ignored",
            },
            default_camera="/World/Camera",
        )

        assert normalized == {
            "frame": 7,
            "camera": "/World/Camera",
            "stats": {"blank": True, "reason": "solid_color"},
            "image_file": "frame_7.png",
        }


class TestDaemonRecovery:
    def test_recover_shuts_down_daemon_and_warms_up(self):
        daemon = _FakeDaemon()
        r = renderer_module.Renderer.__new__(renderer_module.Renderer)
        r._backend = _BackendWithDaemon(daemon)  # type: ignore[attr-defined]
        r._initialized = True  # type: ignore[attr-defined]
        r._render_lock = threading.RLock()  # type: ignore[attr-defined]

        def mark_initialized() -> bool:
            r._initialized = True  # type: ignore[attr-defined]
            daemon.running = True
            return True

        r.warm_up = Mock(side_effect=mark_initialized)  # type: ignore[method-assign]

        assert r.recover(force=True) is True
        assert daemon.shutdown_calls == 1
        assert r._initialized is True  # type: ignore[attr-defined]
        r.warm_up.assert_called_once_with()  # type: ignore[attr-defined]

    def test_recover_is_single_flight_after_concurrent_recovery_succeeds(self):
        daemon = _FakeDaemon()
        daemon.running = False
        r = renderer_module.Renderer.__new__(renderer_module.Renderer)
        r._backend = _BackendWithDaemon(daemon)  # type: ignore[attr-defined]
        r._initialized = False  # type: ignore[attr-defined]
        r._render_lock = threading.RLock()  # type: ignore[attr-defined]

        first_recovery_entered = threading.Event()
        release_first_recovery = threading.Event()
        warm_up_calls = 0

        def mark_initialized() -> bool:
            nonlocal warm_up_calls
            warm_up_calls += 1
            if warm_up_calls == 1:
                first_recovery_entered.set()
                assert release_first_recovery.wait(timeout=2.0)
            r._initialized = True  # type: ignore[attr-defined]
            daemon.running = True
            return True

        r.warm_up = Mock(side_effect=mark_initialized)  # type: ignore[method-assign]

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(r.recover)
            assert first_recovery_entered.wait(timeout=2.0)

            second = executor.submit(r.recover)
            release_first_recovery.set()

            assert first.result(timeout=2.0) is True
            assert second.result(timeout=2.0) is True

        assert daemon.shutdown_calls == 1
        assert warm_up_calls == 1

    def test_retries_once_after_recoverable_daemon_failure(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        backend.responses = [
            TimeoutError("OvRTX daemon render timed out after 1.0s"),
            _RecordingBackend.success_result(),
        ]
        r.recover = Mock(return_value=True)  # type: ignore[method-assign]

        response = r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )

        assert response["status"] == "success"
        assert backend.render_calls == 2
        r.recover.assert_called_once_with(force=True)  # type: ignore[attr-defined]

    def test_does_not_retry_nonrecoverable_render_error(
        self, renderer_with_stub_backend
    ):
        r, backend = renderer_with_stub_backend
        backend.responses = [RuntimeError("OvRTX daemon render error: bad camera")]
        r.recover = Mock(return_value=True)  # type: ignore[method-assign]

        response = r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )

        assert response["status"] == "exception"
        assert "bad camera" in response["error"]
        assert backend.render_calls == 1
        r.recover.assert_not_called()  # type: ignore[attr-defined]

    def test_reports_exception_when_recovery_fails(self, renderer_with_stub_backend):
        r, backend = renderer_with_stub_backend
        backend.responses = [RuntimeError("OvRTX daemon pipe failed: broken pipe")]
        r.recover = Mock(return_value=False)  # type: ignore[method-assign]

        response = r.render(
            url="data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )

        assert response["status"] == "exception"
        assert response["error"] == "OVRTX daemon recovery failed"
        assert backend.render_calls == 1
        r.recover.assert_called_once_with(force=True)  # type: ignore[attr-defined]
