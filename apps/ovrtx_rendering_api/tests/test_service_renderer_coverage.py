# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional branch coverage for the OVRTX renderer service module."""

from __future__ import annotations

import sys
import threading
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image
from service import renderer as renderer_module


def _nonblank_image() -> Image.Image:
    image = Image.new("RGB", (2, 2), color=(8, 16, 32))
    image.putpixel((1, 0), (64, 16, 32))
    image.putpixel((0, 1), (8, 80, 32))
    image.putpixel((1, 1), (8, 16, 96))
    return image


def _renderer_with_backend(backend: Any):
    renderer = renderer_module.Renderer.__new__(renderer_module.Renderer)
    renderer._backend = backend
    renderer._initialized = False
    renderer._render_lock = threading.RLock()
    renderer._recovery_cooldown_until = 0.0
    return renderer


def _install_fake_pxr_stage(monkeypatch: pytest.MonkeyPatch, stage: Any) -> None:
    pxr_mod = types.ModuleType("pxr")
    pxr_mod.Usd = types.SimpleNamespace(
        Stage=types.SimpleNamespace(Open=lambda _path: stage)
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr_mod)


class _FakeStage:
    def __bool__(self) -> bool:
        return True


class _RecordingRenderBackend:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or {
            "results": [
                {
                    "camera": "/World/Camera",
                    "images": [_nonblank_image()],
                    "sensors": {},
                }
            ]
        }

    def render(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


class _FakeDaemon:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.shutdown_calls = 0

    def _is_running(self) -> bool:
        return self.running

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.running = False

    def lifecycle_snapshot(self) -> dict[str, int | str | None]:
        return {
            "daemon_pid": 42,
            "daemon_completed_renders": 7,
            "daemon_rss_bytes": 1234,
            "daemon_recycle_count": 2,
            "daemon_last_recycle_reason": "rss_limit",
            "daemon_pending_recycle_reason": None,
        }


class TestRendererLifecycleCoverage:
    def test_constructor_uses_backend_arguments_and_initializes_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rendering_mod = types.ModuleType(
            "world_understanding.functions.graphics.rendering"
        )

        class FakeBackend:
            def __init__(
                self,
                *,
                log_level: str,
                num_sensor_updates: int,
                render_mode: str,
            ) -> None:
                self.kwargs = {
                    "log_level": log_level,
                    "num_sensor_updates": num_sensor_updates,
                    "render_mode": render_mode,
                }

        rendering_mod.OvRTXRenderingBackend = FakeBackend
        monkeypatch.setitem(
            sys.modules,
            "world_understanding.functions.graphics.rendering",
            rendering_mod,
        )

        renderer = renderer_module.Renderer(
            log_level="info",
            num_sensor_updates=7,
            render_mode="rt1",
        )

        assert renderer._backend.kwargs == {
            "log_level": "info",
            "num_sensor_updates": 7,
            "render_mode": "rt1",
        }
        assert renderer.is_initialized is False
        assert renderer.is_ready is False

    def test_daemon_running_handles_missing_and_non_callable_daemon(self) -> None:
        renderer = _renderer_with_backend(types.SimpleNamespace())
        assert renderer.daemon_running is False
        assert renderer.daemon_lifecycle == {}

        renderer._backend = types.SimpleNamespace(
            _daemon=types.SimpleNamespace(
                _is_running=True,
                lifecycle_snapshot={"daemon_pid": 42},
            )
        )
        assert renderer.daemon_running is False
        assert renderer.daemon_lifecycle == {}

        renderer._backend = types.SimpleNamespace(
            _daemon=types.SimpleNamespace(lifecycle_snapshot=lambda: "bad")
        )
        assert renderer.daemon_lifecycle == {}

    def test_daemon_lifecycle_returns_daemon_snapshot(self) -> None:
        renderer = _renderer_with_backend(types.SimpleNamespace(_daemon=_FakeDaemon()))

        assert renderer.daemon_lifecycle == {
            "daemon_pid": 42,
            "daemon_completed_renders": 7,
            "daemon_rss_bytes": 1234,
            "daemon_recycle_count": 2,
            "daemon_last_recycle_reason": "rss_limit",
            "daemon_pending_recycle_reason": None,
        }

    def test_warm_up_success_sets_initialized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _RecordingRenderBackend(response={"results": [{"ok": True}]})
        renderer = _renderer_with_backend(backend)
        monkeypatch.setattr(renderer_module, "_build_smoke_stage", lambda: "stage")

        assert renderer.warm_up() is True
        assert renderer.is_initialized is True
        assert backend.calls[0]["stage"] == "stage"
        assert backend.calls[0]["cameras"] == ["/World/Camera"]

    def test_warm_up_failure_paths_leave_renderer_uninitialized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(renderer_module, "_build_smoke_stage", lambda: "stage")

        failing_backend = types.SimpleNamespace(
            render=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        renderer = _renderer_with_backend(failing_backend)
        assert renderer.warm_up() is False
        assert renderer.is_initialized is False

        empty_backend = _RecordingRenderBackend(response={"results": []})
        renderer = _renderer_with_backend(empty_backend)
        assert renderer.warm_up() is False
        assert renderer.is_initialized is False

    def test_recover_skips_during_recent_failure_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = _renderer_with_backend(types.SimpleNamespace())
        renderer._recovery_cooldown_until = 20.0
        monkeypatch.setattr(renderer_module.time, "monotonic", lambda: 10.0)

        assert renderer.recover() is False

    def test_recover_records_cooldown_when_shutdown_and_warmup_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = _renderer_with_backend(types.SimpleNamespace())
        renderer.shutdown = lambda: (_ for _ in ()).throw(RuntimeError("no daemon"))
        renderer.warm_up = lambda: False
        monkeypatch.setattr(renderer_module.time, "monotonic", lambda: 10.0)

        assert renderer.recover(force=True) is False
        assert renderer._recovery_cooldown_until == (
            10.0 + renderer_module._RECOVERY_FAILURE_COOLDOWN_SECONDS
        )


class TestRenderBranchCoverage:
    def test_render_rejects_when_stage_open_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = _renderer_with_backend(_RecordingRenderBackend())
        monkeypatch.setattr(renderer_module, "_fetch_usd", lambda _url, _path: None)
        _install_fake_pxr_stage(monkeypatch, None)

        response = renderer.render(
            "data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=0,
            frame_end=0,
            width=64,
            height=64,
        )

        assert response["status"] == "exception"
        assert "Failed to open USD stage" in response["error"]

    def test_render_maps_sensors_frame_range_and_lock_wait(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        depth = np.array([[1.0, 2.0]], dtype=np.float32)
        backend = _RecordingRenderBackend(
            response={
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [_nonblank_image()],
                        "sensors": {"depth": {3: depth}},
                    }
                ]
            }
        )
        renderer = _renderer_with_backend(backend)
        monkeypatch.setattr(renderer_module, "_fetch_usd", lambda _url, _path: None)
        _install_fake_pxr_stage(monkeypatch, _FakeStage())

        times = [0.0, 0.1, 0.2, 0.4]

        def fake_time() -> float:
            if times:
                return times.pop(0)
            return 0.4

        monkeypatch.setattr(renderer_module.time, "time", fake_time)

        response = renderer.render(
            "data:application/octet-stream;base64,AA==",
            camera_paths=["/World/Camera"],
            frame_start=3,
            frame_end=5,
            width=64,
            height=64,
            sensors=["linear_depth", "depth", "normals"],
        )

        assert response["status"] == "success"
        assert backend.calls[0]["frames"] == "3:5"
        assert backend.calls[0]["sensors"] == ["depth"]
        camera_data = response["images"]["3"]["/World/Camera"]
        assert camera_data["linear_depth"]
        assert camera_data["depth"]
        assert camera_data["normals"] == ""


class TestSmokeStageCoverage:
    def test_build_smoke_stage_uses_expected_usd_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, Any]] = []

        class FakeStage:
            def DefinePrim(self, path: str, prim_type: str):
                calls.append(("define", path, prim_type))
                return (path, prim_type)

            def SetDefaultPrim(self, prim: Any) -> None:
                calls.append(("default", prim))

        class FakeAttrPrim:
            def __init__(self, path: str) -> None:
                self.path = path

            def CreateSizeAttr(self, value: float) -> None:
                calls.append(("size", self.path, value))

            def CreateDisplayColorAttr(self, value: Any) -> None:
                calls.append(("color", self.path, value))

            def CreateFocalLengthAttr(self, value: float) -> None:
                calls.append(("focal", self.path, value))

            def CreateHorizontalApertureAttr(self, value: float) -> None:
                calls.append(("horizontal", self.path, value))

            def CreateVerticalApertureAttr(self, value: float) -> None:
                calls.append(("vertical", self.path, value))

            def CreateClippingRangeAttr(self, value: Any) -> None:
                calls.append(("clip", self.path, value))

            def CreateIntensityAttr(self, value: float) -> None:
                calls.append(("intensity", self.path, value))

        class FakeOp:
            def __init__(self, kind: str) -> None:
                self.kind = kind

            def Set(self, value: Any) -> None:
                calls.append((self.kind, value))

        class FakeXformable:
            def __init__(self, prim: Any) -> None:
                calls.append(("xformable", getattr(prim, "path", prim)))

            def AddTranslateOp(self) -> FakeOp:
                return FakeOp("translate")

            def AddRotateXYZOp(self) -> FakeOp:
                return FakeOp("rotate")

        pxr_mod = types.ModuleType("pxr")
        pxr_mod.Gf = types.SimpleNamespace(
            Vec2f=lambda *values: ("Vec2f", values),
            Vec3d=lambda *values: ("Vec3d", values),
            Vec3f=lambda *values: ("Vec3f", values),
        )
        pxr_mod.Usd = types.SimpleNamespace(
            Stage=types.SimpleNamespace(CreateInMemory=lambda: FakeStage())
        )
        pxr_mod.UsdGeom = types.SimpleNamespace(
            Tokens=types.SimpleNamespace(y="y"),
            SetStageUpAxis=lambda stage, axis: calls.append(("axis", axis)),
            SetStageMetersPerUnit=lambda stage, meters: calls.append(
                ("meters", meters)
            ),
            Cube=types.SimpleNamespace(Define=lambda stage, path: FakeAttrPrim(path)),
            Camera=types.SimpleNamespace(Define=lambda stage, path: FakeAttrPrim(path)),
            Xformable=FakeXformable,
        )
        pxr_mod.UsdLux = types.SimpleNamespace(
            DistantLight=types.SimpleNamespace(
                Define=lambda stage, path: FakeAttrPrim(path)
            )
        )
        monkeypatch.setitem(sys.modules, "pxr", pxr_mod)

        stage = renderer_module._build_smoke_stage()

        assert isinstance(stage, FakeStage)
        assert ("axis", "y") in calls
        assert ("size", "/World/Cube", 1.0) in calls
        assert ("intensity", "/World/KeyLight", 5000.0) in calls


class TestUrlAndRequestCoverage:
    def test_legacy_ipv4_parsing_edge_cases(self) -> None:
        assert renderer_module._parse_legacy_ipv4_part("") is None
        assert renderer_module._parse_legacy_ipv4_part("0xnope") is None
        assert renderer_module._parse_legacy_ipv4_part("09") is None
        assert renderer_module._parse_legacy_ipv4_literal("1.2.3.4.5") is None
        assert str(renderer_module._parse_legacy_ipv4_literal("127.1")) == "127.0.0.1"
        assert str(renderer_module._parse_legacy_ipv4_literal("127.0.1")) == "127.0.0.1"

    def test_iter_resolved_host_ips_handles_dns_errors_and_bad_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            renderer_module.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                renderer_module.socket.gaierror()
            ),
        )
        assert renderer_module._iter_resolved_host_ips("missing.example") == ()

        monkeypatch.setattr(
            renderer_module.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (0, 0, 0, "", ()),
                (0, 0, 0, "", ("not-an-ip", 443)),
                (0, 0, 0, "", ("93.184.216.34", 443)),
                (0, 0, 0, "", ("93.184.216.34", 443)),
            ],
        )

        assert tuple(
            map(str, renderer_module._iter_resolved_host_ips("example.com"))
        ) == ("93.184.216.34",)

    def test_validate_url_target_allows_public_literal_ip(self) -> None:
        renderer_module._validate_url_target("https://93.184.216.34/scene.usd")

    def test_private_address_connections_validate_connected_peer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket_obj = object()
        hints: list[str] = []

        monkeypatch.setattr(
            renderer_module.HTTPConnection,
            "_new_conn",
            lambda _self: socket_obj,
        )
        monkeypatch.setattr(
            renderer_module.HTTPSConnection,
            "_new_conn",
            lambda _self: socket_obj,
        )
        monkeypatch.setattr(
            renderer_module,
            "_validate_connected_socket_peer",
            lambda sock, hint: hints.append(hint),
        )

        assert (
            renderer_module._PrivateAddressBlockingHTTPConnection(
                "example.com"
            )._new_conn()
            is socket_obj
        )
        assert (
            renderer_module._PrivateAddressBlockingHTTPSConnection(
                "secure.example"
            )._new_conn()
            is socket_obj
        )
        assert hints == ["http://example.com", "https://secure.example"]

    def test_pool_manager_adapter_and_safe_get(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = renderer_module._PrivateAddressBlockingPoolManager()
        assert (
            manager.pool_classes_by_scheme["http"]
            is renderer_module._PrivateAddressBlockingHTTPConnectionPool
        )
        assert (
            manager.pool_classes_by_scheme["https"]
            is renderer_module._PrivateAddressBlockingHTTPSConnectionPool
        )

        adapter = renderer_module._PrivateAddressBlockingAdapter()
        adapter.init_poolmanager(1, 2, block=True)
        assert isinstance(
            adapter.poolmanager,
            renderer_module._PrivateAddressBlockingPoolManager,
        )

        class FakeSession:
            def __init__(self) -> None:
                self.trust_env = True
                self.mounts: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, *_args: Any) -> None:
                pass

            def mount(self, prefix: str, _adapter: Any) -> None:
                self.mounts.append(prefix)

            def get(self, url: str, *, timeout: float, allow_redirects: bool):
                assert self.trust_env is False
                assert self.mounts == ["http://", "https://"]
                assert url == "https://example.com/scene.usd"
                assert timeout == 3.0
                assert allow_redirects is False
                return "response"

        monkeypatch.setattr(renderer_module.requests, "Session", FakeSession)

        assert (
            renderer_module._safe_requests_get(
                "https://example.com/scene.usd",
                timeout=3.0,
                allow_redirects=False,
            )
            == "response"
        )

    def test_connected_socket_peer_closes_on_invalid_peer_ip(self) -> None:
        class FakeSocket:
            closed = False

            def getpeername(self):
                return ("not-an-ip", 443)

            def close(self) -> None:
                self.closed = True

        sock = FakeSocket()
        with pytest.raises(ValueError):
            renderer_module._validate_connected_socket_peer(sock, "https://example.com")
        assert sock.closed is True

    def test_url_for_hint(self) -> None:
        assert renderer_module._url_for_hint("https", "example.com") == (
            "https://example.com"
        )


class TestZipAndFetchCoverage:
    def test_parse_zip_size_valid_override(self) -> None:
        assert renderer_module._parse_zip_max_uncompressed_bytes("128") == 128

    def test_bad_zip_is_not_usdz(self, tmp_path: Path) -> None:
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_text("not a zip")

        assert renderer_module._zip_matches_usdz_structure(str(bad_zip)) is False

    def test_extract_zip_handles_directories_and_prefer_first_non_usd_entries(
        self, tmp_path: Path
    ) -> None:
        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("assets/", b"")
            zf.writestr("README.md", "notes")
            zf.writestr("assets/stage.usda", '#usda 1.0\ndef Xform "Root" {}\n')

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        extracted = renderer_module._extract_zip_bundle(
            str(zip_path),
            str(work_dir),
            prefer_first_usd=True,
        )

        assert Path(extracted).name == "stage.usda"
        assert (work_dir / "bundle" / "assets").is_dir()

    def test_fetch_usd_routes_s3_and_https_s3_urls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        downloads: list[tuple[str, str]] = []
        monkeypatch.setattr(
            renderer_module,
            "_download_s3",
            lambda url, dest: downloads.append((url, dest)),
        )

        renderer_module._fetch_usd("s3://bucket/path/scene.usd", str(tmp_path / "a"))
        renderer_module._fetch_usd(
            "https://bucket.s3.us-west-2.amazonaws.com/path/scene.usd",
            str(tmp_path / "b"),
        )

        assert downloads == [
            ("s3://bucket/path/scene.usd", str(tmp_path / "a")),
            ("s3://bucket/path/scene.usd", str(tmp_path / "b")),
        ]

    def test_download_s3_uses_profile_when_configured(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = self._install_fake_boto(monkeypatch, behavior="profile")
        monkeypatch.setenv("AWS_PROFILE", "rendering")

        renderer_module._download_s3("s3://bucket/key.usd", str(tmp_path / "scene.usd"))

        assert calls["sessions"] == ["rendering"]
        assert calls["downloads"] == [
            ("bucket", "key.usd", str(tmp_path / "scene.usd"))
        ]

    def test_download_s3_uses_accessible_default_profile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = self._install_fake_boto(monkeypatch, behavior="default-ok")
        monkeypatch.delenv("AWS_PROFILE", raising=False)

        renderer_module._download_s3("s3://bucket/key.usd", str(tmp_path / "scene.usd"))

        assert calls["sessions"] == ["default"]
        assert calls["head_buckets"] == ["bucket"]
        assert calls["downloads"] == [
            ("bucket", "key.usd", str(tmp_path / "scene.usd"))
        ]

    def test_download_s3_falls_back_when_default_profile_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = self._install_fake_boto(monkeypatch, behavior="default-missing")
        monkeypatch.delenv("AWS_PROFILE", raising=False)

        renderer_module._download_s3("s3://bucket/key.usd", str(tmp_path / "scene.usd"))

        assert calls["sessions"] == ["default", None]
        assert calls["downloads"] == [
            ("bucket", "key.usd", str(tmp_path / "scene.usd"))
        ]

    def test_download_s3_falls_back_when_default_profile_cannot_access_bucket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = self._install_fake_boto(monkeypatch, behavior="default-denied")
        monkeypatch.delenv("AWS_PROFILE", raising=False)

        renderer_module._download_s3("s3://bucket/key.usd", str(tmp_path / "scene.usd"))

        assert calls["sessions"] == ["default", None]
        assert calls["head_buckets"] == ["bucket"]
        assert calls["downloads"] == [
            ("bucket", "key.usd", str(tmp_path / "scene.usd"))
        ]

    @staticmethod
    def _install_fake_boto(
        monkeypatch: pytest.MonkeyPatch,
        *,
        behavior: str,
    ) -> dict[str, list[Any]]:
        calls: dict[str, list[Any]] = {
            "sessions": [],
            "head_buckets": [],
            "downloads": [],
        }
        boto3_mod = types.ModuleType("boto3")
        botocore_mod = types.ModuleType("botocore")
        exceptions_mod = types.ModuleType("botocore.exceptions")

        class ProfileNotFound(Exception):
            pass

        class ClientError(Exception):
            pass

        class FakeClient:
            def __init__(self, profile_name: str | None) -> None:
                self.profile_name = profile_name

            def head_bucket(self, *, Bucket: str) -> None:
                calls["head_buckets"].append(Bucket)
                if behavior == "default-denied" and self.profile_name == "default":
                    raise ClientError("denied")

            def download_file(self, bucket: str, key: str, dest_path: str) -> None:
                calls["downloads"].append((bucket, key, dest_path))

        class FakeSession:
            def __init__(self, profile_name: str | None = None) -> None:
                calls["sessions"].append(profile_name)
                if behavior == "default-missing" and profile_name == "default":
                    raise ProfileNotFound("missing")
                self.profile_name = profile_name

            def client(self, service_name: str) -> FakeClient:
                assert service_name == "s3"
                return FakeClient(self.profile_name)

        boto3_mod.Session = FakeSession
        exceptions_mod.ClientError = ClientError
        exceptions_mod.ProfileNotFound = ProfileNotFound
        botocore_mod.exceptions = exceptions_mod
        monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
        monkeypatch.setitem(sys.modules, "botocore", botocore_mod)
        monkeypatch.setitem(sys.modules, "botocore.exceptions", exceptions_mod)
        return calls


class TestV1ResponseCoverage:
    def test_duplicate_blank_frames_and_warnings_are_deduplicated(self) -> None:
        blank_frame = {
            "frame": 0,
            "camera": "/World/Camera",
            "stats": {"blank": True, "reason": "remote_blank_render"},
        }

        response = renderer_module._to_v1_response(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [_nonblank_image()],
                        "image_frames": [0],
                        "sensors": {},
                        "warnings": ["same warning"],
                        "blank_render_frames": [blank_frame],
                    },
                    {
                        "camera": "/World/Camera",
                        "images": [_nonblank_image()],
                        "image_frames": [0],
                        "sensors": {},
                        "warnings": ["same warning"],
                        "blank_render_frames": [blank_frame],
                    },
                ]
            },
            requested_sensors=[],
            ovrtx_sensors=[],
            frame_start=0,
        )

        assert response["blank_render_frames"] == [blank_frame]
        assert response["warnings"].count("same warning") == 1

    def test_sensor_encoding_missing_frames_and_unsupported_sensors(self) -> None:
        response = renderer_module._to_v1_response(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "images": [_nonblank_image(), _nonblank_image()],
                        "image_frames": [5, "bad"],
                        "sensors": {
                            "depth": {
                                5: np.array([[1.0, 2.0]], dtype=np.float32),
                            }
                        },
                    }
                ]
            },
            requested_sensors=["linear_depth", "depth", "normals"],
            ovrtx_sensors=["depth"],
            frame_start=10,
        )

        assert response["images"]["5"]["/World/Camera"]["linear_depth"]
        assert response["images"]["11"]["/World/Camera"]["linear_depth"] == ""
        assert response["images"]["11"]["/World/Camera"]["normals"] == ""

    def test_blank_frame_normalization_rejects_invalid_entries(self) -> None:
        assert (
            renderer_module._blank_frames_by_frame(
                "not a list",
                default_camera="/World/Camera",
            )
            == {}
        )

        frames = renderer_module._blank_frames_by_frame(
            [
                "bad",
                {"frame": -1},
                {"frame": 1, "camera": 123, "stats": "bad stats"},
                {"frame": 2, "camera": "/Other"},
            ],
            default_camera="/World/Camera",
        )

        assert frames == {
            1: {
                "frame": 1,
                "camera": "/World/Camera",
                "stats": {"blank": True, "reason": "remote_blank_render"},
            }
        }

    def test_string_list_rejects_non_lists(self) -> None:
        assert renderer_module._string_list("warning") == []
