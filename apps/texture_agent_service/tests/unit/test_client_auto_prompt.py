# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from ...client import client as client_module
from ...client.client import SSEMessage, TextureAgentClient, build_arg_parser


class _Response:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        status_code: int = 200,
    ) -> None:
        self._payload = (
            payload if payload is not None else {"session_id": "session-client-strict"}
        )
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(
                f"{self.status_code} Server Error",
                response=response,
            )

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttp:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts.append({"url": url, **kwargs})
        return _Response()


class _FakeStatusHttp:
    def __init__(self, responses: list[_Response | Exception]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses
        self.gets: list[dict[str, Any]] = []

    def post(self, _url: str, **_kwargs: Any) -> _Response:
        return _Response({"session_id": "session-texture"})

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.gets.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize(
    ("auto_prompt_enabled", "expected"),
    [
        (False, "false"),
        (True, "true"),
    ],
)
def test_client_start_pipeline_serializes_auto_prompting(
    auto_prompt_enabled: bool,
    expected: str,
) -> None:
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    session_id = client.start_pipeline(
        session_id="uploaded-session",
        material_textures={"Aluminum_Matte": {"prompt": "weathered aluminum"}},
        auto_prompt_enabled=auto_prompt_enabled,
    )

    assert session_id == "session-client-strict"
    assert fake_http.posts[0]["data"]["session_id"] == "uploaded-session"
    assert fake_http.posts[0]["data"]["auto_prompt_enabled"] == expected
    assert "Aluminum_Matte" in fake_http.posts[0]["data"]["material_textures_json"]


def test_client_start_pipeline_omits_auto_prompting_when_defaulting() -> None:
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    client.start_pipeline(session_id="uploaded-session")

    assert "auto_prompt_enabled" not in fake_http.posts[0]["data"]


def test_client_start_pipeline_serializes_projection_backend_fields() -> None:
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    client.start_pipeline(
        session_id="uploaded-session",
        texture_backend="service",
        texture_endpoint="http://projection-backend",
        backend_engine="fake_projection",
        backend_custom_parameters={"variant": "success_full_pbr"},
        detail_policy="surface_only",
        reference_image_uris=["file:///ref.png"],
        turntable_video_uri="file:///turntable.mp4",
        multiview_image_uris=["file:///view0.png"],
        seed=11631,
        strength=0.8,
        strict_scope=True,
        uv_policy="force_projection",
        uv_scope="target_prims",
        uv_backend="python",
        uv_projection="box",
        uv_overwrite_existing=False,
        uv_rebake_source_albedo=True,
        uv_rebake_size=2048,
        uv_normalize_out_of_range=False,
    )

    data = fake_http.posts[0]["data"]
    assert data["texture_backend"] == "service"
    assert data["texture_endpoint"] == "http://projection-backend"
    assert data["backend_engine"] == "fake_projection"
    assert json.loads(data["backend_custom_parameters_json"]) == {
        "variant": "success_full_pbr"
    }
    assert data["detail_policy"] == "surface_only"
    assert json.loads(data["reference_image_uris_json"]) == ["file:///ref.png"]
    assert data["turntable_video_uri"] == "file:///turntable.mp4"
    assert json.loads(data["multiview_image_uris_json"]) == ["file:///view0.png"]
    assert data["seed"] == "11631"
    assert data["strength"] == "0.8"
    assert data["strict_scope"] == "true"
    assert data["uv_policy"] == "force_projection"
    assert data["uv_scope"] == "target_prims"
    assert data["uv_backend"] == "python"
    assert data["uv_projection"] == "box"
    assert data["uv_overwrite_existing"] == "false"
    assert data["uv_rebake_source_albedo"] == "true"
    assert data["uv_rebake_size"] == "2048"
    assert data["uv_normalize_out_of_range"] == "false"


def test_client_arg_parser_accepts_detail_policy() -> None:
    args = build_arg_parser().parse_args(
        ["scene.usd", "--detail-policy", "surface_only"]
    )

    assert args.detail_policy == "surface_only"


def test_client_arg_parser_accepts_progress_observability_options() -> None:
    args = build_arg_parser().parse_args(
        [
            "--timeout-seconds",
            "60",
            "--reconnect-attempts",
            "0",
            "--reconnect-backoff-seconds",
            "1.5",
            "--max-polls",
            "45",
            "--max-stale-pending-polls",
            "6",
            "--status-output",
            "status.json",
            "scene.usd",
        ]
    )

    assert args.timeout_seconds == 60
    assert args.reconnect_attempts == 0
    assert args.reconnect_backoff_seconds == 1.5
    assert args.max_polls == 45
    assert args.max_stale_pending_polls == 6
    assert args.status_output == "status.json"


def test_client_main_writes_failed_status_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status_path = tmp_path / "status.json"

    class _FailingClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.base_url = "http://texture.test"

        def run_and_monitor(self, **_kwargs: Any) -> tuple[str, dict[str, Any]]:
            return (
                "session-failed",
                {
                    "session_id": "session-failed",
                    "status": "failed",
                    "error": "startup import failed",
                    "failed_step": "pipeline_startup",
                    "failed_step_stats": {"phase": "import"},
                },
            )

    monkeypatch.setattr(client_module, "TextureAgentClient", _FailingClient)

    rc = client_module.main(
        [
            "--base-url",
            "http://texture.test",
            "--status-output",
            str(status_path),
            "scene.usd",
        ]
    )

    assert rc == 1
    assert json.loads(status_path.read_text(encoding="utf-8")) == {
        "session_id": "session-failed",
        "status": "failed",
        "error": "startup import failed",
        "failed_step": "pipeline_startup",
        "failed_step_stats": {"phase": "import"},
    }
    out = capsys.readouterr().out
    assert "Pipeline status: failed" in out
    assert "Failed step: pipeline_startup" in out
    assert "Error: startup import failed" in out


def test_client_start_pipeline_uploads_reference_image(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"image bytes")
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    client.start_pipeline(
        session_id="uploaded-session",
        reference_image_path=str(reference),
    )

    files = fake_http.posts[0]["files"]
    assert files[0][0] == "reference_image_file"
    assert files[0][1][0] == "reference.png"


def test_client_start_pipeline_closes_usd_when_reference_open_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    class _TrackingHandle:
        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> _TrackingHandle:
            return self

        def __exit__(self, *args: object) -> None:
            self.closed = True

    usd_handle = _TrackingHandle()

    def _open(path: str, mode: str) -> _TrackingHandle:
        assert mode == "rb"
        if path == "scene.usdz":
            return usd_handle
        if path == "missing-reference.png":
            raise FileNotFoundError(path)
        raise AssertionError(f"Unexpected path opened: {path}")

    monkeypatch.setattr("builtins.open", _open)

    with pytest.raises(FileNotFoundError):
        client.start_pipeline(
            usd_path="scene.usdz",
            reference_image_path="missing-reference.png",
        )

    assert usd_handle.closed
    assert fake_http.posts == []


def test_client_run_and_monitor_forwards_projection_backend_fields() -> None:
    class _MonitorClient(TextureAgentClient):
        def __init__(self) -> None:
            super().__init__("http://texture.test")
            self.start_kwargs: dict[str, Any] | None = None

        def start_pipeline(self, **kwargs: Any) -> str:  # type: ignore[override]
            self.start_kwargs = kwargs
            return "session-projection"

        def stream_events(
            self, session_id: str, request_timeout: int | None = None
        ) -> Any:
            yield SSEMessage(event="done", data="{}")

        def get_status(self, session_id: str) -> dict[str, str]:
            return {"status": "completed"}

    client = _MonitorClient()

    session_id, status = client.run_and_monitor(
        usd_path="scene.usd",
        texture_backend="service",
        texture_endpoint="http://projection-backend",
        backend_engine="fake_projection",
        backend_custom_parameters={"variant": "roughness_metalness"},
        reference_image_uris=["file:///ref.png"],
        reference_image_path="reference.png",
        turntable_video_uri="file:///turntable.mp4",
        multiview_image_uris=["file:///view0.png"],
        seed=11631,
        strength=0.8,
        strict_scope=True,
        uv_policy="force_projection",
        uv_scope="target_prims",
        uv_backend="python",
        uv_projection="box",
        uv_overwrite_existing=False,
        uv_rebake_source_albedo=True,
        uv_rebake_size=2048,
        uv_normalize_out_of_range=False,
        print_stream=False,
    )

    assert session_id == "session-projection"
    assert status == {"status": "completed"}
    assert client.start_kwargs is not None
    assert client.start_kwargs["texture_backend"] == "service"
    assert client.start_kwargs["texture_endpoint"] == "http://projection-backend"
    assert client.start_kwargs["backend_engine"] == "fake_projection"
    assert client.start_kwargs["backend_custom_parameters"] == {
        "variant": "roughness_metalness"
    }
    assert client.start_kwargs["reference_image_uris"] == ["file:///ref.png"]
    assert client.start_kwargs["reference_image_path"] == "reference.png"
    assert client.start_kwargs["turntable_video_uri"] == "file:///turntable.mp4"
    assert client.start_kwargs["multiview_image_uris"] == ["file:///view0.png"]
    assert client.start_kwargs["seed"] == 11631
    assert client.start_kwargs["strength"] == 0.8
    assert client.start_kwargs["strict_scope"] is True
    assert client.start_kwargs["uv_policy"] == "force_projection"
    assert client.start_kwargs["uv_scope"] == "target_prims"
    assert client.start_kwargs["uv_backend"] == "python"
    assert client.start_kwargs["uv_projection"] == "box"
    assert client.start_kwargs["uv_overwrite_existing"] is False
    assert client.start_kwargs["uv_rebake_source_albedo"] is True
    assert client.start_kwargs["uv_rebake_size"] == 2048
    assert client.start_kwargs["uv_normalize_out_of_range"] is False


def test_client_cli_flag_disables_auto_prompting() -> None:
    args = build_arg_parser().parse_args(
        ["--disable-auto-prompt", "--quiet", "scene.usd"]
    )

    assert args.disable_auto_prompt is True


def test_client_monitor_retries_transient_status_gateway_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    client._http = _FakeStatusHttp(
        [
            _Response(status_code=504),
            _Response({"status": "completed", "overall_percent": 100}),
            _Response({"status": "completed", "overall_percent": 100}),
        ]
    )
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
    )

    assert session_id == "session-texture"
    assert status is not None
    assert status["status"] == "completed"
    assert len(client._http.gets) == 3
    assert client._http.gets[0]["timeout"] == 30


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("status poll timed out"),
        requests.ConnectionError("status poll connection failed"),
    ],
)
def test_client_monitor_retries_transient_status_network_errors(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    client._http = _FakeStatusHttp(
        [
            error,
            _Response({"status": "completed", "overall_percent": 100}),
            _Response({"status": "completed", "overall_percent": 100}),
        ]
    )
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    _session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
    )

    assert status is not None
    assert status["status"] == "completed"
    assert len(client._http.gets) == 3


def test_client_monitor_preserves_last_status_when_final_status_get_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    client._http = _FakeStatusHttp(
        [
            _Response({"status": "completed", "overall_percent": 100}),
            *[_Response(status_code=504) for _ in range(6)],
        ]
    )
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    _session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
    )

    assert status is not None
    assert status["status"] == "completed"
    assert len(client._http.gets) == 7


def test_client_monitor_stops_on_stale_pending_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    pending_status = {
        "status": "pending",
        "overall_progress": {"percent": 0},
        "completed_steps": [],
    }
    client._http = _FakeStatusHttp(
        [
            _Response(pending_status),
            _Response(pending_status),
            _Response(pending_status),
            _Response(pending_status),
        ]
    )
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    _session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
        max_polls=10,
        max_stale_pending_polls=3,
    )

    assert status is not None
    assert status["status"] == "pending"
    assert len(client._http.gets) == 4


def test_client_monitor_skips_sse_when_stale_pending_guard_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    pending_status = {
        "status": "pending",
        "overall_progress": {"percent": 0},
        "completed_steps": [],
    }
    client._http = _FakeStatusHttp(
        [
            _Response(pending_status),
            _Response(pending_status),
            _Response(pending_status),
            _Response(pending_status),
        ]
    )

    def _unexpected_stream(_session_id: str) -> Any:
        raise AssertionError("SSE stream should be skipped for bounded polling")

    monkeypatch.setattr(client, "stream_events", _unexpected_stream)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    _session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
        max_polls=10,
        max_stale_pending_polls=3,
    )

    assert status is not None
    assert status["status"] == "pending"
    assert len(client._http.gets) == 4


def test_client_monitor_stops_on_stale_startup_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    startup_status = {
        "status": "running",
        "overall_progress": {"percent": 0},
        "current_step": {
            "name": "pipeline_startup",
            "progress": {"message": "Preparing texture pipeline workflow"},
        },
        "completed_steps": [],
    }
    client._http = _FakeStatusHttp(
        [
            _Response(startup_status),
            _Response(startup_status),
            _Response(startup_status),
            _Response(startup_status),
        ]
    )
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    _session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
        max_polls=10,
        max_stale_pending_polls=3,
    )

    assert status is not None
    assert status["status"] == "running"
    assert len(client._http.gets) == 4


def test_client_monitor_preserves_sse_done_status_when_final_status_get_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TextureAgentClient("http://texture.test")
    client._http = _FakeStatusHttp([_Response(status_code=504) for _ in range(6)])
    monkeypatch.setattr(
        client,
        "stream_events",
        lambda _session_id: iter(
            [
                SSEMessage(
                    event="done",
                    data='{"session_id": "session-texture", "final_state": "completed"}',
                )
            ]
        ),
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    _session_id, status = client.run_and_monitor(
        usd_path=None,
        s3_uri="s3://bucket/scene.usda",
        print_stream=False,
    )

    assert status is not None
    assert status["status"] == "completed"
    assert len(client._http.gets) == 6
