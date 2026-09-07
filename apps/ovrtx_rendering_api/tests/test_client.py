# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

CLIENT_PATH = Path(__file__).resolve().parents[1] / "client" / "client.py"


def _png(width: int, height: int) -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (width, height)).save(payload, format="PNG")
    return payload.getvalue()


def _load_client_module():
    spec = importlib.util.spec_from_file_location("ovrtx_smoke_client", CLIENT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_client_module_loads_without_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)

    client = _load_client_module()

    assert callable(client.render_smoke)


class _FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        status_code_error: BaseException | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.status_code_error = status_code_error
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True
        if self.status_code_error is not None:
            raise self.status_code_error

    def json(self) -> object:
        return self.payload


def test_encode_usd_as_data_uri(tmp_path: Path) -> None:
    client = _load_client_module()
    usd = tmp_path / "scene.usda"
    usd.write_bytes(b"#usda 1.0\n")

    data_uri = client._encode_usd_as_data_uri(usd)

    prefix, encoded = data_uri.split(",", 1)
    assert prefix == "data:application/octet-stream;base64"
    assert base64.b64decode(encoded) == b"#usda 1.0\n"


def test_build_request_uses_expected_smoke_defaults() -> None:
    client = _load_client_module()

    payload = client._build_request("data:application/octet-stream;base64,AA==")

    assert payload["url"] == "data:application/octet-stream;base64,AA=="
    assert payload["force_render"] is True
    assert payload["render_settings"]["camera_paths"] == ["/World/Camera"]
    assert payload["render_settings"]["frame_range"] == {"start": 0, "end": 0}
    assert payload["render_settings"]["camera_parameters"] == {
        "width": 256,
        "height": 256,
    }


def test_health_check_sends_optional_bearer_token(monkeypatch) -> None:
    client = _load_client_module()
    calls: list[dict] = []
    monkeypatch.setenv("NVCF_INVOKE_VERSION_ID", "version-under-test")

    def fake_get(url: str, *, headers: dict, timeout: float):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeResponse({"gpu_initialized": True})

    monkeypatch.setattr(client.requests, "get", fake_get)

    client.health_check("http://renderer.test/", "secret", 12.5)

    assert calls == [
        {
            "url": "http://renderer.test/health",
            "headers": {
                "Authorization": "Bearer secret",
                "Function-Version-Id": "version-under-test",
            },
            "timeout": 12.5,
        }
    ]


def test_health_check_exits_retryable_when_renderer_is_not_initialized(
    monkeypatch,
) -> None:
    client = _load_client_module()

    monkeypatch.setattr(
        client.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse({"gpu_initialized": False}),
    )

    with pytest.raises(SystemExit) as exc_info:
        client.health_check("http://renderer.test", None, 1.0)

    assert exc_info.value.code == client.RETRYABLE_EXIT_CODE


def test_retryable_503_uses_temporary_failure_exit() -> None:
    client = _load_client_module()
    response = _FakeResponse(
        {"error_code": "renderer_initializing", "retryable": True},
        status_code=503,
    )

    with pytest.raises(SystemExit) as exc_info:
        client._raise_for_status(response)

    assert exc_info.value.code == client.RETRYABLE_EXIT_CODE
    assert response.raised is False


def test_protocol_v3_retry_after_uses_temporary_failure_exit() -> None:
    client = _load_client_module()
    response = _FakeResponse(
        {"detail": "worker response truncated"},
        status_code=503,
        headers={"Retry-After": "1"},
    )

    with pytest.raises(SystemExit) as exc_info:
        client._raise_for_status(response)

    assert exc_info.value.code == client.RETRYABLE_EXIT_CODE
    assert response.raised is False


def test_retry_after_handles_non_object_json_body() -> None:
    client = _load_client_module()
    response = _FakeResponse(
        [],
        status_code=503,
        headers={"Retry-After": "1"},
    )

    with pytest.raises(SystemExit) as exc_info:
        client._raise_for_status(response)

    assert exc_info.value.code == client.RETRYABLE_EXIT_CODE
    assert response.raised is False


def test_gateway_timeout_uses_temporary_failure_exit() -> None:
    client = _load_client_module()
    response = _FakeResponse(
        {},
        status_code=504,
    )

    with pytest.raises(SystemExit) as exc_info:
        client._raise_for_status(response)

    assert exc_info.value.code == client.RETRYABLE_EXIT_CODE
    assert response.raised is False


def test_renderer_not_ready_uses_temporary_failure_exit() -> None:
    client = _load_client_module()
    response = _FakeResponse(
        {"detail": "renderer is not ready"},
        status_code=503,
    )

    with pytest.raises(SystemExit) as exc_info:
        client._raise_for_status(response)

    assert exc_info.value.code == client.RETRYABLE_EXIT_CODE
    assert response.raised is False


def test_non_retryable_503_fails_immediately() -> None:
    client = _load_client_module()
    error = RuntimeError("permanent failure")
    response = _FakeResponse(
        {"error_code": "invalid_render", "retryable": False},
        status_code=503,
        status_code_error=error,
    )

    with pytest.raises(RuntimeError, match="permanent failure"):
        client._raise_for_status(response)

    assert response.raised is True


def test_render_smoke_posts_json_and_counts_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client_module()
    monkeypatch.delenv("NVCF_INVOKE_VERSION_ID", raising=False)
    usd = tmp_path / "scene.usda"
    usd.write_bytes(b"#usda 1.0\n")
    calls: list[dict] = []

    def fake_post(url: str, *, data: str, headers: dict, timeout: float):
        calls.append(
            {
                "url": url,
                "data": json.loads(data),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "status": "success",
                "images": {
                    "0": {
                        "/World/Camera": {
                            "images": "rgb",
                            "depth": "depth",
                        }
                    }
                },
            }
        )

    monkeypatch.setattr(client.requests, "post", fake_post)

    client.render_smoke("http://renderer.test/", usd, "secret", 2.0)

    assert calls[0]["url"] == "http://renderer.test/render"
    assert calls[0]["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    assert calls[0]["timeout"] == 2.0
    assert calls[0]["data"]["url"].startswith("data:application/octet-stream;base64,")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"status": "exception", "error": "boom", "images": {}}, "status=exception"),
        ({"status": "success", "images": {}}, "empty images map"),
    ],
)
def test_render_smoke_exits_on_bad_render_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    message: str,
) -> None:
    client = _load_client_module()
    usd = tmp_path / "scene.usda"
    usd.write_bytes(b"#usda 1.0\n")
    monkeypatch.setattr(
        client.requests,
        "post",
        lambda *_args, **_kwargs: _FakeResponse(payload),
    )

    with pytest.raises(SystemExit, match=message):
        client.render_smoke("http://renderer.test", usd, None, 1.0)


def test_protocol_v3_render_smoke_posts_usdz_and_validates_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client_module()
    monkeypatch.setenv("NVCF_INVOKE_VERSION_ID", "version-under-test")
    usd = tmp_path / "scene.usda"
    usd.write_bytes(b"#usda 1.0\n")
    calls: list[dict] = []

    def fake_post(
        url: str,
        *,
        files: dict,
        data: dict,
        headers: dict,
        timeout: float,
    ) -> _FakeResponse:
        calls.append(
            {
                "url": url,
                "files": files,
                "params": json.loads(data["params"]),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _FakeResponse(
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "frame": 0.0,
                        "image_base64": base64.b64encode(_png(64, 64)).decode("ascii"),
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 8,
                        "active_aov": "LdrColor",
                    }
                ]
            }
        )

    monkeypatch.setattr(client.requests, "post", fake_post)

    client.protocol_v3_render_smoke("http://renderer.test/", usd, "secret", 4.0)

    call = calls[0]
    assert call["url"] == "http://renderer.test/render/upload"
    assert call["headers"] == {
        "Authorization": "Bearer secret",
        "Function-Version-Id": "version-under-test",
    }
    assert call["timeout"] == 4.0
    assert call["params"] == {
        "cameras": ["/World/Camera"],
        "image_width": 64,
        "image_height": 64,
        "mode": "fast",
        "compression": "none",
        "frames": [0.0],
    }
    filename, bundle, content_type = call["files"]["file"]
    assert filename == "smoke_scene.usdz"
    assert content_type == "application/octet-stream"
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.namelist() == ["scene.usda"]
        assert archive.read("scene.usda") == b"#usda 1.0\n"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"results": []}, "no protocol-v3 results"),
        (
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "image_base64": "",
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 8,
                        "active_aov": "LdrColor",
                    }
                ]
            },
            "empty image",
        ),
        (
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "image_base64": base64.b64encode(b"not a png").decode("ascii"),
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 8,
                        "active_aov": "LdrColor",
                    }
                ]
            },
            "invalid PNG image data",
        ),
        (
            {
                "results": [
                    {
                        "camera": "/World/Camera",
                        "image_base64": base64.b64encode(_png(32, 64)).decode("ascii"),
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 8,
                        "active_aov": "LdrColor",
                    }
                ]
            },
            "PNG dimensions 32x64; expected 64x64",
        ),
    ],
)
def test_protocol_v3_render_smoke_exits_on_bad_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    message: str,
) -> None:
    client = _load_client_module()
    usd = tmp_path / "scene.usda"
    usd.write_bytes(b"#usda 1.0\n")
    monkeypatch.setattr(
        client.requests,
        "post",
        lambda *_args, **_kwargs: _FakeResponse(payload),
    )

    with pytest.raises(SystemExit, match=message):
        client.protocol_v3_render_smoke("http://renderer.test", usd, None, 1.0)


def test_main_exits_when_usd_fixture_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client_module()
    missing = tmp_path / "missing.usda"
    monkeypatch.setattr(
        client.sys,
        "argv",
        [
            "client.py",
            "--base-url",
            "http://renderer.test",
            "--usd",
            str(missing),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        client.main()

    assert exc_info.value.code == 1


def test_main_runs_health_and_render_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client_module()
    usd = tmp_path / "scene.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    calls: list[tuple] = []

    monkeypatch.setattr(
        client.sys,
        "argv",
        [
            "client.py",
            "--base-url",
            "http://renderer.test",
            "--usd",
            str(usd),
            "--token",
            "secret",
            "--timeout",
            "3",
            "--require-protocol-v3",
        ],
    )
    monkeypatch.setattr(
        client,
        "health_check",
        lambda *args: calls.append(("health", args)),
    )
    monkeypatch.setattr(
        client,
        "render_smoke",
        lambda *args: calls.append(("render", args)),
    )
    monkeypatch.setattr(
        client,
        "protocol_v3_render_smoke",
        lambda *args: calls.append(("protocol-v3", args)),
    )

    client.main()

    assert calls == [
        ("health", ("http://renderer.test", "secret", 3.0)),
        ("render", ("http://renderer.test", usd, "secret", 3.0)),
        ("protocol-v3", ("http://renderer.test", usd, "secret", 3.0)),
    ]


def test_main_can_skip_health_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client_module()
    usd = tmp_path / "scene.usda"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        client.sys,
        "argv",
        [
            "client.py",
            "--base-url",
            "http://renderer.test",
            "--usd",
            str(usd),
            "--skip-health",
        ],
    )
    monkeypatch.setattr(client, "health_check", lambda *args: calls.append("health"))
    monkeypatch.setattr(client, "render_smoke", lambda *args: calls.append("render"))

    client.main()

    assert calls == ["render"]
