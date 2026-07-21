# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest

CLIENT_PATH = Path(__file__).resolve().parents[1] / "client" / "client.py"


def _load_client_module():
    spec = importlib.util.spec_from_file_location("ovrtx_smoke_client", CLIENT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(
        self,
        payload: dict,
        *,
        status_code_error: BaseException | None = None,
    ) -> None:
        self.payload = payload
        self.status_code_error = status_code_error
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True
        if self.status_code_error is not None:
            raise self.status_code_error

    def json(self) -> dict:
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

    def fake_get(url: str, *, headers: dict, timeout: float):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeResponse({"gpu_initialized": True})

    monkeypatch.setattr(client.requests, "get", fake_get)

    client.health_check("http://renderer.test/", "secret", 12.5)

    assert calls == [
        {
            "url": "http://renderer.test/health",
            "headers": {"Authorization": "Bearer secret"},
            "timeout": 12.5,
        }
    ]


def test_health_check_exits_when_renderer_is_not_initialized(monkeypatch) -> None:
    client = _load_client_module()

    monkeypatch.setattr(
        client.requests,
        "get",
        lambda *_args, **_kwargs: _FakeResponse({"gpu_initialized": False}),
    )

    with pytest.raises(SystemExit, match="gpu_initialized=false"):
        client.health_check("http://renderer.test", None, 1.0)


def test_render_smoke_posts_json_and_counts_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _load_client_module()
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

    client.main()

    assert calls == [
        ("health", ("http://renderer.test", "secret", 3.0)),
        ("render", ("http://renderer.test", usd, "secret", 3.0)),
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
