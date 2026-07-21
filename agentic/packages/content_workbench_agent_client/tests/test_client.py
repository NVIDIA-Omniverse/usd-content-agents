# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Content Workbench agent client helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

from content_workbench_agent_client import client


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_normalize_url_removes_trailing_slash() -> None:
    assert client.normalize_url("http://127.0.0.1:8088/") == "http://127.0.0.1:8088"


def test_post_json_sends_object_payload(monkeypatch: Any) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        observed["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    result = client.post_json(
        "http://127.0.0.1:8088/sessions",
        {"scene_path": "/tmp/asset.usd"},
        timeout=12.5,
    )

    assert result == {"ok": True}
    assert observed == {
        "url": "http://127.0.0.1:8088/sessions",
        "method": "POST",
        "timeout": 12.5,
        "body": {"scene_path": "/tmp/asset.usd"},
    }


def test_download_to_file_writes_bytes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"png bytes"),
    )

    output = tmp_path / "render.png"
    client.download_to_file("http://127.0.0.1:8088/render.png", output)

    assert output.read_bytes() == b"png bytes"


def test_session_url_encodes_session_id() -> None:
    assert (
        client.session_url(
            "http://127.0.0.1:8088/",
            "session/one",
            "scene/snapshot",
        )
        == "http://127.0.0.1:8088/sessions/session%2Fone/scene/snapshot"
    )


def test_artifact_url_accepts_relative_artifact_path() -> None:
    assert (
        client.artifact_url("http://127.0.0.1:8088/", "artifacts/render.png")
        == "http://127.0.0.1:8088/artifacts/render.png"
    )


def test_artifact_url_accepts_same_origin_absolute_url() -> None:
    assert (
        client.artifact_url("http://127.0.0.1:8088/", "http://127.0.0.1:8088/a.png")
        == "http://127.0.0.1:8088/a.png"
    )


def test_artifact_url_rejects_cross_origin_absolute_url() -> None:
    try:
        client.artifact_url("http://127.0.0.1:8088/", "https://example.test/a.png")
    except RuntimeError as exc:
        assert "same origin" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected RuntimeError")


def test_download_agent_api_docs_uses_standard_endpoints(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    observed: list[str] = []

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed.append(f"{request.full_url} timeout={timeout}")
        return FakeResponse(b"doc")

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    docs = client.download_agent_api_docs(
        "http://127.0.0.1:8088/",
        tmp_path,
        timeout=7.0,
    )

    assert observed == [
        "http://127.0.0.1:8088/agent-api timeout=7.0",
        "http://127.0.0.1:8088/agent-api.json timeout=7.0",
        "http://127.0.0.1:8088/agent/capabilities timeout=7.0",
        "http://127.0.0.1:8088/agent/tool-manifest timeout=7.0",
        "http://127.0.0.1:8088/openapi.json timeout=7.0",
    ]
    assert docs == {
        "agent_api_html": str(tmp_path / "agent-api.html"),
        "agent_api_json": str(tmp_path / "agent-api.json"),
        "agent_capabilities": str(tmp_path / "agent-capabilities.json"),
        "agent_tool_manifest": str(tmp_path / "agent-tool-manifest.json"),
        "openapi_json": str(tmp_path / "openapi.json"),
    }
    assert (tmp_path / "agent-api.html").read_bytes() == b"doc"


def test_snapshot_scene_posts_encoded_session_payload(monkeypatch: Any) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        observed["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return FakeResponse(b'{"session_id":"session/one"}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    result = client.snapshot_scene(
        "http://127.0.0.1:8088/",
        "session/one",
        {"include_properties": False},
        timeout=5.0,
    )

    assert result == {"session_id": "session/one"}
    assert observed == {
        "url": "http://127.0.0.1:8088/sessions/session%2Fone/scene/snapshot",
        "method": "POST",
        "timeout": 5.0,
        "body": {"include_properties": False},
    }


def test_restore_scene_posts_to_canonical_endpoint(monkeypatch: Any) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        observed["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return FakeResponse(b'{"status":"restored"}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    result = client.restore_scene(
        "http://127.0.0.1:8088/",
        "session/one",
        {"output_usd_path": "/tmp/out.usda", "output_mode": "layer"},
        timeout=9.0,
    )

    assert result == {"status": "restored"}
    assert observed == {
        "url": "http://127.0.0.1:8088/sessions/session%2Fone/scene/restore",
        "method": "POST",
        "timeout": 9.0,
        "body": {"output_usd_path": "/tmp/out.usda", "output_mode": "layer"},
    }


def test_render_view_requires_image_url(tmp_path: Path, monkeypatch: Any) -> None:
    observed_urls: list[str] = []

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed_urls.append(request.full_url)
        return FakeResponse(b"{}")

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    try:
        client.render_view(
            workbench_url="http://127.0.0.1:8088",
            session_id="session/one",
            output_dir=tmp_path,
            name="initial_top",
            direction="+z",
            width=64,
            height=64,
            render_quality="inspection",
        )
    except RuntimeError as exc:
        assert "missing image_url" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected RuntimeError")

    assert observed_urls == ["http://127.0.0.1:8088/sessions/session%2Fone/render"]


def test_render_view_ignores_non_string_camera_url(
    tmp_path: Path, monkeypatch: Any
) -> None:
    downloaded: list[str] = []

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        if request.get_method() == "POST":
            return FakeResponse(
                b'{"image_url": "artifacts/render.png", "camera_json_url": true}'
            )
        downloaded.append(request.full_url)
        return FakeResponse(b"fake image")

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    result = client.render_view(
        workbench_url="http://127.0.0.1:8088/",
        session_id="session/one",
        output_dir=tmp_path,
        name="initial_top",
        direction="+z",
        width=64,
        height=64,
        render_quality="inspection",
    )

    assert downloaded == ["http://127.0.0.1:8088/artifacts/render.png"]
    assert result["camera_json_path"] is None
    assert result["artifact_download_count"] == 1


def test_render_frames_posts_to_generic_endpoint(monkeypatch: Any) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return FakeResponse(b'{"frame_urls":["/a.png"]}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    result = client.render_frames(
        "http://127.0.0.1:8088",
        "session-1",
        {
            "scene_path": "/tmp/recording.usda",
            "frames": "0:1",
            "camera_path": "+x+y+z",
        },
    )

    assert result == {"frame_urls": ["/a.png"]}
    assert observed == {
        "url": "http://127.0.0.1:8088/sessions/session-1/render-frames",
        "body": {
            "scene_path": "/tmp/recording.usda",
            "frames": "0:1",
            "camera_path": "+x+y+z",
        },
    }


def test_translate_paths_posts_batch_payload(monkeypatch: Any) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["body"] = json.loads((request.data or b"{}").decode("utf-8"))
        return FakeResponse(b'{"results":[]}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    result = client.translate_paths(
        "http://127.0.0.1:8088",
        "session-1",
        [
            {
                "prim_path": "/World/A",
                "source_space": "inspection",
                "target_space": "source",
            }
        ],
    )

    assert result == {"results": []}
    assert observed == {
        "url": "http://127.0.0.1:8088/sessions/session-1/paths/translate:batch",
        "body": {
            "requests": [
                {
                    "prim_path": "/World/A",
                    "source_space": "inspection",
                    "target_space": "source",
                }
            ]
        },
    }


def test_request_json_sanitizes_absolute_paths(monkeypatch: Any) -> None:
    def fake_urlopen(_request: Request, *, timeout: float) -> FakeResponse:
        raise OSError("failed at /home/user/private/asset.usd")

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    try:
        client.get_json("http://127.0.0.1:8088/fail")
    except RuntimeError as exc:
        assert "/home/user/private" not in str(exc)
        assert "<path>" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected RuntimeError")


def test_is_healthy_requires_content_workbench_service(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            b'{"status":"healthy","service":"not-content-workbench"}'
        ),
    )

    assert not client.is_healthy("http://127.0.0.1:8088")


def test_is_healthy_accepts_content_workbench_service(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            b'{"status":"healthy","service":"content-workbench"}'
        ),
    )

    assert client.is_healthy("http://127.0.0.1:8088")


def test_is_healthy_requires_exact_output_root(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        client,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            json.dumps(
                {
                    "status": "healthy",
                    "service": "content-workbench",
                    "output_roots": [str(tmp_path.resolve())],
                }
            ).encode()
        ),
    )

    assert not client.is_healthy(
        "http://127.0.0.1:8088",
        output_root=run_dir,
    )


def test_wait_until_healthy_reports_connection_error(monkeypatch: Any) -> None:
    monotonic_values = iter([0.0, 0.0, 1.0])

    def fake_urlopen(_request: Request, *, timeout: float) -> FakeResponse:
        raise OSError("connection refused")

    monkeypatch.setattr(client, "urlopen", fake_urlopen)
    monkeypatch.setattr(client.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(client.time, "sleep", lambda _seconds: None)

    try:
        client.wait_until_healthy("http://127.0.0.1:8088", timeout_seconds=0.5)
    except RuntimeError as exc:
        assert "Content Workbench endpoint is not healthy" in str(exc)
        assert "connection refused" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected RuntimeError")


def test_wait_until_healthy_requires_exact_output_root(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "run"
    broader_root = tmp_path
    response = FakeResponse(
        json.dumps(
            {
                "status": "healthy",
                "service": "content-workbench",
                "output_roots": [str(broader_root.resolve())],
            }
        ).encode()
    )
    monkeypatch.setattr(client, "urlopen", lambda *_args, **_kwargs: response)

    try:
        client.wait_until_healthy(
            "http://127.0.0.1:8088",
            timeout_seconds=60.0,
            output_root=run_dir,
        )
    except RuntimeError as exc:
        assert "run-scoped output root" in str(exc)
        assert str(run_dir.resolve()) in str(exc)
        assert str(broader_root.resolve()) in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected mismatched output root to fail")

    matching_response = FakeResponse(
        json.dumps(
            {
                "status": "healthy",
                "service": "content-workbench",
                "output_roots": [str(run_dir.resolve())],
            }
        ).encode()
    )
    monkeypatch.setattr(
        client,
        "urlopen",
        lambda *_args, **_kwargs: matching_response,
    )
    client.wait_until_healthy(
        "http://127.0.0.1:8088",
        timeout_seconds=60.0,
        output_root=run_dir,
    )


def test_close_session_encodes_session_id(monkeypatch: Any) -> None:
    observed: dict[str, object] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        return FakeResponse(b"{}")

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    client.close_session("http://127.0.0.1:8088", "session/one", timeout=14.0)

    assert observed == {
        "url": "http://127.0.0.1:8088/sessions/session%2Fone",
        "method": "DELETE",
        "timeout": 14.0,
    }


def test_physics_client_helpers_use_canonical_session_endpoints(
    monkeypatch: Any,
) -> None:
    observed: list[tuple[str, object]] = []

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        observed.append(
            (
                request.full_url,
                json.loads((request.data or b"{}").decode("utf-8")),
            )
        )
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    assert client.inspect_physics_candidates(
        "http://127.0.0.1:8088",
        "session/one",
        {"usd_path": "/tmp/in.usda"},
    ) == {"ok": True}
    assert client.inspect_physics_components(
        "http://127.0.0.1:8088",
        "session/one",
        {"usd_path": "/tmp/in.usda"},
    ) == {"ok": True}
    assert client.inspect_physics_topology(
        "http://127.0.0.1:8088",
        "session/one",
        {"usd_path": "/tmp/in.usda"},
    ) == {"ok": True}
    assert client.apply_physics_topology_plan(
        "http://127.0.0.1:8088",
        "session/one",
        {
            "input_usd_path": "/tmp/in.usda",
            "expected_source_digest": "sha256:test",
            "operations": [],
            "invariants": {
                "enabled_collider_count": 0,
                "reject_articulation_changes": True,
            },
        },
    ) == {"ok": True}
    assert client.apply_physics_schema(
        "http://127.0.0.1:8088",
        "session/one",
        {"predictions_jsonl_path": "/tmp/predictions.jsonl"},
    ) == {"ok": True}
    assert client.validate_physics_runtime(
        "http://127.0.0.1:8088",
        "session/one",
        {"physics_usd_path": "/tmp/physics.usda", "engine": "ovphysx"},
    ) == {"ok": True}
    assert observed == [
        (
            "http://127.0.0.1:8088/sessions/session%2Fone/physics/inspect-mesh-candidates",
            {"usd_path": "/tmp/in.usda"},
        ),
        (
            "http://127.0.0.1:8088/sessions/session%2Fone/physics/inspect-components",
            {"usd_path": "/tmp/in.usda"},
        ),
        (
            "http://127.0.0.1:8088/sessions/session%2Fone/physics/inspect-topology",
            {"usd_path": "/tmp/in.usda"},
        ),
        (
            "http://127.0.0.1:8088/sessions/session%2Fone/physics/apply-topology-plan",
            {
                "input_usd_path": "/tmp/in.usda",
                "expected_source_digest": "sha256:test",
                "operations": [],
                "invariants": {
                    "enabled_collider_count": 0,
                    "reject_articulation_changes": True,
                },
            },
        ),
        (
            "http://127.0.0.1:8088/sessions/session%2Fone/physics/apply-schema",
            {"predictions_jsonl_path": "/tmp/predictions.jsonl"},
        ),
        (
            "http://127.0.0.1:8088/sessions/session%2Fone/physics/validate-runtime",
            {"physics_usd_path": "/tmp/physics.usda", "engine": "ovphysx"},
        ),
    ]
