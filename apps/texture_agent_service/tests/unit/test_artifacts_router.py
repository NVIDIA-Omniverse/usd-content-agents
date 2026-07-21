# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import queue
import threading
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...service.routers import artifacts_router
from ...service.session.manager import SessionManager
from ...service.storage import METADATA_KEY, LocalSessionStore


def _build_artifact_app(tmp_path: Path) -> tuple[TestClient, str]:
    sid = "artifact-session"
    manager = SessionManager(tmp_path)
    session_dir = manager.create_session(sid)

    (session_dir / "cache" / "discovery" / "materials.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (session_dir / "cache" / "artifacts_manifest.json").write_text(
        '{"schema_version":"texture-agent-artifacts.v1"}',
        encoding="utf-8",
    )
    (session_dir / "cache" / "textures" / "albedo.png").write_bytes(b"texture")
    (session_dir / "cache" / "output" / "textured_output.usdz").write_bytes(
        b"usdz",
    )
    (session_dir / "cache" / "renders" / "final.png").write_bytes(b"render")
    (session_dir / "preview" / "preview.png").write_bytes(b"preview")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    return TestClient(app), sid


def test_artifact_endpoints_return_documented_media_types(tmp_path: Path) -> None:
    client, sid = _build_artifact_app(tmp_path)

    expected = {
        f"/artifacts/{sid}/materials": "application/json",
        f"/artifacts/{sid}/manifest": "application/json",
        f"/artifacts/{sid}/textures": "application/zip",
        f"/artifacts/{sid}/textures/albedo.png": "image/png",
        f"/artifacts/{sid}/output": "model/vnd.usdz+zip",
        f"/artifacts/{sid}/renders": "application/zip",
        f"/artifacts/{sid}/renders/final.png": "image/png",
        f"/artifacts/{sid}/preview/preview.png": "image/png",
    }

    for path, media_type in expected.items():
        response = client.get(path)

        assert response.status_code == 200
        assert response.headers["content-type"] == media_type


def test_missing_artifacts_return_json_errors(tmp_path: Path) -> None:
    client, sid = _build_artifact_app(tmp_path)

    expected = {
        f"/artifacts/{sid}-missing/materials": (404, "Session not found"),
        f"/artifacts/{sid}-missing/manifest": (404, "Session not found"),
        f"/artifacts/{sid}-missing/textures/albedo.png": (404, "Session not found"),
        f"/artifacts/{sid}-missing/renders": (404, "Session not found"),
        f"/artifacts/{sid}/textures/missing.png": (
            404,
            "Texture file not found: missing.png",
        ),
        f"/artifacts/{sid}/renders/missing.png": (
            404,
            "Render file not found: missing.png",
        ),
        f"/artifacts/{sid}/preview/missing.png": (
            404,
            "Preview file not found: missing.png",
        ),
        f"/artifacts/{sid}/textures/%5Cescape.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/renders/%5Cescape.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/preview/%5Cescape.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/textures/bad%0D%0Aname.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/renders/bad%0D%0Aname.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/preview/bad%0D%0Aname.png": (400, "Invalid filename"),
    }

    for path, (status_code, detail) in expected.items():
        response = client.get(path)

        assert response.status_code == status_code
        assert response.headers["content-type"] == "application/json"
        assert response.json()["detail"] == detail


def test_invalid_session_id_artifact_route_returns_not_found(tmp_path: Path) -> None:
    client, _ = _build_artifact_app(tmp_path)

    response = client.get("/artifacts/%2E%2E/materials")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == "Session not found"
    assert tmp_path.exists()


def test_unavailable_artifact_collections_return_json_errors(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    sid = "empty-artifact-session"
    session_dir = manager.create_session(sid)

    (session_dir / "cache" / "textures").rmdir()
    (session_dir / "cache" / "renders").rmdir()

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    expected = {
        f"/artifacts/{sid}/materials": "Materials data not available",
        f"/artifacts/{sid}/manifest": "Artifact manifest not available",
        f"/artifacts/{sid}/textures": "Textures not available",
        f"/artifacts/{sid}/textures/albedo.png": "Textures not available",
        f"/artifacts/{sid}/output": "Output USDZ not available",
        f"/artifacts/{sid}/renders": "Renders not available",
        f"/artifacts/{sid}/renders/final.png": "Renders not available",
    }

    for path, detail in expected.items():
        response = client.get(path)

        assert response.status_code == 404
        assert response.headers["content-type"] == "application/json"
        assert response.json()["detail"] == detail


def test_artifact_openapi_documents_binary_media_types(tmp_path: Path) -> None:
    client, _ = _build_artifact_app(tmp_path)
    paths = client.app.openapi()["paths"]

    expected = {
        "/artifacts/{session_id}/textures": "application/zip",
        "/artifacts/{session_id}/textures/{filename}": "image/png",
        "/artifacts/{session_id}/output": "model/vnd.usdz+zip",
        "/artifacts/{session_id}/renders": "application/zip",
        "/artifacts/{session_id}/renders/{filename}": "image/png",
        "/artifacts/{session_id}/preview/{filename}": "image/png",
    }

    for path, media_type in expected.items():
        content = paths[path]["get"]["responses"]["200"]["content"]

        assert list(content) == [media_type]
        assert content[media_type]["schema"] == {
            "type": "string",
            "format": "binary",
        }


def test_shared_store_artifact_uses_public_url_when_available(tmp_path: Path) -> None:
    class PublicUrlStore(LocalSessionStore):
        def make_public_url(
            self,
            session_id: str,
            key: str,
            expires_seconds: int = 3600,
        ) -> str | None:
            return f"https://example.test/{session_id}/{key}"

    store = PublicUrlStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-artifact-url"
    manager.create_session(sid)
    store.put_bytes(
        sid,
        "cache/output/textured_output.usdz",
        b"usdz",
        content_type="model/vnd.usdz+zip",
    )

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    response = client.get(f"/artifacts/{sid}/output", follow_redirects=False)

    assert response.status_code == 307
    assert (
        response.headers["location"]
        == f"https://example.test/{sid}/cache/output/textured_output.usdz"
    )


def test_shared_store_json_artifacts_are_proxied_for_sanitization(
    tmp_path: Path,
) -> None:
    class PublicUrlStore(LocalSessionStore):
        def make_public_url(
            self,
            session_id: str,
            key: str,
            expires_seconds: int = 3600,
        ) -> str | None:
            return f"https://example.test/{session_id}/{key}"

    store = PublicUrlStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-json-proxy"
    manager.create_session(sid)
    payloads = {
        "cache/discovery/materials.json": b'[{"path":"/var/texture-agent/sessions/id/mat"}]',
        "cache/artifacts_manifest.json": b'{"path":"/var/texture-agent/sessions/id/out"}',
    }
    for key, payload in payloads.items():
        store.put_bytes(sid, key, payload, content_type="application/json")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    for endpoint, expected in (
        ("materials", payloads["cache/discovery/materials.json"]),
        ("manifest", payloads["cache/artifacts_manifest.json"]),
    ):
        response = client.get(f"/artifacts/{sid}/{endpoint}", follow_redirects=False)

        assert response.status_code == 200
        assert response.content == expected
        assert "location" not in response.headers


def test_shared_store_texture_zip_streams_from_store(tmp_path: Path) -> None:
    store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-texture-zip"
    manager.create_session(sid)
    store.put_bytes(sid, "cache/textures/albedo.png", b"albedo")
    store.put_bytes(sid, "cache/textures/normal.png", b"normal")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    response = client.get(f"/artifacts/{sid}/textures")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert sorted(zf.namelist()) == [
            "textures/albedo.png",
            "textures/normal.png",
        ]
        assert zf.read("textures/albedo.png") == b"albedo"
        assert zf.read("textures/normal.png") == b"normal"


def test_shared_store_single_texture_streams_without_local_hydration(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-single-texture"
    store.init_session(sid)
    store.put_json(sid, METADATA_KEY, {"session_id": sid, "status": "completed"})
    store.put_bytes(sid, "cache/textures/albedo.png", b"albedo", "image/png")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    response = client.get(f"/artifacts/{sid}/textures/albedo.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"albedo"
    assert not manager.get_session_dir(sid).exists()


def test_shared_store_texture_zip_closes_store_stream_on_disconnect() -> None:
    class ChunkStream:
        def __init__(self) -> None:
            self.remaining = 4096
            self.closed = False

        def read(self, size: int = -1) -> bytes:
            if self.remaining <= 0:
                return b""
            self.remaining -= 1
            return b"x" * min(size, 4096)

        def close(self) -> None:
            self.closed = True

    class StreamingManager:
        def __init__(self, stream: ChunkStream) -> None:
            self.stream = stream

        def open_store_stream(self, session_id: str, key: str) -> ChunkStream | None:
            return self.stream

    stream = ChunkStream()
    generator = cast(
        Generator[bytes, None, None],
        artifacts_router._iter_zip_store_keys(
            cast(SessionManager, StreamingManager(stream)),
            "session",
            ["cache/textures/albedo.png"],
            prefix="textures",
        ),
    )

    assert next(generator)
    generator.close()

    assert stream.closed is True


def test_public_url_artifact_redirect_preserves_missing_json_404(
    tmp_path: Path,
) -> None:
    class PublicUrlStore(LocalSessionStore):
        def make_public_url(
            self,
            session_id: str,
            key: str,
            expires_seconds: int = 3600,
        ) -> str | None:
            return f"https://example.test/{session_id}/{key}"

    store = PublicUrlStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-artifact-no-head"
    manager.create_session(sid)

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    response = client.get(f"/artifacts/{sid}/output", follow_redirects=False)

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == "Output USDZ not available"


def test_zip_queue_writer_basic_contract() -> None:
    import queue
    import threading

    output_queue: queue.Queue[bytes | None] = queue.Queue()
    writer = artifacts_router._ZipQueueWriter(output_queue, threading.Event())

    assert writer.writable() is True
    assert writer.seekable() is False
    assert writer.tell() == 0
    assert writer.write(b"abc") == 3
    assert writer.tell() == 3
    assert output_queue.get_nowait() == b"abc"
    assert writer.write(b"") == 0


def test_zip_queue_writer_retries_full_queue() -> None:
    class RetryQueue:
        def __init__(self) -> None:
            self.calls = 0
            self.items: list[bytes] = []

        def put(self, item: bytes, timeout: float) -> None:
            self.calls += 1
            if self.calls == 1:
                raise queue.Full
            self.items.append(item)

    retry_queue = RetryQueue()
    writer = artifacts_router._ZipQueueWriter(
        cast(queue.Queue[bytes | None], retry_queue),
        threading.Event(),
    )

    assert writer.write(b"abc") == 3
    assert retry_queue.calls == 2
    assert retry_queue.items == [b"abc"]


def test_empty_local_artifact_dirs_return_specific_json_errors(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    sid = "empty-local-artifacts"
    session_dir = manager.create_session(sid)

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    expected = {
        f"/artifacts/{sid}/textures": "No texture files found",
        f"/artifacts/{sid}/renders": "No render files found",
        "/artifacts/missing-session/renders/final.png": "Session not found",
        "/artifacts/missing-session/preview/final.png": "Session not found",
    }

    for path, detail in expected.items():
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["detail"] == detail

    (session_dir / "cache" / "textures" / "folder.png").mkdir()
    response = client.get(f"/artifacts/{sid}/textures/folder.png")
    assert response.status_code == 404
    assert response.json()["detail"] == "Texture file not found: folder.png"


def test_artifact_downloads_reject_symlink_escapes(tmp_path: Path) -> None:
    client, sid = _build_artifact_app(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    session_dir = tmp_path / sid

    for folder, route in (
        ("cache/textures", "textures"),
        ("cache/renders", "renders"),
        ("preview", "preview"),
    ):
        link = session_dir / folder / "escape.png"
        link.unlink(missing_ok=True)
        link.symlink_to(outside)

        response = client.get(f"/artifacts/{sid}/{route}/escape.png")

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid filename"


def test_shared_store_renders_zip_streams_when_local_dir_empty(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-renders-empty-local"
    manager.create_session(sid)
    store.put_bytes(sid, "cache/renders/final.png", b"render")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    response = client.get(f"/artifacts/{sid}/renders")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert zf.read("renders/final.png") == b"render"


def test_shared_store_manifest_render_and_preview_stream_without_hydration(
    tmp_path: Path,
) -> None:
    store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", store=store)
    sid = "shared-artifact-streams"
    store.init_session(sid)
    store.put_json(sid, METADATA_KEY, {"session_id": sid, "status": "completed"})
    store.put_bytes(sid, "cache/artifacts_manifest.json", b'{"ok": true}')
    store.put_bytes(sid, "cache/renders/final.png", b"render")
    store.put_bytes(sid, "preview/final.png", b"preview")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    manifest = client.get(f"/artifacts/{sid}/manifest")
    assert manifest.status_code == 200
    assert manifest.content == b'{"ok": true}'

    render = client.get(f"/artifacts/{sid}/renders/final.png")
    assert render.status_code == 200
    assert render.content == b"render"

    preview = client.get(f"/artifacts/{sid}/preview/final.png")
    assert preview.status_code == 200
    assert preview.content == b"preview"
    assert not manager.get_session_dir(sid).exists()
