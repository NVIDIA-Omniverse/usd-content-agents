# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional branch coverage for service API routers."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from world_understanding.utils.artifacts import (
    OpenArtifactFile,
    is_pipeline_temp_path,
    open_regular_file_no_follow,
)

from ...service.routers import (
    artifacts_router,
    assets_router,
    materials_router,
    sessions_router,
)
from ...service.session.manager import SessionManager
from ...service.storage.local_store import LocalSessionStore

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
    b"\xf6\x178U"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write(path: Path, data: bytes | str = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data)
    else:
        path.write_bytes(data)
    return path


async def _session(
    tmp_path: Path, *routers: object
) -> tuple[SessionManager, str, Path]:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    for router in routers:
        router.set_session_manager(manager)
    return manager, session_id, session_dir


def _expect_http(
    status_code: int, exc_info: pytest.ExceptionInfo[HTTPException]
) -> None:
    assert exc_info.value.status_code == status_code


@dataclass
class _MaterialLibrary:
    id: str
    name: str
    entries: list[dict[str, str]]
    icons: dict[str, str]
    base_dir: str


class _MaterialsConfig:
    def __init__(self, default_library_id: str, libraries: dict[str, _MaterialLibrary]):
        self.default_library_id = default_library_id
        self.material_libraries = libraries
        self.materials = [
            {"name": "Fallback", "description": "fallback", "binding": "/Fallback"}
        ]

    def get_library(self, library_id: str) -> _MaterialLibrary | None:
        return self.material_libraries.get(library_id)


class _RemoteKindStore(LocalSessionStore):
    @property
    def kind(self) -> str:
        return "remote-test"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_materials_router_library_and_icon_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    icon = _write(tmp_path / "default" / "icons" / "aluminum.png", PNG_BYTES)
    direct_icon = _write(tmp_path / "default" / "icons" / "direct.png", PNG_BYTES)
    custom_icon = _write(tmp_path / "custom" / "icons" / "steel.png", PNG_BYTES)
    default = _MaterialLibrary(
        id="default",
        name="Default",
        entries=[
            {
                "name": "Aluminum",
                "description": "metal",
                "binding": "/Aluminum",
                "icon": "icons/aluminum.png",
            },
            {"name": "No Icon", "description": "", "binding": "/NoIcon"},
        ],
        icons={"Aluminum": str(icon.relative_to(tmp_path / "default"))},
        base_dir=str(tmp_path / "default"),
    )
    custom = _MaterialLibrary(
        id="custom",
        name="Custom",
        entries=[
            {
                "name": "Steel",
                "description": "alloy",
                "binding": "/Steel",
                "icon": "icons/steel.png",
            }
        ],
        icons={"Steel": str(custom_icon.relative_to(tmp_path / "custom"))},
        base_dir=str(tmp_path / "custom"),
    )
    monkeypatch.setattr(
        materials_router,
        "config",
        _MaterialsConfig("default", {"custom": custom, "default": default}),
    )

    libraries = await materials_router.list_libraries()
    assert [lib["id"] for lib in libraries["libraries"]] == ["default", "custom"]

    payload = await materials_router.get_library_materials("default")
    assert payload["materials"][0]["name"] == "Aluminum"
    assert payload["materials"][1]["icon_url"] is None

    assert isinstance(
        await materials_router.get_library_material_icon("default", "Aluminum"),
        FileResponse,
    )
    assert isinstance(
        await materials_router.get_library_material_icon(
            "default", str(direct_icon.relative_to(tmp_path / "default"))
        ),
        FileResponse,
    )
    assert isinstance(
        await materials_router.get_material_icon("Aluminum"), FileResponse
    )

    with pytest.raises(HTTPException) as exc_info:
        await materials_router.get_library_materials("missing")
    _expect_http(404, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await materials_router.get_library_material_icon("default", "../secret.png")
    _expect_http(403, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await materials_router.get_library_material_icon("default", "missing.png")
    _expect_http(404, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await materials_router.get_library_material_icon("missing", "Aluminum")
    _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_materials_router_fallback_list_and_template_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(materials_router, "config", _MaterialsConfig("absent", {}))
    materials = await materials_router.list_materials()
    assert materials["materials"] == [
        {
            "name": "Fallback",
            "description": "fallback",
            "binding": "/Fallback",
            "icon_url": None,
            "icon_path": None,
        }
    ]

    fake_module = tmp_path / "pkg" / "service" / "routers" / "materials_router.py"
    monkeypatch.setattr(materials_router, "__file__", str(fake_module))
    with pytest.raises(HTTPException) as exc_info:
        await materials_router.download_materials_template()
    _expect_http(404, exc_info)

    template = tmp_path / "pkg" / "materials" / "default" / "default_materials.zip"
    _write(template, b"zip")
    assert isinstance(
        await materials_router.download_materials_template(), FileResponse
    )


class _FallbackManager:
    class _Store:
        kind = "local"

    def __init__(
        self,
        *,
        public_url: str | None = None,
        data: bytes | None = None,
    ) -> None:
        self.public_url = public_url
        self.data = data
        self.store = self._Store()

    async def get_session_metadata(self, session_id: str) -> None:
        return None

    async def get_session_metadata_versioned(self, session_id: str):
        return types.SimpleNamespace(value={}, version="fixture-v1")

    @staticmethod
    def resolve_published_artifact_key(
        metadata: dict,
        logical_name: str,
        *,
        legacy_key: str | None = None,
    ) -> str | None:
        return legacy_key

    @staticmethod
    def resolve_prediction_report_key(
        metadata: dict,
        *,
        legacy_key: str | None = None,
    ) -> str | None:
        return legacy_key

    async def make_public_url(self, session_id: str, key: str) -> str | None:
        return self.public_url

    async def read_from_store(self, session_id: str, key: str) -> bytes | None:
        return self.data

    async def iter_store_chunks(
        self,
        session_id: str,
        key: str,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes] | None:
        if self.data is None:
            return None

        async def chunks() -> AsyncIterator[bytes]:
            for offset in range(0, len(self.data), chunk_size):
                yield self.data[offset : offset + chunk_size]

        return chunks()

    async def open_local_artifact(
        self,
        _session_id: str,
        local_path: Path,
    ) -> OpenArtifactFile | None:
        if is_pipeline_temp_path(local_path):
            return None
        try:
            with open_regular_file_no_follow(local_path) as (source, metadata):
                descriptor = os.dup(source.fileno())
        except (OSError, RuntimeError, ValueError):
            return None
        return OpenArtifactFile(
            relative_key=str(local_path),
            stream=os.fdopen(descriptor, "rb"),
            metadata=metadata,
        )


class _VersionedFallbackManager(_FallbackManager):
    def __init__(
        self,
        metadata: dict | None,
        *,
        data: bytes | None = None,
        public_url: str | None = None,
        resolved_key: str | None = "logical.bin",
        followup_metadata: dict | None = None,
    ) -> None:
        super().__init__(public_url=public_url, data=data)
        self.metadata = metadata
        self.followup_metadata = followup_metadata
        self.resolved_key = resolved_key
        self.version_calls = 0

    async def get_session_metadata_versioned(self, session_id: str):
        self.version_calls += 1
        value = self.metadata
        if self.version_calls > 1:
            value = self.followup_metadata
        return types.SimpleNamespace(
            value=value,
            version="version" if value is not None else None,
        )

    def resolve_published_artifact_key(
        self,
        metadata: dict,
        logical_name: str,
        *,
        legacy_key: str | None = None,
    ) -> str | None:
        return self.resolved_key

    def resolve_prediction_report_key(
        self,
        metadata: dict,
        *,
        legacy_key: str | None = None,
    ) -> str | None:
        return self.resolved_key


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assets_file_fallback_ladder(tmp_path: Path) -> None:
    assert isinstance(
        await assets_router._serve_file_with_fallback(
            _FallbackManager(public_url="https://example.test/file"),
            "sid",
            "key.png",
            tmp_path / "key.png",
        ),
        RedirectResponse,
    )

    stored = await assets_router._serve_file_with_fallback(
        _FallbackManager(data=b"stored"),
        "sid",
        "key.bin",
        tmp_path / "key.bin",
        filename="key.bin",
    )
    assert isinstance(stored, Response)
    assert stored.headers["content-disposition"] == 'attachment; filename="key.bin"'

    local = _write(tmp_path / "local.unknown", b"local")
    assert isinstance(
        await assets_router._serve_file_with_fallback(
            _FallbackManager(), "sid", "local.unknown", local
        ),
        FileResponse,
    )
    assert (
        await assets_router._serve_file_with_fallback(
            _FallbackManager(), "sid", "missing.png", tmp_path / "missing.png"
        )
        is None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_versioned_file_fallback_guard_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = _VersionedFallbackManager(None)
    assert (
        await artifacts_router._try_serve_file_with_fallback(
            missing,
            "sid",
            "logical.bin",
            tmp_path / "missing.bin",
        )
        is None
    )
    assert (
        await assets_router._serve_file_with_fallback(
            missing,
            "sid",
            "cache/preview/missing.png",
            tmp_path / "missing.png",
        )
        is None
    )

    invalid = _VersionedFallbackManager(
        {"artifact_validity": {"raw_predictions": False}}
    )
    assert (
        await artifacts_router._try_serve_file_with_fallback(
            invalid,
            "sid",
            "logical.bin",
            tmp_path / "invalid.bin",
            artifact="raw_predictions",
        )
        is None
    )

    unresolved = _VersionedFallbackManager({}, resolved_key=None)
    assert (
        await artifacts_router._try_serve_file_with_fallback(
            unresolved,
            "sid",
            "logical.bin",
            tmp_path / "unresolved.bin",
        )
        is None
    )
    unresolved_preview = _VersionedFallbackManager({}, resolved_key=None)
    assert (
        await assets_router._serve_file_with_fallback(
            unresolved_preview,
            "sid",
            "cache/preview/unresolved.png",
            tmp_path / "unresolved.png",
        )
        is None
    )

    stale_artifact = _VersionedFallbackManager(
        {"artifact_validity": {"raw_predictions": True}},
        data=b"stale",
        followup_metadata=None,
    )
    assert (
        await artifacts_router._try_serve_file_with_fallback(
            stale_artifact,
            "sid",
            "logical.bin",
            tmp_path / "stale.bin",
            artifact="raw_predictions",
        )
        is None
    )
    stale_preview = _VersionedFallbackManager(
        {"artifact_validity": {"previews": True}},
        data=b"stale",
        followup_metadata=None,
    )
    assert (
        await assets_router._serve_file_with_fallback(
            stale_preview,
            "sid",
            "cache/preview/stale.png",
            tmp_path / "stale.png",
        )
        is None
    )

    immutable = _VersionedFallbackManager(
        {},
        public_url="https://example.test/immutable",
    )
    assert isinstance(
        await artifacts_router._try_serve_file_with_fallback(
            immutable,
            "sid",
            "runs/1-token/output.usd",
            tmp_path / "output.usd",
        ),
        RedirectResponse,
    )

    artifact_path = _write(tmp_path / "local-artifact.bin", b"artifact")
    current_artifact = _VersionedFallbackManager(
        {"artifact_validity": {"raw_predictions": True}},
        followup_metadata={"artifact_validity": {"raw_predictions": True}},
    )
    response = await artifacts_router._try_serve_file_with_fallback(
        current_artifact,
        "sid",
        "logical.bin",
        artifact_path,
        filename="artifact.bin",
        artifact="raw_predictions",
    )
    assert isinstance(response, Response)
    assert response.headers["content-disposition"] == (
        'attachment; filename="artifact.bin"'
    )

    preview_path = _write(tmp_path / "local-preview.png", b"preview")
    current_preview = _VersionedFallbackManager(
        {"artifact_validity": {"previews": True}},
        followup_metadata={"artifact_validity": {"previews": True}},
        resolved_key="cache/preview/local-preview.png",
    )
    response = await assets_router._serve_file_with_fallback(
        current_preview,
        "sid",
        "cache/preview/local-preview.png",
        preview_path,
        filename="preview.png",
    )
    assert isinstance(response, Response)
    assert response.headers["content-disposition"] == (
        'attachment; filename="preview.png"'
    )

    original_open_local_artifact = _FallbackManager.open_local_artifact

    async def fail_selected_open(
        manager: _FallbackManager,
        session_id: str,
        path: Path,
    ) -> OpenArtifactFile | None:
        if path in {artifact_path, preview_path}:
            return None
        return await original_open_local_artifact(manager, session_id, path)

    monkeypatch.setattr(_FallbackManager, "open_local_artifact", fail_selected_open)
    for helper, manager, key, path, artifact in (
        (
            artifacts_router._try_serve_file_with_fallback,
            _VersionedFallbackManager({"artifact_validity": {"raw_predictions": True}}),
            "logical.bin",
            artifact_path,
            "raw_predictions",
        ),
        (
            assets_router._serve_file_with_fallback,
            _VersionedFallbackManager(
                {"artifact_validity": {"previews": True}},
                resolved_key="cache/preview/local-preview.png",
            ),
            "cache/preview/local-preview.png",
            preview_path,
            None,
        ),
    ):
        kwargs = {"artifact": artifact} if artifact is not None else {}
        assert await helper(manager, "sid", key, path, **kwargs) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assets_generated_reference_paths(tmp_path: Path) -> None:
    manager, session_id, session_dir = await _session(tmp_path, assets_router)

    assert assets_router._get_generated_reference_entry(None) is None
    assert assets_router._get_generated_reference_entry(
        {"generated_reference_images": [{"id": "first"}, {"id": "second"}]}
    ) == {"id": "second"}
    assert (
        assets_router._get_generated_reference_entry(
            {"generated_reference_images": [{"id": "first"}]}, "missing"
        )
        is None
    )

    _write(session_dir / "input" / "generated_ref_0.png", PNG_BYTES)
    assert isinstance(await assets_router.get_generated_ref(session_id), Response)

    await manager.update_session(
        session_id,
        {"generated_reference_images": [{"id": "ref", "key": "input/ref.png"}]},
    )
    _write(session_dir / "input" / "ref.png", PNG_BYTES)
    assert isinstance(
        await assets_router.get_generated_ref_by_id(session_id, "ref"), Response
    )

    await manager.update_session(
        session_id,
        {"generated_reference_images": [{"id": "bad", "key": 42}]},
    )
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_generated_ref(session_id)
    _expect_http(404, exc_info)
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_generated_ref_by_id(session_id, "bad")
    _expect_http(404, exc_info)
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_generated_ref_by_id(session_id, "missing")
    _expect_http(404, exc_info)

    await manager.update_session(
        session_id,
        {"generated_reference_images": [{"id": "gone", "key": "input/gone.png"}]},
    )
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_generated_ref_by_id(session_id, "gone")
    _expect_http(404, exc_info)

    await manager.update_session(session_id, {"generated_reference_images": []})
    (session_dir / "input" / "generated_ref_0.png").unlink()
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_generated_ref(session_id)
    _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assets_input_render_states(tmp_path: Path) -> None:
    manager, session_id, session_dir = await _session(tmp_path, assets_router)
    _write(session_dir / "input" / "input_render.png", PNG_BYTES)
    assert isinstance(await assets_router.get_input_render(session_id), Response)

    failed = str(uuid4())
    await manager.create_session(failed)
    sentinel = "sentinel-preview-backend-secret"
    await manager.update_session(
        failed,
        {"preview_render_status": "failed", "preview_render_error": sentinel},
    )
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_input_render(failed)
    _expect_http(424, exc_info)
    assert exc_info.value.detail == "Input preview render failed"
    assert sentinel not in str(exc_info.value.detail)

    rendering = str(uuid4())
    rendering_dir = await manager.create_session(rendering)
    _write(rendering_dir / ".input_render_config.yaml", "rendering")
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_input_render(rendering)
    _expect_http(503, exc_info)

    missing = str(uuid4())
    await manager.create_session(missing)
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_input_render(missing)
    _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assets_preview_reference_pdf_and_page_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, assets_router)

    _write(session_dir / "cache" / "preview" / "new.png", PNG_BYTES)
    _write(session_dir / "preview" / "old.png", PNG_BYTES)
    await manager.update_session(session_id, {"preview_images": ["new.png"]})
    assert isinstance(
        await assets_router.get_preview_image(session_id, "new.png"), Response
    )
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_preview_image(session_id, "old.png")
    _expect_http(404, exc_info)
    await manager.update_session(
        session_id,
        {"preview_images": ["new.png", "old.png"]},
    )
    assert isinstance(
        await assets_router.get_preview_image(session_id, "old.png"), Response
    )
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_preview_image(session_id, "missing.png")
    _expect_http(404, exc_info)

    previews = await assets_router.list_preview_images(session_id)
    assert previews.total == 2

    _write(session_dir / "cache" / "preview" / "live.png", PNG_BYTES)
    await manager.update_session(
        session_id,
        {
            "preview_images": [],
            "artifact_validity": {"previews": False},
        },
    )

    class _LiveBus:
        async def get_fenced_snapshot(self, requested_session_id: str):
            assert requested_session_id == session_id
            return {
                "status": "running",
                "preview_images": ["live.png"],
                "updated_at": "2026-07-01T00:00:00+00:00",
            }

    monkeypatch.setattr(assets_router, "get_event_bus", lambda: _LiveBus())
    assert isinstance(
        await assets_router.get_preview_image(session_id, "live.png"), Response
    )
    previews = await assets_router.list_preview_images(session_id)
    assert [preview.name for preview in previews.previews] == ["live.png"]

    with pytest.raises(HTTPException) as exc_info:
        await assets_router.list_preview_images(str(uuid4()))
    _expect_http(404, exc_info)

    _write(session_dir / "input" / "reference_images" / "reference_1.png", PNG_BYTES)
    _write(session_dir / "input" / "reference_images" / "reference_2.jpg", PNG_BYTES)
    _write(session_dir / "input" / "reference_images" / "ignore.txt", "ignore")
    assert isinstance(
        await assets_router.get_reference_image(session_id, "reference_1.png"),
        Response,
    )
    references = await assets_router.list_reference_images(session_id)
    assert references["total"] == 2
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_reference_image(session_id, "missing.png")
    _expect_http(404, exc_info)

    _write(session_dir / "input" / "reference_pdfs" / "reference_0000.pdf", b"%PDF")
    assert isinstance(
        await assets_router.get_reference_pdf(session_id, "reference_0000.pdf"),
        Response,
    )
    assert (await assets_router.list_reference_pdfs(session_id))["total"] == 1
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_reference_pdf(session_id, "missing.pdf")
    _expect_http(404, exc_info)

    _write(session_dir / "cache" / "dataset" / "pdf_0" / "page.png", PNG_BYTES)
    _write(session_dir / "cache" / "dataset" / "pdf_text" / "page.png", PNG_BYTES)
    _write(session_dir / "cache" / "dataset" / "pdf_1", "not a dir")
    pages = await assets_router.list_rendered_pdf_pages(session_id)
    assert pages["total"] == 2
    assert isinstance(
        await assets_router.get_rendered_pdf_page(session_id, "pdf_0", "page.png"),
        Response,
    )
    for pdf_index, page_name in [("bad", "page.png"), ("pdf_0", "../page.png")]:
        with pytest.raises(HTTPException) as exc_info:
            await assets_router.get_rendered_pdf_page(session_id, pdf_index, page_name)
        _expect_http(400, exc_info)
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_rendered_pdf_page(session_id, "pdf_0", "missing.png")
    _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_preview_same_name_beats_prior_immutable_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, assets_router)
    await manager.update_session(session_id, {"status": "completed"})
    planned = await manager.get_session_metadata_versioned(session_id)
    assert planned.version is not None
    claim = await manager.claim_regeneration(
        session_id,
        expected_version=planned.version,
    )
    logical_key = "cache/preview/same.png"
    immutable_key = f"{claim.artifact_prefix}/{logical_key}"
    await manager.put_bytes_to_store(session_id, immutable_key, b"old-generation")
    assert await manager.finalize_regeneration_claim(
        session_id,
        claim,
        updates={
            "status": "completed",
            "preview_images": ["same.png"],
            "artifact_validity": {"previews": True},
        },
        artifact_map={logical_key: immutable_key},
    )
    _write(
        session_dir / "cache" / "preview" / "same.png",
        b"current-generation",
    )

    class _LiveBus:
        async def get_fenced_snapshot(self, requested_session_id: str):
            assert requested_session_id == session_id
            return {
                "status": "running",
                "preview_images": ["same.png"],
                "updated_at": "2026-07-01T00:00:00+00:00",
            }

    monkeypatch.setattr(assets_router, "get_event_bus", lambda: _LiveBus())

    response = await assets_router.get_preview_image(session_id, "same.png")

    assert isinstance(response, Response)
    assert response.body == b"current-generation"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_preview_read_and_confirmation_failures_fall_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, assets_router)
    cache_path = _write(
        session_dir / "cache" / "preview" / "live.png",
        b"unreadable-cache",
    )
    fallback_path = _write(
        session_dir / "preview" / "live.png",
        b"live-fallback",
    )
    await manager.update_session(
        session_id,
        {
            "preview_images": [],
            "artifact_validity": {"previews": False},
        },
    )

    class _LiveBus:
        async def get_fenced_snapshot(self, _session_id: str):
            return {
                "status": "running",
                "preview_images": ["live.png"],
                "updated_at": "2026-07-01T00:00:00+00:00",
            }

    monkeypatch.setattr(assets_router, "get_event_bus", lambda: _LiveBus())
    original_read_bytes = Path.read_bytes

    def fail_cache_read(path: Path) -> bytes:
        if path == cache_path:
            raise OSError("cache disappeared")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_cache_read)
    response = await assets_router.get_preview_image(session_id, "live.png")
    assert isinstance(response, Response)
    assert response.body == b"live-fallback"

    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)

    class _VanishingBus:
        def __init__(self) -> None:
            self.calls = 0

        async def get_fenced_snapshot(self, _session_id: str):
            self.calls += 1
            if self.calls == 1:
                return {
                    "status": "running",
                    "preview_images": ["live.png"],
                    "updated_at": "2026-07-01T00:00:00+00:00",
                }
            return None

    vanishing_bus = _VanishingBus()
    monkeypatch.setattr(assets_router, "get_event_bus", lambda: vanishing_bus)
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.get_preview_image(session_id, "live.png")
    _expect_http(404, exc_info)

    fallback_path.unlink()

    class _NoLiveBus:
        async def get_fenced_snapshot(self, _session_id: str):
            return None

    monkeypatch.setattr(assets_router, "get_event_bus", lambda: _NoLiveBus())
    with pytest.raises(HTTPException) as exc_info:
        await assets_router.list_preview_images(session_id)
    _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_artifact_routes_reject_stale_local_files(
    tmp_path: Path,
) -> None:
    store = _RemoteKindStore(str(tmp_path / "remote"))
    manager = SessionManager(tmp_path / "pod-local", store=store)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    artifacts_router.set_session_manager(manager)
    assets_router.set_session_manager(manager)
    _write(
        session_dir / "cache" / "predictions" / "predictions.jsonl",
        '{"id": "/Stale"}\n',
    )
    _write(session_dir / "cache" / "preview" / "stale.png", b"stale-preview")
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "preview_images": ["stale.png"],
            "artifact_validity": {
                "raw_predictions": True,
                "previews": True,
            },
        },
        sync_files=False,
    )

    with pytest.raises(HTTPException) as predictions_exc:
        await artifacts_router.download_predictions(session_id)
    _expect_http(404, predictions_exc)
    with pytest.raises(HTTPException) as preview_exc:
        await assets_router.get_preview_image(session_id, "stale.png")
    _expect_http(404, preview_exc)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fresh_route_read_rejects_artifacts_invalidated_mid_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id, session_dir = await _session(
        tmp_path,
        artifacts_router,
        assets_router,
    )
    _write(
        session_dir / "cache" / "predictions" / "predictions.jsonl",
        '{"id": "/Current"}\n',
    )
    _write(session_dir / "cache" / "preview" / "current.png", b"current-preview")
    await manager.update_session(
        session_id,
        {
            "status": "completed",
            "preview_images": ["current.png"],
            "artifact_validity": {
                "raw_predictions": True,
                "restored_predictions": False,
                "previews": True,
            },
        },
    )
    valid_metadata = await manager.get_session_metadata(session_id)
    assert valid_metadata is not None
    original_read = manager.read_from_store
    original_iter = manager.iter_store_chunks

    async def invalidate_predictions_during_prepare(
        requested_session_id: str,
        key: str,
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes] | None:
        chunks = await original_iter(
            requested_session_id,
            key,
            chunk_size=chunk_size,
        )
        await manager.update_session(
            requested_session_id,
            {
                "artifact_validity": {
                    **valid_metadata["artifact_validity"],
                    "raw_predictions": False,
                }
            },
            sync_files=False,
        )
        return chunks

    monkeypatch.setattr(
        manager, "iter_store_chunks", invalidate_predictions_during_prepare
    )
    with pytest.raises(HTTPException) as predictions_exc:
        await artifacts_router.download_predictions(session_id)
    _expect_http(404, predictions_exc)

    await manager.update_session(
        session_id,
        {"artifact_validity": valid_metadata["artifact_validity"]},
        sync_files=False,
    )
    monkeypatch.setattr(manager, "iter_store_chunks", original_iter)

    async def invalidate_previews_during_read(
        requested_session_id: str,
        key: str,
    ) -> bytes | None:
        data = await original_read(requested_session_id, key)
        await manager.update_session(
            requested_session_id,
            {
                "artifact_validity": {
                    **valid_metadata["artifact_validity"],
                    "previews": False,
                }
            },
            sync_files=False,
        )
        return data

    monkeypatch.setattr(manager, "read_from_store", invalidate_previews_during_read)
    with pytest.raises(HTTPException) as preview_exc:
        await assets_router.get_preview_image(session_id, "current.png")
    _expect_http(404, preview_exc)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assets_missing_session_errors(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    assets_router.set_session_manager(manager)
    missing = str(uuid4())

    calls = [
        lambda: assets_router.get_input_render(missing),
        lambda: assets_router.get_generated_ref(missing),
        lambda: assets_router.get_generated_ref_by_id(missing, "ref"),
        lambda: assets_router.get_preview_image(missing, "preview.png"),
        lambda: assets_router.get_reference_image(missing, "reference.png"),
        lambda: assets_router.list_reference_images(missing),
        lambda: assets_router.get_reference_pdf(missing, "reference.pdf"),
        lambda: assets_router.list_reference_pdfs(missing),
        lambda: assets_router.list_rendered_pdf_pages(missing),
        lambda: assets_router.get_rendered_pdf_page(missing, "pdf_0", "page.png"),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as exc_info:
            await call()
        _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifacts_file_fallback_ladder(tmp_path: Path) -> None:
    assert isinstance(
        await artifacts_router._try_serve_file_with_fallback(
            _FallbackManager(public_url="https://example.test/artifact"),
            "sid",
            "artifact.png",
            tmp_path / "artifact.png",
        ),
        RedirectResponse,
    )

    stored = await artifacts_router._try_serve_file_with_fallback(
        _FallbackManager(data=b"stored"),
        "sid",
        "artifact.usda",
        tmp_path / "artifact.usda",
        filename="artifact.usda",
    )
    assert isinstance(stored, Response)
    assert (
        stored.headers["content-disposition"] == 'attachment; filename="artifact.usda"'
    )

    local = _write(tmp_path / "artifact.mtl", b"local")
    assert isinstance(
        await artifacts_router._try_serve_file_with_fallback(
            _FallbackManager(), "sid", "artifact.mtl", local
        ),
        FileResponse,
    )
    assert (
        await artifacts_router._try_serve_file_with_fallback(
            _FallbackManager(), "sid", "missing.png", tmp_path / "missing.png"
        )
        is None
    )
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router._serve_file_with_fallback(
            _FallbackManager(), "sid", "missing.png", tmp_path / "missing.png"
        )
    _expect_http(404, exc_info)

    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router._serve_file_with_fallback(
            _FallbackManager(),
            "sid",
            "missing.png",
            tmp_path / "missing.png",
            not_found_detail="Custom missing artifact",
        )
    assert exc_info.value.detail == "Custom missing artifact"


@pytest.mark.unit
@pytest.mark.parametrize(
    "media_type",
    ["application/json", "application/problem+json", "application/x-ndjson"],
)
@pytest.mark.asyncio
async def test_structured_artifacts_are_proxied_for_sanitization(
    tmp_path: Path,
    media_type: str,
) -> None:
    manager = _FallbackManager(
        public_url="https://presigned.example.test/manifest",
        data=b'{"output_path":"/var/material-agent/sessions/id/output.usd"}\n',
    )

    response = await artifacts_router._try_serve_file_with_fallback(
        manager,
        "sid",
        "scene/manifest.json",
        tmp_path / "manifest.json",
        media_type=media_type,
    )

    expected_type = (
        StreamingResponse if media_type == "application/x-ndjson" else Response
    )
    assert isinstance(response, expected_type)
    assert not isinstance(response, RedirectResponse)
    if isinstance(response, StreamingResponse):
        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(
            chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks
        )
        assert body == manager.data
    else:
        assert response.body == manager.data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_iterator_is_lazy_during_lineage_recheck_failure(
    tmp_path: Path,
) -> None:
    class FailingLineageManager(_VersionedFallbackManager):
        iterator_started = False

        async def get_session_metadata_versioned(self, session_id: str):
            self.version_calls += 1
            if self.version_calls > 1:
                raise RuntimeError("lineage lookup failed")
            return types.SimpleNamespace(value=self.metadata, version="version")

        async def iter_store_chunks(
            self,
            session_id: str,
            key: str,
            *,
            chunk_size: int,
        ) -> AsyncIterator[bytes] | None:
            async def chunks() -> AsyncIterator[bytes]:
                self.iterator_started = True
                yield b"{}\n"

            return chunks()

    manager = FailingLineageManager({"artifact_validity": {"raw_predictions": True}})

    with pytest.raises(RuntimeError, match="lineage lookup failed"):
        await artifacts_router._try_serve_file_with_fallback(
            manager,
            "sid",
            "logical.bin",
            tmp_path / "artifact.jsonl",
            media_type="application/x-ndjson",
            artifact="raw_predictions",
        )

    assert manager.iterator_started is False


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("alias_kind", ["leaf", "ancestor"])
async def test_material_local_fallback_rejects_pipeline_temp_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, artifacts_router)

    async def no_store_bytes(_session_id: str, _key: str) -> None:
        return None

    monkeypatch.setattr(manager, "read_from_store", no_store_bytes)
    secret_dir = session_dir / "cache" / ".pipeline_temp"
    secret_dir.mkdir(parents=True)
    secret = _write(secret_dir / "credential.json", b"material-secret-sentinel")
    local_path = session_dir / "scene" / "manifest.json"
    if alias_kind == "leaf":
        local_path.parent.mkdir(parents=True)
        local_path.symlink_to(secret)
    else:
        local_path.parent.symlink_to(secret_dir, target_is_directory=True)

    response = await artifacts_router._try_serve_file_with_fallback(
        manager,
        session_id,
        "scene/manifest.json",
        local_path,
        media_type="application/json",
    )

    assert response is None
    assert secret.read_bytes() == b"material-secret-sentinel"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_material_local_response_holds_inode_across_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, artifacts_router)

    async def no_store_bytes(_session_id: str, _key: str) -> None:
        return None

    monkeypatch.setattr(manager, "read_from_store", no_store_bytes)
    local_path = _write(
        session_dir / "scene" / "manifest.json",
        b"safe-material-bytes",
    )
    responses = [
        await artifacts_router._try_serve_file_with_fallback(
            manager,
            session_id,
            "scene/manifest.json",
            local_path,
            media_type="application/json",
        )
        for _ in range(4)
    ]
    assert all(isinstance(response, FileResponse) for response in responses)

    detached = local_path.with_name("manifest.safe")
    local_path.rename(detached)
    secret = _write(
        session_dir / "cache" / ".pipeline_temp" / "credential.json",
        b"material-secret-sentinel",
    )
    local_path.symlink_to(secret)

    async def call_response(
        response: Response,
        method: str,
        headers: list[tuple[bytes, bytes]],
    ) -> list[dict]:
        messages: list[dict] = []

        async def receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict) -> None:
            messages.append(message)

        await response(
            {
                "type": "http",
                "method": method,
                "headers": headers,
                "extensions": {},
            },
            receive,
            send,
        )
        return messages

    get_messages = await call_response(responses[0], "GET", [])  # type: ignore[arg-type]
    range_messages = await call_response(  # type: ignore[arg-type]
        responses[1],
        "GET",
        [(b"range", b"bytes=1-3")],
    )
    head_messages = await call_response(  # type: ignore[arg-type]
        responses[2],
        "HEAD",
        [],
    )

    async def receive_cancelled() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def cancel_on_body(message: dict) -> None:
        if message["type"] == "http.response.body":
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await responses[3](  # type: ignore[operator]
            {
                "type": "http",
                "method": "GET",
                "headers": [],
                "extensions": {},
            },
            receive_cancelled,
            cancel_on_body,
        )
    assert responses[3]._stream.closed  # type: ignore[union-attr]

    get_body = b"".join(
        message.get("body", b"")
        for message in get_messages
        if message["type"] == "http.response.body"
    )
    range_body = b"".join(
        message.get("body", b"")
        for message in range_messages
        if message["type"] == "http.response.body"
    )
    head_body = b"".join(
        message.get("body", b"")
        for message in head_messages
        if message["type"] == "http.response.body"
    )
    assert get_body == b"safe-material-bytes"
    assert range_body == b"afe"
    assert head_body == b""
    assert range_messages[0]["status"] == 206
    assert b"material-secret-sentinel" not in get_body + range_body + head_body


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifacts_output_and_final_render_paths(tmp_path: Path) -> None:
    manager, session_id, session_dir = await _session(tmp_path, artifacts_router)

    _write(session_dir / "output" / "scene_with_materials_flat.usd", "#usda 1.0")
    assert isinstance(await artifacts_router.download_output_usd(session_id), Response)
    (session_dir / "output" / "scene_with_materials_flat.usd").unlink()

    _write(session_dir / "output" / "composed_scene_flat.usd", "#usda 1.0")
    assert isinstance(await artifacts_router.download_output_usd(session_id), Response)
    (session_dir / "output" / "composed_scene_flat.usd").unlink()

    _write(session_dir / "output" / "scene_with_materials.usd", "#usda 1.0")
    assert isinstance(await artifacts_router.download_output_usd(session_id), Response)
    (session_dir / "output" / "scene_with_materials.usd").unlink()
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.download_output_usd(session_id)
    _expect_http(404, exc_info)

    _write(session_dir / "output" / "scene_with_materials.png", PNG_BYTES)
    assert isinstance(
        await artifacts_router.download_final_render(session_id), Response
    )
    (session_dir / "output" / "scene_with_materials.png").unlink()

    _write(session_dir / "output" / "composed_scene_001.png", PNG_BYTES)
    absolute_render = _write(tmp_path / "absolute.png", PNG_BYTES)
    await manager.update_session(
        session_id,
        {
            "pipeline_type": "large_scene",
            "scene": {
                "rendered_images": [
                    "",
                    None,
                    "/",
                    "output/composed_scene_001.png",
                    str(absolute_render),
                    "output/composed_scene_001.png",
                ]
            },
        },
    )
    candidates = artifacts_router._scene_render_candidates(
        await manager.get_session_metadata(session_id), session_dir
    )
    assert len(candidates) == 3
    assert isinstance(
        await artifacts_router.download_final_render(session_id), Response
    )
    (session_dir / "output" / "composed_scene_001.png").unlink()

    await manager.update_session(session_id, {"pipeline_type": "standard", "scene": {}})
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.download_final_render(session_id)
    _expect_http(404, exc_info)

    assert artifacts_router._scene_render_candidates(None, session_dir) == []
    assert artifacts_router._scene_render_candidates({"scene": []}, session_dir) == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifacts_download_endpoints_and_optimization_report(
    tmp_path: Path,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, artifacts_router)
    files_and_calls = [
        (
            session_dir / "cache" / "predictions" / "predictions.jsonl",
            "{}\n",
            artifacts_router.download_predictions(session_id),
        ),
        (
            session_dir / "scene" / "manifest.json",
            "{}",
            artifacts_router.download_scene_manifest(session_id),
        ),
        (
            session_dir / "scene" / "validation_report.json",
            "{}",
            artifacts_router.download_scene_validation_report(session_id),
        ),
        (
            session_dir / "scene" / "predictions.jsonl",
            "{}\n",
            artifacts_router.download_scene_predictions(session_id),
        ),
        (
            session_dir / "cache" / "clusters" / "cluster_map.jsonl",
            "{}\n",
            artifacts_router.download_cluster_map(session_id),
        ),
        (
            session_dir / "cache" / "clusters" / "cluster_report.html",
            "<html></html>",
            artifacts_router.view_cluster_report(session_id),
        ),
        (
            session_dir / "cache" / "clusters" / "cluster_summary.json",
            "{}",
            artifacts_router.download_cluster_summary(session_id),
        ),
        (
            session_dir / "cache" / "clusters" / "dataset_representatives.jsonl",
            "{}\n",
            artifacts_router.download_cluster_representatives(session_id),
        ),
        (
            session_dir / "cache" / "optimized" / "optimized_input.metadata.json",
            "{}",
            artifacts_router.view_optimization_report(session_id),
        ),
    ]
    for path, content, call in files_and_calls:
        _write(path, content)
        assert isinstance(await call, Response)

    restored_path = session_dir / "cache" / "restored" / "restored_predictions.jsonl"
    _write(restored_path, '{"material": "restored"}\n')
    await manager.update_session(
        session_id,
        {"completed_steps": [{"name": "restore_usd"}]},
    )
    restored_response = await artifacts_router.download_predictions(session_id)
    assert isinstance(restored_response, StreamingResponse)
    restored_chunks = [chunk async for chunk in restored_response.body_iterator]
    restored_body = b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode()
        for chunk in restored_chunks
    )
    assert b"restored" in restored_body

    (session_dir / "cache" / "optimized" / "optimized_input.metadata.json").unlink()
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_optimization_report(session_id)
    _expect_http(404, exc_info)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifacts_prediction_report_generation_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, artifacts_router)

    _write(session_dir / "cache" / "predictions" / "prediction_report.html", "<html />")
    assert isinstance(
        await artifacts_router.view_prediction_report(session_id), Response
    )
    (session_dir / "cache" / "predictions" / "prediction_report.html").unlink()

    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(404, exc_info)

    _write(session_dir / "cache" / "predictions" / "predictions.jsonl", '{"p": 1}\n')
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(404, exc_info)

    _write(session_dir / "cache" / "dataset" / "dataset.jsonl", '{"d": 1}\n')

    class _ReportTask:
        def run(self, context: dict, event_bus: object) -> None:
            assert context["predictions"] == [{"p": 1}]
            _write(
                Path(context["output_dir"]) / "prediction_report.html",
                "<html>generated</html>",
            )

    module = types.ModuleType("material_agent.tasks.reporting")
    module.GeneratePredictionReportTask = _ReportTask
    monkeypatch.setitem(sys.modules, "material_agent.tasks.reporting", module)
    generated = await artifacts_router.view_prediction_report(session_id)
    assert isinstance(generated, Response)
    assert b"generated" in generated.body
    metadata = await manager.get_session_metadata(session_id)
    assert metadata is not None
    assert metadata["prediction_report_publication"]["key"].startswith("reports/")

    sentinel = "material-report-publication-sentinel-727"

    async def fail_generate(*args: object, **kwargs: object) -> None:
        raise RuntimeError(sentinel)

    await manager.update_session(
        session_id,
        {
            "prediction_lineage_token": str(uuid4()),
            "artifact_validity": {
                "raw_predictions": True,
                "prediction_report": False,
            },
        },
    )
    monkeypatch.setattr(artifacts_router, "_generate_report_on_demand", fail_generate)
    with caplog.at_level(logging.ERROR, logger=artifacts_router.__name__):
        with pytest.raises(HTTPException) as exc_info:
            await artifacts_router.view_prediction_report(session_id)
    _expect_http(500, exc_info)
    assert exc_info.value.detail == "Report generation failed"
    assert "material_prediction_report_publication_failed" in caplog.text
    assert "phase=local_publication" in caplog.text
    assert sentinel not in caplog.text

    assert manager.store.kind == "local"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifact_lineage_route_guard_and_report_publication_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, session_id, session_dir = await _session(tmp_path, artifacts_router)
    predictions = _write(
        session_dir / "cache" / "predictions" / "predictions.jsonl",
        '{"id": "/Root"}\n',
    )
    dataset = _write(
        session_dir / "cache" / "dataset" / "dataset.jsonl",
        '{"id": "/Root"}\n',
    )
    await manager.update_session(
        session_id,
        {
            "artifact_validity": {
                "raw_predictions": False,
                "restored_predictions": False,
                "prediction_report": False,
            }
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.download_predictions(session_id)
    _expect_http(404, exc_info)
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(404, exc_info)

    await manager.update_session(
        session_id,
        {
            "artifact_validity": {
                "raw_predictions": True,
                "restored_predictions": False,
                "prediction_report": False,
            }
        },
    )

    async def no_lineage(_session_id: str) -> None:
        return None

    monkeypatch.setattr(manager, "capture_prediction_lineage", no_lineage)
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(404, exc_info)

    async def fixed_lineage(_session_id: str) -> str:
        return "lineage"

    monkeypatch.setattr(manager, "capture_prediction_lineage", fixed_lineage)
    original_get_metadata = manager.get_session_metadata
    metadata_calls = 0

    async def metadata_vanishes(requested_session_id: str):
        nonlocal metadata_calls
        metadata_calls += 1
        if metadata_calls > 1:
            return None
        return await original_get_metadata(requested_session_id)

    monkeypatch.setattr(manager, "get_session_metadata", metadata_vanishes)
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(404, exc_info)
    monkeypatch.setattr(manager, "get_session_metadata", original_get_metadata)

    async def staged_report(*_args: object, **_kwargs: object) -> Path:
        path = tmp_path / f"staged-{uuid4()}" / "prediction_report.html"
        _write(path, "<html>staged</html>")
        return path

    monkeypatch.setattr(
        artifacts_router,
        "_generate_report_on_demand",
        staged_report,
    )

    async def reject_publication(*_args: object, **_kwargs: object) -> bool:
        return False

    monkeypatch.setattr(
        manager,
        "mark_prediction_report_valid_if_lineage_matches",
        reject_publication,
    )
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(409, exc_info)

    async def accept_publication(*_args: object, **_kwargs: object) -> bool:
        return True

    monkeypatch.setattr(
        manager,
        "mark_prediction_report_valid_if_lineage_matches",
        accept_publication,
    )

    async def missing_publication(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        artifacts_router,
        "_try_serve_file_with_fallback",
        missing_publication,
    )
    with pytest.raises(HTTPException) as exc_info:
        await artifacts_router.view_prediction_report(session_id)
    _expect_http(500, exc_info)
    assert predictions.is_file() and dataset.is_file()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_report_builder_cleans_staging_directory_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _write(tmp_path / "predictions" / "predictions.jsonl", "{}\n")
    dataset = _write(tmp_path / "dataset" / "dataset.jsonl", "{}\n")

    class _FailingReportTask:
        def run(self, _context: dict, _event_bus: object) -> None:
            raise RuntimeError("report failed")

    module = types.ModuleType("material_agent.tasks.reporting")
    module.GeneratePredictionReportTask = _FailingReportTask
    monkeypatch.setitem(sys.modules, "material_agent.tasks.reporting", module)

    with pytest.raises(RuntimeError, match="report failed"):
        await artifacts_router._generate_report_on_demand(
            tmp_path,
            predictions,
            dataset,
            "lineage",
        )
    report_builds = predictions.parent / ".report-builds"
    assert not report_builds.exists() or not list(report_builds.iterdir())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_artifacts_missing_session_errors(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    artifacts_router.set_session_manager(manager)
    missing = str(uuid4())
    calls = [
        lambda: artifacts_router.download_output_usd(missing),
        lambda: artifacts_router.download_final_render(missing),
        lambda: artifacts_router.download_predictions(missing),
        lambda: artifacts_router.download_scene_manifest(missing),
        lambda: artifacts_router.download_scene_validation_report(missing),
        lambda: artifacts_router.download_scene_predictions(missing),
        lambda: artifacts_router.download_cluster_map(missing),
        lambda: artifacts_router.view_cluster_report(missing),
        lambda: artifacts_router.download_cluster_summary(missing),
        lambda: artifacts_router.download_cluster_representatives(missing),
        lambda: artifacts_router.view_optimization_report(missing),
        lambda: artifacts_router.view_prediction_report(missing),
    ]
    for call in calls:
        with pytest.raises(HTTPException) as exc_info:
            await call()
        _expect_http(404, exc_info)


class _UsageManager:
    def __init__(self) -> None:
        self.deleted = 0
        self.exists = True

    async def list_sessions(self) -> list[str]:
        return ["missing", "old", "future", "invalid", "upload", "complete", "failed"]

    async def get_session_metadata(self, session_id: str) -> dict | None:
        metadata = {
            "missing": None,
            "old": {
                "created_at": "2025-01-01T00:00:00+00:00",
                "status": "completed",
            },
            "future": {
                "created_at": "2027-01-01T00:00:00+00:00",
                "status": "completed",
            },
            "invalid": {
                "created_at": "not-a-date",
                "status": "running",
                "filename": "bad.usd",
            },
            "upload": {
                "created_at": "2026-01-02T00:00:00+00:00",
                "status": "uploading",
            },
            "complete": {
                "created_at": "2026-01-03T00:00:00+00:00",
                "updated_at": "2026-01-03T00:00:01+00:00",
                "status": "completed",
                "user_email": "a@example.com",
                "asset": {"filename": "asset.usd"},
                "duration_seconds": 10,
                "step_timings": {"predict": 2},
                "config": {"vlm_model": "model"},
                "results": {
                    "original_prim_count": 1,
                    "prims_processed": 2,
                    "predictions_made": 3,
                    "materials_applied": 4,
                    "images_generated": 5,
                },
            },
            "failed": {
                "created_at": "2026-01-04T00:00:00+00:00",
                "status": "failed",
                "user_email": "b@example.com",
                "filename": "failed.usd",
            },
        }
        return metadata[session_id]

    async def get_session_metadata_batch(
        self, session_ids: list[str]
    ) -> list[dict | None]:
        return [
            await self.get_session_metadata(session_id) for session_id in session_ids
        ]

    async def session_exists(self, session_id: str) -> bool:
        return self.exists

    async def delete_session(self, session_id: str) -> bool:
        self.deleted += 1
        return self.deleted > 2

    async def cleanup_stale_local_cache(self, max_age_hours: float) -> int:
        assert max_age_hours == 3
        return 4

    async def cleanup_expired_sessions(self) -> int:
        return 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sessions_router_usage_and_admin_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _UsageManager()
    sessions_router.set_session_manager(manager)

    naive = datetime(2026, 1, 1)
    assert sessions_router._ensure_utc(naive).tzinfo == UTC
    aware = datetime(2026, 1, 1, tzinfo=UTC)
    assert sessions_router._ensure_utc(aware).tzinfo == UTC

    with pytest.raises(HTTPException) as exc_info:
        await sessions_router.get_usage_stats(from_date="not-a-date")
    _expect_http(400, exc_info)

    usage = await sessions_router.get_usage_stats(
        from_date="2026-01-01T00:00:00+00:00",
        to_date="2026-12-31T00:00:00+00:00",
        user_email=None,
    )
    assert usage["total_sessions"] == 3
    assert usage["total_completed"] == 1
    assert usage["total_failed"] == 1
    assert usage["by_user"]["a@example.com"]["avg_duration_seconds"] == 10
    assert usage["by_asset"]["asset.usd"]["step_avg_durations"] == {"predict": 2}

    filtered = await sessions_router.get_usage_stats(
        from_date=None,
        to_date=None,
        user_email="nobody@example.com",
    )
    assert filtered["total_sessions"] == 0

    listed = await sessions_router.list_sessions(limit=2, offset=1)
    assert listed["limit"] == 2
    assert listed["offset"] == 1

    assert (await sessions_router.get_session("complete"))["status"] == "completed"
    manager.exists = False
    with pytest.raises(HTTPException) as exc_info:
        await sessions_router.delete_session("missing")
    _expect_http(404, exc_info)
    manager.exists = True

    class _RunningRegistry:
        active_count = 1
        max_concurrent = 2

        def is_running(self, session_id: str) -> bool:
            return True

        async def cancel(self, session_id: str) -> None:
            self.cancelled = session_id

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(sessions_router, "get_job_registry", lambda: _RunningRegistry())
    monkeypatch.setattr(sessions_router.asyncio, "sleep", no_sleep)
    assert await sessions_router.delete_session("complete") is None

    class _ToctouManager(_UsageManager):
        def __init__(self) -> None:
            super().__init__()
            self.exists_checks = 0

        async def session_exists(self, session_id: str) -> bool:
            self.exists_checks += 1
            return self.exists_checks == 1

        async def delete_session(self, session_id: str) -> bool:
            return False

    sessions_router.set_session_manager(_ToctouManager())
    assert await sessions_router.delete_session("complete") is None

    manager = _UsageManager()
    sessions_router.set_session_manager(manager)
    manager.exists = True

    async def never_delete(session_id: str) -> bool:
        return False

    manager.delete_session = never_delete  # type: ignore[method-assign]
    with pytest.raises(HTTPException) as exc_info:
        await sessions_router.delete_session("complete")
    _expect_http(500, exc_info)

    cleanup = await sessions_router.trigger_cleanup(max_age_hours=3)
    assert cleanup == {
        "cleaned_local_cache": 4,
        "expired_sessions_removed": 5,
        "max_age_hours": 3,
    }

    with pytest.raises(HTTPException) as exc_info:
        await sessions_router.get_session("missing")
    _expect_http(404, exc_info)
