# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads for texture pipeline outputs."""

import io
import logging
import queue
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from ..session.manager import SessionManager
from .common import JSON_RESPONSE

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/artifacts", tags=["artifacts"])

ZIP_RESPONSE = {
    "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}
}
USDZ_RESPONSE = {
    "content": {
        "model/vnd.usdz+zip": {"schema": {"type": "string", "format": "binary"}}
    }
}
PNG_RESPONSE = {
    "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}
}
STREAM_CHUNK_SIZE = 1024 * 1024
ZIP_QUEUE_PUT_TIMEOUT_SECONDS = 1.0

# Global session manager (initialized by main app)
session_manager: SessionManager | None = None


def _requires_sanitizing_proxy(media_type: str) -> bool:
    """Keep structured text artifacts behind the public response boundary."""
    normalized = media_type.partition(";")[0].strip().lower()
    return normalized in {
        "application/json",
        "application/x-ndjson",
    } or normalized.endswith("+json")


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


def _zip_directory(directory: Path, prefix: str = "", pattern: str = "*") -> io.BytesIO:
    """Create a ZIP archive of matching files in a directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file():
                arcname = f"{prefix}/{file_path.name}" if prefix else file_path.name
                zf.write(file_path, arcname)
    buffer.seek(0)
    return buffer


def _store_response(
    manager: SessionManager,
    session_id: str,
    key: str,
    media_type: str,
    filename: str | None = None,
) -> Response | None:
    """Return a response for a shared-store object, if available."""
    if not _requires_sanitizing_proxy(media_type):
        public_url = manager.make_store_public_url(session_id, key)
        if public_url:
            return RedirectResponse(public_url, status_code=307)

    stream = manager.open_store_stream(session_id, key)
    if stream is None:
        return None

    headers = {}
    if filename:
        headers["Content-Disposition"] = f"attachment; filename={filename}"
    return StreamingResponse(
        _iter_and_close(stream),
        media_type=media_type,
        headers=headers,
    )


def _validate_filename(filename: str) -> None:
    if (
        "/" in filename
        or "\\" in filename
        or filename in (".", "..")
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in filename)
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")


def _iter_and_close(stream: BinaryIO) -> Iterator[bytes]:
    try:
        while True:
            chunk = stream.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    finally:
        stream.close()


class _ZipQueueWriter(io.RawIOBase):
    def __init__(
        self,
        output_queue: queue.Queue[bytes | None],
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self._queue = output_queue
        self._stop_event = stop_event
        self._offset = 0

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._offset

    def write(self, data: Any) -> int:
        if data:
            while not self._stop_event.is_set():
                try:
                    self._queue.put(
                        bytes(data),
                        timeout=ZIP_QUEUE_PUT_TIMEOUT_SECONDS,
                    )
                    break
                except queue.Full:  # pragma: no cover - backpressure retry loop
                    continue
            else:
                raise BrokenPipeError("ZIP stream consumer disconnected")
            self._offset += len(data)
        return len(data)


def _iter_zip_store_keys(
    manager: SessionManager,
    session_id: str,
    keys: list[str],
    prefix: str,
) -> Iterator[bytes]:
    """Stream a ZIP archive from shared-store keys without spooling to disk."""
    output_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    errors: list[BaseException] = []
    active_streams: list[BinaryIO] = []
    active_streams_lock = threading.Lock()

    def _put_sentinel() -> None:  # pragma: no cover - stream shutdown race
        while not stop_event.is_set():
            try:
                output_queue.put(None, timeout=ZIP_QUEUE_PUT_TIMEOUT_SECONDS)
                return
            except queue.Full:
                continue

    def _produce() -> None:
        try:
            writer = _ZipQueueWriter(output_queue, stop_event)
            with zipfile.ZipFile(writer, "w", zipfile.ZIP_DEFLATED) as zf:
                for key in sorted(keys):
                    if stop_event.is_set():  # pragma: no cover - disconnect race
                        break
                    stream = manager.open_store_stream(session_id, key)
                    if stream is None:  # pragma: no cover - deleted store key race
                        continue
                    with active_streams_lock:
                        active_streams.append(stream)
                    try:
                        arcname = (
                            f"{prefix}/{Path(key).name}" if prefix else Path(key).name
                        )
                        with zf.open(arcname, "w") as dest:
                            while True:
                                if stop_event.is_set():
                                    raise BrokenPipeError(
                                        "ZIP stream consumer disconnected"
                                    )
                                chunk = stream.read(STREAM_CHUNK_SIZE)
                                if not chunk:
                                    break
                                dest.write(chunk)
                    finally:
                        with active_streams_lock:
                            if stream in active_streams:
                                active_streams.remove(stream)
                        stream.close()
        except BrokenPipeError:
            return
        except Exception as exc:
            errors.append(exc)
        finally:
            _put_sentinel()

    producer = threading.Thread(target=_produce, daemon=True)
    producer.start()
    try:
        while True:
            item = output_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        producer.join(timeout=2.0)
        if producer.is_alive():  # pragma: no cover - stuck producer cleanup
            with active_streams_lock:
                streams = list(active_streams)
                active_streams.clear()
            for stream in streams:
                stream.close()
            logger.warning(
                "ZIP stream producer for session %s did not stop cleanly; "
                "closed %d active store stream(s)",
                session_id,
                len(streams),
            )
    if errors:  # pragma: no cover - producer exception handoff
        raise errors[0]


def _zip_store_png_response(
    manager: SessionManager,
    session_id: str,
    store_prefix: str,
    archive_prefix: str,
    filename_stem: str,
) -> StreamingResponse | None:
    keys = [
        key
        for key in manager.list_store_keys(session_id, store_prefix)
        if key.endswith(".png")
    ]
    if not keys:
        return None
    return StreamingResponse(
        _iter_zip_store_keys(manager, session_id, keys, prefix=archive_prefix),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename={filename_stem}_{session_id[:8]}.zip"
            )
        },
    )


@router.get(
    "/{session_id}/materials",
    responses={200: JSON_RESPONSE, 404: JSON_RESPONSE},
)
def download_materials(session_id: str):
    """Download discovered materials JSON file."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    materials_path = manager.get_artifact_path(session_id, "materials")
    if not materials_path or not materials_path.exists():
        response = _store_response(
            manager,
            session_id,
            "cache/discovery/materials.json",
            "application/json",
            "materials.json",
        )
        if response is None:
            raise HTTPException(status_code=404, detail="Materials data not available")
        return response  # pragma: no cover - remote render ZIP without local dir

    return FileResponse(
        materials_path,
        media_type="application/json",
        filename="materials.json",
    )


@router.get(
    "/{session_id}/manifest",
    responses={200: JSON_RESPONSE, 404: JSON_RESPONSE},
)
def download_manifest(session_id: str) -> FileResponse:
    """Download the run artifact manifest JSON file."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    manifest_path = manager.get_artifact_path(session_id, "manifest")
    if not manifest_path or not manifest_path.exists():
        response = _store_response(
            manager,
            session_id,
            "cache/artifacts_manifest.json",
            "application/json",
            "artifacts_manifest.json",
        )
        if response is None:
            raise HTTPException(
                status_code=404, detail="Artifact manifest not available"
            )
        return response  # pragma: no cover - remote render ZIP without local dir

    return FileResponse(
        manifest_path,
        media_type="application/json",
        filename="artifacts_manifest.json",
    )


@router.get(
    "/{session_id}/textures",
    response_class=Response,
    responses={200: ZIP_RESPONSE, 404: JSON_RESPONSE},
)
def download_textures_zip(session_id: str):
    """Download all blended textures as a ZIP archive."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    textures_dir = manager.get_artifact_dir(session_id, "textures")
    if not textures_dir or not textures_dir.exists():
        response = _zip_store_png_response(
            manager, session_id, "cache/textures/", "textures", "textures"
        )
        if response is None:
            raise HTTPException(status_code=404, detail="Textures not available")
        return response

    png_files = list(textures_dir.glob("*.png"))
    if not png_files:
        response = _zip_store_png_response(
            manager, session_id, "cache/textures/", "textures", "textures"
        )
        if response is not None:
            return response
        raise HTTPException(status_code=404, detail="No texture files found")

    buffer = _zip_directory(textures_dir, prefix="textures", pattern="*.png")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=textures_{session_id[:8]}.zip"
        },
    )


@router.get(
    "/{session_id}/textures/{filename}",
    response_class=Response,
    responses={200: PNG_RESPONSE, 400: JSON_RESPONSE, 404: JSON_RESPONSE},
)
def download_single_texture(session_id: str, filename: str):
    """Download a single texture file."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    _validate_filename(filename)

    textures_dir = manager.get_artifact_dir(session_id, "textures")
    if not textures_dir:
        textures_dir = manager.get_session_dir(session_id) / "cache" / "textures"
        if not manager.list_store_keys(session_id, "cache/textures/"):
            raise HTTPException(status_code=404, detail="Textures not available")

    file_path = textures_dir / filename

    if not file_path.resolve().is_relative_to(textures_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path, media_type="image/png")

    if not file_path.exists():
        response = _store_response(
            manager,
            session_id,
            f"cache/textures/{filename}",
            "image/png",
            filename,
        )
        if response is None:
            raise HTTPException(
                status_code=404, detail=f"Texture file not found: {filename}"
            )
        return response

    raise HTTPException(status_code=404, detail=f"Texture file not found: {filename}")


@router.get(
    "/{session_id}/output",
    response_class=Response,
    responses={200: USDZ_RESPONSE, 404: JSON_RESPONSE},
)
def download_output(session_id: str):
    """Download the textured output as a self-contained USDZ archive.

    The USDZ bundles the USD file with all texture images into a single
    download — no separate texture download needed.
    """
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    usdz_path = manager.get_artifact_path(session_id, "output_usdz")
    if not usdz_path or not usdz_path.exists():
        response = _store_response(
            manager,
            session_id,
            "cache/output/textured_output.usdz",
            "model/vnd.usdz+zip",
            "textured_output.usdz",
        )
        if response is None:
            raise HTTPException(status_code=404, detail="Output USDZ not available")
        return response

    return FileResponse(
        usdz_path,
        media_type="model/vnd.usdz+zip",
        filename="textured_output.usdz",
    )


@router.get(
    "/{session_id}/renders",
    response_class=Response,
    responses={200: ZIP_RESPONSE, 404: JSON_RESPONSE},
)
def download_renders_zip(session_id: str):
    """Download all rendered images as a ZIP archive."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    renders_dir = manager.get_artifact_dir(session_id, "renders")
    if not renders_dir or not renders_dir.exists():
        response = _zip_store_png_response(
            manager, session_id, "cache/renders/", "renders", "renders"
        )
        if response is None:
            raise HTTPException(status_code=404, detail="Renders not available")
        return response  # pragma: no cover - remote render ZIP without local dir

    png_files = list(renders_dir.glob("*.png"))
    if not png_files:
        response = _zip_store_png_response(
            manager, session_id, "cache/renders/", "renders", "renders"
        )
        if response is not None:
            return response
        raise HTTPException(status_code=404, detail="No render files found")

    buffer = _zip_directory(renders_dir, prefix="renders", pattern="*.png")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=renders_{session_id[:8]}.zip"
        },
    )


@router.get(
    "/{session_id}/renders/{filename}",
    response_class=Response,
    responses={200: PNG_RESPONSE, 400: JSON_RESPONSE, 404: JSON_RESPONSE},
)
def download_single_render(session_id: str, filename: str):
    """Download a single rendered image."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    _validate_filename(filename)

    renders_dir = manager.get_artifact_dir(session_id, "renders")
    if not renders_dir:
        renders_dir = manager.get_session_dir(session_id) / "cache" / "renders"
        if not manager.list_store_keys(session_id, "cache/renders/"):
            raise HTTPException(status_code=404, detail="Renders not available")

    file_path = renders_dir / filename

    if not file_path.resolve().is_relative_to(renders_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists() or not file_path.is_file():
        response = _store_response(
            manager,
            session_id,
            f"cache/renders/{filename}",
            "image/png",
            filename,
        )
        if response is None:
            raise HTTPException(
                status_code=404, detail=f"Render file not found: {filename}"
            )
        return response

    return FileResponse(file_path, media_type="image/png")


@router.get(
    "/{session_id}/preview/{filename}",
    response_class=Response,
    responses={200: PNG_RESPONSE, 400: JSON_RESPONSE, 404: JSON_RESPONSE},
)
def download_preview(session_id: str, filename: str):
    """Download a preview/thumbnail image."""
    manager = get_session_manager()

    if not manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    preview_dir = session_dir / "preview"

    _validate_filename(filename)

    file_path = preview_dir / filename

    if not file_path.resolve().is_relative_to(preview_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists() or not file_path.is_file():
        response = _store_response(
            manager,
            session_id,
            f"preview/{filename}",
            "image/png",
            filename,
        )
        if response is None:
            raise HTTPException(
                status_code=404, detail=f"Preview file not found: {filename}"
            )
        return response

    return FileResponse(file_path, media_type="image/png")
