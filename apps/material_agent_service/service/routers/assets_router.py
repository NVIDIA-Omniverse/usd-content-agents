# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assets API endpoints - Images and previews."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from world_understanding.utils.held_file_response import HeldFileResponse

from ..artifact_lineage import artifact_is_valid
from ..models.responses import PreviewImage, PreviewList
from ..runtime.bus import get_event_bus
from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/assets", tags=["assets"])

# Content type mapping for common file extensions
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
}

# Global session manager (initialized by main app)
session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


async def _live_preview_snapshot(session_id: str) -> dict | None:
    """Return current-run preview evidence while the local pipeline is active."""
    snapshot = await get_event_bus().get_fenced_snapshot(session_id)
    if snapshot and snapshot.get("status") in {"pending", "running", "cancelling"}:
        return snapshot
    return None


def _get_generated_reference_entry(
    metadata: dict | None, reference_id: str | None = None
) -> dict | None:
    if not metadata:
        return None

    generated_refs = metadata.get("generated_reference_images", [])
    if reference_id is None:
        return generated_refs[-1] if generated_refs else None

    for ref in generated_refs:
        if ref.get("id") == reference_id:
            return ref
    return None


async def _serve_file_with_fallback(
    manager: SessionManager,
    session_id: str,
    key: str,
    local_path: Path,
    media_type: str | None = None,
    filename: str | None = None,
) -> Response | FileResponse | RedirectResponse | None:
    """Serve a file with fallback from presigned URL → store → local.

    Args:
        manager: Session manager
        session_id: Session identifier
        key: Store key for the file
        local_path: Local filesystem path
        media_type: MIME type (auto-detected if None)
        filename: Download filename (for Content-Disposition)

    Returns:
        Response object (redirect, streaming, or file response), or None if not found
    """
    # Auto-detect media type from extension if not provided
    if media_type is None:
        suffix = local_path.suffix.lower()
        media_type = CONTENT_TYPES.get(suffix, "application/octet-stream")

    metadata_snapshot = await manager.get_session_metadata_versioned(session_id)
    metadata = metadata_snapshot.value
    if metadata is None or metadata_snapshot.version is None:
        return None
    run_scoped = key.startswith(("cache/preview/", "preview/"))
    if run_scoped and not artifact_is_valid(metadata, "previews"):
        return None

    def resolve_store_key(current_metadata: dict) -> str | None:
        return (
            manager.resolve_published_artifact_key(
                current_metadata,
                key,
                legacy_key=key,
            )
            if run_scoped
            else key
        )

    store_key = resolve_store_key(metadata)
    if store_key is None:
        return None

    async def publication_is_still_current() -> bool:
        if not run_scoped:
            return True
        current = await manager.get_session_metadata_versioned(session_id)
        if current.value is None or current.version is None:
            return False
        return artifact_is_valid(current.value, "previews") and (
            resolve_store_key(current.value) == store_key
        )

    # 1. Try presigned URL (redirect)
    url = await manager.make_public_url(session_id, store_key)
    if url and await publication_is_still_current():
        return RedirectResponse(url, status_code=302)

    # 2. Try reading from store (streaming response)
    data = await manager.read_from_store(session_id, store_key)
    if data is not None and await publication_is_still_current():
        headers = {}
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(content=data, media_type=media_type, headers=headers)

    # 3. Fallback to local file
    if run_scoped and (
        manager.store.kind != "local"
        or (metadata and metadata.get("published_artifacts") is not None)
    ):
        return None
    local_artifact = await manager.open_local_artifact(session_id, local_path)
    if local_artifact is None:
        return None
    if run_scoped:
        try:
            local_data = local_artifact.stream.read()
        finally:
            local_artifact.stream.close()
        if not await publication_is_still_current():
            return None
        headers = {}
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(content=local_data, media_type=media_type, headers=headers)
    return HeldFileResponse(
        local_artifact,
        media_type=media_type,
        filename=filename,
    )


@router.api_route("/{session_id}/input-render", methods=["GET", "HEAD"])
async def get_input_render(session_id: str):
    """Get the input USD render (before material assignment).

    This preview is generated automatically after upload to show the original scene.

    Args:
        session_id: Session identifier

    Returns:
        PNG image of the input scene

    Raises:
        404: If session not found or render not yet complete
        503: If render is still in progress
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = "input/input_render.png"
    session_dir = manager.get_session_dir(session_id)
    input_render_path = session_dir / key

    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        input_render_path,
        media_type="image/png",
    )
    if response:
        return response

    metadata = await manager.get_session_metadata(session_id)
    if metadata and metadata.get("preview_render_status") == "failed":
        raise HTTPException(status_code=424, detail="Input preview render failed")

    # Check if it's still rendering
    temp_config = session_dir / ".input_render_config.yaml"
    if temp_config.exists() or (
        metadata and metadata.get("preview_render_status") == "rendering"
    ):
        raise HTTPException(status_code=503, detail="Input render still in progress")

    raise HTTPException(status_code=404, detail="Input render not available")


@router.api_route(
    "/{session_id}/generated-ref", methods=["GET", "HEAD"], response_model=None
)
async def get_generated_ref(session_id: str):
    """Get the AI-generated reference image.

    This image is generated interactively by the user via the
    generate-reference-image endpoint before pipeline submission.

    Args:
        session_id: Session identifier

    Returns:
        PNG image of the generated reference

    Raises:
        404: If session not found or image not generated
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    generated_ref = _get_generated_reference_entry(metadata)
    key = generated_ref.get("key") if generated_ref else "input/generated_ref_0.png"
    if not isinstance(key, str):
        raise HTTPException(
            status_code=404, detail="Generated reference image not available"
        )
    session_dir = manager.get_session_dir(session_id)
    generated_ref_path = session_dir / key

    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        generated_ref_path,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404, detail="Generated reference image not available"
    )


@router.api_route(
    "/{session_id}/generated-ref/{reference_id}",
    methods=["GET", "HEAD"],
    response_model=None,
)
async def get_generated_ref_by_id(session_id: str, reference_id: str):
    """Get a generated reference image by its explicit reference ID."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    generated_ref = _get_generated_reference_entry(metadata, reference_id)
    if not generated_ref:
        raise HTTPException(status_code=404, detail="Generated reference not found")

    key = generated_ref.get("key")
    if not isinstance(key, str):
        raise HTTPException(status_code=404, detail="Generated reference not found")

    session_dir = manager.get_session_dir(session_id)
    generated_ref_path = session_dir / key

    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        generated_ref_path,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404, detail="Generated reference image not available"
    )


@router.api_route("/{session_id}/preview/{image_name}", methods=["GET", "HEAD"])
async def get_preview_image(session_id: str, image_name: str):
    """Get a preview image (thumbnail) from the rendering process.

    Thumbnails are 128×128 resized versions stored in cache/preview/.

    Args:
        session_id: Session identifier
        image_name: Preview image filename

    Returns:
        PNG image
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    persisted_preview = bool(
        artifact_is_valid(metadata, "previews")
        and metadata
        and image_name in (metadata.get("preview_images") or [])
    )
    live_snapshot = await _live_preview_snapshot(session_id)
    live_preview = bool(
        live_snapshot and image_name in (live_snapshot.get("preview_images") or [])
    )
    if not persisted_preview and not live_preview:
        raise HTTPException(
            status_code=404,
            detail=f"Preview image not found: {image_name}",
        )
    session_dir = manager.get_session_dir(session_id)

    if live_preview:
        # The EventBus name belongs to the in-flight local generation. Prefer
        # its exact bytes over any prior immutable publication with the same
        # logical filename.
        for live_path in (
            session_dir / "cache" / "preview" / image_name,
            session_dir / "preview" / image_name,
        ):
            if live_path.is_file():
                try:
                    live_data = live_path.read_bytes()
                except OSError:
                    continue
                confirmed_snapshot = await _live_preview_snapshot(session_id)
                if confirmed_snapshot and image_name in (
                    confirmed_snapshot.get("preview_images") or []
                ):
                    return Response(content=live_data, media_type="image/png")
                break

    # Try cache/preview/ first (new event-driven path)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        f"cache/preview/{image_name}",
        session_dir / "cache" / "preview" / image_name,
        media_type="image/png",
    )
    if response:
        return response

    # Fallback to preview/ (old path for backward compatibility)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        f"preview/{image_name}",
        session_dir / "preview" / image_name,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404, detail=f"Preview image not found: {image_name}"
    )


@router.get("/{session_id}/previews", response_model=PreviewList)
async def list_preview_images(session_id: str) -> PreviewList:
    """List all available preview images.

    Args:
        session_id: Session identifier

    Returns:
        List of preview images with URLs
    """
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")
    live_snapshot = await _live_preview_snapshot(session_id)
    live_preview_images = list((live_snapshot or {}).get("preview_images") or [])
    persisted_preview_images = (
        list(metadata.get("preview_images") or [])
        if artifact_is_valid(metadata, "previews")
        else []
    )
    preview_images = list(
        dict.fromkeys([*persisted_preview_images, *live_preview_images])
    )
    if not preview_images and not artifact_is_valid(metadata, "previews"):
        raise HTTPException(
            status_code=404,
            detail="Preview images are not available for the current pipeline run",
        )

    previews = [
        PreviewImage(
            name=img,
            url=f"/assets/{session_id}/preview/{img}",
            prim_path=None,  # Could extract from filename if needed
            created_at=(live_snapshot or metadata)["updated_at"],  # Approximate
        )
        for img in preview_images
    ]

    return PreviewList(session_id=session_id, previews=previews, total=len(previews))


@router.get("/{session_id}/reference/{image_name}")
async def get_reference_image(session_id: str, image_name: str):
    """Get a reference image uploaded for this session.

    Reference images are stored in input/reference_images/ and used by the VLM
    to understand the target appearance/materials of the asset.

    Args:
        session_id: Session identifier
        image_name: Reference image filename (e.g., reference_1.png)

    Returns:
        Image file (PNG or JPG)
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = f"input/reference_images/{image_name}"
    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        session_dir / key,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Reference image not found")


@router.get("/{session_id}/references")
async def list_reference_images(session_id: str):
    """List all reference images uploaded for this session.

    Args:
        session_id: Session identifier

    Returns:
        JSON list of reference image filenames with URLs
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    ref_dir = manager.get_session_dir(session_id) / "input" / "reference_images"

    references = []
    if ref_dir.exists():
        # Get all image files
        for img_path in sorted(ref_dir.glob("reference_*.*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                references.append(
                    {
                        "name": img_path.name,
                        "url": f"/assets/{session_id}/reference/{img_path.name}",
                    }
                )

    return {
        "session_id": session_id,
        "references": references,
        "total": len(references),
    }


@router.get("/{session_id}/reference-pdf/{pdf_name}")
async def get_reference_pdf(session_id: str, pdf_name: str):
    """Get a reference PDF uploaded for this session.

    Reference PDFs are stored in input/reference_pdfs/ and converted to page
    images during the prepare_dataset step as specification evidence. The page
    images remain outside visual-model media.

    Args:
        session_id: Session identifier
        pdf_name: Reference PDF filename (e.g., reference_0000.pdf)

    Returns:
        PDF file
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = f"input/reference_pdfs/{pdf_name}"
    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        session_dir / key,
        media_type="application/pdf",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="Reference PDF not found")


@router.get("/{session_id}/reference-pdfs")
async def list_reference_pdfs(session_id: str):
    """List all reference PDFs uploaded for this session.

    Args:
        session_id: Session identifier

    Returns:
        JSON list of reference PDF filenames with URLs
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    pdf_dir = manager.get_session_dir(session_id) / "input" / "reference_pdfs"

    pdfs = []
    if pdf_dir.exists():
        # Get all PDF files
        for pdf_path in sorted(pdf_dir.glob("reference_*.pdf")):
            pdfs.append(
                {
                    "name": pdf_path.name,
                    "url": f"/assets/{session_id}/reference-pdf/{pdf_path.name}",
                }
            )

    return {
        "session_id": session_id,
        "pdfs": pdfs,
        "total": len(pdfs),
    }


@router.get("/{session_id}/pdf-pages")
async def list_rendered_pdf_pages(session_id: str):
    """List rendered PDF page images for this session.

    PDF pages are converted to images during the prepare_dataset step
    and stored in cache/dataset/pdf_0/, pdf_1/, etc.

    Args:
        session_id: Session identifier

    Returns:
        JSON list of rendered page images grouped by PDF index
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    dataset_dir = manager.get_session_dir(session_id) / "cache" / "dataset"
    pages: list[dict] = []

    if dataset_dir.exists():
        for pdf_dir in sorted(dataset_dir.glob("pdf_*")):
            if not pdf_dir.is_dir():
                continue
            for img_path in sorted(pdf_dir.glob("*.png")):
                pages.append(
                    {
                        "name": img_path.name,
                        "pdf_index": pdf_dir.name,
                        "url": f"/assets/{session_id}/pdf-page/{pdf_dir.name}/{img_path.name}",
                    }
                )

    return {
        "session_id": session_id,
        "pages": pages,
        "total": len(pages),
    }


@router.get("/{session_id}/pdf-page/{pdf_index}/{page_name}")
async def get_rendered_pdf_page(session_id: str, pdf_index: str, page_name: str):
    """Get a single rendered PDF page image.

    Args:
        session_id: Session identifier
        pdf_index: PDF directory name (e.g., pdf_0)
        page_name: Image filename (e.g., spec_page_001.png)

    Returns:
        PNG image file
    """
    import re

    if not re.match(r"^pdf_\d+$", pdf_index):
        raise HTTPException(status_code=400, detail="Invalid pdf_index format")
    if ".." in page_name or "/" in page_name:
        raise HTTPException(status_code=400, detail="Invalid page_name")

    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = f"cache/dataset/{pdf_index}/{page_name}"
    session_dir = manager.get_session_dir(session_id)
    response = await _serve_file_with_fallback(
        manager,
        session_id,
        key,
        session_dir / key,
        media_type="image/png",
    )
    if response:
        return response

    raise HTTPException(status_code=404, detail="PDF page image not found")
