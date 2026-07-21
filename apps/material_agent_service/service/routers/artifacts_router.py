# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads and reports."""

import asyncio
import hashlib
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.held_file_response import HeldFileResponse

from ..artifact_lineage import artifact_is_valid
from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/artifacts", tags=["artifacts"])

# Content type mapping for common file extensions
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".usd": "application/octet-stream",
    ".usda": "text/plain",
    ".usdc": "application/octet-stream",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".html": "text/html",
    ".pdf": "application/pdf",
}

# Global session manager (initialized by main app)
session_manager: SessionManager | None = None
STORE_STREAM_CHUNK_SIZE = 1024 * 1024


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


async def _require_valid_artifact_lineage(
    manager: SessionManager,
    session_id: str,
    artifact: str,
) -> dict[str, Any] | None:
    """Reject artifacts retained from an invalidated pipeline generation."""
    metadata = await manager.get_session_metadata(session_id)
    if not artifact_is_valid(metadata, artifact):
        raise HTTPException(
            status_code=404,
            detail="Artifact is not available for the current pipeline run",
        )
    return metadata


async def _try_serve_file_with_fallback(
    manager: SessionManager,
    session_id: str,
    key: str,
    local_path: Path,
    media_type: str | None = None,
    filename: str | None = None,
    artifact: str | None = None,
) -> Response | FileResponse | RedirectResponse | StreamingResponse | None:
    """Serve a file with fallback from presigned URL → store → local.

    Args:
        manager: Session manager
        session_id: Session identifier
        key: Store key for the file
        local_path: Local filesystem path
        media_type: MIME type (auto-detected if None)
        filename: Download filename (for Content-Disposition)

    Returns:
        Response object (redirect, streaming, or file response), or ``None`` if
        the artifact is not available anywhere.
    """
    # Auto-detect media type from extension if not provided
    if media_type is None:
        suffix = local_path.suffix.lower()
        media_type = CONTENT_TYPES.get(suffix, "application/octet-stream")

    metadata_snapshot = await manager.get_session_metadata_versioned(session_id)
    metadata = metadata_snapshot.value
    if metadata is None or metadata_snapshot.version is None:
        return None
    if artifact is not None and not artifact_is_valid(metadata, artifact):
        return None
    immutable_key = key.startswith(("runs/", "reports/"))

    def resolve_store_key(current_metadata: dict[str, Any]) -> str | None:
        if immutable_key:
            return key
        if key == "cache/predictions/prediction_report.html":
            return manager.resolve_prediction_report_key(
                current_metadata,
                legacy_key=key,
            )
        return manager.resolve_published_artifact_key(
            current_metadata,
            key,
            legacy_key=key,
        )

    store_key = resolve_store_key(metadata)
    if store_key is None:
        return None

    async def publication_is_still_current() -> bool:
        if artifact is None:
            return True
        current = await manager.get_session_metadata_versioned(session_id)
        if current.value is None or current.version is None:
            return False
        return artifact_is_valid(current.value, artifact) and (
            resolve_store_key(current.value) == store_key
        )

    # 1. Try presigned URL (redirect). JSON and NDJSON must be proxied so the
    # public response middleware can sanitize persisted session-local fields.
    if not _requires_sanitizing_proxy(media_type):
        url = await manager.make_public_url(session_id, store_key)
        if url and await publication_is_still_current():
            return RedirectResponse(url, status_code=302)

    # 2. Keep record-oriented artifacts streaming so sanitization stays bounded.
    if _requires_sanitizing_proxy(media_type) and (
        media_type.partition(";")[0].strip().lower() == "application/x-ndjson"
    ):
        stream = await manager.iter_store_chunks(
            session_id,
            store_key,
            chunk_size=STORE_STREAM_CHUNK_SIZE,
        )
        if stream is not None:
            if await publication_is_still_current():
                headers = {}
                if filename:
                    headers["Content-Disposition"] = (
                        f'attachment; filename="{filename}"'
                    )
                return StreamingResponse(
                    stream,
                    media_type=media_type,
                    headers=headers,
                )

    # 3. Try reading a single-value object from the store.
    data = await manager.read_from_store(session_id, store_key)
    if data is not None and await publication_is_still_current():
        headers = {}
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(content=data, media_type=media_type, headers=headers)

    # 4. Fallback to local file
    if manager.store.kind != "local" or (
        metadata is not None and metadata.get("published_artifacts") is not None
    ):
        return None
    local_artifact = await manager.open_local_artifact(session_id, local_path)
    if local_artifact is None:
        return None
    if artifact is not None:
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


async def _serve_file_with_fallback(
    manager: SessionManager,
    session_id: str,
    key: str,
    local_path: Path,
    media_type: str | None = None,
    filename: str | None = None,
    not_found_detail: str = "Artifact not found",
    artifact: str | None = None,
) -> Response | FileResponse | RedirectResponse | StreamingResponse:
    """Serve a file with fallback from presigned URL → store → local.

    Raises:
        HTTPException: If file not found anywhere.
    """
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        key,
        local_path,
        media_type=media_type,
        filename=filename,
        artifact=artifact,
    )
    if response is not None:
        return response

    raise HTTPException(status_code=404, detail=not_found_detail)


def _scene_render_candidates(
    metadata: dict[str, Any] | None,
    session_dir: Path,
) -> list[tuple[str, Path, str]]:
    """Return store keys/local paths for large-scene render fallbacks."""
    candidates: list[tuple[str, Path, str]] = []
    scene_metadata = metadata.get("scene", {}) if metadata else {}
    rendered_images = (
        scene_metadata.get("rendered_images", [])
        if isinstance(scene_metadata, dict)
        else []
    )
    if isinstance(rendered_images, list):
        for image in rendered_images:
            if not isinstance(image, str) or not image:
                continue
            image_path = Path(image)
            filename = image_path.name
            if not filename:
                continue
            local_path = image_path if image_path.is_absolute() else session_dir / image
            candidates.append((f"output/{filename}", local_path, filename))
            mirrored_path = session_dir / "output" / filename
            if mirrored_path != local_path:
                candidates.append((f"output/{filename}", mirrored_path, filename))

    for image_path in sorted((session_dir / "output").glob("composed_scene_*.png")):
        candidates.append((f"output/{image_path.name}", image_path, image_path.name))

    unique: list[tuple[str, Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, local_path, filename in candidates:
        marker = (key, str(local_path))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append((key, local_path, filename))
    return unique


async def _generate_report_on_demand(
    session_dir: Path,
    predictions_path: Path,
    dataset_path: Path,
    prediction_lineage: str,
) -> Path:
    """Generate prediction HTML report on-demand.

    This is called only when the /report endpoint is accessed, preventing
    blocking operations during the predict step.

    Args:
        session_dir: Session directory
        predictions_path: Path to predictions.jsonl
        dataset_path: Path to dataset.jsonl
    """

    # Load predictions
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    # Load dataset
    dataset = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # Import lazily because report generation pulls in material-agent runtime deps.
    from material_agent.tasks.reporting import GeneratePredictionReportTask

    task = GeneratePredictionReportTask()

    # Prepare context
    lineage_digest = hashlib.sha256(prediction_lineage.encode("utf-8")).hexdigest()
    report_dir = (
        predictions_path.parent / ".report-builds" / f"{lineage_digest}-{uuid.uuid4()}"
    )
    report_context = {
        "predictions": predictions,
        "failed_predictions": [],
        "dataset": dataset,
        "output_dir": str(report_dir),
        "dataset_path": str(dataset_path),
    }

    # Run report generation in thread pool (blocks this coroutine but not event loop)
    try:
        await asyncio.to_thread(task.run, report_context, None)
        staged_report = report_dir / "prediction_report.html"
        if not staged_report.is_file():
            raise RuntimeError(
                "Report generator did not produce prediction_report.html"
            )
        return staged_report
    except BaseException:
        shutil.rmtree(report_dir, ignore_errors=True)
        raise


@router.get("/{session_id}/output")
async def download_output_usd(session_id: str):
    """Download flattened output USD file with applied materials.

    Returns the flattened USD file that was sent to rendering, not the layered version.
    This ensures the downloaded file matches what was actually rendered.

    Args:
        session_id: Session identifier

    Returns:
        Flattened USD file as download
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    session_dir = manager.get_session_dir(session_id)

    if artifact_is_valid(metadata, "rendered_output_usd"):
        # Try flattened version first.
        response = await _try_serve_file_with_fallback(
            manager,
            session_id,
            "output/scene_with_materials_flat.usd",
            session_dir / "output" / "scene_with_materials_flat.usd",
            filename="scene_with_materials_flat.usd",
            artifact="rendered_output_usd",
        )
        if response:
            return response

        # Large-scene rendering writes this sibling flat file before mirroring.
        response = await _try_serve_file_with_fallback(
            manager,
            session_id,
            "output/composed_scene_flat.usd",
            session_dir / "output" / "composed_scene_flat.usd",
            filename="scene_with_materials_flat.usd",
            artifact="rendered_output_usd",
        )
        if response:
            return response

    # Fallback to non-flattened version
    if artifact_is_valid(metadata, "applied_output_usd"):
        logger.warning(
            f"Flattened USD not found for {session_id[:8]}, "
            "trying non-flattened version"
        )
        response = await _try_serve_file_with_fallback(
            manager,
            session_id,
            "output/scene_with_materials.usd",
            session_dir / "output" / "scene_with_materials.usd",
            filename="scene_with_materials.usd",
            artifact="applied_output_usd",
        )
        if response:
            return response

    raise HTTPException(
        status_code=404,
        detail="Output USD not available. Pipeline may not be completed.",
    )


@router.api_route("/{session_id}/final-render", methods=["GET", "HEAD"])
async def download_final_render(session_id: str):
    """Download final render image (output USD with materials applied).

    Args:
        session_id: Session identifier

    Returns:
        Final render PNG image
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    metadata = await _require_valid_artifact_lineage(
        manager,
        session_id,
        "final_render",
    )
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        "output/scene_with_materials.png",
        session_dir / "output" / "scene_with_materials.png",
        media_type="image/png",
        artifact="final_render",
    )
    if response:
        return response

    if metadata and metadata.get("pipeline_type") == "large_scene":
        for key, local_path, filename in _scene_render_candidates(
            metadata,
            session_dir,
        ):
            response = await _try_serve_file_with_fallback(
                manager,
                session_id,
                key,
                local_path,
                media_type="image/png",
                filename=filename,
                artifact="final_render",
            )
            if response:
                return response

    raise HTTPException(
        status_code=404,
        detail="Final render not available. Pipeline may not have completed the render step.",
    )


@router.get("/{session_id}/predictions")
async def download_predictions(session_id: str):
    """Download predictions JSONL file.

    Args:
        session_id: Session identifier

    Returns:
        Predictions JSONL file
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    metadata = await manager.get_session_metadata(session_id)
    if artifact_is_valid(metadata, "restored_predictions"):
        response = await _try_serve_file_with_fallback(
            manager,
            session_id,
            "cache/restored/restored_predictions.jsonl",
            session_dir / "cache" / "restored" / "restored_predictions.jsonl",
            media_type="application/x-ndjson",
            filename="predictions.jsonl",
            artifact="restored_predictions",
        )
        if response:
            return response
    if not artifact_is_valid(metadata, "raw_predictions"):
        raise HTTPException(
            status_code=404,
            detail="Predictions are not available for the current pipeline run",
        )
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/predictions/predictions.jsonl",
        session_dir / "cache" / "predictions" / "predictions.jsonl",
        media_type="application/x-ndjson",
        filename="predictions.jsonl",
        not_found_detail="Predictions not available yet",
        artifact="raw_predictions",
    )


@router.get("/{session_id}/scene-manifest")
async def download_scene_manifest(session_id: str):
    """Download the large-scene manifest JSON file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "scene/manifest.json",
        session_dir / "scene" / "manifest.json",
        media_type="application/json",
        filename="manifest.json",
        not_found_detail="Scene manifest not available",
    )


@router.get("/{session_id}/scene-validation-report")
async def download_scene_validation_report(session_id: str):
    """Download the large-scene validation report JSON file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "scene/validation_report.json",
        session_dir / "scene" / "validation_report.json",
        media_type="application/json",
        filename="validation_report.json",
        not_found_detail="Scene validation report not available",
    )


@router.get("/{session_id}/scene-predictions")
async def download_scene_predictions(session_id: str):
    """Download collated large-scene per-asset predictions JSONL."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "scene/predictions.jsonl",
        session_dir / "scene" / "predictions.jsonl",
        media_type="application/x-ndjson",
        filename="scene_predictions.jsonl",
        not_found_detail="Scene predictions not available",
    )


@router.get("/{session_id}/cluster-map")
async def download_cluster_map(session_id: str):
    """Download the prim clustering map JSONL file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    await _require_valid_artifact_lineage(manager, session_id, "cluster_map")
    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/cluster_map.jsonl",
        session_dir / "cache" / "clusters" / "cluster_map.jsonl",
        media_type="application/x-ndjson",
        filename="cluster_map.jsonl",
        not_found_detail="Cluster map not available",
        artifact="cluster_map",
    )


@router.get("/{session_id}/cluster-report")
async def view_cluster_report(session_id: str):
    """View the prim clustering HTML report."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    await _require_valid_artifact_lineage(manager, session_id, "cluster_report")
    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/cluster_report.html",
        session_dir / "cache" / "clusters" / "cluster_report.html",
        media_type="text/html",
        not_found_detail="Cluster report not available",
        artifact="cluster_report",
    )


@router.get("/{session_id}/cluster-summary")
async def download_cluster_summary(session_id: str):
    """Download the lightweight prim clustering summary JSON file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    await _require_valid_artifact_lineage(manager, session_id, "cluster_summary")
    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/cluster_summary.json",
        session_dir / "cache" / "clusters" / "cluster_summary.json",
        media_type="application/json",
        filename="cluster_summary.json",
        not_found_detail="Cluster summary not available",
        artifact="cluster_summary",
    )


@router.get("/{session_id}/cluster-representatives")
async def download_cluster_representatives(session_id: str):
    """Download the representative-only dataset used for clustered prediction."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    await _require_valid_artifact_lineage(
        manager,
        session_id,
        "cluster_representatives",
    )
    session_dir = manager.get_session_dir(session_id)
    return await _serve_file_with_fallback(
        manager,
        session_id,
        "cache/clusters/dataset_representatives.jsonl",
        session_dir / "cache" / "clusters" / "dataset_representatives.jsonl",
        media_type="application/x-ndjson",
        filename="dataset_representatives.jsonl",
        not_found_detail="Cluster representatives not available",
        artifact="cluster_representatives",
    )


@router.get("/{session_id}/optimization-report")
async def view_optimization_report(session_id: str):
    """View optimization JSON report in browser.

    Args:
        session_id: Session identifier

    Returns:
        Optimization report JSON
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    key = "cache/optimized/optimized_input.metadata.json"
    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "optimized" / "optimized_input.metadata.json"

    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        key,
        report_path,
        media_type="application/json",
    )
    if response:
        return response

    raise HTTPException(
        status_code=404,
        detail="Optimization report is not available. Pipeline may not have completed the optimization step.",
    )


@router.get("/{session_id}/report")
async def view_prediction_report(session_id: str):
    """View prediction HTML report in browser.

    Generates the report on-demand if it doesn't exist yet.
    This prevents blocking the predict step with heavy HTML generation.

    Args:
        session_id: Session identifier

    Returns:
        Prediction report HTML served for viewing (not download)
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = await manager.get_session_metadata(session_id)
    key = "cache/predictions/prediction_report.html"
    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "predictions" / "prediction_report.html"

    if artifact_is_valid(metadata, "prediction_report"):
        # Serve only HTML produced for the active prediction lineage.
        response = await _try_serve_file_with_fallback(
            manager,
            session_id,
            key,
            report_path,
            media_type="text/html",
            artifact="prediction_report",
        )
        if response:
            return response

    if not artifact_is_valid(metadata, "raw_predictions"):
        raise HTTPException(
            status_code=404,
            detail="Prediction report inputs are not available for the current run",
        )

    prediction_lineage = await manager.capture_prediction_lineage(session_id)
    if prediction_lineage is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction report inputs are not available for the current run",
        )
    metadata = await manager.get_session_metadata(session_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Report doesn't exist anywhere - try to generate on-demand
    logger.info(f"Report not found for {session_id[:8]}, generating on-demand...")

    # Check if predictions exist
    predictions_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"

    # Always refresh canonical report inputs from the shared store.  Ordinary
    # sync intentionally skips existing local files, which can leave another
    # service instance holding inputs from an older prediction generation.
    refreshed_inputs: dict[Path, bool] = {}
    for key_to_refresh, local_path in (
        ("cache/predictions/predictions.jsonl", predictions_path),
        ("cache/dataset/dataset.jsonl", dataset_path),
    ):
        published_key = manager.resolve_published_artifact_key(
            metadata,
            key_to_refresh,
            legacy_key=key_to_refresh,
        )
        current_data = (
            await manager.read_from_store(session_id, published_key)
            if published_key is not None
            else None
        )
        if current_data is not None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path = local_path.with_suffix(f"{local_path.suffix}.refresh")
            pending_path.write_bytes(current_data)
            pending_path.replace(local_path)
        refreshed_inputs[local_path] = current_data is not None

    trust_local_inputs = (
        manager.store.kind == "local" and metadata.get("published_artifacts") is None
    )
    if not predictions_path.exists() or (
        not trust_local_inputs and not refreshed_inputs[predictions_path]
    ):
        raise HTTPException(status_code=404, detail="Predictions not available yet")

    if not dataset_path.exists() or (
        not trust_local_inputs and not refreshed_inputs[dataset_path]
    ):
        raise HTTPException(status_code=404, detail="Dataset not available")

    # Generate into a lineage-specific staging path.  The canonical local/store
    # file is published only after the lineage check succeeds under the session
    # metadata lock.
    staged_report: Path | None = None
    try:
        staged_report = await _generate_report_on_demand(
            session_dir,
            predictions_path,
            dataset_path,
            prediction_lineage,
        )
        logger.info(f"✓ Report generated on-demand for {session_id[:8]}")
    except Exception:
        log_durable_failure(
            logger,
            "material_prediction_report_publication_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Report generation failed",
        ) from None

    assert staged_report is not None
    try:
        report_published = (
            await manager.mark_prediction_report_valid_if_lineage_matches(
                session_id,
                prediction_lineage,
                staged_report,
                report_key=key,
            )
        )
    finally:
        shutil.rmtree(staged_report.parent, ignore_errors=True)
    if not report_published:
        raise HTTPException(
            status_code=409,
            detail=(
                "Prediction lineage changed while the report was generated; "
                "retry after the active run finishes"
            ),
        )

    # Resolve the CAS-published immutable pointer; the staging directory has
    # already been removed and no mutable canonical report is written.
    response = await _try_serve_file_with_fallback(
        manager,
        session_id,
        key,
        report_path,
        media_type="text/html",
        artifact="prediction_report",
    )
    if response is None:
        raise HTTPException(
            status_code=500,
            detail="Published prediction report is unavailable",
        )
    return response
