# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads and reports."""

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.held_file_response import HeldFileResponse

from ..session.cache_publications import (
    CACHE_NAMESPACES,
    PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD,
    PREDICTION_REPORT_PUBLICATION_ID_FIELD,
    parse_cache_publications,
    prediction_report_publication_key,
    prediction_report_publication_path,
)
from ..session.manager import (
    JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS,
    PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME,
    SessionManager,
)

logger = logging.getLogger(__name__)

ARTIFACT_STREAM_CHUNK_SIZE = 1024 * 1024

# Create router
router = APIRouter(prefix="/artifacts", tags=["artifacts"])

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


def _iter_stream_chunks(stream: BinaryIO) -> Iterator[bytes]:
    """Read a store-backed artifact in fixed-size chunks."""

    try:
        while chunk := stream.read(ARTIFACT_STREAM_CHUNK_SIZE):
            yield chunk
    finally:
        stream.close()


def _copy_stream_to_path(stream: BinaryIO, destination: Path) -> None:
    """Copy a bound store stream to one report-generation snapshot."""

    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(
                stream,
                output,
                length=ARTIFACT_STREAM_CHUNK_SIZE,
            )
    finally:
        stream.close()


async def _generate_report_on_demand(
    session_dir: Path, predictions_path: Path, dataset_path: Path
) -> None:
    """Generate prediction HTML report on-demand."""
    predictions = []
    with open(predictions_path) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    dataset = []
    with open(dataset_path) as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    import sys

    service_dir = Path(__file__).parent.parent.parent
    apps_dir = service_dir.parent
    repo_root = apps_dir.parent
    for path in [str(apps_dir), str(repo_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)

    from joint_agent.tasks.reporting import GeneratePredictionReportTask

    task = GeneratePredictionReportTask()

    report_context = {
        "predictions": predictions,
        "failed_predictions": [],
        "dataset": dataset,
        "output_dir": str(predictions_path.parent),
        "dataset_path": str(dataset_path),
    }

    await asyncio.to_thread(task.run, report_context, None)


async def _maintain_report_claim(
    manager: SessionManager,
    session_id: str,
    run_id: str,
) -> None:
    """Keep an on-demand legacy report generation claim alive."""

    while True:
        await asyncio.sleep(manager.run_claim_heartbeat_seconds)
        if not await manager.renew_run(session_id, run_id):
            return


async def _drain_report_operation(awaitable: Awaitable[Any]) -> Any:
    """Drain thread-backed report work before cancellation releases its claim."""

    task = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc

    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


async def _require_report_claim(
    manager: SessionManager,
    session_id: str,
    run_id: str,
    claim_task: asyncio.Task[None],
) -> None:
    """Renew the exact report claim immediately before publishing mutable data."""

    if claim_task.done() or not await manager.renew_run(session_id, run_id):
        raise HTTPException(
            status_code=409,
            detail="Prediction report generation lost session ownership",
        )


async def _cleanup_report_claim(
    manager: SessionManager,
    session_id: str,
    run_id: str,
    claim_task: asyncio.Task[None],
) -> None:
    """Stop the heartbeat and release the report claim as one drained cleanup."""

    claim_task.cancel()
    await asyncio.gather(claim_task, return_exceptions=True)
    await manager.release_run(session_id, run_id)


def _materialize_prediction_report_publication(
    session_dir: Path,
    run_id: str,
) -> Path:
    """Copy one trusted report into a run-unique local publication."""

    session_root = session_dir.resolve()
    source = session_dir / "cache" / "predictions" / "report.html"
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("Prediction report source is not a regular file")
    if not source.resolve().is_relative_to(session_root):
        raise RuntimeError("Prediction report source escapes the session directory")

    publication = prediction_report_publication_path(session_dir, run_id)
    publication.parent.mkdir(parents=True, exist_ok=False)
    if not publication.parent.resolve().is_relative_to(session_root):
        raise RuntimeError(
            "Prediction report publication escapes the session directory"
        )

    temporary = publication.with_name(f".{publication.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(publication)
    finally:
        temporary.unlink(missing_ok=True)
    return publication


async def _serve_artifact(
    manager: SessionManager,
    session_id: str,
    artifact_type: str,
    media_type: str,
    filename: str,
) -> FileResponse | StreamingResponse:
    """Serve an artifact from local disk or store (S3)."""
    if artifact_type.startswith("joint_rigger_"):
        metadata = await manager.get_session_metadata(session_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if metadata.get("status") != "completed":
            raise HTTPException(
                status_code=404,
                detail=f"{artifact_type.capitalize()} not available for current run",
            )
        artifact_flags = (metadata.get("results") or {}).get("joint_rigger_artifacts")
        if isinstance(artifact_flags, dict) and not artifact_flags.get(artifact_type):
            raise HTTPException(
                status_code=404,
                detail=f"{artifact_type.capitalize()} not available for current run",
            )

    joint_artifact = artifact_type in JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS
    immutable_local = (
        await manager.get_immutable_local_artifact_stream_with_filename(
            session_id,
            artifact_type,
        )
        if joint_artifact
        else None
    )
    if immutable_local is not None:
        local_artifact, local_filename = immutable_local
        served_filename = (
            local_filename if artifact_type == "joint_rigger_output" else filename
        )
        return HeldFileResponse(
            local_artifact,
            media_type=media_type,
            filename=served_filename,
        )

    # Fall back to store (S3 — works cross-instance)
    served_filename = filename
    if artifact_type == "joint_rigger_output":
        selected = await manager.get_artifact_stream_with_filename(
            session_id,
            artifact_type,
        )
        if selected is None:
            stream = None
        else:
            stream, served_filename = selected
    else:
        stream = await manager.get_artifact_stream(session_id, artifact_type)
    if stream:
        return StreamingResponse(
            _iter_stream_chunks(stream),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{served_filename}"'
            },
            background=BackgroundTask(stream.close),
        )

    raise HTTPException(
        status_code=404, detail=f"{artifact_type.capitalize()} not available"
    )


@router.get("/{session_id}/predictions")
async def download_predictions(session_id: str):
    """Download predictions JSONL file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "predictions",
        "application/x-ndjson",
        "predictions.jsonl",
    )


@router.get("/{session_id}/articulation-candidates")
async def download_articulation_candidates(session_id: str):
    """Download Stage 2 articulation candidates JSON."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "articulation_candidates",
        "application/json",
        "articulation_candidates.json",
    )


@router.get("/{session_id}/articulation-report")
async def view_articulation_report(session_id: str):
    """View Stage 2 articulation candidate HTML report."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    stream = await manager.get_artifact_stream(session_id, "articulation_report")
    if stream:
        return StreamingResponse(
            _iter_stream_chunks(stream),
            media_type="text/html",
            background=BackgroundTask(stream.close),
        )

    raise HTTPException(status_code=404, detail="Articulation report not available")


@router.get("/{session_id}/report")
async def view_prediction_report(session_id: str):
    """View prediction HTML report in browser.

    Generates the report on-demand if it doesn't exist yet.
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    bound_stream = await manager.get_artifact_stream(session_id, "prediction_report")
    if bound_stream:
        return StreamingResponse(
            _iter_stream_chunks(bound_stream),
            media_type="text/html",
            background=BackgroundTask(bound_stream.close),
        )

    report_run_id = uuid4().hex
    if not await manager.reserve_legacy_cache_run(session_id, report_run_id):
        raise HTTPException(status_code=404, detail="Prediction report not available")

    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "predictions" / "report.html"
    claim_task = asyncio.create_task(
        _maintain_report_claim(manager, session_id, report_run_id),
        name=f"prediction-report-claim-{session_id[:8]}",
    )
    try:
        metadata = await manager.get_session_metadata(session_id)
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=404, detail="Session not found")
        cache_publications = parse_cache_publications(metadata)
        if cache_publications is not None and not all(
            namespace in cache_publications for namespace in CACHE_NAMESPACES
        ):
            raise HTTPException(
                status_code=404, detail="Prediction report not available"
            )

        if cache_publications is None:
            pulled = await _drain_report_operation(
                manager.sync_from_store(
                    session_id,
                    prefix="cache/",
                    overwrite=True,
                )
            )
            if pulled > 0:
                logger.info(
                    f"Refreshed {pulled} legacy artifact(s) for report generation"
                )

        if cache_publications is not None or not report_path.exists():
            logger.info(
                f"Report not found for {session_id[:8]}, generating on-demand..."
            )

            predictions_path = (
                session_dir / "cache" / "predictions" / "predictions.jsonl"
            )
            dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"

            with TemporaryDirectory(
                prefix=".prediction-report-",
                dir=session_dir,
            ) as snapshot_dir_text:
                snapshot_dir = Path(snapshot_dir_text)
                snapshot_predictions = snapshot_dir / "predictions.jsonl"
                snapshot_dataset = snapshot_dir / "dataset.jsonl"
                if cache_publications is None:
                    if not predictions_path.exists():
                        raise HTTPException(
                            status_code=404,
                            detail="Predictions not available yet",
                        )
                    if not dataset_path.exists():
                        raise HTTPException(
                            status_code=404,
                            detail="Dataset not available",
                        )
                    await _drain_report_operation(
                        asyncio.to_thread(
                            shutil.copy2,
                            predictions_path,
                            snapshot_predictions,
                        )
                    )
                    await _drain_report_operation(
                        asyncio.to_thread(
                            shutil.copy2,
                            dataset_path,
                            snapshot_dataset,
                        )
                    )
                else:
                    predictions_stream = await manager.get_artifact_stream(
                        session_id,
                        "predictions",
                    )
                    if predictions_stream is None:
                        raise HTTPException(
                            status_code=404,
                            detail="Predictions not available yet",
                        )
                    await _drain_report_operation(
                        asyncio.to_thread(
                            _copy_stream_to_path,
                            predictions_stream,
                            snapshot_predictions,
                        )
                    )
                    dataset_stream = await manager.get_artifact_stream(
                        session_id,
                        "dataset",
                    )
                    if dataset_stream is None:
                        raise HTTPException(
                            status_code=404,
                            detail="Dataset not available",
                        )
                    await _drain_report_operation(
                        asyncio.to_thread(
                            _copy_stream_to_path,
                            dataset_stream,
                            snapshot_dataset,
                        )
                    )
                await _drain_report_operation(
                    _generate_report_on_demand(
                        snapshot_dir,
                        snapshot_predictions,
                        snapshot_dataset,
                    )
                )
                snapshot_report = snapshot_dir / "report.html"
                await _require_report_claim(
                    manager,
                    session_id,
                    report_run_id,
                    claim_task,
                )
                report_path.parent.mkdir(parents=True, exist_ok=True)
                snapshot_report.replace(report_path)
                logger.info(f"Report generated on-demand for {session_id[:8]}")

        await _require_report_claim(
            manager,
            session_id,
            report_run_id,
            claim_task,
        )
        await _drain_report_operation(
            asyncio.to_thread(
                _materialize_prediction_report_publication,
                session_dir,
                report_run_id,
            )
        )
        await _drain_report_operation(
            manager.sync_to_store(
                session_id,
                prefix=prediction_report_publication_key(report_run_id),
                overwrite=False,
            )
        )
        accepted = await _drain_report_operation(
            manager.terminalize_and_release_run(
                session_id,
                report_run_id,
                {
                    "status": "completed",
                    PREDICTION_REPORT_PUBLICATION_ID_FIELD: report_run_id,
                    PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD: cache_publications,
                },
            )
        )
        if not accepted:
            raise HTTPException(
                status_code=409,
                detail="Prediction report generation lost session ownership",
            )
    except HTTPException:
        raise
    except Exception:
        log_durable_failure(
            logger,
            "joint_prediction_report_publication_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Report generation failed",
        ) from None
    finally:
        await _drain_report_operation(
            _cleanup_report_claim(
                manager,
                session_id,
                report_run_id,
                claim_task,
            )
        )

    report_stream = await manager.get_artifact_stream(session_id, "prediction_report")
    if report_stream:
        return StreamingResponse(
            _iter_stream_chunks(report_stream),
            media_type="text/html",
            background=BackgroundTask(report_stream.close),
        )
    raise HTTPException(status_code=404, detail="Prediction report not available")


@router.get("/{session_id}/dataset")
async def download_dataset(session_id: str):
    """Download dataset JSONL file."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "dataset",
        "application/x-ndjson",
        "dataset.jsonl",
    )


@router.get("/{session_id}/joint-rigger-output")
async def download_joint_rigger_output(session_id: str):
    """Download the generated USD from the Joint Rigger apply step."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "joint_rigger_output",
        "application/octet-stream",
        PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME,
    )


@router.get("/{session_id}/joint-rigger-diagnostics")
async def download_joint_rigger_diagnostics(session_id: str):
    """Download Joint Rigger diagnostics JSON."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "joint_rigger_diagnostics",
        "application/json",
        "joint_rigger_diagnostics.json",
    )


@router.get("/{session_id}/joint-rigger-validation")
async def download_joint_rigger_validation(session_id: str):
    """Download Joint Rigger generated-USD validation JSON."""
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return await _serve_artifact(
        manager,
        session_id,
        "joint_rigger_validation",
        "application/json",
        "joint_rigger_validation.json",
    )
