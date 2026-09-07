# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifacts API endpoints - Downloads and reports."""

import asyncio
import json
import logging
import shutil
import tempfile
import zipfile
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from world_understanding.functions.graphics.so_export import (
    legacy_portable_sidecar_name,
    portable_sidecar_name,
)
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    OpenArtifactFile,
    iter_open_regular_files,
    open_confined_directory,
    open_regular_file_no_follow,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.held_file_response import (
    HeldFileResponse,
    open_held_artifact_file,
)

from ..session.manager import SessionManager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/artifacts", tags=["artifacts"])

_USD_MEDIA_TYPES = {
    ".usd": "application/octet-stream",
    ".usda": "text/plain",
    ".usdc": "application/octet-stream",
    ".usdz": "model/vnd.usdz+zip",
}
_ZIP_MEDIA_TYPE = "application/zip"
_ZIP_COPY_CHUNK_SIZE = 1024 * 1024
_INSPECTABLE_USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})

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

    from physics_agent.tasks.reporting import GeneratePredictionReportTask

    task = GeneratePredictionReportTask()

    report_context = {
        "predictions": predictions,
        "failed_predictions": [],
        "dataset": dataset,
        "output_dir": str(predictions_path.parent),
        "dataset_path": str(dataset_path),
    }

    await asyncio.to_thread(task.run, report_context, None)


async def _serve_s3_prediction_report(
    manager: SessionManager,
    session_id: str,
) -> HeldFileResponse:
    """Render from one immutable publication without touching live worker state."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="physics-report-snapshot-"))
    report_artifact: OpenArtifactFile | None = None
    try:
        await manager.store.sync_to_local(
            session_id,
            str(snapshot_dir),
            prefix=(
                "cache/predictions/",
                "cache/dataset/dataset.jsonl",
            ),
        )
        report_key = "cache/predictions/report.html"
        report_path = snapshot_dir / report_key
        predictions_path = snapshot_dir / "cache" / "predictions" / "predictions.jsonl"
        dataset_path = snapshot_dir / "cache" / "dataset" / "dataset.jsonl"
        if not report_path.exists():
            if not predictions_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail="Predictions not available yet",
                )
            if not dataset_path.exists():
                raise HTTPException(status_code=404, detail="Dataset not available")

            await _generate_report_on_demand(
                snapshot_dir,
                predictions_path,
                dataset_path,
            )
        report_artifact = open_held_artifact_file(snapshot_dir, report_key)
        response = HeldFileResponse(
            report_artifact,
            media_type="text/html",
            background=BackgroundTask(_cleanup_temp_tree, snapshot_dir),
        )
        report_artifact = None
        return response
    except HTTPException:
        if report_artifact is not None:
            report_artifact.stream.close()
        _cleanup_temp_tree(snapshot_dir)
        raise
    except Exception:
        if report_artifact is not None:
            report_artifact.stream.close()
        _cleanup_temp_tree(snapshot_dir)
        log_durable_failure(
            logger,
            "physics_prediction_report_publication_failed",
            phase=FailurePhase.LOCAL_PUBLICATION,
            retryable=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Report generation failed",
        ) from None
    except BaseException:
        if report_artifact is not None:
            report_artifact.stream.close()
        _cleanup_temp_tree(snapshot_dir)
        raise


async def _serve_artifact(
    manager: SessionManager,
    session_id: str,
    artifact_type: str,
    media_type: str,
    filename: str,
) -> FileResponse | StreamingResponse:
    """Serve an artifact from local disk or store (S3)."""
    # S3 sessions must resolve through one committed immutable manifest. A
    # replica-local path may belong to an older generation after failover.
    if getattr(manager.store, "kind", "local") != "s3":
        local_artifact = await manager.get_local_artifact_stream(
            session_id,
            artifact_type,
        )
        if local_artifact is not None:
            artifact, _ = local_artifact
            return HeldFileResponse(
                artifact,
                media_type=media_type,
                filename=filename,
            )

    # Fall back to store (S3 — works cross-instance)
    stream = await manager.get_artifact_stream(session_id, artifact_type)
    if stream:
        return StreamingResponse(
            stream,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            background=BackgroundTask(stream.close),
        )

    raise HTTPException(
        status_code=404, detail=f"{artifact_type.capitalize()} not available"
    )


def _usd_media_type(path_or_key: str | Path) -> str:
    return _USD_MEDIA_TYPES.get(
        Path(path_or_key).suffix.lower(),
        "application/octet-stream",
    )


def _new_temp_zip_path() -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    try:
        return Path(handle.name)
    finally:
        handle.close()


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to remove temporary artifact bundle %s", path)


def _cleanup_temp_tree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to remove temporary artifact snapshot %s", path)


def _zip_file_response(zip_path: Path, filename: str) -> FileResponse:
    return FileResponse(
        zip_path,
        media_type=_ZIP_MEDIA_TYPE,
        filename=filename,
        background=BackgroundTask(_cleanup_temp_file, zip_path),
    )


def _output_usd_bundle_filename(output_name: str) -> str:
    return f"{Path(output_name).stem}_bundle.zip"


def _local_output_sidecar_dir(output_path: Path) -> Path | None:
    """Return the first existing sidecar actually authored by ``output_path``."""
    current = output_path.parent / portable_sidecar_name(output_path)
    legacy = output_path.parent / legacy_portable_sidecar_name(output_path)
    candidates = [
        candidate
        for candidate in dict.fromkeys((current, legacy))
        if candidate.exists() or candidate.is_symlink()
    ]
    if not candidates:
        return None
    try:
        with open_regular_file_no_follow(output_path) as (stream, _metadata):
            for candidate in candidates:
                if not _stream_authors_sidecar_directory(
                    stream,
                    output_path.name,
                    candidate.name,
                ):
                    continue
                if candidate.is_symlink() or not candidate.is_dir():
                    raise ArtifactPathError(
                        "Output USD authors an unsafe sidecar directory"
                    )
                return candidate
    except (ArtifactPathError, OSError, RuntimeError, ValueError):
        logger.warning("Could not safely inspect output USD for sidecars")
        raise
    return None


def _validate_archive_relpath(path: PurePosixPath) -> PurePosixPath:
    if path.is_absolute() or not path.parts:
        raise ValueError(f"Unsafe ZIP archive path: {path}")

    for part in path.parts:
        if part in {"", ".", ".."} or "\\" in part or ":" in part:
            raise ValueError(f"Unsafe ZIP archive path: {path}")

    return path


def _archive_name_for_output_file(output_name: str) -> str:
    return _validate_archive_relpath(PurePosixPath(output_name)).as_posix()


def _archive_name_for_sidecar(sidecar_dir_name: str, sidecar_rel: PurePosixPath) -> str:
    safe_rel = _validate_archive_relpath(sidecar_rel)
    archive_path = PurePosixPath(sidecar_dir_name) / safe_rel
    return _validate_archive_relpath(archive_path).as_posix()


def _copy_stream_to_archive(
    archive: zipfile.ZipFile,
    archive_name: str,
    stream: BinaryIO,
) -> None:
    """Copy one stream into a ZIP entry without an unbounded read."""

    with archive.open(archive_name, "w", force_zip64=True) as destination:
        shutil.copyfileobj(stream, destination, length=_ZIP_COPY_CHUNK_SIZE)


def _authored_asset_uses_sidecar_directory(
    authored_path: str,
    sidecar_name: str,
) -> bool:
    """Return whether one canonical relative asset path enters ``sidecar_name``."""

    try:
        path = _validate_archive_relpath(PurePosixPath(authored_path))
    except ValueError:
        return False
    return len(path.parts) > 1 and path.parts[0] == sidecar_name


def _stream_authors_sidecar_directory(
    stream: BinaryIO,
    output_name: str,
    sidecar_name: str,
) -> bool:
    """Inspect one held USD layer for authored paths into a legacy sidecar.

    The layer is staged with bounded memory and opened only as an anonymous Sdf
    layer. ``ModifyAssetPaths`` visits asset attributes and composition arcs
    without opening or resolving any referenced dependency layers. The caller
    retains stream ownership and its original position.
    """

    safe_output_name = PurePosixPath(output_name).name
    if Path(safe_output_name).suffix.lower() not in _INSPECTABLE_USD_SUFFIXES:
        raise ArtifactPathError(
            "Output format cannot be inspected for legacy-sidecar references"
        )

    try:
        original_position = stream.tell()
        stream.seek(0)
    except (OSError, ValueError) as exc:
        raise ArtifactPathError(
            "Output USD stream cannot be inspected for legacy-sidecar references"
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="physics-usd-inspect-") as temp_dir:
            staged_path = Path(temp_dir) / safe_output_name
            with staged_path.open("wb") as staged:
                shutil.copyfileobj(
                    stream,
                    staged,
                    length=_ZIP_COPY_CHUNK_SIZE,
                )

            from pxr import Sdf, Usd, UsdUtils

            # Importing Usd registers the USD/USDA/USDC Sdf file formats.
            _ = Usd.GetVersion()
            layer = Sdf.Layer.OpenAsAnonymous(str(staged_path))
            if layer is None:
                raise ArtifactPathError(
                    "Output USD layer could not be inspected for legacy sidecars"
                )

            found = False

            def inspect_path(authored_path: str) -> str:
                nonlocal found
                if _authored_asset_uses_sidecar_directory(
                    authored_path,
                    sidecar_name,
                ):
                    found = True
                return authored_path

            UsdUtils.ModifyAssetPaths(
                layer,
                inspect_path,
                keepEmptyPathsInArrays=True,
            )
            return found
    except Exception as exc:
        raise ArtifactPathError(
            "Output USD layer could not be inspected for legacy sidecars"
        ) from exc
    finally:
        stream.seek(original_position)


def _write_local_output_usd_bundle(output_path: Path) -> Path | None:
    sidecar_dir = _local_output_sidecar_dir(output_path)
    if sidecar_dir is None:
        return None

    sidecar_files = sorted(path for path in sidecar_dir.rglob("*") if path.is_file())
    if not sidecar_files:
        return None

    zip_path = _new_temp_zip_path()
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_path, _archive_name_for_output_file(output_path.name))
            for path in sidecar_files:
                rel = PurePosixPath(path.relative_to(sidecar_dir).as_posix())
                try:
                    archive_name = _archive_name_for_sidecar(sidecar_dir.name, rel)
                except ValueError:
                    logger.warning("Skipping unsafe sidecar archive path: %s", path)
                    continue
                archive.write(path, archive_name)
    except Exception:
        _cleanup_temp_file(zip_path)
        raise

    return zip_path


def _write_open_output_usd_bundle(
    storage_root: Path,
    session_id: str,
    output_artifact: OpenArtifactFile,
    relative_key: str,
) -> Path | None:
    """Bundle one held output and safely traversed sidecars, if any exist."""

    output_name = PurePosixPath(relative_key).name
    sidecar_names = (
        portable_sidecar_name(relative_key),
        legacy_portable_sidecar_name(relative_key),
    )
    with open_confined_directory(storage_root) as root_descriptor:
        sidecars = None
        try:
            first_sidecar = None
            sidecar_name = sidecar_names[0]
            sidecar_prefix = ""
            for candidate_name in dict.fromkeys(sidecar_names):
                if not _stream_authors_sidecar_directory(
                    output_artifact.stream,
                    output_name,
                    candidate_name,
                ):
                    continue
                candidate_prefix = (
                    f"{session_id}/{PurePosixPath(relative_key).parent.as_posix()}/"
                    f"{candidate_name}/"
                )
                candidate_sidecars = iter_open_regular_files(
                    root_descriptor,
                    prefix=candidate_prefix,
                )
                try:
                    first_sidecar = next(candidate_sidecars, None)
                except BaseException:
                    candidate_sidecars.close()
                    raise
                if first_sidecar is not None:
                    sidecar_name = candidate_name
                    sidecar_prefix = candidate_prefix
                    sidecars = candidate_sidecars
                    break
                candidate_sidecars.close()
            if first_sidecar is None:
                return None
            assert sidecars is not None

            zip_path = _new_temp_zip_path()
            try:
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                    _copy_stream_to_archive(
                        archive,
                        _archive_name_for_output_file(output_name),
                        output_artifact.stream,
                    )
                    for sidecar in chain((first_sidecar,), sidecars):
                        sidecar_relative = PurePosixPath(
                            sidecar.relative_key.removeprefix(sidecar_prefix)
                        )
                        _copy_stream_to_archive(
                            archive,
                            _archive_name_for_sidecar(
                                sidecar_name,
                                sidecar_relative,
                            ),
                            sidecar.stream,
                        )
            except Exception:
                _cleanup_temp_file(zip_path)
                raise
            return zip_path
        finally:
            if sidecars is not None:
                sidecars.close()


def _store_output_sidecar_prefix(output_key: str) -> str:
    key_path = PurePosixPath(output_key)
    sidecar_dir = key_path.parent / portable_sidecar_name(output_key)
    return f"{sidecar_dir.as_posix().rstrip('/')}/"


def _legacy_store_output_sidecar_prefix(output_key: str) -> str:
    key_path = PurePosixPath(output_key)
    sidecar_dir = key_path.parent / legacy_portable_sidecar_name(output_key)
    return f"{sidecar_dir.as_posix().rstrip('/')}/"


async def _list_store_output_sidecar_keys(
    manager: SessionManager,
    session_id: str,
    output_key: str,
) -> list[str]:
    prefixes = tuple(
        dict.fromkeys(
            (
                _store_output_sidecar_prefix(output_key),
                _legacy_store_output_sidecar_prefix(output_key),
            )
        )
    )
    stream: BinaryIO | None = None
    try:
        for prefix in prefixes:
            keys = sorted(await manager.store.list_keys(session_id, prefix=prefix))
            if not keys:
                continue
            if stream is None:
                stream = await manager.store.open_read(session_id, output_key)
            if _stream_authors_sidecar_directory(
                stream,
                PurePosixPath(output_key).name,
                PurePosixPath(prefix.rstrip("/")).name,
            ):
                return keys
    finally:
        if stream is not None:
            stream.close()
    return []


def _archive_name_for_store_sidecar(output_key: str, sidecar_key: str) -> str:
    output_path = PurePosixPath(output_key)
    sidecar_path = PurePosixPath(sidecar_key)
    for sidecar_name in dict.fromkeys(
        (
            portable_sidecar_name(output_key),
            legacy_portable_sidecar_name(output_key),
        )
    ):
        sidecar_dir = output_path.parent / sidecar_name
        try:
            sidecar_rel = sidecar_path.relative_to(sidecar_dir)
        except ValueError:
            continue
        return _archive_name_for_sidecar(sidecar_dir.name, sidecar_rel)
    raise ValueError(f"Sidecar key does not match output USD: {sidecar_key}")


async def _write_store_output_usd_bundle(
    manager: SessionManager,
    session_id: str,
    output_key: str,
    sidecar_keys: list[str],
) -> Path:
    zip_path = _new_temp_zip_path()
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            stream = await manager.store.open_read(session_id, output_key)
            try:
                _copy_stream_to_archive(
                    archive,
                    _archive_name_for_output_file(PurePosixPath(output_key).name),
                    stream,
                )
            finally:
                stream.close()

            for sidecar_key in sidecar_keys:
                try:
                    archive_name = _archive_name_for_store_sidecar(
                        output_key,
                        sidecar_key,
                    )
                except ValueError:
                    logger.warning(
                        "Skipping unsafe sidecar store key in output bundle: %s",
                        sidecar_key,
                    )
                    continue

                stream = await manager.store.open_read(session_id, sidecar_key)
                try:
                    _copy_stream_to_archive(archive, archive_name, stream)
                finally:
                    stream.close()
    except BaseException:
        _cleanup_temp_file(zip_path)
        raise

    return zip_path


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


@router.get("/{session_id}/report")
async def view_prediction_report(session_id: str):
    """View prediction HTML report in browser.

    Generates the report on-demand if it doesn't exist yet.
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    if getattr(manager.store, "kind", "local") == "s3":
        return await _serve_s3_prediction_report(manager, session_id)

    session_dir = manager.get_session_dir(session_id)
    report_path = session_dir / "cache" / "predictions" / "report.html"

    if not report_path.exists():
        logger.info(f"Report not found for {session_id[:8]}, generating on-demand...")

        predictions_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
        dataset_path = session_dir / "cache" / "dataset" / "dataset.jsonl"

        # Pull from store (S3) if files are missing locally (cross-instance case)
        if not predictions_path.exists() or not dataset_path.exists():
            pulled = await manager.sync_from_store(session_id, prefix="cache/")
            if pulled > 0:
                logger.info(
                    f"Pulled {pulled} artifact(s) from store for report generation"
                )

        if not predictions_path.exists():
            raise HTTPException(status_code=404, detail="Predictions not available yet")

        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not available")

        try:
            await _generate_report_on_demand(
                session_dir, predictions_path, dataset_path
            )
            logger.info(f"Report generated on-demand for {session_id[:8]}")
        except Exception:
            log_durable_failure(
                logger,
                "physics_prediction_report_publication_failed",
                phase=FailurePhase.LOCAL_PUBLICATION,
                retryable=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Report generation failed",
            ) from None

    report_artifact = await manager.open_local_artifact_key(
        session_id,
        "cache/predictions/report.html",
    )
    if report_artifact is None:
        raise HTTPException(status_code=404, detail="Prediction report not available")
    return HeldFileResponse(report_artifact, media_type="text/html")


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


async def _serve_s3_output_usd_snapshot(
    manager: SessionManager,
    session_id: str,
) -> Response:
    """Serve root USD and sidecars from one pinned publication manifest."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="physics-output-snapshot-"))
    output_artifact: OpenArtifactFile | None = None
    try:
        expected_suffix = await manager._expected_output_usd_suffix(
            session_id,
            snapshot_dir,
        )
        await manager.store.sync_to_local(
            session_id,
            str(snapshot_dir),
            prefix="cache/physics/",
        )
        candidates: list[tuple[OpenArtifactFile, str]] = []
        for suffix in _USD_MEDIA_TYPES:
            relative_key = f"cache/physics/scene_physics{suffix}"
            try:
                artifact = open_held_artifact_file(snapshot_dir, relative_key)
            except (OSError, RuntimeError, ValueError):
                continue
            candidates.append((artifact, relative_key))
        if not candidates:
            raise HTTPException(status_code=404, detail="Output USD not available")

        # Download mtimes reflect transfer order, not authoring order. Match
        # the manager's deterministic USD suffix preference instead.
        output_artifact, relative_key = min(
            candidates,
            key=lambda item: (
                PurePosixPath(item[1]).suffix.lower() != expected_suffix,
                tuple(_USD_MEDIA_TYPES).index(PurePosixPath(item[1]).suffix.lower()),
            ),
        )
        for candidate, _candidate_key in candidates:
            if candidate is not output_artifact:
                candidate.stream.close()

        filename = PurePosixPath(relative_key).name
        bundle_path = _write_open_output_usd_bundle(
            snapshot_dir.parent,
            snapshot_dir.name,
            output_artifact,
            relative_key,
        )
        if bundle_path is not None:
            output_artifact.stream.close()
            output_artifact = None
            _cleanup_temp_tree(snapshot_dir)
            return _zip_file_response(
                bundle_path,
                _output_usd_bundle_filename(filename),
            )
        response = HeldFileResponse(
            output_artifact,
            media_type=_usd_media_type(filename),
            filename=filename,
            background=BackgroundTask(_cleanup_temp_tree, snapshot_dir),
        )
        output_artifact = None
        return response
    except HTTPException:
        if output_artifact is not None:
            output_artifact.stream.close()
        _cleanup_temp_tree(snapshot_dir)
        raise
    except BaseException:
        if output_artifact is not None:
            output_artifact.stream.close()
        _cleanup_temp_tree(snapshot_dir)
        raise


@router.get("/{session_id}/output-usd")
async def download_output_usd(session_id: str) -> Response:
    # Annotated as the starlette Response base class rather than the
    # concrete `FileResponse | StreamingResponse` union — FastAPI treats
    # endpoint-level union return annotations as a pydantic response_model
    # and fails at route registration ("Invalid args for response field").
    # Response covers both subclasses for MyPy while letting FastAPI pass
    # the result through untouched.
    """Download the simulation-ready USD written by the apply_physics step.

    Returned only when the pipeline has completed with apply_physics enabled
    (the service default). USD, USDA, and USDC inputs keep their suffix; USDZ
    inputs default to USDA so runtime-resolved MDL shader references remain
    asset paths. When flattened outputs localize dependencies beside the root,
    this endpoint returns a ZIP bundle containing the root layer and sidecar
    asset directory; otherwise it returns the single USD artifact. The output
    is augmented with UsdPhysics schemas (RigidBodyAPI, CollisionAPI, MassAPI,
    MaterialAPI) on each predicted prim, plus a PhysicsScene. Consumable by
    PhysX / Isaac.
    """
    manager = get_session_manager()

    if not await manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    if getattr(manager.store, "kind", "local") == "s3":
        return await _serve_s3_output_usd_snapshot(manager, session_id)

    local_output = (
        await manager.get_local_artifact_stream(session_id, "output_usd")
        if getattr(manager.store, "kind", "local") != "s3"
        else None
    )
    if local_output is not None:
        output_artifact, relative_key = local_output
        filename = PurePosixPath(relative_key).name
        try:
            try:
                bundle_path = _write_open_output_usd_bundle(
                    manager.storage_path,
                    session_id,
                    output_artifact,
                    relative_key,
                )
            except ArtifactPathError:
                logger.warning("Refusing unsafe local output USD sidecar tree")
                raise
            if bundle_path:
                output_artifact.stream.close()
                return _zip_file_response(
                    bundle_path,
                    _output_usd_bundle_filename(filename),
                )
            return HeldFileResponse(
                output_artifact,
                media_type=_usd_media_type(filename),
                filename=filename,
            )
        except BaseException:
            output_artifact.stream.close()
            raise

    keys = await manager.list_artifact_keys(session_id, "output_usd")
    if keys:
        key = keys[0]
        filename = PurePosixPath(key).name
        sidecar_keys = await _list_store_output_sidecar_keys(manager, session_id, key)
        if sidecar_keys:
            bundle_path = await _write_store_output_usd_bundle(
                manager,
                session_id,
                key,
                sidecar_keys,
            )
            return _zip_file_response(
                bundle_path,
                _output_usd_bundle_filename(filename),
            )

        stream = await manager.get_artifact_stream(
            session_id,
            "output_usd",
            key=key,
        )
        if stream:
            return StreamingResponse(
                stream,
                media_type=_usd_media_type(filename),
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                background=BackgroundTask(stream.close),
            )

    raise HTTPException(status_code=404, detail="Output USD not available")
