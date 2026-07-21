# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for joint agent pipeline executions.

Delegates all persistence to a pluggable SessionStore (local or S3).
All public methods are async.
"""

import asyncio
import json
import logging
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from world_understanding.utils.artifacts import (
    ArtifactPathError,
    OpenArtifactFile,
    open_held_confined_artifact,
    remove_confined_tree,
)
from world_understanding.utils.durable_diagnostics import (
    FailurePhase,
    log_durable_failure,
)
from world_understanding.utils.session_paths import (
    confined_session_path,
    safe_listed_session_ids,
)

from ..progress import (
    SERVICE_DEFAULT_TOTAL_STEPS,
    STEP_COMPLETION_PERCENTS,
    STEP_NUMBERS,
    step_display_name,
    step_overall_percent,
)
from ..storage import LocalSessionStore, SessionStore
from ..storage.base import METADATA_KEY
from .cache_publications import (
    CACHE_NAMESPACES,
    CACHE_PUBLICATIONS_FIELD,
    PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD,
    PREDICTION_REPORT_PUBLICATION_ID_FIELD,
    bound_cache_artifact_key,
    parse_cache_publications,
    prediction_report_publication_key,
)

logger = logging.getLogger(__name__)

# Session IDs are server-generated UUID4 strings but are also accepted back from
# URL path parameters (e.g. GET /sessions/{id}/...), so they must be validated
# before reaching any code that builds a filesystem path or storage key from
# them. The pattern is intentionally case-insensitive to tolerate normal UUID
# casing variance; it still rejects `../`, `/`, empty, and non-hex inputs.
_SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME = "rigged.usdz"
JOINT_RIGGER_OUTPUT_FILENAMES = (
    PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME,
    "rigged.usd",
)
JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD = "joint_rigger_artifact_keys"
JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD = "joint_rigger_publication_id"
JOINT_RIGGER_PUBLICATION_PREFIX = "artifacts/joint_rigger"
_JOINT_RIGGER_PUBLICATION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ACTIVE_RUN_ID_FIELD = "active_run_id"
ACTIVE_RUN_EXPIRES_AT_FIELD = "active_run_expires_at"
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_METADATA_CAS_ATTEMPTS = 16
DEFAULT_RUN_CLAIM_LEASE_SECONDS = 300.0
DEFAULT_RUN_CLAIM_HEARTBEAT_SECONDS = 60.0
JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS: dict[str, tuple[str, ...]] = {
    "joint_rigger_output": tuple(
        f"cache/joint_rigger/{filename}" for filename in JOINT_RIGGER_OUTPUT_FILENAMES
    ),
    "joint_rigger_diagnostics": ("cache/joint_rigger/joint_rigger_diagnostics.json",),
    "joint_rigger_validation": ("cache/joint_rigger/joint_rigger_validation.json",),
}
_ARTIFACT_RELATIVE_PATHS: dict[str, tuple[str, ...]] = {
    "predictions": ("cache/predictions/predictions.jsonl",),
    "prediction_report": ("cache/predictions/report.html",),
    "articulation_candidates": ("cache/predictions/articulation_candidates.json",),
    "articulation_report": ("cache/predictions/articulation_candidates.html",),
    "dataset": ("cache/dataset/dataset.jsonl",),
    **JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS,
}


class InvalidSessionIdError(ValueError):
    """Raised when a session_id fails format validation.

    Subclasses ValueError so existing `except ValueError` / `pytest.raises(ValueError)`
    keeps working, but lets FastAPI's exception handler target just this class
    instead of swallowing every ValueError in the app.
    """


def _validate_session_id(session_id: str) -> str:
    """Validate that session_id has UUID shape; reject otherwise."""
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise InvalidSessionIdError(f"Invalid session_id: {session_id!r}")
    return session_id


def _observed_total_steps(metadata: dict[str, Any]) -> int:
    """Return the largest configured step number already observed by metadata."""
    progress = metadata.get("overall_progress", {})
    total_steps = int(progress.get("total_steps") or SERVICE_DEFAULT_TOTAL_STEPS)
    for completed_step in metadata.get("completed_steps", []):
        if not isinstance(completed_step, dict):
            continue
        step_number = STEP_NUMBERS.get(str(completed_step.get("name", "")))
        if step_number is not None and step_number > total_steps:
            total_steps = step_number
    current_step = metadata.get("current_step")
    if isinstance(current_step, dict):
        step_number = STEP_NUMBERS.get(str(current_step.get("name", "")))
        if step_number is not None and step_number > total_steps:
            total_steps = step_number
    return total_steps


def _artifact_publication_marker(metadata: dict[str, Any]) -> tuple[str, str, str]:
    """Return a stable identity for the completed artifact publication."""

    results = metadata.get("results")
    if isinstance(results, dict):
        publication_id = results.get(JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD)
        if isinstance(
            publication_id, str
        ) and _JOINT_RIGGER_PUBLICATION_ID_PATTERN.fullmatch(publication_id):
            return ("publication", publication_id, "")
    return (
        "metadata",
        str(metadata.get("completed_at") or ""),
        str(metadata.get("updated_at") or ""),
    )


def _legacy_cache_marker(metadata: dict[str, Any]) -> tuple[str, str, str]:
    """Identify one idle legacy cache snapshot for post-open revalidation."""

    return (
        "legacy-cache",
        str(metadata.get("completed_at") or ""),
        str(metadata.get("updated_at") or ""),
    )


class SessionManager:
    """Manages pipeline sessions and their artifacts.

    Wraps a SessionStore for persistence and keeps a local directory
    for pipeline working data (GPU rendering needs fast local I/O).
    """

    def __init__(
        self,
        storage_path: Path | str,
        ttl_hours: int = 24,
        store: SessionStore | None = None,
        run_claim_lease_seconds: float = DEFAULT_RUN_CLAIM_LEASE_SECONDS,
        run_claim_heartbeat_seconds: float = DEFAULT_RUN_CLAIM_HEARTBEAT_SECONDS,
    ):
        if not math.isfinite(run_claim_lease_seconds) or run_claim_lease_seconds <= 0:
            raise ValueError("run_claim_lease_seconds must be finite and positive")
        if (
            not math.isfinite(run_claim_heartbeat_seconds)
            or run_claim_heartbeat_seconds <= 0
        ):
            raise ValueError("run_claim_heartbeat_seconds must be finite and positive")
        if run_claim_heartbeat_seconds >= run_claim_lease_seconds:
            raise ValueError(
                "run_claim_heartbeat_seconds must be shorter than the run claim lease"
            )
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self.store = store or LocalSessionStore(root_dir=str(self.storage_path))
        self.run_claim_lease_seconds = float(run_claim_lease_seconds)
        self.run_claim_heartbeat_seconds = float(run_claim_heartbeat_seconds)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock for safe read-modify-write."""
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def create_session(
        self, session_id: str, config: dict[str, Any] | None = None
    ) -> Path:
        """Create a new session with local dirs and store entry."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)

        # Create local directory structure (pipeline needs fast local I/O)
        (session_dir / "input").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "dataset").mkdir(parents=True, exist_ok=True)
        (session_dir / "cache" / "predictions").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)

        # Initialize store entry
        await self.store.init_session(session_id)

        metadata: dict[str, Any] = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": SERVICE_DEFAULT_TOTAL_STEPS,
                "percent": 0,
                "estimated_remaining_seconds": None,
            },
            "preview_images": [],
            CACHE_PUBLICATIONS_FIELD: {},
            "can_cancel": True,
            "elapsed_seconds": 0,
            "config": config or {},
            "ttl_expires_at": (
                datetime.now(UTC) + timedelta(hours=self.ttl_hours)
            ).isoformat(),
        }

        await self.store.put_json(session_id, METADATA_KEY, metadata)
        logger.info(f"Created session: {session_id}")
        return session_dir

    def get_session_dir(self, session_id: str) -> Path:
        """Get path to local session directory."""
        return confined_session_path(
            self.storage_path,
            _validate_session_id(session_id),
        )

    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists in the store."""
        session_id = _validate_session_id(session_id)
        return await self.store.exists(session_id, METADATA_KEY)

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata from store."""
        session_id = _validate_session_id(session_id)
        return await self.store.get_json(session_id, METADATA_KEY)

    async def reserve_run(self, session_id: str, run_id: str) -> bool:
        """Atomically reserve or reclaim one leased run across service instances."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return False
                encoded, metadata = document
                claim = self._parse_active_run_claim(metadata)
                if claim is None:
                    if self._has_active_run_claim_fields(metadata):
                        return False
                elif claim[1] > self._claim_now():
                    return False

                self._apply_metadata_updates(metadata, {})
                self._set_active_run_claim(metadata, run_id)
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return True
            raise RuntimeError(
                f"Could not reserve contended session metadata for {session_id}"
            )

    async def reserve_legacy_cache_run(self, session_id: str, run_id: str) -> bool:
        """Atomically reserve one completed, unclaimed cache snapshot for a report."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return False
                encoded, metadata = document
                cache_publications = parse_cache_publications(metadata)
                has_complete_cache_snapshot = cache_publications is None or all(
                    namespace in cache_publications for namespace in CACHE_NAMESPACES
                )
                if (
                    metadata.get("status") != "completed"
                    or not has_complete_cache_snapshot
                ):
                    return False
                claim = self._parse_active_run_claim(metadata)
                if claim is None:
                    if self._has_active_run_claim_fields(metadata):
                        return False
                elif claim[1] > self._claim_now():
                    return False

                self._apply_metadata_updates(metadata, {})
                self._set_active_run_claim(metadata, run_id)
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return True
            raise RuntimeError(
                f"Could not reserve legacy cache metadata for {session_id}"
            )

    async def renew_run(self, session_id: str, run_id: str) -> bool:
        """Extend a run lease only while the same generation still owns it."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return False
                encoded, metadata = document
                claim = self._parse_active_run_claim(metadata)
                if claim is None or claim[0] != run_id or claim[1] <= self._claim_now():
                    return False

                self._set_active_run_claim(metadata, run_id)
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return True
            raise RuntimeError(f"Could not renew contended run claim for {session_id}")

    async def terminalize_and_release_run(
        self,
        session_id: str,
        run_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Commit terminal metadata and release an exact, possibly expired claim."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return False
                encoded, metadata = document
                claim = self._parse_active_run_claim(metadata)
                if claim is None or claim[0] != run_id:
                    return False
                if (
                    metadata.get("status") == "cancelling"
                    and updates.get("status") != "cancelled"
                ):
                    return False

                self._apply_metadata_updates(metadata, updates)
                self._clear_active_run_claim(metadata)
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return True
            raise RuntimeError(
                f"Could not terminalize contended session metadata for {session_id}"
            )

    async def is_run_current(self, session_id: str, run_id: str) -> bool:
        """Return whether the run owns the unexpired metadata lease."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        document = await self._read_metadata_document(session_id)
        if document is None:
            return False
        claim = self._parse_active_run_claim(document[1])
        return bool(claim and claim[0] == run_id and claim[1] > self._claim_now())

    async def update_session_for_run(
        self,
        session_id: str,
        run_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Apply metadata only while the accepted run still owns the session."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return False
                encoded, metadata = document
                claim = self._parse_active_run_claim(metadata)
                if claim is None or claim[0] != run_id or claim[1] <= self._claim_now():
                    return False
                self._apply_metadata_updates(metadata, updates)
                self._set_active_run_claim(metadata, run_id)
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return True
            raise RuntimeError(
                f"Could not update contended session metadata for {session_id}"
            )

    async def release_run(self, session_id: str, run_id: str) -> bool:
        """Release the active claim only when it is still owned by this run."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return False
                encoded, metadata = document
                claim = self._parse_active_run_claim(metadata)
                if claim is None or claim[0] != run_id:
                    return False

                self._clear_active_run_claim(metadata)
                self._apply_metadata_updates(metadata, {})
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return True
            raise RuntimeError(
                f"Could not release contended session metadata for {session_id}"
            )

    @staticmethod
    def _claim_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _parse_active_run_claim(
        metadata: dict[str, Any],
    ) -> tuple[str, datetime] | None:
        run_id = metadata.get(ACTIVE_RUN_ID_FIELD)
        expires_at_text = metadata.get(ACTIVE_RUN_EXPIRES_AT_FIELD)
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
            return None
        if not isinstance(expires_at_text, str):
            return None
        try:
            expires_at = datetime.fromisoformat(expires_at_text)
            if expires_at.tzinfo is None:
                return None
            return run_id, expires_at.astimezone(UTC)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _has_active_run_claim_fields(metadata: dict[str, Any]) -> bool:
        return (
            ACTIVE_RUN_ID_FIELD in metadata or ACTIVE_RUN_EXPIRES_AT_FIELD in metadata
        )

    def _set_active_run_claim(self, metadata: dict[str, Any], run_id: str) -> None:
        expires_at = self._claim_now() + timedelta(seconds=self.run_claim_lease_seconds)
        metadata[ACTIVE_RUN_ID_FIELD] = run_id
        metadata[ACTIVE_RUN_EXPIRES_AT_FIELD] = expires_at.isoformat()

    @staticmethod
    def _clear_active_run_claim(metadata: dict[str, Any]) -> None:
        metadata.pop(ACTIVE_RUN_ID_FIELD, None)
        metadata.pop(ACTIVE_RUN_EXPIRES_AT_FIELD, None)

    async def _read_key_bytes(self, session_id: str, key: str) -> bytes | None:
        try:
            stream = await self.store.open_read(session_id, key)
        except FileNotFoundError:
            return None
        try:
            return stream.read()
        finally:
            stream.close()

    async def _read_metadata_document(
        self, session_id: str
    ) -> tuple[bytes, dict[str, Any]] | None:
        encoded = await self._read_key_bytes(session_id, METADATA_KEY)
        if encoded is None:
            return None
        metadata = json.loads(encoded)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid session metadata for {session_id}")
        return encoded, metadata

    @staticmethod
    def _encode_metadata(metadata: dict[str, Any]) -> bytes:
        return json.dumps(metadata).encode("utf-8")

    async def _compare_and_swap_metadata(
        self,
        session_id: str,
        expected: bytes,
        metadata: dict[str, Any],
    ) -> bool:
        return await self.store.compare_and_swap_bytes(
            session_id,
            METADATA_KEY,
            expected,
            self._encode_metadata(metadata),
            "application/json",
        )

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session metadata (read-modify-write with lock)."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    logger.warning(f"Cannot update non-existent session: {session_id}")
                    return
                encoded, metadata = document
                self._apply_metadata_updates(metadata, updates)
                if await self.store.compare_and_swap_bytes(
                    session_id,
                    METADATA_KEY,
                    encoded,
                    self._encode_metadata(metadata),
                    "application/json",
                ):
                    return
            raise RuntimeError(
                f"Could not update contended session metadata for {session_id}"
            )

    @staticmethod
    def _apply_metadata_updates(
        metadata: dict[str, Any], updates: dict[str, Any]
    ) -> None:
        metadata.update(updates)
        if metadata.get("status") == "completed":
            progress = metadata.setdefault("overall_progress", {})
            total_steps = _observed_total_steps(metadata)
            progress["total_steps"] = total_steps
            progress["current_step"] = total_steps
            progress["percent"] = 100
        metadata["updated_at"] = datetime.now(UTC).isoformat()

        created_at = datetime.fromisoformat(metadata["created_at"])
        now = datetime.now(UTC)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        metadata["elapsed_seconds"] = int((now - created_at).total_seconds())

    async def update_step_progress(
        self,
        session_id: str,
        step_name: str,
        progress: dict[str, Any],
    ) -> None:
        """Update progress for current step."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return
                encoded, metadata = document
                step_num = STEP_NUMBERS.get(step_name, 0)

                current_step_info = metadata.get("current_step")
                if current_step_info and current_step_info.get("name") == step_name:
                    started_at = datetime.fromisoformat(current_step_info["started_at"])
                    if started_at.tzinfo is None:
                        started_at = started_at.replace(tzinfo=UTC)
                    elapsed = int((datetime.now(UTC) - started_at).total_seconds())
                    current_step_info["progress"] = progress
                    current_step_info["elapsed_seconds"] = elapsed
                else:
                    current_step_info = {
                        "name": step_name,
                        "display_name": step_display_name(step_name),
                        "started_at": datetime.now(UTC).isoformat(),
                        "progress": progress,
                        "elapsed_seconds": 0,
                    }

                metadata["current_step"] = current_step_info

                if step_num > 0:
                    step_progress_percent = progress.get("percent", 0)
                    overall_percent = step_overall_percent(
                        step_name, step_progress_percent
                    )

                    if step_num > metadata["overall_progress"]["total_steps"]:
                        metadata["overall_progress"]["total_steps"] = step_num
                    metadata["overall_progress"]["current_step"] = step_num
                    if overall_percent is not None:
                        metadata["overall_progress"]["percent"] = overall_percent

                metadata["updated_at"] = datetime.now(UTC).isoformat()
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return
            raise RuntimeError(
                f"Could not update contended step progress for {session_id}"
            )

    async def mark_step_completed(
        self,
        session_id: str,
        step_name: str,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as completed."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return
                encoded, metadata = document
                current_step_info = metadata.get("current_step")
                if not current_step_info or current_step_info["name"] != step_name:
                    return

                started_at = datetime.fromisoformat(current_step_info["started_at"])
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=UTC)
                completed_at = datetime.now(UTC)
                duration = int((completed_at - started_at).total_seconds())

                completed_step = {
                    "name": step_name,
                    "display_name": current_step_info["display_name"],
                    "started_at": current_step_info["started_at"],
                    "completed_at": completed_at.isoformat(),
                    "duration_seconds": duration,
                    "stats": stats or {},
                }

                if "completed_steps" not in metadata:
                    metadata["completed_steps"] = []
                metadata["completed_steps"].append(completed_step)

                if "timings" not in metadata:
                    metadata["timings"] = {}
                metadata["timings"][step_name] = duration

                metadata["current_step"] = None

                completed_count = len(metadata["completed_steps"])
                total_steps = metadata["overall_progress"]["total_steps"]
                step_number = STEP_NUMBERS.get(step_name, completed_count)
                if step_number > total_steps:
                    metadata["overall_progress"]["total_steps"] = step_number
                    total_steps = step_number
                metadata["overall_progress"]["current_step"] = min(
                    step_number, total_steps
                )

                if step_name in STEP_COMPLETION_PERCENTS:
                    metadata["overall_progress"]["percent"] = STEP_COMPLETION_PERCENTS[
                        step_name
                    ]
                elif completed_count >= metadata["overall_progress"]["total_steps"]:
                    metadata["overall_progress"]["percent"] = 100

                metadata["updated_at"] = datetime.now(UTC).isoformat()
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return
            raise RuntimeError(
                f"Could not mark contended step complete for {session_id}"
            )

    async def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return
                encoded, metadata = document
                preview_images = metadata.setdefault("preview_images", [])
                if image_name in preview_images:
                    return
                preview_images.append(image_name)
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return
            raise RuntimeError(
                f"Could not add contended preview image for {session_id}"
            )

    async def update_preview_images(
        self, session_id: str, image_names: list[str]
    ) -> None:
        """Update the list of preview images."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    return
                encoded, metadata = document
                metadata["preview_images"] = image_names
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    return
            raise RuntimeError(
                f"Could not update contended preview images for {session_id}"
            )

    @staticmethod
    def _cancel_request_key(run_id: str) -> str:
        return f".cancel/{run_id}"

    async def is_cancelled(self, session_id: str, run_id: str | None = None) -> bool:
        """Check whether the selected run has a cross-instance cancel request."""

        session_id = _validate_session_id(session_id)
        if run_id is None:
            metadata = await self.get_session_metadata(session_id)
            if not isinstance(metadata, dict):
                return False
            active_run_id = metadata.get(ACTIVE_RUN_ID_FIELD)
            if not isinstance(active_run_id, str) or not _RUN_ID_PATTERN.fullmatch(
                active_run_id
            ):
                return False
            run_id = active_run_id
        elif not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        return await self.store.exists(
            session_id,
            self._cancel_request_key(run_id),
        )

    async def is_cancellation_accepted(self, session_id: str, run_id: str) -> bool:
        """Return whether an exact run's durable cancellation was accepted."""

        session_id = _validate_session_id(session_id)
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        if not await self.store.exists(
            session_id,
            self._cancel_request_key(run_id),
        ):
            return False
        metadata = await self.get_session_metadata(session_id)
        if not isinstance(metadata, dict) or metadata.get("status") != "cancelling":
            return False
        claim = self._parse_active_run_claim(metadata)
        return bool(claim and claim[0] == run_id)

    async def request_cancellation(
        self,
        session_id: str,
        run_id: str | None = None,
    ) -> bool:
        """Request cancellation only while the selected run still owns the lease."""

        session_id = _validate_session_id(session_id)
        if run_id is not None and not _RUN_ID_PATTERN.fullmatch(run_id):
            return False
        selected_run_id = run_id
        marker_written = False
        lock = self._get_lock(session_id)
        async with lock:
            for _ in range(_METADATA_CAS_ATTEMPTS):
                document = await self._read_metadata_document(session_id)
                if document is None:
                    logger.warning(f"Cannot cancel non-existent session: {session_id}")
                    return False
                encoded, metadata = document
                claim = self._parse_active_run_claim(metadata)
                if selected_run_id is None:
                    if claim is None:
                        return False
                    selected_run_id = claim[0]
                if (
                    claim is None
                    or claim[0] != selected_run_id
                    or claim[1] <= self._claim_now()
                    or metadata.get("status") not in {"pending", "running"}
                ):
                    return False

                if not marker_written:
                    await self.store.put_bytes(
                        session_id,
                        self._cancel_request_key(selected_run_id),
                        b"",
                    )
                    marker_written = True
                self._apply_metadata_updates(metadata, {"status": "cancelling"})
                if await self._compare_and_swap_metadata(
                    session_id,
                    encoded,
                    metadata,
                ):
                    logger.info(
                        "Cancellation requested for session/run: %s/%s",
                        session_id,
                        selected_run_id,
                    )
                    return True
            raise RuntimeError(
                f"Could not cancel contended session metadata for {session_id}"
            )

    async def get_artifact_path(
        self, session_id: str, artifact_type: str
    ) -> Path | None:
        """Get path to a local session artifact."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)

        relative_paths, publication_marker = await self._artifact_lookup_for_session(
            session_id,
            artifact_type,
        )
        for relative_path in relative_paths:
            path = session_dir / relative_path
            if path.is_file():
                current_paths, current_marker = await self._artifact_lookup_for_session(
                    session_id,
                    artifact_type,
                )
                if (
                    relative_path in current_paths
                    and current_marker == publication_marker
                ):
                    return path

        return None

    async def get_immutable_local_artifact_path_with_filename(
        self,
        session_id: str,
        artifact_type: str,
    ) -> tuple[Path, str] | None:
        """Return the path of a statically safe immutable local publication."""

        selected = await self.get_immutable_local_artifact_stream_with_filename(
            session_id,
            artifact_type,
        )
        if selected is None:
            return None
        artifact, filename = selected
        artifact.stream.close()
        return self.storage_path / artifact.relative_key, filename

    async def get_immutable_local_artifact_stream_with_filename(
        self,
        session_id: str,
        artifact_type: str,
    ) -> tuple[OpenArtifactFile, str] | None:
        """Hold a revalidated immutable publication for response streaming."""

        if (
            self.store.kind != "local"
            or not isinstance(self.store, LocalSessionStore)
            or self.store.root.resolve() != self.storage_path.resolve()
            or artifact_type not in JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS
        ):
            return None

        relative_paths, publication_marker = await self._artifact_lookup_for_session(
            session_id,
            artifact_type,
        )
        if not publication_marker or publication_marker[0] != "publication":
            return None

        session_dir = self.get_session_dir(session_id)
        for relative_path in relative_paths:
            try:
                artifact = await asyncio.to_thread(
                    open_held_confined_artifact,
                    session_dir,
                    relative_path,
                )
            except (ArtifactPathError, OSError, RuntimeError, ValueError):
                continue
            try:
                current_paths, current_marker = await self._artifact_lookup_for_session(
                    session_id,
                    artifact_type,
                )
            except BaseException:
                artifact.stream.close()
                raise
            if relative_path in current_paths and current_marker == publication_marker:
                return artifact, Path(relative_path).name
            artifact.stream.close()
        return None

    async def _artifact_lookup_for_session(
        self,
        session_id: str,
        artifact_type: str,
    ) -> tuple[tuple[str, ...], tuple[str, str, str] | None]:
        """Resolve candidate keys and the completed publication identity."""

        relative_paths = _ARTIFACT_RELATIVE_PATHS.get(artifact_type, ())
        if artifact_type not in JOINT_RIGGER_ARTIFACT_RELATIVE_PATHS:
            metadata = await self.get_session_metadata(session_id)
            if not isinstance(metadata, dict):
                return (), None
            cache_publications = parse_cache_publications(metadata)
            report_publication_matches_cache = (
                artifact_type == "prediction_report"
                and PREDICTION_REPORT_PUBLICATION_ID_FIELD in metadata
                and (
                    cache_publications is None
                    or metadata.get(PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD)
                    == cache_publications
                )
            )
            if report_publication_matches_cache:
                publication_id = metadata.get(PREDICTION_REPORT_PUBLICATION_ID_FIELD)
                if metadata.get("status") != "completed" or not isinstance(
                    publication_id, str
                ):
                    return (), None
                try:
                    publication_key = prediction_report_publication_key(publication_id)
                except ValueError:
                    logger.warning(
                        "Ignoring invalid prediction report publication for %s",
                        session_id[:8],
                    )
                    return (), None
                return (
                    (publication_key,),
                    ("prediction-report", publication_id, ""),
                )
            if cache_publications is None:
                legacy_run_is_idle = metadata.get(
                    "status"
                ) == "completed" and not self._has_active_run_claim_fields(metadata)
                if not legacy_run_is_idle:
                    return (), None
                return relative_paths, _legacy_cache_marker(metadata)
            if metadata.get("status") != "completed" or not relative_paths:
                return (), None

            namespace = Path(relative_paths[0]).parts[1]
            run_id = cache_publications.get(namespace)
            if run_id is None:
                logger.warning(
                    "Ignoring %s without a valid cache publication for %s",
                    artifact_type,
                    session_id[:8],
                )
                return (), None
            bound_paths = tuple(
                bound_cache_artifact_key(metadata, relative_path)
                for relative_path in relative_paths
            )
            if any(bound_path is None for bound_path in bound_paths):
                return (), None
            return (
                tuple(str(bound_path) for bound_path in bound_paths),
                ("cache", namespace, run_id),
            )

        metadata = await self.get_session_metadata(session_id)
        if not isinstance(metadata, dict):
            return (), None
        if metadata.get("status") != "completed":
            return (), None
        publication_marker = _artifact_publication_marker(metadata)

        results = metadata.get("results")
        if not isinstance(results, dict):
            # Completed sessions created before exact-key binding retain the
            # established USDZ-first compatibility lookup.
            return relative_paths, publication_marker
        if JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD not in results:
            if JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD in results:
                logger.warning(
                    "Ignoring Joint Rigger publication without artifact bindings "
                    "for %s",
                    session_id[:8],
                )
                return (), publication_marker
            # Completed sessions created before exact-key binding retain the
            # established USDZ-first compatibility lookup.
            return relative_paths, publication_marker

        if JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD not in results:
            allowed_bound_paths = relative_paths
        else:
            publication_id_value = results.get(JOINT_RIGGER_PUBLICATION_ID_RESULT_FIELD)
            if not isinstance(
                publication_id_value, str
            ) or not _JOINT_RIGGER_PUBLICATION_ID_PATTERN.fullmatch(
                publication_id_value
            ):
                logger.warning(
                    "Ignoring invalid Joint Rigger publication identity for %s: %r",
                    session_id[:8],
                    publication_id_value,
                )
                return (), publication_marker
            publication_relative_paths = relative_paths
            if artifact_type == "joint_rigger_output":
                publication_relative_paths = (
                    f"cache/joint_rigger/{PREFERRED_JOINT_RIGGER_OUTPUT_FILENAME}",
                )
            allowed_bound_paths = tuple(
                f"{JOINT_RIGGER_PUBLICATION_PREFIX}/{publication_id_value}/"
                f"{Path(relative_path).name}"
                for relative_path in publication_relative_paths
            )
        artifact_keys = results.get(JOINT_RIGGER_ARTIFACT_KEYS_RESULT_FIELD)
        if not isinstance(artifact_keys, dict):
            logger.warning(
                "Ignoring malformed Joint Rigger artifact bindings for %s",
                session_id[:8],
            )
            return (), publication_marker

        bound_key = artifact_keys.get(artifact_type)
        if bound_key is None:
            return (), publication_marker
        if not isinstance(bound_key, str) or bound_key not in allowed_bound_paths:
            logger.warning(
                "Ignoring invalid %s binding for %s: %r",
                artifact_type,
                session_id[:8],
                bound_key,
            )
            return (), publication_marker
        return (bound_key,), publication_marker

    async def _get_artifact_store_selection(
        self,
        session_id: str,
        artifact_type: str,
    ) -> tuple[str, tuple[str, str, str] | None] | None:
        """Return one available key and its completed publication identity."""

        session_id = _validate_session_id(session_id)
        relative_paths, publication_marker = await self._artifact_lookup_for_session(
            session_id,
            artifact_type,
        )
        for key in relative_paths:
            if await self.store.exists(session_id, key):
                return key, publication_marker
        return None

    async def _get_artifact_store_key(
        self, session_id: str, artifact_type: str
    ) -> str | None:
        """Return the preferred available store key for one artifact type."""

        selection = await self._get_artifact_store_selection(
            session_id,
            artifact_type,
        )
        return selection[0] if selection is not None else None

    async def get_artifact_stream(
        self, session_id: str, artifact_type: str
    ) -> IO[bytes] | None:
        """Get artifact as a byte stream from store (works for S3)."""
        resolved = await self.get_artifact_stream_with_filename(
            session_id,
            artifact_type,
        )
        return resolved[0] if resolved is not None else None

    async def get_artifact_stream_with_filename(
        self, session_id: str, artifact_type: str
    ) -> tuple[IO[bytes], str] | None:
        """Open one selected store artifact and return its matching filename."""

        selection = await self._get_artifact_store_selection(
            session_id,
            artifact_type,
        )
        if selection is None:
            return None
        key, publication_marker = selection

        try:
            stream = await self.store.open_read(session_id, key)
        except FileNotFoundError:
            return None
        if publication_marker is not None:
            try:
                current_paths, current_marker = await self._artifact_lookup_for_session(
                    session_id,
                    artifact_type,
                )
            except BaseException:
                stream.close()
                raise
            if key not in current_paths or current_marker != publication_marker:
                stream.close()
                return None
        return stream, Path(key).name

    async def get_artifact_filename(
        self, session_id: str, artifact_type: str
    ) -> str | None:
        """Return the filename that matches the locally or remotely served bytes."""

        local_path = await self.get_artifact_path(session_id, artifact_type)
        if local_path is not None:
            return local_path.name
        key = await self._get_artifact_store_key(session_id, artifact_type)
        return Path(key).name if key is not None else None

    async def has_artifact(self, session_id: str, artifact_type: str) -> bool:
        """Return whether an artifact is available locally or in the store."""
        if await self.get_artifact_path(session_id, artifact_type):
            return True

        return await self._get_artifact_store_key(session_id, artifact_type) is not None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from store and local disk."""
        session_id = _validate_session_id(session_id)
        try:
            await self.store.delete_session(session_id)
        except Exception:
            log_durable_failure(
                logger,
                "session_store_delete_failed",
                phase=FailurePhase.ROLLBACK,
                retryable=True,
            )
            return False

        # Also clean up local directory (with retry for transient failures)
        session_dir = self.get_session_dir(session_id)
        for attempt in range(3):
            try:
                await asyncio.to_thread(
                    remove_confined_tree,
                    session_dir,
                    self.storage_path,
                )
                break
            except (OSError, RuntimeError, ValueError):
                if attempt == 2:
                    log_durable_failure(
                        logger,
                        "session_local_delete_failed",
                        phase=FailurePhase.ROLLBACK,
                        retryable=True,
                    )
                else:
                    await asyncio.sleep(0.5 * (attempt + 1))

        # Clean up lock
        self._locks.pop(session_id, None)

        logger.info(f"Deleted session: {session_id}")
        return True

    async def list_sessions(self) -> list[str]:
        """List all session IDs from the store."""
        return safe_listed_session_ids(await self.store.list_sessions())

    async def sync_to_store(
        self,
        session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        """Sync local session files to the store (uploads to S3 if configured)."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            return 0
        return await self.store.sync_from_local(
            session_id,
            str(session_dir),
            prefix=prefix,
            overwrite=overwrite,
        )

    async def sync_from_store(
        self,
        session_id: str,
        prefix: str = "",
        *,
        overwrite: bool = False,
    ) -> int:
        """Pull files from the store to local session directory (downloads from S3 if configured)."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return await self.store.sync_to_local(
            session_id,
            str(session_dir),
            prefix=prefix,
            overwrite=overwrite,
        )

    async def cleanup_expired_sessions(self) -> int:
        """Remove sessions past their TTL."""
        cleaned = 0
        now = datetime.now(UTC)

        session_ids = await self.list_sessions()
        for session_id in session_ids:
            metadata = await self.get_session_metadata(session_id)
            if not metadata:
                continue

            expires_at_str = metadata.get("ttl_expires_at")
            if expires_at_str:
                expires_at = datetime.fromisoformat(expires_at_str)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                if now > expires_at:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    if await self.delete_session(session_id):
                        cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired sessions")

        return cleaned
