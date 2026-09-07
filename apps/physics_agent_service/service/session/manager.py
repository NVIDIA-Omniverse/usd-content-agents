# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for physics agent pipeline executions.

Delegates all persistence to a pluggable SessionStore (local or S3).
All public methods are async.
"""

import asyncio
import logging
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from physics_agent.config.usd_suffixes import (
    USD_ARTIFACT_EXTENSIONS,
    default_apply_physics_output_suffix,
)
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

from ..runtime.progress import (
    STEP_COMPLETION_PERCENT,
    STEP_DISPLAY_NAMES,
    STEP_NUMBER,
    STEP_WEIGHTS,
    TOTAL_VISIBLE_STEPS,
)
from ..storage import LocalSessionStore, SessionGeneration, SessionStore
from ..storage.base import METADATA_KEY, CompletedSessionSnapshot

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
_TERMINAL_SESSION_STATUSES = {"completed", "failed", "cancelled"}
_PIPELINE_TERMINAL_CLAIM_KEY = ".pipeline-terminal-claim"
_LEASE_HEARTBEAT_RETRY_SECONDS = 1.0
_LEASE_HEARTBEAT_MAX_RETRY_SECONDS = 30.0


class InvalidSessionIdError(ValueError):
    """Raised when a session_id fails format validation.

    Subclasses ValueError so existing `except ValueError` / `pytest.raises(ValueError)`
    keeps working, but lets FastAPI's exception handler target just this class
    instead of swallowing every ValueError in the app.
    """


class SessionStoreDeletionError(RuntimeError):
    """Raised when a terminal-only session delete fails in the store."""


def _validate_session_id(session_id: str) -> str:
    """Validate that session_id has UUID shape; reject otherwise."""
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise InvalidSessionIdError(f"Invalid session_id: {session_id!r}")
    return session_id


def _usd_suffix_from_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    suffix = Path(value).suffix.lower()
    return suffix if suffix in USD_ARTIFACT_EXTENSIONS else None


def _configured_output_usd_suffix(config: dict[str, Any]) -> str | None:
    candidates = []
    steps = config.get("steps")
    if isinstance(steps, dict):
        candidates.append(steps.get("apply_physics"))
    step_configs = config.get("step_configs")
    if isinstance(step_configs, dict):
        candidates.append(step_configs.get("apply_physics"))
    candidates.append(config.get("apply_physics"))

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        suffix = _usd_suffix_from_path(candidate.get("output_usd_path"))
        if suffix:
            return suffix
    return None


def _is_terminal_session(metadata: dict[str, Any]) -> bool:
    return metadata.get("status") in _TERMINAL_SESSION_STATUSES


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
    ):
        self.storage_path = Path(storage_path)
        self.ttl_hours = ttl_hours
        self.store = store or LocalSessionStore(root_dir=str(self.storage_path))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock for safe read-modify-write."""
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def begin_generation(self, session_id: str) -> SessionGeneration | None:
        """Atomically assign a new durable run generation when supported."""
        session_id = _validate_session_id(session_id)
        begin = getattr(self.store, "begin_generation", None)
        if begin is None:
            return None
        generation = await begin(session_id)
        adopt_local = getattr(self.store, "adopt_local_generation", None)
        if adopt_local is not None:
            try:
                await adopt_local(
                    session_id,
                    str(self.get_session_dir(session_id)),
                    generation,
                )
            except asyncio.CancelledError:
                # Let the caller observe the committed claim and enter its
                # generation rollback path before cancellation is delivered at
                # the next suspension point.
                task = asyncio.current_task()
                if task is None:
                    raise
                asyncio.get_running_loop().call_soon(task.cancel)
            except Exception:
                # The durable claim already succeeded, but a caller may have
                # selected replica-local input before reaching this method.
                # Fail closed by cancelling at its next suspension point, after
                # it has observed the claim and enabled generation rollback.
                log_durable_failure(
                    logger,
                    "physics_local_generation_adoption_failed",
                    phase=FailurePhase.PERSISTENCE_VERIFICATION,
                    retryable=True,
                )
                task = asyncio.current_task()
                if task is None:
                    raise
                asyncio.get_running_loop().call_soon(task.cancel)
        return generation

    async def maintain_generation_lease(
        self,
        session_id: str,
        interval_seconds: float = 30.0,
        *,
        capacity_ready: asyncio.Event | None = None,
    ) -> bool:
        """Renew a queued generation and prove ownership at capacity handoff."""
        session_id = _validate_session_id(session_id)
        owns_generation = getattr(self.store, "owns_active_generation", None)
        if owns_generation is None:
            await (capacity_ready or asyncio.Event()).wait()
            return True
        consecutive_failures = 0
        try:
            while True:
                try:
                    owns = await owns_generation(session_id)
                except Exception:  # noqa: BLE001 - persistence may recover
                    consecutive_failures += 1
                    retry_seconds = min(
                        _LEASE_HEARTBEAT_RETRY_SECONDS
                        * (2 ** min(consecutive_failures - 1, 5)),
                        _LEASE_HEARTBEAT_MAX_RETRY_SECONDS,
                    )
                    logger.warning(
                        "Generation lease renewal failed for %s; retrying in %.1fs "
                        "(attempt %d)",
                        session_id,
                        retry_seconds,
                        consecutive_failures,
                        exc_info=True,
                    )
                    # Never drop an accepted queued request merely because the
                    # persistence check is temporarily unavailable. Once the
                    # store recovers, either ownership is still ours and the
                    # lease is renewed, or a successor is observed and this
                    # heartbeat returns so the stale worker is discarded.
                    await asyncio.sleep(retry_seconds)
                    continue
                if not owns:
                    return False
                consecutive_failures = 0
                if capacity_ready is not None:
                    if capacity_ready.is_set():
                        return True
                    try:
                        await asyncio.wait_for(
                            capacity_ready.wait(),
                            timeout=interval_seconds,
                        )
                    except TimeoutError:
                        continue
                    # Capacity becoming available is only a wake-up signal.
                    # Loop once more for an ownership read at handoff time.
                    continue
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            return False

    async def _update_metadata_document(
        self,
        session_id: str,
        updater: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        update_json = getattr(self.store, "update_json", None)
        if update_json is not None:
            return await update_json(session_id, METADATA_KEY, updater)
        metadata = await self.store.get_json(session_id, METADATA_KEY)
        if metadata is None:
            return None
        updated = updater(metadata)
        if updated is not None:
            await self.store.put_json(session_id, METADATA_KEY, updated)
            return updated
        return metadata

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
        (session_dir / "cache" / "physics").mkdir(parents=True, exist_ok=True)
        (session_dir / "preview").mkdir(parents=True, exist_ok=True)

        # Initialize store entry
        await self.store.init_session(session_id)

        metadata = {
            "session_id": session_id,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "pending",
            "current_step": None,
            "completed_steps": [],
            "overall_progress": {
                "current_step": 0,
                "total_steps": TOTAL_VISIBLE_STEPS,
                "percent": 0,
                "estimated_remaining_seconds": None,
            },
            "preview_images": [],
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

    async def update_session(self, session_id: str, updates: dict[str, Any]) -> None:
        """Update session metadata with backend CAS and generation fencing."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:

            def apply_updates(metadata: dict[str, Any]) -> dict[str, Any]:
                metadata.update(updates)
                metadata["updated_at"] = datetime.now(UTC).isoformat()

                created_at = datetime.fromisoformat(metadata["created_at"])
                now = datetime.now(UTC)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                metadata["elapsed_seconds"] = int((now - created_at).total_seconds())
                return metadata

            metadata = await self._update_metadata_document(session_id, apply_updates)
            if metadata is None:
                logger.warning(f"Cannot update non-existent session: {session_id}")

    async def update_session_if_not_cancelled(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Atomically persist a terminal outcome unless cancellation won."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:

            def apply_updates(metadata: dict[str, Any]) -> dict[str, Any]:
                metadata.update(updates)
                metadata["updated_at"] = datetime.now(UTC).isoformat()

                created_at = datetime.fromisoformat(metadata["created_at"])
                now = datetime.now(UTC)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                metadata["elapsed_seconds"] = int((now - created_at).total_seconds())
                return metadata

            conditional_update = getattr(
                self.store,
                "update_json_if_not_cancelled",
                None,
            )
            if conditional_update is not None:
                metadata = await conditional_update(
                    session_id,
                    METADATA_KEY,
                    apply_updates,
                )
                return metadata is not None

            if await self.store.exists(session_id, ".cancel"):
                return False
            metadata = await self._update_metadata_document(
                session_id,
                apply_updates,
            )
            return metadata is not None

    async def restore_session_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Restore an exact startup snapshot without rewriting its timestamps."""
        session_id = _validate_session_id(session_id)
        required = {"session_id", "created_at", "status"}
        if not required.issubset(metadata) or metadata.get("session_id") != session_id:
            raise ValueError("Cannot restore an incomplete session metadata snapshot")
        lock = self._get_lock(session_id)
        async with lock:
            restored = await self._update_metadata_document(
                session_id,
                lambda _current: deepcopy(metadata),
            )
            if restored is None:
                raise FileNotFoundError(session_id)

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

            def apply_progress(metadata: dict[str, Any]) -> dict[str, Any]:
                step_info = {
                    "display": STEP_DISPLAY_NAMES.get(step_name, step_name),
                    "step_num": STEP_NUMBER.get(step_name, 0),
                }
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
                        "display_name": step_info["display"],
                        "started_at": datetime.now(UTC).isoformat(),
                        "progress": progress,
                        "elapsed_seconds": 0,
                    }
                metadata["current_step"] = current_step_info

                step_num = step_info["step_num"]
                if step_num > 0:
                    step_progress_percent = progress.get("percent", 0)
                    weights = STEP_WEIGHTS.get(step_name)
                    if weights is not None:
                        start, end = weights
                        overall_percent = start + int(
                            (end - start) * step_progress_percent / 100
                        )
                    else:
                        overall_percent = step_progress_percent
                    metadata["overall_progress"]["current_step"] = step_num
                    metadata["overall_progress"]["percent"] = min(100, overall_percent)
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                return metadata

            await self._update_metadata_document(session_id, apply_progress)

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

            def apply_completion(metadata: dict[str, Any]) -> dict[str, Any] | None:
                current_step_info = metadata.get("current_step")
                if not current_step_info or current_step_info["name"] != step_name:
                    return None
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
                metadata.setdefault("completed_steps", []).append(completed_step)
                metadata.setdefault("timings", {})[step_name] = duration
                metadata["current_step"] = None
                metadata["overall_progress"]["current_step"] = max(
                    metadata["overall_progress"].get("current_step", 0),
                    STEP_NUMBER.get(step_name, len(metadata["completed_steps"])),
                )
                current_percent = metadata["overall_progress"].get("percent", 0)
                snapped = STEP_COMPLETION_PERCENT.get(step_name)
                if snapped is not None:
                    metadata["overall_progress"]["percent"] = max(
                        current_percent, snapped
                    )
                metadata["updated_at"] = datetime.now(UTC).isoformat()
                return metadata

            await self._update_metadata_document(session_id, apply_completion)

    async def add_preview_image(self, session_id: str, image_name: str) -> None:
        """Add a preview image to the session."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:

            def add_image(metadata: dict[str, Any]) -> dict[str, Any] | None:
                images = metadata.setdefault("preview_images", [])
                if image_name in images:
                    return None
                images.append(image_name)
                return metadata

            await self._update_metadata_document(session_id, add_image)

    async def update_preview_images(
        self, session_id: str, image_names: list[str]
    ) -> None:
        """Update the list of preview images."""
        session_id = _validate_session_id(session_id)
        lock = self._get_lock(session_id)
        async with lock:

            def replace_images(metadata: dict[str, Any]) -> dict[str, Any]:
                metadata["preview_images"] = image_names
                return metadata

            await self._update_metadata_document(session_id, replace_images)

    async def is_cancelled(self, session_id: str) -> bool:
        """Check if session has been cancelled (works cross-instance via store)."""
        session_id = _validate_session_id(session_id)
        owns_generation = getattr(self.store, "owns_active_generation", None)
        if owns_generation is not None and not await owns_generation(session_id):
            return True
        return await self.store.exists(session_id, ".cancel")

    async def clear_cancellation(self, session_id: str) -> None:
        """Clear a durable cancellation marker before reusing a session."""
        session_id = _validate_session_id(session_id)
        await self.store.delete_key(session_id, ".cancel")

    async def claim_pipeline_terminal_state(
        self,
        session_id: str,
        status: str,
    ) -> str:
        """Atomically choose one terminal outcome across service instances."""
        session_id = _validate_session_id(session_id)
        if status not in _TERMINAL_SESSION_STATUSES:
            raise ValueError(f"Invalid terminal pipeline status: {status}")

        encoded = status.encode("ascii")
        if await self.store.put_bytes_if_absent(
            session_id,
            _PIPELINE_TERMINAL_CLAIM_KEY,
            encoded,
        ):
            return status

        stream = await self.store.open_read(
            session_id,
            _PIPELINE_TERMINAL_CLAIM_KEY,
        )
        try:
            winner = stream.read().decode("ascii")
        finally:
            stream.close()
        if winner not in _TERMINAL_SESSION_STATUSES:
            raise RuntimeError("Invalid persisted pipeline terminal claim")
        return winner

    async def clear_pipeline_terminal_claim(self, session_id: str) -> None:
        """Clear the terminal claim before a terminal session is reused."""
        session_id = _validate_session_id(session_id)
        await self.store.delete_key(session_id, _PIPELINE_TERMINAL_CLAIM_KEY)

    async def request_pipeline_cancellation(self, session_id: str) -> bool:
        """Atomically compete with completion/failure for pipeline termination."""
        session_id = _validate_session_id(session_id)
        if not await self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return False

        generation_cancel = getattr(self.store, "request_generation_cancellation", None)
        if generation_cancel is not None and not await generation_cancel(
            session_id,
            update_status=False,
        ):
            return False

        # Publish the worker signal first. If another terminal writer already
        # won, remove the losing signal so it cannot affect a later reuse.
        if generation_cancel is None:
            await self.store.put_bytes(session_id, ".cancel", b"")
        winner = await self.claim_pipeline_terminal_state(session_id, "cancelled")
        if winner != "cancelled":
            await self.store.delete_key(session_id, ".cancel")
            return False

        logger.info(f"Pipeline cancellation accepted for session: {session_id}")
        return True

    async def request_cancellation(self, session_id: str) -> None:
        """Request cancellation — visible to all instances via store."""
        session_id = _validate_session_id(session_id)
        if not await self.session_exists(session_id):
            logger.warning(f"Cannot cancel non-existent session: {session_id}")
            return

        generation_cancel = getattr(self.store, "request_generation_cancellation", None)
        if generation_cancel is not None:
            if not await generation_cancel(session_id, update_status=True):
                return
        else:
            await self.store.put_bytes(session_id, ".cancel", b"")
            await self.update_session(session_id, {"status": "cancelling"})
        logger.info(f"Cancellation requested for session: {session_id}")

    async def get_artifact_path(
        self, session_id: str, artifact_type: str
    ) -> Path | None:
        """Get path to a local session artifact."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)

        selected = await self.get_local_artifact_stream(session_id, artifact_type)
        if selected is None:
            return None
        artifact, relative_key = selected
        artifact.stream.close()
        return session_dir / relative_key

    async def open_local_artifact_key(
        self,
        session_id: str,
        relative_key: str,
    ) -> OpenArtifactFile | None:
        """Open one session key through a held, no-follow descriptor chain."""

        session_id = _validate_session_id(session_id)
        try:
            session_dir = self.get_session_dir(session_id)
            return await asyncio.to_thread(
                open_held_confined_artifact,
                session_dir,
                relative_key,
            )
        except (ArtifactPathError, OSError, RuntimeError, ValueError):
            return None

    async def get_local_artifact_stream(
        self,
        session_id: str,
        artifact_type: str,
    ) -> tuple[OpenArtifactFile, str] | None:
        """Open one well-known local artifact and return its canonical key."""

        session_id = _validate_session_id(session_id)

        artifact_map = {
            "predictions": "cache/predictions/predictions.jsonl",
            "dataset": "cache/dataset/dataset.jsonl",
        }

        if artifact_type == "output_usd":
            session_dir = self.get_session_dir(session_id)
            suffix = await self._expected_output_usd_suffix(session_id, session_dir)
            if suffix:
                relative_key = f"cache/physics/scene_physics{suffix}"
                artifact = await self.open_local_artifact_key(
                    session_id,
                    relative_key,
                )
                return (artifact, relative_key) if artifact is not None else None

            candidates = [
                f"cache/physics/scene_physics{candidate_suffix}"
                for candidate_suffix in USD_ARTIFACT_EXTENSIONS
            ]
            opened: list[tuple[OpenArtifactFile, str]] = []
            for relative_key in candidates:
                artifact = await self.open_local_artifact_key(
                    session_id,
                    relative_key,
                )
                if artifact is not None:
                    opened.append((artifact, relative_key))
            if not opened:
                return None
            selected = max(opened, key=lambda item: item[0].metadata.st_mtime)
            for artifact, _ in opened:
                if artifact is not selected[0]:
                    artifact.stream.close()
            return selected

        relative_key = artifact_map.get(artifact_type)
        if relative_key:
            artifact = await self.open_local_artifact_key(session_id, relative_key)
            if artifact is not None:
                return artifact, relative_key

        return None

    async def get_artifact_stream(
        self, session_id: str, artifact_type: str, key: str | None = None
    ) -> IO[bytes] | None:
        """Get artifact as a byte stream from store (works for S3)."""
        session_id = _validate_session_id(session_id)
        key_map = {
            "predictions": "cache/predictions/predictions.jsonl",
            "dataset": "cache/dataset/dataset.jsonl",
        }
        if artifact_type == "output_usd":
            keys = await self.list_artifact_keys(session_id, "output_usd")
            if key is not None and key not in keys:
                return None
            key = key or (keys[0] if keys else None)
        else:
            key = key_map.get(artifact_type)
        if not key or not await self.store.exists(session_id, key):
            return None

        return await self.store.open_read(session_id, key)

    async def list_artifact_keys(
        self, session_id: str, artifact_type: str
    ) -> list[str]:
        """List store keys for an artifact type."""
        session_id = _validate_session_id(session_id)
        if artifact_type != "output_usd":
            return []

        keys = await self.store.list_keys(session_id, prefix="cache/physics/")
        expected = {
            f"cache/physics/scene_physics{suffix}" for suffix in USD_ARTIFACT_EXTENSIONS
        }
        # Keep .usdz in the candidate set for older sessions and lower-level
        # explicit output paths; the expected suffix below narrows new
        # unified-pipeline sessions to the default contract.
        matched = [key for key in keys if key in expected]
        suffix = await self._expected_output_usd_suffix(
            session_id,
            self.get_session_dir(session_id),
        )
        if suffix:
            exact = f"cache/physics/scene_physics{suffix}"
            return [exact] if exact in matched else []

        return sorted(
            matched,
            key=lambda key: USD_ARTIFACT_EXTENSIONS.index(Path(key).suffix.lower()),
        )

    async def list_store_keys(self, session_id: str, prefix: str = "") -> list[str]:
        """List persisted session keys beneath an optional prefix."""
        session_id = _validate_session_id(session_id)
        return await self.store.list_keys(session_id, prefix=prefix)

    async def _expected_output_usd_suffix(
        self, session_id: str, session_dir: Path
    ) -> str | None:
        metadata = await self.store.get_json(session_id, METADATA_KEY)
        config = metadata.get("config", {}) if metadata else {}
        if isinstance(config, dict):
            suffix = _configured_output_usd_suffix(config)
            if suffix:
                return suffix
            input_config = config.get("input")
            if isinstance(input_config, dict):
                suffix = _usd_suffix_from_path(input_config.get("usd_path"))
                if suffix:
                    return default_apply_physics_output_suffix(suffix)
            for key in ("usd_path", "input_usd_path", "input_usd"):
                suffix = _usd_suffix_from_path(config.get(key))
                if suffix:
                    return default_apply_physics_output_suffix(suffix)

        input_dir = session_dir / "input"
        for suffix in USD_ARTIFACT_EXTENSIONS:
            if (input_dir / f"scene{suffix}").exists():
                return default_apply_physics_output_suffix(suffix)
        if input_dir.exists():
            for path in sorted(input_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in USD_ARTIFACT_EXTENSIONS:
                    return default_apply_physics_output_suffix(path.suffix.lower())

        input_keys = await self.store.list_keys(session_id, prefix="input/")
        for suffix in USD_ARTIFACT_EXTENSIONS:
            if f"input/scene{suffix}" in input_keys:
                return default_apply_physics_output_suffix(suffix)
        for key in sorted(input_keys):
            suffix = Path(key).suffix.lower()
            if suffix in USD_ARTIFACT_EXTENSIONS:
                return default_apply_physics_output_suffix(suffix)

        return None

    async def delete_session(
        self,
        session_id: str,
        *,
        require_terminal: bool = False,
    ) -> bool:
        """Delete a session from store and local disk."""
        return await self._delete_session(
            session_id,
            require_terminal=require_terminal,
            raise_store_errors=False,
        )

    async def delete_terminal_session(self, session_id: str) -> bool:
        """Delete a terminal session, surfacing store errors to the API."""
        return await self._delete_session(
            session_id,
            require_terminal=True,
            raise_store_errors=True,
        )

    async def _delete_session(
        self,
        session_id: str,
        *,
        require_terminal: bool,
        raise_store_errors: bool,
    ) -> bool:
        session_id = _validate_session_id(session_id)
        try:
            if require_terminal:
                if not await self.store.delete_session_if_terminal(session_id):
                    return False
            else:
                await self.store.delete_session(session_id)
        except Exception as error:
            log_durable_failure(
                logger,
                "session_store_delete_failed",
                phase=FailurePhase.ROLLBACK,
                retryable=True,
            )
            if raise_store_errors:
                raise SessionStoreDeletionError(
                    f"Failed to delete session from store: {session_id}"
                ) from error
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
        prefix: str | tuple[str, ...] = "",
    ) -> int:
        """Publish one local artifact snapshot to the configured store."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        if not session_dir.exists():
            if self.store.kind == "s3":
                raise FileNotFoundError(session_dir)
            return 0
        return await self.store.sync_from_local(
            session_id, str(session_dir), prefix=prefix
        )

    async def sync_from_store(
        self,
        session_id: str,
        prefix: str | tuple[str, ...] = "",
    ) -> int:
        """Pull files from the store to local session directory (downloads from S3 if configured)."""
        session_id = _validate_session_id(session_id)
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return await self.store.sync_to_local(
            session_id, str(session_dir), prefix=prefix
        )

    async def snapshot_completed_publication(
        self,
        session_id: str,
        destination_dir: Path,
        *,
        prefix: str | tuple[str, ...] = "",
    ) -> CompletedSessionSnapshot:
        """Hydrate metadata and artifacts proven to share one completed view."""
        session_id = _validate_session_id(session_id)
        snapshot_completed = getattr(
            self.store,
            "sync_completed_publication_to_local",
            None,
        )
        if snapshot_completed is None:
            raise RuntimeError(
                "The configured session store cannot snapshot a completed publication"
            )
        return await snapshot_completed(
            session_id,
            str(destination_dir),
            prefix=prefix,
        )

    async def cleanup_expired_sessions(self) -> int:
        """Remove terminal sessions past their TTL."""
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
                    if not _is_terminal_session(metadata):
                        logger.info(
                            "Skipping expired active session during cleanup: %s "
                            "(status=%s)",
                            session_id,
                            metadata.get("status"),
                        )
                        continue
                    logger.info(f"Cleaning up expired session: {session_id}")
                    if await self.delete_session(session_id, require_terminal=True):
                        cleaned += 1

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired sessions")

        return cleaned

    async def cleanup_stale_local_cache(self, max_age_hours: float = 24.0) -> int:
        """Remove stale local session cache after syncing terminal sessions."""
        skip_session_ids: set[str] = set()
        for session_id in await self.list_sessions():
            metadata = await self.get_session_metadata(session_id)
            if metadata and not _is_terminal_session(metadata):
                skip_session_ids.add(session_id)

        return await self.store.cleanup_stale_local_sessions(
            str(self.storage_path),
            max_age_hours=max_age_hours,
            skip_session_ids=skip_session_ids,
        )
