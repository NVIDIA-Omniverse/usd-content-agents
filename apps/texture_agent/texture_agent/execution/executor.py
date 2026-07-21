# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded execution of immutable Texture Plan units."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from texture_agent.execution.models import (
    TextureArtifactRef,
    TextureExecutionCheckpoint,
    TextureExecutionStatus,
    TextureExecutionSummary,
    TextureUnitExecutionRecord,
    TextureUnitExecutionResult,
    TextureUnitExecutionState,
)
from texture_agent.planning import (
    TexturePlan,
    TexturePlanUnit,
    validate_texture_plan_payload,
)


class TextureExecutionCheckpointStore(Protocol):
    """Persistence boundary used after every unit state transition."""

    def load(self) -> TextureExecutionCheckpoint | None: ...

    def save(self, checkpoint: TextureExecutionCheckpoint) -> None: ...


class FileTextureExecutionCheckpointStore:
    """Atomic local-filesystem checkpoint store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> TextureExecutionCheckpoint | None:
        if not self.path.is_file():
            return None
        return TextureExecutionCheckpoint.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )

    def save(self, checkpoint: TextureExecutionCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                checkpoint.model_dump_json(indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class TextureExecutionCancelled(RuntimeError):
    """Raised cooperatively by a unit runner after cancellation is requested."""


class TextureExecutionTimedOut(TimeoutError):
    """Raised when a unit runner exceeds the immutable per-unit timeout."""


class CancellationToken:
    """Thread-safe cooperative cancellation token."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class TextureUnitExecutionContext:
    """Attempt-local controls passed to the backend-facing unit runner."""

    unit_id: str
    attempt: int
    timeout_seconds: int
    cancellation_token: CancellationToken
    external_cancellation_check: Callable[[], bool] | None = None
    started_monotonic: float = 0.0

    def is_cancelled(self) -> bool:
        return self.cancellation_token.is_cancelled() or bool(
            self.external_cancellation_check and self.external_cancellation_check()
        )

    def remaining_seconds(self) -> float:
        elapsed = time.monotonic() - self.started_monotonic
        return max(0.0, self.timeout_seconds - elapsed)

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise TextureExecutionCancelled(
                f"Texture unit {self.unit_id} was cancelled"
            )

    def raise_if_timed_out(self) -> None:
        if self.remaining_seconds() <= 0:
            raise TextureExecutionTimedOut(
                f"Texture unit {self.unit_id} exceeded its "
                f"{self.timeout_seconds}s timeout"
            )


TextureUnitRunner = Callable[
    [TexturePlanUnit, TextureUnitExecutionContext],
    TextureUnitExecutionResult | Mapping[str, str],
]
TextureExecutionProgressCallback = Callable[[TextureExecutionCheckpoint], None]
TextureArtifactValidator = Callable[[TextureUnitExecutionResult], bool]


def texture_plan_fingerprint(plan: TexturePlan) -> str:
    """Hash the complete immutable plan artifact for safe checkpoint reuse."""
    canonical = json.dumps(
        plan.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _default_artifact_validator(result: TextureUnitExecutionResult) -> bool:
    """Validate local artifacts while accepting durable remote URIs."""
    for artifact in result.artifacts:
        parsed = urlparse(artifact.uri)
        if parsed.scheme and parsed.scheme != "file":
            continue
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else artifact.uri)
        if not path.is_file():
            return False
        if artifact.sha256 is not None:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != artifact.sha256:
                return False
    return True


class BoundedTextureExecutor:
    """Execute only selected units from a validated immutable plan.

    The executor owns scheduling, caching, checkpointing, and selection
    enforcement. Backend-specific code remains in the supplied ``unit_runner``
    and receives the plan's per-unit timeout plus a cooperative cancellation
    token. A runner must use that timeout for blocking backend calls.
    """

    def __init__(
        self,
        *,
        plan: TexturePlan | Mapping[str, object] | str | bytes,
        checkpoint_store: TextureExecutionCheckpointStore,
        unit_runner: TextureUnitRunner,
        cancellation_token: CancellationToken | None = None,
        external_cancellation_check: Callable[[], bool] | None = None,
        artifact_validator: TextureArtifactValidator | None = None,
        progress_callback: TextureExecutionProgressCallback | None = None,
    ) -> None:
        self.plan = validate_texture_plan_payload(plan)
        if not self.plan.decision.execution_allowed:
            raise ValueError(
                "Texture plan is not approved for execution: "
                f"{self.plan.decision.state.value}"
            )
        self.checkpoint_store = checkpoint_store
        self.unit_runner = unit_runner
        self.cancellation_token = cancellation_token or CancellationToken()
        self.external_cancellation_check = external_cancellation_check
        self.artifact_validator = artifact_validator or _default_artifact_validator
        self.progress_callback = progress_callback
        self.plan_fingerprint = texture_plan_fingerprint(self.plan)
        self._plan_unit_by_id = {
            unit.unit_id: unit for unit in self.plan.selected_units
        }
        self._selected_unit_ids = tuple(self._plan_unit_by_id)

    def execute(
        self,
        *,
        resume: bool = False,
        regenerate_unit_ids: Sequence[str] = (),
    ) -> TextureExecutionSummary:
        """Execute, resume, or regenerate an exact selected-unit subset."""
        regenerate_ids = tuple(regenerate_unit_ids)
        if len(regenerate_ids) != len(set(regenerate_ids)):
            raise ValueError("regenerate_unit_ids must not contain duplicates")
        unknown = sorted(set(regenerate_ids) - set(self._selected_unit_ids))
        if unknown:
            raise ValueError(
                "Regeneration requested unit IDs outside the approved plan: "
                + ", ".join(unknown)
            )

        should_load = resume or bool(regenerate_ids)
        checkpoint = self._load_or_create_checkpoint(load_existing=should_load)
        records = {record.unit_id: record for record in checkpoint.records}
        requested_ids = regenerate_ids or self._selected_unit_ids

        executed_ids: list[str] = []
        cache_hit_ids: list[str] = []
        pending_ids: list[str] = []
        for unit_id in requested_ids:
            record = records[unit_id]
            accepted = record.accepted_result
            use_cache = (
                not regenerate_ids
                and accepted is not None
                and self.artifact_validator(accepted)
            )
            if use_cache:
                records[unit_id] = record.model_copy(
                    update={
                        "state": TextureUnitExecutionState.COMPLETED,
                        "cache_hit_count": record.cache_hit_count + 1,
                        "last_error": None,
                    }
                )
                cache_hit_ids.append(unit_id)
            else:
                if accepted is not None:
                    records[unit_id] = record.model_copy(
                        update={"accepted_result": None}
                    )
                pending_ids.append(unit_id)

        checkpoint = self._save_records(checkpoint, records)
        active: dict[Future[TextureUnitExecutionResult], tuple[str, float]] = {}
        pending_index = 0
        abandon_executor = False

        executor = ThreadPoolExecutor(
            max_workers=self.plan.execution.max_concurrency,
            thread_name_prefix="texture-unit",
        )
        try:
            while pending_index < len(pending_ids) or active:
                while (
                    pending_index < len(pending_ids)
                    and len(active) < self.plan.execution.max_concurrency
                    and not self._is_cancelled()
                ):
                    unit_id = pending_ids[pending_index]
                    pending_index += 1
                    record = records[unit_id]
                    attempt = record.attempts + 1
                    started_at = datetime.now(UTC)
                    records[unit_id] = record.model_copy(
                        update={
                            "state": TextureUnitExecutionState.RUNNING,
                            "attempts": attempt,
                            "last_error": None,
                            "started_at": started_at,
                            "finished_at": None,
                        }
                    )
                    checkpoint = self._save_records(checkpoint, records)
                    future = executor.submit(
                        self._execute_one,
                        self._plan_unit_by_id[unit_id],
                        attempt,
                    )
                    active[future] = (
                        unit_id,
                        time.monotonic() + self.plan.execution.unit_timeout_seconds,
                    )
                    executed_ids.append(unit_id)

                if not active:
                    break

                now = time.monotonic()
                next_deadline = min(deadline for _, deadline in active.values())
                wait_timeout = min(0.1, max(0.0, next_deadline - now))
                done, _ = wait(
                    active,
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    unit_id, _deadline = active.pop(future)
                    record = records[unit_id]
                    finished_at = datetime.now(UTC)
                    try:
                        result = future.result()
                    except TextureExecutionCancelled as exc:
                        records[unit_id] = record.model_copy(
                            update={
                                "state": TextureUnitExecutionState.CANCELLED,
                                "last_error": str(exc),
                                "finished_at": finished_at,
                            }
                        )
                    except Exception as exc:
                        records[unit_id] = record.model_copy(
                            update={
                                "state": TextureUnitExecutionState.FAILED,
                                "last_error": (
                                    f"{type(exc).__name__}: texture unit execution "
                                    "failed"
                                ),
                                "finished_at": finished_at,
                            }
                        )
                    else:
                        records[unit_id] = record.model_copy(
                            update={
                                "state": TextureUnitExecutionState.COMPLETED,
                                "accepted_result": result,
                                "last_error": None,
                                "finished_at": finished_at,
                            }
                        )
                    checkpoint = self._save_records(checkpoint, records)

                timed_out = [
                    (future, unit_id)
                    for future, (unit_id, deadline) in list(active.items())
                    if time.monotonic() >= deadline
                ]
                if timed_out:
                    self.cancellation_token.cancel()
                    abandon_executor = True
                    finished_at = datetime.now(UTC)
                    for future, unit_id in timed_out:
                        active.pop(future, None)
                        future.cancel()
                        record = records[unit_id]
                        records[unit_id] = record.model_copy(
                            update={
                                "state": TextureUnitExecutionState.FAILED,
                                "last_error": (
                                    "TextureExecutionTimedOut: texture unit "
                                    "exceeded its immutable timeout"
                                ),
                                "finished_at": finished_at,
                            }
                        )
                    for future, (unit_id, _deadline) in list(active.items()):
                        active.pop(future, None)
                        future.cancel()
                        record = records[unit_id]
                        records[unit_id] = record.model_copy(
                            update={
                                "state": TextureUnitExecutionState.CANCELLED,
                                "last_error": (
                                    "Cancelled after another texture unit timed out"
                                ),
                                "finished_at": finished_at,
                            }
                        )
                    checkpoint = self._save_records(checkpoint, records)
                    break

                if self._is_cancelled() and active:
                    abandon_executor = True
                    finished_at = datetime.now(UTC)
                    for future, (unit_id, _deadline) in list(active.items()):
                        active.pop(future, None)
                        future.cancel()
                        record = records[unit_id]
                        records[unit_id] = record.model_copy(
                            update={
                                "state": TextureUnitExecutionState.CANCELLED,
                                "last_error": "Texture unit execution cancelled",
                                "finished_at": finished_at,
                            }
                        )
                    checkpoint = self._save_records(checkpoint, records)
                    break

            if self._is_cancelled():
                for unit_id in pending_ids[pending_index:]:
                    record = records[unit_id]
                    records[unit_id] = record.model_copy(
                        update={
                            "state": TextureUnitExecutionState.CANCELLED,
                            "last_error": "Cancelled before execution",
                            "finished_at": datetime.now(UTC),
                        }
                    )
                checkpoint = self._save_records(checkpoint, records)
        finally:
            executor.shutdown(wait=not abandon_executor, cancel_futures=True)

        return self._summary(
            checkpoint,
            requested_ids=requested_ids,
            executed_ids=tuple(executed_ids),
            cache_hit_ids=tuple(cache_hit_ids),
        )

    def _execute_one(
        self,
        unit: TexturePlanUnit,
        attempt: int,
    ) -> TextureUnitExecutionResult:
        context = TextureUnitExecutionContext(
            unit_id=unit.unit_id,
            attempt=attempt,
            timeout_seconds=self.plan.execution.unit_timeout_seconds,
            cancellation_token=self.cancellation_token,
            external_cancellation_check=self.external_cancellation_check,
            started_monotonic=time.monotonic(),
        )
        context.raise_if_cancelled()
        raw_result = self.unit_runner(unit, context)
        context.raise_if_timed_out()
        if isinstance(raw_result, TextureUnitExecutionResult):
            result = raw_result
        else:
            result = TextureUnitExecutionResult(
                unit_id=unit.unit_id,
                artifacts=tuple(
                    TextureArtifactRef(name=name, uri=uri)
                    for name, uri in sorted(raw_result.items())
                ),
            )
        if result.unit_id != unit.unit_id:
            raise ValueError(
                f"Runner returned result for {result.unit_id}; expected {unit.unit_id}"
            )
        if not self.artifact_validator(result):
            raise RuntimeError(
                f"Runner returned missing or invalid artifacts for {unit.unit_id}"
            )
        return result

    def _is_cancelled(self) -> bool:
        if self.cancellation_token.is_cancelled():
            return True
        if self.external_cancellation_check and self.external_cancellation_check():
            self.cancellation_token.cancel()
            return True
        return False

    def _load_or_create_checkpoint(
        self,
        *,
        load_existing: bool,
    ) -> TextureExecutionCheckpoint:
        checkpoint = self.checkpoint_store.load() if load_existing else None
        if checkpoint is not None:
            if checkpoint.plan_fingerprint != self.plan_fingerprint:
                raise ValueError(
                    "Execution checkpoint belongs to a different immutable Texture Plan"
                )
            if checkpoint.selected_unit_ids != self._selected_unit_ids:
                raise ValueError(
                    "Execution checkpoint selected units do not match the Texture Plan"
                )
            return checkpoint

        now = datetime.now(UTC)
        checkpoint = TextureExecutionCheckpoint(
            plan_schema_version=self.plan.schema_version,
            plan_fingerprint=self.plan_fingerprint,
            selected_unit_ids=self._selected_unit_ids,
            records=tuple(
                TextureUnitExecutionRecord(unit_id=unit_id)
                for unit_id in self._selected_unit_ids
            ),
            created_at=now,
            updated_at=now,
        )
        self.checkpoint_store.save(checkpoint)
        return checkpoint

    def _save_records(
        self,
        checkpoint: TextureExecutionCheckpoint,
        records: Mapping[str, TextureUnitExecutionRecord],
    ) -> TextureExecutionCheckpoint:
        updated = checkpoint.model_copy(
            update={
                "records": tuple(
                    records[unit_id] for unit_id in self._selected_unit_ids
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        self.checkpoint_store.save(updated)
        if self.progress_callback is not None:
            self.progress_callback(updated)
        return updated

    def _summary(
        self,
        checkpoint: TextureExecutionCheckpoint,
        *,
        requested_ids: tuple[str, ...],
        executed_ids: tuple[str, ...],
        cache_hit_ids: tuple[str, ...],
    ) -> TextureExecutionSummary:
        records = {record.unit_id: record for record in checkpoint.records}
        accepted = tuple(
            unit_id
            for unit_id in self._selected_unit_ids
            if records[unit_id].accepted_result is not None
            and self.artifact_validator(records[unit_id].accepted_result)
        )
        failed = tuple(
            unit_id
            for unit_id in requested_ids
            if records[unit_id].state == TextureUnitExecutionState.FAILED
        )
        cancelled = tuple(
            unit_id
            for unit_id in requested_ids
            if records[unit_id].state == TextureUnitExecutionState.CANCELLED
        )
        remaining = tuple(
            unit_id for unit_id in self._selected_unit_ids if unit_id not in accepted
        )
        accepted_requested = sum(unit_id in accepted for unit_id in requested_ids)
        if failed and accepted_requested:
            status = TextureExecutionStatus.PARTIAL
        elif failed:
            status = TextureExecutionStatus.FAILED
        elif cancelled:
            status = TextureExecutionStatus.CANCELLED
        elif remaining:
            status = TextureExecutionStatus.PARTIAL
        else:
            status = TextureExecutionStatus.COMPLETED
        return TextureExecutionSummary(
            status=status,
            plan_fingerprint=self.plan_fingerprint,
            requested_unit_ids=requested_ids,
            executed_unit_ids=executed_ids,
            cache_hit_unit_ids=cache_hit_ids,
            accepted_unit_ids=accepted,
            failed_unit_ids=failed,
            cancelled_unit_ids=cancelled,
            remaining_unit_ids=remaining,
            records=checkpoint.records,
        )
