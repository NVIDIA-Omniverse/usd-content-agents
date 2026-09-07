# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durable agent-side observation memory for content workflows.

The journal and immutable objects are canonical.  SQLite is a rebuildable,
run-local retrieval projection and never authorizes workflow mutations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
import uuid
import warnings
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .artifacts import atomic_write_json

MEMORY_EVENT_SCHEMA_VERSION = "content-agent-memory.event.v1"
MEMORY_OBSERVATION_SCHEMA_VERSION = "content-agent-memory.observation.v1"
MEMORY_MANIFEST_SCHEMA_VERSION = "content-agent-memory.run.v1"
MEMORY_LEASE_SCHEMA_VERSION = "content-agent-memory.lease.v1"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTEXT_CARDS = 50
_DEFAULT_CONTEXT_CARDS = 12
_MAX_SEARCH_RESULTS = 50
_MAX_INSPECT_OBSERVATIONS = 8
_DEFAULT_INSPECT_BYTES = 64 * 1024 * 1024
MAX_INSPECT_BYTES = 64 * 1024 * 1024
_DEFAULT_ARTIFACT_BYTES = 256 * 1024 * 1024
_COPY_CHUNK_SIZE = 1024 * 1024

MemoryOutcomeClassification = Literal[
    "matched", "contradicted", "ambiguous", "not_checked"
]
MemoryImportance = Literal["low", "normal", "high", "critical"]
MemoryRetention = Literal["scratch", "evidence", "pinned", "exported"]
MemoryArtifactProvenance = Literal["native", "imported", "reconstructed"]
_RETENTION_PRIORITY: dict[MemoryRetention, int] = {
    "scratch": 0,
    "evidence": 1,
    "pinned": 2,
    "exported": 3,
}


class MemoryError(RuntimeError):
    """Base error for durable observation memory."""


class JournalCorruptionError(MemoryError):
    """Raised when the canonical journal cannot be replayed safely."""


class MemoryArtifactError(MemoryError):
    """Raised when an artifact cannot be imported or verified."""


class MemoryBoundsError(MemoryError):
    """Raised when a retrieval request exceeds a hard bound."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _non_empty(value: str, *, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    return normalized


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        from world_understanding.utils.artifacts import fsync_directory

        fsync_directory(path)
        return
    directory_fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class MemoryArtifactInput(_StrictModel):
    """One local artifact to import before recording an observation."""

    path: Path
    role: str
    media_type: str
    retention: MemoryRetention = "evidence"
    provenance: MemoryArtifactProvenance = "imported"

    @field_validator("path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            raise ValueError("memory artifact input paths must be absolute")
        return candidate

    @field_validator("role", "media_type")
    @classmethod
    def _required_text(cls, value: str, info: Any) -> str:
        return _non_empty(value, label=info.field_name)


class MemoryArtifactReference(_StrictModel):
    """Content-addressed artifact metadata stored in an observation."""

    role: str
    media_type: str
    byte_size: int = Field(ge=0)
    sha256: str
    extension: str
    retention: MemoryRetention
    provenance: MemoryArtifactProvenance

    @field_validator("sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("extension")
    @classmethod
    def _extension(cls, value: str) -> str:
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", value):
            raise ValueError("extension must be a short lowercase suffix")
        return value


class MemorySceneIdentity(_StrictModel):
    session_id: str | None = None
    scene_revision_id: str | None = None
    source_scene_sha256: str | None = None
    camera_sha256: str | None = None

    @field_validator("source_scene_sha256", "camera_sha256")
    @classmethod
    def _optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("scene digests must be lowercase SHA-256 values")
        return value


class MemoryInteraction(_StrictModel):
    operation: str
    request_event_id: str | None = None
    response_event_id: str | None = None
    target_object_ids: tuple[str, ...] = ()
    target_prim_paths: tuple[str, ...] = ()

    @field_validator("operation")
    @classmethod
    def _operation(cls, value: str) -> str:
        return _non_empty(value, label="operation")


class MemoryExpectation(_StrictModel):
    expected_changes: tuple[str, ...] = ()
    forbidden_changes: tuple[str, ...] = ()


class MemoryOutcome(_StrictModel):
    classification: MemoryOutcomeClassification
    summary: str = Field(max_length=2000)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("summary")
    @classmethod
    def _summary(cls, value: str) -> str:
        return _non_empty(value, label="outcome summary")


class RememberRequest(_StrictModel):
    """Agent-authored meaning attached to durable observable evidence."""

    workflow: str
    phase: str
    interaction: MemoryInteraction
    outcome: MemoryOutcome
    expectation: MemoryExpectation = Field(default_factory=MemoryExpectation)
    scene: MemorySceneIdentity | None = None
    artifacts: tuple[MemoryArtifactInput, ...] = ()
    parent_observation_id: str | None = None
    turn_id: str | None = None
    importance: MemoryImportance = "normal"
    tags: tuple[str, ...] = ()

    @field_validator("workflow", "phase")
    @classmethod
    def _required_names(cls, value: str, info: Any) -> str:
        return _non_empty(value, label=info.field_name)

    @field_validator("tags")
    @classmethod
    def _normalized_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        tags = tuple(dict.fromkeys(tag.strip() for tag in value if tag.strip()))
        if len(tags) > 32:
            raise ValueError("an observation may contain at most 32 tags")
        return tags


class ObservationRecord(_StrictModel):
    schema_version: Literal["content-agent-memory.observation.v1"] = (
        MEMORY_OBSERVATION_SCHEMA_VERSION
    )
    observation_id: str
    run_id: str
    parent_observation_id: str | None = None
    turn_id: str | None = None
    workflow: str
    phase: str
    scene: MemorySceneIdentity | None = None
    interaction: MemoryInteraction
    expectation: MemoryExpectation
    artifacts: tuple[MemoryArtifactReference, ...] = ()
    outcome: MemoryOutcome
    importance: MemoryImportance = "normal"
    tags: tuple[str, ...] = ()
    created_at: str


class ObservationCard(_StrictModel):
    observation_id: str
    sequence: int = Field(ge=1)
    phase: str
    operation: str
    targets: tuple[str, ...]
    outcome: MemoryOutcomeClassification
    summary: str
    artifact_roles: tuple[str, ...]
    importance: MemoryImportance
    tags: tuple[str, ...]
    pinned: bool = False


class MemoryEvent(_StrictModel):
    schema_version: Literal["content-agent-memory.event.v1"] = (
        MEMORY_EVENT_SCHEMA_VERSION
    )
    event_id: str
    run_id: str
    sequence: int = Field(ge=1)
    time: str
    kind: Literal["observation", "pin", "unpin"]
    phase: str
    observation: dict[str, Any]


class MemoryContextResult(_StrictModel):
    run_id: str
    cards: tuple[ObservationCard, ...]
    pinned_observation_ids: tuple[str, ...]
    unresolved_observation_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class MemorySearchQuery(_StrictModel):
    text: str | None = None
    workflow: str | None = None
    phase: str | None = None
    scene_revision_id: str | None = None
    operation: str | None = None
    outcome: MemoryOutcomeClassification | None = None
    importance: MemoryImportance | None = None
    tag: str | None = None
    target: str | None = None
    artifact_role: str | None = None
    sequence_min: int | None = Field(default=None, ge=1)
    sequence_max: int | None = Field(default=None, ge=1)
    created_before: str | None = None
    phase_not: str | None = None
    limit: int = Field(default=12, ge=1, le=_MAX_SEARCH_RESULTS)

    @model_validator(mode="after")
    def _sequence_range(self) -> MemorySearchQuery:
        if (
            self.sequence_min is not None
            and self.sequence_max is not None
            and self.sequence_min > self.sequence_max
        ):
            raise ValueError("sequence_min must not exceed sequence_max")
        return self


class MaterializedMemoryArtifact(_StrictModel):
    observation_id: str
    role: str
    media_type: str
    byte_size: int
    sha256: str
    path: Path


class MemoryInspectResult(_StrictModel):
    run_id: str
    observations: tuple[ObservationRecord, ...]
    artifacts: tuple[MaterializedMemoryArtifact, ...]
    lease_id: str | None = None
    expires_at: str | None = None


def resolve_memory_root(
    explicit_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Resolve the configured managed memory root."""

    default_repo_root: Path | None = None
    if explicit_root is not None:
        selected = Path(explicit_root).expanduser()
    elif os.getenv("WU_AGENT_MEMORY_ROOT"):
        selected = Path(os.environ["WU_AGENT_MEMORY_ROOT"]).expanduser()
    elif repo_root is not None:
        default_repo_root = Path(repo_root).expanduser()
        selected = default_repo_root / "runs" / ".memory"
    else:
        default_repo_root = _discover_repo_root(Path.cwd())
        selected = default_repo_root / "runs" / ".memory"

    if default_repo_root is not None:
        legacy_root = default_repo_root / "agentic" / "runs" / ".memory"
        if legacy_root.exists() and not selected.exists():
            warnings.warn(
                "Existing pre-0.6 agent memory was found at "
                f"{legacy_root}. The new default is {selected}; set "
                "WU_AGENT_MEMORY_ROOT to the existing directory to retain it.",
                RuntimeWarning,
                stacklevel=2,
            )
    if create:
        selected.mkdir(parents=True, exist_ok=True)
    return selected.resolve()


def _discover_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "agentic").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    raise MemoryError(
        "Cannot discover repository root; pass memory_root or repo_root explicitly"
    )


def _ensure_managed_directory(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise MemoryError(f"Managed path escapes memory root: {path}") from exc
    current = root
    if current.is_symlink() or not current.is_dir():
        raise MemoryError(f"Memory root must be a real directory: {root}")
    for component in relative.parts:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MemoryError(f"Managed directory is unsafe: {current}")


def _safe_name(value: str) -> str:
    normalized = _SAFE_NAME_RE.sub("_", value).strip("._")
    return normalized[:80] or "artifact"


def _canonical_extension(media_type: str, source_suffix: str) -> str:
    normalized = media_type.lower().split(";", 1)[0].strip()
    suffix = source_suffix.lower()
    fixed = {
        "image/png": (".png", frozenset({".png"})),
        "image/jpeg": (".jpg", frozenset({".jpg", ".jpeg"})),
        "application/json": (".json", frozenset({".json"})),
        "application/yaml": (".yaml", frozenset({".yaml", ".yml"})),
        "text/yaml": (".yaml", frozenset({".yaml", ".yml"})),
        "text/plain": (".txt", frozenset({".txt"})),
        "text/markdown": (".md", frozenset({".md", ".markdown"})),
    }
    if normalized in fixed:
        canonical, allowed = fixed[normalized]
        if suffix not in allowed:
            raise MemoryArtifactError(
                f"{media_type} artifacts require one of: {', '.join(sorted(allowed))}"
            )
        return canonical
    if normalized in {"model/vnd.usd", "application/usd"}:
        if suffix not in {".usd", ".usda", ".usdc", ".usdz"}:
            raise MemoryArtifactError("USD artifacts require a USD file extension")
        return suffix
    if normalized == "application/octet-stream":
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            return ".bin"
        return suffix
    raise MemoryArtifactError(f"Unsupported memory artifact media type: {media_type}")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_SIZE):
            digest.update(chunk)
            byte_size += len(chunk)
    return digest.hexdigest(), byte_size


class AgentMemory:
    """One run's journal, object ingestion, index, and bounded retrieval API."""

    def __init__(
        self,
        *,
        run_id: str,
        memory_root: str | Path | None = None,
        repo_root: str | Path | None = None,
        max_artifact_bytes: int = _DEFAULT_ARTIFACT_BYTES,
    ) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(
                "run_id must contain only letters, digits, '.', '_', or '-'"
            )
        if max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")
        self.run_id = run_id
        self.root = resolve_memory_root(memory_root, repo_root=repo_root)
        self.run_dir = self.root / run_id
        self.objects_root = self.root / ".objects" / "sha256"
        self.cache_root = self.root / ".cache"
        self.locks_root = self.root / ".locks"
        self.journal_path = self.run_dir / "trace" / "events.jsonl"
        self.index_path = self.run_dir / "memory" / "index.sqlite"
        self.max_artifact_bytes = max_artifact_bytes

        for directory in (
            self.objects_root,
            self.cache_root,
            self.locks_root,
            self.run_dir / "trace",
            self.run_dir / "memory",
        ):
            _ensure_managed_directory(self.root, directory)
        self._run_lock = FileLock(str(self.locks_root / f"run-{run_id}.lock"))
        self._objects_lock = FileLock(str(self.locks_root / "objects.lock"))
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        manifest_path = self.run_dir / "manifest.json"
        with self._run_lock:
            if manifest_path.is_symlink():
                raise MemoryError(
                    f"Memory run manifest must not be a symlink: {manifest_path}"
                )
            if manifest_path.is_file():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != MEMORY_MANIFEST_SCHEMA_VERSION:
                    raise MemoryError("Memory run manifest schema is unsupported")
                if payload.get("run_id") != self.run_id:
                    raise MemoryError("Memory run manifest run_id does not match")
                return
            if manifest_path.exists():
                raise MemoryError(f"Memory run manifest is not a file: {manifest_path}")
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": MEMORY_MANIFEST_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "created_at": _utc_now(),
                },
            )

    def remember(self, request: RememberRequest) -> ObservationRecord:
        """Import evidence and append one meaningful observation."""

        if request.parent_observation_id is not None:
            self._observation_record(request.parent_observation_id)
        ingested = tuple(self._ingest_artifact(item) for item in request.artifacts)
        unique_artifacts: dict[tuple[str, str], MemoryArtifactReference] = {}
        for artifact in ingested:
            key = (artifact.role, artifact.sha256)
            existing = unique_artifacts.get(key)
            if existing is None or (
                _RETENTION_PRIORITY[artifact.retention]
                > _RETENTION_PRIORITY[existing.retention]
            ):
                unique_artifacts[key] = artifact
        artifacts = tuple(unique_artifacts.values())
        record = ObservationRecord(
            observation_id=_new_id("obs"),
            run_id=self.run_id,
            parent_observation_id=request.parent_observation_id,
            turn_id=request.turn_id,
            workflow=request.workflow,
            phase=request.phase,
            scene=request.scene,
            interaction=request.interaction,
            expectation=request.expectation,
            artifacts=artifacts,
            outcome=request.outcome,
            importance=request.importance,
            tags=request.tags,
            created_at=_utc_now(),
        )
        self._append_memory_event(
            kind="observation",
            phase=request.phase,
            observation=record.model_dump(mode="json"),
        )
        return record

    def pin(self, observation_id: str) -> ObservationCard:
        """Append a pin event and return the updated card."""

        self._observation_record(observation_id)
        self._append_memory_event(
            kind="pin",
            phase="memory",
            observation={"observation_id": observation_id},
        )
        return self._card_for_id(observation_id)

    def unpin(self, observation_id: str) -> ObservationCard:
        """Append an unpin event and return the updated card."""

        self._observation_record(observation_id)
        self._append_memory_event(
            kind="unpin",
            phase="memory",
            observation={"observation_id": observation_id},
        )
        return self._card_for_id(observation_id)

    def context(self, *, limit: int = _DEFAULT_CONTEXT_CARDS) -> MemoryContextResult:
        """Return a bounded working set, prioritizing pins and unresolved evidence."""

        if not 1 <= limit <= _MAX_CONTEXT_CARDS:
            raise MemoryBoundsError(
                f"context limit must be in [1, {_MAX_CONTEXT_CARDS}]"
            )
        self._ensure_index_synced()
        with closing(self._connect_index()) as connection, connection:
            pinned = self._select_cards(
                connection,
                "WHERE pinned = 1 ORDER BY sequence DESC",
                (),
                limit,
            )
            unresolved = self._select_cards(
                connection,
                "WHERE outcome IN ('contradicted', 'ambiguous') ORDER BY sequence DESC",
                (),
                limit,
            )
            recent = self._select_cards(
                connection,
                "ORDER BY sequence DESC",
                (),
                limit,
            )
        cards: list[ObservationCard] = []
        seen: set[str] = set()
        for candidate in (*pinned, *unresolved, *recent):
            if candidate.observation_id in seen:
                continue
            seen.add(candidate.observation_id)
            cards.append(candidate)
            if len(cards) == limit:
                break
        return MemoryContextResult(
            run_id=self.run_id,
            cards=tuple(cards),
            pinned_observation_ids=tuple(card.observation_id for card in pinned),
            unresolved_observation_ids=tuple(
                card.observation_id for card in unresolved
            ),
        )

    def search(self, query: MemorySearchQuery) -> tuple[ObservationCard, ...]:
        """Search one run through exact filters and bounded FTS."""

        self._ensure_index_synced()
        joins: list[str] = []
        clauses: list[str] = []
        values: list[Any] = []
        if query.text:
            fts_query = self._fts_query(query.text)
            joins.append(
                "JOIN observation_fts f ON f.observation_id = o.observation_id"
            )
            clauses.append("observation_fts MATCH ?")
            values.append(fts_query)
        for field, value in (
            ("workflow", query.workflow),
            ("phase", query.phase),
            ("scene_revision_id", query.scene_revision_id),
            ("operation", query.operation),
            ("outcome", query.outcome),
            ("importance", query.importance),
        ):
            if value is not None:
                clauses.append(f"o.{field} = ?")
                values.append(value)
        if query.sequence_min is not None:
            clauses.append("o.sequence >= ?")
            values.append(query.sequence_min)
        if query.sequence_max is not None:
            clauses.append("o.sequence <= ?")
            values.append(query.sequence_max)
        if query.created_before is not None:
            clauses.append("json_extract(o.record_json, '$.created_at') <= ?")
            values.append(query.created_before)
        if query.phase_not is not None:
            clauses.append("o.phase != ?")
            values.append(query.phase_not)
        if query.tag:
            clauses.append(
                "EXISTS (SELECT 1 FROM tags t WHERE t.observation_id = "
                "o.observation_id AND t.tag = ?)"
            )
            values.append(query.tag)
        if query.target:
            clauses.append(
                "EXISTS (SELECT 1 FROM targets t WHERE t.observation_id = "
                "o.observation_id AND t.target = ?)"
            )
            values.append(query.target)
        if query.artifact_role:
            clauses.append(
                "EXISTS (SELECT 1 FROM artifacts a WHERE a.observation_id = "
                "o.observation_id AND a.role = ?)"
            )
            values.append(query.artifact_role)
        sql = "SELECT DISTINCT o.* FROM observations o " + " ".join(joins)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY o.sequence DESC LIMIT ?"
        values.append(query.limit)
        with closing(self._connect_index()) as connection, connection:
            rows = connection.execute(sql, values).fetchall()
        return tuple(self._card_from_row(row) for row in rows)

    def count_observations(
        self,
        *,
        workflow: str | None = None,
        phase_not: str | None = None,
        tag: str | None = None,
    ) -> int:
        """Count matching observations without applying retrieval result limits."""

        self._ensure_index_synced()
        clauses: list[str] = []
        values: list[str] = []
        if workflow is not None:
            clauses.append("workflow = ?")
            values.append(workflow)
        if phase_not is not None:
            clauses.append("phase != ?")
            values.append(phase_not)
        if tag is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM tags t WHERE "
                "t.observation_id = observations.observation_id AND t.tag = ?)"
            )
            values.append(tag)
        sql = "SELECT COUNT(*) FROM observations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with closing(self._connect_index()) as connection, connection:
            row = connection.execute(sql, values).fetchone()
        return int(row[0])

    def inspect(
        self,
        observation_ids: tuple[str, ...] | list[str],
        *,
        artifact_roles: tuple[str, ...] | list[str] = (),
        max_bytes: int = _DEFAULT_INSPECT_BYTES,
    ) -> MemoryInspectResult:
        """Return full records and materialize only explicitly requested roles."""

        self._cleanup_expired_leases()
        selected_ids = tuple(dict.fromkeys(observation_ids))
        if not selected_ids:
            raise ValueError("inspect requires at least one observation_id")
        if len(selected_ids) > _MAX_INSPECT_OBSERVATIONS:
            raise MemoryBoundsError(
                f"inspect accepts at most {_MAX_INSPECT_OBSERVATIONS} observations"
            )
        if max_bytes <= 0:
            raise MemoryBoundsError("inspect max_bytes must be positive")
        if max_bytes > MAX_INSPECT_BYTES:
            raise MemoryBoundsError(
                f"inspect max_bytes cannot exceed {MAX_INSPECT_BYTES}"
            )
        selected_roles = set(artifact_roles)
        records = tuple(self._observation_record(item) for item in selected_ids)
        references = [
            (record.observation_id, artifact)
            for record in records
            for artifact in record.artifacts
            if artifact.role in selected_roles
        ]
        total_bytes = sum(artifact.byte_size for _, artifact in references)
        if total_bytes > max_bytes:
            raise MemoryBoundsError(
                f"inspect requested {total_bytes} bytes, exceeding {max_bytes}"
            )
        if not references:
            return MemoryInspectResult(
                run_id=self.run_id,
                observations=records,
                artifacts=(),
            )

        lease_id = _new_id("lease")
        lease_dir = self.cache_root / lease_id
        _ensure_managed_directory(self.root, lease_dir)
        expires_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        materialized: list[MaterializedMemoryArtifact] = []
        for index, (observation_id, reference) in enumerate(references):
            source = self._object_path(reference)
            self._verify_object(source, reference)
            target_name = (
                f"{index:02d}-{observation_id[:16]}-{_safe_name(reference.role)}-"
                f"{reference.sha256[:12]}"
                f"{reference.extension}"
            )
            target = lease_dir / target_name
            shutil.copyfile(source, target)
            copied_digest, copied_size = _sha256_file(target)
            if copied_digest != reference.sha256 or copied_size != reference.byte_size:
                target.unlink(missing_ok=True)
                raise MemoryArtifactError("Materialized artifact verification failed")
            materialized.append(
                MaterializedMemoryArtifact(
                    observation_id=observation_id,
                    role=reference.role,
                    media_type=reference.media_type,
                    byte_size=reference.byte_size,
                    sha256=reference.sha256,
                    path=target,
                )
            )
        atomic_write_json(
            lease_dir / "manifest.json",
            {
                "schema_version": MEMORY_LEASE_SCHEMA_VERSION,
                "lease_id": lease_id,
                "run_id": self.run_id,
                "expires_at": expires_at,
                "artifacts": [item.model_dump(mode="json") for item in materialized],
            },
        )
        return MemoryInspectResult(
            run_id=self.run_id,
            observations=records,
            artifacts=tuple(materialized),
            lease_id=lease_id,
            expires_at=expires_at,
        )

    def _cleanup_expired_leases(self) -> None:
        """Opportunistically remove expired, well-formed materialization leases."""

        now = datetime.now(UTC)
        for lease_dir in self.cache_root.iterdir():
            if lease_dir.is_symlink() or not lease_dir.is_dir():
                continue
            manifest_path = lease_dir / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(manifest["expires_at"])
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if expires_at.tzinfo is None:
                continue
            if expires_at <= now:
                shutil.rmtree(lease_dir)

    def rebuild_index(self) -> None:
        """Delete and rebuild the derived SQLite projection from the journal."""

        with self._run_lock:
            self._rebuild_index_locked(self._load_events())

    def recover_torn_final_line(self) -> Path | None:
        """Preserve and truncate an incomplete final journal line explicitly."""

        with self._run_lock:
            flags = (
                os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                journal_fd = os.open(self.journal_path, flags)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise JournalCorruptionError(
                    "Memory journal could not be opened without following links"
                ) from exc
            with os.fdopen(journal_fd, "r+b") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise JournalCorruptionError(
                        "Memory journal must be a singly linked regular file"
                    )
                payload = stream.read()
                if not payload or payload.endswith(b"\n"):
                    return None
                boundary = payload.rfind(b"\n") + 1
                damaged = payload[boundary:]
                backup = (
                    self.journal_path.parent / f"events.torn-{uuid.uuid4().hex}.bin"
                )
                with backup.open("xb") as backup_stream:
                    backup_stream.write(damaged)
                    backup_stream.flush()
                    os.fsync(backup_stream.fileno())
                _fsync_directory(self.journal_path.parent)
                stream.truncate(boundary)
                stream.flush()
                os.fsync(stream.fileno())
            self._rebuild_index_locked(self._load_events())
            return backup

    def _ingest_artifact(self, item: MemoryArtifactInput) -> MemoryArtifactReference:
        source = item.path
        try:
            source_stat = source.lstat()
        except FileNotFoundError as exc:
            raise MemoryArtifactError(
                f"Memory artifact does not exist: {source}"
            ) from exc
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
            raise MemoryArtifactError(
                f"Memory artifact must be a regular non-symlink file: {source}"
            )
        if source_stat.st_size > self.max_artifact_bytes:
            raise MemoryArtifactError(
                f"Memory artifact exceeds {self.max_artifact_bytes} bytes: {source}"
            )
        extension = _canonical_extension(item.media_type, source.suffix)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=".memory-object-",
            suffix=".tmp",
            dir=self.objects_root,
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        byte_size = 0
        try:
            source_fd = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened_stat = os.fstat(source_fd)
                if not stat.S_ISREG(opened_stat.st_mode) or (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                ) != (source_stat.st_dev, source_stat.st_ino):
                    raise MemoryArtifactError(
                        f"Memory artifact changed while opening: {source}"
                    )
                with os.fdopen(temporary_fd, "wb", closefd=False) as target_stream:
                    while chunk := os.read(source_fd, _COPY_CHUNK_SIZE):
                        byte_size += len(chunk)
                        if byte_size > self.max_artifact_bytes:
                            raise MemoryArtifactError(
                                "Memory artifact exceeded the size limit while reading"
                            )
                        digest.update(chunk)
                        target_stream.write(chunk)
                    target_stream.flush()
                    os.fsync(target_stream.fileno())
                final_stat = os.fstat(source_fd)
                if (
                    final_stat.st_size != opened_stat.st_size
                    or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
                ):
                    raise MemoryArtifactError(
                        f"Memory artifact changed while reading: {source}"
                    )
            finally:
                os.close(source_fd)
            os.close(temporary_fd)
            temporary_fd = -1
            sha256 = digest.hexdigest()
            with self._objects_lock:
                destination = self._existing_or_new_object_path(sha256, extension)
                if destination.exists():
                    existing_digest, existing_size = _sha256_file(destination)
                    if existing_digest != sha256 or existing_size != byte_size:
                        raise MemoryArtifactError(
                            f"Stored object failed verification: {destination}"
                        )
                else:
                    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    os.replace(temporary_path, destination)
                    _fsync_directory(destination.parent)
                extension = destination.suffix
            return MemoryArtifactReference(
                role=item.role,
                media_type=item.media_type,
                byte_size=byte_size,
                sha256=sha256,
                extension=extension,
                retention=item.retention,
                provenance=item.provenance,
            )
        finally:
            if temporary_fd >= 0:
                os.close(temporary_fd)
            temporary_path.unlink(missing_ok=True)

    def _existing_or_new_object_path(self, digest: str, extension: str) -> Path:
        parent = self.objects_root / digest[:2]
        if parent.exists():
            matches = tuple(parent.glob(f"{digest}.*"))
            if len(matches) > 1:
                raise MemoryArtifactError(
                    f"Multiple stored objects share digest {digest}"
                )
            if matches:
                return matches[0]
        return parent / f"{digest}{extension}"

    def _object_path(self, reference: MemoryArtifactReference) -> Path:
        candidate = (
            self.objects_root
            / reference.sha256[:2]
            / f"{reference.sha256}{reference.extension}"
        )
        try:
            candidate.relative_to(self.objects_root)
        except ValueError as exc:  # pragma: no cover - digest validation guards this
            raise MemoryArtifactError(
                "Stored object path escaped the object root"
            ) from exc
        return candidate

    @staticmethod
    def _verify_object(path: Path, reference: MemoryArtifactReference) -> None:
        if path.is_symlink() or not path.is_file():
            raise MemoryArtifactError(f"Stored object is unavailable: {path}")
        digest, byte_size = _sha256_file(path)
        if digest != reference.sha256 or byte_size != reference.byte_size:
            raise MemoryArtifactError(f"Stored object digest mismatch: {path}")

    def _append_memory_event(
        self,
        *,
        kind: Literal["observation", "pin", "unpin"],
        phase: str,
        observation: dict[str, Any],
    ) -> MemoryEvent:
        with self._run_lock:
            events = self._load_events()
            self._ensure_index_synced_locked(events)
            sequence = 1 + max(
                (event.sequence for event in events if isinstance(event, MemoryEvent)),
                default=0,
            )
            event = MemoryEvent(
                event_id=_new_id("evt"),
                run_id=self.run_id,
                sequence=sequence,
                time=_utc_now(),
                kind=kind,
                phase=phase,
                observation=observation,
            )
            self._append_journal_line(event)
            try:
                self._project_event(event)
            except sqlite3.DatabaseError:
                self._rebuild_index_locked((*events, event))
            return event

    def _append_journal_line(self, event: MemoryEvent) -> None:
        _ensure_managed_directory(self.root, self.journal_path.parent)
        try:
            self.journal_path.lstat()
            journal_existed = True
        except FileNotFoundError:
            journal_existed = False
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = os.open(self.journal_path, flags, 0o600)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise MemoryError("Memory journal must be a singly linked regular file")
            payload = (
                json.dumps(
                    event.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("Memory journal append made no forward progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        if not journal_existed:
            _fsync_directory(self.journal_path.parent)

    def _load_events(self) -> tuple[MemoryEvent | dict[str, Any], ...]:
        if not self.journal_path.exists():
            return ()
        if self.journal_path.is_symlink() or not self.journal_path.is_file():
            raise JournalCorruptionError("Memory journal is not a regular file")
        payload = self.journal_path.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise JournalCorruptionError(
                "Memory journal has a torn final line; recover it explicitly"
            )
        events: list[MemoryEvent | dict[str, Any]] = []
        expected_sequence = 1
        event_ids: set[str] = set()
        observation_ids: set[str] = set()
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                raw = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JournalCorruptionError(
                    f"Memory journal line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(raw, dict):
                raise JournalCorruptionError(
                    f"Memory journal line {line_number} is not an object"
                )
            if raw.get("schema_version") == MEMORY_EVENT_SCHEMA_VERSION:
                try:
                    event = MemoryEvent.model_validate(raw)
                except ValueError as exc:
                    raise JournalCorruptionError(
                        f"Memory journal line {line_number} has an invalid event"
                    ) from exc
                if event.run_id != self.run_id:
                    raise JournalCorruptionError(
                        f"Memory journal line {line_number} belongs to another run"
                    )
                if event.sequence != expected_sequence:
                    raise JournalCorruptionError(
                        "Memory journal sequence is not strictly contiguous"
                    )
                if event.event_id in event_ids:
                    raise JournalCorruptionError(
                        f"Memory journal repeats event_id {event.event_id}"
                    )
                event_ids.add(event.event_id)
                if event.kind == "observation":
                    try:
                        record = ObservationRecord.model_validate(event.observation)
                    except ValueError as exc:
                        raise JournalCorruptionError(
                            f"Memory journal line {line_number} has an invalid observation"
                        ) from exc
                    if record.run_id != self.run_id:
                        raise JournalCorruptionError(
                            f"Memory journal line {line_number} observation belongs "
                            "to another run"
                        )
                    if record.observation_id in observation_ids:
                        raise JournalCorruptionError(
                            "Memory journal repeats observation_id "
                            f"{record.observation_id}"
                        )
                    observation_ids.add(record.observation_id)
                expected_sequence += 1
                events.append(event)
            else:
                events.append(raw)
        return tuple(events)

    def _connect_index(self) -> sqlite3.Connection:
        expected_identity = self._prepare_index_file()
        connection = sqlite3.connect(self.index_path)
        try:
            metadata = self.index_path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino) != expected_identity
            ):
                raise MemoryError(
                    "Memory index changed while opening; refusing unsafe SQLite file"
                )
        except Exception:
            connection.close()
            raise
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        self._create_index_schema(connection)
        return connection

    def _prepare_index_file(self) -> tuple[int, int]:
        """Create or validate the derived index before SQLite may follow its path."""

        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.index_path}{suffix}")
            try:
                metadata = sidecar.lstat()
            except FileNotFoundError:
                continue
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise MemoryError(
                    f"Memory index sidecar must be a singly linked regular file: {sidecar}"
                )

        try:
            metadata = self.index_path.lstat()
        except FileNotFoundError:
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            fd = os.open(self.index_path, flags, 0o600)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise MemoryError(
                        "Memory index must be a singly linked regular file"
                    )
            finally:
                os.close(fd)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise MemoryError(
                f"Memory index must be a singly linked regular file: {self.index_path}"
            )
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _create_index_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                time TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                workflow TEXT NOT NULL,
                phase TEXT NOT NULL,
                scene_revision_id TEXT,
                operation TEXT NOT NULL,
                outcome TEXT NOT NULL,
                summary TEXT NOT NULL,
                importance TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                observation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                media_type TEXT NOT NULL,
                retention TEXT NOT NULL,
                PRIMARY KEY (observation_id, role, sha256),
                FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
            );
            CREATE TABLE IF NOT EXISTS targets (
                observation_id TEXT NOT NULL,
                target TEXT NOT NULL,
                PRIMARY KEY (observation_id, target),
                FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
            );
            CREATE TABLE IF NOT EXISTS tags (
                observation_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (observation_id, tag),
                FOREIGN KEY (observation_id) REFERENCES observations(observation_id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS observation_fts USING fts5(
                observation_id UNINDEXED,
                text,
                tokenize = 'unicode61'
            );
            """
        )

    def _ensure_index_synced(self) -> None:
        with self._run_lock:
            events = self._load_events()
            self._ensure_index_synced_locked(events)

    def _ensure_index_synced_locked(
        self,
        events: tuple[MemoryEvent | dict[str, Any], ...]
        | list[MemoryEvent | dict[str, Any]],
    ) -> None:
        memory_events = tuple(
            event for event in events if isinstance(event, MemoryEvent)
        )
        try:
            with closing(self._connect_index()) as connection, connection:
                row = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) AS sequence, "
                    "COUNT(*) AS event_count FROM events"
                ).fetchone()
                indexed_sequence = int(row["sequence"])
                indexed_event_count = int(row["event_count"])
        except sqlite3.DatabaseError:
            self._remove_index_files()
            indexed_sequence = -1
            indexed_event_count = -1
        expected_sequence = memory_events[-1].sequence if memory_events else 0
        if indexed_sequence != expected_sequence or indexed_event_count != len(
            memory_events
        ):
            self._rebuild_index_locked(events)

    def _remove_index_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.index_path}{suffix}").unlink(missing_ok=True)

    def _rebuild_index_locked(
        self,
        events: tuple[MemoryEvent | dict[str, Any], ...]
        | list[MemoryEvent | dict[str, Any]],
    ) -> None:
        self._remove_index_files()
        with closing(self._connect_index()) as connection, connection:
            for event in events:
                if isinstance(event, MemoryEvent):
                    self._project_event(event, connection=connection)

    def _project_event(
        self,
        event: MemoryEvent,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        owns_connection = connection is None
        selected = connection or self._connect_index()
        try:
            selected.execute(
                "INSERT INTO events(sequence, event_id, kind, time) VALUES (?, ?, ?, ?)",
                (event.sequence, event.event_id, event.kind, event.time),
            )
            if event.kind == "observation":
                record = ObservationRecord.model_validate(event.observation)
                selected.execute(
                    """
                    INSERT INTO observations(
                        observation_id, sequence, workflow, phase,
                        scene_revision_id, operation, outcome, summary,
                        importance, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.observation_id,
                        event.sequence,
                        record.workflow,
                        record.phase,
                        (
                            record.scene.scene_revision_id
                            if record.scene is not None
                            else None
                        ),
                        record.interaction.operation,
                        record.outcome.classification,
                        record.outcome.summary,
                        record.importance,
                        record.model_dump_json(),
                    ),
                )
                artifact_rows: list[tuple[str, str, str, str, str]] = []
                seen_artifacts: set[tuple[str, str]] = set()
                for item in record.artifacts:
                    artifact_key = (item.role, item.sha256)
                    if artifact_key in seen_artifacts:
                        continue
                    seen_artifacts.add(artifact_key)
                    artifact_rows.append(
                        (
                            record.observation_id,
                            item.role,
                            item.sha256,
                            item.media_type,
                            item.retention,
                        )
                    )
                selected.executemany(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                    artifact_rows,
                )
                targets = tuple(
                    dict.fromkeys(
                        (
                            *record.interaction.target_object_ids,
                            *record.interaction.target_prim_paths,
                        )
                    )
                )
                selected.executemany(
                    "INSERT INTO targets VALUES (?, ?)",
                    [(record.observation_id, target) for target in targets],
                )
                selected.executemany(
                    "INSERT INTO tags VALUES (?, ?)",
                    [(record.observation_id, tag) for tag in record.tags],
                )
                searchable = " ".join(
                    (
                        record.outcome.summary,
                        *record.expectation.expected_changes,
                        *record.expectation.forbidden_changes,
                        *record.tags,
                    )
                )
                selected.execute(
                    "INSERT INTO observation_fts(observation_id, text) VALUES (?, ?)",
                    (record.observation_id, searchable),
                )
            else:
                observation_id = str(event.observation.get("observation_id") or "")
                cursor = selected.execute(
                    "UPDATE observations SET pinned = ? WHERE observation_id = ?",
                    (1 if event.kind == "pin" else 0, observation_id),
                )
                if cursor.rowcount != 1:
                    raise JournalCorruptionError(
                        f"{event.kind} references an unknown observation"
                    )
            selected.commit()
        finally:
            if owns_connection:
                selected.close()

    @staticmethod
    def _fts_query(text: str) -> str:
        terms = _TOKEN_RE.findall(text)[:20]
        if not terms:
            raise ValueError("memory search text contains no searchable terms")
        return " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms)

    def _observation_record(self, observation_id: str) -> ObservationRecord:
        self._ensure_index_synced()
        with closing(self._connect_index()) as connection, connection:
            row = connection.execute(
                "SELECT record_json FROM observations WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown observation_id: {observation_id}")
        return ObservationRecord.model_validate_json(row["record_json"])

    def _card_for_id(self, observation_id: str) -> ObservationCard:
        self._ensure_index_synced()
        with closing(self._connect_index()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM observations WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown observation_id: {observation_id}")
        return self._card_from_row(row)

    def _select_cards(
        self,
        connection: sqlite3.Connection,
        suffix: str,
        values: tuple[Any, ...],
        limit: int,
    ) -> tuple[ObservationCard, ...]:
        rows = connection.execute(
            f"SELECT * FROM observations {suffix} LIMIT ?",
            (*values, limit),
        ).fetchall()
        return tuple(self._card_from_row(row) for row in rows)

    @staticmethod
    def _card_from_row(row: sqlite3.Row) -> ObservationCard:
        record = ObservationRecord.model_validate_json(row["record_json"])
        targets = tuple(
            dict.fromkeys(
                (
                    *record.interaction.target_object_ids,
                    *record.interaction.target_prim_paths,
                )
            )
        )
        return ObservationCard(
            observation_id=record.observation_id,
            sequence=int(row["sequence"]),
            phase=record.phase,
            operation=record.interaction.operation,
            targets=targets,
            outcome=record.outcome.classification,
            summary=record.outcome.summary,
            artifact_roles=tuple(item.role for item in record.artifacts),
            importance=record.importance,
            tags=record.tags,
            pinned=bool(row["pinned"]),
        )


__all__ = [
    "AgentMemory",
    "JournalCorruptionError",
    "MaterializedMemoryArtifact",
    "MemoryArtifactError",
    "MemoryArtifactInput",
    "MemoryArtifactReference",
    "MemoryBoundsError",
    "MemoryContextResult",
    "MemoryError",
    "MemoryExpectation",
    "MemoryInspectResult",
    "MemoryInteraction",
    "MemoryOutcome",
    "MemorySceneIdentity",
    "MemorySearchQuery",
    "ObservationCard",
    "ObservationRecord",
    "RememberRequest",
    "resolve_memory_root",
]
