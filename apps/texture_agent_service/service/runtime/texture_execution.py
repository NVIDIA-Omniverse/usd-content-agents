# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable Texture Plan execution checkpoints for service sessions."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from texture_agent.execution import TextureArtifactRef, TextureExecutionCheckpoint

from ..storage.base import SessionStore

TEXTURE_EXECUTION_CHECKPOINT_KEY = "cache/execution/texture_execution_checkpoint.json"
TEXTURE_EXECUTION_ACCEPTED_PREFIX = "cache/execution/accepted"
_SESSION_ARTIFACT_SCHEME = "session-artifact"
_ACCEPTED_PREFIX_PARTS = PurePosixPath(TEXTURE_EXECUTION_ACCEPTED_PREFIX).parts


def _validate_session_artifact_key(key: str) -> str:
    path = PurePosixPath(unquote(key))
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or ".." in parts
        or "\\" in path.as_posix()
        or parts[: len(_ACCEPTED_PREFIX_PARTS)] != _ACCEPTED_PREFIX_PARTS
    ):
        raise ValueError("Invalid texture execution session artifact key")
    return path.as_posix()


def _validate_artifact_name(name: str) -> None:
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise ValueError("Invalid texture execution artifact name")


class SessionTextureExecutionCheckpointStore:
    """Persist checkpoints and accepted artifacts through a session store.

    Accepted unit artifacts are copied under a deterministic session prefix
    before the checkpoint advertises them. A replacement worker hydrates that
    prefix and receives local artifact paths, so resume can reuse completed
    units even when the original worker filesystem disappeared.
    """

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        local_session_dir: str | Path,
        key: str = TEXTURE_EXECUTION_CHECKPOINT_KEY,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.local_session_dir = Path(local_session_dir)
        self.key = key
        self._persisted_attempts: dict[str, int] = {}

    def load(self) -> TextureExecutionCheckpoint | None:
        payload = self.store.get_json(self.session_id, self.key)
        if payload is None:
            return None
        self.store.sync_to_local(
            self.session_id,
            str(self.local_session_dir),
            f"{TEXTURE_EXECUTION_ACCEPTED_PREFIX}/",
        )
        checkpoint = TextureExecutionCheckpoint.model_validate(payload)
        localized_records = []
        for record in checkpoint.records:
            self._persisted_attempts[record.unit_id] = record.attempts
            result = record.accepted_result
            if result is None:
                localized_records.append(record)
                continue
            localized_records.append(
                record.model_copy(
                    update={
                        "accepted_result": result.model_copy(
                            update={
                                "artifacts": tuple(
                                    self._localize_artifact(artifact)
                                    for artifact in result.artifacts
                                )
                            }
                        )
                    }
                )
            )
        return checkpoint.model_copy(update={"records": tuple(localized_records)})

    def save(self, checkpoint: TextureExecutionCheckpoint) -> None:
        persisted_records = []
        for record in checkpoint.records:
            result = record.accepted_result
            if result is None:
                persisted_records.append(record)
                continue
            should_upload = (
                record.state.value == "completed"
                and self._persisted_attempts.get(record.unit_id) != record.attempts
            )
            persisted_artifacts = tuple(
                self._persist_artifact(
                    record.unit_id,
                    artifact,
                    upload=should_upload,
                )
                for artifact in result.artifacts
            )
            if should_upload:
                self._persisted_attempts[record.unit_id] = record.attempts
            persisted_records.append(
                record.model_copy(
                    update={
                        "accepted_result": result.model_copy(
                            update={"artifacts": persisted_artifacts}
                        )
                    }
                )
            )
        persisted = checkpoint.model_copy(update={"records": tuple(persisted_records)})
        self.store.put_json(
            self.session_id,
            self.key,
            persisted.model_dump(mode="json"),
        )

    def _persist_artifact(
        self,
        unit_id: str,
        artifact: TextureArtifactRef,
        *,
        upload: bool,
    ) -> TextureArtifactRef:
        parsed = urlparse(artifact.uri)
        if parsed.scheme == _SESSION_ARTIFACT_SCHEME:
            _validate_session_artifact_key(parsed.path.lstrip("/"))
            return artifact
        if parsed.scheme and parsed.scheme != "file":
            return artifact
        path = Path(unquote(parsed.path) if parsed.scheme == "file" else artifact.uri)
        if not path.is_file():
            return artifact
        try:
            path.resolve().relative_to(self.local_session_dir.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Refusing to persist a texture artifact outside the session directory"
            ) from exc
        suffix = path.suffix.lower() or ".bin"
        _validate_artifact_name(artifact.name)
        key = f"{TEXTURE_EXECUTION_ACCEPTED_PREFIX}/{unit_id}/{artifact.name}{suffix}"
        key = _validate_session_artifact_key(key)
        if upload:
            self.store.put_file(self.session_id, key, str(path))
        return artifact.model_copy(
            update={"uri": f"{_SESSION_ARTIFACT_SCHEME}:///{key}"}
        )

    def _localize_artifact(self, artifact: TextureArtifactRef) -> TextureArtifactRef:
        parsed = urlparse(artifact.uri)
        if parsed.scheme != _SESSION_ARTIFACT_SCHEME:
            return artifact
        key = _validate_session_artifact_key(parsed.path.lstrip("/"))
        local_path = (self.local_session_dir / key).resolve()
        try:
            local_path.relative_to(self.local_session_dir.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Refusing to localize a texture artifact outside the session directory"
            ) from exc
        return artifact.model_copy(update={"uri": str(local_path)})
