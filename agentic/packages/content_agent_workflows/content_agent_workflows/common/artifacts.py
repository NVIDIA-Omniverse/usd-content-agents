# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic JSON and content-addressed artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_DIGEST_SCHEMA = "content-agent-workflows.artifact-set-digest.v1"
_CHUNK_SIZE = 1024 * 1024


def resolve_artifact_path(path: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve an artifact path, interpreting relative paths from ``base_dir``."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact is not a file: {resolved}")

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _update_file_digest(digest: Any, path: Path, logical_path: str) -> None:
    digest.update(b"file\0")
    digest.update(logical_path.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    digest.update(b"\0")


def artifact_set_digest(
    paths: Iterable[str | Path],
    *,
    base_dir: Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Digest a deterministic set of files/directories plus optional metadata."""

    resolved_paths = sorted(
        {resolve_artifact_path(path, base_dir=base_dir) for path in paths},
        key=str,
    )
    digest = hashlib.sha256()
    digest.update(_DIGEST_SCHEMA.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            dict(metadata or {}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    digest.update(b"\0")

    for root in resolved_paths:
        if not root.exists():
            raise FileNotFoundError(f"Artifact does not exist: {root}")
        digest.update(b"root\0")
        digest.update(str(root).encode("utf-8"))
        digest.update(b"\0")
        if root.is_file():
            _update_file_digest(digest, root, root.name)
            continue
        if not root.is_dir():
            raise ValueError(f"Unsupported artifact type: {root}")

        digest.update(b"directory\0")
        entries = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            if entry.is_dir():
                digest.update(b"dir\0")
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
            elif entry.is_file():
                _update_file_digest(digest, entry, relative)
            else:
                raise ValueError(f"Unsupported artifact type: {entry}")
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text through a same-directory atomic replacement."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, resolved)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return resolved


def atomic_write_json(path: str | Path, payload: BaseModel | Mapping[str, Any]) -> Path:
    """Write stable JSON through a same-directory atomic replacement."""

    if isinstance(payload, BaseModel):
        document = payload.model_dump(mode="json")
    else:
        document = dict(payload)
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return atomic_write_text(path, text)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {resolved}")
    return payload


def phase_result_digest(result: BaseModel, *, result_path: str | Path) -> str:
    """Digest a phase result's claims and every artifact it references."""

    artifact_paths = getattr(result, "artifact_paths", None)
    if not isinstance(artifact_paths, list):
        raise TypeError("Phase result must expose an artifact_paths list")
    metadata = result.model_dump(mode="json", exclude={"output_digest"})
    return artifact_set_digest(
        artifact_paths,
        base_dir=Path(result_path).expanduser().resolve().parent,
        metadata=metadata,
    )


def seal_phase_result[ModelT: BaseModel](result: ModelT, path: str | Path) -> ModelT:
    """Compute a phase output digest and atomically write the sealed result."""

    output_digest = phase_result_digest(result, result_path=path)
    sealed = result.model_copy(update={"output_digest": output_digest})
    atomic_write_json(path, sealed)
    return sealed
