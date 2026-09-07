# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical artifacts for trusted external-runtime tuning and refinement."""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
from pathlib import Path
from typing import Any

from physics_agent.tuning.types import TrialRecord

from .types import ExternalTuneSpec

ARTIFACT_SCHEMA_VERSION = 1

QUALIFICATION = "qualification.json"
QUALIFICATION_FRAMES = "qualification_frames.json"
APPROVAL = "approval.json"
RUN_SPEC = "run_spec.json"
HISTORY = "history.jsonl"
BEST_PARAMS = "best_params.json"
RESULTS = "external_tune_results.json"
BEST_RECORDING = "best_recording.usd"
MANIFEST = "manifest.json"

_PUBLIC_TUNE_FILES = frozenset(
    {
        QUALIFICATION,
        QUALIFICATION_FRAMES,
        APPROVAL,
        RUN_SPEC,
        HISTORY,
        BEST_PARAMS,
        RESULTS,
        BEST_RECORDING,
    }
)
_PUBLIC_TUNE_PREFIXES = ("qualification_frames/", "outputs/", "render/")
_MEDIA_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".png": "image/png",
    ".usd": "model/vnd.usd",
    ".usda": "model/vnd.usd",
    ".usdc": "model/vnd.usd",
}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def file_sha256(path: Path) -> str:
    """Return a prefixed SHA-256 digest for one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_descriptor(path: Path, *, relative_path: str) -> dict[str, Any]:
    """Return portable integrity metadata for a persisted artifact."""

    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "path": relative_path,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": media_type,
    }


def external_trial_payload(
    record: TrialRecord,
    *,
    include_internal_artifacts: bool = True,
) -> dict[str, Any]:
    """Serialize one trial with a common core and optional BYOR internals."""

    replicas: list[dict[str, Any]] = []
    for replica in record.replicas:
        payload = replica.to_dict()
        if not include_internal_artifacts:
            payload.pop("artifacts", None)
            payload.pop("artifact_metadata", None)
            payload.pop("trial_dir", None)
        replicas.append(payload)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "mode": "external_runtime",
        "trial_index": record.trial_index,
        "params": record.params,
        "score": record.optimizer_score,
        "backend_metrics": record.backend_metrics,
        "duration_seconds": record.duration_seconds,
        "failed": record.failed,
        "error": record.error,
        "objective_value": record.objective_value,
        "optimizer_score": record.optimizer_score,
        "success": record.success,
        "replicas": replicas,
    }


def summarize_external_trials(
    history: list[TrialRecord],
) -> list[dict[str, Any]]:
    """Return the compact top-trial payload shared by refine prompts and judges."""

    return [
        {
            "trial_index": record.trial_index,
            "params": record.params,
            "objective_value": record.objective_value,
            "optimizer_loss": (
                record.optimizer_score
                if math.isfinite(record.optimizer_score)
                else None
            ),
            "failed": record.failed,
        }
        for record in sorted(history, key=lambda item: item.optimizer_score)[:8]
    ]


def run_spec_payload(
    spec: ExternalTuneSpec,
    *,
    qualification_digest: str,
    fixed_params: dict[str, float],
    optimizer_used: str | None,
) -> dict[str, Any]:
    """Serialize execution choices not covered by qualification approval."""

    active_names = {parameter.name for parameter in spec.params}
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": "external_tune",
        "task": spec.task,
        "qualification_digest": qualification_digest,
        "parameter_catalog": [
            {"name": parameter.name, "integer": parameter.integer}
            for parameter in spec.parameter_catalog
        ],
        "active_parameters": [
            {
                "name": parameter.name,
                "min": parameter.min_value,
                "max": parameter.max_value,
                "integer": parameter.integer,
            }
            for parameter in spec.params
        ],
        "fixed_params": {
            name: float(value)
            for name, value in fixed_params.items()
            if name not in active_names
        },
        "objective": {
            "name": spec.objective.name,
            "unit": spec.objective.unit,
            "direction": spec.objective.direction,
            "failure_penalty": spec.objective.failure_penalty,
        },
        "optimizer": {
            "requested": spec.optimizer.name,
            "resolved": optimizer_used,
            "max_trials": spec.optimizer.max_trials,
            "seed": spec.optimizer.seed,
            "replicas": spec.optimizer.replicas,
            "replica_seed": spec.optimizer.replica_seed,
            "replica_seeds": list(spec.optimizer.replica_seeds),
        },
        "publish_artifacts": list(spec.publish_artifacts),
    }


def write_run_spec(
    output_dir: Path,
    spec: ExternalTuneSpec,
    *,
    qualification_digest: str,
    fixed_params: dict[str, float],
    optimizer_used: str | None,
) -> Path:
    """Persist the complete mutable execution contract for one sweep."""

    return _write_json(
        output_dir / RUN_SPEC,
        run_spec_payload(
            spec,
            qualification_digest=qualification_digest,
            fixed_params=fixed_params,
            optimizer_used=optimizer_used,
        ),
    )


def write_manifest(root: Path) -> Path:
    """Write an integrity manifest for every file in a curated directory."""

    root = root.resolve()
    manifest_path = root / MANIFEST
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path == manifest_path:
            continue
        relative = path.relative_to(root).as_posix()
        entries[relative] = file_descriptor(path, relative_path=relative)
    return _write_json(
        manifest_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": "external_refine_final",
            "artifacts": entries,
        },
    )


def collect_public_tune_artifacts(output_dir: Path) -> list[str]:
    """Return only canonical external-tune files intended for API download."""

    output_dir = output_dir.resolve()
    artifacts: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative in _PUBLIC_TUNE_FILES or relative.startswith(_PUBLIC_TUNE_PREFIXES):
            artifacts.append(relative)
    return artifacts


__all__ = [
    "APPROVAL",
    "ARTIFACT_SCHEMA_VERSION",
    "BEST_PARAMS",
    "BEST_RECORDING",
    "HISTORY",
    "MANIFEST",
    "QUALIFICATION",
    "QUALIFICATION_FRAMES",
    "RESULTS",
    "RUN_SPEC",
    "collect_public_tune_artifacts",
    "external_trial_payload",
    "file_descriptor",
    "file_sha256",
    "run_spec_payload",
    "summarize_external_trials",
    "write_manifest",
    "write_run_spec",
]
