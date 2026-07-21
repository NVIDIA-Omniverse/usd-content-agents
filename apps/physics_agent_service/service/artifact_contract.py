# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Availability-backed artifact contracts for tune and refine sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class ArtifactSpec:
    logical_name: str
    key: str
    media_type: str
    download_name: str


TUNE_ARTIFACT_SPECS = (
    ArtifactSpec(
        "best_params", "tune/best_params.json", "application/json", "best_params.json"
    ),
    ArtifactSpec(
        "tune_results",
        "tune/tune_results.json",
        "application/json",
        "tune_results.json",
    ),
    ArtifactSpec(
        "history", "tune/history.jsonl", "application/x-ndjson", "history.jsonl"
    ),
    ArtifactSpec("report", "tune/report.md", "text/markdown", "report.md"),
    ArtifactSpec(
        "tuned_usd",
        "tune/tuned_physics.usd",
        "application/octet-stream",
        "tuned_physics.usd",
    ),
    ArtifactSpec(
        "visual_comparison", "tune/comparison.png", "image/png", "comparison.png"
    ),
)

REFINE_ARTIFACT_SPECS = (
    ArtifactSpec(
        "refine_summary",
        "refine/refine_summary.json",
        "application/json",
        "refine_summary.json",
    ),
    ArtifactSpec(
        "final_scenario",
        "refine/final/scenario.yaml",
        "application/x-yaml",
        "scenario.yaml",
    ),
    ArtifactSpec(
        "final_best_params",
        "refine/final/best_params.json",
        "application/json",
        "best_params.json",
    ),
    ArtifactSpec(
        "final_tune_results",
        "refine/final/tune_results.json",
        "application/json",
        "tune_results.json",
    ),
    ArtifactSpec(
        "final_history",
        "refine/final/history.jsonl",
        "application/x-ndjson",
        "history.jsonl",
    ),
    ArtifactSpec(
        "final_judge_result",
        "refine/final/judge_result.json",
        "application/json",
        "judge_result.json",
    ),
    ArtifactSpec(
        "final_tuned_usd",
        "refine/final/tuned_physics.usd",
        "application/octet-stream",
        "tuned_physics.usd",
    ),
    ArtifactSpec(
        "final_recording_usd",
        "refine/final/recording.usd",
        "application/octet-stream",
        "recording.usd",
    ),
    ArtifactSpec(
        "final_report", "refine/final/report.md", "text/markdown", "report.md"
    ),
    ArtifactSpec(
        "final_visual_comparison",
        "refine/final/comparison.png",
        "image/png",
        "comparison.png",
    ),
)

_TUNE_ARTIFACT_KEYS = frozenset(spec.key for spec in TUNE_ARTIFACT_SPECS) | {
    "tune/tuned_physics.usda",
}
_REFINE_ARTIFACT_KEYS = frozenset(spec.key for spec in REFINE_ARTIFACT_SPECS) | {
    "refine/final/tuned_physics.usda",
}


def collect_artifact_manifest(session_dir: Path, prefix: str) -> list[str]:
    """Return the exact files produced beneath a session artifact directory."""
    normalized_prefix = prefix.strip("/")
    root = session_dir / normalized_prefix
    if not root.is_dir():
        return []
    return sorted(
        f"{normalized_prefix}/{path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def collect_public_artifact_manifest(session_dir: Path, kind: str) -> list[str]:
    """Collect only files exposed by the tune or refine download contract."""
    manifest = collect_artifact_manifest(session_dir, kind)
    if kind == "tune":
        return [key for key in manifest if key in _TUNE_ARTIFACT_KEYS]
    if kind == "refine":
        return [key for key in manifest if key in _REFINE_ARTIFACT_KEYS]
    raise ValueError(f"Unsupported artifact kind: {kind}")


async def available_artifact_keys(
    manager: Any,
    session_id: str,
    metadata: dict[str, Any],
    prefix: str,
) -> set[str]:
    """Return produced artifact keys that are currently present in the store."""
    normalized_prefix = f"{prefix.strip('/')}/"
    stored = set(await manager.list_store_keys(session_id, prefix=normalized_prefix))
    manifest = metadata.get("artifact_manifest")
    if not isinstance(manifest, list):
        # Legacy sessions did not persist a manifest. The store listing is the
        # authoritative evidence available for those sessions.
        return {key for key in stored if key.startswith(normalized_prefix)}
    produced = {
        key
        for key in manifest
        if isinstance(key, str) and key.startswith(normalized_prefix)
    }
    return stored & produced


def artifact_name_from_key(key: str, prefix: str) -> str:
    normalized_prefix = f"{prefix.strip('/')}/"
    if not key.startswith(normalized_prefix):
        raise ValueError(f"Artifact key is outside {normalized_prefix}: {key}")
    return key[len(normalized_prefix) :]


def is_safe_artifact_name(name: str) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
