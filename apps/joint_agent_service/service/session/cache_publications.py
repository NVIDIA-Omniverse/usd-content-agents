# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable cache-publication bindings for cross-instance regeneration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

CACHE_PUBLICATIONS_FIELD = "cache_publications"
CACHE_PUBLICATION_PREFIX = "artifacts/run_cache"
CACHE_NAMESPACES = ("dataset", "predictions")
PREDICTION_REPORT_PUBLICATION_ID_FIELD = "prediction_report_publication_id"
PREDICTION_REPORT_CACHE_PUBLICATIONS_FIELD = "prediction_report_cache_publications"
PREDICTION_REPORT_PUBLICATION_PREFIX = "artifacts/prediction_reports"
PIPELINE_CONFIG_PUBLICATION_ID_FIELD = "pipeline_config_publication_id"
PIPELINE_CONFIG_PUBLICATION_SHA256_FIELD = "pipeline_config_publication_sha256"
PIPELINE_CONFIG_PUBLICATION_PREFIX = "artifacts/pipeline_configs"

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def cache_publication_prefix(run_id: str, namespace: str) -> str:
    """Return the immutable store prefix for one run-owned cache namespace."""

    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid cache publication run ID: {run_id!r}")
    if namespace not in CACHE_NAMESPACES:
        raise ValueError(f"Invalid cache publication namespace: {namespace!r}")
    return f"{CACHE_PUBLICATION_PREFIX}/{run_id}/cache/{namespace}/"


def cache_publication_path(
    session_dir: Path,
    run_id: str,
    namespace: str,
) -> Path:
    """Return the local directory matching one immutable publication prefix."""

    return session_dir / cache_publication_prefix(run_id, namespace).rstrip("/")


def prediction_report_publication_key(run_id: str) -> str:
    """Return the immutable key for one on-demand prediction report."""

    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid prediction report publication ID: {run_id!r}")
    return f"{PREDICTION_REPORT_PUBLICATION_PREFIX}/{run_id}/report.html"


def prediction_report_publication_path(session_dir: Path, run_id: str) -> Path:
    """Return the local path for one immutable on-demand report."""

    return session_dir / prediction_report_publication_key(run_id)


def pipeline_config_publication_key(run_id: str) -> str:
    """Return the immutable key for one accepted pipeline configuration."""

    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid pipeline config publication ID: {run_id!r}")
    return f"{PIPELINE_CONFIG_PUBLICATION_PREFIX}/{run_id}/config.yaml"


def pipeline_config_publication_path(session_dir: Path, run_id: str) -> Path:
    """Return the local path for one immutable pipeline configuration."""

    return session_dir / pipeline_config_publication_key(run_id)


def parse_cache_publications(metadata: dict[str, Any]) -> dict[str, str] | None:
    """Validate cache bindings, distinguishing legacy absence from malformed data."""

    if CACHE_PUBLICATIONS_FIELD not in metadata:
        return None
    value = metadata[CACHE_PUBLICATIONS_FIELD]
    if not isinstance(value, dict):
        return {}

    bindings: dict[str, str] = {}
    for namespace, run_id in value.items():
        if (
            namespace not in CACHE_NAMESPACES
            or not isinstance(run_id, str)
            or not _RUN_ID_PATTERN.fullmatch(run_id)
        ):
            return {}
        bindings[namespace] = run_id
    return bindings


def bound_cache_artifact_key(
    metadata: dict[str, Any],
    relative_path: str,
) -> str | None:
    """Resolve one stable cache path through the selected immutable snapshot."""

    path = Path(relative_path)
    parts = path.parts
    if len(parts) < 3 or parts[0] != "cache" or parts[1] not in CACHE_NAMESPACES:
        return None
    bindings = parse_cache_publications(metadata)
    if bindings is None:
        return relative_path
    run_id = bindings.get(parts[1])
    if run_id is None:
        return None
    return f"{cache_publication_prefix(run_id, parts[1])}{Path(*parts[2:]).as_posix()}"
