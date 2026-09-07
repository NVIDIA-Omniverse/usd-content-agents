# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry evidence composition index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
)

GEOMETRY_EVIDENCE_BUNDLE_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-evidence-bundle.v3"
)


class GeometryEvidenceReference(BaseModel):
    """One immutable upstream report or evidence artifact reference."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    path: str
    sha256: str
    producer: str
    claim_scope: str
    status: str
    severity: Literal["info", "warning", "error"] = "info"
    schema_version: str | None = None


class GeometryEvidenceBundle(BaseModel):
    """Composition index without reinterpreting upstream validator results."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GEOMETRY_EVIDENCE_BUNDLE_SCHEMA_VERSION
    source_asset: str
    geometry_usd: str
    geometry_validation_status: str
    sim_ready_status: str
    optimization_policy: str
    effective_optimization_policy: str | None = None
    runtime_validation_mode: str
    source_bundle: dict[str, Any] | None = None
    artifacts: list[GeometryEvidenceReference] = Field(default_factory=list)


def _schema_version(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("schema_version")
    return str(value) if value else None


def evidence_reference(
    *,
    kind: str,
    path: str | Path,
    producer: str,
    claim_scope: str,
    status: str,
    severity: Literal["info", "warning", "error"] = "info",
) -> GeometryEvidenceReference:
    """Build a content-addressed evidence reference."""

    resolved = Path(path).resolve()
    return GeometryEvidenceReference(
        kind=kind,
        path=str(resolved),
        sha256=file_sha256(resolved),
        producer=producer,
        claim_scope=claim_scope,
        status=status,
        severity=severity,
        schema_version=_schema_version(resolved),
    )


def write_geometry_evidence_bundle(
    bundle: GeometryEvidenceBundle, output_dir: Path
) -> Path:
    """Write the canonical geometry evidence composition index."""

    path = output_dir / "geometry_evidence_bundle.json"
    atomic_write_json(path, bundle)
    return path
