# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral validation for external geometry-authoring evidence."""

from __future__ import annotations

import os
from pathlib import Path

from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryRepresentationBinding,
    GeometrySourceBundle,
    geometry_source_bundle_id,
)

from content_agent_workflows.common.artifacts import read_contained_artifact

_MAX_SOURCE_MANIFEST_BYTES = 16 * 1024 * 1024


def load_external_source_bundle(path: str | Path) -> GeometrySourceBundle:
    """Load one canonical source manifest without accepting legacy receipts."""

    raw = Path(path).expanduser()
    absolute = Path(os.path.abspath(raw))
    resolved = absolute.resolve(strict=True)
    if resolved != absolute or not resolved.is_file():
        raise ValueError(
            "Geometry source manifest must be a regular file without symlinks"
        )
    captured = read_contained_artifact(
        resolved.parent,
        resolved.name,
        max_bytes=_MAX_SOURCE_MANIFEST_BYTES,
        capture_bytes=True,
    )
    assert captured.data is not None
    bundle = GeometrySourceBundle.model_validate_json(captured.data)
    expected_bundle_id = geometry_source_bundle_id(
        provider=bundle.producer,
        source_revision=bundle.source_revision,
        coordinate_system=bundle.coordinate_system,
        representations=bundle.representations,
        parts=bundle.parts,
        parameters=bundle.parameters,
        verification_assertions=bundle.verification_assertions,
        provenance=bundle.provenance,
        rights=bundle.rights,
    )
    if bundle.bundle_id != expected_bundle_id:
        raise ValueError("Geometry source bundle identity is invalid")
    for representation in bundle.representations:
        candidate = read_contained_artifact(
            resolved.parent,
            representation.artifact.path,
        )
        if (
            candidate.sha256 != representation.artifact.sha256
            or candidate.size_bytes != representation.artifact.size_bytes
        ):
            raise ValueError(
                "Geometry source representation differs from its immutable identity"
            )
    for artifact in bundle.provenance.input_artifacts:
        candidate = read_contained_artifact(resolved.parent, artifact.path)
        if (
            candidate.sha256 != artifact.sha256
            or candidate.size_bytes != artifact.size_bytes
        ):
            raise ValueError(
                "Geometry source provenance input differs from its immutable identity"
            )
    return bundle


def source_artifact_path(
    manifest_path: str | Path,
    artifact: GeometryArtifactBinding,
) -> Path:
    """Resolve one bound artifact while confining it to the manifest directory."""

    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    root = manifest.parent
    raw = Path(artifact.path).expanduser()
    absolute = Path(os.path.abspath(raw if raw.is_absolute() else root / raw))
    resolved = absolute.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
    ):
        raise ValueError("Geometry source artifact escapes its manifest directory")
    return resolved


def source_representation_path(
    manifest_path: str | Path,
    representation: GeometryRepresentationBinding,
) -> Path:
    """Resolve one representation while confining it to the manifest directory."""

    return source_artifact_path(manifest_path, representation.artifact)


def select_source_usd(
    manifest_path: str | Path,
    bundle: GeometrySourceBundle,
) -> tuple[GeometryRepresentationBinding, Path]:
    """Select the single render-oriented USD representation for Geometry."""

    candidates = [
        item
        for item in bundle.representations
        if item.role == "render_geometry" and item.format in {"usd", "usda", "usdc"}
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Geometry source bundle requires exactly one render-oriented USD representation"
        )
    representation = candidates[0]
    path = source_representation_path(manifest_path, representation)
    return representation, path


def validate_external_source_manifest(
    path: str | Path,
    *,
    output_path: str | Path,
    output_sha256: str,
    output_size_bytes: int,
) -> GeometrySourceBundle:
    """Validate a canonical source bundle and its selected output identity."""

    bundle = load_external_source_bundle(path)
    _representation, selected = select_source_usd(path, bundle)
    output = Path(output_path).expanduser().resolve(strict=True)
    if selected != output:
        raise ValueError("Geometry source bundle does not select the output")
    captured = read_contained_artifact(output.parent, output.name)
    if captured.size_bytes != output_size_bytes or captured.sha256 != output_sha256:
        raise ValueError("Geometry source bundle output identity changed")
    return bundle


__all__ = [
    "load_external_source_bundle",
    "select_source_usd",
    "source_artifact_path",
    "source_representation_path",
    "validate_external_source_manifest",
]
