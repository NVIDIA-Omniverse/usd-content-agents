# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare geometry and reference conditioning for material creation backends."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import threading
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError
from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.material_library_generation.creation_contract import (
    MATERIAL_CREATION_SCHEMA_VERSION,
    CreateMaterialRequest,
    MaterialColorSpace,
    MaterialConditioningArtifact,
    MaterialConditioningArtifactSource,
    MaterialConditioningKind,
    MaterialCreationError,
    MaterialCreationErrorCode,
    PreparedMaterialConditioning,
)

MATERIAL_CONDITIONING_MANIFEST_NAME = "material_conditioning_manifest.json"
MATERIAL_CONDITIONING_SCHEMA_VERSION = "material-agent-conditioning.v1"
REAL_SEED_MATERIAL_SCHEMA_VERSION = "material-agent-real-seed.v1"
OVRTX_CONDITIONING_SCHEMA_VERSION = "material-agent-ovrtx-conditioning.v1"
_CONDITIONING_DIGEST_BYTES = 12
_HASH_CHUNK_SIZE = 1024 * 1024
_RENDER_VIEW_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DISALLOWED_REAL_EVIDENCE_VALUES = frozenset(
    {
        "deterministic",
        "deterministic_fixture",
        "fake",
        "mock",
        "placeholder",
        "recipe_synthesized",
        "simulate",
        "simulated",
        "simulation",
        "synthetic",
    }
)
_REAL_EVIDENCE_SOURCES = frozenset(
    {
        MaterialConditioningArtifactSource.SOURCE_DERIVED,
        MaterialConditioningArtifactSource.RENDERER_DERIVED,
        MaterialConditioningArtifactSource.RECIPE_REFERENCE,
        MaterialConditioningArtifactSource.REQUEST_REFERENCE,
    }
)
_STEP1X_MATERIAL_CREATION_BACKEND = "step1x_material_anything"


class MaterialConditioningEvidenceMode(StrEnum):
    """Whether prepared conditioning may use deterministic fixture artifacts."""

    DETERMINISTIC_FIXTURE = "deterministic_fixture"
    REAL_EVIDENCE = "real_evidence"


@dataclass(frozen=True)
class RealMaterialConditioningInputs:
    """Explicit seed and OVRTX manifests required by real-evidence mode."""

    seed_manifest_path: Path
    seed_manifest_sha256: str
    ovrtx_manifest_path: Path

    def __post_init__(self) -> None:
        seed_manifest_path = Path(self.seed_manifest_path).resolve()
        ovrtx_manifest_path = Path(self.ovrtx_manifest_path).resolve()
        seed_manifest_sha256 = self.seed_manifest_sha256.strip().lower()
        if _SHA256_RE.fullmatch(seed_manifest_sha256) is None:
            raise ValueError("seed_manifest_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "seed_manifest_path", seed_manifest_path)
        object.__setattr__(self, "seed_manifest_sha256", seed_manifest_sha256)
        object.__setattr__(self, "ovrtx_manifest_path", ovrtx_manifest_path)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> RealMaterialConditioningInputs:
        seed_manifest_path = _resolve_config_path(
            data.get("seed_manifest_path"),
            base_dir=base_dir,
            field_name="seed_manifest_path",
        )
        ovrtx_manifest_path = _resolve_config_path(
            data.get("ovrtx_manifest_path"),
            base_dir=base_dir,
            field_name="ovrtx_manifest_path",
        )
        return cls(
            seed_manifest_path=seed_manifest_path,
            seed_manifest_sha256=str(data.get("seed_manifest_sha256", "")),
            ovrtx_manifest_path=ovrtx_manifest_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_manifest_path": self.seed_manifest_path.as_posix(),
            "seed_manifest_sha256": self.seed_manifest_sha256,
            "ovrtx_manifest_path": self.ovrtx_manifest_path.as_posix(),
        }


@dataclass(frozen=True)
class MaterialConditioningOptions:
    """Conditioning artifact options."""

    image_size: int = 64
    render_views: tuple[str, ...] = ("oblique",)
    include_normal: bool = True
    include_depth: bool = True
    include_segmentation: bool = True
    include_source_albedo: bool = True
    evidence_mode: MaterialConditioningEvidenceMode = (
        MaterialConditioningEvidenceMode.DETERMINISTIC_FIXTURE
    )
    real_evidence: RealMaterialConditioningInputs | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.image_size <= 4096:
            raise ValueError("image_size must be in [1, 4096]")
        evidence_mode = MaterialConditioningEvidenceMode(self.evidence_mode)
        views = tuple(view.strip() for view in self.render_views)
        if (
            not views
            and evidence_mode is MaterialConditioningEvidenceMode.DETERMINISTIC_FIXTURE
        ):
            raise ValueError("render_views must contain at least one view")
        if any(not view for view in views):
            raise ValueError("render_views must contain non-empty names")
        if any(_RENDER_VIEW_RE.fullmatch(view) is None for view in views):
            raise ValueError("render_views must contain only safe filename characters")
        if len(views) != len(set(views)):
            raise ValueError("render_views must not contain duplicate names")
        object.__setattr__(self, "evidence_mode", evidence_mode)
        object.__setattr__(self, "render_views", views)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        base_dir: Path | None = None,
    ) -> MaterialConditioningOptions:
        values = dict(data or {})
        mode = MaterialConditioningEvidenceMode(
            values.get(
                "evidence_mode",
                MaterialConditioningEvidenceMode.DETERMINISTIC_FIXTURE,
            )
        )
        real_data = values.get("real_evidence")
        if real_data is not None and not isinstance(real_data, Mapping):
            raise TypeError("conditioning.real_evidence must be a mapping")
        real_evidence = (
            RealMaterialConditioningInputs.from_dict(real_data, base_dir=base_dir)
            if real_data is not None
            else None
        )
        default_views: tuple[str, ...] = (
            ()
            if mode is MaterialConditioningEvidenceMode.REAL_EVIDENCE
            else ("oblique",)
        )
        render_views = values.get("render_views", default_views)
        if isinstance(render_views, str):
            render_views = (render_views,)
        if not isinstance(render_views, list | tuple):
            raise TypeError("conditioning.render_views must be a list or tuple")
        return cls(
            image_size=int(values.get("image_size", 64)),
            render_views=tuple(str(view) for view in render_views),
            include_normal=bool(
                values.get(
                    "include_normal",
                    mode is not MaterialConditioningEvidenceMode.REAL_EVIDENCE,
                )
            ),
            include_depth=bool(
                values.get(
                    "include_depth",
                    mode is not MaterialConditioningEvidenceMode.REAL_EVIDENCE,
                )
            ),
            include_segmentation=bool(
                values.get(
                    "include_segmentation",
                    mode is not MaterialConditioningEvidenceMode.REAL_EVIDENCE,
                )
            ),
            include_source_albedo=bool(values.get("include_source_albedo", True)),
            evidence_mode=mode,
            real_evidence=real_evidence,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "image_size": self.image_size,
            "render_views": list(self.render_views),
            "include_normal": self.include_normal,
            "include_depth": self.include_depth,
            "include_segmentation": self.include_segmentation,
            "include_source_albedo": self.include_source_albedo,
            "evidence_mode": self.evidence_mode.value,
        }
        if self.real_evidence is not None:
            data["real_evidence"] = self.real_evidence.to_dict()
        return data


@dataclass(frozen=True)
class MaterialConditioningResult:
    """Prepared conditioning plus manifest metadata."""

    request_id: str
    output_dir: Path
    scoped_usd_path: Path
    manifest_path: Path
    conditioning: PreparedMaterialConditioning
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "output_dir": self.output_dir.as_posix(),
            "scoped_usd_path": self.scoped_usd_path.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "conditioning": self.conditioning.to_dict(),
            "metadata": dict(self.metadata),
        }


def prepare_material_conditioning(
    request: CreateMaterialRequest,
    output_dir: Path,
    *,
    options: MaterialConditioningOptions | None = None,
    cancel_event: threading.Event | None = None,
) -> MaterialConditioningResult:
    """Prepare scoped USD plus image/reference conditioning artifacts.

    This function never mutates ``request.source_usd``.  It writes a scoped copy
    of the composed stage and hides imageable prims outside the requested target
    subtrees so downstream renderers/backends consume the same target scope. The
    default ``deterministic_fixture`` evidence mode preserves the WP4 behavior
    by writing deterministic placeholder PNGs. Step1X requests default to
    fail-closed ``real_evidence`` mode and require explicit, hashed seed-material
    and OVRTX manifests. No deterministic or recipe-synthesized artifact is
    accepted by that path.
    """

    _raise_if_conditioning_cancelled(request, cancel_event)
    active_options = options or _default_options_for_request(request)
    if (
        request.backend == _STEP1X_MATERIAL_CREATION_BACKEND
        and active_options.evidence_mode
        is not MaterialConditioningEvidenceMode.REAL_EVIDENCE
    ):
        _raise_real_evidence_input_error(
            request,
            "Step1X requires real_evidence mode; deterministic fixtures are "
            "not backend inputs",
        )
    source_usd = request.source_usd
    if not source_usd.exists():
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            f"source USD does not exist: {source_usd}",
            backend=request.backend,
        )
    source_digest = _sha256_file(source_usd)
    _raise_if_conditioning_cancelled(request, cancel_event)
    if (
        request.source_usd_sha256 is not None
        and request.source_usd_sha256 != source_digest
    ):
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            "source_usd_sha256 does not match source_usd contents",
            backend=request.backend,
        )

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            f"failed to open source USD: {source_usd}",
            backend=request.backend,
        )
    scope = _collect_target_scope(stage, request.target_prim_paths)
    _raise_if_conditioning_cancelled(request, cancel_event)
    _reject_planned_placeholder_evidence_in_real_mode(request, active_options)
    real_preflight: _RealEvidencePreflight | None = None
    if active_options.evidence_mode is MaterialConditioningEvidenceMode.REAL_EVIDENCE:
        real_preflight = _preflight_real_evidence(
            request,
            source_stage=stage,
            source_usd_sha256=source_digest,
            options=active_options,
            cancel_event=cancel_event,
        )

    run_dir = (
        Path(output_dir)
        / _conditioning_run_name(
            request,
            active_options,
            source_usd_sha256=source_digest,
            ovrtx_manifest_sha256=(
                real_preflight.ovrtx.manifest_sha256
                if real_preflight is not None
                else None
            ),
        )
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    scoped_usd_path = run_dir / "scoped.usda"
    _write_scoped_usd(stage, request.target_prim_paths, scoped_usd_path)
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)

    artifacts: list[MaterialConditioningArtifact] = []

    real_evidence_metadata: dict[str, Any] | None = None
    if real_preflight is not None:
        real_artifacts, real_evidence_metadata = _prepare_real_evidence_artifacts(
            request,
            scoped_usd_path=scoped_usd_path,
            run_dir=run_dir,
            preflight=real_preflight,
            cancel_event=cancel_event,
        )
        artifacts.extend(real_artifacts)
    else:
        artifacts.extend(
            _prepare_deterministic_fixture_artifacts(
                request,
                run_dir=run_dir,
                options=active_options,
                cancel_event=cancel_event,
            )
        )

    artifacts.insert(
        0,
        _artifact(
            MaterialConditioningKind.SCOPED_USD,
            scoped_usd_path,
            sha256=_sha256_file(scoped_usd_path),
            evidence_source=MaterialConditioningArtifactSource.SOURCE_DERIVED,
            evidence_source_detail=(
                "source USD copy scoped to the requested target prim paths"
            ),
        ),
    )
    artifacts.extend(_reference_artifacts(request))
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
    _reject_non_real_evidence_in_real_mode(
        request,
        artifacts=tuple(artifacts),
        evidence_mode=active_options.evidence_mode,
    )
    conditioning = PreparedMaterialConditioning.for_request(
        request,
        artifacts=tuple(artifacts),
    )
    metadata = _json_dict(
        {
            "schema_version": MATERIAL_CONDITIONING_SCHEMA_VERSION,
            "creation_schema_version": MATERIAL_CREATION_SCHEMA_VERSION,
            "evidence_mode": active_options.evidence_mode.value,
            "request": request.to_dict(),
            "source_usd_sha256": source_digest,
            "target_scope": scope,
            "options": active_options.to_dict(),
            "artifact_provenance": _artifact_provenance(
                tuple(artifacts),
                evidence_mode=active_options.evidence_mode,
            ),
            "real_evidence": real_evidence_metadata,
            "conditioning": conditioning.to_dict(),
        }
    )
    manifest_path = run_dir / MATERIAL_CONDITIONING_MANIFEST_NAME
    _write_json(manifest_path, metadata)
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
    return MaterialConditioningResult(
        request_id=request.request_id,
        output_dir=run_dir,
        scoped_usd_path=scoped_usd_path,
        manifest_path=manifest_path,
        conditioning=conditioning,
        metadata=metadata,
    )


def _raise_if_conditioning_cancelled(
    request: CreateMaterialRequest,
    cancel_event: threading.Event | None,
    *,
    cleanup_dir: Path | None = None,
) -> None:
    if cancel_event is None or not cancel_event.is_set():
        return
    if cleanup_dir is not None:
        shutil.rmtree(cleanup_dir, ignore_errors=True)
    raise MaterialCreationError(
        MaterialCreationErrorCode.CANCELLED,
        "material conditioning cancelled",
        backend=request.backend,
    )


def _artifact(
    kind: MaterialConditioningKind,
    path: Path,
    *,
    color_space: MaterialColorSpace | None = None,
    view: str | None = None,
    sha256: str | None = None,
    evidence_source: MaterialConditioningArtifactSource | None = None,
    evidence_source_detail: str | None = None,
) -> MaterialConditioningArtifact:
    return MaterialConditioningArtifact(
        kind=kind,
        uri=path.as_posix(),
        color_space=color_space,
        view=view,
        sha256=sha256,
        evidence_source=evidence_source,
        evidence_source_detail=evidence_source_detail,
    )


def _default_options_for_request(
    request: CreateMaterialRequest,
) -> MaterialConditioningOptions:
    if request.backend != _STEP1X_MATERIAL_CREATION_BACKEND:
        return MaterialConditioningOptions()
    return MaterialConditioningOptions(
        render_views=(),
        include_normal=False,
        include_depth=False,
        include_segmentation=False,
        include_source_albedo=True,
        evidence_mode=MaterialConditioningEvidenceMode.REAL_EVIDENCE,
    )


def _prepare_deterministic_fixture_artifacts(
    request: CreateMaterialRequest,
    *,
    run_dir: Path,
    options: MaterialConditioningOptions,
    cancel_event: threading.Event | None,
) -> tuple[MaterialConditioningArtifact, ...]:
    artifacts: list[MaterialConditioningArtifact] = []
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
    uv_layout = run_dir / "uv_layout.png"
    uv_mask = run_dir / "uv_mask.png"
    _write_rgb_png(uv_layout, (64, 128, 255), options.image_size)
    _write_rgb_png(uv_mask, (255, 255, 255), options.image_size)
    artifacts.extend(
        [
            _artifact(
                MaterialConditioningKind.UV_LAYOUT,
                uv_layout,
                color_space=MaterialColorSpace.RAW,
                view="uv_layout",
                sha256=_sha256_file(uv_layout),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture UV layout placeholder",
            ),
            _artifact(
                MaterialConditioningKind.UV_MASK,
                uv_mask,
                color_space=MaterialColorSpace.RAW,
                view="uv_mask",
                sha256=_sha256_file(uv_mask),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture UV mask placeholder",
            ),
        ]
    )

    if options.include_normal:
        _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
        normal = run_dir / "normal.png"
        _write_rgb_png(normal, (128, 128, 255), options.image_size)
        artifacts.append(
            _artifact(
                MaterialConditioningKind.NORMAL,
                normal,
                color_space=MaterialColorSpace.RAW,
                view="normal",
                sha256=_sha256_file(normal),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture normal placeholder",
            )
        )
    if options.include_depth:
        _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
        depth = run_dir / "depth.png"
        _write_rgb_png(depth, (128, 128, 128), options.image_size)
        artifacts.append(
            _artifact(
                MaterialConditioningKind.DEPTH,
                depth,
                color_space=MaterialColorSpace.RAW,
                view="depth",
                sha256=_sha256_file(depth),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture depth placeholder",
            )
        )
    if options.include_segmentation:
        _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
        segmentation = run_dir / "segmentation.png"
        _write_rgb_png(segmentation, (255, 255, 255), options.image_size)
        artifacts.append(
            _artifact(
                MaterialConditioningKind.SEGMENTATION,
                segmentation,
                color_space=MaterialColorSpace.RAW,
                view="target_mask",
                sha256=_sha256_file(segmentation),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture segmentation placeholder",
            )
        )

    recipe_rgb = _rgb_from_recipe(request.recipe.base_color_hint)
    for view in options.render_views:
        _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
        render = run_dir / f"render_{view}.png"
        _write_rgb_png(render, recipe_rgb, options.image_size)
        artifacts.append(
            _artifact(
                MaterialConditioningKind.RENDER,
                render,
                color_space=MaterialColorSpace.SRGB,
                view=view,
                sha256=_sha256_file(render),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture render placeholder",
            )
        )

    if options.include_source_albedo:
        _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
        source_albedo = run_dir / "source_albedo.png"
        _write_rgb_png(source_albedo, recipe_rgb, options.image_size)
        artifacts.append(
            _artifact(
                MaterialConditioningKind.SOURCE_ALBEDO,
                source_albedo,
                color_space=MaterialColorSpace.SRGB,
                view="source_albedo",
                sha256=_sha256_file(source_albedo),
                evidence_source=MaterialConditioningArtifactSource.PLACEHOLDER,
                evidence_source_detail="deterministic fixture source-albedo placeholder",
            )
        )
    return tuple(artifacts)


@dataclass(frozen=True)
class _SeedMaterialPackage:
    manifest_path: Path
    manifest_sha256: str
    package_id: str
    package_revision: str
    material_usd_path: Path
    material_usd_sha256: str
    material_prim_path: str
    source_albedo_path: Path
    source_albedo_sha256: str
    source_uri: str
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class _OvrtxRenderEvidence:
    path: Path
    sha256: str
    view: str


@dataclass(frozen=True)
class _OvrtxEvidence:
    manifest_path: Path
    manifest_sha256: str
    provider_revision: str
    request_id: str
    request_sha256: str
    request: dict[str, Any]
    renders: tuple[_OvrtxRenderEvidence, ...]


@dataclass(frozen=True)
class _RealEvidencePreflight:
    seed: _SeedMaterialPackage
    mesh_st: dict[str, Any]
    ovrtx: _OvrtxEvidence


def _preflight_real_evidence(
    request: CreateMaterialRequest,
    *,
    source_stage: Any,
    source_usd_sha256: str,
    options: MaterialConditioningOptions,
    cancel_event: threading.Event | None,
) -> _RealEvidencePreflight:
    inputs = options.real_evidence
    if inputs is None:
        _raise_real_evidence_input_error(
            request,
            "explicit seed-material and OVRTX manifests are required",
        )
    if not options.include_source_albedo:
        _raise_real_evidence_input_error(request, "source albedo is required")

    seed = _load_seed_material_package(request, inputs)
    _raise_if_conditioning_cancelled(request, cancel_event)
    mesh_st = _collect_mesh_st_provenance(
        request,
        source_stage=source_stage,
        source_usd_sha256=source_usd_sha256,
    )
    _raise_if_conditioning_cancelled(request, cancel_event)
    ovrtx = _load_ovrtx_evidence(
        request,
        inputs,
        source_usd_sha256=source_usd_sha256,
        seed_manifest_sha256=seed.manifest_sha256,
    )
    _raise_if_conditioning_cancelled(request, cancel_event)
    return _RealEvidencePreflight(seed=seed, mesh_st=mesh_st, ovrtx=ovrtx)


def _prepare_real_evidence_artifacts(
    request: CreateMaterialRequest,
    *,
    scoped_usd_path: Path,
    run_dir: Path,
    preflight: _RealEvidencePreflight,
    cancel_event: threading.Event | None,
) -> tuple[tuple[MaterialConditioningArtifact, ...], dict[str, Any]]:
    seed = preflight.seed
    mesh_st = preflight.mesh_st
    ovrtx = preflight.ovrtx

    material_copy = run_dir / "seed_material.usda"
    source_albedo = run_dir / f"source_albedo{seed.source_albedo_path.suffix.lower()}"
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
    _copy_evidence_file(
        request,
        seed.material_usd_path,
        material_copy,
        expected_sha256=seed.material_usd_sha256,
        label="seed material USD",
    )
    _copy_evidence_file(
        request,
        seed.source_albedo_path,
        source_albedo,
        expected_sha256=seed.source_albedo_sha256,
        label="seed source albedo",
    )
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
    _retarget_seed_material_albedo(
        request,
        material_usd_path=material_copy,
        material_prim_path=seed.material_prim_path,
        source_albedo_name=source_albedo.name,
    )
    material_path, bound_mesh_paths = _bind_seed_material(
        request,
        scoped_usd_path=scoped_usd_path,
        seed_material_path=material_copy,
        seed_material_prim_path=seed.material_prim_path,
        package_id=seed.package_id,
    )
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)

    mesh_st_path = run_dir / "mesh_st_provenance.json"
    _write_real_evidence_json(request, mesh_st_path, mesh_st)
    ovrtx_manifest_copy = run_dir / "ovrtx_request_manifest.json"
    _copy_evidence_file(
        request,
        ovrtx.manifest_path,
        ovrtx_manifest_copy,
        expected_sha256=ovrtx.manifest_sha256,
        label="OVRTX request manifest",
    )
    _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)

    artifacts = [
        _artifact(
            MaterialConditioningKind.SEED_MATERIAL,
            material_copy,
            sha256=_sha256_file(material_copy),
            evidence_source=MaterialConditioningArtifactSource.SOURCE_DERIVED,
            evidence_source_detail=(
                f"verified seed package {seed.package_id}@{seed.package_revision}"
            ),
        ),
        _artifact(
            MaterialConditioningKind.SOURCE_ALBEDO,
            source_albedo,
            color_space=MaterialColorSpace.SRGB,
            view="source_albedo",
            sha256=_sha256_file(source_albedo),
            evidence_source=MaterialConditioningArtifactSource.SOURCE_DERIVED,
            evidence_source_detail=f"verified source asset {seed.source_uri}",
        ),
        _artifact(
            MaterialConditioningKind.MESH_ST,
            mesh_st_path,
            sha256=_sha256_file(mesh_st_path),
            evidence_source=MaterialConditioningArtifactSource.SOURCE_DERIVED,
            evidence_source_detail="authored target-mesh st and topology provenance",
        ),
        _artifact(
            MaterialConditioningKind.RENDER_REQUEST,
            ovrtx_manifest_copy,
            sha256=_sha256_file(ovrtx_manifest_copy),
            evidence_source=MaterialConditioningArtifactSource.RENDERER_DERIVED,
            evidence_source_detail=(f"OVRTX request manifest {ovrtx.request_sha256}"),
        ),
    ]
    copied_renders: list[dict[str, Any]] = []
    for render in ovrtx.renders:
        _raise_if_conditioning_cancelled(request, cancel_event, cleanup_dir=run_dir)
        suffix = render.path.suffix.lower() or ".png"
        copied_path = run_dir / "ovrtx" / f"render_{render.view}{suffix}"
        _copy_evidence_file(
            request,
            render.path,
            copied_path,
            expected_sha256=render.sha256,
            label=f"OVRTX render {render.view}",
        )
        artifacts.append(
            _artifact(
                MaterialConditioningKind.RENDER,
                copied_path,
                color_space=MaterialColorSpace.SRGB,
                view=render.view,
                sha256=_sha256_file(copied_path),
                evidence_source=MaterialConditioningArtifactSource.RENDERER_DERIVED,
                evidence_source_detail=(
                    f"OVRTX output for request {ovrtx.request_sha256}"
                ),
            )
        )
        copied_renders.append(
            {
                "view": render.view,
                "path": copied_path.as_posix(),
                "sha256": render.sha256,
            }
        )

    return tuple(artifacts), {
        "seed_material": {
            "package_id": seed.package_id,
            "package_revision": seed.package_revision,
            "manifest_path": seed.manifest_path.as_posix(),
            "manifest_sha256": seed.manifest_sha256,
            "material_usd_sha256": seed.material_usd_sha256,
            "source_albedo_sha256": seed.source_albedo_sha256,
            "source_uri": seed.source_uri,
            "source_metadata": seed.source_metadata,
            "material_path": material_path,
            "bound_mesh_paths": bound_mesh_paths,
        },
        "mesh_st": mesh_st,
        "ovrtx": {
            "manifest_path": ovrtx.manifest_path.as_posix(),
            "manifest_sha256": ovrtx.manifest_sha256,
            "provider_revision": ovrtx.provider_revision,
            "request_id": ovrtx.request_id,
            "request_sha256": ovrtx.request_sha256,
            "request": ovrtx.request,
            "renders": copied_renders,
        },
    }


def _load_seed_material_package(
    request: CreateMaterialRequest,
    inputs: RealMaterialConditioningInputs,
) -> _SeedMaterialPackage:
    manifest_path = inputs.seed_manifest_path
    if not manifest_path.is_file():
        _raise_real_evidence_input_error(
            request, f"seed manifest does not exist: {manifest_path}"
        )
    data, manifest_sha256 = _read_hashed_json_object(
        request,
        manifest_path,
        label="seed manifest",
    )
    if manifest_sha256 != inputs.seed_manifest_sha256:
        _raise_real_evidence_input_error(
            request, "seed manifest SHA-256 does not match the configured digest"
        )
    _reject_disallowed_manifest_values(request, data, label="seed manifest")
    if data.get("schema_version") != REAL_SEED_MATERIAL_SCHEMA_VERSION:
        _raise_real_evidence_input_error(
            request, "seed manifest has an unsupported schema_version"
        )

    package_id = _required_manifest_string(request, data, "package_id")
    package_revision = _required_manifest_string(request, data, "package_revision")
    material_data = _required_manifest_mapping(request, data, "material_usd")
    albedo_data = _required_manifest_mapping(request, data, "source_albedo")
    source_data = _required_manifest_mapping(request, data, "source")
    source_kind = _required_manifest_string(request, source_data, "kind")
    if source_kind not in {"approved_s3", "checked_in"}:
        _raise_real_evidence_input_error(
            request,
            "seed source kind must be approved_s3 or checked_in; synthetic sources "
            "are not accepted",
        )
    source_uri = _required_manifest_string(request, source_data, "uri")
    if source_kind == "approved_s3" and urlparse(source_uri).scheme != "s3":
        _raise_real_evidence_input_error(
            request, "approved_s3 seed source must provide an s3:// URI"
        )
    source_metadata = {
        "kind": source_kind,
        "uri": source_uri,
        "etag": _required_manifest_string(request, source_data, "etag"),
        "last_modified": _required_manifest_string(
            request, source_data, "last_modified"
        ),
    }
    if isinstance(source_data.get("content_type"), str):
        source_metadata["content_type"] = source_data["content_type"]
    if isinstance(source_data.get("byte_size"), int):
        source_metadata["byte_size"] = source_data["byte_size"]

    package_root = manifest_path.parent.resolve()
    material_usd_path = _resolve_manifest_member(
        request,
        package_root,
        _required_manifest_string(request, material_data, "path"),
        label="seed material USD",
    )
    source_albedo_path = _resolve_manifest_member(
        request,
        package_root,
        _required_manifest_string(request, albedo_data, "path"),
        label="seed source albedo",
    )
    material_usd_sha256 = _required_sha256(
        request, material_data, "sha256", label="seed material USD"
    )
    source_albedo_sha256 = _required_sha256(
        request, albedo_data, "sha256", label="seed source albedo"
    )
    _verify_manifest_file(
        request,
        material_usd_path,
        material_usd_sha256,
        label="seed material USD",
    )
    _verify_manifest_file(
        request,
        source_albedo_path,
        source_albedo_sha256,
        label="seed source albedo",
    )
    albedo_size = _verify_image_file(
        request,
        source_albedo_path,
        label="seed source albedo",
        reject_flat=True,
    )
    if (
        isinstance(source_data.get("byte_size"), int)
        and source_albedo_path.stat().st_size != source_data["byte_size"]
    ):
        _raise_real_evidence_input_error(
            request, "seed source albedo byte_size does not match its manifest"
        )
    for axis, actual in zip(("width", "height"), albedo_size, strict=True):
        expected = albedo_data.get(axis)
        if isinstance(expected, int) and expected != actual:
            _raise_real_evidence_input_error(
                request, f"seed source albedo {axis} does not match its manifest"
            )
    material_prim_path = _required_manifest_string(
        request, material_data, "material_prim_path"
    )
    _validate_seed_material_usd(
        request,
        material_usd_path=material_usd_path,
        material_prim_path=material_prim_path,
        source_albedo_path=source_albedo_path,
    )
    return _SeedMaterialPackage(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        package_id=package_id,
        package_revision=package_revision,
        material_usd_path=material_usd_path,
        material_usd_sha256=material_usd_sha256,
        material_prim_path=material_prim_path,
        source_albedo_path=source_albedo_path,
        source_albedo_sha256=source_albedo_sha256,
        source_uri=source_uri,
        source_metadata=source_metadata,
    )


def _load_ovrtx_evidence(
    request: CreateMaterialRequest,
    inputs: RealMaterialConditioningInputs,
    *,
    source_usd_sha256: str,
    seed_manifest_sha256: str,
) -> _OvrtxEvidence:
    manifest_path = inputs.ovrtx_manifest_path
    if not manifest_path.is_file():
        _raise_real_evidence_input_error(
            request, f"OVRTX request manifest does not exist: {manifest_path}"
        )
    data, manifest_sha256 = _read_hashed_json_object(
        request,
        manifest_path,
        label="OVRTX request manifest",
    )
    _reject_disallowed_manifest_values(request, data, label="OVRTX request manifest")
    if data.get("schema_version") != OVRTX_CONDITIONING_SCHEMA_VERSION:
        _raise_real_evidence_input_error(
            request, "OVRTX request manifest has an unsupported schema_version"
        )
    if data.get("provider") != "ovrtx":
        _raise_real_evidence_input_error(
            request, "real conditioning requires provider=ovrtx"
        )
    provider_revision = _required_manifest_string(request, data, "provider_revision")
    request_id = _required_manifest_string(request, data, "request_id")
    if request_id != request.request_id:
        _raise_real_evidence_input_error(
            request, "OVRTX request ID does not match this request"
        )
    if data.get("simulate") is not False:
        _raise_real_evidence_input_error(
            request, "OVRTX request manifest must record simulate=false"
        )
    request_data = _required_manifest_mapping(request, data, "request")
    renderer_data = _required_manifest_mapping(request, request_data, "renderer")
    if renderer_data.get("backend") != "ovrtx":
        _raise_real_evidence_input_error(
            request, "OVRTX request renderer backend must be ovrtx"
        )
    request_sha256 = _required_sha256(
        request, data, "request_sha256", label="OVRTX request"
    )
    if request_sha256 != _sha256_json(request_data):
        _raise_real_evidence_input_error(
            request, "OVRTX request SHA-256 does not match its request payload"
        )
    if request_data.get("source_usd_sha256") != source_usd_sha256:
        _raise_real_evidence_input_error(
            request, "OVRTX request source USD SHA-256 does not match this request"
        )
    if request_data.get("seed_manifest_sha256") != seed_manifest_sha256:
        _raise_real_evidence_input_error(
            request, "OVRTX request seed manifest SHA-256 does not match"
        )
    if tuple(request_data.get("target_prim_paths", ())) != request.target_prim_paths:
        _raise_real_evidence_input_error(
            request, "OVRTX request target prim paths do not match this request"
        )

    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        _raise_real_evidence_input_error(
            request, "OVRTX request manifest requires render artifacts"
        )
    renders: list[_OvrtxRenderEvidence] = []
    seen_views: set[str] = set()
    manifest_root = manifest_path.parent.resolve()
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping) or raw.get("kind") != "render":
            _raise_real_evidence_input_error(
                request, "OVRTX artifacts must be render mappings"
            )
        if raw.get("evidence_source") != "renderer_derived":
            _raise_real_evidence_input_error(
                request, "OVRTX render artifacts must be renderer_derived"
            )
        view = _required_manifest_string(request, raw, "view")
        if _RENDER_VIEW_RE.fullmatch(view) is None or view in seen_views:
            _raise_real_evidence_input_error(
                request, "OVRTX render views must be unique safe names"
            )
        seen_views.add(view)
        path = _resolve_manifest_member(
            request,
            manifest_root,
            _required_manifest_string(request, raw, "path"),
            label=f"OVRTX render {view}",
        )
        sha256 = _required_sha256(request, raw, "sha256", label=f"OVRTX render {view}")
        _verify_manifest_file(request, path, sha256, label=f"OVRTX render {view}")
        _verify_image_file(request, path, label=f"OVRTX render {view}")
        renders.append(_OvrtxRenderEvidence(path=path, sha256=sha256, view=view))
    return _OvrtxEvidence(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        provider_revision=provider_revision,
        request_id=request_id,
        request_sha256=request_sha256,
        request=dict(request_data),
        renders=tuple(renders),
    )


def _collect_mesh_st_provenance(
    request: CreateMaterialRequest,
    *,
    source_stage: Any,
    source_usd_sha256: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target_path in request.target_prim_paths:
        for prim in Usd.PrimRange(
            source_stage.GetPrimAtPath(target_path),
            Usd.TraverseInstanceProxies(),
        ):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            prim_path = str(prim.GetPath())
            if prim_path in seen:
                continue
            seen.add(prim_path)
            mesh = UsdGeom.Mesh(prim)
            bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if bound_material and bound_material.GetPrim():
                _raise_real_evidence_input_error(
                    request,
                    f"target mesh already has a bound material: {prim_path}",
                )
            primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            if not primvar or not primvar.HasAuthoredValue():
                _raise_real_evidence_input_error(
                    request, f"target mesh has no authored st primvar: {prim_path}"
                )
            st_type = primvar.GetTypeName()
            if st_type not in {
                Sdf.ValueTypeNames.Float2Array,
                Sdf.ValueTypeNames.TexCoord2fArray,
            }:
                _raise_real_evidence_input_error(
                    request,
                    f"target mesh st primvar must be a float2 array: {prim_path}",
                )
            values = primvar.GetAttr().Get()
            if values is None or len(values) == 0:
                _raise_real_evidence_input_error(
                    request, f"target mesh has an empty st primvar: {prim_path}"
                )
            st_values = _usd_array_to_json(values)
            st_indices = _usd_array_to_json(primvar.GetIndicesAttr().Get() or ())
            face_vertex_counts = _usd_array_to_json(
                mesh.GetFaceVertexCountsAttr().Get() or ()
            )
            face_vertex_indices = _usd_array_to_json(
                mesh.GetFaceVertexIndicesAttr().Get() or ()
            )
            points_payload = _usd_array_to_json(mesh.GetPointsAttr().Get() or ())
            if not face_vertex_counts or sum(face_vertex_counts) != len(
                face_vertex_indices
            ):
                _raise_real_evidence_input_error(
                    request, f"target mesh has invalid topology: {prim_path}"
                )
            if not points_payload or any(
                index < 0 or index >= len(points_payload)
                for index in face_vertex_indices
            ):
                _raise_real_evidence_input_error(
                    request, f"target mesh has invalid point indices: {prim_path}"
                )
            interpolation = str(primvar.GetInterpolation())
            expected_st_count = {
                "constant": 1,
                "uniform": len(face_vertex_counts),
                "vertex": len(points_payload),
                "varying": len(points_payload),
                "faceVarying": len(face_vertex_indices),
            }.get(interpolation)
            if expected_st_count is None:
                _raise_real_evidence_input_error(
                    request,
                    f"target mesh has unsupported st interpolation {interpolation!r}: "
                    f"{prim_path}",
                )
            flattened = primvar.ComputeFlattened()
            flattened_st_values = (
                _usd_array_to_json(flattened) if flattened is not None else []
            )
            if len(flattened_st_values) != expected_st_count:
                _raise_real_evidence_input_error(
                    request,
                    f"target mesh st cardinality does not match {interpolation} "
                    f"topology: {prim_path}",
                )
            st_payload = {
                "values": st_values,
                "indices": st_indices,
                "flattened_values": flattened_st_values,
                "interpolation": interpolation,
                "element_size": int(primvar.GetElementSize()),
            }
            topology_payload = {
                "face_vertex_counts": face_vertex_counts,
                "face_vertex_indices": face_vertex_indices,
            }
            st_sha256 = _sha256_json(st_payload)
            topology_sha256 = _sha256_json(topology_payload)
            points_sha256 = _sha256_json(points_payload)
            entries.append(
                {
                    "mesh_prim_path": prim_path,
                    "primvar_name": "st",
                    "type_name": str(st_type),
                    "interpolation": st_payload["interpolation"],
                    "element_size": st_payload["element_size"],
                    "indexed": bool(primvar.IsIndexed()),
                    "value_count": len(st_values),
                    "index_count": len(st_indices),
                    "flattened_value_count": len(flattened_st_values),
                    "st_sha256": st_sha256,
                    "topology_sha256": topology_sha256,
                    "points_sha256": points_sha256,
                    "mesh_identity_sha256": _sha256_json(
                        {
                            "mesh_prim_path": prim_path,
                            "st_sha256": st_sha256,
                            "topology_sha256": topology_sha256,
                            "points_sha256": points_sha256,
                        }
                    ),
                }
            )
    if not entries:
        _raise_real_evidence_input_error(
            request, "real conditioning requires at least one target mesh"
        )
    return {
        "schema_version": "material-agent-mesh-st-provenance.v1",
        "source_usd_path": request.source_usd.as_posix(),
        "source_usd_sha256": source_usd_sha256,
        "target_prim_paths": list(request.target_prim_paths),
        "meshes": entries,
    }


def _bind_seed_material(
    request: CreateMaterialRequest,
    *,
    scoped_usd_path: Path,
    seed_material_path: Path,
    seed_material_prim_path: str,
    package_id: str,
) -> tuple[str, list[str]]:
    stage = Usd.Stage.Open(str(scoped_usd_path))
    if stage is None:
        _raise_real_evidence_input_error(
            request, f"failed to reopen scoped USD copy: {scoped_usd_path}"
        )
    target_meshes: list[Any] = []
    seen: set[str] = set()
    for target_path in request.target_prim_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target_path)):
            if not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()) in seen:
                continue
            seen.add(str(prim.GetPath()))
            bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if bound_material and bound_material.GetPrim():
                _raise_real_evidence_input_error(
                    request,
                    f"target mesh already has a bound material: {prim.GetPath()}",
                )
            target_meshes.append(prim)
    root_path = "/__MaterialCreationConditioning"
    UsdGeom.Scope.Define(stage, root_path)
    UsdGeom.Scope.Define(stage, f"{root_path}/Looks")
    material_path = f"{root_path}/Looks/{_safe_prim_name(package_id)}"
    material = UsdShade.Material.Define(stage, material_path)
    material.GetPrim().GetReferences().AddReference(
        seed_material_path.name,
        seed_material_prim_path,
    )
    for prim in target_meshes:
        if not UsdShade.MaterialBindingAPI.Apply(prim).Bind(material):
            _raise_real_evidence_input_error(
                request,
                f"failed to bind seed material to target mesh: {prim.GetPath()}",
            )
    try:
        stage.GetRootLayer().Save()
    except Exception as exc:
        _raise_real_evidence_input_error(
            request, f"failed to save seed-material binding: {exc}"
        )
    return material_path, sorted(seen)


def _retarget_seed_material_albedo(
    request: CreateMaterialRequest,
    *,
    material_usd_path: Path,
    material_prim_path: str,
    source_albedo_name: str,
) -> None:
    stage = Usd.Stage.Open(material_usd_path.as_posix())
    if stage is None:
        _raise_real_evidence_input_error(
            request, f"failed to reopen copied seed material: {material_usd_path}"
        )
    material_prim = stage.GetPrimAtPath(material_prim_path)
    texture_inputs = []
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() == "UsdUVTexture":
            texture_inputs.append(shader.GetInput("file"))
    if len(texture_inputs) != 1 or not texture_inputs[0]:
        _raise_real_evidence_input_error(
            request, "seed material must contain exactly one albedo UsdUVTexture"
        )
    if not texture_inputs[0].Set(Sdf.AssetPath(source_albedo_name)):
        _raise_real_evidence_input_error(
            request, "failed to retarget copied seed material source albedo"
        )
    try:
        stage.GetRootLayer().Save()
    except Exception as exc:
        _raise_real_evidence_input_error(
            request, f"failed to save copied seed material: {exc}"
        )


def _copy_evidence_file(
    request: CreateMaterialRequest,
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
) -> None:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    except OSError as exc:
        _raise_real_evidence_input_error(
            request, f"failed to copy {label} into conditioning: {exc}"
        )
    if _sha256_file(destination) != expected_sha256:
        _raise_real_evidence_input_error(
            request, f"copied {label} SHA-256 changed during conditioning"
        )


def _write_real_evidence_json(
    request: CreateMaterialRequest,
    path: Path,
    payload: dict[str, Any],
) -> None:
    try:
        _write_json(path, payload)
    except OSError as exc:
        _raise_real_evidence_input_error(
            request, f"failed to write real-evidence provenance: {exc}"
        )


def _validate_seed_material_usd(
    request: CreateMaterialRequest,
    *,
    material_usd_path: Path,
    material_prim_path: str,
    source_albedo_path: Path,
) -> None:
    sdf_path = Sdf.Path(material_prim_path)
    if not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
        _raise_real_evidence_input_error(
            request, "seed material_prim_path must be an absolute USD prim path"
        )
    stage = Usd.Stage.Open(material_usd_path.as_posix())
    if stage is None:
        _raise_real_evidence_input_error(
            request, f"failed to open seed material USD: {material_usd_path}"
        )
    material_prim = stage.GetPrimAtPath(material_prim_path)
    if not material_prim or not material_prim.IsA(UsdShade.Material):
        _raise_real_evidence_input_error(
            request, "seed material USD does not contain the declared material prim"
        )
    try:
        expected_albedo = Path(
            os.path.relpath(source_albedo_path, material_usd_path.parent)
        ).as_posix()
    except ValueError as exc:
        _raise_real_evidence_input_error(
            request, f"seed source albedo cannot be resolved from material USD: {exc}"
        )
    texture_assets: list[str] = []
    texture_shaders: list[Any] = []
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        value = shader.GetInput("file").Get()
        if isinstance(value, Sdf.AssetPath) and value.path:
            texture_assets.append(value.path)
            texture_shaders.append(shader)
    if texture_assets != [expected_albedo]:
        _raise_real_evidence_input_error(
            request,
            "seed material USD must reference only the declared source albedo",
        )
    st_input = texture_shaders[0].GetInput("st")
    st_sources, invalid_st_sources = (
        st_input.GetConnectedSources() if st_input else ([], [])
    )
    if invalid_st_sources or len(st_sources) != 1:
        _raise_real_evidence_input_error(
            request, "seed material source albedo must sample the st primvar"
        )
    reader = UsdShade.Shader(st_sources[0].source.GetPrim())
    if (
        not reader
        or reader.GetIdAttr().Get() != "UsdPrimvarReader_float2"
        or reader.GetInput("varname").Get() != "st"
    ):
        _raise_real_evidence_input_error(
            request, "seed material source albedo must sample the st primvar"
        )
    surface_output = UsdShade.Material(material_prim).GetSurfaceOutput()
    surface_sources, invalid_surface_sources = (
        surface_output.GetConnectedSources() if surface_output else ([], [])
    )
    if invalid_surface_sources or len(surface_sources) != 1:
        _raise_real_evidence_input_error(
            request, "seed material source albedo must drive the material surface"
        )
    surface_shader = UsdShade.Shader(surface_sources[0].source.GetPrim())
    if not surface_shader:
        _raise_real_evidence_input_error(
            request, "seed material source albedo must drive the material surface"
        )
    texture_prim = texture_shaders[0].GetPrim()
    drives_surface = False
    for shader_input in surface_shader.GetInputs():
        input_sources, _ = shader_input.GetConnectedSources()
        if any(
            source.source.GetPrim() == texture_prim and str(source.sourceName) == "rgb"
            for source in input_sources
        ):
            drives_surface = True
            break
    if not drives_surface:
        _raise_real_evidence_input_error(
            request, "seed material source albedo must drive the material surface"
        )


def _resolve_config_path(
    value: Any,
    *,
    base_dir: Path | None,
    field_name: str,
) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required")
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()


def _resolve_manifest_member(
    request: CreateMaterialRequest,
    package_root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    path = (package_root / relative_path).resolve()
    if not path.is_relative_to(package_root):
        _raise_real_evidence_input_error(
            request, f"{label} must stay within its manifest directory"
        )
    return path


def _read_hashed_json_object(
    request: CreateMaterialRequest,
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        data = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _raise_real_evidence_input_error(request, f"failed to read {label}: {exc}")
    if not isinstance(data, dict):
        _raise_real_evidence_input_error(request, f"{label} must contain an object")
    return cast(dict[str, Any], data), hashlib.sha256(payload).hexdigest()


def _reject_disallowed_manifest_values(
    request: CreateMaterialRequest,
    value: Any,
    *,
    label: str,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, str):
        normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized).lower().strip("_")
        bounded = f"_{normalized}_"
        disallowed = next(
            (
                marker
                for marker in _DISALLOWED_REAL_EVIDENCE_VALUES
                if f"_{marker}_" in bounded
            ),
            None,
        )
        if disallowed is not None:
            _raise_real_evidence_input_error(
                request,
                f"{label} uses disallowed {disallowed!r} evidence at "
                f"{'.'.join(path) or '<root>'}",
            )
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            child_path = (*path, key)
            if key in {"simulate", "simulate_mode", "simulation"} and child is True:
                _raise_real_evidence_input_error(
                    request,
                    f"{label} enables simulate mode at {'.'.join(child_path)}",
                )
            if child is True and any(
                marker in key for marker in _DISALLOWED_REAL_EVIDENCE_VALUES
            ):
                _raise_real_evidence_input_error(
                    request,
                    f"{label} enables disallowed evidence flag at "
                    f"{'.'.join(child_path)}",
                )
            _reject_disallowed_manifest_values(
                request,
                child,
                label=label,
                path=child_path,
            )
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _reject_disallowed_manifest_values(
                request,
                child,
                label=label,
                path=(*path, f"[{index}]"),
            )


def _required_manifest_mapping(
    request: CreateMaterialRequest,
    data: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        _raise_real_evidence_input_error(
            request, f"real-evidence manifest requires a {key} object"
        )
    return value


def _required_manifest_string(
    request: CreateMaterialRequest,
    data: Mapping[str, Any],
    key: str,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        _raise_real_evidence_input_error(
            request, f"real-evidence manifest requires non-empty {key}"
        )
    return value.strip()


def _required_sha256(
    request: CreateMaterialRequest,
    data: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> str:
    value = _required_manifest_string(request, data, key).lower()
    if _SHA256_RE.fullmatch(value) is None:
        _raise_real_evidence_input_error(
            request, f"{label} must declare a lowercase SHA-256 digest"
        )
    return value


def _verify_manifest_file(
    request: CreateMaterialRequest,
    path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> None:
    if not path.is_file():
        _raise_real_evidence_input_error(request, f"{label} is missing: {path}")
    if _sha256_file(path) != expected_sha256:
        _raise_real_evidence_input_error(
            request, f"{label} SHA-256 does not match its manifest"
        )


def _verify_image_file(
    request: CreateMaterialRequest,
    path: Path,
    *,
    label: str,
    reject_flat: bool = False,
) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            size = image.size
            if reject_flat:
                extrema = image.convert("RGB").getextrema()
                if all(low == high for low, high in extrema):
                    _raise_real_evidence_input_error(
                        request, f"{label} must not be a flat-color image"
                    )
            else:
                image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        _raise_real_evidence_input_error(
            request, f"{label} is not a decodable image: {exc}"
        )
    return int(size[0]), int(size[1])


def _usd_array_to_json(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if isinstance(value, bool | int | float | str):
            result.append(value)
            continue
        result.append([float(component) for component in value])
    return result


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raise_real_evidence_input_error(
    request: CreateMaterialRequest,
    message: str,
) -> NoReturn:
    raise MaterialCreationError(
        MaterialCreationErrorCode.INVALID_REQUEST,
        f"real_evidence conditioning rejected input: {message}",
        backend=request.backend,
    )


def _safe_prim_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "Material"
    if name[0].isdigit():
        name = f"Material_{name}"
    return name


def _reject_planned_placeholder_evidence_in_real_mode(
    request: CreateMaterialRequest,
    options: MaterialConditioningOptions,
) -> None:
    if options.evidence_mode is not MaterialConditioningEvidenceMode.REAL_EVIDENCE:
        return
    rejected_kinds = _planned_placeholder_kind_values(options)
    if rejected_kinds:
        _raise_non_real_evidence_error(request, rejected_kinds=rejected_kinds)


def _planned_placeholder_kind_values(
    options: MaterialConditioningOptions,
) -> tuple[str, ...]:
    kinds: set[str] = set()
    if options.render_views:
        kinds.add(MaterialConditioningKind.RENDER.value)
    if options.include_normal:
        kinds.add(MaterialConditioningKind.NORMAL.value)
    if options.include_depth:
        kinds.add(MaterialConditioningKind.DEPTH.value)
    if options.include_segmentation:
        kinds.add(MaterialConditioningKind.SEGMENTATION.value)
    return tuple(sorted(kinds))


def _reject_non_real_evidence_in_real_mode(
    request: CreateMaterialRequest,
    *,
    artifacts: tuple[MaterialConditioningArtifact, ...],
    evidence_mode: MaterialConditioningEvidenceMode,
) -> None:
    if evidence_mode is not MaterialConditioningEvidenceMode.REAL_EVIDENCE:
        return
    rejected_kinds = tuple(
        sorted(
            {
                artifact.kind.value
                for artifact in artifacts
                if artifact.evidence_source not in _REAL_EVIDENCE_SOURCES
                or artifact.sha256 is None
                or artifact.evidence_source_detail is None
            }
        )
    )
    if rejected_kinds:
        _raise_non_real_evidence_error(request, rejected_kinds=rejected_kinds)


def _raise_non_real_evidence_error(
    request: CreateMaterialRequest,
    *,
    rejected_kinds: tuple[str, ...],
) -> None:
    raise MaterialCreationError(
        MaterialCreationErrorCode.INVALID_REQUEST,
        "real_evidence conditioning mode accepts only source-derived, "
        "renderer-derived, recipe-reference, or request-reference artifacts "
        "with SHA-256 and source metadata; rejected placeholder, synthetic, "
        "unhashed, or unclassified evidence for: " + ", ".join(rejected_kinds),
        backend=request.backend,
    )


def _artifact_provenance(
    artifacts: tuple[MaterialConditioningArtifact, ...],
    *,
    evidence_mode: MaterialConditioningEvidenceMode,
) -> list[dict[str, Any]]:
    provenance: list[dict[str, Any]] = []
    for artifact in artifacts:
        entry: dict[str, Any] = {
            "kind": artifact.kind.value,
            "uri": artifact.uri,
            "evidence_mode": evidence_mode.value,
        }
        if artifact.view is not None:
            entry["view"] = artifact.view
        if artifact.sha256 is not None:
            entry["sha256"] = artifact.sha256
        if artifact.evidence_source is not None:
            entry["evidence_source"] = artifact.evidence_source.value
        if artifact.evidence_source_detail is not None:
            entry["evidence_source_detail"] = artifact.evidence_source_detail
        provenance.append(entry)
    return provenance


def _collect_target_scope(
    stage: Any, target_prim_paths: tuple[str, ...]
) -> dict[str, Any]:
    missing = [
        path for path in target_prim_paths if not stage.GetPrimAtPath(path).IsValid()
    ]
    if missing:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            "target prim path does not exist: " + ", ".join(missing),
        )

    target_types: dict[str, str] = {}
    mesh_paths: list[str] = []
    mesh_paths_with_uvs: list[str] = []
    for target_path in target_prim_paths:
        target = stage.GetPrimAtPath(target_path)
        target_types[target_path] = str(target.GetTypeName())
        for prim in Usd.PrimRange(target, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh_path = str(prim.GetPath())
            mesh_paths.append(mesh_path)
            if _mesh_has_uvs(prim):
                mesh_paths_with_uvs.append(mesh_path)

    return {
        "target_prim_paths": list(target_prim_paths),
        "target_prim_types": target_types,
        "target_mesh_paths": mesh_paths,
        "mesh_count": len(mesh_paths),
        "mesh_paths_with_uvs": mesh_paths_with_uvs,
        "uv_mesh_count": len(mesh_paths_with_uvs),
        "all_target_meshes_have_uvs": bool(mesh_paths)
        and len(mesh_paths) == len(mesh_paths_with_uvs),
    }


def _mesh_has_uvs(prim: Any) -> bool:
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    return bool(primvar and primvar.HasAuthoredValue())


def _write_scoped_usd(
    stage: Any,
    target_prim_paths: tuple[str, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.Export(str(output_path)):
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            f"failed to export scoped USD copy: {output_path}",
        )
    scoped_stage = Usd.Stage.Open(str(output_path))
    if scoped_stage is None:
        raise MaterialCreationError(
            MaterialCreationErrorCode.INVALID_REQUEST,
            f"failed to reopen scoped USD copy: {output_path}",
        )
    scoped_stage = _deinstance_for_visibility(scoped_stage, output_path)
    for prim in scoped_stage.Traverse():
        prim_path = str(prim.GetPath())
        if _is_related_to_targets(prim_path, target_prim_paths):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    scoped_stage.GetRootLayer().Save()


def _deinstance_for_visibility(scoped_stage: Any, output_path: Path) -> Any:
    while True:
        changed = False
        for prim in list(scoped_stage.Traverse()):
            if prim.IsInstance() or prim.IsInstanceable():
                prim.SetInstanceable(False)
                changed = True
        if not changed:
            return scoped_stage
        scoped_stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(output_path))
        if reopened is None:
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                f"failed to reopen de-instanced scoped USD copy: {output_path}",
            )
        scoped_stage = reopened


def _is_related_to_targets(prim_path: str, target_prim_paths: tuple[str, ...]) -> bool:
    return any(
        prim_path == target
        or prim_path.startswith(f"{target}/")
        or target.startswith(f"{prim_path}/")
        for target in target_prim_paths
    )


def _reference_artifacts(
    request: CreateMaterialRequest,
) -> tuple[MaterialConditioningArtifact, ...]:
    artifacts: list[MaterialConditioningArtifact] = []
    seen: set[str] = set()

    def append_reference(
        uri: str,
        *,
        source: MaterialConditioningArtifactSource,
        detail: str,
    ) -> None:
        if uri in seen:
            return
        seen.add(uri)
        sha256 = _sha256_uri(uri)
        artifacts.append(
            MaterialConditioningArtifact(
                kind=MaterialConditioningKind.REFERENCE_IMAGE,
                uri=uri,
                color_space=MaterialColorSpace.SRGB,
                sha256=sha256,
                evidence_source=source,
                evidence_source_detail=detail,
            )
        )

    for uri in request.recipe.reference_image_uris:
        append_reference(
            _normalize_reference_uri(uri),
            source=MaterialConditioningArtifactSource.RECIPE_REFERENCE,
            detail="recipe reference image URI",
        )
    for uri in request.reference_image_uris:
        append_reference(
            _normalize_reference_uri(uri),
            source=MaterialConditioningArtifactSource.REQUEST_REFERENCE,
            detail="request-local reference image URI",
        )
    return tuple(artifacts)


def _normalize_reference_uri(uri: str) -> str:
    normalized = uri.strip()
    if not normalized:
        raise ValueError("reference_image_uris must be a non-empty string")
    return normalized


def _sha256_uri(uri: str) -> str | None:
    parsed = urlparse(uri)
    if parsed.scheme:
        if parsed.scheme != "file":
            return None
        if parsed.netloc not in ("", "localhost"):
            return None
        path = Path(unquote(parsed.path))
        return _sha256_file(path) if path.is_file() else None
    if "://" in uri:
        return None
    path = Path(uri)
    return _sha256_file(path) if path.is_file() else None


def _rgb_from_recipe(
    base_color_hint: tuple[float, float, float],
) -> tuple[int, int, int]:
    channels = tuple(int(max(0.0, min(1.0, value)) * 255) for value in base_color_hint)
    return (channels[0], channels[1], channels[2])


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _write_rgb_png(path: Path, rgb: tuple[int, int, int], size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    scanline = b"\x00" + bytes(rgb) * size
    payload = scanline * size
    path.write_bytes(
        signature
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(payload, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conditioning_run_name(
    request: CreateMaterialRequest,
    options: MaterialConditioningOptions,
    *,
    source_usd_sha256: str,
    ovrtx_manifest_sha256: str | None,
) -> str:
    payload = {
        "schema_version": MATERIAL_CONDITIONING_SCHEMA_VERSION,
        "options": options.to_dict(),
        "source_usd_sha256": source_usd_sha256,
        "ovrtx_manifest_sha256": ovrtx_manifest_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{request.request_id}-{digest[:_CONDITIONING_DIGEST_BYTES]}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(payload, sort_keys=True)))
