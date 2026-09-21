# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral artifact preparation for the geometry workflow."""

from __future__ import annotations

import math
import re
import shutil
import struct
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException
from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryCoordinateSystem,
    GeometryRepresentationBinding,
    geometry_source_bundle_id,
)
from geometry_authoring_contracts.source_dependencies import (
    DEPENDENCY_REPRESENTATION_ROLES,
    PROCESSABLE_REPRESENTATION_ROLES,
    validate_materialized_source_dependencies,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from content_agent_workflows.common.artifacts import (
    file_sha256,
    read_contained_artifact,
    snapshot_contained_artifact,
)
from content_agent_workflows.convert_to_usd.workflow import (
    ConvertToUsdWorkflowInput,
    converter_reference_for_source_extension,
    is_existing_usd,
    run_convert_to_usd_workflow,
)

from .authoring_provider import GeometrySourceBundle

SourceAuthoringMode = Literal[
    "auto",
    "opaque_import",
    "parametric_recovery",
    "direct_preserve",
    "shared_conversion",
    "lossless_gltf",
]
SourceFidelityTier = Literal[
    "opaque_cad_import",
    "parametric_recovery_request",
    "direct_usd_preserved",
    "mesh_3mf_import",
    "shared_conversion_noneditable",
    "source_gltf_preserved",
    "unsupported_requires_converter",
]

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
OPAQUE_IMPORT_SUFFIXES = {".step", ".stp", ".stl", ".obj"}
RECOVERABLE_MESH_SUFFIXES = {".stl", ".obj"}
_UNITLESS_MESH_SUFFIXES = RECOVERABLE_MESH_SUFFIXES | {".ply"}
THREEMF_SUFFIXES = {".3mf"}
BREP_SUFFIXES = {".brep", ".iges", ".igs", ".step", ".stp"}
_THREEMF_CORE_NS = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"
THREEMF_MODEL_XML_MAX_BYTES = 64 * 1024 * 1024
GEOMETRY_SOURCE_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_COORDINATE_EXTENT_REL_TOLERANCE = 1e-4
_COORDINATE_EXTENT_ABS_TOLERANCE_M = 1e-9


class PreparedGeometrySource(BaseModel):
    """Normalized geometry source plus the fidelity contract used to create it."""

    model_config = ConfigDict(extra="forbid")

    fidelity_tier: SourceFidelityTier
    source_authoring_mode: SourceAuthoringMode
    status: Literal["prepared", "admitted", "unsupported"] = "prepared"
    source_path: str | None = None
    prepared_usd_path: str | None = None
    original_input_path: str | None = None
    lossy: bool = False
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_prepared(self) -> bool:
        return self.status == "prepared" and self.prepared_usd_path is not None


class GeometrySourceBundleAdmissionError(ValueError):
    """A typed failure while admitting an immutable geometry source bundle."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class AdmittedGeometrySourceBundle:
    """Digest-verified source bundle ready for deterministic preparation."""

    bundle: GeometrySourceBundle
    selected_representation: GeometryRepresentationBinding
    selected_source: Path
    selected_source_sha256: str
    manifest_sha256: str
    representation_records: tuple[dict[str, Any], ...]
    provenance_input_records: tuple[dict[str, Any], ...]


def verify_geometry_source_identity(
    *,
    source_path: Path | None,
    source_manifest_path: Path | None,
    expected_source_sha256: str | None,
    expected_source_manifest_sha256: str | None,
) -> None:
    """Fail when a digest-bound Geometry input changed before it is read."""

    for label, path, expected in (
        ("source", source_path, expected_source_sha256),
        ("source manifest", source_manifest_path, expected_source_manifest_sha256),
    ):
        if expected is None:
            continue
        if path is None:
            raise ValueError(f"Expected {label} SHA-256 without a {label} path")
        raw = Path(path).expanduser()
        absolute = raw.absolute()
        try:
            resolved = absolute.resolve(strict=True)
        except OSError as exc:
            raise FileNotFoundError(
                f"Geometry {label} is not a file: {absolute}"
            ) from exc
        if resolved != absolute:
            raise ValueError(f"Geometry {label} path must not traverse a symlink")
        captured = read_contained_artifact(
            resolved.parent,
            resolved.name,
            allow_hardlinks=True,
        )
        observed = captured.sha256
        if observed != expected:
            raise RuntimeError(
                f"Geometry {label} changed after request admission: "
                f"expected {expected}, observed {observed}"
            )


def _source_bundle_representation(
    bundle: GeometrySourceBundle,
    *,
    requested_id: str | None,
    requested_role: str | None,
):
    if requested_id is not None:
        matches = [
            item
            for item in bundle.representations
            if item.representation_id == requested_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "geometry.source.v1 does not contain the requested representation "
                f"{requested_id!r}"
            )
        if requested_role is not None and matches[0].role != requested_role:
            raise ValueError(
                "geometry.source.v1 representation role differs from the request"
            )
        return _require_processable_representation(matches[0])
    if requested_role is not None:
        matches = [
            item for item in bundle.representations if item.role == requested_role
        ]
        if len(matches) != 1:
            raise ValueError(
                "geometry.source.v1 must contain exactly one representation for "
                f"requested role {requested_role!r}"
            )
        return _require_processable_representation(matches[0])
    for role in (
        "render_geometry",
        "design_exchange",
        "collision_candidate",
        "reference",
    ):
        matches = [item for item in bundle.representations if item.role == role]
        if matches:
            return matches[0]
    raise ValueError(
        "geometry.source.v1 has no processable exported representation; native "
        "and supporting source artifacts are retained but never executed"
    )


def _require_processable_representation(
    representation: GeometryRepresentationBinding,
) -> GeometryRepresentationBinding:
    if representation.role not in PROCESSABLE_REPRESENTATION_ROLES:
        raise ValueError(
            "geometry.source.v1 native, supporting, and metadata artifacts are retained "
            "with the bundle but are never executed or interpreted as root geometry"
        )
    return representation


def _load_geometry_source_bundle(
    manifest_path: Path,
    *,
    requested_id: str | None,
    requested_role: str | None,
) -> tuple[GeometrySourceBundle, Any, Path, str, str]:
    raw_manifest = manifest_path.expanduser().absolute()
    manifest = raw_manifest.resolve(strict=True)
    if manifest != raw_manifest:
        raise ValueError("Geometry source manifest path must not traverse a symlink")
    try:
        captured_manifest = read_contained_artifact(
            manifest.parent,
            manifest.name,
            max_bytes=GEOMETRY_SOURCE_MANIFEST_MAX_BYTES,
            capture_bytes=True,
        )
        assert captured_manifest.data is not None
        bundle = GeometrySourceBundle.model_validate_json(captured_manifest.data)
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise ValueError(f"Invalid geometry.source.v1 manifest: {exc}") from exc
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
        raise ValueError("geometry.source.v1 bundle identity is invalid")
    representation = _source_bundle_representation(
        bundle,
        requested_id=requested_id,
        requested_role=requested_role,
    )
    try:
        captured_artifact = read_contained_artifact(
            manifest.parent,
            representation.artifact.path,
            allow_hardlinks=False,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Geometry source representation is not a contained regular file"
        ) from exc
    if captured_artifact.size_bytes != representation.artifact.size_bytes:
        raise ValueError("Geometry source representation byte size changed")
    if captured_artifact.sha256 != representation.artifact.sha256:
        raise ValueError("Geometry source representation SHA-256 changed")
    return (
        bundle,
        representation,
        captured_artifact.path,
        captured_artifact.sha256,
        captured_manifest.sha256,
    )


def admit_geometry_source_bundle(
    *,
    source_manifest_path: Path,
    source_path: Path | None,
    snapshot_parent: Path,
    source_representation_id: str | None = None,
    source_representation_role: str | None = None,
    generated_usd_path: Path | None = None,
) -> AdmittedGeometrySourceBundle:
    """Verify and isolate one immutable bundle's consumable closure."""

    try:
        (
            source_bundle,
            selected_representation,
            bundled_source,
            bundled_source_sha256,
            source_manifest_sha256,
        ) = _load_geometry_source_bundle(
            source_manifest_path,
            requested_id=source_representation_id,
            requested_role=source_representation_role,
        )
    except ValueError as exc:
        raise GeometrySourceBundleAdmissionError(
            str(exc), error_code="geometry_source_bundle_invalid"
        ) from exc
    if source_path is not None and source_path.resolve() != bundled_source:
        raise GeometrySourceBundleAdmissionError(
            "source_path conflicts with the selected source-bundle representation",
            error_code="geometry_source_bundle_ambiguous",
        )
    if generated_usd_path is not None:
        raise GeometrySourceBundleAdmissionError(
            "generated_usd_path cannot override an immutable source bundle",
            error_code="geometry_source_bundle_ambiguous",
        )
    if source_bundle.coordinate_system.handedness != "right":
        raise GeometrySourceBundleAdmissionError(
            "Left-handed geometry source bundles require an explicit handedness converter.",
            error_code="geometry_source_coordinate_system_unsupported",
        )

    captured_representations: list[tuple[GeometryRepresentationBinding, Path]] = []
    captured_provenance_inputs: list[tuple[GeometryArtifactBinding, Path]] = []
    try:
        package_root = source_manifest_path.resolve().parent
        for representation in source_bundle.representations:
            captured = read_contained_artifact(
                package_root,
                representation.artifact.path,
                allow_hardlinks=False,
            )
            if (
                captured.size_bytes != representation.artifact.size_bytes
                or captured.sha256 != representation.artifact.sha256
            ):
                raise ValueError(
                    "Geometry source representation changed after provider publication"
                )
            captured_representations.append((representation, captured.path))

        for artifact in source_bundle.provenance.input_artifacts:
            try:
                captured = read_contained_artifact(
                    package_root,
                    artifact.path,
                    allow_hardlinks=False,
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "Geometry source provenance input is not a contained regular file"
                ) from exc
            if (
                captured.size_bytes != artifact.size_bytes
                or captured.sha256 != artifact.sha256
            ):
                raise ValueError(
                    "Geometry source provenance input changed after provider publication"
                )
            captured_provenance_inputs.append((artifact, captured.path))

        dependency_files = tuple(
            path
            for representation, path in captured_representations
            if representation.role in DEPENDENCY_REPRESENTATION_ROLES
        )
        for representation, path in captured_representations:
            if representation.role not in PROCESSABLE_REPRESENTATION_ROLES:
                continue
            validate_materialized_source_dependencies(
                path,
                package_root=package_root,
                package_files=dependency_files,
            )

        snapshot_parent.mkdir(parents=True, exist_ok=True)
        snapshot_root = Path(
            tempfile.mkdtemp(prefix="geometry-source-", dir=snapshot_parent.resolve())
        ).resolve()
        snapshot_paths: dict[str, Path] = {}
        snapshot_paths_by_artifact: dict[str, Path] = {}
        representation_records: list[dict[str, Any]] = []
        provenance_input_records: list[dict[str, Any]] = []
        try:
            for representation, published_path in captured_representations:
                record = representation.model_dump(mode="json")
                if representation.role not in DEPENDENCY_REPRESENTATION_ROLES:
                    representation_records.append(
                        {
                            **record,
                            "materialized_path": None,
                            "published_path": str(published_path),
                        }
                    )
                    continue
                relative = Path(*PurePosixPath(representation.artifact.path).parts)
                snapshot = snapshot_contained_artifact(
                    package_root,
                    representation.artifact.path,
                    snapshot_root / relative,
                    destination_run_dir=snapshot_root,
                    expected_sha256=representation.artifact.sha256,
                    expected_size_bytes=representation.artifact.size_bytes,
                    allow_hardlinks=False,
                )
                snapshot_paths[representation.representation_id] = snapshot.path
                snapshot_paths_by_artifact[representation.artifact.path] = snapshot.path
                representation_records.append(
                    {
                        **record,
                        "materialized_path": str(snapshot.path),
                    }
                )

            for artifact, _published_path in captured_provenance_inputs:
                snapshot_path = snapshot_paths_by_artifact.get(artifact.path)
                if snapshot_path is None:
                    relative = Path(*PurePosixPath(artifact.path).parts)
                    snapshot = snapshot_contained_artifact(
                        package_root,
                        artifact.path,
                        snapshot_root / relative,
                        destination_run_dir=snapshot_root,
                        expected_sha256=artifact.sha256,
                        expected_size_bytes=artifact.size_bytes,
                        allow_hardlinks=False,
                    )
                    snapshot_path = snapshot.path
                    snapshot_paths_by_artifact[artifact.path] = snapshot_path
                provenance_input_records.append(
                    {
                        **artifact.model_dump(mode="json"),
                        "materialized_path": str(snapshot_path),
                    }
                )

            snapshot_dependency_files = tuple(snapshot_paths.values())
            for representation, _published_path in captured_representations:
                if representation.role not in PROCESSABLE_REPRESENTATION_ROLES:
                    continue
                validate_materialized_source_dependencies(
                    snapshot_paths[representation.representation_id],
                    package_root=snapshot_root,
                    package_files=snapshot_dependency_files,
                )
            bundled_source = snapshot_paths[selected_representation.representation_id]
        except Exception:
            shutil.rmtree(snapshot_root, ignore_errors=True)
            raise
    except (OSError, ValueError) as exc:
        raise GeometrySourceBundleAdmissionError(
            str(exc), error_code="geometry_source_bundle_invalid"
        ) from exc

    return AdmittedGeometrySourceBundle(
        bundle=source_bundle,
        selected_representation=selected_representation,
        selected_source=bundled_source,
        selected_source_sha256=bundled_source_sha256,
        manifest_sha256=source_manifest_sha256,
        representation_records=tuple(representation_records),
        provenance_input_records=tuple(provenance_input_records),
    )


def prepare_geometry_source(
    *,
    source_path: Path | None,
    source_manifest_path: Path | None = None,
    source_representation_id: str | None = None,
    source_representation_role: str | None = None,
    generated_usd_path: Path | None,
    prompt: str | None,
    image_path: Path | None,
    output_dir: Path,
    source_authoring_mode: SourceAuthoringMode = "auto",
    allow_lossy_recovery: bool = False,
    param_overrides: dict[str, Any] | None = None,
    install_missing_converters: bool = False,
    converter_timeout_s: float = 120.0,
    expected_source_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    derived_source_path: Path | None = None,
    expected_derived_source_sha256: str | None = None,
) -> PreparedGeometrySource:
    """Normalize user/generator geometry into a USD artifact with explicit fidelity."""

    verify_geometry_source_identity(
        source_path=source_path,
        source_manifest_path=source_manifest_path,
        expected_source_sha256=expected_source_sha256,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if (derived_source_path is None) != (expected_derived_source_sha256 is None):
        raise ValueError(
            "derived_source_path and expected_derived_source_sha256 must be supplied together"
        )
    if derived_source_path is not None:
        if source_manifest_path is None:
            raise ValueError("A derived source requires an admitted source bundle")
        verify_geometry_source_identity(
            source_path=derived_source_path,
            source_manifest_path=None,
            expected_source_sha256=expected_derived_source_sha256,
            expected_source_manifest_sha256=None,
        )
    if source_manifest_path is not None:
        try:
            admitted = admit_geometry_source_bundle(
                source_manifest_path=source_manifest_path,
                source_path=source_path,
                snapshot_parent=output_dir / "source_prep" / "admitted_bundles",
                source_representation_id=source_representation_id,
                source_representation_role=source_representation_role,
                generated_usd_path=generated_usd_path,
            )
        except GeometrySourceBundleAdmissionError as exc:
            return _unsupported(
                source_manifest_path,
                source_authoring_mode,
                str(exc),
                error_code=exc.error_code,
            )
        source_bundle = admitted.bundle
        preparation_source = admitted.selected_source
        preparation_source_sha256 = admitted.selected_source_sha256
        if derived_source_path is not None:
            preparation_source = derived_source_path.resolve()
            if (
                admitted.selected_source.suffix.lower() not in BREP_SUFFIXES
                or preparation_source.suffix.lower() not in BREP_SUFFIXES
            ):
                raise ValueError(
                    "Native repair derivatives require B-rep source and output formats"
                )
            assert expected_derived_source_sha256 is not None
            preparation_source_sha256 = expected_derived_source_sha256
        prepared = prepare_geometry_source(
            source_path=preparation_source,
            generated_usd_path=None,
            prompt=prompt,
            image_path=image_path,
            output_dir=output_dir,
            source_authoring_mode=source_authoring_mode,
            allow_lossy_recovery=allow_lossy_recovery,
            param_overrides=param_overrides,
            install_missing_converters=install_missing_converters,
            converter_timeout_s=converter_timeout_s,
            expected_source_sha256=preparation_source_sha256,
        )
        if prepared.is_prepared:
            try:
                _apply_declared_coordinate_system(
                    prepared,
                    coordinate_system=source_bundle.coordinate_system,
                    source_path=preparation_source,
                    source_suffix=preparation_source.suffix.lower(),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                return _unsupported(
                    source_manifest_path,
                    source_authoring_mode,
                    str(exc),
                    error_code="geometry_source_coordinate_system_invalid",
                )
            assert prepared.prepared_usd_path is not None
            prepared_usd = Path(prepared.prepared_usd_path).resolve(strict=True)
            if prepared_usd == preparation_source:
                verify_geometry_source_identity(
                    source_path=prepared_usd,
                    source_manifest_path=None,
                    expected_source_sha256=preparation_source_sha256,
                    expected_source_manifest_sha256=None,
                )
                prepared.metadata["prepared_usd_sha256"] = preparation_source_sha256
            else:
                prepared.metadata["prepared_usd_sha256"] = file_sha256(prepared_usd)
        prepared.metadata["source_bundle"] = {
            "schema_version": source_bundle.schema_version,
            "bundle_id": source_bundle.bundle_id,
            "producer": source_bundle.producer.model_dump(mode="json"),
            "source_revision": source_bundle.source_revision,
            "coordinate_system": source_bundle.coordinate_system.model_dump(
                mode="json"
            ),
            "selected_representation": admitted.selected_representation.model_dump(
                mode="json"
            ),
            "representations": list(admitted.representation_records),
            "parts": [item.model_dump(mode="json") for item in source_bundle.parts],
            "parameters": [
                item.model_dump(mode="json") for item in source_bundle.parameters
            ],
            "verification_assertions": [
                item.model_dump(mode="json")
                for item in source_bundle.verification_assertions
            ],
            "provenance": source_bundle.provenance.model_dump(mode="json"),
            "provenance_input_artifacts": list(admitted.provenance_input_records),
            "rights": source_bundle.rights.model_dump(mode="json"),
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": admitted.manifest_sha256,
        }
        if derived_source_path is not None:
            prepared.metadata["source_derivation"] = {
                "kind": "native_repair",
                "source_sha256": admitted.selected_source_sha256,
                "derived_sha256": expected_derived_source_sha256,
            }
        prepared.metadata["semantic_parts"] = [
            item.model_dump(mode="json") for item in source_bundle.parts
        ]
        prepared.metadata["parameter_ranges"] = {
            item.name: {
                "value": item.value,
                "unit": item.unit,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "description": item.description,
            }
            for item in source_bundle.parameters
        }
        prepared.metadata["verification_assertions"] = [
            item.model_dump(mode="json")
            for item in source_bundle.verification_assertions
        ]
        return prepared
    prep_dir = output_dir / "source_prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    if generated_usd_path is not None:
        return _prepare_direct_usd(Path(generated_usd_path))

    if source_path is None:
        return PreparedGeometrySource(
            fidelity_tier="unsupported_requires_converter",
            source_authoring_mode=source_authoring_mode,
            status="unsupported",
            errors=[
                "Geometry generation did not provide a source artifact. Configure "
                "an authoring provider and pass its immutable source bundle or USD."
            ],
            metadata={
                "error_code": "requires_generation_artifact",
                "prompt": prompt,
                "image_path": str(image_path) if image_path else None,
            },
        )

    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Geometry source artifact not found: {source}")
    if source.suffix.lower() == ".py":
        return _unsupported(
            source,
            source_authoring_mode,
            (
                "Executable source is never run by the geometry workflow. "
                "An authoring provider must export a supported geometry artifact first."
            ),
            error_code="arbitrary_python_not_executed",
        )
    suffix = source.suffix.lower()
    existing_usd = is_existing_usd(source)
    mode_error = _source_authoring_mode_error(
        source,
        source_authoring_mode,
        existing_usd=existing_usd,
    )
    if mode_error:
        return _unsupported(
            source,
            source_authoring_mode,
            mode_error,
            error_code="source_authoring_mode_mismatch",
        )

    if existing_usd:
        return _prepare_direct_usd(source)
    if source_authoring_mode == "lossless_gltf":
        from .lossless_gltf import import_static_gltf

        destination = prep_dir / "source_preserved.usdc"
        receipt = import_static_gltf(source, destination)
        return PreparedGeometrySource(
            fidelity_tier="source_gltf_preserved",
            source_authoring_mode="lossless_gltf",
            source_path=str(source),
            original_input_path=str(source),
            prepared_usd_path=str(destination),
            metadata={"source_format": source.suffix.lower().lstrip("."), "lossless_gltf": receipt},
        )
    if suffix in THREEMF_SUFFIXES and source_authoring_mode in {
        "auto",
        "opaque_import",
    }:
        return _prepare_generic_3mf(source, prep_dir)

    if source_authoring_mode == "parametric_recovery" or (
        source_authoring_mode == "auto"
        and allow_lossy_recovery
        and suffix in RECOVERABLE_MESH_SUFFIXES
    ):
        return _prepare_parametric_recovery(
            source,
            prep_dir,
            allow_lossy_recovery=allow_lossy_recovery,
        )
    if source_authoring_mode == "opaque_import" and suffix in OPAQUE_IMPORT_SUFFIXES:
        return _prepare_opaque_import(source, prep_dir)
    if converter_reference_for_source_extension(source) is not None:
        return _prepare_shared_conversion(
            source,
            prep_dir,
            install_missing=install_missing_converters,
            timeout_s=converter_timeout_s,
        )
    return _unsupported(
        source,
        source_authoring_mode,
        f"Unsupported geometry source extension {suffix!r}.",
        error_code="unsupported_source_extension",
    )


def _stage_extent_in_stage_units(stage: Any, Usd: Any, UsdGeom: Any) -> list[float]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    world_range = cache.ComputeWorldBound(root).ComputeAlignedRange()
    if world_range.IsEmpty():
        raise ValueError("Converted provider stage has no measurable geometry bounds")
    minimum = world_range.GetMin()
    maximum = world_range.GetMax()
    extent = [float(maximum[index] - minimum[index]) for index in range(3)]
    if not all(math.isfinite(value) and value >= 0.0 for value in extent) or not any(
        value > 0.0 for value in extent
    ):
        raise ValueError("Converted provider stage has invalid geometry bounds")
    return extent


def _brep_extent_in_meters(source_path: Path) -> tuple[list[float], float]:
    try:
        from geometry_repair.brep import inspect_brep
        from OCP.UnitsMethods import UnitsMethods
    except Exception as exc:
        raise RuntimeError(
            "Native CAD coordinate verification requires an authorized B-rep backend; "
            "supply a provider-generated USD or mesh representation with verified units"
        ) from exc

    _shape, metrics = inspect_brep(source_path)
    minimum = metrics.bbox_min_source_units
    maximum = metrics.bbox_max_source_units
    if metrics.error or minimum is None or maximum is None:
        raise ValueError("Native CAD bounds could not verify converted stage units")
    cascade_meters_per_unit = float(UnitsMethods.GetCasCadeLengthUnit_s()) * 0.001
    extent = [
        float(maximum[index] - minimum[index]) * cascade_meters_per_unit
        for index in range(3)
    ]
    if (
        not math.isfinite(cascade_meters_per_unit)
        or cascade_meters_per_unit <= 0.0
        or not all(math.isfinite(value) and value >= 0.0 for value in extent)
        or not any(value > 0.0 for value in extent)
    ):
        raise ValueError("Native CAD bounds are invalid for coordinate verification")
    return extent, cascade_meters_per_unit


def _converted_brep_coordinate_evidence(
    *,
    source_path: Path,
    stage: Any,
    declared_meters_per_unit: float,
    Usd: Any,
    UsdGeom: Any,
) -> dict[str, Any]:
    source_extent_m, cascade_meters_per_unit = _brep_extent_in_meters(source_path)
    stage_extent_units = _stage_extent_in_stage_units(stage, Usd, UsdGeom)
    stage_extent_if_declared_m = [
        value * declared_meters_per_unit for value in stage_extent_units
    ]
    ordered_source = sorted(source_extent_m)
    ordered_stage = sorted(stage_extent_if_declared_m)
    maximum_extent_m = max(*ordered_source, *ordered_stage)
    absolute_tolerance_m = max(
        _COORDINATE_EXTENT_ABS_TOLERANCE_M,
        maximum_extent_m * _COORDINATE_EXTENT_REL_TOLERANCE,
    )
    matched = all(
        math.isclose(
            source_value,
            stage_value,
            rel_tol=_COORDINATE_EXTENT_REL_TOLERANCE,
            abs_tol=absolute_tolerance_m,
        )
        for source_value, stage_value in zip(ordered_source, ordered_stage, strict=True)
    )
    return {
        "status": "matched" if matched else "mismatch",
        "method": "native_brep_to_converted_usd_bounds",
        "source_extent_m": source_extent_m,
        "stage_extent_units": stage_extent_units,
        "stage_extent_if_declared_m": stage_extent_if_declared_m,
        "cascade_meters_per_unit": cascade_meters_per_unit,
        "relative_tolerance": _COORDINATE_EXTENT_REL_TOLERANCE,
        "absolute_tolerance_m": absolute_tolerance_m,
    }


def _apply_declared_coordinate_system(
    prepared: PreparedGeometrySource,
    *,
    coordinate_system: GeometryCoordinateSystem,
    source_path: Path,
    source_suffix: str,
) -> None:
    """Apply or verify provider-declared metrics on the prepared USD stage."""

    if prepared.prepared_usd_path is None:
        raise ValueError("Prepared geometry has no USD artifact")
    from pxr import Usd, UsdGeom

    usd_path = Path(prepared.prepared_usd_path).resolve(strict=True)
    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError("Prepared geometry is not a readable USD stage")
    authored_up_axis = stage.HasAuthoredMetadata("upAxis")
    authored_meters = stage.HasAuthoredMetadata("metersPerUnit")
    observed_up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    observed_meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    unitless_source = source_suffix in _UNITLESS_MESH_SUFFIXES
    direct_usd = source_suffix in USD_SUFFIXES
    meters_mismatch = authored_meters and not math.isclose(
        observed_meters,
        coordinate_system.meters_per_unit,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    coordinate_evidence: dict[str, Any] | None = None
    correct_converted_meters = False

    if direct_usd and (not authored_up_axis or not authored_meters):
        raise ValueError(
            "Provider USD must author upAxis and metersPerUnit to match its source bundle"
        )
    if not unitless_source:
        if authored_up_axis and observed_up_axis != coordinate_system.up_axis:
            raise ValueError("Provider stage upAxis contradicts its source bundle")
        if meters_mismatch:
            if direct_usd or source_suffix not in BREP_SUFFIXES:
                raise ValueError(
                    "Provider stage metersPerUnit contradicts its source bundle"
                )
            coordinate_evidence = _converted_brep_coordinate_evidence(
                source_path=source_path,
                stage=stage,
                declared_meters_per_unit=coordinate_system.meters_per_unit,
                Usd=Usd,
                UsdGeom=UsdGeom,
            )
            if coordinate_evidence["status"] != "matched":
                raise ValueError(
                    "Converted CAD stage metersPerUnit contradicts its source bundle, "
                    "and native bounds do not prove a metadata-only converter error"
                )
            correct_converted_meters = True

    application = {
        "source_suffix": source_suffix,
        "unitless_source": unitless_source,
        "declared": coordinate_system.model_dump(mode="json"),
        "observed_before": {
            "up_axis": observed_up_axis,
            "meters_per_unit": observed_meters,
            "up_axis_authored": authored_up_axis,
            "meters_per_unit_authored": authored_meters,
        },
        "native_bounds_evidence": coordinate_evidence,
        "converter_metadata_corrected": correct_converted_meters,
    }
    if direct_usd:
        custom_data = dict(stage.GetRootLayer().customLayerData)
        retained = custom_data.get("geometrySourceCoordinateSystem")
        if retained is not None:
            if not isinstance(retained, Mapping):
                raise ValueError(
                    "Provider USD geometrySourceCoordinateSystem must be a dictionary"
                )
            try:
                retained_coordinates = GeometryCoordinateSystem.model_validate(
                    dict(retained)
                )
            except ValidationError as exc:
                raise ValueError(
                    "Provider USD geometrySourceCoordinateSystem is invalid"
                ) from exc
            if retained_coordinates != coordinate_system:
                raise ValueError(
                    "Provider USD geometrySourceCoordinateSystem contradicts its "
                    "source bundle"
                )
            application["retained_source_coordinates"] = "matched"
        else:
            application["retained_source_coordinates"] = "missing"
        prepared.metadata["coordinate_system_application"] = application
        return

    axis_token = {
        "X": UsdGeom.Tokens.x,
        "Y": UsdGeom.Tokens.y,
        "Z": UsdGeom.Tokens.z,
    }[coordinate_system.up_axis]
    if unitless_source or not authored_up_axis:
        UsdGeom.SetStageUpAxis(stage, axis_token)
    if unitless_source or not authored_meters or correct_converted_meters:
        UsdGeom.SetStageMetersPerUnit(stage, coordinate_system.meters_per_unit)
    custom_data = dict(stage.GetRootLayer().customLayerData)
    custom_data["geometrySourceCoordinateSystem"] = coordinate_system.model_dump(
        mode="json"
    )
    stage.GetRootLayer().customLayerData = custom_data
    if not stage.GetRootLayer().Save():
        raise RuntimeError("Could not persist provider-declared source coordinates")
    if correct_converted_meters:
        prepared.warnings.append(
            "Converted CAD unit metadata was corrected after native bounds verification."
        )
    prepared.metadata["coordinate_system_application"] = application


def _source_authoring_mode_error(
    source: Path,
    mode: SourceAuthoringMode,
    *,
    existing_usd: bool,
) -> str | None:
    if mode == "auto":
        return None
    suffix = source.suffix.lower()
    compatible = {
        "lossless_gltf": suffix in {".gltf", ".glb"},
        "direct_preserve": existing_usd,
        "parametric_recovery": suffix in RECOVERABLE_MESH_SUFFIXES,
        "opaque_import": suffix in OPAQUE_IMPORT_SUFFIXES | THREEMF_SUFFIXES,
        "shared_conversion": converter_reference_for_source_extension(source)
        is not None,
    }
    if compatible.get(mode, False):
        return None
    return f"Authoring mode {mode!r} is incompatible with source {source.name!r}."


def _prepare_direct_usd(source: Path) -> PreparedGeometrySource:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Geometry USD artifact not found: {source}")
    if not is_existing_usd(source):
        raise RuntimeError(f"Expected a readable USD artifact, got {source.name!r}.")
    return PreparedGeometrySource(
        fidelity_tier="direct_usd_preserved",
        source_authoring_mode="direct_preserve",
        source_path=str(source),
        prepared_usd_path=str(source),
        original_input_path=str(source),
        metadata={
            "source_format": source.suffix.lower().lstrip("."),
            "contract": "Existing USD is preserved as the source-of-record.",
        },
    )


def _prepare_shared_conversion(
    source: Path,
    prep_dir: Path,
    *,
    install_missing: bool,
    timeout_s: float,
) -> PreparedGeometrySource:
    """Convert an external source through the shared conversion workflow."""

    conversion_dir = prep_dir / "conversion"
    output_usd = conversion_dir / f"{source.stem}.converted.usda"
    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(
            source_asset_path=source,
            output_dir=conversion_dir,
            output_usd_path=output_usd,
            install_missing=install_missing,
            converter_timeout_s=timeout_s,
        )
    )
    metadata = {
        "source_format": result.source_format,
        "contract": (
            "The shared conversion workflow produced USD, but the external "
            "source remains the source-of-record and native CAD editability is "
            "not claimed."
        ),
        "conversion": {
            "selected_converter": result.selected_converter,
            "request_path": result.request_path,
            "probe_path": result.converter_probe_path,
            "report_path": result.conversion_report_path,
            "validation_report_path": result.validation_report_path,
            "manifest_path": result.manifest_path,
            "status": result.validation_status,
        },
    }
    if not result.success or not result.output_usd_path:
        errors = [result.error or "Shared conversion did not produce usable USD."]
        return PreparedGeometrySource(
            fidelity_tier="shared_conversion_noneditable",
            source_authoring_mode="shared_conversion",
            status="unsupported",
            source_path=str(source),
            original_input_path=str(source),
            lossy=True,
            errors=errors,
            metadata=metadata,
        )
    return PreparedGeometrySource(
        fidelity_tier="shared_conversion_noneditable",
        source_authoring_mode="shared_conversion",
        source_path=str(source),
        prepared_usd_path=str(Path(result.output_usd_path).resolve()),
        original_input_path=str(source),
        lossy=True,
        warnings=[
            "External source was converted to USD without claiming native CAD editability."
        ],
        metadata=metadata,
    )


def _prepare_opaque_import(
    source: Path,
    prep_dir: Path,
) -> PreparedGeometrySource:
    prepared = _prepare_shared_conversion(
        source,
        prep_dir,
        install_missing=False,
        timeout_s=120.0,
    )
    prepared.source_authoring_mode = "opaque_import"
    prepared.metadata["opaque_import"] = True
    return prepared


def _prepare_parametric_recovery(
    source: Path,
    prep_dir: Path,
    *,
    allow_lossy_recovery: bool,
) -> PreparedGeometrySource:
    if not allow_lossy_recovery:
        return _unsupported(
            source,
            "parametric_recovery",
            "Parametric recovery would be lossy; pass allow_lossy_recovery=True to opt in.",
            error_code="lossy_recovery_requires_opt_in",
            lossy=True,
        )
    return _unsupported(
        source,
        "parametric_recovery",
        (
            "Mesh-to-parametric recovery requires an explicitly configured authoring "
            "provider that returns an immutable source bundle."
        ),
        error_code="authoring_provider_unavailable",
        lossy=True,
    )


def _prepare_generic_3mf(source: Path, prep_dir: Path) -> PreparedGeometrySource:
    converted = _convert_3mf_to_usda(
        source,
        prep_dir / f"{source.stem}.mesh.usda",
        asset_name=_sanitize_usd_identifier(source.stem),
    )
    material_hints = _mesh_material_hints(converted["objects"])
    return PreparedGeometrySource(
        fidelity_tier="mesh_3mf_import",
        source_authoring_mode="opaque_import",
        source_path=str(source.resolve()),
        prepared_usd_path=str(Path(converted["usd_path"]).resolve()),
        original_input_path=str(source.resolve()),
        metadata={
            "source_format": "3mf",
            "contract": (
                "3MF geometry and color groups are imported through the built-in "
                "deterministic mesh bridge; native parametric editability remains upstream."
            ),
            "intermediate_3mf_path": str(source.resolve()),
            "mesh_cleanup": _mesh_cleanup_summary(converted["objects"]),
            "semantic_parts": _mesh_semantic_parts(converted["objects"]),
            "material_hints": material_hints,
            "parameter_ranges": {},
            "variant_parameters": [],
            "verification_assertions": [],
            "articulation_hints": {"joints": []},
        },
    )


def _convert_3mf_to_usda(
    threemf_path: Path,
    usd_path: Path,
    *,
    asset_name: str,
) -> dict[str, Any]:
    objects = _parse_3mf_objects(threemf_path)
    if not objects:
        raise RuntimeError(f"3MF contains no mesh objects: {threemf_path}")
    _write_usda_meshes(objects, usd_path, asset_name=asset_name)
    return {
        "usd_path": str(usd_path.resolve()),
        "objects": objects,
        "object_count": len(objects),
        "vertex_count": sum(len(item["points"]) for item in objects),
        "triangle_count": sum(len(item["indices"]) // 3 for item in objects),
    }


def _parse_3mf_objects(threemf_path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(threemf_path) as archive:
        model_name = next(
            (name for name in archive.namelist() if name.endswith(".model")),
            None,
        )
        if model_name is None:
            raise RuntimeError(f"3MF archive has no .model payload: {threemf_path}")
        model_info = archive.getinfo(model_name)
        if model_info.file_size > THREEMF_MODEL_XML_MAX_BYTES:
            raise RuntimeError(
                "3MF model XML payload exceeds "
                f"{THREEMF_MODEL_XML_MAX_BYTES} bytes: {model_name}"
            )
        with archive.open(model_info) as stream:
            model_xml = stream.read(THREEMF_MODEL_XML_MAX_BYTES + 1)
        if len(model_xml) > THREEMF_MODEL_XML_MAX_BYTES:
            raise RuntimeError(
                "3MF model XML payload exceeds "
                f"{THREEMF_MODEL_XML_MAX_BYTES} bytes: {model_name}"
            )
        try:
            root = DefusedET.fromstring(model_xml, forbid_dtd=True)
        except (DefusedXmlException, DefusedET.ParseError) as exc:
            raise RuntimeError(
                f"3MF model XML is unsafe or invalid: {model_name}: {exc}"
            ) from exc
    unit_to_mm = _3mf_unit_to_millimeters(root.get("unit"))

    colorgroups: dict[str, list[str]] = {}
    for node in root.iter():
        if not node.tag.endswith("colorgroup"):
            continue
        group_id = node.get("id")
        if not group_id:
            continue
        colorgroups[group_id] = [
            str(child.get("color") or "#c8c8c8")
            for child in node
            if child.tag.endswith("color")
        ]

    resources = root.find(f"{_THREEMF_CORE_NS}resources")
    if resources is None:
        return []
    objects_by_id: dict[str, dict[str, Any]] = {}
    for obj in resources.findall(f"{_THREEMF_CORE_NS}object"):
        object_id = obj.get("id")
        if object_id is None:
            continue
        object_name = obj.get("name") or f"object_{object_id}"
        mesh = obj.find(f"{_THREEMF_CORE_NS}mesh")
        if mesh is not None:
            vertices = mesh.find(f"{_THREEMF_CORE_NS}vertices")
            triangles = mesh.find(f"{_THREEMF_CORE_NS}triangles")
            if vertices is None or triangles is None:
                continue
            points = [
                (
                    float(vertex.get("x") or 0.0) * unit_to_mm,
                    float(vertex.get("y") or 0.0) * unit_to_mm,
                    float(vertex.get("z") or 0.0) * unit_to_mm,
                )
                for vertex in vertices.findall(f"{_THREEMF_CORE_NS}vertex")
            ]
            object_color = _resolve_3mf_color(obj, colorgroups)
            triangle_groups: dict[str, list[int]] = {}
            for triangle in triangles.findall(f"{_THREEMF_CORE_NS}triangle"):
                color = _resolve_3mf_color(
                    triangle,
                    colorgroups,
                    default=object_color,
                    fallback_pid=obj.get("pid"),
                )
                triangle_groups.setdefault(color, []).extend(
                    [
                        int(triangle.get("v1") or 0),
                        int(triangle.get("v2") or 0),
                        int(triangle.get("v3") or 0),
                    ]
                )
            mesh_objects: list[dict[str, Any]] = []
            multi_material = len(triangle_groups) > 1
            for color, indices in triangle_groups.items():
                material_suffix = (
                    f"_{color.lstrip('#').lower()}" if multi_material else ""
                )
                mesh_object = _clean_3mf_mesh_object(
                    {
                        "id": object_id,
                        "name": f"{object_name}{material_suffix}",
                        "color": color,
                        "points": points,
                        "indices": indices,
                    }
                )
                mesh_objects.append(
                    {
                        "type": "mesh",
                        "id": object_id,
                        "name": mesh_object["name"],
                        "color": color,
                        "points": mesh_object["points"],
                        "indices": mesh_object["indices"],
                        "cleanup": mesh_object["cleanup"],
                    }
                )
            if not mesh_objects:
                continue
            if len(mesh_objects) == 1:
                objects_by_id[object_id] = mesh_objects[0]
            else:
                objects_by_id[object_id] = {
                    "type": "mesh_group",
                    "id": object_id,
                    "name": object_name,
                    "meshes": mesh_objects,
                }
            continue

        components = obj.find(f"{_THREEMF_CORE_NS}components")
        if components is None:
            continue
        component_items: list[dict[str, Any]] = []
        for component in components.findall(f"{_THREEMF_CORE_NS}component"):
            component_object_id = component.get("objectid")
            if component_object_id is None:
                continue
            component_items.append(
                {
                    "objectid": component_object_id,
                    "transform": _parse_3mf_transform(
                        component.get("transform"),
                        unit_to_mm=unit_to_mm,
                    ),
                }
            )
        objects_by_id[object_id] = {
            "type": "components",
            "id": object_id,
            "name": object_name,
            "components": component_items,
        }

    build = root.find(f"{_THREEMF_CORE_NS}build")
    if build is None:
        ordered: list[dict[str, Any]] = []
        for object_id in objects_by_id:
            ordered.extend(
                _instantiate_3mf_object(
                    object_id,
                    objects_by_id,
                    _identity_3mf_transform(),
                    name_prefix=[],
                    stack=(),
                )
            )
        return ordered
    ordered = []
    for index, item in enumerate(build.findall(f"{_THREEMF_CORE_NS}item"), start=1):
        object_id = item.get("objectid")
        if object_id not in objects_by_id:
            continue
        instances = _instantiate_3mf_object(
            object_id,
            objects_by_id,
            _parse_3mf_transform(item.get("transform"), unit_to_mm=unit_to_mm),
            name_prefix=[],
            stack=(),
        )
        if len(instances) > 1:
            for instance in instances:
                instance["name"] = f"build_{index}_{instance['name']}"
        ordered.extend(instances)
    if ordered:
        return ordered
    fallback: list[dict[str, Any]] = []
    for object_id in objects_by_id:
        fallback.extend(
            _instantiate_3mf_object(
                object_id,
                objects_by_id,
                _identity_3mf_transform(),
                name_prefix=[],
                stack=(),
            )
        )
    return fallback


def _3mf_unit_to_millimeters(unit: str | None) -> float:
    normalized = (unit or "millimeter").strip().lower()
    return {
        "micron": 0.001,
        "millimeter": 1.0,
        "centimeter": 10.0,
        "inch": 25.4,
        "foot": 304.8,
        "meter": 1000.0,
    }.get(normalized, 1.0)


def _resolve_3mf_color(
    node: Any,
    colorgroups: dict[str, list[str]],
    *,
    default: str = "#c8c8c8",
    fallback_pid: str | None = None,
) -> str:
    pid = node.get("pid") or fallback_pid
    index_text = node.get("pindex")
    if index_text is None:
        vertex_indices = [node.get("p1"), node.get("p2"), node.get("p3")]
        explicit_indices = [item for item in vertex_indices if item is not None]
        if explicit_indices:
            index_text = explicit_indices[0]
    if pid in colorgroups and index_text is not None:
        try:
            return colorgroups[pid][int(index_text)]
        except (IndexError, ValueError):
            return default
    return default


Matrix3x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def _identity_3mf_transform() -> Matrix3x4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
    )


def _parse_3mf_transform(value: str | None, *, unit_to_mm: float = 1.0) -> Matrix3x4:
    if not value:
        return _identity_3mf_transform()
    values = [float(item) for item in value.split()]
    if len(values) != 12 or not all(math.isfinite(item) for item in values):
        raise RuntimeError("3MF transform must contain twelve finite values.")
    return (
        (values[0], values[3], values[6], values[9] * unit_to_mm),
        (values[1], values[4], values[7], values[10] * unit_to_mm),
        (values[2], values[5], values[8], values[11] * unit_to_mm),
    )


def _compose_3mf_transform(parent: Matrix3x4, child: Matrix3x4) -> Matrix3x4:
    return (
        (
            parent[0][0] * child[0][0]
            + parent[0][1] * child[1][0]
            + parent[0][2] * child[2][0],
            parent[0][0] * child[0][1]
            + parent[0][1] * child[1][1]
            + parent[0][2] * child[2][1],
            parent[0][0] * child[0][2]
            + parent[0][1] * child[1][2]
            + parent[0][2] * child[2][2],
            parent[0][0] * child[0][3]
            + parent[0][1] * child[1][3]
            + parent[0][2] * child[2][3]
            + parent[0][3],
        ),
        (
            parent[1][0] * child[0][0]
            + parent[1][1] * child[1][0]
            + parent[1][2] * child[2][0],
            parent[1][0] * child[0][1]
            + parent[1][1] * child[1][1]
            + parent[1][2] * child[2][1],
            parent[1][0] * child[0][2]
            + parent[1][1] * child[1][2]
            + parent[1][2] * child[2][2],
            parent[1][0] * child[0][3]
            + parent[1][1] * child[1][3]
            + parent[1][2] * child[2][3]
            + parent[1][3],
        ),
        (
            parent[2][0] * child[0][0]
            + parent[2][1] * child[1][0]
            + parent[2][2] * child[2][0],
            parent[2][0] * child[0][1]
            + parent[2][1] * child[1][1]
            + parent[2][2] * child[2][1],
            parent[2][0] * child[0][2]
            + parent[2][1] * child[1][2]
            + parent[2][2] * child[2][2],
            parent[2][0] * child[0][3]
            + parent[2][1] * child[1][3]
            + parent[2][2] * child[2][3]
            + parent[2][3],
        ),
    )


def _transform_3mf_point(
    transform: Matrix3x4,
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        transform[0][0] * point[0]
        + transform[0][1] * point[1]
        + transform[0][2] * point[2]
        + transform[0][3],
        transform[1][0] * point[0]
        + transform[1][1] * point[1]
        + transform[1][2] * point[2]
        + transform[1][3],
        transform[2][0] * point[0]
        + transform[2][1] * point[1]
        + transform[2][2] * point[2]
        + transform[2][3],
    )


def _instantiate_3mf_object(
    object_id: str,
    objects_by_id: dict[str, dict[str, Any]],
    transform: Matrix3x4,
    *,
    name_prefix: list[str],
    stack: tuple[str, ...],
) -> list[dict[str, Any]]:
    if object_id in stack:
        chain = " -> ".join((*stack, object_id))
        raise RuntimeError(f"3MF component cycle detected: {chain}")
    record = objects_by_id.get(object_id)
    if record is None:
        return []
    if record.get("type") in {"mesh", "mesh_group"}:
        mesh_records = (
            [record]
            if record.get("type") == "mesh"
            else list(cast(list[dict[str, Any]], record.get("meshes") or []))
        )
        instances: list[dict[str, Any]] = []
        for mesh_record in mesh_records:
            name_parts = [*name_prefix, str(mesh_record["name"])]
            name = "_".join(part for part in name_parts if part)
            instances.append(
                {
                    "id": mesh_record["id"],
                    "name": name,
                    "color": mesh_record["color"],
                    "points": [
                        _transform_3mf_point(transform, point)
                        for point in cast(
                            list[tuple[float, float, float]], mesh_record["points"]
                        )
                    ],
                    "indices": list(cast(list[int], mesh_record["indices"])),
                    "cleanup": dict(mesh_record.get("cleanup") or {}),
                }
            )
        return instances
    instances: list[dict[str, Any]] = []
    child_prefix = [*name_prefix, str(record["name"])]
    for component in cast(list[dict[str, Any]], record.get("components") or []):
        instances.extend(
            _instantiate_3mf_object(
                str(component["objectid"]),
                objects_by_id,
                _compose_3mf_transform(
                    transform,
                    cast(Matrix3x4, component["transform"]),
                ),
                name_prefix=child_prefix,
                stack=(*stack, object_id),
            )
        )
    return instances


def _clean_3mf_mesh_object(obj: dict[str, Any]) -> dict[str, Any]:
    """Conservatively normalize 3MF triangles before USD handoff."""

    points = cast(list[tuple[float, float, float]], obj["points"])
    indices = cast(list[int], obj["indices"])
    weld_tolerance_mm = 1e-9
    point_map: dict[tuple[int, int, int], int] = {}
    compact_points: list[tuple[float, float, float]] = []
    remap: list[int] = []
    duplicate_vertices = 0
    for point in points:
        point_key = tuple(
            int(round(float(coordinate) / weld_tolerance_mm)) for coordinate in point
        )
        existing = point_map.get(point_key)
        if existing is not None:
            duplicate_vertices += 1
            remap.append(existing)
            continue
        point_map[point_key] = len(compact_points)
        remap.append(point_map[point_key])
        compact_points.append(point)

    clean_indices: list[int] = []
    seen_faces: set[tuple[int, int, int]] = set()
    invalid_faces = 0
    degenerate_faces = 0
    duplicate_faces = 0
    for offset in range(0, len(indices), 3):
        tri = indices[offset : offset + 3]
        if len(tri) != 3 or any(index < 0 or index >= len(remap) for index in tri):
            invalid_faces += 1
            continue
        remapped = (remap[tri[0]], remap[tri[1]], remap[tri[2]])
        if (
            len(set(remapped)) != 3
            or _triangle_area2(
                compact_points[remapped[0]],
                compact_points[remapped[1]],
                compact_points[remapped[2]],
            )
            <= 1e-18
        ):
            degenerate_faces += 1
            continue
        face_key = tuple(sorted(remapped))
        if face_key in seen_faces:
            duplicate_faces += 1
            continue
        seen_faces.add(face_key)
        clean_indices.extend(remapped)

    used_indices = sorted(set(clean_indices))
    used_remap = {old: new for new, old in enumerate(used_indices)}
    final_points = [compact_points[index] for index in used_indices]
    final_indices = [used_remap[index] for index in clean_indices]
    obj["points"] = final_points
    obj["indices"] = final_indices
    obj["cleanup"] = {
        "input_vertex_count": len(points),
        "input_triangle_count": len(indices) // 3,
        "duplicate_vertices_removed": duplicate_vertices,
        "invalid_faces_removed": invalid_faces,
        "degenerate_faces_removed": degenerate_faces,
        "duplicate_faces_removed": duplicate_faces,
        "unused_vertices_removed": len(compact_points) - len(used_indices),
        "output_vertex_count": len(final_points),
        "output_triangle_count": len(final_indices) // 3,
    }
    return obj


def _triangle_area2(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    ux, uy, uz = (b[index] - a[index] for index in range(3))
    vx, vy, vz = (c[index] - a[index] for index in range(3))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return nx * nx + ny * ny + nz * nz


def _write_usda_meshes(
    objects: list[dict[str, Any]],
    usd_path: Path,
    *,
    asset_name: str,
) -> None:
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    root_name = _sanitize_usd_identifier(asset_name or usd_path.stem)
    used: set[str] = set()
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{root_name}"',
        "    metersPerUnit = 0.001",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{root_name}"',
        "{",
        '    def Scope "Looks"',
        "    {",
    ]
    mesh_names: list[tuple[str, dict[str, Any]]] = []
    for obj in objects:
        mesh_name = _dedupe_name(_sanitize_usd_identifier(str(obj["name"])), used)
        mesh_names.append((mesh_name, obj))
        material_name = f"mat_{mesh_name}"
        rgb = _hex_to_linear_rgb(str(obj.get("color") or "#c8c8c8"))
        metallic = 1.0 if _mesh_name_is_metallic(str(obj["name"])) else 0.0
        roughness = 0.35 if metallic else 0.55
        lines.extend(
            [
                f'        def Material "{material_name}"',
                "        {",
                f"            token outputs:surface.connect = </{root_name}/Looks/{material_name}/PreviewSurface.outputs:surface>",
                '            def Shader "PreviewSurface"',
                "            {",
                '                uniform token info:id = "UsdPreviewSurface"',
                f"                color3f inputs:diffuseColor = ({rgb[0]:.6f}, {rgb[1]:.6f}, {rgb[2]:.6f})",
                f"                float inputs:metallic = {metallic:.1f}",
                f"                float inputs:roughness = {roughness:.2f}",
                "                token outputs:surface",
                "            }",
                "        }",
            ]
        )
    lines.append("    }")
    for mesh_name, obj in mesh_names:
        points = cast(list[tuple[float, float, float]], obj["points"])
        indices = cast(list[int], obj["indices"])
        world_extent = _points_extent(points)
        local_origin = tuple(
            (world_extent[0][axis] + world_extent[1][axis]) * 0.5 for axis in range(3)
        )
        local_points = [
            tuple(point[axis] - local_origin[axis] for axis in range(3))
            for point in points
        ]
        serialized_mesh = _clean_3mf_mesh_object(
            {
                "points": [
                    tuple(_float32(value) for value in point) for point in local_points
                ],
                "indices": indices,
            }
        )
        local_points = cast(list[tuple[float, float, float]], serialized_mesh["points"])
        indices = cast(list[int], serialized_mesh["indices"])
        obj["points"] = [
            tuple(point[axis] + local_origin[axis] for axis in range(3))
            for point in local_points
        ]
        obj["indices"] = indices
        obj["cleanup"] = _merge_mesh_cleanup(
            cast(dict[str, Any], obj.get("cleanup") or {}),
            cast(dict[str, Any], serialized_mesh["cleanup"]),
        )
        counts = [3] * (len(indices) // 3)
        extent = _points_extent(local_points)
        rgb = _hex_to_linear_rgb(str(obj.get("color") or "#c8c8c8"))
        lines.extend(
            [
                f'    def Mesh "{mesh_name}" (',
                '        prepend apiSchemas = ["MaterialBindingAPI"]',
                "    )",
                "    {",
                f"        rel material:binding = </{root_name}/Looks/mat_{mesh_name}>",
                f"        double3 xformOp:translate = ({_fmt_double_vec(local_origin)})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                '        uniform token orientation = "rightHanded"',
                '        uniform token subdivisionScheme = "none"',
                *(
                    ['        uniform token purpose = "guide"']
                    if _mesh_name_is_guide(str(obj["name"]))
                    else []
                ),
                f"        float3[] extent = [({_fmt_vec(extent[0])}), ({_fmt_vec(extent[1])})]",
                f"        int[] faceVertexCounts = [{', '.join(str(item) for item in counts)}]",
                f"        int[] faceVertexIndices = [{', '.join(str(item) for item in indices)}]",
                f"        point3f[] points = [{', '.join(f'({_fmt_vec(point)})' for point in local_points)}]",
                f"        color3f[] primvars:displayColor = [({rgb[0]:.6f}, {rgb[1]:.6f}, {rgb[2]:.6f})] (",
                '            interpolation = "constant"',
                "        )",
                "    }",
            ]
        )
    lines.append("}")
    lines.append("")
    usd_path.write_text("\n".join(lines), encoding="utf-8")


def _points_extent(
    points: list[tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [point[0] for point in points] or [0.0]
    ys = [point[1] for point in points] or [0.0]
    zs = [point[2] for point in points] or [0.0]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _fmt_vec(point: tuple[float, float, float]) -> str:
    return ", ".join(format(value, ".9g") for value in point)


def _fmt_double_vec(point: tuple[float, float, float]) -> str:
    return ", ".join(format(value, ".17g") for value in point)


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _merge_mesh_cleanup(
    initial: dict[str, Any], serialized: dict[str, Any]
) -> dict[str, Any]:
    removal_keys = (
        "duplicate_vertices_removed",
        "invalid_faces_removed",
        "degenerate_faces_removed",
        "duplicate_faces_removed",
        "unused_vertices_removed",
    )
    return {
        "input_vertex_count": int(
            initial.get("input_vertex_count")
            or serialized.get("input_vertex_count")
            or 0
        ),
        "input_triangle_count": int(
            initial.get("input_triangle_count")
            or serialized.get("input_triangle_count")
            or 0
        ),
        **{
            key: int(initial.get(key) or 0) + int(serialized.get(key) or 0)
            for key in removal_keys
        },
        "output_vertex_count": int(serialized.get("output_vertex_count") or 0),
        "output_triangle_count": int(serialized.get("output_triangle_count") or 0),
    }


def _hex_to_linear_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.strip().lstrip("#")
    if len(value) < 6:
        value = "c8c8c8"
    raw = tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
    return tuple(_srgb_channel_to_linear(channel / 255.0) for channel in raw)


def _srgb_channel_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _sanitize_usd_identifier(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "Asset"
    if sanitized[0].isdigit():
        sanitized = f"p_{sanitized}"
    return sanitized


def _dedupe_name(name: str, used: set[str]) -> str:
    base = name
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _mesh_semantic_parts(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(obj["name"]),
            "role": _semantic_role_from_name(str(obj["name"])),
            "source_object_id": str(obj.get("id") or ""),
            "material": f"imported_color_{str(obj.get('color') or '#c8c8c8').lstrip('#').lower()}",
        }
        for obj in objects
    ]


def _mesh_material_hints(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: dict[str, dict[str, Any]] = {}
    for obj in objects:
        color = str(obj.get("color") or "#c8c8c8").lower()
        material_name = f"imported_color_{color.lstrip('#')}"
        hints.setdefault(
            material_name,
            {
                "name": material_name,
                "color": color,
                "metallic": _mesh_name_is_metallic(str(obj["name"])),
                "parts": [],
            },
        )
        hints[material_name]["parts"].append(str(obj["name"]))
    return list(hints.values())


def _mesh_cleanup_summary(objects: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {
        "object_count": len(objects),
        "input_vertex_count": 0,
        "input_triangle_count": 0,
        "duplicate_vertices_removed": 0,
        "invalid_faces_removed": 0,
        "degenerate_faces_removed": 0,
        "duplicate_faces_removed": 0,
        "unused_vertices_removed": 0,
        "output_vertex_count": 0,
        "output_triangle_count": 0,
    }
    for obj in objects:
        cleanup = cast(dict[str, Any], obj.get("cleanup") or {})
        for key in totals:
            if key == "object_count":
                continue
            totals[key] += int(cleanup.get(key) or 0)
    return totals


def _semantic_role_from_name(name: str) -> str:
    role = re.sub(r"[^A-Za-z0-9]+", "_", name.lower()).strip("_")
    return role or "imported_part"


def _mesh_name_is_metallic(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "metal",
            "prong",
            "pin",
            "contact",
            "blade",
            "shield",
            "gold",
            "copper",
            "brass",
            "spring",
        )
    )


def _mesh_name_is_guide(name: str) -> bool:
    lowered = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return any(
        token in lowered
        for token in (
            "bench_mat",
            "service_mat",
            "table_contact_pad",
            "task_label",
            "hub_label",
            "mat_edge",
        )
    ) or lowered.endswith("_label")


def _unsupported(
    source: Path,
    source_authoring_mode: SourceAuthoringMode,
    message: str,
    *,
    error_code: str,
    lossy: bool = False,
    confidence: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> PreparedGeometrySource:
    return PreparedGeometrySource(
        fidelity_tier="unsupported_requires_converter",
        source_authoring_mode=source_authoring_mode,
        status="unsupported",
        source_path=str(source),
        original_input_path=str(source),
        lossy=lossy,
        confidence=confidence,
        errors=[message],
        metadata={
            "source_format": source.suffix.lower().lstrip("."),
            "error_code": error_code,
            **(metadata or {}),
        },
    )
