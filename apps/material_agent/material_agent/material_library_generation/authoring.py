# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic authoring for one portable material package.

This module is intentionally orchestration-free.  Agentic skills and fixed tasks
may decide what a material should become, but both publish that decision through
the same request, graph-editing, packaging, and validation boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from material_agent.material_library_generation.schema import (
    GeneratedMaterial,
    MaterialRecipe,
    MaterialRecipeSemantics,
    TextureMapSet,
)
from material_agent.material_library_generation.source_graph import (
    MaterialGraphEdit,
    MaterialGraphEditError,
    MaterialGraphInfo,
    inspect_material_graph,
    write_edited_material_graphs,
)
from material_agent.material_library_generation.usd_authoring import (
    MaterialAuthoringError,
    require_material_authoring_prerequisites,
    write_material_library_usd,
)
from material_agent.material_library_generation.validation import (
    ValidationResult,
    validate_generated_material_library,
)
from material_agent.material_profiles import normalize_material_profile

MATERIAL_AUTHORING_SCHEMA_VERSION = "material-agent-author.v1"
MATERIAL_AUTHORING_MANIFEST_NAME = "material_authoring_manifest.json"
MATERIAL_LIBRARY_NAME = "material_library.usda"
MATERIAL_LIST_MANIFEST_NAME = "materials.yaml"
_SUPPORTED_USD_SUFFIXES = frozenset((".usd", ".usda", ".usdc", ".usdz"))
_SHA256_RE_LENGTH = 64


class MaterialAuthoringOperation(StrEnum):
    """Deterministic operations supported by the package authoring core."""

    CREATE = "create"
    MODIFY = "modify"


class MaterialRepresentationPolicy(StrEnum):
    """Representation behavior for a source-backed material operation."""

    PRESERVE = "preserve"


class MaterialPackageAuthoringError(RuntimeError):
    """Raised when a requested material package cannot be authored safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _canonical_prim_path(value: str, *, field_name: str) -> str:
    path = _non_empty(value, field_name=field_name)
    if not path.startswith("/") or path == "/" or path.endswith("/") or "//" in path:
        raise ValueError(f"{field_name} must be a canonical absolute USD prim path")
    return path


def _canonical_prim_paths(values: tuple[str, ...]) -> tuple[str, ...]:
    paths = tuple(
        _canonical_prim_path(value, field_name="target_prim_paths") for value in values
    )
    if len(paths) != len(set(paths)):
        raise ValueError("target_prim_paths must not contain duplicates")
    return tuple(sorted(paths))


def _validate_sha256(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_RE_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class SourceMaterialReference:
    """Exact source material graph used by a modify operation."""

    usd_path: Path
    material_prim_path: str
    usd_sha256: str | None = None

    def __post_init__(self) -> None:
        usd_path = Path(self.usd_path).expanduser().resolve()
        if not usd_path.is_absolute():
            raise ValueError("source material usd_path must be absolute")
        if usd_path.suffix.lower() not in _SUPPORTED_USD_SUFFIXES:
            raise ValueError("source material must be a USD, USDA, USDC, or USDZ file")
        if not usd_path.is_file():
            raise FileNotFoundError(f"source material USD does not exist: {usd_path}")
        digest = self.usd_sha256
        actual = _sha256(usd_path)
        if digest is not None:
            expected = _validate_sha256(digest, field_name="source.usd_sha256")
            if actual != expected:
                raise ValueError("source material USD digest does not match usd_sha256")
        object.__setattr__(self, "usd_path", usd_path)
        object.__setattr__(
            self,
            "material_prim_path",
            _canonical_prim_path(
                self.material_prim_path,
                field_name="source.material_prim_path",
            ),
        )
        object.__setattr__(self, "usd_sha256", actual)

    @property
    def effective_usd_sha256(self) -> str:
        """Return the source digest frozen when this reference was created."""

        assert self.usd_sha256 is not None
        return self.usd_sha256

    def to_dict(self) -> dict[str, str]:
        return {
            "usd_path": self.usd_path.as_posix(),
            "material_prim_path": self.material_prim_path,
            "usd_sha256": self.effective_usd_sha256,
        }


def _resolved_textures(textures: TextureMapSet | None) -> TextureMapSet | None:
    if textures is None:
        return None
    resolved = TextureMapSet(
        albedo=Path(textures.albedo).expanduser().resolve(),
        normal=Path(textures.normal).expanduser().resolve(),
        orm=Path(textures.orm).expanduser().resolve(),
    )
    for channel, path in (
        ("albedo", resolved.albedo),
        ("normal", resolved.normal),
        ("orm", resolved.orm),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{channel} texture does not exist: {path}")
    return resolved


@dataclass(frozen=True)
class MaterialAuthoringRequest:
    """One model-free material package authoring request."""

    operation: MaterialAuthoringOperation
    recipe: MaterialRecipe
    source: SourceMaterialReference | None = None
    provenance_source: SourceMaterialReference | None = None
    textures: TextureMapSet | None = None
    target_prim_paths: tuple[str, ...] = ()
    material_profile: str = "auto"
    recipe_semantics: MaterialRecipeSemantics = (
        MaterialRecipeSemantics.LITERAL_SHADER_VALUES
    )
    representation_policy: MaterialRepresentationPolicy = (
        MaterialRepresentationPolicy.PRESERVE
    )
    schema_version: str = MATERIAL_AUTHORING_SCHEMA_VERSION
    _texture_sha256: tuple[tuple[str, str], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        operation = MaterialAuthoringOperation(self.operation)
        policy = MaterialRepresentationPolicy(self.representation_policy)
        recipe_semantics = MaterialRecipeSemantics(self.recipe_semantics)
        self.recipe.validate()
        if operation is MaterialAuthoringOperation.CREATE and self.source is not None:
            raise ValueError("create authoring must not include a source material")
        if operation is MaterialAuthoringOperation.MODIFY and self.source is None:
            raise ValueError("modify authoring requires a source material")
        if (
            operation is MaterialAuthoringOperation.MODIFY
            and self.provenance_source is not None
        ):
            raise ValueError(
                "modify authoring records its provenance through source; "
                "provenance_source is create-only"
            )
        if (
            operation is MaterialAuthoringOperation.MODIFY
            and recipe_semantics is not MaterialRecipeSemantics.LITERAL_SHADER_VALUES
        ):
            raise ValueError("modify authoring requires literal shader values")
        profile = normalize_material_profile(self.material_profile)
        if profile == "display_color":
            raise ValueError("display_color is assignment-only and cannot be authored")
        if self.schema_version != MATERIAL_AUTHORING_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {MATERIAL_AUTHORING_SCHEMA_VERSION!r}"
            )
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "recipe_semantics", recipe_semantics)
        object.__setattr__(self, "representation_policy", policy)
        object.__setattr__(self, "material_profile", profile)
        textures = _resolved_textures(self.textures)
        object.__setattr__(self, "textures", textures)
        object.__setattr__(
            self,
            "_texture_sha256",
            tuple(
                (channel, _sha256(path)) for channel, path in _texture_items(textures)
            ),
        )
        object.__setattr__(
            self,
            "target_prim_paths",
            _canonical_prim_paths(tuple(self.target_prim_paths)),
        )

    def _identity_payload(self) -> dict[str, Any]:
        textures = self.textures
        texture_sha256 = dict(self._texture_sha256)
        return {
            "schema_version": self.schema_version,
            "operation": self.operation.value,
            "recipe": self.recipe.to_dict(),
            "source": self.source.to_dict() if self.source is not None else None,
            "provenance_source": (
                self.provenance_source.to_dict()
                if self.provenance_source is not None
                else None
            ),
            "textures": (
                {
                    "albedo": {
                        "path": textures.albedo.as_posix(),
                        "sha256": texture_sha256["albedo"],
                    },
                    "normal": {
                        "path": textures.normal.as_posix(),
                        "sha256": texture_sha256["normal"],
                    },
                    "orm": {
                        "path": textures.orm.as_posix(),
                        "sha256": texture_sha256["orm"],
                    },
                }
                if textures is not None
                else None
            ),
            "target_prim_paths": self.target_prim_paths,
            "material_profile": self.material_profile,
            "recipe_semantics": self.recipe_semantics.value,
            "representation_policy": self.representation_policy.value,
        }

    @property
    def request_id(self) -> str:
        payload = json.dumps(
            self._identity_payload(), sort_keys=True, separators=(",", ":")
        )
        return f"ma_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "request_id": self.request_id}


@dataclass(frozen=True)
class MaterialPackage:
    """One validated portable material package."""

    request_id: str
    operation: MaterialAuthoringOperation
    material_id: str
    material_prim_path: str
    material_usd_path: Path
    materials_manifest_path: Path
    authoring_manifest_path: Path
    material_list_entry: dict[str, Any] = field(compare=False, hash=False)
    textures: TextureMapSet | None = None
    material_profile: str = "auto"
    representation: str = "scalar_pbr"
    topology_sha256: str | None = None
    artifact_sha256: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )
    validation: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "operation": self.operation.value,
            "material_id": self.material_id,
            "material_prim_path": self.material_prim_path,
            "material_usd_path": self.material_usd_path.as_posix(),
            "materials_manifest_path": self.materials_manifest_path.as_posix(),
            "authoring_manifest_path": self.authoring_manifest_path.as_posix(),
            "material_list_entry": dict(self.material_list_entry),
            "textures": (
                {
                    "albedo": self.textures.albedo.as_posix(),
                    "normal": self.textures.normal.as_posix(),
                    "orm": self.textures.orm.as_posix(),
                }
                if self.textures is not None
                else None
            ),
            "material_profile": self.material_profile,
            "representation": self.representation,
            "topology_sha256": self.topology_sha256,
            "artifact_sha256": dict(self.artifact_sha256),
            "validation": dict(self.validation),
        }


def _copy_png(source: Path, destination: Path, *, channel: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            image.load()
            if image.mode in {"F", "I"} or image.mode.startswith("I;16"):
                raise MaterialPackageAuthoringError(
                    f"{channel} texture must use 8-bit channels"
                )
            target_mode = (
                "RGBA"
                if channel == "albedo"
                and ("A" in image.getbands() or "transparency" in image.info)
                else "RGB"
            )
            converted = image.convert(target_mode)
            if converted.width <= 0 or converted.height <= 0:
                raise MaterialPackageAuthoringError(
                    f"{channel} texture has invalid dimensions"
                )
            if source.resolve() == destination.resolve() and image.format == "PNG":
                return destination
            converted.save(destination, format="PNG")
    except (OSError, UnidentifiedImageError) as exc:
        raise MaterialPackageAuthoringError(
            f"{channel} texture is not a decodable image: {source}"
        ) from exc
    return destination


def _texture_items(
    textures: TextureMapSet | None,
) -> tuple[tuple[str, Path], ...]:
    if textures is None:
        return ()
    return (
        ("albedo", textures.albedo),
        ("normal", textures.normal),
        ("orm", textures.orm),
    )


def _verify_frozen_textures(request: MaterialAuthoringRequest) -> None:
    expected = dict(request._texture_sha256)
    for channel, path in _texture_items(request.textures):
        if _sha256(path) != expected[channel]:
            raise MaterialPackageAuthoringError(
                f"{channel} texture no longer matches the frozen digest"
            )


def _package_textures(
    textures: TextureMapSet | None,
    *,
    package_dir: Path,
    material_id: str,
) -> TextureMapSet | None:
    if textures is None:
        return None
    texture_dir = package_dir / "textures" / material_id
    return TextureMapSet(
        albedo=_copy_png(textures.albedo, texture_dir / "albedo.png", channel="albedo"),
        normal=_copy_png(textures.normal, texture_dir / "normal.png", channel="normal"),
        orm=_copy_png(textures.orm, texture_dir / "orm.png", channel="orm"),
    )


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-6)


def _validate_preserved_optics(
    source: MaterialGraphInfo,
    recipe: MaterialRecipe,
) -> None:
    requested = recipe.pbr_hints
    differences = [
        name
        for name, source_value, requested_value in (
            ("opacity", source.opacity, requested.opacity),
            ("transmission", source.transmission, requested.transmission),
            ("ior", source.ior, requested.ior),
        )
        if not _same_float(source_value, requested_value)
    ]
    if bool(source.thin_walled) != bool(requested.thin_walled):
        differences.append("thin_walled")
    if differences:
        raise MaterialPackageAuthoringError(
            "modify currently preserves optical controls; target recipe changes "
            + ", ".join(differences)
        )


def _validate_no_authoring_input_references(
    published: MaterialGraphInfo,
    staging_dir: Path,
) -> None:
    if published.textures is None:
        return
    staging_root = staging_dir.resolve()
    staged_channels = [
        channel
        for channel, path in (
            ("albedo", published.textures.albedo),
            ("normal", published.textures.normal),
            ("orm", published.textures.orm),
        )
        if path.resolve().is_relative_to(staging_root)
    ]
    if staged_channels:
        raise MaterialPackageAuthoringError(
            "published material references temporary authoring input texture(s): "
            + ", ".join(staged_channels)
        )


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")


def _write_materials_manifest(
    path: Path,
    library_path: Path,
    entry: dict[str, Any],
) -> Path:
    data = {
        "library_path": _relative_path(library_path, path.parent),
        "entries": [entry],
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return path


def _material_list_entry(
    request: MaterialAuthoringRequest,
    *,
    source_info: MaterialGraphInfo | None,
) -> dict[str, Any]:
    recipe = request.recipe
    entry: dict[str, Any] = {
        "name": recipe.name,
        "description": recipe.description,
        "binding": recipe.binding,
        "source": (
            "modified"
            if request.operation is MaterialAuthoringOperation.MODIFY
            else "generated"
        ),
        "generation_id": recipe.material_id,
        "authoring_request_id": request.request_id,
        "authoring_operation": request.operation.value,
        "authoring_manifest": MATERIAL_AUTHORING_MANIFEST_NAME,
    }
    if request.target_prim_paths:
        entry["target_prim_paths"] = list(request.target_prim_paths)
    if recipe.intended_parts:
        entry["intended_parts"] = [
            part.semantic_label for part in recipe.intended_parts
        ]
    if source_info is not None:
        entry["prototype_source"] = {
            "library_path": source_info.usd_path.as_posix(),
            "binding": source_info.material_prim_path,
            "material_profile": source_info.material_profile,
            "representation": source_info.representation,
            "topology_sha256": source_info.topology_sha256,
        }
    return entry


def _write_package_files(
    request: MaterialAuthoringRequest,
    package_dir: Path,
    *,
    material_list_entry: dict[str, Any] | None = None,
) -> MaterialPackage:
    _verify_frozen_textures(request)
    package_dir.mkdir(parents=True, exist_ok=True)
    library_path = package_dir / MATERIAL_LIBRARY_NAME
    materials_manifest_path = package_dir / MATERIAL_LIST_MANIFEST_NAME
    authoring_manifest_path = package_dir / MATERIAL_AUTHORING_MANIFEST_NAME
    source_info: MaterialGraphInfo | None = None

    if request.operation is MaterialAuthoringOperation.MODIFY:
        assert request.source is not None
        source_digest = request.source.effective_usd_sha256
        if _sha256(request.source.usd_path) != source_digest:
            raise MaterialPackageAuthoringError(
                "source material USD no longer matches the frozen digest"
            )
        source_info = inspect_material_graph(
            request.source.usd_path,
            request.source.material_prim_path,
        )
        if request.material_profile not in {"auto", source_info.material_profile}:
            raise MaterialPackageAuthoringError(
                "modify preserves the source material profile: "
                f"requested={request.material_profile}, "
                f"source={source_info.material_profile}"
            )
        _validate_preserved_optics(source_info, request.recipe)
        if source_info.representation == "scalar_pbr" and request.textures is not None:
            raise MaterialPackageAuthoringError(
                "representation_policy='preserve' forbids adding textures to a "
                "scalar source material"
            )
        if source_info.representation == "textured_pbr" and request.textures is None:
            raise MaterialPackageAuthoringError(
                "textured material modification requires complete replacement "
                "textures; connected scalar controls cannot satisfy the request"
            )
        authoring_inputs_dir = package_dir / ".authoring_inputs"
        packaged_inputs = _package_textures(
            request.textures,
            package_dir=authoring_inputs_dir,
            material_id=request.recipe.material_id,
        )
        (published_info,) = write_edited_material_graphs(
            library_path,
            (
                MaterialGraphEdit(
                    source_usd=request.source.usd_path,
                    source_material_prim_path=request.source.material_prim_path,
                    target_material_prim_path=request.recipe.binding,
                    base_color=request.recipe.base_color_hint,
                    roughness=request.recipe.pbr_hints.roughness,
                    metallic=request.recipe.pbr_hints.metallic,
                    textures=packaged_inputs,
                ),
            ),
        )
        _validate_no_authoring_input_references(published_info, authoring_inputs_dir)
        shutil.rmtree(authoring_inputs_dir, ignore_errors=True)
        if _sha256(request.source.usd_path) != source_digest:
            raise MaterialPackageAuthoringError(
                "source material USD changed while the package was authored"
            )
        if published_info.representation != source_info.representation:
            raise MaterialPackageAuthoringError(
                "published material changed source representation"
            )
        if published_info.topology_sha256 != source_info.topology_sha256:
            raise MaterialPackageAuthoringError(
                "published material changed source graph topology"
            )
    else:
        packaged_textures = _package_textures(
            request.textures,
            package_dir=package_dir,
            material_id=request.recipe.material_id,
        )
        require_material_authoring_prerequisites(request.material_profile)
        write_material_library_usd(
            library_path,
            (GeneratedMaterial(recipe=request.recipe, textures=packaged_textures),),
            material_profile=request.material_profile,
            recipe_semantics=request.recipe_semantics,
        )
        published_info = inspect_material_graph(
            library_path,
            request.recipe.binding,
            allow_textured_omnipbr=True,
        )

    _verify_frozen_textures(request)
    entry = (
        dict(material_list_entry)
        if material_list_entry is not None
        else _material_list_entry(request, source_info=source_info)
    )
    _write_materials_manifest(materials_manifest_path, library_path, entry)
    validation = validate_generated_material_library(materials_manifest_path)
    _require_portable_validation(validation)
    artifact_sha256: dict[str, Any] = {
        "material_usd_path": _sha256(library_path),
        "materials_manifest_path": _sha256(materials_manifest_path),
        "textures": (
            {
                channel: _sha256(path)
                for channel, path in _texture_items(published_info.textures)
            }
            if published_info.textures is not None
            else None
        ),
    }
    package = MaterialPackage(
        request_id=request.request_id,
        operation=request.operation,
        material_id=request.recipe.material_id,
        material_prim_path=request.recipe.binding,
        material_usd_path=library_path,
        materials_manifest_path=materials_manifest_path,
        authoring_manifest_path=authoring_manifest_path,
        material_list_entry=entry,
        textures=published_info.textures,
        material_profile=published_info.material_profile,
        representation=published_info.representation,
        topology_sha256=published_info.topology_sha256,
        artifact_sha256=artifact_sha256,
        validation={
            "status": "ok",
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            **validation.metadata,
            "cache_hit": False,
        },
    )
    manifest_data = {
        "schema_version": MATERIAL_AUTHORING_SCHEMA_VERSION,
        "request": request.to_dict(),
        "material_package": _relativize_package(package, package_dir),
    }
    with authoring_manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest_data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return package


def _relativize_package(package: MaterialPackage, package_dir: Path) -> dict[str, Any]:
    data = package.to_dict()
    for key in (
        "material_usd_path",
        "materials_manifest_path",
        "authoring_manifest_path",
    ):
        data[key] = _relative_path(Path(data[key]), package_dir)
    textures = data.get("textures")
    if isinstance(textures, dict):
        data["textures"] = {
            channel: _relative_path(Path(path), package_dir)
            for channel, path in textures.items()
        }
    validation = data.get("validation")
    if isinstance(validation, dict) and validation.get("library_path"):
        validation["library_path"] = _relative_path(
            Path(validation["library_path"]), package_dir
        )
    return data


def _resolve_package_path(value: str, package_dir: Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (package_dir / path).resolve()
    if not resolved.is_relative_to(package_dir.resolve()):
        raise ValueError("material package path escapes the package directory")
    return resolved


def _read_material_package(
    package_dir: Path,
) -> tuple[MaterialPackage, dict[str, Any]]:
    manifest_path = package_dir / MATERIAL_AUTHORING_MANIFEST_NAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("authoring manifest root must be a mapping")
    if data.get("schema_version") != MATERIAL_AUTHORING_SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    stored_request = data["request"]
    if not isinstance(stored_request, dict):
        raise ValueError("request must be a mapping")
    result = data["material_package"]
    if not isinstance(result, dict):
        raise ValueError("material_package must be a mapping")
    recorded_validation = result.get("validation")
    if (
        not isinstance(recorded_validation, dict)
        or recorded_validation.get("status") != "ok"
        or recorded_validation.get("errors") not in ([], ())
    ):
        raise ValueError("authoring manifest does not record successful validation")
    authored_manifest_path = _resolve_package_path(
        str(result["authoring_manifest_path"]), package_dir
    )
    if authored_manifest_path != manifest_path.resolve():
        raise ValueError("authoring_manifest_path does not reference its manifest")
    texture_data = result.get("textures")
    textures = (
        TextureMapSet(
            albedo=_resolve_package_path(str(texture_data["albedo"]), package_dir),
            normal=_resolve_package_path(str(texture_data["normal"]), package_dir),
            orm=_resolve_package_path(str(texture_data["orm"]), package_dir),
        )
        if isinstance(texture_data, dict)
        else None
    )
    package = MaterialPackage(
        request_id=str(result["request_id"]),
        operation=MaterialAuthoringOperation(str(result["operation"])),
        material_id=str(result["material_id"]),
        material_prim_path=str(result["material_prim_path"]),
        material_usd_path=_resolve_package_path(
            str(result["material_usd_path"]), package_dir
        ),
        materials_manifest_path=_resolve_package_path(
            str(result["materials_manifest_path"]), package_dir
        ),
        authoring_manifest_path=manifest_path.resolve(),
        material_list_entry=dict(result["material_list_entry"]),
        textures=textures,
        material_profile=str(result["material_profile"]),
        representation=str(result["representation"]),
        topology_sha256=result.get("topology_sha256"),
        artifact_sha256=dict(result.get("artifact_sha256") or {}),
        validation=dict(recorded_validation),
    )
    if stored_request.get("request_id") != package.request_id:
        raise ValueError("request_id mismatch inside authoring manifest")
    return package, stored_request


def _require_portable_validation(validation: ValidationResult) -> None:
    """Reject invalid or nonportable material-library validation results."""

    if not validation.ok:
        raise MaterialPackageAuthoringError(
            "material package validation failed: " + "; ".join(validation.errors)
        )
    nonportable_assets = [
        warning
        for warning in validation.warnings
        if warning.startswith("non-relative asset path")
    ]
    if nonportable_assets:
        raise MaterialPackageAuthoringError(
            "material package is not portable: " + "; ".join(nonportable_assets)
        )


def _validate_loaded_package(
    package: MaterialPackage,
    *,
    cache_hit: bool,
) -> MaterialPackage:
    validation = validate_generated_material_library(package.materials_manifest_path)
    _require_portable_validation(validation)
    materials_data = yaml.safe_load(
        package.materials_manifest_path.read_text(encoding="utf-8")
    )
    library_value = (
        materials_data.get("library_path") if isinstance(materials_data, dict) else None
    )
    if not isinstance(library_value, str) or not library_value.strip():
        raise MaterialPackageAuthoringError(
            "material package manifest requires library_path"
        )
    manifest_library = Path(library_value).expanduser()
    manifest_library = (
        manifest_library.resolve()
        if manifest_library.is_absolute()
        else (package.materials_manifest_path.parent / manifest_library).resolve()
    )
    package_root = package.authoring_manifest_path.parent.resolve()
    if (
        not manifest_library.is_relative_to(package_root)
        or manifest_library != package.material_usd_path
    ):
        raise MaterialPackageAuthoringError(
            "material package manifest library_path does not identify the returned "
            "material USD"
        )
    entries = (
        materials_data.get("entries") if isinstance(materials_data, dict) else None
    )
    if not isinstance(entries, list) or len(entries) != 1:
        raise MaterialPackageAuthoringError(
            "material package must contain exactly one manifest entry"
        )
    entry = entries[0]
    if (
        not isinstance(entry, dict)
        or entry.get("binding") != package.material_prim_path
        or package.material_list_entry.get("binding") != package.material_prim_path
    ):
        raise MaterialPackageAuthoringError(
            "material package manifest bindings are inconsistent"
        )
    if package.representation == "scalar_pbr" and package.textures is not None:
        raise MaterialPackageAuthoringError(
            "scalar material package must not declare textures"
        )
    if package.representation == "textured_pbr" and package.textures is None:
        raise MaterialPackageAuthoringError(
            "textured material package requires complete texture maps"
        )
    info = inspect_material_graph(
        package.material_usd_path,
        package.material_prim_path,
        allow_textured_omnipbr=(package.operation is MaterialAuthoringOperation.CREATE),
    )
    if info.material_profile != package.material_profile:
        raise MaterialPackageAuthoringError("material profile evidence is stale")
    if info.representation != package.representation:
        raise MaterialPackageAuthoringError("material representation evidence is stale")
    if info.topology_sha256 != package.topology_sha256:
        raise MaterialPackageAuthoringError("material topology evidence is stale")
    if package.textures is not None and info.textures != package.textures:
        raise MaterialPackageAuthoringError("material texture evidence is stale")
    expected_artifacts = package.artifact_sha256
    expected_texture_digests = expected_artifacts.get("textures")
    observed_texture_digests = (
        {channel: _sha256(path) for channel, path in _texture_items(package.textures)}
        if package.textures is not None
        else None
    )
    if (
        expected_artifacts.get("material_usd_path")
        != _sha256(package.material_usd_path)
        or expected_artifacts.get("materials_manifest_path")
        != _sha256(package.materials_manifest_path)
        or expected_texture_digests != observed_texture_digests
    ):
        raise MaterialPackageAuthoringError(
            "material package artifact digest evidence is stale"
        )
    return replace(
        package,
        textures=info.textures,
        validation={
            **package.validation,
            "status": "ok",
            "errors": [],
            "warnings": list(validation.warnings),
            **validation.metadata,
            "cache_hit": cache_hit,
        },
    )


def load_material_package(package_dir: str | Path) -> MaterialPackage:
    """Load and revalidate a published package without trusting its manifest."""

    destination = Path(package_dir).expanduser().resolve()
    try:
        package, _stored_request = _read_material_package(destination)
        return _validate_loaded_package(package, cache_hit=False)
    except MaterialPackageAuthoringError:
        raise
    except Exception as exc:
        raise MaterialPackageAuthoringError(
            f"material authoring package is invalid: {exc}"
        ) from exc


def _load_cached_package(
    request: MaterialAuthoringRequest,
    package_dir: Path,
) -> MaterialPackage:
    try:
        package, stored_request = _read_material_package(package_dir)
        if stored_request.get("request_id") != request.request_id:
            raise ValueError("request_id mismatch")
        return _validate_loaded_package(package, cache_hit=True)
    except MaterialPackageAuthoringError:
        raise
    except Exception as exc:
        raise MaterialPackageAuthoringError(
            f"existing material authoring package is not reusable: {exc}"
        ) from exc


def author_material_package(
    request: MaterialAuthoringRequest,
    package_dir: str | Path,
    *,
    overwrite: bool = False,
) -> MaterialPackage:
    """Author and validate one package without owning planning or retry policy."""

    destination = Path(package_dir).expanduser().resolve()
    manifest_path = destination / MATERIAL_AUTHORING_MANIFEST_NAME
    if manifest_path.is_file() and not overwrite:
        return _load_cached_package(request, destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "material package already exists without a reusable authoring manifest"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.authoring-", dir=destination.parent
        )
    )
    try:
        staged_package = _write_package_files(request, staging)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
        return replace(
            staged_package,
            material_usd_path=destination / MATERIAL_LIBRARY_NAME,
            materials_manifest_path=destination / MATERIAL_LIST_MANIFEST_NAME,
            authoring_manifest_path=destination / MATERIAL_AUTHORING_MANIFEST_NAME,
            textures=(
                TextureMapSet(
                    albedo=(
                        destination
                        / staged_package.textures.albedo.relative_to(staging)
                    ),
                    normal=(
                        destination
                        / staged_package.textures.normal.relative_to(staging)
                    ),
                    orm=(
                        destination / staged_package.textures.orm.relative_to(staging)
                    ),
                )
                if staged_package.textures is not None
                else None
            ),
            validation={
                **staged_package.validation,
                "library_path": str(destination / MATERIAL_LIBRARY_NAME),
            },
        )
    except (MaterialGraphEditError, MaterialAuthoringError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise MaterialPackageAuthoringError(str(exc)) from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_material_package_files(
    request: MaterialAuthoringRequest,
    package_dir: str | Path,
    *,
    material_list_entry: dict[str, Any] | None = None,
) -> MaterialPackage:
    """Write authoring files into an existing backend-owned package directory.

    This is the narrow composition seam used by creation backends after they have
    produced channel artifacts.  It does not clean, replace, retry, or cache the
    directory; those lifecycle decisions remain with the caller.
    """

    destination = Path(package_dir).expanduser().resolve()
    try:
        return _write_package_files(
            request,
            destination,
            material_list_entry=material_list_entry,
        )
    except (MaterialGraphEditError, MaterialAuthoringError) as exc:
        raise MaterialPackageAuthoringError(str(exc)) from exc


__all__ = [
    "MATERIAL_AUTHORING_MANIFEST_NAME",
    "MATERIAL_AUTHORING_SCHEMA_VERSION",
    "MaterialAuthoringOperation",
    "MaterialAuthoringRequest",
    "MaterialPackage",
    "MaterialPackageAuthoringError",
    "MaterialRecipeSemantics",
    "MaterialRepresentationPolicy",
    "SourceMaterialReference",
    "author_material_package",
    "load_material_package",
    "write_material_package_files",
]
