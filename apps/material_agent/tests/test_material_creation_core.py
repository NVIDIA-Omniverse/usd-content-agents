# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material creation orchestration and packaging."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

pxr = pytest.importorskip("pxr")

from pxr import Usd  # noqa: E402

import material_agent.material_library_generation.creation as creation_module  # noqa: E402
from material_agent.material_library_generation import (  # noqa: E402
    BackendMaterialResult,
    CreateMaterialRequest,
    FakeMaterialBackendBehavior,
    FakeMaterialCreationBackend,
    IntendedPart,
    MaterialChannel,
    MaterialConditioningArtifact,
    MaterialConditioningKind,
    MaterialCreationBackendRegistry,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialRecipe,
    PBRHints,
    PreparedMaterialConditioning,
    create_material_package,
)


def _source_usd(tmp_path: Path) -> Path:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    return source.resolve()


def _recipe(*, material_id: str = "satin_blue_plastic") -> MaterialRecipe:
    return MaterialRecipe(
        id=material_id,
        name="Satin Blue Plastic",
        description="Satin blue opaque plastic for the exterior housing.",
        appearance_prompt="satin blue molded plastic with subtle scuffs",
        color="blue",
        material="plastic",
        finish="satin",
        base_color_hint=(0.08, 0.18, 0.72),
        pbr_hints=PBRHints(metallic=0.0, roughness=0.42),
        intended_parts=(
            IntendedPart(
                semantic_label="main housing",
                evidence="Planner selected the exterior housing.",
                prim_path_hints=("/World/Asset/Housing",),
            ),
        ),
    )


def _request(
    tmp_path: Path,
    *,
    target_prim_paths: tuple[str, ...] = ("/World/Asset/Housing",),
) -> CreateMaterialRequest:
    return CreateMaterialRequest(
        source_usd=_source_usd(tmp_path),
        target_prim_paths=target_prim_paths,
        recipe=_recipe(),
        texture_size=64,
        backend="auto",
        source_usd_sha256="0" * 64,
    )


def _registry(backend: FakeMaterialCreationBackend) -> MaterialCreationBackendRegistry:
    registry = MaterialCreationBackendRegistry()
    registry.register(backend, make_default=True)
    return registry


def _conditioning(
    request: CreateMaterialRequest,
    marker: str,
) -> PreparedMaterialConditioning:
    return PreparedMaterialConditioning.for_request(
        request,
        artifacts=(
            MaterialConditioningArtifact(
                kind=MaterialConditioningKind.SCOPED_USD,
                uri=f"fixture://{marker}.usda",
            ),
        ),
    )


def _rewrite_creation_manifest(package_dir: Path, data: dict[str, Any]) -> None:
    (package_dir / "material_creation_manifest.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ConditioningBlindFakeBackend(FakeMaterialCreationBackend):
    """Fake backend variant that omits current conditioning from provenance."""

    def create(
        self,
        request: CreateMaterialRequest,
        *,
        output_dir: Path,
        conditioning: PreparedMaterialConditioning | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BackendMaterialResult:
        return super().create(
            request,
            output_dir=output_dir,
            conditioning=None,
            cancel_event=cancel_event,
        )


def test_registry_resolves_default_and_named_backends() -> None:
    registry = MaterialCreationBackendRegistry()
    backend = FakeMaterialCreationBackend()

    with pytest.raises(MaterialCreationError) as missing_default:
        registry.resolve("auto")
    assert missing_default.value.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE

    registry.register(backend, make_default=True)
    assert registry.backend_names == ("fake",)
    assert registry.default_backend_name == "fake"
    assert registry.resolve("auto") is backend
    assert registry.resolve("fake") is backend

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeMaterialCreationBackend())

    replacement = FakeMaterialCreationBackend()
    registry.register(replacement, replace_existing=True)
    assert registry.resolve("fake") is replacement


def test_create_material_package_writes_portable_package_and_manifest(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = FakeMaterialCreationBackend()
    package_dir = tmp_path / "package"

    created = create_material_package(
        request,
        package_dir,
        registry=_registry(backend),
    )

    assert backend.calls == [request.request_id]
    assert created.material_id == request.recipe.material_id
    assert created.material_prim_path == request.recipe.binding
    assert created.material_usd_path == package_dir / "material_library.usda"
    assert created.creation_manifest_path == (
        package_dir / "material_creation_manifest.json"
    )
    assert set(created.texture_paths) == {"albedo", "normal", "orm"}
    assert created.validation["cache_hit"] is False

    manifest = yaml.safe_load((package_dir / "materials.yaml").read_text())
    assert manifest["library_path"] == "material_library.usda"
    assert manifest["entries"] == [created.material_list_entry.to_dict()]
    assert manifest["entries"][0]["creation_manifest"] == (
        "material_creation_manifest.json"
    )

    creation_manifest = json.loads(
        (package_dir / "material_creation_manifest.json").read_text()
    )
    assert creation_manifest["schema_version"] == "material-agent-create.v1"
    assert creation_manifest["request"]["request_id"] == request.request_id
    assert creation_manifest["backend_result"]["provenance"]["backend"] == "fake"
    assert creation_manifest["created_material"]["material_usd_path"] == (
        "material_library.usda"
    )
    assert creation_manifest["created_material"]["texture_paths"]["albedo"] == (
        "textures/satin_blue_plastic/albedo.png"
    )

    stage = Usd.Stage.Open(str(created.material_usd_path))
    assert stage is not None
    assert stage.GetPrimAtPath(request.recipe.binding)


def test_create_material_package_reuses_matching_cached_manifest(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = FakeMaterialCreationBackend()
    registry = _registry(backend)
    package_dir = tmp_path / "package"

    created = create_material_package(request, package_dir, registry=registry)
    cached = create_material_package(request, package_dir, registry=registry)

    assert backend.calls == [request.request_id]
    assert cached.provenance.cache_key == created.provenance.cache_key
    assert cached.validation["cache_hit"] is True


def test_create_material_package_rejects_cache_when_conditioning_changes(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = FakeMaterialCreationBackend()
    registry = _registry(backend)
    package_dir = tmp_path / "package"

    create_material_package(
        request,
        package_dir,
        registry=registry,
        conditioning=_conditioning(request, "first"),
    )

    with pytest.raises(MaterialCreationError) as error:
        create_material_package(
            request,
            package_dir,
            registry=registry,
            conditioning=_conditioning(request, "second"),
        )

    assert error.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "conditioning" in str(error.value)
    assert backend.calls == [request.request_id]


def test_create_material_package_rejects_fresh_result_with_stale_conditioning(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"

    with pytest.raises(MaterialCreationError) as error:
        create_material_package(
            request,
            package_dir,
            registry=_registry(ConditioningBlindFakeBackend()),
            conditioning=_conditioning(request, "current"),
        )

    assert error.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "conditioning" in str(error.value)
    assert not package_dir.exists()


def test_create_material_package_rejects_different_request_for_existing_manifest(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    registry = _registry(FakeMaterialCreationBackend())
    create_material_package(request, package_dir, registry=registry)

    changed_scope = _request(
        tmp_path,
        target_prim_paths=("/World/Asset/OtherHousing",),
    )
    with pytest.raises(MaterialCreationError) as error:
        create_material_package(changed_scope, package_dir, registry=registry)

    assert error.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "different creation request" in str(error.value)


def test_cached_manifest_rejects_material_file_outside_package(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    registry = _registry(FakeMaterialCreationBackend())
    create_material_package(request, package_dir, registry=registry)
    external_usd = tmp_path / "external.usda"
    external_usd.write_text("#usda 1.0\n", encoding="utf-8")
    manifest_path = package_dir / "material_creation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_material"]["material_usd_path"] = external_usd.as_posix()
    _rewrite_creation_manifest(package_dir, manifest)

    with pytest.raises(MaterialCreationError) as error:
        create_material_package(request, package_dir, registry=registry)

    assert error.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "outside the package directory" in str(error.value)


def test_cached_manifest_rejects_texture_artifact_outside_package(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    registry = _registry(FakeMaterialCreationBackend())
    create_material_package(request, package_dir, registry=registry)
    external_png = tmp_path / "external.png"
    external_png.write_bytes(b"not-a-real-png")
    manifest_path = package_dir / "material_creation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_material"]["texture_artifacts"][0]["path"] = (
        external_png.as_posix()
    )
    _rewrite_creation_manifest(package_dir, manifest)

    with pytest.raises(MaterialCreationError) as error:
        create_material_package(request, package_dir, registry=registry)

    assert error.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "outside the package directory" in str(error.value)


def test_create_material_package_rejects_existing_directory_without_manifest(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    with pytest.raises(MaterialCreationError) as error:
        create_material_package(
            request,
            package_dir,
            registry=_registry(FakeMaterialCreationBackend()),
        )

    assert error.value.code is MaterialCreationErrorCode.INVALID_OUTPUT
    assert "without a reusable creation manifest" in str(error.value)


def test_create_material_package_overwrites_existing_directory_when_requested(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    stale_file = package_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")

    created = create_material_package(
        request,
        package_dir,
        registry=_registry(FakeMaterialCreationBackend()),
        overwrite=True,
    )

    assert not stale_file.exists()
    assert created.creation_manifest_path.is_file()


def test_create_material_package_cleans_partial_output_on_backend_failure(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    registry = _registry(
        FakeMaterialCreationBackend(FakeMaterialBackendBehavior.FAILURE)
    )

    with pytest.raises(MaterialCreationError) as error:
        create_material_package(request, package_dir, registry=registry)

    assert error.value.code is MaterialCreationErrorCode.BACKEND_FAILURE
    assert not package_dir.exists()


def test_degraded_normal_backend_gets_explicit_packaging_fallback(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    package_dir = tmp_path / "package"
    backend = FakeMaterialCreationBackend(FakeMaterialBackendBehavior.DEGRADED_NORMAL)

    created = create_material_package(
        request,
        package_dir,
        registry=_registry(backend),
    )

    assert created.texture_paths["normal"].is_file()
    assert created.texture_paths["normal"].name == "normal.png"
    assert any(
        degradation.code.value == "missing_normal"
        for degradation in created.degradations
    )
    assert created.material_list_entry.creation_cache_key == (
        created.provenance.cache_key
    )

    creation_manifest = json.loads(
        (package_dir / "material_creation_manifest.json").read_text()
    )
    backend_channels = {
        artifact["channel"]
        for artifact in creation_manifest["backend_result"]["artifacts"]
    }
    created_channels = {
        artifact["channel"]
        for artifact in creation_manifest["created_material"]["texture_artifacts"]
    }
    assert backend_channels == {"albedo", "orm"}
    assert created_channels == {"albedo", "normal", "orm"}
    assert creation_manifest["created_material"]["texture_paths"]["normal"] == (
        "textures/satin_blue_plastic/normal.png"
    )


def test_creation_manifest_path_helpers_handle_relative_and_external_paths(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    external_path = tmp_path / "external.png"

    assert (
        creation_module._relative_or_original(
            "textures/material/albedo.png",
            package_dir,
        )
        == "textures/material/albedo.png"
    )
    assert (
        creation_module._relative_or_original(
            str(external_path),
            package_dir,
        )
        == external_path.as_posix()
    )
    assert (
        creation_module._resolve_package_path(
            str(external_path),
            package_dir,
        )
        == external_path
    )
