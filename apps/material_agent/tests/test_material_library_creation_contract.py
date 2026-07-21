# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for geometry-conditioned material creation."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from material_agent.material_library_generation import (
    BackendMaterialResult,
    CreatedMaterial,
    CreatedMaterialListEntry,
    CreateMaterialRequest,
    FakeMaterialBackendBehavior,
    FakeMaterialCreationBackend,
    IntendedPart,
    MaterialAction,
    MaterialArtifactLayout,
    MaterialChannel,
    MaterialChannelArtifact,
    MaterialChannelComponent,
    MaterialChannelSource,
    MaterialColorSpace,
    MaterialComponentProvenance,
    MaterialConditioningArtifact,
    MaterialConditioningKind,
    MaterialCreationDiagnostic,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    MaterialCreationProvenance,
    MaterialDegradation,
    MaterialDegradationCode,
    MaterialDiagnosticSeverity,
    MaterialRecipe,
    NormalConvention,
    ORMPacking,
    PBRHints,
    PreparedMaterialConditioning,
    intended_part_prim_path_hints,
)
from material_agent.material_library_generation import (
    fake_backend as fake_backend_module,
)


def _source_usd(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    return source


def _recipe(
    *,
    material_id: str = "satin_blue_plastic",
    reference_image_uris: tuple[str, ...] = (),
    pbr_hints: PBRHints | None = None,
) -> MaterialRecipe:
    return MaterialRecipe(
        id=material_id,
        name="Satin Blue Plastic",
        description="Satin blue opaque plastic for the exterior housing.",
        appearance_prompt="satin blue molded plastic with subtle scuffs",
        color="blue",
        material="plastic",
        finish="satin",
        base_color_hint=(0.08, 0.18, 0.72),
        pbr_hints=pbr_hints or PBRHints(metallic=0.0, roughness=0.42),
        reference_image_uris=reference_image_uris,
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
    recipe: MaterialRecipe | None = None,
    target_prim_paths: tuple[str, ...] = ("/World/Asset/Housing",),
    reference_image_uris: tuple[str, ...] = (),
    seed: int | None = None,
) -> CreateMaterialRequest:
    return CreateMaterialRequest(
        source_usd=_source_usd(tmp_path),
        target_prim_paths=target_prim_paths,
        recipe=recipe or _recipe(),
        reference_image_uris=reference_image_uris,
        creation_mode=MaterialCreationMode.ASSET_UV,
        texture_size=64,
        backend="fake",
        seed=seed,
        source_usd_sha256="0" * 64,
    )


def _conditioning(
    request: CreateMaterialRequest,
    tmp_path: Path,
) -> PreparedMaterialConditioning:
    artifacts = [
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.SCOPED_USD,
            uri=(tmp_path / "scoped.usda").as_posix(),
            sha256="1" * 64,
        ),
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.UV_MASK,
            uri=(tmp_path / "uv-mask.png").as_posix(),
            color_space=MaterialColorSpace.RAW,
            sha256="2" * 64,
        ),
    ]
    artifacts.extend(
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.REFERENCE_IMAGE,
            uri=uri,
            color_space=MaterialColorSpace.SRGB,
        )
        for uri in request.effective_reference_image_uris
    )
    return PreparedMaterialConditioning.for_request(
        request,
        artifacts=tuple(artifacts),
    )


def _created_material(
    request: CreateMaterialRequest,
    result: BackendMaterialResult,
    tmp_path: Path,
) -> CreatedMaterial:
    entry = CreatedMaterialListEntry.for_request(
        request,
        creation_manifest="material_creation_manifest.json",
        provenance=result.provenance,
    )
    return CreatedMaterial(
        material_id=request.recipe.material_id,
        material_prim_path=request.recipe.binding,
        material_usd_path=tmp_path / "package" / "material_library.usda",
        creation_manifest_path=tmp_path / "package" / "material_creation_manifest.json",
        texture_artifacts=result.artifacts,
        material_list_entry=entry,
        preview_paths=result.preview_paths,
        validation={"ok": True},
        provenance=result.provenance,
        degradations=result.degradations,
    )


def test_action_contract_and_request_identity_are_stable(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        target_prim_paths=("/World/Asset/Z", "/World/Asset/A"),
    )

    assert MaterialAction.ASSIGN_EXISTING.value == "assign_existing"
    assert MaterialAction.CREATE_NEW.value == "create_new"
    assert MaterialAction.MODIFY_EXISTING.value == "modify_existing"
    assert request.target_prim_paths == ("/World/Asset/A", "/World/Asset/Z")
    assert request.request_id.startswith("mc_")
    assert request.effective_seed == int(request.request_id[3:11], 16)
    assert request.reuse_key == "satin_blue_plastic"
    assert intended_part_prim_path_hints(request.recipe) == ("/World/Asset/Housing",)


def test_request_rejects_invalid_core_inputs(tmp_path: Path) -> None:
    recipe = _recipe()
    source = _source_usd(tmp_path)
    kwargs = {
        "source_usd": source,
        "target_prim_paths": ("/World/Asset/Housing",),
        "recipe": recipe,
    }

    with pytest.raises(ValueError, match="absolute path"):
        CreateMaterialRequest(**{**kwargs, "source_usd": Path("relative.usda")})
    with pytest.raises(ValueError, match="USD, USDA, USDC, or USDZ"):
        CreateMaterialRequest(**{**kwargs, "source_usd": tmp_path / "asset.txt"})
    with pytest.raises(ValueError, match="at least one"):
        CreateMaterialRequest(**{**kwargs, "target_prim_paths": ()})
    with pytest.raises(ValueError, match="duplicate USD prim paths"):
        CreateMaterialRequest(
            **{**kwargs, "target_prim_paths": ("/World/A", "/World/A")}
        )
    with pytest.raises(ValueError, match="reference_image_uris"):
        CreateMaterialRequest(
            **{
                **kwargs,
                "reference_image_uris": ("file:///ref.png", "file:///ref.png"),
            }
        )
    with pytest.raises(ValueError, match="texture_size"):
        CreateMaterialRequest(**{**kwargs, "texture_size": 0})
    with pytest.raises(ValueError, match="seed"):
        CreateMaterialRequest(**{**kwargs, "seed": -1})
    with pytest.raises(ValueError, match="source_usd_sha256"):
        CreateMaterialRequest(**{**kwargs, "source_usd_sha256": "not-a-digest"})
    with pytest.raises(ValueError, match="schema_version"):
        CreateMaterialRequest(**{**kwargs, "schema_version": "old"})
    with pytest.raises(ValueError, match="reference_image_uris"):
        CreateMaterialRequest(
            **{
                **kwargs,
                "recipe": _recipe(reference_image_uris=("   ",)),
            }
        )


def test_optional_references_merge_recipe_and_request_order(tmp_path: Path) -> None:
    recipe = _recipe(reference_image_uris=("file:///recipe.png",))
    request = _request(
        tmp_path,
        recipe=recipe,
        reference_image_uris=("file:///recipe.png", "file:///explicit.png"),
    )
    conditioning = _conditioning(request, tmp_path)

    assert request.effective_reference_image_uris == (
        "file:///recipe.png",
        "file:///explicit.png",
    )
    assert conditioning.reference_image_uris == request.effective_reference_image_uris
    assert conditioning.to_dict()["artifacts"][-2]["uri"] == "file:///recipe.png"
    assert conditioning.to_dict()["artifacts"][-1]["uri"] == "file:///explicit.png"


def test_conditioning_artifact_and_scope_guards(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(ValueError, match="must declare a color space"):
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.UV_MASK,
            uri="file:///uv-mask.png",
        )
    with pytest.raises(ValueError, match="must not declare a color space"):
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.SCOPED_USD,
            uri="file:///scoped.usda",
            color_space=MaterialColorSpace.RAW,
        )
    with pytest.raises(ValueError, match="sha256"):
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.SCOPED_USD,
            uri="file:///scoped.usda",
            sha256="bad",
        )
    view_artifact = MaterialConditioningArtifact(
        kind=MaterialConditioningKind.RENDER,
        uri="file:///render.png",
        color_space=MaterialColorSpace.SRGB,
        view="front",
    )
    assert view_artifact.to_dict()["view"] == "front"

    wrong_target = PreparedMaterialConditioning(
        request_id=request.request_id,
        target_prim_paths=("/World/Other",),
        artifacts=(
            MaterialConditioningArtifact(
                kind=MaterialConditioningKind.SCOPED_USD,
                uri="file:///scoped.usda",
            ),
        ),
    )
    with pytest.raises(ValueError, match="target scope"):
        wrong_target.validate_request(request)
    with pytest.raises(ValueError, match="target scope"):
        MaterialCreationProvenance.for_request(
            request,
            backend="fake",
            backend_revision="fake-material-backend.v1",
            model_revisions=("fake-material-model@v1",),
            duration_seconds=0.0,
            conditioning=wrong_target,
        )

    wrong_refs = PreparedMaterialConditioning(
        request_id=request.request_id,
        target_prim_paths=request.target_prim_paths,
        artifacts=(
            MaterialConditioningArtifact(
                kind=MaterialConditioningKind.SCOPED_USD,
                uri="file:///scoped.usda",
            ),
            MaterialConditioningArtifact(
                kind=MaterialConditioningKind.REFERENCE_IMAGE,
                uri="file:///other.png",
                color_space=MaterialColorSpace.SRGB,
            ),
        ),
        reference_image_uris=("file:///other.png",),
    )
    with pytest.raises(ValueError, match="references"):
        wrong_refs.validate_request(request)


def test_fake_backend_success_fixture_packages_created_entry(tmp_path: Path) -> None:
    request = _request(tmp_path, seed=123)
    conditioning = _conditioning(request, tmp_path)
    backend = FakeMaterialCreationBackend()
    layout = MaterialArtifactLayout(tmp_path / "package", request.recipe.material_id)

    result = backend.create(
        request,
        output_dir=tmp_path / "package",
        conditioning=conditioning,
    )
    created = _created_material(request, result, tmp_path)

    assert backend.calls == [request.request_id]
    assert layout.material_usd_path.name == "material_library.usda"
    assert layout.materials_manifest_path.name == "materials.yaml"
    assert layout.creation_manifest_path.name == "material_creation_manifest.json"
    assert created.material_list_entry.to_dict()["source"] == "generated"
    assert created.material_list_entry.to_dict()["target_prim_paths"] == [
        "/World/Asset/Housing"
    ]
    assert set(created.texture_paths) == {"albedo", "normal", "orm"}
    assert created.texture_paths["albedo"].is_file()
    assert created.texture_paths["orm"].is_file()
    assert created.texture_paths["normal"].is_file()
    assert created.provenance.seed == 123
    assert created.degradations[0].code is MaterialDegradationCode.NEUTRAL_AO
    assert created.to_dict()["texture_paths"]["albedo"].endswith("/albedo.png")
    assert result.to_dict()["provenance"]["request_id"] == request.request_id
    assert isinstance(hash(created), int)


def test_fake_backend_records_per_component_orm_provenance(tmp_path: Path) -> None:
    result = FakeMaterialCreationBackend().create(
        _request(tmp_path),
        output_dir=tmp_path / "package",
    )
    orm = result.artifact(MaterialChannel.ORM)

    assert orm is not None
    assert orm.packing is ORMPacking.OCCLUSION_ROUGHNESS_METALLIC
    provenance = {
        component.component: component.source for component in orm.component_provenance
    }
    assert provenance == {
        MaterialChannelComponent.OCCLUSION: MaterialChannelSource.NEUTRAL_FALLBACK,
        MaterialChannelComponent.ROUGHNESS: MaterialChannelSource.RECIPE_HINT,
        MaterialChannelComponent.METALLIC: MaterialChannelSource.RECIPE_HINT,
    }


def test_channel_artifact_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    component = MaterialComponentProvenance(
        component=MaterialChannelComponent.BASE_COLOR,
        source=MaterialChannelSource.MODEL_GENERATED,
        source_detail="test",
    )
    with pytest.raises(ValueError, match="albedo must use"):
        MaterialChannelArtifact(
            channel=MaterialChannel.ALBEDO,
            path=tmp_path / "albedo.png",
            color_space=MaterialColorSpace.RAW,
            component_provenance=(component,),
        )
    with pytest.raises(ValueError, match="normal must be"):
        MaterialChannelArtifact(
            channel=MaterialChannel.NORMAL,
            path=tmp_path / "normal.png",
            color_space=MaterialColorSpace.RAW,
            component_provenance=(
                MaterialComponentProvenance(
                    component=MaterialChannelComponent.TANGENT_NORMAL,
                    source=MaterialChannelSource.MODEL_GENERATED,
                    source_detail="test",
                ),
            ),
        )
    with pytest.raises(ValueError, match="orm must declare"):
        MaterialChannelArtifact(
            channel=MaterialChannel.ORM,
            path=tmp_path / "orm.png",
            color_space=MaterialColorSpace.RAW,
            component_provenance=(
                MaterialComponentProvenance(
                    component=MaterialChannelComponent.OCCLUSION,
                    source=MaterialChannelSource.NEUTRAL_FALLBACK,
                    source_detail="test",
                ),
                MaterialComponentProvenance(
                    component=MaterialChannelComponent.ROUGHNESS,
                    source=MaterialChannelSource.RECIPE_HINT,
                    source_detail="test",
                ),
                MaterialComponentProvenance(
                    component=MaterialChannelComponent.METALLIC,
                    source=MaterialChannelSource.RECIPE_HINT,
                    source_detail="test",
                ),
            ),
        )
    with pytest.raises(ValueError, match="component provenance"):
        MaterialChannelArtifact(
            channel=MaterialChannel.ALBEDO,
            path=tmp_path / "albedo.png",
            color_space=MaterialColorSpace.SRGB,
            component_provenance=(
                MaterialComponentProvenance(
                    component=MaterialChannelComponent.ROUGHNESS,
                    source=MaterialChannelSource.RECIPE_HINT,
                    source_detail="test",
                ),
            ),
        )
    with pytest.raises(ValueError, match="duplicate components"):
        MaterialChannelArtifact(
            channel=MaterialChannel.ALBEDO,
            path=tmp_path / "albedo.png",
            color_space=MaterialColorSpace.SRGB,
            component_provenance=(component, component),
        )


def test_degraded_normal_fixture_requires_explicit_degradation(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeMaterialCreationBackend(
        FakeMaterialBackendBehavior.DEGRADED_NORMAL
    ).create(request, output_dir=tmp_path / "package")

    assert result.artifact(MaterialChannel.NORMAL) is None
    assert any(
        degradation.code is MaterialDegradationCode.MISSING_NORMAL
        for degradation in result.degradations
    )
    _created_material(request, result, tmp_path)


def test_backend_result_rejects_missing_channel_without_degradation(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    provenance = MaterialCreationProvenance.for_request(
        request,
        backend="fake",
        backend_revision="fake-material-backend.v1",
        model_revisions=("fake-material-model@v1",),
        duration_seconds=0.0,
    )
    albedo = MaterialChannelArtifact(
        channel=MaterialChannel.ALBEDO,
        path=tmp_path / "albedo.png",
        color_space=MaterialColorSpace.SRGB,
        component_provenance=(
            MaterialComponentProvenance(
                component=MaterialChannelComponent.BASE_COLOR,
                source=MaterialChannelSource.MODEL_GENERATED,
                source_detail="test",
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing normal requires"):
        BackendMaterialResult(artifacts=(albedo,), provenance=provenance)

    wrong_channel_degradation = MaterialDegradation(
        code=MaterialDegradationCode.MISSING_NORMAL,
        channels=(MaterialChannel.ORM,),
        message="Normal is missing.",
        fallback="Continue without normal.",
    )
    with pytest.raises(ValueError, match="missing normal requires"):
        BackendMaterialResult(
            artifacts=(albedo,),
            provenance=provenance,
            degradations=(wrong_channel_degradation,),
        )


def test_created_material_rejects_cross_field_mismatches(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = FakeMaterialCreationBackend().create(
        request,
        output_dir=tmp_path / "package",
    )
    entry = CreatedMaterialListEntry.for_request(
        request,
        creation_manifest="material_creation_manifest.json",
        provenance=result.provenance,
    )

    with pytest.raises(ValueError, match="generation_id"):
        CreatedMaterial(
            material_id="other_material",
            material_prim_path=request.recipe.binding,
            material_usd_path=tmp_path / "package" / "material_library.usda",
            creation_manifest_path=tmp_path
            / "package"
            / "material_creation_manifest.json",
            texture_artifacts=result.artifacts,
            material_list_entry=entry,
            preview_paths=result.preview_paths,
            validation={"ok": True},
            provenance=result.provenance,
            degradations=result.degradations,
        )
    with pytest.raises(ValueError, match="material_prim_path"):
        CreatedMaterial(
            material_id=request.recipe.material_id,
            material_prim_path="/World/Looks/Other",
            material_usd_path=tmp_path / "package" / "material_library.usda",
            creation_manifest_path=tmp_path
            / "package"
            / "material_creation_manifest.json",
            texture_artifacts=result.artifacts,
            material_list_entry=entry,
            preview_paths=result.preview_paths,
            validation={"ok": True},
            provenance=result.provenance,
            degradations=result.degradations,
        )
    with pytest.raises(ValueError, match="cache keys"):
        CreatedMaterial(
            material_id=request.recipe.material_id,
            material_prim_path=request.recipe.binding,
            material_usd_path=tmp_path / "package" / "material_library.usda",
            creation_manifest_path=tmp_path
            / "package"
            / "material_creation_manifest.json",
            texture_artifacts=result.artifacts,
            material_list_entry=entry,
            preview_paths=result.preview_paths,
            validation={"ok": True},
            provenance=replace(result.provenance, cache_key="1" * 64),
            degradations=result.degradations,
        )
    with pytest.raises(ValueError, match="target scopes"):
        CreatedMaterial(
            material_id=request.recipe.material_id,
            material_prim_path=request.recipe.binding,
            material_usd_path=tmp_path / "package" / "material_library.usda",
            creation_manifest_path=tmp_path
            / "package"
            / "material_creation_manifest.json",
            texture_artifacts=result.artifacts,
            material_list_entry=entry,
            preview_paths=result.preview_paths,
            validation={"ok": True},
            provenance=replace(
                result.provenance,
                target_prim_paths=("/World/Asset/Other",),
            ),
            degradations=result.degradations,
        )
    with pytest.raises(ValueError, match="duplicate channels"):
        CreatedMaterial(
            material_id=request.recipe.material_id,
            material_prim_path=request.recipe.binding,
            material_usd_path=tmp_path / "package" / "material_library.usda",
            creation_manifest_path=tmp_path
            / "package"
            / "material_creation_manifest.json",
            texture_artifacts=(*result.artifacts, result.artifacts[0]),
            material_list_entry=entry,
            preview_paths=result.preview_paths,
            validation={"ok": True},
            provenance=result.provenance,
            degradations=result.degradations,
        )
    with pytest.raises(ValueError, match="requires albedo"):
        CreatedMaterial(
            material_id=request.recipe.material_id,
            material_prim_path=request.recipe.binding,
            material_usd_path=tmp_path / "package" / "material_library.usda",
            creation_manifest_path=tmp_path
            / "package"
            / "material_creation_manifest.json",
            texture_artifacts=tuple(
                artifact
                for artifact in result.artifacts
                if artifact.channel is not MaterialChannel.ALBEDO
            ),
            material_list_entry=entry,
            preview_paths=result.preview_paths,
            validation={"ok": True},
            provenance=result.provenance,
            degradations=result.degradations,
        )


def test_failure_and_unsupported_fixtures_do_not_return_packages(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    with pytest.raises(MaterialCreationError) as failure:
        FakeMaterialCreationBackend(FakeMaterialBackendBehavior.FAILURE).create(
            request,
            output_dir=tmp_path / "failure",
        )
    assert failure.value.code is MaterialCreationErrorCode.BACKEND_FAILURE
    assert failure.value.retryable is True
    assert failure.value.diagnostics[0].phase == "generation"

    unsupported = _request(
        tmp_path,
        recipe=_recipe(pbr_hints=PBRHints(opacity=0.8, transmission=0.2)),
    )
    with pytest.raises(MaterialCreationError) as unsupported_error:
        FakeMaterialCreationBackend().create(
            unsupported,
            output_dir=tmp_path / "unsupported",
        )
    assert (
        unsupported_error.value.code is MaterialCreationErrorCode.UNSUPPORTED_MATERIAL
    )
    assert unsupported_error.value.retryable is False

    with pytest.raises(MaterialCreationError) as explicit_unsupported_error:
        FakeMaterialCreationBackend(FakeMaterialBackendBehavior.UNSUPPORTED).create(
            request,
            output_dir=tmp_path / "explicit-unsupported",
        )
    assert (
        explicit_unsupported_error.value.code
        is MaterialCreationErrorCode.UNSUPPORTED_MATERIAL
    )
    assert explicit_unsupported_error.value.retryable is False


def test_fake_backend_unsupported_check_treats_absent_pbr_hints_as_supported() -> None:
    class RecipeWithoutHints:
        pbr_hints: object | None = None

    class RequestWithoutHints:
        recipe: RecipeWithoutHints = RecipeWithoutHints()

    request = cast(CreateMaterialRequest, RequestWithoutHints())

    assert fake_backend_module._is_unsupported(request) is False


def test_cancelled_backend_and_error_serialization(tmp_path: Path) -> None:
    cancel_event = threading.Event()
    cancel_event.set()
    backend = FakeMaterialCreationBackend()

    with pytest.raises(MaterialCreationError) as cancelled:
        backend.create(
            _request(tmp_path),
            output_dir=tmp_path / "cancelled",
            cancel_event=cancel_event,
        )

    assert backend.calls == []
    assert cancelled.value.to_dict()["code"] == MaterialCreationErrorCode.CANCELLED


def test_diagnostics_and_degradations_are_serializable_and_hashable() -> None:
    diagnostic = MaterialCreationDiagnostic(
        code="TEST",
        message="diagnostic",
        severity=MaterialDiagnosticSeverity.WARNING,
        phase="validation",
        channels=(MaterialChannel.ORM,),
        retryable=True,
        details={"detail": "value"},
    )
    degradation = MaterialDegradation(
        code=MaterialDegradationCode.NEUTRAL_AO,
        channels=(MaterialChannel.ORM,),
        message="AO was unavailable.",
        fallback="Use neutral AO.",
    )

    assert diagnostic.to_dict()["details"] == {"detail": "value"}
    assert degradation.to_dict()["fallback"] == "Use neutral AO."
    assert isinstance(hash(diagnostic), int)


def test_reuse_key_is_recipe_scoped_but_cache_key_detects_conflicts(
    tmp_path: Path,
) -> None:
    recipe = _recipe()
    first = _request(
        tmp_path / "first",
        recipe=recipe,
        target_prim_paths=("/World/Asset/Housing",),
    )
    second = _request(
        tmp_path / "second",
        recipe=recipe,
        target_prim_paths=("/World/Asset/Door",),
    )

    assert first.reuse_key == second.reuse_key == recipe.material_id
    assert first.request_id != second.request_id

    backend = FakeMaterialCreationBackend()
    first_result = backend.create(first, output_dir=tmp_path / "first-package")
    second_result = backend.create(second, output_dir=tmp_path / "second-package")

    assert first_result.provenance.cache_key != second_result.provenance.cache_key
    assert (
        CreatedMaterialListEntry.for_request(
            first,
            creation_manifest="material_creation_manifest.json",
            provenance=first_result.provenance,
        ).reuse_key
        == recipe.material_id
    )
