# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

import material_agent.material_library_generation.conditioning as conditioning_module
from material_agent.material_library_generation.conditioning import (
    MATERIAL_CONDITIONING_MANIFEST_NAME,
    MATERIAL_CONDITIONING_SCHEMA_VERSION,
    OVRTX_CONDITIONING_SCHEMA_VERSION,
    REAL_SEED_MATERIAL_SCHEMA_VERSION,
    MaterialConditioningEvidenceMode,
    MaterialConditioningOptions,
    RealMaterialConditioningInputs,
    _reject_non_real_evidence_in_real_mode,
    prepare_material_conditioning,
)
from material_agent.material_library_generation.creation_contract import (
    CreateMaterialRequest,
    MaterialColorSpace,
    MaterialConditioningArtifact,
    MaterialConditioningArtifactSource,
    MaterialConditioningKind,
    MaterialCreationError,
    MaterialCreationErrorCode,
)
from material_agent.material_library_generation.schema import (
    IntendedPart,
    MaterialRecipe,
    PBRHints,
)


def test_prepare_material_conditioning_writes_scoped_backend_inputs(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    source_digest = _sha256(source_usd)
    recipe_ref = _write_reference(tmp_path / "recipe_ref.png")
    request_ref = _write_reference(tmp_path / "request_ref.png")
    file_ref = _write_reference(tmp_path / "file ref.png")
    file_ref_uri = file_ref.as_uri()
    remote_file_ref = "file://example.invalid/reference.png"
    malformed_ref = "://not-a-local-path"
    remote_ref = "https://example.invalid/reference.png"
    request = _request(
        source_usd,
        reference_image_uris=(recipe_ref.as_posix(),),
        request_reference_image_uris=(
            request_ref.as_posix(),
            file_ref_uri,
            remote_file_ref,
            malformed_ref,
            remote_ref,
        ),
        source_usd_sha256=source_digest,
    )

    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=MaterialConditioningOptions(image_size=4, render_views=("oblique",)),
    )

    result.conditioning.validate_request(request)
    assert result.request_id == request.request_id
    assert result.output_dir.name.startswith(f"{request.request_id}-")
    assert result.manifest_path.name == MATERIAL_CONDITIONING_MANIFEST_NAME
    assert result.to_dict()["metadata"]["schema_version"] == (
        MATERIAL_CONDITIONING_SCHEMA_VERSION
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest == result.metadata
    assert manifest["evidence_mode"] == (
        MaterialConditioningEvidenceMode.DETERMINISTIC_FIXTURE.value
    )
    assert manifest["options"]["evidence_mode"] == (
        MaterialConditioningEvidenceMode.DETERMINISTIC_FIXTURE.value
    )
    assert manifest["source_usd_sha256"] == source_digest
    assert manifest["target_scope"]["target_mesh_paths"] == ["/World/Asset/Body"]
    assert manifest["target_scope"]["all_target_meshes_have_uvs"] is True

    kinds = [artifact.kind for artifact in result.conditioning.artifacts]
    assert kinds[:7] == [
        MaterialConditioningKind.SCOPED_USD,
        MaterialConditioningKind.UV_LAYOUT,
        MaterialConditioningKind.UV_MASK,
        MaterialConditioningKind.NORMAL,
        MaterialConditioningKind.DEPTH,
        MaterialConditioningKind.SEGMENTATION,
        MaterialConditioningKind.RENDER,
    ]
    assert MaterialConditioningKind.SOURCE_ALBEDO in kinds
    assert kinds[-6:] == [
        MaterialConditioningKind.REFERENCE_IMAGE,
        MaterialConditioningKind.REFERENCE_IMAGE,
        MaterialConditioningKind.REFERENCE_IMAGE,
        MaterialConditioningKind.REFERENCE_IMAGE,
        MaterialConditioningKind.REFERENCE_IMAGE,
        MaterialConditioningKind.REFERENCE_IMAGE,
    ]

    for artifact in result.conditioning.artifacts:
        if artifact.sha256 is not None and "://" not in artifact.uri:
            assert artifact.sha256 == _sha256(Path(artifact.uri))

    reference_artifacts = [
        artifact
        for artifact in result.conditioning.artifacts
        if artifact.kind is MaterialConditioningKind.REFERENCE_IMAGE
    ]
    assert [artifact.uri for artifact in reference_artifacts] == [
        recipe_ref.as_posix(),
        request_ref.as_posix(),
        file_ref_uri,
        remote_file_ref,
        malformed_ref,
        remote_ref,
    ]
    assert reference_artifacts[0].sha256 == _sha256(recipe_ref)
    assert reference_artifacts[1].sha256 == _sha256(request_ref)
    assert reference_artifacts[2].sha256 == _sha256(file_ref)
    assert reference_artifacts[3].sha256 is None
    assert reference_artifacts[4].sha256 is None
    assert reference_artifacts[5].sha256 is None
    assert reference_artifacts[0].evidence_source == (
        MaterialConditioningArtifactSource.RECIPE_REFERENCE
    )
    assert all(
        artifact.evidence_source is MaterialConditioningArtifactSource.REQUEST_REFERENCE
        for artifact in reference_artifacts[1:]
    )

    scoped_stage = Usd.Stage.Open(result.scoped_usd_path.as_posix())
    assert scoped_stage is not None
    target = UsdGeom.Imageable(scoped_stage.GetPrimAtPath("/World/Asset/Body"))
    sibling = UsdGeom.Imageable(scoped_stage.GetPrimAtPath("/World/Asset/Other"))
    assert target.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert sibling.ComputeVisibility() == UsdGeom.Tokens.invisible
    assert _sha256(source_usd) == source_digest


def test_prepare_material_conditioning_rejects_placeholders_in_real_evidence_mode(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd)
    output_dir = tmp_path / "conditioning"

    with pytest.raises(MaterialCreationError) as error:
        prepare_material_conditioning(
            request,
            output_dir,
            options=MaterialConditioningOptions(
                image_size=2,
                evidence_mode=MaterialConditioningEvidenceMode.REAL_EVIDENCE,
            ),
        )

    assert error.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert error.value.backend == "fake"
    message = str(error.value)
    assert "real_evidence" in message
    assert "placeholder" in message
    assert "render" in message
    assert not output_dir.exists()


def test_step1x_conditioning_rejects_missing_explicit_real_inputs(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")

    with pytest.raises(MaterialCreationError, match="explicit seed-material") as error:
        prepare_material_conditioning(request, tmp_path / "conditioning")

    assert error.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert error.value.backend == "step1x_material_anything"
    assert "OVRTX" in str(error.value)


def test_step1x_default_real_conditioning_accepts_large_texture_request(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        texture_size=16384,
    )
    output_dir = tmp_path / "conditioning"

    with pytest.raises(MaterialCreationError, match="explicit seed-material"):
        prepare_material_conditioning(request, output_dir)

    assert not output_dir.exists()


def test_step1x_conditioning_rejects_explicit_deterministic_fixture_mode(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")

    with pytest.raises(MaterialCreationError, match="deterministic fixtures"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=MaterialConditioningOptions(image_size=2),
        )


def test_step1x_conditioning_uses_verified_seed_mesh_st_and_ovrtx_evidence(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)

    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=options,
    )

    assert result.metadata["evidence_mode"] == "real_evidence"
    kinds = [artifact.kind for artifact in result.conditioning.artifacts]
    assert kinds == [
        MaterialConditioningKind.SCOPED_USD,
        MaterialConditioningKind.SEED_MATERIAL,
        MaterialConditioningKind.SOURCE_ALBEDO,
        MaterialConditioningKind.MESH_ST,
        MaterialConditioningKind.RENDER_REQUEST,
        MaterialConditioningKind.RENDER,
    ]
    assert all(artifact.sha256 for artifact in result.conditioning.artifacts)
    assert all(
        artifact.evidence_source is not MaterialConditioningArtifactSource.PLACEHOLDER
        for artifact in result.conditioning.artifacts
    )
    metadata = result.metadata["real_evidence"]
    assert metadata["seed_material"]["source_metadata"]["kind"] == "approved_s3"
    assert metadata["ovrtx"]["request"]["source_usd_sha256"] == _sha256(source_usd)
    assert metadata["ovrtx"]["provider_revision"] == "temporary-test-fixture"
    assert metadata["ovrtx"]["request_id"] == request.request_id
    assert metadata["mesh_st"]["meshes"][0]["primvar_name"] == "st"
    assert metadata["mesh_st"]["meshes"][0]["interpolation"] == "faceVarying"
    assert metadata["mesh_st"]["meshes"][0]["value_count"] == 4
    assert metadata["mesh_st"]["meshes"][0]["flattened_value_count"] == 4

    source_albedo = next(
        artifact
        for artifact in result.conditioning.artifacts
        if artifact.kind is MaterialConditioningKind.SOURCE_ALBEDO
    )
    assert source_albedo.evidence_source is (
        MaterialConditioningArtifactSource.SOURCE_DERIVED
    )
    assert source_albedo.sha256 == _sha256(Path(source_albedo.uri))
    assert Path(source_albedo.uri).parent == result.scoped_usd_path.parent
    scoped_stage = Usd.Stage.Open(result.scoped_usd_path.as_posix())
    assert scoped_stage is not None
    target = scoped_stage.GetPrimAtPath("/World/Asset/Body")
    material, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
    assert material
    assert str(material.GetPath()).startswith("/__MaterialCreationConditioning/Looks/")
    texture_paths = []
    for prim in Usd.PrimRange(material.GetPrim()):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() == "UsdUVTexture":
            texture_paths.append(shader.GetInput("file").Get().path)
    assert texture_paths == [Path(source_albedo.uri).name]
    assert (result.scoped_usd_path.parent / texture_paths[0]).is_file()


def test_real_conditioning_run_identity_tracks_ovrtx_manifest(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None

    first = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=options,
    )
    ovrtx_manifest = options.real_evidence.ovrtx_manifest_path
    manifest = json.loads(ovrtx_manifest.read_text(encoding="utf-8"))
    manifest["provider_revision"] = "temporary-test-fixture-revision-2"
    ovrtx_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    second = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=options,
    )

    assert first.output_dir != second.output_dir
    assert (
        first.metadata["real_evidence"]["ovrtx"]["manifest_sha256"]
        != (second.metadata["real_evidence"]["ovrtx"]["manifest_sha256"])
    )


def test_real_conditioning_reads_each_manifest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_paths = {
        options.real_evidence.seed_manifest_path,
        options.real_evidence.ovrtx_manifest_path,
    }
    reads = dict.fromkeys(manifest_paths, 0)
    read_bytes = Path.read_bytes

    def count_manifest_read(path: Path) -> bytes:
        if path in reads:
            reads[path] += 1
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", count_manifest_read)

    prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=options,
    )

    assert reads == dict.fromkeys(manifest_paths, 1)


def test_real_conditioning_rejects_missing_seed_albedo(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    seed_manifest = json.loads(
        options.real_evidence.seed_manifest_path.read_text(encoding="utf-8")
    )
    albedo = (
        options.real_evidence.seed_manifest_path.parent
        / seed_manifest["source_albedo"]["path"]
    )
    albedo.unlink()
    output_dir = tmp_path / "conditioning"

    with pytest.raises(MaterialCreationError, match="seed source albedo is missing"):
        prepare_material_conditioning(
            request,
            output_dir,
            options=options,
        )
    assert not output_dir.exists()


def test_real_conditioning_rejects_recipe_synthesized_seed_source(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.seed_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["kind"] = "recipe_synthesized"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    options = MaterialConditioningOptions(
        render_views=(),
        include_normal=False,
        include_depth=False,
        include_segmentation=False,
        evidence_mode=MaterialConditioningEvidenceMode.REAL_EVIDENCE,
        real_evidence=RealMaterialConditioningInputs(
            seed_manifest_path=manifest_path,
            seed_manifest_sha256=_sha256(manifest_path),
            ovrtx_manifest_path=options.real_evidence.ovrtx_manifest_path,
        ),
    )

    with pytest.raises(MaterialCreationError, match="recipe_synthesized"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_seed_member_path_escape(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.seed_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_albedo"]["path"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    options = replace(
        options,
        real_evidence=replace(
            options.real_evidence,
            seed_manifest_sha256=_sha256(manifest_path),
        ),
    )

    with pytest.raises(MaterialCreationError, match="must stay within"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_requires_source_albedo_option(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = replace(
        _write_real_conditioning_inputs(tmp_path, request),
        include_source_albedo=False,
    )

    with pytest.raises(MaterialCreationError, match="source albedo is required"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_missing_seed_manifest(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    options = replace(
        options,
        real_evidence=replace(
            options.real_evidence,
            seed_manifest_path=tmp_path / "missing-seed-manifest.json",
        ),
    )

    with pytest.raises(MaterialCreationError, match="seed manifest does not exist"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_seed_manifest_config_digest(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    options = replace(
        options,
        real_evidence=replace(
            options.real_evidence,
            seed_manifest_sha256="0" * 64,
        ),
    )

    with pytest.raises(MaterialCreationError, match="configured digest"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("schema", "unsupported schema_version"),
        ("source_kind", "source kind must be"),
        ("source_uri", "must provide an s3:// URI"),
        ("byte_size", "byte_size"),
        ("width", "width"),
        ("mapping", "requires a material_usd object"),
        ("string", "requires non-empty package_id"),
        ("sha256", "lowercase SHA-256"),
        ("prim_path", "absolute USD prim path"),
        ("missing_prim", "declared material prim"),
        ("texture_reference", "only the declared source albedo"),
        ("flat_albedo", "must not be a flat-color image"),
        ("st_connection", "must sample the st primvar"),
        ("st_reader", "must sample the st primvar"),
        ("surface_output", "must drive the material surface"),
        ("surface_nodegraph", "must drive the material surface"),
        ("surface_connection", "must drive the material surface"),
        ("json_object", "must contain an object"),
    ],
)
def test_real_conditioning_rejects_invalid_seed_manifest_contract(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.seed_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if case == "schema":
        manifest["schema_version"] = "unsupported"
    elif case == "source_kind":
        manifest["source"]["kind"] = "external"
    elif case == "source_uri":
        manifest["source"]["uri"] = "https://example.test/albedo.png"
    elif case == "byte_size":
        manifest["source"]["byte_size"] = 1
    elif case == "width":
        manifest["source_albedo"]["width"] = 3
    elif case == "mapping":
        manifest["material_usd"] = []
    elif case == "string":
        manifest["package_id"] = ""
    elif case == "sha256":
        manifest["material_usd"]["sha256"] = "not-a-digest"
    elif case == "prim_path":
        manifest["material_usd"]["material_prim_path"] = "Seed/Material"
    elif case == "missing_prim":
        manifest["material_usd"]["material_prim_path"] = "/Seed/Missing"
    elif case == "texture_reference":
        material_path = manifest_path.parent / manifest["material_usd"]["path"]
        material_path.write_text(
            material_path.read_text(encoding="utf-8").replace(
                "textures/real_source_albedo.png",
                "textures/other.png",
            ),
            encoding="utf-8",
        )
        manifest["material_usd"]["sha256"] = _sha256(material_path)
    elif case == "flat_albedo":
        albedo_path = manifest_path.parent / manifest["source_albedo"]["path"]
        Image.new("RGB", (2, 2), (120, 130, 140)).save(albedo_path)
        manifest["source_albedo"]["sha256"] = _sha256(albedo_path)
        manifest["source"]["byte_size"] = albedo_path.stat().st_size
    elif case in {
        "st_connection",
        "st_reader",
        "surface_output",
        "surface_nodegraph",
        "surface_connection",
    }:
        material_path = manifest_path.parent / manifest["material_usd"]["path"]
        material_text = material_path.read_text(encoding="utf-8")
        if case == "st_connection":
            material_text = material_text.replace(
                "float2 inputs:st.connect = </Seed/Material/UVReader.outputs:result>",
                "float2 inputs:st = (0, 0)",
            )
        elif case == "st_reader":
            material_text = material_text.replace(
                'token inputs:varname = "st"',
                'token inputs:varname = "st1"',
            )
        elif case == "surface_output":
            material_text = material_text.replace(
                "token outputs:surface.connect = "
                "</Seed/Material/Surface.outputs:surface>",
                "token outputs:surface",
            )
        elif case == "surface_nodegraph":
            material_text = material_text.replace(
                "</Seed/Material/Surface.outputs:surface>",
                "</Seed/Material/Graph.outputs:surface>",
            ).replace(
                '        def Shader "UVReader"',
                '        def NodeGraph "Graph"\n'
                "        {\n"
                "            token outputs:surface\n"
                "        }\n"
                '        def Shader "UVReader"',
            )
        else:
            material_text = material_text.replace(
                "color3f inputs:diffuseColor.connect = "
                "</Seed/Material/Albedo.outputs:rgb>",
                "color3f inputs:diffuseColor = (0.5, 0.5, 0.5)",
            )
        material_path.write_text(material_text, encoding="utf-8")
        manifest["material_usd"]["sha256"] = _sha256(material_path)
    elif case == "json_object":
        manifest_path.write_text("[]", encoding="utf-8")
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(case)

    if case != "json_object":
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    options = replace(
        options,
        real_evidence=replace(
            options.real_evidence,
            seed_manifest_sha256=_sha256(manifest_path),
        ),
    )

    with pytest.raises(MaterialCreationError, match=expected_message):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_target_mesh_without_st(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/Asset/Other",),
    )
    options = _write_real_conditioning_inputs(tmp_path, request)

    with pytest.raises(MaterialCreationError, match="no authored st"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_existing_target_material(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    stage = Usd.Stage.Open(source_usd.as_posix())
    assert stage is not None
    mesh = stage.GetPrimAtPath("/World/Asset/Body")
    material = UsdShade.Material.Define(stage, "/World/Looks/Existing")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    stage.GetRootLayer().Save()
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    output_dir = tmp_path / "conditioning"

    with pytest.raises(MaterialCreationError, match="already has a bound material"):
        prepare_material_conditioning(request, output_dir, options=options)

    assert not output_dir.exists()


def test_real_conditioning_rejects_st_cardinality_mismatch(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    stage = Usd.Stage.Open(source_usd.as_posix())
    assert stage is not None
    mesh = stage.GetPrimAtPath("/World/Asset/Body")
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    primvar.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    stage.GetRootLayer().Save()
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)

    with pytest.raises(MaterialCreationError, match="st cardinality"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("empty_st", "empty st primvar"),
        ("topology", "invalid topology"),
        ("point_indices", "invalid point indices"),
        ("interpolation", "unsupported st interpolation"),
        ("type", "must be a float2 array"),
    ],
)
def test_mesh_st_provenance_rejects_invalid_mesh_data(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    stage = Usd.Stage.Open(source_usd.as_posix())
    assert stage is not None
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Asset/Body"))
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    if case == "empty_st":
        primvar.Set([])
    elif case == "topology":
        mesh.GetFaceVertexCountsAttr().Set([3])
    elif case == "point_indices":
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 9])
    elif case == "interpolation":
        primvar.GetAttr().SetMetadata("interpolation", "unsupported")
    elif case == "type":
        mesh.GetPrim().RemoveProperty("primvars:st")
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            conditioning_module.Sdf.ValueTypeNames.Float3Array,
            UsdGeom.Tokens.faceVarying,
        )
        primvar.Set([(0.0, 0.0, 0.0)] * 4)
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(case)
    request = _request(source_usd, backend="step1x_material_anything")

    with pytest.raises(MaterialCreationError, match=expected_message):
        conditioning_module._collect_mesh_st_provenance(
            request,
            source_stage=stage,
            source_usd_sha256=_sha256(source_usd),
        )


def test_mesh_st_provenance_skips_non_meshes_and_duplicate_targets(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    stage = Usd.Stage.Open(source_usd.as_posix())
    assert stage is not None
    stage.RemovePrim("/World/Asset/Other")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/Asset", "/World/Asset/Body"),
    )

    provenance = conditioning_module._collect_mesh_st_provenance(
        request,
        source_stage=stage,
        source_usd_sha256=_sha256(source_usd),
    )

    assert [mesh["mesh_prim_path"] for mesh in provenance["meshes"]] == [
        "/World/Asset/Body"
    ]


def test_mesh_st_provenance_requires_a_target_mesh(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    stage = Usd.Stage.Open(source_usd.as_posix())
    assert stage is not None
    stage.DefinePrim("/World/Empty", "Xform")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/Empty",),
    )

    with pytest.raises(MaterialCreationError, match="at least one target mesh"):
        conditioning_module._collect_mesh_st_provenance(
            request,
            source_stage=stage,
            source_usd_sha256=_sha256(source_usd),
        )


def test_bind_seed_material_skips_non_meshes_and_rejects_existing_binding(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scoped.usda")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/Asset", "/World/Asset/Body"),
    )
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest = json.loads(
        options.real_evidence.seed_manifest_path.read_text(encoding="utf-8")
    )
    packaged_seed = (
        options.real_evidence.seed_manifest_path.parent
        / manifest["material_usd"]["path"]
    )
    seed_material = tmp_path / "seed_material.usda"
    seed_material.write_bytes(packaged_seed.read_bytes())

    material_path, bound_meshes = conditioning_module._bind_seed_material(
        request,
        scoped_usd_path=source_usd,
        seed_material_path=seed_material,
        seed_material_prim_path="/Seed/Material",
        package_id="123 seed",
    )

    assert material_path.endswith("/Material_123_seed")
    assert "/World/Asset/Body" in bound_meshes
    with pytest.raises(MaterialCreationError, match="already has a bound material"):
        conditioning_module._bind_seed_material(
            request,
            scoped_usd_path=source_usd,
            seed_material_path=seed_material,
            seed_material_prim_path="/Seed/Material",
            package_id="123 seed",
        )


def test_bind_seed_material_rejects_failed_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scoped.usda")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/Asset/Body",),
    )
    real_binding_api = conditioning_module.UsdShade.MaterialBindingAPI

    class FailingBindingAPI:
        def __init__(self, prim) -> None:
            self._delegate = real_binding_api(prim)

        def ComputeBoundMaterial(self):
            return self._delegate.ComputeBoundMaterial()

        @staticmethod
        def Apply(_prim):
            return SimpleNamespace(Bind=lambda _material: False)

    monkeypatch.setattr(
        conditioning_module.UsdShade,
        "MaterialBindingAPI",
        FailingBindingAPI,
    )

    with pytest.raises(MaterialCreationError, match="failed to bind seed material"):
        conditioning_module._bind_seed_material(
            request,
            scoped_usd_path=source_usd,
            seed_material_path=tmp_path / "seed_material.usda",
            seed_material_prim_path="/Seed/Material",
            package_id="seed",
        )


def test_real_conditioning_rejects_non_real_ovrtx_manifest(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.ovrtx_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["simulate"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MaterialCreationError, match="simulate mode"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_fake_renderer_request(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.ovrtx_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["request"]["renderer"]["backend"] = "fake"
    manifest["request_sha256"] = _json_sha256(manifest["request"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MaterialCreationError, match="disallowed 'fake' evidence"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_ovrtx_render_hash_mismatch(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.ovrtx_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    render_path = manifest_path.parent / manifest["artifacts"][0]["path"]
    render_path.write_bytes(b"changed after render")

    with pytest.raises(MaterialCreationError, match="OVRTX render front SHA-256"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_real_conditioning_rejects_missing_ovrtx_manifest(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    options = replace(
        options,
        real_evidence=replace(
            options.real_evidence,
            ovrtx_manifest_path=tmp_path / "missing-ovrtx-manifest.json",
        ),
    )

    with pytest.raises(MaterialCreationError, match="does not exist"):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("schema", "unsupported schema_version"),
        ("provider", "provider=ovrtx"),
        ("request_id", "request ID does not match"),
        ("simulate", "simulate=false"),
        ("request_sha256", "request SHA-256"),
        ("source_sha256", "source USD SHA-256"),
        ("seed_sha256", "seed manifest SHA-256"),
        ("target_paths", "target prim paths"),
        ("artifacts", "requires render artifacts"),
        ("artifact_kind", "must be render mappings"),
        ("artifact_source", "must be renderer_derived"),
        ("artifact_view", "unique safe names"),
        ("generic_value", "disallowed 'placeholder' evidence"),
        ("compound_value", "disallowed 'simulated' evidence"),
        ("dotted_value", "disallowed 'fake' evidence"),
        ("camel_value", "disallowed 'fake' evidence"),
        ("generic_flag", "disallowed evidence flag"),
        ("simulated_backend", "disallowed 'simulated' evidence"),
        ("renderer_backend", "renderer backend must be ovrtx"),
    ],
)
def test_real_conditioning_rejects_invalid_ovrtx_manifest_contract(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    assert options.real_evidence is not None
    manifest_path = options.real_evidence.ovrtx_manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if case == "schema":
        manifest["schema_version"] = "unsupported"
    elif case == "provider":
        manifest["provider"] = "renderer"
    elif case == "request_id":
        manifest["request_id"] = "another-request"
    elif case == "simulate":
        manifest.pop("simulate")
    elif case == "request_sha256":
        manifest["request_sha256"] = "0" * 64
    elif case == "source_sha256":
        manifest["request"]["source_usd_sha256"] = "0" * 64
    elif case == "seed_sha256":
        manifest["request"]["seed_manifest_sha256"] = "0" * 64
    elif case == "target_paths":
        manifest["request"]["target_prim_paths"] = ["/World/Other"]
    elif case == "artifacts":
        manifest["artifacts"] = []
    elif case == "artifact_kind":
        manifest["artifacts"][0]["kind"] = "normal"
    elif case == "artifact_source":
        manifest["artifacts"][0]["evidence_source"] = "external"
    elif case == "artifact_view":
        manifest["artifacts"][0]["view"] = "../front"
    elif case == "generic_value":
        manifest["request"]["mode"] = "placeholder"
    elif case == "compound_value":
        manifest["request"]["mode"] = "simulated render output"
    elif case == "dotted_value":
        manifest["request"]["mode"] = "fake.renderer"
    elif case == "camel_value":
        manifest["request"]["mode"] = "fakeRenderer"
    elif case == "generic_flag":
        manifest["request"]["mock"] = True
    elif case == "simulated_backend":
        manifest["request"]["renderer"]["backend"] = "simulated"
    elif case == "renderer_backend":
        manifest["request"]["renderer"]["backend"] = "blender"
    else:  # pragma: no cover - parameter list is exhaustive
        raise AssertionError(case)
    if case in {
        "source_sha256",
        "seed_sha256",
        "target_paths",
        "generic_value",
        "compound_value",
        "dotted_value",
        "camel_value",
        "generic_flag",
        "simulated_backend",
        "renderer_backend",
    }:
        manifest["request_sha256"] = _json_sha256(manifest["request"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MaterialCreationError, match=expected_message):
        prepare_material_conditioning(
            request,
            tmp_path / "conditioning",
            options=options,
        )


def test_usd_conditioning_helpers_reject_unopenable_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    monkeypatch.setattr(conditioning_module.Usd.Stage, "Open", lambda *_args: None)

    with pytest.raises(MaterialCreationError, match="scoped USD copy"):
        conditioning_module._bind_seed_material(
            request,
            scoped_usd_path=source_usd,
            seed_material_path=tmp_path / "seed.usda",
            seed_material_prim_path="/Seed/Material",
            package_id="seed",
        )
    with pytest.raises(MaterialCreationError, match="copied seed material"):
        conditioning_module._retarget_seed_material_albedo(
            request,
            material_usd_path=tmp_path / "seed.usda",
            material_prim_path="/Seed/Material",
            source_albedo_name="source.png",
        )
    with pytest.raises(MaterialCreationError, match="failed to open seed material"):
        conditioning_module._validate_seed_material_usd(
            request,
            material_usd_path=tmp_path / "seed.usda",
            material_prim_path="/Seed/Material",
            source_albedo_path=tmp_path / "source.png",
        )


def test_retarget_seed_material_requires_one_uv_texture(tmp_path: Path) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    seed_material = tmp_path / "seed.usda"
    seed_material.write_text(
        '#usda 1.0\ndef Material "Seed" {}\n',
        encoding="utf-8",
    )

    with pytest.raises(MaterialCreationError, match="exactly one"):
        conditioning_module._retarget_seed_material_albedo(
            request,
            material_usd_path=seed_material,
            material_prim_path="/Seed",
            source_albedo_name="source.png",
        )


@pytest.mark.parametrize(
    ("set_result", "save_error", "expected_message"),
    [
        (False, None, "failed to retarget"),
        (True, OSError("read-only layer"), "failed to save copied seed material"),
    ],
)
def test_retarget_seed_material_surfaces_usd_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    set_result: bool,
    save_error: OSError | None,
    expected_message: str,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")

    class FakeInput:
        def __bool__(self) -> bool:
            return True

        def Set(self, _value) -> bool:
            return set_result

    class FakePrim:
        def IsA(self, _schema) -> bool:
            return True

    class FakeShader:
        def GetIdAttr(self):
            return SimpleNamespace(Get=lambda: "UsdUVTexture")

        def GetInput(self, _name):
            return FakeInput()

    class FakeLayer:
        def Save(self) -> None:
            if save_error is not None:
                raise save_error

    class FakeStage:
        def GetPrimAtPath(self, _path):
            return FakePrim()

        def GetRootLayer(self):
            return FakeLayer()

    monkeypatch.setattr(
        conditioning_module.Usd.Stage,
        "Open",
        lambda *_args: FakeStage(),
    )
    monkeypatch.setattr(
        conditioning_module.Usd,
        "PrimRange",
        lambda _prim: (FakePrim(),),
    )
    monkeypatch.setattr(
        conditioning_module.UsdShade,
        "Shader",
        lambda _prim: FakeShader(),
    )

    with pytest.raises(MaterialCreationError, match=expected_message):
        conditioning_module._retarget_seed_material_albedo(
            request,
            material_usd_path=tmp_path / "seed.usda",
            material_prim_path="/Seed/Material",
            source_albedo_name="source.png",
        )


def test_copy_evidence_file_rejects_changed_copy_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    monkeypatch.setattr(conditioning_module, "_sha256_file", lambda _path: "0" * 64)

    with pytest.raises(MaterialCreationError, match="changed during conditioning"):
        conditioning_module._copy_evidence_file(
            request,
            source,
            tmp_path / "copied.bin",
            expected_sha256="1" * 64,
            label="test evidence",
        )


def test_real_evidence_mode_rejects_missing_artifact_provenance(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd)
    artifacts = (
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.SCOPED_USD,
            uri="scoped.usda",
            evidence_source=MaterialConditioningArtifactSource.SOURCE_DERIVED,
        ),
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.RENDER,
            uri="render.png",
            color_space=MaterialColorSpace.SRGB,
            view="front",
        ),
    )

    with pytest.raises(MaterialCreationError) as error:
        _reject_non_real_evidence_in_real_mode(
            request,
            artifacts=artifacts,
            evidence_mode=MaterialConditioningEvidenceMode.REAL_EVIDENCE,
        )

    assert error.value.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert "real_evidence" in str(error.value)
    assert "render" in str(error.value)


def test_prepare_material_conditioning_manifest_records_artifact_provenance(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    recipe_ref = _write_reference(tmp_path / "recipe_ref.png")
    request_ref = _write_reference(tmp_path / "request_ref.png")
    request = _request(
        source_usd,
        reference_image_uris=(recipe_ref.as_posix(),),
        request_reference_image_uris=(recipe_ref.as_posix(), request_ref.as_posix()),
    )

    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=MaterialConditioningOptions(
            image_size=2,
            render_views=("front",),
            include_normal=False,
            include_depth=False,
            include_segmentation=False,
        ),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    provenance_by_kind = {
        item["kind"]: item for item in manifest["artifact_provenance"]
    }
    assert (
        provenance_by_kind[MaterialConditioningKind.SCOPED_USD.value]["evidence_source"]
        == MaterialConditioningArtifactSource.SOURCE_DERIVED.value
    )
    assert (
        provenance_by_kind[MaterialConditioningKind.UV_LAYOUT.value]["evidence_source"]
        == MaterialConditioningArtifactSource.PLACEHOLDER.value
    )
    assert (
        provenance_by_kind[MaterialConditioningKind.RENDER.value]["evidence_source"]
        == MaterialConditioningArtifactSource.PLACEHOLDER.value
    )
    assert (
        provenance_by_kind[MaterialConditioningKind.SOURCE_ALBEDO.value][
            "evidence_source"
        ]
        == MaterialConditioningArtifactSource.PLACEHOLDER.value
    )
    reference_provenance = [
        item
        for item in manifest["artifact_provenance"]
        if item["kind"] == MaterialConditioningKind.REFERENCE_IMAGE.value
    ]
    assert request.effective_reference_image_uris == (
        recipe_ref.as_posix(),
        request_ref.as_posix(),
    )
    assert [item["uri"] for item in reference_provenance] == [
        recipe_ref.as_posix(),
        request_ref.as_posix(),
    ]
    assert [item["evidence_source"] for item in reference_provenance] == [
        MaterialConditioningArtifactSource.RECIPE_REFERENCE.value,
        MaterialConditioningArtifactSource.REQUEST_REFERENCE.value,
    ]
    assert all(
        item["evidence_mode"]
        == MaterialConditioningEvidenceMode.DETERMINISTIC_FIXTURE.value
        for item in manifest["artifact_provenance"]
    )
    render_artifact = next(
        artifact
        for artifact in manifest["conditioning"]["artifacts"]
        if artifact["kind"] == MaterialConditioningKind.RENDER.value
    )
    assert render_artifact["evidence_source"] == (
        MaterialConditioningArtifactSource.PLACEHOLDER.value
    )


def test_prepare_material_conditioning_respects_optional_artifact_settings(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, target_prim_paths=("/World/Asset",))

    default_result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=MaterialConditioningOptions(image_size=2),
    )
    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=MaterialConditioningOptions(
            image_size=2,
            render_views=("front", "side"),
            include_normal=False,
            include_depth=False,
            include_segmentation=False,
            include_source_albedo=False,
        ),
    )

    assert result.output_dir != default_result.output_dir
    assert default_result.output_dir.name.startswith(f"{request.request_id}-")
    assert result.output_dir.name.startswith(f"{request.request_id}-")
    kinds = [artifact.kind for artifact in result.conditioning.artifacts]
    assert kinds == [
        MaterialConditioningKind.SCOPED_USD,
        MaterialConditioningKind.UV_LAYOUT,
        MaterialConditioningKind.UV_MASK,
        MaterialConditioningKind.RENDER,
        MaterialConditioningKind.RENDER,
    ]
    assert [artifact.view for artifact in result.conditioning.artifacts[-2:]] == [
        "front",
        "side",
    ]
    assert result.metadata["target_scope"]["target_mesh_paths"] == [
        "/World/Asset/Body",
        "/World/Asset/Other",
    ]
    assert result.metadata["target_scope"]["all_target_meshes_have_uvs"] is False


def test_prepare_material_conditioning_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    missing_source = tmp_path / "missing.usda"

    with pytest.raises(MaterialCreationError) as missing_source_error:
        prepare_material_conditioning(_request(missing_source), tmp_path / "out")
    assert missing_source_error.value.code is MaterialCreationErrorCode.INVALID_REQUEST

    with pytest.raises(MaterialCreationError, match="source_usd_sha256"):
        prepare_material_conditioning(
            _request(source_usd, source_usd_sha256="0" * 64),
            tmp_path / "out",
        )

    with pytest.raises(MaterialCreationError) as missing_targets:
        prepare_material_conditioning(
            _request(
                source_usd,
                target_prim_paths=(
                    "/World/Asset/Missing",
                    "/World/Asset/AlsoMissing",
                ),
            ),
            tmp_path / "out",
        )
    assert "/World/Asset/Missing" in str(missing_targets.value)
    assert "/World/Asset/AlsoMissing" in str(missing_targets.value)


def test_prepare_material_conditioning_honors_preexisting_cancellation(
    tmp_path: Path,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    cancel_event = threading.Event()
    cancel_event.set()
    output_dir = tmp_path / "conditioning"

    with pytest.raises(MaterialCreationError) as error:
        prepare_material_conditioning(
            _request(source_usd),
            output_dir,
            cancel_event=cancel_event,
        )

    assert error.value.code is MaterialCreationErrorCode.CANCELLED
    assert not output_dir.exists()


def test_prepare_material_conditioning_cleans_partial_cancelled_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    cancel_event = threading.Event()
    output_dir = tmp_path / "conditioning"
    write_scoped_usd = conditioning_module._write_scoped_usd

    def write_then_cancel(*args, **kwargs) -> None:
        write_scoped_usd(*args, **kwargs)
        cancel_event.set()

    monkeypatch.setattr(
        conditioning_module,
        "_write_scoped_usd",
        write_then_cancel,
    )

    with pytest.raises(MaterialCreationError) as error:
        prepare_material_conditioning(
            _request(source_usd),
            output_dir,
            cancel_event=cancel_event,
        )

    assert error.value.code is MaterialCreationErrorCode.CANCELLED
    assert output_dir.is_dir()
    assert not tuple(output_dir.iterdir())


def test_real_conditioning_cancels_during_preflight_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_usd = _write_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, backend="step1x_material_anything")
    options = _write_real_conditioning_inputs(tmp_path, request)
    cancel_event = threading.Event()
    output_dir = tmp_path / "conditioning"
    load_seed = conditioning_module._load_seed_material_package

    def load_then_cancel(*args, **kwargs):
        seed = load_seed(*args, **kwargs)
        cancel_event.set()
        return seed

    monkeypatch.setattr(
        conditioning_module,
        "_load_seed_material_package",
        load_then_cancel,
    )

    with pytest.raises(MaterialCreationError) as error:
        prepare_material_conditioning(
            request,
            output_dir,
            options=options,
            cancel_event=cancel_event,
        )

    assert error.value.code is MaterialCreationErrorCode.CANCELLED
    assert not output_dir.exists()


def test_conditioning_options_validate_values() -> None:
    with pytest.raises(ValueError, match="image_size"):
        MaterialConditioningOptions(image_size=0)
    with pytest.raises(ValueError, match="at least one"):
        MaterialConditioningOptions(render_views=())
    with pytest.raises(ValueError, match="non-empty"):
        MaterialConditioningOptions(render_views=("oblique", " "))
    with pytest.raises(ValueError, match="safe filename"):
        MaterialConditioningOptions(render_views=("front", "../side"))
    with pytest.raises(ValueError, match="duplicate"):
        MaterialConditioningOptions(render_views=("oblique", "oblique"))
    assert (
        MaterialConditioningOptions(
            render_views=(),
            evidence_mode=MaterialConditioningEvidenceMode.REAL_EVIDENCE,
        ).render_views
        == ()
    )
    assert MaterialConditioningOptions.from_dict(
        {"render_views": "front"}
    ).render_views == ("front",)
    with pytest.raises(TypeError, match="real_evidence"):
        MaterialConditioningOptions.from_dict({"real_evidence": []})
    with pytest.raises(TypeError, match="render_views"):
        MaterialConditioningOptions.from_dict({"render_views": 1})


def test_conditioning_options_resolve_relative_real_evidence_paths(
    tmp_path: Path,
) -> None:
    options = MaterialConditioningOptions.from_dict(
        {
            "evidence_mode": "real_evidence",
            "real_evidence": {
                "seed_manifest_path": "seed_manifest.json",
                "seed_manifest_sha256": "a" * 64,
                "ovrtx_manifest_path": "ovrtx_manifest.json",
            },
        },
        base_dir=tmp_path,
    )

    assert options.real_evidence is not None
    assert (
        options.real_evidence.seed_manifest_path
        == (tmp_path / "seed_manifest.json").resolve()
    )
    assert (
        options.real_evidence.ovrtx_manifest_path
        == (tmp_path / "ovrtx_manifest.json").resolve()
    )


def test_prepare_material_conditioning_deinstances_before_visibility(
    tmp_path: Path,
) -> None:
    source_usd = _write_instanced_source_usd(tmp_path / "scene.usda")
    request = _request(source_usd, target_prim_paths=("/World/AssetInstance/Body",))

    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=MaterialConditioningOptions(image_size=2),
    )

    scoped_stage = Usd.Stage.Open(result.scoped_usd_path.as_posix())
    assert scoped_stage is not None
    instance_root = scoped_stage.GetPrimAtPath("/World/AssetInstance")
    target = UsdGeom.Imageable(scoped_stage.GetPrimAtPath("/World/AssetInstance/Body"))
    sibling = UsdGeom.Imageable(
        scoped_stage.GetPrimAtPath("/World/AssetInstance/Other")
    )
    assert instance_root.IsInstanceable() is False
    assert target.ComputeVisibility() == UsdGeom.Tokens.inherited
    assert sibling.ComputeVisibility() == UsdGeom.Tokens.invisible


def test_real_conditioning_collects_mesh_st_from_instance_root(
    tmp_path: Path,
) -> None:
    source_usd = _write_instanced_source_usd(tmp_path / "scene.usda")
    source_stage = Usd.Stage.Open(source_usd.as_posix())
    assert source_stage is not None
    assert source_stage.RemovePrim("/World/Prototype/Other")
    source_stage.GetRootLayer().Save()
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/AssetInstance",),
    )
    options = _write_real_conditioning_inputs(tmp_path, request)

    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=options,
    )

    assert result.metadata["target_scope"]["target_mesh_paths"] == [
        "/World/AssetInstance/Body"
    ]
    assert (
        result.metadata["real_evidence"]["mesh_st"]["meshes"][0]["mesh_prim_path"]
        == "/World/AssetInstance/Body"
    )
    scoped_stage = Usd.Stage.Open(result.scoped_usd_path.as_posix())
    assert scoped_stage is not None
    material, _ = UsdShade.MaterialBindingAPI(
        scoped_stage.GetPrimAtPath("/World/AssetInstance/Body")
    ).ComputeBoundMaterial()
    assert material


def test_real_conditioning_deinstances_nested_target_mesh(
    tmp_path: Path,
) -> None:
    source_usd = _write_nested_instanced_source_usd(tmp_path / "scene.usda")
    request = _request(
        source_usd,
        backend="step1x_material_anything",
        target_prim_paths=("/World/AssetInstance",),
    )
    options = _write_real_conditioning_inputs(tmp_path, request)

    result = prepare_material_conditioning(
        request,
        tmp_path / "conditioning",
        options=options,
    )

    scoped_stage = Usd.Stage.Open(result.scoped_usd_path.as_posix())
    assert scoped_stage is not None
    assert not scoped_stage.GetPrimAtPath("/World/AssetInstance").IsInstanceable()
    nested = scoped_stage.GetPrimAtPath("/World/AssetInstance/NestedInstance")
    assert not nested.IsInstanceable()
    body = scoped_stage.GetPrimAtPath("/World/AssetInstance/NestedInstance/Body")
    material, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial()
    assert material
    assert (
        result.metadata["real_evidence"]["mesh_st"]["meshes"][0]["mesh_prim_path"]
        == "/World/AssetInstance/NestedInstance/Body"
    )


def _write_real_conditioning_inputs(
    tmp_path: Path,
    request: CreateMaterialRequest,
) -> MaterialConditioningOptions:
    seed_dir = tmp_path / "seed_package"
    texture_dir = seed_dir / "textures"
    texture_dir.mkdir(parents=True)
    source_albedo = texture_dir / "real_source_albedo.png"
    source_image = Image.new("RGB", (2, 2), (120, 130, 140))
    source_image.putpixel((1, 1), (121, 130, 140))
    source_image.save(source_albedo)
    material_usd = seed_dir / "seed_material.usda"
    material_usd.write_text(
        """#usda 1.0
(
    defaultPrim = "Seed"
)
def Scope "Seed"
{
    def Material "Material"
    {
        token outputs:surface.connect = </Seed/Material/Surface.outputs:surface>
        def Shader "UVReader"
        {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
        def Shader "Albedo"
        {
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @textures/real_source_albedo.png@
            float2 inputs:st.connect = </Seed/Material/UVReader.outputs:result>
            float3 outputs:rgb
        }
        def Shader "Surface"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor.connect = </Seed/Material/Albedo.outputs:rgb>
            float inputs:metallic = 1
            float inputs:roughness = 0.4
            token outputs:surface
        }
    }
}
""",
        encoding="utf-8",
    )
    seed_manifest = seed_dir / "seed_manifest.json"
    seed_manifest.write_text(
        json.dumps(
            {
                "schema_version": REAL_SEED_MATERIAL_SCHEMA_VERSION,
                "package_id": "temporary-real-input-test-fixture",
                "package_revision": "test-only-revision",
                "material_usd": {
                    "path": material_usd.name,
                    "sha256": _sha256(material_usd),
                    "material_prim_path": "/Seed/Material",
                },
                "source_albedo": {
                    "path": "textures/real_source_albedo.png",
                    "sha256": _sha256(source_albedo),
                },
                "source": {
                    "kind": "approved_s3",
                    "uri": "s3://test-only/real-source-albedo.png",
                    "etag": "test-only-etag",
                    "last_modified": "2026-06-30T00:00:00Z",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    ovrtx_manifest = _write_ovrtx_manifest(
        tmp_path,
        request,
        seed_manifest_sha256=_sha256(seed_manifest),
    )
    return MaterialConditioningOptions(
        image_size=4,
        render_views=(),
        include_normal=False,
        include_depth=False,
        include_segmentation=False,
        include_source_albedo=True,
        evidence_mode=MaterialConditioningEvidenceMode.REAL_EVIDENCE,
        real_evidence=RealMaterialConditioningInputs(
            seed_manifest_path=seed_manifest,
            seed_manifest_sha256=_sha256(seed_manifest),
            ovrtx_manifest_path=ovrtx_manifest,
        ),
    )


def _write_ovrtx_manifest(
    tmp_path: Path,
    request: CreateMaterialRequest,
    *,
    seed_manifest_sha256: str,
) -> Path:
    ovrtx_dir = tmp_path / "ovrtx_evidence"
    ovrtx_dir.mkdir(exist_ok=True)
    render = ovrtx_dir / "front.png"
    Image.new("RGB", (2, 2), (90, 100, 110)).save(render)
    request_data = {
        "source_usd_sha256": _sha256(request.source_usd),
        "seed_manifest_sha256": seed_manifest_sha256,
        "target_prim_paths": list(request.target_prim_paths),
        "renderer": {"backend": "ovrtx", "image_width": 512, "image_height": 512},
    }
    manifest = ovrtx_dir / "request_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": OVRTX_CONDITIONING_SCHEMA_VERSION,
                "provider": "ovrtx",
                "provider_revision": "temporary-test-fixture",
                "request_id": request.request_id,
                "simulate": False,
                "request": request_data,
                "request_sha256": _json_sha256(request_data),
                "artifacts": [
                    {
                        "kind": "render",
                        "path": render.name,
                        "view": "front",
                        "sha256": _sha256(render),
                        "evidence_source": "renderer_derived",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def _request(
    source_usd: Path,
    *,
    target_prim_paths: tuple[str, ...] = ("/World/Asset/Body",),
    reference_image_uris: tuple[str, ...] = (),
    request_reference_image_uris: tuple[str, ...] = (),
    source_usd_sha256: str | None = None,
    backend: str = "fake",
    texture_size: int = 1024,
) -> CreateMaterialRequest:
    recipe = MaterialRecipe(
        name="Matte blue plastic",
        description="Opaque matte blue plastic.",
        appearance_prompt="matte blue plastic with fine molded texture",
        color="blue",
        material="plastic",
        finish="matte",
        base_color_hint=(0.05, 0.18, 0.62),
        pbr_hints=PBRHints(roughness=0.72, metallic=0.0),
        reference_image_uris=reference_image_uris,
        intended_parts=(
            IntendedPart(
                semantic_label="body",
                evidence="unit test target",
                prim_path_hints=target_prim_paths,
            ),
        ),
    )
    return CreateMaterialRequest(
        source_usd=source_usd.resolve(),
        source_usd_sha256=source_usd_sha256,
        target_prim_paths=target_prim_paths,
        recipe=recipe,
        reference_image_uris=request_reference_image_uris,
        backend=backend,
        texture_size=texture_size,
        seed=482,
    )


def _write_source_usd(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Xform "Asset"
    {
        def Mesh "Body"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0)
            ]
            texCoord2f[] primvars:st = [
                (0, 0), (1, 0), (1, 1), (0, 1)
            ] (
                interpolation = "faceVarying"
            )
        }
        def Mesh "Other"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [
                (-1, -1, 0.0),
                (-0.75, -1, 0.0),
                (-0.75, -0.75, 0.0),
                (-1, -0.75, 0.0)
            ]
        }
    }
}
""",
        encoding="utf-8",
    )
    return path


def _write_instanced_source_usd(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Xform "Prototype"
    {
        def Mesh "Body"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0)
            ]
            texCoord2f[] primvars:st = [
                (0, 0), (1, 0), (1, 1), (0, 1)
            ] (
                interpolation = "faceVarying"
            )
        }
        def Mesh "Other"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [
                (-1, -1, 0.0),
                (-0.75, -1, 0.0),
                (-0.75, -0.75, 0.0),
                (-1, -0.75, 0.0)
            ]
        }
    }
    def Xform "AssetInstance" (
        prepend references = </World/Prototype>
        instanceable = true
    )
    {
    }
}
""",
        encoding="utf-8",
    )
    return path


def _write_nested_instanced_source_usd(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Xform "LeafPrototype"
    {
        def Mesh "Body"
        {
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0)
            ]
            texCoord2f[] primvars:st = [
                (0, 0), (1, 0), (1, 1), (0, 1)
            ] (
                interpolation = "faceVarying"
            )
        }
    }
    def Xform "ContainerPrototype"
    {
        def Xform "NestedInstance" (
            prepend references = </World/LeafPrototype>
            instanceable = true
        )
        {
        }
    }
    def Xform "AssetInstance" (
        prepend references = </World/ContainerPrototype>
        instanceable = true
    )
    {
    }
}
""",
        encoding="utf-8",
    )
    return path


def _write_reference(path: Path) -> Path:
    path.write_bytes(b"reference-image")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
