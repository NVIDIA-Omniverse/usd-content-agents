# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryAuthoringProviderIdentity,
    GeometryCoordinateSystem,
    GeometryRepresentationBinding,
    GeometryRightsAssertion,
    GeometrySourceBundle,
    GeometrySourceProvenance,
    geometry_source_bundle_id,
)

from content_agent_workflows.geometry import (
    GeometryWorkflowInput,
    route_geometry_request,
    run_geometry_workflow,
)
from content_agent_workflows.geometry import source_prep as geometry_source_prep
from content_agent_workflows.geometry import workflow as geometry_workflow


def _tiny_usda(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Asset"
{
    def Mesh "male_block"
    {
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 4, 0]
        point3f[] points = [(-0.01, -0.01, -0.01), (0.01, -0.01, -0.01), (0.01, 0.01, -0.01), (-0.01, 0.01, -0.01), (-0.01, -0.01, 0.01), (0.01, -0.01, 0.01), (0.01, 0.01, 0.01), (-0.01, 0.01, 0.01)]
    }
}
""",
        encoding="utf-8",
    )
    return path


def _source_bundle_manifest(
    manifest: Path,
    artifact: Path,
    *,
    artifact_path: str | None = None,
    sha256: str | None = None,
    role: str = "render_geometry",
    representation_id: str = "render-usd",
    representation_format: str = "usda",
    media_type: str = "model/vnd.usda",
    coordinate_system: GeometryCoordinateSystem | None = None,
    additional_representations: tuple[GeometryRepresentationBinding, ...] = (),
    provenance_input_artifacts: tuple[GeometryArtifactBinding, ...] = (),
) -> Path:
    binding = GeometryArtifactBinding(
        path=artifact_path or artifact.name,
        sha256=sha256 or hashlib.sha256(artifact.read_bytes()).hexdigest(),
        size_bytes=artifact.stat().st_size,
    )
    producer = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    coordinate_system = coordinate_system or GeometryCoordinateSystem(
        meters_per_unit=1.0, up_axis="Z", forward_axis="+Y"
    )
    representations = (
        GeometryRepresentationBinding(
            representation_id=representation_id,
            role=role,
            format=representation_format,
            media_type=media_type,
            artifact=binding,
        ),
        *additional_representations,
    )
    provenance = GeometrySourceProvenance(
        request_digest="a" * 64,
        input_artifacts=provenance_input_artifacts,
    )
    rights = GeometryRightsAssertion(
        assertion="Authorized fixture for workflow validation."
    )
    bundle = GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=producer,
            source_revision="immutable-revision-1",
            coordinate_system=coordinate_system,
            representations=representations,
            provenance=provenance,
            rights=rights,
        ),
        producer=producer,
        source_revision="immutable-revision-1",
        coordinate_system=coordinate_system,
        representations=representations,
        provenance=provenance,
        rights=rights,
    )
    manifest.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def _repair_cube_usda(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Asset"
{
    def Mesh "cube"
    {
        uniform token subdivisionScheme = "none"
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        int[] faceVertexCounts = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        int[] faceVertexIndices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
        point3f[] points = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]
    }
}
""",
        encoding="utf-8",
    )
    return path


def _tiny_3mf(path: Path) -> Path:
    model = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter"
  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
  xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <resources>
    <m:colorgroup id="1">
      <m:color color="#c77a35" />
      <m:color color="#2d3338" />
    </m:colorgroup>
    <object id="2" name="rotor copper winding" pid="1" pindex="0">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0" />
          <vertex x="10" y="0" z="0" />
          <vertex x="0" y="10" z="0" />
        </vertices>
        <triangles><triangle v1="0" v2="1" v3="2" /></triangles>
      </mesh>
    </object>
    <object id="3" name="steel shaft pin" pid="1" pindex="1">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="1" />
          <vertex x="5" y="0" z="1" />
          <vertex x="0" y="5" z="1" />
        </vertices>
        <triangles><triangle v1="0" v2="1" v3="2" /></triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="2" />
    <item objectid="3" />
  </build>
</model>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("3D/3dmodel.model", model)
    return path


def _component_3mf(path: Path) -> Path:
    model = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter"
  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="2" name="triangle tooth">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0" />
          <vertex x="1" y="0" z="0" />
          <vertex x="0" y="1" z="0" />
        </vertices>
        <triangles><triangle v1="0" v2="1" v3="2" /></triangles>
      </mesh>
    </object>
    <object id="10" name="translated assembly">
      <components>
        <component objectid="2" transform="1 0 0 0 1 0 0 0 1 10 0 0" />
      </components>
    </object>
  </resources>
  <build>
    <item objectid="10" transform="1 0 0 0 1 0 0 0 1 0 20 0" />
  </build>
</model>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("3D/3dmodel.model", model)
    return path


def test_geometry_policy_routes_provided_cad_and_catalog_text() -> None:
    provided = route_geometry_request(source_path="part.step")
    assert provided.route == "provided_cad_repair"
    assert provided.source_category == "convertible_external"
    assert not provided.requires_generation

    retired_source = route_geometry_request(source_path="fixture.legacy-cad-source")
    assert retired_source.route == "provided_cad_repair"
    assert retired_source.input_modality == "provided_cad"
    assert retired_source.source_category == "unsupported_external"
    assert not retired_source.requires_generation

    forgejs = route_geometry_request(source_path="motor_assembly.forge.js")
    assert forgejs.route == "provided_cad_repair"
    assert forgejs.input_modality == "provided_cad"
    assert forgejs.source_category == "unsupported_external"
    assert not forgejs.requires_generation

    generated = route_geometry_request(generated_usd_path="generated.usda")
    assert generated.route == "provided_cad_repair"
    assert generated.input_modality == "provided_cad"
    assert generated.source_category == "generated_usd"
    assert not generated.requires_generation

    generated_with_source = route_geometry_request(
        source_path="scan.stl",
        generated_usd_path="generated.usda",
    )
    assert generated_with_source.route == "provided_cad_repair"
    assert generated_with_source.input_modality == "provided_cad"
    assert generated_with_source.source_path == "generated.usda"
    assert not generated_with_source.requires_generation

    catalog = route_geometry_request(prompt="Generate McMaster part number 2584N111")
    assert catalog.route == "class_backed_asset_gen"
    assert catalog.source_category == "class_backed_generation"
    assert catalog.optional_bridge == "asset_gen"

    generic_3mf = route_geometry_request(source_path="part.3mf")
    assert generic_3mf.source_category == "convertible_external"


@pytest.mark.parametrize(
    ("filename", "expected_category"),
    [
        ("part.forge.js", "unsupported_external"),
        ("part.usd", "existing_usd"),
        ("part.usdz", "existing_usd"),
        ("part.step", "convertible_external"),
        ("part.glb", "convertible_external"),
        ("part.fbx", "convertible_external"),
        ("part.3mf", "convertible_external"),
        ("part.urdf", "convertible_external"),
        ("part.mjcf", "convertible_external"),
        ("part.stl", "mesh_recovery_candidate"),
        ("part.obj", "mesh_recovery_candidate"),
    ],
)
def test_geometry_policy_source_categories_match_shared_conversion_boundary(
    filename: str,
    expected_category: str,
) -> None:
    decision = route_geometry_request(source_path=filename)

    assert decision.source_category == expected_category
    assert decision.requires_generation is False


def test_geometry_policy_routes_mujoco_xml_through_shared_conversion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model.xml"
    source.write_text(
        '<mujoco model="fixture"><worldbody /></mujoco>\n', encoding="utf-8"
    )

    decision = route_geometry_request(source_path=source)

    assert decision.source_category == "convertible_external"
    assert decision.requires_generation is False


def test_geometry_policy_does_not_relabel_unknown_source_as_generation() -> None:
    decision = route_geometry_request(
        source_path="part.unknown-cad",
        prompt="Preserve this supplied part",
    )

    assert decision.route == "provided_cad_repair"
    assert decision.source_category == "unsupported_external"
    assert decision.requires_generation is False
    assert decision.source_path == "part.unknown-cad"


def test_geometry_policy_rejects_non_usd_generated_usd_artifact() -> None:
    decision = route_geometry_request(generated_usd_path="generated.glb")

    assert decision.route == "provided_cad_repair"
    assert decision.source_category == "unsupported_external"
    assert decision.requires_generation is False


def test_source_bundle_preparation_preserves_provider_metadata(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.is_prepared
    assert prepared.prepared_usd_path is not None
    prepared_source = Path(prepared.prepared_usd_path)
    assert prepared_source != source.resolve()
    assert prepared_source.read_bytes() == source.read_bytes()
    assert prepared_source.is_relative_to(
        (tmp_path / "out" / "source_prep" / "admitted_bundles").resolve()
    )
    expected_bundle = json.loads(manifest.read_text(encoding="utf-8"))
    assert prepared.metadata["source_bundle"] == {
        "schema_version": "geometry.source.v1",
        "bundle_id": expected_bundle["bundle_id"],
        "producer": {
            "provider_id": "fixture-provider",
            "provider_version": "1.0",
        },
        "source_revision": "immutable-revision-1",
        "coordinate_system": {
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "forward_axis": "+Y",
            "handedness": "right",
        },
        "selected_representation": {
            "representation_id": "render-usd",
            "role": "render_geometry",
            "format": "usda",
            "media_type": "model/vnd.usda",
            "artifact": {
                "path": "provider-output.usda",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size_bytes": source.stat().st_size,
            },
            "root_prim_path": None,
        },
        "representations": [
            {
                "representation_id": "render-usd",
                "role": "render_geometry",
                "format": "usda",
                "media_type": "model/vnd.usda",
                "artifact": {
                    "path": "provider-output.usda",
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "size_bytes": source.stat().st_size,
                },
                "root_prim_path": None,
                "materialized_path": str(prepared_source),
            }
        ],
        "parts": [],
        "parameters": [],
        "verification_assertions": [],
        "provenance": {
            "request_digest": "a" * 64,
            "parent_bundle_id": None,
            "input_artifacts": [],
            "upstream_edit_uri": None,
        },
        "provenance_input_artifacts": [],
        "rights": {
            "authorized_for_requested_use": True,
            "assertion": "Authorized fixture for workflow validation.",
            "license_identifier": None,
        },
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def test_source_bundle_snapshots_digest_bound_provenance_inputs(
    tmp_path: Path,
) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    reference = tmp_path / "reference.png"
    reference_bytes = b"immutable image input"
    reference.write_bytes(reference_bytes)
    reference_binding = GeometryArtifactBinding(
        path=reference.name,
        sha256=hashlib.sha256(reference_bytes).hexdigest(),
        size_bytes=len(reference_bytes),
    )
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        provenance_input_artifacts=(reference_binding,),
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.is_prepared
    records = prepared.metadata["source_bundle"]["provenance_input_artifacts"]
    assert len(records) == 1
    snapshot = Path(records[0]["materialized_path"])
    assert snapshot.is_relative_to(
        (tmp_path / "out" / "source_prep" / "admitted_bundles").resolve()
    )
    assert snapshot.read_bytes() == reference_bytes
    reference.write_bytes(b"changed after admission")
    assert snapshot.read_bytes() == reference_bytes
    evidence = geometry_workflow._source_record_artifacts(prepared)
    assert any(
        item.kind == "source_provenance_input" and Path(item.path) == snapshot
        for item in evidence
    )


@pytest.mark.parametrize("failure", ("missing", "digest_drift"))
def test_source_bundle_rejects_unverifiable_provenance_inputs(
    tmp_path: Path,
    failure: str,
) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    reference = tmp_path / "reference.png"
    if failure == "digest_drift":
        reference.write_bytes(b"current bytes")
    binding = GeometryArtifactBinding(
        path=reference.name,
        sha256=hashlib.sha256(b"declared bytes").hexdigest(),
        size_bytes=len(b"declared bytes"),
    )
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        provenance_input_artifacts=(binding,),
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "provenance input" in prepared.errors[0]


def test_direct_provider_usd_retains_declared_axes_on_workflow_output(
    tmp_path: Path,
) -> None:
    source = _tiny_usda(tmp_path / "provider-y-up.usda")
    source.write_text(
        source.read_text(encoding="utf-8").replace('upAxis = "Z"', 'upAxis = "Y"'),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    declared = GeometryCoordinateSystem(
        meters_per_unit=1.0,
        up_axis="Y",
        forward_axis="+Z",
    )
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        coordinate_system=declared,
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            source_manifest_path=manifest,
            expected_source_sha256=source_sha256,
            expected_source_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            output_dir=tmp_path / "out",
            optimization_policy="skip",
            canonicalize_stage_metrics=True,
            run_runtime_validation=False,
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    from pxr import Usd

    assert result.optimized_usd_path is not None, result.error
    output = Path(result.optimized_usd_path)
    assert output.resolve() != source.resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256
    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadNone)
    assert stage is not None
    assert stage.GetRootLayer().customLayerData[
        "geometrySourceCoordinateSystem"
    ] == declared.model_dump(mode="json")
    canonical = {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "forward_axis": "-Y",
        "handedness": "right",
    }
    assert (
        stage.GetRootLayer().customLayerData["geometryCanonicalCoordinateSystem"]
        == canonical
    )
    optimization = json.loads(
        Path(result.optimization_metadata_path or "").read_text(encoding="utf-8")
    )
    retention = optimization["source_coordinate_system_retention"]
    assert retention["status"] == "pass"
    assert retention["immutable_source_sha256"] == source_sha256
    assert retention["coordinate_system"] == declared.model_dump(mode="json")
    assert Path(retention["report_path"]).is_file()
    assert (
        optimization["stage_metric_normalization"]["canonical_coordinate_system"]
        == canonical
    )


def test_declared_coordinates_override_unitless_converter_assumptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "provider-output.stl"
    source.write_bytes(b"solid fixture\nendsolid fixture\n")
    converted = _tiny_usda(tmp_path / "converted.usda")
    declared = GeometryCoordinateSystem(
        meters_per_unit=0.001,
        up_axis="Y",
        forward_axis="+Z",
    )
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        coordinate_system=declared,
    )

    def fake_conversion(
        captured_source: Path,
        _prep_dir: Path,
        *,
        install_missing: bool,
        timeout_s: float,
    ) -> geometry_source_prep.PreparedGeometrySource:
        assert captured_source != source
        assert captured_source.read_bytes() == source.read_bytes()
        assert install_missing is False
        assert timeout_s == 120.0
        return geometry_source_prep.PreparedGeometrySource(
            fidelity_tier="shared_conversion_noneditable",
            source_authoring_mode="shared_conversion",
            source_path=str(captured_source),
            prepared_usd_path=str(converted),
            original_input_path=str(captured_source),
        )

    monkeypatch.setattr(
        geometry_source_prep,
        "_prepare_shared_conversion",
        fake_conversion,
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    from pxr import Usd, UsdGeom

    assert prepared.is_prepared
    stage = Usd.Stage.Open(str(converted), load=Usd.Stage.LoadNone)
    assert stage is not None
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Y"
    assert float(UsdGeom.GetStageMetersPerUnit(stage)) == 0.001
    assert stage.GetRootLayer().customLayerData["geometrySourceCoordinateSystem"] == (
        declared.model_dump(mode="json")
    )


def test_declared_coordinates_override_unitless_ply_converter_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "provider-output.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    converted = _tiny_usda(tmp_path / "converted.usda")
    prepared = geometry_source_prep.PreparedGeometrySource(
        fidelity_tier="shared_conversion_noneditable",
        source_authoring_mode="shared_conversion",
        source_path=str(source),
        prepared_usd_path=str(converted),
        original_input_path=str(source),
    )
    declared = GeometryCoordinateSystem(
        meters_per_unit=0.001,
        up_axis="Y",
        forward_axis="+Z",
    )

    geometry_source_prep._apply_declared_coordinate_system(
        prepared,
        coordinate_system=declared,
        source_path=source,
        source_suffix=".ply",
    )

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(converted), load=Usd.Stage.LoadNone)
    assert stage is not None
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Y"
    assert float(UsdGeom.GetStageMetersPerUnit(stage)) == 0.001
    assert prepared.metadata["coordinate_system_application"]["unitless_source"] is True


def _write_step_box(path: Path, *, size_mm: float = 20.0) -> Path:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(
        BRepPrimAPI_MakeBox(size_mm, size_mm, size_mm).Shape(),
        STEPControl_AsIs,
    )
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def test_converted_brep_corrects_unit_metadata_only_after_bounds_match(
    tmp_path: Path,
) -> None:
    source = _write_step_box(tmp_path / "provider-output.step")
    converted = _tiny_usda(tmp_path / "converted.usda")
    converted.write_text(
        converted.read_text().replace("metersPerUnit = 1", "metersPerUnit = 0.001"),
        encoding="utf-8",
    )
    prepared = geometry_source_prep.PreparedGeometrySource(
        fidelity_tier="shared_conversion_noneditable",
        source_authoring_mode="shared_conversion",
        source_path=str(source),
        prepared_usd_path=str(converted),
        original_input_path=str(source),
    )

    geometry_source_prep._apply_declared_coordinate_system(
        prepared,
        coordinate_system=GeometryCoordinateSystem(
            meters_per_unit=1.0,
            up_axis="Z",
            forward_axis="+Y",
        ),
        source_path=source,
        source_suffix=".step",
    )

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(converted), load=Usd.Stage.LoadNone)
    assert stage is not None
    assert float(UsdGeom.GetStageMetersPerUnit(stage)) == 1.0
    evidence = prepared.metadata["coordinate_system_application"]
    assert evidence["converter_metadata_corrected"] is True
    assert evidence["native_bounds_evidence"]["status"] == "matched"
    assert "native bounds verification" in prepared.warnings[-1]


def test_converted_brep_rejects_unit_metadata_when_bounds_do_not_match(
    tmp_path: Path,
) -> None:
    source = _write_step_box(tmp_path / "provider-output.step", size_mm=200.0)
    converted = _tiny_usda(tmp_path / "converted.usda")
    converted.write_text(
        converted.read_text().replace("metersPerUnit = 1", "metersPerUnit = 0.001"),
        encoding="utf-8",
    )
    prepared = geometry_source_prep.PreparedGeometrySource(
        fidelity_tier="shared_conversion_noneditable",
        source_authoring_mode="shared_conversion",
        source_path=str(source),
        prepared_usd_path=str(converted),
        original_input_path=str(source),
    )

    with pytest.raises(ValueError, match="native bounds do not prove"):
        geometry_source_prep._apply_declared_coordinate_system(
            prepared,
            coordinate_system=GeometryCoordinateSystem(
                meters_per_unit=1.0,
                up_axis="Z",
                forward_axis="+Y",
            ),
            source_path=source,
            source_suffix=".step",
        )


def test_provider_usd_rejects_coordinate_metadata_drift(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        coordinate_system=GeometryCoordinateSystem(
            meters_per_unit=0.001,
            up_axis="Z",
            forward_axis="+Y",
        ),
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == (
        "geometry_source_coordinate_system_invalid"
    )
    assert "metersPerUnit contradicts" in prepared.errors[0]


def test_provider_usd_rejects_retained_axis_drift(tmp_path: Path) -> None:
    from pxr import Usd

    source = _tiny_usda(tmp_path / "provider-output.usda")
    stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    assert stage is not None
    custom_data = dict(stage.GetRootLayer().customLayerData)
    custom_data["geometrySourceCoordinateSystem"] = {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "forward_axis": "-Y",
        "handedness": "right",
    }
    stage.GetRootLayer().customLayerData = custom_data
    assert stage.GetRootLayer().Save()
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        coordinate_system=GeometryCoordinateSystem(
            meters_per_unit=1.0,
            up_axis="Z",
            forward_axis="+Y",
        ),
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == (
        "geometry_source_coordinate_system_invalid"
    )
    assert "geometrySourceCoordinateSystem contradicts" in prepared.errors[0]


@pytest.mark.parametrize(
    ("representation_id", "representation_role"),
    (("render-usd", None), (None, "native_source")),
)
def test_source_bundle_never_executes_explicit_native_source_selection(
    tmp_path: Path,
    representation_id: str | None,
    representation_role: str | None,
) -> None:
    source = _tiny_usda(tmp_path / "provider-native-source.usda")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        role="native_source",
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        source_representation_id=representation_id,
        source_representation_role=representation_role,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "never executed" in prepared.errors[0]


def test_source_bundle_rejects_representation_path_escape(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    outside = _tiny_usda(tmp_path / "outside.usda")
    manifest = _source_bundle_manifest(
        bundle_dir / "geometry.source.json",
        outside,
        artifact_path="../outside.usda",
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "contained regular file" in prepared.errors[0]


def test_source_bundle_rejects_symlinked_representation(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    alias = tmp_path / "provider-alias.usda"
    alias.symlink_to(source)
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        artifact_path=alias.name,
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "contained regular file" in prepared.errors[0]


def test_source_bundle_rejects_symlinked_manifest(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
    )
    alias = tmp_path / "geometry.source.alias.json"
    alias.symlink_to(manifest)

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=alias,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "must not traverse a symlink" in prepared.errors[0]


def test_source_bundle_rejects_artifact_digest_drift(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "provider-output.usda")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        sha256="0" * 64,
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "SHA-256 changed" in prepared.errors[0]


@pytest.mark.parametrize("dependency", ("../outside.usda", "unlisted.usda"))
def test_source_bundle_rejects_unbound_usd_dependency(
    tmp_path: Path,
    dependency: str,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    _tiny_usda(tmp_path / "outside.usda")
    _tiny_usda(bundle_dir / "unlisted.usda")
    source = bundle_dir / "root.usda"
    source.write_text(
        f"""#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [@{dependency}@]
)

def Xform "Asset" {{}}
""",
        encoding="utf-8",
    )
    manifest = _source_bundle_manifest(
        bundle_dir / "geometry.source.json",
        source,
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "dependency" in prepared.errors[0]


def test_source_bundle_rejects_unbound_udim_tile(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = bundle_dir / "root.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Asset"
{
    custom asset texture = @albedo.<UDIM>.png@
}
""",
        encoding="utf-8",
    )
    admitted_tile = bundle_dir / "albedo.1001.png"
    admitted_tile.write_bytes(b"admitted tile")
    (bundle_dir / "albedo.1002.png").write_bytes(b"unbound tile")
    tile_representation = GeometryRepresentationBinding(
        representation_id="albedo-1001",
        role="supporting_asset",
        format="png",
        media_type="image/png",
        artifact=GeometryArtifactBinding(
            path=admitted_tile.name,
            sha256=hashlib.sha256(admitted_tile.read_bytes()).hexdigest(),
            size_bytes=admitted_tile.stat().st_size,
        ),
    )
    manifest = _source_bundle_manifest(
        bundle_dir / "geometry.source.json",
        source,
        additional_representations=(tile_representation,),
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "absent from its package" in prepared.errors[0]


def test_source_bundle_parses_only_exact_ply_header_terminator(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = bundle_dir / "mesh.ply"
    source.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "comment fake end_header marker",
                "comment TextureFile ../outside.png",
                "element vertex 0",
                "end_header",
                "",
            )
        ),
        encoding="ascii",
    )
    (tmp_path / "outside.png").write_bytes(b"outside texture")
    manifest = _source_bundle_manifest(
        bundle_dir / "geometry.source.json",
        source,
        representation_format="ply",
        media_type="application/x-ply",
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "traversal dependency" in prepared.errors[0]


def test_source_bundle_snapshots_sidecars_before_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    source = bundle_dir / "mesh.gltf"
    source.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "buffer.bin", "byteLength": 14}],
            }
        ),
        encoding="utf-8",
    )
    sidecar = bundle_dir / "buffer.bin"
    admitted_bytes = b"admitted bytes"
    sidecar.write_bytes(admitted_bytes)
    sidecar_representation = GeometryRepresentationBinding(
        representation_id="mesh-buffer",
        role="supporting_asset",
        format="bin",
        media_type="application/octet-stream",
        artifact=GeometryArtifactBinding(
            path=sidecar.name,
            sha256=hashlib.sha256(admitted_bytes).hexdigest(),
            size_bytes=len(admitted_bytes),
        ),
    )
    manifest = _source_bundle_manifest(
        bundle_dir / "geometry.source.json",
        source,
        representation_format="gltf",
        media_type="model/gltf+json",
        additional_representations=(sidecar_representation,),
    )
    consumed_paths: list[Path] = []

    def inspect_conversion_snapshot(
        captured_source: Path,
        _prep_dir: Path,
        *,
        install_missing: bool,
        timeout_s: float,
    ) -> geometry_source_prep.PreparedGeometrySource:
        assert install_missing is False
        assert timeout_s == 120.0
        sidecar.write_bytes(b"changed after admission")
        captured_sidecar = captured_source.parent / sidecar.name
        assert captured_source != source
        assert captured_sidecar.read_bytes() == admitted_bytes
        consumed_paths.extend((captured_source, captured_sidecar))
        return geometry_source_prep.PreparedGeometrySource(
            fidelity_tier="shared_conversion_noneditable",
            source_authoring_mode="shared_conversion",
            status="unsupported",
            source_path=str(captured_source),
            original_input_path=str(captured_source),
            errors=["conversion intentionally stopped after snapshot inspection"],
        )

    monkeypatch.setattr(
        geometry_source_prep,
        "_prepare_shared_conversion",
        inspect_conversion_snapshot,
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert len(consumed_paths) == 2
    snapshot_root = consumed_paths[0].parent
    assert snapshot_root.is_relative_to(
        (tmp_path / "out" / "source_prep" / "admitted_bundles").resolve()
    )
    assert all(path.is_relative_to(snapshot_root) for path in consumed_paths)


def test_source_bundle_does_not_admit_native_source_as_usd_dependency(
    tmp_path: Path,
) -> None:
    root = _tiny_usda(tmp_path / "root.usda")
    native = _tiny_usda(tmp_path / "native.usda")
    root.write_text(
        root.read_text(encoding="utf-8").replace(
            'defaultPrim = "Asset"',
            'defaultPrim = "Asset"\n    subLayers = [@native.usda@]',
        ),
        encoding="utf-8",
    )
    native_representation = GeometryRepresentationBinding(
        representation_id="native-source",
        role="native_source",
        format="usda",
        media_type="model/vnd.usda",
        artifact=GeometryArtifactBinding(
            path=native.name,
            sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            size_bytes=native.stat().st_size,
        ),
    )
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        root,
        additional_representations=(native_representation,),
    )

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        source_manifest_path=manifest,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "geometry_source_bundle_invalid"
    assert "absent from its package" in prepared.errors[0]


def test_generic_3mf_uses_builtin_deterministic_mesh_bridge(tmp_path: Path) -> None:
    source = _tiny_3mf(tmp_path / "generic.3mf")

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=source,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
    )

    assert prepared.is_prepared
    assert prepared.fidelity_tier == "mesh_3mf_import"
    assert prepared.source_authoring_mode == "opaque_import"
    assert prepared.metadata["mesh_cleanup"]["object_count"] == 2


def test_auto_mesh_recovery_requires_explicit_lossy_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "scan.stl"
    source.write_text("solid scan\nendsolid scan\n", encoding="utf-8")
    calls: list[str] = []

    def fake_recovery(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append("recovery")
        return geometry_source_prep.PreparedGeometrySource(
            fidelity_tier="parametric_recovery_request",
            source_authoring_mode="parametric_recovery",
            source_path=str(source),
            prepared_usd_path=str(tmp_path / "recovered.usda"),
            original_input_path=str(source),
            lossy=True,
        )

    def fake_conversion(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        calls.append("conversion")
        return geometry_source_prep.PreparedGeometrySource(
            fidelity_tier="shared_conversion_noneditable",
            source_authoring_mode="shared_conversion",
            source_path=str(source),
            prepared_usd_path=str(tmp_path / "converted.usda"),
            original_input_path=str(source),
            lossy=True,
        )

    monkeypatch.setattr(
        geometry_source_prep, "_prepare_parametric_recovery", fake_recovery
    )
    monkeypatch.setattr(
        geometry_source_prep, "_prepare_shared_conversion", fake_conversion
    )

    geometry_source_prep.prepare_geometry_source(
        source_path=source,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "converted-out",
    )
    geometry_source_prep.prepare_geometry_source(
        source_path=source,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "recovered-out",
        allow_lossy_recovery=True,
    )

    assert calls == ["conversion", "recovery"]


@pytest.mark.parametrize(
    ("filename", "mode"),
    [
        ("part.step", "parametric_recovery"),
        ("part.legacy-cad-source", "shared_conversion"),
        ("part.3mf", "direct_preserve"),
    ],
)
def test_explicit_source_authoring_mode_must_match_source_type(
    tmp_path: Path,
    filename: str,
    mode: str,
) -> None:
    source = tmp_path / filename
    source.write_text("source", encoding="utf-8")

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=source,
        generated_usd_path=None,
        prompt=None,
        image_path=None,
        output_dir=tmp_path / "out",
        source_authoring_mode=mode,  # type: ignore[arg-type]
    )

    assert prepared.status == "unsupported"
    assert prepared.metadata["error_code"] == "source_authoring_mode_mismatch"


def test_generated_usd_reports_actual_direct_preserve_mode(tmp_path: Path) -> None:
    generated = _tiny_usda(tmp_path / "generated.usda")

    prepared = geometry_source_prep.prepare_geometry_source(
        source_path=None,
        generated_usd_path=generated,
        prompt="generated by an upstream authoring provider",
        image_path=None,
        output_dir=tmp_path / "out",
        source_authoring_mode="opaque_import",
    )

    assert prepared.is_prepared
    assert prepared.fidelity_tier == "direct_usd_preserved"
    assert prepared.source_authoring_mode == "direct_preserve"


def test_failed_shared_conversion_paths_are_returned_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "part.step"
    source.write_text("STEP", encoding="utf-8")
    probe = tmp_path / "converter_probe.json"
    report = tmp_path / "conversion_report.json"
    probe.write_text("{}\n", encoding="utf-8")
    report.write_text("{}\n", encoding="utf-8")
    prepared = geometry_source_prep.PreparedGeometrySource(
        fidelity_tier="shared_conversion_noneditable",
        source_authoring_mode="shared_conversion",
        status="unsupported",
        source_path=str(source),
        original_input_path=str(source),
        lossy=True,
        errors=["converter unavailable"],
        metadata={
            "conversion": {
                "probe_path": str(probe),
                "report_path": str(report),
                "status": "blocked",
            }
        },
    )
    monkeypatch.setattr(
        geometry_workflow, "prepare_geometry_source", lambda **_kwargs: prepared
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(source_path=source, output_dir=tmp_path / "out")
    )

    assert result.success is False
    assert result.handoff_manifest_path is None
    assert result.conversion_probe_path == str(probe)
    assert result.conversion_report_path == str(report)


def test_generic_3mf_rejects_oversize_model_xml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threemf = tmp_path / "oversize.3mf"
    with zipfile.ZipFile(threemf, "w") as archive:
        archive.writestr("3D/3dmodel.model", "x" * 16)
    monkeypatch.setattr(geometry_source_prep, "THREEMF_MODEL_XML_MAX_BYTES", 8)

    with pytest.raises(RuntimeError, match="model XML payload exceeds"):
        geometry_source_prep._parse_3mf_objects(threemf)


def test_generic_3mf_rejects_dtd_and_external_entities(tmp_path: Path) -> None:
    threemf = tmp_path / "unsafe.3mf"
    model = b"""<?xml version="1.0"?>
<!DOCTYPE model [<!ENTITY file SYSTEM "file:///etc/passwd">]>
<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="unsafe">&file;</metadata><resources/><build/>
</model>
"""
    with zipfile.ZipFile(threemf, "w") as archive:
        archive.writestr("3D/3dmodel.model", model)

    with pytest.raises(RuntimeError, match="model XML is unsafe or invalid"):
        geometry_source_prep._parse_3mf_objects(threemf)


def test_generic_3mf_applies_model_units_and_preserves_triangle_materials(
    tmp_path: Path,
) -> None:
    model = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="inch"
  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
  xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <resources>
    <m:colorgroup id="1">
      <m:color color="#ff0000" />
      <m:color color="#0000ff" />
    </m:colorgroup>
    <object id="2" name="two color plate" pid="1" pindex="0">
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0" />
          <vertex x="1" y="0" z="0" />
          <vertex x="0" y="1" z="0" />
          <vertex x="1" y="1" z="0" />
        </vertices>
        <triangles>
          <triangle v1="0" v2="1" v3="2" pid="1" p1="0" p2="0" p3="0" />
          <triangle v1="1" v2="3" v3="2" pid="1" p1="1" p2="1" p3="1" />
        </triangles>
      </mesh>
    </object>
  </resources>
  <build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 1 0 0" /></build>
</model>
"""
    threemf = tmp_path / "colored_inches.3mf"
    with zipfile.ZipFile(threemf, "w") as archive:
        archive.writestr("3D/3dmodel.model", model)

    objects = geometry_source_prep._parse_3mf_objects(threemf)

    assert {obj["color"] for obj in objects} == {"#ff0000", "#0000ff"}
    assert {obj["name"] for obj in objects} == {
        "build_1_two color plate_ff0000",
        "build_1_two color plate_0000ff",
    }
    assert all(len(obj["indices"]) == 3 for obj in objects)
    assert any((25.4, 0.0, 0.0) in obj["points"] for obj in objects)
    assert any((50.8, 0.0, 0.0) in obj["points"] for obj in objects)


def test_generic_3mf_mesh_cleanup_removes_invalid_and_duplicate_triangles() -> None:
    cleaned = geometry_source_prep._clean_3mf_mesh_object(
        {
            "id": "mesh",
            "name": "mesh",
            "color": "#c8c8c8",
            "points": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
            ],
            "indices": [
                0,
                1,
                2,
                0,
                1,
                2,
                0,
                2,
                3,
                0,
                4,
                4,
                0,
                1,
                99,
            ],
        }
    )

    assert cleaned["points"] == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    assert cleaned["indices"] == [0, 1, 2]
    assert cleaned["cleanup"] == {
        "input_vertex_count": 5,
        "input_triangle_count": 5,
        "duplicate_vertices_removed": 1,
        "invalid_faces_removed": 1,
        "degenerate_faces_removed": 2,
        "duplicate_faces_removed": 1,
        "unused_vertices_removed": 1,
        "output_vertex_count": 3,
        "output_triangle_count": 1,
    }


def test_generic_3mf_mesh_cleanup_welds_numerically_equivalent_vertices() -> None:
    cleaned = geometry_source_prep._clean_3mf_mesh_object(
        {
            "id": "mesh",
            "name": "mesh",
            "color": "#c8c8c8",
            "points": [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0 + 4e-10, 0.0),
            ],
            "indices": [0, 1, 2, 0, 1, 3],
        }
    )

    assert cleaned["points"] == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    assert cleaned["indices"] == [0, 1, 2]
    assert cleaned["cleanup"]["duplicate_vertices_removed"] == 1
    assert cleaned["cleanup"]["duplicate_faces_removed"] == 1


def test_3mf_usd_writer_recenters_meshes_to_preserve_small_remote_features(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    usd_path = tmp_path / "remote_detail.usda"
    geometry_source_prep._write_usda_meshes(
        [
            {
                "id": "remote",
                "name": "remote detail",
                "color": "#c8c8c8",
                "points": [
                    (1_000_000.0, 2_000_000.0, 3_000_000.0),
                    (1_000_000.01, 2_000_000.0, 3_000_000.0),
                    (1_000_000.0, 2_000_000.01, 3_000_000.0),
                ],
                "indices": [0, 1, 2],
            }
        ],
        usd_path,
        asset_name="remote_asset",
    )

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath("/remote_asset/remote_detail")
    points = list(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    translation = prim.GetAttribute("xformOp:translate").Get()
    assert float(points[1][0] - points[0][0]) == pytest.approx(0.01)
    assert float(points[2][1] - points[0][1]) == pytest.approx(0.01)
    assert tuple(float(value) for value in translation) == pytest.approx(
        (1_000_000.005, 2_000_000.005, 3_000_000.0)
    )


def test_generic_3mf_components_and_build_transforms_are_preserved(
    tmp_path: Path,
) -> None:
    objects = geometry_source_prep._parse_3mf_objects(
        _component_3mf(tmp_path / "component.3mf")
    )

    assert len(objects) == 1
    assert objects[0]["name"] == "translated assembly_triangle tooth"
    assert objects[0]["points"] == [
        (10.0, 20.0, 0.0),
        (11.0, 20.0, 0.0),
        (10.0, 21.0, 0.0),
    ]
    assert objects[0]["indices"] == [0, 1, 2]


def test_geometry_workflow_rejects_nonfinite_tessellation_tolerance(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        GeometryWorkflowInput(
            source_path=tmp_path / "source.json",
            output_dir=tmp_path / "out",
            usd_tessellation_tolerance=float("inf"),
        )


def test_geometry_workflow_rejects_unsupported_output_suffix(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must end in .usd, .usda, or .usdc"):
        GeometryWorkflowInput(
            source_path=tmp_path / "source.usda",
            output_dir=tmp_path / "out",
            output_usd_path=tmp_path / "out" / "geometry.usdz",
        )


def test_static_visual_profile_does_not_require_physics_materials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cad_verifier import sim_ready

    source = _tiny_usda(tmp_path / "static_visual.usda")

    monkeypatch.setattr(
        sim_ready,
        "audit_usd_mesh_topology",
        lambda _path: {"status": "pass", "issues": []},
    )
    monkeypatch.setattr(
        sim_ready,
        "audit_usd_physics_authoring",
        lambda _path: {"status": "fail", "issues": ["missing rigid body"]},
    )
    monkeypatch.setattr(
        sim_ready,
        "audit_usd_physics_materials",
        lambda _path: {"status": "fail", "issues": ["missing physics material"]},
    )

    checks, failures, warnings, _artifacts, _report_path = (
        geometry_workflow._cad_preflight_checks(
            source,
            tmp_path,
            sim_ready.STATIC_VISUAL_PROFILE_ID,
        )
    )
    assert [check.name for check in checks] == ["cad_preflight.mesh_topology"]
    assert failures == []
    assert warnings == []

    rigid_checks, rigid_failures, _warnings, _artifacts, _report_path = (
        geometry_workflow._cad_preflight_checks(
            source,
            tmp_path,
            sim_ready.RIGID_DYNAMIC_PROFILE_ID,
        )
    )
    assert {check.name for check in rigid_checks} == {
        "cad_preflight.mesh_topology",
        "cad_preflight.downstream_physics_delegation",
    }
    delegation = next(
        check
        for check in rigid_checks
        if check.name == "cad_preflight.downstream_physics_delegation"
    )
    assert delegation.status == "not_evaluated"
    assert delegation.metadata["delegate"] == "content-workflow-physics"
    assert rigid_failures == []

    legacy_checks, legacy_failures, _warnings, _artifacts, _report_path = (
        geometry_workflow._cad_preflight_checks(
            source,
            tmp_path,
            sim_ready.RIGID_DYNAMIC_PROFILE_ID,
            include_legacy_physics=True,
        )
    )
    assert {check.name for check in legacy_checks} == {
        "cad_preflight.mesh_topology",
        "cad_preflight.physics_authoring",
        "cad_preflight.physics_materials",
    }
    assert legacy_failures == ["missing rigid body", "missing physics material"]

    (
        unknown_profile_checks,
        unknown_profile_failures,
        _warnings,
        _artifacts,
        _report_path,
    ) = geometry_workflow._cad_preflight_checks(
        source,
        tmp_path,
        "legacy.custom-profile.v1",
        include_legacy_physics=True,
    )
    assert {check.name for check in unknown_profile_checks} == {
        "cad_preflight.mesh_topology",
        "cad_preflight.physics_authoring",
        "cad_preflight.physics_materials",
    }
    assert unknown_profile_failures == [
        "missing rigid body",
        "missing physics material",
    ]


def test_cad_preflight_preserves_not_evaluated_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cad_verifier import sim_ready

    source = _tiny_usda(tmp_path / "not_evaluated.usda")
    monkeypatch.setattr(
        sim_ready,
        "audit_usd_mesh_topology",
        lambda _path: {"status": "not_evaluated", "issues": []},
    )

    checks, failures, warnings, artifacts, report_path = (
        geometry_workflow._cad_preflight_checks(
            source,
            tmp_path,
            sim_ready.STATIC_VISUAL_PROFILE_ID,
        )
    )

    assert checks[0].status == "not_evaluated"
    assert failures == []
    assert warnings == []
    assert artifacts[0].metadata["status"] == "not_evaluated"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["report"] == report["reports"]["mesh_topology"]
    assert "compatibility_schema_versions" not in report


def test_cad_preflight_extracts_structured_issue_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cad_verifier import sim_ready

    source = _tiny_usda(tmp_path / "structured-issue.usda")
    monkeypatch.setattr(
        sim_ready,
        "audit_usd_mesh_topology",
        lambda _path: {
            "status": "fail",
            "issues": [
                {
                    "code": "mesh.non_manifold",
                    "message": "Mesh has non-manifold edges.",
                }
            ],
        },
    )

    checks, failures, warnings, _artifacts, _report_path = (
        geometry_workflow._cad_preflight_checks(
            source,
            tmp_path,
            sim_ready.STATIC_VISUAL_PROFILE_ID,
        )
    )

    assert checks[0].failures == ["Mesh has non-manifold edges."]
    assert failures == ["Mesh has non-manifold edges."]
    assert warnings == []


def test_geometry_workflow_writes_handoff_manifest(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out",
            run_runtime_validation=False,
        )
    )
    assert not result.success
    assert result.validation_status == "fail"
    assert result.optimized_usd_path
    assert result.handoff_manifest_path
    manifest = json.loads(
        Path(result.handoff_manifest_path).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "content-agent-workflows.geometry-handoff.v3"
    assert manifest["source_route"]["route"] == "provided_cad_repair"
    assert manifest["physics_hints"]["delegate"] == "content-workflow-physics"
    assert "preferred_collision" not in manifest["physics_hints"]
    assert manifest["brep_usd"] is None
    assert manifest["compatibility"]["legacy_mesh_geometry_field"] == "mesh_usd"
    assert manifest["compatibility"]["legacy_brep_mesh_alias_removed"] is True
    assert manifest["compatibility"]["brep_source_field"] == "brep_source"
    assert "brep_source" in manifest["compatibility"]["brep_usd_behavior"]
    assert manifest["sim_ready_status"] == "not_evaluated"
    assert manifest["source_fidelity"]["fidelity_tier"] == "direct_usd_preserved"
    assert manifest["asset_audit"]["passed"] is True
    assert Path(result.asset_audit_path or "").exists()
    assert Path(result.authoring_contract_path or "").exists()
    assert Path(result.validation_evidence_path or "").exists()
    assert Path(result.evidence_bundle_path or "").exists()
    evidence = json.loads(
        Path(result.validation_evidence_path or "").read_text(encoding="utf-8")
    )
    assert evidence["validation_tier"] == "T1_basic_stability"
    assert evidence["sim_ready_status"] == "not_evaluated"


def test_geometry_workflow_returns_result_on_early_route_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_route(**_kwargs):  # noqa: ANN003
        raise RuntimeError("route exploded")

    monkeypatch.setattr(geometry_workflow, "route_geometry_request", fail_route)

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            prompt="Generate a simple fixture",
            output_dir=tmp_path / "out",
            run_runtime_validation=False,
        )
    )

    assert result.success is False
    assert result.validation_status == "fail"
    assert "route exploded" in (result.error or "")
    assert result.route.route == "text_to_cad_generate"


def test_geometry_workflow_surfaces_optimizer_fallback_as_validation_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")

    def fake_optimize_geometry(
        *,
        source_usd,
        output_usd,
        policy,
        backend,
        optimization_config,
        protected_semantic_prim_paths,
        expected_source_sha256,
    ):
        del backend, optimization_config, protected_semantic_prim_paths
        assert expected_source_sha256 == geometry_workflow.file_sha256(source)
        shutil.copy2(source_usd, output_usd)
        metadata_path = Path(str(output_usd) + ".optimization.json")
        metadata = {
            "source_usd": str(source_usd),
            "output_usd": str(output_usd),
            "backend": "world_understanding.OptimizeUSDTask.local",
            "policy": policy,
            "status": "optimization_unavailable",
            "artifact_role": "normalized_copy",
            "degraded_reason": "Scene Optimizer unavailable in test.",
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return {**metadata, "metadata_path": str(metadata_path)}

    def pass_cad_preflight_checks(
        usd_path: Path,
        output_dir: Path,
        profile_id: str,
        **_kwargs: object,
    ):
        return (
            [
                geometry_workflow.ValidationCheck(
                    name="cad_preflight.mesh_topology",
                    status="pass",
                    summary="forced pass",
                    metadata={"profile_id": profile_id},
                )
            ],
            [],
            [],
            [
                geometry_workflow.EvidenceArtifact(
                    kind="geometry_usd",
                    path=str(usd_path),
                    description="Geometry USD under test.",
                )
            ],
            output_dir / "cad_preflight_report.json",
        )

    def pass_runtime_check(*, usd_path: Path, output_dir: Path, params):
        del usd_path, output_dir, params
        return (
            geometry_workflow.ValidationCheck(
                name="runtime_loadability",
                status="pass",
                summary="forced pass",
            ),
            [],
            [],
            [],
            None,
        )

    monkeypatch.setattr(
        geometry_workflow.scene_ops,
        "optimize_geometry",
        fake_optimize_geometry,
    )
    monkeypatch.setattr(
        geometry_workflow,
        "_cad_preflight_checks",
        pass_cad_preflight_checks,
    )
    monkeypatch.setattr(geometry_workflow, "_runtime_check", pass_runtime_check)

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out_optimizer_fallback",
            run_asset_audit=False,
            run_shared_usd_validation=False,
        )
    )

    assert result.success
    assert result.validation_status == "conditional"
    evidence = json.loads(
        Path(result.validation_evidence_path or "").read_text(encoding="utf-8")
    )
    optimizer_check = next(
        check
        for check in evidence["checks"]
        if check["name"] == "usd_cli_scene_optimizer"
    )
    assert optimizer_check["status"] == "warning"
    assert "not claimed as optimized" in optimizer_check["warnings"][0]
    assert evidence["sim_ready_status"] == "not_evaluated"
    bundle = json.loads(
        Path(result.evidence_bundle_path or "").read_text(encoding="utf-8")
    )
    optimizer_reference = next(
        artifact
        for artifact in bundle["artifacts"]
        if artifact["kind"] == "optimization_metadata"
    )
    assert optimizer_reference["status"] == "optimization_unavailable"
    assert optimizer_reference["severity"] == "warning"


def test_optimize_geometry_marks_no_output_fallback_as_contract_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    output = tmp_path / "optimized.usda"
    output.write_text("stale output", encoding="utf-8")
    monkeypatch.setattr(
        geometry_workflow.scene_ops.OptimizeUSDTask,
        "run",
        lambda _self, context: {
            **context,
            "optimization_success": True,
            "optimization_metadata": {},
        },
    )

    result = geometry_workflow.scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
    )

    assert output.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert result["status"] == "optimization_unavailable"
    assert result["backend"] == "world_understanding.OptimizeUSDTask.local"
    assert "without writing output" in result["degraded_reason"]


def test_binary_usdc_export_validates_temporary_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf

    source = _tiny_usda(tmp_path / "source.usda")
    output = tmp_path / "geometry.usdc"
    prior_output = b"previous accepted output"
    output.write_bytes(prior_output)
    real_open_as_anonymous = Sdf.Layer.OpenAsAnonymous

    def reject_temporary(identifier: str) -> object | None:
        if Path(identifier).name.startswith(".geometry.usdc."):
            return None
        return real_open_as_anonymous(identifier)

    monkeypatch.setattr(Sdf.Layer, "OpenAsAnonymous", reject_temporary)

    with pytest.raises(RuntimeError, match="could not reopen binary USDC"):
        geometry_workflow.scene_ops._export_binary_usdc(source, output)

    assert output.read_bytes() == prior_output


def test_runtime_validation_builds_temp_proxy_when_rigid_body_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "imported.usda")
    runtime_dir = tmp_path / "runtime"

    from content_agent_workflows.runtime_validation import workflow as runtime_workflow

    def fake_optimize(_self, context):  # noqa: ANN001, ANN202
        assert (
            context["optimization_config"]["scene_optimizer_settings"][
                "enable_deinstance"
            ]
            is True
        )
        shutil.copy2(context["input_usd_path"], context["output_usd_path"])
        return {
            **context,
            "optimization_success": True,
            "optimization_metadata": {"operations_executed": ["deinstance"]},
        }

    def fake_inspect_mesh_candidates(_path):  # noqa: ANN001, ANN202
        return {
            "candidates": [
                {
                    "prim_path": "/Asset/male_block",
                    "prim_name": "male_block",
                    "type_name": "Mesh",
                },
                {
                    "prim_path": "/Asset/guide_clearance",
                    "prim_name": "guide_clearance",
                    "type_name": "Mesh",
                },
            ]
        }

    def fake_apply_schema(
        *,
        usd_path,
        predictions_jsonl_path,
        output_usd_path,
        collision_approximation,
    ):  # noqa: ANN001, ANN202
        assert Path(usd_path).name.endswith(".deinstanced.usda")
        assert collision_approximation == "convexHull"
        records = [
            json.loads(line)
            for line in Path(predictions_jsonl_path)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(records) == 1
        assert records[0]["id"] == "/Asset/male_block"
        assert records[0]["source"].endswith("temporary_loadability_proxy")
        shutil.copy2(usd_path, output_usd_path)
        return {"rigid_body_count": 1, "collision_count": 1}

    def fake_validate_runtime(
        *,
        physics_usd,
        output_dir,
        engine,
        duration_s,
        dt,
        sample_fps,
        drop_height_m,
    ):  # noqa: ANN001, ANN202
        del duration_s, dt, sample_fps, drop_height_m
        assert engine == "fake"
        assert Path(physics_usd).name.endswith(".runtime-proxy.usda")
        report = Path(output_dir) / "runtime_validation_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"engine": "fake"}\n', encoding="utf-8")
        return {
            "engine": engine,
            "runtime_report": str(report),
            "failures": [],
            "warnings": [],
            "evidence_artifacts": [
                {
                    "kind": "runtime_report",
                    "path": str(report),
                    "description": "Runtime report.",
                }
            ],
        }

    monkeypatch.setattr(runtime_workflow.OptimizeUSDTask, "run", fake_optimize)
    monkeypatch.setattr(
        runtime_workflow, "_guide_mesh_paths", lambda _path: {"/Asset/guide_clearance"}
    )
    monkeypatch.setattr(
        runtime_workflow, "inspect_mesh_candidates", fake_inspect_mesh_candidates
    )
    monkeypatch.setattr(runtime_workflow, "apply_schema", fake_apply_schema)
    monkeypatch.setattr(runtime_workflow, "validate_runtime", fake_validate_runtime)

    result = runtime_workflow.run_runtime_validation(
        runtime_workflow.RuntimeValidationRequest(
            asset_path=source,
            output_dir=runtime_dir,
            mode="temporary_loadability_proxy",
            engine="fake",
        )
    )

    assert result.failures == []
    assert result.status == "warning"
    assert result.temporary_proxy_used is True
    assert result.claim_scope == "temporary_geometry_loadability_only"
    assert result.metadata["temporary_proxy"]["prediction_count"] == 1
    assert result.metadata["temporary_proxy"]["guide_mesh_count"] == 1
    assert Path(result.runtime_validation_usd or "").name.endswith(
        ".runtime-proxy.usda"
    )
    artifact_kinds = {artifact["kind"] for artifact in result.evidence_artifacts}
    assert {
        "runtime_report",
        "runtime_proxy_deinstanced_usd",
        "runtime_proxy_predictions",
        "runtime_validation_usd",
    } <= artifact_kinds


def test_geometry_workflow_success_tracks_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")

    def fail_cad_preflight_checks(*args, **kwargs):
        del args, kwargs
        return (
            [
                geometry_workflow.ValidationCheck(
                    name="forced_validation_failure",
                    status="fail",
                    summary="forced failure",
                )
            ],
            ["forced failure"],
            [],
            [],
            tmp_path / "cad_preflight_report.json",
        )

    monkeypatch.setattr(
        geometry_workflow,
        "_cad_preflight_checks",
        fail_cad_preflight_checks,
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out_failed_validation",
            run_runtime_validation=False,
            fail_on_validation_error=False,
        )
    )

    assert not result.success
    assert result.validation_status == "fail"


def test_geometry_workflow_reports_missing_generation_artifact(tmp_path: Path) -> None:
    result = run_geometry_workflow(
        GeometryWorkflowInput(
            prompt="Generate a precise rigid fixture",
            output_dir=tmp_path / "out_missing",
            run_runtime_validation=False,
        )
    )

    assert not result.success
    assert result.source_fidelity_tier == "unsupported_requires_converter"
    assert "Configure an authoring provider" in (result.error or "")


def test_geometry_workflow_reports_unsupported_converter_requirement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.urdf"
    source.write_text("<robot name='placeholder' />", encoding="utf-8")

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out_urdf",
            run_runtime_validation=False,
        )
    )

    assert not result.success
    assert result.route.source_category == "convertible_external"
    assert result.source_fidelity_tier == "shared_conversion_noneditable"
    assert "urdf_usd_converter" in (result.error or "")
    assert (
        tmp_path / "out_urdf" / "source_prep" / "conversion" / "conversion_report.json"
    ).exists()


def test_geometry_workflow_composes_profiled_repair_evidence(tmp_path: Path) -> None:
    source = _repair_cube_usda(tmp_path / "cube.usda")

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out",
            optimization_policy="skip",
            repair_mode="auto",
            repair_profile="rigid_pick_place",
            repair_collision_runtime_engine="fake",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is True, result.error
    assert result.repair_outcome == "conditional"
    assert Path(result.repair_certificate_path or "").is_file()
    assert Path(result.repair_source_format_validation_path or "").is_file()
    assert Path(result.repair_collision_usd_path or "").is_file()
    assert Path(result.repair_correspondence_path or "").is_file()
    assert Path(result.repair_protected_feature_candidates_path or "").is_file()
    assert Path(result.repair_manifold_seam_analysis_path or "").is_file()
    manifest = json.loads(Path(result.handoff_manifest_path or "").read_text())
    assert manifest["geometry_repair"]["outcome"] == "conditional"
    assert manifest["geometry_repair"]["claim_scope"] == (
        "geometry_repair.rigid_pick_place"
    )
    assert Path(manifest["geometry_repair"]["correspondence"]).is_file()
    assert Path(manifest["geometry_repair"]["source_format_validation"]).is_file()
    assert Path(manifest["geometry_repair"]["protected_feature_candidates"]).is_file()
    repaired_optimizer_input = Path(manifest["geometry_repair"]["render_usd"]).resolve()
    derived_from = manifest["workflow_geometry"]["derived_from"]
    assert derived_from["role"] == "geometry_repair_render"
    assert Path(derived_from["path"]) == repaired_optimizer_input
    assert derived_from["sha256"] == geometry_workflow.file_sha256(
        repaired_optimizer_input
    )
    evidence = json.loads(Path(result.validation_evidence_path or "").read_text())
    repair_check = next(
        check for check in evidence["checks"] if check["name"] == "geometry_repair"
    )
    assert repair_check["status"] == "warning"
    assert any(
        artifact["kind"] == "geometry_repair_source_format_validation"
        for artifact in repair_check["evidence_artifacts"]
    )
    assert evidence["sim_ready_status"] == "not_evaluated"


def test_changed_repair_certificate_requires_outer_visual_review(
    tmp_path: Path,
) -> None:
    certificate = tmp_path / "repair_certificate.json"
    certificate.write_text(
        json.dumps({"render_geometry_changed": True}),
        encoding="utf-8",
    )

    class _Result:
        certificate_path = str(certificate)

    assert geometry_workflow._repair_visual_review_required(_Result()) is True
    certificate.write_text(
        json.dumps({"render_geometry_changed": False}),
        encoding="utf-8",
    )
    assert geometry_workflow._repair_visual_review_required(_Result()) is False


def test_geometry_workflow_validates_advanced_repair_profile_contract(
    tmp_path: Path,
) -> None:
    advanced = {"profile": "articulated_rigid"}
    request = GeometryWorkflowInput(
        source_path=tmp_path / "source.usda",
        output_dir=tmp_path / "out",
        repair_profile="articulated_rigid",
        repair_advanced_profile=advanced,
    )
    assert request.repair_advanced_profile == advanced

    with pytest.raises(ValueError, match="must match repair_profile"):
        GeometryWorkflowInput(
            source_path=tmp_path / "source.usda",
            output_dir=tmp_path / "mismatch",
            repair_profile="contact_rich",
            repair_advanced_profile=advanced,
        )


def test_geometry_workflow_normalizes_legacy_sdf_worker_ids(tmp_path: Path) -> None:
    request = GeometryWorkflowInput.model_validate(
        {
            "source_path": tmp_path / "source.usda",
            "output_dir": tmp_path / "out",
            "repair_enabled_workers": ("openvdb_rebuild",),
        }
    )
    assert request.repair_enabled_workers == ["sdf_rebuild"]

    with pytest.raises(ValueError, match="duplicate canonical worker 'sdf_rebuild'"):
        GeometryWorkflowInput.model_validate(
            {
                "source_path": tmp_path / "source.usda",
                "output_dir": tmp_path / "out",
                "repair_enabled_workers": ("sdf_rebuild", "openvdb_rebuild"),
            }
        )


def test_manifest_only_brep_enters_native_repair_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fixture.step"
    source.write_bytes(b"digest-bound STEP fixture")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        role="design_exchange",
        representation_id="design-step",
        representation_format="step",
        media_type="model/step",
    )
    admitted_sources: list[Path] = []

    def stop_after_native_admission(request: object) -> None:
        admitted_sources.append(Path(getattr(request, "source_path")).resolve())
        raise RuntimeError("native repair gate reached")

    monkeypatch.setattr(
        geometry_workflow,
        "run_geometry_repair",
        stop_after_native_admission,
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_manifest_path=manifest,
            source_representation_id="design-step",
            expected_source_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            output_dir=tmp_path / "out",
            repair_mode="diagnose",
            optimization_policy="skip",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert len(admitted_sources) == 1
    assert admitted_sources[0] != source.resolve()
    assert admitted_sources[0].read_bytes() == source.read_bytes()
    assert admitted_sources[0].is_relative_to(
        (tmp_path / "out" / "source_prep" / "admitted_bundles").resolve()
    )
    assert result.success is False
    assert result.error == "native repair gate reached"


def test_manifest_native_repair_rejects_changed_accepted_derivative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "fixture.step"
    source.write_bytes(b"digest-bound STEP fixture")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        role="design_exchange",
        representation_id="design-step",
        representation_format="step",
        media_type="model/step",
    )
    derivative = tmp_path / "repaired.step"

    def return_tampered_derivative(_request: object) -> object:
        derivative.write_bytes(b"verified native repair")
        verified_sha256 = hashlib.sha256(derivative.read_bytes()).hexdigest()
        derivative.write_bytes(b"changed after worker verification")
        return SimpleNamespace(
            outcome="certified",
            attempts=[
                SimpleNamespace(
                    status="accepted",
                    output_path=str(derivative),
                    output_sha256=verified_sha256,
                )
            ],
        )

    monkeypatch.setattr(
        geometry_workflow,
        "run_geometry_repair",
        return_tampered_derivative,
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_manifest_path=manifest,
            source_representation_id="design-step",
            expected_source_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            output_dir=tmp_path / "out",
            repair_mode="auto",
            optimization_policy="skip",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is False
    assert result.error == "Accepted native repair output changed after verification"


@pytest.mark.parametrize("output_path", [None, "missing.step"])
def test_manifest_native_repair_rejects_missing_accepted_derivative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_path: str | None,
) -> None:
    source = tmp_path / "fixture.step"
    source.write_bytes(b"digest-bound STEP fixture")
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        role="design_exchange",
        representation_id="design-step",
        representation_format="step",
        media_type="model/step",
    )

    monkeypatch.setattr(
        geometry_workflow,
        "run_geometry_repair",
        lambda _request: SimpleNamespace(
            outcome="certified",
            attempts=[
                SimpleNamespace(
                    status="accepted",
                    output_path=(
                        str(tmp_path / output_path) if output_path is not None else None
                    ),
                    output_sha256="0" * 64,
                )
            ],
        ),
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_manifest_path=manifest,
            source_representation_id="design-step",
            expected_source_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            output_dir=tmp_path / "out",
            repair_mode="auto",
            optimization_policy="skip",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is False
    assert result.error == "Accepted native repair output is missing"


@pytest.mark.skipif(
    shutil.which("usd-convert-cad") is None
    and not (Path(sys.executable).parent / "usd-convert-cad").is_file(),
    reason="usd-convert-cad is not installed",
)
def test_geometry_workflow_runs_manifest_native_brep_gate_before_usd_repair(
    tmp_path: Path,
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    source = tmp_path / "box.step"
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), STEPControl_AsIs)
    assert writer.Write(str(source)) == IFSelect_RetDone
    manifest = _source_bundle_manifest(
        tmp_path / "geometry.source.json",
        source,
        role="design_exchange",
        representation_id="design-step",
        representation_format="step",
        media_type="model/step",
        coordinate_system=GeometryCoordinateSystem(
            meters_per_unit=0.001,
            up_axis="Z",
            forward_axis="+Y",
        ),
    )

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_manifest_path=manifest,
            source_representation_id="design-step",
            expected_source_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            output_dir=tmp_path / "out",
            repair_mode="auto",
            repair_profile="visual_only",
            repair_production_use=False,
            optimization_policy="skip",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is True, result.error
    assert result.source_bundle_id is not None
    assert result.source_representation_id == "design-step"
    assert result.native_repair_certificate_path
    assert result.repair_outcome == "certified"
    assert "/geometry_repair_native/attempts/attempt-00/" in str(
        result.prepared_source_path
    )
    diagnosis = json.loads(Path(result.repair_diagnosis_path or "").read_text())
    assert diagnosis["metrics"]["mesh"]["indexed_boundary_edge_count"] > 0
    assert diagnosis["metrics"]["mesh"]["boundary_edge_count"] == 0
    assert diagnosis["metrics"]["mesh"]["self_intersection_status"] == "pass"
