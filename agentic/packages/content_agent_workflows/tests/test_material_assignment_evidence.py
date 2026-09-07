# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for workflow-owned Material evidence artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from content_agent_workflows.common.scene_correspondence import SceneOptimizerPathMap
from content_agent_workflows.material_assignment.evidence import (
    MaterialCandidatePolicy,
    _authoring_key,
    _shape_hint,
    build_material_authoring_evidence,
)
from content_agent_workflows.material_assignment.finalizer import (
    MaterialDecisionPolicyError,
    finalize_material_policy,
    normalize_material_decision_policy,
)
from content_agent_workflows.material_assignment.grounding import (
    candidate_grounding_evidence,
    decode_segmentation_picks,
    sample_grounding_pixels,
    write_material_grounding_diagnostics,
)
from content_agent_workflows.material_assignment.manifest import (
    MaterialManifestEntry,
    ResolvedMaterialManifest,
)


def test_material_evidence_preserves_legacy_candidate_and_context_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        """#usda 1.0
def Xform "Asset"
{
    def Mesh "BluePanel"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        color3f[] primvars:displayColor = [(0.1, 0.2, 0.9)]
    }
}
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "materials.yaml"
    manifest.write_text("entries: []\n", encoding="utf-8")
    library = tmp_path / "materials.usda"
    library.write_text("#usda 1.0\n", encoding="utf-8")

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="workflow-test",
        source_usd=source,
        materials_yaml=manifest,
        materials_usd=library,
        policy=MaterialCandidatePolicy(skip_invisible=False),
        respect_existing_material_bindings=False,
    )

    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    assert visible["schema_version"] == "content-agents.visible-candidate-prims.v1"
    assert visible["path_space"] == "source"
    assert visible["traversal"]["stage_gprim_count"] == 1
    assert visible["candidates"][0]["source_path"] == "/Asset/BluePanel"
    assert visible["candidates"][0]["runtime_paths"] == ["/Asset/BluePanel"]

    context = json.loads(Path(artifacts["material_authoring_context"]).read_text())
    assert (
        context["schema_version"]
        == "content-agent-workflows.material-authoring-context.v1"
    )
    assert context["material_candidate_policy"]["skip_invisible"] is False
    seed = json.loads(Path(artifacts["material_assignment_seed"]).read_text())
    assert seed["coverage"]["candidate_visible_prim_count"] == 1
    assert Path(artifacts["visible_candidate_table"]).is_file()


def test_material_evidence_preserves_empty_candidate_contract(tmp_path: Path) -> None:
    source = tmp_path / "empty.usda"
    source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="empty",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(),
        respect_existing_material_bindings=False,
    )

    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    seed = json.loads(Path(artifacts["material_assignment_seed"]).read_text())
    assert visible["candidate_visible_prim_count"] == 0
    assert visible["traversal"]["stage_gprim_count"] == 0
    assert visible["candidates"] == []
    assert seed["coverage"]["candidate_visible_prim_count"] == 0
    assert seed["assignments"] == []


def test_material_evidence_rejects_unsupported_renderable_gprim(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cube.usda"
    source.write_text(
        '#usda 1.0\ndef Xform "Asset"\n{\n    def Cube "Body" {}\n}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"non-Mesh Gprims: /Asset/Body \(Cube\)"):
        build_material_authoring_evidence(
            run_dir=tmp_path / "run",
            session_id="unsupported-gprim",
            source_usd=source,
            materials_yaml=None,
            materials_usd=None,
            policy=MaterialCandidatePolicy(),
            respect_existing_material_bindings=False,
        )


def test_material_evidence_ignores_basis_curves(tmp_path: Path) -> None:
    source = tmp_path / "curves.usda"
    source.write_text(
        """#usda 1.0
def Xform "Asset"
{
    def Mesh "Body"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }
    def BasisCurves "Seam"
    {
        point3f[] points = [(0, 0, 0), (1, 1, 0)]
        int[] curveVertexCounts = [2]
        token type = "linear"
        token wrap = "nonperiodic"
    }
}
""",
        encoding="utf-8",
    )

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="ignore-curves",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(),
        respect_existing_material_bindings=False,
    )

    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    assert [candidate["source_path"] for candidate in visible["candidates"]] == [
        "/Asset/Body"
    ]
    assert visible["traversal"]["stage_gprim_count"] == 2
    assert visible["excluded_non_candidates"] == [
        {"reason": "ignored_basis_curves", "count": 1}
    ]


def test_material_evidence_rejects_nonexistent_root(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(_triangle_scene("/Asset", "Panel"), encoding="utf-8")

    with pytest.raises(ValueError, match="root prim does not exist: /Missing"):
        build_material_authoring_evidence(
            run_dir=tmp_path / "run",
            session_id="missing-root",
            source_usd=source,
            materials_yaml=None,
            materials_usd=None,
            policy=MaterialCandidatePolicy(root_prim_path="/Missing"),
            respect_existing_material_bindings=False,
        )


def test_material_evidence_uses_inspection_stage_and_restores_source_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    inspection = tmp_path / "inspection.usda"
    source.write_text(_triangle_scene("/Source", "Panel"), encoding="utf-8")
    inspection.write_text(
        _triangle_scene("/Optimized", "Panel", extent=2), encoding="utf-8"
    )
    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="optimized",
        source_usd=source,
        inspection_usd=inspection,
        correspondence=SceneOptimizerPathMap(
            source_to_inspection_map={"/Source/Panel": ["/Optimized/Panel"]},
            inspection_to_source_map={"/Optimized/Panel": ["/Source/Panel"]},
        ),
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(material_candidate_space="inspection"),
        respect_existing_material_bindings=False,
    )
    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    candidate = visible["candidates"][0]
    assert visible["inspection_usd"] == str(inspection.resolve())
    assert visible["path_space"] == "inspection"
    assert candidate["runtime_path"] == "/Optimized/Panel"
    assert candidate["source_paths"] == ["/Source/Panel"]
    assert candidate["bounds_size"] and candidate["shape_hint"] == "thin_panel"
    context = json.loads(Path(artifacts["material_authoring_context"]).read_text())
    group = context["candidate_groups"][0]
    assert group["runtime_space"] == "inspection"
    assert group["runtime_paths"] == ["/Optimized/Panel"]
    assert group["inspection_paths"] == ["/Optimized/Panel"]
    assert group["source_paths"] == ["/Source/Panel"]
    seed = json.loads(Path(artifacts["material_assignment_seed"]).read_text())
    assignment = seed["assignments"][0]
    assert assignment["runtime_space"] == "inspection"
    assert assignment["runtime_prim_paths"] == ["/Optimized/Panel"]
    assert assignment["source_prim_paths"] == ["/Source/Panel"]
    assert assignment["prim_paths"] == ["/Optimized/Panel"]


def test_material_evidence_restores_source_occurrences_for_optimized_instances(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    inspection = tmp_path / "inspection.usda"
    source.write_text(
        "#usda 1.0\n"
        + _instance_template("Template")
        + 'def Xform "Copy" (prepend references = </Template>) {}\n',
        encoding="utf-8",
    )
    inspection.write_text(
        "#usda 1.0\n"
        + 'def Xform "Optimized"\n{\n'
        + _triangle_scene("", "Body").removeprefix("#usda 1.0\n")
        + _triangle_scene("", "Trim").removeprefix("#usda 1.0\n")
        + "}\n",
        encoding="utf-8",
    )
    correspondence = SceneOptimizerPathMap(
        source_to_inspection_map={
            "/Template/Body": ["/Optimized/Body"],
            "/Template/Trim": ["/Optimized/Trim"],
        },
        inspection_to_source_map={
            "/Optimized/Body": ["/Template/Body"],
            "/Optimized/Trim": ["/Template/Trim"],
        },
    )

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="optimized-instance",
        source_usd=source,
        inspection_usd=inspection,
        correspondence=correspondence,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(material_candidate_space="inspection"),
        respect_existing_material_bindings=False,
    )

    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert [candidate["source_path"] for candidate in candidates] == [
        "/Template/Body",
        "/Template/Trim",
    ]
    assert [candidate["runtime_path"] for candidate in candidates] == [
        "/Optimized/Body",
        "/Optimized/Trim",
    ]
    assert [candidate["original_source_paths"] for candidate in candidates] == [
        ["/Copy/Body"],
        ["/Copy/Trim"],
    ]


def test_material_shape_hints_distinguish_flat_panel_from_slender_bar() -> None:
    panel_shape = _shape_hint([1.0, 0.8, 0.02])
    bar_shape = _shape_hint([5.0, 0.2, 0.2])

    assert panel_shape == "thin_panel"
    assert bar_shape == "slender_bar"
    assert _shape_hint([1.0, 0.8, 0.7]) == "blocky"
    assert _shape_hint([5.0, 0.5, 0.02]) == "irregular"
    assert _shape_hint([1.0, 0.0, 0.0]) == "thin_or_degenerate"
    common = {
        "semantic_hint": "generic_geometry",
        "display_color_label": None,
        "parent": "/Asset/Parts",
        "size_hint": "medium",
    }
    assert _authoring_key({**common, "shape_hint": panel_shape}) == (
        "path:parts:thin_panel:medium"
    )
    assert _authoring_key({**common, "shape_hint": bar_shape}) == (
        "path:parts:slender_bar:medium"
    )


def test_material_evidence_translates_source_root_to_optimized_inspection(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    inspection = tmp_path / "inspection.usda"
    source.write_text(_triangle_scene("/Source", "Panel"), encoding="utf-8")
    inspection.write_text(_triangle_scene("/Optimized", "Panel"), encoding="utf-8")
    correspondence = SceneOptimizerPathMap(
        source_to_inspection_map={"/Source": ["/Optimized"]},
        inspection_to_source_map={"/Optimized": ["/Source"]},
    )
    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="optimized-root",
        source_usd=source,
        inspection_usd=inspection,
        correspondence=correspondence,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            material_candidate_space="inspection", root_prim_path="/Source"
        ),
        respect_existing_material_bindings=False,
    )
    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    assert visible["traversal"]["root_prim_path"] == "/Optimized"
    assert visible["candidates"][0]["source_path"] == "/Source/Panel"


def test_material_evidence_rejects_ambiguous_optimized_root(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    inspection = tmp_path / "inspection.usda"
    source.write_text(_triangle_scene("/Source", "Panel"), encoding="utf-8")
    inspection.write_text(_triangle_scene("/A", "Panel"), encoding="utf-8")
    with pytest.raises(
        ValueError, match="Ambiguous optimizer correspondence for source root"
    ):
        build_material_authoring_evidence(
            run_dir=tmp_path / "run",
            session_id="ambiguous-root",
            source_usd=source,
            inspection_usd=inspection,
            correspondence=SceneOptimizerPathMap(
                source_to_inspection_map={"/Source": ["/A", "/B"]}
            ),
            materials_yaml=None,
            materials_usd=None,
            policy=MaterialCandidatePolicy(
                material_candidate_space="inspection", root_prim_path="/Source"
            ),
            respect_existing_material_bindings=False,
        )


def test_material_evidence_fails_closed_for_ambiguous_inspection_mapping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    inspection = tmp_path / "inspection.usda"
    scene = _triangle_scene("", "Panel")
    source.write_text(scene, encoding="utf-8")
    inspection.write_text(
        scene.replace('"Panel"', '"OptimizedPanel"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Ambiguous optimizer correspondence"):
        build_material_authoring_evidence(
            run_dir=tmp_path / "run",
            session_id="optimized",
            source_usd=source,
            inspection_usd=inspection,
            correspondence=SceneOptimizerPathMap(
                inspection_to_source_map={"/OptimizedPanel": ["/Source/A", "/Source/B"]}
            ),
            materials_yaml=None,
            materials_usd=None,
            policy=MaterialCandidatePolicy(material_candidate_space="inspection"),
            respect_existing_material_bindings=False,
        )


def test_material_evidence_records_traversal_flags_and_effective_inherited_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        """#usda 1.0
def Scope "Materials"
{
    def Material "Existing"
    {
    }
}
def Xform "Asset" {
    rel material:binding = </Materials/Existing>
    def Mesh "Panel" {
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
    }
    def Mesh "__Prototype_Helper" {
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
    }
}
""",
        encoding="utf-8",
    )
    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="flags",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            skip_instances=False, skip_prototypes=True, skip_invisible=False
        ),
        respect_existing_material_bindings=True,
    )
    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    assert visible["traversal"] == {
        "stage": str(source.resolve()),
        "root_prim_path": "/",
        "skip_instances": False,
        "skip_prototypes": True,
        "skip_invisible": False,
        "instance_proxy_traversal": True,
        "stage_gprim_count": 2,
    }
    assert [item["source_path"] for item in visible["candidates"]] == ["/Asset/Panel"]
    candidate = visible["candidates"][0]
    assert candidate["material_binding_type"] == "inherited"
    assert candidate["binding_source_path"] == "/Asset"
    context = json.loads(Path(artifacts["material_authoring_context"]).read_text())
    assert (
        context["candidate_groups"][0]["recommended_coverage_status"]
        == "preserved_existing"
    )


def test_material_evidence_maps_native_instance_parts_to_distinct_source_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        "#usda 1.0\n"
        + _instance_template("Template")
        + 'def Xform "Copy" (prepend references = </Template>) {}\n',
        encoding="utf-8",
    )
    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="native-instance",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            skip_instances=True,
            skip_prototypes=True,
            skip_invisible=False,
        ),
        respect_existing_material_bindings=False,
    )

    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert [candidate["source_path"] for candidate in candidates] == [
        "/Template/Body",
        "/Template/Trim",
    ]
    assert [candidate["runtime_path"] for candidate in candidates] == [
        "/Copy/Body",
        "/Copy/Trim",
    ]
    assert [candidate["original_source_paths"] for candidate in candidates] == [
        ["/Copy/Body"],
        ["/Copy/Trim"],
    ]
    assert all(candidate["instance_collapsed"] for candidate in candidates)
    assert all(not candidate["deinstance_root_paths"] for candidate in candidates)
    assert not any(
        candidate["runtime_path"].startswith("/Template/") for candidate in candidates
    )


def test_material_evidence_keeps_collapsed_candidates_scoped_to_instance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        "#usda 1.0\n"
        + _instance_template("Template")
        + 'def Xform "Copy" (prepend references = </Template>) {}\n'
        + 'def Xform "OtherCopy" (prepend references = </Template>) {}\n',
        encoding="utf-8",
    )

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="scoped-native-instance",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            root_prim_path="/Copy",
            skip_instances=True,
            skip_prototypes=True,
            skip_invisible=False,
        ),
        respect_existing_material_bindings=False,
    )

    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert [candidate["source_path"] for candidate in candidates] == [
        "/Template/Body",
        "/Template/Trim",
    ]
    assert [candidate["runtime_path"] for candidate in candidates] == [
        "/Copy/Body",
        "/Copy/Trim",
    ]
    assert [candidate["original_source_paths"] for candidate in candidates] == [
        ["/Copy/Body"],
        ["/Copy/Trim"],
    ]
    assert all(candidate["instance_collapsed"] for candidate in candidates)


def test_material_evidence_keeps_included_instance_candidates_independent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        "#usda 1.0\n"
        + _instance_template("Template")
        + 'def Xform "Copy" (prepend references = </Template>) {}\n',
        encoding="utf-8",
    )
    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="included-instance",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            skip_instances=False,
            skip_prototypes=False,
            skip_invisible=False,
        ),
        respect_existing_material_bindings=False,
    )

    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert {candidate["source_path"] for candidate in candidates} == {
        "/Copy/Body",
        "/Copy/Trim",
        "/Template/Body",
        "/Template/Trim",
    }
    assert all(
        candidate["original_source_paths"] == [candidate["source_path"]]
        for candidate in candidates
    )


def test_material_evidence_never_emits_runtime_prototype_candidates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(
        "#usda 1.0\n"
        + _instance_template("Template")
        + 'def Xform "Copy" (prepend references = </Template>) {}\n',
        encoding="utf-8",
    )

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="runtime-prototype",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            skip_instances=True,
            skip_prototypes=False,
            skip_invisible=False,
        ),
        respect_existing_material_bindings=False,
    )

    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert candidates
    assert all(
        "/__Prototype_" not in candidate["source_path"] for candidate in candidates
    )
    assert all(
        "/__Prototype_" not in candidate["runtime_path"] for candidate in candidates
    )


def test_material_evidence_treats_pseudo_root_as_containing_all_prims(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.usda"
    source.write_text(_triangle_scene("/World", "Panel"), encoding="utf-8")

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="pseudo-root",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(root_prim_path="/"),
        respect_existing_material_bindings=False,
    )

    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert [candidate["source_path"] for candidate in candidates] == ["/World/Panel"]


def test_material_evidence_does_not_repurpose_non_material_subset_families(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    source = tmp_path / "subsets.usda"
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Panel")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 1, 3, 2])
    material_faces = UsdGeom.Subset.Define(stage, "/Asset/Panel/MaterialFaces")
    material_faces.CreateElementTypeAttr(UsdGeom.Tokens.face)
    material_faces.CreateFamilyNameAttr(UsdShade.Tokens.materialBind)
    material_faces.CreateIndicesAttr([0])
    semantic_faces = UsdGeom.Subset.Define(stage, "/Asset/Panel/SemanticFaces")
    semantic_faces.CreateElementTypeAttr(UsdGeom.Tokens.face)
    semantic_faces.CreateFamilyNameAttr("semanticPartition")
    semantic_faces.CreateIndicesAttr([1])
    stage.GetRootLayer().Save()

    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="subset-families",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(),
        respect_existing_material_bindings=False,
    )
    visible = json.loads(Path(artifacts["visible_candidates"]).read_text())
    paths = {candidate["source_path"] for candidate in visible["candidates"]}

    assert "/Asset/Panel/MaterialFaces" in paths
    assert "/Asset/Panel" in paths
    assert "/Asset/Panel/SemanticFaces" not in paths
    assert {row["reason"]: row["count"] for row in visible["excluded_non_candidates"]}[
        "non_material_subset_family"
    ] == 1


@pytest.mark.parametrize("skip_instances", [True, False])
def test_material_evidence_preserves_external_instance_parts_for_deinstance(
    tmp_path: Path,
    skip_instances: bool,
) -> None:
    external = tmp_path / "external.usda"
    external.write_text(
        "#usda 1.0\n" + _instance_template("ExternalTemplate"), encoding="utf-8"
    )
    source = tmp_path / "asset.usda"
    source.write_text(
        "#usda 1.0\n"
        + 'def Xform "ExternalCopy" (\n    instanceable = true\n    prepend references = @external.usda@</ExternalTemplate>\n) {}\n',
        encoding="utf-8",
    )
    artifacts = build_material_authoring_evidence(
        run_dir=tmp_path / "run",
        session_id="external-instance",
        source_usd=source,
        materials_yaml=None,
        materials_usd=None,
        policy=MaterialCandidatePolicy(
            skip_instances=skip_instances,
            skip_prototypes=True,
            skip_invisible=False,
        ),
        respect_existing_material_bindings=False,
    )
    candidates = json.loads(Path(artifacts["visible_candidates"]).read_text())[
        "candidates"
    ]
    assert [candidate["source_path"] for candidate in candidates] == [
        "/ExternalCopy/Body",
        "/ExternalCopy/Trim",
    ]
    assert [candidate["runtime_path"] for candidate in candidates] == [
        "/ExternalCopy/Body",
        "/ExternalCopy/Trim",
    ]
    assert all(not candidate["instance_collapsed"] for candidate in candidates)
    assert all(
        candidate["deinstance_root_paths"] == ["/ExternalCopy"]
        for candidate in candidates
    )


def _instance_template(name: str) -> str:
    return (
        f'def Xform "{name}" (instanceable = true)\n{{\n'
        '    def Mesh "Body"\n    {\n'
        "        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]\n"
        "        int[] faceVertexCounts = [3]\n"
        "        int[] faceVertexIndices = [0,1,2]\n"
        "    }\n"
        '    def Mesh "Trim"\n    {\n'
        "        point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]\n"
        "        int[] faceVertexCounts = [3]\n"
        "        int[] faceVertexIndices = [0,1,2]\n"
        "    }\n}\n"
    )


def _triangle_scene(root: str, name: str, *, extent: int = 1) -> str:
    indent = "    " if root else ""
    open_root = f'def Xform "{root.strip("/")}"\n{{\n' if root else ""
    close_root = "}\n" if root else ""
    return (
        "#usda 1.0\n"
        + open_root
        + f'{indent}def Mesh "{name}"\n{indent}{{\n'
        + f"{indent}    point3f[] points = [(0, 0, 0), ({extent}, 0, 0), (0, {extent}, 0)]\n"
        + f"{indent}    int[] faceVertexCounts = [3]\n"
        + f"{indent}    int[] faceVertexIndices = [0, 1, 2]\n"
        + f"{indent}}}\n"
        + close_root
    )


def test_finalizer_enforces_candidate_coverage_before_scene_authoring() -> None:
    candidates = {
        "schema_version": "content-agents.visible-candidate-prims.v1",
        "path_space": "source",
        "candidate_visible_prim_count": 1,
        "candidates": [
            {"source_path": "/Asset/Panel", "source_paths": ["/Asset/Panel"]}
        ],
    }
    manifest = ResolvedMaterialManifest(
        manifest_path=Path("materials.yaml"),
        library_path=Path("materials.usda"),
        entries=(MaterialManifestEntry("paint", "", "/Materials/Paint", ()),),
    )
    decision = {
        "schema_version": "content-agents.material-decision-patch.v1",
        "material_assignments": [
            {
                "material_name": "paint",
                "material_path": "/Materials/Paint",
                "source_prim_paths": ["/Asset/Panel"],
            }
        ],
        "reviewed_no_override": [],
    }
    result = finalize_material_policy(
        decision,
        candidates=candidates,
        manifest=manifest,
        respect_existing_material_bindings=False,
    )
    assert result.coverage["unassigned_visible_prim_count"] == 0

    decision["material_assignments"][0]["material_name"] = "missing"
    try:
        finalize_material_policy(
            decision,
            candidates=candidates,
            manifest=manifest,
            respect_existing_material_bindings=False,
        )
    except MaterialDecisionPolicyError as exc:
        assert exc.errors[0].code == "unknown_material"
    else:  # pragma: no cover - assertion support
        raise AssertionError(
            "unknown library material must be rejected before authoring"
        )


def _manifest() -> ResolvedMaterialManifest:
    return ResolvedMaterialManifest(
        manifest_path=Path("materials.yaml"),
        library_path=Path("materials.usda"),
        entries=(
            MaterialManifestEntry(
                "paint", "white paint", "/Materials/Paint", ("paint", "white")
            ),
            MaterialManifestEntry("metal", "steel", "/Materials/Metal", ("metal",)),
        ),
    )


def _assignment(
    name: str, paths: list[str], *, material: str = "paint"
) -> dict[str, object]:
    return {
        "family": name,
        "material_name": material,
        "material_path": f"/Materials/{material.title()}",
        "prim_paths": paths,
    }


@pytest.mark.parametrize(
    ("case", "candidates", "patch", "expected_assignments", "expected_rejections"),
    [
        pytest.param(
            "unambiguous runtime alias translates to source target",
            {
                "path_space": "source",
                "candidates": [{"source_path": "/A", "runtime_path": "/Optimized/A"}],
            },
            {"material_assignments": [_assignment("panel", ["/Optimized/A"])]},
            [["/A"]],
            0,
            id="alias-translation",
        ),
        pytest.param(
            "ambiguous alias is rejected instead of fanning out",
            {
                "path_space": "source",
                "candidates": [
                    {"source_path": "/A", "runtime_path": "/Optimized/Shared"},
                    {"source_path": "/B", "runtime_path": "/Optimized/Shared"},
                ],
            },
            {"material_assignments": [_assignment("shared", ["/Optimized/Shared"])]},
            [],
            1,
            id="ambiguous-alias",
        ),
        pytest.param(
            "same-material shared runtime candidates coalesce atomically",
            {
                "path_space": "source",
                "candidates": [
                    {"source_path": "/A", "runtime_path": "/Optimized/Shared"},
                    {"source_path": "/B", "runtime_path": "/Optimized/Shared"},
                ],
            },
            {
                "material_assignments": [
                    _assignment("a", ["/A"]),
                    _assignment("b", ["/B"]),
                ]
            },
            [["/A", "/B"]],
            0,
            id="shared-alias-coalesce",
        ),
        pytest.param(
            "groups spanning independent aliases retain every component",
            {
                "path_space": "source",
                "candidates": [
                    {"source_path": "/a1", "runtime_path": "/Optimized/A"},
                    {"source_path": "/a2", "runtime_path": "/Optimized/A"},
                    {"source_path": "/b1", "runtime_path": "/Optimized/B"},
                    {"source_path": "/b2", "runtime_path": "/Optimized/B"},
                ],
            },
            {
                "material_assignments": [
                    _assignment("first", ["/a1", "/b1"]),
                    _assignment("second", ["/a2", "/b2"]),
                ]
            },
            [["/a1", "/a2", "/b1", "/b2"]],
            0,
            id="shared-alias-multiple-components",
        ),
        pytest.param(
            "conflicting shared runtime candidates reject all affected paths",
            {
                "path_space": "source",
                "candidates": [
                    {"source_path": "/A", "runtime_path": "/Optimized/Shared"},
                    {"source_path": "/B", "runtime_path": "/Optimized/Shared"},
                ],
            },
            {
                "material_assignments": [
                    _assignment("a", ["/A"], material="paint"),
                    _assignment("b", ["/B"], material="metal"),
                ]
            },
            [],
            2,
            id="shared-alias-conflict",
        ),
        pytest.param(
            "partial shared runtime assignment is rejected",
            {
                "path_space": "source",
                "candidates": [
                    {"source_path": "/A", "runtime_path": "/Optimized/Shared"},
                    {"source_path": "/B", "runtime_path": "/Optimized/Shared"},
                ],
            },
            {"material_assignments": [_assignment("a", ["/A"])]},
            [],
            1,
            id="shared-alias-partial",
        ),
        pytest.param(
            "overlapping shared aliases are rejected",
            {
                "path_space": "source",
                "candidates": [
                    {
                        "source_path": "/A",
                        "runtime_paths": ["/Optimized/One", "/Optimized/Two"],
                    },
                    {"source_path": "/B", "runtime_path": "/Optimized/One"},
                    {"source_path": "/C", "runtime_path": "/Optimized/Two"},
                ],
            },
            {
                "material_assignments": [
                    _assignment("a", ["/A"]),
                    _assignment("b", ["/B"]),
                    _assignment("c", ["/C"]),
                ]
            },
            [],
            3,
            id="overlapping-shared-aliases",
        ),
        pytest.param(
            "instance-collapsed shared inspection alias is rejected",
            {
                "path_space": "source",
                "candidates": [
                    {
                        "source_path": "/A",
                        "runtime_path": "/Optimized/Shared",
                        "instance_collapsed": True,
                        "runtime_space": "inspection",
                    },
                    {
                        "source_path": "/B",
                        "runtime_path": "/Optimized/Shared",
                        "instance_collapsed": True,
                        "runtime_space": "inspection",
                    },
                ],
            },
            {
                "material_assignments": [
                    _assignment("a", ["/A"]),
                    _assignment("b", ["/B"]),
                ]
            },
            [],
            2,
            id="instance-collapsed-shared-alias",
        ),
        pytest.param(
            "source target retains multi-source authoring fanout",
            {
                "path_space": "source",
                "candidates": [
                    {
                        "source_path": "/Prototype/A",
                        "source_paths": ["/Prototype/A", "/Instance/A"],
                    }
                ],
            },
            {"material_assignments": [_assignment("a", ["/Prototype/A"])]},
            [["/Prototype/A"]],
            0,
            id="source-fanout",
        ),
        pytest.param(
            "inspection-space runtime target remains canonical",
            {
                "path_space": "inspection",
                "candidates": [
                    {"source_path": "/Source/A", "runtime_path": "/Runtime/A"}
                ],
            },
            {"material_assignments": [_assignment("a", ["/Runtime/A"])]},
            [["/Runtime/A"]],
            0,
            id="inspection-target",
        ),
        pytest.param(
            "same-material inspection candidates sharing a source coalesce",
            {
                "path_space": "inspection",
                "candidates": [
                    {"source_path": "/Source/Shared", "runtime_path": "/Runtime/A"},
                    {"source_path": "/Source/Shared", "runtime_path": "/Runtime/B"},
                ],
            },
            {
                "material_assignments": [
                    _assignment("a", ["/Runtime/A"]),
                    _assignment("b", ["/Runtime/B"]),
                ]
            },
            [["/Runtime/A", "/Runtime/B"]],
            0,
            id="inspection-shared-source-coalesce",
        ),
        pytest.param(
            "conflicting inspection candidates sharing a source reject",
            {
                "path_space": "inspection",
                "candidates": [
                    {"source_path": "/Source/Shared", "runtime_path": "/Runtime/A"},
                    {"source_path": "/Source/Shared", "runtime_path": "/Runtime/B"},
                ],
            },
            {
                "material_assignments": [
                    _assignment("a", ["/Runtime/A"], material="paint"),
                    _assignment("b", ["/Runtime/B"], material="metal"),
                ]
            },
            [],
            2,
            id="inspection-shared-source-conflict",
        ),
        pytest.param(
            "partial inspection candidates sharing a source reject",
            {
                "path_space": "inspection",
                "candidates": [
                    {"source_path": "/Source/Shared", "runtime_path": "/Runtime/A"},
                    {"source_path": "/Source/Shared", "runtime_path": "/Runtime/B"},
                ],
            },
            {"material_assignments": [_assignment("a", ["/Runtime/A"])]},
            [],
            1,
            id="inspection-shared-source-partial",
        ),
    ],
)
def test_normalized_finalizer_preserves_alias_and_fanout_policy(
    case: str,
    candidates: dict[str, object],
    patch: dict[str, object],
    expected_assignments: list[list[str]],
    expected_rejections: int,
) -> None:
    del case
    result = normalize_material_decision_policy(
        patch,
        candidates=candidates,
        manifest=_manifest(),
        respect_existing_material_bindings=False,
    )
    assert [
        group["prim_paths"] for group in result.material_assignments
    ] == expected_assignments
    assert len(result.rejected_groups) == expected_rejections


def test_shared_alias_groups_spanning_components_merge_family_once() -> None:
    result = normalize_material_decision_policy(
        {
            "material_assignments": [
                _assignment("first", ["/a1", "/b1"]),
                _assignment("second", ["/a2", "/b2"]),
            ]
        },
        candidates={
            "path_space": "source",
            "candidates": [
                {"source_path": "/a1", "runtime_path": "/Optimized/A"},
                {"source_path": "/a2", "runtime_path": "/Optimized/A"},
                {"source_path": "/b1", "runtime_path": "/Optimized/B"},
                {"source_path": "/b2", "runtime_path": "/Optimized/B"},
            ],
        },
        manifest=_manifest(),
        respect_existing_material_bindings=False,
    )

    assert result.material_assignments[0]["family"] == "first / second"


def test_inspection_shared_source_coalesce_preserves_all_path_spaces() -> None:
    result = normalize_material_decision_policy(
        {
            "material_assignments": [
                _assignment("a", ["/Runtime/A"]),
                _assignment("b", ["/Runtime/B"]),
            ]
        },
        candidates={
            "path_space": "inspection",
            "candidates": [
                {"source_path": "/Source/Shared", "runtime_path": "/Runtime/A"},
                {"source_path": "/Source/Shared", "runtime_path": "/Runtime/B"},
            ],
        },
        manifest=_manifest(),
        respect_existing_material_bindings=False,
    )

    assert len(result.material_assignments) == 1
    assignment = result.material_assignments[0]
    assert assignment["prim_paths"] == ["/Runtime/A", "/Runtime/B"]
    assert assignment["runtime_prim_paths"] == ["/Runtime/A", "/Runtime/B"]
    assert assignment["source_prim_paths"] == ["/Source/Shared"]


@pytest.mark.parametrize(
    ("candidates", "patch", "reason"),
    [
        pytest.param(
            {
                "path_space": "source",
                "candidates": [{"source_path": "/Rail", "shape_hint": "slender_bar"}],
            },
            {"material_assignments": [_assignment("rail", ["/Rail"])]},
            "slender-bar",
            id="slender-bar-painted",
        ),
        pytest.param(
            {
                "path_space": "source",
                "candidates": [
                    {
                        "source_path": f"/Part{i}",
                        "shape_hint": "thin_panel" if i % 2 else "blocky",
                    }
                    for i in range(4)
                ],
            },
            {
                "material_assignments": [
                    _assignment("mixed", [f"/Part{i}" for i in range(4)])
                ]
            },
            "mixed-shape",
            id="broad-painted-mixed",
        ),
        pytest.param(
            {
                "path_space": "source",
                "candidates": [
                    {
                        "source_path": f"/Part{i}",
                        "shape_hint": "thin_panel" if i % 2 else "blocky",
                    }
                    for i in range(17)
                ],
            },
            {
                "material_assignments": [
                    _assignment(
                        "large", [f"/Part{i}" for i in range(17)], material="metal"
                    )
                ]
            },
            "large mixed",
            id="large-mixed",
        ),
    ],
)
def test_normalized_finalizer_enforces_structured_guardrails(
    candidates: dict[str, object], patch: dict[str, object], reason: str
) -> None:
    result = normalize_material_decision_policy(
        patch,
        candidates=candidates,
        manifest=_manifest(),
        respect_existing_material_bindings=False,
    )
    assert not result.material_assignments
    assert reason in result.rejected_groups[0]["rejection_reason"]


def test_normalized_finalizer_rejects_clean_slate_no_override_and_duplicates() -> None:
    candidates = {"path_space": "source", "candidates": [{"source_path": "/A"}]}
    result = normalize_material_decision_policy(
        {
            "material_assignments": [
                _assignment("first", ["/A"]),
                _assignment("second", ["/A"]),
            ],
            "reviewed_no_override": [{"family": "keep", "prim_paths": ["/A"]}],
        },
        candidates=candidates,
        manifest=_manifest(),
        respect_existing_material_bindings=False,
    )
    assert [group["family"] for group in result.material_assignments] == ["first"]
    assert len(result.rejected_groups) == 2


def test_grounding_helpers_keep_pixel_and_candidate_policy_outside_scene_tool(
    tmp_path: Path,
) -> None:
    image = tmp_path / "render.png"
    Image.new("RGB", (16, 16), "black").save(image)
    assert sample_grounding_pixels(image, issue_text="too dark", max_points=2)
    assert (
        candidate_grounding_evidence(
            {
                "candidates": [
                    {
                        "source_path": "/Asset/Rail",
                        "path_tokens": ["rail"],
                        "semantic_hint": "rail_bar",
                    }
                ]
            },
            issue_text="dark rail",
        )[0]["source_path"]
        == "/Asset/Rail"
    )
    segmentation = tmp_path / "seg.png"
    Image.new("RGB", (1, 1), (5, 10, 15)).save(segmentation)
    legend = tmp_path / "seg.legend.txt"
    legend.write_text("/Asset/Rail\trgb(5,10,15)\n", encoding="utf-8")
    assert decode_segmentation_picks(
        segmentation_path=segmentation,
        legend_path=legend,
        sample_points=[{"x": 0, "y": 0}],
    )[0]["prim_paths"] == ["/Asset/Rail"]

    legend.write_text("@n12\trgb(5,10,15)\n", encoding="utf-8")
    assert decode_segmentation_picks(
        segmentation_path=segmentation,
        legend_path=legend,
        sample_points=[{"x": 0, "y": 0}],
        label_paths={"@n12": "/Asset/Rail"},
    )[0]["prim_paths"] == ["/Asset/Rail"]
    assert (
        decode_segmentation_picks(
            segmentation_path=segmentation,
            legend_path=legend,
            sample_points=[{"x": 0, "y": 0}],
        )[0]["prim_paths"]
        == []
    )


def test_grounding_diagnostic_never_claims_pixel_pick_without_adapter(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "run" / "raw"
    raw.mkdir(parents=True)
    (raw / "visible_candidate_prims.json").write_text(
        json.dumps({"candidates": []}), encoding="utf-8"
    )
    (raw / "final_render_records.json").write_text(
        json.dumps({"renders": []}), encoding="utf-8"
    )
    result = write_material_grounding_diagnostics(
        run_dir=tmp_path / "run",
        validation_iteration=2,
        unresolved_issues=["dark panel"],
    )
    assert result is not None
    record = json.loads(Path(result["aggregate"]).read_text())
    assert record["latest"]["status"] == "completed"
    assert record["latest"]["operation_counts"]["pick_calls"] == 0


def test_grounding_diagnostic_decodes_usd_cli_segmentation_legend(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    beauty = run_dir / "final.png"
    Image.new("RGB", (4, 4), "black").save(beauty)
    segmentation = run_dir / "segmentation.png"
    Image.new("RGB", (4, 4), (9, 8, 7)).save(segmentation)
    legend = run_dir / "segmentation.legend.txt"
    legend.write_text("@n12\trgb(9,8,7)\n", encoding="utf-8")
    (raw / "visible_candidate_prims.json").write_text(
        json.dumps({"candidates": []}), encoding="utf-8"
    )
    (raw / "final_render_records.json").write_text(
        json.dumps(
            {
                "renders": [
                    {
                        "name": "final_top",
                        "image_path": str(beauty),
                        "segmentation_path": str(segmentation),
                        "segmentation_legend_path": str(legend),
                        "segmentation_response": {
                            "data": {
                                "segmentation_legend_paths": {"@n12": "/Asset/Panel"}
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = write_material_grounding_diagnostics(
        run_dir=run_dir,
        validation_iteration=3,
        unresolved_issues=["dark panel"],
    )
    assert result is not None
    record = json.loads(Path(result["aggregate"]).read_text())
    view = record["latest"]["issues"][0]["views"][0]
    assert view["picked_source_paths"] == ["/Asset/Panel"]
    assert record["latest"]["operation_counts"]["pick_calls"] > 0


def test_grounding_diagnostic_rejects_recorded_artifacts_outside_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    beauty = outside / "beauty.png"
    segmentation = outside / "segmentation.png"
    legend = outside / "segmentation.legend.txt"
    Image.new("RGB", (4, 4), "black").save(beauty)
    Image.new("RGB", (4, 4), (9, 8, 7)).save(segmentation)
    legend.write_text("@n12\trgb(9,8,7)\n", encoding="utf-8")
    (raw / "visible_candidate_prims.json").write_text(
        json.dumps({"candidates": []}), encoding="utf-8"
    )
    (raw / "final_render_records.json").write_text(
        json.dumps(
            {
                "renders": [
                    {
                        "name": "untrusted",
                        "image_path": str(beauty),
                        "segmentation_path": str(segmentation),
                        "segmentation_legend_path": str(legend),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = write_material_grounding_diagnostics(
        run_dir=run_dir,
        validation_iteration=4,
        unresolved_issues=["dark panel"],
    )

    assert result is not None
    record = json.loads(Path(result["aggregate"]).read_text(encoding="utf-8"))
    view = record["latest"]["issues"][0]["views"][0]
    assert view["image_path"] is None
    assert view["segmentation_path"] is None
    assert view["segmentation_legend_path"] is None
    assert view["pick_results"] == []
    assert view["skip_reason"] == "missing_segmentation_evidence"


def test_grounding_diagnostic_does_not_follow_record_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    outside = tmp_path / "outside-records.json"
    outside.write_text(
        json.dumps({"renders": [{"image_path": "/etc/passwd"}]}),
        encoding="utf-8",
    )
    (raw / "visible_candidate_prims.json").write_text("{}", encoding="utf-8")
    (raw / "final_render_records.json").symlink_to(outside)

    result = write_material_grounding_diagnostics(
        run_dir=run_dir,
        validation_iteration=2,
        unresolved_issues=["unresolved material"],
    )

    assert result is not None
    record = json.loads(Path(result["aggregate"]).read_text(encoding="utf-8"))
    assert record["latest"]["issues"][0]["views"] == []


def test_grounding_diagnostic_retains_prior_runs_and_replaces_same_iteration(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    (raw / "visible_candidate_prims.json").write_text(
        json.dumps({"candidates": []}), encoding="utf-8"
    )
    (raw / "final_render_records.json").write_text(
        json.dumps({"renders": []}), encoding="utf-8"
    )

    write_material_grounding_diagnostics(
        run_dir=run_dir,
        validation_iteration=1,
        unresolved_issues=["old first issue"],
    )
    write_material_grounding_diagnostics(
        run_dir=run_dir,
        validation_iteration=2,
        unresolved_issues=["second issue"],
    )
    result = write_material_grounding_diagnostics(
        run_dir=run_dir,
        validation_iteration=1,
        unresolved_issues=["replacement first issue"],
    )

    assert result is not None
    aggregate = json.loads(Path(result["aggregate"]).read_text(encoding="utf-8"))
    runs_by_iteration = {run["validation_iteration"]: run for run in aggregate["runs"]}
    assert set(runs_by_iteration) == {1, 2}
    assert runs_by_iteration[1]["issues"][0]["issue_text"] == (
        "replacement first issue"
    )
    assert runs_by_iteration[2]["issues"][0]["issue_text"] == "second issue"
    assert aggregate["latest"] == runs_by_iteration[1]
