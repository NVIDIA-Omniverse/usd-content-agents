# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdShade

from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.texture import (
    ProvidedImageTextureApplyLeaf,
    TextureAcceptanceCriteria,
    TextureExecutionResult,
    TextureGeneratorInputs,
    TextureGeneratorLeafRequest,
    TextureInspectionResult,
    TextureInspectionUnit,
    TextureOperationSelection,
    TextureOuterPlan,
    TextureOuterReviewInput,
    TextureOuterUnitReview,
    TexturePlanDocument,
    TexturePlanTarget,
    TexturePreservationConstraints,
    TextureProvidedImageArtifact,
    TextureProvidedImageProducer,
    TextureReferenceArtifact,
    TextureUnitArtifact,
    TextureUnitRenderEvidence,
    build_texture_capability_request,
    collect_texture_candidate_evidence,
    invoke_texture_generator,
    prepare_texture_scope,
    publish_texture_candidate,
    record_texture_outer_review,
    request_texture_critique,
    request_texture_provider_proposal,
    texture_plan_digest,
    validate_texture_preparation,
)


def _binding(path: Path) -> ExecutionArtifactBinding:
    resolved = path.resolve()
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _source(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
        }
    }

    def Mesh "Panel" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </World/Looks/Paint>
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }
}
""",
        encoding="utf-8",
    )
    return path


class _ProviderFreeInspector:
    def __init__(self) -> None:
        self.calls = 0

    def inspect(self, *, request: Any, plan: Any, output_dir: Path) -> Any:
        self.calls += 1
        before = output_dir / "before.png"
        before.write_bytes(b"provider-free-before")
        facts = output_dir / "inspection.json"
        facts.write_text("{}\n", encoding="utf-8")
        units = tuple(
            TextureInspectionUnit(
                unit_id=unit.unit_id,
                material_prim_paths=unit.material_prim_paths,
                material_alias_paths=tuple(
                    str(path)
                    for path in unit.model_dump(mode="json").get(
                        "material_alias_paths", ()
                    )
                ),
                member_prim_paths=unit.member_prim_paths,
                member_subset_paths=unit.member_subset_paths,
                uv_status="ready",
                uv_facts={"interpolation": "vertex", "finite_values": True},
                # Advisory seed deliberately differs from the later outer input.
                proposed_generator_inputs=TextureGeneratorInputs(
                    backend="advisory-service",
                    prompt="provider proposed blue paint",
                ),
            )
            for unit in plan.selected_units
        )
        return TextureInspectionResult(
            source=_binding(Path(request.source_asset)),
            proposal_plan_digest=texture_plan_digest(plan),
            units=units,
            before_render_artifacts=(_binding(before),),
            inspection_artifacts=(_binding(facts),),
            reference_artifacts=request.reference_artifacts,
            capability_constraints=("surface texturing only",),
            renderer_metadata={"provider": "fake-ovrtx"},
            tool_metadata={"provider": "deterministic-test"},
        )


class _ExplicitGenerator:
    provider_id = "explicit-generator"
    capability_id = "test.generate.v1"

    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls = 0
        self.operation_root_mode_before_generate: int | None = None

    def generate(self, request: Any) -> TextureExecutionResult:
        self.calls += 1
        output_dir = Path(request.output_dir)
        self.operation_root_mode_before_generate = stat.S_IMODE(
            output_dir.parent.stat().st_mode
        )
        output_dir.mkdir(parents=True)
        candidate = output_dir / "candidate.usda"
        shutil.copy2(self.source, candidate)
        artifacts: list[TextureUnitArtifact] = []
        for unit_id in request.target_unit_ids:
            texture = output_dir / f"{unit_id}_albedo.png"
            texture.write_bytes(f"texture:{unit_id}".encode())
            artifacts.append(
                TextureUnitArtifact(
                    unit_id=unit_id,
                    artifact_paths=(str(texture),),
                    metadata={"backend": self.provider_id},
                )
            )
        return TextureExecutionResult(
            requested_unit_ids=request.target_unit_ids,
            unit_artifacts=tuple(artifacts),
            output_asset_path=str(candidate),
            metadata={"backend": self.provider_id},
        )


class _ProposalProvider:
    def __init__(self, proposal: Any) -> None:
        self.proposal = proposal
        self.calls = 0

    def plan(self, _request: Any) -> Any:
        self.calls += 1
        return self.proposal

    def export_resume_state(self, _plan: Any) -> dict[str, Any]:
        return {}


class _DeterministicCollector:
    def __init__(self) -> None:
        self.calls = 0

    def collect_candidate_evidence(self, **kwargs: Any) -> Any:
        self.calls += 1
        output_dir = Path(kwargs["output_dir"])
        static = output_dir / "scope_invariants.json"
        static.write_text('{"passed": true}\n', encoding="utf-8")
        units: list[TextureUnitRenderEvidence] = []
        for unit_id in kwargs["unit_ids"]:
            source = output_dir / f"{unit_id}_source.png"
            candidate = output_dir / f"{unit_id}_candidate.png"
            source.write_bytes(f"source:{unit_id}".encode())
            candidate.write_bytes(f"candidate:{unit_id}".encode())
            units.append(
                TextureUnitRenderEvidence(
                    unit_id=unit_id,
                    source_images=(_binding(source),),
                    candidate_images=(_binding(candidate),),
                )
            )
        return (
            tuple(units),
            (_binding(static),),
            {"provider": "fake-ovrtx", "semantic_assessment": "not_evaluated"},
        )


def _write_model(path: Path, value: Any) -> ExecutionArtifactBinding:
    atomic_write_json(path, value)
    return _binding(path)


def test_provider_free_prepare_records_unselected_leaves_without_calling_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path / "source.usda")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"exact-reference")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("provider was constructed during Texture preparation")

    monkeypatch.setattr(
        "content_agent_workflows.texture.client.TextureAgentServiceClient.__init__",
        forbidden,
    )
    monkeypatch.setattr(
        "content_agent_workflows.texture.scene_validation.VlmTextureVisualAssessor.__init__",
        forbidden,
    )
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Apply the exact red reference graphic.",
        material_prim_paths=("/World/Looks/Paint",),
        reference_artifacts=(("appearance_reference", reference),),
        operations=TextureOperationSelection(),
    )
    inspector = _ProviderFreeInspector()
    preparation, binding = prepare_texture_scope(request, inspector=inspector)

    assert inspector.calls == 1
    assert stat.S_IMODE(Path(request.output_dir).stat().st_mode) == 0o700
    assert binding.path.endswith("texture_preparation.json")
    assert preparation.scope_plan.selected_unit_ids
    assert preparation.request.reference_artifacts == (
        TextureReferenceArtifact(
            role="appearance_reference",
            artifact=_binding(reference),
        ),
    )
    for operation in (
        "propose",
        "generate",
        "evidence",
        "critique",
        "review",
        "publish",
    ):
        outcome = preparation.operation_status.outcome(operation)  # type: ignore[arg-type]
        assert outcome.state == "not_requested"
        assert outcome.artifact is None


def test_prepare_texture_scope_rejects_foreign_usd_cli_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.texture import scene_validation

    source = _source(tmp_path / "source.usda")
    operation_root = tmp_path / "run" / "preparation"
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=operation_root,
        intent="Preserve the selected surface.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=TextureOperationSelection(),
    )

    class ForeignUsdCliSession:
        @staticmethod
        def preflight_package_route() -> None:
            raise RuntimeError("editable from a foreign source")

    monkeypatch.setattr(
        scene_validation,
        "WorkflowUsdCliSession",
        ForeignUsdCliSession,
    )
    inspector = scene_validation.LiveUsdCliTextureValidator(
        assessor=None,
        validation_policy_id="test.foreign-route.v1",
        directions=("+z",),
    )

    with pytest.raises(RuntimeError, match="editable from a foreign source"):
        prepare_texture_scope(request, inspector=inspector)

    assert not (tmp_path / "run").exists()
    assert not operation_root.exists()
    assert not (operation_root / "capability_request.json").exists()


def test_provider_free_prepare_retains_selected_material_alias_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "internal-instance.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/Root")
    prototype = UsdGeom.Xform.Define(stage, "/Root/Prototype")
    body = UsdGeom.Cube.Define(stage, "/Root/Prototype/Body")
    UsdGeom.Scope.Define(stage, "/Root/Prototype/Looks")
    material = UsdShade.Material.Define(stage, "/Root/Prototype/Looks/Paint")
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)
    instance = UsdGeom.Xform.Define(stage, "/Root/Instance").GetPrim()
    assert instance.GetReferences().AddInternalReference(prototype.GetPath())
    assert instance.SetInstanceable(True)
    assert stage.GetRootLayer().Save()
    alias_path = "/Root/Instance/Looks/Paint"
    assert stage.GetPrimAtPath(alias_path).IsInstanceProxy()

    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Apply brushed paint only to the selected composed material.",
        material_prim_paths=(alias_path,),
        operations=TextureOperationSelection(),
    )
    preparation, _binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )

    selected = preparation.scope_plan.selected_units[0]
    assert alias_path in selected.model_extra["material_alias_paths"]
    assert alias_path in preparation.inspection.units[0].material_alias_paths


def test_cached_preparation_cannot_replace_the_frozen_request(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.usda")
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Preserve the exact selected surface.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=TextureOperationSelection(),
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    validate_texture_preparation(
        preparation,
        preparation_binding=preparation_binding,
        expected_request=request,
    )

    changed_request = preparation.request.model_copy(
        update={"intent": "Replace the frozen intent after preparation."}
    )
    changed = preparation.model_copy(
        update={
            "request": changed_request,
            "request_digest": canonical_json_digest(changed_request),
        }
    )
    changed_path = tmp_path / "changed_preparation.json"
    atomic_write_json(changed_path, changed)

    with pytest.raises(ValueError, match="differs from the frozen request"):
        validate_texture_preparation(
            changed,
            preparation_binding=_binding(changed_path),
            expected_request=request,
        )


def test_cached_preparation_requires_renderer_provenance_on_replay(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.usda")
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Preserve the exact selected surface.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=TextureOperationSelection(),
    )
    preparation, _preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    changed = preparation.model_copy(
        update={
            "inspection": preparation.inspection.model_copy(
                update={"renderer_metadata": {}}
            )
        }
    )
    changed_path = tmp_path / "missing-renderer-provenance.json"
    atomic_write_json(changed_path, changed)

    with pytest.raises(ValueError, match="requires renderer provenance"):
        validate_texture_preparation(
            changed,
            preparation_binding=_binding(changed_path),
            expected_request=request,
        )


def test_outer_plan_without_service_proposal_runs_explicit_generator_and_publication(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.usda")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"exact-reference")
    operations = TextureOperationSelection(
        generate="requested",
        evidence="requested",
        critique="not_requested",
        review="requested",
        publish="requested",
    )
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Apply the exact warm red reference graphic.",
        material_prim_paths=("/World/Looks/Paint",),
        reference_artifacts=(("appearance_reference", reference),),
        operations=operations,
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    outer_inputs = TextureGeneratorInputs(
        backend="explicit-generator",
        engine="fixture-v2",
        prompt="warm red label with white border",
        seed=17,
        reference_artifacts=tuple(
            item.artifact for item in request.reference_artifacts
        ),
    )
    targets = tuple(
        TexturePlanTarget(
            unit_id=unit.unit_id,
            material_prim_paths=unit.material_prim_paths,
            member_prim_paths=unit.member_prim_paths,
            member_subset_paths=unit.member_subset_paths,
            requested_appearance="warm red product label",
            generator_inputs=outer_inputs,
        )
        for unit in preparation.inspection.units
    )
    outer_plan = TextureOuterPlan(
        preparation=preparation_binding,
        source=request.source,
        scope_plan_digest=preparation.scope_plan_digest,
        advisory_proposal=None,
        reference_artifacts=request.reference_artifacts,
        operations=operations,
        targets=targets,
        preservation=TexturePreservationConstraints(),
        acceptance=TextureAcceptanceCriteria(
            appearance_requirements=("match the warm red reference",),
        ),
        capability_constraints=preparation.inspection.capability_constraints,
        stop_policy="Publish only after exact outer acceptance.",
    )
    outer_plan_binding = _write_model(
        Path(request.output_dir) / "texture_outer_plan.json",
        outer_plan,
    )
    generator = _ExplicitGenerator(source)
    substituted_plan_binding = _write_model(
        Path(request.output_dir) / "substituted_outer_plan.json",
        outer_plan.model_copy(update={"stop_policy": "A substituted stop policy."}),
    )
    with pytest.raises(
        ValueError, match="outer Texture plan binding contains another typed packet"
    ):
        invoke_texture_generator(
            outer_plan,
            outer_plan_binding=substituted_plan_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            generator=generator,
        )
    assert generator.calls == 0

    operation_root = Path(request.output_dir)
    operation_root.chmod(0o750)
    generation, generation_binding = invoke_texture_generator(
        outer_plan,
        outer_plan_binding=outer_plan_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        generator=generator,
    )
    assert generator.calls == 1
    assert generator.operation_root_mode_before_generate == 0o750
    assert stat.S_IMODE(operation_root.stat().st_mode) == 0o750
    assert generation.provider_proposal is None
    assert generation.generator_inputs == (outer_inputs,)
    assert (
        preparation.inspection.units[0].proposed_generator_inputs
        != generation.generator_inputs[0]
    )
    assert generation.operation_status.outcome("propose").state == "not_requested"

    graph_generator = _ExplicitGenerator(source)
    graph_operation_root = tmp_path / "graph-native"
    invoke_texture_generator(
        outer_plan,
        outer_plan_binding=outer_plan_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        generator=graph_generator,
        output_dir=graph_operation_root,
    )
    assert graph_generator.calls == 1
    assert graph_generator.operation_root_mode_before_generate == 0o700
    assert stat.S_IMODE(graph_operation_root.stat().st_mode) == 0o700

    evidence, evidence_binding = collect_texture_candidate_evidence(
        outer_plan,
        generation,
        preparation=preparation,
        preparation_binding=preparation_binding,
        outer_plan_binding=outer_plan_binding,
        generation_binding=generation_binding,
        collector=_DeterministicCollector(),
    )
    assert evidence.renderer_metadata["semantic_assessment"] == "not_evaluated"
    assert evidence.operation_status.outcome("critique").state == "not_requested"

    class ForbiddenCritic:
        provider_id = "must-not-run"
        capability_id = "test.critique.v1"

        def critique(self, **_kwargs: Any) -> Any:
            raise AssertionError("unselected Texture critique was invoked")

    with pytest.raises(ValueError, match="Texture critique was not requested"):
        request_texture_critique(
            evidence,
            evidence_binding=evidence_binding,
            intent=request.intent,
            provider=ForbiddenCritic(),
        )

    visuals = (
        *(item.artifact for item in request.reference_artifacts),
        *(
            binding
            for unit in evidence.unit_evidence
            for binding in (*unit.source_images, *unit.candidate_images)
        ),
    )
    review_input = TextureOuterReviewInput(
        outer_plan=outer_plan_binding,
        generation=generation_binding,
        candidate_evidence=evidence_binding,
        candidate=generation.candidate,
        reference_artifacts=request.reference_artifacts,
        unit_reviews=tuple(
            TextureOuterUnitReview(
                unit_id=unit_id,
                disposition="accept",
                rationale="Direct comparison matches the requested reference.",
            )
            for unit_id in outer_plan.target_unit_ids
        ),
        inspected_visual_artifacts=visuals,
        findings=("Outer multimodal review accepted every exact target.",),
    )
    reviewed_image = Path(evidence.unit_evidence[0].candidate_images[0].path)
    reviewed_image_bytes = reviewed_image.read_bytes()
    reviewed_image.write_bytes(b"tampered-after-evidence")
    with pytest.raises(ValueError, match="Texture visual evidence .* bytes changed"):
        record_texture_outer_review(
            review_input,
            outer_plan=outer_plan,
            generation=generation,
            evidence=evidence,
        )
    reviewed_image.write_bytes(reviewed_image_bytes)
    review, review_binding = record_texture_outer_review(
        review_input,
        outer_plan=outer_plan,
        generation=generation,
        evidence=evidence,
    )
    receipt, _receipt_binding = publish_texture_candidate(
        review,
        review_binding=review_binding,
        request_binding=_binding(Path(request.output_dir) / "capability_request.json"),
        preparation=preparation,
        preparation_binding=preparation_binding,
        outer_plan=outer_plan,
        outer_plan_binding=outer_plan_binding,
        generation=generation,
        generation_binding=generation_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
        publication_path=tmp_path / "published.usda",
    )
    assert receipt.published_asset.sha256 == generation.candidate.sha256
    assert receipt.reference_artifacts == request.reference_artifacts
    assert receipt.operation_status.outcome("publish").state == "completed"
    assert stat.S_IMODE(operation_root.stat().st_mode) == 0o750

    Path(generation.generator_artifacts[0].path).write_bytes(b"tampered-generator")
    with pytest.raises(ValueError, match="Texture generator result bytes changed"):
        publish_texture_candidate(
            review,
            review_binding=review_binding,
            request_binding=_binding(
                Path(request.output_dir) / "capability_request.json"
            ),
            preparation=preparation,
            preparation_binding=preparation_binding,
            outer_plan=outer_plan,
            outer_plan_binding=outer_plan_binding,
            generation=generation,
            generation_binding=generation_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            publication_path=tmp_path / "must-not-publish.usda",
        )
    assert not (tmp_path / "must-not-publish.usda").exists()


def test_apply_provided_image_never_calls_provider_and_binds_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path / "source.usda")
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    relative_dependency = tmp_path / "relative_dependency.usda"
    relative_dependency.write_text(
        '#usda 1.0\nover "World"\n{\n}\n',
        encoding="utf-8",
    )
    existing_texture = tmp_path / "existing-albedo.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(existing_texture)
    source_layer = Sdf.Layer.FindOrOpen(str(source))
    assert source_layer is not None
    source_layer.subLayerPaths.append(relative_dependency.name)
    source_layer.Save()

    source_stage = Usd.Stage.Open(str(source))
    assert source_stage is not None
    source_material = UsdShade.Material(
        source_stage.GetPrimAtPath("/World/Looks/Paint")
    )
    existing_texture_shader = UsdShade.Shader.Define(
        source_stage,
        "/World/Looks/Paint/ExistingTexture",
    )
    existing_texture_shader.CreateIdAttr("UsdUVTexture")
    existing_texture_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(existing_texture.name)
    )
    materialx_surface = UsdShade.Shader.Define(
        source_stage,
        "/World/Looks/Paint/ExistingMaterialXSurface",
    )
    materialx_surface.CreateIdAttr("ND_standard_surface_surfaceshader")
    materialx_output = materialx_surface.CreateOutput(
        "out",
        Sdf.ValueTypeNames.Token,
    )
    source_material.CreateSurfaceOutput("mtlx").ConnectToSource(materialx_output)
    preserved = UsdGeom.Cube.Define(source_stage, "/World/PreservedPanel")
    UsdShade.MaterialBindingAPI.Apply(preserved.GetPrim()).Bind(source_material)
    source_stage.GetRootLayer().Save()
    reference = tmp_path / "appearance_reference.png"
    Image.new("RGB", (64, 64), (96, 82, 68)).save(reference)
    provided_path = tmp_path / "outer_generated_albedo.png"
    Image.new("RGB", (64, 64), (212, 73, 29)).save(provided_path)
    operations = TextureOperationSelection(
        generate="requested",
        evidence="requested",
        review="requested",
        publish="requested",
    )
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Apply the exact outer-generated orange image to Panel only.",
        prim_paths=("/World/Panel",),
        reference_artifacts=(("appearance_reference", reference),),
        operations=operations,
        texture_size=64,
    )
    assert {
        item.path: item.model_dump(mode="json") for item in request.source_dependencies
    } == {
        str(path.resolve()): _binding(path).model_dump(mode="json")
        for path in (existing_texture, relative_dependency)
    }
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    unit = preparation.inspection.units[0]
    provided = TextureProvidedImageArtifact(
        unit_id=unit.unit_id,
        channel="albedo",
        artifact=_binding(provided_path),
        producer=TextureProvidedImageProducer(
            provider="outer-image-tool",
            capability="image.generate.v1",
            invocation_id="fresh-external-invocation",
            provenance={"model_alias": "real-provider-alias"},
        ),
    )
    outer_inputs = TextureGeneratorInputs(
        execution_mode="apply_provided",
        backend=ProvidedImageTextureApplyLeaf.provider_id,
        prompt="Outer-authored orange surface intent.",
        texture_size=64,
        reference_artifacts=tuple(
            item.artifact for item in request.reference_artifacts
        ),
        provided_images=(provided,),
    )
    outer_plan = TextureOuterPlan(
        preparation=preparation_binding,
        source=request.source,
        scope_plan_digest=preparation.scope_plan_digest,
        reference_artifacts=request.reference_artifacts,
        operations=operations,
        targets=(
            TexturePlanTarget(
                unit_id=unit.unit_id,
                material_prim_paths=unit.material_prim_paths,
                member_prim_paths=unit.member_prim_paths,
                member_subset_paths=unit.member_subset_paths,
                requested_appearance="Use the exact outer-generated orange image.",
                generator_inputs=outer_inputs,
            ),
        ),
        preservation=TexturePreservationConstraints(),
        acceptance=TextureAcceptanceCriteria(
            appearance_requirements=("show the exact supplied orange surface",),
        ),
        capability_constraints=preparation.inspection.capability_constraints,
        stop_policy="Publish only after exact outer acceptance.",
    )
    outer_plan_binding = _write_model(
        Path(request.output_dir) / "texture_outer_plan.json",
        outer_plan,
    )

    class ForbiddenGenerator:
        provider_id = ProvidedImageTextureApplyLeaf.provider_id
        capability_id = ProvidedImageTextureApplyLeaf.capability_id

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _request: Any) -> Any:
            self.calls += 1
            raise AssertionError("stale provided image reached the selected leaf")

    forbidden_generator = ForbiddenGenerator()
    original = provided_path.read_bytes()
    provided_path.write_bytes(b"changed-before-apply")
    with pytest.raises(ValueError, match="Texture provided image .* bytes changed"):
        invoke_texture_generator(
            outer_plan,
            outer_plan_binding=outer_plan_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            generator=forbidden_generator,
        )
    assert forbidden_generator.calls == 0
    provided_path.write_bytes(original)

    provider_calls = 0

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("apply-provided called a Texture provider")

    monkeypatch.setattr(
        "content_agent_workflows.texture.client.TextureAgentServiceClient.plan",
        forbidden_provider,
    )
    monkeypatch.setattr(
        "content_agent_workflows.texture.client.TextureAgentServiceClient.execute_outer_plan",
        forbidden_provider,
    )
    generation, generation_binding = invoke_texture_generator(
        outer_plan,
        outer_plan_binding=outer_plan_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        generator=ProvidedImageTextureApplyLeaf(),
    )
    assert provider_calls == 0
    assert generation.execution_mode == "apply_provided"
    assert generation.generator_provider == "outer_provided_image_apply"
    assert generation.generator_capability == "texture.apply-provided.v1"
    assert generation.provided_images == (provided,)
    assert generation.execution.metadata["provider_invoked"] is False
    assert generation.candidate.path.endswith(".usdz")
    with zipfile.ZipFile(generation.candidate.path) as candidate_package:
        texture_members = {
            name: candidate_package.read(name)
            for name in candidate_package.namelist()
            if Path(name).suffix.lower() == ".png"
        }
        assert set(texture_members.values()) == {
            original,
            existing_texture.read_bytes(),
        }
        provided_members = tuple(
            name for name, contents in texture_members.items() if contents == original
        )
        assert len(provided_members) == 1
    unit_artifact = generation.execution.unit_artifacts[0]
    unit_receipt = json.loads(Path(unit_artifact.artifact_paths[1]).read_text())
    expected_member = provided_members[0]
    expected_member_binding = {
        "package_member_path": expected_member,
        "package_member_sha256": provided.artifact.sha256,
        "package_member_size_bytes": provided.artifact.size_bytes,
    }
    assert {
        key: unit_receipt[key] for key in expected_member_binding
    } == expected_member_binding
    assert {
        key: unit_artifact.metadata[key] for key in expected_member_binding
    } == expected_member_binding
    assert {item["path"]: item for item in unit_receipt["source_dependencies"]} == {
        str(path.resolve()): _binding(path).model_dump(mode="json")
        for path in (existing_texture, relative_dependency)
    }
    cloned_material_path = f"/World/Looks/{unit.unit_id}"
    assert unit_receipt["authored_shader_paths"] == [
        f"{cloned_material_path}/OuterProvidedAlbedo_{provided.artifact.sha256[:12]}"
    ]
    candidate_stage = Usd.Stage.Open(generation.candidate.path)
    assert candidate_stage is not None
    candidate_material = UsdShade.Material(
        candidate_stage.GetPrimAtPath(cloned_material_path)
    )
    connected_surface_contexts = tuple(
        output.GetFullName()
        for output in candidate_material.GetOutputs()
        if output.GetBaseName() == "surface" and output.GetConnectedSources()[0]
    )
    assert connected_surface_contexts == ("outputs:surface",)
    selected_material = UsdShade.MaterialBindingAPI(
        candidate_stage.GetPrimAtPath("/World/Panel")
    ).ComputeBoundMaterial()[0]
    preserved_material = UsdShade.MaterialBindingAPI(
        candidate_stage.GetPrimAtPath("/World/PreservedPanel")
    ).ComputeBoundMaterial()[0]
    assert str(selected_material.GetPath()) == cloned_material_path
    assert str(preserved_material.GetPath()) == "/World/Looks/Paint"
    original_material = UsdShade.Material(
        candidate_stage.GetPrimAtPath("/World/Looks/Paint")
    )
    assert tuple(
        output.GetFullName()
        for output in original_material.GetOutputs()
        if output.GetConnectedSources()[0]
    ) == ("outputs:mtlx:surface",)

    provided_path.write_bytes(b"changed-after-apply")
    with pytest.raises(ValueError, match="Texture provided image .* bytes changed"):
        collect_texture_candidate_evidence(
            outer_plan,
            generation,
            preparation=preparation,
            preparation_binding=preparation_binding,
            outer_plan_binding=outer_plan_binding,
            generation_binding=generation_binding,
            collector=_DeterministicCollector(),
        )
    provided_path.write_bytes(original)

    evidence, evidence_binding = collect_texture_candidate_evidence(
        outer_plan,
        generation,
        preparation=preparation,
        preparation_binding=preparation_binding,
        outer_plan_binding=outer_plan_binding,
        generation_binding=generation_binding,
        collector=_DeterministicCollector(),
    )
    assert evidence.provided_images == (provided,)
    visuals = (
        *(item.artifact for item in request.reference_artifacts),
        *(item.artifact for item in generation.provided_images),
        *(
            binding
            for item in evidence.unit_evidence
            for binding in (*item.source_images, *item.candidate_images)
        ),
    )
    review_input = TextureOuterReviewInput(
        outer_plan=outer_plan_binding,
        generation=generation_binding,
        candidate_evidence=evidence_binding,
        candidate=generation.candidate,
        reference_artifacts=request.reference_artifacts,
        provided_images=generation.provided_images,
        unit_reviews=(
            TextureOuterUnitReview(
                unit_id=unit.unit_id,
                disposition="accept",
                rationale="The candidate uses the exact outer-provided image.",
            ),
        ),
        inspected_visual_artifacts=visuals,
        findings=("Outer review accepted the digest-bound supplied image.",),
    )
    review, review_binding = record_texture_outer_review(
        review_input,
        outer_plan=outer_plan,
        generation=generation,
        evidence=evidence,
    )
    published_path = tmp_path / "published.usdz"
    provided_path.write_bytes(b"changed-before-publication")
    with pytest.raises(ValueError, match="Texture provided image .* bytes changed"):
        publish_texture_candidate(
            review,
            review_binding=review_binding,
            request_binding=_binding(
                Path(request.output_dir) / "capability_request.json"
            ),
            preparation=preparation,
            preparation_binding=preparation_binding,
            outer_plan=outer_plan,
            outer_plan_binding=outer_plan_binding,
            generation=generation,
            generation_binding=generation_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            publication_path=published_path,
        )
    assert not published_path.exists()
    provided_path.write_bytes(original)
    with pytest.raises(ValueError, match="generation requires evidence.*publication"):
        TextureOperationSelection(
            generate="requested",
            evidence="requested",
            review="requested",
            publish="not_requested",
        )

    real_link = os.link
    source_bytes = source.read_bytes()

    def drift_source_during_promotion(staging: Path, output: Path) -> None:
        real_link(staging, output)
        source.write_bytes(b"changed-during-publication-promotion")

    monkeypatch.setattr(os, "link", drift_source_during_promotion)
    with pytest.raises(ValueError, match="Texture source bytes changed"):
        publish_texture_candidate(
            review,
            review_binding=review_binding,
            request_binding=_binding(
                Path(request.output_dir) / "capability_request.json"
            ),
            preparation=preparation,
            preparation_binding=preparation_binding,
            outer_plan=outer_plan,
            outer_plan_binding=outer_plan_binding,
            generation=generation,
            generation_binding=generation_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            publication_path=published_path,
        )
    assert not published_path.exists()
    source.write_bytes(source_bytes)

    def publish_concurrent_winner(staging: Path, output: Path) -> None:
        Path(output).write_bytes(b"concurrent-winner")
        real_link(staging, output)

    monkeypatch.setattr(os, "link", publish_concurrent_winner)
    with pytest.raises(FileExistsError):
        publish_texture_candidate(
            review,
            review_binding=review_binding,
            request_binding=_binding(
                Path(request.output_dir) / "capability_request.json"
            ),
            preparation=preparation,
            preparation_binding=preparation_binding,
            outer_plan=outer_plan,
            outer_plan_binding=outer_plan_binding,
            generation=generation,
            generation_binding=generation_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            publication_path=published_path,
        )
    assert published_path.read_bytes() == b"concurrent-winner"
    published_path.unlink()
    monkeypatch.setattr(os, "link", real_link)
    receipt, _receipt_binding = publish_texture_candidate(
        review,
        review_binding=review_binding,
        request_binding=_binding(Path(request.output_dir) / "capability_request.json"),
        preparation=preparation,
        preparation_binding=preparation_binding,
        outer_plan=outer_plan,
        outer_plan_binding=outer_plan_binding,
        generation=generation,
        generation_binding=generation_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
        publication_path=published_path,
    )
    assert receipt.provided_images == (provided,)
    assert receipt.published_asset.sha256 == generation.candidate.sha256

    symlink_path = tmp_path / "provided-symlink.png"
    symlink_path.symlink_to(provided_path)
    symlink_binding = ExecutionArtifactBinding(
        path=str(symlink_path),
        sha256=provided.artifact.sha256,
        size_bytes=provided.artifact.size_bytes,
    )
    symlink_provided = provided.model_copy(
        update={"artifact": symlink_binding},
    )
    symlink_inputs = outer_inputs.model_copy(
        update={"provided_images": (symlink_provided,)},
    )
    symlink_target = outer_plan.targets[0].model_copy(
        update={"generator_inputs": symlink_inputs},
    )
    symlink_plan = outer_plan.model_copy(update={"targets": (symlink_target,)})
    symlink_plan_binding = _write_model(
        Path(request.output_dir) / "texture_outer_plan_symlink.json",
        symlink_plan,
    )
    with pytest.raises(ValueError, match="must not be a symlink"):
        invoke_texture_generator(
            symlink_plan,
            outer_plan_binding=symlink_plan_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            generator=forbidden_generator,
        )
    assert forbidden_generator.calls == 0


def test_apply_provided_rejects_flattened_instance_proxy_member(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    prototype_path = tmp_path / "prototype.usda"
    prototype = Usd.Stage.CreateNew(str(prototype_path))
    UsdGeom.Xform.Define(prototype, "/Prototype")
    UsdGeom.Scope.Define(prototype, "/Prototype/Looks")
    material = UsdShade.Material.Define(prototype, "/Prototype/Looks/Paint")
    mesh = UsdGeom.Mesh.Define(prototype, "/Prototype/Panel")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    assert prototype.GetRootLayer().Save()

    source_path = tmp_path / "source.usda"
    source = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.Xform.Define(source, "/World")
    instance = UsdGeom.Xform.Define(source, "/World/Instance").GetPrim()
    assert instance.GetReferences().AddReference(prototype_path.name, "/Prototype")
    assert instance.SetInstanceable(True)
    assert source.GetRootLayer().Save()

    flattened = Usd.Stage.Open(source.Flatten())
    member_path = "/World/Instance/Panel"
    assert flattened.GetPrimAtPath(member_path).IsInstanceProxy()
    with pytest.raises(ValueError, match="read-only instance proxy"):
        ProvidedImageTextureApplyLeaf._bound_material_path(flattened, member_path)


def test_apply_provided_maps_unique_internal_instance_proxy_member(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    from content_agent_workflows.texture.scope_validation import (
        _authorable_instance_path,
        _stage_prims,
    )

    source_path = tmp_path / "source.usda"
    source = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.Xform.Define(source, "/World")
    UsdGeom.Xform.Define(source, "/World/Prototype")
    UsdGeom.Scope.Define(source, "/World/Prototype/Looks")
    material = UsdShade.Material.Define(
        source,
        "/World/Prototype/Looks/Paint",
    )
    mesh = UsdGeom.Mesh.Define(source, "/World/Prototype/Panel")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    instance = UsdGeom.Xform.Define(source, "/World/Instance").GetPrim()
    assert instance.GetReferences().AddInternalReference("/World/Prototype")
    assert instance.SetInstanceable(True)
    assert source.GetRootLayer().Save()

    flattened = Usd.Stage.Open(source.Flatten())
    member_path = "/World/Instance/Panel"
    assert flattened.GetPrimAtPath(member_path).IsInstanceProxy()
    authorable_path = _authorable_instance_path(flattened, member_path)
    assert authorable_path == "/Flattened_Prototype_1/Panel"
    assert authorable_path not in _stage_prims(flattened)
    assert flattened.GetPrimAtPath(authorable_path).IsValid()
    assert (
        ProvidedImageTextureApplyLeaf._bound_material_path(
            flattened,
            authorable_path,
        )
        == "/Flattened_Prototype_1/Looks/Paint"
    )


def test_scope_validation_keys_internal_instance_sources_by_stable_owner(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.texture.scope_validation import (
        _authorable_instance_path,
        validate_texture_scope_invariants,
    )

    unit_a = "tu_11111111111111111111"
    unit_b = "tu_22222222222222222222"
    plan = TexturePlanDocument.model_validate(
        {
            "schema_version": "texture-agent-plan.v1",
            "counts": {"selected_unit_count": 2},
            "decision": {"state": "ready", "execution_allowed": True},
            "selected_units": [
                {
                    "unit_id": unit_a,
                    "unit_mode": "per_material",
                    "material_prim_paths": ["/World/Looks/MaterialA"],
                    "member_prim_paths": ["/World/InstanceA/Mesh"],
                    "member_subset_paths": [],
                },
                {
                    "unit_id": unit_b,
                    "unit_mode": "per_material",
                    "material_prim_paths": ["/World/Looks/MaterialB"],
                    "member_prim_paths": ["/World/InstanceB/Mesh"],
                    "member_subset_paths": [],
                },
            ],
        }
    )

    def create_flattened_stage(
        path: Path,
        *,
        owner_order: tuple[str, str],
        author_unit_materials: bool,
    ) -> None:
        stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/Looks")
        source_materials = (
            UsdShade.Material.Define(stage, "/World/Looks/MaterialA"),
            UsdShade.Material.Define(stage, "/World/Looks/MaterialB"),
        )
        if author_unit_materials:
            for unit_id in (unit_a, unit_b):
                UsdShade.Material.Define(stage, f"/World/Looks/{unit_id}")
        for index, suffix in enumerate(("A", "B")):
            prototype_root = f"/World/Prototype{suffix}"
            UsdGeom.Xform.Define(stage, prototype_root)
            local_material = UsdShade.Material.Define(
                stage,
                f"{prototype_root}/Looks/Diffuse",
            )
            local_surface = UsdShade.Shader.Define(
                stage,
                f"{prototype_root}/Looks/Diffuse/PreviewSurface",
            )
            local_surface.CreateIdAttr("UsdPreviewSurface")
            local_surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            local_material.CreateSurfaceOutput().ConnectToSource(
                local_surface.ConnectableAPI(),
                "surface",
            )
            local_relationship = local_material.GetPrim().CreateRelationship(
                "texture:testSource"
            )
            local_relationship.AddTarget(
                Sdf.Path(
                    f"{prototype_root}/Looks/Diffuse/PreviewSurface.outputs:surface"
                )
            )
            mesh = UsdGeom.Mesh.Define(stage, f"{prototype_root}/Mesh")
            mesh.CreatePointsAttr(
                [(-1.0, -1.0, float(index)), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
            )
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                source_materials[index]
            )
        for suffix in owner_order:
            owner = UsdGeom.Xform.Define(stage, f"/World/Instance{suffix}").GetPrim()
            prototype_root = f"/World/Prototype{suffix}"
            assert owner.GetReferences().AddInternalReference(prototype_root)
            assert owner.SetInstanceable(True)
        flattened = stage.Flatten()
        assert flattened.Export(str(path))
        if not author_unit_materials:
            return
        output_stage = Usd.Stage.Open(str(path))
        assert output_stage is not None
        for index, suffix in enumerate(("A", "B")):
            authorable_member = _authorable_instance_path(
                output_stage,
                f"/World/Instance{suffix}/Mesh",
            )
            output_material = UsdShade.Material(
                output_stage.GetPrimAtPath(f"/World/Looks/{(unit_a, unit_b)[index]}")
            )
            UsdShade.MaterialBindingAPI.Apply(
                output_stage.GetPrimAtPath(authorable_member)
            ).Bind(output_material)
        assert output_stage.GetRootLayer().Save()

    source = tmp_path / "source.usda"
    output = tmp_path / "output.usda"
    create_flattened_stage(
        source,
        owner_order=("A", "B"),
        author_unit_materials=False,
    )
    create_flattened_stage(
        output,
        owner_order=("B", "A"),
        author_unit_materials=True,
    )

    report = validate_texture_scope_invariants(
        source_asset_path=source,
        output_asset_path=output,
        plan=plan,
    )

    assert report.passed
    assert report.structure_unchanged_outside_target
    assert report.bindings_unchanged

    output_stage = Usd.Stage.Open(str(output))
    assert output_stage is not None
    output_member = Sdf.Path(
        _authorable_instance_path(output_stage, "/World/InstanceA/Mesh")
    )
    output_prototype_root = output_member.GetParentPath()
    assert output_prototype_root != Sdf.Path("/Flattened_Prototype_1")
    output_material = output_prototype_root.AppendPath("Looks/Diffuse")
    relationship = output_stage.GetPrimAtPath(output_material).GetRelationship(
        "texture:testSource"
    )
    assert relationship.SetTargets(
        [Sdf.Path("/World/InstanceA/Looks/Diffuse/PreviewSurface.outputs:surface")]
    )
    assert output_stage.GetRootLayer().Save()

    retargeted = validate_texture_scope_invariants(
        source_asset_path=source,
        output_asset_path=output,
        plan=plan,
    )

    assert not retargeted.passed
    assert not retargeted.non_target_materials_unchanged
    assert any(
        violation.code == "material.non_target_changed"
        and violation.prim_path == "/World/InstanceA/Looks/Diffuse"
        for violation in retargeted.violations
    )

    assert relationship.SetTargets(
        [
            Sdf.Path(
                str(
                    output_material.AppendChild("PreviewSurface").AppendProperty(
                        "outputs:surface"
                    )
                )
            )
        ]
    )
    assert output_stage.GetRootLayer().Save()
    UsdGeom.Xform.Define(
        output_stage,
        str(output_prototype_root.AppendChild("Unexpected")),
    )
    assert output_stage.GetRootLayer().Save()

    changed = validate_texture_scope_invariants(
        source_asset_path=source,
        output_asset_path=output,
        plan=plan,
    )

    assert not changed.passed
    assert not changed.structure_unchanged_outside_target
    assert any(
        violation.code == "structure.changed_outside_target"
        and violation.prim_path.endswith("/Unexpected")
        for violation in changed.violations
    )


def test_scope_validation_preserves_authored_instance_source_and_observer_paths(
    tmp_path: Path,
) -> None:
    from content_agent_workflows.texture.scope_validation import (
        validate_texture_scope_invariants,
    )

    unit_id = "tu_11111111111111111111"
    plan = TexturePlanDocument.model_validate(
        {
            "schema_version": "texture-agent-plan.v1",
            "counts": {"selected_unit_count": 1},
            "decision": {"state": "ready", "execution_allowed": True},
            "selected_units": [
                {
                    "unit_id": unit_id,
                    "unit_mode": "per_material",
                    "material_prim_paths": ["/World/Looks/Material"],
                    "member_prim_paths": ["/World/Instance/Mesh"],
                    "member_subset_paths": [],
                }
            ],
        }
    )

    def create_stage(
        path: Path,
        *,
        authored_source_root: str,
        author_unit_material: bool,
    ) -> None:
        stage = Usd.Stage.CreateNew(str(path))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Scope.Define(stage, "/World/Looks")
        source_material = UsdShade.Material.Define(stage, "/World/Looks/Material")
        bound_material = source_material
        if author_unit_material:
            bound_material = UsdShade.Material.Define(
                stage,
                f"/World/Looks/{unit_id}",
            )
        UsdGeom.Xform.Define(stage, authored_source_root)
        shader = UsdShade.Shader.Define(stage, f"{authored_source_root}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mesh = UsdGeom.Mesh.Define(stage, f"{authored_source_root}/Mesh")
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        observer = UsdGeom.Scope.Define(stage, "/World/Observer").GetPrim()
        observer.CreateRelationship("texture:observedSurface").AddTarget(
            shader.GetPath().AppendProperty("outputs:surface")
        )
        owner = UsdGeom.Xform.Define(stage, "/World/Instance").GetPrim()
        assert owner.GetReferences().AddInternalReference(authored_source_root)
        assert owner.SetInstanceable(True)
        assert stage.GetRootLayer().Save()

    source = tmp_path / "authored-source.usda"
    output = tmp_path / "authored-output.usda"
    create_stage(
        source,
        authored_source_root="/World/AuthoredPrototypeA",
        author_unit_material=False,
    )
    create_stage(
        output,
        authored_source_root="/World/AuthoredPrototypeB",
        author_unit_material=True,
    )

    report = validate_texture_scope_invariants(
        source_asset_path=source,
        output_asset_path=output,
        plan=plan,
    )

    assert not report.passed
    assert not report.structure_unchanged_outside_target
    assert any(
        violation.code == "scope.authored_state_changed_outside_target"
        and violation.prim_path == "/World/Observer"
        for violation in report.violations
    )


def test_apply_provided_generates_for_unique_internal_instance_proxy_member(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from texture_agent.functions.material_discovery import (
        TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP,
    )

    source_path = tmp_path / "source.usda"
    source = Usd.Stage.CreateNew(str(source_path))
    UsdGeom.Xform.Define(source, "/World")
    UsdGeom.Xform.Define(source, "/World/Prototype")
    UsdGeom.Scope.Define(source, "/World/Prototype/Looks")
    material = UsdShade.Material.Define(
        source,
        "/World/Prototype/Looks/Paint",
    )
    mesh = UsdGeom.Mesh.Define(source, "/World/Prototype/Panel")
    mesh.CreatePointsAttr([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    primvars = UsdGeom.PrimvarsAPI(mesh)
    primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    ).Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    instance = UsdGeom.Xform.Define(source, "/World/Instance").GetPrim()
    assert instance.GetReferences().AddInternalReference("/World/Prototype")
    assert instance.SetInstanceable(True)
    assert source.GetRootLayer().Save()

    request = build_texture_capability_request(
        source_asset=source_path,
        output_dir=tmp_path / "run",
        intent="Apply one exact image to the unique instance panel.",
        prim_paths=("/World/Instance/Panel",),
        operations=TextureOperationSelection(
            generate="requested",
            evidence="requested",
            review="requested",
            publish="requested",
        ),
        texture_size=64,
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    unit = preparation.scope_plan.selected_units[0]
    assert unit.material_prim_paths == ("/World/Instance/Looks/Paint",)
    assert unit.member_prim_paths == ("/World/Instance/Panel",)

    provided_path = tmp_path / "provided.png"
    Image.new("RGB", (64, 64), (92, 104, 116)).save(provided_path)
    provided = TextureProvidedImageArtifact(
        unit_id=unit.unit_id,
        channel="albedo",
        artifact=_binding(provided_path),
        producer=TextureProvidedImageProducer(
            provider="outer-image-tool",
            capability="image.generate.v1",
            invocation_id="unique-internal-instance",
        ),
    )
    generator_inputs = TextureGeneratorInputs(
        execution_mode="apply_provided",
        backend=ProvidedImageTextureApplyLeaf.provider_id,
        prompt="Use the exact supplied image.",
        texture_size=64,
        provided_images=(provided,),
    )
    leaf_request = TextureGeneratorLeafRequest(
        outer_plan=preparation_binding,
        preparation=preparation_binding,
        source=request.source,
        source_dependencies=request.source_dependencies,
        intent=request.intent,
        scope_plan=preparation.scope_plan,
        target_unit_ids=(unit.unit_id,),
        generator_inputs=(generator_inputs,),
        output_dir=str(tmp_path / "apply"),
    )

    execution = ProvidedImageTextureApplyLeaf().generate(leaf_request)

    candidate = Usd.Stage.Open(execution.output_asset_path)
    assert candidate is not None
    authorable_member = candidate.GetPrimAtPath("/Flattened_Prototype_1/Panel")
    bound_material = UsdShade.MaterialBindingAPI(
        authorable_member
    ).ComputeBoundMaterial()[0]
    assert str(bound_material.GetPath()) == (
        f"/Flattened_Prototype_1/Looks/{unit.unit_id}"
    )
    aliases = bound_material.GetPrim().GetRelationship(
        TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP
    )
    assert {str(path) for path in aliases.GetTargets()} == {
        "/World/Instance/Looks/Paint"
    }


def test_packaged_provided_image_readback_rejects_member_substitution(
    tmp_path: Path,
) -> None:
    from pxr import Sdf

    provided_path = tmp_path / "provided.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(provided_path)
    supplied_binding = _binding(provided_path)
    member = "textures/unit/albedo.png"
    alternate_member = "textures/unit/alternate.png"
    candidate = tmp_path / "candidate.usdz"
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(member, provided_path.read_bytes())
        archive.writestr(alternate_member, b"different")

    valid = ProvidedImageTextureApplyLeaf._verify_packaged_image_member(
        candidate=candidate,
        asset_path=Sdf.AssetPath(member, f"{candidate}[{member}]"),
        supplied_binding=supplied_binding,
        shader_path="/World/Looks/Paint/OuterProvidedAlbedo_test",
    )
    assert valid == {
        "package_member_path": member,
        "package_member_sha256": supplied_binding.sha256,
        "package_member_size_bytes": supplied_binding.size_bytes,
    }

    with pytest.raises(ValueError, match="package member is missing"):
        ProvidedImageTextureApplyLeaf._verify_packaged_image_member(
            candidate=candidate,
            asset_path=Sdf.AssetPath(
                "textures/unit/missing.png",
                f"{candidate}[textures/unit/missing.png]",
            ),
            supplied_binding=supplied_binding,
            shader_path="/World/Looks/Paint/OuterProvidedAlbedo_missing",
        )

    with pytest.raises(ValueError, match="substituted package member"):
        ProvidedImageTextureApplyLeaf._verify_packaged_image_member(
            candidate=candidate,
            asset_path=Sdf.AssetPath(member, f"{candidate}[{alternate_member}]"),
            supplied_binding=supplied_binding,
            shader_path="/World/Looks/Paint/OuterProvidedAlbedo_swapped",
        )

    substituted_package = tmp_path / "substituted.usdz"
    with zipfile.ZipFile(
        substituted_package,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(member, provided_path.read_bytes())
    with pytest.raises(ValueError, match="outside its exact candidate package"):
        ProvidedImageTextureApplyLeaf._verify_packaged_image_member(
            candidate=candidate,
            asset_path=Sdf.AssetPath(
                member,
                f"{substituted_package}[{member}]",
            ),
            supplied_binding=supplied_binding,
            shader_path="/World/Looks/Paint/OuterProvidedAlbedo_substituted",
        )

    mismatched_binding = supplied_binding.model_copy(update={"sha256": "f" * 64})
    with pytest.raises(ValueError, match="bytes differ from its supplied binding"):
        ProvidedImageTextureApplyLeaf._verify_packaged_image_member(
            candidate=candidate,
            asset_path=Sdf.AssetPath(member, f"{candidate}[{member}]"),
            supplied_binding=mismatched_binding,
            shader_path="/World/Looks/Paint/OuterProvidedAlbedo_mismatched",
        )


def test_copy_exact_replaces_existing_extracted_member(tmp_path: Path) -> None:
    source = tmp_path / "reviewed.png"
    destination = tmp_path / "extracted" / "textures" / "albedo.png"
    source.write_bytes(b"reviewed-texture-bytes")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"prior-texture-bytes")

    ProvidedImageTextureApplyLeaf._copy_exact(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert file_sha256(destination) == file_sha256(source)
    assert not destination.with_name(f".{destination.name}.replacement").exists()


def test_apply_provided_repackages_prior_candidate_without_nested_usdz(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from pxr import Usd, UsdShade
    from texture_agent.functions.material_discovery import (
        TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP,
        discover_effective_materials,
    )

    from content_agent_workflows.texture.agentic_readiness import (
        TexturePlanUnitDisposition,
        _narrow_texture_scope_plan,
    )

    source = tmp_path / "two-materials.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "First" {}
        def Material "Second" {}
    }
    def Mesh "FirstPanel" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </World/Looks/First>
        point3f[] points = [(-1, -1, 0), (0, -1, 0), (0, 1, 0), (-1, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }
    def Mesh "SecondPanel" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </World/Looks/Second>
        point3f[] points = [(0, -1, 0), (1, -1, 0), (1, 1, 0), (0, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }
}
""",
        encoding="utf-8",
    )
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Apply one exact image to each selected material.",
        material_prim_paths=("/World/Looks/First", "/World/Looks/Second"),
        operations=TextureOperationSelection(
            generate="requested",
            evidence="requested",
            review="requested",
            publish="requested",
        ),
        texture_size=64,
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    units = preparation.scope_plan.selected_units
    assert len(units) == 2
    narrowed = _narrow_texture_scope_plan(
        preparation.scope_plan,
        (
            TexturePlanUnitDisposition(
                unit_id=units[0].unit_id,
                material_prim_paths=units[0].material_prim_paths,
                member_prim_paths=units[0].member_prim_paths,
                member_subset_paths=units[0].member_subset_paths,
                action="preserve",
                rationale="Verify one narrowed deterministic scope.",
            ),
        ),
    )
    assert narrowed.counts.selected_unit_count == 1
    assert narrowed.counts.selected_material_count == 1
    assert narrowed.counts.planned_generation_job_count == 1
    images: list[Path] = []
    provided: list[TextureProvidedImageArtifact] = []
    for index, unit in enumerate(units):
        image = tmp_path / f"provided-{index}.png"
        Image.new("RGB", (64, 64), (20 + index * 80, 40, 60)).save(image)
        images.append(image)
        provided.append(
            TextureProvidedImageArtifact(
                unit_id=unit.unit_id,
                channel="albedo",
                artifact=_binding(image),
                producer=TextureProvidedImageProducer(
                    provider="outer-image-tool",
                    capability="image.generate.v1",
                    invocation_id=f"provided-{index}",
                ),
            )
        )

    current_source = request.source
    candidates: list[Path] = []
    for index, unit in enumerate(units):
        generator_inputs = TextureGeneratorInputs(
            execution_mode="apply_provided",
            backend=ProvidedImageTextureApplyLeaf.provider_id,
            prompt=f"Exact image {index}.",
            texture_size=64,
            provided_images=(provided[index],),
        )
        disposition = TexturePlanUnitDisposition(
            unit_id=unit.unit_id,
            material_prim_paths=unit.material_prim_paths,
            member_prim_paths=unit.member_prim_paths,
            member_subset_paths=unit.member_subset_paths,
            action="apply_provided",
            rationale="Apply the exact selected image once.",
            requested_appearance=f"Apply exact image {index}.",
            generator_inputs=generator_inputs,
        )
        leaf_request = TextureGeneratorLeafRequest(
            outer_plan=preparation_binding,
            preparation=preparation_binding,
            source=current_source,
            source_dependencies=(),
            intent=request.intent,
            scope_plan=_narrow_texture_scope_plan(
                preparation.scope_plan,
                (disposition,),
            ),
            target_unit_ids=(unit.unit_id,),
            generator_inputs=(generator_inputs,),
            output_dir=str(tmp_path / f"apply-{index}"),
        )
        execution = ProvidedImageTextureApplyLeaf().generate(leaf_request)
        candidate = Path(execution.output_asset_path)
        candidates.append(candidate)
        current_source = _binding(candidate)

    final_candidate = candidates[-1]
    with zipfile.ZipFile(final_candidate) as package:
        members = package.namelist()
        assert not any(name.endswith(".usdz") for name in members)
        packaged_pngs = {
            package.read(name)
            for name in members
            if Path(name).suffix.lower() == ".png"
        }
    assert packaged_pngs == {image.read_bytes() for image in images}
    stage = Usd.Stage.Open(str(final_candidate))
    assert stage is not None
    authored_files = []
    for prim in stage.Traverse():
        if "OuterProvidedAlbedo" not in str(prim.GetPath()):
            continue
        value = UsdShade.Shader(prim).GetInput("file").Get()
        assert value.resolvedPath.startswith(f"{final_candidate.resolve()}[")
        authored_files.append(value.path)
    assert len(authored_files) == 2
    discovery = discover_effective_materials(stage)
    assert {
        alias
        for material in discovery.effective_materials
        for alias in material.material_alias_paths
        if alias in {"/World/Looks/First", "/World/Looks/Second"}
    } == {"/World/Looks/First", "/World/Looks/Second"}
    for semantic_path in ("/World/Looks/First", "/World/Looks/Second"):
        generated = next(
            material
            for material in discovery.effective_materials
            if semantic_path in material.material_alias_paths
        )
        relation = stage.GetPrimAtPath(generated.prim_path).GetRelationship(
            TEXTURE_SEMANTIC_MATERIAL_ALIASES_RELATIONSHIP
        )
        assert semantic_path in {str(path) for path in relation.GetTargets()}
    assert "_EnqueueDependency" not in capfd.readouterr().err


def test_apply_provided_rejects_scope_identity_and_service_substitution(
    tmp_path: Path,
) -> None:
    image = tmp_path / "provided.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(image)
    unit_id = "tu_0123456789abcdef0123"
    producer = TextureProvidedImageProducer(
        provider="outer-image-tool",
        capability="image.generate.v1",
        invocation_id="fresh-invocation",
    )
    provided = TextureProvidedImageArtifact(
        unit_id=unit_id,
        channel="albedo",
        artifact=_binding(image),
        producer=producer,
    )
    with pytest.raises(ValueError, match="apply_provided inputs require"):
        TextureGeneratorInputs(
            execution_mode="apply_provided",
            backend="outer_provided_image_apply",
            prompt="orange",
        )
    with pytest.raises(ValueError, match="engine, seed, or provider parameters"):
        TextureGeneratorInputs(
            execution_mode="apply_provided",
            backend="outer_provided_image_apply",
            engine="must-not-run",
            prompt="orange",
            provided_images=(provided,),
        )
    for field, value in (("channel", "normal"), ("role", "appearance_reference")):
        payload = provided.model_dump(mode="json")
        payload[field] = value
        with pytest.raises(ValueError):
            TextureProvidedImageArtifact.model_validate(payload)
    with pytest.raises(ValueError, match="exactly one ordered image"):
        TexturePlanTarget(
            unit_id="tu_aaaaaaaaaaaaaaaaaaaa",
            material_prim_paths=("/World/Looks/Paint",),
            member_prim_paths=("/World/Panel",),
            requested_appearance="orange",
            generator_inputs=TextureGeneratorInputs(
                execution_mode="apply_provided",
                backend="outer_provided_image_apply",
                prompt="orange",
                texture_size=64,
                provided_images=(provided,),
            ),
        )
    with pytest.raises(ValueError, match="exactly one ordered image"):
        TexturePlanTarget(
            unit_id=unit_id,
            material_prim_paths=("/World/Looks/Paint",),
            member_prim_paths=("/World/Panel",),
            requested_appearance="orange",
            generator_inputs=TextureGeneratorInputs(
                execution_mode="apply_provided",
                backend="outer_provided_image_apply",
                prompt="orange",
                texture_size=64,
                provided_images=(provided, provided),
            ),
        )
    with pytest.raises(ValueError, match="cannot include provided candidate images"):
        TextureGeneratorInputs(
            backend="service-generator",
            prompt="orange",
            provided_images=(provided,),
        )

    source = _source(tmp_path / "source.usda")
    second_image = tmp_path / "provided-second.png"
    Image.new("RGB", (64, 64), (4, 5, 6)).save(second_image)
    second_unit_id = "tu_aaaaaaaaaaaaaaaaaaaa"
    second_provided = provided.model_copy(
        update={
            "unit_id": second_unit_id,
            "artifact": _binding(second_image),
        }
    )
    scope_payload = {
        "schema_version": "texture-agent-plan.v1",
        "request": {
            "source": {
                "source_asset": str(source),
                "source_asset_sha256": _binding(source).sha256,
            },
            "discovery_mode": "explicit",
            "unit_mode": "per_material",
            "explicit_material_paths": ["/World/Looks/Paint"],
            "explicit_prim_paths": [],
        },
        "counts": {"selected_unit_count": 2},
        "selected_units": [
            {
                "unit_id": unit_id,
                "material_prim_paths": ["/World/Looks/Paint"],
                "member_prim_paths": ["/World/Panel"],
                "member_subset_paths": [],
            },
            {
                "unit_id": second_unit_id,
                "material_prim_paths": ["/World/Looks/Paint"],
                "member_prim_paths": ["/World/Panel"],
                "member_subset_paths": [],
            },
        ],
        "decision": {"state": "ready", "execution_allowed": True},
    }
    scope_plan = TexturePlanDocument.model_validate(scope_payload)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    preparation_path = tmp_path / "preparation.json"
    preparation_path.write_text("{}\n", encoding="utf-8")
    leaf_request = TextureGeneratorLeafRequest(
        outer_plan=_binding(plan_path),
        preparation=_binding(preparation_path),
        source=_binding(source),
        intent="apply exact supplied images",
        scope_plan=scope_plan,
        target_unit_ids=(unit_id, second_unit_id),
        generator_inputs=(
            TextureGeneratorInputs(
                execution_mode="apply_provided",
                backend=ProvidedImageTextureApplyLeaf.provider_id,
                prompt="first",
                texture_size=64,
                provided_images=(provided,),
            ),
            TextureGeneratorInputs(
                execution_mode="apply_provided",
                backend=ProvidedImageTextureApplyLeaf.provider_id,
                prompt="second",
                texture_size=64,
                provided_images=(second_provided,),
            ),
        ),
        output_dir=str(tmp_path / "overlap-output"),
    )
    with pytest.raises(ValueError, match="overlap material path"):
        ProvidedImageTextureApplyLeaf().generate(leaf_request)


def test_texture_capability_binding_rejects_symlinked_source(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.usda")
    symlink = tmp_path / "source-link.usda"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="must not be a symlink"):
        build_texture_capability_request(
            source_asset=symlink,
            output_dir=tmp_path / "run",
            intent="must fail before preparation",
            operations=TextureOperationSelection(),
        )


def test_service_proposal_uses_same_auto_prompt_scope_as_preparation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.usda")
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Propose an appearance for the exact explicit material.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=TextureOperationSelection(propose="requested"),
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )

    class RecordingProvider:
        def __init__(self) -> None:
            self.request: Any | None = None

        def plan(self, workflow_request: Any) -> TexturePlanDocument:
            self.request = workflow_request
            return preparation.scope_plan

        def export_resume_state(self, _plan: TexturePlanDocument) -> dict[str, Any]:
            return {"session_id": "proposal-session"}

    provider = RecordingProvider()
    proposal, _proposal_binding = request_texture_provider_proposal(
        preparation,
        preparation_binding=preparation_binding,
        provider=provider,
        provider_id="recording-service",
    )
    assert provider.request is not None
    assert provider.request.metadata["auto_prompt_enabled"] is True
    assert (
        proposal.proposal.selected_unit_ids == preparation.scope_plan.selected_unit_ids
    )


def test_reference_drift_is_rejected_before_outer_plan_execution(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.usda")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"original")
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Use the reference.",
        material_prim_paths=("/World/Looks/Paint",),
        reference_artifacts=(("appearance_reference", reference),),
        operations=TextureOperationSelection(
            generate="requested",
            evidence="requested",
            review="requested",
            publish="requested",
        ),
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    reference.write_bytes(b"changed")
    inputs = TextureGeneratorInputs(
        backend="explicit-generator",
        prompt="use changed reference",
        reference_artifacts=tuple(
            item.artifact for item in request.reference_artifacts
        ),
    )
    unit = preparation.inspection.units[0]
    plan = TextureOuterPlan(
        preparation=preparation_binding,
        source=request.source,
        scope_plan_digest=preparation.scope_plan_digest,
        reference_artifacts=request.reference_artifacts,
        operations=request.operations,
        targets=(
            TexturePlanTarget(
                unit_id=unit.unit_id,
                material_prim_paths=unit.material_prim_paths,
                member_prim_paths=unit.member_prim_paths,
                requested_appearance="changed",
                generator_inputs=inputs,
            ),
        ),
        preservation=TexturePreservationConstraints(),
        acceptance=TextureAcceptanceCriteria(appearance_requirements=("match",)),
        capability_constraints=preparation.inspection.capability_constraints,
        stop_policy="review first",
    )
    plan_binding = _write_model(Path(request.output_dir) / "plan.json", plan)
    generator = _ExplicitGenerator(source)
    with pytest.raises(
        ValueError, match="reference appearance_reference bytes changed"
    ):
        invoke_texture_generator(
            plan,
            outer_plan_binding=plan_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            generator=generator,
        )
    assert generator.calls == 0


def test_source_drift_is_rejected_before_proposal_provider_call(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path / "source.usda")
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Request advisory Texture semantics.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=TextureOperationSelection(propose="requested"),
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    provider = _ProposalProvider(preparation.scope_plan)
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# changed after preparation\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Texture source bytes changed"):
        request_texture_provider_proposal(
            preparation,
            preparation_binding=preparation_binding,
            provider=provider,
            provider_id="advisory-service",
        )
    assert provider.calls == 0


def test_dependency_drift_is_rejected_before_generator_leaf_call(
    tmp_path: Path,
) -> None:
    dependency = tmp_path / "shared_geometry.usda"
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "SharedGeometry" {}\n',
        encoding="utf-8",
    )
    source = _source(tmp_path / "source.usda")
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            '    defaultPrim = "World"\n',
            '    defaultPrim = "World"\n    subLayers = [@shared_geometry.usda@]\n',
            1,
        ),
        encoding="utf-8",
    )
    request = build_texture_capability_request(
        source_asset=source,
        output_dir=tmp_path / "run",
        intent="Generate against the exact composed source.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=TextureOperationSelection(
            generate="requested",
            evidence="requested",
            review="requested",
            publish="requested",
        ),
    )
    preparation, preparation_binding = prepare_texture_scope(
        request,
        inspector=_ProviderFreeInspector(),
    )
    unit = preparation.inspection.units[0]
    outer_plan = TextureOuterPlan(
        preparation=preparation_binding,
        source=request.source,
        scope_plan_digest=preparation.scope_plan_digest,
        reference_artifacts=request.reference_artifacts,
        operations=request.operations,
        targets=(
            TexturePlanTarget(
                unit_id=unit.unit_id,
                material_prim_paths=unit.material_prim_paths,
                member_prim_paths=unit.member_prim_paths,
                member_subset_paths=unit.member_subset_paths,
                requested_appearance="exact composed source",
                generator_inputs=TextureGeneratorInputs(
                    backend="explicit-generator",
                    prompt="preserve the exact composed source",
                ),
            ),
        ),
        preservation=TexturePreservationConstraints(),
        acceptance=TextureAcceptanceCriteria(appearance_requirements=("exact",)),
        capability_constraints=preparation.inspection.capability_constraints,
        stop_policy="review first",
    )
    outer_plan_binding = _write_model(
        Path(request.output_dir) / "plan.json",
        outer_plan,
    )
    dependency.write_text(
        '#usda 1.0\n\ndef Xform "ChangedGeometry" {}\n',
        encoding="utf-8",
    )
    generator = _ExplicitGenerator(source)

    with pytest.raises(RuntimeError, match="dependency closure identity changed"):
        invoke_texture_generator(
            outer_plan,
            outer_plan_binding=outer_plan_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            generator=generator,
        )
    assert generator.calls == 0
