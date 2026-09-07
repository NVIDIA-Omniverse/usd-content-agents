# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from texture_agent.functions.detail_policy import apply_detail_policy_to_prompt

import content_agent_workflows.texture.agentic_readiness as texture_agentic_readiness
from content_agent_workflows.asset_composition import (
    AssetCompositionStateError,
    bind_usd_dependency_closure,
)
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.texture import (
    TextureAcceptanceCriteria,
    TextureAdapterCallLedger,
    TextureAgenticAdapterResult,
    TextureAgenticCleanupReceipt,
    TextureAgenticEvidenceReceipt,
    TextureAgenticEvidenceRequirements,
    TextureAgenticGeneratorLeafAdapter,
    TextureAgenticPlan,
    TextureAgenticPublicationReceipt,
    TextureAgenticReviewReceipt,
    TextureAgenticSavedStageReadback,
    TextureAgenticUnitArtifacts,
    TextureAgenticUnitReview,
    TextureCapabilityRequest,
    TextureExecutionResult,
    TextureGeneratorInputs,
    TextureInspectionResult,
    TextureInspectionUnit,
    TextureOperationOutcome,
    TextureOperationSelection,
    TextureOperationStatus,
    TexturePlanCounts,
    TexturePlanDecision,
    TexturePlanDocument,
    TexturePlanSelectedUnit,
    TexturePlanUnitDisposition,
    TexturePreparationPacket,
    TexturePreservationConstraints,
    TextureProvidedImageArtifact,
    TextureProvidedImageProducer,
    TextureUnitArtifact,
    TextureUnitRenderEvidence,
    accept_texture_agentic_plan,
    bind_texture_agentic_artifact,
    execute_texture_agentic_plan,
    prepare_texture_agentic_source,
    record_texture_agentic_cleanup,
    record_texture_agentic_evidence,
    record_texture_agentic_publication,
    record_texture_agentic_review,
    record_texture_agentic_saved_stage_readback,
    seal_texture_agentic_terminal_receipt,
    validate_texture_agentic_evidence,
    validate_texture_agentic_plan,
    validate_texture_agentic_saved_stage_readback,
    validate_texture_agentic_source_preparation,
)

UNIT_A = "tu_aaaaaaaaaaaaaaaaaaaa"
UNIT_B = "tu_bbbbbbbbbbbbbbbbbbbb"
UNIT_C = "tu_cccccccccccccccccccc"


def _write(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _binding(path: Path) -> ExecutionArtifactBinding:
    return bind_texture_agentic_artifact(path)


def test_source_preparation_authors_missing_uvs_before_reasoning(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.usda"
    albedo = _write(tmp_path / "albedo.png", b"source-albedo")
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 1, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    material = UsdShade.Material.Define(stage, "/World/Looks/Material")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    texture = UsdShade.Shader.Define(stage, "/World/Looks/Material/Albedo")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(albedo.name)
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    assert stage.GetRootLayer().Save()

    receipt, receipt_binding = prepare_texture_agentic_source(
        source,
        output_dir=tmp_path / "source-preparation",
        target_prim_paths=("/World/Mesh",),
    )

    validate_texture_agentic_source_preparation(
        receipt,
        receipt_binding=receipt_binding,
    )
    assert receipt.original_source == _binding(source)
    assert receipt.effective_source != receipt.original_source
    assert receipt.provider_invoked is False
    reopened = Usd.Stage.Open(receipt.effective_source.path)
    primvar = UsdGeom.PrimvarsAPI(
        reopened.GetPrimAtPath("/World/Mesh")
    ).FindPrimvarWithInheritance("st")
    assert primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying
    packaged_texture = (
        UsdShade.Shader(reopened.GetPrimAtPath("/World/Looks/Material/Albedo"))
        .GetInput("file")
        .Get()
    )
    assert packaged_texture.path
    assert packaged_texture.resolvedPath
    assert bind_usd_dependency_closure(receipt.effective_source.path) == []

    repeated, _ = prepare_texture_agentic_source(
        source,
        output_dir=tmp_path / "source-preparation-repeat",
        target_prim_paths=("/World/Mesh",),
    )
    assert repeated.effective_source.sha256 == receipt.effective_source.sha256
    assert "_EnqueueDependency" not in capfd.readouterr().err


def test_source_preparation_cleans_localization_scratch_on_packaging_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 1, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    material = UsdShade.Material.Define(stage, "/World/Looks/Material")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    assert stage.GetRootLayer().Save()

    def fail_after_extract(
        _package_root: Path,
        _root_member: Path,
        output_path: Path,
    ) -> None:
        output_path.write_bytes(b"partial package")
        raise RuntimeError("injected package write failure")

    monkeypatch.setattr(
        "world_understanding.utils.usd.package.write_usdz_package_from_directory",
        fail_after_extract,
    )
    output_dir = tmp_path / "source-preparation"

    with pytest.raises(RuntimeError, match="injected package write failure"):
        prepare_texture_agentic_source(
            source,
            output_dir=output_dir,
            target_prim_paths=("/World/Mesh",),
        )

    assert not (output_dir / ".localized_texture.usdz").exists()
    assert not (output_dir / ".localized_package").exists()
    assert not (output_dir / "prepared_texture.usdz").exists()


def _operation_status(inspection: ExecutionArtifactBinding) -> TextureOperationStatus:
    return TextureOperationStatus(
        operations=(
            TextureOperationOutcome(
                operation="inspect",
                state="completed",
                artifact=inspection,
                detail="provider-free preparation completed",
            ),
            *(
                TextureOperationOutcome(
                    operation=operation,
                    state="not_requested",
                    detail="semantic selection is deferred to the reasoning child",
                )
                for operation in (
                    "propose",
                    "generate",
                    "evidence",
                    "critique",
                    "review",
                    "publish",
                )
            ),
        )
    )


def _preparation(
    tmp_path: Path,
    *,
    with_source_dependency: bool = False,
) -> tuple[TexturePreparationPacket, ExecutionArtifactBinding]:
    dependency = tmp_path / "dependency.usda"
    if with_source_dependency:
        _write(
            dependency,
            b'#usda 1.0\n\ndef Xform "Dependency" {}\n',
        )
    source_path = _write(
        tmp_path / "source.usda",
        (
            b"#usda 1.0\n(\n    subLayers = [@dependency.usda@]\n)\n"
            if with_source_dependency
            else b"#usda 1.0\n"
        ),
    )
    source = _binding(source_path)
    before = _binding(_write(tmp_path / "before.png", b"ovrtx"))
    facts = _binding(_write(tmp_path / "facts.json", b"{}\n"))
    provided = _binding(_write(tmp_path / "provided.png", b"provided-image"))
    units = (
        TextureInspectionUnit(
            unit_id=UNIT_A,
            material_prim_paths=("/World/Looks/A",),
            member_prim_paths=("/World/A",),
            uv_status="ready",
            proposed_generator_inputs=TextureGeneratorInputs(
                backend="selected-provider",
                prompt=apply_detail_policy_to_prompt(
                    "Texture only exact prepared surfaces.", "surface_only"
                ),
            ),
        ),
        TextureInspectionUnit(
            unit_id=UNIT_B,
            material_prim_paths=("/World/Looks/B",),
            member_prim_paths=("/World/B",),
            uv_status="ready",
            proposed_generator_inputs=TextureGeneratorInputs(
                backend="not_requested",
                prompt=apply_detail_policy_to_prompt(
                    "provider-neutral preparation", "surface_only"
                ),
            ),
        ),
        TextureInspectionUnit(
            unit_id=UNIT_C,
            material_prim_paths=("/World/Looks/C",),
            member_prim_paths=("/World/C",),
            uv_status="ready",
            proposed_generator_inputs=TextureGeneratorInputs(
                backend="selected-provider",
                prompt=apply_detail_policy_to_prompt(
                    "Texture only exact prepared surfaces.", "surface_only"
                ),
            ),
        ),
    )
    scope_plan = TexturePlanDocument(
        counts=TexturePlanCounts(selected_unit_count=3),
        selected_units=tuple(
            TexturePlanSelectedUnit.model_validate(
                {
                    "unit_id": unit.unit_id,
                    "material_prim_paths": list(unit.material_prim_paths),
                    "member_prim_paths": list(unit.member_prim_paths),
                    "member_subset_paths": list(unit.member_subset_paths),
                }
            )
            for unit in units
        ),
        decision=TexturePlanDecision(state="ready", execution_allowed=True),
        request={
            "discovery_mode": "explicit",
            "unit_mode": "per_material",
            "explicit_material_paths": [
                path for unit in units for path in unit.material_prim_paths
            ],
            "explicit_prim_paths": [],
        },
    )
    request = TextureCapabilityRequest(
        source=source,
        source_dependencies=tuple(bind_usd_dependency_closure(source_path)),
        output_dir=str(tmp_path / "run"),
        intent="Texture only exact prepared surfaces.",
        material_prim_paths=tuple(
            path for unit in units for path in unit.material_prim_paths
        ),
        operations=TextureOperationSelection(),
        metadata={
            "content_workflow_cli_texture_agentic_provided_candidates": [
                {
                    "target_path": "/World/Looks/C",
                    "artifact": provided.model_dump(mode="json"),
                    "producer": {
                        "provider": "outer-image-generator",
                        "capability": "image.generate.v1",
                        "invocation_id": "outer-image-001",
                        "provenance": {},
                    },
                }
            ]
        },
    )
    inspection = TextureInspectionResult(
        source=source,
        proposal_plan_digest="1" * 64,
        units=units,
        before_render_artifacts=(before,),
        inspection_artifacts=(facts,),
        capability_constraints=(
            "surface texturing only",
            "target scope is immutable after acceptance",
        ),
        renderer_metadata={"renderer": "ovrtx", "current_run": True},
    )
    packet = TexturePreparationPacket(
        request=request,
        request_digest="2" * 64,
        scope_plan=scope_plan,
        scope_plan_digest="1" * 64,
        inspection=inspection,
        operation_status=_operation_status(facts),
    )
    path = tmp_path / "run" / "texture_preparation.json"
    atomic_write_json(path, packet)
    return packet, _binding(path)


def _mixed_plan(
    tmp_path: Path,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
) -> TextureAgenticPlan:
    provided_path = _write(tmp_path / "provided.png", b"provided-image")
    provided = TextureProvidedImageArtifact(
        unit_id=UNIT_C,
        channel="albedo",
        artifact=_binding(provided_path),
        producer=TextureProvidedImageProducer(
            provider="outer-image-generator",
            capability="image.generate.v1",
            invocation_id="outer-image-001",
        ),
    )
    units = preparation.inspection.units
    return TextureAgenticPlan(
        preparation=preparation_binding,
        source=preparation.request.source,
        scope_plan_digest=preparation.scope_plan_digest,
        dispositions=(
            TexturePlanUnitDisposition(
                unit_id=UNIT_A,
                material_prim_paths=units[0].material_prim_paths,
                member_prim_paths=units[0].member_prim_paths,
                action="generate",
                rationale="The requested appearance requires one generated texture.",
                requested_appearance="Lightly scuffed brushed metal.",
                generator_inputs=TextureGeneratorInputs(
                    backend="selected-provider",
                    prompt=apply_detail_policy_to_prompt(
                        "Lightly scuffed brushed metal.", "surface_only"
                    ),
                ),
            ),
            TexturePlanUnitDisposition(
                unit_id=UNIT_B,
                material_prim_paths=units[1].material_prim_paths,
                member_prim_paths=units[1].member_prim_paths,
                action="preserve",
                rationale="The existing authored surface already matches intent.",
            ),
            TexturePlanUnitDisposition(
                unit_id=UNIT_C,
                material_prim_paths=units[2].material_prim_paths,
                member_prim_paths=units[2].member_prim_paths,
                action="apply_provided",
                rationale="Use the exact outer-provided candidate image.",
                requested_appearance="Exact supplied blue coating.",
                generator_inputs=TextureGeneratorInputs(
                    execution_mode="apply_provided",
                    backend="outer_provided_image_apply",
                    prompt=apply_detail_policy_to_prompt(
                        "Exact supplied blue coating.", "surface_only"
                    ),
                    provided_images=(provided,),
                ),
            ),
        ),
        preservation=TexturePreservationConstraints(),
        acceptance=TextureAcceptanceCriteria(
            appearance_requirements=("Match the requested per-unit appearance.",),
        ),
        evidence=TextureAgenticEvidenceRequirements(
            required_views=("+x-y+z", "+z"),
        ),
        capability_constraints=preparation.inspection.capability_constraints,
    )


def _accept(
    tmp_path: Path,
    preparation: TexturePreparationPacket,
    preparation_binding: ExecutionArtifactBinding,
    plan: TextureAgenticPlan,
) -> tuple[Any, ExecutionArtifactBinding]:
    plan_path = tmp_path / "run" / "texture_plan.json"
    atomic_write_json(plan_path, plan)
    return accept_texture_agentic_plan(
        plan_path,
        preparation=preparation,
        preparation_binding=preparation_binding,
        output_path=tmp_path / "run" / "accepted_texture_plan.json",
    )


def test_agentic_plan_accepts_mixed_actions_and_exact_prepared_order(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)

    validate_texture_agentic_plan(
        plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        plan,
    )

    assert accepted.plan.unit_ids == (UNIT_A, UNIT_B, UNIT_C)
    assert accepted.plan.units_for("generate") == (UNIT_A,)
    assert accepted.plan.units_for("preserve") == (UNIT_B,)
    assert accepted.plan.units_for("apply_provided") == (UNIT_C,)
    assert accepted_binding.path.endswith("accepted_texture_plan.json")


def test_agentic_plan_maps_material_alias_policies_to_exact_prepared_units(
    tmp_path: Path,
) -> None:
    preparation, _ = _preparation(tmp_path)
    aliases = (
        "/World/InstanceA/Looks/A",
        "/World/InstanceB/Looks/B",
        "/World/InstanceC/Looks/C",
    )
    units = tuple(
        unit.model_copy(update={"material_alias_paths": (alias,)})
        for unit, alias in zip(preparation.inspection.units, aliases, strict=True)
    )
    inspection = preparation.inspection.model_copy(update={"units": units})
    provided_inventory = preparation.request.metadata[
        "content_workflow_cli_texture_agentic_provided_candidates"
    ]
    assert isinstance(provided_inventory, list)
    provided_entry = {**provided_inventory[0], "target_path": aliases[2]}
    request = preparation.request.model_copy(
        update={
            "metadata": {
                **preparation.request.metadata,
                "content_workflow_cli_texture_agentic_action_policy": {
                    aliases[0]: "generate",
                    aliases[1]: "preserve",
                    aliases[2]: "apply_provided",
                },
                "content_workflow_cli_texture_agentic_appearance_policy": {
                    aliases[0]: "Lightly scuffed brushed metal.",
                    aliases[2]: "Exact supplied blue coating.",
                },
                "content_workflow_cli_texture_agentic_provided_candidates": [
                    provided_entry
                ],
            }
        }
    )
    preparation = preparation.model_copy(
        update={"request": request, "inspection": inspection}
    )
    preparation_path = tmp_path / "run" / "alias_texture_preparation.json"
    atomic_write_json(preparation_path, preparation)
    preparation_binding = _binding(preparation_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)

    validate_texture_agentic_plan(
        plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )


def test_agentic_plan_enforces_exact_per_unit_appearance_policy(
    tmp_path: Path,
) -> None:
    preparation, _ = _preparation(tmp_path)
    request = preparation.request.model_copy(
        update={
            "metadata": {
                **preparation.request.metadata,
                "content_workflow_cli_texture_agentic_appearance_policy": {
                    "/World/A": "Lightly scuffed brushed metal.",
                    "/World/C": "Exact supplied blue coating.",
                },
            }
        }
    )
    preparation = preparation.model_copy(update={"request": request})
    preparation_path = tmp_path / "run" / "appearance_texture_preparation.json"
    atomic_write_json(preparation_path, preparation)
    preparation_binding = _binding(preparation_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)

    validate_texture_agentic_plan(
        plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )

    paraphrased = plan.model_copy(
        update={
            "dispositions": (
                plan.dispositions[0].model_copy(
                    update={"requested_appearance": "Brushed metal with scuffs."}
                ),
                *plan.dispositions[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="exact per-unit appearance policy"):
        validate_texture_agentic_plan(
            paraphrased,
            preparation=preparation,
            preparation_binding=preparation_binding,
        )


def test_agentic_plan_coalesces_agreeing_actions_for_one_per_material_unit(
    tmp_path: Path,
) -> None:
    preparation, _ = _preparation(tmp_path)
    first_unit = preparation.inspection.units[0].model_copy(
        update={"member_prim_paths": ("/World/A", "/World/A2")}
    )
    inspection = preparation.inspection.model_copy(
        update={
            "units": (first_unit, *preparation.inspection.units[1:]),
        }
    )
    first_selected = preparation.scope_plan.selected_units[0].model_copy(
        update={"member_prim_paths": first_unit.member_prim_paths}
    )
    scope_plan = preparation.scope_plan.model_copy(
        update={
            "selected_units": (
                first_selected,
                *preparation.scope_plan.selected_units[1:],
            )
        }
    )
    metadata = {
        **preparation.request.metadata,
        "content_workflow_cli_texture_agentic_action_policy": {
            "/World/A": "generate",
            "/World/A2": "generate",
        },
    }
    request = preparation.request.model_copy(update={"metadata": metadata})
    preparation = preparation.model_copy(
        update={
            "request": request,
            "scope_plan": scope_plan,
            "inspection": inspection,
        }
    )
    preparation_path = tmp_path / "run" / "shared_texture_preparation.json"
    atomic_write_json(preparation_path, preparation)
    preparation_binding = _binding(preparation_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)

    validate_texture_agentic_plan(
        plan,
        preparation=preparation,
        preparation_binding=preparation_binding,
    )

    conflicting_request = request.model_copy(
        update={
            "metadata": {
                **metadata,
                "content_workflow_cli_texture_agentic_action_policy": {
                    "/World/A": "generate",
                    "/World/A2": "preserve",
                },
            }
        }
    )
    conflicting = preparation.model_copy(update={"request": conflicting_request})
    conflicting_path = tmp_path / "run" / "conflicting_texture_preparation.json"
    atomic_write_json(conflicting_path, conflicting)
    conflicting_binding = _binding(conflicting_path)
    conflicting_plan = plan.model_copy(update={"preparation": conflicting_binding})

    with pytest.raises(ValueError, match="targets for one prepared unit conflict"):
        validate_texture_agentic_plan(
            conflicting_plan,
            preparation=conflicting,
            preparation_binding=conflicting_binding,
        )


def test_agentic_plan_revalidates_complete_source_dependency_closure(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(
        tmp_path,
        with_source_dependency=True,
    )
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    dependency = tmp_path / "dependency.usda"
    dependency.write_text(
        dependency.read_text(encoding="utf-8") + "\n# changed after inspection\n",
        encoding="utf-8",
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="dependency closure identity changed",
    ):
        validate_texture_agentic_plan(
            plan,
            preparation=preparation,
            preparation_binding=preparation_binding,
        )


def test_agentic_plan_rejects_cross_unit_generation_prompt(tmp_path: Path) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    generate = plan.dispositions[0]
    assert generate.generator_inputs is not None
    changed_inputs = generate.generator_inputs.model_copy(
        update={
            "prompt": apply_detail_policy_to_prompt(
                "Exact supplied blue coating.", "surface_only"
            )
        }
    )
    changed = generate.model_copy(update={"generator_inputs": changed_inputs})
    plan = plan.model_copy(update={"dispositions": (changed, *plan.dispositions[1:])})

    with pytest.raises(ValueError, match="unit-specific requested appearance"):
        validate_texture_agentic_plan(
            plan,
            preparation=preparation,
            preparation_binding=preparation_binding,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("omission", "every prepared unit"),
        ("unsafe_target", "target is unsafe"),
        ("stale_preparation", "stale preparation"),
    ],
)
def test_agentic_plan_fails_before_execution(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    if failure == "omission":
        plan = plan.model_copy(update={"dispositions": plan.dispositions[:-1]})
    elif failure == "unsafe_target":
        changed = plan.dispositions[0].model_copy(
            update={"member_prim_paths": ("/World/Unsafe",)}
        )
        plan = plan.model_copy(
            update={"dispositions": (changed, *plan.dispositions[1:])}
        )
    else:
        plan = plan.model_copy(
            update={
                "preparation": preparation_binding.model_copy(
                    update={"sha256": "f" * 64}
                )
            }
        )

    with pytest.raises(ValueError, match=message):
        validate_texture_agentic_plan(
            plan,
            preparation=preparation,
            preparation_binding=preparation_binding,
        )


class _RecordingAdapter:
    def __init__(self, action: str, calls: list[tuple[str, tuple[str, ...]]]) -> None:
        self.action = action
        self.adapter_id = f"fixture.{action}.v1"
        self.calls = calls

    def execute(
        self,
        *,
        accepted_plan: Any,
        preparation: Any,
        units: tuple[TexturePlanUnitDisposition, ...],
        input_asset: ExecutionArtifactBinding,
        output_dir: Path,
    ) -> TextureAgenticAdapterResult:
        del accepted_plan, preparation
        unit_ids = tuple(item.unit_id for item in units)
        self.calls.append((self.action, unit_ids))
        output = _write(
            output_dir / f"{self.action}.usda",
            Path(input_asset.path).read_bytes() + self.action.encode("utf-8"),
        )
        evidence = _write(output_dir / f"{self.action}.json", b"{}\n")
        return TextureAgenticAdapterResult(
            action=self.action,
            unit_ids=unit_ids,
            adapter_id=self.adapter_id,
            input_asset=input_asset,
            output_asset=_binding(output),
            unit_artifacts=tuple(
                TextureAgenticUnitArtifacts(
                    unit_id=unit_id,
                    artifacts=(_binding(output),),
                    metadata={"action": self.action},
                )
                for unit_id in unit_ids
            ),
            evidence_artifacts=(_binding(evidence),),
        )


class _TrustedLeafFixture:
    capability_id = "fixture.trusted-leaf.v1"

    def __init__(
        self,
        *,
        provider_id: str,
        calls: list[Any],
        retry_count: int = 0,
        allow_resume: bool = False,
    ) -> None:
        self.provider_id = provider_id
        self.calls = calls
        self.retry_count = retry_count
        self.allow_resume = allow_resume

    def generate(self, request: Any) -> TextureExecutionResult:
        self.calls.append(request)
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=self.allow_resume)
        output = output_dir / "candidate.usda"
        output.write_bytes(
            Path(request.source.path).read_bytes()
            + f"\n# {self.provider_id}\n".encode()
        )
        unit_artifacts: list[TextureUnitArtifact] = []
        for unit_id in request.target_unit_ids:
            artifact = output_dir / f"{unit_id}.json"
            artifact.write_text("{}\n", encoding="utf-8")
            unit_artifacts.append(
                TextureUnitArtifact(
                    unit_id=unit_id,
                    artifact_paths=(str(artifact),),
                )
            )
        return TextureExecutionResult(
            requested_unit_ids=request.target_unit_ids,
            unit_artifacts=tuple(unit_artifacts),
            output_asset_path=str(output),
            retry_count=self.retry_count,
        )


def test_outer_dispatch_constructs_and_calls_only_selected_adapters(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    constructions: list[str] = []
    calls: list[tuple[str, tuple[str, ...]]] = []

    def factory(action: str):
        def build() -> _RecordingAdapter:
            constructions.append(action)
            return _RecordingAdapter(action, calls)

        return build

    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": factory("generate"),
            "apply_provided": factory("apply_provided"),
        },
        output_dir=tmp_path / "run" / "execution",
    )

    assert constructions == ["generate", "apply_provided"]
    assert calls == [("generate", (UNIT_A,)), ("apply_provided", (UNIT_C,))]
    assert tuple(record.action for record in ledger.records) == (
        "generate",
        "apply_provided",
    )
    assert UNIT_B not in {unit for record in ledger.records for unit in record.unit_ids}
    assert ledger.unresolved_unit_ids == ()
    assert ledger_binding.path.endswith("texture_adapter_ledger.json")


def test_outer_dispatch_persists_successful_prefix_before_later_adapter_failure(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    class FailingApplyAdapter:
        adapter_id = "fixture.apply_provided.failure.v1"

        def execute(self, **_kwargs: Any) -> TextureAgenticAdapterResult:
            raise RuntimeError("synthetic apply failure")

    output_dir = tmp_path / "run" / "partial-execution"
    with pytest.raises(RuntimeError, match="synthetic apply failure"):
        execute_texture_agentic_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            adapter_factories={
                "generate": lambda: _RecordingAdapter("generate", calls),
                "apply_provided": FailingApplyAdapter,
            },
            output_dir=output_dir,
        )

    ledger = TextureAdapterCallLedger.model_validate_json(
        (output_dir / "texture_adapter_ledger.json").read_text(encoding="utf-8")
    )
    assert calls == [("generate", (UNIT_A,))]
    assert tuple(record.action for record in ledger.records) == ("generate",)
    assert ledger.final_candidate == ledger.records[0].output_asset

    resumed, _ = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: _RecordingAdapter("generate", calls),
            "apply_provided": lambda: _RecordingAdapter("apply_provided", calls),
        },
        output_dir=output_dir,
    )

    assert calls == [
        ("generate", (UNIT_A,)),
        ("apply_provided", (UNIT_C,)),
    ]
    assert tuple(record.action for record in resumed.records) == (
        "generate",
        "apply_provided",
    )


def test_production_leaf_adapter_narrows_and_chains_exact_selected_units(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    generate_calls: list[Any] = []
    apply_calls: list[Any] = []
    ledger, _ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: TextureAgenticGeneratorLeafAdapter(
                action="generate",
                leaf_factory=lambda: _TrustedLeafFixture(
                    provider_id="selected-provider",
                    calls=generate_calls,
                ),
                adapter_id="selected-provider:fixture.trusted-leaf.v1",
            ),
            "apply_provided": lambda: TextureAgenticGeneratorLeafAdapter(
                action="apply_provided",
                leaf_factory=lambda: _TrustedLeafFixture(
                    provider_id="outer_provided_image_apply",
                    calls=apply_calls,
                ),
                adapter_id="outer_provided_image_apply:fixture.trusted-leaf.v1",
            ),
        },
        output_dir=tmp_path / "run" / "production-execution",
    )

    assert len(generate_calls) == len(apply_calls) == 1
    assert generate_calls[0].target_unit_ids == (UNIT_A,)
    assert generate_calls[0].scope_plan.selected_unit_ids == (UNIT_A,)
    generate_scope = generate_calls[0].scope_plan.model_dump(mode="json")["request"]
    assert generate_scope["explicit_material_paths"] == ["/World/Looks/A"]
    assert generate_scope["explicit_prim_paths"] == []
    assert apply_calls[0].target_unit_ids == (UNIT_C,)
    assert apply_calls[0].scope_plan.selected_unit_ids == (UNIT_C,)
    apply_scope = apply_calls[0].scope_plan.model_dump(mode="json")["request"]
    assert apply_scope["explicit_material_paths"] == ["/World/Looks/C"]
    assert apply_scope["explicit_prim_paths"] == []
    assert apply_calls[0].source == ledger.records[0].output_asset
    assert ledger.records[1].input_asset == ledger.records[0].output_asset


def test_production_leaf_adapter_preserves_explicit_material_alias_scope(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    aliases = {
        UNIT_A: "/World/InstanceA/Looks/A",
        UNIT_B: "/World/InstanceB/Looks/B",
        UNIT_C: "/World/InstanceC/Looks/C",
    }
    inspection = preparation.inspection.model_copy(
        update={
            "units": tuple(
                unit.model_copy(
                    update={"material_alias_paths": (aliases[unit.unit_id],)}
                )
                for unit in preparation.inspection.units
            )
        }
    )
    scope_payload = preparation.scope_plan.model_dump(mode="python")
    for selected_unit in scope_payload["selected_units"]:
        selected_unit["material_alias_paths"] = [aliases[selected_unit["unit_id"]]]
    scope_payload["request"]["explicit_material_paths"] = list(aliases.values())
    preparation = preparation.model_copy(
        update={
            "inspection": inspection,
            "scope_plan": TexturePlanDocument.model_validate(scope_payload),
        }
    )
    atomic_write_json(preparation_binding.path, preparation)
    preparation_binding = _binding(Path(preparation_binding.path))
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    generate_calls: list[Any] = []
    apply_calls: list[Any] = []

    execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: TextureAgenticGeneratorLeafAdapter(
                action="generate",
                leaf_factory=lambda: _TrustedLeafFixture(
                    provider_id="selected-provider",
                    calls=generate_calls,
                ),
                adapter_id="selected-provider:fixture.trusted-leaf.v1",
            ),
            "apply_provided": lambda: TextureAgenticGeneratorLeafAdapter(
                action="apply_provided",
                leaf_factory=lambda: _TrustedLeafFixture(
                    provider_id="outer_provided_image_apply",
                    calls=apply_calls,
                ),
                adapter_id="outer_provided_image_apply:fixture.trusted-leaf.v1",
            ),
        },
        output_dir=tmp_path / "run" / "alias-production-execution",
    )

    assert len(generate_calls) == len(apply_calls) == 1
    generate_scope = generate_calls[0].scope_plan.model_dump(mode="json")["request"]
    assert generate_scope["explicit_material_paths"] == [aliases[UNIT_A]]
    assert generate_scope["explicit_prim_paths"] == []
    apply_scope = apply_calls[0].scope_plan.model_dump(mode="json")["request"]
    assert apply_scope["explicit_material_paths"] == [aliases[UNIT_C]]
    assert apply_scope["explicit_prim_paths"] == []


def test_production_leaf_adapter_preserves_exact_explicit_prim_scope(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    scope_payload = preparation.scope_plan.model_dump(mode="python")
    scope_payload["request"]["explicit_material_paths"] = []
    scope_payload["request"]["explicit_prim_paths"] = [
        "/World/A",
        "/World/B",
        "/World/C",
    ]
    preparation = preparation.model_copy(
        update={"scope_plan": TexturePlanDocument.model_validate(scope_payload)}
    )
    atomic_write_json(preparation_binding.path, preparation)
    preparation_binding = _binding(Path(preparation_binding.path))
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    generate_calls: list[Any] = []
    apply_calls: list[Any] = []
    execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: TextureAgenticGeneratorLeafAdapter(
                action="generate",
                leaf_factory=lambda: _TrustedLeafFixture(
                    provider_id="selected-provider",
                    calls=generate_calls,
                ),
                adapter_id="selected-provider:fixture.trusted-leaf.v1",
            ),
            "apply_provided": lambda: TextureAgenticGeneratorLeafAdapter(
                action="apply_provided",
                leaf_factory=lambda: _TrustedLeafFixture(
                    provider_id="outer_provided_image_apply",
                    calls=apply_calls,
                ),
                adapter_id="outer_provided_image_apply:fixture.trusted-leaf.v1",
            ),
        },
        output_dir=tmp_path / "run" / "prim-scoped-execution",
    )

    generate_scope = generate_calls[0].scope_plan.model_dump(mode="json")["request"]
    assert generate_scope["explicit_material_paths"] == []
    assert generate_scope["explicit_prim_paths"] == ["/World/A"]
    apply_scope = apply_calls[0].scope_plan.model_dump(mode="json")["request"]
    assert apply_scope["explicit_material_paths"] == []
    assert apply_scope["explicit_prim_paths"] == ["/World/C"]


def test_outer_dispatch_replays_digest_bound_completed_prefix_on_resume(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    plan = plan.model_copy(
        update={
            "dispositions": (
                plan.dispositions[0],
                *(
                    item.model_copy(
                        update={
                            "action": "preserve",
                            "rationale": "Keep exact prepared material state.",
                            "requested_appearance": None,
                            "generator_inputs": None,
                        }
                    )
                    for item in plan.dispositions[1:]
                ),
            )
        }
    )
    accepted, accepted_binding = _accept(
        tmp_path, preparation, preparation_binding, plan
    )
    calls: list[Any] = []

    def factory() -> TextureAgenticGeneratorLeafAdapter:
        return TextureAgenticGeneratorLeafAdapter(
            action="generate",
            leaf_factory=lambda: _TrustedLeafFixture(
                provider_id="selected-provider",
                calls=calls,
                allow_resume=True,
            ),
            adapter_id="selected-provider:fixture.trusted-leaf.v1",
        )

    output_dir = tmp_path / "run" / "resumable-execution"
    first, _ = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={"generate": factory},
        output_dir=output_dir,
    )
    resumed, _ = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={"generate": factory},
        output_dir=output_dir,
    )

    assert len(calls) == 1
    assert resumed.records == first.records
    assert resumed.final_candidate == first.final_candidate


def test_production_leaf_adapter_rejects_retry_as_second_attempt(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    with pytest.raises(ValueError, match="rejects provider retries"):
        execute_texture_agentic_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            adapter_factories={
                "generate": lambda: TextureAgenticGeneratorLeafAdapter(
                    action="generate",
                    leaf_factory=lambda: _TrustedLeafFixture(
                        provider_id="selected-provider",
                        calls=[],
                        retry_count=1,
                    ),
                    adapter_id="selected-provider:fixture.trusted-leaf.v1",
                )
            },
            output_dir=tmp_path / "run" / "retry-execution",
        )


def test_preserve_only_constructs_no_mutating_adapter(tmp_path: Path) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    plan = plan.model_copy(
        update={
            "dispositions": tuple(
                item.model_copy(
                    update={
                        "action": "preserve",
                        "rationale": "Keep exact prepared material state.",
                        "requested_appearance": None,
                        "generator_inputs": None,
                    }
                )
                for item in plan.dispositions
            )
        }
    )
    accepted, accepted_binding = _accept(
        tmp_path, preparation, preparation_binding, plan
    )

    def forbidden_factory() -> _RecordingAdapter:
        pytest.fail("preserve-only plan constructed a mutating adapter")

    ledger, _ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": forbidden_factory,
            "apply_provided": forbidden_factory,
        },
        output_dir=tmp_path / "run" / "execution",
    )

    assert ledger.records == ()
    assert ledger.final_candidate == preparation.request.source


def test_preserve_only_packages_layered_source_before_readback(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(
        tmp_path,
        with_source_dependency=True,
    )
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    plan = plan.model_copy(
        update={
            "dispositions": tuple(
                item.model_copy(
                    update={
                        "action": "preserve",
                        "rationale": "Keep exact prepared material state.",
                        "requested_appearance": None,
                        "generator_inputs": None,
                    }
                )
                for item in plan.dispositions
            )
        }
    )
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        plan,
    )
    output_dir = tmp_path / "run" / "layered-preserve-execution"

    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={},
        output_dir=output_dir,
    )

    assert ledger.records == ()
    assert ledger.final_candidate != preparation.request.source
    assert ledger.prepared_candidate == ledger.final_candidate
    assert Path(ledger.final_candidate.path).suffix == ".usdz"
    assert bind_usd_dependency_closure(ledger.final_candidate.path) == []
    candidate_stage = Usd.Stage.Open(ledger.final_candidate.path)
    assert candidate_stage is not None
    assert candidate_stage.GetPrimAtPath("/Dependency").IsValid()

    replayed, replayed_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={},
        output_dir=output_dir,
    )
    assert replayed == ledger
    assert replayed_binding == ledger_binding


def test_preserve_only_recovers_after_candidate_precedes_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation, preparation_binding = _preparation(
        tmp_path,
        with_source_dependency=True,
    )
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    plan = plan.model_copy(
        update={
            "dispositions": tuple(
                item.model_copy(
                    update={
                        "action": "preserve",
                        "rationale": "Keep exact prepared material state.",
                        "requested_appearance": None,
                        "generator_inputs": None,
                    }
                )
                for item in plan.dispositions
            )
        }
    )
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        plan,
    )
    output_dir = tmp_path / "run" / "interrupted-layered-preserve"
    original_write_packet = texture_agentic_readiness._write_packet
    interrupted = False

    def fail_first_ledger_write(path: str | Path, packet: Any) -> Any:
        nonlocal interrupted
        if Path(path).name == "texture_adapter_ledger.json" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption before ledger persistence")
        return original_write_packet(path, packet)

    monkeypatch.setattr(
        texture_agentic_readiness,
        "_write_packet",
        fail_first_ledger_write,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_texture_agentic_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            adapter_factories={},
            output_dir=output_dir,
        )

    candidate_root = output_dir / "00-preserve" / "candidate"
    candidate_path = candidate_root / "preserved-texture-candidate.usdz"
    assert candidate_root.is_dir()
    candidate_binding = bind_texture_agentic_artifact(candidate_path)
    assert (candidate_root / "preserve_candidate_receipt.json").is_file()
    assert not (output_dir / "texture_adapter_ledger.json").exists()

    original_rmtree = texture_agentic_readiness.shutil.rmtree

    def forbid_completed_candidate_rebuild(
        path: str | Path, *args: Any, **kwargs: Any
    ) -> None:
        if Path(path) == candidate_root:
            pytest.fail("completed preserve candidate was rebuilt")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        texture_agentic_readiness.shutil,
        "rmtree",
        forbid_completed_candidate_rebuild,
    )

    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={},
        output_dir=output_dir,
    )

    assert ledger_binding.path.endswith("texture_adapter_ledger.json")
    assert ledger.records == ()
    assert ledger.prepared_candidate == ledger.final_candidate
    assert ledger.final_candidate == candidate_binding
    assert bind_usd_dependency_closure(ledger.final_candidate.path) == []


def test_apply_provided_constructs_no_generator_adapter(tmp_path: Path) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    plan = plan.model_copy(
        update={
            "dispositions": (
                plan.dispositions[0].model_copy(
                    update={
                        "action": "preserve",
                        "rationale": "Preserve this prepared unit.",
                        "requested_appearance": None,
                        "generator_inputs": None,
                    }
                ),
                plan.dispositions[1],
                plan.dispositions[2],
            )
        }
    )
    accepted, accepted_binding = _accept(
        tmp_path, preparation, preparation_binding, plan
    )
    constructed: list[str] = []
    calls: list[tuple[str, tuple[str, ...]]] = []

    def forbidden_generate_factory() -> _RecordingAdapter:
        pytest.fail("apply-provided plan constructed a generator adapter")

    def apply_factory() -> _RecordingAdapter:
        constructed.append("apply_provided")
        return _RecordingAdapter("apply_provided", calls)

    ledger, _ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": forbidden_generate_factory,
            "apply_provided": apply_factory,
        },
        output_dir=tmp_path / "run" / "execution",
    )

    assert constructed == ["apply_provided"]
    assert calls == [("apply_provided", (UNIT_C,))]
    assert tuple(record.action for record in ledger.records) == ("apply_provided",)


def test_unavailable_selected_provider_fails_closed_without_fallback(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    constructed: list[str] = []

    def unselected_apply_factory() -> _RecordingAdapter:
        constructed.append("apply_provided")
        return _RecordingAdapter("apply_provided", [])

    with pytest.raises(RuntimeError, match="unavailable without fallback: generate"):
        execute_texture_agentic_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            adapter_factories={"apply_provided": unselected_apply_factory},
            output_dir=tmp_path / "run" / "execution",
        )

    assert constructed == []


def test_execution_rejects_a_plan_changed_after_outer_freeze(tmp_path: Path) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    changed_plan = accepted.plan.model_copy(
        update={
            "dispositions": (
                accepted.plan.dispositions[0].model_copy(
                    update={"rationale": "Changed after outer acceptance."}
                ),
                *accepted.plan.dispositions[1:],
            )
        }
    )
    atomic_write_json(accepted.proposal.path, changed_plan)

    with pytest.raises(ValueError, match="proposed Texture plan bytes changed"):
        execute_texture_agentic_plan(
            accepted,
            accepted_binding=accepted_binding,
            preparation=preparation,
            preparation_binding=preparation_binding,
            adapter_factories={},
            output_dir=tmp_path / "run" / "execution",
        )


def _evidence_and_review(
    tmp_path: Path,
    accepted: Any,
    accepted_binding: ExecutionArtifactBinding,
    ledger: TextureAdapterCallLedger,
    ledger_binding: ExecutionArtifactBinding,
) -> tuple[
    TextureAgenticSavedStageReadback,
    ExecutionArtifactBinding,
    TextureAgenticEvidenceReceipt,
    ExecutionArtifactBinding,
    TextureAgenticReviewReceipt,
    ExecutionArtifactBinding,
]:
    unit_evidence: list[TextureUnitRenderEvidence] = []
    for unit_id in accepted.plan.unit_ids:
        sources = tuple(
            _binding(_write(tmp_path / "review" / f"{unit_id}-source-{view}.png"))
            for view in accepted.plan.evidence.required_views
        )
        candidates = tuple(
            _binding(_write(tmp_path / "review" / f"{unit_id}-candidate-{view}.png"))
            for view in accepted.plan.evidence.required_views
        )
        unit_evidence.append(
            TextureUnitRenderEvidence(
                unit_id=unit_id,
                source_images=sources,
                candidate_images=candidates,
            )
        )
    static = _binding(_write(tmp_path / "review" / "scope.json", b"{}\n"))
    readback = TextureAgenticSavedStageReadback(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        saved_stage=ledger.final_candidate,
        scope_plan_digest=accepted.plan.scope_plan_digest,
        verification_artifacts=(static,),
    )
    readback, readback_binding = record_texture_agentic_saved_stage_readback(
        readback,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        output_path=tmp_path / "run" / "texture_saved_stage_readback.json",
    )
    evidence = TextureAgenticEvidenceReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        view_names=accepted.plan.evidence.required_views,
        unit_evidence=tuple(unit_evidence),
        static_evidence=(static,),
        saved_stage_readback=readback_binding,
        renderer_metadata={
            "renderer": "ovrtx",
            "current_run": True,
            "directions": list(accepted.plan.evidence.required_views),
        },
    )
    evidence, evidence_binding = record_texture_agentic_evidence(
        evidence,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
        output_path=tmp_path / "run" / "texture_agentic_evidence.json",
    )
    visuals = tuple(
        binding
        for item in unit_evidence
        for binding in (*item.source_images, *item.candidate_images)
    )
    review = TextureAgenticReviewReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        evidence=evidence_binding,
        candidate=ledger.final_candidate,
        plan_digest=accepted.proposal_digest,
        unit_reviews=tuple(
            TextureAgenticUnitReview(
                unit_id=item.unit_id,
                disposition=(
                    "unresolved"
                    if item.action == "defer"
                    else "reject"
                    if item.action == "reject"
                    else "accept"
                ),
                rationale="Exact current-run evidence satisfies the plan.",
            )
            for item in accepted.plan.dispositions
        ),
        inspected_visual_artifacts=visuals,
        findings=("All exact current-run views accepted.",),
    )
    review, review_binding = record_texture_agentic_review(
        review,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
        output_path=tmp_path / "run" / "texture_agentic_review.json",
    )
    return (
        readback,
        readback_binding,
        evidence,
        evidence_binding,
        review,
        review_binding,
    )


def test_persisted_readback_and_evidence_revalidate_for_resume(tmp_path: Path) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: _RecordingAdapter("generate", calls),
            "apply_provided": lambda: _RecordingAdapter("apply_provided", calls),
        },
        output_dir=tmp_path / "run" / "execution",
    )
    (
        readback,
        readback_binding,
        evidence,
        evidence_binding,
        _review,
        _review_binding,
    ) = _evidence_and_review(
        tmp_path,
        accepted,
        accepted_binding,
        ledger,
        ledger_binding,
    )

    validate_texture_agentic_saved_stage_readback(
        readback,
        readback_binding=readback_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
    )
    validate_texture_agentic_evidence(
        evidence,
        evidence_binding=evidence_binding,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        readback=readback,
        readback_binding=readback_binding,
    )
    Path(evidence.unit_evidence[0].candidate_images[0].path).write_bytes(
        b"tampered-after-evidence-persistence"
    )
    with pytest.raises(ValueError, match="current-run OVRTX evidence .* bytes changed"):
        validate_texture_agentic_evidence(
            evidence,
            evidence_binding=evidence_binding,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            readback=readback,
            readback_binding=readback_binding,
        )


def test_review_rehashes_nested_visual_evidence_after_child(tmp_path: Path) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: _RecordingAdapter("generate", calls),
            "apply_provided": lambda: _RecordingAdapter("apply_provided", calls),
        },
        output_dir=tmp_path / "run" / "execution",
    )
    (
        _readback,
        _readback_binding,
        evidence,
        evidence_binding,
        review,
        _review_binding,
    ) = _evidence_and_review(
        tmp_path,
        accepted,
        accepted_binding,
        ledger,
        ledger_binding,
    )
    Path(review.inspected_visual_artifacts[0].path).write_bytes(
        b"review-child-mutated-visual"
    )

    with pytest.raises(
        ValueError,
        match=r"Texture review visual evidence .* bytes changed",
    ):
        record_texture_agentic_review(
            review,
            accepted=accepted,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            output_path=tmp_path / "run" / "second_texture_agentic_review.json",
        )


def test_terminal_receipt_requires_exact_evidence_review_and_publication(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        _mixed_plan(tmp_path, preparation, preparation_binding),
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: _RecordingAdapter("generate", calls),
            "apply_provided": lambda: _RecordingAdapter("apply_provided", calls),
        },
        output_dir=tmp_path / "run" / "execution",
    )
    (
        _readback,
        readback_binding,
        evidence,
        evidence_binding,
        review,
        review_binding,
    ) = _evidence_and_review(
        tmp_path, accepted, accepted_binding, ledger, ledger_binding
    )
    request = _binding(_write(tmp_path / "run" / "capability_request.json", b"{}\n"))
    published = _binding(
        _write(
            tmp_path / "run" / "published.usda",
            Path(ledger.final_candidate.path).read_bytes(),
        )
    )
    publication_verification = _binding(
        _write(tmp_path / "run" / "publication-verification.json", b"{}\n")
    )
    publication = TextureAgenticPublicationReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        evidence=evidence_binding,
        saved_stage_readback=readback_binding,
        review=review_binding,
        published_asset=published,
        verification_artifacts=(publication_verification,),
    )
    publication, publication_binding = record_texture_agentic_publication(
        publication,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
        readback_binding=readback_binding,
        review=review,
        review_binding=review_binding,
        output_path=tmp_path / "run" / "texture_agentic_publication.json",
    )
    cleanup = TextureAgenticCleanupReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        status="completed",
        retained_artifacts=(published,),
    )
    cleanup, cleanup_binding = record_texture_agentic_cleanup(
        cleanup,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        output_path=tmp_path / "run" / "texture_agentic_cleanup.json",
    )

    terminal, terminal_binding = seal_texture_agentic_terminal_receipt(
        request=request,
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
        review=review,
        review_binding=review_binding,
        publication=publication,
        publication_binding=publication_binding,
        cleanup=cleanup,
        cleanup_binding=cleanup_binding,
        disposition="published",
        output_path=tmp_path / "run" / "texture_terminal_receipt.json",
    )

    assert terminal.disposition == "published"
    assert terminal.saved_stage_readback == readback_binding
    assert terminal.publication == publication_binding
    assert terminal.cleanup == cleanup_binding
    assert terminal_binding.path.endswith("texture_terminal_receipt.json")


def test_reject_or_defer_blocks_publication(
    tmp_path: Path,
) -> None:
    preparation, preparation_binding = _preparation(tmp_path)
    plan = _mixed_plan(tmp_path, preparation, preparation_binding)
    deferred = plan.dispositions[1].model_copy(
        update={"action": "defer", "rationale": "Required input is unavailable."}
    )
    plan = plan.model_copy(
        update={"dispositions": (plan.dispositions[0], deferred, plan.dispositions[2])}
    )
    accepted, accepted_binding = _accept(
        tmp_path,
        preparation,
        preparation_binding,
        plan,
    )
    calls: list[tuple[str, tuple[str, ...]]] = []
    ledger, ledger_binding = execute_texture_agentic_plan(
        accepted,
        accepted_binding=accepted_binding,
        preparation=preparation,
        preparation_binding=preparation_binding,
        adapter_factories={
            "generate": lambda: _RecordingAdapter("generate", calls),
            "apply_provided": lambda: _RecordingAdapter("apply_provided", calls),
        },
        output_dir=tmp_path / "run" / "execution",
    )
    (
        _readback,
        readback_binding,
        evidence,
        evidence_binding,
        review,
        review_binding,
    ) = _evidence_and_review(
        tmp_path, accepted, accepted_binding, ledger, ledger_binding
    )
    publication = TextureAgenticPublicationReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        evidence=evidence_binding,
        saved_stage_readback=readback_binding,
        review=review_binding,
        published_asset=ledger.final_candidate,
        verification_artifacts=(readback_binding,),
    )

    with pytest.raises(ValueError, match="unresolved Texture units block publication"):
        record_texture_agentic_publication(
            publication,
            accepted_binding=accepted_binding,
            ledger=ledger,
            ledger_binding=ledger_binding,
            evidence=evidence,
            evidence_binding=evidence_binding,
            readback_binding=readback_binding,
            review=review,
            review_binding=review_binding,
            output_path=tmp_path / "run" / "must-not-publication.json",
        )

    cleanup = TextureAgenticCleanupReceipt(
        accepted_plan=accepted_binding,
        adapter_ledger=ledger_binding,
        candidate=ledger.final_candidate,
        status="completed",
        retained_artifacts=(ledger.final_candidate,),
    )
    cleanup, cleanup_binding = record_texture_agentic_cleanup(
        cleanup,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        output_path=tmp_path / "run" / "texture_agentic_cleanup.json",
    )
    terminal, _terminal_binding = seal_texture_agentic_terminal_receipt(
        request=_binding(_write(tmp_path / "run" / "capability_request.json", b"{}\n")),
        accepted=accepted,
        accepted_binding=accepted_binding,
        ledger=ledger,
        ledger_binding=ledger_binding,
        evidence=evidence,
        evidence_binding=evidence_binding,
        review=review,
        review_binding=review_binding,
        publication=None,
        publication_binding=None,
        cleanup=cleanup,
        cleanup_binding=cleanup_binding,
        disposition="blocked",
        issue_codes=("unresolved_unit",),
        output_path=tmp_path / "run" / "texture_terminal_receipt.json",
    )

    assert ledger.unresolved_unit_ids == (UNIT_B,)
    assert terminal.disposition == "blocked"
