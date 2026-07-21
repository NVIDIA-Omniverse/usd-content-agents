# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import BaseModel, ValidationError

from world_understanding.agentic.validation_scaffold import (
    DraftValidationIssue,
    DraftValidationResult,
    create_draft_validation_request,
    run_validation_scaffold,
)
from world_understanding.rendering_backend_contract import RENDERING_BACKEND_NAMES
from world_understanding.validation import (
    SUPPORTED_RENDER_BACKENDS,
    V1_TEMPLATE_NAMES,
    VALIDATION_RENDERING_BACKEND_NAMES,
    ValidationContractError,
    ValidationEvidence,
    ValidationFocusConfig,
    ValidationInputGroups,
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationProject,
    ValidationRenderConfig,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateDefinition,
    ValidationTemplateRegistry,
    ValidationTemplateResult,
    aggregate_validation_verdict,
    create_default_template_registry,
    validation_result_from_scaffold_result,
)
from world_understanding.validation import models as validation_models
from world_understanding.validation.json_normalization import (
    StructuredJsonNormalizer,
)


class _ExternalMetadataModel(BaseModel):
    path: Path


class _UnsupportedMetadataValue:
    pass


def test_validation_request_round_trips_json(tmp_path: Path) -> None:
    usd_path = tmp_path / "asset.usda"
    request = ValidationRequest(
        task_description="Validate that the generated cart looks correct.",
        inputs=(usd_path,),
        project=ValidationProject(name="cart", working_dir=tmp_path / "run"),
        render=ValidationRenderConfig(
            backend="ovrtx",
            image_width=512,
            image_height=512,
            views=("front", "side"),
            animation_frames=(1, 12),
        ),
        requested_templates=("render_valid", "look_right"),
        policy={"reference_image": tmp_path / "reference.png"},
        metadata={"source": "unit-test"},
    )

    loaded = ValidationRequest.model_validate_json(request.model_dump_json())

    assert loaded.schema_version == "1.0"
    assert loaded.inputs == (str(usd_path),)
    assert loaded.project.working_dir == str(tmp_path / "run")
    assert loaded.render.views == ("front", "side")
    assert loaded.render.animation_frames == (1, 12)
    assert loaded.policy["reference_image"] == str(tmp_path / "reference.png")


def test_validation_models_reexport_constants() -> None:
    assert validation_models.SCHEMA_VERSION == "1.0"
    assert validation_models.ISSUE_CODE_PATTERN.startswith("^[a-z]")


def test_validation_render_backend_schema_advertises_capability_subset() -> None:
    assert VALIDATION_RENDERING_BACKEND_NAMES == ("remote", "ovrtx")
    assert SUPPORTED_RENDER_BACKENDS == frozenset({"remote", "ovrtx"})
    assert set(VALIDATION_RENDERING_BACKEND_NAMES) < set(RENDERING_BACKEND_NAMES)

    backend_schema = ValidationRenderConfig.model_json_schema()["properties"]["backend"]
    assert "remote, ovrtx" in backend_schema["description"]
    assert "structured unknown-versus-unsupported" in backend_schema["description"]
    assert "enum" not in backend_schema

    # Schema clients must be able to submit values that runtime code then
    # distinguishes as unknown or canonical-but-unsupported.
    assert ValidationRenderConfig(backend="typo").backend == "typo"


def test_validation_template_definition_normalizer_edges() -> None:
    definition = ValidationTemplateDefinition(
        name="render_valid",
        description="Render validation.",
        required_input_kinds="usd",
        optional_input_kinds=None,
    )
    assert definition.required_input_kinds == ("usd",)
    assert definition.optional_input_kinds == ()

    registry = ValidationTemplateRegistry((definition,))
    assert registry.definitions() == (definition,)


def test_validation_request_requires_task_and_input() -> None:
    with pytest.raises(ValidationError):
        ValidationRequest(task_description="", inputs=("asset.usda",))

    with pytest.raises(ValidationError):
        ValidationRequest(task_description="Validate asset.", inputs=())


def test_invalid_contract_shapes_raise_validation_error() -> None:
    request = ValidationRequest(
        task_description="Validate asset.",
        inputs=("asset.usda",),
    )
    plan = ValidationPlan(
        steps=(
            ValidationPlanStep(
                template_name="render_valid",
                reason="requested",
            ),
        )
    )

    with pytest.raises(ValidationError):
        ValidationRequest(task_description="Validate asset.", inputs=123)

    with pytest.raises(ValidationError):
        ValidationRequest(
            task_description="Validate asset.",
            inputs=("asset.usda",),
            policy=[],
        )

    with pytest.raises(ValidationError):
        ValidationFocusConfig(prim_paths=123)

    with pytest.raises(ValidationError):
        ValidationPlan(
            steps=(
                ValidationPlanStep(
                    template_name="render_valid",
                    reason="requested",
                ),
            ),
            artifact_paths=[],
        )

    with pytest.raises(ValidationError):
        ValidationResult(
            verdict="pass",
            request=request,
            plan=plan,
            artifact_paths=[],
        )

    with pytest.raises(ValidationError):
        ValidationTemplateDefinition(
            name="render_valid",
            description="Render validation.",
            required_input_kinds=123,
        )


def test_validation_model_normalizers_cover_scalar_and_none_edges() -> None:
    decorator = validation_models.field_validator("metadata", check_fields=False)
    assert callable(decorator)

    project = ValidationProject(name="asset", working_dir=None)
    assert project.working_dir is None

    planner = validation_models.ValidationPlannerConfig(
        metadata={"project": project, "path": Path("asset.usda")}
    )
    assert planner.metadata == {
        "project": {"name": "asset", "working_dir": None, "session_id": None},
        "path": "asset.usda",
    }
    assert validation_models.ValidationPlannerConfig(metadata=None).metadata == {}

    render = ValidationRenderConfig(
        views=123,
        animation_frames=7,
        metadata=None,
    )
    assert render.views == "123"
    assert render.animation_frames == "7"
    assert render.metadata == {}

    assert ValidationFocusConfig(prim_paths=None).prim_paths == ()
    assert ValidationFocusConfig(prim_paths="/World/Cube").prim_paths == (
        "/World/Cube",
    )

    request = ValidationRequest(
        task_description="Validate asset.",
        inputs=Path("asset.usda"),
        requested_templates=None,
    )
    assert request.inputs == ("asset.usda",)
    assert request.requested_templates == ()
    assert ValidationRequest(
        task_description="Validate asset.",
        inputs=("asset.usda",),
        requested_templates="render_valid",
    ).requested_templates == ("render_valid",)

    validation_input = validation_models.ValidationInput(
        original="asset.usda",
        path=Path("asset.usda"),
        kind="usd",
        image_paths=None,
    )
    assert validation_input.path == "asset.usda"
    assert validation_input.image_paths == ()
    assert validation_models.ValidationInput(
        original="asset.usda",
        path="asset.usda",
        kind="usd",
        image_paths=Path("preview.png"),
    ).image_paths == ("preview.png",)

    plan = ValidationPlan(
        steps=(ValidationPlanStep(template_name="render_valid"),),
        focus_prim_paths=Path("/World/Cube"),
        artifact_paths=None,
    )
    assert plan.focus_prim_paths == ("/World/Cube",)
    assert plan.artifact_paths == {}

    evidence = ValidationEvidence(kind="render", path=None, metadata=None)
    assert evidence.path is None
    assert evidence.metadata == {}

    result = ValidationResult(
        verdict="pass",
        request=request,
        plan=plan,
        artifact_paths=None,
    )
    assert result.artifact_paths == {}

    groups = ValidationInputGroups.from_inventory_dict({"items": "not-a-sequence"})
    assert groups.items == ()


def test_validation_contract_json_normalizer_preserves_tail_behavior() -> None:
    external_model = _ExternalMetadataModel(path=Path("model.usda"))
    unsupported = _UnsupportedMetadataValue()

    planner = validation_models.ValidationPlannerConfig(
        metadata={
            7: {
                "values": [
                    Path("asset.usda"),
                    external_model,
                    None,
                    "label",
                    3,
                    1.5,
                    True,
                    unsupported,
                ]
            }
        }
    )

    assert planner.metadata["7"]["values"][:-1] == [
        "asset.usda",
        {"path": "model.usda"},
        None,
        "label",
        3,
        1.5,
        True,
    ]
    assert planner.metadata["7"]["values"][-1] is unsupported

    with pytest.raises(ValueError, match="Expected a mapping"):
        StructuredJsonNormalizer().mapping([])


def test_default_template_registry_defines_v1_allowlist() -> None:
    registry = create_default_template_registry()

    assert registry.names() == V1_TEMPLATE_NAMES
    assert registry.get("look_right").required_capabilities == ("vlm",)
    assert registry.get("render_valid").issue_code_namespaces == ("render", "ovrtx")
    physical_behavior = registry.get("physical_behavior")
    assert physical_behavior.required_capabilities == ()
    assert physical_behavior.optional_input_kinds == (
        "time_sampled_usd",
        "animation_usd",
        "video",
        "sampled_video_frame",
        "simulation_json",
        "trajectory_metrics",
        "refine_summary",
    )
    assert physical_behavior.output_evidence == (
        "resolution",
        "available_evidence",
        "refine_summaries",
        "behavior_summary",
    )

    with pytest.raises(ValidationContractError, match="Unknown validation template"):
        registry.validate_template_names(("made_up",))

    with pytest.raises(ValidationContractError, match="already registered"):
        ValidationTemplateRegistry(
            (
                ValidationTemplateDefinition(
                    name="render_valid",
                    description="First definition.",
                ),
                ValidationTemplateDefinition(
                    name="render_valid",
                    description="Duplicate definition.",
                ),
            )
        )


def test_input_groups_preserve_reference_paths_and_ignore_future_item_keys() -> None:
    groups = ValidationInputGroups.from_inventory_dict(
        {
            "items": [
                {
                    "original": "render.png",
                    "path": "render.png",
                    "kind": "image",
                    "future_inventory_key": "ignored",
                }
            ],
            "image_paths": ["render.png"],
            "reference_image_paths": ["reference.png"],
        }
    )

    assert groups.image_paths == ("render.png",)
    assert groups.reference_image_paths == ("reference.png",)
    assert groups.items[0].path == "render.png"

    with pytest.raises(ValidationError):
        ValidationInputGroups.from_inventory_dict(
            {"items": [{"original": "bad", "path": None, "kind": "image"}]}
        )


def test_validation_result_round_trips_json() -> None:
    request = ValidationRequest(
        task_description="Validate render evidence.",
        inputs=("render.png",),
        requested_templates=("render_valid",),
    )
    plan = ValidationPlan(
        steps=(
            ValidationPlanStep(
                template_name="render_valid",
                reason="requested by caller",
                inputs_needed=("images_or_render_bundle_or_usd",),
            ),
        ),
        reasoning_summary="Rules planner selected render_valid.",
    )
    issue = ValidationIssue(
        code="render.low_contrast",
        severity="warn",
        message="Render contrast is low.",
        template_name="render_valid",
    )
    template_result = ValidationTemplateResult(
        template_name="render_valid",
        status="warn",
        issues=(issue,),
        metrics={"image_count": 1},
        evidence={"image_paths": ["render.png"]},
    )
    result = ValidationResult(
        verdict="warn",
        request=request,
        plan=plan,
        template_results=(template_result,),
        issues=(issue,),
        metrics={"render_valid": {"image_count": 1}},
        evidence={"render_valid": {"image_paths": ["render.png"]}},
        artifact_paths={"validation_result": "run/validation_result.json"},
    )

    loaded = ValidationResult.model_validate_json(result.model_dump_json())

    assert loaded.verdict == "warn"
    assert loaded.template_results[0].passed is False
    assert loaded.issues[0].code == "render.low_contrast"
    assert loaded.artifact_paths["validation_result"] == "run/validation_result.json"


def test_aggregate_verdict_and_evidence_contract() -> None:
    assert aggregate_validation_verdict(()) == "planned"

    warn_issue = ValidationIssue(
        code="visual.judge_unavailable",
        severity="warn",
        message="VLM judge is unavailable.",
    )
    warn_result = ValidationTemplateResult(
        template_name="look_right",
        status="skipped",
        issues=(warn_issue,),
        evidence_items=(
            ValidationEvidence(
                kind="image",
                path=Path("render.png"),
                metadata={"source": Path("render.png")},
            ),
        ),
    )
    fail_result = ValidationTemplateResult(
        template_name="render_valid",
        status="passed",
        issues=(
            ValidationIssue(
                code="render.image_decode_failed",
                severity="fail",
                message="Image did not decode.",
            ),
        ),
    )
    needs_refinement_result = ValidationTemplateResult(
        template_name="physical_behavior",
        status="needs_refinement",
        issues=(
            ValidationIssue(
                code="physics.behavior_needs_refinement",
                severity="warn",
                message="Behavior judge requested another refinement iteration.",
            ),
        ),
    )
    pass_result = ValidationTemplateResult(
        template_name="render_valid",
        status="passed",
    )

    assert aggregate_validation_verdict((warn_result,)) == "warn"
    assert (
        aggregate_validation_verdict((needs_refinement_result,)) == "needs_refinement"
    )
    assert (
        aggregate_validation_verdict((warn_result, needs_refinement_result))
        == "needs_refinement"
    )
    assert aggregate_validation_verdict((warn_result, fail_result)) == "fail"
    assert (
        aggregate_validation_verdict((needs_refinement_result, fail_result)) == "fail"
    )
    assert aggregate_validation_verdict((pass_result,)) == "pass"
    assert warn_result.evidence_items[0].path == "render.png"
    assert warn_result.evidence_items[0].metadata["source"] == "render.png"


def test_empty_plan_and_bad_issue_code_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ValidationPlan(steps=())

    with pytest.raises(ValidationError):
        ValidationIssue(code="BAD", severity="warn", message="bad code")


def test_scaffold_result_maps_to_stable_validation_result(tmp_path: Path) -> None:
    image_path, scaffold_result = _build_scaffold_result(tmp_path)
    stable_result = validation_result_from_scaffold_result(scaffold_result)
    loaded = ValidationResult.model_validate_json(stable_result.model_dump_json())

    assert loaded.schema_version == "1.0"
    assert loaded.verdict == "pass"
    assert loaded.request.inputs == (str(image_path),)
    assert [step.template_name for step in loaded.plan.steps] == ["render_valid"]
    assert loaded.plan.input_groups.image_paths == (str(image_path),)
    assert loaded.template_results[0].status == "passed"
    assert loaded.template_results[0].metadata["adapter"] == "render_valid"
    assert loaded.artifact_paths["validation_result"].endswith("validation_result.json")
    assert "artifact_paths" not in loaded.metadata


def test_scaffold_compat_rejects_unknown_status(tmp_path: Path) -> None:
    _, scaffold_result = _build_scaffold_result(tmp_path)
    bad_result = scaffold_result.template_results[0].__class__(
        template_name="render_valid",
        status="partial",  # type: ignore[arg-type]
    )
    patched_result = scaffold_result.__class__(
        verdict=scaffold_result.verdict,
        request=scaffold_result.request,
        plan=scaffold_result.plan,
        template_results=(bad_result,),
    )

    with pytest.raises(ValidationContractError, match="Unknown scaffold"):
        validation_result_from_scaffold_result(patched_result)


def test_scaffold_compat_rejects_unknown_issue_severity(tmp_path: Path) -> None:
    _, scaffold_result = _build_scaffold_result(tmp_path)
    bad_issue = DraftValidationIssue(
        code="render.unknown",
        severity="fatal",  # type: ignore[arg-type]
        message="Unknown severity.",
    )
    bad_result = scaffold_result.template_results[0].__class__(
        template_name="render_valid",
        status="failed",
        issues=(bad_issue,),
    )
    patched_result = scaffold_result.__class__(
        verdict=scaffold_result.verdict,
        request=scaffold_result.request,
        plan=scaffold_result.plan,
        template_results=(bad_result,),
    )

    with pytest.raises(
        ValidationContractError, match="Unknown scaffold issue severity"
    ):
        validation_result_from_scaffold_result(patched_result)


def test_scaffold_compat_private_helper_edges() -> None:
    from world_understanding.validation import scaffold_compat

    assert scaffold_compat._artifact_paths_from_inventory({}) == {}
    assert scaffold_compat._mapping_attr(object(), "metadata") == {}
    assert scaffold_compat._issue_severity("info") == "info"


def _build_scaffold_result(tmp_path: Path) -> tuple[Path, DraftValidationResult]:
    image_path = _write_valid_image(tmp_path / "render.png")
    scaffold_request = create_draft_validation_request(
        task_description="Validate render evidence.",
        inputs=(image_path,),
        working_dir=tmp_path / "run",
        requested_templates=("render_valid",),
    )
    return image_path, run_validation_scaffold(scaffold_request)


def _write_valid_image(path: Path) -> Path:
    image = PILImage.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 31, 31], fill=(255, 0, 0))
    draw.rectangle([32, 0, 63, 31], fill=(0, 255, 0))
    draw.rectangle([0, 32, 31, 63], fill=(0, 0, 255))
    draw.rectangle([32, 32, 63, 63], fill=(255, 255, 0))
    image.save(path)
    return path
