# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for material refinement and variation orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade
from world_understanding.functions.graphics.mock_rendering import MockRenderingBackend

from material_agent.api.material_refinement import (
    MaterialRefinementInput,
    run_material_refinement_api,
)
from material_agent.material_refinement import (
    MaterialRefinementConfig,
    MaterialVariationConfig,
    RestTextureVariationGenerator,
    TextureVariationArtifacts,
    TextureVariationRequest,
    TextureVariationSettings,
    run_material_refinement,
    run_material_variations,
)

_APPROVE = """**Critique:**
The material matches the goal.
**Score:** 9
**Decision:** APPROVE
**Improvement Suggestions:**
None.
"""
_CONTINUE = """**Critique:**
The surface is too glossy and needs more visible texture.
**Score:** 4
**Decision:** CONTINUE
**Improvement Suggestions:**
Increase roughness and add fine mottling.
"""


class _FakeGenerator:
    name = "deterministic-texture-variation-fake"

    def __init__(self, *, fixed_channel: int | None = None) -> None:
        self.requests: list[TextureVariationRequest] = []
        self.fixed_channel = fixed_channel

    def generate(
        self,
        request: TextureVariationRequest,
        *,
        output_dir: Path,
        cancel_check: Any = None,
    ) -> TextureVariationArtifacts:
        self.requests.append(request)
        output_dir.mkdir(parents=True, exist_ok=True)
        channel = (
            self.fixed_channel
            if self.fixed_channel is not None
            else int(round(request.strength * 255))
        )
        albedo = output_dir / "albedo.png"
        normal = output_dir / "normal.png"
        orm = output_dir / "orm.png"
        Image.new("RGB", (8, 8), (channel, 64, 255 - channel)).save(albedo)
        Image.new("RGB", (8, 8), (128, 128, 255)).save(normal)
        Image.new("RGB", (8, 8), (255, channel, 0)).save(orm)
        return TextureVariationArtifacts(
            albedo_path=albedo,
            normal_path=normal,
            orm_path=orm,
            variant_asset_uri=request.source_asset_path.as_uri(),
            metadata={"seed": request.seed, "strength": request.strength},
            diagnostics=(),
        )


class _SequenceVlm:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if str(kwargs.get("system_prompt", "")).startswith(
            "You infer normalized PBR controls"
        ):
            prompt = str(kwargs["final_prompt"])
            encoded_bounds = prompt.split("Allowed target ranges:\n", 1)[1].split(
                "\n\nReturn exactly", 1
            )[0]
            bounds = json.loads(encoded_bounds)

            def bounded(name: str, requested: float) -> float:
                return min(bounds[name]["max"], max(bounds[name]["min"], requested))

            return json.dumps(
                {
                    "base_color": [
                        bounded("base_color_r", 0.6),
                        bounded("base_color_g", 0.25),
                        bounded("base_color_b", 0.4),
                    ],
                    "roughness": bounded("roughness", 0.6),
                    "metallic": bounded("metallic", 0.0),
                    "reasoning": "Deterministic prompt-target test inference.",
                }
            )
        return self.responses.pop(0)


def _config_data(output_dir: Path, *, refinements: int = 1) -> dict[str, Any]:
    source_texture_dir = output_dir.parent / f"{output_dir.name}-source-textures"
    source_texture_dir.mkdir(parents=True, exist_ok=True)
    source_albedo = source_texture_dir / "albedo.png"
    source_normal = source_texture_dir / "normal.png"
    source_orm = source_texture_dir / "orm.png"
    Image.new("RGB", (8, 8), (115, 51, 153)).save(source_albedo)
    Image.new("RGB", (8, 8), (128, 128, 255)).save(source_normal)
    Image.new("RGB", (8, 8), (255, 89, 0)).save(source_orm)
    return {
        "goal": {
            "appearance_prompt": "satin blue painted metal",
        },
        "source": {
            "material_id": "blue_painted_metal",
            "name": "Blue Painted Metal",
            "description": "A source painted-metal material.",
            "material_profile": "preview_surface",
            "base_color": [0.45, 0.2, 0.6],
            "roughness": 0.35,
            "metallic": 0.0,
            "textures": {
                "albedo": source_albedo.as_posix(),
                "normal": source_normal.as_posix(),
                "orm": source_orm.as_posix(),
            },
        },
        "variation": {
            "strength_min": 0.2,
            "strength_max": 1.0,
            "texture_size": 8,
        },
        "optimization": {
            "name": "random",
            "max_trials": 3,
            "seed": 7,
            "max_refinements": refinements,
        },
        "render": {
            "backend": "mock",
            "image_width": 32,
            "image_height": 32,
            "camera_corners": ["+x+y+z"],
        },
        "judge": {
            "vlm": {"backend": "fake", "model": "test-vlm"},
            "score_threshold": 0.7,
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "output_dir": output_dir.as_posix(),
    }


def _run(
    config: MaterialRefinementConfig,
    generator: _FakeGenerator,
    vlm: _SequenceVlm,
):
    return run_material_refinement(
        config,
        generator=generator,
        rendering_backend=MockRenderingBackend(),
        vlm_judge=vlm,
    )


def test_config_uses_source_material_and_variation_search_controls(
    tmp_path: Path,
) -> None:
    config = MaterialRefinementConfig.from_mapping(
        _config_data(tmp_path / "output"), base_dir=tmp_path
    )

    assert config.goal.appearance_prompt == "satin blue painted metal"
    assert config.source.name == "Blue Painted Metal"
    assert config.source.base_color == pytest.approx((0.45, 0.2, 0.6))
    assert config.source.roughness == pytest.approx(0.35)
    assert config.variation.endpoint is None
    assert config.source.representation == "textured_pbr"
    assert config.source.textures is not None
    assert [parameter.name for parameter in config.search_params] == [
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "roughness",
        "metallic",
        "strength",
    ]
    assert config.search_params[-1].min_value == pytest.approx(0.2)
    assert config.search_params[-1].max_value == pytest.approx(1.0)


def test_config_defaults_to_auto_optimizer(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "output")
    data["optimization"].pop("name")

    config = MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)

    assert config.optimizer.name == "auto"


def test_config_rejects_generation_recipe_input(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "output")
    data["recipe"] = {
        "name": "Wrong abstraction",
        "description": "Generation-only metadata.",
        "appearance_prompt": "This must not be accepted by refinement.",
    }

    with pytest.raises(ValueError, match=r"unknown field\(s\): recipe"):
        MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)


def test_scalar_pbr_input_produces_scalar_pbr_output(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "output", refinements=0)
    data["source"].pop("textures")
    data["goal"]["appearance_prompt"] = "clear frosted blue painted metal"
    config = MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)
    generator = _FakeGenerator()

    result = _run(config, generator, _SequenceVlm(_APPROVE))

    assert result.approved is True
    assert generator.requests == []
    assert "strength" not in result.best_params
    assert not any(result.best_material_dir.rglob("*.png"))
    assert "albedo" not in result.best_artifacts
    stage = Usd.Stage.Open(str(result.best_artifacts["material_usd"]))
    shader_ids = {
        str(prim.GetAttribute("info:id").Get())
        for prim in stage.Traverse()
        if prim.GetAttribute("info:id").IsValid()
    }
    assert shader_ids == {"UsdPreviewSurface"}
    preview = next(
        UsdShade.Shader(prim)
        for prim in stage.Traverse()
        if prim.GetAttribute("info:id").Get() == "UsdPreviewSurface"
    )
    assert tuple(preview.GetInput("diffuseColor").Get()) == pytest.approx(
        (
            result.best_params["base_color_r"],
            result.best_params["base_color_g"],
            result.best_params["base_color_b"],
        )
    )
    assert preview.GetInput("roughness").Get() == pytest.approx(
        result.best_params["roughness"]
    )
    assert preview.GetInput("metallic").Get() == pytest.approx(
        result.best_params["metallic"]
    )
    assert preview.GetInput("opacity").Get() == pytest.approx(config.source.opacity)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["source_representation"] == "scalar_pbr"


def test_default_generator_requires_sanitized_service_endpoint() -> None:
    settings = TextureVariationSettings.from_mapping(
        {"endpoint": "https://example.test", "engine": "step1x"}
    )
    assert settings.endpoint == "https://example.test"
    assert settings.engine == "step1x"

    with pytest.raises(ValueError, match="endpoint is required"):
        RestTextureVariationGenerator(TextureVariationSettings())
    with pytest.raises(ValueError, match="must not contain credentials"):
        RestTextureVariationGenerator(
            TextureVariationSettings(endpoint="https://user:secret@example.test")
        )


def test_refinement_runs_shared_tuning_then_approves_winner(tmp_path: Path) -> None:
    config = MaterialRefinementConfig.from_mapping(
        _config_data(tmp_path / "output"), base_dir=tmp_path
    )
    generator = _FakeGenerator()
    vlm = _SequenceVlm(_APPROVE)

    result = _run(config, generator, vlm)

    assert result.approved is True
    assert result.trial_count == 3
    assert len(generator.requests) == 3
    assert len(result.attempts) == 1
    assert result.best_artifacts["albedo"].is_file()
    assert result.best_artifacts["rendered_image_1"].is_file()
    assert result.source_swatch_path.is_file()
    source_stage = Usd.Stage.Open(str(result.source_swatch_path))
    source_mesh = UsdGeom.Mesh.Get(source_stage, "/World/MaterialSwatch")
    assert source_mesh.GetPrim().IsValid()
    assert len(source_mesh.GetPointsAttr().Get()) == 8066
    assert len(source_mesh.GetFaceVertexCountsAttr().Get()) == 8192
    source_st = UsdGeom.PrimvarsAPI(source_mesh).GetPrimvar("st")
    assert source_st.GetInterpolation() == UsdGeom.Tokens.faceVarying
    assert len(source_st.Get()) == sum(source_mesh.GetFaceVertexCountsAttr().Get())
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["generator"] == generator.name
    assert summary["proxy_objective"]["approval_authority"] is False
    assert summary["history_sha256"]
    source_evidence = json.loads(
        (config.output_dir / "source" / "source.json").read_text(encoding="utf-8")
    )
    assert "recipe" not in source_evidence
    assert source_evidence["source"]["name"] == "Blue Painted Metal"
    assert source_evidence["source"]["base_color"] == [0.45, 0.2, 0.6]
    assert source_evidence["authoring_material"]["binding"].startswith("/World/Looks/")

    first_trial = json.loads(
        result.history_path.read_text(encoding="utf-8").splitlines()[0]
    )
    replica = first_trial["replicas"][0]
    for channel in ("albedo", "normal", "orm"):
        candidate_path = Path(replica["artifacts"][channel])
        service_path = Path(replica["artifacts"][f"service_{channel}"])
        assert candidate_path.is_file()
        assert service_path.is_file()
        assert candidate_path != service_path
        assert replica["artifact_metadata"][channel]["sha256"]
        assert replica["artifact_metadata"][f"service_{channel}"]["sha256"]
    assert replica["metrics"]["candidate_map_validation"]["width"] == 8
    assert (
        "This optimizer trial must target" in replica["metadata"]["generation_prompt"]
    )


def test_prompt_only_goal_uses_model_inferred_bounded_target(
    tmp_path: Path,
) -> None:
    data = _config_data(tmp_path / "output", refinements=0)
    data["optimization"]["max_trials"] = 1
    data["goal"] = {"appearance_prompt": "subtly weathered blue paint"}
    config = MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)

    result = _run(config, _FakeGenerator(), _SequenceVlm(_APPROVE))

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    resolved = summary["target_inference"]["resolved_features"]
    assert resolved["base_color"] == pytest.approx((0.6, 0.25, 0.4))
    assert resolved["roughness"] == pytest.approx(0.6)
    assert resolved["metallic"] == pytest.approx(0.0)
    assert resolved["color_source"] == "appearance_prompt_inference"
    assert result.best_params["base_color_r"] == pytest.approx(0.6)
    assert result.best_params["base_color_g"] == pytest.approx(0.25)
    assert result.best_params["base_color_b"] == pytest.approx(0.4)
    assert result.best_params["roughness"] == pytest.approx(0.6)
    assert result.best_params["metallic"] == pytest.approx(0.0)
    assert summary["target_inference"]["inference"]["raw_response"]
    assert summary["goal"] == {"appearance_prompt": "subtly weathered blue paint"}
    assert json.loads((config.output_dir / "goal.json").read_text()) == summary["goal"]
    assert (
        json.loads((config.output_dir / "target_inference.json").read_text())
        == summary["target_inference"]
    )


def test_rejection_revises_prompt_and_starts_fresh_tuning_sweep(
    tmp_path: Path,
) -> None:
    class RevisingVlm(_SequenceVlm):
        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            if not str(kwargs.get("system_prompt", "")).startswith(
                "You infer normalized PBR controls"
            ):
                return super().generate_with_image_caption_pairs(**kwargs)
            prompt = str(kwargs["final_prompt"])
            if "Increase roughness and add fine mottling" not in prompt:
                return super().generate_with_image_caption_pairs(**kwargs)
            self.calls.append(kwargs)
            return json.dumps(
                {
                    "base_color": [0.8, 0.5, 0.2],
                    "roughness": 0.8,
                    "metallic": 0.0,
                    "reasoning": (
                        "The review requests a target outside the initial bounds."
                    ),
                }
            )

    config = MaterialRefinementConfig.from_mapping(
        _config_data(tmp_path / "output", refinements=1), base_dir=tmp_path
    )
    generator = _FakeGenerator()
    vlm = RevisingVlm(_CONTINUE, _APPROVE)

    result = _run(config, generator, vlm)

    assert result.approved is True
    assert result.trial_count == 6
    assert len(result.attempts) == 2
    first_prompts = {request.prompt for request in generator.requests[:3]}
    second_prompts = {request.prompt for request in generator.requests[3:]}
    assert len(first_prompts) == 3
    assert all("satin blue painted metal" in prompt for prompt in first_prompts)
    assert all("This optimizer trial must target" in prompt for prompt in first_prompts)
    assert len(second_prompts) == 3
    assert all(
        "Increase roughness and add fine mottling" in prompt
        for prompt in second_prompts
    )
    assert result.attempts[0].optimizer_seed == 7
    assert result.attempts[1].optimizer_seed == 10
    assert result.attempts[0].target_revision is not None
    revised_space = result.attempts[0].revised_search_space
    assert revised_space is not None
    initial_space = result.attempts[0].search_space
    assert (
        revised_space.bounds("base_color_r")[1]
        > initial_space.bounds("base_color_r")[1]
    )
    assert (
        revised_space.bounds("base_color_g")[1]
        > initial_space.bounds("base_color_g")[1]
    )
    assert (
        revised_space.bounds("base_color_b")[0]
        < initial_space.bounds("base_color_b")[0]
    )
    assert revised_space.bounds("roughness")[1] > initial_space.bounds("roughness")[1]
    assert result.attempts[1].search_space == revised_space
    assert result.attempts[1].target_features.base_color == pytest.approx(
        (0.8, 0.5, 0.2)
    )
    assert result.attempts[1].target_features.roughness == pytest.approx(0.8)
    second_attempt_trials = [
        json.loads(line)
        for line in result.history_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["attempt"] == 2
    ]
    assert second_attempt_trials[0]["params"]["base_color_r"] == pytest.approx(0.8)
    assert second_attempt_trials[0]["params"]["base_color_g"] == pytest.approx(0.5)
    assert second_attempt_trials[0]["params"]["base_color_b"] == pytest.approx(0.2)
    assert second_attempt_trials[0]["params"]["roughness"] == pytest.approx(0.8)
    target_evidence = json.loads(
        (config.output_dir / "target_inference.json").read_text(encoding="utf-8")
    )
    assert len(target_evidence["revisions"]) == 1
    assert target_evidence["resolved_features"]["roughness"] == pytest.approx(0.8)
    assert target_evidence["revisions"][0]["expanded_controls"] == [
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "roughness",
    ]


def test_variation_set_calls_same_refinement_core_and_carries_prior_render(
    tmp_path: Path,
) -> None:
    data = _config_data(tmp_path / "variations", refinements=0)
    data["variation_set"] = {
        "description": "Explore distinct blue paint finishes.",
        "requested_count": 2,
        "minimum_feature_distance": 0.02,
        "diversity_weight": 2.0,
    }
    config = MaterialVariationConfig.from_mapping(data, base_dir=tmp_path)
    generator = _FakeGenerator()
    vlm = _SequenceVlm(_APPROVE, _APPROVE)

    result = run_material_variations(
        config,
        generator=generator,
        rendering_backend=MockRenderingBackend(),
        vlm_judge=vlm,
    )

    assert result.success is True
    assert len(result.slots) == 2
    assert all(slot.result is not None for slot in result.slots)
    assert result.material_library_path is not None
    assert result.material_library_path.is_file()
    assert result.materials_manifest_path is not None
    assert result.materials_manifest_path.is_file()
    assert len(result.selected_materials) == 2
    assert len({material.binding for material in result.selected_materials}) == 2
    second_captions = next(
        [caption for caption, _path in call["image_caption_pairs"]]
        for call in vlm.calls
        if any(
            caption.startswith("Previously Approved Variations")
            for caption, _path in call["image_caption_pairs"]
        )
    )
    assert "Previously Approved Variations (1):" in second_captions
    assert "Requested slot: 2 of 2" in generator.requests[3].prompt
    second_result = result.slots[1].result
    assert second_result is not None
    valid_replicas = [
        replica
        for line in second_result.history_path.read_text(encoding="utf-8").splitlines()
        for replica in json.loads(line)["replicas"]
        if replica["success"]
    ]
    assert valid_replicas
    assert any(
        replica["metrics"]["diversity_penalty"] > 0.0
        and replica["metrics"]["combined_objective"]
        > replica["metrics"]["objective"]["value"]
        for replica in valid_replicas
    )


def test_scalar_variation_set_never_introduces_textures(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "variations", refinements=0)
    data["source"].pop("textures")
    data["variation_set"] = {
        "description": "Explore distinct scalar color and sheen treatments.",
        "requested_count": 2,
        "minimum_feature_distance": 0.0,
    }
    config = MaterialVariationConfig.from_mapping(data, base_dir=tmp_path)
    generator = _FakeGenerator()

    result = run_material_variations(
        config,
        generator=generator,
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_SequenceVlm(_APPROVE, _APPROVE),
    )

    assert result.success is True
    assert generator.requests == []
    assert result.material_library_path is not None
    assert not any(result.material_library_path.parent.rglob("*.png"))
    assert all(material.textures is None for material in result.selected_materials)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert all(
        material["representation"] == "scalar_pbr" and "textures" not in material
        for material in manifest["selected_materials"]
    )
    stage = Usd.Stage.Open(str(result.material_library_path))
    shader_ids = [
        str(prim.GetAttribute("info:id").Get())
        for prim in stage.Traverse()
        if prim.GetAttribute("info:id").IsValid()
    ]
    assert shader_ids == ["UsdPreviewSurface", "UsdPreviewSurface"]


def test_variation_cancellation_preserves_manifest(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "variations", refinements=0)
    data["variation_set"] = {"requested_count": 2}
    config = MaterialVariationConfig.from_mapping(data, base_dir=tmp_path)

    result = run_material_variations(
        config,
        cancel_check=lambda: True,
        generator=_FakeGenerator(),
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_SequenceVlm(_APPROVE),
    )

    assert result.status == "cancelled"
    assert result.manifest_path.is_file()
    assert json.loads(result.manifest_path.read_text())["status"] == "cancelled"


def test_variation_rejects_deterministic_near_duplicates(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "variations", refinements=0)
    data["variation_set"] = {
        "requested_count": 2,
        "minimum_feature_distance": 0.1,
    }
    data["search"] = {
        "max_color_delta": 0.0,
        "max_roughness_delta": 0.0,
        "max_mixed_metallic_delta": 0.0,
    }
    data["variation"]["strength_min"] = 0.5
    data["variation"]["strength_max"] = 0.5
    config = MaterialVariationConfig.from_mapping(data, base_dir=tmp_path)

    result = run_material_variations(
        config,
        generator=_FakeGenerator(fixed_channel=128),
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_SequenceVlm(_APPROVE),
    )

    assert result.status == "incomplete"
    assert result.termination_reason == "slot_incomplete"
    assert result.material_library_path is None
    assert result.slots[1].error is not None


def test_public_api_projects_refinement_result(tmp_path: Path) -> None:
    generator = _FakeGenerator()
    output = run_material_refinement_api(
        MaterialRefinementInput(
            config=_config_data(tmp_path / "output", refinements=0),
            generator=generator,
            rendering_backend=MockRenderingBackend(),
            vlm_judge=_SequenceVlm(_APPROVE),
        )
    )

    assert output.success is True
    assert output.approved is True
    assert output.trial_count == 3
    assert output.attempt_count == 1
    assert output.summary_path is not None and output.summary_path.is_file()


@pytest.mark.parametrize(
    ("filename", "contract"),
    (
        ("material_refinement_example.yaml", MaterialRefinementConfig),
        (
            "material_refinement_preview_surface_graph_example.yaml",
            MaterialRefinementConfig,
        ),
        ("material_refinement_openpbr_example.yaml", MaterialRefinementConfig),
        (
            "material_refinement_omnipbr_mdl_example.yaml",
            MaterialRefinementConfig,
        ),
        ("material_variation_example.yaml", MaterialVariationConfig),
    ),
)
def test_checked_in_configs_pass_real_contract_validation(
    filename: str,
    contract: type[MaterialRefinementConfig] | type[MaterialVariationConfig],
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / filename
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    parsed = contract.from_mapping(data, base_dir=config_path.parent)

    assert parsed is not None
    optimizer = (
        parsed.refinement.optimizer
        if isinstance(parsed, MaterialVariationConfig)
        else parsed.optimizer
    )
    refinement = (
        parsed.refinement if isinstance(parsed, MaterialVariationConfig) else parsed
    )
    assert optimizer.name == "auto"
    assert set(data["goal"]) == {"appearance_prompt"}
    assert refinement.goal.to_dict() == {
        "appearance_prompt": refinement.goal.appearance_prompt
    }
    assert refinement.judge.vlm["backend"] == "nim"
    assert refinement.judge.vlm["model"] == "moonshotai/kimi-k3"


def test_refinement_examples_use_one_shared_appearance_goal() -> None:
    config_root = Path(__file__).parents[1] / "configs"
    filenames = (
        "material_refinement_example.yaml",
        "material_refinement_preview_surface_graph_example.yaml",
        "material_refinement_openpbr_example.yaml",
        "material_refinement_omnipbr_mdl_example.yaml",
    )

    prompts = {
        yaml.safe_load(config_root.joinpath(filename).read_text(encoding="utf-8"))[
            "goal"
        ]["appearance_prompt"]
        for filename in filenames
    }

    assert prompts == {
        "Satin cobalt-blue painted steel with soft broad highlights and a "
        "physically plausible dielectric response."
    }
