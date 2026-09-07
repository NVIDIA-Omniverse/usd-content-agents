# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary and failure-path coverage for material refinement."""

from __future__ import annotations

import io
import json
import urllib.error
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image, UnidentifiedImageError
from world_understanding.functions.graphics.mock_rendering import MockRenderingBackend
from world_understanding.optimization import OptimizationCancelledError

import material_agent.api.material_refinement as api_module
import material_agent.material_refinement.artifacts as artifacts_module
import material_agent.material_refinement.runner as runner_module
import material_agent.material_refinement.texture_variation as texture_module
import material_agent.material_refinement.variation as variation_module
import material_agent.material_refinement.visual as visual_module
from material_agent.material_library_generation.schema import TextureMapSet
from material_agent.material_refinement import (
    MaterialFeatures,
    MaterialObjectiveWeights,
    MaterialRefinementCancelled,
    MaterialRefinementConfig,
    MaterialRefinementDependencyError,
    MaterialRefinementError,
    MaterialRefinementGoal,
    MaterialRefinementJudgeSettings,
    MaterialRefinementRenderSettings,
    MaterialVariationConfig,
    MaterialVariationGoal,
    RestTextureVariationGenerator,
    TextureVariationArtifacts,
    TextureVariationError,
    TextureVariationRequest,
    TextureVariationSettings,
    material_feature_distance,
    resolve_target_features,
    run_material_refinement,
    run_material_variations,
    score_material_features,
)

_APPROVE = """**Critique:**
The material matches the goal.
**Score:** 9
**Decision:** APPROVE
**Improvement Suggestions:**
None.
"""
_CONTINUE = """**Critique:**
The material remains too glossy.
**Score:** 4
**Decision:** CONTINUE
**Improvement Suggestions:**
Increase roughness.
"""


class _FakeGenerator:
    name = "edge-test-texture-variation"

    def generate(
        self,
        request: TextureVariationRequest,
        *,
        output_dir: Path,
        cancel_check: Any = None,
    ) -> TextureVariationArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        channel = int(round(request.strength * 255))
        albedo = output_dir / "albedo.png"
        normal = output_dir / "normal.png"
        orm = output_dir / "orm.png"
        Image.new("RGB", (4, 4), (channel, 64, 255 - channel)).save(albedo)
        Image.new("RGB", (4, 4), (128, 128, 255)).save(normal)
        Image.new("RGB", (4, 4), (255, channel, 0)).save(orm)
        return TextureVariationArtifacts(
            albedo_path=albedo,
            normal_path=normal,
            orm_path=orm,
            variant_asset_uri=request.source_asset_path.as_uri(),
            metadata={},
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


def _config_data(
    output_dir: Path,
    *,
    max_trials: int = 1,
    max_refinements: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_texture_dir = output_dir.parent / f"{output_dir.name}-source-textures"
    source_texture_dir.mkdir(parents=True, exist_ok=True)
    source_albedo = source_texture_dir / "albedo.png"
    source_normal = source_texture_dir / "normal.png"
    source_orm = source_texture_dir / "orm.png"
    Image.new("RGB", (4, 4), (115, 51, 153)).save(source_albedo)
    Image.new("RGB", (4, 4), (128, 128, 255)).save(source_normal)
    Image.new("RGB", (4, 4), (255, 89, 0)).save(source_orm)
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
            "texture_size": 4,
        },
        "optimization": {
            "name": "random",
            "max_trials": max_trials,
            "seed": 7,
            "max_refinements": max_refinements,
        },
        "render": {
            "backend": "mock",
            "image_width": 24,
            "image_height": 24,
            "camera_corners": ["+x+y+z"],
        },
        "judge": {
            "vlm": {"backend": "fake", "model": "test-vlm"},
            "score_threshold": 0.7,
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "output_dir": output_dir.as_posix(),
        "overwrite": overwrite,
    }


def _config(tmp_path: Path, **kwargs: Any) -> MaterialRefinementConfig:
    return MaterialRefinementConfig.from_mapping(
        _config_data(tmp_path / "output", **kwargs), base_dir=tmp_path
    )


def _run(
    config: MaterialRefinementConfig,
    *,
    generator: Any | None = None,
    vlm: Any | None = None,
    **kwargs: Any,
):
    return run_material_refinement(
        config,
        generator=generator or _FakeGenerator(),
        rendering_backend=MockRenderingBackend(),
        vlm_judge=vlm or _SequenceVlm(_APPROVE),
        **kwargs,
    )


def _variation_request(source: Path, reference: Path | None = None):
    return TextureVariationRequest(
        source_asset_path=source,
        material_path="/Looks/Test",
        prompt="painted steel",
        reference_image_paths=(reference,) if reference is not None else (),
        strength=0.75,
        seed=42,
        variant_name="painted_steel_001",
    )


class _BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_prompt_goal_render_string_and_replica_seed_run_end_to_end(
    tmp_path: Path,
) -> None:
    data = _config_data(tmp_path / "ignored")
    data["goal"] = {"appearance_prompt": "satin blue coating"}
    data["render"]["camera_corners"] = "+x+y+z"
    data["optimization"]["replica_seed"] = 101
    config = MaterialRefinementConfig.from_mapping(
        data,
        base_dir=tmp_path,
        output_dir_override=Path("reference-output"),
    )

    result = _run(config)

    assert result.approved is True
    assert result.attempts[0].representative_seed == 101
    assert config.render.to_task_config()["camera_corners"] == ["+x+y+z"]
    goal_evidence = json.loads((config.output_dir / "goal.json").read_text())
    assert goal_evidence == {"appearance_prompt": "satin blue coating"}
    inference_evidence = json.loads(
        (config.output_dir / "target_inference.json").read_text()
    )
    assert inference_evidence["inference"]["target_features"]["base_color"]


def test_objective_scalar_paths_and_fail_closed_inputs() -> None:
    weights = MaterialObjectiveWeights(color=0.0, roughness=3.0, metallic=1.0)
    target = MaterialFeatures(base_color=None, roughness=0.25, metallic=0.0)
    generated = MaterialFeatures(base_color=None, roughness=0.75, metallic=1.0)
    scalar_target = resolve_target_features(
        MaterialRefinementGoal(appearance_prompt="rough neutral coating"),
        MaterialFeatures(None, 0.25, 0.0, "source"),
    )

    score = score_material_features(target, generated, weights)

    assert scalar_target.base_color is None
    assert scalar_target.color_source == "source_albedo"
    assert score.value == pytest.approx(0.625)
    assert score.to_dict()["normalized_weights"] == {
        "roughness": 0.75,
        "metallic": 0.25,
    }
    assert weights.to_dict() == {
        "color": 0.0,
        "roughness": 3.0,
        "metallic": 1.0,
    }
    assert target.to_dict()["base_color"] is None
    with pytest.raises(ValueError, match="base color"):
        score_material_features(
            MaterialFeatures((0.0, 0.0, 0.0), None, None),
            generated,
            MaterialObjectiveWeights(),
        )
    with pytest.raises(ValueError, match="roughness"):
        score_material_features(
            MaterialFeatures(None, 0.5, None),
            MaterialFeatures(None, None, None),
            MaterialObjectiveWeights(),
        )
    with pytest.raises(ValueError, match="metallic"):
        score_material_features(
            MaterialFeatures(None, None, 0.5),
            MaterialFeatures(None, None, None),
            MaterialObjectiveWeights(),
        )
    with pytest.raises(ValueError, match="positively weighted"):
        score_material_features(
            target,
            generated,
            MaterialObjectiveWeights(color=0.0, roughness=0.0, metallic=0.0),
        )
    with pytest.raises(ValueError, match="shared positively weighted"):
        material_feature_distance(
            MaterialFeatures(None, None, None),
            MaterialFeatures(None, None, None),
            MaterialObjectiveWeights(),
        )


def test_terminal_candidate_selection_ignores_non_finite_proxy_scores() -> None:
    def candidate(
        objective_value: float | None,
        *,
        score: float = 0.5,
        approved: bool = False,
    ) -> Any:
        return SimpleNamespace(
            verdict=SimpleNamespace(approved=approved, score=score),
            trial=SimpleNamespace(objective_value=objective_value),
        )

    valid = candidate(0.25, score=0.4)
    selected = runner_module._select_terminal_candidate(
        [candidate(None, score=0.9), candidate(float("nan"), score=0.8), valid]
    )

    assert selected is valid
    assert (
        runner_module._select_terminal_candidate(
            [candidate(None, score=0.9, approved=True), valid]
        )
        is valid
    )
    with pytest.raises(MaterialRefinementError, match="proxy score is not finite"):
        runner_module._select_terminal_candidate(
            [candidate(None), candidate(float("inf"))]
        )


def test_visual_dependency_factories_and_reference_judgment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = object()
    monkeypatch.setattr(
        visual_module, "create_rendering_backend", lambda *_args: backend
    )
    render_settings = MaterialRefinementRenderSettings.from_mapping({"backend": "mock"})
    assert visual_module.provision_rendering_backend(render_settings) is backend

    with pytest.raises(ValueError, match="backend/provider"):
        visual_module.provision_vlm_judge(MaterialRefinementJudgeSettings(vlm={}))
    expected_vlm = _SequenceVlm(_APPROVE)
    monkeypatch.setattr(
        visual_module.ModelProvisioningTask,
        "create_vlm",
        lambda _self, _settings: expected_vlm,
    )
    judge_settings = MaterialRefinementJudgeSettings(
        vlm={"backend": "fake", "model": "judge"}
    )
    assert visual_module.provision_vlm_judge(judge_settings) is expected_vlm

    rendered = tmp_path / "rendered.png"
    Image.new("RGB", (2, 2), "blue").save(rendered)
    long_critique = "x" * 240
    vlm = _SequenceVlm(
        f"**Critique:** {long_critique}\n**Score:** 9\n**Decision:** APPROVE"
    )
    verdict = visual_module.judge_rendered_material(
        target=MaterialRefinementGoal(appearance_prompt="blue paint"),
        render_evidence=visual_module.MaterialRenderEvidence(
            swatch_usd_path=tmp_path / "swatch.usda",
            flattened_usd_path=None,
            rendered_image_paths=(rendered,),
            backend="mock",
            rendering_stats={},
            render_validation=(),
        ),
        settings=judge_settings,
        vlm_judge=vlm,
        iteration=1,
        previous_feedback=None,
    )

    assert verdict.approved is True
    assert verdict.image_caption_pairs[0][0] == "Rendered Material Swatch - View 1:"
    assert verdict.reasoning.endswith("...")
    normalized_prompt = " ".join(verdict.prompt.split())
    assert "Controllable properties in this run" in normalized_prompt
    assert "Do not lower the material score" in normalized_prompt
    assert "do not recommend changing them" in normalized_prompt
    assert "where 7 or higher is acceptable" in normalized_prompt


def test_identical_renders_receive_fresh_visual_judgments(tmp_path: Path) -> None:
    data = _config_data(
        tmp_path / "output",
        max_trials=1,
        max_refinements=1,
    )
    config = MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)
    vlm = _SequenceVlm(
        "**Critique:** first review\n**Score:** 2\n**Decision:** CONTINUE",
        "**Critique:** second review\n**Score:** 3\n**Decision:** CONTINUE",
    )

    result = _run(config, vlm=vlm)

    visual_calls = [
        call
        for call in vlm.calls
        if not str(call.get("system_prompt", "")).startswith(
            "You infer normalized PBR controls"
        )
    ]
    assert result.termination_reason == "max_iterations"
    assert len(result.attempts) == 2
    assert len(visual_calls) == 2
    first_render = result.attempts[0].render_evidence
    second_render = result.attempts[1].render_evidence
    assert first_render is not None and second_render is not None
    assert (
        first_render.rendered_image_paths[0].read_bytes()
        == second_render.rendered_image_paths[0].read_bytes()
    )
    assert result.attempts[0].judge_verdict is not None
    assert result.attempts[1].judge_verdict is not None
    assert result.attempts[0].judge_verdict.score == pytest.approx(0.2)
    assert result.attempts[1].judge_verdict.score == pytest.approx(0.3)


def test_inferred_controls_clamp_only_float_boundary_drift() -> None:
    minimum = 0.6710000038146973

    assert visual_module._bounded_inferred_value(
        0.671,
        field_name="base_color_g",
        bounds=(minimum, 1.0),
    ) == pytest.approx(minimum)
    with pytest.raises(ValueError, match="outside"):
        visual_module._bounded_inferred_value(
            0.67,
            field_name="base_color_g",
            bounds=(minimum, 1.0),
        )


def test_material_map_validation_and_portable_copy(tmp_path: Path) -> None:
    source_dir = tmp_path / "service"
    source_dir.mkdir()
    albedo = source_dir / "albedo.png"
    normal = source_dir / "normal.png"
    orm = source_dir / "orm.png"
    source_albedo = Image.new("RGB", (2, 2), (25, 50, 75))
    source_albedo.putpixel((1, 1), (125, 150, 175))
    source_albedo.save(albedo)
    Image.new("RGB", (2, 2), (128, 128, 255)).save(normal)
    source_orm = Image.new("RGB", (2, 2), (80, 20, 10))
    source_orm.putpixel((1, 1), (160, 40, 30))
    source_orm.save(orm)
    source = TextureMapSet(albedo=albedo, normal=normal, orm=orm)

    output, source_validation, output_validation = (
        artifacts_module.materialize_candidate_maps(
            source,
            output_dir=tmp_path / "candidate",
        )
    )
    features = runner_module.extract_material_features(
        albedo_path=output.albedo, orm_path=output.orm
    )

    assert source_validation.width == output_validation.width == 2
    assert output_validation.to_dict()["sha256"]["normal"]
    assert features.base_color == pytest.approx((0.0585606, 0.10016885, 0.1599427))
    assert features.roughness == pytest.approx(25 / 255)
    assert features.metallic == pytest.approx(15 / 255)
    with Image.open(output.albedo) as candidate_albedo:
        assert candidate_albedo.convert("RGB").getpixel((1, 1)) == (125, 150, 175)
    with Image.open(output.orm) as candidate_orm:
        assert candidate_orm.convert("RGB").getpixel((0, 0))[0] == 80

    with pytest.raises(ValueError, match="distinct"):
        artifacts_module.validate_material_maps(
            TextureMapSet(albedo=albedo, normal=albedo, orm=orm)
        )
    mismatched = source_dir / "mismatched.png"
    Image.new("RGB", (3, 2), "white").save(mismatched)
    with pytest.raises(ValueError, match="matching dimensions"):
        artifacts_module.validate_material_maps(
            TextureMapSet(albedo=albedo, normal=normal, orm=mismatched)
        )
    corrupt = source_dir / "corrupt.png"
    corrupt.write_text("not an image", encoding="ascii")
    with pytest.raises(UnidentifiedImageError):
        artifacts_module.validate_material_maps(
            TextureMapSet(albedo=albedo, normal=normal, orm=corrupt)
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        artifacts_module.validate_material_maps(
            TextureMapSet(
                albedo=albedo,
                normal=normal,
                orm=source_dir / "missing.png",
            )
        )


def test_rest_generator_polls_and_localizes_completed_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="ascii")
    reference = tmp_path / "reference.png"
    Image.new("RGB", (1, 1), "white").save(reference)
    maps: dict[str, Path] = {}
    for name, color in (
        ("albedo", "red"),
        ("normal", "blue"),
        ("orm", "green"),
    ):
        path = tmp_path / f"service-{name}.png"
        Image.new("RGB", (1, 1), color).save(path)
        maps[name] = path
    settings = TextureVariationSettings(
        endpoint="http://example.test/",
        engine="test-engine",
        poll_interval_seconds=0.001,
    )
    generator = RestTextureVariationGenerator(settings)
    monkeypatch.setattr(
        generator,
        "_upload_asset",
        lambda path: f"http://example.test/assets/{path.name}",
    )
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request_json(method: str, url: str, payload=None):
        calls.append((method, url, payload))
        if method == "POST":
            return {"job_id": "job-1", "status": "queued"}
        return {
            "job_id": "job-1",
            "status": "completed",
            "result": {
                "variant_asset_uri": source.as_uri(),
                "variant_name": "painted_steel_001",
                "generated_textures": {
                    name: path.as_uri() for name, path in maps.items()
                },
                "metadata": {"engine": "test"},
                "diagnostics": [{"kind": "complete"}],
            },
        }

    monkeypatch.setattr(generator, "_request_json", request_json)

    artifacts = generator.generate(
        _variation_request(source, reference), output_dir=tmp_path / "localized"
    )

    assert generator.name == "texture-variation-api"
    assert [call[0] for call in calls] == ["POST", "GET"]
    assert calls[0][2]["source_asset_uri"] == (
        "http://example.test/assets/source_material.usdz"
    )
    assert calls[0][2]["conditioning"]["reference_image_uris"] == [
        "http://example.test/assets/reference.png"
    ]
    assert calls[0][2]["configuration"]["engine"] == "test-engine"
    assert artifacts.metadata == {"engine": "test"}
    assert artifacts.diagnostics == ({"kind": "complete"},)
    assert artifacts.albedo_path.is_file()


def test_rest_generator_cancellation_timeout_and_failed_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="ascii")
    generator = RestTextureVariationGenerator(
        TextureVariationSettings(endpoint="https://example.test", timeout_seconds=0.5)
    )

    with pytest.raises(TextureVariationError, match="cancelled"):
        generator.generate(
            _variation_request(source),
            output_dir=tmp_path / "cancelled-before-submit",
            cancel_check=lambda: True,
        )
    with pytest.raises(FileNotFoundError, match="source USD"):
        generator.generate(
            _variation_request(tmp_path / "missing.usda"),
            output_dir=tmp_path / "missing",
        )
    monkeypatch.setattr(
        generator,
        "_upload_asset",
        lambda path: f"https://example.test/assets/{path.name}",
    )

    poll_cancelled = iter((False, True))
    cancel_calls: list[str] = []

    def queued_or_cancel(method: str, url: str, payload=None):
        cancel_calls.append(method)
        return {"job_id": "job-cancel", "status": "queued"}

    monkeypatch.setattr(generator, "_request_json", queued_or_cancel)
    with pytest.raises(TextureVariationError, match="cancelled"):
        generator.generate(
            _variation_request(source),
            output_dir=tmp_path / "cancelled-while-polling",
            cancel_check=lambda: next(poll_cancelled),
        )
    assert cancel_calls == ["POST", "DELETE"]

    timeout_calls: list[str] = []

    def queued_or_timeout(method: str, url: str, payload=None):
        timeout_calls.append(method)
        return {"job_id": "job-timeout", "status": "queued"}

    monotonic = iter((0.0, 1.0))
    real_monotonic = texture_module.time.monotonic
    monkeypatch.setattr(generator, "_request_json", queued_or_timeout)
    monkeypatch.setattr(texture_module.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(TextureVariationError, match="timed out"):
        generator.generate(
            _variation_request(source), output_dir=tmp_path / "timed-out"
        )
    assert timeout_calls == ["POST", "DELETE"]
    monkeypatch.setattr(texture_module.time, "monotonic", real_monotonic)

    monkeypatch.setattr(
        generator,
        "_request_json",
        lambda *_args, **_kwargs: {
            "job_id": "job-failed",
            "status": "failed",
            "error_message": "backend unavailable",
        },
    )
    with pytest.raises(TextureVariationError, match="backend unavailable"):
        generator.generate(_variation_request(source), output_dir=tmp_path / "failed")


def test_rest_generator_validates_job_and_swallows_cancel_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="ascii")
    generator = RestTextureVariationGenerator(
        TextureVariationSettings(endpoint="http://example.test")
    )
    monkeypatch.setattr(
        generator,
        "_upload_asset",
        lambda path: f"http://example.test/assets/{path.name}",
    )
    monkeypatch.setattr(
        generator,
        "_request_json",
        lambda *_args, **_kwargs: {"job_id": "", "status": "completed"},
    )
    with pytest.raises(TextureVariationError, match="no job id"):
        generator.generate(_variation_request(source), output_dir=tmp_path / "no-job")

    monkeypatch.setattr(
        generator,
        "_request_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TextureVariationError("cancel failed")
        ),
    )
    generator._cancel_job("job-1")


def test_rest_json_and_artifact_transport_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generator = RestTextureVariationGenerator(
        TextureVariationSettings(endpoint="http://example.test")
    )
    monkeypatch.setattr(
        texture_module.urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *_open_args, **_open_kwargs: _BytesResponse(b'{"status": "ok"}')
        ),
    )
    assert generator._request_json("POST", "http://example.test/jobs", {"x": 1}) == {
        "status": "ok"
    }

    upload_source = tmp_path / "source.usdz"
    upload_source.write_bytes(b"package")
    monkeypatch.setattr(
        texture_module.urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *_open_args, **_open_kwargs: _BytesResponse(
                b'{"asset_uri": "http://example.test/v1/texture-variation-assets/va-1/source.usdz", "byte_size": 7}'
            )
        ),
    )
    assert generator._upload_asset(upload_source).endswith("/va-1/source.usdz")

    monkeypatch.setattr(
        texture_module.urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *_open_args, **_open_kwargs: _BytesResponse(b"[]")
        ),
    )
    with pytest.raises(TextureVariationError, match="non-object"):
        generator._request_json("GET", "http://example.test/jobs")

    monkeypatch.setattr(
        texture_module.urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *_open_args, **_open_kwargs: (_ for _ in ()).throw(
                urllib.error.URLError("offline")
            )
        ),
    )
    with pytest.raises(TextureVariationError, match="URLError"):
        generator._request_json("GET", "http://example.test/jobs")

    monkeypatch.setattr(
        texture_module.urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *_open_args, **_open_kwargs: _BytesResponse(b"texture-bytes")
        ),
    )
    downloaded = generator._localize_uri(
        "http://example.test/albedo.png", tmp_path / "http" / "albedo.png"
    )
    assert downloaded.read_bytes() == b"texture-bytes"

    with pytest.raises(TextureVariationError, match="configured endpoint origin"):
        generator._localize_uri(
            "http://assets.example.test/albedo.png",
            tmp_path / "wrong-host.png",
        )

    request = texture_module.urllib.request.Request("http://example.test/start")
    handler = texture_module._SameOriginRedirectHandler(("http", "example.test", 80))
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "http://example.test/final",
    )
    assert redirected is not None and redirected.full_url.endswith("/final")
    with pytest.raises(TextureVariationError, match="configured endpoint origin"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://internal.example/final",
        )
    with pytest.raises(TextureVariationError, match="artifact URI is invalid"):
        texture_module._require_same_origin(
            "not-an-http-uri", ("http", "example.test", 80)
        )

    limited = RestTextureVariationGenerator(
        TextureVariationSettings(
            endpoint="http://example.test",
            max_artifact_bytes=4,
        )
    )
    with pytest.raises(TextureVariationError, match="max_artifact_bytes"):
        limited._localize_uri(
            "http://example.test/large.png",
            tmp_path / "large.png",
        )
    assert not (tmp_path / "large.png").exists()
    local_large = tmp_path / "local-large.png"
    local_large.write_bytes(b"12345")
    with pytest.raises(TextureVariationError, match="max_artifact_bytes"):
        limited._localize_uri(
            local_large.as_uri(),
            tmp_path / "local-large-copy.png",
        )
    assert not (tmp_path / "local-large-copy.png").exists()

    monkeypatch.setattr(
        texture_module.urllib.request,
        "build_opener",
        lambda *_args: SimpleNamespace(
            open=lambda *_open_args, **_open_kwargs: (_ for _ in ()).throw(
                urllib.error.URLError("offline")
            )
        ),
    )
    with pytest.raises(TextureVariationError, match="failed to download"):
        generator._localize_uri(
            "http://example.test/offline.png",
            tmp_path / "offline.png",
        )

    local = tmp_path / "local.png"
    local.write_bytes(b"local")
    assert generator._localize_uri(local.as_uri(), local) == local
    converted: list[str] = []
    monkeypatch.setattr(
        texture_module.urllib.request,
        "url2pathname",
        lambda value: converted.append(value) or local.as_posix(),
    )
    assert generator._localize_uri("file://server/share/map.png", local) == local
    assert converted == ["//server/share/map.png"]
    copied = generator._localize_uri(local.as_posix(), tmp_path / "copy.png")
    assert copied.read_bytes() == b"local"
    with pytest.raises(TextureVariationError, match="unsupported"):
        generator._localize_uri("s3://bucket/map.png", tmp_path / "s3.png")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        generator._localize_uri(
            (tmp_path / "absent.png").as_posix(), tmp_path / "absent-copy.png"
        )


def test_rest_result_uses_normalized_map_fallback_and_requires_all_maps(
    tmp_path: Path,
) -> None:
    generator = RestTextureVariationGenerator(
        TextureVariationSettings(endpoint="http://example.test")
    )
    maps: dict[str, dict[str, str]] = {}
    for name in ("albedo", "normal", "orm"):
        source = tmp_path / f"{name}.png"
        source.write_bytes(name.encode("ascii"))
        maps[name] = {"uri": source.as_uri()}
    result = {
        "variant_asset_uri": "file:///tmp/variant.usda",
        "variant_name": "variant",
        "generated_textures": {"albedo": None, "normal": None, "orm": None},
        "maps": maps,
    }

    localized = generator._localize_result(result, tmp_path / "fallback")

    assert localized.orm_path.read_bytes() == b"orm"
    with pytest.raises(TextureVariationError, match="missing required maps"):
        generator._localize_result(
            {**result, "maps": {"albedo": maps["albedo"]}},
            tmp_path / "incomplete",
        )


def test_endpoint_and_output_safety_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        RestTextureVariationGenerator(
            TextureVariationSettings(endpoint="relative/service")
        )
    with pytest.raises(ValueError, match="query"):
        RestTextureVariationGenerator(
            TextureVariationSettings(endpoint="https://example.test?token=secret")
        )

    config = _config(tmp_path)
    config.output_dir.write_text("not a directory", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not a directory"):
        runner_module._prepare_output_dir(config)


def test_config_rejects_unknown_keys_and_invalid_variation_types(
    tmp_path: Path,
) -> None:
    mutations = (
        lambda data: data.update({"optimzation": {}}),
        lambda data: data["goal"].update({"colour": [0.1, 0.2, 0.3]}),
        lambda data: data["goal"].update({"base_color": [0.1, 0.2, 0.3]}),
        lambda data: data["goal"].update({"roughness": 0.5}),
        lambda data: data["goal"].update({"metallic": 0.0}),
        lambda data: data["goal"].update({"reference_image": "target.png"}),
        lambda data: data["goal"].update({"weights": {"colour": 1.0}}),
        lambda data: data["variation"].update({"strenth_min": 0.1}),
        lambda data: data.update({"search": {"max_colour_delta": 0.1}}),
        lambda data: data["source"].update({"texture_width": 4}),
        lambda data: data["optimization"].update({"trials": 1}),
        lambda data: data["judge"].update({"threshold": 0.8}),
    )
    for index, mutate in enumerate(mutations):
        data = _config_data(tmp_path / f"unknown-{index}")
        mutate(data)
        with pytest.raises(ValueError, match="unknown field"):
            MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)

    variation_data = _config_data(tmp_path / "unknown-variation")
    variation_data["variation_set"] = {"requested_count": 1, "distance": 0.2}
    with pytest.raises(ValueError, match="unknown field"):
        MaterialVariationConfig.from_mapping(variation_data, base_dir=tmp_path)

    with pytest.raises(ValueError, match="description"):
        MaterialVariationGoal(description=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="minimum_feature_distance"):
        MaterialVariationGoal(minimum_feature_distance=True)
    with pytest.raises(TypeError, match="diversity_weight"):
        MaterialVariationGoal(diversity_weight=False)

    with pytest.raises(ValueError, match="at most 8 camera views"):
        MaterialRefinementRenderSettings.from_mapping(
            {"camera_corners": ["+x+y+z"] * 9}
        )


def test_measured_source_bounds_preserve_metalness_class(tmp_path: Path) -> None:
    search = _config(tmp_path).search

    assert search.metallic_bounds(0.0) == (0.0, 0.0)
    assert search.metallic_bounds(1.0) == (1.0, 1.0)
    low = search.metallic_bounds(0.05)
    mixed = search.metallic_bounds(0.5)
    high = search.metallic_bounds(0.95)
    assert 0.0 <= low[0] <= low[1] <= 0.1
    assert 0.1 < mixed[0] <= mixed[1] < 0.9
    assert 0.9 <= high[0] <= high[1] <= 1.0


def test_out_of_bounds_prompt_inference_writes_failure_summary(tmp_path: Path) -> None:
    class OutOfBoundsVlm(_SequenceVlm):
        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            if str(kwargs.get("system_prompt", "")).startswith(
                "You infer normalized PBR controls"
            ):
                return json.dumps(
                    {
                        "base_color": [2.0, 2.0, 2.0],
                        "roughness": 2.0,
                        "metallic": 2.0,
                    }
                )
            return super().generate_with_image_caption_pairs(**kwargs)

    config = _config(tmp_path)

    with pytest.raises(
        MaterialRefinementDependencyError, match="appearance target inference failed"
    ):
        _run(config, vlm=OutOfBoundsVlm(_APPROVE))

    summary = json.loads(config.output_dir.joinpath("summary.json").read_text())
    assert summary["success"] is False
    assert summary["termination_reason"] == "target_inference_failed"
    assert summary["source_features"]["base_color"]


def test_failed_target_revision_stops_with_judged_evidence(tmp_path: Path) -> None:
    class InvalidRevisionVlm(_SequenceVlm):
        inference_calls = 0

        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            if str(kwargs.get("system_prompt", "")).startswith(
                "You infer normalized PBR controls"
            ):
                self.inference_calls += 1
                if self.inference_calls == 2:
                    self.calls.append(kwargs)
                    return "{}"
            return super().generate_with_image_caption_pairs(**kwargs)

    config = _config(tmp_path, max_refinements=1)

    result = _run(config, vlm=InvalidRevisionVlm(_CONTINUE))

    assert result.termination_reason == "target_revision_failed"
    assert result.success is False
    assert len(result.attempts) == 1
    assert result.attempts[0].judge_verdict is not None
    assert "required JSON" in str(result.attempts[0].error)
    revision = json.loads(
        config.output_dir.joinpath(
            "attempts/attempt_001/evidence/target_revision.json"
        ).read_text(encoding="utf-8")
    )
    assert revision["error_type"] == "ValueError"


def test_cancellation_during_target_revision_preserves_judged_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_inference = runner_module.infer_material_target
    inference_calls = 0

    def cancel_second_inference(**kwargs: Any):
        nonlocal inference_calls
        inference_calls += 1
        if inference_calls == 2:
            raise OptimizationCancelledError("cancel target revision")
        return real_inference(**kwargs)

    monkeypatch.setattr(runner_module, "infer_material_target", cancel_second_inference)
    config = _config(tmp_path, max_refinements=1)

    result = _run(config, vlm=_SequenceVlm(_CONTINUE))

    assert result.termination_reason == "cancelled"
    assert result.success is False
    assert len(result.attempts) == 1
    assert result.attempts[0].cancelled is True
    assert result.best_artifacts["rendered_image_1"].is_file()


def test_source_material_failure_writes_terminal_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "_build_source_material",
        lambda _config: (_ for _ in ()).throw(RuntimeError("source unavailable")),
    )

    with pytest.raises(MaterialRefinementDependencyError, match="source unavailable"):
        _run(config)

    summary = json.loads(config.output_dir.joinpath("summary.json").read_text())
    assert summary["success"] is False
    assert summary["termination_reason"] == "source_material_failed"
    assert Path(summary["goal_path"]).is_file()


def test_missing_source_texture_fails_closed(tmp_path: Path) -> None:
    data = _config_data(tmp_path / "strict-source")
    data["source"]["textures"]["normal"] = "missing-normal.png"

    with pytest.raises(FileNotFoundError, match="source texture does not exist"):
        MaterialRefinementConfig.from_mapping(data, base_dir=tmp_path)


def test_marked_overwrite_and_runner_helpers(tmp_path: Path) -> None:
    config = _config(tmp_path, overwrite=True)
    first = _run(config)
    stale = config.output_dir / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    second = _run(config)

    assert first.approved and second.approved
    assert not stale.exists()

    class EmptyMessageError(RuntimeError):
        def __str__(self) -> str:
            return ""

    assert runner_module._safe_error(EmptyMessageError()) == "EmptyMessageError"
    assert runner_module._candidate_prompt("base", iteration=1, feedback=None) == "base"
    calls: list[dict[str, float]] = []
    wrapped = runner_module._priority_first_runner(
        ({"strength": 0.75}, {"strength": 0.5}),
        lambda *_args, **_kwargs: pytest.fail("unexpected"),
    )
    wrapped(
        SimpleNamespace(params=()),
        lambda params: calls.append(params) or 0.0,
        max_trials=1,
        seed=7,
    )
    assert calls == [{"strength": 0.75}]

    cancelled_calls: list[dict[str, float]] = []
    wrapped(
        SimpleNamespace(params=()),
        lambda params: cancelled_calls.append(params) or 0.0,
        max_trials=2,
        seed=7,
        cancel_check=lambda: True,
    )
    assert cancelled_calls == []


def test_failed_generation_records_a_safe_failed_replica(tmp_path: Path) -> None:
    class FailingGenerator:
        name = "failing-generator"

        def generate(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError("generation failed")

    config = _config(tmp_path)

    with pytest.raises(MaterialRefinementError, match="visual evaluation failed"):
        _run(config, generator=FailingGenerator())

    history = json.loads(config.output_dir.joinpath("history.jsonl").read_text())
    assert history["failed"] is True
    error_path = Path(history["replicas"][0]["artifacts"]["error"])
    assert json.loads(error_path.read_text())["error"] == "generation failed"


def test_optimizer_cancellation_before_judgment_is_preserved(tmp_path: Path) -> None:
    class CancellingGenerator:
        name = "cancelling-generator"

        def generate(self, *_args: Any, **_kwargs: Any):
            raise OptimizationCancelledError("stop")

    config = _config(tmp_path)

    with pytest.raises(MaterialRefinementCancelled, match="visual judgment"):
        _run(config, generator=CancellingGenerator())

    summary = json.loads(config.output_dir.joinpath("summary.json").read_text())
    assert summary["termination_reason"] == "cancelled"
    assert summary["attempts"][0]["cancelled"] is True


def test_cancellation_during_visual_evaluation_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner_module,
        "render_material_swatch",
        lambda **_kwargs: (_ for _ in ()).throw(
            OptimizationCancelledError("cancel render")
        ),
    )
    config = _config(tmp_path)

    with pytest.raises(MaterialRefinementCancelled, match="visual judgment"):
        _run(config)

    assert (config.output_dir / "summary.json").is_file()


def test_cancellation_after_judgment_returns_evidence_and_cancelled_api(
    tmp_path: Path,
) -> None:
    cancelled = False

    class CancellingVlm(_SequenceVlm):
        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            nonlocal cancelled
            response = super().generate_with_image_caption_pairs(**kwargs)
            if not str(kwargs.get("system_prompt", "")).startswith(
                "You infer normalized PBR controls"
            ):
                cancelled = True
            return response

    output = api_module.run_material_refinement_api(
        api_module.MaterialRefinementInput(
            config=_config_data(tmp_path / "cancelled-after-judge"),
            cancel_checker=lambda: cancelled,
            generator=_FakeGenerator(),
            rendering_backend=MockRenderingBackend(),
            vlm_judge=CancellingVlm(_APPROVE),
        )
    )

    assert output.success is False
    assert output.cancelled is True
    assert output.termination_reason == "cancelled"
    summary = json.loads(output.summary_path.read_text())
    assert summary["success"] is False
    assert output.best_artifacts["judge_verdict"].is_file()
    assert output.best_artifacts["rendered_image_1"].is_file()


def test_visual_failure_writes_evidence_and_stops(tmp_path: Path) -> None:
    class SkippedRenderTask:
        def run(self, context: dict[str, Any], object_store: Any = None):
            return {**context, "rendering_skipped": True}

    config = _config(tmp_path)

    with pytest.raises(MaterialRefinementError, match="visual evaluation failed"):
        _run(config, render_task=SkippedRenderTask())

    error_path = config.output_dir / "attempts/attempt_001/evidence/error.json"
    assert json.loads(error_path.read_text())["error_type"] == "RuntimeError"


def test_later_visual_failure_retains_evidence_but_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_render = runner_module.render_material_swatch
    render_calls = 0

    def fail_second_render(**kwargs: Any):
        nonlocal render_calls
        render_calls += 1
        if render_calls == 2:
            raise RuntimeError("renderer became unavailable")
        return original_render(**kwargs)

    monkeypatch.setattr(runner_module, "render_material_swatch", fail_second_render)
    output = api_module.run_material_refinement_api(
        api_module.MaterialRefinementInput(
            config=_config_data(tmp_path / "later-visual-failure", max_refinements=1),
            generator=_FakeGenerator(),
            rendering_backend=MockRenderingBackend(),
            vlm_judge=_SequenceVlm(_CONTINUE),
        )
    )

    assert output.success is False
    assert output.cancelled is False
    assert output.termination_reason == "visual_evaluation_failed"
    assert output.diagnostic == (
        "Material refinement stopped: visual_evaluation_failed"
    )
    assert output.best_artifacts["rendered_image_1"].is_file()
    summary = json.loads(output.summary_path.read_text())
    assert summary["success"] is False


def test_service_artifact_uri_is_redacted_from_history(tmp_path: Path) -> None:
    class CredentialUriGenerator(_FakeGenerator):
        def generate(
            self,
            request: TextureVariationRequest,
            *,
            output_dir: Path,
            cancel_check: Any = None,
        ) -> TextureVariationArtifacts:
            result = super().generate(
                request, output_dir=output_dir, cancel_check=cancel_check
            )
            return replace(
                result,
                variant_asset_uri=(
                    "https://user:credential-value@example.test/output.usda"
                    "?token=credential-value"
                ),
            )

    config = _config(tmp_path)
    result = _run(config, generator=CredentialUriGenerator())

    history = result.history_path.read_text(encoding="utf-8")
    assert "credential-value" not in history
    assert '"variant_asset_uri": "<redacted>"' in history


def test_variation_output_overwrite_callback_cancel_and_unapproved_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = _config_data(tmp_path / "variations")
    data["variation_set"] = {"requested_count": 1}
    config = MaterialVariationConfig.from_mapping(data, base_dir=tmp_path)
    config.refinement.output_dir.mkdir()
    config.refinement.output_dir.joinpath(".material-variation-output").write_text(
        "old", encoding="ascii"
    )
    config.refinement.output_dir.joinpath("stale").write_text("old")
    overwrite_config = replace(
        config, refinement=replace(config.refinement, overwrite=True)
    )
    assert variation_module._prepare_output(overwrite_config).is_dir()
    assert not overwrite_config.refinement.output_dir.joinpath("stale").exists()

    unapproved_data = _config_data(tmp_path / "unapproved")
    unapproved_data["variation_set"] = {"requested_count": 1}
    unapproved = MaterialVariationConfig.from_mapping(
        unapproved_data, base_dir=tmp_path
    )
    callbacks: list[tuple[int, int]] = []
    result = run_material_variations(
        unapproved,
        on_trial=lambda slot, attempt, _trial: callbacks.append((slot, attempt)),
        generator=_FakeGenerator(),
        rendering_backend=MockRenderingBackend(),
        vlm_judge=_SequenceVlm(_CONTINUE),
    )
    assert result.status == "incomplete"
    assert result.slots[0].result is not None
    assert callbacks == [(1, 1)]

    cancelled_data = _config_data(tmp_path / "cancelled")
    cancelled_data["variation_set"] = {"requested_count": 1}
    cancelled = MaterialVariationConfig.from_mapping(cancelled_data, base_dir=tmp_path)
    monkeypatch.setattr(
        variation_module,
        "run_material_refinement",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MaterialRefinementCancelled("cancelled in slot")
        ),
    )
    cancelled_result = run_material_variations(cancelled)
    assert cancelled_result.status == "cancelled"
    assert cancelled_result.slots[0].error == "cancelled in slot"

    cancelled_after_judge = False

    class CancellingVlm(_SequenceVlm):
        def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
            nonlocal cancelled_after_judge
            response = super().generate_with_image_caption_pairs(**kwargs)
            if not str(kwargs.get("system_prompt", "")).startswith(
                "You infer normalized PBR controls"
            ):
                cancelled_after_judge = True
            return response

    judged_data = _config_data(tmp_path / "cancelled-after-judge-variation")
    judged_data["variation_set"] = {"requested_count": 1}
    judged = MaterialVariationConfig.from_mapping(judged_data, base_dir=tmp_path)
    monkeypatch.setattr(
        variation_module, "run_material_refinement", run_material_refinement
    )
    judged_result = run_material_variations(
        judged,
        cancel_check=lambda: cancelled_after_judge,
        generator=_FakeGenerator(),
        rendering_backend=MockRenderingBackend(),
        vlm_judge=CancellingVlm(_APPROVE),
    )
    assert judged_result.status == "cancelled"
    assert judged_result.slots[0].result is not None
    assert judged_result.slots[0].result.best_artifacts["judge_verdict"].is_file()


def test_variation_goal_and_output_type_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="description"):
        MaterialVariationGoal(description="")
    with pytest.raises(ValueError, match="requested_count"):
        MaterialVariationGoal(requested_count=17)
    with pytest.raises(ValueError, match="minimum_feature_distance"):
        MaterialVariationGoal(minimum_feature_distance=float("nan"))
    with pytest.raises(ValueError, match="diversity_weight"):
        MaterialVariationGoal(diversity_weight=-1.0)

    data = _config_data(tmp_path / "bad-config")
    data["variation_set"] = []
    with pytest.raises(TypeError, match="variation_set"):
        MaterialVariationConfig.from_mapping(data, base_dir=tmp_path)

    file_output_data = _config_data(tmp_path / "file-output")
    file_output_data["variation_set"] = {"requested_count": 1}
    file_output = MaterialVariationConfig.from_mapping(
        file_output_data, base_dir=tmp_path
    )
    file_output.refinement.output_dir.write_text("file", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not a directory"):
        variation_module._prepare_output(file_output)


def test_api_loads_yaml_and_projects_variation_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config_data(tmp_path / "api")))
    params = api_module.MaterialRefinementInput(
        config=config_path,
        output_dir_override=tmp_path / "override",
        config_path=config_path,
    )
    data, base_dir = api_module._load_mapping(params.config, params.config_path)
    assert data["goal"]["appearance_prompt"] == "satin blue painted metal"
    assert base_dir == tmp_path

    variation_result = SimpleNamespace(
        success=False,
        status="incomplete",
        termination_reason="slot_incomplete",
        selected_materials=(),
        manifest_path=tmp_path / "manifest.json",
        material_library_path=None,
        materials_manifest_path=None,
        selected_render_paths=(),
    )
    monkeypatch.setattr(
        api_module,
        "run_material_variations",
        lambda *_args, **_kwargs: variation_result,
    )
    output = api_module.run_material_variations_api(params)
    assert output.success is False
    assert output.status == "incomplete"
    assert output.diagnostic == "Variation set is incomplete"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        api_module._load_mapping(invalid, None)


@pytest.mark.parametrize(
    ("error", "cancelled", "diagnostic"),
    (
        (
            MaterialRefinementCancelled("cancelled"),
            True,
            "Material refinement was cancelled",
        ),
        (
            MaterialRefinementDependencyError("missing optimizer"),
            False,
            "missing optimizer",
        ),
        (RuntimeError("secret failure"), False, "inspect local logs"),
    ),
)
def test_refinement_api_failure_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    cancelled: bool,
    diagnostic: str,
) -> None:
    monkeypatch.setattr(api_module.logger, "exception", lambda *_args: None)
    monkeypatch.setattr(
        api_module,
        "run_material_refinement",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    output = api_module.run_material_refinement_api(
        api_module.MaterialRefinementInput(config=_config_data(tmp_path / "api"))
    )
    assert output.success is False
    assert output.cancelled is cancelled
    assert diagnostic in (output.diagnostic or "")


@pytest.mark.parametrize(
    ("operation", "controller"),
    (
        (api_module.run_material_refinement_api, "material-refinement controller"),
        (api_module.run_material_variations_api, "material-variation controller"),
    ),
)
def test_compatibility_api_deprecation_warnings_name_the_controller(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
    controller: str,
) -> None:
    monkeypatch.setattr(api_module.logger, "exception", lambda *_args: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        operation(api_module.MaterialRefinementInput(config={"invalid": True}))

    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert controller in str(caught[0].message)


@pytest.mark.parametrize(
    "operation",
    (
        api_module.run_material_refinement_api,
        api_module.run_material_variations_api,
    ),
)
def test_compatibility_api_preserves_result_contract_when_warnings_are_errors(
    monkeypatch: pytest.MonkeyPatch,
    operation: Any,
) -> None:
    monkeypatch.setattr(api_module.logger, "exception", lambda *_args: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = operation(api_module.MaterialRefinementInput(config={"invalid": True}))

    assert result.success is False
    assert result.error is not None


def test_variation_api_failure_and_convenience_wrappers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api_module.logger, "exception", lambda *_args: None)
    monkeypatch.setattr(
        api_module,
        "run_material_variations",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    failed = api_module.run_material_variations_api(
        api_module.MaterialRefinementInput(config=_config_data(tmp_path / "failed"))
    )
    assert failed.success is False
    assert failed.error == "Material variation failed"

    refinement_sentinel = api_module.MaterialRefinementOutput(success=True)
    variation_sentinel = api_module.MaterialVariationOutput(success=True)
    refinement_inputs: list[api_module.MaterialRefinementInput] = []
    variation_inputs: list[api_module.MaterialRefinementInput] = []
    monkeypatch.setattr(
        api_module,
        "run_material_refinement_api",
        lambda params: refinement_inputs.append(params) or refinement_sentinel,
    )
    monkeypatch.setattr(
        api_module,
        "run_material_variations_api",
        lambda params: variation_inputs.append(params) or variation_sentinel,
    )
    config = _config_data(tmp_path / "convenience")
    output_dir = tmp_path / "override"
    assert (
        api_module.refine_material(config, output_dir=output_dir) is refinement_sentinel
    )
    assert (
        api_module.create_material_variations(config, output_dir=output_dir)
        is variation_sentinel
    )
    assert refinement_inputs[0].output_dir_override == output_dir
    assert variation_inputs[0].output_dir_override == output_dir
