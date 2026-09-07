# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from apps.texture_gen_service_common import WeatheringControls
from apps.texture_gen_step1x_service.weathering import (
    WeatheringPlan,
    _density_mask,
    _effect_score,
    _local_image_path,
    _quality_failures,
    _repetition_peak,
    _rgb,
    _shape_score,
    _shift_without_wrap,
    apply_weathering_plan,
    plan_weathering_request,
)
from PIL import Image


def test_prompt_first_planner_keeps_normalized_controls_internal() -> None:
    plan = plan_weathering_request(
        text_prompt=(
            "localized rust around worn lower joints; exclude the rubber feet"
        ),
        controls=WeatheringControls(
            protected_mask_uri="file:///protected.png",
        ),
        strength=0.6,
    )

    assert plan is not None
    assert plan.effect == "rust"
    assert plan.density == 0.2
    assert plan.direction_degrees == 90.0
    assert plan.severity == 0.6
    assert plan.protected_mask_uri == "file:///protected.png"

    top_plan = plan_weathering_request(
        text_prompt="light dust on the upper surfaces",
        controls=None,
        strength=0.4,
    )
    assert top_plan is not None
    assert top_plan.directionality == 0.35
    assert top_plan.direction_degrees == -90.0


def test_prompt_first_planner_fails_closed_on_ambiguous_or_orphan_controls() -> None:
    with pytest.raises(ValueError, match="multiple weathering classes"):
        plan_weathering_request(
            text_prompt="add rust and dust to this material",
            controls=None,
            strength=0.8,
        )
    with pytest.raises(ValueError, match="mask controls require"):
        plan_weathering_request(
            text_prompt="clean blue paint",
            controls=WeatheringControls(editable_mask_uri="file:///editable.png"),
            strength=0.8,
        )

    assert (
        plan_weathering_request(
            text_prompt="clean blue paint",
            controls=None,
            strength=0.8,
        )
        is None
    )


def _write_rgb(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)
    return path


def _candidate_images(tmp_path: Path, *, effect: str) -> tuple[Path, Path, Path]:
    size = 64
    source = np.full((size, size, 3), 72, dtype=np.uint8)
    rng = np.random.default_rng(1045 if effect == "rust" else 1046)
    variation = rng.integers(0, 180, size=(size, size), dtype=np.uint8)
    if effect == "rust":
        candidate = np.stack(
            [np.clip(90 + variation, 0, 255), 45 + variation // 4, 25 + variation // 8],
            axis=2,
        )
    else:
        candidate = np.stack(
            [100 + variation // 2, 92 + variation // 2, 75 + variation // 3],
            axis=2,
        )
    orm = np.empty_like(source)
    orm[:, :, 0] = 255
    orm[:, :, 1] = 225
    orm[:, :, 2] = 255 if effect == "rust" else 0
    return (
        _write_rgb(tmp_path / "source.png", source),
        _write_rgb(tmp_path / "candidate.png", candidate),
        _write_rgb(tmp_path / "candidate_orm.png", orm),
    )


def test_rust_localizes_albedo_and_correlated_pbr_channels(tmp_path: Path) -> None:
    source, candidate, orm = _candidate_images(tmp_path, effect="rust")
    editable_path = tmp_path / "editable.png"
    Image.new("L", (64, 64), 255).save(editable_path)
    protected = np.zeros((64, 64), dtype=np.uint8)
    protected[:, 48:] = 255
    protected_path = tmp_path / "protected.png"
    Image.fromarray(protected, mode="L").save(protected_path)

    outputs = apply_weathering_plan(
        albedo_uri=candidate.as_uri(),
        orm_uri=orm.as_uri(),
        source_albedo_path=source,
        source_orm_path=None,
        source_roughness=0.25,
        source_metalness=1.0,
        spec=WeatheringPlan(
            effect="rust",
            density=0.3,
            scale=0.08,
            severity=0.8,
            editable_mask_uri=editable_path.as_uri(),
            protected_mask_uri=protected_path.as_uri(),
        ),
        output_dir=tmp_path / "out",
    )

    evidence = outputs.metadata
    assert evidence["status"] == "pass", evidence["failures"]
    assert evidence["metrics"]["roughness_mask_correlation"] >= 0.5
    assert evidence["metrics"]["metallic_mask_correlation"] <= -0.5
    assert evidence["metrics"]["protected_max_delta"] == 0.0
    assert evidence["artifacts"]["weathering_mask_sha256"]
    assert outputs.auxiliary_artifacts["masks"]["editable"]["sha256"]
    with Image.open(Path(outputs.orm_uri.removeprefix("file://"))) as image:
        final_orm = np.asarray(image.convert("RGB"))
    assert np.all(final_orm[:, 48:, :] == np.array([255, 63, 255]))


def test_dust_preserves_dielectric_metallic_and_correlates_roughness(
    tmp_path: Path,
) -> None:
    source, candidate, orm = _candidate_images(tmp_path, effect="dust")
    outputs = apply_weathering_plan(
        albedo_uri=candidate.as_uri(),
        orm_uri=orm.as_uri(),
        source_albedo_path=source,
        source_orm_path=None,
        source_roughness=0.4,
        source_metalness=0.0,
        spec=WeatheringPlan(
            effect="dust",
            density=0.35,
            scale=0.16,
            directionality=0.35,
            direction_degrees=0.0,
            severity=0.7,
        ),
        output_dir=tmp_path / "out",
    )

    assert outputs.metadata["status"] == "pass", outputs.metadata["failures"]
    assert outputs.metadata["metrics"]["roughness_mask_correlation"] >= 0.5
    with Image.open(Path(outputs.orm_uri.removeprefix("file://"))) as image:
        metallic = np.asarray(image.convert("RGB"))[:, :, 2]
    assert not np.any(metallic)


def test_rust_rejects_dielectric_source(tmp_path: Path) -> None:
    source, candidate, orm = _candidate_images(tmp_path, effect="rust")
    with pytest.raises(ValueError, match="metallic source"):
        apply_weathering_plan(
            albedo_uri=candidate.as_uri(),
            orm_uri=orm.as_uri(),
            source_albedo_path=source,
            source_orm_path=None,
            source_roughness=0.5,
            source_metalness=0.0,
            spec=WeatheringPlan(
                effect="rust",
                density=0.2,
                scale=0.12,
                directionality=0.25,
                direction_degrees=0.0,
                severity=0.8,
            ),
            output_dir=tmp_path / "out",
        )


def test_rust_respects_metallic_mask_as_material_boundary(tmp_path: Path) -> None:
    source, candidate, orm = _candidate_images(tmp_path, effect="rust")
    source_orm = np.empty((64, 64, 3), dtype=np.uint8)
    source_orm[:, :, 0] = 255
    source_orm[:, :, 1] = 64
    source_orm[:, :32, 2] = 255
    source_orm[:, 32:, 2] = 0
    source_orm_path = _write_rgb(tmp_path / "source_orm.png", source_orm)

    outputs = apply_weathering_plan(
        albedo_uri=candidate.as_uri(),
        orm_uri=orm.as_uri(),
        source_albedo_path=source,
        source_orm_path=source_orm_path,
        source_roughness=0.25,
        source_metalness=1.0,
        spec=WeatheringPlan(
            effect="rust",
            density=0.3,
            scale=0.08,
            severity=0.8,
        ),
        output_dir=tmp_path / "out",
    )

    assert outputs.metadata["status"] == "pass", outputs.metadata["failures"]
    assert outputs.metadata["metrics"]["material_boundary_max_delta"] == 0.0
    with Image.open(Path(outputs.albedo_uri.removeprefix("file://"))) as image:
        final_albedo = np.asarray(image.convert("RGB"))
    with Image.open(source) as image:
        source_albedo = np.asarray(image.convert("RGB"))
    assert np.array_equal(final_albedo[:, 32:, :], source_albedo[:, 32:, :])


def test_repetition_metric_detects_periodic_tiling() -> None:
    yy, xx = np.mgrid[:128, :128]
    periodic = ((xx // 8 + yy // 8) % 2).astype(np.float32)
    assert _repetition_peak(periodic, 0.05) > 0.65


def test_image_helpers_accept_plain_paths_and_resize(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 3), (10, 20, 30)).save(source)

    assert _local_image_path(str(source), label="source") == source.resolve()
    assert _rgb(source, size=(4, 5)).shape == (5, 4, 3)

    with pytest.raises(ValueError, match="not a readable local file"):
        _local_image_path("https://example.invalid/source.png", label="source")


@pytest.mark.parametrize("effect", ["wear", "dirt"])
def test_effect_score_supports_non_rust_weathering(effect: str) -> None:
    source = np.zeros((2, 2, 3), dtype=np.float32)
    generated = np.full((2, 2, 3), 0.5, dtype=np.float32)

    score = _effect_score(generated, source, effect)

    assert score.shape == (2, 2)
    assert np.all(score == 1.0)


def test_directional_shape_does_not_wrap_across_atlas_edges() -> None:
    score = np.zeros((16, 16), dtype=np.float32)
    score[8, 0] = 1.0
    shaped = _shape_score(
        score,
        WeatheringPlan(
            effect="wear",
            scale=0.5,
            directionality=1.0,
            direction_degrees=0.0,
        ),
    )

    assert shaped.shape == score.shape
    assert not np.any(_shift_without_wrap(score, 16, 0))
    shifted = _shift_without_wrap(score, 2, -1)
    assert shifted[7, 2] == 1.0
    assert shifted[8, 0] == 0.0


def test_density_and_repetition_degenerate_limits_are_deterministic() -> None:
    score = np.arange(64, dtype=np.float32).reshape(8, 8)

    assert not np.any(_density_mask(score, 0.0))
    assert np.all(_density_mask(score, 1.0))
    assert _repetition_peak(np.ones((16, 16), dtype=np.float32), 0.1) == 0.0
    assert _repetition_peak(score, 0.1) == 0.0


def test_quality_contract_reports_every_fail_closed_reason() -> None:
    rust_failures = _quality_failures(
        effect="rust",
        localization=0.5,
        protected_delta=1.0,
        material_boundary_delta=1.0,
        repetition_peak=1.0,
        roughness_correlation=None,
        metallic_correlation=0.0,
        metallic_max_delta=1.0,
    )
    dust_failures = _quality_failures(
        effect="dust",
        localization=1.0,
        protected_delta=0.0,
        material_boundary_delta=0.0,
        repetition_peak=0.0,
        roughness_correlation=1.0,
        metallic_correlation=None,
        metallic_max_delta=1.0,
    )

    assert rust_failures == [
        "weathering changes are not localized to the effect mask",
        "protected pixels changed",
        "weathering crossed a source material boundary",
        "weathering mask has a strong repeated spatial peak",
        "roughness does not correlate with localized weathering",
        "metallic does not decrease with localized rust",
    ]
    assert dust_failures == ["non-rust weathering changed metallic"]
