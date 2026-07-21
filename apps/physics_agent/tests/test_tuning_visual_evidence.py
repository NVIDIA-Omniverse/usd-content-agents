# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for physics visual-judge evidence preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_agent.tuning.visual_evidence import (
    JudgeVisualEvidence,
    backend_supports_reasoning_effort,
    generated_frame_caption,
    has_reference_media,
    prepare_reference_media,
    resolve_default_judge_vlm,
    sample_visual_evidence_items,
    validate_visual_frame_count,
    write_comparison_contact_sheet,
)


def test_unknown_backend_does_not_support_reasoning_effort() -> None:
    assert backend_supports_reasoning_effort("missing-vlm-backend") is False


def test_prepare_reference_images_copies_and_captions(tmp_path: Path) -> None:
    src = tmp_path / "input.png"
    src.write_bytes(b"fake image bytes")

    evidence = prepare_reference_media(
        reference_images=[src],
        reference_descriptions=["target pose"],
        output_dir=tmp_path / "out",
    )

    assert evidence.has_reference_media is True
    assert len(evidence.reference_image_caption_pairs) == 1
    caption, copied = evidence.reference_image_caption_pairs[0]
    assert caption == "Reference Image 1: target pose"
    assert copied == (
        tmp_path / "out" / "reference_media" / "images" / "reference_image_01.png"
    )
    assert copied.read_bytes() == b"fake image bytes"


def test_prepare_reference_media_validates_description_count(tmp_path: Path) -> None:
    src = tmp_path / "input.png"
    src.write_bytes(b"fake image bytes")

    with pytest.raises(ValueError, match="reference_descriptions"):
        prepare_reference_media(
            reference_images=[src],
            reference_descriptions=[],
            output_dir=tmp_path / "out",
        )


def test_prepare_reference_media_validates_files_and_extensions(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match="reference image not found"):
        prepare_reference_media(reference_images=[missing], output_dir=tmp_path / "out")

    bad_ext = tmp_path / "reference.gif"
    bad_ext.write_bytes(b"not supported")
    with pytest.raises(ValueError, match="Unsupported reference image extension"):
        prepare_reference_media(reference_images=[bad_ext], output_dir=tmp_path / "out")


def test_prepare_reference_video_captions_include_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake video bytes")

    def fake_extract_frames(
        _video_path: Path,
        output_dir: Path,
        *,
        n: int,
    ) -> list[Path]:
        assert n == 3
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = output_dir / "frame_0001__t500.png"
        frame.write_bytes(b"fake frame bytes")
        return [frame]

    monkeypatch.setattr(
        "world_understanding.functions.cv.video_frames.extract_frames",
        fake_extract_frames,
    )

    evidence = prepare_reference_media(
        reference_videos=[src],
        reference_video_descriptions=["target motion"],
        output_dir=tmp_path / "out",
        frames_per_video=3,
    )

    assert evidence.reference_image_caption_pairs[0][0] == (
        "Reference Video 1 - Frame 1 (t=0.500s): target motion"
    )


def test_generated_frame_caption_includes_timestamp() -> None:
    assert generated_frame_caption(2, "frame_0002__t1250.png") == (
        "Generated Physics Output - Frame 2 (t=1.250s):"
    )


def test_sample_visual_evidence_items_preserves_endpoints(tmp_path: Path) -> None:
    references = tuple(
        (f"Reference Image {idx}: target", tmp_path / f"ref_{idx:02d}.png")
        for idx in range(1, 21)
    )
    generated = tuple(tmp_path / f"frame_{idx:04d}.png" for idx in range(1, 61))
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=references,
        generated_image_paths=generated,
    )

    reference_items, generated_items = sample_visual_evidence_items(
        evidence,
        max_reference_images=5,
        max_generated_images=7,
    )

    assert len(reference_items) == 5
    assert len(generated_items) == 7
    assert reference_items[0] == references[0]
    assert reference_items[-1] == references[-1]
    assert generated_items[0] == ("Generated Physics Output - Frame 1:", generated[0])
    assert generated_items[-1] == (
        "Generated Physics Output - Frame 60:",
        generated[-1],
    )


@pytest.mark.parametrize("value", [0, 65, True, 3.0, "3"])
def test_validate_visual_frame_count_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="frames"):
        validate_visual_frame_count("frames", value)  # type: ignore[arg-type]


def test_write_comparison_contact_sheet_and_metadata(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")

    reference = tmp_path / "reference.png"
    generated = tmp_path / "frame_0001__t250.png"
    Image.new("RGB", (8, 8), "red").save(reference)
    Image.new("RGB", (8, 8), "blue").save(generated)

    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(("Reference Image 1: target", reference),),
        generated_image_paths=(generated,),
    )
    comparison_path, comparison_error = write_comparison_contact_sheet(
        evidence,
        tmp_path / "comparison.png",
    )
    evidence = evidence.with_comparison_image(
        comparison_path,
        comparison_error=comparison_error,
    )

    assert comparison_path == tmp_path / "comparison.png"
    assert comparison_path.exists()
    assert comparison_error is None
    assert evidence.to_metadata() == {
        "reference_images": [
            {"caption": "Reference Image 1: target", "path": str(reference)}
        ],
        "generated_images": [
            {
                "caption": "Generated Physics Output - Frame 1 (t=0.250s):",
                "path": str(generated),
            }
        ],
        "comparison_image": str(comparison_path),
        "reference_error": None,
        "generated_error": None,
        "comparison_error": None,
    }


def test_write_comparison_contact_sheet_reports_invalid_limits(tmp_path: Path) -> None:
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(
            ("Reference Image 1: target", tmp_path / "r.png"),
        ),
        generated_image_paths=(tmp_path / "g.png",),
    )

    comparison_path, comparison_error = write_comparison_contact_sheet(
        evidence,
        tmp_path / "comparison.png",
        max_reference_images=0,
    )

    assert comparison_path is None
    assert comparison_error is not None
    assert "max_reference_images" in comparison_error


def test_write_comparison_contact_sheet_reports_all_image_failures(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL.Image")

    missing_reference = tmp_path / "missing_reference.png"
    missing_generated = tmp_path / "missing_generated.png"
    evidence = JudgeVisualEvidence(
        reference_image_caption_pairs=(
            ("Reference Image 1: target", missing_reference),
        ),
        generated_image_paths=(missing_generated,),
    )

    comparison_path, comparison_error = write_comparison_contact_sheet(
        evidence,
        tmp_path / "comparison.png",
    )

    assert comparison_path is None
    assert comparison_error is not None
    assert "missing_reference.png" in comparison_error
    assert "missing_generated.png" in comparison_error


def test_default_judge_vlm_filters_reasoning_effort_for_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.api import defaults

    captured: dict[str, object] = {}

    def fake_create_vlm(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(defaults, "DEFAULT_VLM_BACKEND", "nim")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MODEL", "model")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MAX_TOKENS", 123)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(
        "world_understanding.agentic.config.get_api_key_for_model_config",
        lambda _backend, _config, _model_type: "key",
    )
    monkeypatch.setattr(
        "world_understanding.functions.models.vision_language_models.create_vlm",
        fake_create_vlm,
    )

    resolve_default_judge_vlm()

    assert captured["backend"] == "nim"
    assert captured["api_key"] == "key"
    assert "reasoning_effort" not in captured


def test_default_judge_vlm_honors_nim_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.api import defaults

    for env_var in (
        "WU_VLM_NIM_BASE_URL",
        "PA_VLM_NIM_BASE_URL",
        "TA_VLM_NIM_BASE_URL",
        "MA_VLM_NIM_BASE_URL",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("PA_VLM_NIM_BASE_URL", "http://localhost:9000/v1")

    captured: dict[str, object] = {}

    def fake_create_vlm(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(defaults, "DEFAULT_VLM_BACKEND", "openai")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MODEL", "model")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MAX_TOKENS", 123)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(
        "world_understanding.agentic.config.get_api_key_for_model_config",
        lambda _backend, _config, _model_type: "not-used",
    )
    monkeypatch.setattr(
        "world_understanding.functions.models.vision_language_models.create_vlm",
        fake_create_vlm,
    )

    resolve_default_judge_vlm()

    assert captured["backend"] == "nim"
    assert captured["base_url"] == "http://localhost:9000/v1"
    assert captured["api_key"] == "not-used"
    assert "reasoning_effort" not in captured


def test_default_judge_vlm_honors_configured_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.api import defaults

    for env_var in (
        "WU_VLM_NIM_BASE_URL",
        "PA_VLM_NIM_BASE_URL",
        "TA_VLM_NIM_BASE_URL",
        "MA_VLM_NIM_BASE_URL",
    ):
        monkeypatch.delenv(env_var, raising=False)

    captured: dict[str, object] = {}

    monkeypatch.setattr(defaults, "DEFAULT_VLM_BACKEND", "openai")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MODEL", "custom-model")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MAX_TOKENS", 123)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_BASE_URL", "https://vlm.example.test/v1")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_API_KEY", None)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_API_KEY_ENV", "CUSTOM_VLM_KEY")
    monkeypatch.setenv("CUSTOM_VLM_KEY", "endpoint-key")
    monkeypatch.setattr(
        "world_understanding.functions.models.vision_language_models.create_vlm",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    resolve_default_judge_vlm()

    assert captured["backend"] == "openai"
    assert captured["model"] == "custom-model"
    assert captured["base_url"] == "https://vlm.example.test/v1"
    assert captured["api_key"] == "endpoint-key"
    assert captured["reasoning_effort"] == "high"


def test_default_judge_vlm_rejects_unset_configured_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from physics_agent.api import defaults

    for env_var in (
        "WU_VLM_NIM_BASE_URL",
        "PA_VLM_NIM_BASE_URL",
        "TA_VLM_NIM_BASE_URL",
        "MA_VLM_NIM_BASE_URL",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(defaults, "DEFAULT_VLM_BACKEND", "openai")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MODEL", "custom-model")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_MAX_TOKENS", 123)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_REASONING_EFFORT", "high")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_BASE_URL", "https://vlm.example.test/v1")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_API_KEY", None)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_API_KEY_ENV", "MISSING_CUSTOM_VLM_KEY")
    monkeypatch.delenv("MISSING_CUSTOM_VLM_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="^configured API key environment variable is not set or empty$",
    ) as exc_info:
        resolve_default_judge_vlm()
    assert "MISSING_CUSTOM_VLM_KEY" not in str(exc_info.value)


def test_has_reference_media_detects_images_or_videos(tmp_path: Path) -> None:
    assert has_reference_media() is False
    assert has_reference_media(reference_images=[tmp_path / "ref.png"]) is True
    assert has_reference_media(reference_videos=[tmp_path / "ref.mp4"]) is True
