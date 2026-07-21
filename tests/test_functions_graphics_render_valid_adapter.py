# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the provisional render_valid scaffold adapter."""

import json
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw

from world_understanding.functions.graphics.render_valid_adapter import (
    RENDER_EVIDENCE_MISSING,
    RENDER_FRAME_ID_MISMATCH,
    RENDER_VALID_TEMPLATE,
    _tag_issues,
    run_render_valid_adapter,
)
from world_understanding.functions.graphics.render_validation import (
    OVRTX_CAMERA_RESPONSE_MISSING,
    OVRTX_MISSING_FRAME,
    OVRTX_RENDER_ARTIFACT_DETECTED,
    RENDER_BLANK_IMAGE,
    RENDER_IDENTICAL_ANIMATION_FRAMES,
    RENDER_MALFORMED_RESPONSE,
    RENDER_MISSING_OUTPUT,
    RENDER_SUSPECTED_ERROR_MATERIAL,
)


def _save_image(path: Path, image: PILImage.Image) -> Path:
    image.save(path)
    return path


def _normal_image() -> PILImage.Image:
    image = PILImage.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 31, 31], fill=(255, 0, 0))
    draw.rectangle([32, 0, 63, 31], fill=(0, 255, 0))
    draw.rectangle([0, 32, 31, 63], fill=(0, 0, 255))
    draw.rectangle([32, 32, 63, 63], fill=(255, 255, 0))
    return image


def _issue_codes(result: dict[str, object]) -> list[str]:
    issues = result["issues"]
    assert isinstance(issues, list)
    return [str(issue["code"]) for issue in issues]


def test_adapter_passes_valid_image_and_render_response(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "camera_a_0.png", _normal_image())
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "image_files": [str(image_path)],
            }
        ]
    }

    result = run_render_valid_adapter(
        image_paths=[image_path],
        render_response=response,
        expected_cameras=["/CameraA"],
        expected_frames=[0],
        backend="ovrtx",
    )

    json.dumps(result)
    assert result["template"] == RENDER_VALID_TEMPLATE
    assert result["status"] == "pass"
    assert result["issues"] == []
    assert result["metrics"]["image_count"] == 1
    assert result["metrics"]["render_response_image_count"] == 1
    assert result["metadata"]["render_response"]["backend"] == "ovrtx"
    assert result["metadata"]["render_response"]["output_paths"] == [str(image_path)]


def test_adapter_accepts_bare_image_path_string(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "camera.png", _normal_image())

    result = run_render_valid_adapter(image_paths=str(image_path))

    assert result["status"] == "pass"
    assert result["metrics"]["image_count"] == 1
    assert result["evidence"]["image_paths"] == [str(image_path)]


def test_adapter_reports_missing_image_as_failure(tmp_path: Path) -> None:
    missing_image = tmp_path / "missing.png"

    result = run_render_valid_adapter(image_paths=[missing_image])

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]
    assert result["issues"][0]["template"] == RENDER_VALID_TEMPLATE
    assert result["issues"][0]["check"] == "image"
    assert result["issues"][0]["evidence_index"] == 0
    json.dumps(result)


def test_adapter_reports_red_error_material_render_as_failure(
    tmp_path: Path,
) -> None:
    image = PILImage.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 55, 55], fill=(255, 0, 0))
    image_path = _save_image(tmp_path / "red_fallback.png", image)

    result = run_render_valid_adapter(
        image_paths=[image_path],
        backend="remote",
        detect_error_material_artifacts=True,
    )

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_SUSPECTED_ERROR_MATERIAL]
    assert result["issues"][0]["check"] == "image"
    assert result["issues"][0]["evidence_index"] == 0
    assert (
        result["results"]["images"][0]["metrics"]["red_dominant_nonblack_ratio"] == 1.0
    )
    json.dumps(result)


def test_adapter_allows_red_foreground_by_default(
    tmp_path: Path,
) -> None:
    image = PILImage.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 55, 55], fill=(255, 0, 0))
    image_path = _save_image(tmp_path / "red_asset.png", image)

    result = run_render_valid_adapter(image_paths=[image_path], backend="remote")

    assert result["status"] == "pass"
    assert result["issues"] == []
    assert (
        result["results"]["images"][0]["metrics"]["red_dominant_nonblack_ratio"] == 1.0
    )
    json.dumps(result)


def test_adapter_reports_ovrtx_response_missing_camera_and_frame(
    tmp_path: Path,
) -> None:
    image_path = _save_image(tmp_path / "camera_a_0.png", _normal_image())
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "image_files": [str(image_path)],
            }
        ]
    }

    result = run_render_valid_adapter(
        render_response=response,
        expected_cameras=["/CameraA", "/CameraB"],
        expected_frames=[0, 1],
        backend="ovrtx",
    )

    assert result["status"] == "fail"
    assert _issue_codes(result) == [
        RENDER_MISSING_OUTPUT,
        OVRTX_CAMERA_RESPONSE_MISSING,
        RENDER_MISSING_OUTPUT,
        OVRTX_MISSING_FRAME,
    ]
    assert {issue["check"] for issue in result["issues"]} == {"render_response"}


def test_adapter_detects_duplicate_animation_frames(tmp_path: Path) -> None:
    frame_0 = _save_image(tmp_path / "frame_0.png", _normal_image())
    frame_1 = _save_image(tmp_path / "frame_1.png", _normal_image())

    result = run_render_valid_adapter(
        animation_frame_paths=[frame_0, frame_1],
        frame_ids=[0, 1],
    )

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_IDENTICAL_ANIMATION_FRAMES]
    assert result["metrics"]["animation_frame_count"] == 2
    assert result["results"]["animation_frames"][0]["passed"]
    assert result["results"]["duplicate_frames"]["pairs"][0]["first_frame"] == 0
    assert result["results"]["duplicate_frames"]["pairs"][0]["second_frame"] == 1


def test_adapter_passes_distinct_animation_frames(tmp_path: Path) -> None:
    frame_0 = _save_image(tmp_path / "frame_0.png", _normal_image())
    distinct = _normal_image()
    draw = ImageDraw.Draw(distinct)
    draw.rectangle([0, 0, 63, 63], outline=(0, 0, 0), width=8)
    frame_1 = _save_image(tmp_path / "frame_1.png", distinct)

    result = run_render_valid_adapter(
        animation_frame_paths=[frame_0, frame_1],
        frame_ids=[0, 1],
    )

    assert result["status"] == "pass"
    assert result["metrics"]["animation_frame_count"] == 2
    assert result["issues"] == []
    assert result["results"]["duplicate_frames"]["passed"]


def test_adapter_reports_missing_animation_frame(tmp_path: Path) -> None:
    missing_frame = tmp_path / "missing_frame.png"

    result = run_render_valid_adapter(animation_frame_paths=[missing_frame])

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]
    assert result["issues"][0]["check"] == "animation_frame"
    assert result["issues"][0]["evidence_index"] == 0


def test_adapter_reports_frame_id_mismatch(tmp_path: Path) -> None:
    frame_0 = _save_image(tmp_path / "frame_0.png", _normal_image())
    frame_1 = _save_image(tmp_path / "frame_1.png", _normal_image())

    result = run_render_valid_adapter(
        animation_frame_paths=[frame_0, frame_1],
        frame_ids=[0],
    )

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_FRAME_ID_MISMATCH]
    assert result["issues"][0]["check"] == "duplicate_frames"
    assert result["issues"][0]["details"] == {
        "frame_id_count": 1,
        "animation_frame_count": 2,
    }


def test_adapter_skips_empty_animation_frame_list() -> None:
    result = run_render_valid_adapter(animation_frame_paths=[])

    assert result["status"] == "skipped"
    assert _issue_codes(result) == [RENDER_EVIDENCE_MISSING]


def test_adapter_reports_malformed_render_response() -> None:
    result = run_render_valid_adapter(render_response=42)

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_MALFORMED_RESPONSE]
    assert result["issues"][0]["check"] == "render_response"


def test_adapter_validates_images_embedded_in_render_response() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "images": [PILImage.new("RGB", (32, 32), (0, 0, 0))],
            }
        ]
    }

    result = run_render_valid_adapter(
        render_response=response,
        expected_cameras=["/CameraA"],
        expected_frames=[0],
        backend="ovrtx",
    )

    assert result["status"] == "fail"
    assert _issue_codes(result) == [RENDER_BLANK_IMAGE, OVRTX_RENDER_ARTIFACT_DETECTED]
    assert {issue["check"] for issue in result["issues"]} == {"render_response_image"}
    assert result["metrics"]["render_response_image_count"] == 1


def test_adapter_skips_unresolved_relative_response_image_path() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "image_files": ["camera_a_0.png"],
            }
        ]
    }

    result = run_render_valid_adapter(
        render_response=response,
        expected_cameras=["/CameraA"],
        expected_frames=[0],
    )

    assert result["status"] == "pass"
    assert result["metrics"]["render_response_image_count"] == 0
    assert result["evidence"]["render_response_images"] == []


def test_adapter_resolves_relative_response_image_path_with_output_dir(
    tmp_path: Path,
) -> None:
    _save_image(tmp_path / "camera_a_0.png", _normal_image())
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "image_files": ["camera_a_0.png"],
            }
        ]
    }

    result = run_render_valid_adapter(
        render_response=response,
        expected_cameras=["/CameraA"],
        expected_frames=[0],
        render_output_dir=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["metrics"]["render_response_image_count"] == 1
    assert result["evidence"]["render_response_images"] == [
        str(tmp_path / "camera_a_0.png")
    ]


def test_adapter_handles_single_entry_response_without_results_or_image_keys() -> None:
    result = run_render_valid_adapter(
        render_response={
            "camera": "/CameraA",
            "frames": [0],
        },
        expected_cameras=["/CameraA"],
    )

    assert result["status"] == "fail"
    assert result["metrics"]["render_response_image_count"] == 0
    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]


def test_adapter_extracts_scalar_response_image_entry(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "camera_a_0.png", _normal_image())

    result = run_render_valid_adapter(
        render_response={
            "camera": "/CameraA",
            "frames": [0],
            "outputs": str(image_path),
        },
        expected_cameras=["/CameraA"],
    )

    assert result["status"] == "fail"
    assert result["metrics"]["render_response_image_count"] == 1
    assert result["evidence"]["render_response_images"] == [str(image_path)]
    assert _issue_codes(result) == [RENDER_MALFORMED_RESPONSE]


def test_adapter_collects_response_images_from_later_keys(tmp_path: Path) -> None:
    _save_image(tmp_path / "camera_a_0.png", _normal_image())
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "images": [],
                "output_paths": ["camera_a_0.png"],
            }
        ]
    }

    result = run_render_valid_adapter(
        render_response=response,
        expected_cameras=["/CameraA"],
        expected_frames=[0],
        render_output_dir=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["metrics"]["render_response_image_count"] == 1
    assert result["evidence"]["render_response_images"] == [
        str(tmp_path / "camera_a_0.png")
    ]


def test_adapter_ignores_non_mapping_response_entries_and_unsupported_images() -> None:
    result = run_render_valid_adapter(
        render_response=[
            123,
            {
                "camera": "/CameraA",
                "frames": [0],
                "images": ["", 42],
            },
        ],
    )

    assert result["status"] == "fail"
    assert result["metrics"]["render_response_image_count"] == 0
    assert _issue_codes(result) == [
        RENDER_MALFORMED_RESPONSE,
        RENDER_MISSING_OUTPUT,
    ]


def test_adapter_ignores_non_sequence_results_when_extracting_images() -> None:
    result = run_render_valid_adapter(render_response={"results": "not-a-sequence"})

    assert result["status"] == "fail"
    assert result["metrics"]["render_response_image_count"] == 0
    assert _issue_codes(result) == [RENDER_MALFORMED_RESPONSE]


def test_adapter_does_not_double_count_response_image_aliases(
    tmp_path: Path,
) -> None:
    _save_image(tmp_path / "camera_a_0.png", _normal_image())
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0, 1],
                "images": [_normal_image()],
                "output_paths": ["camera_a_0.png"],
            }
        ]
    }

    result = run_render_valid_adapter(
        render_response=response,
        expected_cameras=["/CameraA"],
        expected_frames=[0, 1],
        render_output_dir=tmp_path,
    )

    assert result["status"] == "fail"
    assert result["metrics"]["render_response_image_count"] == 1
    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]


def test_tag_issues_ignores_non_issue_shapes() -> None:
    assert _tag_issues("not-a-list", check="image") == []
    assert _tag_issues(
        [object(), {"code": "render.example"}],
        check="image",
        evidence_index=2,
    ) == [
        {
            "code": "render.example",
            "template": RENDER_VALID_TEMPLATE,
            "check": "image",
            "evidence_index": 2,
        }
    ]


def test_adapter_skips_when_no_evidence_is_supplied() -> None:
    result = run_render_valid_adapter()

    json.dumps(result)
    assert result["status"] == "skipped"
    assert _issue_codes(result) == [RENDER_EVIDENCE_MISSING]
    assert result["issues"][0]["severity"] == "warning"
    assert result["issues"][0]["template"] == RENDER_VALID_TEMPLATE
    assert result["issues"][0]["check"] == "evidence"
    assert result["metrics"]["issue_count"] == 1
    assert result["results"]["images"] == []
    assert result["results"]["animation_frames"] == []
    assert result["results"]["render_response_images"] == []
