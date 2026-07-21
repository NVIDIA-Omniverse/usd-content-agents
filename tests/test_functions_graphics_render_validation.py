# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for render evidence validation utilities."""

from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw

from world_understanding.functions.graphics.render_validation import (
    OVRTX_CAMERA_RESPONSE_MISSING,
    OVRTX_MISSING_FRAME,
    OVRTX_RENDER_ARTIFACT_DETECTED,
    OVRTX_TIME_SAMPLE_VISIBILITY_FAILED,
    RENDER_BLANK_IMAGE,
    RENDER_IDENTICAL_ANIMATION_FRAMES,
    RENDER_IMAGE_DECODE_FAILED,
    RENDER_LOW_CONTRAST,
    RENDER_MALFORMED_RESPONSE,
    RENDER_MISSING_OUTPUT,
    RENDER_SUSPECTED_ERROR_MATERIAL,
    DuplicateFrameValidationResult,
    ImageValidationResult,
    RenderResponseValidationResult,
    detect_duplicate_frames,
    extract_render_response_metadata,
    validate_image_artifact,
    validate_render_response,
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


def _issue_codes(
    result: ImageValidationResult
    | DuplicateFrameValidationResult
    | RenderResponseValidationResult,
) -> list[str]:
    return [issue.code for issue in result.issues]


def test_valid_normal_image_passes(tmp_path: Path) -> None:
    image_path = _save_image(tmp_path / "normal.png", _normal_image())

    result = validate_image_artifact(image_path)

    assert result.passed
    assert result.readable
    assert result.width == 64
    assert result.height == 64
    assert result.metrics["unique_colors"] == 4
    assert result.metrics["red_dominant_nonblack_ratio"] == 0.25


def test_missing_image_path_fails_as_missing_output(tmp_path: Path) -> None:
    result = validate_image_artifact(tmp_path / "missing.png")

    assert not result.passed
    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]


def test_corrupt_image_fails_decode_check(tmp_path: Path) -> None:
    image_path = tmp_path / "corrupt.png"
    image_path.write_text("not an image", encoding="utf-8")

    result = validate_image_artifact(image_path)

    assert not result.passed
    assert _issue_codes(result) == [RENDER_IMAGE_DECODE_FAILED]


def test_blank_gray_image_fails_with_near_blank_kind(tmp_path: Path) -> None:
    image_path = _save_image(
        tmp_path / "blank.png",
        PILImage.new("RGB", (32, 32), (128, 128, 128)),
    )

    result = validate_image_artifact(image_path)

    assert not result.passed
    assert _issue_codes(result) == [RENDER_BLANK_IMAGE]
    assert result.issues[0].details["kind"] == "near_blank"


def test_all_black_and_all_white_images_are_classified(tmp_path: Path) -> None:
    black_path = _save_image(
        tmp_path / "black.png",
        PILImage.new("RGB", (32, 32), (0, 0, 0)),
    )
    white_path = _save_image(
        tmp_path / "white.png",
        PILImage.new("RGB", (32, 32), (255, 255, 255)),
    )

    black = validate_image_artifact(black_path)
    white = validate_image_artifact(white_path)

    assert _issue_codes(black) == [RENDER_BLANK_IMAGE]
    assert black.issues[0].details["kind"] == "all_black"
    assert _issue_codes(white) == [RENDER_BLANK_IMAGE]
    assert white.issues[0].details["kind"] == "all_white"


def test_low_contrast_image_fails_without_blank_classification(tmp_path: Path) -> None:
    image = PILImage.new("RGB", (64, 64))
    pixels = image.load()
    for x in range(64):
        value = 120 + (x % 11)
        for y in range(64):
            pixels[x, y] = (value, value, value)
    image_path = _save_image(tmp_path / "low_contrast.png", image)

    result = validate_image_artifact(image_path)

    assert not result.passed
    assert _issue_codes(result) == [RENDER_LOW_CONTRAST]


def test_saturated_red_foreground_fails_as_suspected_error_material(
    tmp_path: Path,
) -> None:
    image = PILImage.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 55, 55], fill=(255, 0, 0))
    image_path = _save_image(tmp_path / "red_fallback.png", image)

    result = validate_image_artifact(
        image_path,
        backend="ovrtx",
        detect_error_material_artifacts=True,
    )

    assert not result.passed
    assert _issue_codes(result) == [
        RENDER_SUSPECTED_ERROR_MATERIAL,
        OVRTX_RENDER_ARTIFACT_DETECTED,
    ]
    assert result.issues[0].details["red_dominant_nonblack_ratio"] == 1.0


def test_non_saturated_red_foreground_does_not_trip_error_material_heuristic(
    tmp_path: Path,
) -> None:
    image = PILImage.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 55, 55], fill=(180, 0, 0))
    image_path = _save_image(tmp_path / "red_asset.png", image)

    result = validate_image_artifact(
        image_path,
        backend="ovrtx",
        detect_error_material_artifacts=True,
    )

    assert result.passed
    assert result.metrics["red_dominant_nonblack_ratio"] == 0.0


def test_full_red_frame_opt_in_fails_as_suspected_error_material(
    tmp_path: Path,
) -> None:
    image_path = _save_image(
        tmp_path / "solid_red_fallback.png",
        PILImage.new("RGB", (64, 64), (255, 0, 0)),
    )

    result = validate_image_artifact(
        image_path,
        backend="ovrtx",
        detect_error_material_artifacts=True,
    )

    assert not result.passed
    assert _issue_codes(result) == [
        RENDER_SUSPECTED_ERROR_MATERIAL,
        OVRTX_RENDER_ARTIFACT_DETECTED,
    ]
    assert result.issues[0].details["red_dominant_nonblack_ratio"] == 1.0


def test_red_dominant_foreground_passes_without_error_material_opt_in(
    tmp_path: Path,
) -> None:
    image = PILImage.new("RGB", (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([8, 8, 55, 55], fill=(255, 0, 0))
    image_path = _save_image(tmp_path / "red_asset.png", image)

    result = validate_image_artifact(image_path)

    assert result.passed
    assert result.metrics["red_dominant_nonblack_ratio"] == 1.0


def test_ovrtx_backend_adds_artifact_issue_for_bad_image(tmp_path: Path) -> None:
    image_path = _save_image(
        tmp_path / "blank.png",
        PILImage.new("RGB", (32, 32), (128, 128, 128)),
    )

    result = validate_image_artifact(image_path, backend="ovrtx")

    assert _issue_codes(result) == [RENDER_BLANK_IMAGE, OVRTX_RENDER_ARTIFACT_DETECTED]


def test_duplicate_frames_detect_identical_and_near_identical(
    tmp_path: Path,
) -> None:
    frame_0 = _save_image(tmp_path / "frame_0.png", _normal_image())
    near_duplicate = _normal_image()
    draw = ImageDraw.Draw(near_duplicate)
    draw.rectangle([0, 0, 4, 4], fill=(254, 0, 0))
    frame_1 = _save_image(tmp_path / "frame_1.png", near_duplicate)
    frame_2 = _save_image(
        tmp_path / "frame_2.png",
        PILImage.new("RGB", (64, 64), (0, 0, 0)),
    )

    result = detect_duplicate_frames(
        [frame_0, frame_1, frame_2],
        frame_ids=[0, 1, 2],
    )

    assert not result.passed
    assert _issue_codes(result) == [RENDER_IDENTICAL_ANIMATION_FRAMES]
    assert len(result.pairs) == 1
    assert result.pairs[0].first_frame == 0
    assert result.pairs[0].second_frame == 1


def test_duplicate_frames_report_decode_failure(tmp_path: Path) -> None:
    valid_frame = _save_image(tmp_path / "valid.png", _normal_image())
    corrupt_frame = tmp_path / "corrupt.png"
    corrupt_frame.write_text("not an image", encoding="utf-8")

    result = detect_duplicate_frames([valid_frame, corrupt_frame], backend="ovrtx")

    assert not result.passed
    assert _issue_codes(result) == [
        RENDER_IMAGE_DECODE_FAILED,
        OVRTX_RENDER_ARTIFACT_DETECTED,
    ]


def test_valid_mock_render_response_passes() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0, 1],
                "image_files": ["camera_a_0.png", "camera_a_1.png"],
            },
            {
                "camera": "/CameraB",
                "frames": [0, 1],
                "image_files": ["camera_b_0.png", "camera_b_1.png"],
            },
        ]
    }

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA", "/CameraB"],
        expected_frames=[0, 1],
    )

    assert result.passed
    assert result.metadata.cameras == ["/CameraA", "/CameraB"]
    assert result.metadata.frames == [0, 1]
    assert result.metadata.output_paths == [
        "camera_a_0.png",
        "camera_a_1.png",
        "camera_b_0.png",
        "camera_b_1.png",
    ]


def test_render_response_uses_later_image_keys_when_earlier_key_empty() -> None:
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

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA"],
        expected_frames=[0],
    )

    assert result.passed
    assert result.metadata.output_paths == ["camera_a_0.png"]


def test_render_response_does_not_double_count_non_empty_image_aliases() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0, 1],
                "images": [object()],
                "output_paths": ["camera_a_0.png"],
            }
        ]
    }

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA"],
        expected_frames=[0, 1],
    )

    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]
    assert result.issues[0].details["missing_frames"] == [1]
    assert result.issues[0].details["image_count"] == 1


def test_render_response_missing_frame_fails_with_ovrtx_code() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0, 1],
                "image_files": ["camera_a_0.png", "camera_a_1.png"],
            }
        ]
    }

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA"],
        expected_frames=[0, 1, 2],
        backend="ovrtx",
    )

    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT, OVRTX_MISSING_FRAME]
    assert result.issues[0].details["missing_frames"] == [2]


def test_render_response_frame_metadata_does_not_hide_missing_images() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0, 1, 2],
                "image_files": ["camera_a_0.png"],
            }
        ]
    }

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA"],
        expected_frames=[0, 1, 2],
        backend="ovrtx",
    )

    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT, OVRTX_MISSING_FRAME]
    assert result.issues[0].details["missing_frames"] == [1, 2]
    assert result.issues[0].details["image_count"] == 1


def test_render_response_sparse_frame_metadata_reports_actual_missing_frame() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0, 2],
                "image_files": ["camera_a_0.png", "camera_a_2.png"],
            }
        ]
    }

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA"],
        expected_frames=[0, 1, 2],
    )

    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]
    assert result.issues[0].details["missing_frames"] == [1]
    assert result.issues[0].details["rendered_frames"] == [0, 2]


def test_render_response_missing_camera_fails_with_ovrtx_code() -> None:
    response = {
        "results": [
            {
                "camera": "/CameraA",
                "frames": [0],
                "image_files": ["camera_a_0.png"],
            }
        ]
    }

    result = validate_render_response(
        response,
        expected_cameras=["/CameraA", "/CameraB"],
        expected_frames=[0],
        backend="ovrtx",
    )

    assert _issue_codes(result) == [
        RENDER_MISSING_OUTPUT,
        OVRTX_CAMERA_RESPONSE_MISSING,
    ]
    assert result.issues[0].details["camera"] == "/CameraB"


def test_render_response_missing_image_entry_fails() -> None:
    response = {"results": [{"camera": "/Camera", "frames": [0], "image_files": [""]}]}

    result = validate_render_response(
        response,
        expected_cameras=["/Camera"],
        expected_frames=[0],
    )

    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT]
    assert result.issues[0].details["image_index"] == 0


def test_malformed_render_response_fails() -> None:
    result = validate_render_response({"results": {"camera": "/Camera"}})

    assert not result.passed
    assert _issue_codes(result) == [RENDER_MALFORMED_RESPONSE]
    assert result.metadata.malformed


def test_single_camera_entry_shape_is_supported() -> None:
    response = {"camera": "/Camera", "frames": [0], "images": [object()]}

    result = validate_render_response(
        response,
        expected_cameras=["/Camera"],
        expected_frames=[0],
    )

    assert result.passed
    assert result.metadata.cameras == ["/Camera"]


def test_metadata_records_sensors_failures_and_ovrtx_visibility_issue() -> None:
    response = {
        "results": [
            {
                "camera": "/Camera",
                "frames": [0],
                "image_files": ["camera_0.png"],
                "sensor_files": {"depth": {"0": "depth.npy"}},
                "error": "time-sampled visibility failed while rendering frame 0",
            }
        ]
    }

    metadata = extract_render_response_metadata(response, backend="ovrtx")
    result = validate_render_response(
        response,
        expected_cameras=["/Camera"],
        expected_frames=[0],
        backend="ovrtx",
    )

    assert metadata.backend == "ovrtx"
    assert metadata.sensors == ["depth"]
    assert metadata.failures == [
        "time-sampled visibility failed while rendering frame 0"
    ]
    assert _issue_codes(result) == [OVRTX_TIME_SAMPLE_VISIBILITY_FAILED]


def test_image_below_minimum_dimensions_reports_missing_output(tmp_path: Path) -> None:
    image_path = _save_image(
        tmp_path / "too_small.png",
        PILImage.new("RGB", (1, 2), (255, 0, 0)),
    )

    result = validate_image_artifact(image_path, min_width=2, min_height=3)

    assert _issue_codes(result) == [RENDER_MISSING_OUTPUT, RENDER_BLANK_IMAGE]
    assert result.issues[0].details == {
        "width": 1,
        "height": 2,
        "min_width": 2,
        "min_height": 3,
    }


def test_image_analysis_sampling_stride_is_recorded(tmp_path: Path) -> None:
    image_path = _save_image(
        tmp_path / "sampled.png",
        PILImage.new("RGB", (10, 10), (120, 120, 120)),
    )

    result = validate_image_artifact(image_path, max_analysis_pixels=10)

    assert result.metrics["sampled_pixels"] == 10


def test_duplicate_frames_single_frame_passes() -> None:
    result = detect_duplicate_frames([_normal_image()])

    assert result.passed
    assert result.pairs == []
    assert result.issues == []


def test_render_response_reports_non_mapping_entry_and_empty_results() -> None:
    malformed_entry = validate_render_response({"results": [object()]})
    empty_results = validate_render_response({"results": []})

    assert _issue_codes(malformed_entry) == [
        RENDER_MALFORMED_RESPONSE,
        RENDER_MALFORMED_RESPONSE,
    ]
    assert malformed_entry.issues[0].subject == "results[0]"
    assert _issue_codes(empty_results) == [RENDER_MALFORMED_RESPONSE]
    assert "does not contain any camera" in empty_results.issues[0].message


def test_render_response_top_level_sequence_is_supported() -> None:
    result = validate_render_response(
        [{"camera": "/Camera", "images": [object()]}],
        expected_cameras=["/Camera"],
    )

    assert result.passed
    assert result.metadata.cameras == ["/Camera"]


def test_render_response_reports_missing_and_duplicate_camera_entries() -> None:
    response = {
        "results": [
            {"images": [object()]},
            {"camera": "/Camera", "images": [object()]},
            {"camera": "/Camera", "images": [object()]},
        ]
    }

    result = validate_render_response(response)

    assert _issue_codes(result) == [
        RENDER_MALFORMED_RESPONSE,
        RENDER_MALFORMED_RESPONSE,
    ]
    assert result.issues[0].subject == "results[0]"
    assert result.issues[1].subject == "/Camera"


def test_render_response_image_entry_edge_shapes() -> None:
    no_image_key = validate_render_response({"results": [{"camera": "/Camera"}]})
    scalar_images = validate_render_response(
        {"results": [{"camera": "/Camera", "images": "single.png"}]}
    )
    empty_images = validate_render_response(
        {"results": [{"camera": "/Camera", "images": []}]}
    )
    none_image = validate_render_response(
        {"results": [{"camera": "/Camera", "images": [None]}]}
    )
    none_then_output = validate_render_response(
        {
            "results": [
                {
                    "camera": "/Camera",
                    "images": None,
                    "output_paths": ["camera.png"],
                }
            ]
        }
    )

    assert _issue_codes(no_image_key) == [RENDER_MISSING_OUTPUT]
    assert _issue_codes(scalar_images) == [RENDER_MALFORMED_RESPONSE]
    assert _issue_codes(empty_images) == [RENDER_MISSING_OUTPUT]
    assert _issue_codes(none_image) == [RENDER_MISSING_OUTPUT]
    assert none_image.issues[0].details["image_index"] == 0
    assert none_then_output.passed
    assert none_then_output.metadata.output_paths == ["camera.png"]


def test_render_response_infers_missing_frames_from_metadata_counts() -> None:
    without_expected_frames = validate_render_response(
        {
            "results": [
                {
                    "camera": "/Camera",
                    "frames": [0, 1],
                    "images": ["frame_0.png"],
                }
            ]
        }
    )
    without_rendered_frames = validate_render_response(
        {"results": [{"camera": "/Camera", "images": ["frame_0.png"]}]},
        expected_frames=[0, 1],
    )

    assert _issue_codes(without_expected_frames) == [RENDER_MISSING_OUTPUT]
    assert without_expected_frames.issues[0].details["missing_frames"] == [1]
    assert _issue_codes(without_rendered_frames) == [RENDER_MISSING_OUTPUT]
    assert without_rendered_frames.issues[0].details["missing_frames"] == [1]


def test_render_response_metadata_coerces_scalar_and_mapping_fields() -> None:
    metadata = extract_render_response_metadata(
        {
            "cameras": "TopCamera",
            "frames": 3,
            "sensors": ["depth", "normal"],
            "failures": "top failure",
            "errors": {"worker": "boom"},
            "results": [
                {
                    "camera_path": "/CameraA",
                    "frame_ids": [{"frame": 1}],
                    "sensors": 7,
                    "images": "inline.png",
                    "failures": ["list failure", 4],
                },
                {
                    "camera_name": "/CameraB",
                    "frames": object(),
                    "image_files": ["out.png"],
                    "failures": 5,
                },
            ],
        }
    )

    assert metadata.cameras == ["TopCamera", "/CameraA", "/CameraB"]
    assert metadata.frames[0] == 3
    assert metadata.frames[1] == "{'frame': 1}"
    assert len(metadata.frames) == 3
    assert metadata.sensors == ["depth", "normal", "7"]
    assert metadata.output_paths == ["out.png"]
    assert metadata.failures == [
        "top failure",
        "worker: boom",
        "list failure",
        "4",
        "5",
    ]
