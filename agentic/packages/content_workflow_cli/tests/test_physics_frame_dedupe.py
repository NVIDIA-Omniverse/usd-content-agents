# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-identical validation frames must be collapsed before the review turn.

A drop that actually comes to rest renders byte-identical PNGs for every frame
after motion stops. The codex API rejects a request carrying more than roughly
20-25 byte-identical images with an opaque ``400 Bad Request``, so the assets
whose physics settles best are exactly the ones whose visual review cannot run.

Measured on RoboCasa drops: banana 36 duplicate frames and pear 43 out of 91
both failed deterministically; apple, apricot, orange and peach had zero
duplicates and all passed. Forcing 30 duplicates into orange's frames made it
fail too, and adding noise to banana's frames made them pass -- while inflating
the payload from 13.8 MB to 58.3 MB, which also rules out request size.
"""

from pathlib import Path

from content_workflow_cli.runner import _dedupe_frame_image_inputs


def _write_frames(tmp_path: Path, contents: list[bytes]) -> list[str]:
    paths = []
    for index, blob in enumerate(contents):
        path = tmp_path / f"frame_{index:04d}.png"
        path.write_bytes(blob)
        paths.append(str(path))
    return paths


def test_identical_trailing_frames_collapse_to_one(tmp_path: Path) -> None:
    """The settled tail is one image, not thirty rejected ones."""

    frames = _write_frames(
        tmp_path, [f"moving{i}".encode() for i in range(10)] + [b"at-rest"] * 30
    )

    inputs = _dedupe_frame_image_inputs(frames)

    assert len(inputs) == 11
    assert inputs[-1]["label"] == (
        "Physics validation frames 11-40 (consecutive frames render identically)"
    )
    # The label must not assert the body is at rest: on the assets that trigger
    # this the repeats are a period-4 cycle, so "static" would be false.
    assert "static" not in inputs[-1]["label"]


def test_unique_frames_are_untouched(tmp_path: Path) -> None:
    """Assets that keep moving must keep every frame and their plain labels."""

    frames = _write_frames(tmp_path, [f"frame{i}".encode() for i in range(40)])

    inputs = _dedupe_frame_image_inputs(frames)

    assert len(inputs) == 40
    assert inputs[0]["label"] == "Physics validation frame 1"
    assert inputs[-1]["label"] == "Physics validation frame 40"


def test_duplicate_count_stays_under_the_api_limit(tmp_path: Path) -> None:
    """Regression guard on the real failure: banana sent 36 duplicates of 91."""

    frames = _write_frames(
        tmp_path, [f"m{i}".encode() for i in range(55)] + [b"rest"] * 36
    )

    inputs = _dedupe_frame_image_inputs(frames)

    paths = [item["path"] for item in inputs]
    assert len(paths) == len(set(paths)), "no duplicate image may reach the request"
    assert len(inputs) == 56


def test_non_adjacent_duplicates_do_not_claim_a_static_span(tmp_path: Path) -> None:
    """A body that rocks back through the same pose must not be labelled as a
    contiguous static span -- frames 2 and 4 differ, so "frames 1-5 identical"
    would be a false statement to the reviewing agent."""

    frames = _write_frames(tmp_path, [b"a", b"b", b"a", b"c", b"a"])

    inputs = _dedupe_frame_image_inputs(frames)

    assert [item["path"] for item in inputs] == [frames[0], frames[1], frames[3]]
    assert inputs[0]["label"] == (
        "Physics validation frame 1 (identical render also at frames 3, 5)"
    )
    assert "1-5" not in inputs[0]["label"]


def test_unreadable_frames_are_passed_through(tmp_path: Path) -> None:
    """A missing frame must not crash the review turn or silently vanish."""

    frames = _write_frames(tmp_path, [b"a", b"b"])
    frames.append(str(tmp_path / "does_not_exist.png"))

    inputs = _dedupe_frame_image_inputs(frames)

    assert len(inputs) == 3
    assert inputs[-1]["label"] == "Physics validation frame 3"
