# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared assertions for Physics Agent service image definitions."""


def assert_locked_tuning_and_ovphysx_profiles(text: str) -> None:
    """Assert that a service image uses the reviewed runtime lock profiles."""
    required_markers = (
        "ARG TARGETARCH",
        "${TARGETARCH:-amd64}",
        "amd64) tuning_lock=pylock.physics-tuning-runtime.toml",
        "arm64) tuning_lock=pylock.physics-tuning-runtime.aarch64.toml",
        "amd64) ovphysx_lock=pylock.ovphysx-runtime.toml",
        "arm64) ovphysx_lock=pylock.ovphysx-runtime.aarch64.toml",
        'echo "Unsupported TARGETARCH: ${TARGETARCH}"',
        "--require-hashes --no-deps",
        'touch "${WU_OVPHYSX_VENV_DIR}/.wu-ovphysx-runtime-ready"',
        "is_botorch_available",
        "physics_agent[tuning] install missing BoTorch support",
    )
    forbidden_markers = (
        '-e ".[tuning]"',
        "ovphysx==",
        "imageio imageio-ffmpeg",
    )
    for marker in required_markers:
        assert marker in text, f"Physics service image is missing {marker!r}"
    for marker in forbidden_markers:
        assert marker not in text, f"Physics service image contains {marker!r}"
