# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from apps.physics_agent_service.tests.dockerfile_assertions import (
    assert_locked_tuning_and_ovphysx_profiles,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_lock(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _package_versions(lock: dict[str, Any]) -> dict[str, str]:
    return {package["name"]: package["version"] for package in lock["packages"]}


def _artifacts(lock: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for package in lock["packages"]:
        if "sdist" in package:
            artifacts.append(package["sdist"])
        artifacts.extend(package.get("wheels", []))
        if "archive" in package:
            artifacts.append(package["archive"])
    return artifacts


def test_physics_service_image_uses_locked_tuning_and_ovphysx_profiles() -> None:
    text = (REPO_ROOT / "apps/physics_agent_service/Dockerfile").read_text(
        encoding="utf-8"
    )

    assert_locked_tuning_and_ovphysx_profiles(text)
    for native_library in ("libgl1", "libgomp1", "libopengl0", "libx11-6", "libxt6"):
        assert native_library in text
    assert "from ovphysx import PhysX" in text
    assert "PhysX(device='cpu')" in text


def test_physics_runtime_profiles_pin_optimizer_video_and_daemon_stacks() -> None:
    runtime_dir = REPO_ROOT / "apps/physics_agent/runtime"

    tuning_lock = _load_lock(runtime_dir / "pylock.physics-tuning-runtime.toml")
    tuning_arm64_lock = _load_lock(
        runtime_dir / "pylock.physics-tuning-runtime.aarch64.toml"
    )
    daemon_lock = _load_lock(runtime_dir / "pylock.ovphysx-runtime.toml")
    daemon_arm64_lock = _load_lock(runtime_dir / "pylock.ovphysx-runtime.aarch64.toml")

    tuning_packages = _package_versions(tuning_lock)
    tuning_arm64_packages = _package_versions(tuning_arm64_lock)
    daemon_packages = _package_versions(daemon_lock)
    daemon_arm64_packages = _package_versions(daemon_arm64_lock)

    assert tuning_packages["botorch"] == "0.17.2"
    assert tuning_packages["torch"] == "2.12.1+cpu"
    assert tuning_packages["imageio"] == "2.37.2"
    assert tuning_packages["imageio-ffmpeg"] == "0.6.0"
    assert "cuda-toolkit" not in tuning_packages
    assert daemon_packages == {
        "numpy": "2.4.4",
        "ovphysx": "0.4.13",
        "packaging": "23.2",
    }
    assert tuning_arm64_packages == tuning_packages
    assert daemon_arm64_packages == daemon_packages
    assert all(
        artifact.get("hashes", {}).get("sha256")
        for artifact in [
            *_artifacts(tuning_lock),
            *_artifacts(tuning_arm64_lock),
            *_artifacts(daemon_lock),
            *_artifacts(daemon_arm64_lock),
        ]
    )

    arm64_artifact_urls = [
        artifact["url"]
        for artifact in [*_artifacts(tuning_arm64_lock), *_artifacts(daemon_arm64_lock)]
    ]
    assert not any("x86_64" in url for url in arm64_artifact_urls)
    assert any(
        "torch-2.12.1%2Bcpu" in url and "aarch64" in url for url in arm64_artifact_urls
    )
    assert any(
        "ovphysx-0.4.13" in url and "aarch64" in url for url in arm64_artifact_urls
    )
