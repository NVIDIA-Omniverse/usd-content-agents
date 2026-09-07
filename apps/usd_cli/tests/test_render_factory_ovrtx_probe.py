# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-OVRTX availability probes for supported host runtimes."""

from __future__ import annotations

import glob as glob_module
import platform as platform_module

import pytest
from usd_core.render import factory as factory_module


def _fake_filesystem(monkeypatch, *, existing: set[str], globs: dict[str, list[str]]):
    """Resolve only ``existing`` paths and ``globs`` patterns.

    ``factory`` imports ``glob``, ``os``, and ``platform`` inside the probe, so
    patch those modules directly; ``Path`` is a module-level import and is patched
    on the factory module itself.
    """

    class _FakePath:
        def __init__(self, value: str) -> None:
            self._value = str(value).replace("\\", "/").rstrip("/")

        def __truediv__(self, other: str) -> "_FakePath":
            return _FakePath(f"{self._value}/{other}")

        def exists(self) -> bool:
            return self._value in existing

        def is_file(self) -> bool:
            return self._value in existing

    monkeypatch.setattr(factory_module, "Path", _FakePath)
    monkeypatch.setattr(glob_module, "glob", lambda pattern: globs.get(pattern, []))


@pytest.fixture(autouse=True)
def _linux_host(monkeypatch):
    monkeypatch.setattr(platform_module, "system", lambda: "Linux")


def test_native_linux_nvidia_driver_is_available(monkeypatch) -> None:
    _fake_filesystem(monkeypatch, existing={"/proc/driver/nvidia/version"}, globs={})
    assert factory_module._ovrtx_available() is True


def test_native_linux_nvidia_device_node_is_available(monkeypatch) -> None:
    _fake_filesystem(
        monkeypatch,
        existing=set(),
        globs={"/dev/nvidia[0-9]*": ["/dev/nvidia0"]},
    )
    assert factory_module._ovrtx_available() is True


def test_native_linux_without_gpu_is_unavailable(monkeypatch) -> None:
    _fake_filesystem(monkeypatch, existing=set(), globs={})
    assert factory_module._ovrtx_available() is False


def test_wsl2_dxg_and_icd_files_do_not_claim_local_ovrtx(monkeypatch) -> None:
    """An ICD manifest does not give WSL2 a usable Vulkan device."""
    _fake_filesystem(
        monkeypatch,
        existing={"/dev/dxg"},
        globs={
            "/usr/lib/wsl/lib/libnvidia-ml.so*": [
                "/usr/lib/wsl/lib/libnvidia-ml.so.1"
            ],
            "/etc/vulkan/icd.d/nvidia_icd*.json": [
                "/etc/vulkan/icd.d/nvidia_icd.json"
            ],
        },
    )
    assert factory_module._ovrtx_available() is False


def test_unsupported_host_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(platform_module, "system", lambda: "Darwin")
    assert factory_module._ovrtx_available() is False
