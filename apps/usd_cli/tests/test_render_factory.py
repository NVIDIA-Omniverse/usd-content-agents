# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform qualification tests for renderer auto-selection."""

from __future__ import annotations

import platform

from usd_core.render import factory


def test_ovrtx_available_on_windows_with_nvidia_driver(tmp_path, monkeypatch):
    system_root = tmp_path / "Windows"
    driver = system_root / "System32" / "nvcuda.dll"
    driver.parent.mkdir(parents=True)
    driver.touch()
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setenv("SystemRoot", str(system_root))

    assert factory._ovrtx_available() is True


def test_ovrtx_unavailable_on_windows_without_nvidia_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))

    assert factory._ovrtx_available() is False
