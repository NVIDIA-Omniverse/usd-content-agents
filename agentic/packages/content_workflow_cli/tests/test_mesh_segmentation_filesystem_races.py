# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-race regressions for mesh-segmentation evidence discovery."""

from __future__ import annotations

from pathlib import Path

from content_workflow_cli.mesh_segmentation_runner import (
    _first_nonzero_hypothesis_path,
)


def _raise_for_path(
    monkeypatch,
    *,
    unavailable_path: Path,
) -> None:
    original_stat = Path.stat

    def racy_stat(path: Path, *args, **kwargs):
        if path == unavailable_path:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racy_stat)


def test_first_nonzero_hypothesis_skips_entry_deleted_after_glob(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    unavailable = run_dir / "hypotheses" / "rev-000" / "face_labels.u32le"
    available = run_dir / "hypotheses" / "rev-001" / "face_labels.u32le"
    unavailable.parent.mkdir(parents=True)
    available.parent.mkdir(parents=True)
    unavailable.write_bytes(b"\x01\x00\x00\x00")
    available.write_bytes(b"\x01\x00\x00\x00")
    _raise_for_path(monkeypatch, unavailable_path=unavailable)

    assert _first_nonzero_hypothesis_path(run_dir) == available
