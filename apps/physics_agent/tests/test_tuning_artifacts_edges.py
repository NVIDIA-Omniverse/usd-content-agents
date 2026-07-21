# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted edge coverage for tuning artifact helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import physics_agent.tuning.artifacts as artifacts_mod
from physics_agent.tuning.artifacts import (
    _append_visual_evidence_md,
    _atomic_write_text,
    _inline_code,
)


def test_atomic_write_failure_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_replace(_src: Path, _dst: Path) -> None:
        raise RuntimeError("replace failed")

    monkeypatch.setattr(artifacts_mod.os, "replace", raise_replace)
    with pytest.raises(RuntimeError, match="replace failed"):
        _atomic_write_text(tmp_path / "out.json", "{}")
    assert list(tmp_path.glob(".*.tmp")) == []


def test_visual_evidence_markdown_edges() -> None:
    lines: list[str] = []
    _append_visual_evidence_md(lines, {})
    assert lines == []

    _append_visual_evidence_md(
        lines,
        {
            "comparison_image": "comparison.png",
            "reference_images": [
                {"path": "ref.png", "caption": "reference caption"},
                {"path": "ref-no-caption.png"},
                "skip",
            ],
            "generated_images": [
                {"path": "gen.png", "caption": "generated caption"},
                {"path": "gen-no-caption.png"},
            ],
            "reference_error": "ref failed",
            "generated_error": "gen failed",
            "comparison_error": "comparison failed",
        },
    )
    text = "\n".join(lines)
    assert "Comparison image" in text
    assert "Reference media" in text
    assert "Generated frames" in text
    assert "`ref-no-caption.png`" in text
    assert "`gen-no-caption.png`" in text
    assert "ref failed" in text
    assert "gen failed" in text
    assert "comparison failed" in text

    assert _inline_code("a `quoted` value") == "`` a `quoted` value ``"
