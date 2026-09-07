# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for geometry-repair artifact preservation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from geometry_repair import artifacts
from geometry_repair.artifacts import classify_unresolved_dependencies, preserve_source


def test_dependency_discovery_failure_with_material_suffix_blocks_geometry() -> None:
    failure = "dependency discovery failed: RuntimeError: cannot inspect missing_texture.png"

    geometry, material = classify_unresolved_dependencies(["/asset/missing_texture.png", failure])

    assert geometry == [failure]
    assert material == ["/asset/missing_texture.png"]


@pytest.mark.parametrize(
    "mutated_content",
    [b"BBBB", b"dependency-grew"],
    ids=["digest-changed", "size-changed"],
)
def test_dependency_copy_fails_closed_when_source_mutates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_content: bytes,
) -> None:
    source = tmp_path / "source.obj"
    source.write_bytes(b"source")
    dependency = tmp_path / "dependency.bin"
    dependency.write_bytes(b"AAAA")
    real_copy2 = shutil.copy2

    def mutate_then_copy(
        source_path: str | Path,
        target_path: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> str | Path:
        if Path(source_path) == dependency:
            dependency.write_bytes(mutated_content)
        return real_copy2(source_path, target_path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(artifacts.shutil, "copy2", mutate_then_copy)

    with pytest.raises(RuntimeError, match="Dependency source or preserved copy changed"):
        preserve_source(
            source,
            tmp_path / "repair",
            dependency_paths=[dependency],
            create_resolved_snapshot=False,
        )

    assert not (tmp_path / "repair" / "source" / "source_manifest.json").exists()
