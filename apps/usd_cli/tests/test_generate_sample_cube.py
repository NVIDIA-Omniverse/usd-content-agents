# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped Cube sample is exactly what its generator writes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf  # noqa: E402  (after importorskip)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_sample_cube.py"
CUBE = ROOT / "sample_assets" / "simready" / "Cube"
LAYERS = (
    "cube.usda",
    "Payload/Contents.usda",
    "Payload/Materials.usda",
    "Payload/MaterialsLibrary.usda",
    "Payload/Geometry.usdc",
)


def _generator():
    spec = importlib.util.spec_from_file_location("generate_sample_cube", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", LAYERS)
def test_checked_in_cube_matches_generator(tmp_path: Path, relative: str) -> None:
    generator = _generator()
    generator.generate(tmp_path)

    expected = Sdf.Layer.OpenAsAnonymous(str(tmp_path / relative))
    shipped = Sdf.Layer.OpenAsAnonymous(str(CUBE / relative))

    assert shipped.ExportToString() == expected.ExportToString()
    assert shipped.customLayerData["creator"] == generator.CREATOR
