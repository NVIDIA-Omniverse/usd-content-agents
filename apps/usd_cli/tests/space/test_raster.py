# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rasterised multi-layer solid heightfield (esd.raster) -> free spans.

Verifies the properties the single-ray sampler could not guarantee:
  * a raised deck yields TWO layered free spans (on the floor under it, and on
    the deck top) — multi-layer, not a single column;
  * a tall lateral beam marks its column solid over its whole height, so NO free
    span passes through it (the fix for the absurd through-structure placements);
  * a thin slat that would fall between ray samples is captured (conservative
    2D overlap), and CPU == CUDA bit-for-bit.
"""


import numpy as np
import pytest


wp = pytest.importorskip("warp")
from usd_core.space.heightfield import build_spans_raster  # noqa: E402
from usd_core.space.device_mesh import DeviceMesh  # noqa: E402
from space_synthetic import scene_from_boxes  # noqa: E402


def _scene():
    # floor slab, a raised deck over x in [0,2], and a tall beam at x ~= 3.
    return scene_from_boxes(
        [
            ([-1, -1, -0.1], [5, 3, 0.0]),   # floor            (obj 0)
            ([0.0, 0.0, 1.0], [2.0, 3, 1.1]),  # raised deck     (obj 1)
            ([3.0, 0.0, 0.0], [3.1, 3, 2.0]),  # tall beam/upright (obj 2)
        ],
        names=["floor", "deck", "beam"],
    )


def _spans(sp, x, y):
    ox, oy = sp["origin"]
    cs = sp["cell_size"]
    ix = int((x - ox) / cs)
    iy = int((y - oy) / cs)
    c = int(sp["span_count"][ix, iy])
    return [(round(float(sp["span_min"][ix, iy, k]), 3),
             round(float(sp["span_max"][ix, iy, k]), 3)) for k in range(c)]


def _build(device="cpu", cht=0.2):
    rc = DeviceMesh(_scene(), device=device)
    return build_spans_raster(rc, (-1, -1, -0.5), (5, 3, 3.0), 0.2, cht)


def test_raised_deck_is_two_layers():
    sp = _build()
    # Column over the deck: free on the floor (up to the deck underside) AND free
    # on the deck top (up to the ceiling).
    assert _spans(sp, 1.0, 1.5) == [(0.0, 1.0), (1.1, 3.0)]


def test_open_floor_single_span():
    sp = _build()
    assert _spans(sp, 4.0, 1.5) == [(0.0, 3.0)]


def test_no_free_span_through_beam():
    sp = _build()
    # The tall beam occupies [0,2]; only the space ABOVE it is free — nothing
    # passes through the beam (the lateral-blindness fix).
    spans = _spans(sp, 3.05, 1.5)
    assert spans == [(2.0, 3.0)], spans
    assert all(lo >= 2.0 for lo, _ in spans)


def test_cpu_cuda_parity():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    cpu = _build(device="cpu")
    cuda = _build(device="cuda")
    assert np.array_equal(cpu["span_count"], cuda["span_count"])
    assert np.allclose(cpu["span_min"], cuda["span_min"], atol=1e-5)
    assert np.allclose(cpu["span_max"], cuda["span_max"], atol=1e-5)
