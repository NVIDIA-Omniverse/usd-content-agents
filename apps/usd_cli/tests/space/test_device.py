# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`--device auto` resolution — original tests, not ported from upstream.

The GPU-present branches run for real on a CUDA host and are skipped elsewhere; the
GPU-absent branches are exercised everywhere by faking the probe, because a CI box
without a GPU must still prove that `auto` degrades to `cpu` and that an explicit
`cuda` fails with a sentence rather than inside a kernel launch.
"""

from __future__ import annotations

import pytest

wp = pytest.importorskip("warp")
from usd_core.space.device import (  # noqa: E402
    DEVICES,
    cuda_available,
    resolved_device,
)
from usd_core.space.errors import SpaceInvalidArgument  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not cuda_available(), reason="no CUDA device available to Warp")


def test_auto_never_leaks_through():
    """Callers downstream index Warp by device string; `auto` is not one."""
    assert resolved_device("auto") in ("cpu", "cuda")
    assert resolved_device(None) in ("cpu", "cuda")
    assert resolved_device("") in ("cpu", "cuda")


def test_explicit_cpu_is_honoured():
    """`--device cpu` must stay cpu even where cuda is available.

    This is the reproducibility escape hatch: the two devices are each deterministic
    but can differ on geometry dense enough to overflow the per-cell solid-layer
    buffer, so a caller that needs a stable answer has to be able to pin cpu.
    """
    assert resolved_device("cpu") == "cpu"
    assert resolved_device("CPU") == "cpu", "device should be case-insensitive"


def test_unknown_device_is_rejected_by_name():
    with pytest.raises(SpaceInvalidArgument) as excinfo:
        resolved_device("gpu")
    message = str(excinfo.value)
    assert "gpu" in message, "the rejection must quote what was actually passed"
    for name in DEVICES:
        assert name in message, "the rejection must list the accepted devices"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    """The branch a GPU-less CI box takes."""
    monkeypatch.setattr("usd_core.space.device.cuda_available", lambda: False)
    assert resolved_device("auto") == "cpu"


def test_explicit_cuda_without_cuda_fails_clearly(monkeypatch):
    """Fail here, not deep inside a Warp launch.

    A machine can have an NVIDIA device node and still not be able to launch (no
    driver, or a Warp build without CUDA), which is why the probe asks Warp rather
    than looking for /dev/nvidia*.
    """
    monkeypatch.setattr("usd_core.space.device.cuda_available", lambda: False)
    with pytest.raises(SpaceInvalidArgument) as excinfo:
        resolved_device("cuda")
    message = str(excinfo.value)
    assert "cuda" in message.lower()
    assert "--device cpu" in message, "the error should name the way out"


def test_probe_survives_a_broken_warp(monkeypatch):
    """A probe must never be the thing that fails."""
    class Boom:
        @staticmethod
        def is_cuda_available():
            raise RuntimeError("driver exploded")

    monkeypatch.setattr("usd_core.space._warp.warp", lambda: Boom())
    assert cuda_available() is False


def test_probe_reports_false_without_the_space_extra(monkeypatch):
    from usd_core.space._warp import SpaceUnavailable

    def no_warp():
        raise SpaceUnavailable("not installed")

    monkeypatch.setattr("usd_core.space._warp.warp", no_warp)
    assert cuda_available() is False


@requires_cuda
def test_auto_picks_cuda_when_available():
    assert resolved_device("auto") == "cuda"


@requires_cuda
def test_cpu_and_cuda_agree_on_simple_geometry():
    """Parity where it is guaranteed: no cell overflows the solid-layer buffer.

    Dense scenes are a different matter — see `device.py`'s module docstring — so this
    pins the case that must never drift, rather than claiming a parity that does not
    hold in general.
    """
    import sys

    sys.path.insert(0, "tests")
    from space_synthetic import scene_from_boxes

    from usd_core.space.heightfield import MAX_SOLID
    from usd_core.space.support import detect_support_regions

    scene = scene_from_boxes([
        ((0.0, 0.0, -0.02), (1.0, 1.0, 0.0)),
        ((0.2, 0.2, 0.0), (0.5, 0.5, 0.15)),
    ])
    scope = ((0, 0, -0.05), (1, 1, 0.6))
    obj = (0.1, 0.1, 0.05)
    out = {d: detect_support_regions(scene, *scope, 0.02, obj, device=d)
           for d in ("cpu", "cuda")}

    from usd_core.space.device_mesh import DeviceMesh
    from usd_core.space.heightfield import build_spans_raster
    spans = build_spans_raster(DeviceMesh(scene, device="cpu"), *scope, 0.02, obj[2])
    assert spans["solid_overflow_cells"] == 0, (
        f"fixture must stay under MAX_SOLID={MAX_SOLID} for parity to be guaranteed")

    assert len(out["cpu"].regions) == len(out["cuda"].regions)
    for a, b in zip(out["cpu"].regions, out["cuda"].regions):
        assert a.support_z == pytest.approx(b.support_z, abs=1e-5)
        assert sorted(a.cells) == sorted(b.cells)
