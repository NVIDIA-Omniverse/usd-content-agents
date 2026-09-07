# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the `--device` axis (auto | cpu | cuda) for the space verbs.

Mirrors `usd_core.render.factory`'s shape for the `renderer` axis — a cheap
availability probe, a `_resolve_auto`, and a public `resolved_device()` that never
raises — so the two "pick a backend" axes in the repo read the same way.

Why `auto` is worth having, measured on this repo's own assets (2x RTX 6000 Ada):

    Siemens PCB, 2.3M triangles, `space support`   cpu 1.27s   cuda 0.27s   4.7x
    synthetic grid, 145k triangles                 cpu 0.100s  cuda 0.047s  2.1x
    synthetic grid, 204 triangles                  cpu 0.071s  cuda 0.057s  1.2x

CUDA was not slower at any size tested, so `auto` does not need a size threshold. The
one cost is a ~0.11s CUDA context + kernel-module load on the first query in a process;
the daemon is long-lived, so it is paid once and amortised over every later query.

**Resolved devices are reported, never silent.** `space` puts the concrete device in
its response summary the way `render` reports `backend`. That matters more here than
for rendering: CPU and CUDA are not guaranteed bit-identical on dense geometry. The
rasteriser appends a cell's solid layers with an atomic counter and keeps at most
`heightfield.MAX_SOLID` of them, so on a scene where cells overflow that buffer, *which*
layers survive depends on thread order and the two devices can disagree. Measured on
the Siemens PCB (4165 of 42180 cells overflow, up to 1808 layers in one cell): 54
regions on CPU vs 53 on CUDA, 0.67% difference in total support area. Each device is
internally deterministic — repeated runs agree — but they are not interchangeable at
that precision. Pass `--device cpu` explicitly when a result has to be reproducible
across machines.
"""

from __future__ import annotations

from .errors import SpaceInvalidArgument

#: The devices `--device` accepts. `auto` is resolved by `resolved_device`.
DEVICES = ("auto", "cpu", "cuda")


def cuda_available() -> bool:
    """True when Warp can actually launch on a CUDA device.

    Asks Warp rather than probing for an NVIDIA device node the way
    `render.factory._ovrtx_available` does. The render probe is deliberately
    import-light because importing ovrtx is expensive; here Warp is already being
    imported to run the kernels, and only Warp knows whether it has a usable CUDA
    toolchain — a machine can have `/dev/nvidia0` and still fail to launch.
    """
    from ._warp import SpaceUnavailable, warp

    try:
        wp = warp()
    except SpaceUnavailable:
        return False  # no `space` extra at all; the caller reports that separately
    try:
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001 — a probe must never be the thing that fails
        return False


def resolved_device(requested: str | None) -> str:
    """The concrete device a space query will run on. Never returns `"auto"`.

    Raises `SpaceInvalidArgument` for an unknown device, or for an explicit `cuda`
    on a host that cannot run it — failing here with a sentence beats failing inside
    a Warp kernel launch with a stack trace.
    """
    device = (requested or "auto").lower().strip()
    if device not in DEVICES:
        raise SpaceInvalidArgument(
            f"unknown --device '{requested}' (expected one of {', '.join(DEVICES)})")
    if device == "cuda" and not cuda_available():
        raise SpaceInvalidArgument(
            "--device cuda was requested but Warp reports no usable CUDA device "
            "(no NVIDIA GPU, no driver, or a Warp build without CUDA support). "
            "Use --device cpu, or --device auto to pick whatever is available.")
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    return device
