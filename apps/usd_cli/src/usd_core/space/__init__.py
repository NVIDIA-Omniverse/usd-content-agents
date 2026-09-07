# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empty-space detection — where an object *fits* and where it can *rest*.

The counterpart to `usd_core.spatial`: those verbs (`raycast`, `nearest`, `within`,
`overlapping`, `distance`) locate **objects** from their bounds; this locates the
**absence** of objects from actual triangles. Two read-only queries, one detector:

- `space free`    — how much room is left above each surface
- `space support` — where one object of a given size can rest stably

Both run `support.detect_support_regions` over the same unit (a surface and the
clearance above it) and differ only in the height they report and how they rank.
`--container` scopes either to a named container; with `--exclude-others` that is
the container's capacity.

Adapted from NVIDIA EmptySpaceDetectionDemo under Apache-2.0. The retained
license text is in ``licenses/LICENSE.emptyspacedetection.Apache-2.0``; adapted
source files retain their SPDX notices.

**Nothing is imported here.** The compute layer pulls in NVIDIA Warp, which costs
hundreds of ms to initialize (CUDA device enumeration + JIT cache warm) and is an
optional extra. `usd_core.session` imports must stay cheap and must not hard-fail
on a host without `[space]`, so the space commands import `adapter` / `support` /
`geometry` lazily at call time. Import them the same way from new code:

    from usd_core.space.adapter import scene_geometry   # inside the function
"""

from __future__ import annotations

__all__ = ["SpaceError", "SpaceInvalidArgument", "SpaceOutOfRange", "SpaceUnavailable"]


def __getattr__(name: str):
    """Expose the exception types without importing the numpy/warp compute layer.

    These are the one thing callers need at module scope — `usd_cli.main` and
    `usd_server.app` catch them to shape the error envelope — and they live in a
    pure-stdlib module, so serving them lazily keeps `usd_core.space` free of any
    import cost while `from usd_core.space import SpaceUnavailable` still works.
    """
    if name in ("SpaceError", "SpaceInvalidArgument", "SpaceOutOfRange"):
        from . import errors

        return getattr(errors, name)
    if name == "SpaceUnavailable":
        from ._warp import SpaceUnavailable

        return SpaceUnavailable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
