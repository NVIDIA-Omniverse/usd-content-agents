# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics utilities — both static USD inspection and live ovphysx-daemon helpers.

This package is intentionally pxr-free at the boundary that talks to ovphysx:
it is imported by code paths that either run inside the ovphysx daemon
(which has its own bundled USD) or in processes that haven't loaded
usd-core yet. Mixing pxr (usd-core 0.26.5) and ovphysx (bundled USD 25.11)
in the same process raises a USD-version incompatibility error at ovphysx
bootstrap; the daemon pattern in ``ovphysx_daemon.py`` is the long-running
workaround.

Static physics-USD inspection helpers (``physics_sanity``) live alongside
the daemon helpers in this package and may be imported freely from
pxr-using contexts.
"""

from world_understanding.functions.physics.ovphysx_daemon import (
    OvPhysXDaemonError,
    OvPhysXDaemonUnavailableError,
    _OvPhysXDaemon,
    ovphysx_runtime_available,
)
from world_understanding.functions.physics.physics_sanity import (
    PhysicsSanityFinding,
    PhysicsSanityResult,
    infer_physics_expected,
    inspect_usd_physics,
)

__all__ = [
    "OvPhysXDaemonError",
    "OvPhysXDaemonUnavailableError",
    "PhysicsSanityFinding",
    "PhysicsSanityResult",
    "_OvPhysXDaemon",
    "infer_physics_expected",
    "inspect_usd_physics",
    "ovphysx_runtime_available",
]
