# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private process-limit handshake for the isolated OvPhysX daemon."""

from __future__ import annotations

import os
from collections.abc import MutableMapping

try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows has no RLIMIT_AS.
    _resource = None  # type: ignore[assignment]


OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV = "_WU_OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE"


def relax_address_space_limit_from_environment(
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Raise this daemon's soft RLIMIT_AS to its inherited hard ceiling.

    The parent client adds the private one-shot marker only for an explicitly
    opted-in daemon launch. Consuming it keeps the capability out of the daemon's
    own children and leaves upstream hard limits authoritative.
    """

    environment = os.environ if environ is None else environ
    requested = environment.pop(OVPHYSX_DAEMON_RELAX_ADDRESS_SPACE_ENV, None)
    if requested != "1" or _resource is None:
        return False
    _soft, hard = _resource.getrlimit(_resource.RLIMIT_AS)
    _resource.setrlimit(_resource.RLIMIT_AS, (hard, hard))
    return True
