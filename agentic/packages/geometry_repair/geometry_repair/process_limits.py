# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable process resource-limit helpers."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from contextlib import contextmanager

ADDRESS_SPACE_LIMIT_MODE_HARD = "hard"
ADDRESS_SPACE_LIMIT_MODE_SOFT = "soft"
ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE = "worker_process_tree"
OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE = "collision_worker_parent_and_non_ovphysx_children"
OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION = (
    "ovphysx_daemon_only_may_raise_soft_limit_to_inherited_hard_ceiling_for_cuda_"
    "virtual_address_reservations"
)


def bounded_process_environment(*, deterministic_seed: int = 0) -> dict[str, str]:
    """Return an environment that bounds native thread pools and allocator arenas."""

    return {
        **os.environ,
        "MALLOC_ARENA_MAX": "2",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONHASHSEED": str(deterministic_seed),
        "PXR_WORK_THREAD_LIMIT": "1",
        "TBB_NUM_THREADS": "1",
    }


def limit_cpu_affinity(cpu_count: int = 1) -> None:
    """Restrict the process to a deterministic subset of its allowed CPUs when supported."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    if get_affinity is None or set_affinity is None:  # pragma: no cover - non-Linux hosts.
        return
    try:
        allowed = sorted(get_affinity(0))
        selected = allowed[: max(1, int(cpu_count))]
        if selected:
            set_affinity(0, selected)
    except OSError:
        return


@contextmanager
def temporary_cpu_affinity(cpu_count: int = 1) -> Iterator[None]:
    """Restrict the calling thread while work runs, then restore its allowed CPUs."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    original: set[int] | None = None
    if get_affinity is not None and set_affinity is not None:
        try:
            allowed = set(get_affinity(0))
            selected = sorted(allowed)[: max(1, int(cpu_count))]
            if selected and set(selected) != allowed:
                set_affinity(0, selected)
                original = allowed
        except OSError:
            pass
    try:
        yield
    finally:
        if original is not None:
            try:
                set_affinity(0, original)
            except OSError:
                pass


def limit_address_space(memory_mb: int) -> None:
    """Apply a hard POSIX address-space limit when the host supports it."""

    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX development hosts.
        return
    requested = int(memory_mb) * 1024 * 1024
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective = requested if hard in {-1, resource.RLIM_INFINITY} else min(requested, hard)
    resource.setrlimit(resource.RLIMIT_AS, (effective, effective))


def limit_address_space_soft(memory_mb: int) -> None:
    """Apply a POSIX address-space cap while preserving the inherited hard ceiling.

    This is reserved for workers that must launch a trusted child whose runtime
    legitimately reserves more virtual than resident memory. The worker remains
    capped; only that child may opt in to raising its inherited soft limit.
    """

    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX development hosts.
        return
    requested = int(memory_mb) * 1024 * 1024
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    effective = requested if hard in {-1, resource.RLIM_INFINITY} else min(requested, hard)
    resource.setrlimit(resource.RLIMIT_AS, (effective, hard))


def limit_file_size(max_bytes: int) -> None:
    """Apply a hard POSIX output-file limit when the host supports it."""

    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX development hosts.
        return
    requested = max(1, int(max_bytes))
    _soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    effective = requested if hard in {-1, resource.RLIM_INFINITY} else min(requested, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (effective, effective))


def limit_cpu_time(seconds: float) -> None:
    """Apply a hard POSIX CPU-time limit when the host supports it."""

    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX development hosts.
        return
    requested = max(1, int(math.ceil(seconds)))
    _soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    effective = requested if hard in {-1, resource.RLIM_INFINITY} else min(requested, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (effective, effective))
