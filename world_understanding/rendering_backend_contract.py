# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight canonical contracts for USD rendering backends.

This module deliberately does not import renderer implementations. Schemas,
CLIs, and independently packaged agents can therefore share backend names,
capability validation, limits, and errors without loading USD or optional GPU
dependencies.
"""

from __future__ import annotations

from collections.abc import Collection

RENDERING_BACKEND_NAMES: tuple[str, ...] = ("remote", "warp", "ovrtx", "mock")
SUPPORTED_RENDERING_BACKENDS: frozenset[str] = frozenset(RENDERING_BACKEND_NAMES)
MAX_REMOTE_RENDER_WORKERS = 32


class RenderingBackendContractError(ValueError):
    """Base error for invalid USD rendering-backend selections."""


class UnknownRenderingBackendError(RenderingBackendContractError):
    """Raised when a selector is not part of the canonical backend registry."""


class UnsupportedRenderingBackendError(RenderingBackendContractError):
    """Raised when a canonical backend is unavailable on a narrower surface."""


class RemoteRenderingSlotTimeoutError(TimeoutError):
    """Raised when the process-wide remote-render slot cannot be acquired."""


def validate_remote_render_max_workers(max_workers: object) -> int:
    """Return a bounded remote-render worker count or raise ``ValueError``."""
    if (
        not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or not 1 <= max_workers <= MAX_REMOTE_RENDER_WORKERS
    ):
        raise ValueError(
            "Remote render max_workers must be an integer between 1 and "
            f"{MAX_REMOTE_RENDER_WORKERS}"
        )
    return max_workers


def validate_rendering_backend_name(backend_type: object) -> str:
    """Return a canonical backend name or raise a clear configuration error."""
    if not isinstance(backend_type, str) or (
        backend_type not in SUPPORTED_RENDERING_BACKENDS
    ):
        supported = ", ".join(sorted(SUPPORTED_RENDERING_BACKENDS))
        raise UnknownRenderingBackendError(
            f"Unknown rendering backend: {backend_type}. "
            f"Supported backends: {supported}"
        )
    return backend_type


def rendering_backend_subset(*backend_names: str) -> tuple[str, ...]:
    """Return a validated capability subset in canonical registry order."""
    requested: set[str] = set()
    for backend_name in backend_names:
        canonical_name = validate_rendering_backend_name(backend_name)
        if canonical_name in requested:
            raise ValueError(
                f"Duplicate rendering backend in capability subset: {canonical_name}"
            )
        requested.add(canonical_name)
    return tuple(
        backend_name
        for backend_name in RENDERING_BACKEND_NAMES
        if backend_name in requested
    )


def validate_rendering_backend_for_surface(
    backend_type: object,
    supported_backends: Collection[str],
    *,
    surface: str,
) -> str:
    """Validate a selector globally, then against one capability-restricted surface.

    Validation order is intentional: callers can distinguish a typo from a
    canonical backend that the selected surface cannot safely support.
    """
    canonical_subset = rendering_backend_subset(*tuple(supported_backends))
    try:
        canonical_name = validate_rendering_backend_name(backend_type)
    except UnknownRenderingBackendError as exc:
        supported = ", ".join(canonical_subset)
        raise UnknownRenderingBackendError(
            f"{exc}. Supported by {surface}: {supported}"
        ) from exc
    if canonical_name not in canonical_subset:
        supported = ", ".join(canonical_subset)
        raise UnsupportedRenderingBackendError(
            f"Rendering backend {canonical_name!r} is recognized but unsupported "
            f"by {surface}. Supported backends: {supported}"
        )
    return canonical_name
