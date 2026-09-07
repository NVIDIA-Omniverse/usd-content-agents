# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry-point registry for optional optimizer implementations."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata

from .contracts import SUPPORTED_OPTIMIZERS

logger = logging.getLogger(__name__)

OptimizerRunner = Callable[..., None]
AvailabilityCheck = Callable[[], bool]

_OPTIMIZER_PLUGIN_GROUP = "world_understanding.optimizers"
_BUILT_IN_OPTIMIZERS = frozenset(SUPPORTED_OPTIMIZERS)
_optimizer_plugins_lock = threading.RLock()
_optimizer_plugins_scanned = False
_loaded_optimizer_plugins: set[str] = set()
_failed_optimizer_plugins: dict[str, str] = {}


@dataclass(frozen=True)
class OptimizerPlugin:
    """One installed optimizer implementation and its availability contract."""

    runner: OptimizerRunner
    is_available: AvailabilityCheck
    unavailable_message: str


_optimizer_plugins: dict[str, OptimizerPlugin] = {}


def load_optimizer_plugins() -> tuple[str, ...]:
    """Load installed optimizer registrars through package entry points."""

    global _optimizer_plugins_scanned

    with _optimizer_plugins_lock:
        if _optimizer_plugins_scanned:
            return tuple(sorted(_loaded_optimizer_plugins))

        entry_points = metadata.entry_points(group=_OPTIMIZER_PLUGIN_GROUP)
        for entry_point in entry_points:
            plugin_id = f"{entry_point.name}:{entry_point.value}"
            if (
                plugin_id in _loaded_optimizer_plugins
                or plugin_id in _failed_optimizer_plugins
            ):
                continue
            # One broken plugin must not take down built-in optimizers or other
            # installed plugins, so load failures are recorded and skipped.
            registered_before = dict(_optimizer_plugins)
            try:
                registrar = entry_point.load()
                if not callable(registrar):
                    raise TypeError(
                        f"Optimizer plugin {entry_point.name!r} must be callable"
                    )
                registrar()
            except Exception as error:
                _optimizer_plugins.clear()
                _optimizer_plugins.update(registered_before)
                _failed_optimizer_plugins[plugin_id] = str(error)
                logger.warning(
                    "Skipping broken optimizer plugin %s: %s", plugin_id, error
                )
                continue
            _loaded_optimizer_plugins.add(plugin_id)
        _optimizer_plugins_scanned = True
        return tuple(sorted(_loaded_optimizer_plugins))


def list_failed_optimizer_plugins() -> dict[str, str]:
    """Return load errors keyed by plugin identifier for diagnostics."""

    with _optimizer_plugins_lock:
        return dict(_failed_optimizer_plugins)


def register_optimizer(
    name: str,
    runner: OptimizerRunner,
    *,
    is_available: AvailabilityCheck | None = None,
    unavailable_message: str | None = None,
) -> None:
    """Register an optional optimizer without changing the public package."""

    normalized = name.strip() if isinstance(name, str) else ""
    if not normalized:
        raise ValueError("optimizer name must not be empty")
    if normalized in _BUILT_IN_OPTIMIZERS:
        raise ValueError(f"cannot replace built-in optimizer {normalized!r}")
    if not callable(runner):
        raise TypeError("optimizer runner must be callable")
    availability = is_available or (lambda: True)
    if not callable(availability):
        raise TypeError("optimizer availability check must be callable")
    message = unavailable_message or f"Optimizer {normalized!r} is unavailable"
    with _optimizer_plugins_lock:
        _optimizer_plugins[normalized] = OptimizerPlugin(
            runner=runner,
            is_available=availability,
            unavailable_message=message,
        )


def get_registered_optimizer(name: str) -> OptimizerPlugin | None:
    """Return an installed optimizer plugin, loading entry points first."""

    load_optimizer_plugins()
    with _optimizer_plugins_lock:
        return _optimizer_plugins.get(name)


def is_optimizer_registered(name: str) -> bool:
    """Return whether an optional optimizer is installed."""

    return get_registered_optimizer(name) is not None


def list_registered_optimizer_names() -> tuple[str, ...]:
    """Return the names of installed optional optimizers."""

    load_optimizer_plugins()
    with _optimizer_plugins_lock:
        return tuple(sorted(_optimizer_plugins))


def list_loaded_optimizer_plugins() -> tuple[str, ...]:
    """Return identifiers for successfully loaded optimizer entry points."""

    with _optimizer_plugins_lock:
        return tuple(sorted(_loaded_optimizer_plugins))


__all__ = [
    "AvailabilityCheck",
    "OptimizerPlugin",
    "OptimizerRunner",
    "get_registered_optimizer",
    "is_optimizer_registered",
    "list_failed_optimizer_plugins",
    "list_loaded_optimizer_plugins",
    "list_registered_optimizer_names",
    "load_optimizer_plugins",
    "register_optimizer",
]
