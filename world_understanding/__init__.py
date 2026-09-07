# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""World Understanding - Python library for Vision-Language Agents."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

# Version info
try:
    __version__ = version("world-understanding")
except PackageNotFoundError:
    __version__ = "0.0.1-dev"

__all__ = ["agentic", "functions", "nat", "registry", "tools", "utils", "__version__"]

_LAZY_SUBMODULES = frozenset(__all__[:-1])


def __getattr__(name: str) -> ModuleType:
    """Load public subpackages only when callers actually request them."""
    if name not in _LAZY_SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
