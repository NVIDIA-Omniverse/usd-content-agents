# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the shared material-authoring graph primitives.

The graph implementation belongs to ``material_library_generation`` because it is
used by both fixed and agentic authoring.  Keep this module while the rendered
refinement API remains supported so existing imports do not break.
"""

from material_agent.material_library_generation import source_graph as _source_graph
from material_agent.material_library_generation.source_graph import *  # noqa: F403

__all__ = _source_graph.__all__


def __getattr__(name: str) -> object:
    """Forward private compatibility accesses used by older integrations."""

    return getattr(_source_graph, name)
