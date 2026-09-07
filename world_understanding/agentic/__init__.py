# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic framework for workflow orchestration."""

from importlib import import_module

from . import (
    agents,
    base,
    cli,
    config,
    dataset,
    domain_tasks,
    harness,
    session,
    tasks,
    workflows,
)

__all__ = [
    "agents",
    "base",
    "cli",
    "config",
    "create_usd_dataset_workflow",
    "dataset",
    "domain_tasks",
    "harness",
    "session",
    "tasks",
    "workflows",
]


def __getattr__(name: str) -> object:
    """Load the USD workflow only when callers request that public helper."""

    if name != "create_usd_dataset_workflow":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    workflow = import_module(f"{__name__}.usd_workflows").create_usd_dataset_workflow
    globals()[name] = workflow
    return workflow
