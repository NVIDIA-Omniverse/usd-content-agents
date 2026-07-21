# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime infrastructure for event-driven pipeline execution.

This module contains:
- EventBus: SSE event streaming to web clients
- JobRegistry: Async job lifecycle management
- ProgressEvent: Event model for SSE
"""

from .bus import EventBus, get_event_bus
from .events import ProgressEvent, StepState
from .registry import JobRegistry, get_job_registry
from .texture_execution import (
    TEXTURE_EXECUTION_ACCEPTED_PREFIX,
    TEXTURE_EXECUTION_CHECKPOINT_KEY,
    SessionTextureExecutionCheckpointStore,
)

__all__ = [
    "EventBus",
    "JobRegistry",
    "ProgressEvent",
    "SessionTextureExecutionCheckpointStore",
    "StepState",
    "TEXTURE_EXECUTION_ACCEPTED_PREFIX",
    "TEXTURE_EXECUTION_CHECKPOINT_KEY",
    "get_event_bus",
    "get_job_registry",
]
