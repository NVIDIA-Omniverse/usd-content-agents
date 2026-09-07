# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage-aware camera-rig analysis for usd-cli.

The package deliberately keeps USD ingestion, the Newton/Warp observation backend,
and product policy separate.  Importing :mod:`usd_core` therefore does not import
Newton, Warp, or initialize CUDA; those imports happen only when an analysis command
constructs a backend.
"""

from usd_core.camera_analysis.contracts import (
    AnalysisRole,
    CameraBatchRequest,
    CameraIR,
    CameraObservation,
    CameraPose,
    MeshResource,
    RayHits,
    SceneAnalysisIR,
    SceneAnalysisPolicy,
    ShapeIR,
    VisibilityBackend,
)
from usd_core.camera_analysis.cancellation import (
    CameraAnalysisCancelled,
    cancellation_scope,
    check_cancelled,
    defer_cancellation,
)
from usd_core.camera_analysis.look_at import LookAtConfig, LookAtResult

__all__ = [
    "AnalysisRole",
    "CameraBatchRequest",
    "CameraAnalysisCancelled",
    "CameraIR",
    "CameraObservation",
    "CameraPose",
    "LookAtConfig",
    "LookAtResult",
    "MeshResource",
    "RayHits",
    "SceneAnalysisIR",
    "SceneAnalysisPolicy",
    "ShapeIR",
    "VisibilityBackend",
    "cancellation_scope",
    "check_cancelled",
    "defer_cancellation",
]
