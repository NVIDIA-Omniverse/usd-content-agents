# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the shared Joint Rigger artifact transaction."""

from world_understanding.functions.physics.joint_rigger.artifacts import (
    StagedArtifact,
    promote_staged_artifacts,
    remove_artifact,
)

__all__ = ["StagedArtifact", "promote_staged_artifacts", "remove_artifact"]
