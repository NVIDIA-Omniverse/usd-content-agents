# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared producer/consumer artifact contract for mesh segmentation."""

from __future__ import annotations

from typing import Literal

MeshSegmentationWorkflowMode = Literal["targeted", "recognition"]

MESH_SEGMENTATION_REQUIRED_SKILLS = (
    "content-workflow-mesh-segmentation",
    "image-generation",
    "usd-cli",
)

REQUIRED_FINAL_ARTIFACTS = (
    "part_plan.json",
    "segments.json",
    "prepare/topology.json",
    "fragments/fragment_manifest.json",
    "fragments/fragment_ids.u32le",
    "fragments/fragment_ids.npy",
    "fragments/fragment_adjacency.npy",
    "fragments/fragment_statistics.npz",
    "fragments/fragment_colors.npy",
    "fragments/fragments.usdc",
    "state/final_labels.u32le",
    "final/segmented.usdc",
    "final/export_manifest.json",
    "final/renders/render_manifest.json",
    "final/report.md",
)

# Per-part initializer choices and visual findings are validated through their
# bound falsification evidence and final report. The current producer does not
# emit a standalone visual-quality-assessment artifact.
REQUIRED_TARGETED_FINAL_ARTIFACTS = (
    "final/face_labels.u32le",
    "final/segment_manifest.json",
    "final/target_evidence.json",
    "final/topology_validation.json",
    "final/random_color_validation_renders/render_validation.json",
)

REQUIRED_RECOGNITION_FINAL_ARTIFACTS = (
    "part_work_queue.json",
    "part_lock_manifest.json",
    "live_sequential_gate.json",
    "part_review_manifest.json",
    "final/face_labels.u32le",
)

CONTINUATION_ARTIFACT = "continuation_seed/manifest.json"


def required_mesh_segmentation_artifacts(
    mode: MeshSegmentationWorkflowMode,
    *,
    resumable: bool = False,
) -> tuple[str, ...]:
    """Return the exact ordered terminal artifact contract for one run mode."""

    if mode == "targeted":
        mode_artifacts = REQUIRED_TARGETED_FINAL_ARTIFACTS
    elif mode == "recognition":
        mode_artifacts = REQUIRED_RECOGNITION_FINAL_ARTIFACTS
    else:
        raise ValueError(f"unsupported mesh-segmentation workflow mode: {mode!r}")
    continuation = (CONTINUATION_ARTIFACT,) if resumable else ()
    return (*REQUIRED_FINAL_ARTIFACTS, *mode_artifacts, *continuation)


__all__ = [
    "CONTINUATION_ARTIFACT",
    "MESH_SEGMENTATION_REQUIRED_SKILLS",
    "MeshSegmentationWorkflowMode",
    "REQUIRED_FINAL_ARTIFACTS",
    "REQUIRED_RECOGNITION_FINAL_ARTIFACTS",
    "REQUIRED_TARGETED_FINAL_ARTIFACTS",
    "required_mesh_segmentation_artifacts",
]
