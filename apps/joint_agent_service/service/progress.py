# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared progress metadata for Joint Agent service status reporting."""

SERVICE_DEFAULT_TOTAL_STEPS = 7

STEP_DISPLAY_NAMES = {
    "optimize_usd": "Optimizing USD",
    "identify_asset": "Identifying Asset",
    "analyze_structure": "Analyzing Structure",
    "build_dataset_usd": "Rendering USD Scene",
    "build_dataset_prepare_dataset": "Preparing Dataset",
    "predict": "Running VLM Predictions",
    "consistency_pass": "Checking Prediction Consistency",
    "infer_articulation_candidates": "Inferring Articulation Candidates",
    "restore_usd": "Restoring Predictions",
    "apply_joint_rigger": "Applying Joint Rigger",
}

# `optimize_usd` and `identify_asset` both occupy the first progress slot because
# service runs can start with either path. The named `current_step` metadata still
# carries the exact active step for UI display.
STEP_NUMBERS = {
    "optimize_usd": 1,
    "identify_asset": 1,
    "analyze_structure": 2,
    "build_dataset_usd": 3,
    "build_dataset_prepare_dataset": 4,
    "predict": 5,
    "consistency_pass": 6,
    "infer_articulation_candidates": 7,
    "restore_usd": 8,
    "apply_joint_rigger": 9,
}

STEP_PROGRESS_WEIGHTS = {
    "optimize_usd": (0, 5),
    "identify_asset": (5, 15),
    "analyze_structure": (15, 25),
    "build_dataset_usd": (25, 60),
    "build_dataset_prepare_dataset": (60, 70),
    "predict": (70, 88),
    "consistency_pass": (88, 92),
    "infer_articulation_candidates": (92, 96),
    "restore_usd": (96, 98),
    "apply_joint_rigger": (98, 99),
}

STEP_COMPLETION_PERCENTS = {
    step_name: bounds[1] for step_name, bounds in STEP_PROGRESS_WEIGHTS.items()
}


def step_display_name(step_name: str) -> str:
    """Return the user-facing display name for a pipeline step."""
    return STEP_DISPLAY_NAMES.get(step_name, step_name)


def step_overall_percent(step_name: str, step_percent: int | float) -> int | None:
    """Map a step-local percent into the overall service progress range."""
    if step_name not in STEP_PROGRESS_WEIGHTS:
        return None
    start, end = STEP_PROGRESS_WEIGHTS[step_name]
    return min(100, start + int((end - start) * step_percent / 100))
