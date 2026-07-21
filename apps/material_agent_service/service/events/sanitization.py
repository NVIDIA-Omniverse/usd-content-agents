# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound service-facing step completion payloads without changing raw results."""

from __future__ import annotations

from typing import Any


def bounded_step_completion_data(data: dict[str, Any]) -> dict[str, Any]:
    """Remove scene-sized ID evidence from service progress payloads."""
    outputs = data.get("outputs")
    if not isinstance(outputs, dict):
        return data

    bounded_data = dict(data)
    bounded_outputs = dict(outputs)

    restore_stats = bounded_outputs.get("restore_stats")
    if isinstance(restore_stats, dict):
        bounded_restore_stats = dict(restore_stats)
        for evidence_key, count_key in (
            ("restored_prim_sources", "restored_prim_source_count"),
            ("uncovered_originals", "uncovered_original_count"),
            ("unconsumed_predictions", "unconsumed_prediction_count"),
            ("mapping_warnings", "mapping_warning_count"),
        ):
            evidence = bounded_restore_stats.pop(evidence_key, None)
            if isinstance(evidence, dict | list | set | tuple):
                bounded_restore_stats[count_key] = len(evidence)
        bounded_outputs["restore_stats"] = bounded_restore_stats

    assignment_stats = bounded_outputs.get("assignment_stats")
    if isinstance(assignment_stats, dict):
        bounded_assignment_stats = dict(assignment_stats)
        for evidence_key, count_key in (
            ("bound_prim_ids", "bound_prim_count"),
            ("unbound_prim_ids", "unbound_prim_count"),
        ):
            evidence = bounded_assignment_stats.pop(evidence_key, None)
            if isinstance(evidence, list | set | tuple):
                bounded_assignment_stats[count_key] = len(evidence)
        bounded_outputs["assignment_stats"] = bounded_assignment_stats

    bounded_data["outputs"] = bounded_outputs
    return bounded_data
