# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.tasks.apply_completion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import material_agent.tasks.apply_completion as completion_mod
from material_agent.tasks.apply_completion import ApplyCompletionTask


def _patch_listener(monkeypatch: pytest.MonkeyPatch) -> Mock:
    listener = Mock()
    monkeypatch.setattr(
        completion_mod,
        "get_listener",
        lambda context, logger_name=None: listener,
    )
    return listener


def test_apply_completion_with_unresolved_and_multi_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _patch_listener(monkeypatch)
    context = {
        "unique_materials": ["Steel", "Plastic", "Glass"],
        "matched_materials": {
            "Steel": ["/Looks/SteelA", "/Looks/SteelB"],
            "Plastic": [],
            "Glass": ["/Looks/Glass"],
        },
        "unresolved_materials": ["Plastic"],
        "search_stats": {"successful_queries": 2, "total_queries": 4},
        "resolved_materials": {"Steel": "/tmp/steel.mdl"},
        "download_stats": {"resolved": 3},
        "materials_applied": {"meshA": "Steel"},
        "assignment_stats": {"total_prims": 7, "unknown": 2},
        "output_usd_path": Path("/tmp/output.usd"),
        "layer_only": True,
        "rendered_image_path": Path("/tmp/render.png"),
        "rendered_image_paths": [Path("/tmp/a.png"), Path("/tmp/b.png")],
        "rendering_skipped": False,
    }

    result = ApplyCompletionTask().run(context)
    summary = result["summary"]

    assert result["application_complete"] is True
    assert summary == {
        "materials_identified": 3,
        "materials_with_matches": 2,
        "materials_unresolved": 1,
        "total_matches_found": 3,
        "search_success_rate": 50.0,
        "materials_resolved": 1,
        "paths_resolved": 3,
        "materials_applied_to_usd": 1,
        "prims_with_materials": 7,
        "unknown_material_predictions": 2,
        "output_mode": "Layer only",
        "output_path": str(Path("/tmp/output.usd")),
        "rendered_image_path": str(Path("/tmp/render.png")),
        "rendered_image_paths": [str(Path("/tmp/a.png")), str(Path("/tmp/b.png"))],
        "rendering_skipped": False,
    }
    listener.warning.assert_any_call("    Unresolved materials:")
    listener.warning.assert_any_call("      - Plastic")
    listener.warning.assert_any_call(
        "  - Unknown material predictions: 2 prim(s) could not be classified"
    )
    listener.info.assert_any_call("  • Rendered images (2 views):")


def test_apply_completion_defaults_and_single_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _patch_listener(monkeypatch)

    result = ApplyCompletionTask().run(
        {
            "matched_materials": {"Steel": ["/Looks/Steel"]},
            "search_stats": {"successful_queries": 0, "total_queries": 0},
            "assignment_stats": {"unknown": "not-a-count"},
            "rendered_image_paths": [Path("/tmp/one.png")],
            "rendering_skipped": False,
        }
    )
    summary = result["summary"]

    assert summary["materials_identified"] == 0
    assert summary["materials_with_matches"] == 1
    assert summary["materials_unresolved"] == 0
    assert summary["total_matches_found"] == 1
    assert summary["search_success_rate"] == 0.0
    assert summary["materials_resolved"] == 0
    assert summary["paths_resolved"] == 0
    assert summary["materials_applied_to_usd"] == 0
    assert summary["prims_with_materials"] == 0
    assert summary["unknown_material_predictions"] == 0
    assert summary["output_mode"] == "Full stage"
    assert summary["output_path"] is None
    assert summary["rendered_image_path"] is None
    assert summary["rendered_image_paths"] == [str(Path("/tmp/one.png"))]
    assert summary["rendering_skipped"] is False
    listener.info.assert_any_call(f"  • Rendered image: {Path('/tmp/one.png')}")
