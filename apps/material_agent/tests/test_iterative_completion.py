# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for iterative apply completion output hardening."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from pxr import Sdf

from material_agent.tasks.iterative_completion import IterativeApplyCompletionTask


def _create_empty_layer(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(path))
    layer.Save()
    return path


def test_copy_usd_with_updated_paths_drops_unsafe_sublayers(tmp_path: Path) -> None:
    """Generated final USD should not preserve resolver URI or host paths."""
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "final"
    source_dir.mkdir()
    dest_dir.mkdir()
    safe_sublayer = _create_empty_layer(source_dir / "layers" / "safe.usda")
    outside_sublayer = _create_empty_layer(tmp_path / "outside.usda")

    source_usd = source_dir / "iteration_output.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_usd))
    source_layer.subLayerPaths = [
        os.path.relpath(safe_sublayer, source_dir),
        "https://metadata.example.invalid/latest",
        "file:///var/run/secrets/kubernetes.io/serviceaccount/token",
        "/etc/shadow",
        os.path.relpath(outside_sublayer, source_dir),
        "C:/Users/secret/materials.usda",
    ]
    source_layer.Save()

    dest_usd = dest_dir / "final.usda"
    listener = MagicMock()

    IterativeApplyCompletionTask()._copy_usd_with_updated_paths(
        source_usd,
        dest_usd,
        listener,
    )

    dest_layer = Sdf.Layer.FindOrOpen(str(dest_usd))
    assert dest_layer is not None
    assert list(dest_layer.subLayerPaths) == [
        os.path.relpath(safe_sublayer.resolve(), dest_dir).replace("\\", "/")
    ]

    warning_messages = " ".join(
        str(call.args[0]) for call in listener.warning.call_args_list
    )
    assert "Dropping unsafe sublayer path" in warning_messages
    assert "resolver URI schemes are not allowed" in warning_messages
    assert "absolute asset path is outside source directory" in warning_messages
    assert "asset path escapes its source directory" in warning_messages


def test_copy_usd_with_updated_paths_drops_unsafe_references_and_payloads(
    tmp_path: Path,
) -> None:
    """Final root USD should not preserve unsafe refs or payloads."""
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "final"
    source_dir.mkdir()
    dest_dir.mkdir()
    safe_ref = _create_empty_layer(source_dir / "refs" / "safe_ref.usda")
    safe_payload = _create_empty_layer(source_dir / "payloads" / "safe_payload.usda")
    outside_ref = _create_empty_layer(tmp_path / "outside_ref.usda")

    source_usd = source_dir / "iteration_output.usda"
    source_layer = Sdf.Layer.CreateNew(str(source_usd))
    prim = Sdf.CreatePrimInLayer(source_layer, "/World/Asset")
    prim.specifier = Sdf.SpecifierDef
    prim.referenceList.prependedItems = [
        Sdf.Reference(os.path.relpath(safe_ref, source_dir)),
        Sdf.Reference("https://metadata.example.invalid/ref.usda"),
        Sdf.Reference(os.path.relpath(outside_ref, source_dir)),
    ]
    prim.payloadList.prependedItems = [
        Sdf.Payload(os.path.relpath(safe_payload, source_dir)),
        Sdf.Payload("file:///etc/shadow"),
    ]
    source_layer.Save()

    dest_usd = dest_dir / "final.usda"
    listener = MagicMock()

    IterativeApplyCompletionTask()._copy_usd_with_updated_paths(
        source_usd,
        dest_usd,
        listener,
    )

    dest_layer = Sdf.Layer.FindOrOpen(str(dest_usd))
    assert dest_layer is not None
    dest_prim = dest_layer.GetObjectAtPath("/World/Asset")
    assert isinstance(dest_prim, Sdf.PrimSpec)

    assert [item.assetPath for item in dest_prim.referenceList.prependedItems] == [
        os.path.relpath(safe_ref.resolve(), dest_dir).replace("\\", "/")
    ]
    assert [item.assetPath for item in dest_prim.payloadList.prependedItems] == [
        os.path.relpath(safe_payload.resolve(), dest_dir).replace("\\", "/")
    ]

    exported = dest_usd.read_text()
    assert "https://metadata.example.invalid" not in exported
    assert "file:///etc/shadow" not in exported
    assert "outside_ref" not in exported


def test_iterative_completion_run_copies_final_outputs(tmp_path: Path) -> None:
    iter_dir = tmp_path / "iterations" / "iteration_1"
    final_iteration_output = _create_empty_layer(iter_dir / "iteration_output.usda")
    flattened = _create_empty_layer(iter_dir / "iteration_output_render_flat.usd")
    renders_dir = iter_dir / "renders"
    renders_dir.mkdir()
    (renders_dir / "view.png").write_bytes(b"png")
    final_output = tmp_path / "final" / "asset.usda"
    stale_renders = final_output.parent / "renders"
    stale_renders.mkdir(parents=True)
    (stale_renders / "old.png").write_bytes(b"old")
    listener = MagicMock()

    context = {
        "iteration_count": 1,
        "termination_reason": "approved",
        "final_iteration": {
            "judge_score": 4.0,
            "materials_applied_count": 2,
            "prims_with_materials": 3,
        },
        "all_iteration_outputs": [str(final_iteration_output)],
        "final_output_usd_path": str(final_output),
    }

    with patch(
        "material_agent.tasks.iterative_completion.get_listener",
        return_value=listener,
    ):
        result = IterativeApplyCompletionTask().run(context)

    assert result["iterative_apply_complete"] is True
    assert result["final_output_path"] == str(final_output)
    assert result["summary"] == {
        "iteration_count": 1,
        "termination_reason": "approved",
        "final_score": 4.0,
        "final_materials_applied": 2,
        "final_prims_with_materials": 3,
        "all_iteration_outputs": [str(final_iteration_output)],
        "final_output_path": str(final_output),
    }
    assert final_output.exists()
    assert (final_output.parent / "asset_render_flat.usd").read_bytes() == (
        flattened.read_bytes()
    )
    assert (final_output.parent / "renders" / "view.png").read_bytes() == b"png"
    assert not (final_output.parent / "renders" / "old.png").exists()


def test_iterative_completion_run_handles_missing_or_implicit_final_output(
    tmp_path: Path,
) -> None:
    listener = MagicMock()
    missing_output = tmp_path / "missing.usda"
    final_output = tmp_path / "final.usda"

    with patch(
        "material_agent.tasks.iterative_completion.get_listener",
        return_value=listener,
    ):
        missing_result = IterativeApplyCompletionTask().run(
            {
                "iteration_count": 1,
                "all_iteration_outputs": [str(missing_output)],
                "final_output_usd_path": str(final_output),
            }
        )
        implicit_result = IterativeApplyCompletionTask().run(
            {
                "iteration_count": 1,
                "all_iteration_outputs": [str(missing_output)],
            }
        )

    assert missing_result["final_output_path"] == str(missing_output)
    assert implicit_result["final_output_path"] == str(missing_output)
    assert listener.warning.call_args.args[0].startswith(
        "Final iteration output not found"
    )


def test_iterative_completion_sanitize_empty_asset_path() -> None:
    listener = MagicMock()
    assert (
        IterativeApplyCompletionTask()._sanitize_layer_asset_path(
            "",
            Path("/source"),
            Path("/dest"),
            listener,
            "asset path",
        )
        == ""
    )
    listener.warning.assert_not_called()
