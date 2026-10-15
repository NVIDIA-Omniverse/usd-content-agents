# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish accepted material assignments over the original USD geometry."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import file_sha256


def publish_material_layer(
    *,
    input_usd_path: Path,
    materials_usd_path: Path,
    output_usd_path: Path,
    assignments: list[dict[str, Any]],
    source_aliases: dict[str, set[str]],
    candidate_path_space: str | None,
    publish_output: Callable[[Path, Path, Callable[[Path], None]], None],
    validate_output: Callable[[Path], None],
) -> None:
    """Build a material overlay from canonical source-space assignments.

    The existing flat usd-cli output remains the workflow's render and review
    evidence. This function publishes a separate layer over the original input.
    """

    try:
        from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask
    except ImportError as exc:
        raise RuntimeError(
            "--layer-only requires the content-workflow-cli[materials] extra."
        ) from exc

    from pxr import Usd, UsdShade

    input_usd_path = input_usd_path.resolve()
    materials_usd_path = materials_usd_path.resolve()
    output_usd_path = output_usd_path.resolve()
    if output_usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("--layer-only output must use .usd, .usda, or .usdc.")
    source_stage = Usd.Stage.Open(str(input_usd_path))
    library_stage = Usd.Stage.Open(str(materials_usd_path))
    if source_stage is None or library_stage is None:
        raise ValueError(
            "Layer-only output requires readable source and material USDs."
        )

    resolved_materials: dict[str, str] = {}
    bindings: dict[str, str] = {}
    for group in assignments:
        if group.get("coverage_status") not in {None, "material_assignment"}:
            continue
        name = group.get("material_name")
        material_path = group.get("material_path")
        if not isinstance(name, str) or not name or not isinstance(material_path, str):
            raise ValueError("Accepted material assignment is missing its material.")
        material_prim = library_stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
            raise ValueError(f"Material is absent from the library: {material_path}")
        previous_material = resolved_materials.setdefault(name, material_path)
        if previous_material != material_path:
            raise ValueError(f"Material name maps to multiple library paths: {name}")

        source_paths = group.get("source_prim_paths")
        if not source_paths and candidate_path_space == "source":
            source_paths = group.get("prim_paths")
        if not isinstance(source_paths, list) or not source_paths:
            raise ValueError("Accepted assignment has no source-space prim paths.")
        for source_path in source_paths:
            if not isinstance(source_path, str) or not source_path.startswith("/"):
                raise ValueError("Accepted assignment has an invalid source prim path.")
            paths = [source_path]
            if not source_stage.GetPrimAtPath(source_path).IsValid():
                paths = sorted(source_aliases.get(source_path, set()))
            if not paths or any(
                not source_stage.GetPrimAtPath(path).IsValid() for path in paths
            ):
                raise ValueError(
                    f"Source prim is absent from the original USD: {source_path}"
                )
            for path in paths:
                previous_name = bindings.setdefault(path, name)
                if previous_name != name:
                    raise ValueError(f"Conflicting materials for source prim: {path}")

    with tempfile.TemporaryDirectory(
        prefix="content-workflow-material-layer-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        predictions = temporary_root / "predictions.jsonl"
        predictions.write_text(
            "".join(
                json.dumps({"id": path, "materials": {"material": name}}) + "\n"
                for path, name in sorted(bindings.items())
            ),
            encoding="utf-8",
        )
        layer_path = temporary_root / "material_layer.usda"
        source_digest = file_sha256(input_usd_path)
        result = ApplyMaterialsToUSDTask().run(
            {
                "input_usd_path": str(input_usd_path),
                "output_usd_path": str(layer_path),
                "predictions_path": str(predictions),
                "resolved_materials": resolved_materials,
                "is_library_based_mapping": True,
                "material_library_path": str(materials_usd_path),
                "layer_only": True,
                "flatten_output": False,
                "apply_allow_empty_predictions": True,
                "apply_fail_on_unknown_material": True,
            }
        )
        if file_sha256(input_usd_path) != source_digest:
            raise RuntimeError("Material layer authoring modified the source USD.")
        statistics = result["assignment_stats"]
        if statistics["failed"] or set(statistics["bound_prim_ids"]) != set(bindings):
            raise RuntimeError(
                "Material layer did not bind every accepted source prim."
            )

        def validate_staged_layer(staged_path: Path) -> None:
            validate_output(staged_path)
            staged_stage = Usd.Stage.Open(str(staged_path))
            if (
                staged_stage is None
                or str(input_usd_path) not in staged_stage.GetRootLayer().subLayerPaths
            ):
                raise RuntimeError(
                    "Published material layer does not sublayer the original USD."
                )
            for path, name in bindings.items():
                prim = staged_stage.GetPrimAtPath(path)
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if (
                    not material
                    or str(material.GetPath()) != result["materials_applied"][name]
                ):
                    raise RuntimeError(
                        f"Published material layer lost binding for {path}."
                    )

        publish_output(layer_path, output_usd_path, validate_staged_layer)
