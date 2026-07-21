# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-scoped validity for artifacts that may outlive a regeneration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactLineageRule:
    """Steps that invalidate an artifact and outputs that revalidate it."""

    invalidated_by: frozenset[str]
    emitted_outputs: Mapping[str, tuple[str, ...]]


_PREDICTION_STEPS = frozenset(
    {
        "predict",
        "benchmark",
        "expand_cluster_predictions",
        "validate_predictions",
        "harmonize_predictions",
    }
)

ARTIFACT_LINEAGE: dict[str, ArtifactLineageRule] = {
    "raw_predictions": ArtifactLineageRule(
        invalidated_by=_PREDICTION_STEPS,
        emitted_outputs={step: ("predictions_path",) for step in _PREDICTION_STEPS},
    ),
    # Report HTML is produced on demand by the service, not by the prediction step.
    "prediction_report": ArtifactLineageRule(
        invalidated_by=_PREDICTION_STEPS,
        emitted_outputs={},
    ),
    "restored_predictions": ArtifactLineageRule(
        invalidated_by=frozenset({"restore_usd"}),
        emitted_outputs={"restore_usd": ("restored_predictions_path",)},
    ),
    "applied_output_usd": ArtifactLineageRule(
        invalidated_by=frozenset({"apply", "refine"}),
        emitted_outputs={
            "apply": ("output_usd_path",),
            "refine": ("output_usd_path", "final_output_path"),
        },
    ),
    "rendered_output_usd": ArtifactLineageRule(
        invalidated_by=frozenset({"render"}),
        emitted_outputs={"render": ("flattened_usd_path",)},
    ),
    "final_render": ArtifactLineageRule(
        invalidated_by=frozenset({"render"}),
        emitted_outputs={"render": ("rendered_image_path", "rendered_image_paths")},
    ),
    "cluster_map": ArtifactLineageRule(
        invalidated_by=frozenset({"cluster_prims"}),
        emitted_outputs={"cluster_prims": ("cluster_map_path",)},
    ),
    "cluster_report": ArtifactLineageRule(
        invalidated_by=frozenset({"cluster_prims"}),
        emitted_outputs={"cluster_prims": ("cluster_report_path",)},
    ),
    "cluster_summary": ArtifactLineageRule(
        invalidated_by=frozenset({"cluster_prims"}),
        emitted_outputs={"cluster_prims": ("cluster_summary_path",)},
    ),
    "cluster_representatives": ArtifactLineageRule(
        invalidated_by=frozenset({"cluster_prims"}),
        emitted_outputs={"cluster_prims": ("dataset_representatives_path",)},
    ),
    "previews": ArtifactLineageRule(
        invalidated_by=frozenset({"build_dataset_usd"}),
        emitted_outputs={"build_dataset_usd": ("num_images",)},
    ),
}

ARTIFACT_PRODUCER_STEPS: dict[str, frozenset[str]] = {
    artifact: rule.invalidated_by for artifact, rule in ARTIFACT_LINEAGE.items()
}

ARTIFACT_CANONICAL_KEYS: dict[str, tuple[str, ...]] = {
    "raw_predictions": ("cache/predictions/predictions.jsonl",),
    "prediction_report": ("cache/predictions/prediction_report.html",),
    "restored_predictions": ("cache/restored/restored_predictions.jsonl",),
    "applied_output_usd": ("output/scene_with_materials.usd",),
    "rendered_output_usd": (
        "output/scene_with_materials_flat.usd",
        "output/composed_scene_flat.usd",
    ),
    "final_render": ("output/scene_with_materials.png",),
    "cluster_map": ("cache/clusters/cluster_map.jsonl",),
    "cluster_report": ("cache/clusters/cluster_report.html",),
    "cluster_summary": ("cache/clusters/cluster_summary.json",),
    "cluster_representatives": ("cache/clusters/dataset_representatives.jsonl",),
}


def initial_artifact_validity() -> dict[str, bool]:
    """Return the validity contract for a run that has emitted no artifacts."""
    return dict.fromkeys(ARTIFACT_LINEAGE, False)


def artifact_is_valid(metadata: Mapping[str, Any] | None, artifact: str) -> bool:
    """Return whether an existing artifact belongs to the active run lineage."""
    if artifact not in ARTIFACT_LINEAGE:
        raise ValueError(f"Unknown artifact lineage group: {artifact}")
    if not metadata:
        return True
    validity = metadata.get("artifact_validity")
    if isinstance(validity, Mapping) and artifact in validity:
        return bool(validity[artifact])
    if artifact == "restored_predictions" and "restored_predictions_valid" in metadata:
        return bool(metadata["restored_predictions_valid"])
    return True


def current_artifact_validity(
    metadata: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Materialize the full validity contract, including legacy defaults."""
    return {
        artifact: artifact_is_valid(metadata, artifact) for artifact in ARTIFACT_LINEAGE
    }


def invalidate_artifacts_for_steps(
    metadata: Mapping[str, Any] | None,
    invalidated_steps: Iterable[str],
) -> dict[str, bool]:
    """Invalidate artifacts whose producers will be rerun or discarded."""
    invalidated = set(invalidated_steps)
    validity = current_artifact_validity(metadata)
    for artifact, rule in ARTIFACT_LINEAGE.items():
        if rule.invalidated_by & invalidated:
            validity[artifact] = False
    return validity


def revalidate_artifacts_for_completed_steps(
    metadata: Mapping[str, Any] | None,
    completed_steps: Iterable[str],
    step_results: Mapping[str, Any] | None,
    *,
    verified_artifacts: set[str] | None = None,
) -> dict[str, bool]:
    """Revalidate artifacts backed by outputs emitted in this run."""
    completed = set(completed_steps)
    results = step_results if isinstance(step_results, Mapping) else {}
    stored_validity = metadata.get("artifact_validity") if metadata else None
    if isinstance(stored_validity, Mapping):
        validity = current_artifact_validity(metadata)
    else:
        # Route reads retain legacy default-true compatibility, but execution
        # must never turn missing lineage metadata into evidence that every
        # artifact was produced by the active run.
        validity = initial_artifact_validity()
        if metadata and "restored_predictions_valid" in metadata:
            validity["restored_predictions"] = bool(
                metadata["restored_predictions_valid"]
            )
    for artifact, rule in ARTIFACT_LINEAGE.items():
        if verified_artifacts is not None and artifact not in verified_artifacts:
            continue
        for step, output_keys in rule.emitted_outputs.items():
            outputs = results.get(step)
            if (
                step in completed
                and isinstance(outputs, Mapping)
                and any(outputs.get(key) for key in output_keys)
            ):
                validity[artifact] = True
                break
    return validity


def emitted_artifacts_for_completed_steps(
    completed_steps: Iterable[str],
    step_results: Mapping[str, Any] | None,
) -> set[str]:
    """Return artifacts for which this execution emitted explicit evidence."""
    completed = set(completed_steps)
    results = step_results if isinstance(step_results, Mapping) else {}
    emitted: set[str] = set()
    for artifact, rule in ARTIFACT_LINEAGE.items():
        for step, output_keys in rule.emitted_outputs.items():
            outputs = results.get(step)
            if (
                step in completed
                and isinstance(outputs, Mapping)
                and any(outputs.get(key) for key in output_keys)
            ):
                emitted.add(artifact)
                break
    return emitted


def set_artifact_validity(
    metadata: Mapping[str, Any] | None,
    artifact: str,
    valid: bool,
) -> dict[str, bool]:
    """Return a full validity contract with one artifact updated."""
    if artifact not in ARTIFACT_LINEAGE:
        raise ValueError(f"Unknown artifact lineage group: {artifact}")
    validity = current_artifact_validity(metadata)
    validity[artifact] = valid
    return validity
