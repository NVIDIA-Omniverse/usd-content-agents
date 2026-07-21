# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for run-scoped artifact lineage."""

from pathlib import Path
from uuid import uuid4

import pytest

from ...service.artifact_lineage import (
    artifact_is_valid,
    current_artifact_validity,
    emitted_artifacts_for_completed_steps,
    initial_artifact_validity,
    invalidate_artifacts_for_steps,
    revalidate_artifacts_for_completed_steps,
    set_artifact_validity,
)
from ...service.session.manager import SessionManager
from ...service.workers import executor


def test_legacy_route_defaults_and_new_run_defaults_are_distinct() -> None:
    assert all(current_artifact_validity({}).values())
    assert not any(initial_artifact_validity().values())
    assert artifact_is_valid({"restored_predictions_valid": False}, "raw_predictions")
    assert not artifact_is_valid(
        {"restored_predictions_valid": False},
        "restored_predictions",
    )
    with pytest.raises(ValueError, match="Unknown artifact"):
        artifact_is_valid({}, "unknown")


def test_invalidation_and_exact_output_revalidation_fail_closed() -> None:
    current = {"artifact_validity": dict.fromkeys(initial_artifact_validity(), True)}
    invalidated = invalidate_artifacts_for_steps(current, ["predict"])
    assert not invalidated["raw_predictions"]
    assert not invalidated["prediction_report"]
    assert invalidated["cluster_map"]

    metadata = {"artifact_validity": initial_artifact_validity()}
    missing_output = revalidate_artifacts_for_completed_steps(
        metadata,
        ["predict", "apply", "render"],
        {
            "predict": {"predictions_path": None},
            "apply": {"output_usd_path": ""},
            "render": {"flattened_usd_path": None},
        },
    )
    assert not any(missing_output.values())

    exact_output = revalidate_artifacts_for_completed_steps(
        metadata,
        ["predict", "apply", "render"],
        {
            "predict": {"predictions_path": "predictions.jsonl"},
            "apply": {"output_usd_path": "scene.usd"},
            "render": {
                "flattened_usd_path": "flat.usd",
                "rendered_image_paths": ["render.png"],
            },
        },
        verified_artifacts={
            "raw_predictions",
            "applied_output_usd",
            "rendered_output_usd",
        },
    )
    assert exact_output["raw_predictions"]
    assert exact_output["applied_output_usd"]
    assert exact_output["rendered_output_usd"]
    assert not exact_output["final_render"]
    assert not exact_output["prediction_report"]


def test_execution_without_contract_does_not_resurrect_unproduced_groups() -> None:
    validity = revalidate_artifacts_for_completed_steps(
        {},
        ["predict"],
        {
            "predict": {"predictions_path": "new.jsonl"},
            "create_materials": {"predictions_path": "created.jsonl"},
        },
    )
    assert validity["raw_predictions"]
    assert not validity["restored_predictions"]
    assert not validity["applied_output_usd"]
    assert not validity["rendered_output_usd"]
    assert not validity["prediction_report"]

    legacy_restored = revalidate_artifacts_for_completed_steps(
        {"restored_predictions_valid": True},
        [],
        {},
    )
    assert legacy_restored["restored_predictions"]
    assert not legacy_restored["raw_predictions"]


def test_emitted_and_single_group_helpers() -> None:
    emitted = emitted_artifacts_for_completed_steps(
        ["restore_usd", "apply"],
        {
            "restore_usd": {"restored_predictions_path": "restored.jsonl"},
            "apply": {"output_usd_path": "scene.usd"},
        },
    )
    assert emitted == {"restored_predictions", "applied_output_usd"}

    updated = set_artifact_validity({}, "prediction_report", False)
    assert not updated["prediction_report"]
    assert updated["raw_predictions"]
    with pytest.raises(ValueError, match="Unknown artifact"):
        set_artifact_validity({}, "unknown", True)


@pytest.mark.asyncio
async def test_report_publish_checks_lineage_before_publishing_immutable_pointer(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session_id = str(uuid4())
    session_dir = await manager.create_session(session_id)
    canonical = session_dir / "cache" / "predictions" / "prediction_report.html"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("new-R2", encoding="utf-8")
    stale_stage = tmp_path / "stale-R1.html"
    stale_stage.write_text("old-R1", encoding="utf-8")
    await manager.update_session(
        session_id,
        {
            "prediction_lineage_token": "R2",
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
                "prediction_report": True,
            },
        },
    )

    assert not await manager.mark_prediction_report_valid_if_lineage_matches(
        session_id,
        "R1",
        stale_stage,
    )
    assert canonical.read_text(encoding="utf-8") == "new-R2"

    current_stage = tmp_path / "current-R2.html"
    current_stage.write_text("current-R2", encoding="utf-8")
    await manager.update_session(
        session_id,
        {
            "artifact_validity": {
                **initial_artifact_validity(),
                "raw_predictions": True,
            }
        },
    )
    assert await manager.mark_prediction_report_valid_if_lineage_matches(
        session_id,
        "R2",
        current_stage,
    )
    # Mutable canonical bytes are not part of publication and therefore cannot
    # be overwritten by a stale builder after a lineage CAS.
    assert canonical.read_text(encoding="utf-8") == "new-R2"
    metadata = await manager.get_session_metadata(session_id)
    assert metadata["artifact_validity"]["prediction_report"] is True
    publication = metadata["prediction_report_publication"]
    assert publication["prediction_lineage_token"] == "R2"
    assert publication["key"].startswith("reports/R2/")
    assert (
        await manager.read_from_store(session_id, publication["key"]) == b"current-R2"
    )


@pytest.mark.asyncio
async def test_current_outputs_overwrite_store_and_missing_outputs_fail_closed(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    paths = {
        "cache/predictions/predictions.jsonl": b"new-raw",
        "cache/restored/restored_predictions.jsonl": b"new-restored",
        "output/scene_with_materials.usd": b"new-output",
    }
    for key, data in paths.items():
        path = session_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    class _OverwriteStoreManager:
        def __init__(self) -> None:
            self.objects = {key: b"old" for key in paths}

        async def put_file_to_store(
            self,
            session_id: str,
            key: str,
            file_path: str,
        ) -> None:
            self.objects[key] = Path(file_path).read_bytes()

    manager = _OverwriteStoreManager()
    step_results = {
        "predict": {"predictions_path": "new"},
        "restore_usd": {"restored_predictions_path": "new"},
        "apply": {"output_usd_path": "new"},
    }
    promoted = await executor._promote_current_run_artifacts(
        manager,
        "session",
        session_dir,
        ["predict", "restore_usd", "apply"],
        step_results,
    )
    assert promoted == {
        "raw_predictions",
        "restored_predictions",
        "applied_output_usd",
    }
    assert manager.objects == paths

    raw_path = session_dir / "cache" / "predictions" / "predictions.jsonl"
    raw_path.unlink()
    manager.objects["cache/predictions/predictions.jsonl"] = b"old-remote-raw"
    promoted = await executor._promote_current_run_artifacts(
        manager,
        "session",
        session_dir,
        ["predict"],
        {"predict": {"predictions_path": "claimed-but-missing"}},
    )
    assert "raw_predictions" not in promoted
    validity = revalidate_artifacts_for_completed_steps(
        {"artifact_validity": initial_artifact_validity()},
        ["predict"],
        {"predict": {"predictions_path": "claimed-but-missing"}},
        verified_artifacts=promoted,
    )
    assert not validity["raw_predictions"]
    assert manager.objects["cache/predictions/predictions.jsonl"] == b"old-remote-raw"


def test_cached_upstream_completion_is_not_a_current_run_publication() -> None:
    assert executor._current_run_completed_steps(
        {"steps": {"apply": {}}},
        ["predict", "restore_usd", "apply"],
    ) == ["apply"]
