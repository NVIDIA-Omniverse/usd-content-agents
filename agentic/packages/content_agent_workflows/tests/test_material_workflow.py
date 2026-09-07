# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.material_assignment.workflow import (
    MaterialGenerationRequest,
    MaterialVqaGoal,
    execute_material_generation,
    partition_material_assignments,
)


def _group(name: str, paths: list[str]) -> dict[str, Any]:
    return {
        "material_name": name,
        "material_path": f"/Looks/{name}",
        "prim_paths": paths,
        "runtime_prim_paths": [f"/Runtime{path}" for path in paths],
    }


def test_partition_material_assignments_bounds_groups_and_targets() -> None:
    groups = [_group(f"m{index}", [f"/P{index}"]) for index in range(257)]
    groups.append(_group("large", [f"/Large/P{index}" for index in range(4097)]))

    batches = partition_material_assignments(groups, path_key="prim_paths")

    assert len(batches) == 3
    assert [len(batch) for batch in batches] == [256, 2, 1]
    assert all(len(batch) <= 256 for batch in batches)
    assert all(
        sum(len(group["prim_paths"]) for group in batch) <= 4096 for batch in batches
    )
    assert [
        path for batch in batches for group in batch for path in group["prim_paths"]
    ] == [path for group in groups for path in group["prim_paths"]]


def test_material_vqa_goal_reuses_verified_review_and_loops_surgically() -> None:
    goal = MaterialVqaGoal(
        max_iterations=3,
        delegated_review_required=True,
        verified_scene_session=True,
    )

    verified = goal.evaluate_initial(
        candidate_count=4,
        satisfied=True,
        systematic_unfixable=False,
        reviewed_render_signature="same",
        current_render_signature="same",
    )
    repair = goal.evaluate_initial(
        candidate_count=4,
        satisfied=False,
        systematic_unfixable=False,
        reviewed_render_signature=None,
        current_render_signature="current",
    )
    review = goal.evaluate_initial(
        candidate_count=4,
        satisfied=True,
        systematic_unfixable=False,
        reviewed_render_signature=None,
        current_render_signature="current",
    )
    continue_after_change = goal.evaluate_iteration(
        iteration=2,
        returncode=0,
        satisfied=True,
        systematic_unfixable=False,
        previous_signature="before",
        next_signature="pass",
        delegated_review_covers_final_renders=False,
    )
    final = goal.evaluate_iteration(
        iteration=3,
        returncode=0,
        satisfied=True,
        systematic_unfixable=False,
        previous_signature="pass",
        next_signature="pass",
        delegated_review_covers_final_renders=True,
    )

    assert verified.action == "succeed"
    assert repair.action == "continue"
    assert repair.status == "repair_required"
    assert review.status == "delegated_vision_review_required"
    assert continue_after_change.action == "continue"
    assert final.action == "succeed"
    assert tuple(goal.repair_iterations) == (2, 3)


class _FakeSceneSession:
    def __init__(
        self,
        *,
        fail_apply_number: int | None = None,
        fail_checkpoint_load: bool = False,
    ) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.opened: list[Path] = []
        self.apply_count = 0
        self.fail_apply_number = fail_apply_number
        self.fail_checkpoint_load = fail_checkpoint_load

    def open(self, scene: Path, *, force_reload: bool = False) -> dict[str, Any]:
        assert force_reload is True
        self.opened.append(Path(scene))
        return {"ok": True}

    def run_json(
        self,
        args: list[str],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        assert timeout_seconds > 0
        self.calls.append(tuple(args))
        if args[:2] == ["appearance", "clear"]:
            return {
                "schema_version": "1",
                "ok": True,
                "command": "appearance.clear",
                "summary": {"clear": True},
                "data": {
                    "audit": {
                        "clear": True,
                        "overlay_active": True,
                        "counts": {"material_bindings": 0},
                    }
                },
            }
        if args[0] == "material-apply":
            self.apply_count += 1
            if self.apply_count == self.fail_apply_number:
                return {"ok": False}
            plan = json.loads(Path(args[1]).read_text())
            assert len(plan["material_assignments"]) <= 256
            assert (
                sum(
                    len(group[args[args.index("--path-key") + 1]])
                    for group in plan["material_assignments"]
                )
                <= 4096
            )
            return {"schema_version": "1", "ok": True, "command": "material.apply"}
        if args[:2] == ["checkpoint", "load"] and self.fail_checkpoint_load:
            return {"ok": False}
        if args[0] == "save":
            Path(args[1]).write_text("#usda 1.0\n")
            return {"schema_version": "1", "ok": True, "command": "save"}
        if args[0] == "render" and "--seg" in args:
            output = Path(args[args.index("--output") + 1])
            return {
                "schema_version": "1",
                "ok": True,
                "command": "render",
                "summary": {"renderer": "ovrtx"},
                "data": {
                    "results": [
                        {"path": str(output / "beauty_0.png")},
                        {"path": str(output / "beauty_1.png")},
                    ]
                },
                "artifacts": [
                    {"label": "segmentation:0", "path": str(output / "seg_0.png")},
                    {"label": "segmentation:1", "path": str(output / "seg_1.png")},
                ],
            }
        return {"schema_version": "1", "ok": True, "command": args[0]}


def _generation_request(
    tmp_path: Path,
    *,
    groups: list[dict[str, Any]],
    respect_existing: bool,
    inspection: bool,
    stage_gprim_count: int = 0,
) -> MaterialGenerationRequest:
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "source.usda"
    library = raw / "library.usda"
    inspection_path = raw / "inspection.usda"
    source.write_text("#usda 1.0\n")
    library.write_text("#usda 1.0\n")
    inspection_path.write_text("#usda 1.0\n")
    patch = {
        "schema_version": "content-agents.material-decision-patch.v1",
        "candidate_count": sum(len(group["prim_paths"]) for group in groups),
        "material_assignments": groups,
        "reviewed_no_override": [],
    }
    patch_path = raw / "material_decision_patch.json"
    patch_path.write_text(json.dumps(patch))
    return MaterialGenerationRequest(
        run_dir=tmp_path,
        staged_source=source,
        staged_library=library,
        decision_patch_path=patch_path,
        output_path=tmp_path / "output" / "materialized.usda",
        final_render_dir=tmp_path / "final_renders",
        decision_patch=patch,
        iteration=1,
        timeout_seconds=5,
        respect_existing_material_bindings=respect_existing,
        source_focus_prim_path="/Source",
        inspection_focus_prim_path="/Runtime",
        inspection_usd_path=inspection_path if inspection else None,
        inspection_usd_sha256=(file_sha256(inspection_path) if inspection else None),
        stage_gprim_count=stage_gprim_count,
    )


def test_execute_material_generation_uses_domain_stage_and_complete_vqa_set(
    tmp_path: Path,
) -> None:
    session = _FakeSceneSession()
    request = _generation_request(
        tmp_path,
        groups=[_group("metal", ["/Part"])],
        respect_existing=False,
        inspection=True,
    )

    result = execute_material_generation(session=session, request=request)

    assert session.opened[0] == request.inspection_usd_path
    assert session.opened[-1] == request.output_path
    apply_call = next(call for call in session.calls if call[0] == "material-apply")
    assert apply_call[apply_call.index("--path-key") + 1] == "runtime_prim_paths"
    assert result.payload["path_key"] == "runtime_prim_paths"
    assert result.payload["responses"]["verification_render"] is not None
    assert result.payload["responses"]["turntable_render"] is not None
    assert result.payload["output_usd"]["sha256"] == file_sha256(request.output_path)


def test_execute_material_generation_rejects_filtered_empty_scope_before_open(
    tmp_path: Path,
) -> None:
    session = _FakeSceneSession()
    request = _generation_request(
        tmp_path,
        groups=[],
        respect_existing=False,
        inspection=True,
        stage_gprim_count=1,
    )

    with pytest.raises(ValueError, match="filters excluded all stage geometry"):
        execute_material_generation(session=session, request=request)

    assert session.opened == []
    assert session.calls == []
    assert not request.output_path.exists()


@pytest.mark.parametrize("fail_checkpoint_load", [False, True])
def test_execute_material_generation_restores_after_batch_failure(
    tmp_path: Path, fail_checkpoint_load: bool
) -> None:
    session = _FakeSceneSession(
        fail_apply_number=2,
        fail_checkpoint_load=fail_checkpoint_load,
    )
    request = _generation_request(
        tmp_path,
        groups=[_group("metal", [f"/Part{index}" for index in range(4097)])],
        respect_existing=True,
        inspection=False,
    )

    with pytest.raises(ValueError, match="batch 2 failed"):
        execute_material_generation(session=session, request=request)

    assert session.apply_count == 2
    assert any(call[:2] == ("checkpoint", "load") for call in session.calls)
    assert session.opened.count(request.staged_source) == (
        2 if fail_checkpoint_load else 1
    )
    assert not request.output_path.exists()
