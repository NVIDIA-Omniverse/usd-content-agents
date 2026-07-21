# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused WP2 tests for deterministic plan construction and gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

from texture_agent.cli import app
from texture_agent.config import unified_config
from texture_agent.planning import (
    TextureDetailPolicy,
    TexturePlanDecisionState,
    TexturePlanRequest,
    TexturePlanSource,
    TextureUnitMode,
    build_texture_plan,
)
from texture_agent.tasks.generate_prompts import GeneratePromptsTask
from texture_agent.tasks.plan_textures import (
    PlanTexturesTask,
    TexturePlanRejectedError,
    backend_default_texture_cap,
)
from texture_agent.workflows import factory as workflow_factory


@dataclass
class _Material:
    prim_path: str
    name: str
    bound_prim_paths: list[str] = field(default_factory=list)
    bound_subset_paths: list[str] = field(default_factory=list)
    material_alias_paths: list[str] = field(default_factory=list)


def _materials(count: int) -> list[_Material]:
    return [
        _Material(
            prim_path=f"/World/Looks/Material_{index:03d}",
            name=f"Material_{index:03d}",
            bound_prim_paths=[f"/World/Geometry/Mesh_{index:03d}"],
        )
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("count", "backend_cap", "override", "state", "allowed"),
    [
        (16, 16, None, TexturePlanDecisionState.READY, True),
        (17, 16, None, TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE, False),
        (32, 32, None, TexturePlanDecisionState.READY, True),
        (33, 32, None, TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE, False),
        (64, 32, 64, TexturePlanDecisionState.READY, True),
        (65, 32, None, TexturePlanDecisionState.UNSUPPORTED, False),
    ],
)
def test_planner_enforces_contract_boundaries(
    count: int,
    backend_cap: int,
    override: int | None,
    state: TexturePlanDecisionState,
    allowed: bool,
) -> None:
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset="scene.usd"),
        backend_default_cap=backend_cap,
        operator_override_cap=override,
    )

    plan = build_texture_plan(
        request,
        discovered_materials=_materials(count),
        auto_prompt_enabled=True,
    )

    assert plan.counts.authored_material_count == count
    assert plan.counts.effective_bound_material_count == count
    assert plan.counts.selected_unit_count == count
    assert plan.counts.planned_generation_job_count == count
    assert plan.decision.state is state
    assert plan.decision.execution_allowed is allowed
    if count == 65:
        assert plan.decision.consolidation_required is True
        assert plan.decision.explicit_narrowing_required is True
        assert "Consolidate" in plan.decision.recommended_actions[0]


def test_planner_is_path_deterministic_with_duplicate_display_names() -> None:
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset="scene.usd"),
    )
    materials = [
        _Material("/World/B/Looks/Paint", "Paint", ["/World/B/Mesh"]),
        _Material("/World/A/Looks/Paint", "Paint", ["/World/A/Mesh"]),
    ]

    plan = build_texture_plan(
        request,
        discovered_materials=materials,
        material_textures={"Paint": {"prompt": "paint"}},
    )
    reversed_plan = build_texture_plan(
        request,
        discovered_materials=list(reversed(materials)),
        material_textures={"Paint": {"prompt": "paint"}},
    )

    assert [unit.unit_id for unit in plan.selected_units] == [
        unit.unit_id for unit in reversed_plan.selected_units
    ]
    assert plan.counts.selected_material_count == 2
    assert len({unit.unit_id for unit in plan.selected_units}) == 2


def test_strict_material_scope_records_skips() -> None:
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset="scene.usd"),
    )
    plan = build_texture_plan(
        request,
        discovered_materials=_materials(3),
        material_textures={"Material_001": {"prompt": "brushed steel"}},
        auto_prompt_enabled=False,
    )

    assert plan.counts.selected_unit_count == 1
    assert plan.counts.skipped_item_count == 2
    assert {item.reason_code for item in plan.skipped_items} == {"not_requested"}


def test_planner_consumes_typed_effective_discovery_contract() -> None:
    from texture_agent.functions.material_discovery import (
        EffectiveMaterialDiscovery,
        MaterialDiscoverySkip,
        MaterialInfo,
    )

    effective_material = MaterialInfo(
        prim_path="/World/Looks/Paint",
        name="Paint",
        material_alias_paths=[
            "/World/InstanceA/Looks/Paint",
            "/World/InstanceB/Looks/Paint",
        ],
        bound_prim_paths=["/World/Mesh"],
        bound_subset_paths=["/World/Mesh/PaintedFaces"],
    )
    unused_material = MaterialInfo(
        prim_path="/World/Looks/Unused",
        name="Unused",
    )
    discovery = EffectiveMaterialDiscovery(
        authored_materials=(effective_material, unused_material),
        effective_materials=(effective_material,),
        renderable_prim_paths=("/World/Mesh",),
        renderable_subset_paths=("/World/Mesh/PaintedFaces",),
        skipped_materials=(
            MaterialDiscoverySkip(
                material_prim_path="/World/Looks/Unused",
                material_name="Unused",
                reason_code="not_effectively_bound",
                reason="No renderable scene member uses this material.",
            ),
            MaterialDiscoverySkip(
                material_prim_path="relative/bad",
                material_name="Bad",
                reason_code="ignored",
                reason="Relative paths are ignored by the plan skip adapter.",
            ),
        ),
    )
    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset="scene.usd"),
    )

    plan = build_texture_plan(
        request,
        discovered_materials=discovery.authored_materials,
        effective_discovery=discovery,
        auto_prompt_enabled=True,
    )

    assert plan.counts.authored_material_count == 2
    assert plan.counts.renderable_prim_count == 1
    assert plan.counts.renderable_subset_count == 1
    assert plan.counts.effective_bound_material_count == 1
    assert plan.selected_units[0].member_subset_paths == ("/World/Mesh/PaintedFaces",)
    assert plan.skipped_items[0].reason_code == "not_effectively_bound"
    assert len(plan.skipped_items) == 1

    explicit_plan = build_texture_plan(
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            discovery_mode="explicit",
            explicit_material_paths=("/World/InstanceB/Looks/Paint",),
        ),
        discovered_materials=discovery.authored_materials,
        effective_discovery=discovery,
        auto_prompt_enabled=True,
    )
    assert explicit_plan.counts.selected_unit_count == 1
    assert explicit_plan.selected_units[0].material_prim_paths == (
        "/World/Looks/Paint",
    )


def test_planner_covers_explicit_unit_modes_and_scoped_prompt_policy() -> None:
    from texture_agent.functions.material_discovery import (
        EffectiveMaterialDiscovery,
        MaterialDiscoverySkip,
    )

    material = _Material(
        prim_path="/World/Looks/Paint",
        name="Paint",
        bound_prim_paths=["/World/Mesh"],
        bound_subset_paths=["/World/Mesh/Subset"],
    )
    material.material_alias_paths = ["/World/AltLooks/Paint"]  # type: ignore[attr-defined]
    unbound = _Material(
        prim_path="/World/Looks/Unbound",
        name="Unbound",
    )
    discovery = EffectiveMaterialDiscovery(
        authored_materials=(material, unbound),
        effective_materials=(material,),
        renderable_prim_paths=("/World/Mesh",),
        renderable_subset_paths=("/World/Mesh/Subset",),
        skipped_materials=(
            MaterialDiscoverySkip(
                material_prim_path="not-an-absolute-path",
                material_name="BadSkip",
                reason_code="ignored",
                reason="Invalid skip path should be ignored.",
            ),
        ),
    )

    per_prim = build_texture_plan(
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            discovery_mode="all_authored",
            unit_mode="per_prim",
        ),
        discovered_materials=discovery.authored_materials,
        effective_discovery=discovery,
        material_textures={
            "/World/AltLooks/Paint": {
                "detail_policy": "default",
                "per_prim": {"Mesh": {"detail_policy": "surface_only"}},
            },
            "Unbound": {"material_path": "/World/Looks/Other"},
        },
        auto_prompt_enabled=True,
    )

    assert {unit.selection_reason_code for unit in per_prim.selected_units} == {
        "all_authored"
    }
    by_member = {
        unit.member_prim_paths or unit.member_subset_paths: unit
        for unit in per_prim.selected_units
    }
    assert by_member[("/World/Mesh",)].detail_policy == (
        TextureDetailPolicy.SURFACE_ONLY
    )
    assert by_member[("/World/Mesh/Subset",)].member_subset_paths == (
        "/World/Mesh/Subset",
    )
    assert per_prim.decision.state is TexturePlanDecisionState.UNSUPPORTED
    assert per_prim.decision.execution_allowed is False
    assert per_prim.skipped_items[0].reason_code == "no_renderable_member"
    assert len(per_prim.skipped_items) == 1

    per_group = build_texture_plan(
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            unit_mode="per_group",
        ),
        discovered_materials=(material,),
        auto_prompt_enabled=True,
    )

    assert per_group.selected_units[0].unit_mode is TextureUnitMode.PER_GROUP
    assert per_group.selected_units[0].group_key == "/World/Looks/Paint"
    assert per_group.decision.state is TexturePlanDecisionState.UNSUPPORTED
    assert per_group.decision.execution_allowed is False


def test_planner_ignores_non_sequence_member_path_shapes() -> None:
    class _MalformedMaterial:
        prim_path = "/World/Looks/Paint"
        name = "Paint"
        bound_prim_paths = "/World/Mesh"
        bound_subset_paths = b"/World/Mesh/Subset"

    plan = build_texture_plan(
        TexturePlanRequest(source=TexturePlanSource(source_asset="scene.usd")),
        discovered_materials=(_MalformedMaterial(),),
        auto_prompt_enabled=True,
    )

    assert plan.selected_units == ()
    assert plan.skipped_items[0].reason_code == "outside_discovery_scope"
    assert plan.decision.state is TexturePlanDecisionState.REQUIRES_NARROWING
    assert plan.decision.execution_allowed is False


def test_generate_prompts_only_expands_plan_selected_materials(tmp_path: Path) -> None:
    materials = _materials(3)
    plan = build_texture_plan(
        TexturePlanRequest(source=TexturePlanSource(source_asset="scene.usd")),
        discovered_materials=materials,
        material_textures={"Material_001": {"prompt": "brushed steel"}},
        auto_prompt_enabled=False,
    )

    context = GeneratePromptsTask().run(
        {
            "discovered_materials": materials,
            "material_textures": {"Material_001": {"prompt": "brushed steel"}},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_material"},
            "texture_plan": plan,
            "working_dir": str(tmp_path),
        }
    )

    assert len(context["discovered_materials"]) == 3
    assert [material.name for material in context["texture_plan_scoped_materials"]] == [
        "Material_001"
    ]
    assert [unit.material_info.prim_path for unit in context["prim_texture_units"]] == [
        "/World/Looks/Material_001"
    ]

    no_spec = GeneratePromptsTask().run(
        {
            "discovered_materials": materials,
            "material_textures": {},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_material"},
            "texture_plan": plan,
            "working_dir": str(tmp_path / "no-spec"),
        }
    )
    assert no_spec["prim_texture_units"] == []


def test_generate_prompts_scopes_path_spec_and_per_prim_members(
    tmp_path: Path,
) -> None:
    material = _Material(
        prim_path="/World/Looks/Paint",
        name="Paint",
        bound_prim_paths=["/World/MeshA", "/World/MeshB"],
    )
    material.material_alias_paths = ["/World/Alias/Paint"]
    plan = build_texture_plan(
        TexturePlanRequest(
            source=TexturePlanSource(source_asset="scene.usd"),
            discovery_mode="explicit",
            unit_mode="per_prim",
            explicit_prim_paths=("/World/MeshB",),
        ),
        discovered_materials=(material,),
        auto_prompt_enabled=False,
        material_textures={"/World/Alias/Paint": {"prompt": "brushed paint"}},
    )

    context = GeneratePromptsTask().run(
        {
            "discovered_materials": [material],
            "material_textures": {"/World/Alias/Paint": {"prompt": "brushed paint"}},
            "auto_prompt_config": {"enabled": False},
            "texture_config": {"mode": "per_prim"},
            "texture_plan": plan,
            "working_dir": str(tmp_path),
        }
    )

    assert context["material_textures"] == {
        "/World/Alias/Paint": {"prompt": "brushed paint"}
    }
    assert context["texture_plan_scoped_materials"][0].bound_prim_paths == [
        "/World/MeshB"
    ]
    assert [unit.prim_path for unit in context["prim_texture_units"]] == [
        "/World/MeshB"
    ]
    assert [unit.key for unit in context["prim_texture_units"]] == [
        "World_Alias_Paint__MeshB"
    ]


def test_rejected_workflow_persists_plan_before_any_backend_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from texture_agent.workflows import factory

    context = {
        "usd_path": str(tmp_path / "scene.usd"),
        "working_dir": str(tmp_path / "run"),
        "texture_config": {"backend": "simple_image_gen", "size": 1024},
        "planning_config": {},
        "material_textures": {},
        "auto_prompt_config": {"enabled": True},
        "steps": {"generate_textures": {"max_workers": 4}},
        "config": {"input": {}},
    }
    backend_calls: list[str] = []

    class _Discover:
        name = "Discover"
        description = "fixture discovery"

        def run(self, current: dict) -> dict:
            current["discovered_materials"] = _materials(33)
            return current

    class _Backend:
        name = "Backend"
        description = "must not run"

        def run(self, current: dict) -> dict:  # pragma: no cover - gate proof
            backend_calls.append("called")
            return current

    monkeypatch.setattr(
        factory,
        "STEP_ORDER",
        [
            "discover_materials",
            "plan_textures",
            "generate_prompts",
            "generate_textures",
        ],
    )
    monkeypatch.setattr(
        factory,
        "_STEP_TASKS",
        {
            "discover_materials": _Discover,
            "plan_textures": PlanTexturesTask,
            "generate_prompts": _Backend,
            "generate_textures": _Backend,
        },
    )

    with pytest.raises(TexturePlanRejectedError):
        factory.run_pipeline(context)

    assert backend_calls == []
    assert Path(context["texture_plan_path"]).is_file()
    assert context["texture_plan"].decision.state == (
        TexturePlanDecisionState.REQUIRES_OPERATOR_OVERRIDE
    )


def test_plan_only_keeps_rejected_plan_inspectable(tmp_path: Path) -> None:
    context = {
        "usd_path": str(tmp_path / "scene.usd"),
        "working_dir": str(tmp_path / "run"),
        "texture_config": {"backend": "simple_image_gen", "size": 1024},
        "planning_config": {"plan_only": True},
        "material_textures": {},
        "auto_prompt_config": {"enabled": True},
        "discovered_materials": _materials(65),
        "steps": {"generate_textures": {"max_workers": 4}},
        "config": {"input": {}},
    }

    result = PlanTexturesTask().run(context)

    assert result["texture_plan"].decision.state is TexturePlanDecisionState.UNSUPPORTED
    assert result["texture_plan"].decision.execution_allowed is False


def test_plan_task_maps_input_prim_paths_to_explicit_prim_scope(
    tmp_path: Path,
) -> None:
    context = {
        "usd_path": str(tmp_path / "scene.usd"),
        "working_dir": str(tmp_path / "run"),
        "texture_config": {"backend": "simple_image_gen", "size": 1024},
        "planning_config": {"plan_only": True},
        "material_textures": {"Material_001": {"prompt": "brushed steel"}},
        "auto_prompt_config": {"enabled": False},
        "discovered_materials": _materials(3),
        "steps": {"generate_textures": {"max_workers": 4}},
        "config": {"input": {"prim_paths": ["/World/Geometry/Mesh_001"]}},
    }

    result = PlanTexturesTask().run(context)
    plan = result["texture_plan"]

    assert plan.request.explicit_material_paths == ()
    assert plan.request.explicit_prim_paths == ("/World/Geometry/Mesh_001",)
    assert [unit.display_name for unit in plan.selected_units] == ["Material_001"]


def test_manifest_planning_section_records_operator_override(tmp_path: Path) -> None:
    from texture_agent.functions.artifact_manifest import _planning_section

    request = TexturePlanRequest(
        source=TexturePlanSource(source_asset="scene.usd"),
        operator_override_cap=64,
    )
    plan = build_texture_plan(
        request,
        discovered_materials=_materials(33),
        auto_prompt_enabled=True,
    )
    plan_path = tmp_path / "texture_plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

    section = _planning_section(
        {"texture_plan": plan, "texture_plan_path": str(plan_path)},
        tmp_path,
    )

    assert section["execution_allowed"] is True
    assert section["counts"]["planned_generation_job_count"] == 33
    assert section["limits"]["operator_override_cap"] == 64
    assert section["limits"]["hard_cap"] == 64


def test_backend_defaults_distinguish_global_and_uv_aware_paths() -> None:
    assert backend_default_texture_cap({"backend": "simple_image_gen"}) == 32
    assert backend_default_texture_cap({"backend": "service"}) == 16
    assert (
        backend_default_texture_cap(
            {"backend": "service", "engine": "simple_image_gen"}
        )
        == 32
    )
    assert backend_default_texture_cap({"backend": "step1x"}) == 16
    assert (
        backend_default_texture_cap({"backend": "custom", "engine": "step1x-v1"}) == 16
    )
    assert (
        backend_default_texture_cap(
            {"backend": "custom", "uv_policy": "force_projection"}
        )
        == 16
    )


def test_plan_cli_builds_plan_only_context_and_prints_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}
    plan = build_texture_plan(
        TexturePlanRequest(source=TexturePlanSource(source_asset="scene.usd")),
        discovered_materials=_materials(1),
        auto_prompt_enabled=True,
    )

    monkeypatch.setattr(
        unified_config,
        "load_config",
        lambda path: {"texture": {}, "planning": {}},
    )

    def _context(config: dict) -> dict:
        captured["config"] = config
        return {}

    monkeypatch.setattr(unified_config, "config_to_context", _context)
    monkeypatch.setattr(
        workflow_factory,
        "run_pipeline",
        lambda context, **kwargs: {
            "texture_plan": plan,
            "texture_plan_path": str(tmp_path / "texture_plan.json"),
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "plan",
            str(config_path),
            "--discovery-mode",
            "explicit",
            "--material-paths",
            "/World/Looks/Material_000",
            "--operator-override-cap",
            "64",
        ],
    )

    assert result.exit_code == 0
    assert "Decision: ready" in result.stdout
    planning = captured["config"]["planning"]  # type: ignore[index]
    assert planning["plan_only"] is True
    assert planning["discovery_mode"] == "explicit"
    assert planning["explicit_material_paths"] == ["/World/Looks/Material_000"]
    assert planning["operator_override_cap"] == 64


def test_plan_cli_prints_reasons_and_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("input: {}\n", encoding="utf-8")
    plan = build_texture_plan(
        TexturePlanRequest(source=TexturePlanSource(source_asset="scene.usd")),
        discovered_materials=_materials(33),
        auto_prompt_enabled=True,
    )

    monkeypatch.setattr(
        unified_config,
        "load_config",
        lambda path: {"texture": {}, "planning": {}},
    )
    monkeypatch.setattr(unified_config, "config_to_context", lambda config: {})
    monkeypatch.setattr(
        workflow_factory,
        "run_pipeline",
        lambda context, **kwargs: {
            "texture_plan": plan,
            "texture_plan_path": str(tmp_path / "texture_plan.json"),
        },
    )

    result = CliRunner().invoke(app, ["plan", str(config_path)])

    assert result.exit_code == 0
    assert "Reason:" in result.stdout
    assert "Action:" in result.stdout
