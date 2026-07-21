# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WP1 tests for material assignment/create/modify decision policy."""

from __future__ import annotations

import pytest
from world_understanding.utils.object_store import InMemoryObjectStore

from material_agent.material_library_generation.creation_contract import MaterialAction
from material_agent.materials import FALLBACK_MATERIAL_NAME, UNKNOWN_MATERIAL_SENTINEL
from material_agent.tasks.material_creation_policy import (
    MaterialCreationIntent,
    MaterialDecision,
    MaterialDecisionPlan,
    MaterialDecisionPolicyTask,
    MaterialPolicyConflict,
    plan_material_actions,
)


def _match(path: str = "/World/Looks/Existing") -> list[dict[str, object]]:
    return [
        {
            "source_path": path,
            "s3_path": None,
            "dependencies": [],
            "metadata": {"source": "test"},
        }
    ]


def test_adequate_existing_matches_never_create() -> None:
    plan = plan_material_actions(
        [
            {"id": "/World/Geom/Panel", "materials": {"material": "Steel"}},
            {"id": "/World/Geom/Tire", "vlm_response": {"material": "Rubber"}},
        ],
        matched_materials={
            "Steel": _match("/World/Looks/Steel"),
            "Rubber": _match("/World/Looks/Rubber"),
        },
    )

    assert [decision.action for decision in plan.decisions] == [
        MaterialAction.ASSIGN_EXISTING,
        MaterialAction.ASSIGN_EXISTING,
    ]
    assert plan.creation_intents == ()
    assert plan.stats == {
        "assign_existing": 2,
        "create_new": 0,
        "modify_existing": 0,
        "creation_intents": 0,
        "conflicts": 0,
    }


def test_explicit_creation_is_honored_even_when_match_exists() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/Handle",
                "materials": {
                    "material": "Blue textured plastic",
                    "action": "create_new",
                    "appearance_prompt": "blue molded plastic with fine pebble texture",
                    "description": "custom handle grip material",
                    "color": "blue",
                    "finish": "matte",
                    "pbr_hints": {"roughness": 0.82, "metallic": 0.0},
                },
            }
        ],
        matched_materials={"Blue textured plastic": _match()},
    )

    assert len(plan.decisions) == 1
    decision = plan.decisions[0]
    assert decision.action is MaterialAction.CREATE_NEW
    assert decision.matched_existing is True
    assert decision.explicit_action is MaterialAction.CREATE_NEW
    assert decision.creation_intent_id == plan.creation_intents[0].intent_id

    intent = plan.creation_intents[0]
    assert intent.explicit is True
    assert intent.target_prim_paths == ("/World/Geom/Handle",)
    assert intent.recipe.name == "Blue textured plastic"
    assert intent.recipe.appearance_prompt == (
        "blue molded plastic with fine pebble texture"
    )
    assert intent.recipe.finish == "matte"


def test_repeated_compatible_unresolved_requirements_reuse_one_creation_intent() -> (
    None
):
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/LeftGrip",
                "materials": {
                    "material": "Matte black rubber",
                    "appearance_prompt": "matte black rubber with subtle tread grain",
                    "description": "soft black rubber grip",
                },
            },
            {
                "id": "/World/Geom/RightGrip",
                "materials": {
                    "material": "Matte black rubber",
                    "appearance_prompt": "matte black rubber with subtle tread grain",
                    "description": "soft black rubber grip",
                },
            },
        ],
        matched_materials={"Matte black rubber": []},
    )

    assert [decision.action for decision in plan.decisions] == [
        MaterialAction.CREATE_NEW,
        MaterialAction.CREATE_NEW,
    ]
    assert len(plan.creation_intents) == 1
    intent = plan.creation_intents[0]
    assert intent.reuse_key == "matte_black_rubber"
    assert intent.target_prim_paths == (
        "/World/Geom/LeftGrip",
        "/World/Geom/RightGrip",
    )
    assert intent.decision_indices == (0, 1)
    assert [decision.creation_intent_id for decision in plan.decisions] == [
        intent.intent_id,
        intent.intent_id,
    ]
    assert [part.prim_path_hints[0] for part in intent.recipe.intended_parts] == [
        "/World/Geom/LeftGrip",
        "/World/Geom/RightGrip",
    ]


def test_modify_existing_remains_distinct_from_create_new() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/Cover",
                "materials": {
                    "material": "Painted metal",
                    "action": "modify_existing",
                    "reason": "needs a scratched variant of the supplied material",
                },
            }
        ],
        matched_materials={"Painted metal": _match("/World/Looks/PaintedMetal")},
    )

    assert len(plan.decisions) == 1
    assert plan.decisions[0].action is MaterialAction.MODIFY_EXISTING
    assert plan.decisions[0].matched_existing is True
    assert plan.creation_intents == ()


def test_task_wrapper_writes_policy_context_from_object_store() -> None:
    store = InMemoryObjectStore()
    store.set(
        "predictions",
        [
            {
                "id": "/World/Geom/Seat",
                "materials": {
                    "material": "Woven fabric",
                    "appearance_prompt": "dark woven fabric with visible threads",
                    "description": "fabric seat upholstery",
                },
            }
        ],
    )
    context = MaterialDecisionPolicyTask().run(
        {
            "matched_materials": {"Woven fabric": []},
            "material_creation_policy": {"allow_creation": True},
        },
        store,
    )

    assert context["material_creation_policy_stats"]["create_new"] == 1
    assert context["material_action_decisions"][0]["action"] == "create_new"
    assert context["material_creation_intents"][0]["reuse_key"] == "woven_fabric"
    assert context["material_creation_policy_conflicts"] == []
    assert context["material_decision_policy_result"]["conflicts"] == []


def test_task_wrapper_propagates_allow_creation_false() -> None:
    context = MaterialDecisionPolicyTask().run(
        {
            "predictions": [
                {"id": "/World/Geom/Panel", "materials": {"material": "New paint"}}
            ],
            "matched_materials": {"New paint": []},
            "material_creation_policy": {"allow_creation": False},
        }
    )

    assert context["material_creation_intents"] == []
    assert context["material_action_decisions"][0]["action"] == "assign_existing"
    assert context["material_creation_policy_conflicts"][0]["code"] == (
        "creation_disabled_unresolved_material"
    )


def test_task_wrapper_rejects_non_boolean_allow_creation() -> None:
    with pytest.raises(ValueError, match="allow_creation must be a boolean"):
        MaterialDecisionPolicyTask().run(
            {
                "predictions": [
                    {"id": "/World/Geom/Panel", "materials": {"material": "New paint"}}
                ],
                "material_creation_policy": {"allow_creation": "false"},
            }
        )


def test_missing_create_target_surfaces_policy_conflict() -> None:
    plan = plan_material_actions(
        [{"materials": {"material": "Unmapped ceramic", "action": "create_new"}}],
        matched_materials={"Unmapped ceramic": []},
    )

    assert plan.decisions[0].action is MaterialAction.CREATE_NEW
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "missing_target_prim_path"
    assert plan.conflicts[0].reuse_key == "unmapped_ceramic"
    assert plan.to_dict()["conflicts"][0]["reuse_key"] == "unmapped_ceramic"


def test_invalid_creation_recipe_surfaces_conflict_without_crashing() -> None:
    plan = plan_material_actions(
        [
            {"id": "/World/Geom/A", "materials": {"material": "!!!"}},
            {
                "id": "/World/Geom/B",
                "materials": {
                    "material": "Bad roughness",
                    "pbr_hints": {"roughness": "not-a-number"},
                },
            },
        ],
        matched_materials={"!!!": [], "Bad roughness": []},
    )

    assert [decision.action for decision in plan.decisions] == [
        MaterialAction.CREATE_NEW,
        MaterialAction.CREATE_NEW,
    ]
    assert plan.creation_intents == ()
    assert [conflict.code for conflict in plan.conflicts] == [
        "invalid_creation_recipe",
        "invalid_creation_recipe",
    ]
    assert plan.conflicts[0].prim_paths == ("/World/Geom/A",)
    assert plan.conflicts[1].prim_paths == ("/World/Geom/B",)


def test_explicit_action_without_material_name_surfaces_conflict() -> None:
    plan = plan_material_actions(
        [{"id": "/World/Geom/A", "action": "create_new"}],
        matched_materials={},
    )

    assert plan.decisions == ()
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "missing_material_name"
    assert plan.conflicts[0].prim_paths == ("/World/Geom/A",)


def test_explicit_assign_existing_does_not_fall_through_to_creation() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/A",
                "materials": {
                    "material": "Missing library material",
                    "action": "assign_existing",
                },
            }
        ],
        matched_materials={"Missing library material": []},
    )

    assert plan.decisions[0].action is MaterialAction.ASSIGN_EXISTING
    assert plan.decisions[0].matched_existing is False
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "missing_existing_material_match"


def test_modify_existing_without_match_surfaces_conflict() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/A",
                "materials": {
                    "material": "Missing variant base",
                    "action": "modify_existing",
                },
            }
        ],
        matched_materials={"Missing variant base": []},
    )

    assert plan.decisions[0].action is MaterialAction.MODIFY_EXISTING
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "missing_existing_material_match"


def test_creation_disabled_unresolved_material_is_flagged() -> None:
    plan = plan_material_actions(
        [{"id": "/World/Geom/A", "materials": {"material": "Unmapped fabric"}}],
        matched_materials={"Unmapped fabric": []},
        allow_creation=False,
    )

    assert plan.decisions[0].action is MaterialAction.ASSIGN_EXISTING
    assert plan.decisions[0].matched_existing is False
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "creation_disabled_unresolved_material"


def test_explicit_create_is_suppressed_when_creation_disabled() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/A",
                "materials": {
                    "material": "Unmapped fabric",
                    "action": "create_new",
                },
            }
        ],
        matched_materials={"Unmapped fabric": []},
        allow_creation=False,
    )

    assert plan.decisions[0].action is MaterialAction.ASSIGN_EXISTING
    assert plan.decisions[0].explicit_action is MaterialAction.CREATE_NEW
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "creation_disabled_explicit_create"


@pytest.mark.parametrize(
    "material_name",
    [UNKNOWN_MATERIAL_SENTINEL, FALLBACK_MATERIAL_NAME],
)
def test_fallback_material_is_not_created_when_explicit(material_name: str) -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/Unknown",
                "materials": {
                    "material": material_name,
                    "action": "create_new",
                },
            }
        ],
        matched_materials={},
    )

    assert plan.decisions[0].material == FALLBACK_MATERIAL_NAME
    assert plan.decisions[0].action is MaterialAction.ASSIGN_EXISTING
    assert plan.decisions[0].explicit_action is MaterialAction.CREATE_NEW
    assert plan.creation_intents == ()
    assert plan.conflicts == ()


def test_reuse_key_recipe_conflict_suppresses_ambiguous_intents() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/A",
                "materials": {
                    "material": "Custom coating",
                    "appearance_prompt": "smooth glossy custom coating",
                    "description": "glossy coating",
                },
            },
            {
                "id": "/World/Geom/B",
                "materials": {
                    "material": "Custom coating",
                    "appearance_prompt": "rough chipped custom coating",
                    "description": "rough coating",
                },
            },
        ],
        matched_materials={"Custom coating": []},
    )

    assert len(plan.decisions) == 2
    assert plan.creation_intents == ()
    assert plan.conflicts[0].code == "reuse_key_recipe_conflict"
    assert plan.conflicts[0].reuse_key == "custom_coating"
    assert plan.conflicts[0].prim_paths == ("/World/Geom/A", "/World/Geom/B")


def test_top_level_material_record_uses_material_name_for_recipe_id() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/PaintedPanel",
                "material": "Painted metal",
                "material_type": "metal",
                "description": "painted metal panel",
                "appearance_prompt": "smooth painted metal panel",
            }
        ],
        matched_materials={"Painted metal": []},
    )

    assert plan.creation_intents[0].reuse_key == "painted_metal"
    assert plan.creation_intents[0].recipe.material_id == "painted_metal"
    assert plan.creation_intents[0].recipe.material == "metal"


def test_nested_payload_id_is_not_used_as_recipe_id() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/PaintedPanel",
                "materials": {
                    "id": "/World/Geom/NestedIdentifier",
                    "material": "Painted metal",
                    "description": "painted metal panel",
                    "appearance_prompt": "smooth painted metal panel",
                },
            }
        ],
        matched_materials={"Painted metal": []},
    )

    assert plan.creation_intents[0].reuse_key == "painted_metal"
    assert plan.creation_intents[0].recipe.material_id == "painted_metal"


def test_nested_prediction_shapes_and_legacy_action_flags_are_supported() -> None:
    plan = plan_material_actions(
        {
            "results": [
                "Steel",
                42,
                {"id": "/World/Geom/StringMaterial", "materials": "Rubber"},
                {"id": "/World/Geom/StringResponse", "vlm_response": "Leather"},
                {"id": "/World/Geom/Empty", "materials": {"material": ""}},
                {
                    "id": "/World/Geom/BlankWithName",
                    "materials": {
                        "material": "",
                        "name": "Named coating",
                        "description": "coating with blank material fallback",
                        "appearance_prompt": "named coating",
                    },
                },
                {
                    "id": "/World/Geom/TopLevelMaterial",
                    "predicted_material": "Top level ceramic",
                    "materials": {
                        "description": "ceramic material from top-level key",
                        "appearance_prompt": "smooth white ceramic",
                    },
                },
                {
                    "/World/Geom/SlashChild": {
                        "materials": {
                            "name": "Natural wood",
                            "description": "warm wood grip",
                            "appearance_prompt": "warm wood with visible grain",
                            "reference_images": [
                                "wood-ref.png",
                                {"uri": "wood-uri.png"},
                                {"path": "wood-path.png"},
                                7,
                            ],
                        }
                    }
                },
                {
                    "id": "/World/Geom/LegacyCreate",
                    "materials": {
                        "material": "Custom coating",
                        "create_material": True,
                        "description": "custom coating",
                        "appearance_prompt": "smooth custom coating",
                        "reference_image_uris": "coating-ref.png",
                    },
                },
                {
                    "id": "/World/Geom/LegacyModify",
                    "materials": {
                        "material": "Existing coating",
                        "modify_material": True,
                    },
                },
                {
                    "id": "/World/Geom/Unknown",
                    "materials": {"material": "__UNKNOWN__"},
                },
                {
                    "id": "/World/Geom/NoRefs",
                    "materials": {
                        "material": "No reference coating",
                        "description": "coating without valid references",
                        "appearance_prompt": "plain coating",
                        "reference_images": 7,
                    },
                },
            ]
        },
        matched_materials={
            "Steel": _match("/World/Looks/Steel"),
            "Rubber": _match("/World/Looks/Rubber"),
            "Existing coating": _match("/World/Looks/ExistingCoating"),
        },
        resolved_materials={"Leather": "/World/Looks/Leather"},
    )

    decisions_by_material = {decision.material: decision for decision in plan.decisions}
    assert decisions_by_material["Steel"].action is MaterialAction.ASSIGN_EXISTING
    assert decisions_by_material["Rubber"].action is MaterialAction.ASSIGN_EXISTING
    assert decisions_by_material["Leather"].matched_existing is True
    assert (
        decisions_by_material["Top level ceramic"].prim_path
        == "/World/Geom/TopLevelMaterial"
    )
    assert decisions_by_material["Named coating"].prim_path == (
        "/World/Geom/BlankWithName"
    )
    assert decisions_by_material["Natural wood"].prim_path == "/World/Geom/SlashChild"
    assert decisions_by_material["Natural wood"].recipe is not None
    assert decisions_by_material["Natural wood"].recipe.reference_image_uris == (
        "wood-ref.png",
        "wood-uri.png",
        "wood-path.png",
    )
    assert (
        decisions_by_material["Custom coating"].explicit_action
        is MaterialAction.CREATE_NEW
    )
    assert decisions_by_material["Custom coating"].recipe is not None
    assert decisions_by_material["Custom coating"].recipe.reference_image_uris == (
        "coating-ref.png",
    )
    assert (
        decisions_by_material["Existing coating"].action
        is MaterialAction.MODIFY_EXISTING
    )
    assert (
        decisions_by_material[FALLBACK_MATERIAL_NAME].action
        is MaterialAction.ASSIGN_EXISTING
    )
    assert (
        decisions_by_material[FALLBACK_MATERIAL_NAME].prim_path == "/World/Geom/Unknown"
    )
    assert decisions_by_material["No reference coating"].recipe is not None
    assert (
        decisions_by_material["No reference coating"].recipe.reference_image_uris == ()
    )
    decision_prim_paths = {decision.prim_path for decision in plan.decisions}
    assert "/World/Geom/Empty" not in decision_prim_paths
    assert all(
        conflict.prim_paths != ("/World/Geom/Empty",) for conflict in plan.conflicts
    )
    assert len(plan.decisions) == 10


def test_bare_reference_image_mapping_is_preserved() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/Panel",
                "materials": {
                    "material": "Mapped reference coating",
                    "description": "coating with one mapped reference",
                    "appearance_prompt": "coating with single reference",
                    "reference_images": {"uri": "single-reference.png"},
                },
            }
        ],
        matched_materials={"Mapped reference coating": []},
    )

    assert plan.creation_intents[0].recipe.reference_image_uris == (
        "single-reference.png",
    )


def test_container_predictions_and_invalid_action_values_are_ignored() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/Outer",
                "items": [
                    {
                        "id": "/World/Geom/NestedFabric",
                        "materials": {
                            "material": "Nested fabric",
                            "requested_action": "not-a-real-action",
                            "description": "nested fabric material",
                            "appearance_prompt": "nested woven fabric",
                        },
                    }
                ],
            }
        ],
        matched_materials={"Nested fabric": []},
    )

    assert len(plan.decisions) == 1
    assert plan.decisions[0].action is MaterialAction.CREATE_NEW
    assert plan.decisions[0].explicit_action is None
    assert plan.conflicts == ()


def test_action_only_container_propagates_action_to_nested_predictions() -> None:
    plan = plan_material_actions(
        [
            {
                "action": "create_new",
                "items": [
                    {
                        "id": "/World/Geom/NestedPaint",
                        "materials": {
                            "material": "Nested custom paint",
                            "description": "custom paint inherited from parent action",
                            "appearance_prompt": "custom glossy nested paint",
                        },
                    }
                ],
                "/World/Geom/StringPaint": "String custom paint",
            }
        ],
        matched_materials={"Nested custom paint": [], "String custom paint": []},
    )

    decisions_by_material = {decision.material: decision for decision in plan.decisions}
    assert len(plan.decisions) == 2
    assert len(plan.creation_intents) == 2
    assert decisions_by_material["Nested custom paint"].prim_path == (
        "/World/Geom/NestedPaint"
    )
    assert decisions_by_material["Nested custom paint"].action is (
        MaterialAction.CREATE_NEW
    )
    assert decisions_by_material["Nested custom paint"].explicit_action is (
        MaterialAction.CREATE_NEW
    )
    assert decisions_by_material["String custom paint"].prim_path == (
        "/World/Geom/StringPaint"
    )
    assert decisions_by_material["String custom paint"].action is (
        MaterialAction.CREATE_NEW
    )
    assert decisions_by_material["String custom paint"].explicit_action is (
        MaterialAction.CREATE_NEW
    )
    assert plan.conflicts == ()


def test_slash_prefixed_siblings_get_distinct_creation_intent_ids() -> None:
    plan = plan_material_actions(
        {
            "/World/Geom/BluePanel": {
                "materials": {
                    "material": "Blue custom coating",
                    "description": "blue panel coating",
                    "appearance_prompt": "smooth blue custom coating",
                }
            },
            "/World/Geom/RedPanel": {
                "materials": {
                    "material": "Red custom coating",
                    "description": "red panel coating",
                    "appearance_prompt": "smooth red custom coating",
                }
            },
        },
        matched_materials={
            "Blue custom coating": [],
            "Red custom coating": [],
        },
    )

    decisions_by_prim = {decision.prim_path: decision for decision in plan.decisions}
    intents_by_reuse_key = {
        intent.reuse_key: intent for intent in plan.creation_intents
    }

    assert [decision.prediction_index for decision in plan.decisions] == [0, 1]
    assert len(plan.creation_intents) == 2
    assert (
        decisions_by_prim["/World/Geom/BluePanel"].creation_intent_id
        == intents_by_reuse_key["blue_custom_coating"].intent_id
    )
    assert (
        decisions_by_prim["/World/Geom/RedPanel"].creation_intent_id
        == intents_by_reuse_key["red_custom_coating"].intent_id
    )


def test_material_decision_plan_round_trips_from_dict() -> None:
    plan = plan_material_actions(
        [
            {
                "id": "/World/Geom/Panel",
                "materials": {
                    "material": "Roundtrip coating",
                    "description": "roundtrip coating material",
                    "appearance_prompt": "roundtrip coating",
                    "reference_images": {"path": "roundtrip-reference.png"},
                },
            }
        ],
        matched_materials={"Roundtrip coating": []},
    )

    restored = MaterialDecisionPlan.from_dict(plan.to_dict())

    assert restored.to_dict() == plan.to_dict()


def test_material_policy_conflict_round_trips_from_dict() -> None:
    payload = {
        "code": "reuse_key_recipe_conflict",
        "message": "conflicting recipes",
        "reuse_key": "shared_coating",
        "prim_paths": ["/World/Geom/A", "/World/Geom/B"],
    }

    conflict = MaterialPolicyConflict.from_dict(payload)

    assert conflict.to_dict() == payload


def test_policy_result_from_dict_rejects_invalid_shapes() -> None:
    with pytest.raises(TypeError, match="decision plan"):
        MaterialDecisionPlan.from_dict([])
    with pytest.raises(TypeError, match="decision entries"):
        MaterialDecision.from_dict([])
    with pytest.raises(TypeError, match="intent entries"):
        MaterialCreationIntent.from_dict([])
    with pytest.raises(ValueError, match="requires a recipe"):
        MaterialCreationIntent.from_dict({"recipe": None})
    with pytest.raises(TypeError, match="conflict entries"):
        MaterialPolicyConflict.from_dict([])


def test_task_wrapper_records_conflict_warning_context() -> None:
    store = InMemoryObjectStore()
    store.set("predictions", [{"id": "/World/Geom/A", "action": "create_new"}])

    context = MaterialDecisionPolicyTask().run(
        {"material_creation_policy": {"allow_creation": True}},
        store,
    )

    assert context["material_creation_policy_stats"]["conflicts"] == 1
    assert context["material_creation_policy_conflicts"][0]["code"] == (
        "missing_material_name"
    )
