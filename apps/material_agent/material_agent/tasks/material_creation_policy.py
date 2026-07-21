# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision policy for assigning, creating, or modifying materials.

WP1 owns only the agent decision boundary.  It ranks existing material matches
first, normalizes unmet requirements into the WP0 ``MaterialRecipe`` schema, and
groups compatible creation requests for run-local reuse.  Backend execution,
material packaging, and workflow insertion are owned by later packages.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.material_library_generation.creation_contract import MaterialAction
from material_agent.material_library_generation.schema import (
    IntendedPart,
    MaterialRecipe,
    PBRHints,
    make_material_id,
)
from material_agent.materials import (
    FALLBACK_MATERIAL_NAME,
    PREDICTION_CONTAINER_KEYS,
    PREDICTION_ID_KEYS,
    PREDICTION_MATERIAL_KEYS,
    is_actionable_material_name,
    is_unknown_material_name,
    normalize_material_name,
)

logger = logging.getLogger(__name__)


_CREATE_ACTIONS = {"create", "create_new", "generate", "generate_new", "new"}
_MODIFY_ACTIONS = {"modify", "modify_existing", "variation", "texture_variation"}
_ASSIGN_ACTIONS = {"assign", "assign_existing", "existing", "use_existing"}
_ACTION_KEYS = (
    "material_action",
    "action",
    "requested_action",
    "assignment_action",
)


@dataclass(frozen=True)
class MaterialPolicyConflict:
    """Non-fatal planning conflict that later integration must not ignore."""

    code: str
    message: str
    reuse_key: str | None = None
    prim_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "prim_paths": list(self.prim_paths),
        }
        if self.reuse_key is not None:
            data["reuse_key"] = self.reuse_key
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialPolicyConflict:
        if not isinstance(data, Mapping):
            raise TypeError("material policy conflict entries must be mappings")
        reuse_key = data.get("reuse_key")
        return cls(
            code=str(data.get("code", "")),
            message=str(data.get("message", "")),
            reuse_key=reuse_key if isinstance(reuse_key, str) else None,
            prim_paths=_string_tuple(data.get("prim_paths")),
        )


@dataclass(frozen=True)
class MaterialDecision:
    """Action selected for one predicted material requirement."""

    prediction_index: int
    material: str
    action: MaterialAction
    matched_existing: bool
    reason: str
    prim_path: str | None = None
    reuse_key: str | None = None
    creation_intent_id: str | None = None
    explicit_action: MaterialAction | None = None
    recipe: MaterialRecipe | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "prediction_index": self.prediction_index,
            "material": self.material,
            "action": self.action.value,
            "matched_existing": self.matched_existing,
            "reason": self.reason,
        }
        if self.prim_path is not None:
            data["prim_path"] = self.prim_path
        if self.reuse_key is not None:
            data["reuse_key"] = self.reuse_key
        if self.creation_intent_id is not None:
            data["creation_intent_id"] = self.creation_intent_id
        if self.explicit_action is not None:
            data["explicit_action"] = self.explicit_action.value
        if self.recipe is not None:
            data["recipe"] = self.recipe.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialDecision:
        if not isinstance(data, Mapping):
            raise TypeError("material decision entries must be mappings")
        explicit_action = data.get("explicit_action")
        recipe_data = data.get("recipe")
        recipe = (
            MaterialRecipe.from_dict(dict(recipe_data))
            if isinstance(recipe_data, Mapping)
            else None
        )
        return cls(
            prediction_index=int(data.get("prediction_index", 0)),
            material=str(data.get("material", "")),
            action=MaterialAction(str(data.get("action"))),
            matched_existing=bool(data.get("matched_existing", False)),
            reason=str(data.get("reason", "")),
            prim_path=_optional_text(data.get("prim_path")),
            reuse_key=_optional_text(data.get("reuse_key")),
            creation_intent_id=_optional_text(data.get("creation_intent_id")),
            explicit_action=(
                MaterialAction(str(explicit_action))
                if explicit_action is not None
                else None
            ),
            recipe=recipe,
        )


@dataclass(frozen=True)
class MaterialCreationIntent:
    """Run-local creation intent shared by compatible prim requirements."""

    intent_id: str
    reuse_key: str
    recipe: MaterialRecipe
    target_prim_paths: tuple[str, ...]
    decision_indices: tuple[int, ...]
    explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "reuse_key": self.reuse_key,
            "recipe": self.recipe.to_dict(),
            "target_prim_paths": list(self.target_prim_paths),
            "decision_indices": list(self.decision_indices),
            "explicit": self.explicit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialCreationIntent:
        if not isinstance(data, Mapping):
            raise TypeError("material creation intent entries must be mappings")
        recipe_data = data.get("recipe")
        if not isinstance(recipe_data, Mapping):
            raise ValueError("material creation intent requires a recipe mapping")
        return cls(
            intent_id=str(data.get("intent_id", "")),
            reuse_key=str(data.get("reuse_key", "")),
            recipe=MaterialRecipe.from_dict(dict(recipe_data)),
            target_prim_paths=_string_tuple(data.get("target_prim_paths")),
            decision_indices=tuple(
                int(index)
                for index in data.get("decision_indices", ())
                if isinstance(index, int | str)
            ),
            explicit=bool(data.get("explicit", False)),
        )


@dataclass(frozen=True)
class MaterialDecisionPlan:
    """Policy result consumed by later workflow and creation packages."""

    decisions: tuple[MaterialDecision, ...]
    creation_intents: tuple[MaterialCreationIntent, ...]
    conflicts: tuple[MaterialPolicyConflict, ...] = ()

    @property
    def stats(self) -> dict[str, int]:
        counts = {
            MaterialAction.ASSIGN_EXISTING.value: 0,
            MaterialAction.CREATE_NEW.value: 0,
            MaterialAction.MODIFY_EXISTING.value: 0,
        }
        for decision in self.decisions:
            counts[decision.action.value] += 1
        counts["creation_intents"] = len(self.creation_intents)
        counts["conflicts"] = len(self.conflicts)
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "creation_intents": [intent.to_dict() for intent in self.creation_intents],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MaterialDecisionPlan:
        if not isinstance(data, Mapping):
            raise TypeError("material decision plan must be a mapping")
        return cls(
            decisions=tuple(
                MaterialDecision.from_dict(decision)
                for decision in data.get("decisions", ())
                if isinstance(decision, Mapping)
            ),
            creation_intents=tuple(
                MaterialCreationIntent.from_dict(intent)
                for intent in data.get("creation_intents", ())
                if isinstance(intent, Mapping)
            ),
            conflicts=tuple(
                MaterialPolicyConflict.from_dict(conflict)
                for conflict in data.get("conflicts", ())
                if isinstance(conflict, Mapping)
            ),
        )


def plan_material_actions(
    predictions_data: Iterable[Any],
    *,
    matched_materials: Mapping[str, Sequence[Any]] | None = None,
    resolved_materials: Mapping[str, str] | None = None,
    allow_creation: bool = True,
) -> MaterialDecisionPlan:
    """Select material actions after existing material retrieval has run.

    Existing matches always win unless the prediction explicitly requests
    ``create_new`` or ``modify_existing``.  Unresolved actionable requirements are
    normalized into ``MaterialRecipe`` creation intents when creation is allowed.
    """

    if not isinstance(allow_creation, bool):
        raise ValueError("allow_creation must be a boolean")

    matched_by_name = _normalize_mapping_keys(matched_materials or {})
    resolved_by_name = _normalize_mapping_keys(resolved_materials or {})

    decisions: list[MaterialDecision] = []
    conflicts: list[MaterialPolicyConflict] = []

    for index, prediction in enumerate(_iter_prediction_records(predictions_data)):
        explicit_action = _explicit_action_from_prediction(prediction)
        material = _selected_material_from_prediction(prediction)
        prim_path = _prediction_prim_path(prediction)
        if material is None:
            if explicit_action is not None:
                conflicts.append(
                    MaterialPolicyConflict(
                        code="missing_material_name",
                        message=(
                            "explicit material action was ignored because the "
                            "prediction did not include an actionable material name"
                        ),
                        prim_paths=_prim_paths_for_conflict(prim_path),
                    )
                )
            continue

        has_existing = material == FALLBACK_MATERIAL_NAME or _has_existing_match(
            material,
            matched_materials=matched_by_name,
            resolved_materials=resolved_by_name,
        )

        if material == FALLBACK_MATERIAL_NAME:
            action = MaterialAction.ASSIGN_EXISTING
            reason = "default fallback material selected"
        elif explicit_action is MaterialAction.CREATE_NEW:
            if allow_creation:
                action = MaterialAction.CREATE_NEW
                reason = "explicit creation requested"
            else:
                action = MaterialAction.ASSIGN_EXISTING
                reason = "creation disabled; explicit creation suppressed"
        elif explicit_action is MaterialAction.MODIFY_EXISTING:
            action = MaterialAction.MODIFY_EXISTING
            reason = "explicit modification requested"
        elif explicit_action is MaterialAction.ASSIGN_EXISTING:
            action = MaterialAction.ASSIGN_EXISTING
            reason = "explicit existing-material assignment requested"
        elif has_existing:
            action = MaterialAction.ASSIGN_EXISTING
            reason = "existing material match found"
        elif allow_creation:
            action = MaterialAction.CREATE_NEW
            reason = "no adequate existing material match found"
        else:
            action = MaterialAction.ASSIGN_EXISTING
            reason = "creation disabled; leaving unresolved material for existing path"

        recipe = None
        reuse_key = None
        if action is MaterialAction.CREATE_NEW:
            try:
                recipe = _recipe_from_prediction(prediction, material, prim_path)
            except Exception as exc:
                conflicts.append(
                    MaterialPolicyConflict(
                        code="invalid_creation_recipe",
                        message=(
                            "creation prediction could not be normalized into a "
                            f"valid MaterialRecipe: {exc}"
                        ),
                        prim_paths=_prim_paths_for_conflict(prim_path),
                    )
                )
            else:
                reuse_key = recipe.material_id
                if prim_path is None:
                    conflicts.append(
                        MaterialPolicyConflict(
                            code="missing_target_prim_path",
                            message=(
                                "creation requires an absolute target prim path "
                                "before a CreateMaterialRequest can be built"
                            ),
                            reuse_key=reuse_key,
                        )
                    )
        elif action is MaterialAction.MODIFY_EXISTING and not has_existing:
            conflicts.append(
                MaterialPolicyConflict(
                    code="missing_existing_material_match",
                    message=(
                        "modify_existing requires a matched existing material "
                        "before texture variation can run"
                    ),
                    prim_paths=_prim_paths_for_conflict(prim_path),
                )
            )
        elif action is MaterialAction.ASSIGN_EXISTING and (
            not has_existing
            or (explicit_action is MaterialAction.CREATE_NEW and not allow_creation)
        ):
            if explicit_action is MaterialAction.ASSIGN_EXISTING:
                code = "missing_existing_material_match"
                message = "assign_existing selected without a matched existing material"
            elif explicit_action is MaterialAction.CREATE_NEW:
                code = "creation_disabled_explicit_create"
                message = "create_new requested while material creation is disabled"
            else:
                code = "creation_disabled_unresolved_material"
                message = "assign_existing selected without a matched existing material"
            conflicts.append(
                MaterialPolicyConflict(
                    code=code,
                    message=message,
                    prim_paths=_prim_paths_for_conflict(prim_path),
                )
            )

        decisions.append(
            MaterialDecision(
                prediction_index=index,
                prim_path=prim_path,
                material=material,
                action=action,
                matched_existing=has_existing,
                reason=reason,
                reuse_key=reuse_key,
                explicit_action=explicit_action,
                recipe=recipe,
            )
        )

    creation_intents, reuse_conflicts = _build_creation_intents(decisions)
    conflicts.extend(reuse_conflicts)
    decisions = _attach_creation_intent_ids(decisions, creation_intents)

    return MaterialDecisionPlan(
        decisions=tuple(decisions),
        creation_intents=tuple(creation_intents),
        conflicts=tuple(conflicts),
    )


class MaterialDecisionPolicyTask(Task):
    """Task wrapper around :func:`plan_material_actions`."""

    def __init__(self) -> None:
        self.name = "MaterialDecisionPolicy"
        self.description = "Select assign/create/modify material actions"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)
        predictions_data = context.get("predictions_data")
        if predictions_data is None:
            predictions_data = context.get("predictions")
        if predictions_data is None and object_store is not None:
            predictions_data = object_store.get("predictions")
        if predictions_data is None:
            raise ValueError(
                "predictions_data or predictions must be provided before "
                "material decision policy can run"
            )

        policy_config = context.get("material_creation_policy", {})
        if not isinstance(policy_config, dict):
            raise ValueError("material_creation_policy must be a mapping when set")
        allow_creation_value = policy_config.get("allow_creation", True)
        if not isinstance(allow_creation_value, bool):
            raise ValueError(
                "material_creation_policy.allow_creation must be a boolean"
            )
        allow_creation = allow_creation_value

        plan = plan_material_actions(
            predictions_data,
            matched_materials=context.get("matched_materials"),
            resolved_materials=context.get("resolved_materials"),
            allow_creation=allow_creation,
        )
        result = plan.to_dict()
        context["material_decision_policy_result"] = result
        context["material_action_decisions"] = result["decisions"]
        context["material_creation_intents"] = result["creation_intents"]
        context["material_creation_policy_conflicts"] = result["conflicts"]
        context["material_creation_policy_stats"] = result["stats"]

        listener.info(
            "Material decision policy complete: "
            f"{result['stats'][MaterialAction.ASSIGN_EXISTING.value]} assign, "
            f"{result['stats'][MaterialAction.CREATE_NEW.value]} create, "
            f"{result['stats'][MaterialAction.MODIFY_EXISTING.value]} modify"
        )
        if plan.conflicts:
            listener.warning(
                f"Material decision policy produced {len(plan.conflicts)} conflict(s)"
            )
        return context


def _iter_prediction_records(
    predictions_data: Iterable[Any],
) -> Iterable[dict[str, Any]]:
    if isinstance(predictions_data, dict | str):
        values: Iterable[Any] = (predictions_data,)
    else:
        values = predictions_data

    for index, value in enumerate(values):
        yield from _iter_prediction_value(
            value,
            inherited_prim_path=None,
            index=index,
            inherited_action=None,
        )


def _iter_prediction_value(
    value: Any,
    *,
    inherited_prim_path: str | None,
    index: int,
    inherited_action: MaterialAction | None,
) -> Iterable[dict[str, Any]]:
    if isinstance(value, str):
        record = {"material": value, "id": inherited_prim_path or f"index:{index}"}
        if inherited_action is not None:
            record["action"] = inherited_action.value
        yield record
        return
    if isinstance(value, list):
        for child_index, item in enumerate(value):
            yield from _iter_prediction_value(
                item,
                inherited_prim_path=inherited_prim_path,
                index=child_index,
                inherited_action=inherited_action,
            )
        return
    if not isinstance(value, dict):
        return

    record = dict(value)
    if inherited_prim_path is not None and not _prediction_prim_path(record):
        record["id"] = inherited_prim_path

    selected_material = _selected_material_from_prediction(record)
    explicit_action = _explicit_action_from_prediction(record)
    if selected_material is not None:
        if explicit_action is None and inherited_action is not None:
            record["action"] = inherited_action.value
        yield record
        return

    child_action = explicit_action or inherited_action
    yielded_child = False
    for container_key in PREDICTION_CONTAINER_KEYS:
        container = value.get(container_key)
        if container is not None:
            for child_record in _iter_prediction_value(
                container,
                inherited_prim_path=inherited_prim_path,
                index=index,
                inherited_action=child_action,
            ):
                yielded_child = True
                yield child_record

    for key, child in value.items():
        if isinstance(key, str) and key.startswith("/"):
            for child_record in _iter_prediction_value(
                child,
                inherited_prim_path=key,
                index=index,
                inherited_action=child_action,
            ):
                yielded_child = True
                yield child_record

    if explicit_action is not None and not yielded_child:
        yield record


def _normalize_mapping_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        normalize_material_name(key): value
        for key, value in mapping.items()
        if isinstance(key, str)
    }


def _has_existing_match(
    material: str,
    *,
    matched_materials: Mapping[str, Sequence[Any]],
    resolved_materials: Mapping[str, str],
) -> bool:
    resolved = resolved_materials.get(material)
    if isinstance(resolved, str) and resolved.strip():
        return True
    matches = matched_materials.get(material)
    return bool(matches)


def _material_payload(prediction: Mapping[str, Any]) -> Mapping[str, Any]:
    materials = prediction.get("materials")
    if isinstance(materials, Mapping):
        return materials
    vlm_response = prediction.get("vlm_response")
    if isinstance(vlm_response, Mapping):
        return vlm_response
    return prediction


def _selected_material_from_prediction(prediction: Mapping[str, Any]) -> str | None:
    payload = _material_payload(prediction)
    materials = prediction.get("materials")
    if isinstance(materials, str):
        selected = _normalize_selected_material(materials)
        if selected is not None:
            return selected
    vlm_response = prediction.get("vlm_response")
    if isinstance(vlm_response, str):
        selected = _normalize_selected_material(vlm_response)
        if selected is not None:
            return selected

    for key in (*PREDICTION_MATERIAL_KEYS, "name"):
        value = payload.get(key)
        if isinstance(value, str):
            selected = _normalize_selected_material(value)
            if selected is not None:
                return selected
        value = prediction.get(key)
        if isinstance(value, str):
            selected = _normalize_selected_material(value)
            if selected is not None:
                return selected
    return None


def _normalize_selected_material(material: str) -> str | None:
    if is_unknown_material_name(material):
        return cast(str, FALLBACK_MATERIAL_NAME)
    if not is_actionable_material_name(material):
        return None
    return cast(str, normalize_material_name(material))


def _explicit_action_from_prediction(
    prediction: Mapping[str, Any],
) -> MaterialAction | None:
    payload = _material_payload(prediction)
    for container in (payload, prediction):
        for key in _ACTION_KEYS:
            value = container.get(key)
            action = _material_action_from_value(value)
            if action is not None:
                return action
        if container.get("create_material") is True:
            return MaterialAction.CREATE_NEW
        if container.get("modify_material") is True:
            return MaterialAction.MODIFY_EXISTING
    return None


def _material_action_from_value(value: Any) -> MaterialAction | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _CREATE_ACTIONS:
        return MaterialAction.CREATE_NEW
    if normalized in _MODIFY_ACTIONS:
        return MaterialAction.MODIFY_EXISTING
    if normalized in _ASSIGN_ACTIONS:
        return MaterialAction.ASSIGN_EXISTING
    return None


def _prediction_prim_path(prediction: Mapping[str, Any]) -> str | None:
    for key in PREDICTION_ID_KEYS:
        value = prediction.get(key)
        if isinstance(value, str) and value.startswith("/"):
            return value
    return None


def _recipe_from_prediction(
    prediction: Mapping[str, Any],
    material: str,
    prim_path: str | None,
) -> MaterialRecipe:
    payload = _material_payload(prediction)
    reason = _first_text(
        payload.get("reason"),
        prediction.get("reason"),
        payload.get("evidence"),
        prediction.get("evidence"),
    )
    description = _first_text(
        payload.get("description"),
        prediction.get("description"),
        reason,
        f"Material requirement for {material}",
    )
    appearance_prompt = _first_text(
        payload.get("appearance_prompt"),
        payload.get("prompt"),
        prediction.get("appearance_prompt"),
        prediction.get("prompt"),
        description,
        material,
    )
    pbr_hints = PBRHints.from_dict(
        _mapping_or_none(payload.get("pbr_hints"))
        or _mapping_or_none(prediction.get("pbr_hints"))
    )
    intended_parts: tuple[IntendedPart, ...] = ()
    if prim_path is not None:
        label = _first_text(
            payload.get("semantic_label"),
            prediction.get("semantic_label"),
            payload.get("part"),
            prediction.get("part"),
            prim_path.rsplit("/", 1)[-1],
            material,
        )
        intended_parts = (
            IntendedPart(
                semantic_label=label,
                evidence=reason or description,
                prim_path_hints=(prim_path,),
            ),
        )

    material_id = _first_text(payload.get("material_id"), prediction.get("material_id"))

    recipe = MaterialRecipe(
        id=_first_text(material_id, material),
        name=_first_text(payload.get("name"), material),
        description=description,
        appearance_prompt=appearance_prompt,
        color=_optional_text(payload.get("color") or prediction.get("color")),
        material=_optional_text(
            payload.get("material_type") or prediction.get("material_type")
        ),
        finish=_optional_text(payload.get("finish") or prediction.get("finish")),
        pbr_hints=pbr_hints,
        reference_image_uris=_string_tuple(
            payload.get("reference_image_uris")
            or prediction.get("reference_image_uris")
            or payload.get("reference_images")
            or prediction.get("reference_images")
        ),
        intended_parts=intended_parts,
    )
    recipe.validate()
    return recipe


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        uri = value.get("uri") or value.get("path")
        return (uri,) if isinstance(uri, str) else ()
    if not isinstance(value, Iterable):
        return ()

    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, Mapping):
            uri = item.get("uri") or item.get("path")
            if isinstance(uri, str):
                items.append(uri)
    return tuple(items)


def _prim_paths_for_conflict(prim_path: str | None) -> tuple[str, ...]:
    return (prim_path,) if prim_path is not None else ()


def _build_creation_intents(
    decisions: Sequence[MaterialDecision],
) -> tuple[list[MaterialCreationIntent], list[MaterialPolicyConflict]]:
    grouped: dict[tuple[str, str], list[MaterialDecision]] = {}
    fingerprints_by_reuse_key: dict[str, set[str]] = {}
    conflicts: list[MaterialPolicyConflict] = []
    conflicting_reuse_keys: set[str] = set()

    for decision in decisions:
        if decision.action is not MaterialAction.CREATE_NEW or decision.recipe is None:
            continue
        if decision.prim_path is None:
            continue
        reuse_key = decision.reuse_key or decision.recipe.material_id
        fingerprint = _recipe_fingerprint(decision.recipe)
        grouped.setdefault((reuse_key, fingerprint), []).append(decision)
        fingerprints_by_reuse_key.setdefault(reuse_key, set()).add(fingerprint)

    for reuse_key, fingerprints in fingerprints_by_reuse_key.items():
        if len(fingerprints) > 1:
            conflicting_reuse_keys.add(reuse_key)
            prim_paths = tuple(
                sorted(
                    {
                        decision.prim_path
                        for (key, _), values in grouped.items()
                        if key == reuse_key
                        for decision in values
                        if decision.prim_path is not None
                    }
                )
            )
            conflicts.append(
                MaterialPolicyConflict(
                    code="reuse_key_recipe_conflict",
                    message=(
                        "multiple incompatible creation recipes share one "
                        "run-local reuse key"
                    ),
                    reuse_key=reuse_key,
                    prim_paths=prim_paths,
                )
            )

    intents: list[MaterialCreationIntent] = []
    for (reuse_key, fingerprint), values in sorted(grouped.items()):
        if reuse_key in conflicting_reuse_keys:
            continue
        recipe = _merge_recipe_intended_parts(values)
        target_paths = tuple(
            sorted({value.prim_path for value in values if value.prim_path is not None})
        )
        intent_id = _intent_id(reuse_key, fingerprint, target_paths)
        intents.append(
            MaterialCreationIntent(
                intent_id=intent_id,
                reuse_key=reuse_key,
                recipe=recipe,
                target_prim_paths=target_paths,
                decision_indices=tuple(value.prediction_index for value in values),
                explicit=any(
                    value.explicit_action is MaterialAction.CREATE_NEW
                    for value in values
                ),
            )
        )
    return intents, conflicts


def _attach_creation_intent_ids(
    decisions: Sequence[MaterialDecision],
    intents: Sequence[MaterialCreationIntent],
) -> list[MaterialDecision]:
    intent_by_decision = {
        decision_index: intent.intent_id
        for intent in intents
        for decision_index in intent.decision_indices
    }
    return [
        MaterialDecision(
            prediction_index=decision.prediction_index,
            prim_path=decision.prim_path,
            material=decision.material,
            action=decision.action,
            matched_existing=decision.matched_existing,
            reason=decision.reason,
            reuse_key=decision.reuse_key,
            creation_intent_id=intent_by_decision.get(decision.prediction_index),
            explicit_action=decision.explicit_action,
            recipe=decision.recipe,
        )
        for decision in decisions
    ]


def _recipe_fingerprint(recipe: MaterialRecipe) -> str:
    payload = recipe.to_dict()
    payload.pop("intended_parts", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _intent_id(
    reuse_key: str,
    recipe_fingerprint: str,
    target_prim_paths: tuple[str, ...],
) -> str:
    payload = {
        "reuse_key": reuse_key,
        "recipe_fingerprint": recipe_fingerprint,
        "target_prim_paths": target_prim_paths,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"mci_{digest[:16]}"


def _merge_recipe_intended_parts(
    decisions: Sequence[MaterialDecision],
) -> MaterialRecipe:
    recipes = [decision.recipe for decision in decisions if decision.recipe is not None]
    if not recipes:
        raise ValueError("cannot merge an empty creation recipe group")

    base = recipes[0]
    intended_parts: list[IntendedPart] = []
    seen_parts: set[tuple[str, tuple[str, ...]]] = set()
    for recipe in recipes:
        for part in recipe.intended_parts:
            key = (part.semantic_label, part.prim_path_hints)
            if key not in seen_parts:
                intended_parts.append(part)
                seen_parts.add(key)

    return MaterialRecipe(
        id=make_material_id(base.material_id),
        name=base.name,
        description=base.description,
        appearance_prompt=base.appearance_prompt,
        color=base.color,
        material=base.material,
        finish=base.finish,
        base_color_hint=base.base_color_hint,
        pbr_hints=base.pbr_hints,
        reference_image_uris=base.reference_image_uris,
        intended_parts=tuple(intended_parts),
        priority=base.priority,
    )
