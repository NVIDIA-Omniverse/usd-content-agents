# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release-gate coverage for finalized Joint Rigger core branches."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from world_understanding.functions.physics.joint_rigger.models import (
    DIAGNOSTICS_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    FieldDecisionV1,
    FieldProvenanceV1,
    JointRiggerContractError,
    JointRiggerDiagnosticsV1,
    JointRiggerInputV2,
)
from world_understanding.utils.model_auth import ModelAuthenticationFailure

author = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.author"
)
inference = importlib.import_module(
    "world_understanding.functions.classification.inference"
)
models = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.models"
)
rigid_links = importlib.import_module(
    "world_understanding.functions.physics.joint_rigger.rigid_links"
)


def _mimic_joint(
    joint_id: str,
    *,
    joint_type: str = "revolute",
    reference_joint_id: str | None = None,
) -> Any:
    mimic = (
        None
        if reference_joint_id is None
        else SimpleNamespace(reference_joint_id=reference_joint_id)
    )
    return SimpleNamespace(
        topology=SimpleNamespace(joint_id=joint_id, joint_type=joint_type),
        mimic=mimic,
    )


@pytest.mark.parametrize(
    ("joints", "message"),
    [
        (
            (_mimic_joint("follower", reference_joint_id="follower"),),
            "cannot reference itself",
        ),
        (
            (_mimic_joint("follower", reference_joint_id="missing"),),
            "must name another plan joint",
        ),
        (
            (
                _mimic_joint("follower", reference_joint_id="reference"),
                _mimic_joint("reference", joint_type="spherical"),
            ),
            "cannot name a spherical joint",
        ),
    ],
)
def test_v2_mimic_reference_validation_rejects_invalid_targets(
    joints: tuple[Any, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        models._validate_mimic_references(joints)


def test_v2_mimic_reference_validation_accepts_resolved_target() -> None:
    models._validate_mimic_references(
        (
            _mimic_joint("follower", reference_joint_id="reference"),
            _mimic_joint("reference"),
        )
    )


def _articulation_root_request() -> JointRiggerInputV2:
    provenance = FieldProvenanceV1(
        source="template_default",
        evidence="release coverage articulation root",
    )
    root = ArticulationRootPlanV1(
        prim_path="/World",
        provenance=provenance,
    )
    return JointRiggerInputV2.model_construct(
        plan=SimpleNamespace(articulation_roots=(root,), rigid_bodies=())
    )


def test_v2_topology_diagnostics_reports_each_articulation_root() -> None:
    diagnostics = JointRiggerDiagnosticsV1(
        schema_version=DIAGNOSTICS_SCHEMA_VERSION,
        backend_name="release-coverage",
        field_decisions=(
            FieldDecisionV1(
                field="articulation_root",
                disposition="ignored",
                reason_code="legacy_v1_field",
            ),
        ),
    )

    result = author._topology_diagnostics_for_request(
        _articulation_root_request(), diagnostics
    )

    assert tuple(decision.field for decision in result.field_decisions) == (
        "articulation_roots[/World]",
        "rigid_bodies",
    )
    assert result.field_decisions[0].disposition == "accepted"
    assert result.field_decisions[1].disposition == "ignored"


def test_v2_articulation_root_is_applied_to_the_exact_planned_prim() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")

    author._author_v2_articulation_roots(stage, _articulation_root_request())

    assert UsdPhysics.ArticulationRootAPI(stage.GetPrimAtPath("/World"))


def test_authored_aggregate_validation_rejects_missing_body() -> None:
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: None)
    link = SimpleNamespace(body_prim_path="/World/aggregate")

    with pytest.raises(JointRiggerContractError) as error:
        rigid_links._validate_authored_aggregate(
            stage,
            link,
            expected_members=None,
        )

    assert error.value.code == "authored_aggregate_mismatch"


def test_parallel_classification_honors_preexisting_authentication_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlreadyAborted:
        def is_set(self) -> bool:
            return True

        def set(self) -> None:
            pass

    monkeypatch.setattr(inference, "Event", AlreadyAborted)

    with pytest.raises(ModelAuthenticationFailure):
        inference._process_parallel(
            vlm=object(),
            entries=[{"id": "must-not-run"}],
            llm=object(),
            image_base_dir=None,
            system_prompt=None,
            invoke_kwargs=None,
            on_progress=None,
            on_error=None,
            on_result=None,
            on_prediction=None,
            max_workers=1,
            max_retries=1,
            output_key="class",
        )
