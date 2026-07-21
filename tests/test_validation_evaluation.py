# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared validation evaluation signal contracts."""

from pathlib import Path

from pydantic import BaseModel

from world_understanding.validation import (
    EvaluationCorrection,
    EvaluationFinding,
    EvaluationResult,
    EvaluationSubject,
    ValidationEvidence,
    material_self_evaluation_result_from_signals,
)


class _ExternalEvaluationModel(BaseModel):
    path: Path


class _UnsupportedEvaluationValue:
    pass


def test_material_self_evaluation_signals_convert_to_core_result() -> None:
    result = material_self_evaluation_result_from_signals(
        evaluation_signals={
            "schema_version": "material-self-evaluation-signals/v1",
            "prediction_analysis": {
                "status": "completed",
                "symmetry_violations": [
                    {
                        "prim_a": "/World/left_arm",
                        "prim_b": "/World/right_arm",
                        "material_a": "Plastic Black",
                        "material_b": "Car Paint Light Silver",
                        "suggested": "Plastic Black",
                    }
                ],
                "consistency_violations": [],
            },
            "visual_evaluation": {
                "status": "completed",
                "reference_image_paths": ["/tmp/reference.png"],
                "rendered_image_paths": ["/tmp/current.png"],
                "turntable_contact_sheet_image_paths": ["/tmp/contact.png"],
                "visual_grounding_image_paths": ["/tmp/labels.png"],
                "issues": [
                    "Left and right arm shells are not visually consistent.",
                ],
                "label_based_corrections": [
                    {
                        "label_ids": [6, 7],
                        "target_label_ids": [7],
                        "prim_paths": ["/World/right_arm"],
                        "issue": "Right arm should match the left arm.",
                        "suggested_material": "Plastic Black",
                    }
                ],
            },
            "visual_grounding": {
                "status": "completed",
                "packet_path": "/tmp/legend.json",
                "html_report_path": "/tmp/index.html",
            },
        },
        previous_prim_feedback={
            "/World/right_arm": "Use the same material as /World/left_arm.",
        },
        resolved_assignments={"/World/right_arm": "Plastic Black"},
    )

    dumped = result.model_dump(mode="json")

    assert dumped["schema_version"] == "evaluation-signals/v1"
    assert dumped["domain"] == "material_agent.material_assignment"
    assert dumped["status"] == "completed"
    assert {finding["code"] for finding in dumped["findings"]} == {
        "material.symmetry_mismatch",
        "material.visual_issue",
        "material.label_correction",
        "material.prim_feedback",
    }
    assert {item["kind"] for item in dumped["evidence_items"]} == {
        "reference_image",
        "current_render",
        "turntable_contact_sheet",
        "visual_grounding_overlay",
        "visual_grounding_packet",
        "visual_grounding_report",
    }
    assert {correction["subject"]["kind"] for correction in dumped["corrections"]} >= {
        "label_group",
        "prim",
    }


def test_evaluation_models_normalize_singletons_none_and_json_metadata() -> None:
    evidence = ValidationEvidence(kind="image", path="render.png")
    subject = EvaluationSubject(
        kind="asset",
        identifier="asset-1",
        metadata={
            "path": Path("asset.usda"),
            "evidence": evidence,
            "nested": {"path": Path("nested.usda")},
        },
    )
    correction = EvaluationCorrection(
        subject=EvaluationSubject(kind="asset", identifier="asset-2", metadata=None),
        field="material",
        suggested_value=None,
        rationale=None,
    )
    finding = EvaluationFinding(
        code="material.visual_issue",
        message="Render differs from reference.",
        evidence_items=evidence,
    )
    no_evidence_finding = EvaluationFinding(
        code="material.visual_issue",
        message="No evidence is still valid.",
        evidence_items=None,
    )

    empty = EvaluationResult(
        domain="material_agent.material_assignment",
        status="skipped",
        findings=None,
        corrections=None,
        evidence_items=None,
        metadata=None,
    )
    singleton = EvaluationResult(
        domain="material_agent.material_assignment",
        status="completed",
        findings=finding,
        corrections=correction,
        evidence_items=evidence,
    )

    assert subject.metadata == {
        "path": "asset.usda",
        "evidence": evidence.model_dump(mode="json"),
        "nested": {"path": "nested.usda"},
    }
    assert correction.suggested_value is None
    assert correction.rationale is None
    assert correction.subject.metadata == {}
    assert finding.evidence_items == (evidence,)
    assert no_evidence_finding.evidence_items == ()
    assert empty.findings == ()
    assert empty.corrections == ()
    assert empty.evidence_items == ()
    assert empty.metadata == {}
    assert singleton.findings == (finding,)
    assert singleton.corrections == (correction,)
    assert singleton.evidence_items == (evidence,)


def test_evaluation_json_normalizer_preserves_non_validation_models() -> None:
    evidence = ValidationEvidence(kind="image", path="render.png")
    external_model = _ExternalEvaluationModel(path=Path("model.usda"))
    unsupported = _UnsupportedEvaluationValue()

    subject = EvaluationSubject(
        kind="asset",
        identifier="asset-1",
        metadata={
            7: [
                Path("asset.usda"),
                evidence,
                external_model,
                None,
                "label",
                3,
                1.5,
                True,
                unsupported,
            ]
        },
    )

    assert subject.metadata["7"][:2] == [
        "asset.usda",
        evidence.model_dump(mode="json"),
    ]
    assert subject.metadata["7"][2] is external_model
    assert subject.metadata["7"][3:-1] == [None, "label", 3, 1.5, True]
    assert subject.metadata["7"][-1] is unsupported


def test_material_self_evaluation_handles_scalar_mapping_and_duplicate_edges() -> None:
    result = material_self_evaluation_result_from_signals(
        evaluation_signals={
            "schema_version": "material-self-evaluation-signals/v1",
            "prediction_analysis": {
                "status": "completed",
                "consistency_violations": {
                    "group_name": "Bolts",
                    "suggested": "Brushed Steel",
                },
            },
            "visual_evaluation": {
                "status": "completed",
                "reference_image_paths": "reference.png",
                "rendered_image_paths": ["current.png", "current.png"],
                "issues": ["Repeated issue", "Repeated issue", 123, "none"],
                "label_based_corrections": [
                    {
                        "target_label_ids": 7,
                        "issue": "",
                        "suggested_material": "unknown",
                    }
                ],
            },
        },
        previous_prim_feedback={
            "": "ignored",
            "/World/Bolt": "",
        },
        resolved_assignments={
            "": "ignored",
            "/World/Bolt": None,
        },
    )

    dumped = result.model_dump(mode="json")

    assert result.status == "completed"
    assert [item["path"] for item in dumped["evidence_items"]] == [
        "reference.png",
        "current.png",
    ]
    assert [finding["message"] for finding in dumped["findings"]] == [
        "Repeated material group is inconsistent: Bolts.",
        "Repeated issue",
        "123",
        "Label-grounded material issue for 7.",
    ]
    assert [correction["suggested_value"] for correction in dumped["corrections"]] == [
        "Brushed Steel"
    ]

    duplicate_corrections = material_self_evaluation_result_from_signals(
        evaluation_signals={
            "prediction_analysis": {
                "status": "completed",
                "consistency_violations": [
                    {"group_name": "Bolts", "suggested": "Brushed Steel"},
                    {"group_name": "Bolts", "suggested": "Brushed Steel"},
                ],
            },
            "visual_evaluation": {"status": "completed"},
        }
    )
    assert len(duplicate_corrections.corrections) == 1


def test_material_self_evaluation_reports_error_and_skipped_statuses() -> None:
    error = material_self_evaluation_result_from_signals(
        evaluation_signals={
            "prediction_analysis": {"status": "error"},
            "visual_evaluation": {"status": "skipped"},
        }
    )
    skipped = material_self_evaluation_result_from_signals(evaluation_signals={})

    assert error.status == "error"
    assert skipped.status == "skipped"


def test_material_self_evaluation_preserves_partial_status() -> None:
    result = material_self_evaluation_result_from_signals(
        evaluation_signals={
            "schema_version": "material-self-evaluation-signals/v1",
            "prediction_analysis": {"status": "partial"},
            "visual_evaluation": {"status": "skipped"},
        }
    )

    assert result.status == "partial"
