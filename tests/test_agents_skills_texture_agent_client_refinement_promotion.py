# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression contract for Texture Agent client refinement promotion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / ".agents/skills/fixed-pipeline/references/texture-agent-client"
SKILL_PATH = SKILL_ROOT / "reference.md"
VALIDATOR_PATH = SKILL_ROOT / "scripts/validate_refinement_promotion.py"


def _skill_contract() -> str:
    return " ".join(SKILL_PATH.read_text(encoding="utf-8").split())


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "texture_refinement_promotion_validator", VALIDATOR_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(tmp_path: Path, name: str, content: bytes) -> dict[str, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _render_contract(
    tmp_path: Path,
    name: str,
    *,
    source_usd_sha256: str,
    camera_path: str = "/RefinementCamera",
    renderer: str = "world-understanding-ovrtx",
) -> dict[str, str]:
    contract = {
        "schema_version": "texture-agent-ovrtx-render-contract.v1",
        "source_usd_sha256": source_usd_sha256,
        "render_path": "shared_ovrtx_usd",
        "camera": {"path": camera_path, "focal_length_mm": 50.0},
        "lighting": {"preset": "neutral_studio", "exposure": 0.0},
        "background": {"rgba": [0.18, 0.18, 0.18, 1.0]},
        "resolution": {"width": 1024, "height": 1024},
        "ovrtx_metadata": {
            "renderer": renderer,
            "rendered_usd_sha256": source_usd_sha256,
            "request_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "rendered_usd_representation": "flattened_usd",
        },
    }
    return _artifact(
        tmp_path,
        name,
        json.dumps(contract, sort_keys=True).encode("utf-8"),
    )


def _complete_receipt(tmp_path: Path, *, outcome: str, decision: str) -> dict[str, Any]:
    baseline = _artifact(tmp_path, "baseline.usdz", b"olive baseline")
    candidate = _artifact(tmp_path, "candidate.usdz", b"pale candidate")
    reference = _artifact(tmp_path, "reference.png", b"reference")
    baseline_render = _artifact(tmp_path, "baseline.png", b"olive render")
    candidate_render = _artifact(tmp_path, "candidate.png", b"pale render")
    base_color_render = _artifact(tmp_path, "base-color.png", b"base color")
    full_pbr_render = _artifact(tmp_path, "full-pbr.png", b"full pbr")
    accepted = candidate if decision == "promote_candidate" else baseline
    baseline_contract = _render_contract(
        tmp_path,
        "baseline-render-contract.json",
        source_usd_sha256=baseline["sha256"],
    )
    candidate_contract = _render_contract(
        tmp_path,
        "candidate-render-contract.json",
        source_usd_sha256=candidate["sha256"],
    )
    return {
        "schema_version": "texture-agent-refinement-promotion.v1",
        "decision": decision,
        "responsible_stage": "codex_postprocessing",
        "service_session_id": "d8aeb455-9b9e-43d3-84c0-924bbbf53754",
        "baseline": baseline,
        "candidate": candidate,
        "accepted_output": accepted,
        "comparison_status": "complete",
        "comparison": {
            "baseline_render": baseline_render,
            "candidate_render": candidate_render,
            "reference": reference,
            "baseline_render_contract": baseline_contract,
            "candidate_render_contract": candidate_contract,
            "material_findings": [
                {
                    "material": "BodyPaint",
                    "reference_fidelity": outcome,
                    "rationale": "Candidate is paler than the olive reference.",
                }
            ],
            "map_diagnostics": {
                "base_color_only_render": base_color_render,
                "full_pbr_render": full_pbr_render,
                "normal_map_assessment": "shading_only",
                "rationale": "The albedo regression remains without the normal map.",
            },
        },
    }


def _write_receipt(tmp_path: Path, payload: dict[str, Any]) -> Path:
    receipt_path = tmp_path / "refinement_promotion.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    return receipt_path


def test_refinement_requires_visual_evidence_before_candidate_promotion() -> None:
    skill = _skill_contract()

    required_contract = (
        "artifact readiness, not visual acceptance or reference fidelity",
        "Treat a later refinement as a candidate, not as an automatic replacement",
        "same camera, framing,",
        "shared OVRTX USD render path",
        "texture-agent-ovrtx-render-contract.v1",
        "base-color-only diagnostic and the full PBR result",
        "`promote_candidate`,",
        "`reject_keep_baseline`",
        "`blocked_keep_baseline`",
        "Never overwrite the accepted files before this decision",
        "validate_refinement_promotion.py",
    )
    for clause in required_contract:
        assert clause in skill, f"missing refinement promotion clause: {clause!r}"
    assert (
        ".agents/skills/fixed-pipeline/references/texture-agent-client/"
        "scripts/validate_refinement_promotion.py"
    ) in skill


def test_refinement_receipt_separates_service_and_postprocessing() -> None:
    skill = _skill_contract()

    assert "client/Codex post-processing" in skill
    assert "Do not attribute post-processing to the service" in skill
    assert "responsible stage was the service or client/Codex processing" in skill


def test_visual_regression_falls_back_to_prior_accepted_output() -> None:
    skill = _skill_contract()

    assert "Otherwise retain and return the prior accepted output" in skill
    assert "preserve the rejected candidate for diagnosis" in skill
    assert "A refinement is paler, brighter, or less reference-faithful" in skill


def test_regressed_candidate_receipt_keeps_digest_verified_baseline(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    receipt_path = _write_receipt(tmp_path, payload)

    result = validator.validate_receipt(payload, receipt_path=receipt_path)

    assert result["decision"] == "reject_keep_baseline"
    assert result["responsible_stage"] == "codex_postprocessing"
    assert result["accepted_sha256"] == payload["baseline"]["sha256"]


def test_regressed_candidate_cannot_be_promoted(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="promote_candidate",
    )
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(
        validator.ReceiptError,
        match="every material finding to be improved or preserved",
    ):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_improved_candidate_promotes_digest_verified_candidate(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    receipt_path = _write_receipt(tmp_path, payload)

    result = validator.validate_receipt(payload, receipt_path=receipt_path)

    assert result["decision"] == "promote_candidate"
    assert result["accepted_sha256"] == payload["candidate"]["sha256"]


def test_rejection_cannot_publish_candidate_digest(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    payload["accepted_output"] = payload["candidate"]
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="match the baseline digest"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_candidate_promotion_requires_matched_render_contracts(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    contract_artifact = payload["comparison"]["candidate_render_contract"]
    contract_path = tmp_path / contract_artifact["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["camera"]["path"] = "/DifferentCamera"
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    contract_path.write_bytes(encoded)
    contract_artifact["sha256"] = hashlib.sha256(encoded).hexdigest()
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(
        validator.ReceiptError,
        match="verified OVRTX render contracts must match",
    ):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_equal_fabricated_render_contract_digests_are_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    payload["comparison"]["baseline_render_contract"]["sha256"] = "0" * 64
    payload["comparison"]["candidate_render_contract"]["sha256"] = "0" * 64
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="does not match"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_blocked_comparison_keeps_baseline_without_claiming_visual_success(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="blocked",
        decision="reject_keep_baseline",
    )
    payload["decision"] = "blocked_keep_baseline"
    payload["comparison_status"] = "blocked"
    payload["blockers"] = ["Candidate same-camera render is unavailable."]
    payload.pop("comparison")
    receipt_path = _write_receipt(tmp_path, payload)

    result = validator.validate_receipt(payload, receipt_path=receipt_path)

    assert result["decision"] == "blocked_keep_baseline"
    assert result["accepted_sha256"] == payload["baseline"]["sha256"]


def test_blocked_receipt_cannot_embed_completed_comparison(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    payload["decision"] = "blocked_keep_baseline"
    payload["comparison_status"] = "blocked"
    payload["blockers"] = ["OVRTX candidate render is unavailable."]
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="cannot contain a completed"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_complete_receipt_cannot_include_blockers(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    payload["blockers"] = ["Contradicts a complete comparison."]
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="cannot contain blockers"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_complete_comparison_cannot_contain_blocked_material(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="blocked",
        decision="reject_keep_baseline",
    )
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="blocked material findings"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_complete_comparison_cannot_contain_blocked_normal_assessment(
    tmp_path: Path,
) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    payload["comparison"]["map_diagnostics"]["normal_map_assessment"] = "blocked"
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="blocked normal-map assessment"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


@pytest.mark.parametrize("unsafe_path_kind", ["absolute", "parent_traversal"])
def test_artifacts_must_remain_inside_receipt_directory(
    tmp_path: Path,
    unsafe_path_kind: str,
) -> None:
    validator = _load_validator()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    payload = _complete_receipt(
        receipt_dir,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    outside = tmp_path / "outside.usdz"
    outside.write_bytes(b"outside")
    unsafe_path = (
        str(outside.resolve()) if unsafe_path_kind == "absolute" else "../outside.usdz"
    )
    payload["baseline"] = {
        "path": unsafe_path,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    receipt_path = _write_receipt(receipt_dir, payload)

    with pytest.raises(validator.ReceiptError, match="relative|escapes"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_symlinked_artifact_cannot_escape_receipt_directory(tmp_path: Path) -> None:
    validator = _load_validator()
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    payload = _complete_receipt(
        receipt_dir,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    outside = tmp_path / "outside.usdz"
    outside.write_bytes(b"outside")
    symlink = receipt_dir / "outside-link.usdz"
    symlink.symlink_to(outside)
    payload["baseline"] = {
        "path": symlink.name,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    receipt_path = _write_receipt(receipt_dir, payload)

    with pytest.raises(validator.ReceiptError, match="escapes"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_render_contract_must_bind_exact_source_usd(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    contract_artifact = payload["comparison"]["candidate_render_contract"]
    contract_path = tmp_path / contract_artifact["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["source_usd_sha256"] = payload["baseline"]["sha256"]
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    contract_path.write_bytes(encoded)
    contract_artifact["sha256"] = hashlib.sha256(encoded).hexdigest()
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="must match its rendered USD"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_render_contract_must_identify_ovrtx(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    contract_artifact = payload["comparison"]["candidate_render_contract"]
    contract_path = tmp_path / contract_artifact["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["ovrtx_metadata"]["renderer"] = "preview"
    encoded = json.dumps(contract, sort_keys=True).encode("utf-8")
    contract_path.write_bytes(encoded)
    contract_artifact["sha256"] = hashlib.sha256(encoded).hexdigest()
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="must identify OVRTX"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_duplicate_material_findings_are_rejected(tmp_path: Path) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    payload["comparison"]["material_findings"].append(
        dict(payload["comparison"]["material_findings"][0])
    )
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match="duplicate material finding"):
        validator.validate_receipt(payload, receipt_path=receipt_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", "texture-agent-refinement-promotion.v2", "schema_version"),
        ("responsible_stage", "unknown_stage", "responsible_stage"),
    ],
)
def test_unknown_schema_or_responsible_stage_fails_closed(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    validator = _load_validator()
    payload = _complete_receipt(
        tmp_path,
        outcome="improved",
        decision="promote_candidate",
    )
    payload[field] = value
    receipt_path = _write_receipt(tmp_path, payload)

    with pytest.raises(validator.ReceiptError, match=error):
        validator.validate_receipt(payload, receipt_path=receipt_path)


def test_validator_cli_reports_successful_receipt(tmp_path: Path) -> None:
    payload = _complete_receipt(
        tmp_path,
        outcome="regressed",
        decision="reject_keep_baseline",
    )
    receipt_path = _write_receipt(tmp_path, payload)

    completed = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), str(receipt_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert response["valid"] is True
    assert response["decision"] == "reject_keep_baseline"


def test_validator_cli_distinguishes_invalid_receipt_from_io_failure(
    tmp_path: Path,
) -> None:
    invalid_receipt = tmp_path / "invalid.json"
    invalid_receipt.write_text("{not-json", encoding="utf-8")

    invalid = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), str(invalid_receipt)],
        check=False,
        capture_output=True,
        text=True,
    )
    missing = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), str(tmp_path / "missing.json")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert invalid.returncode == 1
    assert json.loads(invalid.stdout)["error_type"] == "receipt"
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["error_type"] == "io"
