#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a digest-bound Texture refinement promotion receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "texture-agent-refinement-promotion.v1"
RENDER_CONTRACT_SCHEMA_VERSION = "texture-agent-ovrtx-render-contract.v1"
OVRTX_RENDER_PATH = "shared_ovrtx_usd"
DECISIONS = {
    "promote_candidate",
    "reject_keep_baseline",
    "blocked_keep_baseline",
}
RESPONSIBLE_STAGES = {"service", "client_workflow", "codex_postprocessing"}
FIDELITY_OUTCOMES = {"improved", "preserved", "regressed", "blocked"}
NORMAL_MAP_ASSESSMENTS = {
    "not_responsible",
    "shading_only",
    "albedo_or_authoring_interaction",
    "not_applicable",
    "blocked",
}
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ReceiptError(ValueError):
    """Raised when a promotion receipt cannot authorize its claimed decision."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{field} must be an object")
    return value


def _nonempty_mapping(value: Any, field: str) -> Mapping[str, Any]:
    mapping = _mapping(value, field)
    if not mapping:
        raise ReceiptError(f"{field} must be a non-empty object")
    return mapping


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiptError(f"{field} must be a non-empty string")
    return value.strip()


def _digest(value: Any, field: str) -> str:
    digest = _text(value, field)
    if not DIGEST_PATTERN.fullmatch(digest):
        raise ReceiptError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise ReceiptError(f"artifact changed while it was being hashed: {path}")
    return digest.hexdigest()


def _canonical_json_sha256(value: Any, field: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptError(f"{field} must contain JSON-compatible values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _artifact(
    value: Any,
    field: str,
    *,
    receipt_dir: Path,
) -> tuple[Path, str]:
    artifact = _mapping(value, field)
    raw_path = _text(artifact.get("path"), f"{field}.path")
    expected_digest = _digest(artifact.get("sha256"), f"{field}.sha256")
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ReceiptError(f"{field}.path must be relative to the receipt")
    path = receipt_dir / relative_path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ReceiptError(f"{field}.path is unavailable: {path}") from exc
    if not resolved.is_relative_to(receipt_dir):
        raise ReceiptError(f"{field}.path escapes the receipt directory: {raw_path}")
    if not resolved.is_file():
        raise ReceiptError(f"{field}.path is not a regular file: {resolved}")
    actual_digest = _sha256_file(resolved)
    if actual_digest != expected_digest:
        raise ReceiptError(
            f"{field}.sha256 does not match {resolved}: "
            f"expected {expected_digest}, got {actual_digest}; the artifact may "
            "have changed after receipt creation or the receipt digest is incorrect"
        )
    return resolved, actual_digest


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReceiptError(f"{field} must be a positive integer")
    return int(value)


def _render_contract(
    value: Any,
    field: str,
    *,
    receipt_dir: Path,
    expected_source_digest: str,
) -> str:
    contract_path, _ = _artifact(value, field, receipt_dir=receipt_dir)
    try:
        contract = _mapping(
            json.loads(contract_path.read_text(encoding="utf-8")), field
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"{field}.path must contain valid JSON: {exc}") from exc

    if contract.get("schema_version") != RENDER_CONTRACT_SCHEMA_VERSION:
        raise ReceiptError(
            f"{field}.schema_version must be {RENDER_CONTRACT_SCHEMA_VERSION!r}"
        )
    source_digest = _digest(
        contract.get("source_usd_sha256"), f"{field}.source_usd_sha256"
    )
    if source_digest != expected_source_digest:
        raise ReceiptError(
            f"{field}.source_usd_sha256 must match its rendered USD artifact"
        )
    render_path = _text(contract.get("render_path"), f"{field}.render_path")
    if render_path != OVRTX_RENDER_PATH:
        raise ReceiptError(f"{field}.render_path must identify {OVRTX_RENDER_PATH!r}")

    camera = _nonempty_mapping(contract.get("camera"), f"{field}.camera")
    lighting = _nonempty_mapping(contract.get("lighting"), f"{field}.lighting")
    background = _nonempty_mapping(contract.get("background"), f"{field}.background")
    resolution = _mapping(contract.get("resolution"), f"{field}.resolution")
    width = _positive_int(resolution.get("width"), f"{field}.resolution.width")
    height = _positive_int(resolution.get("height"), f"{field}.resolution.height")

    ovrtx_metadata = _nonempty_mapping(
        contract.get("ovrtx_metadata"), f"{field}.ovrtx_metadata"
    )
    renderer = _text(ovrtx_metadata.get("renderer"), f"{field}.ovrtx_metadata.renderer")
    if "ovrtx" not in renderer.lower():
        raise ReceiptError(f"{field}.ovrtx_metadata.renderer must identify OVRTX")
    _digest(
        ovrtx_metadata.get("rendered_usd_sha256"),
        f"{field}.ovrtx_metadata.rendered_usd_sha256",
    )
    _digest(
        ovrtx_metadata.get("request_sha256"),
        f"{field}.ovrtx_metadata.request_sha256",
    )
    if ovrtx_metadata.get("error"):
        raise ReceiptError(f"{field}.ovrtx_metadata reports a render error")

    shared_evidence = {
        "render_path": render_path,
        "camera": camera,
        "lighting": lighting,
        "background": background,
        "resolution": {"width": width, "height": height},
        "renderer": renderer,
        "rendered_usd_representation": _text(
            ovrtx_metadata.get("rendered_usd_representation"),
            f"{field}.ovrtx_metadata.rendered_usd_representation",
        ),
    }
    return _canonical_json_sha256(shared_evidence, f"{field}.shared_evidence")


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ReceiptError(f"{field} must be a non-empty list")
    return tuple(_text(item, f"{field}[{index}]") for index, item in enumerate(value))


def validate_receipt(payload: Any, *, receipt_path: Path) -> dict[str, str]:
    """Validate receipt artifacts and return its authorized accepted output."""
    receipt = _mapping(payload, "receipt")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ReceiptError(f"schema_version must be {SCHEMA_VERSION!r}")

    decision = _text(receipt.get("decision"), "decision")
    if decision not in DECISIONS:
        raise ReceiptError(f"decision must be one of {sorted(DECISIONS)}")
    responsible_stage = _text(receipt.get("responsible_stage"), "responsible_stage")
    if responsible_stage not in RESPONSIBLE_STAGES:
        raise ReceiptError(
            f"responsible_stage must be one of {sorted(RESPONSIBLE_STAGES)}"
        )
    service_session_id = _text(receipt.get("service_session_id"), "service_session_id")

    try:
        receipt_dir = receipt_path.parent.resolve(strict=True)
    except OSError as exc:
        raise ReceiptError("receipt directory is unavailable") from exc
    baseline_path, baseline_digest = _artifact(
        receipt.get("baseline"), "baseline", receipt_dir=receipt_dir
    )
    candidate_path, candidate_digest = _artifact(
        receipt.get("candidate"), "candidate", receipt_dir=receipt_dir
    )
    accepted_path, accepted_digest = _artifact(
        receipt.get("accepted_output"),
        "accepted_output",
        receipt_dir=receipt_dir,
    )

    expected_accepted_digest = (
        candidate_digest if decision == "promote_candidate" else baseline_digest
    )
    if accepted_digest != expected_accepted_digest:
        expected_name = "candidate" if decision == "promote_candidate" else "baseline"
        raise ReceiptError(
            f"{decision} requires accepted_output to match the {expected_name} digest"
        )

    comparison_status = _text(receipt.get("comparison_status"), "comparison_status")
    if comparison_status not in {"complete", "blocked"}:
        raise ReceiptError("comparison_status must be 'complete' or 'blocked'")

    if decision == "blocked_keep_baseline":
        if comparison_status != "blocked":
            raise ReceiptError(
                "blocked_keep_baseline requires comparison_status='blocked'"
            )
        if "comparison" in receipt:
            raise ReceiptError(
                "blocked_keep_baseline cannot contain a completed comparison"
            )
        _string_list(receipt.get("blockers"), "blockers")
    else:
        if comparison_status != "complete":
            raise ReceiptError(f"{decision} requires comparison_status='complete'")
        if "blockers" in receipt:
            raise ReceiptError(f"{decision} cannot contain blockers")
        comparison = _mapping(receipt.get("comparison"), "comparison")
        _artifact(
            comparison.get("baseline_render"),
            "comparison.baseline_render",
            receipt_dir=receipt_dir,
        )
        _artifact(
            comparison.get("candidate_render"),
            "comparison.candidate_render",
            receipt_dir=receipt_dir,
        )
        _artifact(
            comparison.get("reference"),
            "comparison.reference",
            receipt_dir=receipt_dir,
        )
        baseline_contract = _render_contract(
            comparison.get("baseline_render_contract"),
            "comparison.baseline_render_contract",
            receipt_dir=receipt_dir,
            expected_source_digest=baseline_digest,
        )
        candidate_contract = _render_contract(
            comparison.get("candidate_render_contract"),
            "comparison.candidate_render_contract",
            receipt_dir=receipt_dir,
            expected_source_digest=candidate_digest,
        )
        if baseline_contract != candidate_contract:
            raise ReceiptError(
                "baseline and candidate verified OVRTX render contracts must match "
                "exactly"
            )

        findings = comparison.get("material_findings")
        if not isinstance(findings, list) or not findings:
            raise ReceiptError("comparison.material_findings must be a non-empty list")
        materials: set[str] = set()
        outcomes: list[str] = []
        for index, raw_finding in enumerate(findings):
            field = f"comparison.material_findings[{index}]"
            finding = _mapping(raw_finding, field)
            material = _text(finding.get("material"), f"{field}.material")
            if material in materials:
                raise ReceiptError(f"duplicate material finding: {material}")
            materials.add(material)
            outcome = _text(
                finding.get("reference_fidelity"), f"{field}.reference_fidelity"
            )
            if outcome not in FIDELITY_OUTCOMES:
                raise ReceiptError(
                    f"{field}.reference_fidelity must be one of "
                    f"{sorted(FIDELITY_OUTCOMES)}"
                )
            _text(finding.get("rationale"), f"{field}.rationale")
            outcomes.append(outcome)

        diagnostics = _mapping(
            comparison.get("map_diagnostics"), "comparison.map_diagnostics"
        )
        _artifact(
            diagnostics.get("base_color_only_render"),
            "comparison.map_diagnostics.base_color_only_render",
            receipt_dir=receipt_dir,
        )
        _artifact(
            diagnostics.get("full_pbr_render"),
            "comparison.map_diagnostics.full_pbr_render",
            receipt_dir=receipt_dir,
        )
        normal_assessment = _text(
            diagnostics.get("normal_map_assessment"),
            "comparison.map_diagnostics.normal_map_assessment",
        )
        if normal_assessment not in NORMAL_MAP_ASSESSMENTS:
            raise ReceiptError(
                "comparison.map_diagnostics.normal_map_assessment must be one of "
                f"{sorted(NORMAL_MAP_ASSESSMENTS)}"
            )
        _text(
            diagnostics.get("rationale"),
            "comparison.map_diagnostics.rationale",
        )

        if "blocked" in outcomes:
            raise ReceiptError(
                "complete comparison cannot contain blocked material findings; "
                "use blocked_keep_baseline with blockers"
            )
        if normal_assessment == "blocked":
            raise ReceiptError(
                "complete comparison cannot contain a blocked normal-map "
                "assessment; use blocked_keep_baseline with blockers"
            )

        if decision == "promote_candidate":
            if any(outcome not in {"improved", "preserved"} for outcome in outcomes):
                raise ReceiptError(
                    "promote_candidate requires every material finding to be "
                    "improved or preserved"
                )
        elif decision == "reject_keep_baseline":
            if "regressed" not in outcomes:
                raise ReceiptError(
                    "reject_keep_baseline requires at least one regressed material "
                    "finding"
                )
        else:  # DECISIONS is validated above; keep future additions fail-closed.
            raise ReceiptError(f"unsupported complete-comparison decision: {decision}")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "responsible_stage": responsible_stage,
        "service_session_id": service_session_id,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "accepted_output": str(accepted_path),
        "accepted_sha256": accepted_digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    try:
        raw_payload = receipt_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            json.dumps(
                {"valid": False, "error_type": "io", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    try:
        payload = json.loads(raw_payload)
        result = validate_receipt(payload, receipt_path=receipt_path)
    except (json.JSONDecodeError, ReceiptError) as exc:
        print(
            json.dumps(
                {"valid": False, "error_type": "receipt", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"valid": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
