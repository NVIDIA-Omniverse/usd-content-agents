# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only bridge from native Physics evidence to Validation.

The bundle freezes evidence bytes, not a verdict. Acceptance still requires the
native typed assessment, clean measured runtime checks and the existing native
OVRTX attestation verifier. No approve/refine result is manufactured here.
This is a bounded single-asset runtime/visual-review contract, not a certificate
of arbitrary user-task behavior, geometry fidelity or physical realism.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "validation.native-physics-behavior-bundle.v1"
ASSESSMENT_SCHEMA = "content-agent-workflows.physics-behavior-assessment.v1"
EVIDENCE_SCHEMA = "content-agent-workflows.validation-evidence.v1"
NATIVE_ROLES = frozenset(
    {
        "native_physics_bundle",
        "physics_behavior_assessment",
        "physics_validation_evidence",
    }
)


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Native evidence JSON exceeds 16 MiB.")

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Native evidence must be a JSON object.")
    return value


def _path(value: Any, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Native evidence path is missing.")
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _native_closure(
    assessment_path: Path, validation_path: Path, run_dir: Path
) -> tuple[Any, dict[str, Any], dict[str, Any], set[Path]]:
    # Import only on this explicitly selected native route. These producer-owned
    # helpers perform reads/attestation only; they launch no model or simulator.
    from content_agent_workflows.common.validation_evidence import ValidationEvidence
    from content_agent_workflows.physics.workflow import PhysicsBehaviorAssessment
    from content_workflow_cli.runner import (
        _capture_physics_visual_evidence_attestation,
        _physics_ovrtx_review_evidence_errors,
        _verify_physics_assessment_evidence_paths,
    )

    raw = _json(assessment_path)
    _check(
        raw.get("schema_version") == ASSESSMENT_SCHEMA,
        "Unsupported native assessment schema.",
    )
    # Require the serialized typed shape, before the producer's permissive LLM
    # normalization/defaults can erase missing or unknown fields.
    expected = {
        "schema_version",
        "status",
        "checked_views",
        "runtime_report",
        "rendered_frames",
        "issues_found",
        "issues_fixed",
        "unresolved_issues",
        "assessment_notes",
    }
    _check(
        set(raw) == expected, "Native assessment is incomplete or has unknown fields."
    )
    for key in (
        "checked_views",
        "rendered_frames",
        "issues_found",
        "issues_fixed",
        "unresolved_issues",
    ):
        _check(
            isinstance(raw[key], list), "Native assessment has a non-list field: " + key
        )
    assessment = PhysicsBehaviorAssessment.model_validate(raw)
    validation = _json(validation_path)
    _check(
        validation.get("schema_version") == EVIDENCE_SCHEMA,
        "Unsupported native ValidationEvidence schema.",
    )
    ValidationEvidence.model_validate(validation)
    _check(
        assessment.status in {"pass", "fixed"} and not assessment.unresolved_issues,
        "Native visual assessment is unresolved or failed.",
    )
    _check(
        bool(assessment.checked_views) and bool(assessment.rendered_frames),
        "Native visual review has no reviewed views/frames.",
    )
    _check(
        validation.get("workflow") == "physics_authoring"
        and validation.get("target_runtime") == "ovphysx",
        "Unsupported native workflow/runtime.",
    )
    _check(validation.get("sim_ready_status") == "pass", "Native runtime is not pass.")
    for key in ("failures", "warnings", "unresolved_issues"):
        _check(validation.get(key) == [], f"Native {key} are missing or nonempty.")
    checks = validation.get("checks", [])
    _check(isinstance(checks, list) and bool(checks), "Missing native checks.")
    _check(
        {
            "physics_properties",
            "runtime_loadability",
            "no_explosions",
            "simulation_visual_review",
        }
        <= {c.get("name") for c in checks},
        "Required native checks are missing.",
    )
    for check in checks:
        _check(
            check.get("status") == "pass"
            and not check.get("failures")
            and not check.get("warnings"),
            "Native check is not a clean pass.",
        )
    report_path = _path(assessment.runtime_report, assessment_path.parent)
    report = _json(report_path)
    _check(
        report.get("engine") == "ovphysx"
        and report.get("failures") == []
        and report.get("warnings") == [],
        "Measured runtime report is not clean.",
    )
    artifacts = list(validation.get("evidence_artifacts", []))
    for check in checks:
        artifacts += check.get("evidence_artifacts", [])
    for kind, path in (
        ("physics_behavior_assessment", assessment_path),
        ("runtime_report", report_path),
    ):
        _check(
            any(
                a.get("kind") == kind
                and _path(a.get("path"), validation_path.parent) == path
                for a in artifacts
            ),
            "Native validation does not reference " + kind,
        )
    visual = [c for c in checks if c["name"] == "simulation_visual_review"]
    _check(
        len(visual) == 1
        and visual[0].get("metadata", {}).get("assessment_status") == assessment.status,
        "Native visual check contradicts its assessment.",
    )
    receipts = {
        _path(a["path"], validation_path.parent)
        for a in artifacts
        if a.get("kind") == "simulation_frame_receipt"
    }
    _check(len(receipts) == 1, "Expected one native frame receipt.")
    receipt_path = next(iter(receipts))
    for artifact in artifacts:
        if artifact.get("kind") == "simulation_frame_receipt":
            _check(
                artifact.get("metadata", {}).get("sha256")
                == _binding(receipt_path)["sha256"],
                "Stale native frame receipt.",
            )
    receipt = _json(receipt_path)
    raw_render_path = (
        run_dir / "raw" / f"physics_render_frames_{receipt['iteration']}.json"
    )
    _check(
        _path(receipt.get("response_artifact", {}).get("path"), receipt_path.parent)
        == raw_render_path.resolve(strict=True),
        "Native review render provenance is outside the bound response.",
    )
    attestation = _capture_physics_visual_evidence_attestation(
        run_dir=run_dir,
        validation_path=validation_path,
        runtime_report_path=report_path,
        rendered_frames=list(assessment.rendered_frames),
        render_receipt_path=receipt_path,
    )
    _verify_physics_assessment_evidence_paths(assessment, attestation)
    errors = _physics_ovrtx_review_evidence_errors(
        run_dir=run_dir,
        iteration=receipt["iteration"],
        runtime_report_path=report_path,
        rendered_frames=list(assessment.rendered_frames),
        assessment=assessment,
    )
    _check(not errors, "Native OVRTX evidence invalid: " + "; ".join(errors))
    closure = {
        assessment_path,
        validation_path,
        report_path,
        *(item[0] for item in attestation.artifacts),
    }
    for key in (
        "physics_usd",
        "scene_usd",
        "recording_usda",
        "trajectory_jsonl",
        "response_path",
    ):
        closure.add(_path(report.get(key), report_path.parent))
    response_path = _path(report["response_path"], report_path.parent)
    response = _json(response_path)
    raw_report_path = _path(response.get("usd_cli_report_path"), response_path.parent)
    closure.add(raw_report_path)
    raw_report = _json(raw_report_path)
    _check(
        raw_report.get("engine") == "ovphysx",
        "Native solver report has the wrong engine.",
    )
    for key in ("scene_usd", "recording_usda", "trajectory_jsonl"):
        _check(
            _path(raw_report.get(key), raw_report_path.parent)
            == _path(report[key], report_path.parent),
            "Native solver report input mismatch: " + key,
        )
    facts = raw_report.get("simulation_facts", {})
    _check(
        facts.get("trajectory_finite") is True
        and facts.get("trajectory_sample_count")
        == response.get("trajectory_sample_count")
        and facts.get("reported_step_count") == response.get("n_steps")
        and facts.get("reported_body_count") == response.get("n_bodies"),
        "Native solver facts are incomplete or contradictory.",
    )
    for key in ("physics_usd", "scene_usd", "recording_usda"):
        _check(
            _path(
                receipt.get("render_inputs", {}).get(key, {}).get("path"),
                receipt_path.parent,
            )
            == _path(report[key], report_path.parent),
            "Render/runtime input mismatch: " + key,
        )
    _check(
        response.get("engine") == "ovphysx" and response.get("status") == "ok",
        "Runtime execution did not succeed.",
    )
    _check(
        type(response.get("n_steps")) is int and response["n_steps"] > 0,
        "Runtime has no executed steps.",
    )
    summary = report.get("summary", {})
    _check(
        summary.get("initial_pose_discontinuity") is False,
        "Runtime initial pose is unresolved.",
    )
    _check(
        response.get("n_bodies")
        == summary.get("loaded_body_count")
        == report.get("acceptance", {}).get("expected_body_count")
        == 1,
        "Native single-body runtime coverage is incomplete.",
    )
    trajectory = _path(report["trajectory_jsonl"], report_path.parent)
    _check(
        trajectory.stat().st_size <= 128 * 1024 * 1024,
        "Native trajectory exceeds the bounded reader limit.",
    )
    rows = [
        json.loads(line) for line in trajectory.read_text().splitlines() if line.strip()
    ]
    _check(
        len(rows) >= 2
        and len(rows)
        == summary.get("n_samples")
        == response.get("trajectory_sample_count"),
        "Incomplete runtime trajectory.",
    )
    times = []
    for index, row in enumerate(rows):
        _check(row.get("frame") == index, "Trajectory frames are incomplete.")
        _check(
            len(row.get("pose", [])) == 7 and len(row.get("vel", [])) == 6,
            "Trajectory state is incomplete.",
        )
        numbers = [row.get("t"), *row["pose"], *row["vel"]]
        _check(
            all(type(n) in (int, float) and math.isfinite(n) for n in numbers),
            "Non-finite trajectory state.",
        )
        times.append(row["t"])
    _check(
        times[0] == 0 and all(b > a for a, b in zip(times, times[1:], strict=False)),
        "Trajectory time coverage is invalid.",
    )
    duration = validation.get("metadata", {}).get("duration_s")
    _check(
        type(duration) in (int, float)
        and duration > 0
        and math.isclose(times[-1], duration, abs_tol=1e-8)
        and math.isclose(summary.get("duration_s", -1), duration, abs_tol=1e-8),
        "Trajectory does not cover the requested runtime duration.",
    )
    return assessment, validation, report, closure


def prepare_native_physics_bundle(
    *, asset: Path, assessment: Path, validation_evidence: Path, run_dir: Path
) -> dict[str, Any]:
    """Freeze verified native input bytes; callers retain this before Validation.

    No provider calls, simulations, success records or source mutations occur.
    Incomplete/failed producer evidence cannot produce a usable bundle.
    """
    assessment, validation_evidence, run_dir, asset = (
        p.resolve(strict=True)
        for p in (assessment, validation_evidence, run_dir, asset)
    )
    review, validation, report, closure = _native_closure(
        assessment, validation_evidence, run_dir
    )
    report_parent = _path(review.runtime_report, assessment.parent).parent
    _check(
        _path(validation.get("asset"), validation_evidence.parent) == asset
        and _path(report.get("physics_usd"), report_parent) == asset,
        "Native asset identity mismatch.",
    )
    _check(
        validation.get("metadata", {}).get("asset_sha256") == _binding(asset)["sha256"],
        "Native asset digest is missing or stale.",
    )
    closure.add(asset)
    return {
        "schema_version": BUNDLE_SCHEMA,
        "run_dir": str(run_dir),
        "asset": str(asset),
        "assessment": str(assessment),
        "validation_evidence": str(validation_evidence),
        "artifacts": [_binding(p) for p in sorted(closure)],
    }


def validate_native_physics_bundle(
    bundle_path: Path, *, usd_paths: Iterable[Path | str]
) -> dict[str, Any]:
    """Return passed/failed directly, never a legacy judge decision."""
    try:
        original_bundle = _binding(bundle_path)
        bundle = _json(bundle_path)
        _check(
            bundle.get("schema_version") == BUNDLE_SCHEMA,
            "Unsupported native bundle schema.",
        )
        paths = {_path(str(p), Path.cwd()) for p in usd_paths}
        asset = _path(bundle.get("asset"), bundle_path.parent)
        _check(
            paths == {asset},
            "Native bundle must cover exactly the requested USD asset.",
        )
        expected = bundle.get("artifacts")
        _check(
            isinstance(expected, list) and bool(expected),
            "Native evidence digest closure is missing.",
        )
        seen = set()
        for item in expected:
            path = _path(item.get("path"), bundle_path.parent)
            _check(path not in seen, "Duplicate native evidence binding.")
            seen.add(path)
            _check(_binding(path) == item, "Native evidence is stale: " + str(path))
        rebuilt = prepare_native_physics_bundle(
            asset=asset,
            assessment=_path(bundle.get("assessment"), bundle_path.parent),
            validation_evidence=_path(
                bundle.get("validation_evidence"), bundle_path.parent
            ),
            run_dir=_path(bundle.get("run_dir"), bundle_path.parent),
        )
        _check(rebuilt == bundle, "Native evidence closure is incomplete or changed.")
        _check(
            _binding(bundle_path) == original_bundle,
            "Native bundle changed during verification.",
        )
        return {
            "status": "passed",
            "kind": "native_physics_behavior",
            "bundle_sha256": _binding(bundle_path)["sha256"],
            "asset_sha256": _binding(asset)["sha256"],
            "native_assessment_status": _json(Path(bundle["assessment"]))["status"],
            "scope": "single_body_native_runtime_and_rendered_behavior_review",
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
        ImportError,
    ) as exc:
        return {
            "status": "failed",
            "kind": "native_physics_behavior",
            "reason": str(exc),
            "error_type": type(exc).__name__,
        }


def native_behavior_result(
    evidence: Iterable[Any], *, usd_paths: Iterable[Path | str]
) -> dict[str, Any] | None:
    """Detect native evidence before legacy status/decision heuristics."""
    native = []
    bundles = []
    for item in evidence:
        payload = {}
        if item.kind == "simulation_json" and item.exists:
            try:
                payload = _json(Path(item.path))
            except (OSError, ValueError):
                pass
        if item.role in NATIVE_ROLES or payload.get("schema_version") in {
            BUNDLE_SCHEMA,
            ASSESSMENT_SCHEMA,
            EVIDENCE_SCHEMA,
        }:
            native.append(item)
            if payload.get("schema_version") == BUNDLE_SCHEMA:
                bundles.append(item)
    if not native:
        return None
    if len(bundles) != 1:
        return {
            "status": "failed",
            "kind": "native_physics_behavior",
            "reason": "Native evidence requires exactly one digest-bound native_physics_bundle; raw assessment status is insufficient.",
        }
    result = validate_native_physics_bundle(Path(bundles[0].path), usd_paths=usd_paths)
    if result["status"] == "passed":
        bound = {
            _path(a["path"], Path(bundles[0].path).parent)
            for a in _json(Path(bundles[0].path))["artifacts"]
        }
        if any(
            Path(item.path).resolve() not in bound
            for item in native
            if item is not bundles[0]
        ):
            return {
                "status": "failed",
                "kind": "native_physics_behavior",
                "reason": "Additional native evidence is outside the validated closure.",
            }
    return result
