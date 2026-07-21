# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for evidence-backed SimReady GSP.001 repair plans."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import content_agent_workflows.simready.conform_profile as conform_profile_module
from content_agent_workflows.simready import (
    SimReadyConformanceInput,
    run_simready_profile_conformance,
)
from content_agent_workflows.simready.models import (
    SIMREADY_GRASP_PLAN_SCHEMA_VERSION,
)


@pytest.fixture(autouse=True)
def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        conform_profile_module,
        "resolve_simready_runtime",
        lambda **_kwargs: SimpleNamespace(
            foundation_root=None,
            foundation_commit=None,
            foundation_spec_root=None,
            warnings=[],
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_asset(
    path: Path,
    *,
    valid_grasp: bool = False,
    invalid_grasp: bool = False,
    conflicting_grasp_prim: bool = False,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    Sdf = pytest.importorskip("pxr.Sdf")

    stage = Usd.Stage.CreateNew(str(path))
    robot = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(robot)
    robot.CreateAttribute("test:marker", Sdf.ValueTypeNames.String, custom=True).Set(
        "preserve-me"
    )
    UsdGeom.Xform.Define(stage, "/robot/handles")
    mesh = UsdGeom.Mesh.Define(stage, "/robot/body")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    if valid_grasp or invalid_grasp:
        grasp = UsdGeom.BasisCurves.Define(stage, "/robot/grasp_identifier_existing")
        grasp.CreateTypeAttr(UsdGeom.Tokens.linear)
        grasp.CreateCurveVertexCountsAttr([2 if valid_grasp else 1])
        grasp.CreatePointsAttr([(0, 0, 0), (0, 0, 1)] if valid_grasp else [(0, 0, 0)])
        grasp.CreateWidthsAttr([0.01])
    if conflicting_grasp_prim:
        stage.DefinePrim("/robot/handles/grasp_identifier_conflict", "Xform")
    assert stage.GetRootLayer().Save()


def _plan_payload(asset: Path) -> dict[str, object]:
    return {
        "schema_version": SIMREADY_GRASP_PLAN_SCHEMA_VERSION,
        "source_asset_sha256": _sha256(asset),
        "default_prim_path": "/robot",
        "provenance": {
            "source": "owner_approved_plan",
            "approved_by": "simready-owner@example.com",
            "evidence": ["review://fixture/robot-handle-v1"],
        },
        "grasp_lines": [
            {
                "prim_path": "/robot/handles/grasp_identifier_primary",
                "coordinate_space": "local",
                "points": [[0, 0, 0], [0, 0, 1]],
                "widths": [0.01],
            }
        ],
    }


def _write_plan(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(
    *,
    asset: Path,
    output_dir: Path,
    plan: Path | None,
    source_asset: Path | None = None,
):
    return run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(output_dir),
            repair_requirements=["GSP.001"],
            grasp_plan_path=str(plan) if plan is not None else None,
            source_asset=str(source_asset) if source_asset is not None else None,
            force=True,
        )
    )


def _receipt(report) -> dict[str, object]:
    return json.loads(Path(report.reports["GSP.001"]).read_text(encoding="utf-8"))


def _assert_blocked(report, *, asset: Path, original_bytes: bytes) -> None:
    assert not report.passed
    assert report.status == "BLOCKED"
    assert report.requirements_repaired == []
    assert report.requirements_blocked == ["GSP.001"]
    assert asset.read_bytes() == original_bytes
    publish_root = Path(report.output_dir) / conform_profile_module.GSP001_OUTPUT_DIR
    assert not publish_root.exists() or list(publish_root.iterdir()) == []


def test_gsp001_authors_multiple_local_lines_deterministically_with_receipt(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    plan_payload = _plan_payload(asset)
    grasp_lines = plan_payload["grasp_lines"]
    assert isinstance(grasp_lines, list)
    grasp_lines.append(
        {
            "prim_path": "/robot/grasp_identifier_secondary",
            "coordinate_space": "local",
            "points": [[1, 0, 0], [1, 1, 0], [1, 2, 0]],
            "widths": [0.02, 0.03, 0.04],
        }
    )
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, plan_payload)

    first = _run(asset=asset, output_dir=tmp_path / "first", plan=plan)
    second = _run(asset=asset, output_dir=tmp_path / "second", plan=plan)

    assert first.passed and second.passed
    assert first.requirements_repaired == ["GSP.001"]
    assert asset.read_bytes() == source_bytes
    assert (
        Path(first.output_usd_path).read_bytes()
        == Path(second.output_usd_path).read_bytes()
    )

    stage = Usd.Stage.Open(first.output_usd_path)
    assert stage is not None
    assert (
        stage.GetPrimAtPath("/robot").GetAttribute("test:marker").Get() == "preserve-me"
    )
    primary = UsdGeom.BasisCurves(
        stage.GetPrimAtPath("/robot/handles/grasp_identifier_primary")
    )
    secondary = UsdGeom.BasisCurves(
        stage.GetPrimAtPath("/robot/grasp_identifier_secondary")
    )
    assert list(primary.GetCurveVertexCountsAttr().Get()) == [2]
    assert len(primary.GetPointsAttr().Get()) == 2
    assert len(primary.GetWidthsAttr().Get()) == 1
    assert primary.GetWidthsInterpolation() == UsdGeom.Tokens.constant
    assert len(primary.GetExtentAttr().Get()) == 2
    assert list(secondary.GetCurveVertexCountsAttr().Get()) == [3]
    assert len(secondary.GetPointsAttr().Get()) == 3
    assert len(secondary.GetWidthsAttr().Get()) == 3
    assert secondary.GetWidthsInterpolation() == UsdGeom.Tokens.vertex
    assert len(secondary.GetExtentAttr().Get()) == 2

    receipt = _receipt(first)
    assert (
        receipt["schema_version"]
        == conform_profile_module.GSP001_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["source_asset_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert receipt["grasp_plan_sha256"] == _sha256(plan)
    assert receipt["readback_verified"] is True
    assert receipt["unrelated_stage_preserved"] is True
    assert receipt["source_asset_preserved"] is True
    assert receipt["publication_outcome"] == "published"
    assert receipt["reused_output"] is False
    assert receipt["source_stage_sha256"] == receipt["output_unrelated_stage_sha256"]
    assert [item["prim_path"] for item in receipt["changes"]] == [
        "/robot/handles/grasp_identifier_primary",
        "/robot/grasp_identifier_secondary",
    ]


def test_gsp001_derivative_propagates_package_root_to_isa001(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))

    report = run_simready_profile_conformance(
        SimReadyConformanceInput(
            asset_path=str(asset),
            output_dir=str(tmp_path / "output"),
            repair_requirements=["GSP.001", "ISA.001"],
            grasp_plan_path=str(plan),
            source_asset=str(asset),
            force=True,
        )
    )

    assert report.passed
    assert report.status == "PASS"
    assert report.requirements_repaired == ["GSP.001", "ISA.001"]
    assert report.requirements_blocked == []
    assert Usd.Stage.Open(report.output_usd_path) is not None


def test_gsp001_binds_plan_to_explicit_source_asset(tmp_path: Path) -> None:
    source_asset = tmp_path / "source.usda"
    _write_asset(source_asset)
    source_bytes = source_asset.read_bytes()
    asset = tmp_path / "working-copy.usda"
    shutil.copy2(source_asset, asset)
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(source_asset))

    report = _run(
        asset=asset,
        output_dir=tmp_path / "output",
        plan=plan,
        source_asset=source_asset,
    )

    assert report.passed
    assert source_asset.read_bytes() == source_bytes
    assert asset.read_bytes() == source_bytes
    receipt = _receipt(report)
    assert receipt["source_asset_path"] == str(source_asset.resolve())
    assert receipt["source_asset_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert receipt["staged_asset_sha256"] == receipt["source_asset_sha256"]


def test_gsp001_extracts_usdz_and_preserves_unrelated_package_members(
    tmp_path: Path,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")

    asset = tmp_path / "asset.usdz"
    with ZipFile(asset, "w") as archive:
        archive.writestr(
            "root.usda",
            """#usda 1.0
(
    defaultPrim = "robot"
)

def Xform "robot"
{
    def Xform "handles" {}
}
""",
        )
        archive.writestr("evidence/handle.txt", b"owner-reviewed evidence")
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    assert report.passed
    assert asset.read_bytes() == source_bytes
    output_root = Path(report.output_usd_path)
    assert output_root.name == "root.usda"
    assert (output_root.parent / "evidence" / "handle.txt").read_bytes() == (
        b"owner-reviewed evidence"
    )
    stage = Usd.Stage.Open(str(output_root))
    assert stage is not None
    assert stage.GetPrimAtPath("/robot/handles/grasp_identifier_primary")
    receipt = _receipt(report)
    assert receipt["source_was_usdz"] is True
    assert receipt["source_asset_sha256"] == hashlib.sha256(source_bytes).hexdigest()


def test_gsp001_preserves_relative_dependency_tree(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")

    payload_layer = tmp_path / "payload.usda"
    payload_layer.write_text(
        """#usda 1.0

def Xform "robot"
{
    def Xform "handles"
    {
    }
}
""",
        encoding="utf-8",
    )
    asset = tmp_path / "asset.usda"
    asset.write_text(
        """#usda 1.0
(
    defaultPrim = "robot"
    subLayers = [@payload.usda@]
)
""",
        encoding="utf-8",
    )
    source_bytes = asset.read_bytes()
    payload_bytes = payload_layer.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    assert report.passed
    assert asset.read_bytes() == source_bytes
    assert payload_layer.read_bytes() == payload_bytes
    output_root = Path(report.output_usd_path)
    assert (output_root.parent / "payload.usda").read_bytes() == payload_bytes
    stage = Usd.Stage.Open(str(output_root))
    assert stage is not None
    assert stage.GetPrimAtPath("/robot/handles/grasp_identifier_primary")
    receipt = _receipt(report)
    assert receipt["source_stage_sha256"] == receipt["output_unrelated_stage_sha256"]


def test_gsp001_existing_valid_line_is_byte_preserving_without_plan(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset, valid_grasp=True)
    source_bytes = asset.read_bytes()

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=None)

    assert report.passed
    assert report.requirements_repaired == ["GSP.001"]
    assert asset.read_bytes() == source_bytes
    assert Path(report.output_usd_path).read_bytes() == source_bytes
    receipt = _receipt(report)
    assert receipt["changes"] == []
    assert receipt["existing_grasp_lines"] == ["/robot/grasp_identifier_existing"]


def test_gsp001_plan_preserves_unplanned_valid_existing_lines(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")

    asset = tmp_path / "asset.usda"
    _write_asset(asset, valid_grasp=True)
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    assert report.passed
    assert asset.read_bytes() == source_bytes
    stage = Usd.Stage.Open(report.output_usd_path)
    assert stage is not None
    existing = stage.GetPrimAtPath("/robot/grasp_identifier_existing")
    assert len(existing.GetAttribute("points").Get()) == 2
    assert list(existing.GetAttribute("widths").Get()) == pytest.approx([0.01])
    assert stage.GetPrimAtPath("/robot/handles/grasp_identifier_primary")
    assert _receipt(report)["existing_grasp_lines"] == [
        "/robot/grasp_identifier_existing"
    ]


def test_gsp001_plan_rejects_valid_existing_target_collision(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset, valid_grasp=True)
    source_bytes = asset.read_bytes()
    payload = _plan_payload(asset)
    lines = payload["grasp_lines"]
    assert isinstance(lines, list)
    assert isinstance(lines[0], dict)
    lines[0]["prim_path"] = "/robot/grasp_identifier_existing"
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, payload)

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "conflicts with an existing prim" in report.steps[0]["reason"]


def test_gsp001_absent_plan_remains_blocked(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=None)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "owner-approved local-coordinate grasp plan" in report.steps[0]["reason"]


@pytest.mark.parametrize(
    "case",
    [
        "extra_field",
        "wrong_version",
        "duplicate_paths",
        "too_few_points",
        "nonfinite_point",
        "float32_overflow",
        "zero_width",
        "width_underflow",
        "ambiguous_width_count",
        "missing_evidence",
        "nonlocal_coordinates",
    ],
)
def test_gsp001_rejects_malformed_or_ambiguous_contracts(
    tmp_path: Path,
    case: str,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    payload = _plan_payload(asset)
    lines = payload["grasp_lines"]
    provenance = payload["provenance"]
    assert isinstance(lines, list)
    assert isinstance(lines[0], dict)
    assert isinstance(provenance, dict)
    line = lines[0]

    if case == "extra_field":
        payload["unexpected"] = True
    elif case == "wrong_version":
        payload["schema_version"] = "content-agent-workflows.simready-grasp-plan.v0"
    elif case == "duplicate_paths":
        lines.append(copy.deepcopy(line))
    elif case == "too_few_points":
        line["points"] = [[0, 0, 0]]
    elif case == "nonfinite_point":
        line["points"] = [[0, 0, 0], [float("nan"), 0, 1]]
    elif case == "float32_overflow":
        line["points"] = [[0, 0, 0], [1e100, 0, 1]]
    elif case == "zero_width":
        line["widths"] = [0]
    elif case == "width_underflow":
        line["widths"] = [1e-100]
    elif case == "ambiguous_width_count":
        line["widths"] = [0.01, 0.02, 0.03]
    elif case == "missing_evidence":
        provenance["evidence"] = []
    elif case == "nonlocal_coordinates":
        line["coordinate_space"] = "world"

    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, payload)
    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "Could not safely author GSP.001 grasp lines" in report.steps[0]["reason"]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("stale_hash", "source_asset_sha256 is stale"),
        ("stale_default", "default_prim_path is stale"),
        ("path_escape", "escapes the default prim"),
        ("default_path", "escapes the default prim"),
        ("whitespace_path", "not a canonical absolute prim path"),
        ("wrong_name", "must start with grasp_identifier"),
        ("missing_parent", "parent must be an existing editable prim"),
    ],
)
def test_gsp001_rejects_stale_identity_and_unsafe_paths(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    payload = _plan_payload(asset)
    lines = payload["grasp_lines"]
    assert isinstance(lines, list)
    assert isinstance(lines[0], dict)
    line = lines[0]

    if case == "stale_hash":
        payload["source_asset_sha256"] = "0" * 64
    elif case == "stale_default":
        payload["default_prim_path"] = "/stale"
    elif case == "path_escape":
        line["prim_path"] = "/robotic/grasp_identifier_escape"
    elif case == "default_path":
        line["prim_path"] = "/robot"
    elif case == "whitespace_path":
        line["prim_path"] = " /robot/handles/grasp_identifier_primary "
    elif case == "wrong_name":
        line["prim_path"] = "/robot/handles/gripper"
    elif case == "missing_parent":
        line["prim_path"] = "/robot/missing/grasp_identifier_orphan"

    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, payload)
    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert reason in report.steps[0]["reason"]


def test_gsp001_rejects_instance_root_as_authored_line_parent(tmp_path: Path) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")

    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    prototype = UsdGeom.Xform.Define(stage, "/Prototype").GetPrim()
    UsdGeom.Xform.Define(stage, "/Prototype/handles")
    robot = UsdGeom.Xform.Define(stage, "/robot").GetPrim()
    robot.GetReferences().AddInternalReference(prototype.GetPath())
    robot.SetInstanceable(True)
    stage.SetDefaultPrim(robot)
    assert stage.GetRootLayer().Save()
    source_bytes = asset.read_bytes()
    payload = _plan_payload(asset)
    lines = payload["grasp_lines"]
    assert isinstance(lines, list)
    assert isinstance(lines[0], dict)
    lines[0]["prim_path"] = "/robot/grasp_identifier_primary"
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, payload)

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "parent must be an existing editable prim" in report.steps[0]["reason"]


def test_gsp001_rejects_staged_bytes_diverging_from_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))
    original_stage_input = conform_profile_module._stage_input

    def stage_then_mutate(*args, **kwargs):
        staged_path, package_root, warnings = original_stage_input(*args, **kwargs)
        staged_path.write_bytes(staged_path.read_bytes() + b"\n# stale staged copy\n")
        return staged_path, package_root, warnings

    monkeypatch.setattr(
        conform_profile_module,
        "_stage_input",
        stage_then_mutate,
    )
    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "no longer matches the exact plan-bound source" in report.steps[0]["reason"]


def test_gsp001_rejects_duplicate_json_keys_and_invalid_utf8(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    payload = _plan_payload(asset)
    remainder = json.dumps(
        {key: value for key, value in payload.items() if key != "schema_version"}
    )[1:]

    duplicate_plan = tmp_path / "duplicate.json"
    duplicate_plan.write_text(
        "{"
        f'"schema_version":"{SIMREADY_GRASP_PLAN_SCHEMA_VERSION}",'
        f'"schema_version":"{SIMREADY_GRASP_PLAN_SCHEMA_VERSION}",' + remainder,
        encoding="utf-8",
    )
    duplicate_report = _run(
        asset=asset,
        output_dir=tmp_path / "duplicate-output",
        plan=duplicate_plan,
    )
    _assert_blocked(duplicate_report, asset=asset, original_bytes=source_bytes)
    assert "duplicate JSON key" in duplicate_report.steps[0]["reason"]

    invalid_utf8_plan = tmp_path / "invalid-utf8.json"
    invalid_utf8_plan.write_bytes(b"\xff")
    invalid_utf8_report = _run(
        asset=asset,
        output_dir=tmp_path / "invalid-utf8-output",
        plan=invalid_utf8_plan,
    )
    _assert_blocked(invalid_utf8_report, asset=asset, original_bytes=source_bytes)
    assert "codec can't decode" in invalid_utf8_report.steps[0]["reason"]


def test_gsp001_rejects_conflicting_existing_grasp_prims(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset, invalid_grasp=True, conflicting_grasp_prim=True)
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "conflicting existing grasp_identifier prims" in report.steps[0]["reason"]
    assert _receipt(report)["invalid_existing_grasp_lines"] == [
        "/robot/grasp_identifier_existing",
        "/robot/handles/grasp_identifier_conflict",
    ]


def test_gsp001_rejects_existing_target_prim(tmp_path: Path) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset, conflicting_grasp_prim=True)
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    payload = _plan_payload(asset)
    lines = payload["grasp_lines"]
    assert isinstance(lines, list)
    assert isinstance(lines[0], dict)
    lines[0]["prim_path"] = "/robot/handles/grasp_identifier_conflict"
    _write_plan(plan, payload)

    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "conflicting existing grasp_identifier prims" in report.steps[0]["reason"]


def test_gsp001_rejects_source_staging_or_plan_mutation_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))
    output_dir = tmp_path / "staged-output"
    original_author = conform_profile_module._author_gsp001_lines

    def mutate_source(**kwargs):
        changes = original_author(**kwargs)
        staged_source = output_dir / "staged" / asset.name
        staged_source.write_bytes(
            staged_source.read_bytes() + b"\n# external mutation\n"
        )
        return changes

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        mutate_source,
    )
    staged_report = _run(asset=asset, output_dir=output_dir, plan=plan)
    _assert_blocked(staged_report, asset=asset, original_bytes=source_bytes)
    assert "Staged USD package changed" in staged_report.steps[0]["reason"]

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        original_author,
    )
    source_output_dir = tmp_path / "source-output"

    def mutate_original_source(**kwargs):
        changes = original_author(**kwargs)
        asset.write_bytes(asset.read_bytes() + b"\n# external source mutation\n")
        return changes

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        mutate_original_source,
    )
    source_report = _run(
        asset=asset,
        output_dir=source_output_dir,
        plan=plan,
    )
    assert not source_report.passed
    assert source_report.requirements_blocked == ["GSP.001"]
    assert "Source asset changed" in source_report.steps[0]["reason"]
    source_publish_root = source_output_dir / conform_profile_module.GSP001_OUTPUT_DIR
    assert not source_publish_root.exists() or list(source_publish_root.iterdir()) == []
    asset.write_bytes(source_bytes)

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        original_author,
    )
    plan_output_dir = tmp_path / "plan-output"

    def mutate_plan(**kwargs):
        changes = original_author(**kwargs)
        plan.write_bytes(plan.read_bytes() + b" ")
        return changes

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        mutate_plan,
    )
    plan_report = _run(asset=asset, output_dir=plan_output_dir, plan=plan)
    _assert_blocked(plan_report, asset=asset, original_bytes=source_bytes)
    assert "Grasp plan changed" in plan_report.steps[0]["reason"]


def test_gsp001_rejects_unplanned_stage_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Sdf = pytest.importorskip("pxr.Sdf")

    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    source_bytes = asset.read_bytes()
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))
    original_author = conform_profile_module._author_gsp001_lines

    def add_unplanned_attribute(**kwargs):
        changes = original_author(**kwargs)
        kwargs["stage"].GetPrimAtPath("/robot").CreateAttribute(
            "test:unplanned",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set("not-in-plan")
        return changes

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        add_unplanned_attribute,
    )
    report = _run(asset=asset, output_dir=tmp_path / "output", plan=plan)

    _assert_blocked(report, asset=asset, original_bytes=source_bytes)
    assert "unplanned stage change" in report.steps[0]["reason"]

    def add_unplanned_planned_prim_metadata(**kwargs):
        changes = original_author(**kwargs)
        kwargs["stage"].GetPrimAtPath(
            "/robot/handles/grasp_identifier_primary"
        ).SetCustomDataByKey("unplanned", "not-in-plan")
        return changes

    monkeypatch.setattr(
        conform_profile_module,
        "_author_gsp001_lines",
        add_unplanned_planned_prim_metadata,
    )
    metadata_report = _run(
        asset=asset,
        output_dir=tmp_path / "metadata-output",
        plan=plan,
    )
    _assert_blocked(metadata_report, asset=asset, original_bytes=source_bytes)
    assert "unplanned prim state" in metadata_report.steps[0]["reason"]


def test_gsp001_publication_reuses_identical_tree_and_removes_read_only_build(
    tmp_path: Path,
) -> None:
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    build_dir = tmp_path / "build"
    nested = build_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    tree_sha256 = conform_profile_module._isa001_tree_sha256(build_dir)
    final_tree = publish_root / tree_sha256
    shutil.copytree(build_dir, final_tree)
    (nested / "asset.usda").chmod(0o400)
    nested.chmod(0o500)

    published, outcome = conform_profile_module._publish_gsp001_tree(
        build_dir=build_dir,
        publish_root=publish_root,
        tree_sha256=tree_sha256,
    )

    assert published == final_tree
    assert outcome == "cache_hit"
    assert not build_dir.exists()


def test_gsp001_publication_distinguishes_concurrent_identical_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    tree_sha256 = conform_profile_module._isa001_tree_sha256(build_dir)
    final_tree = publish_root / tree_sha256

    def publish_concurrent_winner(_self: Path, target: Path) -> None:
        shutil.copytree(build_dir, target)
        raise FileExistsError("simulated concurrent publisher")

    monkeypatch.setattr(Path, "replace", publish_concurrent_winner)

    published, outcome = conform_profile_module._publish_gsp001_tree(
        build_dir=build_dir,
        publish_root=publish_root,
        tree_sha256=tree_sha256,
    )

    assert published == final_tree
    assert outcome == "concurrent_reuse"
    assert not build_dir.exists()


def test_gsp001_publication_does_not_hide_concurrent_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publish_root = tmp_path / "published"
    publish_root.mkdir()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    tree_sha256 = conform_profile_module._isa001_tree_sha256(build_dir)

    def publish_concurrent_winner(_self: Path, target: Path) -> None:
        shutil.copytree(build_dir, target)
        raise FileExistsError("simulated concurrent publisher")

    def fail_cleanup(_build_dir: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(Path, "replace", publish_concurrent_winner)
    monkeypatch.setattr(
        conform_profile_module,
        "_remove_gsp001_build_tree",
        fail_cleanup,
    )

    with pytest.raises(OSError, match="simulated cleanup failure"):
        conform_profile_module._publish_gsp001_tree(
            build_dir=build_dir,
            publish_root=publish_root,
            tree_sha256=tree_sha256,
        )


def test_gsp001_cli_plumbs_grasp_plan_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    asset = tmp_path / "asset.usda"
    _write_asset(asset)
    plan = tmp_path / "grasp-plan.json"
    _write_plan(plan, _plan_payload(asset))

    exit_code = conform_profile_module.main(
        [
            str(asset),
            "--output-dir",
            str(tmp_path / "output"),
            "--repair",
            "GSP.001",
            "--grasp-plan",
            str(plan),
            "--force",
            "--strict",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    receipt = json.loads(
        Path(payload["reports"]["GSP.001"]).read_text(encoding="utf-8")
    )
    assert receipt["grasp_plan_path"] == str(plan)
