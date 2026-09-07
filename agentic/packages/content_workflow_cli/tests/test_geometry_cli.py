# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public Geometry-only launcher."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh
from pxr import Usd, UsdGeom

from content_workflow_cli import geometry_runner
from content_workflow_cli.cli import main


def test_checked_in_quickstart_fixture_is_valid_watertight_usd() -> None:
    fixture = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "geometry"
        / "quickstart"
        / "smoke_bracket.usda"
    )

    assert fixture.read_text(encoding="utf-8").splitlines()[0] == "#usda 1.0"
    stage = Usd.Stage.Open(str(fixture))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath().pathString == "/World"
    mesh = UsdGeom.Mesh.Get(stage, "/World/Geometry")
    assert mesh
    assert "MaterialBindingAPI" in mesh.GetPrim().GetAppliedSchemas()

    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    assert counts and set(counts) == {3}
    triangles = [indices[index : index + 3] for index in range(0, len(indices), 3)]
    inspected = trimesh.Trimesh(
        vertices=list(mesh.GetPointsAttr().Get()),
        faces=triangles,
        process=False,
    )
    assert inspected.is_watertight
    assert inspected.is_volume


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source.usda"
    source.write_text(
        '#usda 1.0\n(defaultPrim = "World" metersPerUnit = 1 upAxis = "Z")\n',
        encoding="utf-8",
    )
    return source


def _result(output_dir: Path, handoff_ready: str = "yes") -> SimpleNamespace:
    payload = {
        "success": handoff_ready != "no",
        "validation_status": (
            "pass"
            if handoff_ready == "yes"
            else "conditional"
            if handoff_ready == "conditional"
            else "fail"
        ),
        "handoff_ready": handoff_ready,
        "geometry_usd_path": str(output_dir / "geometry.usdc"),
        "handoff_manifest_path": str(output_dir / "content_agents_manifest.json"),
        "validation_evidence_path": str(
            output_dir / "geometry_validation_evidence.json"
        ),
        "render_report_path": str(output_dir / "geometry_render_evidence.json"),
        "error": None if handoff_ready != "no" else "geometry rejected",
    }
    return SimpleNamespace(
        **payload,
        model_dump=lambda **_kwargs: payload,
    )


def test_geometry_run_writes_frozen_request_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    output_dir = tmp_path / "run"
    captured: dict[str, object] = {}

    def execute(params):
        captured["params"] = params
        return _result(output_dir)

    monkeypatch.setattr(geometry_runner, "_execute_geometry_workflow", execute)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--prompt",
            "Preserve the mounting bore.",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    params = captured["params"]
    assert params.source_path == source.resolve()
    assert params.expected_source_sha256 == geometry_runner.file_sha256(source)
    assert params.output_dir == output_dir.resolve()
    assert params.render_evidence is True
    assert params.render_backend == "ovrtx"
    assert params.render_preset == "six_view"
    assert params.render_ovrtx_mode == "pt"
    assert params.render_ovrtx_num_sensor_updates == 64
    assert params.runtime_validation_mode == "skip"
    assert params.runtime_engine == "none"

    request = json.loads(
        (output_dir / geometry_runner.GEOMETRY_CLI_REQUEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert request["schema_version"] == "content-workflow-cli.geometry-request.v1"
    assert request["workflow"] == "geometry.run"
    assert request["request"]["source_path"] == str(source.resolve())
    assert len(request["source_sha256"]) == 64
    assert request["request"]["expected_source_sha256"] == request["source_sha256"]
    assert len(request["prompt_sha256"]) == 64

    result = json.loads(
        (output_dir / geometry_runner.GEOMETRY_CLI_RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert result["handoff_ready"] == "yes"
    output = capsys.readouterr().out
    assert "Geometry validation: pass" in output
    assert "Handoff ready: yes" in output
    assert (
        f"Result: {output_dir / geometry_runner.GEOMETRY_CLI_RESULT_FILENAME}" in output
    )


def test_geometry_run_rejects_source_changed_after_request_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    output_dir = tmp_path / "run"
    execute = geometry_runner._execute_geometry_workflow

    def mutate_then_execute(params):
        source.write_text("#usda 1.0\n", encoding="utf-8")
        return execute(params)

    monkeypatch.setattr(
        geometry_runner,
        "_execute_geometry_workflow",
        mutate_then_execute,
    )

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(output_dir),
            "--no-render-evidence",
        ]
    )

    assert code == 1
    result = json.loads(
        (output_dir / geometry_runner.GEOMETRY_CLI_RESULT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert "source changed after request admission" in result["error"]


def test_geometry_run_accepts_remote_ovrtx_final_evidence_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    captured: dict[str, object] = {}

    def execute(params):
        captured["params"] = params
        return _result(output_dir)

    monkeypatch.setattr(geometry_runner, "_execute_geometry_workflow", execute)

    assert (
        main(
            [
                "geometry",
                "run",
                str(_source(tmp_path)),
                "--output-dir",
                str(output_dir),
                "--render-backend",
                "remote",
            ]
        )
        == 0
    )
    assert captured["params"].render_backend == "remote"


@pytest.mark.parametrize(
    ("handoff_ready", "expected_code"),
    [("conditional", 3), ("no", 1)],
)
def test_geometry_run_has_stable_nonpassing_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    handoff_ready: str,
    expected_code: int,
) -> None:
    source = _source(tmp_path)
    output_dir = tmp_path / f"run-{handoff_ready}"
    monkeypatch.setattr(
        geometry_runner,
        "_execute_geometry_workflow",
        lambda _params: _result(output_dir, handoff_ready),
    )

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    assert code == expected_code
    assert json.loads(capsys.readouterr().out)["handoff_ready"] == handoff_ready


def test_geometry_dry_run_freezes_request_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    output_dir = tmp_path / "dry-run"

    def unexpected(_params):
        raise AssertionError("dry run must not execute Geometry")

    monkeypatch.setattr(
        geometry_runner,
        "_execute_geometry_workflow",
        unexpected,
    )

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"] == "geometry.run"
    assert (output_dir / geometry_runner.GEOMETRY_CLI_REQUEST_FILENAME).is_file()
    assert not (output_dir / geometry_runner.GEOMETRY_CLI_RESULT_FILENAME).exists()


def test_geometry_run_rejects_existing_output_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 2
    assert "choose a fresh run path" in capsys.readouterr().err


def test_geometry_run_allows_symlinked_output_parent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(linked_parent / "run"),
            "--dry-run",
        ]
    )

    assert code == 0
    assert capsys.readouterr().err == ""
    assert (
        real_parent / "run" / geometry_runner.GEOMETRY_CLI_REQUEST_FILENAME
    ).is_file()


def test_geometry_run_rejects_symlinked_final_output_component(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    output_dir = tmp_path / "linked-run"
    output_dir.symlink_to(outside, target_is_directory=True)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 2
    assert "choose a fresh run path" in capsys.readouterr().err
    assert not (outside / geometry_runner.GEOMETRY_CLI_REQUEST_FILENAME).exists()


def test_geometry_run_requires_exactly_one_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--source-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert code == 2
    assert "provide exactly one" in capsys.readouterr().err


def test_geometry_required_parts_require_completed_segmentation_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--required-part",
            "handle",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert code == 2
    assert "require --segmentation-run-dir" in capsys.readouterr().err


def test_geometry_run_persists_failure_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)
    output_dir = tmp_path / "failed"

    def fail(_params):
        raise RuntimeError("renderer startup failed")

    monkeypatch.setattr(geometry_runner, "_execute_geometry_workflow", fail)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 2
    failure = json.loads(
        (output_dir / geometry_runner.GEOMETRY_CLI_FAILURE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "renderer startup failed"
    assert "renderer startup failed" in capsys.readouterr().err


def test_geometry_output_usd_is_confined_to_run_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(tmp_path / "run"),
            "--output-usd",
            "../escaped.usdc",
        ]
    )

    assert code == 2
    assert "must resolve inside" in capsys.readouterr().err


def test_geometry_run_rejects_nonpositive_ovrtx_updates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source(tmp_path)

    code = main(
        [
            "geometry",
            "run",
            str(source),
            "--output-dir",
            str(tmp_path / "run"),
            "--ovrtx-sensor-updates",
            "0",
        ]
    )

    assert code == 2
    assert "must be positive" in capsys.readouterr().err
