# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agentic convert-to-USD workflow."""

from __future__ import annotations

import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from content_agent_workflows.convert_to_usd import (
    ConvertToUsdWorkflowInput,
    convert_source_to_usd_file,
    convert_to_usd,
    converter_package_for_source,
    default_output_usd_path,
    is_existing_usd,
    is_mujoco_source,
    preflight_convert_to_usd_dependencies,
    run_convert_to_usd_workflow,
)
from content_agent_workflows.convert_to_usd import cli as convert_cli
from content_agent_workflows.convert_to_usd import workflow as convert_workflow

MINIMAL_USDA = '#usda 1.0\n\ndef Xform "World" {\n}\n'


def test_usd_convert_cad_install_uses_immutable_public_revision() -> None:
    revision = convert_workflow.USD_CONVERT_CAD_REVISION
    install_spec = convert_workflow.USD_CONVERT_CAD_INSTALL_SPEC

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert install_spec == (
        "git+https://github.com/NVIDIA-Omniverse/usd-convert-cad.git@" + revision
    )
    assert (
        convert_workflow.CONVERTER_INSTALL_SPECS["usd-convert-cad"][0] == install_spec
    )


def _write_fake_converter(tmp_path: Path, name: str) -> Path:
    script = tmp_path / name
    script.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / f"{source.stem}.usda").write_text(
    '#usda 1.0\\n\\ndef Xform "World" {\\n}\\n',
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _write_fake_converter_with_sidecar(tmp_path: Path, name: str) -> Path:
    script = tmp_path / name
    script.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

source = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "textures").mkdir(exist_ok=True)
(output_dir / f"{source.stem}.usda").write_text(
    '#usda 1.0\\n\\ndef Xform "World" {\\n}\\n',
    encoding="utf-8",
)
(output_dir / "textures" / "albedo.png").write_text("fake texture\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _write_fake_cad_converter(tmp_path: Path) -> Path:
    script = tmp_path / "usd-convert-cad"
    script.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('#usda 1.0\\n\\ndef Xform "World" {\\n}\\n', encoding="utf-8")
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _patch_fake_usd_export(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, Path]]:
    exports: list[tuple[Path, Path]] = []

    def fake_export(source_usd: Path, output_usd: Path) -> None:
        exports.append((source_usd, output_usd))
        output_usd.write_bytes(b"PXR-USDC fake\n")

    monkeypatch.setattr(convert_workflow, "_export_usd_layer", fake_export)
    return exports


def test_existing_usd_passthrough_writes_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(source_asset_path=source, output_dir=tmp_path / "run")
    )

    assert result.success
    assert result.selected_converter == "existing-usd-passthrough"
    assert result.output_usd_path == str((tmp_path / "run" / "asset.usda").resolve())
    assert Path(result.output_usd_path).read_text(encoding="utf-8") == "#usda 1.0\n"
    assert Path(result.conversion_report_path).exists()

    report = json.loads(Path(result.conversion_report_path).read_text())
    assert report["source_format"] == "usd"
    assert report["converter_reference"] == "existing-usd-passthrough"
    assert report["generated_files"] == ["asset.usda"]
    assert report["errors"] == []


def test_existing_usd_failed_write_does_not_report_stale_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "asset.usda"
    output = tmp_path / "asset_copy.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output.write_text("stale\n", encoding="utf-8")

    def fail_write(_source_usd: Path, _output_usd: Path) -> None:
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(convert_workflow, "_write_primary_usd", fail_write)

    report, _probe = convert_source_to_usd_file(source, output)

    assert not report.passed
    assert report.output_usd_path == ""
    assert report.output_format == "unknown"
    assert report.generated_files == []
    assert "synthetic write failure" in report.errors[0]


def test_file_output_report_sidecars_do_not_mark_primary_written(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "robot.usda"
    source.write_text("<robot name='r' />\n", encoding="utf-8")

    base_report = convert_workflow._report(
        status="failed",
        source_asset=source,
        source_format="urdf",
        converter_reference="urdf-usd-converter",
        converter_tool="urdf_usd_converter",
        converter_command=[],
        output_directory=tmp_path,
        errors=["converter wrote sidecars but not the primary USD"],
    )

    report = convert_workflow._file_output_report(
        base_report,
        output_usd_path=output,
        generated_files=["textures/albedo.png"],
    )

    assert not report.passed
    assert report.output_usd_path == ""
    assert report.output_format == "unknown"
    assert report.generated_files == ["textures/albedo.png"]


def test_file_output_report_existing_file_is_not_generation_proof(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "robot.usda"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    output.write_text("stale\n", encoding="utf-8")

    base_report = convert_workflow._report(
        status="passed",
        source_asset=source,
        source_format="urdf",
        converter_reference="urdf-usd-converter",
        converter_tool="urdf_usd_converter",
        converter_command=[],
        output_directory=tmp_path,
    )

    report = convert_workflow._file_output_report(
        base_report,
        output_usd_path=output,
    )

    assert not report.passed
    assert report.output_usd_path == ""
    assert report.output_format == "unknown"
    assert report.generated_files == []
    assert report.errors == [
        f"primary USD output was not generated by this run: {output}"
    ]


def test_urdf_route_invokes_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.urdf"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(source_asset_path=source, output_dir=tmp_path / "run")
    )

    assert result.success
    assert result.selected_converter == "urdf-usd-converter"
    assert result.source_format == "urdf"
    assert result.output_usd_path is not None
    assert Path(result.output_usd_path).exists()
    assert Path(result.output_usd_path).name == "robot.usda"


def test_mujoco_route_invokes_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "model.xml"
    source.write_text("<mujoco model='m' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "mujoco_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(source_asset_path=source, output_dir=tmp_path / "run")
    )

    assert result.success
    assert result.selected_converter == "mujoco-usd-converter"
    assert result.source_format == "mjcf"
    assert result.output_usd_path is not None
    assert Path(result.output_usd_path).exists()
    assert Path(result.output_usd_path).name == "model.usda"


def test_cad_route_invokes_usd_convert_cad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    _write_fake_cad_converter(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(source_asset_path=source, output_dir=tmp_path / "run")
    )

    assert result.success
    assert result.selected_converter == "usd-convert-cad"
    assert result.source_format == "cad"
    assert result.output_usd_path is not None
    assert Path(result.output_usd_path).exists()
    assert Path(result.output_usd_path).name == "mesh.usda"


def test_converter_tool_resolves_from_active_python_scripts_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    scripts_dir = tmp_path / "venv" / "bin"
    scripts_dir.mkdir(parents=True)
    monkeypatch.setattr(
        convert_workflow.sys,
        "executable",
        str(scripts_dir / "python3"),
    )
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    _write_fake_cad_converter(scripts_dir)

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(source_asset_path=source, output_dir=tmp_path / "run")
    )

    assert result.success
    assert result.selected_converter == "usd-convert-cad"
    report = json.loads(Path(result.conversion_report_path).read_text())
    assert report["converter_command"][0] == str(scripts_dir / "usd-convert-cad")


def test_file_oriented_converter_writes_requested_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "custom_output.usda"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    report, probe = convert_source_to_usd_file(source, output)

    assert report.passed
    assert report.output_usd_path == str(output.resolve())
    assert report.output_directory == str(tmp_path.resolve())
    assert report.generated_files == ["custom_output.usda"]
    assert probe.selected_converter == "urdf-usd-converter"
    assert output.read_text(encoding="utf-8") == MINIMAL_USDA


def test_file_oriented_converter_preserves_sidecar_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "custom_output.usda"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter_with_sidecar(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    report, _probe = convert_source_to_usd_file(source, output)

    assert report.passed
    assert output.read_text(encoding="utf-8") == MINIMAL_USDA
    assert (tmp_path / "textures" / "albedo.png").read_text(encoding="utf-8") == (
        "fake texture\n"
    )
    assert report.generated_files == [
        "custom_output.usda",
        "textures/albedo.png",
    ]
    assert report.sidecar_inputs == ["textures/albedo.png"]


def test_file_oriented_converter_exports_requested_binary_output_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "custom_output.usdc"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    exports = _patch_fake_usd_export(monkeypatch)

    report, _probe = convert_source_to_usd_file(source, output)

    assert report.passed
    assert report.output_usd_path == str(output.resolve())
    assert report.output_format == "usdc"
    assert output.read_bytes().startswith(b"PXR-USDC")
    assert len(exports) == 1
    assert exports[0][0].name == "robot.usda"
    assert exports[0][1] == output.resolve()


def test_file_oriented_converter_reports_usd_import_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.urdf"
    output = tmp_path / "custom_output.usdc"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    def fail_export(_source_usd: Path, _output_usd: Path) -> None:
        raise ImportError("missing pxr")

    monkeypatch.setattr(convert_workflow, "_export_usd_layer", fail_export)

    report, _probe = convert_source_to_usd_file(source, output)

    assert not report.passed
    assert report.output_usd_path == ""
    assert report.output_format == "unknown"
    assert "missing pxr" in report.errors[0]


def test_write_primary_usd_accepts_delimited_asset_identifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    output = tmp_path / "output.usdc"
    source.write_text(MINIMAL_USDA, encoding="utf-8")
    exports = _patch_fake_usd_export(monkeypatch)

    convert_workflow._write_primary_usd(Path(f"{source}@"), output)

    assert output.read_bytes().startswith(b"PXR-USDC")
    assert exports == [(source, output)]


def test_workflow_output_format_writes_usdz_package(tmp_path: Path) -> None:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(
            source_asset_path=source,
            output_dir=tmp_path / "run",
            output_format="usdz",
        )
    )

    assert result.success
    assert result.output_usd_path == str((tmp_path / "run" / "asset.usdz").resolve())
    assert zipfile.is_zipfile(result.output_usd_path)
    report = json.loads(Path(result.conversion_report_path).read_text())
    assert report["output_format"] == "usdz"


def test_cli_defaults_output_to_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "robot.urdf"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)

    exit_code = convert_cli.main([str(source)])

    assert exit_code == 0
    assert (tmp_path / "robot.usda").exists()


def test_cli_output_format_defaults_output_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "robot.urdf"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.chdir(tmp_path)
    exports = _patch_fake_usd_export(monkeypatch)

    exit_code = convert_cli.main([str(source), "--output-format", "usdc"])

    assert exit_code == 0
    assert (tmp_path / "robot.usdc").read_bytes().startswith(b"PXR-USDC")
    assert len(exports) == 1


def test_supported_route_blocks_when_converter_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "robot.urdf"
    source.write_text("<robot name='r' />\n", encoding="utf-8")
    monkeypatch.setattr(convert_workflow, "_dependency_available", lambda _ref: False)
    monkeypatch.setattr(convert_workflow, "_converter_tool_path", lambda _tool: None)

    report, probe = convert_to_usd(source, tmp_path / "run")

    assert not report.passed
    assert report.status == "blocked"
    assert report.converter_reference == "urdf-usd-converter"
    assert probe.selected_converter == "urdf-usd-converter"
    assert "urdf_usd_converter CLI is required" in report.errors[0]
    assert "uv pip install" in report.install_hint
    assert "urdf-usd-converter" in report.install_hint


def test_cad_route_blocks_when_converter_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    monkeypatch.setattr(convert_workflow, "_dependency_available", lambda _ref: False)
    monkeypatch.setattr(convert_workflow, "_converter_tool_path", lambda _tool: None)

    report, probe = convert_to_usd(source, tmp_path / "run")

    assert not report.passed
    assert report.status == "blocked"
    assert report.converter_reference == "usd-convert-cad"
    assert probe.selected_converter == "usd-convert-cad"
    assert "usd-convert-cad CLI is required" in report.errors[0]
    assert "uv pip install" in report.install_hint
    assert convert_workflow.USD_CONVERT_CAD_INSTALL_SPEC in report.install_hint
    assert "https://pypi.nvidia.com" in report.install_hint


def test_preflight_installs_inferred_cad_converter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    calls: list[list[str]] = []
    available = {"value": False}

    def fake_dependency_available(converter_reference: str) -> bool:
        assert converter_reference == "usd-convert-cad"
        return available["value"]

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        available["value"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        convert_workflow,
        "_dependency_available",
        fake_dependency_available,
    )
    monkeypatch.setattr(convert_workflow.subprocess, "run", fake_run)

    report = preflight_convert_to_usd_dependencies(source)

    assert report.passed
    assert report.converter_reference == "usd-convert-cad"
    assert report.source_format == "cad"
    assert report.dependency_available_before is False
    assert report.dependency_available is True
    assert report.install_requested is True
    assert report.install_attempted is True
    assert report.install_command == calls[0]
    assert convert_workflow.USD_CONVERT_CAD_INSTALL_SPEC in calls[0]
    assert "https://pypi.nvidia.com" in calls[0]


def test_preflight_reports_dependency_install_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    monkeypatch.setattr(convert_workflow, "_dependency_available", lambda _ref: False)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(command, timeout=1.0)

    monkeypatch.setattr(convert_workflow.subprocess, "run", fake_run)

    report = preflight_convert_to_usd_dependencies(source)

    assert not report.passed
    assert any(
        "Timed out installing converter dependency" in error for error in report.errors
    )


def test_converter_timeout_ignores_exited_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234
        returncode = None

        def communicate(
            self,
            timeout: float | None = None,
        ) -> tuple[str, str]:
            if timeout is not None:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return "partial stdout", "partial stderr"

    monkeypatch.setattr(
        convert_workflow.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )

    def fail_killpg(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(convert_workflow.os, "killpg", fail_killpg)

    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        convert_workflow._run_converter_command(["fake"], timeout_s=0.1)

    assert exc_info.value.output == "partial stdout"
    assert exc_info.value.stderr == "partial stderr"


def test_convert_file_installs_dependency_after_directory_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "package"
    source_dir.mkdir()
    source = source_dir / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    output = tmp_path / "converted.usda"
    _write_fake_cad_converter(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    calls: list[list[str]] = []
    available = {"value": False}

    def fake_dependency_available(converter_reference: str) -> bool:
        assert converter_reference == "usd-convert-cad"
        return available["value"]

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        available["value"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        convert_workflow,
        "_dependency_available",
        fake_dependency_available,
    )
    monkeypatch.setattr(convert_workflow.subprocess, "run", fake_run)

    report, probe = convert_source_to_usd_file(source_dir, output)

    assert report.passed
    assert probe.selected_converter == "usd-convert-cad"
    assert calls
    assert convert_workflow.USD_CONVERT_CAD_INSTALL_SPEC in calls[0]
    assert output.exists()


def test_preflight_check_only_blocks_without_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "mesh.stl"
    source.write_text("solid mesh\nendsolid mesh\n", encoding="utf-8")
    monkeypatch.setattr(convert_workflow, "_dependency_available", lambda _ref: False)

    report = preflight_convert_to_usd_dependencies(source, install_missing=False)

    assert not report.passed
    assert report.status == "blocked"
    assert report.converter_reference == "usd-convert-cad"
    assert report.install_requested is False
    assert report.install_attempted is False
    assert any("usd-convert-cad CLI is required" in error for error in report.errors)


def test_unsupported_source_writes_blocked_report(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("not a convertible source\n", encoding="utf-8")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(source_asset_path=source, output_dir=tmp_path / "run")
    )

    assert not result.success
    assert result.selected_converter is None
    assert result.output_usd_path is None

    report = json.loads(Path(result.conversion_report_path).read_text())
    assert report["status"] == "blocked"
    assert report["errors"] == [
        "no enabled converter reference reported support for this source asset"
    ]
    assert report["install_hint"] == ""


def test_directory_with_one_supported_source_is_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "package"
    source_dir.mkdir()
    (source_dir / "notes.txt").write_text("ignore me\n", encoding="utf-8")
    (source_dir / "robot.urdf").write_text("<robot name='r' />\n", encoding="utf-8")
    _write_fake_converter(tmp_path, "urdf_usd_converter")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(
            source_asset_path=source_dir,
            output_dir=tmp_path / "run",
        )
    )

    assert result.success
    assert result.selected_converter == "urdf-usd-converter"
    assert result.source_asset.endswith("robot.urdf")


def test_directory_with_multiple_supported_sources_blocks(tmp_path: Path) -> None:
    source_dir = tmp_path / "package"
    source_dir.mkdir()
    (source_dir / "a.urdf").write_text("<robot name='a' />\n", encoding="utf-8")
    (source_dir / "b.urdf").write_text("<robot name='b' />\n", encoding="utf-8")

    result = run_convert_to_usd_workflow(
        ConvertToUsdWorkflowInput(
            source_asset_path=source_dir,
            output_dir=tmp_path / "run",
        )
    )

    assert not result.success
    report = json.loads(Path(result.conversion_report_path).read_text())
    assert "multiple supported source files" in report["errors"][0]


def test_source_detection_helpers(tmp_path: Path) -> None:
    usd = tmp_path / "asset.usda"
    mujoco = tmp_path / "model.xml"
    other_xml = tmp_path / "other.xml"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    mujoco.write_text("<mujoco model='m' />\n", encoding="utf-8")
    other_xml.write_text("<robot name='r' />\n", encoding="utf-8")

    assert is_existing_usd(usd)
    assert is_mujoco_source(mujoco)
    assert not is_mujoco_source(other_xml)
    assert default_output_usd_path(source_asset=usd, cwd=tmp_path) == usd.resolve()
    assert (
        default_output_usd_path(
            source_asset=usd,
            cwd=tmp_path,
            output_format="usdc",
        )
        == (tmp_path / "asset.usdc").resolve()
    )
    assert converter_package_for_source(tmp_path / "robot.urdf") == "urdf-usd-converter"
    assert (
        converter_package_for_source(tmp_path / "robot.mjcf") == "mujoco-usd-converter"
    )
    assert converter_package_for_source(tmp_path / "mesh.stl") == "usd-convert-cad"
    assert converter_package_for_source(tmp_path / "part.step") == "usd-convert-cad"
    assert converter_package_for_source(tmp_path / "part.prt.1") == "usd-convert-cad"
    assert converter_package_for_source(tmp_path / "mesh.stl.bak") is None
    assert converter_package_for_source(mujoco) == "mujoco-usd-converter"
    assert converter_package_for_source(other_xml) is None
