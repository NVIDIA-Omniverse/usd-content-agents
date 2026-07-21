# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the content-workflow-cli CLI wrapper."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from content_workflow_cli.cli import main


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _write_fake_simready_foundation(tmp_path: Path) -> tuple[Path, Path]:
    foundation_root = tmp_path / "simready-foundation"
    spec_root = foundation_root / "nv_core" / "sr_specs" / "docs"
    (spec_root / "capabilities").mkdir(parents=True)
    (spec_root / "features").mkdir(parents=True)
    (spec_root / "profiles").mkdir(parents=True)
    (spec_root / "profiles" / "profiles.toml").write_text(
        '[Prop-Robotics-Neutral]\n"1.0.0" = { features = [] }\n',
        encoding="utf-8",
    )
    venv = tmp_path / "simready-venv"
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    bin_dir.mkdir(parents=True)
    executable = bin_dir / (
        "simready-validate.exe" if os.name == "nt" else "simready-validate"
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"passed": True, "status": "PASS"}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return foundation_root, venv


def _write_error_status_simready_foundation(tmp_path: Path) -> tuple[Path, Path]:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    executable = (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / ("simready-validate.exe" if os.name == "nt" else "simready-validate")
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "passed": False,
    "status": "ERROR",
    "issues": [{"requirement_id": "NP.006", "severity": "ERROR"}],
}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return foundation_root, venv


def test_simready_preflight_cli_reports_json(tmp_path: Path, capsys) -> None:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)

    code = main(
        [
            "preflight",
            "simready-foundation",
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["available_profiles"] == ["Prop-Robotics-Neutral"]


def test_simready_preflight_cli_expands_report_path(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)

    code = main(
        [
            "preflight",
            "simready-foundation",
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--report",
            "~/simready-preflight.json",
        ]
    )

    capsys.readouterr()
    assert code == 0
    assert (home / "simready-preflight.json").exists()
    assert not (cwd / "~" / "simready-preflight.json").exists()


def test_simready_validate_profile_cli_reports_json(tmp_path: Path, capsys) -> None:
    foundation_root, venv = _write_fake_simready_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    code = main(
        [
            "simready",
            "validate-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["profile_target"] == "Prop-Robotics-Neutral@1.0.0"
    assert payload["passed"] is True


def test_simready_validate_profile_cli_non_strict_allows_profile_error_status(
    tmp_path: Path,
    capsys,
) -> None:
    foundation_root, venv = _write_error_status_simready_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")

    code = main(
        [
            "simready",
            "validate-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--venv",
            str(venv),
            "--no-install-missing",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["status"] == "ERROR"
    assert payload["next_step"] == "simready-conform-profile"


def test_simready_conform_profile_cli_reports_json(tmp_path: Path, capsys) -> None:
    foundation_root, _venv = _write_fake_simready_foundation(tmp_path)
    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "World" {\n}\n', encoding="utf-8")
    validation_report = tmp_path / "simready-profile.json"
    validation_report.write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "requirement_id": "NP.006",
                        "severity": "ERROR",
                        "message": "Missing SimReady metadata.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "simready",
            "conform-profile",
            str(asset),
            "--foundation-root",
            str(foundation_root),
            "--validation-report",
            str(validation_report),
            "--output-dir",
            str(tmp_path / "conform"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["status"] == "PASS"
    assert payload["failed_requirements"] == ["NP.006"]
    assert payload["requirements_repaired"] == ["NP.006"]


def test_simready_conform_profile_cli_forwards_physics_fingerprint(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import content_agent_workflows.simready as simready_module

    asset = tmp_path / "asset.usda"
    asset.write_text('#usda 1.0\n\ndef Xform "Asset" {}\n', encoding="utf-8")
    fingerprint = "a" * 64
    captured = {}

    def fake_conformance(params):
        captured["params"] = params
        payload = {
            "passed": True,
            "status": "PASS",
            "output_usd_path": str(asset),
            "report_path": None,
            "requirements_blocked": [],
            "errors": [],
        }
        return SimpleNamespace(**payload, model_dump=lambda **_kwargs: payload)

    monkeypatch.setattr(
        simready_module,
        "run_simready_profile_conformance",
        fake_conformance,
    )

    code = main(
        [
            "simready",
            "conform-profile",
            str(asset),
            "--output-dir",
            str(tmp_path / "conform"),
            "--repair",
            "G3A.HYG.001",
            "--expected-physics-inventory-sha256",
            fingerprint,
            "--json",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert captured["params"].expected_physics_inventory_sha256 == fingerprint


def test_physics_apply_cli_wrapper_runs_fake_validation(
    tmp_path: Path,
    capsys,
) -> None:
    asset = (
        _repo_root()
        / "apps"
        / "physics_agent_service"
        / "tests"
        / "test_data"
        / "simple_cube.usda"
    )

    code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(asset),
            "--output-dir",
            str(tmp_path),
            "--simulation-engine",
            "fake",
            "--duration-s",
            "0.2",
            "--sample-fps",
            "10",
            "--drop-height-m",
            "0.1",
            "--deterministic-workflow",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["success"] is True
    assert Path(payload["physics_usd_path"]).exists()
    assert Path(payload["validation_evidence_path"]).exists()
