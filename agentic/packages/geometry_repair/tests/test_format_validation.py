# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative source-format validation tests."""

from __future__ import annotations

import json
from pathlib import Path

from geometry_repair.format_validation import validate_source_format


def _write_usda(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset" {}
""",
        encoding="utf-8",
    )
    return path


def test_openusd_compliance_checker_emits_hash_bound_report(tmp_path: Path) -> None:
    source = _write_usda(tmp_path / "asset.usda")
    output = tmp_path / "source_format_validation.json"

    report = validate_source_format(source, output)

    assert report.status == "pass"
    assert report.validator == "OpenUSD UsdUtils.ComplianceChecker"
    assert report.validator_version == "25.5"
    assert report.metadata["provider_distribution"] == "usd-exchange"
    assert report.metadata["provider_version"] == "2.3.0"
    assert report.source_sha256
    assert report.metadata["execution_scope"] == "isolated_subprocess"
    assert report.metadata["memory_limit_mb"] == 4096
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_khronos_gltf_validator_output_is_preserved(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "asset.gltf"
    source.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
    executable = tmp_path / "gltf_validator"
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\' \'{"validatorVersion":"2.0-test","issues":'
        '{"numErrors":0,"numWarnings":0,"messages":[]}}\'\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("GEOMETRY_REPAIR_GLTF_VALIDATOR_EXECUTABLE", str(executable))

    report = validate_source_format(source, tmp_path / "gltf_report.json")

    assert report.status == "pass"
    assert report.validator == "Khronos glTF Validator"
    assert report.validator_version == "2.0-test"
    assert report.metadata["returncode"] == 0


def test_missing_gltf_validator_is_not_silently_replaced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "asset.glb"
    source.write_bytes(b"glTF")
    monkeypatch.setenv(
        "GEOMETRY_REPAIR_GLTF_VALIDATOR_EXECUTABLE",
        str(tmp_path / "missing_gltf_validator"),
    )

    report = validate_source_format(source, tmp_path / "missing_report.json")

    assert report.status == "not_evaluated"
    assert "unavailable" in report.errors[0]


def test_gltf_validator_timeout_is_not_misreported_as_source_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "asset.gltf"
    source.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
    executable = tmp_path / "gltf_validator"
    executable.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("GEOMETRY_REPAIR_GLTF_VALIDATOR_EXECUTABLE", str(executable))

    report = validate_source_format(source, tmp_path / "timeout_report.json", timeout_s=0.1)

    assert report.status == "not_evaluated"
    assert "limit" in report.errors[0] or "terminated" in report.errors[0]
