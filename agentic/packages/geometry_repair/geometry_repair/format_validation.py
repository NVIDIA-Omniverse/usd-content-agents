# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative, non-mutating source-format validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, file_sha256
from .models import SourceFormatValidationReport
from .process_limits import (
    bounded_process_environment,
    limit_address_space,
    limit_cpu_time,
    limit_file_size,
)

_USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
_GLTF_SUFFIXES = {".gltf", ".glb"}
_MAX_VALIDATOR_OUTPUT_BYTES = 16 * 1024 * 1024
_VALIDATOR_MEMORY_MB = 4096
_USD_MATERIAL_RULES = {
    "MaterialBindingAPIAppliedChecker",
    "NormalMapTextureChecker",
    "ShaderPropertyTypeConformanceChecker",
    "TextureChecker",
}


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _openusd_runtime_version(version: tuple[int, ...]) -> str:
    """Format OpenUSD's `(0, YY, M)` tuple as its public `YY.M` version."""

    if len(version) >= 3 and version[0] == 0:
        return f"{version[1]}.{version[2]}"
    return ".".join(str(item) for item in version)


def _usd_validation(source: Path) -> SourceFormatValidationReport:
    try:
        from pxr import Usd, UsdUtils
    except Exception as exc:
        return SourceFormatValidationReport(
            source_path=str(source),
            source_sha256=file_sha256(source),
            source_format=source.suffix.lower().lstrip("."),
            status="not_evaluated",
            validator="OpenUSD UsdUtils.ComplianceChecker",
            errors=[f"OpenUSD compliance checker unavailable: {type(exc).__name__}: {exc}"],
        )

    def _run_checker(*, excluded_rules: set[str]) -> tuple[list[str], list[str], list[str]]:
        checker = UsdUtils.ComplianceChecker(
            arkit=False,
            rootPackageOnly=False,
            skipVariants=False,
            verbose=False,
            assetLevelChecks=True,
        )
        if excluded_rules:
            checker._rules = [
                rule for rule in checker._rules if rule.__class__.__name__ not in excluded_rules
            ]
        checker.CheckCompliance(str(source))
        errors = [str(item) for item in checker.GetErrors()]
        failed_checks = [str(item) for item in checker.GetFailedChecks()]
        warnings = [str(item) for item in checker.GetWarnings()]
        return errors, failed_checks, warnings

    excluded_rules: list[str] = []
    full_checker_error: str | None = None
    try:
        errors, failed_checks, warnings = _run_checker(excluded_rules=set())
    except Exception as exc:
        full_checker_error = f"{type(exc).__name__}: {exc}"
        excluded_rules = sorted(_USD_MATERIAL_RULES)
        try:
            errors, failed_checks, warnings = _run_checker(excluded_rules=_USD_MATERIAL_RULES)
            warnings.append(
                "OpenUSD material/shader compliance rules were not evaluated after the full "
                f"checker failed: {full_checker_error}"
            )
        except Exception as fallback_exc:
            return SourceFormatValidationReport(
                source_path=str(source),
                source_sha256=file_sha256(source),
                source_format=source.suffix.lower().lstrip("."),
                status="not_evaluated",
                validator="OpenUSD UsdUtils.ComplianceChecker",
                validator_version=_openusd_runtime_version(Usd.GetVersion()),
                errors=[
                    "OpenUSD compliance execution failed before evidence was complete: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ],
                metadata={
                    "full_checker_error": full_checker_error,
                    "provider_distribution": "usd-exchange",
                    "provider_version": _distribution_version("usd-exchange"),
                },
            )
    errors.extend(item for item in failed_checks if item not in errors)
    return SourceFormatValidationReport(
        source_path=str(source),
        source_sha256=file_sha256(source),
        source_format=source.suffix.lower().lstrip("."),
        status="fail" if errors else "not_evaluated" if excluded_rules else "pass",
        validator="OpenUSD UsdUtils.ComplianceChecker",
        validator_version=_openusd_runtime_version(Usd.GetVersion()),
        errors=errors,
        warnings=warnings,
        metadata={
            "failed_check_count": len(failed_checks),
            "excluded_material_rule_names": excluded_rules,
            "full_checker_error": full_checker_error,
            "provider_distribution": "usd-exchange",
            "provider_version": _distribution_version("usd-exchange"),
        },
    )


def _gltf_validator_executable() -> Path | None:
    configured = os.environ.get("GEOMETRY_REPAIR_GLTF_VALIDATOR_EXECUTABLE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
    discovered = shutil.which("gltf_validator")
    return Path(discovered).resolve() if discovered else None


def _read_bounded_stream(stream: Any) -> str:
    stream.seek(0)
    payload = stream.read(_MAX_VALIDATOR_OUTPUT_BYTES + 1)
    if len(payload) > _MAX_VALIDATOR_OUTPUT_BYTES:
        raise RuntimeError(f"validator output exceeded {_MAX_VALIDATOR_OUTPUT_BYTES} bytes")
    return payload.decode("utf-8", errors="replace")


def _message_severity(message: dict[str, Any]) -> int | None:
    value = message.get("severity")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _gltf_validation(source: Path, *, timeout_s: float) -> SourceFormatValidationReport:
    executable = _gltf_validator_executable()
    if executable is None:
        return SourceFormatValidationReport(
            source_path=str(source),
            source_sha256=file_sha256(source),
            source_format=source.suffix.lower().lstrip("."),
            status="not_evaluated",
            validator="Khronos glTF Validator",
            errors=[
                "Khronos glTF Validator executable is unavailable; set "
                "GEOMETRY_REPAIR_GLTF_VALIDATOR_EXECUTABLE."
            ],
        )
    try:
        with tempfile.TemporaryFile() as stdout_stream, tempfile.TemporaryFile() as stderr_stream:
            completed = subprocess.run(
                [
                    str(executable),
                    "--stdout",
                    "--no-write-timestamp",
                    "--no-absolute-path",
                    str(source),
                ],
                stdout=stdout_stream,
                stderr=stderr_stream,
                timeout=timeout_s,
                check=False,
                start_new_session=True,
            )
            stdout = _read_bounded_stream(stdout_stream)
            stderr = _read_bounded_stream(stderr_stream)
        payload: dict[str, Any] = json.loads(stdout)
        issues = payload.get("issues") if isinstance(payload.get("issues"), dict) else {}
        messages = issues.get("messages") if isinstance(issues.get("messages"), list) else []
        errors = [
            str(item.get("message") or item.get("code") or item)
            for item in messages
            if isinstance(item, dict) and _message_severity(item) == 0
        ]
        warnings = [
            str(item.get("message") or item.get("code") or item)
            for item in messages
            if isinstance(item, dict) and _message_severity(item) != 0
        ]
        if completed.returncode < 0:
            return SourceFormatValidationReport(
                source_path=str(source),
                source_sha256=file_sha256(source),
                source_format=source.suffix.lower().lstrip("."),
                status="not_evaluated",
                validator="Khronos glTF Validator",
                errors=[f"validator was terminated by signal {-completed.returncode}"],
                metadata={"executable": str(executable), "returncode": completed.returncode},
            )
        if completed.returncode != 0 and not errors:
            errors.append(stderr.strip() or f"validator exited {completed.returncode}")
        return SourceFormatValidationReport(
            source_path=str(source),
            source_sha256=file_sha256(source),
            source_format=source.suffix.lower().lstrip("."),
            status="fail" if errors or completed.returncode != 0 else "pass",
            validator="Khronos glTF Validator",
            validator_version=(
                str(payload.get("validatorVersion"))
                if payload.get("validatorVersion") is not None
                else None
            ),
            errors=errors,
            warnings=warnings,
            metadata={
                "executable": str(executable),
                "returncode": completed.returncode,
                "issue_counts": {
                    key: value
                    for key, value in issues.items()
                    if key != "messages" and isinstance(value, int | float | str | bool)
                },
            },
        )
    except subprocess.TimeoutExpired:
        return SourceFormatValidationReport(
            source_path=str(source),
            source_sha256=file_sha256(source),
            source_format=source.suffix.lower().lstrip("."),
            status="not_evaluated",
            validator="Khronos glTF Validator",
            errors=[f"glTF validation exceeded {timeout_s:.3f}s wall-clock limit"],
            metadata={"executable": str(executable), "timeout_s": timeout_s},
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
        return SourceFormatValidationReport(
            source_path=str(source),
            source_sha256=file_sha256(source),
            source_format=source.suffix.lower().lstrip("."),
            status="not_evaluated",
            validator="Khronos glTF Validator",
            errors=[f"glTF validation failed: {type(exc).__name__}: {exc}"],
            metadata={"executable": str(executable)},
        )


def _validator_name(source: Path) -> str:
    return (
        "OpenUSD UsdUtils.ComplianceChecker"
        if source.suffix.lower() in _USD_SUFFIXES
        else "Khronos glTF Validator"
    )


def _isolated_validation(
    source: Path,
    *,
    timeout_s: float,
) -> SourceFormatValidationReport:
    with tempfile.TemporaryDirectory(prefix="geometry-repair-format-validator-") as directory:
        result_path = Path(directory) / "result.json"
        command = [
            sys.executable,
            "-m",
            "geometry_repair.format_validation",
            "--worker",
            "--source",
            str(source),
            "--result",
            str(result_path),
            "--timeout-s",
            str(timeout_s),
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=bounded_process_environment(),
                check=False,
                timeout=max(timeout_s + 5.0, 0.001),
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return SourceFormatValidationReport(
                source_path=str(source),
                source_sha256=file_sha256(source),
                source_format=source.suffix.lower().lstrip("."),
                status="not_evaluated",
                validator=_validator_name(source),
                errors=[f"source-format validator exceeded {timeout_s:.3f}s wall-clock limit"],
                metadata={
                    "timeout_s": timeout_s,
                    "memory_limit_mb": _VALIDATOR_MEMORY_MB,
                    "output_limit_bytes": _MAX_VALIDATOR_OUTPUT_BYTES,
                },
            )
        if completed.returncode != 0 or not result_path.is_file():
            return SourceFormatValidationReport(
                source_path=str(source),
                source_sha256=file_sha256(source),
                source_format=source.suffix.lower().lstrip("."),
                status="not_evaluated",
                validator=_validator_name(source),
                errors=[
                    "source-format validator was terminated or failed before producing "
                    f"evidence (return code {completed.returncode})"
                ],
                metadata={
                    "returncode": completed.returncode,
                    "timeout_s": timeout_s,
                    "memory_limit_mb": _VALIDATOR_MEMORY_MB,
                    "output_limit_bytes": _MAX_VALIDATOR_OUTPUT_BYTES,
                },
            )
        try:
            report = SourceFormatValidationReport.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            return SourceFormatValidationReport(
                source_path=str(source),
                source_sha256=file_sha256(source),
                source_format=source.suffix.lower().lstrip("."),
                status="not_evaluated",
                validator=_validator_name(source),
                errors=[
                    f"source-format validator evidence was invalid: {type(exc).__name__}: {exc}"
                ],
            )
        return report.model_copy(
            update={
                "metadata": {
                    **report.metadata,
                    "execution_scope": "isolated_subprocess",
                    "timeout_s": timeout_s,
                    "memory_limit_mb": _VALIDATOR_MEMORY_MB,
                    "output_limit_bytes": _MAX_VALIDATOR_OUTPUT_BYTES,
                }
            }
        )


def validate_source_format(
    source_path: str | Path,
    output_path: str | Path,
    *,
    timeout_s: float = 120.0,
) -> SourceFormatValidationReport:
    """Validate a source with its authoritative format validator when available."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    source = Path(source_path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in _USD_SUFFIXES | _GLTF_SUFFIXES:
        report = _isolated_validation(source, timeout_s=timeout_s)
    else:
        report = SourceFormatValidationReport(
            source_path=str(source),
            source_sha256=file_sha256(source),
            source_format=suffix.lstrip("."),
            status="not_applicable",
            metadata={"reason": "no independent authoritative validator is configured"},
        )
    target = Path(output_path).expanduser().resolve()
    report = report.model_copy(update={"report_path": str(target)})
    atomic_write_json(target, report)
    return report


def _worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geometry-repair-format-validator")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--timeout-s", required=True, type=float)
    args = parser.parse_args(argv)

    limit_address_space(_VALIDATOR_MEMORY_MB)
    limit_file_size(_MAX_VALIDATOR_OUTPUT_BYTES)
    # Permit interpreter and shared-library startup overhead; wall-clock
    # enforcement remains the tighter limit for the validator operation.
    limit_cpu_time(args.timeout_s + 5.0)
    source = args.source.expanduser().resolve()
    if source.suffix.lower() in _USD_SUFFIXES:
        report = _usd_validation(source)
    elif source.suffix.lower() in _GLTF_SUFFIXES:
        report = _gltf_validation(source, timeout_s=args.timeout_s)
    else:
        return 2
    target = args.result.expanduser().resolve()
    atomic_write_json(target, report.model_copy(update={"report_path": str(target)}))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess.
    sys.exit(_worker_main())
