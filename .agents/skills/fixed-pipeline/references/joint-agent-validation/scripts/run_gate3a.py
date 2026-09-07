#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the diagnostic Gate 3A profile on one Joint Agent USD artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "joint-agent-gate3a-diagnostic-v1"
PROFILE_NAME = "articulated-prop-v1"
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
REQUIRED_EXTENSIONS = (
    "omni.asset_validator.core",
    "isaacsim.asset.validation",
)
PROFILE_CATEGORIES = (
    "Basic",
    "Omni:Basic",
    "Usd:Performance",
    "Usd:Schema",
    "Omni:Geometry",
    "Omni:Material",
    "Omni:Layout",
    "Omni:Skel",
    "Usd:Physics",
    "Omni:SimReady",
    "IsaacSim.PhysicsRules",
    "IsaacSim.SimReadyAssetRules",
)
NOT_APPLICABLE_CATEGORIES = ("AtomicAsset", "IsaacSim.RobotRules")
NOT_APPLICABLE_RULES = (
    "GroundTruthCapabilityChecker",
    "NonVisualSensorCapabilityChecker",
    "VisualSensorCapabilityChecker",
)
ISAAC_PACKAGES = (
    "isaacsim",
    "isaacsim-app",
    "isaacsim-asset",
    "isaacsim-extscache-kit",
    "isaacsim-extscache-physics",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Joint Agent Gate 3A diagnostic profile with Isaac Sim "
            "Asset Validator."
        )
    )
    parser.add_argument("asset", type=Path, help="USD-family asset to validate")
    parser.add_argument(
        "--report",
        type=Path,
        help="Output JSON path; defaults beside the asset",
    )
    parser.add_argument("--max-issues", type=int, default=500)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero for validation warnings or failures",
    )
    return parser.parse_args(argv)


def issue_to_dict(issue: Any) -> dict[str, Any]:
    rule = getattr(issue, "rule", None)
    severity = getattr(issue, "severity", None)
    requirement = getattr(issue, "requirement", None)
    location = getattr(issue, "at", None)
    return {
        "message": getattr(issue, "message", None),
        "severity": getattr(severity, "name", str(severity) if severity else None),
        "rule": getattr(rule, "__name__", None),
        "at": str(location) if location is not None else None,
        "requirement": (
            {
                "code": getattr(requirement, "code", None),
                "version": getattr(requirement, "version", None),
            }
            if requirement is not None
            else None
        ),
    }


def summarize_issues(
    issues: list[Any], rule_to_category: dict[str, str]
) -> dict[str, Any]:
    severity_counts = Counter(
        getattr(getattr(issue, "severity", None), "name", "UNKNOWN") for issue in issues
    )
    rule_counts: dict[str, Counter[str]] = defaultdict(Counter)
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for issue in issues:
        severity = getattr(getattr(issue, "severity", None), "name", "UNKNOWN")
        rule = getattr(getattr(issue, "rule", None), "__name__", "UNKNOWN")
        category = rule_to_category.get(rule, "UNKNOWN")
        rule_counts[rule][severity] += 1
        category_counts[category][severity] += 1

    hard_failure_count = severity_counts.get("ERROR", 0) + severity_counts.get(
        "FAILURE", 0
    )
    if hard_failure_count:
        status = "fail"
    elif severity_counts.get("WARNING", 0):
        status = "warning"
    else:
        status = "pass"
    return {
        "status": status,
        "issue_count": len(issues),
        "hard_failure_count": hard_failure_count,
        "severity_counts": dict(sorted(severity_counts.items())),
        "rule_counts": {
            rule: dict(sorted(counts.items()))
            for rule, counts in sorted(rule_counts.items())
        },
        "category_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_counts.items())
        },
    }


def exit_code(status: str, *, strict: bool) -> int:
    if status in {"blocked", "error"}:
        return 2
    if strict and status != "pass":
        return 1
    return 0


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in ISAAC_PACKAGES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def resolver_working_directory(asset: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(asset.parent)
    try:
        yield
    finally:
        os.chdir(original)


def report_path_for(asset: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    return asset.parent / f"{asset.stem}-gate3a.json"


def report_destination_error(asset: Path, report_path: Path) -> str | None:
    if report_path == asset:
        return "Gate 3A report path must not alias the input asset."
    if report_path.exists() or report_path.is_symlink():
        return (
            "Gate 3A report already exists; choose a fresh path so prior "
            f"evidence is not replaced: {report_path}"
        )
    return None


def base_report(asset: Path, report_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "validator": (
            "Isaac Sim isaacsim.asset.validation via omni.asset_validator.core"
        ),
        "profile": {
            "name": PROFILE_NAME,
            "configured_categories": list(PROFILE_CATEGORIES),
            "not_applicable_categories": list(NOT_APPLICABLE_CATEGORIES),
            "not_applicable_rules": list(NOT_APPLICABLE_RULES),
            "status_basis": (
                "ERROR and FAILURE issues fail the target; WARNING issues "
                "produce warning status."
            ),
        },
        "asset_path": str(asset),
        "asset_sha256": None,
        "asset_sha256_before": None,
        "asset_sha256_after": None,
        "artifact_evidence": {
            "sha256_scope": None,
            "dependency_closure_captured": False,
            "packaged_members_captured": False,
            "diagnostic_only": True,
        },
        "report_path": str(report_path),
        "runtime": {
            "python": sys.executable,
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "package_versions": package_versions(),
            "extensions_enabled": {},
        },
        "registered_categories": [],
        "registered_rules_by_category": {},
        "enabled_categories": [],
        "missing_categories": [],
        "status": "blocked",
        "issue_count": 0,
        "hard_failure_count": 0,
        "severity_counts": {},
        "rule_counts": {},
        "category_counts": {},
        "issues": [],
        "errors": [],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("Gate 3A report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture_asset_sha256_after(asset: Path, payload: dict[str, Any]) -> None:
    try:
        asset_sha256_after = sha256_file(asset)
    except Exception as exc:  # noqa: BLE001 - preserve a structured report.
        payload["status"] = "error"
        payload["hard_failure_count"] = max(1, int(payload["hard_failure_count"]))
        payload["severity_counts"]["ERROR"] = max(
            1, int(payload["severity_counts"].get("ERROR", 0))
        )
        payload["errors"].append(
            "Could not hash input asset after Gate 3A validation: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    payload["asset_sha256_after"] = asset_sha256_after
    if asset_sha256_after != payload["asset_sha256_before"]:
        payload["status"] = "error"
        payload["hard_failure_count"] = max(1, int(payload["hard_failure_count"]))
        payload["errors"].append("Input asset bytes changed during Gate 3A validation.")


def finish(
    report_path: Path,
    payload: dict[str, Any],
    *,
    strict: bool,
    kit_started: bool,
) -> int:
    if payload["asset_sha256_before"] is not None:
        capture_asset_sha256_after(Path(str(payload["asset_path"])), payload)
    try:
        write_report(report_path, payload)
    except OSError as exc:
        sys.stderr.write(f"Could not create Gate 3A report: {exc}\n")
        sys.stderr.flush()
        if kit_started:
            os._exit(2)
        return 2
    summary = {
        "status": payload["status"],
        "issue_count": payload["issue_count"],
        "hard_failure_count": payload["hard_failure_count"],
        "report_path": str(report_path),
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()
    sys.stderr.flush()
    code = exit_code(str(payload["status"]), strict=strict)
    if kit_started:
        os._exit(code)
    return code


def run(args: argparse.Namespace) -> int:
    requested_asset = args.asset.expanduser().resolve(strict=False)
    report_path = report_path_for(requested_asset, args.report)
    if destination_error := report_destination_error(requested_asset, report_path):
        sys.stderr.write(destination_error + "\n")
        return 2
    payload = base_report(requested_asset, report_path)

    try:
        asset = requested_asset.resolve(strict=True)
    except OSError as exc:
        payload["errors"] = [f"Asset path is unavailable: {exc}"]
        return finish(report_path, payload, strict=args.strict, kit_started=False)
    payload["asset_path"] = str(asset)

    if not asset.is_file() or asset.suffix.lower() not in USD_SUFFIXES:
        payload["errors"] = [
            "Gate 3A input must be a regular .usd, .usda, .usdc, or .usdz file."
        ]
        return finish(report_path, payload, strict=args.strict, kit_started=False)

    try:
        asset_sha256_before = sha256_file(asset)
    except Exception as exc:  # noqa: BLE001 - preserve a structured report.
        payload["status"] = "error"
        payload["hard_failure_count"] = 1
        payload["severity_counts"] = {"ERROR": 1}
        payload["errors"] = [
            "Could not hash input asset before Gate 3A validation: "
            f"{type(exc).__name__}: {exc}"
        ]
        return finish(report_path, payload, strict=args.strict, kit_started=False)
    payload["asset_sha256"] = asset_sha256_before
    payload["asset_sha256_before"] = asset_sha256_before
    payload["artifact_evidence"] = {
        "sha256_scope": (
            "usdz_archive_bytes_including_packaged_members"
            if asset.suffix.lower() == ".usdz"
            else "root_layer_bytes_only"
        ),
        "dependency_closure_captured": False,
        "packaged_members_captured": asset.suffix.lower() == ".usdz",
        "diagnostic_only": asset.suffix.lower() != ".usdz",
    }

    if os.environ.get("OMNI_KIT_ACCEPT_EULA") != "YES":
        payload["errors"] = [
            "Review and accept the Isaac Sim license, then set "
            "OMNI_KIT_ACCEPT_EULA=YES before running Gate 3A."
        ]
        return finish(report_path, payload, strict=args.strict, kit_started=False)

    try:
        from isaacsim import SimulationApp
    except Exception as exc:  # noqa: BLE001 - produce a structured blocker.
        payload["errors"] = [
            f"Isaac Sim Python is unavailable: {type(exc).__name__}: {exc}"
        ]
        return finish(report_path, payload, strict=args.strict, kit_started=False)

    launch_config = {
        "headless": True,
        "renderer": "Minimal",
        "create_new_stage": False,
        "disable_viewport_updates": True,
        "extra_args": [
            "--/log/outputStreamLevel=Error",
            "--/telemetry/enableAnonymousData=false",
            "--/privacy/usage=false",
            "--/privacy/performance=false",
            "--/privacy/personalization=false",
        ],
    }
    try:
        _simulation_app = SimulationApp(launch_config)
    except Exception as exc:  # noqa: BLE001 - produce a structured blocker.
        payload["errors"] = [
            f"Isaac Sim Kit could not start: {type(exc).__name__}: {exc}"
        ]
        return finish(report_path, payload, strict=args.strict, kit_started=False)

    start_time = time.monotonic()
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        enabled_extensions = {
            name: bool(manager.set_extension_enabled_immediate(name, True))
            for name in REQUIRED_EXTENSIONS
        }
        payload["runtime"]["extensions_enabled"] = enabled_extensions
        if not all(enabled_extensions.values()):
            payload["errors"] = [
                "Required Isaac Sim Asset Validator extensions could not be enabled."
            ]
            return finish(report_path, payload, strict=args.strict, kit_started=True)

        import isaacsim.asset.validation  # noqa: F401
        import omni.asset_validator.core as asset_validator

        registered_categories = list(
            asset_validator.ValidationRulesRegistry.categories()
        )
        registered_rules = {
            category: [
                rule.__name__
                for rule in asset_validator.ValidationRulesRegistry.rules(category)
            ]
            for category in registered_categories
        }
        missing_categories = [
            category
            for category in PROFILE_CATEGORIES
            if category not in registered_categories
        ]
        payload["registered_categories"] = registered_categories
        payload["registered_rules_by_category"] = registered_rules
        payload["missing_categories"] = missing_categories
        if missing_categories:
            payload["errors"] = [
                "Gate 3A profile categories are missing from this runtime: "
                + ", ".join(missing_categories)
            ]
            return finish(report_path, payload, strict=args.strict, kit_started=True)

        payload["enabled_categories"] = list(PROFILE_CATEGORIES)
        ignored_rules = set(NOT_APPLICABLE_RULES)
        engine = asset_validator.ValidationEngine(init_rules=False, variants=True)
        enabled_rules: dict[str, list[str]] = {}
        for category in PROFILE_CATEGORIES:
            enabled_rules[category] = []
            for rule in asset_validator.ValidationRulesRegistry.rules(category):
                if rule.__name__ in ignored_rules:
                    continue
                engine.enable_rule(rule)
                enabled_rules[category].append(rule.__name__)
        payload["enabled_rules_by_category"] = enabled_rules
        empty_categories = [
            category for category, rules in enabled_rules.items() if not rules
        ]
        if empty_categories:
            payload["status"] = "blocked"
            payload["errors"] = [
                "Gate 3A categories contain no enabled rules: "
                + ", ".join(empty_categories)
            ]
            return finish(report_path, payload, strict=args.strict, kit_started=True)
        rule_to_category = {
            rule_name: category
            for category, rules in registered_rules.items()
            for rule_name in rules
        }

        with resolver_working_directory(asset):
            validation_result = engine.validate(str(asset))
            issues = list(validation_result.issues())
        payload.update(summarize_issues(issues, rule_to_category))
        payload["issues"] = [
            issue_to_dict(issue) for issue in issues[: args.max_issues]
        ]
        payload["issues_truncated"] = max(0, len(issues) - args.max_issues)
    except Exception as exc:  # noqa: BLE001 - preserve validator diagnostics.
        payload["status"] = "error"
        payload["hard_failure_count"] = 1
        payload["severity_counts"] = {"ERROR": 1}
        payload["errors"] = [f"{type(exc).__name__}: {exc}"]
    payload["elapsed_seconds"] = round(time.monotonic() - start_time, 3)
    return finish(report_path, payload, strict=args.strict, kit_started=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_issues < 1:
        raise SystemExit("--max-issues must be at least 1")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
