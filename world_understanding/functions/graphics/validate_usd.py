# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD asset validation using NVIDIA USD Validation.

This module provides local validation without requiring Kit or NVCF.
It uses the usd-validation-nvidia pip package, which runs purely on Python
and OpenUSD.

Install: uv pip install usd-validation-nvidia
"""

import base64
import hashlib
import logging
import sys
import time
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# Category names exposed by usd-validation-nvidia 1.19.x.
AVAILABLE_VALIDATION_CATEGORIES = [
    "Basic",
    "Geometry",
    "Layer",
    "Layout",
    "Material",
    "Other",
    "Physics",
]
DEFAULT_VALIDATION_CATEGORIES = list(AVAILABLE_VALIDATION_CATEGORIES)

# usd-validation-nvidia no longer exposes the legacy Usd:Schema category.
# Schema and USD-structure checks are now spread across these groups, so every
# agent-specific profile keeps them before adding the domain it authors.
USD_SCHEMA_VALIDATION_CATEGORIES = ["Basic", "Layer", "Layout", "Other"]
MATERIAL_VALIDATION_CATEGORIES = [
    *USD_SCHEMA_VALIDATION_CATEGORIES,
    "Material",
]
TEXTURE_VALIDATION_CATEGORIES = [
    *USD_SCHEMA_VALIDATION_CATEGORIES,
    "Material",
]
# Texture UV readiness is validated by the prepare_uvs report; the Geometry
# category also gates mesh topology/normals, which Texture Agent does not author.
PHYSICS_VALIDATION_CATEGORIES = [
    *USD_SCHEMA_VALIDATION_CATEGORIES,
    "Physics",
]

LEGACY_VALIDATION_CATEGORY_ALIASES = {
    "Omni:Basic": ["Basic"],
    "Omni:Geometry": ["Geometry"],
    "Omni:Layout": ["Layout"],
    "Omni:Material": ["Material"],
    "Omni:Physics": ["Physics"],
    "Usd:Performance": USD_SCHEMA_VALIDATION_CATEGORIES,
    "Usd:Schema": USD_SCHEMA_VALIDATION_CATEGORIES,
}


def normalize_validation_categories(categories: list[str]) -> list[str]:
    """Expand legacy validation category names to current public categories."""
    normalized: list[str] = []
    seen: set[str] = set()
    for category in categories:
        for item in LEGACY_VALIDATION_CATEGORY_ALIASES.get(category, [category]):
            if item not in seen:
                seen.add(item)
                normalized.append(item)
    return normalized


def _installed_distribution_owners(
    relative_path: str,
    actual_sha256: str,
) -> set[str]:
    """Return distributions whose RECORD digest matches one installed file."""

    actual_record_digest = (
        base64.urlsafe_b64encode(bytes.fromhex(actual_sha256)).decode().rstrip("=")
    )
    owners: set[str] = set()
    for distribution in metadata.distributions():
        for entry in distribution.files or []:
            if str(entry) != relative_path or entry.hash is None:
                continue
            if entry.hash.mode == "sha256" and entry.hash.value == actual_record_digest:
                owners.add(str(distribution.metadata.get("Name") or "unknown"))
    return owners


def _usd_validation_binary_set_is_coherent() -> bool:
    """Reject mixed pxr extension ownership before loading native validation code."""

    try:
        import pxr

        package_root = Path(next(iter(pxr.__path__))).resolve()
        relative_paths = (
            "pxr/Tf/_tf.so",
            "pxr/UsdValidation/_usdValidation.so",
        )
        owner_sets: list[set[str]] = []
        for relative_path in relative_paths:
            path = package_root.parent / relative_path
            if not path.is_file():
                return True
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            owners = _installed_distribution_owners(relative_path, digest)
            owner_sets.append(owners)
        known_owner_sets = [owners for owners in owner_sets if owners]
        if not known_owner_sets:
            return True
        if len(known_owner_sets) != len(owner_sets):
            return False
        return bool(set.intersection(*known_owner_sets))
    except (OSError, RuntimeError, StopIteration):
        return True


def _install_usd_validation_stub(reason: str) -> None:
    logger.warning(
        "pxr.UsdValidation is unsafe or unavailable (%s); validation rules that "
        "depend on that extension will be skipped.",
        reason,
    )
    module = ModuleType("pxr.UsdValidation")

    class ValidationRegistry:
        def GetOrLoadValidatorByName(self, _name: str) -> None:
            return None

    module.ValidationRegistry = ValidationRegistry  # type: ignore[attr-defined]
    sys.modules["pxr.UsdValidation"] = module


def _ensure_usd_validation_compat() -> None:
    """Apply compatibility shim for unavailable or broken UsdValidation bindings.

    Some OpenUSD providers expose pxr.UsdValidation with broken C++ bindings
    that crash on import, while others may omit it. The usd_validation_nvidia
    library catches ImportError but not TypeError, so we pre-inject a stub
    module to avoid the crash.
    """
    if "pxr.UsdValidation" in sys.modules:
        return

    if not _usd_validation_binary_set_is_coherent():
        _install_usd_validation_stub("mixed OpenUSD extension ownership")
        return

    try:
        from pxr import UsdValidation  # noqa: F401
    except (ImportError, TypeError):
        # Create a stub that makes UsdValidatorAdapter.__contains__ return False
        # This disables the UsdValidation-based rules but keeps all other rules working
        _install_usd_validation_stub("missing or broken bindings")


def is_available() -> bool:
    """Check if the standalone validator is available.

    Returns:
        True if usd-validation-nvidia is installed and importable
    """
    try:
        _ensure_usd_validation_compat()
        from usd_validation_nvidia import (
            CategoryRuleRegistry,
            IssueFixer,  # noqa: F401
            ValidationEngine,  # noqa: F401
        )

        # Category-scoped validation depends on registry access, so availability
        # must cover more than just constructing the validation engine.
        _ = CategoryRuleRegistry().categories

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _registered_rule_categories() -> dict[str, str]:
    """Return the current validator rule-to-category map.

    The usd-validation-nvidia package owns rule names and category membership,
    so deriving this map from its registry prevents stale local tables when
    rules are renamed or added in new package releases.
    """
    try:
        _ensure_usd_validation_compat()
        from usd_validation_nvidia import CategoryRuleRegistry
    except ImportError as exc:
        raise ImportError(
            "usd-validation-nvidia CategoryRuleRegistry is required for "
            "category-scoped validation"
        ) from exc

    registry = CategoryRuleRegistry()
    rule_categories: dict[str, str] = {}
    for category in registry.categories:
        for rule in registry.get_rules(category):
            rule_categories[rule.__name__] = str(category)
    return rule_categories


def clear_registered_rule_categories_cache() -> None:
    """Clear cached validator registry metadata after dependency changes."""
    _registered_rule_categories.cache_clear()


def _load_rule_categories(*, required_for_filtering: bool) -> dict[str, str]:
    """Load rule categories with explicit scoped/unscoped fallback semantics."""
    try:
        return _registered_rule_categories()
    except Exception as exc:
        message = (
            "usd-validation-nvidia CategoryRuleRegistry is required for "
            "category-scoped validation"
        )
        if required_for_filtering:
            raise RuntimeError(message) from exc
        logger.warning(
            "Could not load usd-validation-nvidia CategoryRuleRegistry; "
            "issues will be reported with Unknown categories",
            exc_info=True,
        )
        return {}


def validate_usd(
    input_path: Path | str,
    categories: list[str] | None = None,
    fix: bool = False,
    output_path: Path | str | None = None,
    stage_timeout: float = 180.0,
) -> dict[str, Any]:
    """Validate a USD file using NVIDIA USD Validation.

    Args:
        input_path: Path to the USD file to validate
        categories: Validation categories to check (default: all)
        fix: Attempt to auto-fix issues
        output_path: Path to export the fixed root layer (only used when fix=True)
        stage_timeout: Not used in standalone mode (kept for API compat)

    Returns:
        Dict matching the NVCF /validate response format:
            - status: "success" or "error"
            - validation_time: Time in seconds
            - issues: List of issue dicts
            - summary: Summary dict with counts
            - categories_checked: List of categories checked
            - fixes: List of fix result dicts (empty when fix=False)

    Raises:
        ImportError: If usd-validation-nvidia is not installed
        ValueError: If input file does not exist
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")

    _ensure_usd_validation_compat()

    from usd_validation_nvidia import IssueFixer, ValidationEngine

    if categories:
        categories = normalize_validation_categories(categories)

    start_time = time.time()

    logger.info("Validating USD locally: %s", input_path)
    if categories:
        logger.info("Categories: %s", ", ".join(categories))

    try:
        engine = ValidationEngine()
        results = engine.validate(str(input_path))

        # Collect all issues
        all_issues = list(results.issues())

        # Filter by category if specified. Category-scoped validation needs the
        # package registry so renamed or newly-added rules do not pass through
        # as "Unknown" and escape the requested gate.
        rule_categories: dict[str, str] | None = None
        if categories:
            rule_categories = _load_rule_categories(required_for_filtering=True)
            cat_set = set(categories)
            filtered_issues = []
            for issue in all_issues:
                issue_category = _infer_category(issue, rule_categories)
                # Keep unmapped issues in scoped runs so a registry/package
                # mismatch cannot silently hide validator findings.
                if issue_category in cat_set or issue_category == "Unknown":
                    filtered_issues.append(issue)
        else:
            filtered_issues = all_issues
            categories = list(DEFAULT_VALIDATION_CATEGORIES)
            rule_categories = _load_rule_categories(required_for_filtering=False)

        # Build structured issue list
        issues_list = []
        severity_counts = {"failures": 0, "warnings": 0, "errors": 0}

        for issue in filtered_issues:
            severity = _map_severity(issue.severity)
            severity_counts[f"{severity}s"] = severity_counts.get(f"{severity}s", 0) + 1

            issue_dict: dict[str, Any] = {
                "severity": severity,
                "rule": _get_rule_name(issue),
                "category": _infer_category(issue, rule_categories),
                "message": str(issue.message),
                "at": str(issue.at) if issue.at else None,
                "suggestion": str(issue.suggestion) if issue.suggestion else None,
            }
            issues_list.append(issue_dict)

        is_valid = severity_counts["failures"] == 0 and severity_counts["errors"] == 0

        # Handle fix mode
        fixes_list: list[dict[str, Any]] = []
        if fix and filtered_issues:
            logger.info("Attempting to fix %d issue(s)...", len(filtered_issues))
            fixer = IssueFixer(str(input_path))
            fix_results = fixer.fix(filtered_issues)

            for fix_result in fix_results:
                exception = getattr(fix_result, "exception", None)
                fixes_list.append(
                    {
                        "rule": _get_rule_name(fix_result.issue),
                        "status": fix_result.status.name.lower(),
                        "message": str(exception) if exception else None,
                    }
                )

            # Save fixed stage if output_path provided
            if output_path:
                output_path = Path(output_path)
                if output_path.suffix.lower() == ".usdz":
                    raise RuntimeError(
                        "Fixed USD output_path must be a writable USD layer "
                        "(.usd, .usda, or .usdc), not a USDZ package."
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                asset = getattr(fixer, "asset", None)
                if asset is None:
                    raise RuntimeError("IssueFixer did not return a fixed USD asset")
                root_layer = asset.GetRootLayer()
                if root_layer is None:
                    raise RuntimeError("IssueFixer asset has no root layer to export")

                # IssueFixer works on the opened asset; export the fixed root
                # layer to the requested output path for downstream tasks.
                exported = root_layer.Export(str(output_path))
                if not exported:
                    logger.error("Failed to save fixed stage to: %s", output_path)
                    raise RuntimeError(f"Failed to save fixed stage to: {output_path}")
                logger.info("Saved fixed stage to: %s", output_path)

        validation_time = time.time() - start_time
        logger.info(
            "Local validation completed in %.2fs: %d issues",
            validation_time,
            len(issues_list),
        )

        return {
            "status": "success",
            "validation_time": validation_time,
            "issues": issues_list,
            "summary": {
                "total_issues": len(issues_list),
                "failures": severity_counts["failures"],
                "warnings": severity_counts["warnings"],
                "errors": severity_counts["errors"],
                "is_valid": is_valid,
            },
            "categories_checked": categories,
            "fixes": fixes_list,
        }

    except Exception as e:
        validation_time = time.time() - start_time
        logger.exception("Local validation failed")
        return {
            "status": "error",
            "validation_time": validation_time,
            "error": str(e),
        }


def _map_severity(severity: Any) -> str:
    """Map NVIDIA USD Validation severity enum to string."""
    name = str(severity.name).lower() if hasattr(severity, "name") else str(severity)
    if "failure" in name:
        return "failure"
    elif "warning" in name:
        return "warning"
    elif "error" in name:
        return "error"
    return "warning"


def _get_rule_name(issue: Any) -> str:
    """Extract human-readable rule name from issue."""
    rule = getattr(issue, "rule", None)
    if rule is None:
        return "Unknown"
    # rule is typically a class like
    # <class 'usd_validation_nvidia._geometry_checker.IndexedPrimvarChecker'>
    name = str(rule)
    if "." in name and "'" in name:
        # Extract class name from repr
        name = name.split("'")[1].split(".")[-1]
    return name


def _infer_category(issue: Any, rule_categories: dict[str, str] | None = None) -> str:
    """Infer the category for an issue based on its rule checker class."""
    rule_name = _get_rule_name(issue)
    if rule_categories is None:
        try:
            rule_categories = _registered_rule_categories()
        except Exception:
            return "Unknown"
    return rule_categories.get(rule_name, "Unknown")
