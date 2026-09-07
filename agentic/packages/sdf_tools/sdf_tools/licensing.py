# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed license admission for SDF backend extensions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import LicensePolicyError

_LGPL_PATTERN = re.compile(r"(?:^|[^A-Z])LGPL(?:[^A-Z]|$)|GNU LESSER GENERAL PUBLIC", re.I)
_STRONG_COPYLEFT_PATTERN = re.compile(
    r"(?:^|[^A-Z])(?:AGPL|GPL)(?:[^A-Z]|$)|GNU (?:AFFERO )?GENERAL PUBLIC",
    re.I,
)
_UNKNOWN_LICENSES = frozenset({"", "NOASSERTION", "NONE", "UNKNOWN", "UNLICENSED"})
_NATIVE_ATTESTATION_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ALLOWED_INTRODUCED_LICENSES = frozenset(
    {
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSL-1.0",
        "ISC",
        "MIT",
        "Zlib",
    }
)
_INTRODUCED_SCOPES = frozenset(
    {
        "bundled",
        "runtime",
        "build",
    }
)


class DependencyScope(StrEnum):
    """How a component relates to the backend artifact under review."""

    BUNDLED = "bundled"
    RUNTIME = "runtime"
    BUILD = "build"
    PLATFORM_ABI = "platform_abi"
    PREEXISTING_APPLICATION = "preexisting_application"


@dataclass(frozen=True, slots=True)
class LicenseComponent:
    """One dependency named by a backend's license attestation."""

    name: str
    version: str
    license_expression: str
    scope: DependencyScope
    introduced_by_backend: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "license_expression": self.license_expression,
            "scope": self.scope.value,
            "introduced_by_backend": self.introduced_by_backend,
        }


@dataclass(frozen=True, slots=True)
class BackendLicenseManifest:
    """Reviewed dependency claim attached to a backend descriptor."""

    schema: str
    claim: str
    components: tuple[LicenseComponent, ...]
    native_closure_attestation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim": self.claim,
            "components": [component.as_dict() for component in self.components],
            "native_closure_attestation": self.native_closure_attestation,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def validate_license_manifest(manifest: BackendLicenseManifest) -> None:
    """Reject unknown or LGPL-licensed components introduced by a backend."""

    if manifest.schema != "world-understanding.sdf-backend-license.v1":
        raise LicensePolicyError("unsupported SDF backend license-manifest schema")
    if not manifest.claim:
        raise LicensePolicyError("SDF backend license manifest must state its claim scope")
    if not manifest.components:
        raise LicensePolicyError("SDF backend license manifest must inventory dependencies")

    names: set[str] = set()
    for component in manifest.components:
        if not component.name or component.name in names:
            raise LicensePolicyError("SDF backend license components must have unique names")
        names.add(component.name)
        expression = component.license_expression.strip()
        scope_requires_introduction = component.scope.value in _INTRODUCED_SCOPES
        if component.introduced_by_backend is not scope_requires_introduction:
            expected = "introduced" if scope_requires_introduction else "pre-existing"
            raise LicensePolicyError(
                f"component {component.name!r} has scope {component.scope.value!r} "
                f"and must be classified as {expected}"
            )
        if component.introduced_by_backend and expression.upper() in _UNKNOWN_LICENSES:
            raise LicensePolicyError(
                f"backend-introduced component {component.name!r} has no reviewed SPDX license"
            )
        if component.introduced_by_backend and _LGPL_PATTERN.search(expression):
            raise LicensePolicyError(
                f"backend-introduced component {component.name!r} uses forbidden LGPL terms: "
                f"{expression}"
            )
        if component.introduced_by_backend and _STRONG_COPYLEFT_PATTERN.search(expression):
            raise LicensePolicyError(
                f"backend-introduced component {component.name!r} uses forbidden copyleft terms: "
                f"{expression}"
            )
        if component.introduced_by_backend and expression not in _ALLOWED_INTRODUCED_LICENSES:
            raise LicensePolicyError(
                f"backend-introduced component {component.name!r} has an unapproved SPDX "
                f"license selection: {expression}"
            )

    introduced = [component for component in manifest.components if component.introduced_by_backend]
    if not introduced:
        raise LicensePolicyError("SDF backend manifest has no introduced component inventory")
    if any(component.scope is DependencyScope.BUNDLED for component in manifest.components):
        if not _NATIVE_ATTESTATION_PATTERN.fullmatch(manifest.native_closure_attestation or ""):
            raise LicensePolicyError(
                "SDF backend manifests with bundled native components require a SHA-256 "
                "closure-policy attestation"
            )
