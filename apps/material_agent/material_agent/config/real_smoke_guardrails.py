# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-mock preflight guardrails for Material Agent real-smoke evidence runs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DISALLOWED_BACKEND_VALUES = {"fake", "mock"}
_FAKE_CREATE_MATERIALS_ALIASES = {"auto", "auto-for-test"}
_BACKEND_KEYS = {
    "backend",
    "backend_class",
    "backend_name",
    "embedding_backend",
    "embedding_service",
    "service",
}
_FAKE_BEHAVIOR_KEYS = {"fake_behavior", "fake_material_creation_behavior"}
_SIMULATE_KEYS = {"simulate", "simulate_mode", "simulation"}
_FAKE_CLASS_NAME = "FakeMaterialCreationBackend"
_SUPPORTED_SCAN_SUFFIXES = {
    ".cfg",
    ".conf",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

_BACKEND_DECLARATION_RE = re.compile(
    r"""(?ix)
    ["']?
    (?:backend|backend_class|backend_name|embedding_backend|embedding_service|service)
    ["']?
    \s*[:=]\s*
    ["']?
    (?:fake|mock)
    ["']?
    \b
    """
)
_SIMULATE_LOG_RE = re.compile(
    r"""(?ix)
    (?:\bsimulate\ mode\b|--simulate\b|
       \bsimulate:\s*
       (?:true|yes|on|1|wrote|phase|appended|loaded|0\s+predictions)\b)
    """
)
_FAKE_BACKEND_TEXT_RE = re.compile(
    rf"(?i)(?:{_FAKE_CLASS_NAME}|\bfake material creation backend\b|\bfake backend\b)"
)


@dataclass(frozen=True)
class RealSmokeDisqualifier:
    """One finding that prevents a run from being used as real-smoke evidence."""

    code: str
    location: str
    message: str
    source: str = "config"


class RealSmokeGuardrailError(ValueError):
    """Raised when no-mock/no-fake real-smoke guardrails fail."""

    def __init__(self, disqualifiers: Iterable[RealSmokeDisqualifier]) -> None:
        self.disqualifiers = tuple(disqualifiers)
        super().__init__(_format_error(self.disqualifiers))


def collect_real_smoke_disqualifiers(
    config: Mapping[str, Any] | None,
    *,
    artifact_paths: Iterable[str | Path] | None = None,
    simulate: bool = False,
    scheduled_steps: Iterable[str] | None = None,
) -> tuple[RealSmokeDisqualifier, ...]:
    """Collect fake/mock/simulate markers that disqualify real-smoke evidence.

    The scan is intentionally credential-blind: it reports only locations and
    disqualifier categories, never provider keys or full config sections.
    """

    findings: list[RealSmokeDisqualifier] = []
    scheduled = set(scheduled_steps) if scheduled_steps is not None else None

    if simulate:
        findings.append(
            RealSmokeDisqualifier(
                code="simulate_mode",
                location="pipeline.simulate",
                message="simulate mode cannot produce real-smoke evidence",
            )
        )

    if config is not None:
        _scan_mapping(
            config,
            path=(),
            source="config",
            findings=findings,
            scheduled_steps=scheduled,
        )

    for artifact_path in artifact_paths or ():
        _scan_artifact_path(Path(artifact_path), findings)

    return tuple(findings)


def validate_real_smoke_guardrails(
    config: Mapping[str, Any] | None,
    *,
    artifact_paths: Iterable[str | Path] | None = None,
    simulate: bool = False,
    scheduled_steps: Iterable[str] | None = None,
) -> None:
    """Raise if a config or generated artifact is not eligible as real evidence."""

    disqualifiers = collect_real_smoke_disqualifiers(
        config,
        artifact_paths=artifact_paths,
        simulate=simulate,
        scheduled_steps=scheduled_steps,
    )
    if disqualifiers:
        raise RealSmokeGuardrailError(disqualifiers)


def _scan_mapping(
    value: Any,
    *,
    path: tuple[str, ...],
    source: str,
    findings: list[RealSmokeDisqualifier],
    scheduled_steps: set[str] | None,
) -> None:
    if isinstance(value, Mapping):
        if _disabled_or_unscheduled_step(value, path, scheduled_steps):
            return
        if path == ("steps", "create_materials") and "backend" not in value:
            findings.append(
                RealSmokeDisqualifier(
                    code="fake_backend_default",
                    location=_format_location((*path, "backend")),
                    message=(
                        "an active create_materials step without an explicit backend "
                        "defaults to fake and is not allowed for real-smoke evidence"
                    ),
                    source=source,
                )
            )

        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            normalized_key = _normalize_key(key_text)
            location = _format_location(child_path)

            if _is_backend_key(normalized_key):
                backend_value = _disallowed_backend_value(child)
                if backend_value is not None:
                    findings.append(
                        RealSmokeDisqualifier(
                            code=f"{backend_value}_backend",
                            location=location,
                            message=(
                                f"{backend_value!r} backend value is not allowed "
                                "for real-smoke evidence"
                            ),
                            source=source,
                        )
                    )
                elif _is_fake_create_materials_alias(child, child_path):
                    findings.append(
                        RealSmokeDisqualifier(
                            code="fake_backend_alias",
                            location=location,
                            message=(
                                f"{str(child).strip()!r} resolves to the fake material "
                                "creation backend and is not allowed for real-smoke "
                                "evidence"
                            ),
                            source=source,
                        )
                    )

            if normalized_key in _FAKE_BEHAVIOR_KEYS and child is not None:
                findings.append(
                    RealSmokeDisqualifier(
                        code="fake_material_creation_behavior",
                        location=location,
                        message=(
                            "fake material creation behavior is configured for "
                            "an active scan path"
                        ),
                        source=source,
                    )
                )

            if normalized_key in _SIMULATE_KEYS and child is True:
                findings.append(
                    RealSmokeDisqualifier(
                        code="simulate_mode",
                        location=location,
                        message="simulate mode cannot produce real-smoke evidence",
                        source=source,
                    )
                )

            _scan_mapping(
                child,
                path=child_path,
                source=source,
                findings=findings,
                scheduled_steps=scheduled_steps,
            )
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _scan_mapping(
                child,
                path=(*path, f"[{index}]"),
                source=source,
                findings=findings,
                scheduled_steps=scheduled_steps,
            )
    elif isinstance(value, str) and _FAKE_CLASS_NAME in value:
        findings.append(
            RealSmokeDisqualifier(
                code="fake_material_creation_backend",
                location=_format_location(path),
                message=(
                    "FakeMaterialCreationBackend mention disqualifies "
                    "real-smoke evidence"
                ),
                source=source,
            )
        )


def _scan_artifact_path(path: Path, findings: list[RealSmokeDisqualifier]) -> None:
    if not path.exists():
        findings.append(
            RealSmokeDisqualifier(
                code="missing_artifact",
                location=str(path),
                message="artifact path could not be scanned",
                source=str(path),
            )
        )
        return

    if path.is_file() and path.suffix.lower() not in _SUPPORTED_SCAN_SUFFIXES:
        findings.append(
            RealSmokeDisqualifier(
                code="missing_evidence",
                location=str(path),
                message="artifact file type is not supported for evidence scanning",
                source=str(path),
            )
        )
        return

    scan_files = _scan_files(path)
    if path.is_dir() and not scan_files:
        findings.append(
            RealSmokeDisqualifier(
                code="missing_evidence",
                location=str(path),
                message="artifact directory contained no supported evidence files",
                source=str(path),
            )
        )
        return

    for scan_file in scan_files:
        if not scan_file.is_file():
            continue
        source = str(scan_file)
        try:
            text = scan_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            findings.append(
                RealSmokeDisqualifier(
                    code="unreadable_artifact",
                    location=source,
                    message="artifact path could not be read for guardrail scan",
                    source=source,
                )
            )
            continue

        if _BACKEND_DECLARATION_RE.search(text):
            findings.append(
                RealSmokeDisqualifier(
                    code="artifact_mock_or_fake_backend",
                    location=source,
                    message="artifact contains a mock/fake backend declaration",
                    source=source,
                )
            )

        if _SIMULATE_LOG_RE.search(text):
            findings.append(
                RealSmokeDisqualifier(
                    code="artifact_simulate_mode",
                    location=source,
                    message="artifact contains simulate-mode evidence",
                    source=source,
                )
            )

        if _FAKE_BACKEND_TEXT_RE.search(text):
            findings.append(
                RealSmokeDisqualifier(
                    code="artifact_fake_backend",
                    location=source,
                    message="artifact contains fake material backend evidence",
                    source=source,
                )
            )

        structured = _load_structured_artifact(scan_file, text)
        if structured is not None:
            _scan_mapping(
                structured,
                path=(),
                source=source,
                findings=findings,
                scheduled_steps=None,
            )


def _scan_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    return tuple(
        sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in _SUPPORTED_SCAN_SUFFIXES
        )
    )


def _load_structured_artifact(path: Path, text: str) -> Any | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if suffix in {".json", ".yaml", ".yml"}:
            return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError):
        return None
    return None


def _disabled_or_unscheduled_step(
    value: Mapping[str, Any],
    path: tuple[str, ...],
    scheduled_steps: set[str] | None,
) -> bool:
    if len(path) != 2 or path[0] != "steps":
        return False
    step_name = path[1]
    if scheduled_steps is not None and step_name not in scheduled_steps:
        return True
    return value.get("enabled") is False


def _is_backend_key(key: str) -> bool:
    return key in _BACKEND_KEYS or key.endswith("_backend")


def _disallowed_backend_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in _DISALLOWED_BACKEND_VALUES:
        return normalized
    return None


def _is_fake_create_materials_alias(
    value: Any,
    path: tuple[str, ...],
) -> bool:
    if not isinstance(value, str):
        return False
    normalized_path = tuple(_normalize_key(part) for part in path)
    return normalized_path == ("steps", "create_materials", "backend") and (
        value.strip().lower() in _FAKE_CREATE_MATERIALS_ALIASES
    )


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _format_location(path: tuple[str, ...]) -> str:
    if not path:
        return "<root>"
    location = ""
    for part in path:
        if part.startswith("["):
            location += part
        elif location:
            location += f".{part}"
        else:
            location = part
    return location


def _format_error(disqualifiers: tuple[RealSmokeDisqualifier, ...]) -> str:
    header = (
        "Real-smoke guardrails failed; this run cannot be used as "
        "no-mock/no-fake evidence."
    )
    lines = [header]
    for finding in disqualifiers:
        lines.append(
            f"- {finding.code} at {finding.source}:{finding.location}: "
            f"{finding.message}"
        )
    return "\n".join(lines)
