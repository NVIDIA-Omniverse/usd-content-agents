# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic input identity for trusted local external runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .types import BASE_EXTERNAL_ENVIRONMENT_NAMES, ExternalTuneSpec

_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_PYTHON_ENVIRONMENT_OUTPUT_PREFIX = "PHYSICS_AGENT_RUNTIME_FINGERPRINT_V1="

_PYTHON_ENVIRONMENT_PROBE = r"""
import csv
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import site
import stat
import sys

file_cache = {}


def has_symlink_ancestor(path):
    current = pathlib.Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect installed runtime path: {current}"
            ) from exc
        if stat.S_ISLNK(mode):
            return True
    return False


def installed_file_digest(distribution, relative):
    path = pathlib.Path(distribution.locate_file(relative))
    try:
        resolved = path.resolve(strict=False)
        try:
            mode = resolved.stat().st_mode
        except FileNotFoundError:
            if not has_symlink_ancestor(path):
                raise RuntimeError(f"installed runtime file is missing: {path}")
            digest = hashlib.sha256()
            digest.update(b"missing-through-symlink\0")
            digest.update(str(resolved).encode())
            return "sha256:" + digest.hexdigest(), 0
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"installed runtime path is not a file: {path}")
        cache_key = str(resolved)
        cached = file_cache.get(cache_key)
        if cached is None:
            content_digest = hashlib.sha256()
            size = 0
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content_digest.update(chunk)
                    size += len(chunk)
            cached = ("sha256:" + content_digest.hexdigest(), size)
            file_cache[cache_key] = cached
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"cannot fingerprint installed runtime file: {path}"
        ) from exc
    digest = hashlib.sha256()
    digest.update(str(resolved).encode())
    digest.update(b"\0")
    digest.update(cached[0].encode())
    return "sha256:" + digest.hexdigest(), cached[1]


packages = []
for distribution in metadata.distributions():
    direct_url = distribution.read_text("direct_url.json") or ""
    record = distribution.read_text("RECORD") or ""
    sources = distribution.read_text("SOURCES.txt") or ""
    if record:
        installed_files = [
            row[0]
            for row in csv.reader(record.splitlines())
            if row and row[0]
        ]
    elif sources:
        installed_files = [line for line in sources.splitlines() if line]
    else:
        raise RuntimeError("installed distribution has no file manifest")
    if not installed_files:
        raise RuntimeError("installed distribution has an empty file manifest")
    installed_digest = hashlib.sha256()
    installed_size = 0
    for relative in sorted(installed_files):
        file_digest, file_size = installed_file_digest(distribution, relative)
        installed_digest.update(str(relative).encode())
        installed_digest.update(b"\0")
        installed_digest.update(file_digest.encode())
        installed_digest.update(b"\0")
        installed_digest.update(str(file_size).encode())
        installed_digest.update(b"\0")
        installed_size += file_size
    packages.append(
        {
            "name": str(distribution.metadata.get("Name") or "").lower(),
            "version": str(distribution.version),
            "direct_url_digest": (
                "sha256:" + hashlib.sha256(direct_url.encode()).hexdigest()
                if direct_url
                else None
            ),
            "record_digest": (
                "sha256:" + hashlib.sha256(record.encode()).hexdigest()
                if record
                else None
            ),
            "installed_files_digest": "sha256:" + installed_digest.hexdigest(),
            "installed_file_count": len(installed_files),
            "installed_bytes": installed_size,
        }
    )

site_paths = list(site.getsitepackages())
user_site = site.getusersitepackages()
site_paths.extend([user_site] if isinstance(user_site, str) else user_site)
pth = {}
for raw in set(site_paths):
    path = pathlib.Path(raw)
    if not path.is_dir():
        continue
    for item in path.glob("*.pth"):
        try:
            pth[str(item.resolve())] = (
                "sha256:" + hashlib.sha256(item.read_bytes()).hexdigest()
            )
        except OSError:
            pass

print(
    "PHYSICS_AGENT_RUNTIME_FINGERPRINT_V1="
    + json.dumps(
        {
            "executable": str(pathlib.Path(sys.executable).resolve()),
            "implementation": sys.implementation.name,
            "version": sys.version,
            "prefix": str(pathlib.Path(sys.prefix).resolve()),
            "base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
            "path": list(sys.path),
            "packages": sorted(
                packages,
                key=lambda package: (
                    package["name"],
                    package["version"],
                    package["direct_url_digest"] or "",
                    package["record_digest"] or "",
                    package["installed_files_digest"],
                ),
            ),
            "pth": dict(sorted(pth.items())),
        },
        sort_keys=True,
    )
)
"""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data for stable hashing."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_digest(value: Any) -> str:
    """Return a prefixed SHA-256 digest of canonical JSON data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _fingerprint_environment_value(name: str) -> str:
    if name not in os.environ:
        return "unset"
    return "sha256:" + hashlib.sha256(os.environ[name].encode()).hexdigest()


def _python_environment(spec: ExternalTuneSpec) -> dict[str, Any]:
    allowed = BASE_EXTERNAL_ENVIRONMENT_NAMES | set(spec.runtime.pass_env)
    environment = {name: value for name, value in os.environ.items() if name in allowed}
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            [
                str(spec.runtime.python),
                *spec.runtime.python_args,
                "-c",
                _PYTHON_ENVIRONMENT_PROBE,
            ],
            cwd=str(spec.runtime.cwd),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=spec.runtime.timeout_s,
            shell=False,
        )
        payloads = [
            line.removeprefix(_PYTHON_ENVIRONMENT_OUTPUT_PREFIX)
            for line in completed.stdout.splitlines()
            if line.startswith(_PYTHON_ENVIRONMENT_OUTPUT_PREFIX)
        ]
        if len(payloads) != 1:
            raise RuntimeError(
                "external Python runtime fingerprint probe returned invalid output"
            )
        value = json.loads(payloads[0])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RuntimeError("external Python runtime fingerprint probe failed") from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            "external Python runtime fingerprint probe returned invalid data"
        )
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    found: list[Path] = []
    for root, dirs, names in os.walk(path):
        root_path = Path(root)
        included_dirs = sorted(name for name in dirs if name not in _EXCLUDED_DIRS)
        for name in included_dirs:
            candidate = root_path / name
            if candidate.is_symlink():
                raise ValueError(
                    "runtime fingerprint paths must not contain directory symlinks: "
                    f"{candidate}"
                )
        dirs[:] = included_dirs
        found.extend(root_path / name for name in sorted(names))
    return found


def _qualification_manifest(spec: ExternalTuneSpec) -> dict[str, Any]:
    """Return only the contract covered by qualification approval.

    Search bounds and optimizer settings are execution choices, not properties
    demonstrated by the nominal qualification trial. Keep parameter identity
    and type bound to approval while allowing later optimization/refinement to
    choose different valid ranges and optimizer settings.
    """

    return {
        "schema_version": 1,
        "task": spec.task,
        "runtime": {
            "python": str(spec.runtime.python),
            "script": str(spec.runtime.script),
            "cwd": str(spec.runtime.cwd),
            "timeout_s": spec.runtime.timeout_s,
            "python_args": list(spec.runtime.python_args),
            "extra_args": list(spec.runtime.extra_args),
            "pass_env": list(spec.runtime.pass_env),
            "fingerprint_paths": [str(path) for path in spec.runtime.fingerprint_paths],
            "trial": spec.runtime.trial,
            "max_result_bytes": spec.runtime.max_result_bytes,
            "max_log_bytes": spec.runtime.max_log_bytes,
            "max_artifact_bytes": spec.runtime.max_artifact_bytes,
        },
        "parameter_catalog": [
            {
                "name": parameter.name,
                "integer": parameter.integer,
            }
            for parameter in spec.parameter_catalog
        ],
        "publish_artifacts": list(spec.publish_artifacts),
        "objective": {
            "name": spec.objective.name,
            "unit": spec.objective.unit,
            "direction": spec.objective.direction,
        },
        "qualification": {
            "nominal_params": spec.qualification.nominal_params,
            "seed": spec.qualification.seed,
            "parameter_tolerance": spec.qualification.parameter_tolerance,
        },
        "evidence": (
            {
                "artifact_name": spec.evidence.artifact_name,
                "renderer": spec.evidence.renderer,
                "media_type": spec.evidence.media_type,
                "width": spec.evidence.width,
                "height": spec.evidence.height,
                "fps": spec.evidence.fps,
                "min_frames": spec.evidence.min_frames,
                "require_motion": spec.evidence.require_motion,
                "min_frame_stddev": spec.evidence.min_frame_stddev,
                "min_motion_score": spec.evidence.min_motion_score,
                "camera_position": list(spec.evidence.camera_position),
                "camera_target": list(spec.evidence.camera_target),
                "recording_artifact_name": spec.evidence.recording_artifact_name,
                "playback_renderer": spec.evidence.playback_renderer,
                "max_duration_seconds": spec.evidence.max_duration_seconds,
                "num_sensor_updates": spec.evidence.num_sensor_updates,
                "render_mode": spec.evidence.render_mode,
            }
            if spec.evidence is not None
            else None
        ),
    }


def build_runtime_fingerprint(spec: ExternalTuneSpec) -> dict[str, Any]:
    """Hash declared runtime inputs and environment values passed to trials."""

    declared = [
        spec.runtime.python,
        spec.runtime.script,
        *spec.runtime.fingerprint_paths,
    ]
    file_hashes: dict[str, str] = {}
    seen: set[Path] = set()
    for declared_path in declared:
        for candidate in _files(declared_path):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.is_file():
                continue
            file_hashes[str(candidate.absolute())] = _hash_file(resolved)
    environment_names = BASE_EXTERNAL_ENVIRONMENT_NAMES | set(spec.runtime.pass_env)
    environment = {
        name: _fingerprint_environment_value(name) for name in sorted(environment_names)
    }
    python_environment = _python_environment(spec)
    payload = {
        "schema_version": 1,
        "spec": _qualification_manifest(spec),
        "files": dict(sorted(file_hashes.items())),
        "environment": environment,
        "python_environment": python_environment,
    }
    return {
        "digest": canonical_digest(payload),
        "contract": payload["spec"],
        "files": payload["files"],
        "environment": environment,
        "python_environment": python_environment,
    }


__all__ = ["build_runtime_fingerprint", "canonical_digest", "canonical_json"]
