# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated runtime boundary for the official VoMP inference library."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from world_understanding.utils.credentials import redact_sensitive_path

from physics_agent.integrations.vomp import VompIntegrationError
from physics_agent.integrations.vomp_defaults import (
    DEFAULT_VOMP_ARTIFACT_SHA256,
    DEFAULT_VOMP_REVISION,
)

VOMP_PROTOCOL_VERSION = 1
_RUNTIME_CODE_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".so"})
_ALLOWED_UNTRACKED_PREFIXES = (".venv/", "outputs/", "venv/")
_IGNORED_SCAN_PATHSPEC = (
    ":(exclude).venv",
    ":(exclude).venv/**",
    ":(exclude)outputs",
    ":(exclude)outputs/**",
    ":(exclude)venv",
    ":(exclude)venv/**",
)
_LOWERCASE_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class VompRuntimeConfig:
    """Pinned external VoMP checkout and interpreter."""

    runtime_root: Path
    python_executable: Path
    config_path: Path
    expected_revision: str = DEFAULT_VOMP_REVISION
    expected_artifact_sha256: Mapping[str, str] | None = None
    attention_backend: str = "xformers"
    timeout_seconds: float = 3600.0
    max_complete_voxels: int = 262_144


@dataclass(frozen=True)
class VompRunRequest:
    """Prepared geometry and OVRTX evidence consumed by VoMP."""

    mesh_path: Path
    metadata_path: Path
    output_dir: Path
    output_npz_path: Path
    num_views: int
    seed: int
    voxel_size_normalized: float = 1.0 / 64.0
    feature_image_size: int = 518
    feature_batch_size: int = 16
    save_features: bool = False


@dataclass(frozen=True)
class VompRunResult:
    """Validated outputs returned by the isolated VoMP worker."""

    output_npz_path: Path
    sample_count: int
    voxel_size_m: float
    coordinate_unit_meters: float
    coordinate_offset_m: tuple[float, float, float]
    manifest: dict[str, Any]
    worker_log_path: Path


class VompRunner(Protocol):
    """Minimal library contract used by Physics Agent orchestration."""

    def run(self, request: VompRunRequest) -> VompRunResult:
        """Run official VoMP inference for prepared render evidence."""


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"VoMP worker response contains a non-finite value: {name}")


def _run_worker_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_stream: TextIO,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VompIntegrationError(
            "Unable to attest the configured VoMP checkout revision"
        ) from exc
    return completed.stdout.strip()


def _run_git(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VompIntegrationError(
            "Unable to attest the configured VoMP checkout"
        ) from exc


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attest_artifacts(
    root: Path,
    config_path: Path,
    expected: Mapping[str, str],
) -> None:
    required = set(DEFAULT_VOMP_ARTIFACT_SHA256)
    if set(expected) != required or any(
        not isinstance(value, str) or len(value) != 64 for value in expected.values()
    ):
        raise VompIntegrationError(
            "VoMP expected_artifact_sha256 must pin every required artifact"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VompIntegrationError("Unable to read the configured VoMP config") from exc
    paths = {"config": config_path}
    for key in required - {"config"}:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise VompIntegrationError(f"VoMP config does not define {key}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        paths[key] = path.resolve()
    for key, path in paths.items():
        if not path.is_file() or _sha256(path) != expected[key]:
            raise VompIntegrationError(
                f"Configured VoMP artifact failed attestation: {key}"
            )


def _unsafe_untracked_runtime_paths(paths: list[str]) -> list[str]:
    unsafe: list[str] = []
    for value in paths:
        normalized = value.replace("\\", "/")
        if normalized.startswith(_ALLOWED_UNTRACKED_PREFIXES):
            continue
        if normalized.startswith("vomp/") or Path(normalized).suffix.lower() in (
            _RUNTIME_CODE_SUFFIXES
        ):
            unsafe.append(value)
    return unsafe


def _runtime_untracked_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for ignored in (False, True):
        arguments = [
            "-c",
            "core.quotePath=false",
            "ls-files",
            "-z",
            "--others",
        ]
        if ignored:
            arguments.append("--ignored")
        arguments.extend(["--exclude-standard", "--", ".", *_IGNORED_SCAN_PATHSPEC])
        completed = _run_git(root, arguments)
        if completed.returncode != 0:
            raise VompIntegrationError("Unable to inspect untracked VoMP runtime files")
        paths.extend(entry for entry in completed.stdout.split("\0") if entry)
    return paths


def _validate_runtime(config: VompRuntimeConfig) -> tuple[Path, Path, Path]:
    root = config.runtime_root.expanduser().resolve()
    python = config.python_executable.expanduser()
    if not python.is_absolute():
        python = root / python
    python = Path(os.path.abspath(python))
    config_path = config.config_path.expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    if not root.is_dir():
        raise VompIntegrationError("Configured VoMP runtime root is not a directory")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise VompIntegrationError("Configured VoMP Python is not executable")
    if not config_path.is_file():
        raise VompIntegrationError("Configured VoMP inference config is missing")
    if (
        not config.expected_revision
        or len(config.expected_revision) != 40
        or any(
            character not in _LOWERCASE_HEX for character in config.expected_revision
        )
    ):
        raise VompIntegrationError(
            "VoMP expected_revision must be a full lowercase commit SHA"
        )
    if _git_revision(root) != config.expected_revision:
        raise VompIntegrationError(
            "Configured VoMP checkout does not match expected_revision"
        )
    tracked_diff = _run_git(root, ["diff", "--quiet", "HEAD", "--"])
    if tracked_diff.returncode != 0:
        raise VompIntegrationError("Configured VoMP checkout has tracked modifications")
    unsafe_untracked = _unsafe_untracked_runtime_paths(_runtime_untracked_paths(root))
    if unsafe_untracked:
        raise VompIntegrationError(
            "Configured VoMP checkout has untracked runtime code"
        )
    expected_artifacts = (
        config.expected_artifact_sha256
        if config.expected_artifact_sha256 is not None
        else DEFAULT_VOMP_ARTIFACT_SHA256
    )
    _attest_artifacts(root, config_path, expected_artifacts)
    if not config.timeout_seconds > 0.0:
        raise VompIntegrationError("VoMP timeout_seconds must be positive")
    if config.max_complete_voxels <= 0:
        raise VompIntegrationError("VoMP max_complete_voxels must be positive")
    if config.attention_backend not in {"xformers", "sdpa", "naive"}:
        raise VompIntegrationError("VoMP attention_backend is unsupported")
    return root, python, config_path


class ExternalVompRunner:
    """Run official VoMP with its own interpreter and dependency environment."""

    def __init__(self, config: VompRuntimeConfig) -> None:
        self.config = config

    def run(self, request: VompRunRequest) -> VompRunResult:
        root, python, config_path = _validate_runtime(self.config)
        mesh_path = request.mesh_path.expanduser().resolve()
        metadata_path = request.metadata_path.expanduser().resolve()
        output_dir = request.output_dir.expanduser().resolve()
        output_npz = request.output_npz_path.expanduser().resolve()
        if not mesh_path.is_file():
            raise VompIntegrationError("Prepared VoMP mesh is missing")
        if not metadata_path.is_file():
            raise VompIntegrationError("Prepared VoMP render metadata is missing")
        if request.num_views <= 0:
            raise VompIntegrationError("VoMP num_views must be positive")
        if not 0.0 < request.voxel_size_normalized <= 1.0:
            raise VompIntegrationError(
                "VoMP voxel_size_normalized must be in the interval (0, 1]"
            )
        if request.feature_image_size <= 0 or request.feature_batch_size <= 0:
            raise VompIntegrationError("VoMP feature sizes must be positive")

        output_dir.mkdir(parents=True, exist_ok=True)
        response_path = output_dir / "vomp_worker_response.json"
        request_path = output_dir / "vomp_worker_request.json"
        log_path = output_dir / "vomp_worker.log"
        worker_path = Path(__file__).with_name("vomp_worker.py").resolve()
        payload: dict[str, Any] = {
            "protocolVersion": VOMP_PROTOCOL_VERSION,
            "runtimeRoot": str(root),
            "expectedRevision": self.config.expected_revision,
            "configPath": str(config_path),
            "meshPath": str(mesh_path),
            "metadataPath": str(metadata_path),
            "outputDir": str(output_dir),
            "outputNpzPath": str(output_npz),
            "numViews": request.num_views,
            "seed": request.seed,
            "voxelSizeNormalized": request.voxel_size_normalized,
            "featureImageSize": request.feature_image_size,
            "featureBatchSize": request.feature_batch_size,
            "saveFeatures": request.save_features,
            "maxCompleteVoxels": self.config.max_complete_voxels,
            "expectedArtifactSha256": dict(
                self.config.expected_artifact_sha256
                if self.config.expected_artifact_sha256 is not None
                else DEFAULT_VOMP_ARTIFACT_SHA256
            ),
            "attentionBackend": self.config.attention_backend,
        }
        _write_json_atomic(request_path, payload)
        response_path.unlink(missing_ok=True)

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        command = [
            str(python),
            "-I",
            str(worker_path),
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log_stream:
                completed = _run_worker_process(
                    command,
                    cwd=root,
                    environment=environment,
                    log_stream=log_stream,
                    timeout_seconds=self.config.timeout_seconds,
                )
        except subprocess.TimeoutExpired as exc:
            raise VompIntegrationError(
                "VoMP inference exceeded its configured timeout; worker log: "
                f"{redact_sensitive_path(log_path)}"
            ) from exc
        except OSError as exc:
            raise VompIntegrationError(
                "Unable to start the configured VoMP runtime"
            ) from exc

        try:
            response = json.loads(
                response_path.read_text(encoding="utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise VompIntegrationError(
                "VoMP worker did not return a valid response; worker log: "
                f"{redact_sensitive_path(log_path)}"
            ) from exc
        if not isinstance(response, dict):
            raise VompIntegrationError("VoMP worker returned an invalid response")
        if completed.returncode != 0 or response.get("status") != "success":
            error_type = str(response.get("errorType") or "VoMPWorkerError")
            raise VompIntegrationError(
                f"VoMP worker failed ({error_type}); worker log: "
                f"{redact_sensitive_path(log_path)}"
            )
        if response.get("protocolVersion") != VOMP_PROTOCOL_VERSION:
            raise VompIntegrationError("VoMP worker returned an incompatible protocol")
        if response.get("completeVoxelField") is not True:
            raise VompIntegrationError(
                "VoMP worker did not return a complete voxel field"
            )

        returned_npz_value = response.get("outputNpzPath")
        if not isinstance(returned_npz_value, str) or not returned_npz_value:
            raise VompIntegrationError("VoMP worker returned an invalid NPZ path")
        returned_npz = Path(returned_npz_value).resolve()
        if returned_npz != output_npz or not output_npz.is_file():
            raise VompIntegrationError("VoMP worker returned an unexpected NPZ path")
        if response.get("outputNpzSha256") != _sha256(output_npz):
            raise VompIntegrationError("VoMP worker NPZ failed digest verification")
        offset = response.get("coordinateOffsetM")
        if not isinstance(offset, list) or len(offset) != 3:
            raise VompIntegrationError(
                "VoMP worker returned an invalid coordinate offset"
            )
        try:
            sample_count_value = response["sampleCount"]
            voxel_size_m = float(response["voxelSizeM"])
            coordinate_unit_meters = float(response["coordinateUnitMeters"])
            coordinate_offset_m = tuple(float(value) for value in offset)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise VompIntegrationError(
                "VoMP worker returned invalid numeric metadata"
            ) from exc
        if not isinstance(sample_count_value, int) or isinstance(
            sample_count_value, bool
        ):
            raise VompIntegrationError("VoMP worker returned an invalid sample count")
        sample_count = sample_count_value
        if sample_count <= 0 or sample_count > self.config.max_complete_voxels:
            raise VompIntegrationError("VoMP worker returned an invalid sample count")
        if not math.isfinite(voxel_size_m) or voxel_size_m <= 0.0:
            raise VompIntegrationError("VoMP worker returned an invalid voxel size")
        if not math.isfinite(coordinate_unit_meters) or coordinate_unit_meters <= 0.0:
            raise VompIntegrationError(
                "VoMP worker returned an invalid coordinate unit"
            )
        if not all(math.isfinite(value) for value in coordinate_offset_m):
            raise VompIntegrationError(
                "VoMP worker returned an invalid coordinate offset"
            )
        return VompRunResult(
            output_npz_path=output_npz,
            sample_count=sample_count,
            voxel_size_m=voxel_size_m,
            coordinate_unit_meters=coordinate_unit_meters,
            coordinate_offset_m=coordinate_offset_m,
            manifest=response,
            worker_log_path=log_path,
        )


__all__ = [
    "DEFAULT_VOMP_ARTIFACT_SHA256",
    "DEFAULT_VOMP_REVISION",
    "ExternalVompRunner",
    "VompRunRequest",
    "VompRunResult",
    "VompRunner",
    "VompRuntimeConfig",
]
