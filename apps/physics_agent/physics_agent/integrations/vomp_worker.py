# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated Python 3.10 worker for the official VoMP inference library.

This file intentionally avoids importing Physics Agent. It is launched with the
Python interpreter from a pinned VoMP checkout, whose dependencies conflict with
the parent Physics Agent environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PROTOCOL_VERSION = 1
_CHECKPOINT_KEYS = (
    "geometry_checkpoint_dir",
    "matvae_checkpoint_dir",
    "normalization_params_path",
)
_ALLOWED_UNTRACKED_PREFIXES = (".venv/", "outputs/", "venv/")
_IGNORED_SCAN_PATHSPEC = (
    ":(exclude).venv",
    ":(exclude).venv/**",
    ":(exclude)outputs",
    ":(exclude)outputs/**",
    ":(exclude)venv",
    ":(exclude)venv/**",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=str(path.parent),
        delete=False,
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _resolve_checkpoint_paths(
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    # VoMP resolves these paths from its repository working directory.
    root = Path.cwd().resolve()
    result: dict[str, dict[str, Any]] = {}
    for key in _CHECKPOINT_KEYS:
        configured = config.get(key)
        if not isinstance(configured, str) or not configured:
            raise RuntimeError(f"VoMP config is missing required path: {key}")
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise RuntimeError(f"VoMP runtime artifact is missing: {key}")
        result[key] = {
            "path": str(candidate),
            "sha256": _sha256(candidate),
            "sizeBytes": candidate.stat().st_size,
        }
    result["config"] = {
        "path": str(config_path),
        "sha256": _sha256(config_path),
        "sizeBytes": config_path.stat().st_size,
    }
    return result


def _deduplicate_results(results: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    coordinates = np.asarray(results["voxel_coords_world"])
    _, first_indices, inverse, counts = np.unique(
        coordinates,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    duplicate_groups = np.flatnonzero(counts > 1)
    properties = ("youngs_modulus", "poisson_ratio", "density")
    for group in duplicate_groups:
        rows = inverse == group
        for name in properties:
            values = np.asarray(results[name])[rows]
            if not np.all(values == values[0]):
                raise RuntimeError(
                    f"VoMP produced conflicting {name} values for one voxel cell"
                )

    ordered = np.sort(first_indices)
    input_count = int(coordinates.shape[0])
    for name, value in tuple(results.items()):
        values = np.asarray(value)
        if values.ndim > 0 and values.shape[0] == input_count:
            results[name] = values[ordered]
    results["num_voxels"] = int(len(ordered))
    return {
        "inputRows": input_count,
        "outputRows": int(len(ordered)),
        "removedRows": input_count - int(len(ordered)),
        "duplicateGroups": int(len(duplicate_groups)),
        "maximumMultiplicity": int(counts.max()) if len(counts) else 0,
        "duplicatePropertiesIdentical": True,
    }


def _unsafe_untracked_runtime_paths(paths: Iterable[str]) -> list[str]:
    unsafe: list[str] = []
    for value in paths:
        normalized = value.replace("\\", "/")
        if normalized.startswith(_ALLOWED_UNTRACKED_PREFIXES):
            continue
        if normalized.startswith("vomp/") or Path(normalized).suffix.lower() in {
            ".py",
            ".pyc",
            ".pyo",
            ".so",
        }:
            unsafe.append(value)
    return unsafe


def _runtime_untracked_paths(root: Path) -> list[str]:
    runtime_paths: list[str] = []
    for ignored in (False, True):
        arguments = [
            "git",
            "-C",
            str(root),
            "-c",
            "core.quotePath=false",
            "ls-files",
            "-z",
            "--others",
        ]
        if ignored:
            arguments.append("--ignored")
        arguments.extend(["--exclude-standard", "--", ".", *_IGNORED_SCAN_PATHSPEC])
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        runtime_paths.extend(entry for entry in completed.stdout.split("\0") if entry)
    return runtime_paths


def _validate_watertight_mesh(mesh_path: Path) -> None:
    import trimesh

    mesh = trimesh.load(mesh_path)
    if not hasattr(mesh, "is_watertight") or not hasattr(mesh, "is_winding_consistent"):
        raise RuntimeError("VoMP input did not load as a polygon mesh")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(
            "VoMP mass integration requires a watertight manifold mesh with "
            "consistent winding"
        )


def _run(
    request: dict[str, Any],
) -> dict[str, Any]:  # pragma: no cover - exercised in pinned VoMP environment
    if int(request.get("protocolVersion", -1)) != _PROTOCOL_VERSION:
        raise RuntimeError("unsupported VoMP worker protocol version")
    root = Path(request["runtimeRoot"]).resolve()
    worker_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != worker_directory
    ]
    sys.path.insert(0, str(root))
    attention_backend = request.get("attentionBackend")
    if attention_backend not in {"xformers", "sdpa", "naive"}:
        raise RuntimeError("unsupported VoMP attention backend")
    os.environ["ATTN_BACKEND"] = attention_backend

    if Path.cwd().resolve() != root:
        raise RuntimeError("VoMP worker must run from the attested runtime root")
    revision = _git_revision(root)
    if revision != request["expectedRevision"]:
        raise RuntimeError("VoMP checkout revision does not match the requested pin")
    tracked_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if tracked_diff.returncode != 0:
        raise RuntimeError("VoMP checkout gained tracked modifications")
    unsafe_untracked = _unsafe_untracked_runtime_paths(_runtime_untracked_paths(root))
    if unsafe_untracked:
        raise RuntimeError("VoMP checkout gained untracked runtime code")

    mesh_path = Path(request["meshPath"]).resolve()
    metadata_path = Path(request["metadataPath"]).resolve()
    output_dir = Path(request["outputDir"]).resolve()
    output_npz = Path(request["outputNpzPath"]).resolve()
    config_path = Path(request["configPath"]).resolve()
    for label, path in (
        ("mesh", mesh_path),
        ("render metadata", metadata_path),
        ("VoMP config", config_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} input is missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("r", encoding="utf-8") as stream:
        frames = json.load(stream)
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("render metadata must contain a non-empty frame list")
    if len(frames) != int(request["numViews"]):
        raise RuntimeError("render metadata view count does not match the request")

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    artifacts = _resolve_checkpoint_paths(config_path, config)
    expected_artifacts = request.get("expectedArtifactSha256")
    if not isinstance(expected_artifacts, dict) or set(expected_artifacts) != set(
        artifacts
    ):
        raise RuntimeError("VoMP artifact attestation request is incomplete")
    for key, details in artifacts.items():
        if details["sha256"] != expected_artifacts[key]:
            raise RuntimeError(f"VoMP runtime artifact failed attestation: {key}")
    _validate_watertight_mesh(mesh_path)

    import numpy as np
    import torch
    import vomp
    from vomp.inference import Vomp
    from vomp.inference.utils import (
        denormalize_coords,
        get_mesh_transform_params,
        save_materials,
    )

    seed = int(request["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = Vomp.from_checkpoint(config_path=str(config_path), use_trt=False)
    voxelize = getattr(model, "_voxelize_mesh", None)
    if not callable(voxelize):
        raise RuntimeError(
            "pinned VoMP revision no longer exposes the required voxelization hook"
        )
    voxel_size_normalized = float(request["voxelSizeNormalized"])
    voxel_centers = voxelize(
        str(mesh_path),
        str(output_dir),
        voxel_size=voxel_size_normalized,
        max_voxels=None,
    )
    voxel_count = int(len(voxel_centers))
    maximum = int(request["maxCompleteVoxels"])
    if voxel_count <= 0:
        raise RuntimeError("VoMP voxelization produced no occupied cells")
    if voxel_count > maximum:
        raise RuntimeError(
            f"complete VoMP field has {voxel_count} cells, above configured limit "
            f"{maximum}; refusing to subsample"
        )

    coords, features = model.get_features(
        renders_metadata=frames,
        voxel_centers=voxel_centers,
        output_dir=str(output_dir),
        image_size=int(request["featureImageSize"]),
        batch_size=int(request["featureBatchSize"]),
        save_features=bool(request["saveFeatures"]),
    )
    if int(coords.shape[0]) != voxel_count:
        raise RuntimeError("VoMP feature extraction changed the complete voxel count")
    results = model.predict_materials(
        coords,
        features,
        max_voxels=voxel_count,
        sample_posterior=False,
    )
    if int(results["num_voxels"]) != voxel_count:
        raise RuntimeError("VoMP inference did not preserve the complete voxel field")

    center, scale = get_mesh_transform_params(str(mesh_path))
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("VoMP mesh normalization produced an invalid scale")
    results["voxel_coords_world"] = denormalize_coords(
        np.asarray(results["voxel_coords_world"]),
        np.asarray(center),
        scale,
    )
    deduplication = _deduplicate_results(results)
    results["transform_center"] = np.asarray(center)
    results["transform_scale"] = scale

    temporary_npz = output_npz.with_name(f".{output_npz.stem}.tmp.npz")
    temporary_npz.unlink(missing_ok=True)
    try:
        save_materials(results, str(temporary_npz))
        os.replace(str(temporary_npz), str(output_npz))
    finally:
        temporary_npz.unlink(missing_ok=True)

    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "status": "success",
        "outputNpzPath": str(output_npz),
        "outputNpzSha256": _sha256(output_npz),
        "sampleCount": int(results["num_voxels"]),
        "completeVoxelField": True,
        "voxelSizeNormalized": voxel_size_normalized,
        "voxelSizeM": scale * voxel_size_normalized,
        "coordinateUnitMeters": 1.0,
        "coordinateOffsetM": [0.0, 0.0, 0.0],
        "meshTransform": {
            "centerM": np.asarray(center, dtype=float).tolist(),
            "scaleM": scale,
        },
        "deduplication": deduplication,
        "runtime": {
            "vompRevision": revision,
            "vompModulePath": str(Path(vomp.__file__).resolve()),
            "pythonVersion": sys.version.split()[0],
            "torchVersion": str(torch.__version__),
            "cudaAvailable": bool(torch.cuda.is_available()),
            "cudaVersion": str(torch.version.cuda) if torch.version.cuda else None,
            "attentionBackend": attention_backend,
            "artifacts": artifacts,
        },
    }


def main() -> int:  # pragma: no cover - exercised in the pinned VoMP environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    response_path = Path(args.response).resolve()

    try:
        with request_path.open("r", encoding="utf-8") as stream:
            request = json.load(stream)
        response = _run(request)
        _write_json_atomic(response_path, response)
    except Exception as error:
        response = {
            "protocolVersion": _PROTOCOL_VERSION,
            "status": "error",
            "errorType": type(error).__name__,
            "error": str(error),
        }
        try:
            _write_json_atomic(response_path, response)
        finally:
            traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
