# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit shadow evaluators that cannot participate in production certification."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any

import trimesh

from .artifacts import atomic_write_json, file_sha256
from .mesh_io import load_meshes


def phase2_candidate_decisions() -> dict[str, Any]:
    path = resources.files("geometry_repair").joinpath("phase2_candidate_decisions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def run_cgal_exact_audit(
    source_path: str | Path,
    output_path: str | Path,
    *,
    allow_gpl_evaluation: bool = False,
) -> dict[str, Any]:
    """Run the GPL CGAL binary as an explicit non-certifying shadow evaluator."""

    if not allow_gpl_evaluation:
        raise ValueError("CGAL exact audit requires allow_gpl_evaluation=True")
    configured_executable = os.environ.get("GEOMETRY_REPAIR_CGAL_AUDIT_EXECUTABLE") or shutil.which(
        "geometry_repair_cgal_exact_audit"
    )
    if not configured_executable:
        raise RuntimeError("CGAL exact audit executable is unavailable")
    try:
        executable = Path(configured_executable).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("CGAL exact audit executable is unavailable") from exc
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError("CGAL exact audit executable is unavailable or not executable")
    # The executable is trusted operator configuration, resolved before use;
    # arguments are fixed or generated paths and shell execution is disabled.
    version = subprocess.run(  # nosemgrep
        [executable, "--version"],  # nosemgrep
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout.strip()
    if not version.startswith("geometry-repair-cgal-exact-audit "):
        raise RuntimeError(f"Unexpected CGAL evaluator version: {version!r}")
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    work_dir = output.parent / f"{output.stem}_cgal_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    meshes, _ = load_meshes(source, include_guide_purpose=True)
    records: list[dict[str, Any]] = []
    for index, mesh in enumerate(meshes):
        if not len(mesh.triangles):
            continue
        obj_path = work_dir / f"mesh_{index:04d}.obj"
        trimesh.Trimesh(
            vertices=mesh.world_vertices_m,
            faces=mesh.triangles,
            process=False,
        ).export(obj_path, file_type="obj")
        completed = subprocess.run(  # nosemgrep
            [executable, str(obj_path)],  # nosemgrep
            check=False,
            capture_output=True,
            text=True,
            timeout=300.0,
        )
        record: dict[str, Any] = {
            "mesh_path": mesh.path,
            "input_path": str(obj_path),
            "input_sha256": file_sha256(obj_path),
            "return_code": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
        if completed.returncode == 0:
            record.update(json.loads(completed.stdout))
        records.append(record)
    payload = {
        "schema_version": "geometry-repair.cgal-shadow-audit.v1",
        "claim_scope": "evaluation_only_non_certifying",
        "license": "GPL-3.0-or-later",
        "distribution_status": "not_distributed",
        "evaluator_version": version,
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "status": "pass"
        if records and all(item["return_code"] == 0 for item in records)
        else "fail",
        "meshes": records,
    }
    atomic_write_json(output, payload)
    return payload
