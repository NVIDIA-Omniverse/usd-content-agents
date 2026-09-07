# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.geometry import scene_ops
from content_agent_workflows.geometry import segmentation as segmentation_module
from content_agent_workflows.geometry.segmentation import (
    SEGMENTATION_EXPORT_SCHEMA_VERSION,
    SEGMENTATION_RENDER_SCHEMA_VERSION,
    SEGMENTATION_TERMINAL_SCHEMA_VERSION,
    consume_segmentation_handoff,
    segmentation_validation_check,
)
from content_agent_workflows.geometry.segmentation_routing import route_segmentation
from content_agent_workflows.geometry.workflow import (
    GeometryWorkflowInput,
    _semantic_parts_for_handoff,
    run_geometry_workflow,
)
from content_agent_workflows.mesh_segmentation_contract import (
    REQUIRED_FINAL_ARTIFACTS as _REQUIRED_FINAL_ARTIFACTS,
)
from content_agent_workflows.mesh_segmentation_contract import (
    REQUIRED_RECOGNITION_FINAL_ARTIFACTS as _REQUIRED_RECOGNITION_FINAL_ARTIFACTS,
)
from content_agent_workflows.mesh_segmentation_contract import (
    REQUIRED_TARGETED_FINAL_ARTIFACTS as _REQUIRED_TARGETED_FINAL_ARTIFACTS,
)

_TETRAHEDRON_FACE_VERTEX_INDICES = [
    0,
    2,
    1,
    0,
    1,
    3,
    1,
    2,
    3,
    2,
    0,
    3,
]
_SEMANTIC_PRIM_PATHS = {
    "body": "/Asset/Parts/body",
    "handle": "/Asset/Parts/handle",
}
_CANONICAL_RECOGNITION_ARTIFACTS = (
    *_REQUIRED_FINAL_ARTIFACTS,
    *_REQUIRED_RECOGNITION_FINAL_ARTIFACTS,
)


@dataclass(frozen=True)
class _SegmentationRun:
    source_usd: Path
    run_dir: Path
    segmented_usd: Path
    source_sha256: str
    segmented_sha256: str
    topology_digest: str


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _refresh_segmented_usd_bindings(run: _SegmentationRun) -> None:
    digest = file_sha256(run.segmented_usd)
    export_path = run.run_dir / "final/export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["output_usd_sha256"] = digest
    _write_json(export_path, export)
    render_path = run.run_dir / "final/renders/render_manifest.json"
    render = json.loads(render_path.read_text(encoding="utf-8"))
    render["scene_sha256"] = digest
    _write_json(render_path, render)


def _refresh_usd_cli_receipt_bindings(run: _SegmentationRun) -> None:
    manifest_path = run.run_dir / "final/renders/render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = Path(manifest["usd_cli_command_receipts"])
    checkpoint_path = Path(manifest["usd_cli_receipt_checkpoint"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    receipt_stat = receipt_path.stat()
    checkpoint.update(
        {
            "receipt_device": receipt_stat.st_dev,
            "receipt_inode": receipt_stat.st_ino,
            "receipt_sha256": file_sha256(receipt_path),
            "receipt_size_bytes": receipt_stat.st_size,
        }
    )
    _write_json(checkpoint_path, checkpoint)
    manifest["usd_cli_command_receipts_sha256"] = file_sha256(receipt_path)
    manifest["usd_cli_receipt_checkpoint_sha256"] = file_sha256(checkpoint_path)
    _write_json(manifest_path, manifest)


def _replace_receipted_render_response(
    run: _SegmentationRun,
    *,
    view_index: int,
    replacement: dict[str, Any],
) -> None:
    manifest_path = run.run_dir / "final/renders/render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["renders"][view_index]
    response_path = Path(record["response"])
    original = json.loads(response_path.read_text(encoding="utf-8"))
    _write_json(response_path, replacement)
    record["response_sha256"] = file_sha256(response_path)
    _write_json(manifest_path, manifest)

    receipt_path = Path(manifest["usd_cli_command_receipts"])
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    matching = [receipt for receipt in receipts if receipt.get("response") == original]
    assert len(matching) == 1
    matching[0]["response"] = replacement
    receipt_path.write_text(
        "".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts),
        encoding="utf-8",
    )
    _refresh_usd_cli_receipt_bindings(run)


def _refresh_segments_binding(run: _SegmentationRun) -> None:
    segments_path = run.run_dir / "segments.json"
    export_path = run.run_dir / "final/export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["segments_sha256"] = file_sha256(segments_path)
    _write_json(export_path, export)


def _refresh_source_bindings(run: _SegmentationRun) -> None:
    staged_source = run.run_dir / "inputs/source/source.usda"
    shutil.copy2(run.source_usd, staged_source)
    digest = file_sha256(run.source_usd)

    request_path = run.run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["input_staging"]["asset"]["sha256"] = digest
    request["input_staging"]["asset"]["size_bytes"] = staged_source.stat().st_size
    _write_json(request_path, request)

    for relative in ("prepare/topology.json", "final/export_manifest.json"):
        path = run.run_dir / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_sha256"] = digest
        _write_json(path, payload)

    segments_path = run.run_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    segments["source"]["sha256"] = digest
    _write_json(segments_path, segments)
    _refresh_segments_binding(run)


def _u32le(values: list[int]) -> bytes:
    return b"".join(
        value.to_bytes(4, byteorder="little", signed=False) for value in values
    )


def _assert_pending_ontology_warning(warnings: list[str]) -> None:
    pending = [
        warning
        for warning in warnings
        if "DG-03" in warning and "ontology" in warning.lower()
    ]
    assert len(pending) == 1


def _source_topology_digest() -> str:
    digest = hashlib.sha256()
    for point in [*_tetrahedron_points(0.0), *_tetrahedron_points(2.0)]:
        digest.update(struct.pack("<fff", *(float(value) for value in point)))
    indices = [
        *_TETRAHEDRON_FACE_VERTEX_INDICES,
        *(index + 4 for index in _TETRAHEDRON_FACE_VERTEX_INDICES),
    ]
    for index in indices:
        digest.update(struct.pack("<i", index))
    return f"sha256:{digest.hexdigest()}"


def _tetrahedron_points(x_offset: float) -> Vt.Vec3fArray:
    return Vt.Vec3fArray(
        [
            Gf.Vec3f(x_offset, 0.0, 0.0),
            Gf.Vec3f(x_offset + 1.0, 0.0, 0.0),
            Gf.Vec3f(x_offset, 1.0, 0.0),
            Gf.Vec3f(x_offset, 0.0, 1.0),
        ]
    )


def _define_tetrahedron(
    stage: Usd.Stage,
    prim_path: str,
    *,
    x_offset: float,
    source_face_ids: list[int] | None = None,
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(_tetrahedron_points(x_offset))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3, 3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(_TETRAHEDRON_FACE_VERTEX_INDICES))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if source_face_ids is not None:
        mesh.GetPrim().CreateAttribute(
            "meshSegmentation:sourceFaceIds",
            Sdf.ValueTypeNames.UIntArray,
        ).Set(Vt.UIntArray(source_face_ids))
    return mesh


def _write_source_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    assert stage is not None
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    mesh = UsdGeom.Mesh.Define(stage, "/Asset/SourceMesh")
    points = Vt.Vec3fArray([*_tetrahedron_points(0.0), *_tetrahedron_points(2.0)])
    second_tetrahedron_indices = [
        index + 4 for index in _TETRAHEDRON_FACE_VERTEX_INDICES
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * 8))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray([*_TETRAHEDRON_FACE_VERTEX_INDICES, *second_tetrahedron_indices])
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path


def _write_segmented_usd(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    assert stage is not None
    asset = UsdGeom.Xform.Define(stage, "/Asset")
    UsdGeom.Xform.Define(stage, "/Asset/Parts")
    stage.SetDefaultPrim(asset.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    _define_tetrahedron(
        stage,
        _SEMANTIC_PRIM_PATHS["body"],
        x_offset=0.0,
        source_face_ids=[0, 1, 2, 3],
    )
    _define_tetrahedron(
        stage,
        _SEMANTIC_PRIM_PATHS["handle"],
        x_offset=2.0,
        source_face_ids=[4, 5, 6, 7],
    )
    stage.GetRootLayer().Save()
    return path


@pytest.fixture
def segmentation_run(tmp_path: Path) -> _SegmentationRun:
    source_usd = _write_source_usd(tmp_path / "source.usda")
    run_dir = tmp_path / "mesh-segmentation-run"
    segmented_usd = _write_segmented_usd(run_dir / "final/segmented.usdc")
    source_sha256 = file_sha256(source_usd)
    segmented_sha256 = file_sha256(segmented_usd)
    topology_digest = _source_topology_digest()

    staged_source = run_dir / "inputs/source/source.usda"
    staged_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_usd, staged_source)

    fragment_labels_path = run_dir / "fragments/fragment_ids.u32le"
    fragment_labels_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_labels_path.write_bytes(_u32le([0, 0, 1, 1, 2, 2, 3, 3]))
    face_labels_path = run_dir / "state/final_labels.u32le"
    face_labels_path.parent.mkdir(parents=True, exist_ok=True)
    face_labels_path.write_bytes(_u32le([0, 0, 0, 0, 1, 1, 1, 1]))

    _write_json(
        run_dir / "prepare/topology.json",
        {
            "schema_version": "mesh-segmentation-topology.v1",
            "source_asset": str(staged_source),
            "source_sha256": source_sha256,
            "target_prim_path": "/Asset/SourceMesh",
            "source_face_count": 8,
            "topology_digest": topology_digest,
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "topology_component_count": 2,
            "degenerate_face_count": 0,
            "degenerate_face_ids": [],
        },
    )

    segments_path = _write_json(
        run_dir / "segments.json",
        {
            "schema_version": "mesh-segmentation-segments.v1",
            "source": {
                "path": str(staged_source),
                "sha256": source_sha256,
                "face_count": 8,
            },
            "segments": [
                {
                    "segment_id": 0,
                    "name": "body",
                    "confidence": 0.99,
                    "source_face_ids": [0, 1, 2, 3],
                },
                {
                    "segment_id": 1,
                    "name": "handle",
                    "confidence": 0.97,
                    "source_face_ids": [4, 5, 6, 7],
                },
            ],
        },
    )

    required_artifacts = list(_CANONICAL_RECOGNITION_ARTIFACTS)
    _write_json(
        run_dir / "request.json",
        {
            "schema_version": "content-agents.mesh-segmentation-request.v3",
            "workflow": "mesh-segmentation.run",
            "workflow_mode": "recognition",
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "required_skills": [
                "content-workflow-mesh-segmentation",
                "image-generation",
                "usd-cli",
            ],
            "inputs": {
                "asset": str(staged_source),
                "target_prim": "/Asset/SourceMesh",
                "target_semantic_parts": [],
                "reference_images": [],
                "continuation_seed": None,
            },
            "input_staging": {
                "mode": "single_file_copy",
                "asset": {
                    "source_name": source_usd.name,
                    "staged_path": str(staged_source),
                    "sha256": source_sha256,
                    "size_bytes": staged_source.stat().st_size,
                },
                "reference_images": [],
            },
            "isolation": {
                "fresh_child_thread": True,
                "conversation_context_inherited": False,
                "prior_run_access_allowed": False,
                "working_directory": str(run_dir),
                "permitted_evidence_root": str(run_dir),
            },
            "constraints": {
                "source_asset_edits_allowed": False,
                "semantic_decision_unit": "immutable_fragment",
                "require_fragment_atomicity": True,
                "require_exact_face_provenance": True,
                "part_recognition_required": True,
            },
            "required_final_artifacts": required_artifacts,
        },
    )
    _write_json(
        run_dir / "final/export_manifest.json",
        {
            "schema_version": SEGMENTATION_EXPORT_SCHEMA_VERSION,
            "status": "passed",
            "semantic_decision_unit": "immutable_fragment",
            "exact_source_face_coverage": True,
            "source_asset": str(staged_source),
            "source_sha256": source_sha256,
            "source_face_count": 8,
            "target_prim_path": "/Asset/SourceMesh",
            "topology_digest": topology_digest,
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "segments": str(segments_path),
            "segments_sha256": file_sha256(segments_path),
            "fragment_labels": str(fragment_labels_path),
            "fragment_labels_sha256": file_sha256(fragment_labels_path),
            "face_labels": str(face_labels_path),
            "face_labels_sha256": file_sha256(face_labels_path),
            "fragment_count": 4,
            "fragment_atomicity_conflict_ids": [],
            "output_usd": "final/segmented.usdc",
            "output_usd_sha256": segmented_sha256,
            "output_segments": [
                {
                    "segment_id": 0,
                    "output_prim_path": _SEMANTIC_PRIM_PATHS["body"],
                    "source_face_count": 4,
                    "output_point_count": 4,
                },
                {
                    "segment_id": 1,
                    "output_prim_path": _SEMANTIC_PRIM_PATHS["handle"],
                    "source_face_count": 4,
                    "output_point_count": 4,
                },
            ],
            "limitations": [
                "Fixture export preserves exact source-face geometry and provenance."
            ],
        },
    )
    _write_json(
        run_dir / "final/segment_manifest.json",
        {
            "schema_version": "mesh-segmentation-semantic-manifest.v1",
            "segments": [
                {
                    "segment_id": segment_id,
                    "name": name,
                    "output_prim_path": prim_path,
                }
                for segment_id, (name, prim_path) in enumerate(
                    _SEMANTIC_PRIM_PATHS.items()
                )
            ],
        },
    )
    _write_json(
        run_dir / "final/topology_validation.json",
        {
            "schema_version": "mesh-segmentation-topology-validation.v1",
            "status": "passed",
            "source_face_count": 8,
            "exported_face_count": 8,
            "exact_source_face_coverage": True,
            "overlapping_source_face_ids": [],
            "missing_source_face_ids": [],
            "topology_digest": topology_digest,
        },
    )
    render_dir = run_dir / "final/renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    usd_cli_session_id = "segmentation-session"
    usd_cli_source_revision = "test-usd-cli-revision"
    receipt_records: list[dict[str, Any]] = [
        {
            "schema_version": (
                "content-agent-workflows.mesh-evidence-usd-cli-receipt.v1"
            ),
            "workflow": "mesh-segmentation",
            "session_id": usd_cli_session_id,
            "arguments": ["open", str(segmented_usd)],
            "tool": {
                "name": "usd-cli",
                "source_revision": usd_cli_source_revision,
            },
            "status": "completed",
            "response": {"ok": True},
            "artifact_bindings": [],
        }
    ]
    render_records: list[dict[str, Any]] = []
    for index, direction in enumerate(("+x-y+z", "-x+y+z")):
        name = f"view_{index}"
        image_path = render_dir / f"{name}.png"
        camera_path = render_dir / f"{name}_camera.json"
        response_path = render_dir / f"{name}_response.json"
        position = [float(index), -3.0, 2.0]
        target = [1.0, 0.0, 0.0]
        Image.new(
            "RGB",
            (64, 48),
            color=(28 + index, 36, 48),
        ).save(image_path)
        _write_json(
            camera_path,
            {
                "camera_state": {
                    "position": position,
                    "target": target,
                    "focal_length": 60.0,
                    "horizontal_aperture": 36.0,
                },
                "camera_path": "/World/mesh_evidence",
                "camera_world_transform": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [*position, 1.0],
                ],
                "image_width": 64,
                "image_height": 48,
                "renderer": "ovrtx",
            },
        )
        channel_records: dict[str, dict[str, Any]] = {}
        for channel_name in ("normal", "linear_depth"):
            raw_path = render_dir / f"{name}_{channel_name}.npy"
            preview_path = render_dir / f"{name}_{channel_name}.png"
            raw_path.write_bytes(f"{name}:{channel_name}".encode())
            Image.new("RGB", (64, 48), color=(index, 12, 24)).save(preview_path)
            shape = [48, 64, 3] if channel_name == "normal" else [48, 64]
            channel_records[channel_name] = {
                "aov": channel_name,
                "evidence_role": "auxiliary_cpu_aov",
                "final_render_evidence": False,
                "shape": shape,
                "dtype": "float32",
                "raw": str(raw_path),
                "raw_sha256": file_sha256(raw_path),
                "preview": str(preview_path),
                "preview_sha256": file_sha256(preview_path),
            }
        channel_records["linear_depth"].update(
            {
                "encoding": "camera_space_linear_depth",
                "unit": "meter",
                "preview_encoding": "per_frame_normalized_uint8",
            }
        )
        response_payload = {
            "ok": True,
            "summary": {
                "backend": "ovrtx",
                "ovrtx_render_mode": "rt2",
                "ovrtx_num_sensor_updates": 64,
                "active_aov": "LdrColor",
            },
            "artifacts": [
                {"label": "rgb", "path": str(image_path)},
                {"label": "normals", "path": str(render_dir / f"{name}_normal.png")},
                {
                    "label": "depth",
                    "path": str(render_dir / f"{name}_linear_depth.png"),
                },
                {
                    "label": "linear_depth",
                    "path": str(render_dir / f"{name}_linear_depth.npy"),
                },
            ],
            "data": {"linear_depth_unit": "meter"},
        }
        _write_json(response_path, response_payload)
        receipt_records.append(
            {
                "schema_version": (
                    "content-agent-workflows.mesh-evidence-usd-cli-receipt.v1"
                ),
                "workflow": "mesh-segmentation",
                "session_id": usd_cli_session_id,
                "arguments": [
                    "camera",
                    "create",
                    "--name",
                    "mesh_evidence",
                    "--at",
                    ",".join(str(value) for value in position),
                    "--look-at",
                    ",".join(str(value) for value in target),
                    "--focal",
                    "60.0",
                    "--aperture",
                    "36.0",
                ],
                "tool": {
                    "name": "usd-cli",
                    "source_revision": usd_cli_source_revision,
                },
                "status": "completed",
                "response": {"ok": True},
                "artifact_bindings": [],
            }
        )
        receipt_records.append(
            {
                "schema_version": (
                    "content-agent-workflows.mesh-evidence-usd-cli-receipt.v1"
                ),
                "workflow": "mesh-segmentation",
                "session_id": usd_cli_session_id,
                "arguments": [
                    "render",
                    "--photoreal",
                    "--depth",
                    "--normals",
                    "--res",
                    "64x48",
                    "--output",
                    str(render_dir),
                ],
                "tool": {
                    "name": "usd-cli",
                    "source_revision": usd_cli_source_revision,
                },
                "status": "completed",
                "response": response_payload,
                "artifact_bindings": [
                    {
                        **artifact,
                        "sha256": file_sha256(Path(artifact["path"])),
                        "size_bytes": Path(artifact["path"]).stat().st_size,
                    }
                    for artifact in response_payload["artifacts"]
                ],
            }
        )
        render_records.append(
            {
                "name": name,
                "direction": direction,
                "source_camera": None,
                "image": str(image_path),
                "image_sha256": file_sha256(image_path),
                "camera": str(camera_path),
                "camera_sha256": file_sha256(camera_path),
                "response": str(response_path),
                "response_sha256": file_sha256(response_path),
                "request_seconds": 0.1,
                "renderer": "ovrtx",
                "channels": channel_records,
            }
        )
    receipt_dir = render_dir / ".usd_cli_receipts"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "mesh_evidence_commands.jsonl"
    receipt_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in receipt_records
        ),
        encoding="utf-8",
    )
    receipt_stat = receipt_path.stat()
    checkpoint_path = _write_json(
        receipt_dir / "mesh_evidence_commands.checkpoint.json",
        {
            "schema_version": (
                "content-agent-workflows.mesh-evidence-usd-cli-checkpoint.v1"
            ),
            "workflow": "mesh-segmentation",
            "session_id": usd_cli_session_id,
            "receipt_device": receipt_stat.st_dev,
            "receipt_inode": receipt_stat.st_ino,
            "receipt_sha256": file_sha256(receipt_path),
            "receipt_size_bytes": receipt_stat.st_size,
            "usd_cli_source_revision": usd_cli_source_revision,
        },
    )
    _write_json(
        run_dir / "final/renders/render_manifest.json",
        {
            "schema_version": SEGMENTATION_RENDER_SCHEMA_VERSION,
            "scene": str(segmented_usd),
            "scene_sha256": segmented_sha256,
            "focus": "/Asset",
            "width": 64,
            "height": 48,
            "renderer": "ovrtx",
            "scene_tool": "usd-cli",
            "usd_cli_session_id": usd_cli_session_id,
            "final_render_channels": ["rgb"],
            "auxiliary_cpu_aov_channels": ["normal", "linear_depth"],
            "usd_cli_command_receipts": str(receipt_path),
            "usd_cli_command_receipts_sha256": file_sha256(receipt_path),
            "usd_cli_receipt_checkpoint": str(checkpoint_path),
            "usd_cli_receipt_checkpoint_sha256": file_sha256(checkpoint_path),
            "renders": render_records,
        },
    )
    for relative in required_artifacts:
        artifact = run_dir / relative
        if artifact.exists():
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if relative == "final/face_labels.u32le":
            artifact.write_bytes(face_labels_path.read_bytes())
        else:
            artifact.write_bytes(f"fixture:{relative}".encode())
    _write_json(
        run_dir / "terminal_validation.json",
        {
            "schema_version": SEGMENTATION_TERMINAL_SCHEMA_VERSION,
            "valid": True,
            "status": "passed",
            "semantic_validation_errors": [],
            "required_artifacts": required_artifacts,
            "validated_artifact_count": len(required_artifacts),
        },
    )
    return _SegmentationRun(
        source_usd=source_usd,
        run_dir=run_dir,
        segmented_usd=segmented_usd,
        source_sha256=source_sha256,
        segmented_sha256=segmented_sha256,
        topology_digest=topology_digest,
    )


def test_consume_accepts_mechanically_valid_provisional_segmentation(
    segmentation_run: _SegmentationRun,
) -> None:
    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
        required_parts=["body", "handle"],
    )

    assert handoff.outcome == "conditional"
    assert handoff.requested is True
    assert handoff.required is True
    assert handoff.failures == []
    assert handoff.render_session_id == "segmentation-session"
    assert handoff.render_workspace_dir == str(
        segmentation_run.run_dir / "final/renders/.usd_cli_receipts"
    )
    assert len(handoff.warnings) == 1
    _assert_pending_ontology_warning(handoff.warnings)


def test_existing_semantic_part_uses_the_producer_fidelity_tier(
    segmentation_run: _SegmentationRun,
) -> None:
    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    records = _semantic_parts_for_handoff(
        metadata={
            "semantic_parts": [
                {"name": "body", "role": "body"},
                "authored_label",
            ]
        },
        segmentation=handoff,
    )
    body = next(record for record in records if record["name"] == "body")

    assert body["source_fidelity_tier"] == "exact_source_face_membership"
    assert body["segmentation"]["source_fidelity_tier"] == (
        "exact_source_face_membership"
    )
    authored = next(record for record in records if record["name"] == "authored_label")
    assert authored == {
        "name": "authored_label",
        "role": "design_body",
        "source": "provider_semantic_parts",
    }
    assert handoff.source_asset_sha256 == segmentation_run.source_sha256
    assert handoff.source_face_count == 8
    assert handoff.producer_run_id == segmentation_run.run_dir.name
    assert handoff.producer_workflow_mode == "recognition"
    assert handoff.target_prim_path == "/Asset/SourceMesh"
    assert handoff.topology_digest == segmentation_run.topology_digest
    assert handoff.fragment_labels_sha256 == file_sha256(
        segmentation_run.run_dir / "fragments/fragment_ids.u32le"
    )
    assert handoff.face_labels_sha256 == file_sha256(
        segmentation_run.run_dir / "state/final_labels.u32le"
    )
    assert handoff.segmented_usd_sha256 == segmentation_run.segmented_sha256
    assert Path(handoff.segmented_usd_path or "") == segmentation_run.segmented_usd
    assert {
        part.name: (
            part.output_prim_path,
            part.source_face_count,
            part.output_point_count,
        )
        for part in handoff.parts
    } == {
        "body": (_SEMANTIC_PRIM_PATHS["body"], 4, 4),
        "handle": (_SEMANTIC_PRIM_PATHS["handle"], 4, 4),
    }
    assert all(
        Path(path).is_file()
        for path in [
            handoff.terminal_validation_path,
            handoff.request_path,
            handoff.producer_manifest_path,
            handoff.export_manifest_path,
            handoff.source_topology_path,
            handoff.fragment_labels_path,
            handoff.face_labels_path,
            handoff.topology_validation_path,
            handoff.render_manifest_path,
        ]
        if path
    )


def test_source_bundle_semantic_parts_are_emitted_once(
    segmentation_run: _SegmentationRun,
) -> None:
    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )
    source_part = {
        "part_id": "body-id",
        "name": "body",
        "parent_part_id": None,
        "representation_ids": ["render"],
        "transform": [],
    }

    records = _semantic_parts_for_handoff(
        metadata={
            "source_bundle": {"parts": [source_part]},
            "semantic_parts": [source_part],
        },
        segmentation=handoff,
    )

    body_records = [item for item in records if item.get("name") == "body"]
    assert len(body_records) == 1
    assert body_records[0]["part_id"] == "body-id"
    assert body_records[0]["segmentation"]["source_fidelity_tier"] == (
        "exact_source_face_membership"
    )


def test_semantic_parts_with_distinct_ids_are_not_merged_by_name(
    segmentation_run: _SegmentationRun,
) -> None:
    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    records = _semantic_parts_for_handoff(
        metadata={
            "source_bundle": {
                "parts": [
                    {"part_id": "body-left", "name": "body"},
                    {"part_id": "body-right", "name": "body"},
                ]
            }
        },
        segmentation=handoff,
    )

    assert {item["part_id"] for item in records if item.get("name") == "body"} == {
        "body-left",
        "body-right",
    }


def test_consume_accepts_recognition_run_without_rich_segment_manifest(
    segmentation_run: _SegmentationRun,
) -> None:
    rich_manifest_path = segmentation_run.run_dir / "final/segment_manifest.json"
    rich_manifest_path.unlink()

    terminal_path = segmentation_run.run_dir / "terminal_validation.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["mode"] = "recognition"
    assert "final/segment_manifest.json" not in terminal["required_artifacts"]
    _write_json(terminal_path, terminal)

    request = json.loads(
        (segmentation_run.run_dir / "request.json").read_text(encoding="utf-8")
    )
    assert request["workflow_mode"] == "recognition"
    segments_path = segmentation_run.run_dir / "segments.json"

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
        required_parts=["body", "handle"],
    )

    assert handoff.outcome == "conditional"
    assert handoff.failures == []
    _assert_pending_ontology_warning(handoff.warnings)
    assert handoff.producer_manifest_path == str(segments_path)
    assert "segment_manifest" not in handoff.producer_schema_versions
    assert [part.name for part in handoff.parts] == ["body", "handle"]


@pytest.mark.parametrize("omitted_artifact", _CANONICAL_RECOGNITION_ARTIFACTS)
def test_terminal_required_artifacts_cannot_omit_canonical_artifact(
    segmentation_run: _SegmentationRun,
    omitted_artifact: str,
) -> None:
    terminal_path = segmentation_run.run_dir / "terminal_validation.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["required_artifacts"].remove(omitted_artifact)
    terminal["validated_artifact_count"] = len(terminal["required_artifacts"])
    _write_json(terminal_path, terminal)
    request_path = segmentation_run.run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["required_final_artifacts"].remove(omitted_artifact)
    _write_json(request_path, request)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    failure_text = " ".join(handoff.failures)
    assert omitted_artifact in failure_text
    assert "omit" in failure_text.lower()


def test_consume_rejects_source_digest_mismatch(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
) -> None:
    mismatched_source = tmp_path / "mismatched-source.usda"
    shutil.copy2(segmentation_run.source_usd, mismatched_source)
    with mismatched_source.open("a", encoding="utf-8") as stream:
        stream.write("\n# digest mismatch\n")

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=mismatched_source,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Segmentation source digest does not match the Geometry optimizer input"
    ]


def test_geometry_workflow_reports_rejected_segmentation_digest_as_evidence(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
) -> None:
    mismatched_source = tmp_path / "workflow-mismatched-source.usda"
    shutil.copy2(segmentation_run.source_usd, mismatched_source)
    with mismatched_source.open("a", encoding="utf-8") as stream:
        stream.write("\n# workflow digest mismatch\n")

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=mismatched_source,
            output_dir=tmp_path / "geometry-output",
            optimization_policy="skip",
            segmentation_run_dir=segmentation_run.run_dir,
            segmentation_required=True,
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is False
    assert result.error is None
    assert result.segmentation_outcome == "rejected"
    assert Path(result.geometry_usd_path or "").is_file()
    evidence = json.loads(
        Path(result.validation_evidence_path or "").read_text(encoding="utf-8")
    )
    part_check = next(
        check for check in evidence["checks"] if check["name"] == "part_segregation"
    )
    assert part_check["status"] == "fail"
    assert part_check["failures"] == [
        "Segmentation source digest does not match the Geometry optimizer input"
    ]
    assert "Digest-bound Geometry source changed" not in json.dumps(evidence)


def test_consume_rejects_targeted_run_that_would_drop_sibling_geometry(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.source_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    _define_tetrahedron(
        stage,
        "/Asset/UnrelatedSibling",
        x_offset=4.0,
    )
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Targeted segmentation cannot replace a source containing unrelated "
        "geometry; preserve or compose these prims before Geometry handoff: "
        "['/Asset/UnrelatedSibling']"
    ]


@pytest.mark.parametrize("target_state", ["abstract", "inactive"])
def test_consume_rejects_target_absent_from_composed_render_traversal(
    segmentation_run: _SegmentationRun,
    target_state: str,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.source_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    target = stage.GetPrimAtPath("/Asset/SourceMesh")
    if target_state == "abstract":
        assert target.SetSpecifier(Sdf.SpecifierClass)
    else:
        assert target.SetActive(False)
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Segmentation target is not an active, defined source UsdGeomMesh in "
        "the composed render traversal: /Asset/SourceMesh"
    ]


def test_consume_rejects_targeted_run_that_would_drop_instance_proxy_geometry(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.source_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    stage.CreateClassPrim("/_InstanceTemplate")
    _define_tetrahedron(
        stage,
        "/_InstanceTemplate/Mesh",
        x_offset=4.0,
    )
    instance = UsdGeom.Xform.Define(stage, "/Asset/UnrelatedInstance").GetPrim()
    assert instance.GetReferences().AddInternalReference("/_InstanceTemplate")
    assert instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Targeted segmentation cannot replace a source containing unrelated "
        "geometry; preserve or compose these prims before Geometry handoff: "
        "['/Asset/UnrelatedInstance/Mesh']"
    ]


def test_consume_rejects_targeted_run_that_would_drop_point_instancer(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.source_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    instancer = UsdGeom.PointInstancer.Define(stage, "/Asset/UnrelatedInstancer")
    instancer.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(4.0, 0.0, 0.0)]))
    instancer.CreateProtoIndicesAttr(Vt.IntArray([0]))
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Targeted segmentation cannot replace a source containing unrelated "
        "geometry; preserve or compose these prims before Geometry handoff: "
        "['/Asset/UnrelatedInstancer']"
    ]


def test_consume_rejects_subdivided_source_control_cage(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(str(segmentation_run.source_usd), load=Usd.Stage.LoadNone)
    assert stage is not None
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/SourceMesh"))
    assert mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Mesh segmentation source must use subdivisionScheme=none so source "
        "faces identify rendered triangles exactly"
    ]


def test_consume_rejects_source_with_hidden_hole_faces(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(str(segmentation_run.source_usd), load=Usd.Stage.LoadNone)
    assert stage is not None
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/SourceMesh"))
    assert mesh.CreateHoleIndicesAttr(Vt.IntArray([0]))
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Mesh segmentation source must not use holeIndices because source face "
        "membership must identify rendered triangles exactly"
    ]


def test_consume_rejects_segment_render_semantic_drift(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_SEMANTIC_PRIM_PATHS["body"]))
    assert mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.catmullClark)
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert len(handoff.failures) == 1
    assert (
        "Segment 'body' render semantics do not match the source mesh"
        in (handoff.failures[0])
    )
    assert "'subdivision_scheme': 'catmullClark'" in handoff.failures[0]


def test_consume_rejects_segment_hole_indices(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_SEMANTIC_PRIM_PATHS["body"]))
    assert mesh.CreateHoleIndicesAttr(Vt.IntArray([0]))
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert len(handoff.failures) == 1
    assert (
        "Segment 'body' render semantics do not match the source mesh"
        in (handoff.failures[0])
    )
    assert "'hole_indices': (0,)" in handoff.failures[0]


def test_consume_rejects_segmented_usd_digest_mismatch(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    stage.GetRootLayer().customLayerData = {"tampered_after_export": True}
    stage.GetRootLayer().Save()
    assert file_sha256(segmentation_run.segmented_usd) != (
        segmentation_run.segmented_sha256
    )

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "segmented.usdc digest does not match the export manifest"
    ]


def test_consume_rejects_undeclared_output_mesh(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    _define_tetrahedron(
        stage,
        "/Asset/UndeclaredExtra",
        x_offset=4.0,
    )
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Segmented USD mesh prims do not exactly match output_segments; "
        "undeclared=['/Asset/UndeclaredExtra'], missing=[]"
    ]


def test_consume_rejects_undeclared_non_mesh_boundable_geometry(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    cube = UsdGeom.Cube.Define(stage, "/Asset/InjectedCube")
    cube.CreateSizeAttr(4.0)
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.segmented_usd_path is None
    assert handoff.failures == [
        "Segmented USD contains undeclared non-mesh boundable geometry: "
        "['/Asset/InjectedCube (Cube)']"
    ]


def test_consume_rejects_overlapping_source_face_membership(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    handle_source_ids = stage.GetPrimAtPath(
        _SEMANTIC_PRIM_PATHS["handle"]
    ).GetAttribute("meshSegmentation:sourceFaceIds")
    assert handle_source_ids.Set(Vt.UIntArray([3, 4, 5, 6]))
    stage.GetRootLayer().Save()

    export_path = segmentation_run.run_dir / "final/export_manifest.json"
    export_manifest = json.loads(export_path.read_text(encoding="utf-8"))
    export_manifest["output_usd_sha256"] = file_sha256(segmentation_run.segmented_usd)
    _write_json(export_path, export_manifest)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Segment 'handle' source face 3 is bound to face-label segment 0, not 1"
    ]


def test_consume_rejects_direct_symlink_artifact(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
) -> None:
    segments_path = segmentation_run.run_dir / "segments.json"
    symlink_target = tmp_path / "segments-target.json"
    segments_path.replace(symlink_target)
    segments_path.symlink_to(symlink_target)
    assert segments_path.is_symlink()

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Segmentation artifact must not traverse a symlink: segments.json"
    ]


def test_consume_rejects_hardlinked_segmented_usd(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-segmented.usdc"
    segmentation_run.segmented_usd.replace(outside)
    os.link(outside, segmentation_run.segmented_usd)
    assert segmentation_run.segmented_usd.stat().st_nlink == 2

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Segmentation artifact must not be a hardlink: final/segmented.usdc"
    ]


@pytest.mark.parametrize(
    "dependency_kind",
    ["sublayer", "reference", "payload", "asset_path"],
)
def test_consume_rejects_composed_segmented_usd_dependencies(
    segmentation_run: _SegmentationRun,
    dependency_kind: str,
) -> None:
    dependency = _write_source_usd(segmentation_run.run_dir / "final/dependency.usda")
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    if dependency_kind == "sublayer":
        stage.GetRootLayer().subLayerPaths.append(dependency.name)
    elif dependency_kind == "reference":
        prim = stage.DefinePrim("/DependencyReference", "Xform")
        prim.GetReferences().AddReference(str(dependency), "/Asset")
    elif dependency_kind == "payload":
        prim = stage.DefinePrim("/DependencyPayload", "Xform")
        prim.GetPayloads().AddPayload(str(dependency), "/Asset")
    else:
        attribute = stage.GetPrimAtPath("/Asset").CreateAttribute(
            "malicious:assetPath",
            Sdf.ValueTypeNames.Asset,
        )
        assert attribute.Set(Sdf.AssetPath(str(dependency)))
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert len(handoff.failures) == 1
    assert "self-contained" in handoff.failures[0]


def test_consume_binds_segment_ids_to_each_output_prims_face_labels(
    segmentation_run: _SegmentationRun,
) -> None:
    export_path = segmentation_run.run_dir / "final/export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    first, second = export["output_segments"]
    first["output_prim_path"], second["output_prim_path"] = (
        second["output_prim_path"],
        first["output_prim_path"],
    )
    _write_json(export_path, export)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Segment 'body' source face 4 is bound to face-label segment 1, not 0"
    ]


@pytest.mark.parametrize(
    ("field", "value", "expected_failure"),
    [
        ("up_axis", "Y", "Source topology up_axis does not match source USD"),
        (
            "meters_per_unit",
            0.01,
            "Source topology meters_per_unit does not match source USD",
        ),
    ],
)
def test_consume_rejects_source_metadata_identity_drift(
    segmentation_run: _SegmentationRun,
    field: str,
    value: Any,
    expected_failure: str,
) -> None:
    topology_path = segmentation_run.run_dir / "prepare/topology.json"
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    topology[field] = value
    _write_json(topology_path, topology)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [expected_failure]


@pytest.mark.parametrize("field", ["up_axis", "meters_per_unit"])
def test_consume_rejects_output_stage_metadata_identity_drift(
    segmentation_run: _SegmentationRun,
    field: str,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    if field == "up_axis":
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    else:
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert len(handoff.failures) == 1
    assert (
        "Segmented USD up axis does not match" in handoff.failures[0]
        if field == "up_axis"
        else "Segmented USD metersPerUnit does not match" in handoff.failures[0]
    )


def test_consume_rejects_time_sampled_source_geometry(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(str(segmentation_run.source_usd), load=Usd.Stage.LoadNone)
    assert stage is not None
    points = UsdGeom.Mesh(stage.GetPrimAtPath("/Asset/SourceMesh")).GetPointsAttr()
    assert points.Set(points.Get(), Usd.TimeCode(1.0))
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert len(handoff.failures) == 1
    assert "Segmentation source mesh must be static" in handoff.failures[0]


def test_consume_rejects_time_sampled_output_transform(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    transform = UsdGeom.Xformable(stage.GetPrimAtPath("/Asset/Parts"))
    translate = transform.AddTranslateOp()
    assert translate.Set(Gf.Vec3d(0.0), Usd.TimeCode(1.0))
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert len(handoff.failures) == 1
    assert "Segment 'body' output mesh must be static" in handoff.failures[0]


def test_consume_requires_exact_uint_array_provenance(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    prim = stage.GetPrimAtPath(_SEMANTIC_PRIM_PATHS["body"])
    assert prim.RemoveProperty("meshSegmentation:sourceFaceIds")
    attribute = prim.CreateAttribute(
        "meshSegmentation:sourceFaceIds",
        Sdf.ValueTypeNames.IntArray,
    )
    assert attribute.Set(Vt.IntArray([0, 1, 2, 3]))
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == ["Segment 'body' provenance must be an exact UIntArray"]


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("extra_artifact", "canonical recognition artifact set"),
        ("staged_path", "asset path does not match its staged source"),
        ("staged_size", "source asset staged size is stale"),
        ("staged_digest", "source asset staged digest is stale"),
        ("run_id_format", "invalid run ID"),
        ("run_id_binding", "run_id does not match its run directory"),
        ("run_dir", "run_dir does not match the consumed run"),
        ("mode_vocabulary", "workflow mode is inconsistent"),
    ],
)
def test_consume_rejects_forged_v3_request_bindings(
    segmentation_run: _SegmentationRun,
    mutation: str,
    expected_failure: str,
) -> None:
    request_path = segmentation_run.run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if mutation == "extra_artifact":
        extra = segmentation_run.run_dir / "attacker/extra.bin"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"not canonical")
        request["required_final_artifacts"].append("attacker/extra.bin")
        terminal_path = segmentation_run.run_dir / "terminal_validation.json"
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["required_artifacts"].append("attacker/extra.bin")
        terminal["validated_artifact_count"] += 1
        _write_json(terminal_path, terminal)
    elif mutation == "staged_path":
        request["inputs"]["asset"] = str(
            segmentation_run.run_dir / "inputs/source/forged.usda"
        )
    elif mutation == "staged_size":
        request["input_staging"]["asset"]["size_bytes"] += 1
    elif mutation == "staged_digest":
        request["input_staging"]["asset"]["sha256"] = "0" * 64
    elif mutation == "run_id_format":
        request["run_id"] = "../forged"
    elif mutation == "run_id_binding":
        request["run_id"] = "different-valid-run"
    elif mutation == "run_dir":
        request["run_dir"] = str(segmentation_run.run_dir.parent)
    else:
        request["inputs"]["target_semantic_parts"] = ["body"]
    _write_json(request_path, request)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert len(handoff.failures) == 1
    assert expected_failure in handoff.failures[0]


def test_consume_rejects_targeted_segments_outside_exact_vocabulary(
    segmentation_run: _SegmentationRun,
) -> None:
    artifacts = [*_REQUIRED_FINAL_ARTIFACTS, *_REQUIRED_TARGETED_FINAL_ARTIFACTS]
    for relative in artifacts:
        path = segmentation_run.run_dir / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture:{relative}".encode())

    request_path = segmentation_run.run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["workflow_mode"] = "targeted"
    request["inputs"]["target_semantic_parts"] = ["body", "handle"]
    request["constraints"]["part_recognition_required"] = False
    request["required_final_artifacts"] = artifacts
    _write_json(request_path, request)
    terminal_path = segmentation_run.run_dir / "terminal_validation.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["required_artifacts"] = artifacts
    terminal["validated_artifact_count"] = len(artifacts)
    _write_json(terminal_path, terminal)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Targeted segmentation names do not match the exact request vocabulary "
        "plus other"
    ]


@pytest.mark.parametrize("confidence", ["high", float("nan")])
def test_consume_rejects_malformed_confidence_without_raising(
    segmentation_run: _SegmentationRun,
    confidence: Any,
) -> None:
    segments_path = segmentation_run.run_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    segments["segments"][0]["confidence"] = confidence
    _write_json(segments_path, segments)
    _refresh_segments_binding(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert len(handoff.failures) == 1
    assert "confidence must be a finite number" in handoff.failures[0]


def test_consume_rejects_malformed_scalar_provenance_without_raising(
    segmentation_run: _SegmentationRun,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.segmented_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    prim = stage.GetPrimAtPath(_SEMANTIC_PRIM_PATHS["body"])
    assert prim.RemoveProperty("meshSegmentation:sourceFaceIds")
    attribute = prim.CreateAttribute(
        "meshSegmentation:sourceFaceIds",
        Sdf.ValueTypeNames.UInt,
    )
    assert attribute.Set(0)
    stage.GetRootLayer().Save()
    _refresh_segmented_usd_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == ["Segment 'body' provenance must be an exact UIntArray"]


def test_consume_validates_declared_output_point_count(
    segmentation_run: _SegmentationRun,
) -> None:
    export_path = segmentation_run.run_dir / "final/export_manifest.json"
    export = json.loads(export_path.read_text(encoding="utf-8"))
    export["output_segments"][0]["output_point_count"] = 3
    _write_json(export_path, export)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Segment 'body' output_point_count does not match its mesh"
    ]


def test_consume_rejects_missing_required_semantic_part(
    segmentation_run: _SegmentationRun,
) -> None:
    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
        required_parts=["body", "handle", "wheel"],
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == ["Required semantic parts are missing: ['wheel']"]


def test_consume_rejects_render_evidence_without_ovrtx_attribution(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    render_manifest["renders"][0]["renderer"] = "storm"
    response_path = Path(render_manifest["renders"][0]["response"])
    _write_json(render_manifest_path, render_manifest)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["summary"]["backend"] = "storm"
    _replace_receipted_render_response(
        segmentation_run,
        view_index=0,
        replacement=response,
    )

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.parts == []
    assert handoff.failures == [
        "Segmentation render view view_0 is not attributed to OVRTX"
    ]


def test_consume_rejects_usd_cli_receipts_bound_to_foreign_scene(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    receipt_path = Path(render_manifest["usd_cli_command_receipts"])
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    receipts[0]["arguments"] = ["open", str(segmentation_run.source_usd)]
    receipt_path.write_text(
        "".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts),
        encoding="utf-8",
    )
    _refresh_usd_cli_receipt_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "usd-cli receipts open a scene other than the segmented render scene"
    ]


def test_consume_rejects_usd_cli_session_mismatch(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    render_manifest["usd_cli_session_id"] = "foreign-session"
    _write_json(render_manifest_path, render_manifest)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == ["usd-cli receipt checkpoint does not bind its journal"]


@pytest.mark.parametrize(
    "field",
    [
        "ovrtx_render_mode",
        "active_aov",
    ],
)
def test_consume_rejects_incomplete_ovrtx_response_metadata(
    segmentation_run: _SegmentationRun,
    field: str,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    first = render_manifest["renders"][0]
    response_path = Path(first["response"])
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response["summary"].pop(field)
    _replace_receipted_render_response(
        segmentation_run,
        view_index=0,
        replacement=response,
    )

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Segmentation render view view_0 lacks complete OVRTX metadata: "
        "usd-cli render omitted complete executed OVRTX metadata"
    ]


def test_consume_rejects_ovrtx_metadata_that_disagrees_with_camera(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    first = render_manifest["renders"][0]
    camera_path = Path(first["camera"])
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    camera["image_width"] = 32
    _write_json(camera_path, camera)
    first["camera_sha256"] = file_sha256(camera_path)
    _write_json(render_manifest_path, render_manifest)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Segmentation render view view_0 camera metadata disagrees with OVRTX: "
        "['image_width']"
    ]


def test_consume_rejects_nested_renderer_spoof(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    _write_json(
        render_manifest_path,
        {
            "schema_version": "untrusted-render-manifest.v1",
            "metadata": {"renderer": "ovrtx"},
            "renders": [{"backend": "ovrtx"}],
        },
    )

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == ["Unsupported mesh-segmentation render evidence schema"]


def test_consume_rejects_stale_render_image(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    image_path = Path(render_manifest["renders"][0]["image"])
    image_path.write_bytes(b"not the digest-bound render")

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        f"Render evidence {image_path.name} does not match image_sha256"
    ]


@pytest.mark.parametrize(
    ("artifact_label", "manifest_path", "digest_field"),
    [
        ("rgb", ("image",), "image_sha256"),
        ("normals", ("channels", "normal", "preview"), "preview_sha256"),
        ("depth", ("channels", "linear_depth", "preview"), "preview_sha256"),
        ("linear_depth", ("channels", "linear_depth", "raw"), "raw_sha256"),
    ],
)
def test_consume_rejects_rehashed_render_bytes_not_bound_by_receipt(
    segmentation_run: _SegmentationRun,
    artifact_label: str,
    manifest_path: tuple[str, ...],
    digest_field: str,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    first = render_manifest["renders"][0]
    record: dict[str, Any] = first
    for component in manifest_path[:-1]:
        record = record[component]
    artifact_path = Path(record[manifest_path[-1]])
    if artifact_path.suffix == ".png":
        Image.new("RGB", (64, 48), color=(201, 17, 93)).save(artifact_path)
    else:
        artifact_path.write_bytes(b"replacement auxiliary evidence")
    record[digest_field] = file_sha256(artifact_path)
    _write_json(render_manifest_path, render_manifest)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        f"Segmentation render view view_0 {artifact_label} bytes disagree with "
        "the usd-cli receipt"
    ]


def test_consume_rejects_stale_render_response(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    response_path = Path(render_manifest["renders"][0]["response"])
    _write_json(response_path, {"renderer": "storm"})

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        f"Render evidence {response_path.name} does not match response_sha256"
    ]


def test_consume_rejects_response_paired_with_another_views_receipt(
    segmentation_run: _SegmentationRun,
) -> None:
    manifest_path = segmentation_run.run_dir / "final/renders/render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first, second = manifest["renders"]
    wrong_response = json.loads(Path(second["response"]).read_text(encoding="utf-8"))
    first_response_path = Path(first["response"])
    _write_json(first_response_path, wrong_response)
    first["response_sha256"] = file_sha256(first_response_path)

    # Make the camera semantically match the wrong receipt while keeping its
    # normalized record distinct. The exact response-to-artifact binding must
    # still reject this stale cross-view pairing.
    wrong_camera = json.loads(Path(second["camera"]).read_text(encoding="utf-8"))
    wrong_camera["camera_state"]["evidence_view_id"] = "wrong-receipt-pair"
    first_camera_path = Path(first["camera"])
    _write_json(first_camera_path, wrong_camera)
    first["camera_sha256"] = file_sha256(first_camera_path)
    _write_json(manifest_path, manifest)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Segmentation render view view_0 manifest artifacts disagree with its "
        "usd-cli response: ['depth', 'linear_depth', 'normals', 'rgb']"
    ]


def test_consume_rejects_render_receipt_with_wrong_resolution(
    segmentation_run: _SegmentationRun,
) -> None:
    manifest_path = segmentation_run.run_dir / "final/renders/render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = Path(manifest["usd_cli_command_receipts"])
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    render_receipt = next(
        receipt
        for receipt in receipts
        if receipt.get("arguments", [None])[0] == "render"
    )
    resolution_index = render_receipt["arguments"].index("--res") + 1
    render_receipt["arguments"][resolution_index] = "32x32"
    receipt_path.write_text(
        "".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts),
        encoding="utf-8",
    )
    _refresh_usd_cli_receipt_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Segmentation render view view_0 receipt has the wrong resolution"
    ]


def test_consume_rejects_camera_file_that_disagrees_with_receipt(
    segmentation_run: _SegmentationRun,
) -> None:
    manifest_path = segmentation_run.run_dir / "final/renders/render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = Path(manifest["usd_cli_command_receipts"])
    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    camera_receipt = next(
        receipt
        for receipt in receipts
        if receipt.get("arguments", [None, None])[:2] == ["camera", "create"]
    )
    position_index = camera_receipt["arguments"].index("--at") + 1
    camera_receipt["arguments"][position_index] = "9.0,9.0,9.0"
    receipt_path.write_text(
        "".join(json.dumps(receipt, sort_keys=True) + "\n" for receipt in receipts),
        encoding="utf-8",
    )
    _refresh_usd_cli_receipt_bindings(segmentation_run)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Segmentation render view view_0 position disagrees with its usd-cli "
        "camera receipt"
    ]


def test_consume_rejects_normalized_repeated_render_camera_state(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    first = render_manifest["renders"][0]
    second = render_manifest["renders"][1]
    first_camera = json.loads(Path(first["camera"]).read_text(encoding="utf-8"))
    second_camera_path = Path(second["camera"])
    second_camera = json.loads(second_camera_path.read_text(encoding="utf-8"))
    second_camera["camera_state"] = first_camera["camera_state"]
    second_camera["different_file_metadata"] = "digest must not define viewpoint"
    _write_json(second_camera_path, second_camera)
    second["camera_sha256"] = file_sha256(second_camera_path)
    assert second["camera_sha256"] != first["camera_sha256"]
    _write_json(render_manifest_path, render_manifest)

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Render evidence viewpoints do not use distinct cameras"
    ]


def test_consume_without_required_ovrtx_evidence_is_conditional(
    segmentation_run: _SegmentationRun,
) -> None:
    render_manifest_path = (
        segmentation_run.run_dir / "final/renders/render_manifest.json"
    )
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    for record in render_manifest["renders"]:
        record["renderer"] = "storm"
    _write_json(render_manifest_path, render_manifest)
    for index, record in enumerate(render_manifest["renders"]):
        response_path = Path(record["response"])
        response = json.loads(response_path.read_text(encoding="utf-8"))
        response["summary"]["backend"] = "storm"
        _replace_receipted_render_response(
            segmentation_run,
            view_index=index,
            replacement=response,
        )

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
        require_ovrtx_evidence=False,
    )

    assert handoff.outcome == "conditional"
    assert handoff.failures == []
    assert "Segmentation render evidence lacks explicit OVRTX attribution." in (
        handoff.warnings
    )
    assert len(handoff.warnings) == 2
    _assert_pending_ontology_warning(handoff.warnings)
    assert Path(handoff.segmented_usd_path or "") == segmentation_run.segmented_usd
    assert [part.name for part in handoff.parts] == ["body", "handle"]


def test_required_artifact_lookup_fails_explicitly_without_assert(
    segmentation_run: _SegmentationRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_confined_file = segmentation_module._confined_file

    def return_none_for_terminal(
        run_dir: Path,
        relative: str,
        *,
        required: bool = True,
    ) -> Path | None:
        if relative == "terminal_validation.json":
            return None
        return real_confined_file(run_dir, relative, required=required)

    monkeypatch.setattr(
        segmentation_module,
        "_confined_file",
        return_none_for_terminal,
    )

    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    assert handoff.outcome == "rejected"
    assert handoff.failures == [
        "Missing required segmentation artifact: terminal_validation.json"
    ]


def test_segmentation_validation_requires_routing_decision_artifact(
    segmentation_run: _SegmentationRun,
) -> None:
    handoff = consume_segmentation_handoff(
        run_dir=segmentation_run.run_dir,
        expected_source_usd=segmentation_run.source_usd,
        required=True,
    )

    with pytest.raises(
        ValueError,
        match="routing_decision_path is required at the validation evidence boundary",
    ):
        segmentation_validation_check(handoff)


def test_route_segmentation_rejects_source_without_meshes(tmp_path: Path) -> None:
    source = tmp_path / "empty.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
    stage.GetRootLayer().Save()

    decision = route_segmentation(requested=True, source_usd_path=source)

    assert decision.route == "rejected"
    assert decision.reason_codes == ["no_mesh_prims"]
    assert decision.reasons == ["The source USD contains no mesh prims."]
    assert decision.metrics.unloaded_payload_count == 0


def test_geometry_workflow_integrates_provisional_segmentation_handoff(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
) -> None:
    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=segmentation_run.source_usd,
            output_dir=tmp_path / "geometry-output",
            optimization_policy="preserve_correspondence",
            segmentation_run_dir=segmentation_run.run_dir,
            segmentation_required=True,
            segmentation_required_parts=["body", "handle"],
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is True, result.error
    assert result.validation_status == "conditional"
    assert result.handoff_ready == "conditional"
    assert result.segmentation_outcome == "conditional"
    assert result.segmentation_route == "consume_completed_run"
    assert Path(result.segmentation_routing_path or "").is_file()
    assert Path(result.segmentation_usd_path or "") == segmentation_run.segmented_usd

    manifest_path = Path(result.handoff_manifest_path or "")
    assert manifest_path.name == "content_agents_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["handoff_ready"] == "conditional"
    assert manifest["optimization_policy"] == "preserve_correspondence"
    assert manifest["effective_optimization_policy"] == "skip"
    assert manifest["optimization_status"] == "semantic_lock_skip"
    assert manifest["segmentation"]["outcome"] == "conditional"
    assert manifest["segmentation"]["required"] is True
    assert manifest["segmentation"]["routing"]["route"] == "consume_completed_run"
    assert manifest["segmentation"]["source_asset_sha256"] == (
        segmentation_run.source_sha256
    )
    assert manifest["segmentation"]["segmented_usd"]["sha256"] == (
        segmentation_run.segmented_sha256
    )
    assert manifest["reports"]["part_segregation"] == str(
        segmentation_run.run_dir / "terminal_validation.json"
    )
    assert manifest["workflow_geometry"]["derived_from"]["role"] == (
        "semantic_segmentation"
    )
    assert Path(manifest["workflow_geometry"]["derived_from"]["path"]) == (
        segmentation_run.segmented_usd
    )
    assert {
        (part["name"], part["output_prim_path"], part["source_fidelity_tier"])
        for part in manifest["semantic_parts"]
    } == {
        ("body", _SEMANTIC_PRIM_PATHS["body"], "exact_source_face_membership"),
        (
            "handle",
            _SEMANTIC_PRIM_PATHS["handle"],
            "exact_source_face_membership",
        ),
    }
    evidence_bundle = json.loads(
        Path(result.evidence_bundle_path or "").read_text(encoding="utf-8")
    )
    assert evidence_bundle["optimization_policy"] == "preserve_correspondence"
    assert evidence_bundle["effective_optimization_policy"] == "skip"
    assert manifest["provenance"]["shared_optimization"][
        "protected_semantic_prim_paths"
    ] == list(_SEMANTIC_PRIM_PATHS.values())

    evidence = json.loads(
        Path(result.validation_evidence_path or "").read_text(encoding="utf-8")
    )
    assert evidence["metadata"]["handoff_ready"] == "conditional"
    assert evidence["metadata"]["segmentation"]["outcome"] == "conditional"
    _assert_pending_ontology_warning(evidence["warnings"])
    part_check = next(
        check for check in evidence["checks"] if check["name"] == "part_segregation"
    )
    assert part_check["status"] == "warning"
    assert part_check["metadata"]["part_count"] == 2
    assert {artifact["kind"] for artifact in part_check["evidence_artifacts"]} == {
        "mesh_segmentation_terminal_validation",
        "mesh_segmentation_request",
        "mesh_segmentation_manifest",
        "mesh_segmentation_export_manifest",
        "mesh_segmentation_source_topology",
        "mesh_segmentation_fragment_labels",
        "mesh_segmentation_face_labels",
        "mesh_segmentation_topology_validation",
        "mesh_segmentation_render_manifest",
        "mesh_segmentation_usd",
        "mesh_segmentation_routing",
    }


def test_geometry_workflow_preserves_sibling_geometry_when_targeted_run_rejected(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
) -> None:
    stage = Usd.Stage.Open(
        str(segmentation_run.source_usd),
        load=Usd.Stage.LoadNone,
    )
    assert stage is not None
    _define_tetrahedron(
        stage,
        "/Asset/UnrelatedSibling",
        x_offset=4.0,
    )
    stage.GetRootLayer().Save()
    _refresh_source_bindings(segmentation_run)

    result = run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=segmentation_run.source_usd,
            output_dir=tmp_path / "geometry-output",
            optimization_policy="skip",
            segmentation_run_dir=segmentation_run.run_dir,
            segmentation_required=True,
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert result.success is False
    assert result.segmentation_outcome == "rejected"
    output_stage = Usd.Stage.Open(
        str(result.geometry_usd_path),
        load=Usd.Stage.LoadNone,
    )
    assert output_stage is not None
    assert output_stage.GetPrimAtPath("/Asset/UnrelatedSibling").IsA(UsdGeom.Mesh)


def test_optimizer_preserves_and_reports_semantic_boundaries(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_usd = tmp_path / "optimized-preserved.usdc"

    def fake_run(_self: object, context: dict[str, Any]) -> dict[str, Any]:
        shutil.copy2(context["input_usd_path"], context["output_usd_path"])
        return {
            **context,
            "optimization_success": True,
            "optimization_metadata": {"mock_result": "preserved"},
        }

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", fake_run)
    monkeypatch.setattr(
        scene_ops,
        "_geometry_fidelity_check",
        lambda _source, _output: {"status": "pass", "passed": True},
    )

    optimization = scene_ops.optimize_geometry(
        source_usd=segmentation_run.segmented_usd,
        output_usd=output_usd,
        policy="runtime_efficiency",
        protected_semantic_prim_paths=list(_SEMANTIC_PRIM_PATHS.values()),
    )

    assert optimization["status"] == "completed"
    assert optimization["artifact_role"] == "optimized_geometry"
    assert optimization["shared_optimizer_metadata"] == {"mock_result": "preserved"}
    assert optimization["semantic_prim_boundaries"] == {
        "status": "pass",
        "passed": True,
        "prim_paths": list(_SEMANTIC_PRIM_PATHS.values()),
        "failures": [],
    }
    assert "rejected_output_usd" not in optimization
    assert output_usd.read_bytes() == segmentation_run.segmented_usd.read_bytes()

    persisted = json.loads(
        Path(optimization["metadata_path"]).read_text(encoding="utf-8")
    )
    assert persisted["status"] == "completed"
    assert persisted["semantic_prim_boundaries"]["status"] == "pass"
    assert persisted["semantic_prim_boundaries"]["passed"] is True


def test_preserve_correspondence_skips_incompatible_split_for_semantic_locks(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_usd = tmp_path / "semantic-lock-copy.usdc"

    def unexpected_run(_self: object, _context: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("Scene Optimizer must not run for locked semantic split")

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", unexpected_run)

    optimization = scene_ops.optimize_geometry(
        source_usd=segmentation_run.segmented_usd,
        output_usd=output_usd,
        policy="preserve_correspondence",
        protected_semantic_prim_paths=list(_SEMANTIC_PRIM_PATHS.values()),
    )

    assert optimization["status"] == "semantic_lock_skip"
    assert optimization["artifact_role"] == "normalized_copy"
    assert "does not preserve" in optimization["degraded_reason"]
    assert output_usd.read_bytes() == segmentation_run.segmented_usd.read_bytes()


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (
            "remove_prim",
            f"optimized semantic mesh is missing: {_SEMANTIC_PRIM_PATHS['handle']}",
        ),
        (
            "change_face_membership",
            "optimized semantic source-face membership or ordered face geometry "
            "changed at prim: "
            f"{_SEMANTIC_PRIM_PATHS['body']}",
        ),
        (
            "reorder_face_geometry",
            "optimized semantic source-face membership or ordered face geometry "
            "changed at prim: "
            f"{_SEMANTIC_PRIM_PATHS['body']}",
        ),
    ],
)
def test_optimizer_falls_back_when_protected_semantic_boundaries_change(
    segmentation_run: _SegmentationRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_failure: str,
) -> None:
    output_usd = tmp_path / f"optimized-{mutation}.usdc"

    def fake_run(_self: object, context: dict[str, Any]) -> dict[str, Any]:
        shutil.copy2(context["input_usd_path"], context["output_usd_path"])
        stage = Usd.Stage.Open(context["output_usd_path"], load=Usd.Stage.LoadNone)
        assert stage is not None
        if mutation == "remove_prim":
            assert stage.RemovePrim(_SEMANTIC_PRIM_PATHS["handle"])
        elif mutation == "change_face_membership":
            attribute = stage.GetPrimAtPath(_SEMANTIC_PRIM_PATHS["body"]).GetAttribute(
                "meshSegmentation:sourceFaceIds"
            )
            assert attribute.Set(Vt.UIntArray([0, 1, 2, 7]))
        else:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_SEMANTIC_PRIM_PATHS["body"]))
            indices = list(mesh.GetFaceVertexIndicesAttr().Get())
            reordered = [*indices[3:6], *indices[0:3], *indices[6:]]
            assert mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(reordered))
        stage.GetRootLayer().Save()
        return {
            **context,
            "optimization_success": True,
            "optimization_metadata": {"mock_mutation": mutation},
        }

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", fake_run)
    monkeypatch.setattr(
        scene_ops,
        "_geometry_fidelity_check",
        lambda _source, _output: {"status": "pass", "passed": True},
    )

    optimization = scene_ops.optimize_geometry(
        source_usd=segmentation_run.segmented_usd,
        output_usd=output_usd,
        policy="runtime_efficiency",
        protected_semantic_prim_paths=list(_SEMANTIC_PRIM_PATHS.values()),
    )

    assert optimization["status"] == "semantic_fidelity_fallback"
    assert optimization["artifact_role"] == "normalized_copy"
    assert optimization["protected_semantic_prim_paths"] == list(
        _SEMANTIC_PRIM_PATHS.values()
    )
    semantic_check = optimization["semantic_prim_boundaries"]
    assert semantic_check["status"] == "fail"
    assert semantic_check["passed"] is False
    assert expected_failure in semantic_check["failures"]
    rejected_output = Path(optimization["rejected_output_usd"])
    assert rejected_output.is_file()
    assert output_usd.read_bytes() == segmentation_run.segmented_usd.read_bytes()
    persisted = json.loads(
        Path(optimization["metadata_path"]).read_text(encoding="utf-8")
    )
    assert persisted["status"] == "semantic_fidelity_fallback"
    assert persisted["rejected_output_usd"] == str(rejected_output)
